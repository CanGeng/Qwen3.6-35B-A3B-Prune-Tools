"""LoRA + 4-bit quantization helpers for layerwise training.

Targeted at attention sub-modules (full + linear attention ``nn.Linear``s) of a
single ``Qwen3_5MoeDecoderLayer``. We deliberately avoid PEFT here — PEFT
needs the whole model rooted as a ``PeftModel``, but layerwise loads each
decoder layer in isolation and feeds hidden states through it. Our wrapper is
~50 lines and snapshots back to the original key names, so the downstream
``merge_layer_updates_into_student`` path is untouched.

Three public functions:

* :func:`apply_lora_to_layer`              — in-place wrap matched ``nn.Linear``
* :func:`freeze_base_train_lora`           — set ``requires_grad`` per the design
* :func:`snapshot_layer_with_merged_lora`  — produce a base-shape state_dict
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Norm class names that should always be frozen when LoRA is enabled. Keep
# the strings (not class refs) so we don't import the model's classes here.
_NORM_CLASS_NAMES = (
    "Qwen3_5MoeRMSNorm",
    "Qwen3_5MoeRMSNormGated",
    "FusedRMSNormGated",
    "RMSNorm",
    "LayerNorm",
)


@dataclass(frozen=True)
class LoRASpec:
    """Per-replaced-Linear metadata, indexed by dotted module path under the layer."""
    in_features: int
    out_features: int
    r: int
    alpha: int


def _torch_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16"):
        return torch.float16
    return torch.float32


class LoRAWrapper(nn.Module):
    """Wraps a base ``nn.Linear``-like module with a LoRA adapter.

    forward(x) = base(x) + (dropout(x) @ A.T) @ B.T * scaling

    ``base`` may be a vanilla ``nn.Linear`` or a ``bitsandbytes.nn.Linear4bit``;
    both expose the same call signature. We do not freeze ``base`` here —
    callers run :func:`freeze_base_train_lora` after wiring everything.
    """

    def __init__(
        self,
        base: nn.Module,
        in_features: int,
        out_features: int,
        r: int,
        alpha: int,
        dropout: float,
        adapter_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        # PEFT-standard init: A ~ kaiming_uniform, B = 0 so the adapter starts
        # as a no-op (i.e. forward output equals the base layer).
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        # Adapter parameters live in compute dtype (typically bf16) regardless
        # of how the base is stored; otherwise gradients on a quantized base
        # blow up the dtype matrix on every step.
        self.lora_A.to(adapter_dtype)
        self.lora_B.to(adapter_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        adapter_in = self.dropout(x)
        adapter_in = adapter_in.to(self.lora_A.weight.dtype)
        adapter_out = self.lora_B(self.lora_A(adapter_in)) * self.scaling
        return out + adapter_out.to(out.dtype)


def _is_norm_module(module: nn.Module) -> bool:
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        return True
    name = type(module).__name__
    return name in _NORM_CLASS_NAMES or name.endswith("RMSNorm") or name.endswith("LayerNorm")


def _make_base(
    orig: nn.Linear,
    *,
    load_in_4bit: bool,
    compute_dtype: torch.dtype,
    quant_type: str,
    target_device: torch.device,
) -> nn.Module:
    """Return a Linear4bit (or the original Linear) initialized from ``orig``'s weight."""
    if not load_in_4bit:
        return orig
    try:
        from bitsandbytes.nn import Linear4bit
    except ImportError as e:
        raise RuntimeError(
            "load_in_4bit=True requires bitsandbytes; install it or set load_in_4bit=False"
        ) from e

    out_features, in_features = orig.weight.shape
    has_bias = orig.bias is not None
    # Construct on CPU first; quantization happens at .to(cuda).
    new = Linear4bit(
        in_features,
        out_features,
        bias=has_bias,
        compute_dtype=compute_dtype,
        quant_type=quant_type,
    )
    # Copy fp/bf16 weights into Params4bit.data; .to(device) below quantizes.
    new.weight.data = orig.weight.data.detach().to(dtype=compute_dtype).contiguous()
    if has_bias:
        new.bias.data = orig.bias.data.detach().to(dtype=compute_dtype).contiguous()
    new = new.to(target_device)
    return new


def apply_lora_to_layer(
    layer: nn.Module,
    *,
    target_modules: tuple[str, ...] | list[str],
    r: int,
    alpha: int,
    dropout: float,
    load_in_4bit: bool,
    compute_dtype: torch.dtype | str = torch.bfloat16,
    quant_type: str = "nf4",
) -> dict[str, LoRASpec]:
    """Replace matched ``nn.Linear``s in ``layer`` with ``LoRAWrapper(Linear4bit | Linear)``.

    Returns a dict ``{dotted_module_path: LoRASpec}`` describing which paths
    were wrapped. The dotted path is the wrapper's path, e.g.
    ``"self_attn.q_proj"``; under it sit ``base.weight`` (possibly 4bit),
    ``lora_A.weight``, ``lora_B.weight``.
    """
    if isinstance(compute_dtype, str):
        compute_dtype = _torch_dtype(compute_dtype)
    targets = set(target_modules)
    meta: dict[str, LoRASpec] = {}

    # named_modules() yields entries in registration order; we mutate parents
    # via setattr after collecting the list (mutating during iteration is OK
    # in CPython but the explicit two-pass is safer).
    pending: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for full_name, module in layer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        attr_name = full_name.rsplit(".", 1)[-1]
        if attr_name not in targets:
            continue
        # Find parent module.
        parent_path = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        if parent_path:
            parent = layer.get_submodule(parent_path)
        else:
            parent = layer
        pending.append((parent, attr_name, full_name, module))

    if not pending:
        return meta

    # Pick the device from one of the layer's parameters (LoRA-targeted Linear
    # is materialised by load_layer_to_gpu before we wrap, so its weight is
    # already on the target device).
    sample_param = next(layer.parameters())
    target_device = sample_param.device

    for parent, attr_name, full_name, orig in pending:
        out_features, in_features = orig.weight.shape
        base = _make_base(
            orig,
            load_in_4bit=load_in_4bit,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
            target_device=target_device,
        )
        wrapper = LoRAWrapper(
            base=base,
            in_features=in_features,
            out_features=out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            adapter_dtype=compute_dtype,
        )
        wrapper.to(target_device)
        setattr(parent, attr_name, wrapper)
        meta[full_name] = LoRASpec(
            in_features=in_features,
            out_features=out_features,
            r=r,
            alpha=alpha,
        )

    return meta


def freeze_base_train_lora(layer: nn.Module, lora_meta: dict[str, LoRASpec]) -> None:
    """Apply the layerwise+LoRA freezing policy in place.

    Rules (from plan §2):

    * Norm modules → frozen (weight + bias).
    * For each path in ``lora_meta``: ``base`` (whatever its dtype) → frozen;
      ``lora_A`` / ``lora_B`` → trainable.
    * Everything else stays at ``requires_grad=True`` (assumes caller already
      flipped on grads for the layer, matching layerwise's default behavior).

    ``requires_grad_(True)`` is only legal on floating-point tensors; quantized
    4bit weights (Params4bit -> uint8 storage) are skipped during the default
    "make trainable" pass.
    """
    # 1. Default: every fp param trainable. uint8/int8 (e.g. Params4bit
    # storage) cannot require grad and is left as-is.
    for p in layer.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)

    # 2. Freeze norms.
    for _name, module in layer.named_modules():
        if _is_norm_module(module):
            for p in module.parameters(recurse=False):
                p.requires_grad_(False)

    # 3. Freeze each LoRA wrapper's base; leave A/B trainable.
    for path in lora_meta:
        wrapper = layer.get_submodule(path)
        for p in wrapper.base.parameters():
            # Skip non-floating-point (e.g. Params4bit's uint8 storage —
            # already non-trainable, and requires_grad_ would raise).
            if p.is_floating_point():
                p.requires_grad_(False)
        for p in wrapper.lora_A.parameters():
            p.requires_grad_(True)
        for p in wrapper.lora_B.parameters():
            p.requires_grad_(True)


def _dequantize_base_weight(base: nn.Module) -> torch.Tensor:
    """Return a dense (out, in) tensor for either Linear or Linear4bit."""
    try:
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        Linear4bit = None  # type: ignore[assignment]

    if Linear4bit is not None and isinstance(base, Linear4bit):
        from bitsandbytes.functional import dequantize_4bit
        w = base.weight
        if getattr(w, "quant_state", None) is None:
            # Not yet quantized (e.g. lived on CPU the whole time); just cast.
            return w.data.detach().clone()
        return dequantize_4bit(w.data, w.quant_state).detach().clone()
    return base.weight.data.detach().clone()


def snapshot_layer_with_merged_lora(
    layer: nn.Module,
    lora_meta: dict[str, LoRASpec],
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Return a state_dict with LoRA folded into base, keyed exactly like the
    original (pre-wrap) layer.

    Each wrapped path ``P`` in ``lora_meta`` contributes:
    * ``f"{P}.weight"`` ← dequantize(base.weight) + (B @ A) * scaling
    * ``f"{P}.bias"``   ← base.bias  (if present)

    Every other parameter / buffer is copied through under its original name.
    """
    sd: dict[str, torch.Tensor] = {}
    wrapper_path_prefixes = tuple(f"{p}." for p in lora_meta.keys())

    def _is_under_wrapper(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in wrapper_path_prefixes)

    # Collect non-LoRA parameters and buffers normally.
    for name, p in layer.named_parameters():
        if _is_under_wrapper(name):
            continue
        sd[name] = p.detach().to(out_dtype).cpu().contiguous()
    for name, b in layer.named_buffers():
        if _is_under_wrapper(name):
            continue
        sd[name] = b.detach().to(out_dtype).cpu().contiguous()

    # Rewrite each wrapped path back to its original key form.
    for path, _spec in lora_meta.items():
        wrapper = layer.get_submodule(path)
        base_w = _dequantize_base_weight(wrapper.base).to(out_dtype)
        a = wrapper.lora_A.weight.detach().to(torch.float32)
        b = wrapper.lora_B.weight.detach().to(torch.float32)
        delta = (b @ a).to(out_dtype) * float(wrapper.scaling)
        merged = (base_w + delta.to(base_w.device)).cpu().contiguous()
        sd[f"{path}.weight"] = merged
        if getattr(wrapper.base, "bias", None) is not None:
            sd[f"{path}.bias"] = wrapper.base.bias.detach().to(out_dtype).cpu().contiguous()

    return sd


__all__ = [
    "LoRASpec",
    "LoRAWrapper",
    "apply_lora_to_layer",
    "freeze_base_train_lora",
    "snapshot_layer_with_merged_lora",
]
