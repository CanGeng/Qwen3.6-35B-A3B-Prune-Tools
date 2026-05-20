"""Layer-wise (block-wise) student distillation.

Train the pruned student one block at a time, loading only the active block's
parameters onto the GPU. A *block* spans the layers between two consecutive
``cache_layers`` checkpoints — its input and target hidden states both come
from the existing teacher cache (``cache/teacher_hiddens/{sid}.safetensors``),
so blocks train independently and order-agnostically.

Block 0 spans ``[0 .. cache_layers[0]]`` and uses the embedding output as its
input. Trailing layers past the final cached checkpoint
(``layers > cache_layers[-1]``) have no target and are deferred to phase C
end-to-end fine-tuning.

Memory budget on a 16 GB GPU:

* up to 4 student layers in bf16 (~600 MB / layer) ≈ 2.4 GB params
* 8-bit AdamW state (m + v at 1 byte each) + bf16 grads ≈ params × 2
* gradient checkpointing on the inner layers of multi-layer blocks
* per-sample forward at ``batch_size=1``, seq_len up to ``max_seq_len``
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
import math
import random
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from moe_prune_distill.distill.layer_streamer import (
    build_position_inputs,
    load_embedding_to_gpu,
    load_layer_to_gpu,
    read_shard_index,
    unload_module,
)
from moe_prune_distill.distill.layer_lora import (
    LoRASpec,
    apply_lora_to_layer,
    freeze_base_train_lora,
    snapshot_layer_with_merged_lora,
)
from moe_prune_distill.distill.losses import normalized_hidden_mse, router_kl
from moe_prune_distill.distill.lr_scheduler import build_scheduler
from moe_prune_distill.distill.metrics import (
    batch_token_stats,
    hidden_metrics,
    router_diagnostics,
)
from moe_prune_distill.distill.rollout_cache import (
    RolloutCacheWriter,
    has_rollout,
    load_rollout_input,
)
from moe_prune_distill.distill.teacher_cache import load_sample_cache
from moe_prune_distill.utils.log_format import format_block_banner
from moe_prune_distill.utils.metrics_log import JsonlMetricsWriter
from moe_prune_distill.utils.tensorboard import TensorBoardWriter


_LAYER_PREFIX_FMT = "model.language_model.layers.{idx}."

_METRIC_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "core",
        (
            "loss",
            "ema_h",
            "lr",
            "gn",
        ),
    ),
    (
        "hidden",
        (
            "hidden_mse",
            "cos_loss",
            "nmse",
        ),
    ),
    (
        "router",
        (
            "router_kl",
            "removed_expert_mass",
            "router_entropy",
        ),
    ),
    (
        "stats",
        (
            "valid_tokens",
            "mean_seq_len",
            "student_norm",
            "teacher_norm",
        ),
    ),
)


def _format_metric_value(name: str, value: Any) -> str:
    """Format one scalar metric with stable width for readable logs."""
    if value is None:
        return "None"

    if isinstance(value, torch.Tensor):
        value = value.detach().to(torch.float32).cpu().item()

    if name == "valid_tokens":
        return f"{int(value):>10d}"

    if name == "mean_seq_len":
        return f"{float(value):>10.1f}"

    if name == "lr":
        return f"{float(value):>10.2e}"

    if isinstance(value, int):
        return f"{value:>10d}"

    if isinstance(value, float):
        abs_v = abs(value)
        if abs_v != 0.0 and (abs_v < 1e-4 or abs_v >= 1e4):
            return f"{value:>10.2e}"
        return f"{value:>10.5f}"

    return str(value)


def format_metrics_block(
    *,
    prefix: dict[str, Any],
    scalars: dict[str, Any],
) -> str:
    """Format train / val metrics as a multi-line readable block."""
    block = prefix.get("block", "?")
    step = prefix.get("step", "?")
    mode = str(prefix.get("mode", "?")).strip()

    group_width = 8
    label_width = max([len(k) for k in scalars.keys()] + [1])
    ordered_names: list[str] = []

    lines = [
        "",
        f"block={block} step={step} mode={mode}",
    ]

    for group_name, metric_names in _METRIC_GROUPS:
        group_items = [name for name in metric_names if name in scalars]
        if not group_items:
            continue

        line = f"{group_name:<{group_width}}: "
        # lines.append()
        for name in group_items:
            ordered_names.append(name)
            value = _format_metric_value(name, scalars[name])
            # lines.append(f"    {name:<{label_width}} : {value}")
            line += f"{name:<{label_width}}: {value} | "
        lines.append(line)

    # Keep newly added / unexpected metrics visible instead of silently hiding them.
    remaining_names = [name for name in scalars.keys() if name not in ordered_names]
    if remaining_names:
        lines.append("  other:")
        for name in sorted(remaining_names):
            value = _format_metric_value(name, scalars[name])
            lines.append(f"    {name:<{label_width}} : {value}")

    return "\n".join(lines)

# ====================================================================
# block enumeration
# ====================================================================


@dataclass(frozen=True)
class BlockSpec:
    """One block of the layerwise schedule.

    ``input_layer == -1`` means "feed the embedding output as input".
    ``layer_indices`` are the student layers the block trains; gradients
    flow through every one of them.
    """

    block_id: int
    input_layer: int            # -1 for embed
    output_layer: int           # the layer whose output the loss matches
    layer_indices: tuple[int, ...]


def enumerate_blocks(num_layers: int, cache_layers: Iterable[int]) -> list[BlockSpec]:
    """Partition ``[0..num_layers)`` into contiguous blocks bounded by cache layers.

    Trailing layers past the last cache checkpoint are skipped — they have no
    teacher target and are reserved for phase C end-to-end training.
    """
    cl = sorted(set(int(x) for x in cache_layers if 0 <= int(x) < num_layers))
    if not cl:
        return []
    blocks: list[BlockSpec] = []
    prev = -1
    for bid, out_layer in enumerate(cl):
        layers = tuple(range(prev + 1, out_layer + 1))
        if not layers:
            continue
        blocks.append(
            BlockSpec(
                block_id=bid,
                input_layer=prev,
                output_layer=out_layer,
                layer_indices=layers,
            )
        )
        prev = out_layer
    return blocks


# ====================================================================
# layer state I/O
# ====================================================================


def _layer_state_dict_with_prefix(layer: nn.Module, layer_idx: int) -> dict[str, torch.Tensor]:
    """Return the layer's state_dict with the global ``model.language_model.layers.{i}.`` prefix."""
    prefix = _LAYER_PREFIX_FMT.format(idx=layer_idx)
    return {f"{prefix}{k}": v.detach().cpu().contiguous() for k, v in layer.state_dict().items()}


def save_layer_snapshot(
    out_dir: Path,
    layer_idx: int,
    layer: nn.Module,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write a single layer's prefixed state_dict to ``_layers/layer_{i:03d}.safetensors``.

    Atomic via temp + replace so an interrupted run leaves either the previous
    snapshot or the new one — never a half-written file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = _layer_state_dict_with_prefix(layer, layer_idx)
    out = out_dir / f"layer_{layer_idx:03d}.safetensors"
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_file(sd, str(tmp))
    tmp.replace(out)
    if meta is not None:
        meta_path = out_dir / f"layer_{layer_idx:03d}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def save_layer_snapshot_from_dict(
    out_dir: Path,
    layer_idx: int,
    layer_sd: dict[str, torch.Tensor],
    meta: dict[str, Any] | None = None,
) -> Path:
    """Same as :func:`save_layer_snapshot` but takes an already-built state_dict.

    Keys in ``layer_sd`` must be the layer-local form (e.g. ``self_attn.q_proj.weight``);
    the global ``model.language_model.layers.{i}.`` prefix is added here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = _LAYER_PREFIX_FMT.format(idx=layer_idx)
    sd = {f"{prefix}{k}": v.detach().cpu().contiguous() for k, v in layer_sd.items()}
    out = out_dir / f"layer_{layer_idx:03d}.safetensors"
    tmp = out.with_suffix(out.suffix + ".tmp")
    save_file(sd, str(tmp))
    tmp.replace(out)
    if meta is not None:
        meta_path = out_dir / f"layer_{layer_idx:03d}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def block_done_marker(out_dir: Path, block_id: int) -> Path:
    return out_dir / f"block_{block_id:03d}.done.json"


# ====================================================================
# merge updated layers into a fresh student dir
# ====================================================================


def merge_layer_updates_into_student(
    student_dir: Path,
    updated_layers_dir: Path,
    out_dir: Path,
) -> None:
    """Stream-rewrite ``student_dir`` shards, swapping in updated layer weights.

    Output shards mirror input shards 1:1 (same names) so the index file
    stays valid. Non-layer keys (embed, lm_head, vision tower, norms) and
    layers without a snapshot pass through unchanged.
    """
    student_dir = Path(student_dir)
    updated_layers_dir = Path(updated_layers_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides: dict[str, Path] = {}
    if updated_layers_dir.is_dir():
        for p in sorted(updated_layers_dir.glob("layer_*.safetensors")):
            m = re.fullmatch(r"layer_(\d+)\.safetensors", p.name)
            if not m:
                continue
            with safe_open(str(p), framework="pt", device="cpu") as f:
                for k in f.keys():
                    overrides[k] = p

    weight_map = read_shard_index(student_dir)
    keys_by_shard: dict[str, list[str]] = {}
    for k, sh in weight_map.items():
        keys_by_shard.setdefault(sh, []).append(k)

    new_weight_map: dict[str, str] = {}
    written: list[Path] = []
    for shard_name in sorted(keys_by_shard.keys()):
        src = student_dir / shard_name

        out_sd: dict[str, torch.Tensor] = {}
        with safe_open(str(src), framework="pt", device="cpu") as f:
            for key in keys_by_shard[shard_name]:
                if key in overrides:
                    src_layer_file = overrides[key]
                    snap = load_file(str(src_layer_file))
                    out_sd[key] = snap[key].clone().contiguous()
                    del snap
                else:
                    out_sd[key] = f.get_tensor(key).contiguous()
        dst = out_dir / shard_name
        save_file(out_sd, str(dst))
        written.append(dst)
        for k in out_sd:
            new_weight_map[k] = shard_name
        out_sd.clear()
        del out_sd
        gc.collect()

    if (student_dir / "model.safetensors.index.json").is_file():
        total_size = sum(p.stat().st_size for p in written)
        index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
        (out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    import shutil
    for fname in (
        "config.json",
        "expert_mapping.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        src = student_dir / fname
        if src.is_file():
            shutil.copy2(src, out_dir / fname)


__all_merge__ = ["merge_layer_updates_into_student"]


# ====================================================================
# block trainer
# ====================================================================


@dataclass
class TrainerConfig:
    max_steps: int = 2000
    mse_threshold: float = 1e-3
    patience: int = 400
    learning_rate: float = 5e-5
    optimizer: str = "adamw_8bit"     # adamw_8bit | paged_adamw_8bit | adamw_fp32 | sso | sphere | muon | muon_triton | muon_triton_batched
    use_router_kl: bool = True
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    # When true, 3D MoE expert momentum_buffer is kept in host pinned RAM
    # between steps and staged to/from GPU per step. Saves ~3 GB on the
    # current example.yaml at the cost of +30-50% per-step wall time.
    muon_paged_momentum: bool = False
    router_kl_weight: float = 0.5
    router_temperature: float = 2.0
    save_every_steps: int = 200
    log_every_steps: int = 20
    grad_clip: float = 1.0
    seed: int = 42
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True   # checkpoint each in-block layer (use_reentrant=False)
    # SSO-only knobs (ignored by AdamW variants)
    sso_ns_steps: int = 6
    sso_radius_c: float = 1.0
    sso_radius_mode: str = "preserve"   # paper | preserve (preserve is right for distill)
    sso_msign_dtype: str = "fp32"       # fp32 (paper §5.2) | bf16 (faster, less precise)
    sso_bisect_max_iters: int = 20
    sso_bisect_tol: float = 2e-4
    sso_power_iters: int = 4
    # LR schedule (per-block reset)
    lr_scheduler_type: str = "cosine"   # cosine | linear | constant
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.0
    # Validation / logging
    eval_every_steps: int = 0           # 0 -> reuse log_every_steps
    val_sample_ids: tuple[str, ...] = ()
    train_log_path: Any | None = None
    val_log_path: Any | None = None
    surviving_by_layer: dict[int, list[int]] | None = None
    # TensorBoard (per-block subdirectory; soft dep, no-op if tensorboard not installed)
    tensorboard_enabled: bool = True
    tensorboard_log_dir: str | None = None    # parent dir; per-block subdir is appended
    # Student-rollout chaining: if true, block N+1 reads its input from a
    # post-block-N student forward instead of teacher_cache[input_layer].
    # When true ``rollout_root`` must point at a writable directory.
    use_student_rollout_input: bool = False
    rollout_root: Any | None = None
    rollout_chunk_size: int = 1000
    # LoRA + 4bit (attention sub-modules only). When ``lora_enabled`` is true,
    # LoRA-targeted Linears are wrapped (optionally 4bit) and only A/B train;
    # all norm parameters in the layer are frozen; the rest stays full-FT.
    lora_enabled: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ()
    lora_load_in_4bit: bool = True
    lora_compute_dtype: str = "bfloat16"
    lora_quant_type: str = "nf4"


def _make_optimizer(
    layers: torch.nn.ModuleList,
    cfg: TrainerConfig,
    log: logging.Logger,
):
    """Build the optimizer for the trainable layers.

    ``cfg.optimizer`` selects:

    * ``adamw_8bit``        — bnb 8-bit AdamW (Windows fallback to fp32 AdamW).
    * ``paged_adamw_8bit``  — bnb 8-bit AdamW with optimizer state paged to
                              host RAM (CUDA only). Same hyperparams as
                              ``adamw_8bit`` but ~3 GB less GPU resident.
    * ``adamw_fp32``        — vanilla torch AdamW.
    * ``sso``               — Spectral Sphere Optimizer (paper Algorithm 1).
    * ``sphere``            — MuonSphere variant (SSO with λ=0, retraction kept).
    * ``muon``              — plain Muon (no retraction; sanity baseline).

    For the spectral variants we auto-partition: 2D / 3D matrices (router
    gate, attention proj, MoE expert stacks) go through the matrix branch;
    1D norms / biases get foreach AdamW alongside.
    """
    if cfg.optimizer in ("adamw_8bit", "paged_adamw_8bit"):
        cls_name = (
            "PagedAdamW8bit" if cfg.optimizer == "paged_adamw_8bit" else "AdamW8bit"
        )
        try:
            import bitsandbytes as bnb

            cls = getattr(bnb.optim, cls_name)
            params = [p for p in layers.parameters() if p.requires_grad]
            return cls(params, lr=cfg.learning_rate)
        except Exception as e:
            log.warning("%s unavailable (%s); falling back to fp32 AdamW", cls_name, e)
            cfg = TrainerConfig(**{**cfg.__dict__, "optimizer": "adamw_fp32"})

    if cfg.optimizer == "adamw_fp32":
        params = [p for p in layers.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=cfg.learning_rate)

    if cfg.optimizer in ("muon_triton", "muon_triton_batched"):
        from moe_prune_distill.distill.muon_triton import (
            Muon,
            MuonBatched,
            partition_for_muon,
        )

        muon_params, adamw_params = partition_for_muon(layers.named_parameters())
        cls = MuonBatched if cfg.optimizer == "muon_triton_batched" else Muon
        log.info(
            "Optimizer mode=%s: %d matrix params (Muon branch) + %d scalar/embed params (AdamW)"
            "%s",
            cfg.optimizer, len(muon_params), len(adamw_params),
            " | paged_momentum=ON" if cfg.muon_paged_momentum else "",
        )
        return cls(
            muon_params, adamw_params,
            lr=cfg.learning_rate,
            momentum=cfg.muon_momentum,
            nesterov=True,
            ns_steps=cfg.muon_ns_steps,
            paged_momentum=cfg.muon_paged_momentum,
        )

    if cfg.optimizer in ("sso", "sphere", "muon"):
        from moe_prune_distill.distill.optimizer import SSO, partition_for_sso

        sso_params, adamw_params = partition_for_sso(layers.named_parameters())
        log.info(
            "Optimizer mode=%s radius_mode=%s msign=%s: %d matrix params (SSO branch) + %d scalar/embed params (AdamW)",
            cfg.optimizer,
            cfg.sso_radius_mode,
            cfg.sso_msign_dtype,
            len(sso_params),
            len(adamw_params),
        )
        msign_dtype = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }.get(cfg.sso_msign_dtype.lower())
        if msign_dtype is None:
            raise ValueError(
                f"sso_msign_dtype must be one of fp32 / bf16, got {cfg.sso_msign_dtype!r}"
            )
        return SSO(
            sso_params,
            adamw_params,
            lr=cfg.learning_rate,
            wd=0.1,
            wd_matrix=0.0,
            momentum=0.95,
            nesterov=True,
            ns_steps=cfg.sso_ns_steps,
            radius_c=cfg.sso_radius_c,
            radius_mode=cfg.sso_radius_mode,
            mode=cfg.optimizer,
            msign_dtype=msign_dtype,
            bisect_max_iters=cfg.sso_bisect_max_iters,
            bisect_tol=cfg.sso_bisect_tol,
            power_iters=cfg.sso_power_iters,
        )

    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


class BlockTrainer:
    """Train one ``BlockSpec`` against the existing teacher cache.

    Loads its layers on construction, runs ``run()`` for the configured step
    budget, snapshots the trained layers to ``snapshot_dir``, then unloads.
    """

    def __init__(
        self,
        *,
        block: BlockSpec,
        student_dir: Path,
        student_text_config: Any,
        student_weight_map: dict[str, str],
        cache_dir: Path,
        sample_ids: list[str],
        snapshot_dir: Path,
        device: torch.device | str,
        dtype: torch.dtype,
        cfg: TrainerConfig,
        surviving_by_layer: dict[int, list[int]] | None = None,
        log: logging.Logger | None = None,
        total_blocks: int | None = None,
    ) -> None:
        self.block = block
        self.student_dir = Path(student_dir)
        self.text_config = student_text_config
        self.weight_map = dict(student_weight_map)
        self.cache_dir = Path(cache_dir)
        self.sample_ids = list(sample_ids)
        self.snapshot_dir = Path(snapshot_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.cfg = cfg
        self.surviving_by_layer = surviving_by_layer or {}
        self.log = log or logging.getLogger("moe_prune_distill.layerwise")
        self.total_blocks = total_blocks

        self.embed: nn.Module | None = None
        self.layers: nn.ModuleList | None = None
        # Per-layer LoRA meta (path -> LoRASpec). Empty when lora_enabled=False.
        self._lora_meta_per_layer: dict[int, dict[str, LoRASpec]] = {}

        self._rollout_root: Path | None = (
            Path(cfg.rollout_root) if cfg.rollout_root is not None else None
        )

    # ---- module loading ----

    def _load_modules(self) -> None:
        if self.block.input_layer < 0:
            self.embed = load_embedding_to_gpu(
                self.text_config, self.weight_map, self.student_dir, self.device, self.dtype
            )
            for p in self.embed.parameters():
                p.requires_grad_(False)
        layers: list[nn.Module] = []
        gc_func = (
            partial(torch_checkpoint, use_reentrant=False)
            if self.cfg.gradient_checkpointing
            else None
        )
        for li in self.block.layer_indices:
            layer = load_layer_to_gpu(
                self.text_config, li, self.weight_map, self.student_dir, self.device, self.dtype
            )
            if self.cfg.lora_enabled:
                lora_meta = apply_lora_to_layer(
                    layer,
                    target_modules=tuple(self.cfg.lora_target_modules),
                    r=self.cfg.lora_r,
                    alpha=self.cfg.lora_alpha,
                    dropout=self.cfg.lora_dropout,
                    load_in_4bit=self.cfg.lora_load_in_4bit,
                    compute_dtype=self.cfg.lora_compute_dtype,
                    quant_type=self.cfg.lora_quant_type,
                )
                self._lora_meta_per_layer[li] = lora_meta
                freeze_base_train_lora(layer, lora_meta)
                self.log.info(
                    "block %03d layer %03d: LoRA wrapped %d Linears (r=%d, 4bit=%s)",
                    self.block.block_id,
                    li,
                    len(lora_meta),
                    self.cfg.lora_r,
                    self.cfg.lora_load_in_4bit,
                )
            else:
                for p in layer.parameters():
                    p.requires_grad_(True)
            layer.train()
            if gc_func is not None:
                # Qwen3_5MoeDecoderLayer derives from GradientCheckpointingLayer
                # ([transformers/modeling_layers.py]). Setting these two
                # attributes on the layer is enough — its __call__ wraps
                # super().__call__ in _gradient_checkpointing_func when both
                # gradient_checkpointing and self.training are True.
                layer.gradient_checkpointing = True
                layer._gradient_checkpointing_func = gc_func
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def _trainable_params(self) -> list[nn.Parameter]:
        assert self.layers is not None
        return [p for p in self.layers.parameters() if p.requires_grad]

    # ---- forward through the block ----

    def _block_forward(
        self,
        h_in: torch.Tensor,
        attn_mask_2d: torch.Tensor,
        capture_router_for: set[int] | None = None,
        position_ids_3d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run h_in through every layer in the block, returning (h_out, router_logits_dict).

        ``router_logits_dict`` only contains entries for layers in
        ``capture_router_for``; an empty/None set turns the hook off.

        ``position_ids_3d`` is the (3, B, S) M-RoPE table for VL samples;
        when ``None`` we fall back to plain 1D positions (text-only).
        """
        assert self.layers is not None
        seq_len = h_in.shape[1]
        batch_size = h_in.shape[0]
        pos = build_position_inputs(
            seq_len=seq_len,
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype,
            text_config=self.text_config,
            attention_mask_2d=attn_mask_2d.to(self.device),
            position_ids_3d=position_ids_3d,
        )
        captured: dict[int, torch.Tensor] = {}
        handles: list[Any] = []
        if capture_router_for:
            for li, layer in zip(self.block.layer_indices, self.layers):
                if li not in capture_router_for:
                    continue

                def make_hook(layer_idx: int):
                    def hook(_m, _inp, out):
                        captured[layer_idx] = out[0]

                    return hook

                handles.append(layer.mlp.gate.register_forward_hook(make_hook(li)))

        h = h_in
        for li, layer in zip(self.block.layer_indices, self.layers):
            layer_mask = (
                pos.linear_attn_mask
                if self.text_config.layer_types[li] == "linear_attention"
                else pos.causal_mask
            )
            h = layer(
                h,
                position_embeddings=(pos.cos, pos.sin),
                attention_mask=layer_mask,
                position_ids=pos.text_pos,
                past_key_values=None,
            )

        for handle in handles:
            handle.remove()
        return h, captured

    # ---- one training step ----

    def _build_microbatch(
        self, sids: list[str]
    ) -> tuple[
        torch.Tensor,            # h_in   [B, S, H]
        torch.Tensor,            # h_target [B, S, H]
        torch.Tensor,            # attn_mask [B, S]
        dict[int, torch.Tensor], # teacher router per layer [B, S, E]  (may be empty)
        torch.Tensor | None,     # position_ids_3d [3, B, S] or None when text-only
    ]:
        """Load ``sids``, pad to common seq_len, return stacked tensors on GPU.

        For VL samples the teacher cache stores ``inputs_embeds`` (merged text +
        vision-tower outputs) and ``position_ids_3d``; block 0 uses these
        directly so we never re-run the vision tower at training time.
        """
        cache_layers_to_load = [self.block.output_layer]
        if self.block.input_layer >= 0:
            cache_layers_to_load.append(self.block.input_layer)

        per_sample: list[dict[str, Any]] = []
        capture_router = self.cfg.use_router_kl
        wanted_router_layers: set[int] = set(self.block.layer_indices)

        time_steps = self._time_steps

        for sid in sids:
            if time_steps:
                t0 = time.perf_counter()
            cache = load_sample_cache(self.cache_dir, sid, layers=cache_layers_to_load)
            if time_steps:
                self._t_cache_load += time.perf_counter() - t0
                t0 = time.perf_counter()
            attn = cache["attention_mask"].to(torch.long)
            cached_inputs_embeds = cache.get("inputs_embeds")
            cached_pos3d = cache.get("position_ids_3d")
            is_vl = cached_inputs_embeds is not None
            if self.block.input_layer < 0:
                if is_vl:
                    # VL: use the merged inputs_embeds the streamer cached, so
                    # we don't re-run the vision tower at training time.
                    h_in = cached_inputs_embeds.to(
                        device=self.device, dtype=self.dtype
                    )
                else:
                    assert self.embed is not None
                    ids = cache["input_ids"].to(self.device, dtype=torch.long).unsqueeze(0)
                    with torch.no_grad():
                        h_in = self.embed(ids).to(self.dtype).squeeze(0)
            elif (
                self.cfg.use_student_rollout_input
                and self._rollout_root is not None
                and has_rollout(self._rollout_root, sid)
            ):
                h_in = load_rollout_input(self._rollout_root, sid).to(
                    device=self.device, dtype=self.dtype
                )
            else:
                h_in = cache["hidden"][self.block.input_layer].to(
                    device=self.device, dtype=self.dtype
                )
            h_tgt = cache["hidden"][self.block.output_layer].to(
                device=self.device, dtype=self.dtype
            )
            sample_router: dict[int, torch.Tensor] = {}
            if capture_router and cache["router"]:
                for li, t in cache["router"].items():  # type: ignore[union-attr]
                    li_i = int(li)
                    if li_i in wanted_router_layers:
                        sample_router[li_i] = t.to(device=self.device, dtype=self.dtype)
            per_sample.append(
                {
                    "h_in": h_in,
                    "h_tgt": h_tgt,
                    "attn": attn,
                    "router": sample_router,
                    "pos3d": cached_pos3d,
                }
            )
            if time_steps:
                # Sync so the .to() above actually completes before we attribute
                # its cost to h2d (rather than letting it leak into fwd).
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._t_h2d += time.perf_counter() - t0

        max_len = max(s["h_in"].shape[0] for s in per_sample)
        bsz = len(per_sample)
        hidden_size = per_sample[0]["h_in"].shape[-1]

        h_in_b = torch.zeros((bsz, max_len, hidden_size), device=self.device, dtype=self.dtype)
        h_tgt_b = torch.zeros_like(h_in_b)
        attn_b = torch.zeros((bsz, max_len), dtype=torch.long, device=self.device)

        router_b: dict[int, torch.Tensor] = {}
        if per_sample[0]["router"]:
            for li, t0 in per_sample[0]["router"].items():
                router_b[li] = torch.zeros(
                    (bsz, max_len, t0.shape[-1]), device=self.device, dtype=self.dtype
                )

        pos3d_b: torch.Tensor | None = None
        if any(s["pos3d"] is not None for s in per_sample):
            pos3d_b = torch.zeros((3, bsz, max_len), dtype=torch.long, device=self.device)

        for i, s in enumerate(per_sample):
            sl = s["h_in"].shape[0]
            h_in_b[i, :sl] = s["h_in"]
            h_tgt_b[i, :sl] = s["h_tgt"]
            attn_b[i, :sl] = s["attn"].to(self.device)
            for li, t in s["router"].items():
                if li in router_b:
                    router_b[li][i, :sl] = t
            if pos3d_b is not None:
                if s["pos3d"] is not None:
                    p = s["pos3d"].to(device=self.device, dtype=torch.long)
                    pos3d_b[:, i, :sl] = p[:, :sl]
                    if sl < max_len:
                        tail = p[:, sl - 1] + 1
                        pos3d_b[:, i, sl:] = tail.unsqueeze(-1)
                else:
                    pos3d_b[:, i, :sl] = torch.arange(
                        sl, device=self.device
                    ).unsqueeze(0)

        return h_in_b, h_tgt_b, attn_b, router_b, pos3d_b

    def _microstep_loss(
        self, sids: list[str]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward one micro-batch and return its (unscaled) loss + components."""
        h_in, h_target, attn_mask, t_router, pos3d = self._build_microbatch(sids)

        capture_set: set[int] = set(t_router.keys()) if (
            self.cfg.use_router_kl and t_router
        ) else set()

        h_out, router_caps = self._block_forward(
            h_in, attn_mask, capture_set or None, position_ids_3d=pos3d
        )

        s_hidden = {self.block.output_layer: h_out}
        t_hidden = {self.block.output_layer: h_target}
        loss_h = normalized_hidden_mse(
            s_hidden, t_hidden, weighting="uniform", attention_mask=attn_mask
        )
        loss = loss_h
        comps = {"hidden_mse": float(loss_h.detach().to(torch.float32).cpu())}

        diag = hidden_metrics(s_hidden, t_hidden, attn_mask)
        for k, v in diag.items():
            if k != "hidden_mse":
                comps[k] = v

        if router_caps and self.cfg.use_router_kl and t_router and self.surviving_by_layer:
            B, T = h_in.shape[0], h_in.shape[1]
            s_router: dict[int, torch.Tensor] = {}
            for li, r in router_caps.items():
                # The Qwen3.5 MoE router gate flattens batch×seq into a single
                # dim ([B*T, E]); teacher cache stays as [B, T, E]. Reshape so
                # the loss / diagnostics can broadcast.
                if r.dim() == 2 and r.shape[0] == B * T:
                    r = r.view(B, T, r.shape[-1])
                s_router[int(li)] = r
            t_router_aligned = {
                li: t for li, t in t_router.items() if li in s_router
            }
            if t_router_aligned:
                rkl = router_kl(
                    s_router,
                    t_router_aligned,
                    self.surviving_by_layer,
                    temperature=self.cfg.router_temperature,
                    attention_mask=attn_mask,
                )
                loss = loss + self.cfg.router_kl_weight * rkl
                comps["router_kl"] = float(rkl.detach().to(torch.float32).cpu())
                rdiag = router_diagnostics(
                    s_router, t_router_aligned, self.surviving_by_layer, attn_mask
                )
                comps.update(rdiag)

        comps.update(batch_token_stats(attn_mask))
        comps["loss"] = float(loss.detach().to(torch.float32).cpu())
        return loss, comps

    @torch.no_grad()
    def _evaluate(self, val_ids: list[str]) -> dict[str, float]:
        """Run one no-grad pass over ``val_ids`` and return averaged metrics."""
        if not val_ids:
            return {}
        assert self.layers is not None
        for layer in self.layers:
            layer.eval()
        try:
            sums: dict[str, float] = {}
            counts: dict[str, int] = {}
            bsz = max(1, int(self.cfg.batch_size))
            for start in range(0, len(val_ids), bsz):
                sids = val_ids[start : start + bsz]
                _, comps = self._microstep_loss(sids)
                for k, v in comps.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
            return {k: sums[k] / counts[k] for k in sums}
        finally:
            for layer in self.layers:
                layer.train()

    @torch.no_grad()
    def run_rollout(self, sids: list[str]) -> None:
        """Forward every sample in ``sids`` through the trained block and cache
        the output hidden state at ``block.output_layer``.

        Reuses the loaded ``self.layers`` (avoids a second load_layer_to_gpu).
        Caller is responsible for ensuring this is invoked **after** the final
        snapshot but **before** ``_unload``.
        """
        if not sids or self._rollout_root is None:
            return
        assert self.layers is not None
        writer = RolloutCacheWriter(
            self._rollout_root,
            block_id=self.block.block_id,
            output_layer=self.block.output_layer,
            sample_ids=sids,
            chunk_size=self.cfg.rollout_chunk_size,
        )
        for layer in self.layers:
            layer.eval()
        try:
            bsz = max(1, int(self.cfg.batch_size))
            chunks_seen: set[int] = set()
            chunks_done: set[int] = set()
            for start in range(0, len(sids), bsz):
                batch_sids = sids[start : start + bsz]
                # Build inputs the same way _build_microbatch does, but read the
                # block's INPUT from teacher_cache (or embeddings) — never from
                # the rollout cache, even when use_student_rollout_input is set.
                # Block N's rollout is computed from block N's actual training
                # input source for sid; for block 0 that's embeddings, for
                # block N>0 it's whichever input chained into the loss this run.
                per_sample: list[dict[str, Any]] = []
                for sid in batch_sids:
                    cache = load_sample_cache(
                        self.cache_dir,
                        sid,
                        layers=[self.block.input_layer]
                        if self.block.input_layer >= 0
                        else [],
                    )
                    attn = cache["attention_mask"].to(torch.long)
                    cached_inputs_embeds = cache.get("inputs_embeds")
                    cached_pos3d = cache.get("position_ids_3d")
                    is_vl = cached_inputs_embeds is not None
                    if self.block.input_layer < 0:
                        if is_vl:
                            h_in = cached_inputs_embeds.to(
                                device=self.device, dtype=self.dtype
                            )
                        else:
                            assert self.embed is not None
                            ids = (
                                cache["input_ids"]
                                .to(self.device, dtype=torch.long)
                                .unsqueeze(0)
                            )
                            h_in = self.embed(ids).to(self.dtype).squeeze(0)
                    elif (
                        self.cfg.use_student_rollout_input
                        and has_rollout(self._rollout_root, sid)
                    ):
                        h_in = load_rollout_input(self._rollout_root, sid).to(
                            device=self.device, dtype=self.dtype
                        )
                    else:
                        h_in = cache["hidden"][self.block.input_layer].to(
                            device=self.device, dtype=self.dtype
                        )
                    per_sample.append(
                        {"sid": sid, "h_in": h_in, "attn": attn, "pos3d": cached_pos3d}
                    )

                max_len = max(s["h_in"].shape[0] for s in per_sample)
                hidden_size = per_sample[0]["h_in"].shape[-1]
                bsz_b = len(per_sample)
                h_in_b = torch.zeros(
                    (bsz_b, max_len, hidden_size),
                    device=self.device,
                    dtype=self.dtype,
                )
                attn_b = torch.zeros(
                    (bsz_b, max_len), dtype=torch.long, device=self.device
                )
                pos3d_b: torch.Tensor | None = None
                if any(s["pos3d"] is not None for s in per_sample):
                    pos3d_b = torch.zeros(
                        (3, bsz_b, max_len), dtype=torch.long, device=self.device
                    )
                for i, s in enumerate(per_sample):
                    sl = s["h_in"].shape[0]
                    h_in_b[i, :sl] = s["h_in"]
                    attn_b[i, :sl] = s["attn"].to(self.device)
                    if pos3d_b is not None:
                        if s["pos3d"] is not None:
                            p = s["pos3d"].to(device=self.device, dtype=torch.long)
                            pos3d_b[:, i, :sl] = p[:, :sl]
                            if sl < max_len:
                                tail = p[:, sl - 1] + 1
                                pos3d_b[:, i, sl:] = tail.unsqueeze(-1)
                        else:
                            pos3d_b[:, i, :sl] = torch.arange(
                                sl, device=self.device
                            ).unsqueeze(0)

                h_out, _ = self._block_forward(
                    h_in_b, attn_b, capture_router_for=None, position_ids_3d=pos3d_b
                )

                for i, s in enumerate(per_sample):
                    sl = s["h_in"].shape[0]
                    writer.add(s["sid"], h_out[i, :sl].detach())
                    chunks_seen.add(writer.sid_to_chunk[s["sid"]])

                # Best-effort per-chunk flush: if every sample for some chunk is
                # already buffered, flush it to free RAM. Cheap heuristic: when
                # the *next* batch's first sample lives in a different chunk,
                # the previous chunks are complete (sids list is in declaration
                # order, which sid_to_chunk groups by stride chunk_size).
                if start + bsz < len(sids):
                    next_chunk = writer.sid_to_chunk[sids[start + bsz]]
                    for ch in sorted(chunks_seen):
                        if ch < next_chunk and ch not in chunks_done:
                            writer.flush_chunk(ch)
                            chunks_done.add(ch)
            writer.finalize()
        finally:
            for layer in self.layers:
                layer.train()

    # ---- main run ----

    def run(self) -> dict[str, Any]:
        self._load_modules()
        params = self._trainable_params()
        if not params:
            raise RuntimeError(f"Block {self.block.block_id}: no trainable params")
        n_params = sum(p.numel() for p in params)
        bsz = max(1, int(self.cfg.batch_size))
        accum = max(1, int(self.cfg.gradient_accumulation_steps))
        if self.total_blocks is not None:
            self.log.info(
                format_block_banner(
                    self.block.block_id,
                    self.total_blocks,
                    self.block.layer_indices,
                    n_params,
                )
            )
        self.log.info(
            "block %03d  samples=%d  batch=%d × accum=%d  grad_ckpt=%s  optimizer=%s",
            self.block.block_id,
            len(self.sample_ids),
            bsz,
            accum,
            "on" if self.cfg.gradient_checkpointing else "off",
            self.cfg.optimizer,
        )
        assert self.layers is not None
        optimizer = _make_optimizer(self.layers, self.cfg, self.log)
        warmup_steps = int(self.cfg.max_steps * self.cfg.warmup_ratio)
        scheduler = build_scheduler(
            optimizer,
            type_=self.cfg.lr_scheduler_type,
            num_warmup=warmup_steps,
            num_training=self.cfg.max_steps,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )
        self.log.info(
            "Block %d LR schedule: %s warmup=%d total=%d min_lr_ratio=%.3f",
            self.block.block_id,
            self.cfg.lr_scheduler_type,
            warmup_steps,
            self.cfg.max_steps,
            self.cfg.min_lr_ratio,
        )

        train_writer = (
            JsonlMetricsWriter(self.cfg.train_log_path)
            if self.cfg.train_log_path is not None
            else None
        )
        val_writer = (
            JsonlMetricsWriter(self.cfg.val_log_path)
            if self.cfg.val_log_path is not None
            else None
        )
        tb_dir: Path | None = None
        if self.cfg.tensorboard_enabled and self.cfg.tensorboard_log_dir:
            tb_dir = Path(self.cfg.tensorboard_log_dir) / f"block_{self.block.block_id:03d}"
        tb_train = TensorBoardWriter(
            tb_dir,
            enabled=self.cfg.tensorboard_enabled,
            namespace="train",
        )
        tb_val = TensorBoardWriter(
            tb_dir,
            enabled=self.cfg.tensorboard_enabled and bool(self.cfg.val_sample_ids),
            namespace="val",
        )
        eval_every = (
            self.cfg.eval_every_steps if self.cfg.eval_every_steps > 0
            else self.cfg.log_every_steps
        )
        val_ids = list(self.cfg.val_sample_ids or ())

        rng = random.Random(self.cfg.seed + self.block.block_id)
        ids = list(self.sample_ids)
        rng.shuffle(ids)
        cursor = 0

        def take(n: int) -> list[str]:
            nonlocal cursor, ids
            out: list[str] = []
            while len(out) < n:
                if cursor >= len(ids):
                    rng.shuffle(ids)
                    cursor = 0
                remaining = len(ids) - cursor
                grab = min(n - len(out), remaining)
                out.extend(ids[cursor : cursor + grab])
                cursor += grab
            return out

        ema = None
        ema_beta = 0.95
        best_ema = math.inf
        steps_since_improve = 0
        history: list[dict[str, float]] = []
        step = 0

        # Opt-in step-phase timer (env: MPD_TIME_STEPS=1). Tracks where the
        # wall time of an optimizer step actually goes: cache_load (disk→CPU),
        # h2d (CPU→GPU), fwd, bwd, opt. Reset every log_every_steps so the
        # printed numbers are per-window averages, not per-run cumulatives.
        self._time_steps = bool(int(os.environ.get("MPD_TIME_STEPS", "0") or "0"))
        self._t_cache_load = 0.0
        self._t_h2d = 0.0
        self._t_fwd = 0.0
        self._t_bwd = 0.0
        self._t_opt = 0.0
        timer_window_steps = 0

        try:
            for step in range(1, self.cfg.max_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                comps_acc: dict[str, float] = {}
                for micro in range(accum):
                    sids = take(bsz)
                    if self._time_steps:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_fwd0 = time.perf_counter()
                    loss, comps = self._microstep_loss(sids)
                    if self._time_steps:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        # _microstep_loss includes _build_microbatch (cache_load + h2d
                        # already accounted for separately). The remainder is fwd.
                        self._t_fwd += time.perf_counter() - t_fwd0
                        t_bwd0 = time.perf_counter()
                    (loss / accum).backward()
                    if self._time_steps:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        self._t_bwd += time.perf_counter() - t_bwd0
                    for k, v in comps.items():
                        comps_acc[k] = comps_acc.get(k, 0.0) + v / accum

                grad_norm = float("nan")
                if self.cfg.grad_clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(
                        self._trainable_params(), self.cfg.grad_clip
                    )
                    grad_norm = float(gn) if torch.is_tensor(gn) else float(gn)
                if self._time_steps:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_opt0 = time.perf_counter()
                optimizer.step()
                scheduler.step()
                if self._time_steps:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    self._t_opt += time.perf_counter() - t_opt0
                lr_now = float(scheduler.get_last_lr()[0])
                timer_window_steps += 1

                history.append({"step": step, **comps_acc})
                cur = comps_acc["hidden_mse"]
                ema = cur if ema is None else ema_beta * ema + (1 - ema_beta) * cur
                if ema < best_ema - 1e-6:
                    best_ema = ema
                    steps_since_improve = 0
                else:
                    steps_since_improve += 1

                if step % self.cfg.log_every_steps == 0:
                    if self._time_steps and timer_window_steps > 0:
                        n = timer_window_steps
                        total = (
                            self._t_cache_load + self._t_h2d
                            + self._t_fwd + self._t_bwd + self._t_opt
                        )
                        self.log.info(
                            "\n"
                            "block=%03d step=%6d phase-ms avg/step over %d\n"
                            "  cache : %8.1f\n"
                            "  h2d   : %8.1f\n"
                            "  fwd   : %8.1f\n"
                            "  bwd   : %8.1f\n"
                            "  opt   : %8.1f\n"
                            "  total : %8.1f",
                            self.block.block_id,
                            step,
                            n,
                            1000 * self._t_cache_load / n,
                            1000 * self._t_h2d / n,
                            1000 * self._t_fwd / n,
                            1000 * self._t_bwd / n,
                            1000 * self._t_opt / n,
                            1000 * total / n,
                        )
                        self._t_cache_load = 0.0
                        self._t_h2d = 0.0
                        self._t_fwd = 0.0
                        self._t_bwd = 0.0
                        self._t_opt = 0.0
                        timer_window_steps = 0
                    train_scalars: dict[str, float] = {
                        "loss": comps_acc["loss"],
                        "ema_h": float(ema),
                        "lr": lr_now,
                        "gn": grad_norm,
                    }
                    for k, v in comps_acc.items():
                        if k != "loss":
                            train_scalars[k] = float(v)
                    train_prefix = {
                        "block": f"{self.block.block_id:03d}",
                        "step": f"{step:>6d}",
                        "mode": "train",
                    }
                    self.log.info(
                        format_metrics_block(prefix=train_prefix, scalars=train_scalars)
                    )
                    if train_writer is not None:
                        row = {
                            "step": step,
                            "block": self.block.block_id,
                            "lr": lr_now,
                            "grad_norm": grad_norm,
                            "ema_h": float(ema),
                            **comps_acc,
                        }
                        train_writer.log(row)
                    tb_train.log(
                        {
                            "lr": lr_now,
                            "grad_norm": grad_norm,
                            "ema_h": float(ema),
                            **comps_acc,
                        },
                        step=step,
                    )

                if (
                    val_writer is not None
                    and val_ids
                    and step % eval_every == 0
                ):
                    val_metrics = self._evaluate(val_ids)
                    if val_metrics:
                        val_writer.log(
                            {
                                "step": step,
                                "block": self.block.block_id,
                                "lr": lr_now,
                                **val_metrics,
                            }
                        )
                        tb_val.log({"lr": lr_now, **val_metrics}, step=step)
                        val_scalars = {"lr": lr_now, **{
                            k: float(v) for k, v in val_metrics.items()
                            if isinstance(v, (int, float))
                        }}
                        val_prefix = {
                            "block": f"{self.block.block_id:03d}",
                            "step": f"{step:>6d}",
                            "mode": "  val",
                        }
                        self.log.info(
                            format_metrics_block(prefix=val_prefix, scalars=val_scalars)
                        )

                if step % self.cfg.save_every_steps == 0:
                    self._snapshot(meta={"step": step, "ema_hidden_mse": ema})

                if ema < self.cfg.mse_threshold:
                    self.log.info(
                        "block %d converged at step %d (ema=%.5f < %.5f)",
                        self.block.block_id,
                        step,
                        ema,
                        self.cfg.mse_threshold,
                    )
                    break
                if steps_since_improve >= self.cfg.patience:
                    self.log.info(
                        "block %d early-stop: %d steps no improvement",
                        self.block.block_id,
                        steps_since_improve,
                    )
                    break

            self._snapshot(meta={"step": step, "ema_hidden_mse": ema, "final": True})
            block_done_marker(self.snapshot_dir, self.block.block_id).write_text(
                json.dumps({"final_step": step, "ema_hidden_mse": ema}, indent=2),
                encoding="utf-8",
            )
            if (
                self.cfg.use_student_rollout_input
                and self._rollout_root is not None
            ):
                self.log.info(
                    "block %d: writing student rollout cache for %d samples -> %s",
                    self.block.block_id,
                    len(self.sample_ids),
                    self._rollout_root,
                )
                self.run_rollout(list(self.sample_ids))
            torch.cuda.empty_cache()
        finally:
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            del optimizer
            if train_writer is not None:
                train_writer.close()
            if val_writer is not None:
                val_writer.close()
            tb_train.close()
            tb_val.close()
            self._unload()

        return {"steps": step, "ema_hidden_mse": ema, "history": history}

    def _snapshot(self, meta: dict[str, Any]) -> None:
        assert self.layers is not None
        for li, layer in zip(self.block.layer_indices, self.layers):
            if self.cfg.lora_enabled and self._lora_meta_per_layer.get(li):
                # Merge LoRA delta into base, dequantize 4bit if needed, then
                # write a state_dict whose keys match the original (pre-wrap)
                # layer — keeps merge_layer_updates_into_student transparent.
                merged_sd = snapshot_layer_with_merged_lora(
                    layer,
                    self._lora_meta_per_layer[li],
                    out_dtype=self.dtype,
                )
                save_layer_snapshot_from_dict(
                    self.snapshot_dir, li, merged_sd, meta=meta
                )
            else:
                save_layer_snapshot(self.snapshot_dir, li, layer, meta=meta)

    def _unload(self) -> None:
        if self.layers is not None:
            for layer in self.layers:
                unload_module(layer)
            self.layers = None
        if self.embed is not None:
            unload_module(self.embed)
            self.embed = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "BlockSpec",
    "BlockTrainer",
    "TrainerConfig",
    "block_done_marker",
    "enumerate_blocks",
    "merge_layer_updates_into_student",
    "save_layer_snapshot",
    "save_layer_snapshot_from_dict",
]
