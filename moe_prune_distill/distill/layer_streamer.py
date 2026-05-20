"""Stream teacher MoE inference layer-by-layer to fit a 16GB GPU.

The classical approach — load the whole 35B teacher with ``device_map="auto"``
and forward each sample — pays the layer-swap cost ``num_samples * num_layers``
times. This module flips the loop: load each transformer block to GPU exactly
once and run **every** sample through it before moving to the next layer.

Per-sample hidden states live in a chunked scratch directory between layers
(``scratch_chunk_{j}.cur.safetensors`` holds every sample whose chunk id is
``j``). The embedding pass writes these chunk files; every layer reads each
chunk's ``cur`` file in turn, forwards each sample through the layer, writes
the outputs to a sibling ``next`` file, and atomically renames it over ``cur``.
This trades disk traffic (``2 * N_samples * seq * hidden * 2 bytes``) for an
order-of-magnitude reduction in GPU<->CPU weight movement, while keeping the
on-disk file count proportional to ``N_samples / chunk_size`` instead of
``N_samples`` itself (so ~10000 samples → ~10 scratch files, not 20000).

Teacher cache is emitted in the v2 batched layout (one safetensors per
``(layer, sample-chunk)`` plus an index json) via
:class:`~moe_prune_distill.distill.teacher_cache.BatchedCacheWriter`. Each
``(sample, layer)`` tensor pair is written exactly once at the end of the
relevant layer pass — no read-modify-write, no separate stage→finalize phase.

The API is intentionally narrow:

* :func:`read_shard_index`, :func:`keys_for_layer`, :func:`load_layer_to_gpu`,
  :func:`load_embedding_to_gpu`, :func:`build_position_inputs` are stateless
  helpers that the unit tests exercise individually.
* :class:`LayerStreamer` is a thin orchestrator that wires them together,
  invokes a router hook, and (optionally) writes per-layer chunk cache files
  via :class:`BatchedCacheWriter`.

Currently specific to ``Qwen3_5MoeForConditionalGeneration``: the layer class,
rotary embedding, and mask helpers come from the matching transformers module.
"""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from moe_prune_distill.distill.teacher_cache import BatchedCacheWriter


_EMBED_KEY = "model.language_model.embed_tokens.weight"
_LAYER_PREFIX_FMT = "model.language_model.layers.{idx}."
_VISUAL_PREFIX = "model.visual."


# ====================================================================
# stateless helpers
# ====================================================================


def read_shard_index(teacher_dir: str | Path) -> dict[str, str]:
    """Load ``model.safetensors.index.json``'s ``weight_map``.

    For unsharded teachers (single ``model.safetensors``) we synthesise an
    index by walking the file once, so callers don't need to special-case
    that path.
    """
    teacher_dir = Path(teacher_dir)
    idx_path = teacher_dir / "model.safetensors.index.json"
    if idx_path.is_file():
        return dict(json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"])
    single = teacher_dir / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"no safetensors found under {teacher_dir}")
    with safe_open(str(single), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    return {k: single.name for k in keys}


def keys_for_layer(weight_map: dict[str, str], layer_idx: int) -> dict[str, str]:
    """Subset the weight map to entries belonging to ``layer_idx``."""
    prefix = _LAYER_PREFIX_FMT.format(idx=layer_idx)
    return {k: v for k, v in weight_map.items() if k.startswith(prefix)}


def _materialise_from_shards(
    module: torch.nn.Module,
    keys: dict[str, str],
    teacher_dir: Path,
    device: torch.device | str,
    dtype: torch.dtype,
    strip_prefix: str,
) -> None:
    by_shard: dict[str, list[str]] = {}
    for k, sh in keys.items():
        by_shard.setdefault(sh, []).append(k)
    for shard_name, shard_keys in by_shard.items():
        path = teacher_dir / shard_name
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for k in shard_keys:
                local = k[len(strip_prefix):] if strip_prefix else k
                t = f.get_tensor(k).to(dtype)
                set_module_tensor_to_device(
                    module, local, device, value=t, dtype=dtype
                )


def load_layer_to_gpu(
    text_config: Any,
    layer_idx: int,
    weight_map: dict[str, str],
    teacher_dir: str | Path,
    device: torch.device | str,
    dtype: torch.dtype,
):
    """Materialise a single ``Qwen3_5MoeDecoderLayer`` on ``device``."""
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeDecoderLayer,
    )

    teacher_dir = Path(teacher_dir)
    keys = keys_for_layer(weight_map, layer_idx)
    if not keys:
        raise ValueError(f"no weights for layer {layer_idx} in weight map")
    with torch.device("meta"):
        layer = Qwen3_5MoeDecoderLayer(text_config, layer_idx)
    layer = layer.to(dtype)
    prefix = _LAYER_PREFIX_FMT.format(idx=layer_idx)
    _materialise_from_shards(layer, keys, teacher_dir, device, dtype, prefix)
    layer.eval()
    for p in layer.parameters():
        p.requires_grad_(False)
    return layer


def load_embedding_to_gpu(
    text_config: Any,
    weight_map: dict[str, str],
    teacher_dir: str | Path,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.nn.Embedding:
    """Materialise the input ``embed_tokens`` layer on ``device``."""
    teacher_dir = Path(teacher_dir)
    if _EMBED_KEY not in weight_map:
        raise KeyError(f"weight map missing {_EMBED_KEY}")
    with torch.device("meta"):
        embed = torch.nn.Embedding(
            text_config.vocab_size,
            text_config.hidden_size,
            padding_idx=getattr(text_config, "pad_token_id", None),
        )
    embed = embed.to(dtype)
    _materialise_from_shards(
        embed,
        {_EMBED_KEY: weight_map[_EMBED_KEY]},
        teacher_dir,
        device,
        dtype,
        strip_prefix="model.language_model.embed_tokens.",
    )
    embed.eval()
    for p in embed.parameters():
        p.requires_grad_(False)
    return embed


def load_vision_tower_to_gpu(
    full_config: Any,
    weight_map: dict[str, str],
    teacher_dir: str | Path,
    device: torch.device | str,
    dtype: torch.dtype,
):
    """Materialise the ``Qwen3_5MoeVisionModel`` on ``device``.

    ``full_config`` must be a ``Qwen3_5MoeConfig`` (with ``.vision_config``),
    not just the text sub-config — the vision tower needs its own
    ``hidden_size``/``depth``/etc.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeVisionModel,
        Qwen3_5MoeVisionRotaryEmbedding,
    )

    teacher_dir = Path(teacher_dir)
    keys = {k: v for k, v in weight_map.items() if k.startswith(_VISUAL_PREFIX)}
    if not keys:
        raise KeyError(
            "weight map has no model.visual.* keys; teacher checkpoint is text-only"
        )
    with torch.device("meta"):
        visual = Qwen3_5MoeVisionModel(full_config.vision_config)
    visual = visual.to(dtype)
    _materialise_from_shards(visual, keys, teacher_dir, device, dtype, _VISUAL_PREFIX)
    # Qwen3_5MoeVisionRotaryEmbedding.inv_freq is persistent=False, so it is
    # NOT in the safetensors shards — _materialise_from_shards skips it and
    # the buffer stays on `meta`. Recompute it on-device per its __init__
    # formula so the first vision forward doesn't blow up with
    # "Tensor on device meta is not on the expected device cuda:0".
    for m in visual.modules():
        if isinstance(m, Qwen3_5MoeVisionRotaryEmbedding):
            inv_freq = 1.0 / (
                m.theta
                ** (torch.arange(0, m.dim, 2, dtype=torch.float, device=device) / m.dim)
            )
            m.register_buffer("inv_freq", inv_freq, persistent=False)
    visual.eval()
    for p in visual.parameters():
        p.requires_grad_(False)
    return visual


def unload_module(module: torch.nn.Module) -> None:
    del module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class PositionInputs:
    """Everything a single Qwen3.5 MoE layer needs besides hidden states."""

    cos: torch.Tensor
    sin: torch.Tensor
    text_pos: torch.Tensor      # (1, S) flat text position ids for self_attn
    causal_mask: Any            # 4D mask or None (full-attention layers)
    linear_attn_mask: Any       # 2D mask or None (linear-attention layers)


def build_position_inputs(
    seq_len: int,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    text_config: Any,
    attention_mask_2d: torch.Tensor | None = None,
    position_ids_3d: torch.Tensor | None = None,
) -> PositionInputs:
    """Construct the rotary + mask trio used by every layer in this sample.

    ``attention_mask_2d`` is the standard ``[B, S]`` 0/1 padding mask.
    Linear-attention layers consume the 2D mask directly (or ``None`` if the
    sample has no padding); full-attention layers consume the 4D causal mask
    that ``transformers.masking_utils.create_causal_mask`` builds.

    When ``position_ids_3d`` is provided (shape ``(3, B, S)``), rotary cos/sin
    are computed from it via M-RoPE; the layer's ``position_ids`` arg uses
    the width axis (``position_ids_3d[2]``), which equals plain 1D positions
    inside text-only spans of the sample.
    """
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTextRotaryEmbedding,
    )

    device = torch.device(device)
    if attention_mask_2d is None:
        attention_mask_2d = torch.ones(
            batch_size, seq_len, dtype=torch.long, device=device
        )
    else:
        attention_mask_2d = attention_mask_2d.to(device=device, dtype=torch.long)

    rotary = Qwen3_5MoeTextRotaryEmbedding(text_config).to(device)
    dummy = torch.zeros(batch_size, seq_len, text_config.hidden_size, dtype=dtype, device=device)
    if position_ids_3d is not None:
        position_ids_3d = position_ids_3d.to(device=device, dtype=torch.long)
        cos, sin = rotary(dummy, position_ids_3d)
        text_pos = position_ids_3d[2]
    else:
        text_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        cos, sin = rotary(dummy, text_pos)
    del rotary, dummy

    causal_mask = create_causal_mask(
        config=text_config,
        inputs_embeds=torch.zeros(
            batch_size, seq_len, text_config.hidden_size, dtype=dtype, device=device
        ),
        attention_mask=attention_mask_2d,
        past_key_values=None,
        position_ids=text_pos,
    )

    linear_attn_mask: Any = attention_mask_2d
    if torch.all(attention_mask_2d == 1):
        linear_attn_mask = None

    return PositionInputs(
        cos=cos,
        sin=sin,
        text_pos=text_pos,
        causal_mask=causal_mask,
        linear_attn_mask=linear_attn_mask,
    )


# ====================================================================
# orchestrator
# ====================================================================


@dataclass
class StreamSample:
    """One tokenised sample as the streamer needs it.

    VL fields (``image_grid_thw`` / ``mm_token_type_ids`` / ``position_ids_3d``)
    are populated for VL samples; pure-text samples leave them as ``None``.

    Pixel data follows one of two patterns:

    * **eager** — ``pixel_values`` holds the per-sample patch tensor, ready to
      consume directly. ``image_paths`` and ``teacher_dir`` are ``None``.
      Used when the caller already built ``pixel_values`` (e.g. via
      ``build_vl_inputs``).
    * **lazy** — ``pixel_values`` is ``None``; ``image_paths`` + ``teacher_dir``
      let the streamer's embed pass call ``build_pixel_values(...)`` just
      before forwarding the microbatch, then discard. Avoids holding TB-scale
      pixel tensors in CPU RAM when all N samples are materialised upfront.

    The streamer's embedding pass routes each sample down the right path.
    """

    sid: str
    input_ids: torch.Tensor      # (S,) int64 on CPU
    attention_mask: torch.Tensor  # (S,) int64 on CPU
    pixel_values: torch.Tensor | None = None       # (n_patches, patch_dim) on CPU
    image_grid_thw: torch.Tensor | None = None     # (n_images, 3) int64 on CPU
    mm_token_type_ids: torch.Tensor | None = None  # (S,) int64 on CPU
    position_ids_3d: torch.Tensor | None = None    # (3, S) int64 on CPU
    image_paths: list[str] | None = None           # lazy-mode: image file paths
    teacher_dir: str | None = None                 # lazy-mode: for image_processor lookup


def _scratch_chunk_path(scratch_dir: Path, chunk_id: int) -> Path:
    """Path of the per-chunk scratch file holding ``{sid}.hidden`` keys.

    Mirrors :func:`moe_prune_distill.distill.teacher_cache._chunk_filename`'s
    naming style so chunk ids line up between scratch and cache.
    """
    return scratch_dir / f"scratch_chunk_{int(chunk_id)}.cur.safetensors"


def _topk_count_per_layer(
    router_logits: torch.Tensor, top_k: int, num_experts: int
) -> torch.Tensor:
    """Return a length-``num_experts`` long tensor of top-k selection counts.

    Stays on the same device as ``router_logits`` so the caller can keep
    accumulating without round-tripping to CPU per sample. Float promotion
    happens internally so half-precision logits don't break ``topk``.
    """
    k = min(top_k, num_experts)
    flat = router_logits.reshape(-1, router_logits.shape[-1]).float()
    _, topi = flat.topk(k=k, dim=-1)
    return torch.bincount(topi.reshape(-1), minlength=num_experts).to(torch.long)


class LayerStreamer:
    """Run the teacher layer-by-layer over a list of samples.

    Outputs per layer:

    * ``self.topk_counts[layer, expert]`` — long tensor of top-k selection counts
    * if ``cache_dir`` is set, an appended teacher cache safetensors per sample
      with hidden states (and optionally router logits) for the layers in
      ``cache_layers``. Files are byte-for-byte compatible with
      :func:`save_sample_cache`.

    Caller is expected to feed ``samples`` already tokenised and truncated.
    """

    def __init__(
        self,
        *,
        text_config: Any,
        teacher_dir: str | Path,
        weight_map: dict[str, str],
        samples: Sequence[StreamSample],
        scratch_dir: str | Path,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: str | Path | None = None,
        cache_layers: Iterable[int] | None = None,
        cache_dtype: torch.dtype = torch.float16,
        cache_router_logits: bool = True,
        chunk_size: int = 500,
        batch_size: int = 1,
        log: logging.Logger | None = None,
        full_config: Any | None = None,
        image_token_id: int | None = None,
    ) -> None:
        self.text_config = text_config
        # Full multimodal config (Qwen3_5MoeConfig) — required when any
        # sample carries pixel data so we can instantiate the vision tower.
        self.full_config = full_config
        self.image_token_id = image_token_id
        self.teacher_dir = Path(teacher_dir)
        self.weight_map = dict(weight_map)
        self.samples = list(samples)
        self.scratch_dir = Path(scratch_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_layers = set(cache_layers) if cache_layers is not None else set()
        self.cache_dtype = cache_dtype
        self.cache_router_logits = cache_router_logits
        self.chunk_size = int(chunk_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = int(batch_size)
        self.log = log or logging.getLogger("moe_prune_distill.layer_streamer")

        self.num_layers = int(text_config.num_hidden_layers)
        self.num_experts = int(text_config.num_experts)
        self.top_k = int(getattr(text_config, "num_experts_per_tok", 1) or 1)
        # Long counts buffer kept on-device; flushed to CPU once per layer pass.
        self._topk_counts_dev = torch.zeros(
            self.num_layers, self.num_experts, dtype=torch.long, device=self.device
        )
        self.topk_counts = torch.zeros(self.num_layers, self.num_experts, dtype=torch.long)
        # Lazy rotary table (built once on the first layer pass).
        self._rotary_cos: torch.Tensor | None = None
        self._rotary_sin: torch.Tensor | None = None
        self._rotary_max_seq: int = 0

        self._writer: BatchedCacheWriter | None = None
        if self.cache_dir is not None and self.cache_layers and self.samples:
            self._writer = BatchedCacheWriter(
                self.cache_dir,
                sample_ids=[s.sid for s in self.samples],
                cache_layers=self.cache_layers,
                num_experts=self.num_experts,
                cache_dtype=self.cache_dtype,
                cache_router_logits=self.cache_router_logits,
                chunk_size=self.chunk_size,
            )

        # Chunk layout — same math as BatchedCacheWriter so scratch chunks and
        # cache chunks share boundaries. ``_chunk_to_samples[j]`` lists every
        # sample that lands in chunk j, in the original sample-list order.
        if self._writer is not None:
            self._sid_to_chunk = dict(self._writer.sid_to_chunk)
            self._num_chunks = self._writer.num_chunks
        else:
            self._sid_to_chunk = {
                s.sid: idx // self.chunk_size for idx, s in enumerate(self.samples)
            }
            self._num_chunks = (
                (len(self.samples) + self.chunk_size - 1) // self.chunk_size
                if self.samples
                else 0
            )
        self._chunk_to_samples: list[list[StreamSample]] = [
            [] for _ in range(self._num_chunks)
        ]
        for s in self.samples:
            self._chunk_to_samples[self._sid_to_chunk[s.sid]].append(s)

        # Per-sid 3D positions (only populated for VL samples). Layer pass
        # reads from this to build M-RoPE cos/sin; embed pass forwards them
        # to BatchedCacheWriter.add_meta so layerwise training can resume
        # from block 0 without re-computing positions.
        self._pos3d_per_sid: dict[str, torch.Tensor] = {
            s.sid: s.position_ids_3d
            for s in self.samples
            if s.position_ids_3d is not None
        }
        self._has_vl = any(
            s.pixel_values is not None or s.image_paths for s in self.samples
        )
        if self._has_vl and self.full_config is None:
            raise ValueError(
                "samples include VL data but full_config is None; "
                "construct LayerStreamer with full_config=Qwen3_5MoeConfig(...)"
            )

    # ---- one-shot helpers ----

    def _ensure_rotary_table(self, seq_len: int) -> None:
        """Build (or rebuild) a cos/sin table large enough for ``seq_len``.

        Hoisted out of the per-sample loop so we don't reconstruct
        ``Qwen3_5MoeTextRotaryEmbedding`` 120k+ times per run. The table is
        sized to the largest sample we see and slice-indexed per sample.
        """
        if self._rotary_cos is not None and self._rotary_max_seq >= seq_len:
            return
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextRotaryEmbedding,
        )

        rotary = Qwen3_5MoeTextRotaryEmbedding(self.text_config).to(self.device)
        text_pos = torch.arange(seq_len, device=self.device).unsqueeze(0)
        dummy = torch.zeros(
            1, seq_len, self.text_config.hidden_size, dtype=self.dtype, device=self.device
        )
        cos, sin = rotary(dummy, text_pos)
        self._rotary_cos = cos.detach()
        self._rotary_sin = sin.detach()
        self._rotary_max_seq = seq_len
        del rotary, dummy, text_pos

    def _rotary_for_3d(self, position_ids_3d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cos/sin for a (3, B, S) M-RoPE position tensor.

        ``Qwen3_5MoeTextRotaryEmbedding.forward`` accepts a 3D ``position_ids``
        tensor directly; we just instantiate it on demand for each VL batch
        rather than reusing the 1D-positions table.
        """
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextRotaryEmbedding,
        )

        rotary = Qwen3_5MoeTextRotaryEmbedding(self.text_config).to(self.device)
        # rotary expects shape-only x; build a tiny dummy at (B, S, hidden).
        B = int(position_ids_3d.shape[1])
        S = int(position_ids_3d.shape[2])
        dummy = torch.empty(
            B, S, self.text_config.hidden_size, dtype=self.dtype, device=self.device
        )
        cos, sin = rotary(dummy, position_ids_3d.to(self.device))
        del rotary, dummy
        return cos.detach(), sin.detach()

    def _position_inputs(
        self,
        seq_len: int,
        attention_mask_2d: torch.Tensor,
        position_ids_3d: torch.Tensor | None = None,
    ) -> PositionInputs:
        """Position trio for a (possibly batched) forward.

        ``attention_mask_2d`` may have any batch size ``B`` (1 or more).
        Rotary cos/sin are sliced from the streamer-level cached table at
        shape ``[1, S, head_dim]`` — broadcast across the batch dim by the
        layer's ``apply_rotary_pos_emb``. The 4D causal mask is built at
        the actual batch shape via ``create_causal_mask``.

        When ``position_ids_3d`` is provided (shape ``(3, B, S)``), rotary
        cos/sin are computed per-batch from the 3D positions (M-RoPE) and
        the layer's ``position_ids`` argument carries the (B, S) text axis
        (axis 2 of the 3D positions, which equals plain 1D positions for
        text-only spans inside the sample).
        """
        from transformers.masking_utils import create_causal_mask

        attention_mask_2d = attention_mask_2d.to(device=self.device, dtype=torch.long)
        B = int(attention_mask_2d.shape[0])

        if position_ids_3d is not None:
            position_ids_3d = position_ids_3d.to(device=self.device, dtype=torch.long)
            cos, sin = self._rotary_for_3d(position_ids_3d)
            text_pos = position_ids_3d[2]  # width axis stays monotonic in text spans
        else:
            self._ensure_rotary_table(seq_len)
            cos = self._rotary_cos[:, :seq_len]
            sin = self._rotary_sin[:, :seq_len]
            text_pos = (
                torch.arange(seq_len, device=self.device).unsqueeze(0).expand(B, -1)
            )

        # ``inputs_embeds`` is only used by ``create_causal_mask`` for shape
        # info; an empty allocation is fine and saves the zero-init cost.
        inputs_embeds_dummy = torch.empty(
            B, seq_len, self.text_config.hidden_size, dtype=self.dtype, device=self.device
        )
        causal_mask = create_causal_mask(
            config=self.text_config,
            inputs_embeds=inputs_embeds_dummy,
            attention_mask=attention_mask_2d,
            past_key_values=None,
            position_ids=text_pos,
        )

        linear_attn_mask: Any = attention_mask_2d
        if torch.all(attention_mask_2d == 1):
            linear_attn_mask = None

        return PositionInputs(
            cos=cos,
            sin=sin,
            text_pos=text_pos,
            causal_mask=causal_mask,
            linear_attn_mask=linear_attn_mask,
        )

    def _embed_pass(self) -> None:
        self.log.info("embedding pass: loading embed_tokens to %s", self.device)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        embed = load_embedding_to_gpu(
            self.text_config, self.weight_map, self.teacher_dir, self.device, self.dtype
        )
        visual = None
        if self._has_vl:
            self.log.info("embedding pass: loading vision tower to %s", self.device)
            visual = load_vision_tower_to_gpu(
                self.full_config,
                self.weight_map,
                self.teacher_dir,
                self.device,
                self.dtype,
            )
            merge_size = int(visual.spatial_merge_size)
            if self.image_token_id is None:
                # Fall back to the config field; both attribute names exist
                # depending on transformers version.
                self.image_token_id = int(
                    getattr(self.full_config, "image_token_id", None)
                    or getattr(self.full_config, "image_token_index", 248056)
                )

        with torch.inference_mode():
            for chunk_id, chunk_samples in enumerate(self._chunk_to_samples):
                if not chunk_samples:
                    continue
                buf: dict[str, torch.Tensor] = {}
                for s in chunk_samples:
                    ids = s.input_ids.to(self.device, dtype=torch.long).unsqueeze(0)
                    h = embed(ids).squeeze(0)  # (S, hidden)

                    # Resolve pixel data: eager (s.pixel_values) or lazy
                    # (s.image_paths -> build_pixel_values just-in-time and
                    # release before the next sample so RAM stays bounded).
                    pixel_values = s.pixel_values
                    if pixel_values is None and s.image_paths and visual is not None:
                        from moe_prune_distill.data.vl_processor import build_pixel_values

                        pixel_values = build_pixel_values(
                            s.image_paths, s.teacher_dir or str(self.teacher_dir)
                        )
                        if pixel_values is None:
                            self.log.warning(
                                "embed pass: lazy build_pixel_values failed for sid=%s; "
                                "embedding as text-only",
                                s.sid,
                            )

                    if pixel_values is not None and visual is not None:
                        pv = pixel_values.to(self.device, dtype=visual.dtype)
                        gthw = s.image_grid_thw.to(self.device, dtype=torch.long)
                        vision_out = visual(pv, grid_thw=gthw, return_dict=True)
                        image_embeds = vision_out.pooler_output.to(h.dtype)
                        # Replace each <|image_pad|> token with its
                        # corresponding flattened patch row.
                        image_mask = (s.input_ids.to(self.device) == self.image_token_id)
                        n_image_tokens = int(image_mask.sum().item())
                        if n_image_tokens != image_embeds.shape[0]:
                            self.log.warning(
                                "vl mismatch sid=%s: %d image tokens vs %d patches; "
                                "skipping scatter (sample will train as text-only)",
                                s.sid,
                                n_image_tokens,
                                image_embeds.shape[0],
                            )
                        else:
                            h = h.clone()
                            h[image_mask] = image_embeds
                        del pv, vision_out, image_embeds, pixel_values

                    buf[f"{s.sid}.hidden"] = h.to(self.dtype).detach().cpu().contiguous()
                    if self._writer is not None:
                        # For VL samples we ship the merged inputs_embeds and
                        # 3D positions through to the cache so layerwise
                        # training can skip the vision tower at training time.
                        is_vl = (
                            s.pixel_values is not None or bool(s.image_paths)
                        )
                        self._writer.add_meta(
                            s.sid,
                            s.input_ids,
                            s.attention_mask,
                            inputs_embeds=h.detach() if is_vl else None,
                            position_ids_3d=(
                                s.position_ids_3d if is_vl else None
                            ),
                        )
                cur = _scratch_chunk_path(self.scratch_dir, chunk_id)
                tmp = cur.with_suffix(cur.suffix + ".tmp")
                save_file(buf, str(tmp))
                tmp.replace(cur)
                buf.clear()
        unload_module(embed)
        if visual is not None:
            unload_module(visual)

    def _iter_chunk_batches(
        self, chunk_samples: Sequence[StreamSample]
    ) -> Iterable[list[StreamSample]]:
        """Yield length-sorted batches scoped to one chunk's samples.

        At ``batch_size == 1`` this preserves the chunk's sample order, so
        single-sample runs remain bit-identical to the legacy path. At
        ``batch_size > 1`` we sort by actual sequence length (descending)
        within the chunk and slice: the first batch is the largest, so an
        OOM caused by an over-eager ``batch_size`` shows up immediately.

        VL and text-only samples are never mixed in the same batch — VL
        batches use M-RoPE 3D positions while text batches use the cached
        1D table, and the layer's rotary table can only be one or the other
        at a time.
        """
        if self.batch_size <= 1:
            for s in chunk_samples:
                yield [s]
            return
        vl_samples = [s for s in chunk_samples if s.position_ids_3d is not None]
        text_samples = [s for s in chunk_samples if s.position_ids_3d is None]
        for group in (text_samples, vl_samples):
            if not group:
                continue
            ordered = sorted(
                group,
                key=lambda s: int(s.attention_mask.sum().item()),
                reverse=True,
            )
            for i in range(0, len(ordered), self.batch_size):
                yield ordered[i : i + self.batch_size]

    def _pad_batch_from_buffer(
        self,
        batch: Sequence[StreamSample],
        cur_buf: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor | None]:
        """Right-pad each sample's chunk-resident hidden to the batch max.

        Returns ``(h_in [B, S_pad, H], attention_mask [B, S_pad], lengths,
        position_ids_3d [3, B, S_pad] | None)`` where ``lengths[b]`` is the
        actual (un-padded) seq len for sample b. Pad positions hold zeros in
        ``h_in``, 0 in ``attention_mask``, and the last in-range value in the
        3D position table (so rotary doesn't OOB on the padded suffix).

        ``position_ids_3d`` is ``None`` when the batch is text-only.
        """
        hiddens: list[torch.Tensor] = []
        lengths: list[int] = []
        for s in batch:
            h = cur_buf[s.sid].to(device=self.device, dtype=self.dtype)
            hiddens.append(h)
            lengths.append(int(h.shape[0]))
        max_len = max(lengths)
        hidden_dim = hiddens[0].shape[1]
        B = len(batch)
        h_in = torch.zeros(B, max_len, hidden_dim, dtype=self.dtype, device=self.device)
        attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for b, (h, length) in enumerate(zip(hiddens, lengths)):
            h_in[b, :length] = h
            attn_mask[b, :length] = 1

        pos3d_batch: torch.Tensor | None = None
        if any(s.position_ids_3d is not None for s in batch):
            pos3d_batch = torch.zeros(3, B, max_len, dtype=torch.long, device=self.device)
            for b, (s, length) in enumerate(zip(batch, lengths)):
                if s.position_ids_3d is not None:
                    p = s.position_ids_3d.to(device=self.device, dtype=torch.long)
                    pos3d_batch[:, b, :length] = p[:, :length]
                    if length < max_len:
                        # Continue from the last in-range position so rotary
                        # doesn't see a sudden zero on padded slots.
                        tail = p[:, length - 1] + 1
                        pos3d_batch[:, b, length:] = tail.unsqueeze(-1)
                else:
                    # Text-only sample inside a VL batch — fall back to plain
                    # 1D positions broadcast across all three axes.
                    pos3d_batch[:, b, :length] = torch.arange(
                        length, device=self.device
                    ).unsqueeze(0)
        return h_in, attn_mask, lengths, pos3d_batch

    def _layer_pass(self, layer_idx: int) -> None:
        self.log.info(
            "layer %d/%d (%s): loading",
            layer_idx,
            self.num_layers,
            self.text_config.layer_types[layer_idx],
        )
        layer = load_layer_to_gpu(
            self.text_config,
            layer_idx,
            self.weight_map,
            self.teacher_dir,
            self.device,
            self.dtype,
        )

        captured: list[torch.Tensor] = []

        def hook(_m, _inp, out):
            captured.append(out[0].detach())

        handle = layer.mlp.gate.register_forward_hook(hook)
        cache_this_layer = layer_idx in self.cache_layers and self.cache_dir is not None
        is_linear_attn = self.text_config.layer_types[layer_idx] == "linear_attention"

        with torch.inference_mode():
            for chunk_id, chunk_samples in enumerate(self._chunk_to_samples):
                if not chunk_samples:
                    continue
                cur_path = _scratch_chunk_path(self.scratch_dir, chunk_id)
                cur_raw = load_file(str(cur_path))
                cur_buf: dict[str, torch.Tensor] = {}
                for k, v in cur_raw.items():
                    if k.endswith(".hidden"):
                        cur_buf[k[: -len(".hidden")]] = v
                next_buf: dict[str, torch.Tensor] = {}

                for batch in self._iter_chunk_batches(chunk_samples):
                    h_in, attn_mask, lengths, pos3d_batch = self._pad_batch_from_buffer(
                        batch, cur_buf
                    )
                    B, S, _ = h_in.shape
                    pos = self._position_inputs(S, attn_mask, position_ids_3d=pos3d_batch)
                    layer_mask = (
                        pos.linear_attn_mask if is_linear_attn else pos.causal_mask
                    )

                    captured.clear()
                    h_out = layer(
                        h_in,
                        position_embeddings=(pos.cos, pos.sin),
                        attention_mask=layer_mask,
                        position_ids=pos.text_pos,
                        past_key_values=None,
                    )
                    if not captured:
                        raise RuntimeError(
                            f"router hook did not fire on layer {layer_idx}"
                        )

                    # Router hook outputs flat [B*S, E]; reshape so we can
                    # mask-and-slice per sample.
                    router_logits_flat = captured[0]
                    router_logits = router_logits_flat.view(B, S, self.num_experts)

                    # Top-k counting on valid (non-padding) positions only —
                    # otherwise padded tokens inflate the per-expert counts and
                    # bias the prune step.
                    valid_mask = attn_mask.bool()
                    valid_router = router_logits[valid_mask]
                    self._topk_counts_dev[layer_idx] += _topk_count_per_layer(
                        valid_router, self.top_k, self.num_experts
                    )

                    for b, s in enumerate(batch):
                        length = lengths[b]
                        h_out_b = h_out[b, :length]
                        next_buf[f"{s.sid}.hidden"] = (
                            h_out_b.to(self.dtype).detach().cpu().contiguous()
                        )
                        if cache_this_layer and self._writer is not None:
                            router_b = (
                                router_logits[b, :length]
                                if self.cache_router_logits
                                else None
                            )
                            self._writer.add_layer_sample(
                                s.sid, layer_idx, h_out_b, router_b
                            )

                    del h_in, h_out, router_logits, router_logits_flat, pos

                # Atomic write: tmp → cur. Old cur is replaced in one step,
                # which is safe because we already drained it into cur_buf.
                cur_buf.clear()
                tmp = cur_path.with_suffix(cur_path.suffix + ".tmp")
                save_file(next_buf, str(tmp))
                next_buf.clear()
                tmp.replace(cur_path)

                if cache_this_layer and self._writer is not None:
                    # Free this chunk's ~17 GB of cache RAM before loading the
                    # next chunk, instead of holding every chunk for the layer.
                    self._writer.flush_chunk(layer_idx, chunk_id)

        handle.remove()
        # One CPU sync per layer pass instead of per (sample, layer).
        self.topk_counts[layer_idx] = self._topk_counts_dev[layer_idx].detach().cpu()
        unload_module(layer)
        if cache_this_layer and self._writer is not None:
            # Safety drain in case any chunk was missed (no-op on the happy path).
            self._writer.flush_layer(layer_idx)

    def _cleanup_scratch(self) -> None:
        if not self.scratch_dir.is_dir():
            return
        for p in self.scratch_dir.glob("scratch_chunk_*.safetensors"):
            try:
                p.unlink()
            except OSError:
                pass
        for p in self.scratch_dir.glob("scratch_chunk_*.safetensors.tmp"):
            try:
                p.unlink()
            except OSError:
                pass

    # ---- main entry ----

    def run(self) -> torch.Tensor:
        if not self.samples:
            return self.topk_counts
        self._embed_pass()
        try:
            for i in range(self.num_layers):
                self._layer_pass(i)
            if self._writer is not None:
                self._writer.finalize()
        finally:
            self._cleanup_scratch()
        return self.topk_counts


__all__ = [
    "LayerStreamer",
    "PositionInputs",
    "StreamSample",
    "build_position_inputs",
    "keys_for_layer",
    "load_embedding_to_gpu",
    "load_layer_to_gpu",
    "read_shard_index",
    "unload_module",
]
