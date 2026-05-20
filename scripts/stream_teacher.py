"""Layer-streamed teacher pass: router stats + teacher cache in one shot.

Equivalent to running ``collect_router_stats.py`` followed by
``cache_teacher.py`` on the same dataset, but iterates the loop the other
way around: each transformer block is loaded to GPU exactly once and every
sample is forwarded through it before the block is released. Per-sample
hidden states live in a scratch directory between layers.

Cache output is the **v2 batched layout** (per-layer, sample-chunked
safetensors + ``cache_index.json``) — see
:mod:`moe_prune_distill.distill.teacher_cache` for the schema. Every
downstream reader (``DistillJsonlDataset``, ``layerwise_trainer``) goes
through ``load_sample_cache`` which auto-dispatches between v2 and the
legacy per-sample layout. ``router_stats.json`` is unchanged.

CLI:

    python -m scripts.stream_teacher --config configs/example.yaml \
        --scratch-dir ./cache/scratch_hidden \
        --chunk-size 500 \
        --router-stats-out ./outputs/router_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextConfig

from moe_prune_distill.adapters import detect_adapter
from moe_prune_distill.config import load_config
from moe_prune_distill.data.dataset import JsonlSFTDataset
from moe_prune_distill.distill.layer_streamer import (
    LayerStreamer,
    StreamSample,
    read_shard_index,
)
from moe_prune_distill.distill.router_stats import write_router_stats
from moe_prune_distill.distill.teacher_cache import (
    cache_exists,
    cache_layers_for,
    parse_cache_dtype,
)
from moe_prune_distill.utils.logging import get_logger


def _parse_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"unsupported compute dtype: {name}")


def _as_long_tensor(x: object) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(torch.long)
    return torch.tensor(x, dtype=torch.long)


def _build_samples(
    ds: JsonlSFTDataset, max_samples: int | None, log, teacher_dir: Path
) -> list[StreamSample]:
    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    out: list[StreamSample] = []
    dropped = 0
    for i in tqdm(range(n), desc="building samples (CPU: PIL+image_proc+rope3d)", unit="smp"):
        s = ds[i]
        attn = s["attention_mask"]
        attn_sum = (
            int(attn.sum().item()) if isinstance(attn, torch.Tensor) else int(sum(attn))
        )
        # JsonlSFTDataset's VL fallback for failed vl_processor returns a
        # 1-token row with attention_mask=[0]. Forwarding such a sample
        # through 40 layers wastes GPU time and pollutes the cache index;
        # drop it here. Pure-text samples always have at least one valid
        # token so sum > 0.
        if attn_sum == 0:
            dropped += 1
            continue
        # VL fields are only present when the underlying sample carries images;
        # JsonlSFTDataset's VL path returns pre-computed pixel_values /
        # image_grid_thw / mm_token_type_ids / position_ids_3d alongside the
        # expanded input_ids.
        #
        # In lazy-pixel mode (vl_lazy_pixels=True), ``pixel_values`` is absent
        # and ``image_paths`` is passed through instead — LayerStreamer calls
        # build_pixel_values() per microbatch at embed time so the per-sample
        # patch tensor never lives in CPU RAM.
        pixel_values = s.get("pixel_values")
        image_paths = s.get("image_paths")
        image_grid_thw = s.get("image_grid_thw")
        mm_type = s.get("mm_token_type_ids")
        pos3d = s.get("position_ids_3d")
        out.append(
            StreamSample(
                sid=s["id"],
                input_ids=_as_long_tensor(s["input_ids"]),
                attention_mask=_as_long_tensor(s["attention_mask"]),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=_as_long_tensor(mm_type) if mm_type is not None else None,
                position_ids_3d=_as_long_tensor(pos3d) if pos3d is not None else None,
                image_paths=image_paths,
                teacher_dir=str(teacher_dir) if image_paths else None,
            )
        )
    if dropped:
        log.warning(
            "_build_samples: dropped %d/%d samples (failed vl_processor / empty rows)",
            dropped,
            n,
        )
    return out


def main() -> None:
    log = get_logger()
    p = argparse.ArgumentParser(description="Stream teacher layer-by-layer")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--scratch-dir", type=str, default="./cache/scratch_hidden")
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="forward-batch size per layer pass. Higher = better GPU utilisation "
             "but more activation memory; 1 reproduces the legacy per-sample loop.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="samples per cache_layer chunk file (v2 layout). Lower if RAM-tight; "
             "host RAM peak per active layer ≈ chunk_size × seq × hidden × 2 bytes.",
    )
    p.add_argument(
        "--router-stats-out",
        type=str,
        default=None,
        help="output path for router_stats.json (defaults to <student_dir>/../router_stats.json)",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip samples whose teacher cache already exists (per cache_index.json)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16")
    args = p.parse_args()

    app = load_config(args.config)
    teacher_dir = Path(app.download.local_dir).resolve()
    cfg_path = teacher_dir / "config.json"
    hf = json.loads(cfg_path.read_text(encoding="utf-8"))
    adapter = detect_adapter(hf)
    num_layers = adapter.get_num_layers(hf)
    num_experts = adapter.get_num_experts(hf)
    top_k = adapter.get_num_experts_per_tok(hf)

    if "text_config" not in hf:
        raise SystemExit(
            "stream_teacher currently requires a Qwen3.5 MoE teacher with text_config "
            "(found legacy config); use the cache_teacher.py + collect_router_stats.py "
            "pair instead."
        )
    text_config = Qwen3_5MoeTextConfig(**hf["text_config"])
    # Full multimodal config — required so the streamer can instantiate the
    # vision tower for VL samples. ``Qwen3_5MoeConfig.__init__`` accepts the
    # full HF config dict (with both text_config and vision_config sub-dicts).
    full_config = Qwen3_5MoeConfig(**hf)
    image_token_id = int(hf.get("image_token_id", 248056))

    log.info(
        "Teacher: layers=%d experts=%d top_k=%d",
        num_layers,
        num_experts,
        top_k,
    )

    tok = AutoTokenizer.from_pretrained(str(teacher_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = JsonlSFTDataset(
        app.data.train_file,
        tok,
        max_seq_len=app.data.max_seq_len,
        max_samples=app.data.max_samples,
        teacher_dir=teacher_dir,
        vl_lazy_pixels=True,
    )
    samples = _build_samples(ds, args.max_samples, log, teacher_dir)
    n_total_samples = len(samples)

    cache_dir: Path | None = None
    cache_layers: list[int] = []
    if app.teacher_cache.enabled:
        cache_dir = Path(app.teacher_cache.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_layers = cache_layers_for(
            num_layers,
            app.teacher_cache.cache_layers,
            app.teacher_cache.cache_layer_interval,
        )
        if args.skip_existing:
            samples = [s for s in samples if not cache_exists(cache_dir, s.sid)]
        log.info(
            "Teacher cache enabled: %d layers, %s, %d samples to process",
            len(cache_layers),
            cache_dir,
            len(samples),
        )

    cache_dtype = parse_cache_dtype(app.teacher_cache.cache_dtype)
    compute_dtype = _parse_dtype(args.dtype)
    weight_map = read_shard_index(teacher_dir)

    streamer = LayerStreamer(
        text_config=text_config,
        teacher_dir=teacher_dir,
        weight_map=weight_map,
        samples=samples,
        scratch_dir=Path(args.scratch_dir).resolve(),
        device=args.device if torch.cuda.is_available() else "cpu",
        dtype=compute_dtype,
        cache_dir=cache_dir,
        cache_layers=cache_layers,
        cache_dtype=cache_dtype,
        cache_router_logits=app.teacher_cache.cache_router_logits,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        log=log,
        full_config=full_config,
        image_token_id=image_token_id,
    )
    counts = streamer.run()

    # Decide whether to (over)write router_stats.json. If --skip-existing
    # filtered out every sample, ``counts`` is all zeros — overwriting a
    # prior router_stats with that would silently corrupt the prune step.
    out_path = (
        Path(args.router_stats_out).resolve()
        if args.router_stats_out
        else Path(app.prune.student_dir).resolve().parent / "router_stats.json"
    )
    if args.skip_existing and not samples and out_path.is_file():
        log.info(
            "All samples skipped (existing cache); leaving %s untouched.",
            out_path,
        )
    else:
        write_router_stats(
            out_path,
            model_id=app.download.model_id,
            num_samples=len(samples) if samples else n_total_samples,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            counts=counts,
            target_num_experts=app.prune.target_num_experts,
        )
        log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
