"""Layer-wise distillation training (block-by-block).

Trains the pruned student one ``BlockSpec`` at a time against the existing
teacher cache (``cache/teacher_hiddens/{sid}.safetensors``). Only one block's
parameters live on the GPU at any time. Blocks are independent — input and
output supervision both come from the cache — so a run can be resumed by
re-running with ``--blocks`` to skip already-finished blocks.

When all blocks finish, layer snapshots under ``_layers/`` are merged into
the original student shard layout and written to
``train.layerwise.output_dir`` (default ``./models/student_layerwise``).

CLI:

    python -m scripts.train_layerwise --config configs/example.yaml
    python -m scripts.train_layerwise --config configs/example.yaml \
        --blocks 0,1,2 --max-steps-per-block 1000
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextConfig

from moe_prune_distill.adapters import detect_adapter
from moe_prune_distill.config import load_config
from moe_prune_distill.data.dataset import is_val_id
from moe_prune_distill.distill.layer_streamer import read_shard_index
from moe_prune_distill.distill.layerwise_trainer import (
    BlockTrainer,
    TrainerConfig,
    block_done_marker,
    enumerate_blocks,
    merge_layer_updates_into_student,
)
from moe_prune_distill.distill.teacher_cache import (
    cache_exists,
    cache_layers_for,
)
from moe_prune_distill.distill.trainer import load_expert_mapping
from moe_prune_distill.utils.logging import get_logger


def _parse_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _load_sample_ids(
    train_jsonl: Path,
    cache_dir: Path,
    max_samples: int | None,
    val_split: float = 0.0,
) -> tuple[list[str], list[str]]:
    """Read sample ids from train.jsonl, split into (train_ids, val_ids).

    Only ids whose teacher cache exists are returned. The val partition uses
    the same deterministic id-hash as ``JsonlSFTDataset``, so end-to-end and
    layerwise training share their validation set when both run on the same
    ``train.jsonl``.
    """
    train_ids: list[str] = []
    val_ids: list[str] = []
    with train_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = row.get("id")
            if sid is None:
                continue
            sid = str(sid)
            if not cache_exists(cache_dir, sid):
                continue
            total = len(train_ids) + len(val_ids)
            if max_samples is not None and total >= max_samples:
                break
            if val_split > 0 and is_val_id(sid, val_split):
                val_ids.append(sid)
            else:
                train_ids.append(sid)
    return train_ids, val_ids


def main() -> None:
    log = get_logger()
    p = argparse.ArgumentParser(description="Layer-wise (block-wise) distill training")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--blocks", type=str, default=None,
                   help="comma-separated block ids to run (default: all)")
    p.add_argument("--max-steps-per-block", type=int, default=None)
    p.add_argument("--mse-threshold", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="layerwise micro-batch size (default: layerwise.batch_size)")
    p.add_argument("--gradient-accumulation-steps", type=int, default=None,
                   help="layerwise grad accumulation steps (default: layerwise.gradient_accumulation_steps)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--output-dir", type=str, default=None,
                   help="override layerwise.output_dir")
    p.add_argument("--skip-merge", action="store_true",
                   help="train blocks but don't merge into final student dir")
    p.add_argument("--force", action="store_true",
                   help="re-run blocks even if .done marker exists")
    p.add_argument("--gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_true", default=None,
                   help="enable per-layer gradient checkpointing (default: layerwise.gradient_checkpointing)")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false",
                   help="disable per-layer gradient checkpointing")
    p.add_argument("--optimizer", type=str, default=None,
                   help="override layerwise.optimizer (e.g. paged_adamw_8bit)")
    p.add_argument("--use-student-rollout-input", dest="use_student_rollout_input",
                   action="store_true", default=None,
                   help=(
                       "feed the previous block's student output (post-training, "
                       "eval-mode) as the next block's input; loss target stays "
                       "teacher cache. Forces sequential block execution."
                   ))
    p.add_argument("--no-use-student-rollout-input", dest="use_student_rollout_input",
                   action="store_false",
                   help="disable student-rollout input chaining (default).")
    p.add_argument("--lora", dest="lora_enabled", action="store_true", default=None,
                   help=(
                       "enable LoRA + optional 4bit on attention sub-modules "
                       "(see train.layerwise.lora in the YAML)."
                   ))
    p.add_argument("--no-lora", dest="lora_enabled", action="store_false",
                   help="disable LoRA path (full-FT all in-block params, default).")
    args = p.parse_args()

    app = load_config(args.config)
    lw = app.train.layerwise
    if not lw.enabled:
        log.warning("train.layerwise.enabled=false in config; running anyway")

    student_dir = Path(app.prune.student_dir).resolve()
    cache_dir = Path(app.teacher_cache.cache_dir).resolve()
    out_dir = Path(args.output_dir or lw.output_dir).resolve()
    snapshot_dir = out_dir / "_layers"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    rollout_root = snapshot_dir / "_rollout"

    use_rollout = (
        bool(args.use_student_rollout_input)
        if args.use_student_rollout_input is not None
        else bool(lw.use_student_rollout_input)
    )
    lora_enabled = (
        bool(args.lora_enabled)
        if args.lora_enabled is not None
        else bool(lw.lora.enabled)
    )

    if not cache_dir.is_dir() or not any(cache_dir.glob("*.safetensors")):
        raise SystemExit(
            f"teacher cache empty/missing at {cache_dir}; run scripts/stream_teacher.py first"
        )

    cfg_json = json.loads((student_dir / "config.json").read_text(encoding="utf-8"))
    adapter = detect_adapter(cfg_json)
    num_layers = adapter.get_num_layers(cfg_json)
    if "text_config" not in cfg_json:
        raise SystemExit(
            "scripts.train_layerwise currently requires Qwen3.5 MoE text_config; "
            "legacy configs use the existing scripts/train.py path."
        )
    text_config = Qwen3_5MoeTextConfig(**cfg_json["text_config"])

    cache_layers = cache_layers_for(
        num_layers,
        app.teacher_cache.cache_layers,
        app.teacher_cache.cache_layer_interval,
    )
    blocks = enumerate_blocks(num_layers, cache_layers)
    if not blocks:
        raise SystemExit(f"no blocks: cache_layers={cache_layers}, num_layers={num_layers}")

    if args.blocks is None:
        block_ids = [b.block_id for b in blocks]
    else:
        block_ids = [int(x.strip()) for x in args.blocks.split(",") if x.strip()]
    log.info("Cache layers: %s", cache_layers)
    log.info("Blocks: %d total, running %s", len(blocks), block_ids)
    if lora_enabled:
        log.info(
            "LoRA enabled: r=%d alpha=%d dropout=%.3f targets=%s 4bit=%s (%s/%s)",
            lw.lora.r,
            lw.lora.alpha,
            lw.lora.dropout,
            list(lw.lora.target_modules),
            lw.lora.load_in_4bit,
            lw.lora.bnb_4bit_compute_dtype,
            lw.lora.bnb_4bit_quant_type,
        )

    if use_rollout:
        all_ids = [b.block_id for b in blocks]
        sorted_requested = sorted(block_ids)
        if sorted_requested != block_ids:
            raise SystemExit(
                "use_student_rollout_input requires sequential block execution; "
                f"requested order {block_ids} is not ascending"
            )
        # Must be a contiguous prefix starting at the smallest unfinished or
        # requested block. We enforce: requested ids form a contiguous slice
        # of all_ids. (Skipping leading already-done blocks is allowed because
        # each done block has its rollout cached.)
        if sorted_requested:
            i = all_ids.index(sorted_requested[0])
            expected = all_ids[i : i + len(sorted_requested)]
            if expected != sorted_requested:
                raise SystemExit(
                    "use_student_rollout_input requires a contiguous block range; "
                    f"requested {block_ids} is not contiguous in {all_ids}"
                )

    tok = AutoTokenizer.from_pretrained(str(student_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    sample_ids, val_sample_ids = _load_sample_ids(
        Path(app.data.train_file).resolve(),
        cache_dir,
        app.data.max_samples,
        val_split=app.data.val_split,
    )
    if not sample_ids:
        raise SystemExit(f"no usable samples (cache hits) in {app.data.train_file}")
    log.info(
        "Samples with cache: train=%d val=%d (val_split=%.3f)",
        len(sample_ids),
        len(val_sample_ids),
        app.data.val_split,
    )

    surviving_by_layer = load_expert_mapping(student_dir)
    student_weight_map = read_shard_index(student_dir)
    dtype = _parse_dtype(args.dtype)
    device = args.device if torch.cuda.is_available() else "cpu"

    cfg = TrainerConfig(
        max_steps=args.max_steps_per_block or lw.max_steps_per_block,
        mse_threshold=args.mse_threshold or lw.mse_threshold,
        patience=args.patience or lw.patience,
        learning_rate=args.learning_rate or lw.learning_rate,
        optimizer=args.optimizer or lw.optimizer,
        use_router_kl=lw.use_router_kl,
        router_kl_weight=app.train.losses.router_kl,
        router_temperature=app.train.losses.router_kl_temperature,
        save_every_steps=lw.save_every_steps,
        log_every_steps=lw.log_every_steps,
        seed=app.train.seed,
        batch_size=args.batch_size or lw.batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps or lw.gradient_accumulation_steps
        ),
        gradient_checkpointing=(
            lw.gradient_checkpointing if args.gradient_checkpointing is None
            else bool(args.gradient_checkpointing)
        ),
        sso_ns_steps=lw.sso_ns_steps,
        sso_radius_c=lw.sso_radius_c,
        sso_radius_mode=lw.sso_radius_mode,
        sso_msign_dtype=lw.sso_msign_dtype,
        sso_bisect_max_iters=lw.sso_bisect_max_iters,
        sso_bisect_tol=lw.sso_bisect_tol,
        sso_power_iters=lw.sso_power_iters,
        muon_momentum=lw.muon_momentum,
        muon_ns_steps=lw.muon_ns_steps,
        muon_paged_momentum=lw.muon_paged_momentum,
        lr_scheduler_type=lw.lr_scheduler_type,
        min_lr_ratio=lw.min_lr_ratio,
        warmup_ratio=lw.warmup_ratio,
        eval_every_steps=lw.eval_every_steps,
        tensorboard_enabled=app.train.tensorboard.enabled,
        tensorboard_log_dir=(
            str((Path(app.train.tensorboard.log_dir) / "layerwise").resolve())
            if app.train.tensorboard.enabled
            else None
        ),
        use_student_rollout_input=use_rollout,
        rollout_root=rollout_root if use_rollout else None,
        lora_enabled=lora_enabled,
        lora_r=lw.lora.r,
        lora_alpha=lw.lora.alpha,
        lora_dropout=lw.lora.dropout,
        lora_target_modules=tuple(lw.lora.target_modules),
        lora_load_in_4bit=lw.lora.load_in_4bit,
        lora_compute_dtype=lw.lora.bnb_4bit_compute_dtype,
        lora_quant_type=lw.lora.bnb_4bit_quant_type,
    )

    for block in blocks:
        if block.block_id not in block_ids:
            continue
        marker = block_done_marker(snapshot_dir, block.block_id)
        if marker.is_file() and not args.force:
            log.info("Block %d already done (%s); skip. Pass --force to retrain.",
                     block.block_id, marker.name)
            continue

        if use_rollout and block.input_layer >= 0:
            # We need block N-1's rollout cache to feed block N's input. If
            # the previous block is "done" but its rollout cache is missing
            # (e.g. a partial run before the flag was turned on), refuse to
            # proceed rather than silently falling back to teacher cache.
            from moe_prune_distill.distill.rollout_cache import (
                block_id_for,
                rollout_index_exists,
            )
            prev_bid = block.block_id - 1
            prev_marker = block_done_marker(snapshot_dir, prev_bid)
            if prev_marker.is_file():
                missing_for: list[str] = []
                if not rollout_index_exists(rollout_root):
                    missing_for = list(sample_ids[:1])
                else:
                    for sid in sample_ids:
                        bid = block_id_for(rollout_root, sid)
                        if bid is None or bid < prev_bid:
                            missing_for.append(sid)
                            if len(missing_for) > 3:
                                break
                if missing_for:
                    raise SystemExit(
                        f"use_student_rollout_input is enabled but block {prev_bid} "
                        f"has no rollout cache at {rollout_root}.\n"
                        f"Re-train block {prev_bid} with --force to regenerate the rollout, "
                        f"or run without --use-student-rollout-input."
                    )
        from dataclasses import replace
        block_cfg = replace(
            cfg,
            train_log_path=str(snapshot_dir / f"block_{block.block_id:03d}_train_log.jsonl"),
            val_log_path=(
                str(snapshot_dir / f"block_{block.block_id:03d}_val_log.jsonl")
                if val_sample_ids
                else None
            ),
            val_sample_ids=tuple(val_sample_ids),
        )
        trainer = BlockTrainer(
            block=block,
            student_dir=student_dir,
            student_text_config=text_config,
            student_weight_map=student_weight_map,
            cache_dir=cache_dir,
            sample_ids=sample_ids,
            snapshot_dir=snapshot_dir,
            device=device,
            dtype=dtype,
            cfg=block_cfg,
            surviving_by_layer=surviving_by_layer,
            log=log,
            total_blocks=len(blocks),
        )
        result = trainer.run()
        log.info(
            "block %03d done  steps=%d  ema_hidden_mse=%.5f",
            block.block_id, result["steps"], result["ema_hidden_mse"],
        )
        del trainer, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.skip_merge:
        log.info("--skip-merge: snapshots in %s, no merge done.", snapshot_dir)
        return

    log.info("Merging layer snapshots into %s", out_dir)
    merge_layer_updates_into_student(student_dir, snapshot_dir, out_dir)
    log.info("Layerwise training complete: %s", out_dir)


if __name__ == "__main__":
    main()
