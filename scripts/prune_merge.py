"""Step 3 alt: prune with **expert merging**.

Drop-in replacement for ``scripts/prune.py`` that, before discarding dropped
experts, computes a per-layer dropped→surviving similarity matrix and folds
the dropped experts' weights into the kept ones via a scaled add:

    W_kept_new[k] = W_kept_orig[k] + alpha * Σ_d w[d, k] * W_dropped[d]

Selection of surviving experts (``first_n`` / ``router_top`` / ``manual``)
reuses the same config knobs as ``scripts.prune``. The output student dir,
``expert_mapping.json``, tokenizer files, and shard layout are all
identical to vanilla prune — only the routed_expert_stack tensor values
differ. ``merge_plan.json`` and ``merge_report.md`` are also written under
the student dir so the mixing weights / neuron stats are auditable after
the fact.

Two strategy families:

* **Macro** (`weight_cosine`, `weight_cosine_of_router`, `cooccur`):
  full-tensor dropped→kept mixing.
* **Neuron-level SwiGLU** (`neuron_swiglu_local`,
  `neuron_swiglu_global`): per-neuron super-vector matching, gate_up_proj
  preserved, only down_proj columns absorbed. See
  ``moe_prune_distill.prune.expert_merge.build_neuron_merge_plan``.

Usage:

    python -m scripts.prune_merge --config configs/example.yaml \\
        [--merge-strategy weight_cosine|weight_cosine_of_router|cooccur \\
                          |neuron_swiglu_local|neuron_swiglu_global] \\
        [--merge-alpha FLOAT] \\
        [--merge-tau FLOAT] \\
        [--neuron-sim-threshold FLOAT] \\
        [--neuron-topk INT] \\
        [--router-stats-path PATH] \\
        [--cooccur-path PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moe_prune_distill.adapters import detect_adapter
from moe_prune_distill.config import load_config
from moe_prune_distill.prune.config_editor import write_student_config
from moe_prune_distill.prune.expert_merge import (
    build_merge_plan,
    build_neuron_merge_plan,
    serialize_merge_plan,
    write_merge_report,
)
from moe_prune_distill.prune.expert_selector import surviving_experts_first_n
from moe_prune_distill.prune.slicer import build_expert_mapping_json, prune_state_dict_sharded
from moe_prune_distill.utils.logging import get_logger
from scripts.prune import copy_tokenizer_assets, resolve_surviving


_NEURON_STRATEGIES = ("neuron_swiglu_local", "neuron_swiglu_global")
_MACRO_STRATEGIES = ("weight_cosine", "weight_cosine_of_router", "cooccur")


def main() -> None:
    log = get_logger()
    p = argparse.ArgumentParser(
        description="Prune MoE experts with weight merging (drops folded into kept)"
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--merge-strategy",
        type=str,
        default="neuron_swiglu_global",
        choices=_MACRO_STRATEGIES + _NEURON_STRATEGIES,
        help=(
            "weight_cosine = cosine sim of flattened gate_up_proj+down_proj per expert, "
            "softmax(sim/tau). weight_cosine_of_router = cosine sim of each expert's row "
            "in the router gate matrix (cheap, reflects how the teacher itself separates "
            "experts at routing time). cooccur = read precomputed router pair counts "
            "(requires scripts/collect_router_cooccur.py output). "
            "neuron_swiglu_local = SwiGLU-aware neuron-level merge with two-stage "
            "(router cosine -> per-pair greedy argmax) match. "
            "neuron_swiglu_global = SwiGLU-aware neuron-level merge with global block-Hungarian "
            "match across all surviving experts in the layer."
        ),
    )
    p.add_argument(
        "--merge-alpha",
        type=float,
        default=0.5,
        help="scale on the merged contribution; 0 disables merge (= vanilla prune).",
    )
    p.add_argument(
        "--merge-tau",
        type=float,
        default=0.1,
        help="softmax temperature on cosine sims (lower = sharper; "
             "applies to weight_cosine and weight_cosine_of_router only).",
    )
    p.add_argument(
        "--neuron-sim-threshold",
        type=float,
        default=0.1,
        help="cosine threshold for neuron_swiglu_*; below this a dropped neuron is discarded "
             "rather than injected (avoids adding contradictory knowledge to the kept expert).",
    )
    p.add_argument(
        "--neuron-topk",
        type=int,
        default=8,
        help="top-K candidate hosts per dropped neuron used to seed block-Hungarian "
             "(neuron_swiglu_global only; ignored for local).",
    )
    p.add_argument(
        "--neuron-expert-chunk",
        type=int,
        default=4,
        help="number of dropped experts processed per matmul tile in neuron_swiglu_global "
             "(controls peak memory; lower if you OOM, raise for speed).",
    )
    p.add_argument(
        "--neuron-hungarian-blocks",
        type=int,
        default=8,
        help="number of row-blocks in the block-Hungarian fallback (neuron_swiglu_global only).",
    )
    p.add_argument(
        "--neuron-hungarian-block-threshold",
        type=int,
        default=800,
        help="rows below this run a single Hungarian; above split into --neuron-hungarian-blocks "
             "blocks (neuron_swiglu_global only).",
    )
    p.add_argument(
        "--router-stats-path",
        type=str,
        default=None,
        help="optional router_stats.json; when set with a neuron_swiglu_* strategy, scales "
             "per-(layer, dropped expert) alpha by activation frequency (mean over dropped = 1).",
    )
    p.add_argument(
        "--scratch-dir",
        type=str,
        default=None,
        help="optional cache dir for neuron_swiglu_global top-K candidates (one .pt per layer); "
             "speeds up re-runs that only adjust --neuron-sim-threshold.",
    )
    p.add_argument(
        "--bucket-dir",
        type=str,
        default=None,
        help="dir for streamed per-layer down_proj contribution buckets "
             "(neuron_swiglu_* only). Default: <student_dir>/.neuron_buckets. "
             "Each layer's bucket is ~1.5 GB on Qwen3.5; streaming caps RAM to one at a time.",
    )
    p.add_argument(
        "--keep-buckets",
        action="store_true",
        help="keep --bucket-dir contents after the slicer pass (default: delete).",
    )
    p.add_argument(
        "--cooccur-path",
        type=str,
        default=None,
        help="path to router_cooccur.json (required when --merge-strategy=cooccur).",
    )
    args = p.parse_args()
    app = load_config(args.config)

    teacher_dir = Path(app.download.local_dir).resolve()
    student_dir = Path(app.prune.student_dir).resolve()
    cfg_path = teacher_dir / "config.json"
    hf = json.loads(cfg_path.read_text(encoding="utf-8"))
    adapter = detect_adapter(hf)
    num_layers = adapter.get_num_layers(hf)
    num_experts = adapter.get_num_experts(hf)

    surviving = resolve_surviving(app, num_layers, num_experts, log)
    if surviving is None:
        # first_n strategy → materialise the implicit list so build_merge_plan
        # has something to derive dropped_per_layer from.
        fallback = surviving_experts_first_n(num_experts, app.prune.target_num_experts)
        surviving = {layer: list(fallback) for layer in range(num_layers)}

    if args.merge_strategy in _NEURON_STRATEGIES:
        log.info(
            "Building NEURON merge plan: strategy=%s alpha=%.3f sim_threshold=%.3f%s",
            args.merge_strategy,
            args.merge_alpha,
            args.neuron_sim_threshold,
            f" top_k={args.neuron_topk}" if args.merge_strategy == "neuron_swiglu_global" else "",
        )
        plan = build_neuron_merge_plan(
            teacher_dir,
            adapter,
            hf,
            surviving,
            strategy=args.merge_strategy,
            alpha=args.merge_alpha,
            sim_threshold=args.neuron_sim_threshold,
            top_k=args.neuron_topk,
            expert_chunk=args.neuron_expert_chunk,
            block_threshold=args.neuron_hungarian_block_threshold,
            hungarian_blocks=args.neuron_hungarian_blocks,
            router_stats_path=args.router_stats_path,
            scratch_dir=args.scratch_dir,
            bucket_dir=(
                args.bucket_dir
                if args.bucket_dir is not None
                else (student_dir / ".neuron_buckets")
            ),
            log=log,
        )
        covered = sorted(
            set(plan.neuron_down_contrib) | set(plan.neuron_down_contrib_paths)
        )
        log.info(
            "neuron merge plan: %d layers covered (each rolling %d dropped → %d kept)%s",
            len(covered),
            max((len(plan.dropped_per_layer.get(l, [])) for l in covered), default=0),
            max((len(plan.surviving_per_layer.get(l, [])) for l in covered), default=0),
            f" [buckets streamed via {args.bucket_dir or (student_dir / '.neuron_buckets')}]"
            if plan.neuron_down_contrib_paths
            else "",
        )
    else:
        log.info(
            "Building MACRO merge plan: strategy=%s alpha=%.3f tau=%.3f",
            args.merge_strategy,
            args.merge_alpha,
            args.merge_tau,
        )
        plan = build_merge_plan(
            teacher_dir,
            adapter,
            hf,
            surviving,
            strategy=args.merge_strategy,
            alpha=args.merge_alpha,
            tau=args.merge_tau,
            cooccur_path=args.cooccur_path,
            log=log,
        )
        log.info(
            "macro merge plan: %d layers covered (each rolling %d dropped → %d kept)",
            len(plan.weights),
            max((len(plan.dropped_per_layer.get(l, [])) for l in plan.weights), default=0),
            max((len(plan.surviving_per_layer.get(l, [])) for l in plan.weights), default=0),
        )

    log.info("Writing student config -> %s", student_dir)
    write_student_config(
        cfg_path,
        student_dir,
        adapter,
        app.prune.target_num_experts,
        app.prune.target_num_experts_per_tok,
    )

    log.info("Slicing + merging weights %s -> %s", teacher_dir, student_dir)
    prune_state_dict_sharded(
        teacher_dir,
        student_dir,
        adapter,
        hf,
        app.prune.target_num_experts,
        app.prune.keep_shared_experts,
        surviving_per_layer=surviving,
        merge_plan=plan,
    )

    mapping = build_expert_mapping_json(
        num_layers,
        num_experts,
        app.prune.target_num_experts,
        surviving_per_layer=surviving,
    )
    (student_dir / "expert_mapping.json").write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    plan_path = student_dir / "merge_plan.json"
    plan_path.write_text(
        json.dumps(serialize_merge_plan(plan), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %s", plan_path)

    teacher_arch_list = hf.get("architectures") or []
    teacher_arch = teacher_arch_list[0] if teacher_arch_list else hf.get("model_type")
    report_path = write_merge_report(
        plan,
        student_dir,
        teacher_arch=teacher_arch,
        num_layers_total=num_layers,
        num_experts_total=num_experts,
        target_num_experts=app.prune.target_num_experts,
        target_num_experts_per_tok=app.prune.target_num_experts_per_tok,
    )
    log.info("Wrote %s", report_path)

    copy_tokenizer_assets(teacher_dir, student_dir)

    if (
        args.merge_strategy in _NEURON_STRATEGIES
        and plan.neuron_down_contrib_paths
        and not args.keep_buckets
    ):
        bucket_root = Path(
            args.bucket_dir
            if args.bucket_dir is not None
            else (student_dir / ".neuron_buckets")
        )
        removed = 0
        for path_str in plan.neuron_down_contrib_paths.values():
            p = Path(path_str)
            if p.is_file():
                p.unlink()
                removed += 1
        try:
            bucket_root.rmdir()
        except OSError:
            pass
        log.info("Removed %d streamed bucket file(s) from %s", removed, bucket_root)
    log.info("Prune (with merge) complete: %s", student_dir)


if __name__ == "__main__":
    main()
