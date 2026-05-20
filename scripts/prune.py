"""Step 3: prune teacher into a smaller student (first_n / router_top / manual)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from moe_prune_distill.adapters import detect_adapter
from moe_prune_distill.config import load_config
from moe_prune_distill.prune.config_editor import write_student_config
from moe_prune_distill.prune.expert_selector import (
    load_router_stats,
    parse_manual_experts,
    surviving_experts_first_n,
    surviving_experts_router_top,
)
from moe_prune_distill.prune.slicer import build_expert_mapping_json, prune_state_dict_sharded
from moe_prune_distill.utils.logging import get_logger


_TOKENIZER_FILES = (
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
)


def copy_tokenizer_assets(teacher_dir: Path, student_dir: Path) -> None:
    """Copy tokenizer / preprocessor sidecar files from teacher to student."""
    for name in _TOKENIZER_FILES:
        src = teacher_dir / name
        if src.is_file():
            shutil.copy2(src, student_dir / name)


# Backwards-compat alias for older imports.
_copy_tokenizer_assets = copy_tokenizer_assets


def resolve_surviving(
    app, num_layers: int, num_experts: int, log,
) -> dict[int, list[int]] | None:
    """Resolve surviving expert ids per layer from the prune config.

    Returns ``None`` for the ``first_n`` strategy — the slicer handles the
    fallback implicitly. Re-used by ``scripts/prune_merge.py``.
    """
    target = app.prune.target_num_experts
    if app.prune.expert_selection == "first_n":
        log.info("expert_selection=first_n")
        return None  # fallback path; slicer uses first_n implicitly
    if app.prune.expert_selection == "manual":
        log.info("expert_selection=manual")
        return parse_manual_experts(app.prune.manual_experts, num_layers, num_experts, target)
    if app.prune.expert_selection == "router_top":
        stats_path = (
            Path(app.prune.router_stats_path).resolve()
            if app.prune.router_stats_path
            else Path(app.prune.student_dir).resolve().parent / "router_stats.json"
        )
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"router_top requires {stats_path}; run scripts/collect_router_stats.py first"
            )
        log.info("expert_selection=router_top from %s", stats_path)
        usage = load_router_stats(stats_path)
        return surviving_experts_router_top(usage, num_layers, num_experts, target)
    raise ValueError(f"unknown expert_selection: {app.prune.expert_selection}")


_resolve_surviving = resolve_surviving


def main() -> None:
    log = get_logger()
    p = argparse.ArgumentParser(description="Prune MoE experts (first_n / router_top / manual)")
    p.add_argument("--config", type=str, required=True)
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

    log.info("Writing student config -> %s", student_dir)
    write_student_config(
        cfg_path,
        student_dir,
        adapter,
        app.prune.target_num_experts,
        app.prune.target_num_experts_per_tok,
    )

    log.info("Slicing weights %s -> %s", teacher_dir, student_dir)
    prune_state_dict_sharded(
        teacher_dir,
        student_dir,
        adapter,
        hf,
        app.prune.target_num_experts,
        app.prune.keep_shared_experts,
        surviving_per_layer=surviving,
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

    copy_tokenizer_assets(teacher_dir, student_dir)
    log.info("Prune complete: %s", student_dir)


if __name__ == "__main__":
    main()
