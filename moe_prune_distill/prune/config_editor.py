"""Write modified config.json for pruned student."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from moe_prune_distill.adapters.base import MoEAdapter


def write_student_config(
    teacher_config_path: Path,
    student_dir: Path,
    adapter: MoEAdapter,
    target_num_experts: int,
    target_num_experts_per_tok: int,
) -> None:
    with teacher_config_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = json.load(f)
    new_cfg = adapter.modify_config(cfg, target_num_experts, target_num_experts_per_tok)
    student_dir.mkdir(parents=True, exist_ok=True)
    out = student_dir / "config.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2, ensure_ascii=False)
