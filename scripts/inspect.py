"""Step 1: inspect local HF model (config + adapter + sample keys)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file

from moe_prune_distill.adapters import detect_adapter
from moe_prune_distill.config import load_config


def _iter_sample_keys(teacher_dir: Path, limit: int = 16) -> list[str]:
    index = teacher_dir / "model.safetensors.index.json"
    if index.is_file():
        wm = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        keys = sorted(wm.keys())
        mlp = [k for k in keys if "mlp" in k]
        rest = [k for k in keys if k not in mlp]
        merged = mlp + rest
        return merged[:limit]
    singles = sorted(teacher_dir.glob("*.safetensors"))
    if not singles:
        return []
    sd = load_file(str(singles[0]))
    keys = sorted(sd.keys())
    mlp = [k for k in keys if "mlp" in k]
    rest = [k for k in keys if k not in mlp]
    return (mlp + rest)[:limit]


def _estimate_total_params(teacher_dir: Path) -> int:
    total = 0
    seen: set[str] = set()
    index = teacher_dir / "model.safetensors.index.json"
    if index.is_file():
        wm = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        shard_files = sorted(set(wm.values()))
    else:
        shard_files = sorted({p.name for p in teacher_dir.glob("*.safetensors")})
    for name in shard_files:
        path = teacher_dir / name
        if not path.is_file():
            continue
        sd = load_file(str(path))
        for k, t in sd.items():
            if k in seen:
                continue
            seen.add(k)
            total += int(t.numel())
    return total


def _estimate_pruned_params(teacher_params: int, old_experts: int, new_experts: int) -> int:
    """Rough lower bound: scale non-router params ~linearly with routed expert count."""
    if old_experts <= 0 or new_experts >= old_experts:
        return teacher_params
    return int(teacher_params * (new_experts / old_experts))


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect HF MoE checkpoint")
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    app = load_config(args.config)
    teacher_dir = Path(app.download.local_dir).resolve()
    cfg_path = teacher_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    hf = json.loads(cfg_path.read_text(encoding="utf-8"))
    adapter = detect_adapter(hf)
    layers = adapter.get_num_layers(hf)
    experts = adapter.get_num_experts(hf)
    topk = adapter.get_num_experts_per_tok(hf)
    shared = adapter.has_shared_expert(hf)

    print("=== MoE inspect ===")
    print(f"path: {teacher_dir}")
    print(f"adapter: {adapter.__class__.__name__}")
    arch = hf.get("architectures")
    if arch:
        print(f"architectures: {arch}")
    print(f"model_type (root): {hf.get('model_type')}")
    tc = hf.get("text_config")
    if isinstance(tc, dict):
        print(
            "text_config.model_type:",
            tc.get("model_type"),
            "| layer_types sample:",
            (tc.get("layer_types") or [])[:6],
            "...",
        )
    print(f"num_hidden_layers: {layers}")
    print(f"num_experts (routed): {experts}")
    print(f"num_experts_per_tok: {topk}")
    print(f"has_shared_expert (config heuristic): {shared}")

    sample = _iter_sample_keys(teacher_dir)
    print("sample_keys:")
    for k in sample:
        info = adapter.parse_state_dict_key(k)
        print(f"  {k} -> {info}")

    total_p = _estimate_total_params(teacher_dir)
    new_e = app.prune.target_num_experts
    pruned_est = _estimate_pruned_params(total_p, experts, new_e)
    print(f"approx_total_params (unique tensors): {total_p}")
    print(
        f"approx_params_after_prune (linear expert scaling x{new_e}/{experts}): {pruned_est}"
    )


if __name__ == "__main__":
    main()
