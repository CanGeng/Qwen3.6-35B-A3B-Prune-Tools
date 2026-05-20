"""Shared writer for ``router_stats.json``.

Both :mod:`scripts.collect_router_stats` (legacy, full-model forward) and
:mod:`scripts.stream_teacher` (layer-by-layer streaming) emit the exact same
file by going through this single helper.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def write_router_stats(
    out_path: str | Path,
    *,
    model_id: str,
    num_samples: int,
    num_layers: int,
    num_experts: int,
    top_k: int,
    counts: torch.Tensor,
    target_num_experts: int | None,
) -> Path:
    """Persist a ``router_stats.json`` payload.

    ``counts`` must be a ``[num_layers, num_experts]`` long tensor of top-k
    selection counts. If ``target_num_experts`` is given, the per-layer
    ``recommended`` list will hold its top ``target_num_experts`` ids
    sorted ascending; otherwise all expert ids are returned (sorted by
    descending usage, then ascending id).
    """
    if counts.shape != (num_layers, num_experts):
        raise ValueError(
            f"counts shape {tuple(counts.shape)} != ({num_layers}, {num_experts})"
        )

    layers_out: dict[str, dict] = {}
    for layer in range(num_layers):
        usage = {str(e): int(counts[layer, e].item()) for e in range(num_experts)}
        ranked = sorted(
            range(num_experts), key=lambda e: (-int(counts[layer, e].item()), e)
        )
        if target_num_experts is None:
            recommended = sorted(ranked)
        else:
            recommended = sorted(ranked[:target_num_experts])
        layers_out[str(layer)] = {
            "usage_counts": usage,
            "recommended": recommended,
        }

    payload = {
        "model_id": model_id,
        "num_samples": int(num_samples),
        "num_layers": int(num_layers),
        "num_experts": int(num_experts),
        "top_k": int(top_k),
        "layers": layers_out,
    }

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_path


__all__ = ["write_router_stats"]
