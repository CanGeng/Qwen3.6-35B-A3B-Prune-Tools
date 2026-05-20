"""Expert selection strategies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def surviving_experts_first_n(num_experts: int, target: int) -> list[int]:
    n = min(int(target), int(num_experts))
    return list(range(n))


def surviving_experts_router_top(
    layer_usage: dict[int, dict[int, int] | dict[str, int]],
    num_layers: int,
    num_experts: int,
    target: int,
) -> dict[int, list[int]]:
    """Per-layer top-K experts ranked by usage count.

    Falls back to first_n for layers that have no recorded usage.
    """
    fallback = surviving_experts_first_n(num_experts, target)
    out: dict[int, list[int]] = {}
    for layer in range(num_layers):
        counts_raw = layer_usage.get(layer)
        if counts_raw is None:
            counts_raw = layer_usage.get(str(layer)) if isinstance(layer_usage, dict) else None
        if not counts_raw:
            out[layer] = list(fallback)
            continue
        counts = {int(k): int(v) for k, v in counts_raw.items()}
        ranked = sorted(range(num_experts), key=lambda e: -counts.get(e, 0))
        keep = sorted(ranked[: min(target, num_experts)])
        out[layer] = keep
    return out


def load_router_stats(path: str | Path) -> dict[int, dict[int, int]]:
    """Read collect_router_stats.py output into per-layer count dicts."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    layers_raw = data.get("layers", {})
    out: dict[int, dict[int, int]] = {}
    for k, v in layers_raw.items():
        layer = int(k)
        counts = v.get("usage_counts") or {}
        out[layer] = {int(e): int(c) for e, c in counts.items()}
    return out


def parse_manual_experts(
    spec: Any,
    num_layers: int,
    num_experts: int,
    target: int,
) -> dict[int, list[int]]:
    """Validate manual expert lists. Accepts dict[layer]=ids or list-of-lists."""
    out: dict[int, list[int]] = {}
    if isinstance(spec, dict):
        for k, ids in spec.items():
            layer = int(k.split("_")[-1]) if isinstance(k, str) and not k.isdigit() else int(k)
            out[layer] = sorted(int(x) for x in ids)
    elif isinstance(spec, list):
        for layer, ids in enumerate(spec):
            out[layer] = sorted(int(x) for x in ids)
    else:
        raise ValueError("manual_experts must be dict or list-of-lists")

    fallback = surviving_experts_first_n(num_experts, target)
    final: dict[int, list[int]] = {}
    for layer in range(num_layers):
        ids = out.get(layer, fallback)
        if any(e < 0 or e >= num_experts for e in ids):
            raise ValueError(f"manual_experts layer {layer} contains out-of-range expert id")
        if len(ids) != target:
            raise ValueError(
                f"manual_experts layer {layer} has {len(ids)} ids, expected target={target}"
            )
        final[layer] = sorted(set(ids))
    return final
