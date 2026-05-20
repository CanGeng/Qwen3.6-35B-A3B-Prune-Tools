"""Stub: collect teacher-router top-k pair co-occurrence into ``router_cooccur.json``.

The full implementation will mirror ``scripts/collect_router_stats.py`` /
``scripts/stream_teacher.py``: hook each layer's router gate, take the
top-k indices for every (token, layer) pair, and accumulate

    pair_counts[layer, expert_a, expert_b] += 1   for all a < b in top-k

into a ``[num_layers, num_experts, num_experts]`` long tensor, then dump
into a json with the same per-layer dict format ``router_stats.json`` uses
plus a ``pair_counts`` field. ``scripts/prune_merge.py --merge-strategy
cooccur`` consumes that file.

v1 ships this as a stub so users picking ``--merge-strategy cooccur`` get
a clear error instead of a crash deep in ``build_merge_plan``. The
``weight_cosine`` strategy is fully working without this script.

TODO(prune-merge):
* Lift the ``LayerStreamer`` router-hook accumulation pattern from
  ``stream_teacher.py``, swap the per-expert counter for a pair counter,
  and emit ``router_cooccur.json`` alongside ``router_stats.json``.
* Add a ``--top-k`` override flag (default = teacher config's
  ``num_experts_per_tok``) for ablations.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "scripts/collect_router_cooccur.py is not yet implemented. "
        "Use --merge-strategy weight_cosine in scripts/prune_merge.py for now."
    )


if __name__ == "__main__":
    main()
