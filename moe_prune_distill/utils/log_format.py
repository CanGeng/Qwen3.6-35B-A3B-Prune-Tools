"""Aligned, fixed-width metric line formatter.

Used by both the layerwise block trainer and the end-to-end trainer so
their console / log lines line up column-by-column. Keeps formatting
trivial and side-effect-free; callers handle the actual logger.

Output shape::

    block=03 step=  120 mode=train | loss=  0.012345 ema_h=  0.011820 lr= 5.00e-05 gn=   0.842 hidden_mse=  0.0118  ...

* ``prefix`` — ordered identifying tags (block, step, epoch, mode). Each
  is rendered as ``key=value`` with ``value`` formatted as the caller
  passes it (already a string). They appear left of the ``|`` divider.
* ``scalars`` — numeric metrics. Known keys get fixed widths and formats
  (loss/lr/grad_norm/hidden_mse/router_kl/sft_ce/...); unknown keys fall
  back to ``.4f`` at width 10. Listed in a stable, eyeball-friendly order.
"""

from __future__ import annotations

from collections.abc import Mapping

# (key, width, fmt) — width *includes* the formatted number, no trailing
# space. The caller joins with single spaces; widths above are tuned so
# typical values don't shift columns. Unknown keys fall through to
# (10, ".4f").
_SCALAR_SPEC: dict[str, tuple[int, str]] = {
    # losses / convergence
    "loss":          (10, ".5f"),
    "ema_h":         (10, ".5f"),
    "ema_hidden_mse": (10, ".5f"),
    "hidden_mse":    (10, ".5f"),
    "router_kl":     (9, ".4f"),
    "sft_ce":        (9, ".4f"),
    # optimization
    "lr":            (9, ".2e"),
    "gn":            (8, ".3f"),
    "grad_norm":     (8, ".3f"),
    # token stats
    "valid_tokens":  (7, "d"),
    "mean_seq_len":  (7, ".1f"),
    "max_seq_len":   (5, "d"),
    # diagnostics (router / hidden)
    "router_top1_agree": (6, ".3f"),
    "router_topk_jaccard": (6, ".3f"),
    "hidden_cos":    (6, ".3f"),
    "hidden_rel_err": (7, ".4f"),
}

# Stable display order for known scalars. Anything not listed here is
# appended in alphabetical order at the end.
_SCALAR_ORDER: tuple[str, ...] = (
    "loss",
    "ema_h",
    "ema_hidden_mse",
    "lr",
    "gn",
    "grad_norm",
    "hidden_mse",
    "router_kl",
    "sft_ce",
    "valid_tokens",
    "mean_seq_len",
    "max_seq_len",
    "router_top1_agree",
    "router_topk_jaccard",
    "hidden_cos",
    "hidden_rel_err",
)


def _format_scalar(key: str, value: float | int) -> str:
    width, fmt = _SCALAR_SPEC.get(key, (10, ".4f"))
    if fmt.endswith("d"):
        cell = f"{int(value):>{width}d}"
    else:
        cell = f"{float(value):>{width}{fmt}}"
    return f"{key}={cell}"


def format_metrics_row(
    *,
    prefix: Mapping[str, str],
    scalars: Mapping[str, float | int],
) -> str:
    """Render one aligned metric line. See module docstring for shape."""
    left = " ".join(f"{k}={v}" for k, v in prefix.items())
    seen: set[str] = set()
    cells: list[str] = []
    for k in _SCALAR_ORDER:
        if k in scalars:
            cells.append(_format_scalar(k, scalars[k]))
            seen.add(k)
    extras = sorted(k for k in scalars if k not in seen)
    for k in extras:
        cells.append(_format_scalar(k, scalars[k]))
    right = "  ".join(cells)  # double space between metric cells for scan-readability
    if left and right:
        return f"{left} | {right}"
    return left or right


def format_block_banner(
    block_id: int,
    total_blocks: int,
    layer_indices: tuple[int, ...] | list[int],
    n_params: int,
) -> str:
    """One-line section header printed when a block starts training."""
    layers_s = ",".join(str(li) for li in layer_indices)
    p_b = n_params / 1e9
    if p_b >= 1.0:
        params_s = f"{p_b:.2f}B"
    else:
        params_s = f"{n_params / 1e6:.1f}M"
    body = f" block {block_id:03d}/{total_blocks:03d}  layers=[{layers_s}]  params={params_s} "
    bar = "=" * max(8, 78 - len(body))
    return f"===={body}{bar}"


__all__ = ["format_metrics_row", "format_block_banner"]
