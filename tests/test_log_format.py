"""Tests for the aligned metric line formatter."""

from __future__ import annotations

from moe_prune_distill.utils.log_format import (
    format_block_banner,
    format_metrics_row,
)


def test_format_metrics_row_known_keys_have_fixed_widths() -> None:
    """loss / lr / gn etc. live in stable, padded columns."""
    line = format_metrics_row(
        prefix={"block": "03", "step": "  120", "mode": "train"},
        scalars={"loss": 0.012345, "lr": 5e-5, "gn": 0.842, "ema_h": 0.011820},
    )
    # prefix divider present, prefix order preserved
    assert line.startswith("block=03 step=  120 mode=train | ")
    # known scalar order: loss, ema_h, lr, gn
    cells = line.split(" | ", 1)[1]
    assert cells.index("loss=") < cells.index("ema_h=")
    assert cells.index("ema_h=") < cells.index("lr=")
    assert cells.index("lr=") < cells.index("gn=")
    # right-aligned width (loss spec is .5f, width 10)
    assert "loss=   0.01234" in cells or "loss=   0.01235" in cells


def test_format_metrics_row_unknown_keys_appended_alpha() -> None:
    line = format_metrics_row(
        prefix={"step": "1"},
        scalars={"loss": 1.0, "zeta": 7.0, "alpha": 3.0},
    )
    cells = line.split(" | ", 1)[1]
    assert cells.index("loss=") < cells.index("alpha=")
    assert cells.index("alpha=") < cells.index("zeta=")


def test_format_metrics_row_handles_int_keys() -> None:
    line = format_metrics_row(
        prefix={"step": "1"},
        scalars={"valid_tokens": 4096, "max_seq_len": 1024},
    )
    cells = line.split(" | ", 1)[1]
    # int format spec
    assert "valid_tokens=" in cells
    assert "4096" in cells
    assert "max_seq_len=" in cells
    assert "1024" in cells


def test_format_metrics_row_no_prefix() -> None:
    line = format_metrics_row(prefix={}, scalars={"loss": 1.0})
    assert "|" not in line
    assert "loss=" in line


def test_format_metrics_row_no_scalars() -> None:
    line = format_metrics_row(prefix={"step": "1"}, scalars={})
    assert "|" not in line
    assert line == "step=1"


def test_format_block_banner_shape() -> None:
    line = format_block_banner(3, 10, (12, 13, 14, 15), 1_680_000_000)
    assert "block 003/010" in line
    assert "[12,13,14,15]" in line
    assert "1.68B" in line
    assert line.startswith("====")


def test_format_block_banner_small_params() -> None:
    line = format_block_banner(0, 10, (0, 1), 320_000)
    # under 1B → MiB unit
    assert "0.3M" in line or "0.32M" in line
