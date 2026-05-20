"""Tests for the soft-dep TensorBoardWriter.

Covers three regimes:
1. tensorboard package missing → no-op writer with a single WARNING.
2. tensorboard installed → log() writes scalars under the namespace tag.
3. NaN/Inf and non-numeric values are silently dropped.

When tensorboard isn't importable, only test #1 runs; tests #2 and #3 are
skipped via ``importorskip`` so the suite stays green on minimal envs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from moe_prune_distill.utils.tensorboard import TensorBoardWriter


def test_disabled_writer_is_inert(tmp_path: Path) -> None:
    """enabled=False short-circuits before the soft import; never touches tensorboard."""
    w = TensorBoardWriter(tmp_path, enabled=False, namespace="train")
    assert w.enabled is False
    # Should not raise even though tensorboard might be missing.
    w.log({"loss": 1.0, "step": 1})
    w.flush()
    w.close()
    # nothing written on disk
    assert not list(tmp_path.iterdir())


def test_log_dir_none_disables(tmp_path: Path) -> None:
    w = TensorBoardWriter(None, enabled=True, namespace="train")
    assert w.enabled is False
    w.log({"loss": 1.0, "step": 1})  # no-op
    w.close()


def test_missing_tensorboard_falls_back_to_noop(monkeypatch, tmp_path: Path) -> None:
    """If tensorboard is unavailable, the writer logs a warning and disables itself."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch.utils.tensorboard" or name.startswith("torch.utils.tensorboard."):
            raise ImportError("tensorboard not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    w = TensorBoardWriter(tmp_path, enabled=True, namespace="train")
    assert w.enabled is False
    w.log({"loss": 1.0, "step": 5})  # must not raise
    w.close()


def test_real_tensorboard_writes_scalar(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    pytest.importorskip("torch.utils.tensorboard")

    w = TensorBoardWriter(tmp_path / "run", enabled=True, namespace="train")
    assert w.enabled is True
    w.log({"step": 1, "loss": 0.5, "lr": 1e-4, "grad_norm": 0.7})
    w.log({"step": 2, "loss": 0.4})
    w.flush()
    w.close()

    # SummaryWriter creates a tfevents file under the run dir.
    events = list((tmp_path / "run").glob("events.out.tfevents.*"))
    assert events, "expected at least one tfevents file"
    # File should be non-empty (header + at least one summary).
    assert events[0].stat().st_size > 0


def test_nan_and_non_numeric_dropped_silently(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    pytest.importorskip("torch.utils.tensorboard")

    w = TensorBoardWriter(tmp_path / "run", enabled=True, namespace="t")
    # None of these should raise.
    w.log(
        {
            "step": 1,
            "loss": float("nan"),
            "lr": float("inf"),
            "tag": "this is a string",
            "ok": 0.123,
        }
    )
    w.close()
    events = list((tmp_path / "run").glob("events.out.tfevents.*"))
    assert events


def test_namespace_is_applied_to_tags(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    pytest.importorskip("torch.utils.tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    w = TensorBoardWriter(tmp_path / "run", enabled=True, namespace="train")
    w.log({"step": 1, "loss": 0.5})
    w.close()

    ea = EventAccumulator(str(tmp_path / "run"))
    ea.Reload()
    tags = set(ea.Tags().get("scalars", []))
    assert "train/loss" in tags
    assert "loss" not in tags  # bare tag should NOT exist


def test_step_fallback_to_row(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    pytest.importorskip("torch.utils.tensorboard")

    w = TensorBoardWriter(tmp_path / "run", enabled=True, namespace="t")
    # If row carries 'step', explicit step= can be omitted.
    w.log({"step": 7, "loss": 1.0})
    w.close()
    assert list((tmp_path / "run").glob("events.out.tfevents.*"))


def test_log_without_step_is_silent(tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    pytest.importorskip("torch.utils.tensorboard")
    w = TensorBoardWriter(tmp_path / "run", enabled=True, namespace="t")
    # No step in row, no step= kwarg → call should be a no-op (no exception).
    w.log({"loss": 1.0})
    w.close()
