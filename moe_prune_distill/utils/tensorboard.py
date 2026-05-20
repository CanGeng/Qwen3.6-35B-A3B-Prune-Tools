"""TensorBoard scalar writer with a soft tensorboard dependency.

We treat ``tensorboard`` as optional: import errors fall through to a
silent no-op writer so a developer machine without tensorboard installed
keeps training without a stack trace. The same ``log(...)`` shape works
for both branches, mirroring :class:`JsonlMetricsWriter` so call sites
can hold one of each side-by-side.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


_log = logging.getLogger("moe_prune_distill.tensorboard")


class TensorBoardWriter:
    """Thin wrapper around ``torch.utils.tensorboard.SummaryWriter``.

    * Lazy import — the SummaryWriter is constructed on first use, so a
      disabled / missing-dep instance allocates nothing.
    * Tag namespacing via the ``namespace`` kwarg (e.g. ``train`` ->
      tags become ``train/loss``, ``train/lr``).
    * Robust to non-numeric values: anything that can't be coerced to a
      finite float is silently skipped (TB rejects NaN/Inf with a stack
      trace; we don't want metric noise to crash a long run).
    """

    def __init__(
        self,
        log_dir: str | Path | None,
        *,
        enabled: bool = True,
        namespace: str = "",
    ) -> None:
        self.namespace = namespace.strip("/") if namespace else ""
        self._writer: Any | None = None
        self._enabled = bool(enabled and log_dir is not None)
        self._log_dir = Path(log_dir) if log_dir else None
        self._init_failed = False
        if self._enabled:
            self._try_init()

    def _try_init(self) -> None:
        if self._writer is not None or self._init_failed:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: WPS433
        except Exception as e:  # tensorboard package missing or broken
            _log.warning(
                "tensorboard unavailable (%s); TensorBoard logging disabled. "
                "Install with `pip install tensorboard` to enable.",
                e,
            )
            self._enabled = False
            self._init_failed = True
            return
        try:
            assert self._log_dir is not None
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self._log_dir))
        except Exception as e:
            _log.warning("SummaryWriter init failed (%s); disabling TB", e)
            self._enabled = False
            self._init_failed = True

    @property
    def enabled(self) -> bool:
        return self._enabled and self._writer is not None

    def _scalar_tag(self, key: str) -> str:
        return f"{self.namespace}/{key}" if self.namespace else key

    def log(self, row: dict[str, Any], *, step: int | None = None) -> None:
        """Write every numeric scalar in ``row`` under the writer's namespace.

        ``step`` is the global step. If omitted, falls back to ``row["step"]``;
        if that's also missing, this call is a no-op (TB needs an x value).
        Non-numeric / NaN / Inf values are skipped silently.
        """
        if not self.enabled:
            return
        if step is None:
            step = row.get("step")
        if step is None:
            return
        try:
            step_i = int(step)
        except Exception:
            return
        assert self._writer is not None
        for k, v in row.items():
            if k == "step":
                continue
            f = _to_finite_float(v)
            if f is None:
                continue
            self._writer.add_scalar(self._scalar_tag(str(k)), f, step_i)

    def flush(self) -> None:
        if self.enabled:
            assert self._writer is not None
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass
            self._writer = None
            self._enabled = False

    def __enter__(self) -> "TensorBoardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _to_finite_float(v: Any) -> float | None:
    """Coerce ``v`` to a finite float, or return None if it isn't a usable scalar."""
    if isinstance(v, bool):  # bool is an int subclass; skip to avoid 0/1 noise
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    else:
        try:
            import torch  # local import keeps numpy-only paths happy
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    return None
                f = float(v.detach().cpu().item())
            else:
                return None
        except Exception:
            return None
    import math
    if not math.isfinite(f):
        return None
    return f


__all__ = ["TensorBoardWriter"]
