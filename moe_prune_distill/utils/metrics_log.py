"""Append-only JSONL writer for training metrics.

Lazy directory creation, line-buffered, tensor-coercing. One row per call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, IO


def _coerce(v: Any) -> Any:
    """Best-effort scalarisation so torch tensors / numpy values land as JSON numbers."""
    try:
        import torch  # local import keeps numpy-only environments happy
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.detach().cpu().item())
            return [float(x) for x in v.detach().cpu().flatten().tolist()]
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [_coerce(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _coerce(x) for k, x in v.items()}
    return v


class JsonlMetricsWriter:
    """Append-only JSONL writer. Opens lazily; safe to instantiate before
    output_dir exists.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: IO[str] | None = None

    def _open(self) -> IO[str]:
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        return self._fh

    def log(self, row: dict[str, Any]) -> None:
        fh = self._open()
        coerced = {str(k): _coerce(v) for k, v in row.items()}
        fh.write(json.dumps(coerced, ensure_ascii=False) + "\n")
        fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def __enter__(self) -> "JsonlMetricsWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["JsonlMetricsWriter"]
