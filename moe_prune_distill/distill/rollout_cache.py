"""Per-block student rollout cache for layerwise distillation.

When ``train.layerwise.use_student_rollout_input`` is true, each block N's
trained layers are forwarded once more (eval, no grad) over every training
sample. The resulting hidden states at ``block.output_layer`` are written
here so block N+1 can read them as its **input** (the loss target stays
``teacher_cache[output_layer]``).

On-disk layout (mirrors v2 teacher cache for predictable mmap behavior):

    {root}/
      rollout_index.json
      rollout_block_{NNN}_chunk_{j}.safetensors   # keys: {sid}.hidden -> [seq, hidden]

``rollout_index.json``:

    {
      "version": 1,
      "chunk_size": 1000,
      "samples": {
        "<sid>": {"block_id": 0, "output_layer": 3, "chunk": 0},
        ...
      }
    }

The reader auto-derives the chunk filename from ``(block_id, chunk)``.
Only the most recent block's rollout for each sample is needed at any
given time (block N+1 reads block N's output), but we keep older blocks'
files on disk so a partial run can resume by reading whichever block was
last completed.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROLLOUT_INDEX_NAME = "rollout_index.json"


def _chunk_filename(block_id: int, chunk_id: int) -> str:
    return f"rollout_block_{int(block_id):03d}_chunk_{int(chunk_id)}.safetensors"


class RolloutCacheWriter:
    """Buffer one block's rollout outputs, flush per chunk, finalize an index.

    Sample order at construction time defines the chunk assignment:
    ``sid_to_chunk[sid] = i // chunk_size`` for the i-th declared sample.
    All ``add`` calls for a given chunk should happen before
    :meth:`flush_chunk` to bound memory; :meth:`finalize` drains any
    leftovers and writes the index json.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        block_id: int,
        output_layer: int,
        sample_ids: list[str],
        chunk_size: int = 1000,
        cache_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.block_id = int(block_id)
        self.output_layer = int(output_layer)
        self.sample_ids = list(sample_ids)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.cache_dtype = cache_dtype
        self.sid_to_chunk: dict[str, int] = {
            sid: i // self.chunk_size for i, sid in enumerate(self.sample_ids)
        }
        self.num_chunks = (
            (len(self.sample_ids) + self.chunk_size - 1) // self.chunk_size
            if self.sample_ids
            else 0
        )
        self._buffers: dict[int, dict[str, torch.Tensor]] = {}

    def add(self, sample_id: str, hidden: torch.Tensor) -> None:
        if sample_id not in self.sid_to_chunk:
            raise KeyError(f"sample {sample_id!r} not declared at writer init")
        chunk = self.sid_to_chunk[sample_id]
        buf = self._buffers.setdefault(chunk, {})
        buf[f"{sample_id}.hidden"] = (
            hidden.detach().to(self.cache_dtype).cpu().contiguous()
        )

    def flush_chunk(self, chunk_id: int) -> None:
        chunk_id = int(chunk_id)
        buf = self._buffers.pop(chunk_id, None)
        if not buf:
            return
        self._write_chunk(chunk_id, buf)
        buf.clear()

    def finalize(self) -> None:
        """Drain any leftover chunks, write/merge the index json."""
        for chunk_id, buf in list(self._buffers.items()):
            if buf:
                self._write_chunk(chunk_id, buf)
            self._buffers.pop(chunk_id, None)

        index_path = self.root / ROLLOUT_INDEX_NAME
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            index = {"version": 1, "chunk_size": self.chunk_size, "samples": {}}
        index["chunk_size"] = self.chunk_size
        samples = index.setdefault("samples", {})
        for sid, ch in self.sid_to_chunk.items():
            # Latest writer wins: block N+1 invalidates block N for sid.
            samples[sid] = {
                "block_id": self.block_id,
                "output_layer": self.output_layer,
                "chunk": int(ch),
            }
        tmp = index_path.with_suffix(index_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(index_path)
        _invalidate_index(self.root)

    def _write_chunk(self, chunk_id: int, buf: dict[str, torch.Tensor]) -> None:
        path = self.root / _chunk_filename(self.block_id, chunk_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        save_file(buf, str(tmp))
        tmp.replace(path)


# ---- reader ----

_INDEX_MEMO: dict[str, dict[str, Any]] = {}
_INDEX_LOCK = Lock()
_HANDLES: dict[tuple[str, str], Any] = {}
_HANDLES_LOCK = Lock()


def _read_index(root: Path) -> dict[str, Any]:
    key = str(root.resolve())
    with _INDEX_LOCK:
        cached = _INDEX_MEMO.get(key)
        if cached is not None:
            return cached
    raw = json.loads((root / ROLLOUT_INDEX_NAME).read_text(encoding="utf-8"))
    with _INDEX_LOCK:
        _INDEX_MEMO[key] = raw
    return raw


def _invalidate_index(root: Path) -> None:
    key = str(root.resolve())
    with _INDEX_LOCK:
        _INDEX_MEMO.pop(key, None)
    with _HANDLES_LOCK:
        for k in [hk for hk in _HANDLES if hk[0] == key]:
            _HANDLES.pop(k, None)


def _get_handle(root: Path, filename: str):
    key = (str(root.resolve()), filename)
    with _HANDLES_LOCK:
        h = _HANDLES.get(key)
        if h is not None:
            return h
    h = safe_open(str(root / filename), framework="pt", device="cpu")
    with _HANDLES_LOCK:
        existing = _HANDLES.get(key)
        if existing is not None:
            return existing
        _HANDLES[key] = h
    return h


def rollout_index_exists(root: Path | str) -> bool:
    return (Path(root) / ROLLOUT_INDEX_NAME).is_file()


def has_rollout(root: Path | str, sample_id: str) -> bool:
    root = Path(root)
    if not rollout_index_exists(root):
        return False
    index = _read_index(root)
    return sample_id in index.get("samples", {})


def block_id_for(root: Path | str, sample_id: str) -> int | None:
    root = Path(root)
    if not rollout_index_exists(root):
        return None
    index = _read_index(root)
    entry = index.get("samples", {}).get(sample_id)
    if entry is None:
        return None
    return int(entry.get("block_id"))


def load_rollout_input(root: Path | str, sample_id: str) -> torch.Tensor:
    """Return the latest cached student hidden state for ``sample_id``.

    The chunk file is mmap'd via a pooled ``safe_open`` handle. The caller
    is responsible for moving the tensor to GPU / casting dtype.
    """
    root = Path(root)
    index = _read_index(root)
    entry = index["samples"][sample_id]
    block_id = int(entry["block_id"])
    chunk = int(entry["chunk"])
    h = _get_handle(root, _chunk_filename(block_id, chunk))
    return h.get_tensor(f"{sample_id}.hidden")


__all__ = [
    "ROLLOUT_INDEX_NAME",
    "RolloutCacheWriter",
    "block_id_for",
    "has_rollout",
    "load_rollout_input",
    "rollout_index_exists",
]
