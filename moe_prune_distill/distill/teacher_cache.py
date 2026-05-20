"""Teacher hidden / router cache I/O.

Two on-disk layouts are supported, selected automatically:

* **legacy v1** (one safetensors per sample) — historical and what
  ``cache_teacher.py`` still emits. Schema:
  ``{cache_dir}/{sid}.safetensors`` with keys ``input_ids``,
  ``attention_mask``, ``hidden.layer_<i>``, ``router.layer_<i>``.

* **batched v2** (per-layer, sample-chunked) — emitted by ``stream_teacher``
  via :class:`BatchedCacheWriter`. One header file
  ``cache_meta.safetensors`` holds every sample's ``input_ids`` and
  ``attention_mask``; each cached layer is sharded into one safetensors per
  sample chunk: ``cache_layer_{i}_chunk_{j}.safetensors`` with keys
  ``{sid}.hidden`` and (optionally) ``{sid}.router``. ``cache_index.json``
  records which chunk holds each sample. The chunk_id is sample-stable
  (sample i -> chunk i // chunk_size for **every** cached layer), so all of
  one sample's tensors live at the same chunk index across layers.

:func:`load_sample_cache` and :func:`cache_exists` auto-detect by checking
for ``cache_index.json`` in ``cache_dir`` and dispatch accordingly. The
returned dict shape is identical for both layouts so downstream readers
(``DistillJsonlDataset``, ``layerwise_trainer``) don't notice.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


_HIDDEN_RE = re.compile(r"^hidden\.layer_(\d+)$")
_ROUTER_RE = re.compile(r"^router\.layer_(\d+)$")

# v2 batched-layout filenames
BATCHED_INDEX_NAME = "cache_index.json"
BATCHED_META_NAME = "cache_meta.safetensors"


def cache_layers_for(num_layers: int, mode: str, interval: int) -> list[int]:
    """Resolve cache_layers config to a concrete list of layer indices.

    Supported specs:
    * ``all`` / empty       — every layer.
    * ``every_n`` / ``every_layer_interval`` — ``range(0, N, interval)`` (start-aligned).
    * ``every_<k>``         — ``range(0, N, k)`` (start-aligned).
    * ``block_<k>``         — ``range(k-1, N, k)`` (block-end aligned). Use this
                              when each k-layer group is one architectural unit
                              (e.g. 3 linear-attention + 1 full-attention) and
                              you want the cache checkpoint to land on the last
                              layer of each group, so layerwise blocks span the
                              whole unit instead of straddling boundaries.
    * literal list ``"[a,b,c]"`` or ``"a,b,c"``.
    """
    mode = (mode or "").strip().lower()
    if mode in ("all", ""):
        return list(range(num_layers))
    if mode in ("every_n", "every_layer_interval"):
        n = max(1, int(interval))
        return list(range(0, num_layers, n))
    m = re.fullmatch(r"every_(\d+)", mode)
    if m:
        n = max(1, int(m.group(1)))
        return list(range(0, num_layers, n))
    m = re.fullmatch(r"block_(\d+)", mode)
    if m:
        n = max(1, int(m.group(1)))
        return list(range(n - 1, num_layers, n))
    if mode.startswith("[") or "," in mode:
        try:
            ids = [int(x.strip()) for x in mode.strip("[]").split(",") if x.strip()]
            return sorted(set(ids))
        except ValueError as e:
            raise ValueError(f"unparseable cache_layers spec: {mode}") from e
    raise ValueError(f"unknown cache_layers spec: {mode}")


def cache_path(cache_dir: Path | str, sample_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", sample_id)
    return Path(cache_dir) / f"{safe}.safetensors"


def save_sample_cache(
    cache_dir: Path | str,
    sample_id: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hiddens: dict[int, torch.Tensor],
    routers: dict[int, torch.Tensor] | None,
    dtype: torch.dtype = torch.float16,
) -> Path:
    out = cache_path(cache_dir, sample_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, torch.Tensor] = {
        "input_ids": input_ids.detach().to(torch.int64).cpu().contiguous(),
        "attention_mask": attention_mask.detach().to(torch.int64).cpu().contiguous(),
    }
    for layer, t in hiddens.items():
        payload[f"hidden.layer_{layer}"] = t.detach().to(dtype).cpu().contiguous()
    if routers:
        for layer, t in routers.items():
            payload[f"router.layer_{layer}"] = t.detach().to(dtype).cpu().contiguous()
    save_file(payload, str(out))
    return out


def load_sample_cache(
    cache_dir: Path | str,
    sample_id: str,
    layers: Iterable[int] | None = None,
) -> dict[str, torch.Tensor | dict[int, torch.Tensor]]:
    """Load one cached sample. Returns dict with input_ids/attention_mask/hidden/router.

    Auto-dispatches between v2 batched layout (when ``cache_index.json``
    is present) and the legacy per-sample layout. The return dict has the
    same shape in both cases.
    """
    cache_dir = Path(cache_dir)
    if (cache_dir / BATCHED_INDEX_NAME).is_file():
        return _load_batched(cache_dir, sample_id, layers)
    p = cache_path(cache_dir, sample_id)
    if not p.is_file():
        raise FileNotFoundError(p)
    raw = load_file(str(p))
    hiddens: dict[int, torch.Tensor] = {}
    routers: dict[int, torch.Tensor] = {}
    layer_filter = set(int(x) for x in layers) if layers is not None else None
    for k, v in raw.items():
        m = _HIDDEN_RE.match(k)
        if m:
            li = int(m.group(1))
            if layer_filter is None or li in layer_filter:
                hiddens[li] = v
            continue
        m = _ROUTER_RE.match(k)
        if m:
            li = int(m.group(1))
            if layer_filter is None or li in layer_filter:
                routers[li] = v
            continue
    return {
        "input_ids": raw["input_ids"],
        "attention_mask": raw["attention_mask"],
        "hidden": hiddens,
        "router": routers,
    }


def cache_exists(cache_dir: Path | str, sample_id: str) -> bool:
    cache_dir = Path(cache_dir)
    if (cache_dir / BATCHED_INDEX_NAME).is_file():
        index = _read_batched_index(cache_dir)
        return sample_id in index.get("samples", {})
    return cache_path(cache_dir, sample_id).is_file()


def append_sample_cache(
    cache_dir: Path | str,
    sample_id: str,
    layer: int,
    hidden: torch.Tensor,
    router: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float16,
) -> Path:
    """Add one layer's tensors to a per-sample cache file.

    On first call ``input_ids`` and ``attention_mask`` must be provided so the
    file can be created; subsequent calls (which just merge a new layer onto
    an existing file) ignore them. The on-disk schema matches what
    :func:`save_sample_cache` produces, so a streamed file and a one-shot
    file are byte-for-byte interchangeable.

    Implementation note: on Windows the original file may still be memory-
    mapped while we read it, which blocks an in-place rewrite. We always
    materialise to a sibling temp file and ``replace`` it atomically.
    """
    out = cache_path(cache_dir, sample_id)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.is_file():
        loaded = load_file(str(out))
        merged: dict[str, torch.Tensor] = {k: v.clone().contiguous() for k, v in loaded.items()}
        del loaded
        merged[f"hidden.layer_{layer}"] = (
            hidden.detach().to(dtype).cpu().contiguous()
        )
        if router is not None:
            merged[f"router.layer_{layer}"] = (
                router.detach().to(dtype).cpu().contiguous()
            )
        tmp = out.with_suffix(out.suffix + ".tmp")
        save_file(merged, str(tmp))
        tmp.replace(out)
        return out

    if input_ids is None or attention_mask is None:
        raise ValueError(
            "input_ids and attention_mask are required when creating a new cache file"
        )
    return save_sample_cache(
        cache_dir,
        sample_id,
        input_ids,
        attention_mask,
        {layer: hidden},
        {layer: router} if router is not None else None,
        dtype=dtype,
    )


def parse_cache_dtype(name: str) -> torch.dtype:
    n = (name or "").lower()
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported cache dtype: {name}")


# =====================================================================
# v2 batched layout
# =====================================================================


def _chunk_filename(layer: int, chunk: int) -> str:
    return f"cache_layer_{int(layer)}_chunk_{int(chunk)}.safetensors"


def is_batched_cache(cache_dir: Path | str) -> bool:
    """Return True iff ``cache_dir`` holds a v2 batched layout."""
    return (Path(cache_dir) / BATCHED_INDEX_NAME).is_file()


# Module-level memo for the parsed index and open safetensors handles.
# Each DataLoader worker subprocess gets its own copies (forked / spawned).
# Within a worker we keep handles open across __getitem__ calls so mmap
# random access stays cheap.
_INDEX_MEMO: dict[str, dict[str, Any]] = {}
_INDEX_MEMO_LOCK = Lock()
_HANDLE_POOL: dict[tuple[str, str], Any] = {}
_HANDLE_POOL_LOCK = Lock()


def _read_batched_index(cache_dir: Path) -> dict[str, Any]:
    key = str(cache_dir.resolve())
    with _INDEX_MEMO_LOCK:
        cached = _INDEX_MEMO.get(key)
        if cached is not None:
            return cached
    raw = json.loads((cache_dir / BATCHED_INDEX_NAME).read_text(encoding="utf-8"))
    with _INDEX_MEMO_LOCK:
        _INDEX_MEMO[key] = raw
    return raw


def _invalidate_batched_index(cache_dir: Path) -> None:
    """Drop the parsed-index memo (and any open handles) for ``cache_dir``.

    Call this whenever an existing cache layout changes on disk during a
    Python session — primarily inside tests that rewrite the same dir.
    """
    key = str(cache_dir.resolve())
    with _INDEX_MEMO_LOCK:
        _INDEX_MEMO.pop(key, None)
    with _HANDLE_POOL_LOCK:
        for k in [hk for hk in _HANDLE_POOL if hk[0] == key]:
            _HANDLE_POOL.pop(k, None)


def _get_handle(cache_dir: Path, filename: str):
    """Open (or reuse) a ``safe_open`` handle for ``cache_dir/filename``."""
    key = (str(cache_dir.resolve()), filename)
    with _HANDLE_POOL_LOCK:
        h = _HANDLE_POOL.get(key)
        if h is not None:
            return h
    h = safe_open(str(cache_dir / filename), framework="pt", device="cpu")
    with _HANDLE_POOL_LOCK:
        # Race: another caller may have opened it concurrently. Keep one.
        existing = _HANDLE_POOL.get(key)
        if existing is not None:
            return existing
        _HANDLE_POOL[key] = h
    return h


def _load_batched(
    cache_dir: Path,
    sample_id: str,
    layers: Iterable[int] | None,
) -> dict[str, torch.Tensor | dict[int, torch.Tensor]]:
    index = _read_batched_index(cache_dir)
    samples = index.get("samples", {})
    if sample_id not in samples:
        raise FileNotFoundError(
            f"sample {sample_id} not in {cache_dir}/{BATCHED_INDEX_NAME}"
        )
    chunk = int(samples[sample_id]["chunk"])
    cache_layers_all = [int(li) for li in index.get("cache_layers", [])]
    layer_filter = (
        set(int(x) for x in layers) if layers is not None else set(cache_layers_all)
    )

    meta = _get_handle(cache_dir, BATCHED_META_NAME)
    meta_keys = set(meta.keys())
    input_ids = meta.get_tensor(f"{sample_id}.input_ids")
    attention_mask = meta.get_tensor(f"{sample_id}.attention_mask")
    inputs_embeds = (
        meta.get_tensor(f"{sample_id}.inputs_embeds")
        if f"{sample_id}.inputs_embeds" in meta_keys
        else None
    )
    position_ids_3d = (
        meta.get_tensor(f"{sample_id}.position_ids_3d")
        if f"{sample_id}.position_ids_3d" in meta_keys
        else None
    )

    hiddens: dict[int, torch.Tensor] = {}
    routers: dict[int, torch.Tensor] = {}
    for li in cache_layers_all:
        if li not in layer_filter:
            continue
        h = _get_handle(cache_dir, _chunk_filename(li, chunk))
        hiddens[li] = h.get_tensor(f"{sample_id}.hidden")
        router_key = f"{sample_id}.router"
        if router_key in h.keys():
            routers[li] = h.get_tensor(router_key)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "hidden": hiddens,
        "router": routers,
        "inputs_embeds": inputs_embeds,
        "position_ids_3d": position_ids_3d,
    }


class BatchedCacheWriter:
    """Layer-major, sample-chunked teacher cache writer.

    The streamer feeds tensors in the natural (layer-major, sample-major)
    order; this writer accumulates a chunk's worth in CPU RAM and flushes
    one safetensors per (layer, chunk) when the layer pass finishes. The
    chunking is sample-stable: sample i lives in chunk ``i // chunk_size``
    for **every** cached layer, so a reader can pick the right shard with
    a single integer divide.

    Memory peak per active layer ≈ ``chunk_size × (seq × hidden + seq ×
    num_experts) × dtype_bytes``. With 500 samples / 2048 seq / 4096 hidden
    / fp16 + 256 experts ≈ ~9.4 GB; tune ``chunk_size`` if your host RAM
    is tighter.

    Caller contract:

    1. Call :meth:`add_meta` once per sample (typically during the embed pass).
    2. Inside each cached layer's pass, call :meth:`add_layer_sample` per sample.
    3. Call :meth:`flush_chunk(layer, chunk_id)` after the chunk's samples are
       all buffered to write that chunk file and free its buffer; or fall back
       to :meth:`flush_layer(layer)` to drain every chunk for the layer in one
       call (kept as a safety net).
    4. After all layers are done, call :meth:`finalize` to write
       ``cache_meta.safetensors`` and ``cache_index.json``.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        sample_ids: Iterable[str],
        cache_layers: Iterable[int],
        num_experts: int,
        cache_dtype: torch.dtype = torch.float16,
        cache_router_logits: bool = True,
        chunk_size: int = 500,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_ids = list(sample_ids)
        self.cache_layers = sorted({int(li) for li in cache_layers})
        self.num_experts = int(num_experts)
        self.cache_dtype = cache_dtype
        self.cache_router_logits = bool(cache_router_logits)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.sid_to_chunk: dict[str, int] = {
            sid: idx // self.chunk_size for idx, sid in enumerate(self.sample_ids)
        }
        self.num_chunks = (
            (len(self.sample_ids) + self.chunk_size - 1) // self.chunk_size
            if self.sample_ids
            else 0
        )
        # in-flight per-layer buffers: (layer, chunk) -> {key: tensor}
        self._buffers: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        # accumulating per-sample meta tensors flushed by finalize()
        self._meta: dict[str, torch.Tensor] = {}

    # ---------- writer API ----------

    def add_meta(
        self,
        sample_id: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids_3d: torch.Tensor | None = None,
    ) -> None:
        """Stash a sample's tokens / mask for later inclusion in cache_meta.

        For VL samples, pass the merged ``inputs_embeds`` (text tokens already
        scatter-replaced with vision-tower outputs) and the (3, S)
        ``position_ids_3d`` so the layerwise trainer can resume from block 0
        without re-running the vision tower.
        """
        self._meta[f"{sample_id}.input_ids"] = (
            input_ids.detach().to(torch.int64).cpu().contiguous()
        )
        self._meta[f"{sample_id}.attention_mask"] = (
            attention_mask.detach().to(torch.int64).cpu().contiguous()
        )
        if inputs_embeds is not None:
            self._meta[f"{sample_id}.inputs_embeds"] = (
                inputs_embeds.detach().to(self.cache_dtype).cpu().contiguous()
            )
        if position_ids_3d is not None:
            self._meta[f"{sample_id}.position_ids_3d"] = (
                position_ids_3d.detach().to(torch.int64).cpu().contiguous()
            )

    def add_layer_sample(
        self,
        sample_id: str,
        layer: int,
        hidden: torch.Tensor,
        router: torch.Tensor | None = None,
    ) -> None:
        """Buffer one (sample, layer) pair until ``flush_layer`` is called."""
        if sample_id not in self.sid_to_chunk:
            raise KeyError(
                f"sample {sample_id!r} was not declared at writer init"
            )
        chunk = self.sid_to_chunk[sample_id]
        buf = self._buffers.setdefault((int(layer), chunk), {})
        buf[f"{sample_id}.hidden"] = (
            hidden.detach().to(self.cache_dtype).cpu().contiguous()
        )
        if self.cache_router_logits and router is not None:
            buf[f"{sample_id}.router"] = (
                router.detach().to(self.cache_dtype).cpu().contiguous()
            )

    def flush_chunk(self, layer: int, chunk_id: int) -> None:
        """Write one ``(layer, chunk_id)`` file and drop its buffer.

        Lets callers free RAM as soon as a chunk's samples are all buffered,
        instead of holding every chunk for the layer until end-of-layer.
        Idempotent: re-calling for the same pair after the buffer was drained
        is a no-op.
        """
        layer = int(layer)
        chunk_id = int(chunk_id)
        buf = self._buffers.pop((layer, chunk_id), None)
        if not buf:
            return
        self._write_chunk(layer, chunk_id, buf)
        buf.clear()

    def flush_layer(self, layer: int) -> None:
        """Write every chunk file for ``layer`` and drop its buffers.

        Idempotent when called twice (second pass writes nothing because
        the buffers were already drained).
        """
        layer = int(layer)
        for chunk_id in range(self.num_chunks):
            buf = self._buffers.pop((layer, chunk_id), None)
            if not buf:
                continue
            self._write_chunk(layer, chunk_id, buf)
            buf.clear()

    def finalize(self) -> None:
        """Flush any remaining buffers, write meta + index. Idempotent."""
        # Defensive: drain any leftover buffers (caller forgot flush_layer).
        for (layer, chunk_id), buf in list(self._buffers.items()):
            if buf:
                self._write_chunk(layer, chunk_id, buf)
            self._buffers.pop((layer, chunk_id), None)

        if self._meta:
            meta_path = self.cache_dir / BATCHED_META_NAME
            tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
            save_file(self._meta, str(tmp))
            tmp.replace(meta_path)

        index = {
            "version": 2,
            "cache_layers": list(self.cache_layers),
            "num_experts": self.num_experts,
            "chunk_size": self.chunk_size,
            "num_chunks": self.num_chunks,
            "cache_router_logits": self.cache_router_logits,
            "samples": {sid: {"chunk": ch} for sid, ch in self.sid_to_chunk.items()},
        }
        index_path = self.cache_dir / BATCHED_INDEX_NAME
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _invalidate_batched_index(self.cache_dir)

    # ---------- internals ----------

    def _write_chunk(
        self, layer: int, chunk_id: int, buf: dict[str, torch.Tensor]
    ) -> None:
        path = self.cache_dir / _chunk_filename(layer, chunk_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        save_file(buf, str(tmp))
        tmp.replace(path)


__all__ = [
    "BATCHED_INDEX_NAME",
    "BATCHED_META_NAME",
    "BatchedCacheWriter",
    "append_sample_cache",
    "cache_exists",
    "cache_layers_for",
    "cache_path",
    "is_batched_cache",
    "load_sample_cache",
    "parse_cache_dtype",
    "save_sample_cache",
]
