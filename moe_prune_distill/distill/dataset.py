"""Combined distill dataset/collator: student inputs + lazy teacher cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from moe_prune_distill.data.dataset import JsonlSFTDataset
from moe_prune_distill.distill.teacher_cache import cache_exists, load_sample_cache


class DistillJsonlDataset(Dataset):
    """Wrap JsonlSFTDataset and attach the teacher cache for the same id."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        cache_dir: str | Path,
        cache_layers: list[int] | None,
        max_samples: int | None = None,
        require_cache: bool = True,
        *,
        split: Literal["train", "val", "all"] = "all",
        val_split: float = 0.0,
        teacher_dir: str | Path | None = None,
    ) -> None:
        self.base = JsonlSFTDataset(
            path,
            tokenizer,
            max_seq_len,
            max_samples,
            split=split,
            val_split=val_split,
            teacher_dir=teacher_dir,
        )
        self.cache_dir = Path(cache_dir)
        self.cache_layers = list(cache_layers) if cache_layers is not None else None
        self.require_cache = require_cache
        if require_cache:
            missing = [s.id for s in self.base.samples if not cache_exists(self.cache_dir, s.id)]
            if missing:
                raise FileNotFoundError(
                    f"Teacher cache missing for {len(missing)} samples (e.g. {missing[:3]}); "
                    f"run scripts/cache_teacher.py first or set require_cache=False"
                )

    @property
    def samples(self):
        return self.base.samples

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base[idx]
        sid = item["id"]
        teacher_input_ids = None
        teacher_hidden: dict[int, torch.Tensor] = {}
        teacher_router: dict[int, torch.Tensor] = {}
        if cache_exists(self.cache_dir, sid):
            cache = load_sample_cache(self.cache_dir, sid, layers=self.cache_layers)
            teacher_input_ids = cache["input_ids"]
            teacher_hidden = cache["hidden"]  # type: ignore[assignment]
            teacher_router = cache["router"]  # type: ignore[assignment]
        item["teacher_input_ids"] = teacher_input_ids
        item["teacher_hidden"] = teacher_hidden
        item["teacher_router"] = teacher_router
        return item


def collate_distill(
    features: list[dict],
    pad_id: int,
) -> dict[str, Any]:
    """Pad student tensors and stack teacher caches per layer (right-padded)."""
    max_len = max(len(f["input_ids"]) for f in features)

    def _pad_int(seq: list[int], pad: int) -> list[int]:
        return list(seq) + [pad] * (max_len - len(seq))

    input_ids = torch.tensor(
        [_pad_int(f["input_ids"], pad_id) for f in features], dtype=torch.long
    )
    attn = torch.tensor(
        [_pad_int(f["attention_mask"], 0) for f in features], dtype=torch.long
    )
    labels = torch.tensor(
        [_pad_int(f["labels"], -100) for f in features], dtype=torch.long
    )

    layers = sorted({l for f in features for l in (f.get("teacher_hidden") or {}).keys()})
    t_hidden_per_layer: dict[int, torch.Tensor | None] = {}
    for layer in layers:
        per_sample: list[torch.Tensor] = []
        ok = True
        for f in features:
            h = (f.get("teacher_hidden") or {}).get(layer)
            if h is None:
                ok = False
                break
            if h.dim() == 2:
                pad_len = max_len - h.shape[0]
                if pad_len > 0:
                    pad = torch.zeros(pad_len, h.shape[1], dtype=h.dtype)
                    h = torch.cat([h, pad], dim=0)
                per_sample.append(h)
        t_hidden_per_layer[layer] = torch.stack(per_sample, dim=0) if ok else None

    layers_r = sorted({l for f in features for l in (f.get("teacher_router") or {}).keys()})
    t_router_per_layer: dict[int, torch.Tensor | None] = {}
    for layer in layers_r:
        per_sample = []
        ok = True
        for f in features:
            r = (f.get("teacher_router") or {}).get(layer)
            if r is None:
                ok = False
                break
            if r.dim() == 2:
                pad_len = max_len - r.shape[0]
                if pad_len > 0:
                    pad = torch.zeros(pad_len, r.shape[1], dtype=r.dtype)
                    r = torch.cat([r, pad], dim=0)
                per_sample.append(r)
        t_router_per_layer[layer] = torch.stack(per_sample, dim=0) if ok else None

    out: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "teacher_hidden": {k: v for k, v in t_hidden_per_layer.items() if v is not None},
        "teacher_router": {k: v for k, v in t_router_per_layer.items() if v is not None},
    }

    # VL passthrough — same logic as SFTCollator. Only emit these keys when
    # at least one sample carries pixel data, so pure-text batches stay text.
    has_vl = any(f.get("pixel_values") is not None for f in features)
    if has_vl:
        pixel_chunks: list[torch.Tensor] = []
        grid_chunks: list[torch.Tensor] = []
        mm_rows: list[list[int]] = []
        pos3d_rows: list[list[list[int]]] = []
        for f in features:
            pv = f.get("pixel_values")
            gthw = f.get("image_grid_thw")
            if pv is not None:
                pixel_chunks.append(pv)
            if gthw is not None:
                grid_chunks.append(gthw)
            mm = f.get("mm_token_type_ids")
            if mm is None:
                mm = [0] * len(f["input_ids"])
            else:
                mm = list(mm)
            if len(mm) < max_len:
                mm = mm + [0] * (max_len - len(mm))
            mm_rows.append(mm[:max_len])
            pos3d = f.get("position_ids_3d")
            if pos3d is None:
                pos3d_rows.append([list(range(max_len)) for _ in range(3)])
            else:
                padded: list[list[int]] = []
                for axis in pos3d:
                    a = list(axis)
                    if len(a) < max_len:
                        tail = a[-1] + 1 if a else 0
                        a = a + [tail] * (max_len - len(a))
                    padded.append(a[:max_len])
                pos3d_rows.append(padded)
        if pixel_chunks:
            out["pixel_values"] = torch.cat(pixel_chunks, dim=0).contiguous()
        if grid_chunks:
            out["image_grid_thw"] = (
                torch.cat(grid_chunks, dim=0).to(torch.long).contiguous()
            )
        out["mm_token_type_ids"] = torch.tensor(mm_rows, dtype=torch.long)
        out["position_ids_3d"] = torch.tensor(pos3d_rows, dtype=torch.long)

    return out
