"""PyTorch Dataset for JSONL chat/text samples."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from moe_prune_distill.data.schema import TrainSample
from moe_prune_distill.data.vl_processor import build_vl_inputs, build_vl_inputs_lazy


_log = logging.getLogger("moe_prune_distill.dataset")


def is_val_id(sid: str, val_split: float) -> bool:
    """Deterministic id-hash partition: ``True`` when this id falls in the val slice.

    Stable across runs and independent of sample order so a re-run with a
    different ``max_samples`` still yields the same split for the same ids.
    """
    if val_split <= 0.0:
        return False
    h = int(hashlib.sha1(sid.encode("utf-8")).hexdigest()[:8], 16)
    return (h / 0xFFFFFFFF) < val_split


class JsonlSFTDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        max_samples: int | None = None,
        *,
        split: Literal["train", "val", "all"] = "all",
        val_split: float = 0.0,
        teacher_dir: str | Path | None = None,
        vl_lazy_pixels: bool = False,
    ) -> None:
        if split not in ("train", "val", "all"):
            raise ValueError(f"split must be train|val|all, got {split!r}")
        if not (0.0 <= val_split < 0.5):
            raise ValueError(f"val_split must be in [0, 0.5), got {val_split}")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.val_split = float(val_split)
        # Used by the VL path to look up the matching image processor.
        # Pure-text-only callers can leave this as None.
        self.teacher_dir = str(teacher_dir) if teacher_dir is not None else None
        # Lazy-pixel mode: VL samples return ``image_paths`` instead of an
        # eager ``pixel_values`` tensor, deferring image_processor work to the
        # consumer (LayerStreamer._embed_pass). Keeps CPU RAM bounded when the
        # whole dataset is materialised upfront (372k VL samples × multi-MB
        # pixel_values would otherwise be TB-scale).
        self.vl_lazy_pixels = bool(vl_lazy_pixels)
        loaded: list[TrainSample] = []
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row: dict[str, Any] = json.loads(line)
                loaded.append(TrainSample.from_dict(row))
                if max_samples is not None and len(loaded) >= max_samples:
                    break

        if split == "all" or self.val_split <= 0.0:
            kept = loaded
        elif split == "val":
            kept = [s for s in loaded if is_val_id(str(s.id), self.val_split)]
        else:  # train
            kept = [s for s in loaded if not is_val_id(str(s.id), self.val_split)]
        self.samples = kept

        ids = [s.id for s in self.samples]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate sample ids in training file")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]

        # VL path: sample has images attached. Pixel tensors and 3D mrope
        # position_ids come back alongside the text fields.
        if s.images:
            if self.teacher_dir is None:
                raise RuntimeError(
                    f"sample {s.id!r} has images but JsonlSFTDataset was constructed "
                    "without teacher_dir; pass teacher_dir=... so the image processor "
                    "can be loaded"
                )
            assert s.messages is not None
            builder = build_vl_inputs_lazy if self.vl_lazy_pixels else build_vl_inputs
            vl = builder(
                s.messages,
                list(s.images),
                self.teacher_dir,
                self.max_seq_len,
            )
            if vl is None:
                # Fall back to a tiny placeholder so the DataLoader doesn't
                # crash on a single bad row. We log once per failure.
                _log.warning(
                    "vl_processor returned None for sample %s; emitting empty SFT row",
                    s.id,
                )
                pad_id = self.tokenizer.pad_token_id or 0
                return {
                    "id": s.id,
                    "input_ids": [int(pad_id)],
                    "attention_mask": [0],
                    "labels": [-100],
                }
            return {"id": s.id, **vl}

        if s.text is not None:
            enc = self.tokenizer(
                s.text,
                truncation=True,
                max_length=self.max_seq_len,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            labels = list(input_ids)
            return {"id": s.id, "input_ids": input_ids, "attention_mask": attn, "labels": labels}

        assert s.messages is not None
        try:
            enc = self.tokenizer.apply_chat_template(
                s.messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=self.max_seq_len,
                return_assistant_tokens_mask=True,
            )
            assistant_mask = enc.get("assistant_masks")
        except TypeError:
            enc = self.tokenizer.apply_chat_template(
                s.messages,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=self.max_seq_len,
            )
            assistant_mask = None

        input_ids = enc["input_ids"]
        if not isinstance(input_ids, list):
            input_ids = list(input_ids)
        attn = enc.get("attention_mask")
        if attn is None:
            attn = [1] * len(input_ids)
        elif not isinstance(attn, list):
            attn = list(attn)

        labels: list[int]
        if assistant_mask is not None:
            if not isinstance(assistant_mask, list):
                assistant_mask = list(assistant_mask)
            labels = [
                tid if m else -100 for tid, m in zip(input_ids, assistant_mask, strict=True)
            ]
        else:
            assistant_text = next(
                (m["content"] for m in reversed(s.messages) if m["role"] == "assistant"),
                "",
            )
            a_enc = self.tokenizer(assistant_text, add_special_tokens=False)
            a_ids = a_enc["input_ids"]
            n = min(len(a_ids), len(input_ids))
            labels = [-100] * len(input_ids)
            if n > 0:
                labels[-n:] = input_ids[-n:]

        return {"id": s.id, "input_ids": input_ids, "attention_mask": attn, "labels": labels}
