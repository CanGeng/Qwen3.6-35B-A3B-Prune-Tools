"""Batch collation with padding.

Handles both pure-text SFT samples (``input_ids``/``attention_mask``/
``labels``) and VL samples (those plus ``pixel_values``/``image_grid_thw``/
``mm_token_type_ids``/``position_ids_3d``). VL fields are padded in lockstep
with text fields and concatenated along the image axis where appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase


def _has_vl(features: list[dict]) -> bool:
    return any(f.get("pixel_values") is not None for f in features)


def _pad_position_ids_3d(
    pos3d: list[list[int]] | None,
    max_len: int,
) -> list[list[int]]:
    """Pad a per-sample (3, S) position table to ``max_len`` along the seq axis."""
    if pos3d is None:
        return [list(range(max_len)) for _ in range(3)]
    out: list[list[int]] = []
    for axis in pos3d:
        a = list(axis)
        if len(a) < max_len:
            tail_val = a[-1] + 1 if a else 0
            a = a + [tail_val] * (max_len - len(a))
        out.append(a[:max_len])
    return out


@dataclass
class SFTCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        assert pad_id is not None

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        batch_input: list[list[int]] = []
        batch_attn: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for f in features:
            ids = list(f["input_ids"])
            attn = list(f["attention_mask"])
            lab = list(f["labels"])
            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [pad_id] * pad_len
                attn = attn + [0] * pad_len
                lab = lab + [-100] * pad_len
            batch_input.append(ids)
            batch_attn.append(attn)
            batch_labels.append(lab)

        out: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

        if _has_vl(features):
            pixel_chunks: list[torch.Tensor] = []
            grid_chunks: list[torch.Tensor] = []
            mm_type_rows: list[list[int]] = []
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
                mm_type_rows.append(mm[:max_len])
                pos3d_rows.append(
                    _pad_position_ids_3d(f.get("position_ids_3d"), max_len)
                )
            if pixel_chunks:
                out["pixel_values"] = torch.cat(pixel_chunks, dim=0).contiguous()
            if grid_chunks:
                out["image_grid_thw"] = torch.cat(grid_chunks, dim=0).to(torch.long).contiguous()
            out["mm_token_type_ids"] = torch.tensor(mm_type_rows, dtype=torch.long)
            # position_ids stacks to (B, 3, S); the model side reshapes to
            # (3, B, S) when invoking the rotary embedding, matching HF's
            # ``compute_3d_position_ids`` output convention.
            out["position_ids_3d"] = torch.tensor(pos3d_rows, dtype=torch.long)

        return out
