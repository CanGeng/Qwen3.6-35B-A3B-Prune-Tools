"""Distill training step + helpers (called from scripts/train.py)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from moe_prune_distill.distill.losses import normalized_hidden_mse, router_kl, sft_ce


def load_expert_mapping(student_dir: str | Path) -> dict[int, list[int]]:
    """Read expert_mapping.json into {layer -> surviving_original_ids}."""
    p = Path(student_dir) / "expert_mapping.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[int, list[int]] = {}
    for k, v in raw.items():
        if not k.startswith("layer_"):
            continue
        layer = int(k.split("_")[-1])
        out[layer] = list(v.get("surviving_original_ids", []))
    return out


def _hf_hidden_to_dict(hidden_states, layers: list[int]) -> dict[int, torch.Tensor]:
    """Convert HF model hidden_states tuple (L+1) into {layer -> post-block tensor}."""
    if hidden_states is None:
        return {}
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        idx = layer + 1
        if idx < len(hidden_states):
            out[layer] = hidden_states[idx]
    return out


def _hf_router_to_dict(router_logits, layers: list[int]) -> dict[int, torch.Tensor]:
    if router_logits is None:
        return {}
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        if layer < len(router_logits):
            r = router_logits[layer]
            if isinstance(r, torch.Tensor) and r.ndim >= 2:
                out[layer] = r
    return out


def _restore_router_batch(rl: torch.Tensor, batch: int, seq: int) -> torch.Tensor:
    """HF often returns router logits as [B*T, E]; reshape to [B, T, E] when possible."""
    if rl.dim() == 3:
        return rl
    if rl.dim() == 2 and rl.shape[0] == batch * seq:
        return rl.view(batch, seq, rl.shape[-1])
    return rl


def compute_distill_loss(
    student_out,
    batch: dict[str, Any],
    cache_layers: list[int],
    surviving_by_layer: dict[int, list[int]],
    weights: dict[str, float],
    hidden_layer_weighting: str,
    router_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine hidden MSE + router KL + SFT CE for one student forward output."""
    components: dict[str, float] = {}
    total: torch.Tensor | None = None

    attn = batch.get("attention_mask")

    w_sft = float(weights.get("sft_ce", 0.0))
    if w_sft > 0 and getattr(student_out, "logits", None) is not None:
        ce = sft_ce(student_out.logits, batch["labels"])
        components["sft_ce"] = float(ce.detach().cpu())
        total = ce * w_sft if total is None else total + ce * w_sft

    teacher_hidden: dict[int, torch.Tensor] = batch.get("teacher_hidden") or {}
    w_h = float(weights.get("hidden_mse", 0.0))
    if w_h > 0 and teacher_hidden:
        s_hidden = _hf_hidden_to_dict(getattr(student_out, "hidden_states", None), cache_layers)
        if s_hidden:
            mse = normalized_hidden_mse(
                s_hidden,
                teacher_hidden,
                weighting=hidden_layer_weighting,
                attention_mask=attn,
            )
            components["hidden_mse"] = float(mse.detach().cpu())
            total = mse * w_h if total is None else total + mse * w_h

    teacher_router: dict[int, torch.Tensor] = batch.get("teacher_router") or {}
    w_r = float(weights.get("router_kl", 0.0))
    if w_r > 0 and teacher_router and surviving_by_layer:
        s_router_raw = getattr(student_out, "router_logits", None)
        s_router = _hf_router_to_dict(s_router_raw, list(teacher_router.keys()))
        if s_router:
            B, T = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
            s_router = {k: _restore_router_batch(v, B, T) for k, v in s_router.items()}
            kl = router_kl(
                s_router,
                teacher_router,
                surviving_by_layer,
                temperature=router_temperature,
                attention_mask=attn,
            )
            components["router_kl"] = float(kl.detach().cpu())
            total = kl * w_r if total is None else total + kl * w_r

    if total is None:
        # nothing enabled -> fall back to plain CE so training does not no-op
        ce = sft_ce(student_out.logits, batch["labels"])
        components["sft_ce"] = float(ce.detach().cpu())
        total = ce
    return total, components
