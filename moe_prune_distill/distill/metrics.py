"""Diagnostic-only metrics for distill training.

Pure no-grad helpers. Each returns a ``dict[str, float]`` and silently omits
keys whose inputs are empty so the JSONL row reflects what was actually
available (presence/absence is itself a signal).

These intentionally duplicate a small amount of math from ``losses.py``: the
loss path is already in the autograd graph, and we don't want diagnostics to
hold extra references or interact with grad scaling.
"""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn.functional as F

from moe_prune_distill.distill.losses import _align_shapes


_EPS = 1e-8


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Average ``x`` ([B, T] or [B, T, *]) over the mask; falls back to plain mean."""
    if mask is None:
        return x.mean()
    m = mask.to(device=x.device, dtype=x.dtype)
    if m.dim() == 1:
        m = m.unsqueeze(0)
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    denom = m.sum().clamp_min(1.0)
    return (x * m).sum() / denom


@torch.no_grad()
def hidden_metrics(
    student_hiddens: Dict[int, torch.Tensor],
    teacher_hiddens: Dict[int, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Per-layer diagnostics over matched hidden states.

    Returns ``{hidden_mse, nmse, cos_loss, teacher_norm, student_norm}``.
    All averages reduce uniformly across layers (diagnostic, not loss).
    """
    common = sorted(set(student_hiddens.keys()) & set(teacher_hiddens.keys()))
    if not common:
        return {}

    hmse_sum = 0.0
    nmse_sum = 0.0
    cos_sum = 0.0
    s_norm_sum = 0.0
    t_norm_sum = 0.0
    n = 0
    for layer in common:
        s = student_hiddens[layer]
        t = teacher_hiddens[layer].to(device=s.device, dtype=s.dtype)
        s, t = _align_shapes(s, t)
        s_f = s.float()
        t_f = t.float()

        s_n = F.normalize(s_f, dim=-1)
        t_n = F.normalize(t_f, dim=-1)
        diff_norm_sq = (s_n - t_n).pow(2).sum(dim=-1)        # [B, T]
        diff_sq = (s_f - t_f).pow(2).sum(dim=-1)             # [B, T]
        t_sq = t_f.pow(2).sum(dim=-1)                        # [B, T]
        cos = (s_n * t_n).sum(dim=-1)                        # [B, T]
        s_norm = s_f.norm(dim=-1)                            # [B, T]
        t_norm = t_f.norm(dim=-1)                            # [B, T]

        hmse_sum += float(_masked_mean(diff_norm_sq, attention_mask))
        nmse_sum += float(_masked_mean(diff_sq / (t_sq + _EPS), attention_mask))
        cos_sum += float(_masked_mean(cos, attention_mask))
        s_norm_sum += float(_masked_mean(s_norm, attention_mask))
        t_norm_sum += float(_masked_mean(t_norm, attention_mask))
        n += 1

    inv = 1.0 / n
    return {
        "hidden_mse": hmse_sum * inv,
        "nmse": nmse_sum * inv,
        "cos_loss": 1.0 - cos_sum * inv,
        "teacher_norm": t_norm_sum * inv,
        "student_norm": s_norm_sum * inv,
    }


@torch.no_grad()
def router_diagnostics(
    student_router_logits: Dict[int, torch.Tensor],
    teacher_router_logits: Dict[int, torch.Tensor] | None,
    surviving_by_layer: Dict[int, list[int]] | None,
    attention_mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Student router entropy + teacher mass on already-pruned experts.

    * ``router_entropy`` averaged across layers in ``student_router_logits``.
      Computed on raw logits (no temperature) so it reflects what the model
      actually emits.
    * ``removed_expert_mass`` requires ``teacher_router_logits[layer]`` AND
      ``surviving_by_layer[layer]``; layers without both are skipped.

    Either key is omitted from the result dict if no layer contributed to it.
    """
    out: Dict[str, float] = {}

    # router entropy (student, raw logits)
    if student_router_logits:
        ent_sum = 0.0
        n = 0
        for layer, s in student_router_logits.items():
            if not isinstance(s, torch.Tensor) or s.dim() < 2:
                continue
            p = F.softmax(s.float(), dim=-1)
            h = -(p * p.clamp_min(1e-9).log()).sum(dim=-1)   # [B, T] or [N]
            ent_sum += float(_masked_mean(h, attention_mask))
            n += 1
        if n:
            out["router_entropy"] = ent_sum / n

    # teacher mass on pruned experts
    if teacher_router_logits and surviving_by_layer:
        mass_sum = 0.0
        n = 0
        for layer, t in teacher_router_logits.items():
            surv = surviving_by_layer.get(layer)
            if not surv:
                continue
            if not isinstance(t, torch.Tensor) or t.dim() < 2:
                continue
            idx = torch.tensor(surv, device=t.device, dtype=torch.long)
            p = F.softmax(t.float(), dim=-1)
            kept = p.index_select(dim=-1, index=idx).sum(dim=-1)  # [B, T]
            removed = 1.0 - kept
            mass_sum += float(_masked_mean(removed, attention_mask))
            n += 1
        if n:
            out["removed_expert_mass"] = mass_sum / n

    return out


@torch.no_grad()
def batch_token_stats(attention_mask: torch.Tensor) -> Dict[str, float]:
    """``valid_tokens`` (int sum) and ``mean_seq_len`` (per-sample mean of mask sum)."""
    if attention_mask is None:
        return {}
    m = attention_mask
    if m.dim() == 1:
        m = m.unsqueeze(0)
    valid = int(m.sum().item())
    per_sample = m.sum(dim=-1).float()
    mean_seq = float(per_sample.mean().item())
    return {"valid_tokens": valid, "mean_seq_len": mean_seq}


__all__ = ["hidden_metrics", "router_diagnostics", "batch_token_stats"]
