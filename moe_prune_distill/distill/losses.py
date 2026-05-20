"""Distillation losses: hidden MSE, router KL, SFT CE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _layer_weights(num: int, scheme: str) -> list[float]:
    if num <= 0:
        return []
    if scheme == "uniform":
        return [1.0 / num] * num
    if scheme == "linear_deeper_more":
        raw = [float(i + 1) for i in range(num)]
        s = sum(raw)
        return [r / s for r in raw]
    raise ValueError(f"unknown hidden_layer_weighting: {scheme}")


def _align_shapes(s: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Make student/teacher hidden tensors broadcastable as [B, T, H]."""
    if s.dim() == 2:
        s = s.unsqueeze(0)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    if s.shape[0] != t.shape[0]:
        if t.shape[0] == 1:
            t = t.expand_as(s)
        elif s.shape[0] == 1:
            s = s.expand_as(t)
    return s, t


def normalized_hidden_mse(
        student_hiddens: dict[int, torch.Tensor],
        teacher_hiddens: dict[int, torch.Tensor],
        weighting: str = "linear_deeper_more",
        cos_weight: float = 1.0,
        norm_weight: float = 0.00,
        attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Decoupled Distillation Loss:
    Separates direction (Cosine distance) and magnitude (Norm MSE) to prevent norm inflation.
    """
    common = sorted(set(student_hiddens.keys()) & set(teacher_hiddens.keys()))
    if not common:
        any_t = next(iter(student_hiddens.values()), None)
        device = any_t.device if any_t is not None else torch.device("cpu")
        return torch.zeros((), device=device)
    weights = _layer_weights(len(common), weighting)
    total: torch.Tensor | None = None

    for w, layer in zip(weights, common):
        s = student_hiddens[layer]
        t = teacher_hiddens[layer].to(device=s.device, dtype=s.dtype)
        s, t = _align_shapes(s, t)

        # 必须转为 float 计算对齐，防溢出
        s_float = s.float()
        t_float = t.float()
        # 1. 核心方向损失: 1 - Cosine Similarity
        cos_dist = 1.0 - F.cosine_similarity(s_float, t_float, dim=-1)  # [B, T]
        # 2. 软模长约束损失: (||S||_2 - ||T||_2)^2
        if norm_weight > 0.0:
            # 使用 torch.linalg.vector_norm 是当前 PyTorch 的推荐规范用法
            s_norm = torch.linalg.vector_norm(s_float, dim=-1)
            t_norm = torch.linalg.vector_norm(t_float, dim=-1)
            norm_diff = (s_norm - t_norm).pow(2)  # [B, T]
        else:
            norm_diff = 0.0
        # 解耦组合
        diff = (cos_weight * cos_dist) + (norm_weight * norm_diff)
        # 掩码计算
        if attention_mask is not None:
            m = attention_mask.to(device=diff.device, dtype=diff.dtype)
            if m.dim() == 1:
                m = m.unsqueeze(0)
            denom = m.sum().clamp_min(1.0)
            layer_loss = (diff * m).sum() / denom
        else:
            layer_loss = diff.mean()

        contrib = layer_loss.to(s.dtype) * w
        total = contrib if total is None else total + contrib

    assert total is not None
    return total

def router_kl(
    student_router_logits: dict[int, torch.Tensor],
    teacher_router_logits: dict[int, torch.Tensor],
    surviving_by_layer: dict[int, list[int]],
    temperature: float = 2.0,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Token-level KL between student router and teacher router restricted to surviving experts."""
    common = sorted(set(student_router_logits.keys()) & set(teacher_router_logits.keys()))
    if not common:
        any_t = next(iter(student_router_logits.values()), None)
        device = any_t.device if any_t is not None else torch.device("cpu")
        return torch.zeros((), device=device)

    total: torch.Tensor | None = None
    n_layers = 0
    for layer in common:
        surv = surviving_by_layer.get(layer)
        if not surv:
            continue
        s = student_router_logits[layer]
        t = teacher_router_logits[layer].to(device=s.device, dtype=s.dtype)
        idx = torch.tensor(surv, device=t.device, dtype=torch.long)
        t_sel = t.index_select(dim=-1, index=idx)
        s, t_sel = _align_shapes(s, t_sel)
        if t_sel.shape[-1] != s.shape[-1]:
            raise ValueError(
                f"router shape mismatch layer {layer}: student E={s.shape[-1]}, "
                f"teacher (sliced) E={t_sel.shape[-1]}"
            )
        log_p_s = F.log_softmax(s.float() / temperature, dim=-1)
        p_t = F.softmax(t_sel.float() / temperature, dim=-1)
        kl = (p_t * (p_t.clamp_min(1e-9).log() - log_p_s)).sum(dim=-1)  # [B, T]
        if attention_mask is not None:
            m = attention_mask.to(device=kl.device, dtype=kl.dtype)
            if m.dim() == 1:
                m = m.unsqueeze(0)
            denom = m.sum().clamp_min(1.0)
            layer_kl = (kl * m).sum() / denom
        else:
            layer_kl = kl.mean()
        contrib = (layer_kl * (temperature ** 2)).to(s.dtype)
        total = contrib if total is None else total + contrib
        n_layers += 1
    if total is None or n_layers == 0:
        any_t = next(iter(student_router_logits.values()))
        return torch.zeros((), device=any_t.device, dtype=any_t.dtype)
    return total / n_layers


def sft_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Causal LM cross-entropy with -100 masking."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
