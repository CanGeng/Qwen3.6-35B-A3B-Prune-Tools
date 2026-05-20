"""Cosine / linear / constant LR schedule with warmup, as a single LambdaLR.

Endpoints (post-warmup, with ``progress = (step - warmup) / (total - warmup)``):

* cosine:   λ(0)=0, λ(warmup)=1, λ(total) = min_lr_ratio
* linear:   λ(0)=0, λ(warmup)=1, λ(total) = min_lr_ratio  (linear decay)
* constant: λ(0)=0, λ(warmup)=1, λ(total) = 1
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


_VALID_TYPES = ("cosine", "linear", "constant")


def build_scheduler(
    optimizer: Optimizer,
    *,
    type_: str = "cosine",
    num_warmup: int = 0,
    num_training: int = 1,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    if type_ not in _VALID_TYPES:
        raise ValueError(f"lr_scheduler type must be one of {_VALID_TYPES}, got {type_!r}")
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")
    num_warmup = max(0, int(num_warmup))
    num_training = max(1, int(num_training))

    def lr_lambda(step: int) -> float:
        if num_warmup > 0 and step < num_warmup:
            return float(step) / float(max(1, num_warmup))
        if type_ == "constant":
            return 1.0
        denom = max(1, num_training - num_warmup)
        progress = float(step - num_warmup) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        if type_ == "cosine":
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        # linear
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optimizer, lr_lambda)


__all__ = ["build_scheduler"]
