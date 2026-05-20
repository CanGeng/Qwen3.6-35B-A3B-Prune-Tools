"""Unit tests for distill/lr_scheduler.py."""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from moe_prune_distill.distill.lr_scheduler import build_scheduler


def _fresh_opt(lr: float = 1.0):
    p = torch.nn.Parameter(torch.zeros(2, requires_grad=True))
    opt = torch.optim.SGD([p], lr=lr)
    p.grad = torch.zeros_like(p)
    opt.step()
    return opt


@pytest.fixture(autouse=True)
def _silence_step_order_warning():
    # We invoke scheduler.step() repeatedly to walk the schedule; the per-iter
    # optimizer.step() is irrelevant to the test and adds noise.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*lr_scheduler\.step\(\) before `optimizer\.step\(\).*",
        )
        yield


def test_warmup_endpoint_one():
    opt = _fresh_opt()
    sched = build_scheduler(opt, type_="cosine", num_warmup=10, num_training=100, min_lr_ratio=0.1)
    for _ in range(10):
        sched.step()
    assert math.isclose(sched.get_last_lr()[0], 1.0, rel_tol=1e-6)


def test_cosine_endpoint_min():
    opt = _fresh_opt()
    sched = build_scheduler(opt, type_="cosine", num_warmup=5, num_training=20, min_lr_ratio=0.1)
    for _ in range(20):
        sched.step()
    assert math.isclose(sched.get_last_lr()[0], 0.1, rel_tol=1e-6)


def test_linear_endpoint_min():
    opt = _fresh_opt()
    sched = build_scheduler(opt, type_="linear", num_warmup=5, num_training=20, min_lr_ratio=0.2)
    for _ in range(20):
        sched.step()
    assert math.isclose(sched.get_last_lr()[0], 0.2, rel_tol=1e-6)


def test_constant_post_warmup_one():
    opt = _fresh_opt()
    sched = build_scheduler(opt, type_="constant", num_warmup=3, num_training=30, min_lr_ratio=0.1)
    for _ in range(15):
        sched.step()
    assert math.isclose(sched.get_last_lr()[0], 1.0, rel_tol=1e-6)


def test_warmup_zero_no_division_error():
    opt = _fresh_opt()
    sched = build_scheduler(opt, type_="cosine", num_warmup=0, num_training=10, min_lr_ratio=0.0)
    sched.step()
    # progress = 1/10, lambda must be > 0 and <= 1
    assert 0 < sched.get_last_lr()[0] <= 1.0


def test_cosine_midpoint_close_to_half():
    opt = _fresh_opt()
    # warmup=0 so progress=t/N at step t
    sched = build_scheduler(opt, type_="cosine", num_warmup=0, num_training=10, min_lr_ratio=0.0)
    for _ in range(5):
        sched.step()
    # cos(pi/2) = 0 => lambda = 0.5
    assert math.isclose(sched.get_last_lr()[0], 0.5, rel_tol=1e-3)


def test_invalid_type_raises():
    opt = _fresh_opt()
    with pytest.raises(ValueError):
        build_scheduler(opt, type_="exponential", num_warmup=0, num_training=1, min_lr_ratio=0.1)


def test_invalid_min_lr_raises():
    opt = _fresh_opt()
    with pytest.raises(ValueError):
        build_scheduler(opt, type_="cosine", num_warmup=0, num_training=1, min_lr_ratio=1.5)
