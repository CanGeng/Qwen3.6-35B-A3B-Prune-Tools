"""Muon optimizer with Triton-accelerated Newton-Schulz5 orthogonalization.

Two variants:

* :class:`Muon`        — per-param NS5 using a symmetry-aware ``X @ X.T``
                         Triton kernel (``mmt_kernel``).
* :class:`MuonBatched` — buckets matrix params by shape and runs a batched
                         NS5 (``bmmt_kernel``) so kernel-launch overhead is
                         amortized across many same-shape MoE expert
                         matrices.

Both classes carry an internal foreach AdamW branch for 1D / embedding-like params
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch
import triton
import triton.language as tl


# ====================================================================
# Triton kernels: symmetric A = X @ X.T (and batched flavor)
# ====================================================================


def _autotune_configs():
    return [
        triton.Config(
            {"BLOCK_SIZE_M": blk_m, "BLOCK_SIZE_K": blk_k, "GROUP_SIZE_M": 8},
            num_stages=ns,
            num_warps=nw,
        )
        for blk_m in (32, 64, 128)
        for blk_k in (32, 64)
        for ns in (3, 4, 5)
        for nw in (4, 8)
    ]


@triton.autotune(configs=_autotune_configs(), key=["M", "K"])
@triton.jit
def mmt_kernel(
    x, y, M, K,
    stride_xm, stride_xk, stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Y = X @ X.T (M x M) for X (M x K). Computes lower triangle, mirrors to upper."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if pid_m > pid_n:
        return

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = x + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = x + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        b = tl.load(b_ptrs, mask=mask, other=0.0)
        accumulator = tl.dot(a, tl.permute(b, (1, 0)), accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_xk

    c = accumulator.to(x.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)

    if pid_m < pid_n:
        ct_ptrs = y + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)


def _matmul_transpose_assign(d_in: torch.Tensor, d_out: torch.Tensor) -> None:
    d_in = d_in.contiguous()
    M, K = d_in.shape
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),
    )
    mmt_kernel[grid](
        d_in, d_out, M, K,
        d_in.stride(0), d_in.stride(1),
        d_out.stride(0), d_out.stride(1),
    )


@triton.autotune(configs=_autotune_configs(), key=["M", "K"])
@triton.jit
def bmmt_kernel(
    x_ptr, y_ptr, M, K,
    stride_xb, stride_xm, stride_xk,
    stride_yb, stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Batched Y = X @ X.T  for X[B, M, K]."""
    pid_batch = tl.program_id(axis=1)
    x_ptr += pid_batch * stride_xb
    y_ptr += pid_batch * stride_yb

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if pid_m < pid_n:
        return

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = x_ptr + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        b = tl.load(b_ptrs, mask=mask, other=0.0)
        accumulator = tl.dot(a, tl.permute(b, (1, 0)), accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_xk

    c = accumulator.to(x_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = y_ptr + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)

    if pid_m > pid_n:
        ct_ptrs = y_ptr + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)


def _batched_matmul_transpose(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    B, M, K = x.shape
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        B,
    )
    bmmt_kernel[grid](
        x, y, M, K,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
    )
    return y


# ====================================================================
# NS5 wrappers (Triton inner mat-mul)
# ====================================================================

# Polar Express coefficients (Amsel et al. 2025).
_NS_ABC = (3.4445, -4.7750, 2.0315)


@torch.compile
def zeropower_via_newtonschulz5_triton(
    G: torch.Tensor,
    steps: int = 5,
    *,
    buf1: torch.Tensor | None = None,
    buf2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-matrix NS5 with symmetric Triton mat-mul on the inner ``X @ X.T``.

    ``buf1`` / ``buf2`` are scratch buffers sized ``(M, M)`` where M = min(d_out, d_in).
    Pass them in to avoid allocating fresh ones every call (they're big enough
    that letting the caching allocator churn them creates fragmentation).
    Caller is responsible for shape/dtype/device matching X after the
    "transpose if M > N" path; if either is None it's allocated locally.
    """
    assert G.ndim == 2
    a, b, c = _NS_ABC
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    M, _N = X.shape
    if buf1 is None:
        buf1 = torch.empty((M, M), dtype=X.dtype, device=X.device)
    if buf2 is None:
        buf2 = torch.empty((M, M), dtype=X.dtype, device=X.device)
    for _ in range(steps):
        _matmul_transpose_assign(X, buf1)
        _matmul_transpose_assign(buf1, buf2)
        B = b * buf1 + c * buf2
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.mT
    return X


@torch.compile
def zeropower_via_newtonschulz5_batched_triton(
    G: torch.Tensor,
    steps: int = 5,
    *,
    buf1: torch.Tensor | None = None,
    buf2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched NS5 over a stack of same-shape matrices ``G[B, M, N]``.

    ``buf1`` / ``buf2`` are scratch buffers sized ``(B, size_a, size_a)`` where
    ``size_a = min(M, N)``. Same opt-in pooling story as the per-matrix kernel.
    """
    a, b, c = _NS_ABC
    X = G.bfloat16()
    Batch, M, N = G.shape
    transposed = M > N
    if transposed:
        X = X.mT
        size_a = N
    else:
        size_a = M

    if buf1 is None:
        buf1 = torch.empty((Batch, size_a, size_a), device=X.device, dtype=X.dtype)
    if buf2 is None:
        buf2 = torch.empty((Batch, size_a, size_a), device=X.device, dtype=X.dtype)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        _batched_matmul_transpose(X, buf1)
        _batched_matmul_transpose(buf1, buf2)
        buf1 = b * buf1 + c * buf2
        X = a * X + buf1 @ X

    if transposed:
        X = X.mT
    return X


def newton_schulz(G: torch.Tensor, steps: int) -> torch.Tensor:
    return zeropower_via_newtonschulz5_triton(G, steps)


# ====================================================================
# Optimizer base bits: foreach AdamW step (shared across both classes)
# ====================================================================


def _foreach_adamw_step(
    params: list[torch.nn.Parameter],
    state_lookup,
    *,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    params_with_grad: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    exp_avgs: list[torch.Tensor] = []
    exp_avg_sqs: list[torch.Tensor] = []
    state_steps: list[int] = []

    for p in params:
        if p.grad is None:
            continue
        if p.grad.is_sparse:
            raise RuntimeError("AdamW branch does not support sparse gradients")
        st = state_lookup(p)
        if len(st) == 0:
            st["step"] = 0
            st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            st["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        st["step"] += 1
        params_with_grad.append(p)
        grads.append(p.grad)
        exp_avgs.append(st["exp_avg"])
        exp_avg_sqs.append(st["exp_avg_sq"])
        state_steps.append(st["step"])

    if not params_with_grad:
        return

    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

    step = state_steps[0]
    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step
    step_size = lr / bc1
    bc2_sqrt = math.sqrt(bc2)

    denoms = torch._foreach_sqrt(exp_avg_sqs)
    if bc2_sqrt != 1.0:
        torch._foreach_div_(denoms, bc2_sqrt)
    torch._foreach_add_(denoms, eps)

    if weight_decay != 0:
        torch._foreach_add_(params_with_grad, params_with_grad, alpha=-lr * weight_decay)
    torch._foreach_addcdiv_(params_with_grad, exp_avgs, denoms, value=-step_size)


def _adjust_lr_for_muon(lr: float, shape) -> float:
    """μP-style scaling: lr · 0.2 · √max(d_out, d_in).

    Uses the trailing two dims so 3D expert stacks ``[E, M, N]`` and 4D conv
    weights are scaled by the matrix dims, not the leading batch / channel
    dim that isn't part of the matrix being orthogonalized.
    """
    A, B = shape[-2], shape[-1]
    return lr * 0.2 * math.sqrt(max(A, B))


# ====================================================================
# Scratch + momentum pooling (shared between Muon and MuonBatched)
# ====================================================================

# NS5 needs two ``(B, size_a, size_a)`` workspaces every call. With routed-
# expert stacks ``[128, 1024, 2048]`` that's 256 MB per buffer; reallocating
# one per call per param fragments the caching allocator and inflates peak
# residency. Pool them on the optimizer instance, keyed by shape+dtype+device.
#
# When ``paged_momentum`` is on, the same pool also serves staging tensors
# for 3D expert momentum: the buffer is sized like the param, used in-step
# as a GPU staging area, then released back. Momentum itself lives in host
# pinned memory between steps — see ``_alloc_momentum``.


class _BufPool:
    """Reusable GPU scratch buffers keyed by (shape, dtype, device)."""

    def __init__(self) -> None:
        self._cache: dict[tuple, torch.Tensor] = {}

    @staticmethod
    def _key(shape, dtype: torch.dtype, device: torch.device, tag: str) -> tuple:
        di = device.index if device.index is not None else 0
        return (tuple(shape), dtype, device.type, di, tag)

    def get(
        self,
        shape,
        dtype: torch.dtype,
        device: torch.device,
        *,
        tag: str = "x",
    ) -> torch.Tensor:
        key = self._key(shape, dtype, device, tag)
        buf = self._cache.get(key)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._cache[key] = buf
        return buf

    def get_pair(
        self,
        shape,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.get(shape, dtype, device, tag="ns_a"),
            self.get(shape, dtype, device, tag="ns_b"),
        )


def _alloc_momentum(
    p: torch.Tensor,
    *,
    paged: bool,
    log_warn,
) -> torch.Tensor:
    """Allocate a momentum buffer mirroring ``p.grad``.

    ``paged=False`` → plain GPU clone of ``p.grad`` (current behavior).
    ``paged=True``  → host pinned tensor; per-step we stage to GPU and back.
                      Falls back to GPU clone if pin_memory fails.
    """
    g = p.grad
    if not paged:
        return torch.clone(g).detach()
    try:
        host = torch.empty(g.shape, dtype=g.dtype, device="cpu", pin_memory=True)
        host.copy_(g.detach(), non_blocking=True)
        return host
    except Exception as e:
        log_warn(f"paged_momentum: pin_memory unavailable ({e}); using GPU buffer")
        return torch.clone(g).detach()


# ====================================================================
# Muon (per-param NS, symmetric Triton kernel)
# ====================================================================


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-Schulz (Triton symmetric kernel).

    Args:
        muon_params:   matrix params. ``ndim == 2`` matrices use the per-param
                       NS5 kernel; ``ndim == 3`` expert stacks ``[E, M, N]``
                       are routed through the batched NS5 kernel directly.
                       Anything else falls through to AdamW.
        adamw_params:  fallback params (1D, embeddings, etc.).
        lr, weight_decay, momentum, nesterov: standard SGD-with-momentum knobs;
                       lr applies to both branches.
        ns_steps:      Newton-Schulz iterations per step (5 ≈ paper default).
        adamw_betas, adamw_eps: AdamW second-moment params.
    """

    def __init__(
        self,
        muon_params: Iterable[torch.nn.Parameter],
        adamw_params: Iterable[torch.nn.Parameter] = (),
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        paged_momentum: bool = False,
    ) -> None:
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)
        real_muon, real_adamw = [], list(adamw_params)
        for p in muon_params:
            (real_muon if p.ndim in (2, 3) else real_adamw).append(p)

        groups = [
            dict(
                params=real_muon,
                lr=lr, weight_decay=weight_decay,
                momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            ),
            dict(
                params=real_adamw,
                lr=lr, weight_decay=weight_decay,
                betas=adamw_betas, eps=adamw_eps,
            ),
        ]
        super().__init__(groups, {})
        self._buf_pool = _BufPool()
        self._paged_momentum = bool(paged_momentum)
        self._paged_warned = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._muon_step(self.param_groups[0])
        self._adamw_step(self.param_groups[1])
        return loss

    def _warn_once(self, msg: str) -> None:
        if not self._paged_warned:
            import logging
            logging.getLogger("moe_prune_distill.muon").warning(msg)
            self._paged_warned = True

    def _muon_step(self, group: dict) -> None:
        lr = group["lr"]
        wd = group["weight_decay"]
        mom = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            if g.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients")
            if not torch.isfinite(g).all():
                # Skip the step on NaN / inf grads — preserve momentum buffer
                # so the next clean step recovers.
                continue

            st = self.state[p]
            # ``paged_momentum`` keeps the master momentum on host pinned RAM
            # for 3D expert stacks (the only place big enough to matter); 2D
            # params stay on GPU since the saving doesn't justify the H2D/D2H.
            paged_this = self._paged_momentum and p.ndim == 3
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = _alloc_momentum(
                    p, paged=paged_this, log_warn=self._warn_once
                )
            host_or_gpu_buf = st["momentum_buffer"]

            if paged_this and host_or_gpu_buf.device.type == "cpu":
                gpu_buf = self._buf_pool.get(
                    p.shape, host_or_gpu_buf.dtype, p.device, tag="mom_stage"
                )
                gpu_buf.copy_(host_or_gpu_buf, non_blocking=True)
                buf = gpu_buf
            else:
                buf = host_or_gpu_buf

            buf.mul_(mom).add_(g)
            g_eff = g.add(buf, alpha=mom) if nesterov else buf

            if g_eff.ndim == 2:
                M = min(g_eff.size(0), g_eff.size(1))
                ns_buf1, ns_buf2 = self._buf_pool.get_pair(
                    (M, M), torch.bfloat16, p.device
                )
                u = zeropower_via_newtonschulz5_triton(
                    g_eff, steps=ns_steps, buf1=ns_buf1, buf2=ns_buf2,
                )
            else:  # ndim == 3 — expert stack
                B, M, N = g_eff.shape
                size_a = min(M, N)
                ns_buf1, ns_buf2 = self._buf_pool.get_pair(
                    (B, size_a, size_a), torch.bfloat16, p.device
                )
                u = zeropower_via_newtonschulz5_batched_triton(
                    g_eff.bfloat16().contiguous(),
                    steps=ns_steps,
                    buf1=ns_buf1, buf2=ns_buf2,
                ).to(p.dtype)
            adj_lr = _adjust_lr_for_muon(lr, p.shape)
            p.add_(p, alpha=-lr * wd)
            p.add_(u, alpha=-adj_lr)

            if paged_this and host_or_gpu_buf.device.type == "cpu":
                # Flush staged momentum back to host pinned memory so it
                # survives until the next step.
                host_or_gpu_buf.copy_(buf, non_blocking=True)

    def _adamw_step(self, group: dict) -> None:
        _foreach_adamw_step(
            group["params"], lambda p: self.state[p],
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            eps=group["eps"],
        )


# ====================================================================
# MuonBatched (group same-shape matrices, batched NS5)
# ====================================================================


class MuonBatched(torch.optim.Optimizer):
    """Muon variant that buckets matrix params by shape and runs batched NS5.

    Useful when many MoE expert matrices share the same ``[d_out, d_in]``
    so kernel-launch overhead is amortized across them. Two intake shapes
    are handled:

    * ``ndim == 2`` — buckets up to ``bucket_size`` like-shaped tensors and
      stacks them into ``[B, M, N]`` before the batched NS5 kernel.
    * ``ndim == 3`` — already-stacked expert weights ``[E, M, N]`` (e.g.
      ``mlp.experts.gate_up_proj``); fed straight into the batched kernel
      without re-bucketing, since the leading dim is the expert batch.

    Everything else falls through to the AdamW branch.
    """

    def __init__(
        self,
        muon_params: Iterable[torch.nn.Parameter],
        adamw_params: Iterable[torch.nn.Parameter] = (),
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        bucket_size: int = 16,
        paged_momentum: bool = False,
    ) -> None:
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)
        real_muon, real_adamw = [], list(adamw_params)
        for p in muon_params:
            (real_muon if p.ndim in (2, 3) else real_adamw).append(p)

        groups = [
            dict(
                params=real_muon,
                lr=lr, weight_decay=weight_decay,
                momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                bucket_size=bucket_size,
            ),
            dict(
                params=real_adamw,
                lr=lr, weight_decay=weight_decay,
                betas=adamw_betas, eps=adamw_eps,
            ),
        ]
        super().__init__(groups, {})

        # Bucket only 2D params by shape; 3D params run one-at-a-time through
        # the batched kernel (their leading dim is already the expert batch).
        self.bucketed_indices: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.stacked_3d_indices: list[int] = []
        for idx, p in enumerate(self.param_groups[0]["params"]):
            if p.ndim == 2:
                self.bucketed_indices[tuple(p.shape)].append(idx)
            else:  # ndim == 3
                self.stacked_3d_indices.append(idx)

        self._buf_pool = _BufPool()
        self._paged_momentum = bool(paged_momentum)
        self._paged_warned = False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._muon_step(self.param_groups[0])
        self._adamw_step(self.param_groups[1])
        return loss

    def _warn_once(self, msg: str) -> None:
        if not self._paged_warned:
            import logging
            logging.getLogger("moe_prune_distill.muon").warning(msg)
            self._paged_warned = True

    def _muon_step(self, group: dict) -> None:
        params = group["params"]
        lr = group["lr"]
        wd = group["weight_decay"]
        mom = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        bucket_size = group["bucket_size"]
        if not params:
            return

        # ---- 2D path: bucket like-shaped matrices, stack, batched NS5 ----
        for shape, indices in self.bucketed_indices.items():
            M, N = shape
            adj_lr = _adjust_lr_for_muon(lr, shape)
            stacked = self._buf_pool.get(
                (bucket_size, M, N), torch.bfloat16, params[0].device, tag="bucket"
            )
            size_a = min(M, N)
            for i in range(0, len(indices), bucket_size):
                chunk_idx = indices[i : i + bucket_size]
                chunk_params: list[torch.Tensor] = []
                chunk_grads: list[torch.Tensor] = []
                chunk_bufs: list[torch.Tensor] = []
                for idx in chunk_idx:
                    p = params[idx]
                    if p.grad is None or not torch.isfinite(p.grad).all():
                        continue
                    chunk_params.append(p)
                    chunk_grads.append(p.grad)
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        # 2D momentum stays on GPU regardless of paged flag —
                        # it's small (router gate, attn proj LoRA bases) and
                        # the H2D/D2H roundtrip would dominate the step.
                        st["momentum_buffer"] = torch.clone(p.grad).detach()
                    chunk_bufs.append(st["momentum_buffer"])
                if not chunk_params:
                    continue

                torch._foreach_mul_(chunk_bufs, mom)
                torch._foreach_add_(chunk_bufs, chunk_grads)
                if nesterov:
                    grads_for_ns = torch._foreach_add(chunk_grads, chunk_bufs, alpha=mom)
                else:
                    grads_for_ns = chunk_bufs

                bsz = len(chunk_params)
                torch.stack(grads_for_ns, dim=0, out=stacked[:bsz])
                ns_buf1, ns_buf2 = self._buf_pool.get_pair(
                    (bsz, size_a, size_a), torch.bfloat16, params[0].device
                )
                updates = zeropower_via_newtonschulz5_batched_triton(
                    stacked[:bsz],
                    steps=ns_steps,
                    buf1=ns_buf1, buf2=ns_buf2,
                )
                updates_list = list(torch.unbind(updates, dim=0))

                torch._foreach_add_(chunk_params, chunk_params, alpha=-lr * wd)
                torch._foreach_add_(chunk_params, updates_list, alpha=-adj_lr)

        # ---- 3D path: each param is already [E, M, N]; one batched NS5 call ----
        for idx in self.stacked_3d_indices:
            p = params[idx]
            if p.grad is None or not torch.isfinite(p.grad).all():
                continue
            adj_lr = _adjust_lr_for_muon(lr, p.shape)
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = _alloc_momentum(
                    p, paged=self._paged_momentum, log_warn=self._warn_once
                )
            host_or_gpu_buf = st["momentum_buffer"]

            paged_this = (
                self._paged_momentum and host_or_gpu_buf.device.type == "cpu"
            )
            if paged_this:
                gpu_buf = self._buf_pool.get(
                    p.shape, host_or_gpu_buf.dtype, p.device, tag="mom_stage"
                )
                gpu_buf.copy_(host_or_gpu_buf, non_blocking=True)
                buf = gpu_buf
            else:
                buf = host_or_gpu_buf

            buf.mul_(mom).add_(p.grad)
            g_eff = p.grad.add(buf, alpha=mom) if nesterov else buf

            B, M, N = g_eff.shape
            size_a = min(M, N)
            ns_buf1, ns_buf2 = self._buf_pool.get_pair(
                (B, size_a, size_a), torch.bfloat16, p.device
            )
            update = zeropower_via_newtonschulz5_batched_triton(
                g_eff.bfloat16().contiguous(),
                steps=ns_steps,
                buf1=ns_buf1, buf2=ns_buf2,
            )
            p.add_(p, alpha=-lr * wd)
            p.add_(update.to(p.dtype), alpha=-adj_lr)

            if paged_this:
                host_or_gpu_buf.copy_(buf, non_blocking=True)

    def _adamw_step(self, group: dict) -> None:
        _foreach_adamw_step(
            group["params"], lambda p: self.state[p],
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            beta1=group["betas"][0],
            beta2=group["betas"][1],
            eps=group["eps"],
        )


# ====================================================================
# Convenience: partition named params into (muon, adamw) lists
# ====================================================================


_ADAMW_NAME_HINTS = (
    "embed_tokens",
    "lm_head",
    "embedding",
    "embeddings",
    "wte",
    "wpe",
    "lora_A",
    "lora_B",
)


def partition_for_muon(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    adamw_name_hints: tuple[str, ...] = _ADAMW_NAME_HINTS,
    min_3d_inner_dim: int = 8,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split named params into (muon_matrix, adamw_rest).

    Routing rules (in order):

    * Names matching any of ``adamw_name_hints`` (embeddings, lm_head, …) → AdamW.
    * 2D / 4D params → Muon.
    * 3D params with both inner dims ≥ ``min_3d_inner_dim`` → Muon. This catches
      MoE expert stacks ``[num_experts, d_out, d_in]`` (the dominant param mass
      in MoE models) while excluding depthwise conv1d-style weights
      ``[C, 1, K]`` whose NS5 update would be degenerate.
    * Everything else → AdamW.
    """
    muon: list[torch.nn.Parameter] = []
    adamw: list[torch.nn.Parameter] = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        blacklisted = any(h in name for h in adamw_name_hints)
        if blacklisted:
            adamw.append(p)
            continue
        if p.ndim in (2, 4):
            muon.append(p)
        elif p.ndim == 3 and min(p.shape[1], p.shape[2]) >= min_3d_inner_dim:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


__all__ = [
    "Muon",
    "MuonBatched",
    "newton_schulz",
    "partition_for_muon",
    "zeropower_via_newtonschulz5_triton",
    "zeropower_via_newtonschulz5_batched_triton",
]
