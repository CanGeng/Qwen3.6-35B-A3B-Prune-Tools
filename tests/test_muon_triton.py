"""Tests for the Triton-backed Muon optimizer.

Since the Triton kernels require CUDA + a Triton install, we gate the
end-to-end optimizer tests on both. The pure-Python pieces — partition
helper and learning-rate adjustment — run on CPU regardless.
"""

from __future__ import annotations

import pytest
import torch

triton = pytest.importorskip("triton")

from moe_prune_distill.distill.muon_triton import (
    Muon,
    MuonBatched,
    _adjust_lr_for_muon,
    partition_for_muon,
)


# ====================================================================
# 1. partition_for_muon: routing logic (CPU-only, no Triton needed)
# ====================================================================


def test_partition_for_muon_routes_2d_to_muon() -> None:
    lin = torch.nn.Linear(8, 16)             # weight 2D, bias 1D
    emb = torch.nn.Embedding(32, 16)         # named 'embeddings.weight'
    norm = torch.nn.LayerNorm(16)            # 1D weight, 1D bias

    named = [
        ("matrix.weight", lin.weight),
        ("matrix.bias", lin.bias),
        ("embed_tokens.weight", emb.weight),
        ("ln.weight", norm.weight),
        ("ln.bias", norm.bias),
    ]

    muon, adamw = partition_for_muon(named)
    # Identity comparison: ``in`` on a list of tensors invokes __eq__ which is
    # element-wise, so we check by identity.
    def has(lst, tensor):
        return any(p is tensor for p in lst)

    assert has(muon, lin.weight)
    assert has(adamw, lin.bias)         # 1D bias
    assert has(adamw, emb.weight)       # name hint -> AdamW
    assert has(adamw, norm.weight)      # 1D
    assert has(adamw, norm.bias)        # 1D


def test_partition_skips_frozen_params() -> None:
    w = torch.nn.Parameter(torch.randn(4, 4), requires_grad=False)
    muon, adamw = partition_for_muon([("frozen.weight", w)])
    assert all(p is not w for p in muon)
    assert all(p is not w for p in adamw)


def test_partition_routes_3d_moe_expert_stacks_to_muon() -> None:
    """MoE expert stacks ``[E, M, N]`` (the dominant param mass in MoE
    models) must go through Muon, not AdamW. Conv1d-style ``[C, 1, K]``
    weights, whose NS5 update is degenerate, must stay on AdamW."""
    expert_gate_up = torch.nn.Parameter(torch.randn(128, 1024, 2048))   # 3D, min_inner=1024
    expert_down = torch.nn.Parameter(torch.randn(128, 2048, 512))       # 3D, min_inner=512
    conv1d_w = torch.nn.Parameter(torch.randn(8192, 1, 4))              # 3D, min_inner=1
    router_gate = torch.nn.Parameter(torch.randn(128, 2048))            # 2D
    norm_w = torch.nn.Parameter(torch.randn(2048))                      # 1D

    named = [
        ("mlp.experts.gate_up_proj", expert_gate_up),
        ("mlp.experts.down_proj", expert_down),
        ("linear_attn.conv1d.weight", conv1d_w),
        ("mlp.gate.weight", router_gate),
        ("input_layernorm.weight", norm_w),
    ]
    muon, adamw = partition_for_muon(named)

    def has(lst, t):
        return any(p is t for p in lst)

    assert has(muon, expert_gate_up)
    assert has(muon, expert_down)
    assert has(muon, router_gate)
    assert has(adamw, conv1d_w)         # 3D but inner dim 1 -> AdamW
    assert has(adamw, norm_w)


def test_adjust_lr_scales_by_sqrt_max_dim() -> None:
    # Paper's μP rule: lr · 0.2 · √max(M, N)
    lr = 1e-3
    out = _adjust_lr_for_muon(lr, (16, 64))
    expected = lr * 0.2 * (64 ** 0.5)
    assert abs(out - expected) < 1e-9


def test_adjust_lr_uses_trailing_dims_for_3d() -> None:
    """For 3D expert stacks, the leading dim is the expert batch — only the
    trailing two dims govern the spectral scale."""
    lr = 1e-3
    out = _adjust_lr_for_muon(lr, (128, 1024, 2048))
    expected = lr * 0.2 * (2048 ** 0.5)
    assert abs(out - expected) < 1e-9


# ====================================================================
# 2. Construction sanity (CPU; no .step() call which would touch Triton)
# ====================================================================


def test_muon_construction_partitions_internally() -> None:
    """``Muon`` should re-route 1D params handed to ``muon_params`` over to
    the AdamW branch on its own."""
    w2 = torch.nn.Parameter(torch.randn(8, 16))
    b1 = torch.nn.Parameter(torch.randn(8))
    opt = Muon([w2, b1], [], lr=1e-3)
    assert any(p is w2 for p in opt.param_groups[0]["params"])
    assert any(p is b1 for p in opt.param_groups[1]["params"])


def test_muon_batched_construction_buckets_by_shape() -> None:
    a = torch.nn.Parameter(torch.randn(4, 8))
    b = torch.nn.Parameter(torch.randn(4, 8))
    c = torch.nn.Parameter(torch.randn(8, 16))
    opt = MuonBatched([a, b], [], lr=1e-3, bucket_size=4)
    # Single shape, one bucket of 2
    opt2 = MuonBatched([a, b, c], [], lr=1e-3, bucket_size=4)
    assert len(opt2.bucketed_indices) == 2
    assert sorted(len(v) for v in opt2.bucketed_indices.values()) == [1, 2]
    assert opt2.stacked_3d_indices == []


def test_muon_batched_separates_3d_from_2d_buckets() -> None:
    """3D expert stacks register in ``stacked_3d_indices``, not
    ``bucketed_indices`` — they're already a stack and skip the 2D bucketing
    layer."""
    w2 = torch.nn.Parameter(torch.randn(4, 8))
    expert = torch.nn.Parameter(torch.randn(16, 32, 64))   # 3D
    opt = MuonBatched([w2, expert], [], lr=1e-3, bucket_size=4)
    # Only the 2D param goes into shape-bucketed indices
    assert list(opt.bucketed_indices.keys()) == [(4, 8)]
    assert len(opt.bucketed_indices[(4, 8)]) == 1
    # The 3D stack is registered separately
    assert len(opt.stacked_3d_indices) == 1


# ====================================================================
# 3. End-to-end smoke (CUDA + Triton required)
# ====================================================================


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)


@cuda_only
def test_muon_overfits_linear_regression() -> None:
    """Run a tiny linear regression and check loss drops at least 4×.

    Muon's μP-scaled update has spectral norm ≈ ``lr · 0.2 · √max(M, N)``,
    so on a single-sample regression the per-step move is conservative;
    we check a 4× drop over 500 steps rather than the 10× we'd see with
    AdamW + tuned lr.
    """
    torch.manual_seed(0)
    device = "cuda"
    lin = torch.nn.Linear(16, 8, bias=True).to(device)
    opt = Muon(
        [lin.weight], [lin.bias],
        lr=5e-3, momentum=0.9, ns_steps=5, weight_decay=0.0,
    )
    target = torch.randn(8, device=device)
    x = torch.randn(16, device=device)
    losses: list[float] = []
    for _ in range(500):
        loss = ((lin(x) - target) ** 2).mean()
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0] / 4, (
        f"loss did not drop 4×: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )


@cuda_only
def test_muon_batched_runs_step_on_same_shape_stack() -> None:
    """Two same-shape matrices share one batched NS5 launch via bucketing."""
    torch.manual_seed(0)
    device = "cuda"
    a = torch.nn.Parameter(torch.randn(8, 16, device=device))
    b = torch.nn.Parameter(torch.randn(8, 16, device=device))
    opt = MuonBatched([a, b], [], lr=1e-3, bucket_size=4, ns_steps=4)
    a.grad = torch.randn_like(a)
    b.grad = torch.randn_like(b)
    a_before = a.detach().clone()
    b_before = b.detach().clone()
    opt.step()
    # Both should have moved
    assert not torch.allclose(a, a_before)
    assert not torch.allclose(b, b_before)


@cuda_only
def test_muon_skips_nan_grad_and_keeps_state() -> None:
    """A NaN grad should not corrupt the momentum buffer or write NaN to params."""
    torch.manual_seed(0)
    device = "cuda"
    p = torch.nn.Parameter(torch.randn(8, 16, device=device))
    opt = Muon([p], [], lr=1e-3, momentum=0.9, ns_steps=4)

    p.grad = torch.full_like(p, float("nan"))
    saved = p.detach().clone()
    opt.step()
    assert torch.isfinite(p).all()
    assert torch.allclose(p, saved), "param should be unchanged on NaN grad"


@cuda_only
def test_muon_step_on_3d_expert_stack() -> None:
    """3D MoE expert stacks ``[E, M, N]`` should run through the batched NS5
    kernel directly (Muon path)."""
    torch.manual_seed(0)
    device = "cuda"
    p = torch.nn.Parameter(torch.randn(8, 32, 16, device=device, dtype=torch.bfloat16))
    opt = Muon([p], [], lr=1e-3, momentum=0.9, ns_steps=4, weight_decay=0.0)
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    opt.step()
    assert torch.isfinite(p).all()
    assert not torch.allclose(p, before)


@cuda_only
def test_muon_batched_step_on_3d_expert_stack() -> None:
    """3D expert stacks must also work through MuonBatched without going to
    AdamW. Mirrors the production layout where ``mlp.experts.gate_up_proj``
    is a single ``[E, d_out, d_in]`` parameter."""
    torch.manual_seed(0)
    device = "cuda"
    expert = torch.nn.Parameter(
        torch.randn(8, 32, 16, device=device, dtype=torch.bfloat16)
    )
    w2 = torch.nn.Parameter(torch.randn(16, 32, device=device, dtype=torch.bfloat16))
    opt = MuonBatched([expert, w2], [], lr=1e-3, bucket_size=4, ns_steps=4,
                     weight_decay=0.0)
    expert.grad = torch.randn_like(expert)
    w2.grad = torch.randn_like(w2)
    expert_before = expert.detach().clone()
    w2_before = w2.detach().clone()
    opt.step()
    assert torch.isfinite(expert).all()
    assert torch.isfinite(w2).all()
    assert not torch.allclose(expert, expert_before)
    assert not torch.allclose(w2, w2_before)


# ====================================================================
# 4. Buffer pooling (Plan A) — opt-in NS5 scratch reuse
# ====================================================================


def test_buf_pool_returns_same_tensor_for_repeated_key() -> None:
    """Same (shape, dtype, device, tag) → same tensor object across calls."""
    from moe_prune_distill.distill.muon_triton import _BufPool

    pool = _BufPool()
    a1 = pool.get((4, 8), torch.float32, torch.device("cpu"), tag="x")
    a2 = pool.get((4, 8), torch.float32, torch.device("cpu"), tag="x")
    assert a1 is a2
    b = pool.get((4, 8), torch.float32, torch.device("cpu"), tag="y")
    assert b is not a1   # different tag → different buffer


def test_buf_pool_get_pair_returns_two_distinct_buffers() -> None:
    from moe_prune_distill.distill.muon_triton import _BufPool

    pool = _BufPool()
    p1, p2 = pool.get_pair((4, 4), torch.bfloat16, torch.device("cpu"))
    assert p1 is not p2
    p1b, p2b = pool.get_pair((4, 4), torch.bfloat16, torch.device("cpu"))
    assert p1 is p1b and p2 is p2b   # same call returns same pair


@cuda_only
def test_ns5_kernel_accepts_external_buffers() -> None:
    """Plan A: passing pre-allocated buf1/buf2 must produce the same result as
    the legacy zero-arg path."""
    from moe_prune_distill.distill.muon_triton import (
        zeropower_via_newtonschulz5_batched_triton as f,
    )

    torch.manual_seed(0)
    device = "cuda"
    G = torch.randn(4, 32, 16, device=device, dtype=torch.bfloat16)
    out_default = f(G, steps=4)
    buf1 = torch.empty((4, 16, 16), device=device, dtype=torch.bfloat16)
    buf2 = torch.empty((4, 16, 16), device=device, dtype=torch.bfloat16)
    out_pooled = f(G, steps=4, buf1=buf1, buf2=buf2)
    # Bit-exact: same kernel, same inputs, same accumulator order.
    assert torch.equal(out_default, out_pooled)


@cuda_only
def test_muon_batched_buf_pool_reused_across_steps() -> None:
    """Plan A end-to-end: after two optimizer.step() calls, the pool should
    hold a single NS5 scratch pair (not have grown one per step)."""
    torch.manual_seed(0)
    device = "cuda"
    p = torch.nn.Parameter(
        torch.randn(8, 32, 16, device=device, dtype=torch.bfloat16)
    )
    opt = MuonBatched([p], [], lr=1e-3, ns_steps=4, weight_decay=0.0)
    p.grad = torch.randn_like(p)
    opt.step()
    pool_after_1 = dict(opt._buf_pool._cache)
    p.grad = torch.randn_like(p)
    opt.step()
    pool_after_2 = dict(opt._buf_pool._cache)
    # Same keys both times — pool didn't grow on the second step.
    assert set(pool_after_1.keys()) == set(pool_after_2.keys())
    # And each cached buffer is the same object (no replacement).
    for k in pool_after_1:
        assert pool_after_1[k] is pool_after_2[k]


# ====================================================================
# 5. Paged momentum (Plan B) — opt-in host pinned momentum staging
# ====================================================================


@cuda_only
def test_muon_batched_paged_momentum_keeps_host_buffer() -> None:
    """With ``paged_momentum=True`` on a 3D expert stack, the persistent
    momentum_buffer in optimizer state must live on CPU (pinned) and the
    GPU staging buffer must come from the pool."""
    torch.manual_seed(0)
    device = "cuda"
    p = torch.nn.Parameter(
        torch.randn(8, 32, 16, device=device, dtype=torch.bfloat16)
    )
    opt = MuonBatched(
        [p], [], lr=1e-3, ns_steps=4, weight_decay=0.0, paged_momentum=True,
    )
    p.grad = torch.randn_like(p)
    opt.step()
    mb = opt.state[p]["momentum_buffer"]
    # Either pinned host or fallback GPU clone (driver without pin_memory).
    if mb.device.type == "cpu":
        # Host pinned path: the pool has a "mom_stage" entry sized like p.
        any_stage = any("mom_stage" in str(k) for k in opt._buf_pool._cache.keys())
        assert any_stage, "expected GPU staging buffer in pool when paged"
    else:
        # Fallback path: same shape on GPU, no staging entry.
        assert mb.device.type == "cuda"


@cuda_only
def test_muon_batched_paged_vs_unpaged_param_evolution_close() -> None:
    """Plan B numerical sanity: two MuonBatched instances with identical seeds,
    one with ``paged_momentum=False`` and one ``True``, fed the same grad
    sequence, must produce nearly-identical params after several steps."""
    torch.manual_seed(0)
    device = "cuda"

    def fresh_setup() -> torch.nn.Parameter:
        torch.manual_seed(123)
        return torch.nn.Parameter(
            torch.randn(4, 16, 8, device=device, dtype=torch.bfloat16)
        )

    p_a = fresh_setup()
    p_b = fresh_setup()
    opt_a = MuonBatched([p_a], [], lr=5e-3, ns_steps=4, weight_decay=0.0,
                       paged_momentum=False)
    opt_b = MuonBatched([p_b], [], lr=5e-3, ns_steps=4, weight_decay=0.0,
                       paged_momentum=True)

    torch.manual_seed(7)
    grads = [torch.randn_like(p_a) for _ in range(8)]
    for g in grads:
        p_a.grad = g.clone()
        p_b.grad = g.clone()
        opt_a.step()
        opt_b.step()

    # H2D/D2H is bit-exact memcpy, so any divergence is from numerical re-
    # ordering inside _foreach_* on a fresh staging tensor vs. the long-lived
    # GPU one. Tolerate small bf16 noise.
    diff = (p_a - p_b).abs().to(torch.float32)
    rel = diff.max() / (p_a.abs().to(torch.float32).max() + 1e-6)
    assert rel.item() < 1e-2, f"paged vs unpaged drift too large: rel={rel.item()}"


def test_alloc_momentum_no_gpu_fallback_when_paged_false() -> None:
    """``_alloc_momentum(paged=False)`` is a pure CPU-friendly path used by
    the existing tests; ensure it returns a same-shape tensor without touching
    pinned memory APIs."""
    from moe_prune_distill.distill.muon_triton import _alloc_momentum

    p = torch.nn.Parameter(torch.randn(3, 5))
    p.grad = torch.randn_like(p)
    warned = []
    mb = _alloc_momentum(p, paged=False, log_warn=warned.append)
    assert mb.shape == p.shape
    assert mb.device.type == "cpu"
    assert warned == []


def test_muon_paged_momentum_flag_propagates() -> None:
    """Constructor flag is captured; default is False."""
    w = torch.nn.Parameter(torch.randn(8, 16))
    opt_default = Muon([w], [], lr=1e-3)
    opt_on = Muon([w], [], lr=1e-3, paged_momentum=True)
    assert opt_default._paged_momentum is False
    assert opt_on._paged_momentum is True
    optb_default = MuonBatched([w], [], lr=1e-3)
    optb_on = MuonBatched([w], [], lr=1e-3, paged_momentum=True)
    assert optb_default._paged_momentum is False
    assert optb_on._paged_momentum is True
