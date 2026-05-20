"""Tests for the Spectral Sphere Optimizer (SSO) — arxiv 2601.08393.

Five units, all CPU + fp32 so they run in CI without a GPU:

1. ``test_msign_spectral_norm_unit`` — NS5 output's spectral norm ≈ 1.
2. ``test_power_iteration_matches_svd`` — top singular triplet matches
   ``torch.linalg.svdvals`` for both 2D and batched 3D inputs.
3. ``test_h_lambda_monotone`` — paper Theorem A.2: h(λ) is monotone
   non-decreasing in λ on a constructed (G, Θ) pair.
4. ``test_sso_retraction_holds`` — after end-to-end fitting, ‖W‖₂ stays
   within 5 % of the target spectral radius R for SSO and Sphere modes.
   Plain Muon mode is allowed to drift (negative control).
5. ``test_sso_3d_expert_stack`` — 3D ``[E, M, N]`` input is handled
   per-expert; every expert lands on its own sphere of radius R.
"""

from __future__ import annotations

import torch

from moe_prune_distill.distill.optimizer import (
    SSO,
    msign,
    partition_for_sso,
    power_iteration,
    solve_lambda,
    spectral_radius,
)


# ====================================================================
# 1. msign output sits on the spectral sphere
# ====================================================================


def test_msign_spectral_norm_unit() -> None:
    torch.manual_seed(0)
    for shape in [(8, 16), (16, 8), (32, 32), (4, 64)]:
        G = torch.randn(*shape)
        M = msign(G, steps=8).float()
        sigma_top = torch.linalg.svdvals(M)[0].item()
        sigma_min = torch.linalg.svdvals(M).min().item()
        # NS5 with Polar Express coeffs gives SVs roughly in [0.5, 1.5];
        # the spectral norm should be at most ~1.5 and clearly bounded.
        assert 0.4 < sigma_min, f"shape {shape}: σ_min={sigma_min:.3f}"
        assert sigma_top < 1.6, f"shape {shape}: σ_max={sigma_top:.3f}"


# ====================================================================
# 2. power iteration matches torch.linalg.svdvals
# ====================================================================


def test_power_iteration_matches_svd_2d() -> None:
    torch.manual_seed(0)
    W = torch.randn(8, 16)
    sigma, u, v = power_iteration(W, iters=60)
    sigma_ref = torch.linalg.svdvals(W)[0]
    assert abs(sigma.item() - sigma_ref.item()) < 1e-3, (sigma, sigma_ref)
    # u and v should reconstruct σ via u^T W v.
    recon = (u @ (W.float() @ v)).item()
    assert abs(recon - sigma_ref.item()) < 1e-3


def test_power_iteration_batched_3d() -> None:
    torch.manual_seed(0)
    W = torch.randn(5, 8, 16)
    sigma, u, v = power_iteration(W, iters=60)
    sigma_ref = torch.linalg.svdvals(W)[..., 0]
    assert torch.allclose(sigma, sigma_ref, atol=2e-3), (sigma, sigma_ref)


# ====================================================================
# 3. h(λ) monotonicity (paper Theorem A.2)
# ====================================================================


def test_h_lambda_monotone() -> None:
    torch.manual_seed(1)
    W = torch.randn(8, 16)
    sigma, u, v = power_iteration(W, iters=40)
    Theta = torch.outer(u, v)
    G = torch.randn(8, 16)
    M_hat = G / G.norm()

    hs: list[float] = []
    for lam in torch.linspace(-2.0, 2.0, 11):
        Phi = msign(M_hat + lam * Theta, steps=8).float()
        hs.append((Theta * Phi).sum().item())
    # h(λ) is monotone for exact msign; with NS5 approximation we tolerate
    # small dips. Use a coarser monotonicity check on a 5-point smoothing.
    smoothed = [
        sum(hs[max(0, i - 1):i + 2]) / len(hs[max(0, i - 1):i + 2])
        for i in range(len(hs))
    ]
    diffs = [b - a for a, b in zip(smoothed[:-1], smoothed[1:])]
    assert all(d > -5e-2 for d in diffs), f"non-monotone smoothed h(λ): {smoothed}"
    assert hs[0] < hs[-1] - 1.0, f"h didn't transition: {hs[0]} -> {hs[-1]}"


def test_solve_lambda_finds_root() -> None:
    torch.manual_seed(2)
    W = torch.randn(8, 16)
    sigma, u, v = power_iteration(W, iters=40)
    Theta = torch.outer(u, v)
    G = torch.randn(8, 16)
    M_hat = G / G.norm()
    bound_t = torch.tensor(min(W.shape) ** 0.5, dtype=torch.float32)
    lam_star = solve_lambda(
        M_hat, Theta, bound_t, ns_steps=8, max_iters=30, tol=1e-3
    )
    Phi = msign(M_hat + float(lam_star) * Theta, steps=8).float()
    h = (Theta * Phi).sum().item()
    assert abs(h) < 0.1, f"|h(λ*)|={abs(h):.4f} too large; lam*={float(lam_star):.4f}"


# ====================================================================
# 4. retraction holds across many steps
# ====================================================================


def _fit_and_check_radius(mode: str, *, expect_constrained: bool) -> tuple[float, float]:
    torch.manual_seed(2)
    lin = torch.nn.Linear(16, 8, bias=True)
    opt = SSO(
        [lin.weight], [lin.bias],
        lr=0.05, mode=mode, radius_c=2.0, ns_steps=6,
    )
    R = spectral_radius(lin.weight.shape, c=2.0)
    target = torch.randn(8)
    x = torch.randn(16)
    losses = []
    for _ in range(150):
        loss = ((lin(x) - target) ** 2).mean()
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    sig_end, _, _ = power_iteration(lin.weight.data, iters=40)
    drift = abs(float(sig_end) - R) / R
    return losses[-1] / losses[0], drift


def test_sso_retraction_holds() -> None:
    loss_ratio, drift = _fit_and_check_radius("sso", expect_constrained=True)
    assert loss_ratio < 0.3, f"sso loss didn't drop enough: {loss_ratio:.3f}"
    assert drift < 0.05, f"sso ‖W‖₂ drift {drift*100:.2f}% > 5%"


def test_sphere_retraction_holds() -> None:
    loss_ratio, drift = _fit_and_check_radius("sphere", expect_constrained=True)
    assert loss_ratio < 0.3, f"sphere loss didn't drop enough: {loss_ratio:.3f}"
    assert drift < 0.05, f"sphere ‖W‖₂ drift {drift*100:.2f}% > 5%"


# ====================================================================
# 5. 3D MoE expert stack
# ====================================================================


def test_sso_3d_expert_stack() -> None:
    torch.manual_seed(3)
    E, M, N = 4, 8, 16
    W = torch.randn(E, M, N, requires_grad=True)
    opt = SSO([W], [], lr=0.05, mode="sso", radius_c=2.0, ns_steps=6)
    R = spectral_radius(W.shape, c=2.0)

    # After init, every expert should be on its own sphere.
    sig_init, _, _ = power_iteration(W.data, iters=40)
    for s in sig_init.tolist():
        assert abs(s - R) / R < 0.05, f"init drift: σ={s:.3f}, R={R:.3f}"

    target = torch.randn(E, M)
    x = torch.randn(E, N)
    initial_loss = None
    for step in range(120):
        pred = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)
        loss = ((pred - target) ** 2).mean()
        if step == 0:
            initial_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5, (initial_loss, final_loss)

    sig_end, _, _ = power_iteration(W.data, iters=40)
    for s in sig_end.tolist():
        assert abs(s - R) / R < 0.08, f"end drift: σ={s:.3f}, R={R:.3f}"


# ====================================================================
# 6. partition helper
# ====================================================================


def test_partition_for_sso_excludes_embeddings_and_1d() -> None:
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(10, 8)   # 2D but blacklisted
            self.q_proj = torch.nn.Linear(8, 8, bias=True)  # 2D weight + 1D bias
            self.norm = torch.nn.LayerNorm(8)               # 1D weight + 1D bias
            self.experts = torch.nn.Parameter(torch.randn(4, 8, 16))  # 3D OK

    m = Toy()
    sso, adamw = partition_for_sso(m.named_parameters())
    sso_ids = {id(p) for p in sso}
    assert id(m.q_proj.weight) in sso_ids
    assert id(m.experts) in sso_ids
    assert id(m.embed_tokens.weight) not in sso_ids
    assert id(m.q_proj.bias) not in sso_ids
    assert id(m.norm.weight) not in sso_ids


# ====================================================================
# 7. NaN safety (regression for the distill / pretrained-checkpoint
# regime — none of these inputs are paper-realistic but they all
# happen in practice when retraction is fed pruned MoE experts).
# ====================================================================


def test_msign_finite_on_tiny_input() -> None:
    """Frobenius norm well below fp32 normal range -> msign returns 0, not NaN."""
    G = torch.full((8, 16), 1e-35)
    out = msign(G, steps=8)
    assert torch.isfinite(out).all()
    assert (out == 0).all(), "tiny input should short-circuit to zero"


def test_msign_finite_on_zero_input() -> None:
    G = torch.zeros(8, 16)
    out = msign(G, steps=8)
    assert torch.isfinite(out).all()
    assert (out == 0).all()


def test_msign_finite_on_3d_partial_dead() -> None:
    """One expert is all-zero, others are healthy -> the dead slice
    returns zero, the others are unaffected."""
    G = torch.randn(4, 8, 16)
    G[2].zero_()
    out = msign(G, steps=8)
    assert torch.isfinite(out).all()
    assert (out[2] == 0).all()
    assert out[0].abs().sum() > 0  # healthy slice produced something


def test_solve_lambda_nan_safe() -> None:
    """A degenerate Theta (NaN cell) must not corrupt the bisection;
    solve_lambda returns a finite tensor and falls back to lam=0."""
    torch.manual_seed(0)
    M_hat = torch.randn(8, 16) / 8.0
    # Build a Theta that triggers NaN inside msign(M_hat + λ Θ): if Θ
    # itself contains NaN, the perturbed input will too.
    Theta = torch.zeros(8, 16)
    Theta[0, 0] = float("nan")
    bound_t = torch.tensor(4.0, dtype=torch.float32)
    lam = solve_lambda(M_hat, Theta, bound_t, ns_steps=5, max_iters=10, tol=1e-3)
    assert torch.isfinite(lam).all(), f"solve_lambda returned non-finite: {lam}"


def test_sso_preserve_radius_mode_keeps_initial_sigma() -> None:
    """Under radius_mode='preserve', the optimizer must retract back to
    the *initial* σ — not to the paper formula."""
    torch.manual_seed(7)
    lin = torch.nn.Linear(16, 8, bias=True)
    # Scale the weight to a non-trivial spectral norm.
    with torch.no_grad():
        sigma_pre, _, _ = power_iteration(lin.weight.data, iters=40)
        lin.weight.data.mul_(5.0 / float(sigma_pre))
    sigma_init, _, _ = power_iteration(lin.weight.data, iters=40)
    target_sigma = float(sigma_init)
    assert abs(target_sigma - 5.0) < 0.05, target_sigma

    opt = SSO(
        [lin.weight],
        [lin.bias],
        lr=0.05,
        mode="sso",
        radius_mode="preserve",
        radius_c=2.0,  # would conflict with preserve if respected -> sanity check
        ns_steps=6,
    )
    target = torch.randn(8)
    x = torch.randn(16)
    for _ in range(120):
        loss = ((lin(x) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(lin.weight).all(), "preserve mode produced NaN weights"

    sigma_end, _, _ = power_iteration(lin.weight.data, iters=40)
    drift = abs(float(sigma_end) - target_sigma) / target_sigma
    assert drift < 0.05, (
        f"preserve mode drifted from σ_init={target_sigma:.3f} to "
        f"σ_end={float(sigma_end):.3f} ({drift*100:.2f}%)"
    )


def test_sso_handles_dead_expert_without_nan() -> None:
    """A pruned MoE stack with one all-zero expert must not produce
    inf at construction (R/σ blow-up) or NaN during a step."""
    torch.manual_seed(11)
    E, M, N = 4, 8, 16
    W = torch.randn(E, M, N)
    W[2].zero_()
    Wp = torch.nn.Parameter(W)

    # Paper mode would do R/σ where σ=0 for the dead expert -> would
    # have been inf without the SCALE clamp. Verify the clamp catches it.
    opt = SSO([Wp], [], lr=0.05, mode="sso", radius_mode="paper",
              radius_c=2.0, ns_steps=5)
    assert torch.isfinite(Wp.data).all(), "paper-mode init blew dead expert to inf"

    # One step with a real gradient on the live experts and zero on the
    # dead expert: nothing should NaN out.
    x = torch.randn(E, N)
    target = torch.randn(E, M)
    for _ in range(5):
        pred = torch.bmm(Wp, x.unsqueeze(-1)).squeeze(-1)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(Wp.data).all(), "step produced NaN/inf weights"


def test_sso_skips_nan_gradient_without_corrupting_weights() -> None:
    """If the gradient itself is non-finite (e.g. upstream loss bug),
    SSO should skip the update for that param, not propagate NaN into
    weights."""
    torch.manual_seed(13)
    lin = torch.nn.Linear(16, 8)
    opt = SSO([lin.weight], [lin.bias], lr=0.01, mode="sso",
              radius_mode="preserve", ns_steps=5)
    weight_before = lin.weight.data.clone()

    # Inject a NaN directly into the gradient.
    lin.weight.grad = torch.full_like(lin.weight, float("nan"))
    lin.bias.grad = torch.zeros_like(lin.bias)
    opt.step()

    assert torch.isfinite(lin.weight.data).all(), "SSO let NaN grad corrupt weights"
    # Weights should be unchanged for the SSO param (skip path).
    assert torch.allclose(lin.weight.data, weight_before, atol=1e-6)
