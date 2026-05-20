"""Spectral Sphere Optimizer (SSO) — Xie et al., 2026.

Implements Algorithm 1 of "Controlled LLM Training on Spectral Sphere".
Constrains every 2D / 3D weight matrix to a spectral sphere of radius
``R`` and performs steepest descent in the tangent space of that sphere —
i.e. the update keeps both ``||W||_2 = R`` and ``||Φ||_2 = 1``.

Modes (``mode=``):

* ``sso``    — full Algorithm 1: per-step bisection on λ, retraction to R,
               μP-scaled update.
* ``sphere`` — MuonSphere (§3.4): λ = 0 (skip the bisection), retraction
               kept.

Radius mode (``radius_mode=``):

* ``paper``    — ``R = c · √(d_out / d_in)``, projects every matrix onto
                 that sphere at construction. Matches the paper, intended
                 for from-scratch pretraining.
* ``preserve`` — record each weight's initial top singular value σ₀ once
                 and use ``R = max(σ₀, R_floor)`` per-param thereafter.
                 Does not rescale weights at construction. Intended for
                 finetuning / distillation from a pretrained checkpoint —
                 forcing pretrained weights onto the paper formula
                 destroys learned features (and dead MoE experts whose
                 σ ≈ 0 blow up via R/σ).

For 3D MoE expert stacks ``W ∈ ℝ^{E × M × N}`` the leading axis is a
batch of ``E`` independent matrices; NS5, power iteration and bisection
all run batched over ``E``.

Numerical hardening (none of this is in the paper, but the paper assumes
clean random init — these guards are needed when feeding a pruned
pretrained checkpoint):

* msign runs in fp32 by default (paper §5.2). ``msign_dtype=torch.bfloat16``
  trades precision for speed.
* Frobenius normalization happens in fp32 with an explicit zero-floor:
  if ‖G‖_F < 1e-30, msign returns 0 (no-op update) instead of NaN.
* Power iteration's σ is clamped against NaN/inf and against an absolute
  floor — dead experts produce σ = floor, not σ = 0.
* Retraction scale ``R/σ`` is clamped to ``[1e-6, 1e+3]``; can't blow
  weights to inf.
* Bisection is NaN-aware: a degenerate slice freezes its bracket instead
  of corrupting the bisection state. Final λ* with NaN/inf is replaced
  by 0, degrading that step to MuonSphere instead of NaN.
* Final write-back in ``_sso_step`` skips Φ that contains non-finite
  values. The momentum buffer is preserved so the next step recovers.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Literal

import torch
from torch import Tensor

_log = logging.getLogger(__name__)


# ====================================================================
# 1. Newton-Schulz 5 (msign approximation)
# ====================================================================

# Polar Express coefficients from Amsel et al., 2025.
_NS_ABC = (3.4445, -4.7750, 2.0315)

# Below this Frobenius norm we declare the input degenerate and short-
# circuit msign to zero. fp32's smallest normal is ~1.2e-38; this leaves
# headroom for the X @ X^T squaring inside NS5.
_MSIGN_NORM_FLOOR = 1e-30


def _ns5_inner(X: Tensor, steps: int) -> Tensor:
    """Body of NS5. ``X`` is already normalized to ‖X‖₂ ≤ 1.

    Works for both 2D ``[M, N]`` and batched 3D ``[B, M, N]`` because
    ``@`` and ``transpose(-2, -1)`` broadcast over leading dims.
    """
    a, b, c = _NS_ABC
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X


def _normalize_for_ns(
    G: Tensor, compute_dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Cast to ``compute_dtype`` and divide by Frobenius norm.

    Returns ``(X_normed, mask)`` where ``mask`` is 0 for degenerate
    slices (norm below the floor) and 1 otherwise. Callers multiply
    msign's output by ``mask`` to zero out degenerate slices instead
    of letting them produce NaN.

    The norm is *always* computed in fp32 — bf16 underflows for
    near-zero gradients (anything below ~6e-5 squared and summed).
    """
    if G.ndim == 2:
        norm32 = G.float().norm()
        ok = norm32 > _MSIGN_NORM_FLOOR
        denom = torch.where(
            ok, norm32, torch.ones_like(norm32)
        )  # avoid div-by-zero, mask kills the result anyway
        X = (G.float() / denom).to(compute_dtype)
        return X, ok.to(compute_dtype)
    if G.ndim == 3:
        flat32 = G.float().reshape(G.shape[0], -1)
        norms32 = flat32.norm(dim=1)  # [B]
        ok = norms32 > _MSIGN_NORM_FLOOR  # [B]
        denom = torch.where(ok, norms32, torch.ones_like(norms32))
        X = (G.float() / denom.reshape(-1, 1, 1)).to(compute_dtype)
        return X, ok.to(compute_dtype)
    raise ValueError(f"NS5 supports 2D / 3D only, got ndim={G.ndim}")


def _msign_eager(
    G: Tensor, steps: int, compute_dtype: torch.dtype = torch.float32
) -> Tensor:
    """msign(G) ≈ U V^T via NS5. Returns same shape and dtype as G.

    Computes NS5 in ``compute_dtype`` (fp32 by paper §5.2). Degenerate
    slices (Frobenius norm below floor) return zero.
    """
    X, mask = _normalize_for_ns(G, compute_dtype)
    m, n = X.shape[-2], X.shape[-1]
    transposed = m > n
    if transposed:
        X = X.transpose(-2, -1)
    X = _ns5_inner(X, steps)
    if transposed:
        X = X.transpose(-2, -1)
    if X.ndim == 3:
        X = X * mask.reshape(-1, 1, 1)
    else:
        X = X * mask
    # Belt-and-braces: if NS5 itself produced any non-finite value
    # (saturating fp32 ops near boundary) zero them out rather than
    # propagating NaN into the optimizer state.
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.to(G.dtype)


def _try_compile(fn):
    """Wrap ``fn`` in ``torch.compile`` with a permanent eager fallback.

    On Windows-without-cl, MPS, or anywhere the inductor backend can't
    initialize, the compiled callable raises at *call* time (not build
    time), so we have to catch and downgrade per-call. After the first
    failure we replace the call site so subsequent calls go straight to
    eager — no repeated stacktraces.
    """
    if not hasattr(torch, "compile"):
        return fn
    try:
        compiled = torch.compile(fn, fullgraph=False, dynamic=True)
    except Exception:
        return fn

    state = {"compiled": compiled, "fallen_back": False}

    def wrapper(*args, **kwargs):
        if state["fallen_back"]:
            return fn(*args, **kwargs)
        try:
            return state["compiled"](*args, **kwargs)
        except Exception:
            state["fallen_back"] = True
            return fn(*args, **kwargs)

    return wrapper


_msign = _try_compile(_msign_eager)


def msign(
    G: Tensor,
    steps: int = 5,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Public matrix-sign approximation. Dispatches 2D vs 3D.

    ``compute_dtype`` controls the precision of the NS5 iterations
    (fp32 by paper §5.2; bf16 trades precision for speed).
    """
    if G.ndim not in (2, 3):
        raise ValueError(f"msign: ndim must be 2 or 3, got {G.ndim}")
    return _msign(G, steps, compute_dtype)


# ====================================================================
# 2. Power iteration with warm start (top singular triplet)
# ====================================================================

# Floor for the top singular value. Anything below this we treat as a
# dead direction (e.g. an all-zero pruned MoE expert) — using σ = 0 in
# retraction would blow weights to inf.
_SIGMA_FLOOR = 1e-7


@torch.no_grad()
def power_iteration(
    W: Tensor,
    u: Tensor | None = None,
    v: Tensor | None = None,
    iters: int = 4,
    eps: float = 1e-7,
) -> tuple[Tensor, Tensor, Tensor]:
    """Top singular triplet (σ, u, v) via power iteration on ``W^T W``.

    Supports 2D ``[M, N]`` and 3D ``[B, M, N]`` (batched along B).
    ``u`` / ``v`` are warm-start vectors; if None, sampled from N(0, 1).
    Returns:
        sigma: scalar (2D) or [B] (3D)  — clamped against NaN/inf and floored.
        u    : [M] / [B, M]             — sanitized to N(0,1) on degenerate slices.
        v    : [N] / [B, N]             — sanitized similarly.
    """
    if W.ndim == 2:
        M, N = W.shape
        Wf = W.float()
        if v is None or v.shape != (N,):
            v = torch.randn(N, device=W.device, dtype=torch.float32)
        v = v.to(dtype=torch.float32)
        # Sanitize warm-start vectors (could carry NaN from a previous
        # bad step before we hardened things).
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if v.norm() < eps:
            v = torch.randn(N, device=W.device, dtype=torch.float32)
        for _ in range(iters):
            u_ = Wf @ v
            u_norm = u_.norm().clamp_min(eps)
            u_ = u_ / u_norm
            v_ = Wf.transpose(-1, -2) @ u_
            v_norm = v_.norm().clamp_min(eps)
            v_ = v_ / v_norm
            u, v = u_, v_
        sigma = (u @ (Wf @ v))
        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = sigma.abs().clamp_min(_SIGMA_FLOOR)
        return sigma, u, v

    if W.ndim == 3:
        B, M, N = W.shape
        Wf = W.float()
        if v is None or v.shape != (B, N):
            v = torch.randn(B, N, device=W.device, dtype=torch.float32)
        v = v.to(dtype=torch.float32)
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        # Replace zero-norm rows with random vectors (otherwise the
        # subsequent normalization keeps them at zero forever).
        v_row_norm = v.norm(dim=-1, keepdim=True)
        bad = v_row_norm < eps
        if bad.any():
            v = torch.where(
                bad, torch.randn_like(v), v
            )
        for _ in range(iters):
            u_ = torch.bmm(Wf, v.unsqueeze(-1)).squeeze(-1)
            u_norm = u_.norm(dim=-1, keepdim=True).clamp_min(eps)
            u_ = u_ / u_norm
            v_ = torch.bmm(Wf.transpose(-2, -1), u_.unsqueeze(-1)).squeeze(-1)
            v_norm = v_.norm(dim=-1, keepdim=True).clamp_min(eps)
            v_ = v_ / v_norm
            u, v = u_, v_
        sigma = torch.bmm(
            u.unsqueeze(1), torch.bmm(Wf, v.unsqueeze(-1))
        ).reshape(B)
        sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
        sigma = sigma.abs().clamp_min(_SIGMA_FLOOR)
        return sigma, u, v

    raise ValueError(f"power_iteration: ndim must be 2 or 3, got {W.ndim}")


# ====================================================================
# 3. SSO Lagrange solver: bisect h(λ) = ⟨Θ, msign(M̂ + λΘ)⟩
# ====================================================================


def _h_lambda(
    M_hat: Tensor,
    Theta: Tensor,
    lam: Tensor,
    ns_steps: int,
    msign_dtype: torch.dtype,
) -> Tensor:
    """h(λ) = ⟨Θ, msign(M̂ + λ Θ)⟩.

    Reduction is in fp32 (bf16 sums over millions of elements drift
    badly enough to break the bisection invariant).

    Shapes:
        M_hat, Theta: [..., M, N] (2D or 3D — same shape).
        lam        : scalar for 2D, [B] for 3D.
    Returns:
        scalar (2D) or [B] (3D).
    """
    if M_hat.ndim == 2:
        Phi = msign(M_hat + lam * Theta, steps=ns_steps, compute_dtype=msign_dtype)
        return (Theta.float() * Phi.float()).sum()
    # 3D batched
    lam_b = lam.reshape(-1, 1, 1)
    Phi = msign(M_hat + lam_b * Theta, steps=ns_steps, compute_dtype=msign_dtype)
    return (Theta.float() * Phi.float()).sum(dim=(-1, -2))


@torch.no_grad()
def solve_lambda(
    M_hat: Tensor,
    Theta: Tensor,
    G_nuclear_bound: Tensor,
    *,
    ns_steps: int = 5,
    max_iters: int = 20,
    tol: float = 2e-4,
    msign_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Find λ* such that h(λ*) = 0 via bisection on a paper-sized bracket.

    h is monotone non-decreasing (paper Theorem A.2). The root is bounded
    by ``|λ*| ≤ 2 ‖G‖_*`` (Theorem A.3). We initialize the bracket at
    that bound and run ``max_iters`` bisection steps — no exponential
    bracketing needed.

    NaN / inf handling: if ``h(mid)`` is non-finite for some 3D slice we
    freeze that slice's bracket (don't update lo / hi). At the end we
    sanitize λ* — any remaining non-finite entry is set to 0, degrading
    that step to a MuonSphere update rather than NaN.

    All operations stay on-device. The returned tensor matches Theta's
    layout (scalar for 2D, [B] for 3D).
    """
    is_batched = M_hat.ndim == 3
    device = M_hat.device

    if is_batched:
        # G_nuclear_bound is [B], in fp32.
        bound = (G_nuclear_bound * 2.0).to(torch.float32)
        lo = -bound
        hi = bound.clone()
    else:
        bound_val = float(G_nuclear_bound) * 2.0
        if not math.isfinite(bound_val) or bound_val < 1e-12:
            # Degenerate gradient: λ = 0 is correct (and cheap).
            return torch.zeros((), device=device, dtype=torch.float32)
        lo = torch.full((), -bound_val, device=device, dtype=torch.float32)
        hi = torch.full((), bound_val, device=device, dtype=torch.float32)

    # Sanity-check the theoretical bracket. With exact msign and exact Θ,
    # [-2‖G‖_*, 2‖G‖_*] brackets the root. In practice both msign and Θ
    # are approximated, so a badly degenerate slice may violate this.
    h_lo = _h_lambda(M_hat, Theta, lo, ns_steps=ns_steps, msign_dtype=msign_dtype)
    h_hi = _h_lambda(M_hat, Theta, hi, ns_steps=ns_steps, msign_dtype=msign_dtype)

    if is_batched:
        bracket_ok = torch.isfinite(h_lo) & torch.isfinite(h_hi) & (h_lo <= 0) & (h_hi >= 0)
        # Degenerate slices fall back to λ = 0, i.e. MuonSphere for that slice.
        lo = torch.where(bracket_ok, lo, torch.zeros_like(lo))
        hi = torch.where(bracket_ok, hi, torch.zeros_like(hi))
    else:
        h_lo_val = float(h_lo)
        h_hi_val = float(h_hi)
        if (
            not math.isfinite(h_lo_val)
            or not math.isfinite(h_hi_val)
            or h_lo_val > 0
            or h_hi_val < 0
        ):
            return torch.zeros((), device=device, dtype=torch.float32)

    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        h_mid = _h_lambda(M_hat, Theta, mid, ns_steps=ns_steps, msign_dtype=msign_dtype)

        if is_batched:
            finite = torch.isfinite(h_mid)
            # Freeze bracket on degenerate slices.
            mask_pos = ((h_mid > 0) & finite).to(mid.dtype)
            mask_neg = ((h_mid <= 0) & finite).to(mid.dtype)
            mask_skip = (~finite).to(mid.dtype)
            hi = mask_pos * mid + mask_neg * hi + mask_skip * hi
            lo = mask_neg * mid + mask_pos * lo + mask_skip * lo
            converged = ((hi - lo).abs() < tol).all()
        else:
            h_val = float(h_mid)
            if not math.isfinite(h_val):
                # Freeze: we have no information, just stop bisecting.
                break
            if h_val > 0:
                hi = mid
            else:
                lo = mid
            converged = (hi - lo).abs().item() < tol
        if converged:
            break

    lam_star = 0.5 * (lo + hi)
    # Final sanitize: NaN / inf -> 0 (= MuonSphere step for that slice).
    lam_star = torch.nan_to_num(lam_star, nan=0.0, posinf=0.0, neginf=0.0)
    return lam_star


# ====================================================================
# 4. Spectral radius (R) helper
# ====================================================================


def spectral_radius(shape: torch.Size, c: float = 1.0) -> float:
    """R = c * sqrt(d_out / d_in) for the inner 2D shape (last two dims).

    Used by ``radius_mode='paper'``. The ``preserve`` mode ignores this
    and instead records each weight's initial σ on the optimizer state.
    """
    d_out, d_in = int(shape[-2]), int(shape[-1])
    return float(c) * math.sqrt(d_out / max(d_in, 1))


# Bounds the per-step retraction multiplier R/σ. Even with σ floored at
# _SIGMA_FLOOR, a deeply pruned matrix could try R/σ ~ 1e7; that would
# blow weights to inf in one step. This clamp is a safety net only —
# under normal conditions R/σ ≈ 1.
_SCALE_MIN, _SCALE_MAX = 1e-6, 1e3


# ====================================================================
# 5. The optimizer
# ====================================================================


class SSO(torch.optim.Optimizer):
    """Spectral Sphere Optimizer (+ MuonSphere variants) + foreach AdamW.

    Args:
        sso_params: 2D / 3D matrix params optimized via spectral method.
        adamw_params: everything else (1D, embeddings, etc.).
        lr: base learning rate η. Final update is ``η · R · Φ`` for SSO.
        wd: AdamW weight decay (matrix params have wd=0 by default since
            retraction already bounds ‖W‖₂; pass ``wd_matrix>0`` to override).
        wd_matrix: weight decay for the matrix branch. Paper sets to 0.
        momentum: SGD momentum β for the matrix branch (default 0.95).
        nesterov: whether to use Nesterov-style lookahead. Disabled by
            default because Algorithm 1 uses the momentum buffer directly.
        ns_steps: NS5 iterations per msign call (5–8; paper uses 8).
        radius_c: scalar c in ``R = c · √(d_out / d_in)`` (paper c≈2).
            Only used when ``radius_mode='paper'``.
        radius_mode:
            ``"paper"`` — Algorithm 1: project every matrix onto
                ``R = c · √(d_out / d_in)`` at construction.
            ``"preserve"`` — record each weight's initial σ once and
                use ``R = max(σ_init, σ_floor)`` per-param. No rescaling
                at construction. Right choice for finetuning / distillation
                from a pretrained checkpoint.
        mode: ``"sso" | "sphere"``.
        msign_dtype: NS5 compute dtype. Default fp32 (paper §5.2);
            bf16 trades precision for speed.
        bisect_max_iters: max bisection iterations for λ (SSO only).
        bisect_tol: target ``|hi - lo|`` tolerance (SSO only).
        power_iters: power-iteration steps for warm-started (σ, u, v).
        adamw_betas, adamw_eps: AdamW params (defaults 0.9 / 0.95 / 1e-8).
    """

    VALID_MODES = ("sso", "sphere")
    VALID_RADIUS_MODES = ("paper", "preserve")

    def __init__(
        self,
        sso_params: Iterable[torch.nn.Parameter],
        adamw_params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        wd: float = 0.1,
        wd_matrix: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = False,
        ns_steps: int = 5,
        radius_c: float = 1.0,
        radius_mode: Literal["paper", "preserve"] = "paper",
        mode: str = "sso",
        msign_dtype: torch.dtype = torch.float32,
        bisect_max_iters: int = 20,
        bisect_tol: float = 2e-4,
        power_iters: int = 4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        if radius_mode not in self.VALID_RADIUS_MODES:
            raise ValueError(
                f"radius_mode must be one of {self.VALID_RADIUS_MODES}, got {radius_mode!r}"
            )
        sso_params = list(sso_params)
        adamw_params = list(adamw_params)
        for p in sso_params:
            if p.ndim not in (2, 3):
                raise ValueError(
                    f"SSO param has ndim={p.ndim}; expected 2 or 3 (matrix or expert stack)"
                )

        groups: list[dict] = []
        if sso_params:
            groups.append(
                dict(
                    params=sso_params,
                    lr=lr,
                    wd=wd_matrix,
                    momentum=momentum,
                    nesterov=nesterov,
                    ns_steps=ns_steps,
                    radius_c=radius_c,
                    radius_mode=radius_mode,
                    mode=mode,
                    msign_dtype=msign_dtype,
                    bisect_max_iters=bisect_max_iters,
                    bisect_tol=bisect_tol,
                    power_iters=power_iters,
                    is_sso=True,
                )
            )
        if adamw_params:
            groups.append(
                dict(
                    params=adamw_params,
                    lr=lr,
                    weight_decay=wd,
                    betas=adamw_betas,
                    eps=adamw_eps,
                    step=0,
                    is_sso=False,
                )
            )
        if not groups:
            raise ValueError("SSO: both sso_params and adamw_params are empty")
        super().__init__(groups, {})

        # One-time init: seed (σ, u, v, R) on every matrix param's state.
        # In ``paper`` mode this also rescales p.data toward the sphere.
        if sso_params:
            self._initial_setup()

    # ---- one-time init ----

    @torch.no_grad()
    def _initial_setup(self) -> None:
        """Seed per-param state and (optionally) project onto the sphere.

        For ``radius_mode='paper'``: ``R = c · √(d_out/d_in)``, project
        ``p.data`` toward the sphere (Algorithm 1 line 1). The projection
        scale is safety-clamped for degenerate matrices.

        For ``radius_mode='preserve'``: ``R = max(σ_init, σ_floor)``,
        leave ``p.data`` alone.

        We use 32 power-iteration steps here because at init time we
        need accuracy and don't yet have a warm start.
        """
        for group in self.param_groups:
            if not group.get("is_sso", False):
                continue
            radius_mode = group["radius_mode"]
            for p in group["params"]:
                sigma, u, v = power_iteration(p.data, iters=32)
                if radius_mode == "paper":
                    R = spectral_radius(p.shape, c=group["radius_c"])
                    if p.ndim == 2:
                        scale = float(R) / float(sigma)
                        scale = max(_SCALE_MIN, min(_SCALE_MAX, scale))
                        p.data.mul_(scale)
                        R_state: Tensor = torch.tensor(
                            R, device=p.device, dtype=torch.float32
                        )
                    else:
                        scale = (R / sigma).clamp(_SCALE_MIN, _SCALE_MAX)
                        p.data.mul_(scale.reshape(-1, 1, 1).to(p.dtype))
                        R_state = torch.full(
                            (p.shape[0],), float(R), device=p.device, dtype=torch.float32
                        )
                else:  # preserve
                    if p.ndim == 2:
                        # Floor: dead 2D matrices (shouldn't happen, but be safe).
                        R_state = sigma.detach().clone().clamp_min(_SIGMA_FLOOR)
                    else:
                        R_state = sigma.detach().clone().clamp_min(_SIGMA_FLOOR)

                state = self.state[p]
                state["u"] = u
                state["v"] = v
                state["R"] = R_state

    # ---- main step ----

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = [p for p in group["params"] if p.grad is not None]
            if not params_with_grad:
                continue
            if group.get("is_sso", False):
                self._sso_step(params_with_grad, group)
            else:
                self._adamw_step_foreach(params_with_grad, group)
        return loss

    # ---- SSO branch ----

    def _sso_step(self, params: list[torch.nn.Parameter], group: dict) -> None:
        lr = group["lr"]
        wd = group["wd"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        radius_mode = group["radius_mode"]
        mode = group["mode"]
        msign_dtype = group["msign_dtype"]
        max_iters = group["bisect_max_iters"]
        tol = group["bisect_tol"]
        pwr_iters = group["power_iters"]

        for p in params:
            g = p.grad
            if g.is_sparse:
                raise RuntimeError("SSO does not support sparse gradients")
            if not torch.isfinite(g).all():
                # Skip silently — gradient clipping should have caught
                # this upstream, but if it didn't, taking *any* step on
                # a NaN grad is worse than skipping.
                _log.warning("SSO: skipping param with non-finite grad (shape=%s)", tuple(p.shape))
                continue
            state = self.state[p]

            # Step 4: M_t = β M + (1-β) g (fp32 momentum buffer)
            buf = state.get("momentum_buffer")
            if buf is None:
                buf = torch.zeros_like(
                    p, dtype=torch.float32, memory_format=torch.preserve_format
                )
                state["momentum_buffer"] = buf

            g32 = g.float()
            buf.mul_(momentum).add_(g32, alpha=1.0 - momentum)

            # Algorithm 1 uses M_t directly. Nesterov is kept as an optional
            # variant but disabled by default.
            M_eff = g32.add(buf, alpha=momentum) if nesterov else buf

            # Step 5: M̂ = M / ‖M‖_F
            if M_eff.ndim == 2:
                m_frob = M_eff.norm().clamp_min(1e-7)
                M_hat = M_eff / m_frob
            else:
                flat = M_eff.reshape(M_eff.shape[0], -1)
                m_frob = flat.norm(dim=1).clamp_min(1e-7)
                M_hat = M_eff / m_frob.reshape(-1, 1, 1)

            # Steps 6-8: power iter + retraction to per-param R.
            R_state: Tensor = state["R"]  # scalar (2D) or [B] (3D) fp32
            u_warm = state.get("u")
            v_warm = state.get("v")
            sigma, u, v = power_iteration(
                p.data, u=u_warm, v=v_warm, iters=pwr_iters
            )
            state["u"], state["v"] = u, v

            if p.ndim == 2:
                # scale = R / σ, clamped to safe range.
                scale = (R_state / sigma).clamp(_SCALE_MIN, _SCALE_MAX)
                p.data.mul_(scale.to(p.dtype))
                Theta = torch.outer(u, v).to(M_hat.dtype)
            else:
                scale = (R_state / sigma).clamp(_SCALE_MIN, _SCALE_MAX)
                p.data.mul_(scale.reshape(-1, 1, 1).to(p.dtype))
                Theta = torch.bmm(u.unsqueeze(-1), v.unsqueeze(-2)).to(M_hat.dtype)

            if mode == "sphere":
                # MuonSphere: λ = 0, use matrix sign / polar factor of M̂.
                Phi = msign(M_hat, steps=ns_steps, compute_dtype=msign_dtype)
            else:
                # SSO: bisect for λ such that ⟨Θ, msign(M̂ + λΘ)⟩ = 0.
                # ‖M̂‖_* ≤ √rank gives a coarse bound for the bracket
                # (M̂ is Frobenius-unit, so ‖M̂‖_* ≤ √rank · ‖M̂‖_F = √rank).
                rank = min(p.shape[-2], p.shape[-1])
                if p.ndim == 2:
                    nuc_bound_t = torch.tensor(
                        math.sqrt(rank), device=p.device, dtype=torch.float32
                    )
                else:
                    nuc_bound_t = torch.full(
                        (p.shape[0],), math.sqrt(rank), device=p.device, dtype=torch.float32
                    )
                lam_star = solve_lambda(
                    M_hat, Theta, nuc_bound_t,
                    ns_steps=ns_steps,
                    max_iters=max_iters,
                    tol=tol,
                    msign_dtype=msign_dtype,
                )
                if p.ndim == 2:
                    Phi = msign(
                        M_hat + lam_star.to(M_hat.dtype) * Theta,
                        steps=ns_steps,
                        compute_dtype=msign_dtype,
                    )
                else:
                    Phi = msign(
                        M_hat + lam_star.reshape(-1, 1, 1).to(M_hat.dtype) * Theta,
                        steps=ns_steps,
                        compute_dtype=msign_dtype,
                    )

            # Final guard before write-back.
            if not torch.isfinite(Phi).all():
                _log.warning(
                    "SSO[%s]: non-finite Phi for param shape=%s, skipping update",
                    mode, tuple(p.shape),
                )
                continue

            # Step 12: μP-scaled update W ← W − η R Φ (+ optional matrix wd).
            if wd != 0.0:
                p.data.mul_(1.0 - lr * wd)
            if p.ndim == 2:
                step_scale = (-lr * R_state).to(Phi.dtype)
                p.data.add_(Phi * step_scale)
            else:
                # Per-expert scaling: R_state is [B], so pre-scale Phi.
                step_scale = (-lr * R_state).reshape(-1, 1, 1).to(Phi.dtype)
                p.data.add_(Phi * step_scale)

    # ---- foreach AdamW branch ----

    def _adamw_step_foreach(self, params: list[torch.nn.Parameter], group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]

        exp_avgs: list[Tensor] = []
        exp_avg_sqs: list[Tensor] = []
        grads: list[Tensor] = []
        for p in params:
            state = self.state[p]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            grads.append(p.grad)

        group["step"] = group.get("step", 0) + 1
        step = group["step"]
        bc1 = 1.0 - beta1 ** step
        bc2 = 1.0 - beta2 ** step
        step_size = lr / bc1
        bc2_sqrt = math.sqrt(bc2)

        if wd != 0.0:
            torch._foreach_mul_(params, 1.0 - lr * wd)
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)
        denom = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denom, bc2_sqrt)
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)


# ====================================================================
# 6. Convenience: auto-partition a module's named_parameters
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


def partition_for_sso(
    named_params: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    adamw_name_hints: tuple[str, ...] = _ADAMW_NAME_HINTS,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Split named params into (sso_matrix, adamw_rest).

    Matrix branch takes 2D and 3D params whose names don't match any hint;
    everything else (1D, biases, embedding-like) goes to AdamW.
    """
    sso: list[torch.nn.Parameter] = []
    adamw: list[torch.nn.Parameter] = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        blacklisted = any(h in name for h in adamw_name_hints)
        if not blacklisted and p.ndim in (2, 3):
            sso.append(p)
        else:
            adamw.append(p)
    return sso, adamw


def make_sso_from_module(
    module: torch.nn.Module,
    **kwargs,
) -> SSO:
    """Build SSO by auto-partitioning ``module.named_parameters()``."""
    sso_params, adamw_params = partition_for_sso(module.named_parameters())
    return SSO(sso_params, adamw_params, **kwargs)


__all__ = [
    "SSO",
    "make_sso_from_module",
    "msign",
    "partition_for_sso",
    "power_iteration",
    "solve_lambda",
    "spectral_radius",
]
