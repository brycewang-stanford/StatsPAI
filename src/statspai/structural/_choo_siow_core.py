"""Numerical kernels for the Choo-Siow transferable-utility matching model.

Model (homoskedastic logit heterogeneity, scale 1). Man ``i`` of type
``x`` matched with a woman of type ``y`` gets ``alpha_xy + tau + eps_iy``,
she gets ``gamma_xy - tau + eta_jx``; the shocks are type-I extreme value
and depend on the partner's *type* only (separability). In the large
market the equilibrium matching satisfies

    mu_xy = sqrt(mu_x0 * mu_0y) * exp(Phi_xy / 2),   Phi = alpha + gamma,

with the accounting identities ``n_x = mu_x0 + sum_y mu_xy`` and
``m_y = mu_0y + sum_x mu_xy`` [@choo2006who; @galichon2022cupid].

Everything here works on plain arrays; the public wrapper and the result
class live in ``choo_siow.py`` / ``_choo_siow_result.py``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..exceptions import ConvergenceFailure, IdentificationFailure


def ipfp_equilibrium(
    phi: np.ndarray,
    men: np.ndarray,
    women: np.ndarray,
    *,
    tol: float = 1e-12,
    maxiter: int = 100_000,
) -> Dict[str, np.ndarray]:
    """Solve the Choo-Siow equilibrium by IPFP.

    With ``a_x = sqrt(mu_x0)``, ``b_y = sqrt(mu_0y)`` and
    ``K = exp(Phi / 2)``, the men's margins read
    ``a_x^2 + a_x (K b)_x = n_x``: a quadratic in ``a_x`` given ``b``,
    whose positive root ``a_x = 2 n_x / (B_x + sqrt(B_x^2 + 4 n_x))``
    (``B = K b``) is written without cancellation. Alternating the men's and
    the women's updates is the IPFP of [@galichon2022cupid]; each update
    solves one block of the first-order conditions of a strictly convex
    dual exactly, so the iteration converges.

    Returns ``mu`` (X x Y), ``singles_men`` (X), ``singles_women`` (Y),
    ``n_iter`` and ``max_margin_error`` (largest |margin error| / margin
    over both sexes).
    """
    phi = np.asarray(phi, dtype=float)
    n = np.asarray(men, dtype=float)
    m = np.asarray(women, dtype=float)
    K = np.exp(phi / 2.0)
    # start from everyone single
    b = np.sqrt(m)
    a = np.sqrt(n)
    err = np.inf
    it = 0
    for it in range(1, maxiter + 1):
        B = K @ b
        a = 2.0 * n / (B + np.sqrt(B * B + 4.0 * n))
        C = K.T @ a
        b = 2.0 * m / (C + np.sqrt(C * C + 4.0 * m))
        # the women's margins hold exactly after their update; check men's
        mu = a[:, None] * K * b[None, :]
        err = float(np.max(np.abs(a * a + mu.sum(axis=1) - n) / np.maximum(n, 1e-300)))
        if err < tol:
            break
    else:
        raise ConvergenceFailure(
            f"choo_siow: IPFP did not reach margin error {tol:g} in "
            f"{maxiter} iterations (last {err:.3g}).",
            recovery_hint="Very large |Phi| makes IPFP slow; raise maxiter or "
            "check the surplus values.",
            diagnostics={"max_margin_error": err, "n_iter": maxiter},
        )
    mu = a[:, None] * K * b[None, :]
    err_w = float(np.max(np.abs(b * b + mu.sum(axis=0) - m) / np.maximum(m, 1e-300)))
    return {
        "mu": mu,
        "singles_men": a * a,
        "singles_women": b * b,
        "n_iter": np.asarray(it),
        "max_margin_error": np.asarray(max(err, err_w)),
    }


def nonparametric_phi(
    mu: np.ndarray, mu_x0: np.ndarray, mu_0y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """``Phi_xy = log(mu_xy^2 / (mu_x0 mu_0y))`` and its delta-method covariance.

    ``Phi`` is ``A log(c)`` for the vector ``c`` of household counts
    (couples, single men, single women) with each row of ``A`` summing to
    zero. Under multinomial sampling of households
    ``Cov(log c) ~= diag(1/c) - 11'/H``; the second term is annihilated by
    the zero row sums, so ``Cov(Phi) = A diag(1/c) A'`` exactly at first
    order, i.e.

        Cov(Phi_xy, Phi_x'y') = 4 [xy = x'y'] / mu_xy + [x = x'] / mu_x0
                                + [y = y'] / mu_0y.

    The same holds under Poisson sampling. Cells with ``mu_xy = 0`` get
    ``-inf`` and ``NaN`` (co)variances; the caller warns.
    """
    X, Y = mu.shape
    with np.errstate(divide="ignore"):
        phi = 2.0 * np.log(mu) - np.log(mu_x0)[:, None] - np.log(mu_0y)[None, :]
    inv_xy = np.where(mu > 0, 1.0 / np.where(mu > 0, mu, 1.0), np.nan).ravel()
    xs = np.repeat(np.arange(X), Y)
    ys = np.tile(np.arange(Y), X)
    V = (
        np.diag(4.0 * inv_xy)
        + (xs[:, None] == xs[None, :]) / mu_x0[xs][:, None]
        + (ys[:, None] == ys[None, :]) / mu_0y[ys][:, None]
    )
    bad = ~np.isfinite(inv_xy)
    V[bad, :] = np.nan
    V[:, bad] = np.nan
    return phi, V


def _design(basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Design ``Z`` and weights ``w`` of the weighted Poisson GLM.

    Cells are stacked couples (row-major ``xy``), single men, single
    women. Parameters are ``(u (X), v (Y), beta (K))``. Couples:
    ``eta_xy = (phi_xy' beta - u_x - v_y) / 2`` with weight 2; single men
    ``eta_x0 = -u_x``, single women ``eta_0y = -v_y``, weight 1.
    """
    X, Y, Kb = basis.shape
    XY = X * Y
    phi_mat = basis.reshape(XY, Kb)
    ex = np.kron(np.eye(X), np.ones((Y, 1)))
    ey = np.kron(np.ones((X, 1)), np.eye(Y))
    top = np.hstack([-ex, -ey, phi_mat]) / 2.0
    mid = np.hstack([-np.eye(X), np.zeros((X, Y)), np.zeros((X, Kb))])
    bot = np.hstack([np.zeros((Y, X)), -np.eye(Y), np.zeros((Y, Kb))])
    Z = np.vstack([top, mid, bot])
    w = np.concatenate([2.0 * np.ones(XY), np.ones(X + Y)])
    return Z, w


def poisson_glm_fit(
    basis: np.ndarray,
    mu: np.ndarray,
    mu_x0: np.ndarray,
    mu_0y: np.ndarray,
    *,
    tol: float = 1e-12,
    maxiter: int = 200,
) -> Dict[str, object]:
    """Semilinear Choo-Siow by the weighted Poisson GLM.

    Minimises the convex
    ``F(u, v, beta) = sum_i w_i (exp(eta_i) - c_i eta_i)`` over the stacked
    cells (``c`` = observed counts scaled to one individual), whose
    first-order conditions are (i) the accounting identities for every type
    at the observed margins and (ii) the moment-matching conditions
    ``sum_xy phi_k(x, y) (mu_hat_xy - mu_xy(beta)) = 0`` -- the GLM with
    two-way fixed effects of [@galichon2024estimating]. Newton's method with
    step halving on ``F``.

    The variance is the sandwich ``A^-1 B A^-1``, ``A = Z' diag(w lambda) Z``,
    ``B = Z' diag(w^2 c) Z / N_ind`` (Poisson meat). Multinomial sampling of
    households adds ``-cc'/H`` to ``Var(c)``; that term moves only the
    direction ``(u, v) -> (u - 1, v - 1)``, which leaves ``beta``
    unchanged, so the ``beta`` block is the same under both schemes.
    """
    X, Y, Kb = basis.shape
    Z, w = _design(basis)
    n_ind = float(2.0 * mu.sum() + mu_x0.sum() + mu_0y.sum())
    c = np.concatenate([mu.ravel(), mu_x0, mu_0y]) / n_ind
    if np.linalg.matrix_rank(Z) < Z.shape[1]:
        raise IdentificationFailure(
            "choo_siow: the basis functions are collinear (with each other or "
            "with the type fixed effects), so beta is not identified.",
            recovery_hint="Drop redundant basis functions.",
            diagnostics={"rank": int(np.linalg.matrix_rank(Z)), "n_params": Z.shape[1]},
        )

    # start: exact single-type fixed effects, beta from least squares on
    # the finite nonparametric surplus
    u0 = -np.log(mu_x0 / n_ind)
    v0 = -np.log(mu_0y / n_ind)
    with np.errstate(divide="ignore"):
        phi_np = 2.0 * np.log(mu) - np.log(mu_x0)[:, None] - np.log(mu_0y)[None, :]
    fin = np.isfinite(phi_np).ravel()
    P = basis.reshape(X * Y, Kb)
    if fin.sum() >= Kb:
        beta0 = np.linalg.lstsq(P[fin], phi_np.ravel()[fin], rcond=None)[0]
    else:
        beta0 = np.zeros(Kb)
    gamma = np.concatenate([u0, v0, beta0])

    def F(g: np.ndarray) -> float:
        eta = Z @ g
        return float(np.sum(w * (np.exp(eta) - c * eta)))

    f = F(gamma)
    converged = False
    it = 0
    for it in range(0, maxiter + 1):
        lam = np.exp(Z @ gamma)
        grad = Z.T @ (w * (lam - c))
        H = Z.T @ ((w * lam)[:, None] * Z)
        step = -np.linalg.solve(H, grad)
        dec = float(-grad @ step)  # squared Newton decrement at gamma
        if dec < tol * tol:
            converged = True
            break
        if it == maxiter:
            break
        if dec < 1e-8:
            # inside the quadratic region a full step is safe, and F can no
            # longer resolve the decrease (it is ~dec, below rounding of F)
            gamma = gamma + step
            f = F(gamma)
            continue
        t = 1.0
        for _ in range(60):
            cand = gamma + t * step
            fc = F(cand)
            if np.isfinite(fc) and fc <= f + 1e-4 * t * float(grad @ step):
                break
            t *= 0.5
        gamma, f = cand, fc
    lam = np.exp(Z @ gamma)
    grad = Z.T @ (w * (lam - c))
    A = Z.T @ ((w * lam)[:, None] * Z)
    B = Z.T @ ((w * w * c)[:, None] * Z) / n_ind
    Ainv = np.linalg.inv(A)
    V = Ainv @ B @ Ainv
    V = (V + V.T) / 2.0
    XY = X * Y
    fitted = lam * n_ind
    return {
        "beta": gamma[X + Y :],
        "vcov_beta": V[X + Y :, X + Y :],
        "u": gamma[:X],
        "v": gamma[X : X + Y],
        "fitted_mu": fitted[:XY].reshape(X, Y),
        "fitted_singles_men": fitted[XY : XY + X],
        "fitted_singles_women": fitted[XY + X :],
        "converged": converged,
        "n_iter": it,
        "max_abs_score": float(np.max(np.abs(grad))),
    }
