"""Heckman (1979) two-step SE parity tests.

Before v1.6.6 ``sp.heckman`` reported an ad-hoc HC1-style sandwich that
ignored both (a) the heteroskedasticity induced by selection,
``Var(y|X, D=1) = σ² (1 − ρ² δ_i)``, and (b) the uncertainty in γ̂
(the probit first-stage estimator) propagated through the inverse
Mills ratio λ̂ — the "generated regressor" problem.

This release restores the textbook Heckman (1979) / Greene (2003) /
Wooldridge (2010) two-step analytical variance:

.. math::

   V(\\hat β) = \\hat σ²\\, (X_*'X_*)^{-1}
                \\big[ X_*'(I − \\hat ρ² D_δ) X_*
                       + \\hat ρ²\\, F V̂_γ F' \\big]
                (X_*'X_*)^{-1}

where

- ``X_*`` is the second-stage design including λ̂ as its last column,
- ``δ_i = λ̂_i (λ̂_i + Z_i γ̂) ≥ 0`` (Heckman 1979, eq. 17),
- ``D_δ = diag(δ_i)``,
- ``F = X_*'\\, D_δ\\, Z`` (``k × q``),
- ``V̂_γ = (Z'\\,diag(w_i)\\,Z)^{-1}`` with probit info weights
  ``w_i = φ(Z_iγ̂)²/[Φ(Z_iγ̂)(1 − Φ(Z_iγ̂))]``,
- ``σ̂² = RSS / n + β̂_λ² · mean(δ_i)`` (Greene 2003, eq. 22-21).

References
----------
- Heckman, J. J. (1979). "Sample Selection Bias as a Specification
  Error." *Econometrica*, 47(1), 153–161.
- Greene, W. H. (2003). *Econometric Analysis*, 5th ed., §22.4.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and
  Panel Data*, 2nd ed., §19.6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp


# ---------------------------------------------------------------------------
# DGP with known selection structure and valid exclusion restriction
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def heckman_data():
    """n=2000 units. z1 enters both equations (kept in x), z2 enters only
    selection (valid exclusion restriction). Errors correlated ρ=0.5.
    """
    rng = np.random.default_rng(20260424)
    n = 2000
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    rho = 0.5
    u1 = rng.normal(size=n)
    u2 = rho * u1 + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    # Selection equation (probit latent)
    d_star = 0.2 + 0.6 * z1 + 0.7 * z2 + u1
    d = (d_star > 0).astype(int)
    # Outcome (observed only when d == 1)
    y = 1.0 + 2.0 * z1 + u2
    return pd.DataFrame({"y": y, "z1": z1, "z2": z2, "d": d})


# ---------------------------------------------------------------------------
# Hand-computed Heckman (1979) analytical variance
# ---------------------------------------------------------------------------

def _hand_heckman_variance(df):
    """Compute (β̂, analytical SE) using the Heckman (1979) / Greene
    (2003) formula directly, independent of sp.heckman. Returns
    (beta_vec, se_vec) in the order returned by sp.heckman:
    [const, x_vars..., lambda].
    """
    n = len(df)
    D = df["d"].to_numpy(dtype=float)
    Z = np.column_stack([np.ones(n), df["z1"].to_numpy(), df["z2"].to_numpy()])

    # Probit IRLS — identical kernel to sp.heckman._probit_fit.
    gamma = np.zeros(Z.shape[1])
    for _ in range(100):
        Zg = Z @ gamma
        Phi = np.clip(stats.norm.cdf(Zg), 1e-12, 1 - 1e-12)
        phi = stats.norm.pdf(Zg)
        w = phi ** 2 / (Phi * (1 - Phi))
        score = Z.T @ ((D - Phi) * phi / (Phi * (1 - Phi)))
        H = -(Z.T @ (w[:, None] * Z))
        delta_g = np.linalg.solve(H, score)
        gamma -= delta_g
        if np.max(np.abs(delta_g)) < 1e-10:
            break
    V_gamma = np.linalg.inv(Z.T @ (w[:, None] * Z))  # asymptotic V(γ̂)

    # IMR for all obs; restrict to selected for second stage.
    Zg_all = Z @ gamma
    imr_all = stats.norm.pdf(Zg_all) / np.clip(stats.norm.cdf(Zg_all),
                                               1e-12, 1 - 1e-12)

    sel = D == 1
    Y_v = df.loc[sel, "y"].to_numpy()
    X_v = np.column_stack([np.ones(sel.sum()),
                           df.loc[sel, "z1"].to_numpy(),
                           imr_all[sel]])
    Z_v = Z[sel]
    Zg_v = Zg_all[sel]
    imr_v = imr_all[sel]

    n_sel, k = X_v.shape
    beta = np.linalg.solve(X_v.T @ X_v, X_v.T @ Y_v)
    resid = Y_v - X_v @ beta
    rss = float(resid @ resid)

    beta_lambda = float(beta[-1])
    delta_v = imr_v * (imr_v + Zg_v)   # ≥ 0 by Mills' ratio inequality
    # Greene (2003) eq. 22-21: σ̂² = RSS/n_sel + β_λ² · mean(δ)
    sigma2 = rss / n_sel + beta_lambda ** 2 * float(np.mean(delta_v))
    rho2 = beta_lambda ** 2 / sigma2

    XtX_inv = np.linalg.inv(X_v.T @ X_v)
    # Heteroskedastic contribution X*' (I − ρ² diag(δ)) X*
    w_e = 1.0 - rho2 * delta_v
    M_het = X_v.T @ (w_e[:, None] * X_v)
    # Generated-regressor contribution ρ² · F · V_γ · F'
    F = X_v.T @ (delta_v[:, None] * Z_v)
    Q = rho2 * (F @ V_gamma @ F.T)
    vcv = sigma2 * XtX_inv @ (M_het + Q) @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(vcv), 0.0))
    return beta, se


class TestHeckmanVarianceCorrection:
    """``sp.heckman`` must use the Heckman (1979) / Greene (2003)
    analytical two-step variance, not a naive HC1 sandwich."""

    def test_point_estimates_match_hand_computed(self, heckman_data):
        r = sp.heckman(heckman_data, y="y", x=["z1"],
                       select="d", z=["z1", "z2"])
        beta_hand, _ = _hand_heckman_variance(heckman_data)
        sp_beta = r.detail["coefficient"].to_numpy()
        assert np.allclose(sp_beta, beta_hand, rtol=1e-8, atol=1e-10), (
            f"β̂ mismatch (fix shouldn't change point estimates):\n"
            f"  StatsPAI : {sp_beta}\n"
            f"  hand     : {beta_hand}"
        )

    def test_se_matches_heckman_1979_formula(self, heckman_data):
        r = sp.heckman(heckman_data, y="y", x=["z1"],
                       select="d", z=["z1", "z2"])
        _, se_hand = _hand_heckman_variance(heckman_data)
        sp_se = r.detail["se"].to_numpy()
        assert np.allclose(sp_se, se_hand, rtol=1e-8, atol=1e-10), (
            f"SE mismatch (Heckman 1979 analytical formula):\n"
            f"  StatsPAI : {sp_se}\n"
            f"  hand     : {se_hand}"
        )

    def test_model_info_exposes_sigma_rho(self, heckman_data):
        """``sigma`` and ``rho`` in model_info must use the
        Greene (2003) consistent σ̂² formula, not a naive RSS/(n−k)."""
        r = sp.heckman(heckman_data, y="y", x=["z1"],
                       select="d", z=["z1", "z2"])
        beta_hand, _ = _hand_heckman_variance(heckman_data)
        # σ̂² reconstruction from hand-computed quantities
        D = heckman_data["d"].to_numpy(dtype=float)
        Z = np.column_stack([np.ones(len(heckman_data)),
                             heckman_data["z1"].to_numpy(),
                             heckman_data["z2"].to_numpy()])
        gamma = np.zeros(Z.shape[1])
        for _ in range(100):
            Zg = Z @ gamma
            Phi = np.clip(stats.norm.cdf(Zg), 1e-12, 1 - 1e-12)
            phi = stats.norm.pdf(Zg)
            score = Z.T @ ((D - Phi) * phi / (Phi * (1 - Phi)))
            w_i = phi ** 2 / (Phi * (1 - Phi))
            H = -(Z.T @ (w_i[:, None] * Z))
            g = np.linalg.solve(H, score)
            gamma -= g
            if np.max(np.abs(g)) < 1e-10:
                break
        imr_all = stats.norm.pdf(Z @ gamma) / np.clip(
            stats.norm.cdf(Z @ gamma), 1e-12, 1 - 1e-12)
        sel = D == 1
        delta_v = imr_all[sel] * (imr_all[sel] + (Z @ gamma)[sel])
        Y_v = heckman_data.loc[sel, "y"].to_numpy()
        X_v = np.column_stack([np.ones(sel.sum()),
                               heckman_data.loc[sel, "z1"].to_numpy(),
                               imr_all[sel]])
        resid = Y_v - X_v @ beta_hand
        beta_lambda = float(beta_hand[-1])
        sigma2_hand = (resid @ resid) / sel.sum() + \
            beta_lambda ** 2 * float(np.mean(delta_v))
        sigma_hand = float(np.sqrt(sigma2_hand))
        rho_hand = beta_lambda / sigma_hand

        assert r.model_info["sigma"] == pytest.approx(sigma_hand, rel=1e-8)
        assert r.model_info["rho"] == pytest.approx(rho_hand, rel=1e-8)
