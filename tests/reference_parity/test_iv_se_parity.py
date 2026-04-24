"""IV robust / clustered SE parity tests.

Validates that ``sp.iv.iv`` / ``sp.ivreg`` produce the correct 2SLS
sandwich covariance using the **projected** regressor matrix
:math:`\\hat X = P_W X` in the meat, not the original :math:`X`.

References
----------
- Cameron & Miller (2015). A Practitioner's Guide to Cluster-Robust
  Inference. *Journal of Human Resources*, 50(2), 317-372.
- Baum, Schaffer, Stillman (2007). Enhanced routines for IV/GMM
  estimation and testing. *Stata Journal*, 7(4), 465-506.
- linearmodels.iv.covariance.ClusteredCovariance (docstring &
  implementation at :pycode:`xhat_e = z @ (pinvz @ x) * eps`).

The correct 2SLS cluster "meat" is

.. math::

   \\hat S = \\sum_c \\Big(\\sum_{i\\in c}\\hat x_i \\hat u_i\\Big)
                      \\Big(\\sum_{i\\in c}\\hat x_i \\hat u_i\\Big)^{\\top}

with :math:`\\hat X = P_W X`, where :math:`P_W` is the projection onto
the full instrument set (exog + excluded).  Using the unprojected X in
the meat gives a consistent estimator for OLS, but inflates the 2SLS
cluster SE by a factor depending on how poorly the first stage fits
the endogenous regressor.

The "bread" of the sandwich is :math:`(X'AX)^{-1}` which for k-class
with :math:`A = (1-\\kappa)I + \\kappa P_W` equals
:math:`(\\hat X'\\hat X)^{-1}` when :math:`\\kappa = 1`.  That part
has always been correct in StatsPAI; this test targets only the meat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Deterministic DGP with cluster structure and weak-ish first stage so that
# X_hat and X differ substantively in the endog column.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iv_cluster_data():
    rng = np.random.default_rng(20260424)
    n_clusters = 40
    cluster_size = 25
    n = n_clusters * cluster_size
    cl = np.repeat(np.arange(n_clusters), cluster_size)
    alpha = rng.normal(size=n_clusters)[cl]
    eta = rng.normal(size=n_clusters)[cl]
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x = rng.normal(size=n)
    d = 0.7 * z1 + 0.4 * z2 + 0.3 * x + 0.6 * eta + rng.normal(size=n)
    u = 0.8 * alpha + rng.normal(size=n)
    y = 1.0 + 2.0 * d + 0.5 * x + u
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "x": x, "cl": cl})


def _hand_2sls_sandwich(df, cov_type: str):
    """Closed-form 2SLS sandwich using projected X̂ in the meat.

    Returns the vector of SEs in the order ``[const, x, d]``.  Uses
    Cameron–Miller (2015) cluster correction ``(G/(G-1))((n-1)/(n-k))``
    and Wooldridge HC1 correction ``n/(n-k)`` — matching StatsPAI's
    existing finite-sample conventions so alignment is byte-exact.
    """
    y = df["y"].to_numpy()
    X_exog = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    X_endog = df["d"].to_numpy().reshape(-1, 1)
    Z = np.column_stack([df["z1"].to_numpy(), df["z2"].to_numpy()])
    W = np.column_stack([X_exog, Z])
    X_act = np.column_stack([X_exog, X_endog])
    PW = W @ np.linalg.solve(W.T @ W, W.T)
    X_hat = np.column_stack([X_exog, PW @ X_endog])

    n, k = X_act.shape
    bread = np.linalg.inv(X_hat.T @ X_hat)
    beta = bread @ (X_hat.T @ y)
    resid = y - X_act @ beta

    if cov_type == "cluster":
        cl = df["cl"].to_numpy()
        G = len(np.unique(cl))
        meat = np.zeros((k, k))
        for c in np.unique(cl):
            idx = cl == c
            s = X_hat[idx].T @ resid[idx]
            meat += np.outer(s, s)
        corr = (G / (G - 1)) * ((n - 1) / (n - k))
        vcv = corr * bread @ meat @ bread
    elif cov_type == "hc0":
        meat = X_hat.T @ np.diag(resid ** 2) @ X_hat
        vcv = bread @ meat @ bread
    elif cov_type == "hc1":
        weights = (n / (n - k)) * resid ** 2
        meat = X_hat.T @ np.diag(weights) @ X_hat
        vcv = bread @ meat @ bread
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")

    return np.sqrt(np.maximum(np.diag(vcv), 0.0)), beta


class TestIV2SLSSandwichUsesProjectedX:
    """The 2SLS sandwich meat must use projected X̂ = P_W X, not X."""

    def test_point_estimates_match_hand_computed(self, iv_cluster_data):
        r = sp.ivreg("y ~ (d ~ z1 + z2) + x", data=iv_cluster_data, cluster="cl")
        _, beta_hand = _hand_2sls_sandwich(iv_cluster_data, "cluster")
        # params order in StatsPAI: [Intercept, x, d]
        assert np.allclose(
            np.array([r.params["Intercept"], r.params["x"], r.params["d"]]),
            beta_hand, atol=1e-10,
        )

    def test_cluster_se_matches_projected_meat(self, iv_cluster_data):
        """Cluster SE must match hand-computed projected-X̂ formula."""
        r = sp.ivreg("y ~ (d ~ z1 + z2) + x", data=iv_cluster_data, cluster="cl")
        se_hand, _ = _hand_2sls_sandwich(iv_cluster_data, "cluster")
        sp_se = np.array([r.std_errors["Intercept"],
                          r.std_errors["x"],
                          r.std_errors["d"]])
        assert np.allclose(sp_se, se_hand, rtol=1e-10, atol=1e-10), (
            f"Cluster SE mismatch:\n"
            f"  StatsPAI   : {sp_se}\n"
            f"  projected  : {se_hand}"
        )

    def test_hc0_se_matches_projected_meat(self, iv_cluster_data):
        r = sp.ivreg("y ~ (d ~ z1 + z2) + x", data=iv_cluster_data, robust="hc0")
        se_hand, _ = _hand_2sls_sandwich(iv_cluster_data, "hc0")
        sp_se = np.array([r.std_errors["Intercept"],
                          r.std_errors["x"],
                          r.std_errors["d"]])
        assert np.allclose(sp_se, se_hand, rtol=1e-10, atol=1e-10), (
            f"HC0 SE mismatch:\n"
            f"  StatsPAI   : {sp_se}\n"
            f"  projected  : {se_hand}"
        )

    def test_hc1_se_matches_projected_meat(self, iv_cluster_data):
        r = sp.ivreg("y ~ (d ~ z1 + z2) + x", data=iv_cluster_data, robust="hc1")
        se_hand, _ = _hand_2sls_sandwich(iv_cluster_data, "hc1")
        sp_se = np.array([r.std_errors["Intercept"],
                          r.std_errors["x"],
                          r.std_errors["d"]])
        assert np.allclose(sp_se, se_hand, rtol=1e-10, atol=1e-10), (
            f"HC1 SE mismatch:\n"
            f"  StatsPAI   : {sp_se}\n"
            f"  projected  : {se_hand}"
        )


class TestIVvsLinearmodelsParity:
    """Close parity with linearmodels (after normalising dof conventions)."""

    def test_cluster_se_matches_linearmodels_debiased(self, iv_cluster_data):
        """linearmodels ClusteredCovariance with ``debiased=True`` uses
        ``(G/(G-1))((n-1)/(n-k))`` — identical to StatsPAI's correction —
        so the two packages must agree to machine precision once both
        use the projected X̂ meat.
        """
        lm = pytest.importorskip("linearmodels.iv")
        r_sp = sp.ivreg("y ~ (d ~ z1 + z2) + x",
                        data=iv_cluster_data, cluster="cl")
        r_lm = lm.IV2SLS.from_formula(
            "y ~ 1 + x + [d ~ z1 + z2]", iv_cluster_data
        ).fit(cov_type="clustered", clusters=iv_cluster_data["cl"],
              debiased=True)
        sp_se = np.array([r_sp.std_errors["Intercept"],
                          r_sp.std_errors["x"],
                          r_sp.std_errors["d"]])
        lm_se = np.array([r_lm.std_errors["Intercept"],
                          r_lm.std_errors["x"],
                          r_lm.std_errors["d"]])
        assert np.allclose(sp_se, lm_se, rtol=1e-8, atol=1e-10), (
            f"Cluster SE mismatch:\n"
            f"  StatsPAI      : {sp_se}\n"
            f"  linearmodels  : {lm_se}"
        )

    def test_hc0_se_matches_linearmodels(self, iv_cluster_data):
        """HC0 has no dof correction → should match linearmodels exactly."""
        lm = pytest.importorskip("linearmodels.iv")
        r_sp = sp.ivreg("y ~ (d ~ z1 + z2) + x",
                        data=iv_cluster_data, robust="hc0")
        r_lm = lm.IV2SLS.from_formula(
            "y ~ 1 + x + [d ~ z1 + z2]", iv_cluster_data
        ).fit(cov_type="robust", debiased=False)
        sp_se = np.array([r_sp.std_errors["Intercept"],
                          r_sp.std_errors["x"],
                          r_sp.std_errors["d"]])
        lm_se = np.array([r_lm.std_errors["Intercept"],
                          r_lm.std_errors["x"],
                          r_lm.std_errors["d"]])
        assert np.allclose(sp_se, lm_se, rtol=1e-8, atol=1e-10), (
            f"HC0 SE mismatch:\n"
            f"  StatsPAI      : {sp_se}\n"
            f"  linearmodels  : {lm_se}"
        )
