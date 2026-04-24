"""Standalone LIML (``sp.liml`` / ``sp.iv.liml``) SE parity tests.

The standalone LIML estimator in ``statspai.regression.advanced_iv`` is a
separate code path from the ``_k_class_fit`` dispatcher (fixed in v1.6.4):
users reach it via ``sp.liml(...)`` directly, not via
``sp.ivreg(method='liml')``. This test pins its cluster / robust SE to:

1. A hand-computed sandwich using the k-class transformed regressor
   ``AX = (I − κ M_Z) X``, per the k-class FOC
   ``X' (I − κ M_Z) (y − X β) = 0``.
2. ``linearmodels.iv.IVLIML`` with ``cov_type='clustered', debiased=True``.

Before v1.6.5 the standalone LIML used raw ``X`` in the cluster / robust
meat, identical in character to the 2SLS bug fixed in v1.6.4 but in a
different module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


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


def _hand_liml_cluster_sandwich(df):
    """Hand-computed LIML cluster SE using (I−κM_Z)X in the meat.

    Uses ``scipy.linalg.eigh`` on the symmetric generalized eigenvalue
    problem to pick κ — more numerically stable than the raw
    ``np.linalg.eigvals(inv(A) @ B)`` path and matches ``linearmodels``
    to machine precision on well-conditioned DGPs.

    Returns (SE vector in order [const, x, d], params, kappa).
    """
    from scipy.linalg import eigh as _eigh
    y = df["y"].to_numpy()
    X_exog = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    X_endog = df["d"].to_numpy().reshape(-1, 1)
    Z = np.column_stack([df["z1"].to_numpy(), df["z2"].to_numpy()])
    Z_all = np.column_stack([X_exog, Z])
    X_all = np.column_stack([X_exog, X_endog])
    cl = df["cl"].to_numpy()

    n, k = X_all.shape
    Px = X_exog @ np.linalg.solve(X_exog.T @ X_exog, X_exog.T)
    Mx = np.eye(n) - Px
    Pz = Z_all @ np.linalg.solve(Z_all.T @ Z_all, Z_all.T)
    Mz = np.eye(n) - Pz

    # κ_LIML = smallest root of S_exog v = κ S_full v, both symmetric PSD.
    W = np.column_stack([y.reshape(-1, 1), X_endog])
    S_full = W.T @ Mz @ W
    S_exog = W.T @ Mx @ W
    kappa = float(np.min(_eigh(S_exog, S_full, eigvals_only=True)))

    I_kMz = np.eye(n) - kappa * Mz
    XAX = X_all.T @ I_kMz @ X_all
    beta = np.linalg.solve(XAX, X_all.T @ I_kMz @ y)
    resid = y - X_all @ beta
    bread = np.linalg.inv(XAX)

    AX = I_kMz @ X_all
    G = len(np.unique(cl))
    meat = np.zeros((k, k))
    for c in np.unique(cl):
        idx = cl == c
        s = AX[idx].T @ resid[idx]
        meat += np.outer(s, s)
    corr = (G / (G - 1)) * ((n - 1) / (n - k))
    vcv = corr * bread @ meat @ bread
    return np.sqrt(np.maximum(np.diag(vcv), 0.0)), beta, kappa


class TestStandaloneLIMLSandwichUsesProjectedX:
    """``sp.liml`` cluster SE must use (I−κM_Z)X in the meat, not raw X."""

    def test_point_estimates_match_hand_computed(self, iv_cluster_data):
        r = sp.liml(data=iv_cluster_data, y="y", x_endog=["d"],
                    x_exog=["x"], z=["z1", "z2"], cluster="cl")
        _, beta_hand, _ = _hand_liml_cluster_sandwich(iv_cluster_data)
        sp_beta = np.array([r.params["_cons"], r.params["x"], r.params["d"]])
        assert np.allclose(sp_beta, beta_hand, rtol=1e-6, atol=1e-6), (
            f"Point estimate mismatch (κ may differ between eigenvalue "
            f"solvers, but should agree to float tolerance on this DGP):\n"
            f"  StatsPAI   : {sp_beta}\n"
            f"  hand       : {beta_hand}"
        )

    def test_cluster_se_matches_projected_meat(self, iv_cluster_data):
        r = sp.liml(data=iv_cluster_data, y="y", x_endog=["d"],
                    x_exog=["x"], z=["z1", "z2"], cluster="cl")
        se_hand, _, _ = _hand_liml_cluster_sandwich(iv_cluster_data)
        sp_se = np.array([r.std_errors["_cons"],
                          r.std_errors["x"],
                          r.std_errors["d"]])
        assert np.allclose(sp_se, se_hand, rtol=1e-8, atol=1e-10), (
            f"Cluster SE mismatch (projected-meat formula):\n"
            f"  StatsPAI   : {sp_se}\n"
            f"  hand       : {se_hand}"
        )

    def test_standalone_matches_kclass_dispatcher(self, iv_cluster_data):
        """``sp.liml(...)`` must agree byte-for-byte with
        ``sp.ivreg(..., method='liml')``. Before v1.6.5 the standalone
        entry point diverged on both κ (non-symmetric eigvals) and SE
        (raw X in meat); the fix aligns it with the canonical
        ``_k_class_fit`` implementation.
        """
        r_std = sp.liml(data=iv_cluster_data, y="y", x_endog=["d"],
                        x_exog=["x"], z=["z1", "z2"], cluster="cl")
        r_disp = sp.ivreg("y ~ (d ~ z1 + z2) + x", data=iv_cluster_data,
                          method="liml", cluster="cl")
        std_params = np.array([r_std.params["_cons"],
                               r_std.params["x"],
                               r_std.params["d"]])
        disp_params = np.array([r_disp.params["Intercept"],
                                r_disp.params["x"],
                                r_disp.params["d"]])
        std_se = np.array([r_std.std_errors["_cons"],
                           r_std.std_errors["x"],
                           r_std.std_errors["d"]])
        disp_se = np.array([r_disp.std_errors["Intercept"],
                            r_disp.std_errors["x"],
                            r_disp.std_errors["d"]])
        assert np.allclose(std_params, disp_params, rtol=1e-10, atol=1e-12)
        assert np.allclose(std_se, disp_se, rtol=1e-10, atol=1e-12)


class TestLIMLvsLinearmodelsParity:
    """Close parity with linearmodels.IVLIML with debiased=True.

    StatsPAI uses the influence-function convention AX = (I−κM_Z) X in
    the meat (strict k-class FOC). ``linearmodels.IVLIML`` uses the
    2SLS-style meat X̂ = P_Z X regardless of κ. The two conventions are
    asymptotically equivalent (κ → 1) and coincide exactly at κ = 1
    (2SLS). On this DGP the finite-sample gap at κ ≈ 1.0017 is ~0.13%
    — well within the convention-difference tolerance we allow here.
    """

    def test_cluster_se_close_to_linearmodels_debiased(self, iv_cluster_data):
        lm = pytest.importorskip("linearmodels.iv")
        r_sp = sp.liml(data=iv_cluster_data, y="y", x_endog=["d"],
                       x_exog=["x"], z=["z1", "z2"], cluster="cl")
        r_lm = lm.IVLIML.from_formula(
            "y ~ 1 + x + [d ~ z1 + z2]", iv_cluster_data
        ).fit(cov_type="clustered", clusters=iv_cluster_data["cl"],
              debiased=True)
        # Point estimates use the same κ on both sides — these must match
        # to machine precision.
        assert np.allclose(r_sp.params["d"], r_lm.params["d"],
                           rtol=1e-8, atol=1e-10), (
            f"LIML β[d]: StatsPAI={r_sp.params['d']}, "
            f"linearmodels={r_lm.params['d']}"
        )
        # SEs differ by the AX vs X̂ meat convention; asymptotically
        # equivalent, finite-sample gap ~0.2% at typical κ.
        sp_se = np.array([r_sp.std_errors["_cons"],
                          r_sp.std_errors["x"],
                          r_sp.std_errors["d"]])
        lm_se = np.array([r_lm.std_errors["Intercept"],
                          r_lm.std_errors["x"],
                          r_lm.std_errors["d"]])
        assert np.allclose(sp_se, lm_se, rtol=5e-3, atol=1e-6), (
            f"LIML cluster SE (convention gap AX vs X̂ should be <0.5%):\n"
            f"  StatsPAI      : {sp_se}\n"
            f"  linearmodels  : {lm_se}"
        )
