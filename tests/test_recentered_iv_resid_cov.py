"""sp.recentered_iv(resid_cov=): the efficient instrument with a full Omega.

The instrument is ``(Omega + psi psi')^{-1} (z - mu)`` (Borusyak and Hull,
NBER WP 27845, eq. 9 and Lemma 3) with a non-diagonal
``Omega = E[ee' | w]``. The tests pin it to a dense solve, check that the
equicorrelated cluster form equals the dense matrix it describes and that
both reduce to the diagonal ``resid_var`` path, keep the randomization
test exact (Hodges-Lehmann root = estimate), and show the efficiency gain
on a known-truth design with cluster-correlated errors.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility, StatsPAIError


def _design(seed=7, n=240, n_lines=80, S=60):
    rng = np.random.default_rng(seed)
    cen = rng.uniform(0, 1, n)
    W = (rng.uniform(size=(n, n_lines)) < (0.05 + 0.3 * cen)[:, None]).astype(float)
    g = rng.binomial(1, 0.3, n_lines).astype(float)
    z = W @ g
    cf = np.column_stack([W @ rng.permutation(g) for _ in range(S)])
    eps = 2.0 * cen + rng.normal(scale=0.5, size=n)
    x = 0.7 * z + 0.8 * eps + rng.normal(size=n)
    df = pd.DataFrame(
        {
            "y": 0.5 * x + eps,
            "x": x,
            "z": z,
            "cen": cen,
            "f": rng.integers(0, 8, n),
            "cl": rng.integers(0, 40, n),
            "om": 0.2 + rng.uniform(0, 2, n) * (1 + cen),
            "psi": 2.0 * cen - 1.0,
        }
    )
    return df, cf


def _fit(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.recentered_iv(*args, **kw)


def _zt(df, cf):
    # The point estimate centres at the mean of the S counterfactual draws.
    return df["z"].to_numpy() - cf.mean(axis=1)


def _equicorrelated(df, rho):
    sd = np.sqrt(df["om"].to_numpy())
    same = df["cl"].to_numpy()[:, None] == df["cl"].to_numpy()[None, :]
    R = np.where(same, rho, 0.0)
    np.fill_diagonal(R, 1.0)
    return sd[:, None] * R * sd[None, :]


def _dense_pd(df, seed=2):
    # A positive definite Omega that depends on the exposure only.
    rng = np.random.default_rng(seed)
    n = len(df)
    A = rng.normal(size=(n, 6)) * df["cen"].to_numpy()[:, None]
    return A @ A.T + np.diag(df["om"].to_numpy())


def test_dense_resid_cov_matches_dense_solve():
    df, cf = _design()
    Om = _dense_pd(df)
    r = _fit(df, "y", "x", "z", cf, resid_cov=Om, resid_mean="psi")
    zt = _zt(df, cf)
    psi = df["psi"].to_numpy()
    w = df.assign(zs=np.linalg.solve(Om + np.outer(psi, psi), zt))
    ref = sp.iv("y ~ (x ~ zs)", data=w, robust="hc1")
    assert r.estimate == pytest.approx(float(ref.params["x"]), rel=1e-10)
    assert r.se == pytest.approx(float(ref.std_errors["x"]), rel=1e-10)
    info = r.diagnostics["efficient_instrument"]
    q = float(psi @ np.linalg.solve(Om, psi))
    assert info["resid_cov"] == "dense"
    assert info["resid_var"] is False
    assert info["nu"] == pytest.approx(q / (1 + q), rel=1e-10)
    assert r.diagnostics["ri_hodges_lehmann"] == pytest.approx(r.estimate, rel=1e-10)


def test_diagonal_dense_equals_resid_var_path():
    df, cf = _design()
    a = _fit(df, "y", "x", "z", cf, resid_var="om", resid_mean="psi")
    b = _fit(
        df, "y", "x", "z", cf, resid_cov=np.diag(df["om"].to_numpy()), resid_mean="psi"
    )
    assert b.estimate == pytest.approx(a.estimate, rel=1e-12)
    assert b.se == pytest.approx(a.se, rel=1e-12)
    assert b.diagnostics["ri_pvalue"] == a.diagnostics["ri_pvalue"]
    ea, eb = (
        a.diagnostics["efficient_instrument"],
        b.diagnostics["efficient_instrument"],
    )
    assert eb["nu"] == pytest.approx(ea["nu"], rel=1e-12)


def test_equicorrelated_rho_zero_equals_resid_var_path():
    df, cf = _design()
    a = _fit(df, "y", "x", "z", cf, resid_var="om", resid_mean="psi")
    b = _fit(
        df,
        "y",
        "x",
        "z",
        cf,
        resid_var="om",
        resid_mean="psi",
        resid_cov={"cluster": "cl", "rho": 0.0},
    )
    assert b.estimate == pytest.approx(a.estimate, rel=1e-12)
    assert b.se == pytest.approx(a.se, rel=1e-12)
    assert b.diagnostics["ri_pvalue"] == a.diagnostics["ri_pvalue"]


@pytest.mark.parametrize("rho", [0.45, -0.02])
def test_equicorrelated_equals_the_dense_matrix_it_describes(rho):
    df, cf = _design()
    a = _fit(
        df,
        "y",
        "x",
        "z",
        cf,
        resid_var="om",
        resid_mean="psi",
        resid_cov={"cluster": "cl", "rho": rho},
    )
    b = _fit(
        df, "y", "x", "z", cf, resid_cov=_equicorrelated(df, rho), resid_mean="psi"
    )
    assert a.estimate == pytest.approx(b.estimate, rel=1e-10)
    assert a.se == pytest.approx(b.se, rel=1e-10)
    assert a.diagnostics["ri_pvalue"] == b.diagnostics["ri_pvalue"]
    ea, eb = (
        a.diagnostics["efficient_instrument"],
        b.diagnostics["efficient_instrument"],
    )
    assert ea["nu"] == pytest.approx(eb["nu"], rel=1e-10)
    assert ea["resid_cov"] == "equicorrelated"
    assert ea["rho"] == rho
    assert ea["n_clusters"] == df["cl"].nunique()


def test_resid_cov_with_absorb_keeps_hodges_lehmann():
    df, cf = _design()
    df.loc[[0, 1], "f"] = [100, 101]  # two singletons
    for kw in (
        dict(resid_cov=_dense_pd(df)),
        dict(resid_var="om", resid_cov={"cluster": "cl", "rho": 0.3}),
    ):
        r = _fit(df, "y", "x", "z", cf, absorb="f", resid_mean="psi", **kw)
        assert r.diagnostics["ri_hodges_lehmann"] == pytest.approx(r.estimate, rel=1e-9)


def test_resid_cov_uses_estimation_rows_only():
    # A row dropped for a missing outcome takes its row and column of the
    # dense Omega with it; the result equals the fit on the complete rows.
    df, cf = _design()
    Om = _dense_pd(df)
    df2 = df.copy()
    df2.loc[5, "y"] = np.nan
    a = _fit(df2, "y", "x", "z", cf, resid_cov=Om)
    keep = np.arange(len(df)) != 5
    b = _fit(
        df.loc[keep].reset_index(drop=True),
        "y",
        "x",
        "z",
        cf[keep],
        resid_cov=Om[np.ix_(keep, keep)],
    )
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12)
    assert a.se == pytest.approx(b.se, rel=1e-12)


def test_cluster_correlated_errors_gain_over_diagonal_instrument():
    # z_i = e_i g_i with fair coins, so mu_i = e_i / 2 exactly; beta = 1.
    # Residuals share a cluster component: corr 0.6 within clusters of 5,
    # sd 0.3 + e_i^2. All three instruments are valid; the diagonal Omega
    # removes the heteroskedasticity, the equicorrelated one also the
    # common cluster shock. 150 replications: variance ratios 0.35
    # (diagonal / plain) and 0.42 (equicorrelated / diagonal) observed;
    # the bounds leave room for Monte Carlo noise.
    rng = np.random.default_rng(5)
    n, m, rho, reps = 400, 5, 0.6, 150
    g = np.repeat(np.arange(n // m), m)
    est = {"plain": [], "diag": [], "equi": []}
    for _ in range(reps):
        e = rng.uniform(0.2, 2, n)
        z = e * rng.binomial(1, 0.5, n)
        sd = 0.3 + e**2
        a = rng.normal(size=n // m)[g]
        eps = sd * (np.sqrt(rho) * a + np.sqrt(1 - rho) * rng.normal(size=n))
        df = pd.DataFrame({"y": z + eps, "z": z, "mu": 0.5 * e, "v": sd**2, "g": g})
        cf = np.column_stack([e * rng.binomial(1, 0.5, n) for _ in range(2)])
        kw = dict(expected="mu", ri=False)
        est["plain"].append(_fit(df, "y", "z", "z", cf, **kw).estimate)
        est["diag"].append(_fit(df, "y", "z", "z", cf, resid_var="v", **kw).estimate)
        est["equi"].append(
            _fit(
                df,
                "y",
                "z",
                "z",
                cf,
                resid_var="v",
                resid_cov={"cluster": "g", "rho": rho},
                **kw,
            ).estimate
        )
    est = {k: np.asarray(v) for k, v in est.items()}
    for v in est.values():
        assert abs(v.mean() - 1.0) < 4 * v.std(ddof=1) / np.sqrt(reps)
    var = {k: v.var(ddof=1) for k, v in est.items()}
    assert var["equi"] / var["diag"] < 0.65
    assert var["diag"] / var["plain"] < 0.6


@pytest.mark.parametrize(
    "kw",
    [
        dict(resid_cov=np.ones((5, 5))),  # wrong shape
        dict(resid_cov="not_a_matrix"),  # wrong shape (0-d)
        dict(resid_cov={"cluster": "cl"}),  # missing rho
        dict(resid_cov={"cluster": "cl", "rho": 0.3, "extra": 1}),
        dict(resid_cov={"cluster": "cl", "rho": 0.3}),  # no resid_var
        dict(resid_var="om", resid_cov={"cluster": "cl", "rho": 1.0}),
        dict(resid_var="om", resid_cov={"cluster": "cl", "rho": float("nan")}),
        dict(resid_var="om", resid_cov={"cluster": 3, "rho": 0.3}),
        # rho below -1/(m-1) for the largest cluster
        dict(resid_var="om", resid_cov={"cluster": "cl", "rho": -0.5}),
        dict(resid_var="om", resid_cov={"cluster": "cl", "rho": 0.3}, method="control"),
    ],
)
def test_resid_cov_argument_errors(kw):
    df, cf = _design()
    with pytest.raises(MethodIncompatibility):
        _fit(df, "y", "x", "z", cf, **kw)


def test_dense_resid_cov_errors():
    df, cf = _design()
    n = len(df)
    Om = _dense_pd(df)
    bad_sym = Om.copy()
    bad_sym[0, 1] += 1.0
    not_pd = Om.copy()
    not_pd[:, 0] = not_pd[0, :] = 0.0  # zero variance row
    with_nan = Om.copy()
    with_nan[3, 4] = with_nan[4, 3] = np.nan
    for M in (bad_sym, not_pd, with_nan):
        with pytest.raises(MethodIncompatibility):
            _fit(df, "y", "x", "z", cf, resid_cov=M)
    with pytest.raises(MethodIncompatibility):  # diagonal given twice
        _fit(df, "y", "x", "z", cf, resid_cov=np.eye(n), resid_var="om")
    with pytest.raises(StatsPAIError):  # unknown cluster column
        _fit(
            df,
            "y",
            "x",
            "z",
            cf,
            resid_var="om",
            resid_cov={"cluster": "nope", "rho": 0.1},
        )
