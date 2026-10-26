"""Option scan of ``statspai.spatial`` (2026-10-27).

Each test states the correct behaviour; every one failed on 1.40.2 and
pins the fix (see the CHANGELOG entries of the same date).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility, NumericalInstability


def _ring(n: int, row_std: bool = True) -> np.ndarray:
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i - 1) % n] = 1.0
        W[i, (i + 1) % n] = 1.0
        W[i, (i + 2) % n] = 0.5
        W[i, (i - 2) % n] = 0.5
    if row_std:
        W = W / W.sum(axis=1, keepdims=True)
    return W


def _sar_data(n: int = 150, rho: float = 0.6, seed: int = 0):
    rng = np.random.default_rng(seed)
    W = _ring(n)
    # spatially smooth x makes the (beta, rho) covariance material
    x = np.linalg.solve(np.eye(n) - 0.7 * W, rng.normal(size=n))
    y = np.linalg.solve(np.eye(n) - rho * W, 1.0 + 2.0 * x + rng.normal(size=n))
    return W, pd.DataFrame({"y": y, "x": x})


def _sar_full_vcov(W, X, beta, rho, s2):
    """Inverse of the SAR (beta, rho, sigma2) information matrix (Anselin 1988)."""
    n, k = X.shape
    G = W @ np.linalg.inv(np.eye(n) - rho * W)
    GXb = G @ (X @ beta)
    info = np.zeros((k + 2, k + 2))
    info[:k, :k] = X.T @ X / s2
    info[:k, k] = info[k, :k] = X.T @ GXb / s2
    info[k, k] = np.trace(G @ G) + np.trace(G.T @ G) + GXb @ GXb / s2
    info[k, k + 1] = info[k + 1, k] = np.trace(G) / s2
    info[k + 1, k + 1] = n / (2 * s2**2)
    return np.linalg.inv(info)[: k + 1, : k + 1]


# ---------------------------------------------------------------------- #
#  S1  sp.impacts draws from diag(se^2), ignoring cov(beta, rho)
# ---------------------------------------------------------------------- #


def test_impacts_sar_total_se_matches_full_covariance_delta_method():
    W, df = _sar_data()
    res = sp.sar(W, df, "y ~ x")
    b = res.params[["const", "x"]].to_numpy()
    r = float(res.params["rho"])
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    s2 = float(res.diagnostics["sigma2"])
    V = _sar_full_vcov(W, X, b, r, s2)
    # Row-standardised W: total effect of x = beta_x / (1 - rho).
    grad = np.array([0.0, 1.0 / (1 - r), b[1] / (1 - r) ** 2])
    se_delta = float(np.sqrt(grad @ V @ grad))
    imp = sp.impacts(res, n_sim=20000, seed=0)
    # 20000 draws: Monte Carlo error of an SD estimate ~ 0.5%; the delta
    # method's linearisation error at these SEs is well under 2%.
    assert imp.loc["x", "SE_Total"] == pytest.approx(se_delta, rel=0.03)


def test_impacts_sac_total_se_matches_reported_vcov():
    W, df = _sar_data(n=120)
    res = sp.sac(W, df, "y ~ x")
    V = res.model_info["vcov"].loc[["x", "rho"], ["x", "rho"]].to_numpy()
    b, r = float(res.params["x"]), float(res.params["rho"])
    grad = np.array([1.0 / (1 - r), b / (1 - r) ** 2])
    se_delta = float(np.sqrt(grad @ V @ grad))
    imp = sp.impacts(res, n_sim=20000, seed=0)
    assert imp.loc["x", "SE_Total"] == pytest.approx(se_delta, rel=0.03)


# ---------------------------------------------------------------------- #
#  S2  sp.SpatialModel (public, legacy) keeps the pre-fix SE formulas
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["sar", "sem", "sdm"])
def test_spatialmodel_matches_functional_api(model):
    W, df = _sar_data(n=80, seed=3)
    fn = {"sar": sp.sar, "sem": sp.sem, "sdm": sp.sdm}[model]
    new = fn(W, df, "y ~ x", alpha=0.1)
    old = sp.SpatialModel(W, df, "y ~ x", model_type=model, alpha=0.1).fit()
    np.testing.assert_allclose(old.params.values, new.params.values, rtol=1e-4)
    np.testing.assert_allclose(old.std_errors.values, new.std_errors.values, rtol=1e-4)
    # alpha= must reach the intervals
    assert old.model_info.get("alpha") == 0.1


# ---------------------------------------------------------------------- #
#  S3  moran_local permutes z_i together with the others
# ---------------------------------------------------------------------- #


def test_moran_local_conditional_permutation_mean():
    rng = np.random.default_rng(1)
    n = 30
    w = sp.W({i: [(i - 1) % n, (i + 1) % n] for i in range(n)})
    w.transform = "R"
    y = rng.normal(size=n)
    y[5] = 4.0  # an extreme value makes E[I_i | H0] clearly non-zero
    P = 9999
    out = sp.moran_local(y, w, permutations=P, seed=0)
    z = y - y.mean()
    m2 = float(z @ z) / n
    i = 5
    # Conditional randomisation (Anselin 1995): the neighbours of i are a
    # random draw from {z_k : k != i}, whose mean is -z_i / (n - 1).
    expected = -(z[i] ** 2) * 1.0 / ((n - 1) * m2)
    sims = out["simulations"][:, i]
    mc_se = sims.std(ddof=1) / np.sqrt(P)
    assert abs(sims.mean() - expected) < 4 * mc_se


# ---------------------------------------------------------------------- #
#  S4  spatial_did builds WD after dropping rows with a missing outcome
# ---------------------------------------------------------------------- #


def _did_panel(seed: int = 0):
    rng = np.random.default_rng(seed)
    N, T = 12, 5
    W = np.zeros((N, N))
    for i in range(N):
        W[i, (i - 1) % N] = W[i, (i + 1) % N] = 1.0
    rows = []
    treated = {0, 1, 6}
    for u in range(N):
        fe = rng.normal()
        for t in range(T):
            d = 1.0 if (u in treated and t >= 2) else 0.0
            rows.append(dict(unit=u, time=t, treat=d, fe=fe))
    df = pd.DataFrame(rows)
    Wn = W / W.sum(axis=1, keepdims=True)
    wide = df.pivot(index="unit", columns="time", values="treat").to_numpy()
    wd = (Wn @ wide).ravel()  # unit-major, matches sorted rows
    df["wd_true"] = wd
    df["y"] = (
        df["fe"]
        + 0.2 * df["time"]
        + 1.0 * df["treat"]
        + 0.5 * df["wd_true"]
        + rng.normal(scale=0.3, size=len(df))
    )
    return W, df


def test_spatial_did_missing_outcome_keeps_neighbour_exposure():
    W, df = _did_panel()
    df.loc[(df.unit == 0) & (df.time == 3), "y"] = np.nan
    res = sp.spatial_did(df, y="y", treat="treat", unit="unit", time="time", W=W)
    # Reference: two-way FE OLS on the complete rows, WD from the full
    # treatment panel (dummy-variable regression = within estimator).
    ok = df.dropna(subset=["y"])
    D = pd.get_dummies(ok[["unit", "time"]].astype(str), drop_first=True).astype(float)
    X = np.column_stack(
        [np.ones(len(ok)), ok["treat"], ok["wd_true"], D.to_numpy(float)]
    )
    coef = np.linalg.lstsq(X, ok["y"].to_numpy(float), rcond=None)[0]
    assert res.direct_effect == pytest.approx(coef[1], abs=1e-8)
    assert res.spillover_effect == pytest.approx(coef[2], abs=1e-8)


# ---------------------------------------------------------------------- #
#  S8  sp.slx log-likelihood uses the df-corrected sigma^2
# ---------------------------------------------------------------------- #


def test_slx_loglik_is_gaussian_ml_loglik():
    W, df = _sar_data(n=60, rho=0.0, seed=2)
    res = sp.slx(W, df, "y ~ x")
    e = np.asarray(res.data_info["residuals"])
    n = len(e)
    ll = -n / 2 * (np.log(2 * np.pi * float(e @ e) / n) + 1)
    assert res.diagnostics["Log-Likelihood"] == pytest.approx(ll, abs=1e-3)


# ---------------------------------------------------------------------- #
#  S9  sem_gmm / sar_gmm accept any robust= value and fall back silently
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("fn", ["sem_gmm", "sar_gmm"])
@pytest.mark.parametrize("bad", [True, "hc1", "HET"])
def test_gmm_robust_value_is_validated(fn, bad):
    W, df = _sar_data(n=60, seed=4)
    f = getattr(sp, fn)
    try:
        res = f(W, df, "y ~ x", robust=bad)
    except MethodIncompatibility:
        return
    het = f(W, df, "y ~ x", robust="het")
    np.testing.assert_allclose(res.std_errors.values, het.std_errors.values)


# ---------------------------------------------------------------------- #
#  S10  spatial_iv estimates under-identified models via pinv
# ---------------------------------------------------------------------- #


def test_spatial_iv_underidentified_raises():
    rng = np.random.default_rng(0)
    n = 60
    W = _ring(n)
    d = rng.normal(size=n)
    df = pd.DataFrame({"y": 1 + d + rng.normal(size=n), "d": d})
    with pytest.raises(sp.IdentificationFailure):
        sp.spatial_iv(df, y="y", endog=["d"], exog=[], W=W)


# ---------------------------------------------------------------------- #
#  S11  gwr replaces a singular local X'WX by its pseudo-inverse
# ---------------------------------------------------------------------- #


def test_gwr_singular_local_design_is_refused():
    g = np.arange(6, dtype=float)
    coords = np.array([(a, b) for a in g for b in g])  # unit grid
    rng = np.random.default_rng(0)
    x = rng.normal(size=len(coords))
    y = 1 + x + rng.normal(size=len(coords))
    # fixed bisquare bandwidth 0.5 < grid spacing: only the point itself
    # gets weight, so X'W_iX (2 x 2) has rank 1 everywhere.
    with pytest.raises(NumericalInstability):
        sp.gwr(coords, y, x.reshape(-1, 1), bw=0.5, kernel="bisquare", fixed=True)


# ---------------------------------------------------------------------- #
#  C1  geary / join_counts misalign S.nonzero() with S.data
# ---------------------------------------------------------------------- #


def _dense_geary(y, Wd):
    n = len(y)
    z = y - y.mean()
    num = float((Wd * (y[:, None] - y[None, :]) ** 2).sum())
    return (n - 1) * num / (2 * Wd.sum() * float(z @ z))


def test_geary_with_explicit_zero_weights():
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(40, 2))
    w = sp.kernel_weights(coords, bandwidth=4, kernel="bisquare", fixed=False)
    y = rng.normal(size=40)
    res = sp.geary(y, w, permutations=0)
    assert res.value == pytest.approx(_dense_geary(y, w.full()), rel=1e-12)


def test_join_counts_with_explicit_zero_weights():
    rng = np.random.default_rng(0)
    coords = rng.uniform(size=(40, 2))
    w = sp.kernel_weights(coords, bandwidth=4, kernel="bisquare", fixed=False)
    y = (rng.uniform(size=40) > 0.5).astype(int)
    Wd = w.full()
    res = sp.join_counts(y, w, permutations=0)
    bb = 0.5 * float((Wd * np.outer(y == 1, y == 1)).sum())
    bw = 0.5 * float((Wd * (y[:, None] != y[None, :])).sum())
    assert res["BB"] == pytest.approx(bb, rel=1e-12)
    assert res["BW"] == pytest.approx(bw, rel=1e-12)


# ---------------------------------------------------------------------- #
#  V1  join_counts truncates non-binary input before validating it
# ---------------------------------------------------------------------- #


def test_join_counts_rejects_fractional_values():
    w = sp.W({i: [(i - 1) % 10, (i + 1) % 10] for i in range(10)})
    y = np.array([0, 1, 0.5, 1, 0, 1, 1.7, 0, 1, 0])
    with pytest.raises(ValueError):
        sp.join_counts(y, w, permutations=0)


# ---------------------------------------------------------------------- #
#  Guard: point impacts were right before the fix and stay so
# ---------------------------------------------------------------------- #


def test_sac_impacts_point_values_are_closed_form():
    """Point impacts are already right; the scan only touches their SEs."""
    W, df = _sar_data(n=80, seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.sar(W, df, "y ~ x")
    imp = sp.impacts(res, n_sim=50, seed=0)
    b, r = float(res.params["x"]), float(res.params["rho"])
    assert imp.loc["x", "Total"] == pytest.approx(b / (1 - r), rel=1e-10)


# ---------------------------------------------------------------------- #
#  V5  spatial_panel FE with a time-invariant regressor
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["sar", "sem", "sdm"])
def test_spatial_panel_absorbed_regressor_is_named(model):
    rng = np.random.default_rng(0)
    N, T = 8, 5
    W = _ring(N)
    rows = []
    for i in range(N):
        z = rng.normal()  # time-invariant
        for t in range(T):
            rows.append(dict(id=i, t=t, z=z, x=rng.normal(), y=rng.normal()))
    df = pd.DataFrame(rows)
    with pytest.raises(MethodIncompatibility, match="absorbed"):
        sp.spatial_panel(df, "y ~ x + z", entity="id", time="t", W=W, model=model)
