"""sp.recentered_iv: Conley standard errors and the efficient instrument.

The Conley variance is ``sp.conley`` applied to the same regression, so
the tests pin it to that function, to a dense hand computation, and to
the acreg-pinned ``sp.ivreg(vce='conley')``. The efficient instrument
(Borusyak and Hull, NBER WP 27845, eq. 9 and Lemma 3) is pinned to
a dense ``(Omega + psi psi')^{-1}`` solve and checked for the efficiency
it promises on a known-truth design.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _design(seed=7, n=240, n_lines=80, S=60):
    rng = np.random.default_rng(seed)
    cen = rng.uniform(0, 1, n)
    W = (rng.uniform(size=(n, n_lines)) < (0.05 + 0.3 * cen)[:, None]).astype(float)
    g = rng.binomial(1, 0.3, n_lines).astype(float)
    z = W @ g
    cf = np.column_stack([W @ rng.permutation(g) for _ in range(S)])
    eps = 2.0 * cen + rng.normal(scale=0.5, size=n)
    x = 0.7 * z + 0.8 * eps + rng.normal(size=n)
    y = 0.5 * x + eps
    df = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "cen": cen,
            "f": rng.integers(0, 8, n),
            "lat": rng.uniform(30, 40, n),
            "lon": rng.uniform(100, 115, n),
        }
    )
    # Two singletons for the absorbed path.
    df.loc[[0, 1], "f"] = [100, 101]
    return df, cf


def _fit(*args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.recentered_iv(*args, **kw)


CONLEY = dict(conley_lat="lat", conley_lon="lon", conley_cutoff=400.0)


# --------------------------------------------------------------------
# Defaults unchanged
# --------------------------------------------------------------------


def test_defaults_unchanged_on_fixed_config():
    # Values recorded on main (e82d4561d) before the new arguments were
    # added; bit-identical locally on eight configurations, rtol 1e-12 here
    # for BLAS differences across platforms.
    df, cf = _design()
    df.loc[[0, 1], "f"] = [0, 1]
    r = _fit(df, "y", "x", "z", cf)
    assert r.estimate == pytest.approx(0.5025891936155016, rel=1e-12)
    assert r.se == pytest.approx(0.03928363461684148, rel=1e-12)
    assert r.diagnostics["ri_pvalue"] == pytest.approx(1 / 61, rel=1e-12)
    c = _fit(df, "y", "x", "z", cf, method="control")
    assert c.estimate == pytest.approx(0.4788104029316795, rel=1e-12)
    assert c.se == pytest.approx(0.030891608022699413, rel=1e-12)
    assert "conley" not in r.diagnostics
    assert "efficient_instrument" not in r.diagnostics


# --------------------------------------------------------------------
# Conley
# --------------------------------------------------------------------


def _recentered_frame(df, cf):
    out = df.copy()
    out["mu"] = cf.mean(axis=1)
    out["zt"] = out["z"] - out["mu"]
    return out


@pytest.mark.parametrize("kernel", ["uniform", "bartlett"])
def test_conley_equals_sp_conley_on_recentered_iv(kernel):
    df, cf = _design()
    r = _fit(df, "y", "x", "z", cf, vce="conley", conley_kernel=kernel, **CONLEY)
    w = _recentered_frame(df, cf)
    base = sp.iv("y ~ (x ~ zt)", data=w, robust="hc1")
    ref = sp.conley(base, w, "lat", "lon", 400.0, kernel=kernel)
    assert r.estimate == pytest.approx(float(ref.params["x"]), rel=1e-12)
    assert r.se == pytest.approx(float(ref.std_errors["x"]), rel=1e-10)
    lo, hi = ref.conf_int(alpha=0.05).loc["x"]
    assert r.ci == pytest.approx((lo, hi), rel=1e-10)
    # The unadjusted fit carries the same variance.
    naive = sp.conley(
        sp.iv("y ~ (x ~ z)", data=w, robust="hc1"),
        w,
        "lat",
        "lon",
        400.0,
        kernel=kernel,
    )
    assert r.diagnostics["unadjusted_se"] == pytest.approx(
        float(naive.std_errors["x"]), rel=1e-10
    )
    assert r.diagnostics["conley"] == {
        "cutoff_km": 400.0,
        "kernel": kernel,
        "distance": "haversine",
    }


def test_conley_control_method_and_ols_case():
    df, cf = _design()
    w = _recentered_frame(df, cf)
    r = _fit(df, "y", "x", "z", cf, method="control", vce="conley", **CONLEY)
    ref = sp.conley(sp.iv("y ~ (x ~ z) + mu", data=w), w, "lat", "lon", 400.0)
    assert r.se == pytest.approx(float(ref.std_errors["x"]), rel=1e-10)
    # treat == exposure, method='control' is OLS of y on z and mu.
    o = _fit(df, "y", "z", "z", cf, method="control", vce="conley", **CONLEY)
    ref_o = sp.conley(sp.regress("y ~ z + mu", data=w), w, "lat", "lon", 400.0)
    assert o.estimate == pytest.approx(float(ref_o.params["z"]), rel=1e-12)
    assert o.se == pytest.approx(float(ref_o.std_errors["z"]), rel=1e-10)


def test_conley_absorb_uses_rows_kept_after_singletons():
    df, cf = _design()
    r = _fit(df, "y", "x", "z", cf, absorb="f", vce="conley", **CONLEY)
    assert r.diagnostics["n_dropped_singletons"] == 2
    w = _recentered_frame(df, cf)
    base = sp.iv("y ~ (x ~ zt)", data=w, absorb="f")
    kept = w.iloc[2:].reset_index(drop=True)
    ref = sp.conley(base, kept, "lat", "lon", 400.0)
    assert r.se == pytest.approx(float(ref.std_errors["x"]), rel=1e-10)
    assert r.n_obs == len(kept)


def _haversine(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    h = (
        np.sin(dla / 2) ** 2
        + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


@pytest.mark.parametrize("kernel", ["uniform", "bartlett"])
def test_conley_matches_dense_hand_computation(kernel):
    # V = (Xh'X)^-1 Xh' diag(e) K diag(e) Xh (X'Xh)^-1, K from great-circle
    # distances, written out with an n x n kernel matrix.
    df, cf = _design()
    r = _fit(df, "y", "x", "z", cf, vce="conley", conley_kernel=kernel, **CONLEY)
    w = _recentered_frame(df, cf)
    n = len(w)
    X = np.column_stack([np.ones(n), w["x"]])
    Z = np.column_stack([np.ones(n), w["zt"]])
    Xh = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)
    b = np.linalg.solve(Xh.T @ X, Xh.T @ w["y"].to_numpy())
    e = w["y"].to_numpy() - X @ b
    d = _haversine(w["lat"].to_numpy(), w["lon"].to_numpy())
    K = (d <= 400.0).astype(float)
    if kernel == "bartlett":
        K = K * (1 - d / 400.0)
    s = Xh * e[:, None]
    bread = np.linalg.inv(Xh.T @ X)
    V = bread @ (s.T @ K @ s) @ bread.T
    assert r.estimate == pytest.approx(b[1], rel=1e-10)
    assert r.se == pytest.approx(np.sqrt(V[1, 1]), rel=1e-9)


def test_conley_planar_uniform_equals_acreg_pinned_ivreg():
    # sp.ivreg(vce='conley') is pinned to Stata acreg (planar, uniform).
    df, cf = _design()
    r = _fit(df, "y", "x", "z", cf, vce="conley", conley_distance="planar", **CONLEY)
    w = _recentered_frame(df, cf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sp.ivreg(
            "y ~ (x ~ zt)",
            data=w,
            vce="conley",
            conley_lat="lat",
            conley_lon="lon",
            conley_cutoff=400.0,
        )
    assert r.se == pytest.approx(float(ref.std_errors["x"]), rel=1e-10)


@pytest.mark.parametrize(
    "kw",
    [
        dict(vce="conley", conley_lat="lat", conley_lon="lon"),
        dict(vce="conley", cluster="f", **CONLEY),
        dict(vce="conley", weights="cen", **CONLEY),
        dict(vce="conley", conley_kernel="epanechnikov", **CONLEY),
        dict(vce="conley", conley_lat="lat", conley_lon="lon", conley_cutoff=0.0),
        dict(conley_cutoff=100.0),
    ],
)
def test_conley_argument_errors(kw):
    df, cf = _design()
    with pytest.raises(MethodIncompatibility):
        _fit(df, "y", "x", "z", cf, **kw)


# --------------------------------------------------------------------
# Efficient instrument
# --------------------------------------------------------------------


def _with_variance_model(df, cf):
    rng = np.random.default_rng(11)
    out = df.copy()
    out["om"] = 0.2 + rng.uniform(0, 2, len(df)) * (1 + out["cen"])
    out["psi"] = 2.0 * out["cen"] - 1.0
    return out


def test_efficient_instrument_matches_dense_solve():
    df, cf = _design()
    w = _with_variance_model(_recentered_frame(df, cf), cf)
    r = _fit(
        df.assign(om=w["om"], psi=w["psi"]),
        "y",
        "x",
        "z",
        cf,
        resid_var="om",
        resid_mean="psi",
    )
    E = np.diag(w["om"].to_numpy()) + np.outer(w["psi"], w["psi"])
    w["zs"] = np.linalg.solve(E, w["zt"].to_numpy())
    ref = sp.iv("y ~ (x ~ zs)", data=w, robust="hc1")
    assert r.estimate == pytest.approx(float(ref.params["x"]), rel=1e-10)
    assert r.se == pytest.approx(float(ref.std_errors["x"]), rel=1e-10)
    info = r.diagnostics["efficient_instrument"]
    q = float(np.sum(w["psi"] ** 2 / w["om"]))
    assert info["nu"] == pytest.approx(q / (1 + q), rel=1e-12)
    # The RI statistic uses the same instrument for every draw, so its
    # Hodges-Lehmann root is still the point estimate.
    assert r.diagnostics["ri_hodges_lehmann"] == pytest.approx(r.estimate, rel=1e-10)


def test_constant_resid_var_is_plain_recentering():
    # Omega = c I only rescales z - mu; just-identified IV is invariant.
    df, cf = _design()
    plain = _fit(df, "y", "x", "z", cf)
    eff = _fit(df.assign(om=3.7), "y", "x", "z", cf, resid_var="om")
    assert eff.estimate == pytest.approx(plain.estimate, rel=1e-12)
    assert eff.se == pytest.approx(plain.se, rel=1e-10)
    assert eff.diagnostics["ri_pvalue"] == plain.diagnostics["ri_pvalue"]


def test_efficient_instrument_with_absorb_keeps_hodges_lehmann():
    df, cf = _design()
    w = _with_variance_model(df, cf)
    r = _fit(w, "y", "x", "z", cf, absorb="f", resid_var="om", resid_mean="psi")
    assert r.diagnostics["ri_hodges_lehmann"] == pytest.approx(r.estimate, rel=1e-9)


def test_efficient_instrument_gains_on_known_truth_design():
    # z_i = e_i g_i with fair coins, so mu_i = e_i / 2 exactly. The residual
    # has mean psi_i = 1.5 e_i and sd 0.3 + 1.5 e_i^2 given the exposure;
    # beta = 1. All three instruments are valid (unbiased); weighting by
    # Omega^-1 and then partialling psi cut the variance. 150 replications;
    # the observed ratios are about 0.75 and 0.56 (300 replications), the
    # bounds leave room for Monte Carlo noise.
    rng = np.random.default_rng(3)
    n, reps = 400, 150
    est = {"plain": [], "var": [], "both": []}
    for _ in range(reps):
        e = rng.uniform(0.2, 2, n)
        z = e * rng.binomial(1, 0.5, n)
        sig = 0.3 + 1.5 * e**2
        psi = 1.5 * e
        y = 1.0 * z + psi + sig * rng.normal(size=n)
        df = pd.DataFrame({"y": y, "z": z, "mu": 0.5 * e, "om": sig**2, "psi": psi})
        cf = np.column_stack([e * rng.binomial(1, 0.5, n) for _ in range(2)])
        kw = dict(expected="mu", ri=False)
        est["plain"].append(_fit(df, "y", "z", "z", cf, **kw).estimate)
        est["var"].append(_fit(df, "y", "z", "z", cf, resid_var="om", **kw).estimate)
        est["both"].append(
            _fit(df, "y", "z", "z", cf, resid_var="om", resid_mean="psi", **kw).estimate
        )
    est = {k: np.asarray(v) for k, v in est.items()}
    for v in est.values():
        assert abs(v.mean() - 1.0) < 4 * v.std(ddof=1) / np.sqrt(reps)
    var = {k: v.var(ddof=1) for k, v in est.items()}
    assert var["var"] / var["plain"] < 0.9
    assert var["both"] / var["plain"] < 0.75
    assert var["both"] < var["var"]


@pytest.mark.parametrize(
    "kw",
    [
        dict(resid_mean="cen"),
        dict(resid_var="om", method="control"),
        dict(resid_var="neg"),
        dict(resid_var=np.ones(5)),
    ],
)
def test_efficient_argument_errors(kw):
    df, cf = _design()
    df = df.assign(om=1.0, neg=-1.0)
    with pytest.raises(MethodIncompatibility):
        _fit(df, "y", "x", "z", cf, **kw)
