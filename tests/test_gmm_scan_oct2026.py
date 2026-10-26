"""Option scan (Oct 2026) of ``statspai.gmm`` (general GMM, dynamic panel
GMM, LSDVC).

Each test states the correct behaviour for an option combination that
returned a wrong number, crashed or went unvalidated before the scan.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

# ---------------------------------------------------------------------------
# general GMM
# ---------------------------------------------------------------------------


def _iv_frame(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z1, z2, u = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    x1 = 0.7 * z1 + 0.5 * z2 + u + rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "z1": z1, "z2": z2})


def _moments(theta, data):
    X = np.column_stack([np.ones(len(data)), data["x1"].to_numpy()])
    Z = np.column_stack([np.ones(len(data)), data[["z1", "z2"]].to_numpy()])
    resid = data["y"].to_numpy() - X @ theta
    return resid[:, None] * Z


def test_hac_default_bandwidth_is_the_library_lag_rule():
    """vcov='hac' without a bandwidth weighted no lag (== 'mds')."""
    df = _iv_frame()
    n = len(df)
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    assert lags >= 1
    auto = sp.gmm(_moments, np.zeros(2), df, vcov="hac")
    explicit = sp.gmm(_moments, np.zeros(2), df, vcov="hac", hac_bandwidth=lags + 1)
    np.testing.assert_array_equal(
        auto.std_errors.to_numpy(), explicit.std_errors.to_numpy()
    )
    assert auto.model_info["hac_lags"] == lags
    # bandwidth 1 is the no-lag estimator, identical to 'mds'
    one = sp.gmm(_moments, np.zeros(2), df, vcov="hac", hac_bandwidth=1)
    mds = sp.gmm(_moments, np.zeros(2), df, vcov="mds")
    np.testing.assert_array_equal(one.std_errors.to_numpy(), mds.std_errors.to_numpy())


def test_gmm_small_false_is_normal_inference_and_default_is_t():
    df = _iv_frame()
    n, k = len(df), 2
    t_fit = sp.gmm(_moments, np.zeros(2), df, param_names=["_cons", "x1"])
    z_fit = sp.gmm(_moments, np.zeros(2), df, param_names=["_cons", "x1"], small=False)
    # same estimates and variance
    np.testing.assert_array_equal(t_fit.params.to_numpy(), z_fit.params.to_numpy())
    np.testing.assert_array_equal(
        t_fit.std_errors.to_numpy(), z_fit.std_errors.to_numpy()
    )
    tz = (t_fit.params / t_fit.std_errors).to_numpy()
    np.testing.assert_allclose(
        np.asarray(t_fit.pvalues, float), 2 * stats.t.sf(np.abs(tz), n - k), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(z_fit.pvalues, float), 2 * stats.norm.sf(np.abs(tz)), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(z_fit.conf_int_lower, float),
        z_fit.params.to_numpy() - stats.norm.ppf(0.975) * z_fit.std_errors.to_numpy(),
        rtol=1e-12,
    )
    assert t_fit.inference_rule()["distribution"] == "t"
    assert t_fit.inference_rule()["df"] == n - k
    assert z_fit.inference_rule()["distribution"] == "z"
    with pytest.raises(MethodIncompatibility):
        sp.gmm(_moments, np.zeros(2), df, small="no")


def test_gmm_cluster_labels_with_missing_values_are_refused():
    df = _iv_frame()
    cl = np.repeat(np.arange(40), 10).astype(float)
    cl[7] = np.nan
    with pytest.raises(MethodIncompatibility):
        sp.gmm(_moments, np.zeros(2), df, vcov="cluster", cluster=cl)


# ---------------------------------------------------------------------------
# dynamic panel
# ---------------------------------------------------------------------------


def _dyn_panel(seed: int = 0, n_units: int = 120, n_t: int = 8, beta_x: float = 0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        eta = rng.normal()
        y_prev = eta + rng.normal()
        for t in range(n_t):
            x = rng.normal()
            y = 0.5 * y_prev + beta_x * x + eta + rng.normal()
            rows.append({"id": i, "t": t, "y": y, "x": x, "ind": i % 20})
            y_prev = y
    return pd.DataFrame(rows)


@pytest.mark.parametrize("twostep", [False, True])
def test_xtabond_string_cluster_equals_integer_cluster(twostep):
    df = _dyn_panel()
    df["ind_s"] = "industry_" + df["ind"].astype(str)
    kw = dict(
        y="y",
        x=["x"],
        id="id",
        time="t",
        collapse=True,
        gmm_lags=(2, 4),
        twostep=twostep,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.xtabond(df, cluster="ind", **kw)
        b = sp.xtabond(df, cluster="ind_s", **kw)
    # Codes of string labels sort differently from the integers, so the
    # cluster sums run in another order: equal to rounding.
    np.testing.assert_allclose(
        a.detail["se"].to_numpy(), b.detail["se"].to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(
        a.detail["coefficient"].to_numpy(),
        b.detail["coefficient"].to_numpy(),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# LSDVC bootstrap
# ---------------------------------------------------------------------------


def test_xtlsdvc_bootstrap_world_has_the_fitted_coefficients(monkeypatch):
    """The parametric bootstrap must draw from the fitted process.

    A within regression on a regenerated frame should recover the fitted
    slope on x (large N); a world with x'b left out gives ~0.
    """
    import importlib

    # ``statspai.gmm`` the attribute is the function sp.gmm, so import by name
    lsdvc_mod = importlib.import_module("statspai.gmm.lsdvc")

    df = _dyn_panel(n_units=300, n_t=10, beta_x=1.0)
    frames = []
    orig = lsdvc_mod.xtlsdvc

    def spy(data, *args, **kw):
        if kw.get("bootstrap", 0) == 0:
            frames.append(data.copy())
        return orig(data, *args, **kw)

    monkeypatch.setattr(lsdvc_mod, "xtlsdvc", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = orig(df, y="y", x=["x"], id="id", time="t", bootstrap=2, seed=1)
    assert frames, "the bootstrap made no replication"
    b_x = float(fit.detail["coefficient"].iloc[1])
    boot = frames[0].sort_values(["id", "t"]).copy()
    boot["ly"] = boot.groupby("id")["y"].shift(1)
    boot = boot.dropna(subset=["ly"])
    cols = ["y", "ly", "x"]
    dm = boot[cols] - boot.groupby("id")[cols].transform("mean")
    coef, *_ = np.linalg.lstsq(
        dm[["ly", "x"]].to_numpy(), dm["y"].to_numpy(), rcond=None
    )
    assert abs(coef[1] - b_x) < 0.1


@pytest.mark.parametrize("twostep", [False, True])
def test_xtabond_cluster_with_robust_false_is_refused(twostep):
    """One-step returned sigma2 (Z'HZ)^-1 and ignored cluster=; xtabond2's
    help says cluster() implies robust."""
    df = _dyn_panel()
    with pytest.raises(MethodIncompatibility):
        sp.xtabond(
            df,
            y="y",
            x=["x"],
            id="id",
            time="t",
            collapse=True,
            gmm_lags=(2, 4),
            cluster="ind",
            robust=False,
            twostep=twostep,
        )


def test_onestep_gmm_reports_no_hansen_j():
    """Stata 18: `estat overid` "not available after one-step estimation"
    (gmm ..., onestep winitial(identity)); J at a non-efficient weight is
    not chi2(q - k)."""
    df = _iv_frame()
    one = sp.gmm(_moments, np.zeros(2), df, method="onestep")
    assert one.diagnostics["J_stat"] is None
    assert one.diagnostics["J_p"] is None
    assert "one-step" in one.diagnostics["J_note"]
    two = sp.gmm(_moments, np.zeros(2), df, method="twostep")
    assert np.isfinite(two.diagnostics["J_stat"])
    assert two.diagnostics["J_note"] is None
