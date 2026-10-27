"""sp.standardize_coefs, sp.ibragimov_mueller, sp.conley_sensitivity.

Correctness here is analytic or known-truth; the Stata parity of the same
functions is in tests/reference_parity/test_ivreport_stata_parity.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.fixture(scope="module")
def geo():
    rng = np.random.default_rng(7)
    n = 240
    df = pd.DataFrame(
        {
            "lat": rng.uniform(50, 54, n),
            "lon": rng.uniform(-4, 1, n),
            "z": rng.normal(size=n),
            "w": rng.normal(5, 3, n),
            "g": rng.integers(0, 6, n),
            "wt": rng.uniform(0.5, 2, n),
        }
    )
    v = rng.normal(size=n)
    df["x"] = 0.8 * df.z + 0.1 * df.w + v
    df["y"] = 1 + 0.5 * df.x - 0.2 * df.w + rng.normal(size=n) + 0.5 * v
    return df


def _z(s, w=None):
    if w is None:
        return (s - s.mean()) / s.std(ddof=1)
    w = np.asarray(w, float) * len(s) / np.sum(w)
    m = np.sum(w * s) / len(s)
    return (s - m) / np.sqrt(np.sum(w * (s - m) ** 2) / (len(s) - 1))


# ------------------------------------------------------- standardize_coefs


def test_standardized_weighted_ols_equals_weighted_ols_on_zscores(geo):
    """With [aw] the centred intercept uses weighted means and sds; the whole
    coefficient vector and covariance equal the fit on weighted z-scores."""
    fit = _quiet(sp.regress, "y ~ x + w", data=geo, weights="wt", vce="robust")
    out = _quiet(sp.standardize_coefs, fit)
    dz = geo.assign(**{c: _z(geo[c], geo.wt) for c in ("y", "x", "w")})
    ref = _quiet(sp.regress, "y ~ x + w", data=dz, weights="wt", vce="robust")
    # rounding only (all quantities O(1))
    np.testing.assert_allclose(out.params, ref.params[out.params.index], atol=1e-12)
    np.testing.assert_allclose(
        out.std_errors, ref.std_errors[out.params.index], rtol=1e-10
    )


def test_slope_t_and_p_are_the_fits_own(geo):
    fit = _quiet(sp.iv, "y ~ (x ~ z) + w", data=geo, cluster="g")
    out = _quiet(sp.standardize_coefs, fit)
    for c in ("x", "w"):
        assert out.tvalues[c] == fit.tvalues[c]
        assert out.pvalues[c] == fit.pvalues[c]
        c_k = geo[c].std() / geo.y.std()
        assert out.std_errors[c] == pytest.approx(c_k * fit.std_errors[c], rel=1e-12)
        assert out.conf_int_lower[c] == pytest.approx(
            c_k * fit.conf_int_lower[c], rel=1e-12
        )


def test_variables_subset_and_outcome_false(geo):
    fit = _quiet(sp.regress, "y ~ x + w", data=geo)
    out = _quiet(sp.standardize_coefs, fit, variables=["x"], scale_y=False)
    # x scaled by sd(x) only, w untouched, and the fit on z-scored x matches
    dz = geo.assign(x=_z(geo.x))
    ref = _quiet(sp.regress, "y ~ x + w", data=dz)
    np.testing.assert_allclose(out.params, ref.params[out.params.index], atol=1e-12)
    np.testing.assert_allclose(
        out.std_errors, ref.std_errors[out.params.index], rtol=1e-10
    )


def test_absorbed_fit_needs_data_and_aligned_rows(geo):
    fit = _quiet(sp.hdfe_ols, "y ~ x + w | g", data=geo)
    with pytest.raises(MethodIncompatibility, match="data="):
        sp.standardize_coefs(fit)
    with pytest.raises(MethodIncompatibility, match="estimation sample"):
        sp.standardize_coefs(fit, data=geo.iloc[:-5])


def test_absorbed_fit_needs_every_term_as_a_column(geo):
    fit = _quiet(sp.feols, "y ~ x + x:w | g", data=geo)
    with pytest.raises(ColumnNotFound):
        sp.standardize_coefs(fit, data=geo)


def test_constant_outcome_and_unknown_variable_are_refused(geo):
    fit = _quiet(sp.regress, "y ~ x + w", data=geo)
    with pytest.raises(MethodIncompatibility, match="not slope"):
        sp.standardize_coefs(fit, variables=["nope"])
    flat = geo.assign(y=3.0 + 1e-3 * geo.x)  # sd(y) > 0: allowed
    assert np.isfinite(
        _quiet(sp.standardize_coefs, _quiet(sp.regress, "y ~ x", data=flat)).params["x"]
    )


def test_fit_without_covariance_matrix_reports_no_intercept_se(geo):
    """sp.iv(vce='conley') replaces the SEs without storing V: slopes still
    scale exactly, the centred intercept needs covariances and is NaN with
    a warning naming the reason."""
    fit = _quiet(
        sp.iv,
        "y ~ (x ~ z) + w",
        data=geo,
        vce="conley",
        conley_lat="lat",
        conley_lon="lon",
        conley_cutoff=50,
    )
    with pytest.warns(UserWarning, match="no standard error for the centred"):
        out = sp.standardize_coefs(fit)
    assert np.isnan(out.std_errors["Intercept"])
    c_x = geo.x.std() / geo.y.std()
    assert out.std_errors["x"] == pytest.approx(c_x * fit.std_errors["x"], rel=1e-12)


# ------------------------------------------------------- ibragimov_mueller


def _slope(g):
    X = np.column_stack([np.ones(len(g)), g["x"].to_numpy()])
    return float(np.linalg.lstsq(X, g["y"].to_numpy(), rcond=None)[0][1])


def test_ibragimov_mueller_by_hand():
    """Noise-free groups: each group slope is exactly b_g, so the test is
    the one-sample t-test of (b_1, ..., b_q) computed by hand."""
    slopes = np.array([0.2, 0.9, 0.4, 1.3, 0.7])
    rows = []
    for g, b in enumerate(slopes):
        x = np.linspace(-1, 1, 7) + g
        rows.append(pd.DataFrame({"g": g, "x": x, "y": 2.0 + b * x}))
    df = pd.concat(rows, ignore_index=True)
    res = _quiet(sp.ibragimov_mueller, df, "y ~ x", cluster="g", coef="x", h0=0.3)
    q = len(slopes)
    mean, sd = slopes.mean(), slopes.std(ddof=1)
    t = (mean - 0.3) / (sd / np.sqrt(q))
    assert res.estimate == pytest.approx(mean, abs=1e-12)
    assert res.se == pytest.approx(sd / np.sqrt(q), rel=1e-10)
    assert res.tstat == pytest.approx(t, rel=1e-10)
    assert res.df == q - 1
    assert res.pvalue == pytest.approx(2 * stats.t.sf(abs(t), q - 1), rel=1e-10)
    half = stats.t.ppf(0.975, q - 1) * sd / np.sqrt(q)
    assert res.ci == pytest.approx((mean - half, mean + half), rel=1e-10)
    np.testing.assert_allclose(res.group_estimates["estimate"], slopes, atol=1e-12)
    # the callable path gives the same numbers
    alt = sp.ibragimov_mueller(df, cluster="g", estimator=_slope, h0=0.3)
    assert alt.tstat == pytest.approx(res.tstat, rel=1e-10)


def test_ibragimov_mueller_iv_groups_are_sp_iv_fits(geo):
    res = _quiet(
        sp.ibragimov_mueller,
        geo,
        "y ~ (x ~ z) + w",
        cluster="g",
        coef="x",
        estimator="iv",
    )
    for row in res.group_estimates.itertuples():
        sub = geo[geo.g == row.group]
        ref = _quiet(sp.iv, "y ~ (x ~ z) + w", data=sub).params["x"]
        assert row.estimate == pytest.approx(ref, rel=1e-12)


def test_ibragimov_mueller_size_with_few_heterogeneous_clusters():
    """Known truth: 8 clusters whose noise sd ranges 0.5-7.5, a common shock
    within each cluster, heteroskedastic errors and cluster-specific designs.
    The t(q-1) test on the group slopes is conservative at 5% (Ibragimov and
    Mueller 2010). 1,500 seeded replications: the Monte Carlo SE of a 5%
    rejection rate is 0.0056, so a valid test lands below 0.05 + 3 MC SE =
    0.067 except with probability ~0.001; this design gives ~0.03."""
    rng = np.random.default_rng(20261028)
    q, n_g, reps = 8, 40, 1500
    g = np.repeat(np.arange(q), n_g)
    sd = (0.5 + np.arange(q))[g]
    rejections = 0
    for _ in range(reps):
        x = rng.normal(size=q * n_g) + rng.normal(size=q)[g]
        e = sd * (
            0.7 * rng.normal(size=q)[g] + rng.normal(size=q * n_g) * (1 + np.abs(x))
        )
        df = pd.DataFrame({"g": g, "x": x, "y": 1 + 0.5 * x + e})
        res = sp.ibragimov_mueller(df, cluster="g", estimator=_slope, h0=0.5)
        rejections += res.pvalue < 0.05
    mc_se = np.sqrt(0.05 * 0.95 / reps)
    assert rejections / reps <= 0.05 + 3 * mc_se


def test_ibragimov_mueller_edges():
    df = pd.DataFrame({"g": [0] * 5 + [1] * 5, "x": np.r_[np.arange(5), np.arange(5)]})
    df["y"] = df.x * 0.5
    with pytest.raises(DataInsufficient, match="identical"):
        sp.ibragimov_mueller(df, "y ~ x", cluster="g", coef="x")
    with pytest.raises(DataInsufficient, match="at least two"):
        sp.ibragimov_mueller(df[df.g == 0], "y ~ x", cluster="g", coef="x")
    with pytest.raises(DataInsufficient, match="min_group_size"):
        sp.ibragimov_mueller(df, "y ~ x", cluster="g", coef="x", min_group_size=6)
    with pytest.raises(MethodIncompatibility, match="no 'g'"):
        sp.ibragimov_mueller(
            df.assign(g=[0, 1, None] + [1] * 7), "y ~ x", cluster="g", coef="x"
        )
    with pytest.raises(MethodIncompatibility, match="formula= is not used"):
        sp.ibragimov_mueller(df, "y ~ x", cluster="g", estimator=_slope)
    with pytest.raises(ColumnNotFound):
        sp.ibragimov_mueller(df, "y ~ x", cluster="nope", coef="x")
    bad = df.assign(x=np.r_[np.ones(5), np.arange(5)])  # no variation in group 0
    with pytest.raises(MethodIncompatibility, match="group 0"):
        _quiet(sp.ibragimov_mueller, bad, "y ~ x", cluster="g", coef="x")
    noisy = df.assign(y=df.y + np.r_[0.1, -0.1, 0, 0.2, 0, 0, 0.3, 0, 0, -0.1])
    with pytest.warns(UserWarning, match="0.083"):
        sp.ibragimov_mueller(noisy, "y ~ x", cluster="g", coef="x", alpha=0.10)


# ------------------------------------------------------- conley_sensitivity


def test_conley_sensitivity_rows_are_sp_conley(geo):
    fit = _quiet(sp.iv, "y ~ (x ~ z) + w", data=geo)
    tab = _quiet(
        sp.conley_sensitivity, fit, geo, "lat", "lon", [20, 60, 120], kernel="bartlett"
    )
    for c in (20, 60, 120):
        ref = _quiet(
            sp.conley,
            fit,
            data=geo,
            lat="lat",
            lon="lon",
            dist_cutoff=c,
            kernel="bartlett",
            distance="planar",
        )
        sub = tab.table[tab.table.cutoff_km == c].set_index("term")
        for t in ("x", "w"):
            assert sub.loc[t, "se"] == ref.std_errors[t]
            half = stats.norm.ppf(0.975) * ref.std_errors[t]
            assert sub.loc[t, "ci_upper"] == pytest.approx(
                fit.params[t] + half, rel=1e-12
            )
    assert tab.wide("se").shape == (3, 2)


def test_conley_sensitivity_tiny_cutoff_is_hc0(geo):
    """Only the own pair lies within 1 metre (coordinates are distinct), so
    the uniform-kernel Conley variance is exactly the HC0 sandwich."""
    tab = _quiet(
        sp.conley_sensitivity, "y ~ x + w", geo, "lat", "lon", [0.001], inference="t"
    )
    hc0 = _quiet(sp.regress, "y ~ x + w", data=geo, robust="hc0")
    se = tab.table.set_index("term")["se"]
    for t in ("x", "w"):
        assert se[t] == pytest.approx(hc0.std_errors[t], rel=1e-10)
    # t reference: t(n - k), as sp.conley
    row = tab.table.set_index("term").loc["x"]
    crit = stats.t.ppf(0.975, len(geo) - 3)
    assert row["ci_lower"] == pytest.approx(
        row["estimate"] - crit * row["se"], rel=1e-12
    )


def test_conley_sensitivity_formula_equals_result(geo):
    a = _quiet(sp.conley_sensitivity, "y ~ x + w", geo, "lat", "lon", [50])
    b = _quiet(
        sp.conley_sensitivity,
        _quiet(sp.regress, "y ~ x + w", data=geo),
        geo,
        "lat",
        "lon",
        [50],
    )
    pd.testing.assert_frame_equal(a.table, b.table)


def test_conley_sensitivity_plot(geo):
    import matplotlib

    matplotlib.use("Agg")
    tab = _quiet(sp.conley_sensitivity, "y ~ x + w", geo, "lat", "lon", [25, 50])
    fig = tab.plot()
    assert len(fig.axes) == 2


def test_conley_sensitivity_edges(geo):
    fit = _quiet(sp.regress, "y ~ x + w", data=geo)
    with pytest.raises(MethodIncompatibility, match="positive"):
        sp.conley_sensitivity(fit, geo, "lat", "lon", [0, 10])
    with pytest.raises(MethodIncompatibility, match="duplicates"):
        sp.conley_sensitivity(fit, geo, "lat", "lon", [10, 10])
    with pytest.raises(MethodIncompatibility, match="inference"):
        sp.conley_sensitivity(fit, geo, "lat", "lon", [10], inference="z")
    with pytest.raises(MethodIncompatibility, match="not coefficients"):
        sp.conley_sensitivity(fit, geo, "lat", "lon", [10], terms=["nope"])
    with pytest.raises(MethodIncompatibility):
        sp.conley_sensitivity(fit, geo.iloc[:-1], "lat", "lon", [10])


def test_hdfe_ols_rejects_unknown_conley_kernel(geo):
    with pytest.raises(MethodIncompatibility, match="uniform' or 'bartlett"):
        sp.hdfe_ols(
            "y ~ x | g",
            data=geo,
            vce="conley",
            conley_lat="lat",
            conley_lon="lon",
            conley_cutoff=50,
            conley_kernel="epanechnikov",
        )
