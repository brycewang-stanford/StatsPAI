"""Fixes from the review of Ian Gow's StatsPAI edition of *Empirical
Research in Accounting* (2026-11-21).

Each test states a known truth: an identity between two routes to the same
number, a reference implementation's value, or the caller's own input.
"""

import datetime as dt
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

pl = pytest.importorskip("polars")


# ---------------------------------------------------------------- sp.winsor


def _polars_frame(n=400, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_t(3, n)
    x[::37] = np.nan
    return pl.DataFrame(
        {
            "fyear": rng.integers(2000, 2004, n),
            "x": x,
            "d": [dt.date(2020, 1, 1)] * n,
            "s": ["a"] * n,
            "k": pl.Series(range(n), dtype=pl.Int32),
        }
    ).with_columns(pl.col("x").fill_nan(None))


def test_winsor_returns_polars_with_untouched_columns_as_passed():
    p = _polars_frame()
    out = sp.winsor(p, vars=["x"], replace=True)
    assert isinstance(out, pl.DataFrame)
    assert out.schema == p.schema  # Date stays Date, Int32 stays Int32
    for name in ("fyear", "d", "s", "k"):
        assert out[name].equals(p[name])
    assert out["x"].null_count() == p["x"].null_count()


def test_winsor_polars_values_equal_the_pandas_route():
    p = _polars_frame()
    out = sp.winsor(p, vars=["x"], by="fyear", replace=True)
    ref = sp.winsor(p.to_pandas(), vars=["x"], by="fyear", replace=True)
    np.testing.assert_array_equal(out["x"].to_numpy(), ref["x"].to_numpy(dtype=float))


def test_winsor_new_column_and_lazyframe_and_pipe():
    p = _polars_frame()
    added = p.pipe(sp.winsor, vars=["x"])
    assert added.columns == p.columns + ["x_w"]
    lazy = sp.winsor(p.lazy(), vars=["x"], replace=True)
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(sp.winsor(p, vars=["x"], replace=True))


def test_winsor_trim_missing_values_are_null_in_polars():
    p = _polars_frame()
    out = sp.winsor(p, vars=["x"], trim=True, replace=True)
    assert out["x"].null_count() > p["x"].null_count()
    assert out["x"].is_nan().sum() == 0


def test_winsor_pandas_input_still_returns_pandas():
    df = _polars_frame().to_pandas()
    assert isinstance(sp.winsor(df, vars=["x"]), pd.DataFrame)


def test_winsor_warning_points_at_the_caller():
    p = _polars_frame().with_columns(
        fyear=pl.when(pl.col("k") < 3).then(None).otherwise(pl.col("fyear"))
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sp.winsor(p, vars=["x"], by="fyear")
    assert rec and rec[0].filename == __file__


# ------------------------------------------ coef_only and the spec counter


def _groups(n_groups=30):
    rng = np.random.default_rng(0)
    return [
        pd.DataFrame({"y": rng.normal(size=30 + i), "x": rng.normal(size=30 + i)})
        for i in range(n_groups)
    ]


def test_coef_only_by_group_loop_is_not_a_specification_search():
    sp.reset_spec_counter()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for g in _groups():
            fit = sp.regress("y ~ x", data=g, coef_only=True)
            sp.feols("y ~ x", data=g, coef_only=True)
    assert not [w for w in rec if "distinct specifications" in str(w.message)]
    assert "multiplicity" not in fit.model_info


def test_fits_with_inference_are_still_counted():
    sp.reset_spec_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in _groups(6):
            fit = sp.regress("y ~ x", data=g)
    assert fit.model_info["multiplicity"]["specifications_tried"] == 6
    sp.reset_spec_counter()


# --------------------------------------------------- normal reference


def _reg_data(n=60):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=n), "g": rng.integers(0, 8, n)})
    df["y"] = 0.3 * df["x"] + rng.normal(size=n)
    return df


@pytest.mark.parametrize("kw", [{"robust": "hc1"}, {"cluster": "g"}, {}])
def test_normal_reference_is_the_normal_tail_of_the_same_statistic(kw):
    fit = sp.regress("y ~ x", data=_reg_data(), **kw)
    z = fit.normal_reference()
    stat = fit.params / fit.std_errors
    np.testing.assert_allclose(z.pvalues, 2 * stats.norm.sf(np.abs(stat)), rtol=1e-12)
    half = stats.norm.ppf(0.975) * fit.std_errors
    np.testing.assert_allclose(z.conf_int().iloc[:, 0], fit.params - half, rtol=1e-12)
    pd.testing.assert_series_equal(z.params, fit.params)
    pd.testing.assert_series_equal(z.std_errors, fit.std_errors)
    assert z.inference_rule()["distribution"] == "z"
    assert sp.test(z, "x = 0")["pvalue"] == pytest.approx(z.pvalues["x"], rel=1e-10)


def test_normal_reference_leaves_the_original_fit_alone():
    fit = sp.regress("y ~ x", data=_reg_data(), cluster="g")
    before = fit.pvalues.copy()
    fit.normal_reference()
    pd.testing.assert_series_equal(fit.pvalues, before)
    assert fit.inference_rule()["distribution"] == "t"


def test_normal_reference_matches_statsmodels_robust_default():
    smf = pytest.importorskip("statsmodels.formula.api")
    df = _reg_data()
    z = sp.regress("y ~ x", data=df, robust="hc1").normal_reference()
    ref = smf.ols("y ~ x", df).fit(cov_type="HC1")  # use_t=False
    np.testing.assert_allclose(z.pvalues, ref.pvalues, rtol=1e-10)


def test_normal_reference_refuses_a_fit_without_variance():
    fit = sp.regress("y ~ x", data=_reg_data(), coef_only=True)
    with pytest.raises(MethodIncompatibility):
        fit.normal_reference()


# ----------------------------------------- over-identification row label


def _iv_data(n=400):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(n, 2))
    u = rng.normal(size=n)
    x = z @ [1.0, 1.0] + u + rng.normal(size=n)
    return pd.DataFrame({"y": x + u, "X": x, "z1": z[:, 0], "z2": z[:, 1]})


def _pvalue_rows(*fits):
    text = str(sp.regtable(*fits, diagnostics=True))
    return [line.split("  ")[0] for line in text.splitlines() if "p-value" in line]


def test_regtable_names_the_overid_row_after_the_test_in_it():
    df = _iv_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iid = sp.ivreg("y ~ (X ~ z1 + z2)", data=df, robust="nonrobust")
        rob = sp.ivreg("y ~ (X ~ z1 + z2)", data=df, robust="hc1")
        assert "Sargan p-value" in iid.diagnostics
        assert "Hansen J p-value" not in iid.diagnostics
        assert _pvalue_rows(iid) == ["Sargan p-value"]
        assert _pvalue_rows(rob) == ["Hansen J p-value"]
        assert _pvalue_rows(iid, rob) == ["Overid. test p-value"]


# -------------------------------------------- weak-IV sets, frame first


def test_weak_iv_sets_accept_the_frame_first():
    df = _iv_data()
    kw = dict(y="y", endog="X", instruments=["z1", "z2"])
    for fn in (sp.anderson_rubin_ci, sp.conditional_lr_ci):
        first = fn(df, **kw).as_intervals()
        assert first == fn("y", "X", ["z1", "z2"], data=df).as_intervals()
        assert first == fn(pl.from_pandas(df), **kw).as_intervals()
    with pytest.raises(TypeError, match="both"):
        sp.anderson_rubin_ci(df, data=df, **kw)


# --------------------------------------------- fuzzy RD first-stage F


@pytest.mark.parametrize(
    "kw",
    [{}, {"vce": "hc1"}, {"cluster": "g"}, {"weights": "w"}, {"kernel": "uniform"}],
)
def test_fuzzy_rd_first_stage_F_is_the_sharp_rd_of_the_treatment(kw):
    rng = np.random.default_rng(3)
    n, c = 866, 75
    x = rng.uniform(0, 150, n)
    d = (rng.random(n) < np.where(x >= c, 0.55, 0.25)).astype(float)
    df = pd.DataFrame(
        {
            "y": 1 + 0.5 * d + 0.01 * x + rng.normal(size=n),
            "d": d,
            "x": x,
            "g": rng.integers(0, 60, n),
            "w": rng.uniform(0.5, 2, n),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fuzzy = sp.rdrobust(df, y="y", x="x", c=c, fuzzy="d", **kw)
        h, b = fuzzy.model_info["bandwidth_h"], fuzzy.model_info["bandwidth_b"]
        first = sp.rdrobust(df, y="d", x="x", c=c, h=h, b=b, **kw)
    conv = first.model_info["conventional"]
    assert fuzzy.model_info["first_stage_F"] == pytest.approx(
        (conv["estimate"] / conv["se"]) ** 2, rel=1e-10
    )


# -------------------------------- event study on one security


def _returns(ids, T=1500, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=T)
    mkt = rng.normal(0, 0.01, T)
    frames = [
        pd.DataFrame(
            {
                "id": i,
                "date": dates,
                "mkt": mkt,
                "ret": 0.9 * mkt + rng.normal(0, 0.02, T),
            }
        )
        for i in ids
    ]
    return pd.concat(frames, ignore_index=True), dates


def test_one_security_event_study_has_no_cross_correlation_adjustment():
    data, dates = _returns([1])
    events = pd.DataFrame({"id": 1, "event_date": dates[[400, 430, 460, 700, 900]]})
    with pytest.warns(UserWarning, match="one security"):
        res = sp.abnormal_returns(data, events, model="market", market="mkt")
    assert res.mean_correlation == 0.0
    tests = res.tests
    assert tests.loc["adj_patell", "statistic"] == tests.loc["patell", "statistic"]
    assert tests.loc["kp", "statistic"] == tests.loc["bmp", "statistic"]


def test_repeated_events_of_a_security_do_not_enter_the_correlation():
    # Independent idiosyncratic returns: the cross-correlation of residuals
    # is about zero. Pairing a security with itself on shared estimation
    # days used to add correlations of one.
    data, dates = _returns([1, 2, 3])
    events = pd.DataFrame(
        {"id": [1, 1, 2, 2, 3, 3], "event_date": list(dates[[400, 430]]) * 3}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.abnormal_returns(data, events, model="market", market="mkt")
    assert abs(res.mean_correlation) < 0.05
