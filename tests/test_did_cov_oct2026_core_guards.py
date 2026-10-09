"""Coverage campaign (did, Oct 2026) -- shared primitives and entry-point guards.

* ``did/_core.py``: degenerate inputs of the influence-function standard
  error, the joint Wald test, the cohort-share influence and the R-style
  covariate formula, each against the value it must return.
* ``sp.did_analysis``: the design-based (random adoption timing) methods must
  return the estimator's own result and skip the three parallel-trends
  diagnostics, saying why.
* ``sp.did_multiplegt_dyn``, ``sp.drdid``, ``sp.twfe_decomposition``,
  ``sp.etwfe``: refusals by exception type and message.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.did import _core as C
from statspai.exceptions import DataInsufficient, MethodIncompatibility

sr_mod = importlib.import_module("statspai.did._staggered_rollout")


def _staggered(seed=0, n_units=30, T=8, cohorts=(4, 6, 0), effect=2.0, noise=0.3):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        g = cohorts[i % len(cohorts)]
        a = rng.normal()
        for t in range(1, T + 1):
            d = float(g > 0 and t >= g)
            rows.append(
                {
                    "unit": i,
                    "t": t,
                    "g": g,
                    "d": d,
                    "x": rng.normal(),
                    "y": a + 0.3 * t + effect * d + noise * rng.normal(),
                }
            )
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
#  did/_core.py
# ══════════════════════════════════════════════════════════════════════


def test_influence_function_se_degenerate_inputs():
    assert np.isnan(C.influence_function_se(np.empty(0)))
    out = C.influence_function_se(np.empty((0, 3)))
    assert out.shape == (3,) and np.isnan(out).all()
    # one cluster carries no between-cluster variance
    psi = np.random.default_rng(0).normal(size=(10, 2))
    one = C.influence_function_se(psi, cluster_ids=np.zeros(10, dtype=int))
    assert np.isnan(one).all()


def test_influence_se_did_cluster_length_mismatch():
    psi = np.ones((6, 1))
    with pytest.raises(MethodIncompatibility, match="4 entries for 6"):
        C.influence_se_did(psi, 6, cluster_ids=np.arange(4))
    # every row its own cluster reproduces the unclustered formula
    rng = np.random.default_rng(1)
    psi = rng.normal(size=(12, 2))
    np.testing.assert_allclose(
        C.influence_se_did(psi, 12, cluster_ids=np.arange(12)),
        C.influence_se_did(psi, 12),
        rtol=1e-12,
    )


def test_joint_wald_scalar_and_shape_guard():
    out = C.joint_wald(np.array([2.0]), np.array(4.0))
    # one restriction: W = b^2 / v, chi2(1); the 1e-10 ridge moves it by 1e-11
    assert out["statistic"] == pytest.approx(1.0, abs=1e-9)
    assert out["df"] == 1
    assert out["pvalue"] == pytest.approx(stats.chi2.sf(1.0, 1), abs=1e-9)
    with pytest.raises(ValueError, match="inconsistent with estimates size 2"):
        C.joint_wald(np.array([1.0, 2.0]), np.eye(3))


def test_cohort_share_context_and_weight_influence():
    cells = np.array([4, 6, 4])
    unit_cohorts = np.array([4, 4, 6, 0, 0, 6, 4, 0])
    pg, ind = C.cohort_share_context(cells, unit_cohorts)
    np.testing.assert_allclose(pg, [3 / 8, 2 / 8, 3 / 8])
    assert ind.shape == (8, 3)
    wif = C.weight_influence(pg, ind)
    # the weights sum to one for every sample, so their influence functions
    # sum to zero across cells, and average to zero across units
    np.testing.assert_allclose(wif.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(wif.mean(axis=0), 0.0, atol=1e-12)
    # no treated mass: nothing to differentiate
    zero = C.weight_influence(np.zeros(3), ind)
    assert zero.shape == ind.shape and not zero.any()
    with pytest.raises(ValueError, match="unit_weights has length 3"):
        C.cohort_share_context(cells, unit_cohorts, unit_weights=np.ones(3))


def test_normalize_se_method_auto_without_an_analytic_option():
    got = C.normalize_se_method(
        "auto", supported=["jackknife", "placebo"], function="f"
    )
    assert got == "jackknife"
    assert (
        C.normalize_se_method("auto", supported=["bootstrap", "analytic"], function="f")
        == "analytic"
    )
    # few clusters: a resampling method is preferred when there is one
    assert (
        C.normalize_se_method(
            "auto", supported=["analytic", "bootstrap"], function="f", n_clusters=3
        )
        == "bootstrap"
    )


def test_parallel_trends_block_unknown_label():
    with pytest.raises(ValueError, match="unknown parallel-trends label 'nope'"):
        C.parallel_trends_block("nope")
    label = sorted(C.PT_ASSUMPTIONS)[0]
    assert isinstance(C.parallel_trends_block(label), dict)


def test_covariates_from_formula_branches():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "label": ["a", "b", "a", "b"], "sq": [0.0] * 4}
    )
    out, cols = C.covariates_from_formula(df, "~ x + I(x**2)")
    assert "x" in cols and len(cols) == 2
    built = [c for c in cols if c != "x"][0]
    np.testing.assert_allclose(out[built], df["x"] ** 2)
    # an existing column of the same values is reused untouched
    pd.testing.assert_series_equal(out["x"], df["x"])
    # intercept only: no covariates at all
    same, none = C.covariates_from_formula(df, "~ 1")
    assert none == [] and same is df
    with pytest.raises(MethodIncompatibility, match="could not evaluate"):
        C.covariates_from_formula(df, "~ nope_column")
    # a term whose name is an existing column holding other values
    clash = df.rename(columns={"sq": "I(x ** 2)"})
    with pytest.raises(MethodIncompatibility, match="already exist"):
        C.covariates_from_formula(clash, "~ I(x**2)")


def test_calendar_time_period_dtype_matches_integer_coding():
    df = _staggered()
    ref = sp.sun_abraham(df, y="y", g="g", t="t", i="unit")
    cal = df.copy()
    cal["t"] = [pd.Period(year=1999 + int(v), freq="Y") for v in df["t"]]
    # never-treated units carry a date after the panel ends (the 2100 habit)
    cal["g"] = pd.PeriodIndex(
        [pd.Period(year=1999 + int(v) if v > 0 else 2100, freq="Y") for v in df["g"]]
    )
    cal["t"] = pd.PeriodIndex(cal["t"])
    recoded, info = C.index_calendar_time(cal, "t", "g", function="test")
    np.testing.assert_array_equal(recoded["g"], df["g"])
    np.testing.assert_array_equal(recoded["t"], df["t"])
    assert info["regular"] is True and len(info["periods"]) == 8
    got = sp.sun_abraham(cal, y="y", g="g", t="t", i="unit")
    # the calendar is recoded to 1..P in observed order: identical regression
    assert got.estimate == pytest.approx(ref.estimate, abs=1e-10)
    assert got.se == pytest.approx(ref.se, rel=1e-8)


# ══════════════════════════════════════════════════════════════════════
#  sp.did_analysis, design-based methods
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "method", ["staggered_rollout", "staggered_cs", "staggered_sa"]
)
def test_did_analysis_design_based_methods_delegate_and_skip_pt_diagnostics(method):
    df = _staggered()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.did_analysis(df, y="y", treat="g", time="t", id="unit", method=method)
        direct = getattr(sr_mod, method)(df, y="y", i="unit", t="t", g="g")
    main = out.main_result
    assert out.design == "staggered"
    assert out.event_study_result is None and out.sensitivity is None
    assert main.estimate == pytest.approx(direct.estimate, abs=1e-12)
    assert main.se == pytest.approx(direct.se, rel=1e-12)
    text = "\n".join(out.steps_log)
    assert "Bacon decomposition skipped" in text
    assert "Event study / pre-trend test skipped" in text
    assert "Honest-DiD sensitivity skipped" in text
    assert "random adoption timing" in text


def test_did_analysis_design_based_method_requires_id():
    with pytest.raises(MethodIncompatibility, match="'id' is required"):
        sp.did_analysis(
            _staggered(), y="y", treat="g", time="t", method="staggered_rollout"
        )


# ══════════════════════════════════════════════════════════════════════
#  sp.did_multiplegt_dyn
# ══════════════════════════════════════════════════════════════════════

M_KW = dict(y="y", group="unit", time="t", treatment="d", dynamic=2, n_boot=0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(design=1.5), "design must be a share"),
        (dict(by_path=0), "by_path must be a positive integer"),
        (dict(by_path=True), "by_path must be a positive integer"),
        (dict(design=0.9, continuous=1), "cannot be combined with continuous"),
        (dict(normalized_weights=True), "pass\\s+normalized=True"),
        (dict(by_path=2, se_method="bootstrap"), "by_path= reports analytic"),
    ],
)
def test_dcdh_argument_guards(kwargs, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.did_multiplegt_dyn(_staggered(), **{**M_KW, **kwargs})


def test_dcdh_cluster_with_missing_values_is_refused():
    df = _staggered()
    df["cl"] = df["unit"].astype(float)
    df.loc[df["unit"] == 3, "cl"] = np.nan
    with pytest.raises(MethodIncompatibility, match="has missing values"):
        sp.did_multiplegt_dyn(df, cluster="cl", **M_KW)


def test_dcdh_normalized_weights_columns_sum_to_one():
    df = _staggered()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.did_multiplegt_dyn(
            df,
            se_method="analytic",
            normalized=True,
            normalized_weights=True,
            **M_KW,
        )
    tab = [
        v
        for v in r.model_info.values()
        if isinstance(v, pd.DataFrame) and v.columns.name == "effect"
    ]
    assert len(tab) == 1
    w = tab[0]
    assert list(w.columns) == [1, 2, 3]
    # each normalized effect is a weighted average over treatment lags
    np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-12)
    # binary absorbing treatment: effect l spreads evenly over its l lags
    np.testing.assert_allclose(w[2].dropna(), [0.5, 0.5], atol=1e-12)


def test_dcdh_horizon_beyond_the_panel_is_reported_as_missing():
    # Adoption in period 6 of 8: horizons 0..2 exist, 3 and 4 do not.
    df = _staggered(cohorts=(6, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.did_multiplegt_dyn(df, se_method="analytic", **{**M_KW, "dynamic": 4})
    es = r.model_info["event_study"].set_index("relative_time")
    assert np.isfinite(es.loc[[0, 1, 2], "att"]).all()
    assert es.loc[[3, 4], "att"].isna().all()
    assert (es.loc[[3, 4], "n_switchers"] == 0).all()
    # truth 2, noise 0.3, 15 switchers: 0.5 is several standard errors
    assert r.estimate == pytest.approx(2.0, abs=0.5)


# ══════════════════════════════════════════════════════════════════════
#  wooldridge_did.py: drdid, twfe_decomposition, etwfe
# ══════════════════════════════════════════════════════════════════════


def _two_by_two(seed=0, n=400):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, n)
    post = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    y = 1 + 0.5 * x + 2 * g + 3 * post + 4 * g * post + rng.normal(size=n)
    return pd.DataFrame({"y": y, "treated": g, "post": post, "x": x, "w": 1.0})


def test_drdid_guards_and_unit_weights():
    df = _two_by_two()
    kw = dict(y="y", group="treated", time="post", covariates=["x"])
    with pytest.raises(MethodIncompatibility, match="trim_level must be in"):
        sp.drdid(df, trim_level=0.0, **kw)
    plain = sp.drdid(df, **kw)
    wtd = sp.drdid(df, weights="w", **kw)
    # unit weights are renormalised to mean one: nothing changes
    assert wtd.estimate == pytest.approx(plain.estimate, abs=1e-10)
    assert wtd.se == pytest.approx(plain.se, rel=1e-10)
    # truth 4, n = 400, unit noise: the SE is about 0.2
    assert plain.estimate == pytest.approx(4.0, abs=0.8)


def test_drdid_panel_missing_weight_column_is_named():
    df = _two_by_two().assign(id=lambda d: np.arange(len(d)))
    with pytest.raises(MethodIncompatibility, match="nope"):
        sp.drdid(
            df,
            y="y",
            group="treated",
            time="post",
            covariates=["x"],
            id="id",
            weights="nope",
        )


def test_twfe_decomposition_guards():
    df = _staggered()
    kw = dict(y="y", group="unit", time="t", first_treat="g")
    with pytest.raises(MethodIncompatibility, match="columns not found"):
        sp.twfe_decomposition(df.drop(columns="g"), **kw)
    dup = pd.concat([df, df.iloc[:1]], ignore_index=True)
    with pytest.raises(MethodIncompatibility, match="more than one row"):
        sp.twfe_decomposition(dup, **kw)


def test_etwfe_requires_a_never_treated_unit_by_default():
    df = _staggered(cohorts=(4, 6))
    with pytest.raises(DataInsufficient, match="never-treated"):
        sp.wooldridge_did(df, y="y", group="unit", time="t", first_treat="g")
