"""Regression tests for the October 2026 review of the causal modules.

Each test states a proposition that holds exactly: a known truth on a
noiseless design, equality with the same panel coded another way, invariance
to row order, or a specific exception / warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _panel(seed=2, n=150, periods=8, noise=0.5, cohorts=(0, 4, 6, 7), p=None):
    rng = np.random.default_rng(seed)
    p = p or [1.0 / len(cohorts)] * len(cohorts)
    ids = np.repeat(np.arange(n), periods)
    t = np.tile(np.arange(1, periods + 1), n)
    g = np.repeat(rng.choice(cohorts, n, p=p), periods)
    d = ((g > 0) & (t >= g)).astype(int)
    tau = np.where(d == 1, 1.0 + 0.5 * (t - g), 0.0)
    y = np.repeat(rng.normal(size=n), periods) + 0.3 * t + tau
    y = y + rng.normal(scale=noise, size=n * periods)
    return pd.DataFrame({"id": ids, "t": t, "g": g, "D": d, "y": y, "tau": tau})


def _biennial(df):
    """The same panel with periods two years apart."""
    out = df.copy()
    out["t"] = 1999 + 2 * df["t"]
    out["g"] = np.where(df["g"] > 0, 1999 + 2 * df["g"], 0)
    return out


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# --------------------------------------------------------------- gapped time
def test_sun_abraham_gapped_years_equal_consecutive_coding():
    df = _panel()
    a = sp.sun_abraham(df, "y", "g", "t", "id")
    b = sp.sun_abraham(_biennial(df), "y", "g", "t", "id")
    assert b.estimate == a.estimate
    assert b.se == a.se and b.se > 0
    pd.testing.assert_frame_equal(a.detail, b.detail, check_exact=True)
    info = b.model_info["calendar_time"]
    assert info["periods"] == [2001 + 2 * k for k in range(8)]
    assert info["regular"] is True
    assert "calendar_time" not in a.model_info


@pytest.mark.parametrize("never_only", [True, False])
def test_stacked_did_gapped_years_equal_consecutive_coding(never_only):
    df = _panel()
    kw = dict(window=(-2, 2), never_treated_only=never_only)
    a = sp.stacked_did(df, "y", "id", "t", "g", **kw)
    b = sp.stacked_did(_biennial(df), "y", "id", "t", "g", **kw)
    assert np.isfinite(a.estimate) and b.estimate == a.estimate
    assert b.se == a.se
    pd.testing.assert_frame_equal(a.detail, b.detail, check_exact=True)
    assert b.model_info["calendar_time"]["source"] == "numeric_gaps"


def test_unevenly_spaced_integer_periods_warn():
    df = _panel()
    df = df[df["t"] != 5]  # a hole: no relative time -1 for cohort 6
    with pytest.warns(UserWarning, match="not evenly spaced"):
        res = sp.sun_abraham(df, "y", "g", "t", "id")
    assert res.se > 0


def test_stacked_did_reports_missing_rows():
    df = _panel()
    df.loc[df.index[:7], "y"] = np.nan
    with pytest.warns(UserWarning, match=r"7 row\(s\) of the input"):
        res = sp.stacked_did(df, "y", "id", "t", "g", window=(-2, 2))
    ref = sp.stacked_did(df.dropna(), "y", "id", "t", "g", window=(-2, 2))
    assert res.estimate == ref.estimate


# ------------------------------------------------------------------ gardner
def test_gardner_drops_period_without_untreated_and_recovers_truth():
    df = _panel(noise=0.0, cohorts=(3, 5, 8), p=[0.4, 0.3, 0.3])
    with pytest.warns(UserWarning, match=r"period\(s\) \[8\]") as rec:
        res = sp.gardner_did(df, "y", "id", "t", "g")
    assert any("ATT over the remaining treated cells" in str(w.message) for w in rec)
    truth = df.loc[(df["D"] == 1) & (df["t"] < 8), "tau"].mean()
    assert res.estimate == pytest.approx(truth, abs=1e-10)
    info = res.model_info["unidentified_cells"]
    assert info["periods_without_untreated"] == [8]
    assert info["n_dropped"] == int(((df["D"] == 1) & (df["t"] == 8)).sum())


def test_gardner_drops_always_treated_units():
    df = _panel(noise=0.0, cohorts=(0, 1, 4), p=[0.4, 0.2, 0.4])
    with pytest.warns(UserWarning, match="treated in every row"):
        res = sp.gardner_did(df, "y", "id", "t", "g")
    truth = df.loc[(df["g"] == 4) & (df["D"] == 1), "tau"].mean()
    assert res.estimate == pytest.approx(truth, abs=1e-10)


def test_gardner_identified_panel_reports_nothing():
    df = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sp.gardner_did(df, "y", "id", "t", "g")
    assert res.model_info["unidentified_cells"] is None


def test_gardner_missing_cohort_is_never_treated():
    df = _panel()
    ref = sp.gardner_did(df, "y", "id", "t", "g")
    nan_coded = df.assign(g=df["g"].where(df["g"] > 0))
    res = sp.gardner_did(nan_coded, "y", "id", "t", "g")
    assert res.estimate == ref.estimate and res.se == ref.se
    assert res.model_info["n_units"] == df["id"].nunique()


# ---------------------------------------------------------- cate_gain_curve
def _rct(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    w = rng.binomial(1, 0.5, n)
    y = (1 + 2 * x) * w + rng.normal(size=n)
    return pd.DataFrame({"x": x, "w": w, "y": y})


def test_gain_curve_constant_score_has_zero_gain_whatever_the_row_order():
    df = _rct().sort_values("x", ascending=False).reset_index(drop=True)
    res = sp.cate_gain_curve(df, np.zeros(len(df)), "y", "w", n_boot=20, seed=1)
    assert res.auc == 0.0
    assert np.all(res.curve["cumulative_gain"].to_numpy() == 0.0)
    assert res.auc_ci == (0.0, 0.0)


def test_gain_curve_with_ties_is_row_order_invariant():
    df = _rct().sort_values("x", ascending=False).reset_index(drop=True)
    df["coarse"] = np.round(df["x"], 1)
    a = sp.cate_gain_curve(df, "coarse", "y", "w")
    b = sp.cate_gain_curve(df.sample(frac=1, random_state=3), "coarse", "y", "w")
    assert b.auc == pytest.approx(a.auc, rel=1e-12)
    np.testing.assert_allclose(
        a.curve["cumulative_effect"], b.curve["cumulative_effect"], rtol=1e-10
    )
    # a coarse ranking cannot beat the ranking it was rounded from by luck
    assert a.auc < sp.cate_gain_curve(df, "x", "y", "w").auc


def test_gain_curve_tied_group_enters_with_fractional_weight():
    # Two score groups of four; the first cut (k = 2) falls inside the top
    # group, whose difference in means is 3 in any row order.
    y = np.array([4.0, 6.0, 1.0, 3.0, 9.0, 9.0, 0.0, 0.0] * 2)
    w = np.array([1, 1, 0, 0, 1, 1, 0, 0] * 2)
    s = np.array([1.0] * 4 + [0.0] * 4 + [1.0] * 4 + [0.0] * 4)
    df = pd.DataFrame({"y": y, "w": w})
    res = sp.cate_gain_curve(df, s, "y", "w", n_steps=8, n_quantiles=2)
    assert res.curve["cumulative_effect"].iloc[0] == pytest.approx(3.0)
    assert res.curve["cumulative_effect"].iloc[3] == pytest.approx(3.0)
    assert res.curve["cumulative_effect"].iloc[-1] == pytest.approx(6.0)


def test_gain_curve_input_checks():
    df = _rct(400)
    with pytest.raises(MethodIncompatibility, match="not numeric"):
        sp.cate_gain_curve(df.assign(w=df["w"].map({0: "c", 1: "t"})), "x", "y", "w")
    # a Series is paired with rows by its index
    shuffled = df["x"].sample(frac=1, random_state=0)
    a = sp.cate_gain_curve(df, shuffled, "y", "w")
    b = sp.cate_gain_curve(df, "x", "y", "w")
    assert a.auc == b.auc
    with pytest.raises(MethodIncompatibility, match="index"):
        sp.cate_gain_curve(df, df["x"].set_axis(np.arange(len(df)) + 10_000), "y", "w")


def test_gain_curve_warns_when_top_units_are_one_arm():
    df = _rct(400)
    score = df["x"].to_numpy().copy()
    top = np.argsort(-score)[:12]
    df.loc[top, "w"] = 1
    with pytest.warns(UserWarning, match=r"at \d+ of 100 step"):
        res = sp.cate_gain_curve(df, score, "y", "w")
    assert res.diagnostics["n_steps_without_treatment_variation"] >= 1


# ----------------------------------------------------------- did_multiplegt
def test_did_multiplegt_missing_cells_equal_dropping_them():
    df = _panel(n=100)
    rng = np.random.default_rng(1)
    df.loc[rng.choice(len(df), 15, replace=False), "y"] = np.nan
    df.loc[rng.choice(len(df), 5, replace=False), "D"] = np.nan
    kw = dict(n_boot=8, seed=1, placebo=1, dynamic=1)
    with pytest.warns(Warning, match="rows dropped for missing"):
        res = sp.did_multiplegt(df, "y", "id", "t", "D", **kw)
    ref = _quiet(sp.did_multiplegt, df.dropna(), "y", "id", "t", "D", **kw)
    assert np.isfinite(res.estimate) and res.estimate == ref.estimate
    assert res.se == ref.se
    assert res.model_info["n_rows_dropped_missing"] == int(
        df[["y", "D"]].isna().any(axis=1).sum()
    )


def test_did_multiplegt_without_switchers_is_missing_not_zero():
    df = _panel(n=60)
    df["D"] = (df["g"] > 0).astype(int)  # never changes within a unit
    with pytest.warns(UserWarning, match="DID_M is not estimable"):
        res = sp.did_multiplegt(df, "y", "id", "t", "D", n_boot=5, seed=1)
    assert np.isnan(res.estimate) and np.isnan(res.se) and np.isnan(res.pvalue)
    assert all(np.isnan(v) for v in res.ci)


# ---------------------------------------------------------- twowayfeweights
@pytest.mark.parametrize("value", [1, 3.5])
def test_twowayfeweights_constant_treatment_is_refused(value):
    df = _panel(n=40, periods=6)
    df["D"] = value
    with pytest.raises(DataInsufficient, match="no variation"):
        sp.twowayfeweights(df, "y", "id", "t", "D")


# --------------------------------------------------------- causal discovery
@pytest.mark.parametrize(
    "method", ["pc", "hill_climb", "mmhc", "ges", "fci", "lingam", "notears"]
)
def test_bootstrap_edges_finds_the_collider_for_every_listed_method(method):
    rng = np.random.default_rng(0)
    n = 2000
    a, b = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": a + b + rng.normal(size=n)})
    out = sp.bootstrap_edges(df, method, n_boot=6, seed=1)
    assert set(out["from"]) | set(out["to"]) <= {"a", "b", "c"}
    strength = {
        frozenset((r["from"], r["to"])): r["strength"] for _, r in out.iterrows()
    }
    assert strength[frozenset(("a", "c"))] == 1.0
    assert strength[frozenset(("b", "c"))] == 1.0


def test_hill_climb_refuses_a_constant_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "k": 1.0})
    df["b"] = df["a"] + rng.normal(size=200)
    with pytest.raises(DataInsufficient, match="constant"):
        sp.hill_climb(df)


# ------------------------------------------------------------------ matching
def test_stratified_match_reports_treated_units_without_controls():
    rng = np.random.default_rng(3)
    n = 900
    blk = rng.integers(0, 4, n)
    a = rng.binomial(1, 0.2 + 0.05 * blk)
    y = 2.0 * a + 0.5 * blk + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "a": a, "x": blk.astype(float), "blk": blk})
    df = df[~((df["blk"] == 3) & (df["a"] == 0))]
    n_lost = int(((df["blk"] == 3) & (df["a"] == 1)).sum())
    kw = dict(method="stratify", strata="blk")
    with pytest.warns(UserWarning, match=f"{n_lost} of .* treated units") as rec:
        res = sp.match(df, "y", "a", ["x"], **kw)
    assert any("strata that hold both arms" in str(w.message) for w in rec)
    assert res.model_info["n_treated_outside_strata"] == n_lost
    # the estimate is the one on the strata with both arms
    ref = _quiet(sp.match, df[df["blk"] < 3], "y", "a", ["x"], **kw)
    assert res.estimate == pytest.approx(ref.estimate, rel=1e-12)
    assert ref.model_info["n_treated_outside_strata"] == 0


# --------------------------------------------------------------------- etwfe
@pytest.mark.parametrize("never", [np.inf, 9999, np.nan])
def test_etwfe_never_treated_codings_agree(never):
    df = _panel()
    ref = sp.etwfe(df, "y", "id", "t", "g")
    coded = df.assign(g=np.where(df["g"] > 0, df["g"], never))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sp.etwfe(coded, "y", "id", "t", "g")
    assert res.estimate == ref.estimate and res.se == ref.se


# ------------------------------------------------------------ smaller checks
def _multi_cutoff(seed=0):
    rng = np.random.default_rng(seed)
    cut = rng.choice([30.0, 60.0], 600)
    x = cut + rng.uniform(-20, 20, 600)
    y = 0.02 * x + 1.0 * (x >= cut) + rng.normal(scale=0.3, size=600)
    return pd.DataFrame({"y": y, "x": x, "cut": cut})


def test_rdmcplot_validates_kernel_and_level():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    df = _multi_cutoff()
    kw = dict(y="y", x="x", cutoff_var="cut", p=1, h=10.0)
    with pytest.raises(MethodIncompatibility, match="kernel"):
        sp.rdmcplot(df, kernel="zz", **kw)
    with pytest.raises(MethodIncompatibility, match="ci_level"):
        sp.rdmcplot(df, ci_level=1.5, **kw)
    fig_a, _ = sp.rdmcplot(df, kernel="Uniform", **kw)
    fig_b, _ = sp.rdmcplot(df, kernel="uniform", **kw)
    fig_c, _ = sp.rdmcplot(df, kernel="triangular", **kw)

    def fit(f):
        return np.asarray(f.rdmcplot_data[30.0]["vars_poly"]["rdplot_y"])

    np.testing.assert_array_equal(fit(fig_a), fit(fig_b))
    assert not np.array_equal(fit(fig_b), fit(fig_c))


def test_tmle_known_probability_must_be_interior():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    a = rng.binomial(1, 0.5, n)
    df = pd.DataFrame({"y": 1 + a + x + rng.normal(size=n), "a": a, "x": x})
    for bad in (1.0, 0.0, 1.5):
        with pytest.raises(MethodIncompatibility, match="strictly between 0 and 1"):
            sp.tmle(df, "y", "a", ["x"], g1W=bad)
    with pytest.raises(MethodIncompatibility, match=r"outside\s+\[0, 1\]"):
        sp.tmle(df, "y", "a", ["x"], g1W=np.full(n, 1.2))


def test_synth_ttest_validates_alpha_and_penalization():
    df = sp.california_prop99()
    kw = dict(
        outcome="packspercapita",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )
    with pytest.raises(MethodIncompatibility, match="penalization"):
        sp.synth_ttest(df, penalization=-1, **kw)
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.synth_ttest(df, alpha=1.5, **kw)


def test_rd_optimized_reports_missing_rows():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 400)
    y = 0.5 * x + 1.0 * (x >= 0) + rng.normal(scale=0.3, size=400)
    df = pd.DataFrame({"y": y, "x": x})
    df.loc[:8, "y"] = np.nan
    with pytest.warns(UserWarning, match="dropped 9 of 400 rows"):
        res = sp.rd_optimized(df, "y", "x", M=2.0)
    ref = sp.rd_optimized(df.dropna(), "y", "x", M=2.0)
    assert res.estimate == ref.estimate


def test_bayes_net_reports_missing_rows():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 300)
    b = (a + rng.integers(0, 2, 300)) % 2
    df = pd.DataFrame({"a": a.astype(float), "b": b.astype(float)})
    df.loc[:4, "b"] = np.nan
    with pytest.warns(UserWarning, match="dropped 5 of 300"):
        sp.bayes_net("a -> b", data=df)


def test_dml_did_says_which_text_label_is_the_post_period():
    from sklearn.linear_model import LinearRegression, LogisticRegression

    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    d = rng.binomial(1, 0.5, n)
    pre = x + rng.normal(size=n)
    post = pre + 1.0 + 2.0 * d + rng.normal(size=n)
    long = pd.DataFrame(
        {
            "id": np.tile(np.arange(n), 2),
            "y": np.r_[pre, post],
            "d": np.tile(d, 2),
            "x": np.tile(x, 2),
            "when": ["pre"] * n + ["post"] * n,
            "t01": [0] * n + [1] * n,
        }
    )
    kw = dict(
        y="y",
        treat="d",
        covariates=["x"],
        id="id",
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        random_state=0,
    )
    with pytest.warns(UserWarning, match="'pre' is taken as the post-treatment"):
        txt = sp.dml_did(long, time="when", **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        num = sp.dml_did(long, time="t01", **kw)
    assert txt.estimate == pytest.approx(-num.estimate, rel=1e-10)
    assert num.estimate == pytest.approx(2.0, abs=0.5)
