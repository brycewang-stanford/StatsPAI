"""Coverage gaps (batch B) for sp.did_multiplegt and sp.did_multiplegt_dyn.

Argument validation, horizons the panel cannot support, the per-path and
normalized-weights outputs, and a set of invariances: rows that cannot
contribute (single-observation units, zero-weight units, always-treated
units under ``controls=``, units observed only in an isolated period) must
leave the estimates unchanged.
"""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
import statspai.did.did_multiplegt  # noqa: F401
import statspai.did.did_multiplegt_dyn  # noqa: F401
from statspai.exceptions import DataInsufficient, MethodIncompatibility

M = sys.modules["statspai.did.did_multiplegt"]
DYN = sys.modules["statspai.did.did_multiplegt_dyn"]

K = dict(y="y", group="u", time="t", treatment="d")
BOOT = dict(n_boot=10, seed=1)


def make_panel(seed=0, n=40, T=7):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n):
        first = [3, 4, 5, 99][u % 4]
        fe = rng.normal()
        for t in range(1, T + 1):
            d = float(t >= first)
            rows.append(
                {
                    "u": u,
                    "t": t,
                    "d": d,
                    "y": fe + 0.2 * t + 1.5 * d + rng.normal() * 0.5,
                    "x": rng.normal(),
                    "cl": u // 2,
                    "w": 1.0 + u % 3,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def dyn(df, **kw):
    """Analytic-variance fit; the headline's bootstrap notice is not under test."""
    kw.setdefault("se_method", "analytic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_multiplegt_dyn(df, **K, **kw)


def es(res):
    return res.model_info["event_study"].set_index("relative_time")


# ======================================================================
# sp.did_multiplegt
# ======================================================================
def test_dcdh_placebo_sign_is_validated(panel):
    with pytest.raises(ValueError, match="placebo_sign must be 'stata' or 'r'"):
        sp.did_multiplegt(panel, **K, placebo_sign="python")


def test_dcdh_residualize_without_controls_is_identity():
    dy = np.array([1.0, -2.0, 0.5])
    assert M._residualize(dy, np.empty((3, 0))) is dy
    # with a control, the residual is orthogonal to it and to the constant
    x = np.array([[0.0], [1.0], [3.0]])
    res = M._residualize(dy, x)
    assert abs(res.sum()) < 1e-12 and abs(float(res @ x[:, 0])) < 1e-12


def test_dcdh_period_pair_without_common_units_is_skipped(panel):
    base = sp.did_multiplegt(panel, **K, n_boot=5, seed=1)
    rng = np.random.default_rng(3)
    isolated = pd.DataFrame(
        {"u": np.arange(900, 910), "t": 20, "d": 0.0, "y": rng.normal(size=10)}
    )
    both = sp.did_multiplegt(pd.concat([panel, isolated]), **K, n_boot=5, seed=1)
    assert both.estimate == pytest.approx(base.estimate, rel=1e-12)
    assert both.model_info["n_switchers"] == base.model_info["n_switchers"]


def test_dcdh_placebo_helper_reports_no_cells_on_two_periods(panel):
    two = panel[panel["t"].isin([2, 3])]
    out = M._estimate_placebo(two, "y", "u", "t", "d", None, lag=1)
    assert out["n_cells"] == 0


def test_dcdh_average_cumulative_effect_needs_dynamic_draws():
    assert M._avg_cumulative_effect([], None, 0.05) is None
    assert M._avg_cumulative_effect([{"estimate": 1.0}], None, 0.05) is None


# ======================================================================
# sp.did_multiplegt_dyn: validation
# ======================================================================
@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (dict(design=1.5), r"design must be a share in \(0, 1\]"),
        (dict(design=0), r"design must be a share in \(0, 1\]"),
        (dict(by_path=0), "by_path must be a positive integer"),
        (dict(by_path=True), "by_path must be a positive integer"),
        (dict(design=0.5, continuous=1), "cannot be combined with continuous="),
        (dict(by_path=1, continuous=1), "cannot be combined with continuous="),
        (dict(normalized_weights=True), "pass normalized=True as well"),
        (dict(by_path=1, se_method="bootstrap"), "pass se_method='analytic'"),
    ],
)
def test_dyn_option_combinations_are_rejected(panel, kwargs, fragment):
    with pytest.raises(MethodIncompatibility, match=fragment):
        sp.did_multiplegt_dyn(panel, **K, **BOOT, **kwargs)


def test_dyn_cluster_with_missing_values_is_rejected(panel):
    df = panel.assign(cl=panel["cl"].astype(float))
    df.loc[0, "cl"] = np.nan
    with pytest.raises(MethodIncompatibility, match="has missing values") as ei:
        sp.did_multiplegt_dyn(df, **K, **BOOT, cluster="cl")
    assert ei.value.diagnostics == {"cluster": "cl"}


def test_dyn_effects_equal_range_must_cover_two_effects(panel):
    with pytest.raises(ValueError, match=r"effects_equal=\(5, 9\) selects 0 of"):
        sp.did_multiplegt_dyn(panel, **K, **BOOT, dynamic=2, effects_equal=(5, 9))


def test_dyn_continuous_needs_enough_not_yet_switched_rows(panel):
    with pytest.raises(DataInsufficient, match="degree-3 polynomial"):
        dyn(panel[panel["u"] < 3], continuous=3)


# ======================================================================
# horizons the panel cannot support
# ======================================================================
def test_dyn_unsupported_horizons_are_nan_rows_not_dropped(panel):
    res = dyn(panel, dynamic=6, placebo=3)
    tab = es(res)
    assert tab.index.tolist() == list(range(-3, 7))
    # the latest switch is at t=5 of 7, so effects 5 and 6 have no switcher
    for h in (5, 6):
        assert np.isnan(tab.loc[h, "att"]) and np.isnan(tab.loc[h, "se"])
        assert tab.loc[h, "n_switchers"] == 0 and tab.loc[h, "type"] == "dynamic"
    assert tab.loc[4, "n_switchers"] == 10 and np.isfinite(tab.loc[4, "att"])
    # the headline averages the estimable effects only
    assert res.estimate == pytest.approx(tab.loc[0:4, "att"].mean())
    assert np.isfinite(res.se) and res.se > 0
    # a joint test cannot be formed over effects that include a missing one
    assert res.model_info["joint_effects_test"] is None
    short = dyn(panel, dynamic=4, placebo=3)
    assert short.model_info["joint_effects_test"]["df"] == 5
    assert short.estimate == pytest.approx(res.estimate)


def test_dyn_bootstrap_equality_test_degenerate_cases(panel):
    kw = dict(effects_equal=True, se_method="bootstrap")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = sp.did_multiplegt_dyn(panel, **K, **BOOT, dynamic=0, **kw)
        missing = sp.did_multiplegt_dyn(panel, **K, **BOOT, dynamic=6, **kw)
        few = sp.did_multiplegt_dyn(panel, **K, n_boot=2, seed=1, dynamic=2, **kw)
        ok = sp.did_multiplegt_dyn(panel, **K, **BOOT, dynamic=2, **kw)
    assert one.model_info["effects_equal_test"] is None  # nothing to compare
    assert missing.model_info["effects_equal_test"] is None  # NaN effect
    assert few.model_info["effects_equal_test"] is None  # too few draws
    test = ok.model_info["effects_equal_test"]
    assert test["df"] == 2 and 0.0 <= test["pvalue"] <= 1.0


def test_dyn_bootstrap_draw_without_switchers_is_not_fatal(panel):
    # one switcher among five groups: about a third of the cluster draws
    # contain no switcher at all and must simply contribute nothing
    tiny = panel[panel["u"].isin([0, 3, 7, 11, 15])]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = sp.did_multiplegt_dyn(tiny, **K, n_boot=30, seed=4, dynamic=1)
    assert np.isfinite(res.estimate)
    assert np.isfinite(res.se) and res.se > 0
    assert any("bootstrap replicates failed" in str(w.message) for w in rec)


# ======================================================================
# invariances
# ======================================================================
def test_dyn_single_observation_unit_cannot_contribute(panel):
    extra = panel.iloc[[0]].assign(u=999)
    a, b = dyn(panel), dyn(pd.concat([panel, extra]))
    np.testing.assert_allclose(es(a)["att"], es(b)["att"], rtol=1e-12)


def test_dyn_zero_weight_units_equal_dropping_them(panel):
    zw = panel.copy()
    zw.loc[zw["u"] % 4 == 0, "w"] = 0.0
    a = dyn(zw, weights="w", dynamic=1)
    b = dyn(zw[zw["u"] % 4 != 0], weights="w", dynamic=1)
    np.testing.assert_allclose(es(a)["att"], es(b)["att"], rtol=1e-10)
    np.testing.assert_allclose(es(a)["se"], es(b)["se"], rtol=1e-10)


def test_dyn_controls_ignore_a_baseline_with_no_switch_variation(panel):
    # always-treated groups share one period-one treatment and never switch:
    # no first-difference regression is fitted there and nothing else moves
    always = panel[panel["u"] < 6].assign(u=lambda z: z["u"] + 500, d=1.0)
    a = dyn(panel, controls=["x"])
    b = dyn(pd.concat([panel, always]), controls=["x"])
    np.testing.assert_allclose(es(a)["att"], es(b)["att"], rtol=1e-10)


def test_dyn_controls_need_enough_rows_at_each_baseline(panel):
    rng = np.random.default_rng(8)
    # two groups start treated and switch off at different dates
    off = panel[panel["u"].isin([0, 1])].assign(u=lambda z: z["u"] + 700)
    off["d"] = np.where(off["u"] == 700, off["t"] < 3, off["t"] < 4).astype(float)
    df = pd.concat([panel, off], ignore_index=True)
    names = []
    for j in range(6):
        df[f"c{j}"] = rng.normal(size=len(df))
        names.append(f"c{j}")
    with pytest.raises(DataInsufficient, match="controls=: too few not-yet-switched"):
        dyn(df, controls=names)


def test_dyn_continuous_with_controls_runs_without_slope_term(panel):
    res = dyn(panel, continuous=1, controls=["x"], dynamic=1)
    assert abs(res.estimate - 1.5) < 4 * res.se
    assert (es(res)["se"] > 0).all()


# ======================================================================
# normalized effects, their weights, and treatment paths
# ======================================================================
def test_dyn_normalized_weights_table(panel):
    res = dyn(panel, normalized=True, normalized_weights=True, dynamic=2, placebo=1)
    w = res.model_info["normalized_weights"]
    assert w.columns.tolist() == [1, 2, 3] and w.columns.name == "effect"
    assert w.index.name == "lag"
    # a binary treatment that stays on spreads effect l equally over l lags
    np.testing.assert_allclose(w[3], [1 / 3] * 3)
    np.testing.assert_allclose(w[2].dropna(), [0.5, 0.5])
    assert w[1].dropna().tolist() == [1.0]
    plain = dyn(panel, dynamic=2)
    np.testing.assert_allclose(
        es(res).loc[0:2, "att"].to_numpy(),
        es(plain)["att"].to_numpy() / np.array([1, 2, 3]),
    )


def test_dyn_normalized_effect_is_missing_when_the_dose_is(panel):
    # group 0 switches at t=3; blank its treatment at t=4: the cumulative
    # dose behind effect 1 is then unknown for one switcher
    hole = panel.copy()
    hole.loc[(hole["u"] == 0) & (hole["t"] == 4), "d"] = np.nan
    plain = es(dyn(hole, dynamic=2, placebo=1))
    norm = es(dyn(hole, dynamic=2, placebo=1, normalized=True))
    assert norm.loc[0, "att"] == pytest.approx(plain.loc[0, "att"])
    assert np.isfinite(norm.loc[[1, 2], "att"]).all()


def test_dyn_by_path_average_does_not_depend_on_aggregation_when_binary(panel):
    simple = dyn(panel, by_path=2, dynamic=2).model_info["by_path"]
    switch = dyn(panel, by_path=2, dynamic=2, aggregation="switchers")
    switch = switch.model_info["by_path"]
    assert len(simple) == 1 and simple[0]["path"] == (0.0, 1.0, 1.0, 1.0)
    assert simple[0]["n_switchers"] == 30
    tab = simple[0]["event_study"]
    assert simple[0]["estimate"] == pytest.approx(tab["att"].mean())
    assert switch[0]["estimate"] == pytest.approx(simple[0]["estimate"])
    assert switch[0]["se"] == pytest.approx(simple[0]["se"])
    assert simple[0]["joint_effects_test"]["df"] == 3


def test_dyn_by_path_leaves_out_groups_with_an_incomplete_path(panel):
    hole = panel.copy()
    hole.loc[(hole["u"] == 0) & (hole["t"] == 4), "d"] = np.nan
    paths = dyn(hole, by_path=3, dynamic=2).model_info["by_path"]
    assert paths[0]["path"] == (0.0, 1.0, 1.0, 1.0)
    assert paths[0]["n_switchers"] == 29
    # five effects need periods F-1 .. F+4: only the t=3 switchers have them
    long = dyn(panel, by_path=3, dynamic=4).model_info["by_path"]
    assert long[0]["n_switchers"] == 10 and len(long[0]["path"]) == 6


def test_dyn_cell_centre_rules():
    # arguments: (n_own, mean_own, n_pool, mean_pool)
    assert DYN._cell_centre(4, 2.0, 10, 9.0) == (2.0, pytest.approx(np.sqrt(4 / 3)))
    assert DYN._cell_centre(1, 2.0, 10, 9.0) == (9.0, pytest.approx(np.sqrt(10 / 9)))
    # a lone cluster with no pooled cell to borrow from stays uncentred
    assert DYN._cell_centre(1, 2.0, 1, 9.0) == (0.0, 1.0)
