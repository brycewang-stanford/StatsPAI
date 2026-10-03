"""The placebo and leave-one-out reports of ``sp.synth(method='classic')``
against Stata ``synth2`` (Yan and Chen), which wraps ``synth``.

Fixture: ``_fixtures/textbook_rcm.csv`` (generated data), run through
``_fixtures/_generate_synth2_stata.do`` in Stata 18 with ``synth2`` 2.1.0 and
``synth`` 0.0.8. The specification uses the regression-based predictor
weights (``synth`` without ``nested``), which are deterministic; the nested
search is a non-convex problem on which two optimisers need not agree.

Two things in the Stata output are not the quantities StatsPAI reports.

* ``synth`` stores the donor weights rounded to three decimals and builds
  ``e(Y_synthetic)`` from the rounded weights, so every path, effect and
  post-treatment MSPE ``synth2`` prints carries that rounding. The tests
  rebuild Stata's numbers by rounding our weights the same way; the rounded
  reconstruction agrees to the precision of Stata's ``float`` storage, which
  shows the two solve the same problems. The pre-treatment MSPE comes from
  ``e(RMSPE)``, which ``synth`` computes from the unrounded weights, and
  agrees directly.
* ``synth2`` divides the pre-treatment sum of squared gaps by the variation
  of the *synthetic* path to get its R-squared. ``pre_r2`` here uses the
  variation of the treated unit's outcome. The test reproduces Stata's
  number from its own formula.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"
SPEC = [("y", t, "mean") for t in (2, 5, 8, 11, 14, 17)] + [
    ("y", list(range(1, 16)), "mean"),
    ("y", list(range(16, 31)), "mean"),
]
UNITS = [u for u in range(1, 13) if u != 6]
FLOAT = 2e-6  # Stata keeps the paths in float variables


@pytest.fixture(scope="module")
def G():
    table = pd.read_csv(FIX / "textbook_synth2_Stata.csv", dtype={"key": str})
    return dict(zip(table["key"], table["value"]))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "textbook_rcm.csv")


@pytest.fixture(scope="module")
def wide(data):
    return data.pivot(index="t", columns="unit", values="y")


def fit_with(data, **kw):
    kw.setdefault("v_method", "regression")
    return sp.synth(
        data, "y", "unit", "t", treated_unit=6, treatment_time=31,
        method="classic", special_predictors=SPEC, **kw,
    )  # fmt: skip


@pytest.fixture(scope="module")
def fit(data):
    return fit_with(data, placebo=True, placebo_cutoff=1.5, placebo_time=25, loo=True)


def rounded_path(wide, weights):
    """Synthetic path from weights rounded as Stata's synth stores them."""
    w = weights.dropna().round(3)
    return wide[w.index].to_numpy() @ w.to_numpy()


def test_fit_and_weights(G, fit):
    info = fit.model_info
    gaps = info["gap_table"]
    pre = gaps.loc[~gaps["post_treatment"], "gap"]
    assert np.sqrt((pre**2).mean()) == pytest.approx(G["main.rmse"], rel=1e-9)
    weights = dict(zip(info["weights"]["unit"], info["weights"]["weight"]))
    stata = {int(k.split(".")[1]): v for k, v in G.items() if k.startswith("weight.")}
    assert {u for u, w in weights.items() if w >= 5e-4} == set(stata)
    for unit, w in stata.items():
        assert round(weights[unit], 3) == pytest.approx(w, abs=1e-12)


def test_stata_path_is_the_rounded_weight_path(G, fit, wide):
    info = fit.model_info
    w = info["weights"].set_index("unit")["weight"]
    path = rounded_path(wide, w)
    stata = np.array([G[f"path.main.{t}"] for t in wide.index])
    np.testing.assert_allclose(path, stata, rtol=FLOAT, atol=FLOAT)
    post = wide.index >= 31
    att = float((wide[6].to_numpy() - path)[post].mean())
    assert att == pytest.approx(G["main.att"], rel=FLOAT)
    # the exact weights give a slightly different average
    assert fit.estimate == pytest.approx(G["main.att"], abs=5e-3)


def test_r2_definitions(G, fit, wide):
    info = fit.model_info
    w = info["weights"].set_index("unit")["weight"]
    path = rounded_path(wide, w)
    pre = wide.index < 31
    gap = (wide[6].to_numpy() - path)[pre]
    theirs = 1 - (gap @ gap) / ((path[pre] - path[pre].mean()) ** 2).sum()
    assert theirs == pytest.approx(G["main.r2"], rel=FLOAT)
    actual = wide[6].to_numpy()[pre]
    exact = info["gap_table"].loc[pre, "gap"].to_numpy()
    ours = 1 - (exact @ exact) / ((actual - actual.mean()) ** 2).sum()
    assert info["pre_r2"] == pytest.approx(ours, rel=1e-12)


def test_placebo_unit_table(G, fit, wide):
    info = fit.model_info
    table = info["placebo_table"]
    assert list(table.index) == [6] + UNITS
    for unit in table.index:
        # pre-treatment MSPE: from the unrounded weights on both sides
        assert table.loc[unit, "pre_mspe"] == pytest.approx(
            G[f"mspe.pre.{unit}"], rel=1e-8
        )
        assert table.loc[unit, "pre_mspe_relative"] == pytest.approx(
            G[f"mspe.relative.{unit}"], rel=1e-8
        )
    post = wide.index >= 31
    for unit in UNITS:
        path = rounded_path(wide, info["placebo_weights"].loc[unit])
        post_mspe = float(((wide[unit].to_numpy() - path)[post] ** 2).mean())
        assert post_mspe == pytest.approx(G[f"mspe.post.{unit}"], rel=5e-6)
        # the exact weights move it by the rounding only
        assert table.loc[unit, "post_mspe"] == pytest.approx(
            G[f"mspe.post.{unit}"], rel=2e-2
        )
    assert table["ratio"].to_numpy() == pytest.approx(
        (table["post_mspe"] / table["pre_mspe"]).to_numpy()
    )


def test_placebo_p_values(G, fit):
    info = fit.model_info
    assert sorted(info["placebo_excluded"]) == [2, 3, 4, 10, 12]
    effects = info["placebo_effects"].set_index("time")
    for t in range(31, 41):
        assert effects.loc[t, "effect"] == pytest.approx(G[f"effect.{t}"], abs=5e-3)
        for ours, theirs in (
            ("p_two_sided", "p_two"),
            ("p_right", "p_right"),
            ("p_left", "p_left"),
        ):
            assert effects.loc[t, ours] == pytest.approx(G[f"{theirs}.{t}"], abs=1e-12)
    # rank p-value of the post/pre ratio: treated unit counted, as ADH do
    ratios = info["placebo_table"]["ratio"].to_numpy()
    assert info["placebo_pvalue"] == pytest.approx(np.mean(ratios >= ratios[0]))
    assert fit.pvalue == pytest.approx(info["placebo_pvalue"])
    kept = [u for u in UNITS if u not in info["placebo_excluded"]]
    kept_ratios = info["placebo_table"].loc[[6] + kept, "ratio"].to_numpy()
    assert info["placebo_pvalue_cutoff"] == pytest.approx(
        np.mean(kept_ratios >= kept_ratios[0])
    )


def test_placebo_in_time(G, fit, wide):
    info = fit.model_info
    table = info["placebo_time"]
    assert list(table["time"]) == list(range(25, 41))
    w = info["placebo_time_weights"].set_index("unit")["weight"]
    path = rounded_path(wide, w)
    stata = np.array([G[f"path.time25.{t}"] for t in wide.index])
    np.testing.assert_allclose(path, stata, rtol=FLOAT, atol=FLOAT)
    shown = wide.index >= 25
    np.testing.assert_allclose(table["synthetic"], stata[shown], atol=5e-3)


def test_leave_one_out(G, fit, wide):
    info = fit.model_info
    assert info["loo_units"] == [9, 2, 10, 1]
    paths = np.column_stack(
        [rounded_path(wide, info["loo_weights"][u]) for u in info["loo_units"]]
    )
    low = np.array([G[f"path.loomin.{t}"] for t in wide.index])
    high = np.array([G[f"path.loomax.{t}"] for t in wide.index])
    np.testing.assert_allclose(paths.min(axis=1), low, rtol=FLOAT, atol=FLOAT)
    np.testing.assert_allclose(paths.max(axis=1), high, rtol=FLOAT, atol=FLOAT)
    loo = info["loo"]
    np.testing.assert_allclose(loo["synthetic_min"], low, atol=5e-3)
    np.testing.assert_allclose(loo["synthetic_max"], high, atol=5e-3)
    np.testing.assert_allclose(loo["effect_min"], loo["actual"] - loo["synthetic_max"])
    assert (loo["synthetic_min"] <= loo["synthetic_max"]).all()


def test_post_periods(G, data, wide):
    fit = fit_with(data, placebo=True, post_periods=[31, 32, 33, 34, 35])
    info = fit.model_info
    assert info["n_post_periods"] == 5
    w = info["weights"].set_index("unit")["weight"]
    path = rounded_path(wide, w)
    post = (wide.index >= 31) & (wide.index <= 35)
    gap = (wide[6].to_numpy() - path)[post]
    assert gap.mean() == pytest.approx(G["post.att"], rel=FLOAT)
    assert (gap**2).mean() == pytest.approx(G["post.mspe_post_treated"], rel=5e-6)
    assert info["placebo_table"].loc[6, "ratio"] == pytest.approx(
        G["post.ratio_treated"], rel=2e-2
    )


def test_placebo_on_a_list_of_units(G, data):
    chosen = [1, 2, 3, 4, 5, 7, 8]
    fit = fit_with(data, placebo=True, placebo_units=chosen, placebo_cutoff=1.5)
    info = fit.model_info
    table = info["placebo_table"]
    assert list(table.index) == [6] + chosen
    assert len(table) == G["some.rows"]
    assert table["pre_mspe"].iloc[-1] == pytest.approx(G["some.pre_last"], rel=1e-8)
    effects = info["placebo_effects"].reset_index(drop=True)
    for i in range(10):
        assert effects.loc[i, "p_two_sided"] == pytest.approx(
            G[f"some.p_two.{i + 1}"], abs=1e-12
        )
        assert effects.loc[i, "p_right"] == pytest.approx(
            G[f"some.p_right.{i + 1}"], abs=1e-12
        )
    # the headline p-value is over the same units
    assert fit.pvalue == pytest.approx(
        np.mean(table["ratio"] >= table["ratio"].iloc[0])
    )


def test_pre_periods(G, data):
    spec = [("y", t, "mean") for t in (12, 15, 18, 21, 24, 27)] + [
        ("y", list(range(11, 21)), "mean"),
        ("y", list(range(21, 31)), "mean"),
    ]
    fit = sp.synth(
        data, "y", "unit", "t", treated_unit=6, treatment_time=31,
        method="classic", special_predictors=spec, v_method="regression",
        placebo=False, pre_periods=list(range(11, 31)),
    )  # fmt: skip
    info = fit.model_info
    assert info["n_pre_periods"] == G["short.T0"] == 20
    gaps = info["gap_table"]
    pre = gaps.loc[~gaps["post_treatment"], "gap"]
    assert np.sqrt((pre**2).mean()) == pytest.approx(G["short.rmse"], rel=1e-8)
    weights = dict(zip(info["weights"]["unit"], info["weights"]["weight"]))
    stata = {
        int(k.split(".")[2]): v for k, v in G.items() if k.startswith("short.weight.")
    }
    for unit, w in stata.items():
        assert round(weights[unit], 3) == pytest.approx(w, abs=1e-12)


# ------------------------------------------------------------------ edges
def test_options_that_cannot_be_honoured_raise(data):
    with pytest.raises(MethodIncompatibility, match="placebo=True"):
        fit_with(data, placebo=False, placebo_cutoff=2)
    with pytest.raises(MethodIncompatibility, match="at least 1"):
        fit_with(data, placebo=True, placebo_cutoff=0.5)
    with pytest.raises(MethodIncompatibility, match="before treatment_time"):
        fit_with(data, placebo=False, placebo_time=35)
    with pytest.raises(MethodIncompatibility, match="post_periods"):
        fit_with(data, placebo=False, post_periods=[20])
    with pytest.raises(MethodIncompatibility, match="pre_periods"):
        fit_with(data, placebo=False, pre_periods=[35])
    with pytest.raises(MethodIncompatibility, match="placebo_units"):
        fit_with(data, placebo=True, placebo_units=[6])
    with pytest.raises(MethodIncompatibility, match="placebo=True"):
        fit_with(data, placebo=False, placebo_units=[2])


# ------------------------------------------ rcm with covariates (Hsiao-Zhou)
# Stata `rcm y x`: the covariate of every unit joins the control units'
# outcomes as a candidate predictor. Deterministic, so compared directly.
@pytest.fixture(scope="module")
def data_x():
    return pd.read_csv(FIX / "textbook_rcm_x.csv")


def rcm_x(data_x, **kw):
    return sp.synth(
        data_x, "y", "unit", "t", treated_unit=1, treatment_time=31,
        method="rcm", covariates=["x"], **kw,
    )  # fmt: skip


@pytest.mark.parametrize("selection", ["forward", "best"])
def test_rcm_covariates_selection_path(G, data_x, selection):
    fit = rcm_x(data_x, selection=selection)
    info = fit.model_info
    key = f"rcmx.{selection}"
    assert info["n_candidates"] == G[f"{key}.K_all"] == 23
    assert info["n_selected"] == G[f"{key}.K"]
    assert fit.estimate == pytest.approx(G[f"{key}.att"], rel=1e-7)
    assert info["pre_rmse"] == pytest.approx(G[f"{key}.rmse"], rel=1e-7)
    assert info["pre_r2"] == pytest.approx(G[f"{key}.r2"], rel=1e-7)
    table = info["selection_table"]
    for k in range(1, 24):
        # the path enters covariates from the third step on
        assert table.loc[k, "aic"] == pytest.approx(G[f"{key}.aic.{k}"], rel=1e-6)
        assert table.loc[k, "bic"] == pytest.approx(G[f"{key}.bic.{k}"], rel=1e-6)
        assert table.loc[k, "r2"] == pytest.approx(G[f"{key}.r2.{k}"], rel=1e-6)


def test_rcm_covariates_placebos(G, data_x):
    fit = rcm_x(
        data_x, selection="forward", criterion="bic",
        placebo=True, placebo_cutoff=2, placebo_time=28,
    )  # fmt: skip
    info = fit.model_info
    assert fit.estimate == pytest.approx(G["rcmx.placebo.att"], rel=1e-7)
    stata_order = sorted(range(2, 13), key=str)  # Stata sorts the labels
    table = info["placebo_table"]
    for i, unit in enumerate([1] + stata_order, start=1):
        assert table.loc[unit, "pre_mspe"] == pytest.approx(
            G[f"rcmx.placebo.pre.{i}"], rel=1e-6
        )
        assert table.loc[unit, "post_mspe"] == pytest.approx(
            G[f"rcmx.placebo.post.{i}"], rel=1e-6
        )
    post = fit.detail[fit.detail["post"]].reset_index(drop=True)
    for i in range(10):
        assert post.loc[i, "p_two_sided"] == pytest.approx(
            G[f"rcmx.placebo.p_two.{i + 1}"], abs=1e-6
        )
        assert post.loc[i, "p_right"] == pytest.approx(
            G[f"rcmx.placebo.p_right.{i + 1}"], abs=1e-6
        )
    fake = info["placebo_time"]
    assert len(info["placebo_time_selected"]) == G["rcmx.time28.K"]
    assert fake["effect"].mean() == pytest.approx(G["rcmx.time28.att"], rel=1e-6)


def test_rcm_placebo_on_a_list_of_units(G, data_x):
    fit = rcm_x(
        data_x, selection="forward", criterion="bic",
        placebo=True, placebo_cutoff=2, placebo_units=[2, 3, 4, 5],
    )  # fmt: skip
    assert len(fit.model_info["placebo_table"]) == G["rcmx.some.rows"] == 5
    post = fit.detail[fit.detail["post"]].reset_index(drop=True)
    for i in range(10):
        assert post.loc[i, "p_two_sided"] == pytest.approx(
            G[f"rcmx.some.p_two.{i + 1}"], abs=1e-6
        )
        assert post.loc[i, "p_left"] == pytest.approx(
            G[f"rcmx.some.p_left.{i + 1}"], abs=1e-6
        )


def test_rcm_does_not_select_the_saturated_model(G, data_x):
    """24 pre-treatment periods, 23 candidates. Stata's rcm keeps the model
    with all 23: it has as many coefficients as observations, an R-squared
    of 1, and a BIC that is the logarithm of rounding error (-589). That is
    interpolation, not a fit, so StatsPAI stops at one residual degree of
    freedom. Up to that size the two tables agree."""
    assert G["rcmx.saturated.K"] == 23
    assert G["rcmx.saturated.r2"] == pytest.approx(1.0, abs=1e-12)
    assert G["rcmx.saturated.bic.23"] < -500
    fit = sp.synth(
        data_x, "y", "unit", "t", treated_unit=1, treatment_time=25,
        method="rcm", covariates=["x"], selection="forward", criterion="bic",
    )  # fmt: skip
    table = fit.model_info["selection_table"]
    assert table.index.max() == 22
    for k in (20, 22):
        assert table.loc[k, "bic"] == pytest.approx(
            G[f"rcmx.saturated.bic.{k}"], rel=1e-6
        )
    assert fit.model_info["n_selected"] == 20


def test_rcm_covariate_errors(data_x):
    with pytest.raises(MethodIncompatibility, match="not a column"):
        rcm_x(data_x.drop(columns="x"))
    holes = data_x.copy()
    holes.loc[(holes.unit == 4) & (holes.t == 7), "x"] = np.nan
    with pytest.raises(Exception, match="missing values"):
        rcm_x(holes, selection="forward")
