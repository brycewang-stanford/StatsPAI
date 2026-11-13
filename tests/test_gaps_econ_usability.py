"""Messages and labels an applied economist asked for (2026-11 probe).

* ``sp.callaway_santanna`` with covariates says so when a propensity
  logit has fitted probabilities numerically 0 or 1 (R ``did`` passes on
  ``glm``'s warning for the same situation).
* ``sp.event_study(window=)`` says in its summary that the end points
  pool the periods beyond the window.
* ``sp.xtabond`` prints the z statistics it already computes.
* ``sp.spec_curve`` gives a reason for every specification it leaves out,
  and ``collinear='omit'`` estimates a specification without a redundant
  control instead of leaving it out.
* Native ``sp.feols`` / ``sp.fepois`` collinearity errors name the user's
  ``i()`` / ``C()`` terms, not internal column names.

Numbers against R are in ``tests/reference_parity/test_gaps_econ_parity.py``;
the tests here assert identities and message contents.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    MethodIncompatibility,
    NumericalInstability,
)

DATA = Path(__file__).resolve().parents[1] / "src" / "statspai" / "datasets" / "data"


@pytest.fixture(scope="module")
def castle():
    df = pd.read_csv(DATA / "castle_2013.csv")
    df["g"] = df["effyear"].fillna(0).astype(int)
    return df


@pytest.fixture(scope="module")
def card():
    return pd.read_csv(DATA / "card_1995.csv")


# ---------------------------------------------------------------------------
# callaway_santanna: fitted propensities numerically 0 or 1
# ---------------------------------------------------------------------------


def _separated_panel():
    """Two cohorts. In cohort 3 every treated unit has x above every
    never-treated unit (perfect separation by x); cohort 4 overlaps."""
    rng = np.random.default_rng(20261113)
    rows = []
    for unit in range(120):
        if unit < 6:
            g, x = 3, 5.0 + rng.uniform()
        elif unit < 46:
            g, x = 4, rng.normal()
        else:
            g, x = 0, rng.normal()
        alpha = rng.normal()
        for t in range(1, 6):
            y = alpha + 0.3 * t + 0.2 * x * t + rng.normal(scale=0.5)
            if g and t >= g:
                y += 1.0
            rows.append({"id": unit, "t": t, "g": g, "x": x, "y": y})
    return pd.DataFrame(rows)


def _boundary_warnings(caught):
    return [
        w
        for w in caught
        if issubclass(w.category, AssumptionWarning)
        and "numerically 0 or 1" in str(w.message)
    ]


@pytest.mark.parametrize("estimator", ["dr", "ipw"])
def test_cs_warns_when_a_cohort_is_separated_by_the_covariate(estimator):
    df = _separated_panel()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sp.callaway_santanna(
            df, y="y", g="g", t="t", i="id", x=["x"], estimator=estimator
        )
    (warning,) = _boundary_warnings(caught)
    cells = res.model_info["pscore_boundary_cells"]
    # every cell of the separated cohort, none of the overlapping one
    assert {g for g, _ in cells} == {3}
    n_cohort3 = int((res.detail["group"] == 3).sum())
    assert len(cells) == n_cohort3 == res.model_info["n_pscore_boundary_cells"]
    assert f"in {n_cohort3} of {len(res.detail)} ATT(g,t) cells" in str(warning.message)
    assert "g=3" in str(warning.message)


def test_cs_is_silent_without_separation_and_without_a_propensity_model():
    df = _separated_panel()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # outcome regression fits no propensity model
        reg = sp.callaway_santanna(
            df, y="y", g="g", t="t", i="id", x=["x"], estimator="reg"
        )
        # without the separated cohort the logits are interior
        ok = sp.callaway_santanna(
            df[df["g"] != 3], y="y", g="g", t="t", i="id", x=["x"], estimator="dr"
        )
        none = sp.callaway_santanna(df, y="y", g="g", t="t", i="id")
    assert not _boundary_warnings(caught)
    for res in (reg, ok, none):
        assert res.model_info["pscore_boundary_cells"] == []
        assert res.model_info["n_pscore_boundary_cells"] == 0


def test_cs_castle_covariates_flag_the_small_cohorts(castle):
    """Castle doctrine with four covariates: R ``did`` 2.5.1 reports
    "fitted probabilities numerically 0 or 1" on this call, StatsPAI was
    silent for the fits that still converged. Two covariates separate
    nothing (R is silent there too)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        two = sp.callaway_santanna(
            castle,
            y="l_homicide",
            g="g",
            t="year",
            i="sid",
            x=["l_police", "unemployrt"],
        )
    assert not _boundary_warnings(caught)
    assert two.model_info["n_pscore_boundary_cells"] == 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        four = sp.callaway_santanna(
            castle,
            y="l_homicide",
            g="g",
            t="year",
            i="sid",
            x=["l_police", "unemployrt", "l_income", "poverty"],
        )
    assert len(_boundary_warnings(caught)) == 1
    sizes = castle.groupby("g")["sid"].nunique()
    flagged = {g for g, _ in four.model_info["pscore_boundary_cells"]}
    # five regressors cannot be fitted to a cohort of one or two states
    assert {g for g in sizes.index if 0 < sizes[g] <= 2} <= flagged
    assert 2006 not in flagged  # 13 states


# ---------------------------------------------------------------------------
# event_study: binned end points
# ---------------------------------------------------------------------------


def test_event_study_summary_says_end_points_are_binned(castle):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.event_study(
            castle,
            y="l_homicide",
            treat_time="effyear",
            time="year",
            unit="sid",
            window=(-3, 3),
        )
    rel = (castle["year"] - castle["effyear"]).dropna()
    counts = res.model_info["binned_endpoints"]
    assert counts == {"lower": int((rel < -3).sum()), "upper": int((rel > 3).sum())}
    assert counts["lower"] > 0 and counts["upper"] > 0
    text = res.summary()
    assert "End points are binned (window=(-3, 3))" in text
    assert f"e = -3 also pools {counts['lower']} observations at e < -3" in text
    assert f"e = 3 also pools {counts['upper']} observations at e > 3" in text


def test_event_study_summary_is_silent_when_the_window_covers_the_panel(castle):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.event_study(
            castle,
            y="l_homicide",
            treat_time="effyear",
            time="year",
            unit="sid",
            window=(-9, 5),
        )
    assert res.model_info["binned_endpoints"] == {"lower": 0, "upper": 0}
    assert "End points are binned" not in res.summary()


def test_event_study_one_sided_binning_names_one_end(castle):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.event_study(
            castle,
            y="l_homicide",
            treat_time="effyear",
            time="year",
            unit="sid",
            window=(-9, 2),
        )
    text = res.summary()
    assert res.model_info["binned_endpoints"]["lower"] == 0
    assert "e = 2 also pools" in text and "e = -9 also pools" not in text


# ---------------------------------------------------------------------------
# xtabond: z column
# ---------------------------------------------------------------------------


def test_xtabond_summary_prints_z():
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(80):
        a = rng.normal()
        y = a
        for t in range(7):
            x = rng.normal()
            y = 0.5 * y + 0.3 * x + a + rng.normal()
            rows.append((unit, t, y, x))
    df = pd.DataFrame(rows, columns=["id", "time", "y", "x"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.xtabond(df, y="y", x=["x"])
    text = res.summary()
    header = next(
        line for line in text.splitlines() if line.split()[:1] == ["variable"]
    )
    assert header.split() == ["variable", "coefficient", "se", "z", "pvalue"]
    for _, row in res.detail.iterrows():
        assert row["z"] == pytest.approx(row["coefficient"] / row["se"], rel=1e-12)
        line = next(
            ln for ln in text.splitlines() if ln.split()[:1] == [row["variable"]]
        )
        assert float(line.split()[3]) == pytest.approx(row["z"], abs=5.1e-5)


# ---------------------------------------------------------------------------
# spec_curve: every left-out specification has a reason
# ---------------------------------------------------------------------------

SPEC = dict(
    y="lwage",
    x="educ",
    controls=[[], ["exper"], ["exper", "south"]],
    se_types=["nonrobust", "hc1"],
)


def _subsets(card):
    return {
        "all": None,
        "south": card["south"] == 1,
        "tiny": pd.Series(card.index < 8, index=card.index),
    }


def test_spec_curve_accounts_for_every_specification(card):
    with pytest.warns(AssumptionWarning, match="left out"):
        res = sp.spec_curve(card, subsets=_subsets(card), **SPEC)
    dropped = res.model_info["dropped_specs"]
    # 3 control sets x 2 SE types x 3 subsets
    assert res.n_specs + len(dropped) == 18
    tiny = [d for d in dropped if d["subset"] == "tiny"]
    assert len(tiny) == 6
    assert all("too few observations (n = 8" in d["reason"] for d in tiny)
    assert all(d["collinear"] == [] for d in tiny)
    south = [d for d in dropped if d["subset"] == "south"]
    assert len(south) == 2
    assert all(d["collinear"] == ["south"] for d in south)
    text = res.summary()
    assert f"Left out (not estimable): {len(dropped)}" in text
    assert "south: collinear with 'Intercept'" in text


def test_spec_curve_default_still_drops_the_collinear_cell(card):
    """``collinear='drop'`` is the default: the estimated cells and their
    numbers are those of the call without the option."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = sp.spec_curve(card, subsets=_subsets(card), **SPEC)
        explicit = sp.spec_curve(card, subsets=_subsets(card), collinear="drop", **SPEC)
    pd.testing.assert_frame_equal(default.results_df, explicit.results_df)
    assert "omitted_controls" not in default.results_df.columns
    assert default.n_specs == 10


def test_spec_curve_omit_estimates_the_cell_without_the_redundant_control(card):
    """With ``collinear='omit'`` the cell [exper, south | south == 1] is
    the regression on exper alone in that subsample, which is what Stata
    ``regress`` and fixest report after dropping the constant dummy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.spec_curve(card, subsets=_subsets(card), collinear="omit", **SPEC)
    df = res.results_df
    assert res.n_specs == 12
    for se_type in SPEC["se_types"]:
        pick = (df["subset"] == "south") & (df["se_type"] == se_type)
        cell = df[pick & (df["controls"] == "exper, south")].iloc[0]
        twin = df[pick & (df["controls"] == "exper")].iloc[0]
        assert cell["omitted_controls"] == "south"
        assert twin["omitted_controls"] == ""
        for col in ("estimate", "se", "pvalue", "nobs", "r_squared"):
            assert cell[col] == pytest.approx(twin[col], rel=1e-12)
    # the unestimable cells are still listed
    assert {d["subset"] for d in res.model_info["dropped_specs"]} == {"tiny"}


def test_spec_curve_omit_still_refuses_a_constant_x(card):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.spec_curve(
            card,
            y="lwage",
            x="south",
            controls=[[], ["exper"]],
            se_types=["nonrobust"],
            collinear="omit",
            subsets={"all": None, "south": card["south"] == 1},
        )
    dropped = res.model_info["dropped_specs"]
    assert len(dropped) == 2 and all(d["collinear"] == ["south"] for d in dropped)
    assert set(res.results_df["subset"]) == {"all"}


def test_spec_curve_collinear_option_is_validated(card):
    with pytest.raises(MethodIncompatibility, match="collinear="):
        sp.spec_curve(card, collinear="keep", **SPEC)


# ---------------------------------------------------------------------------
# native feols / fepois: collinearity errors in the user's terms
# ---------------------------------------------------------------------------

_INTERNAL = re.compile(r"__i\d+__")


def _refusal(fn, formula, data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NumericalInstability) as info:
            fn(formula, data, backend="native", **kw)
    return info.value


def test_feols_error_names_c_terms_absorbed_by_the_fixed_effects(castle):
    exc = _refusal(sp.feols, "l_homicide ~ post + C(sid) | sid", castle)
    levels = sorted(castle["sid"].unique())[1:]
    assert exc.diagnostics["absorbed"] == [f"C(sid)[T.{lev}]" for lev in levels]
    for text in (str(exc), exc.message, exc.recovery_hint, repr(exc.to_dict())):
        assert not _INTERNAL.search(text)
    assert "C(sid)[T." in exc.message


def test_feols_error_names_i_terms(castle):
    exc = _refusal(
        sp.feols, "l_homicide ~ post + i(year, ref=2000) | sid + year", castle
    )
    assert exc.diagnostics["absorbed"] == [f"year::{y}" for y in range(2001, 2011)]
    assert not _INTERNAL.search(str(exc))


def test_feols_error_names_mutually_collinear_c_terms(castle):
    exc = _refusal(
        sp.feols,
        "l_homicide ~ post + C(west) + C(south) + C(midwest) + C(northeast)",
        castle,
    )
    assert exc.diagnostics["collinear"] == [
        "C(west)[T.1]",
        "C(south)[T.1]",
        "C(midwest)[T.1]",
        "C(northeast)[T.1]",
    ]
    assert not _INTERNAL.search(str(exc))


def test_feols_fit_with_factor_terms_is_unaffected(castle):
    """The renaming touches refusals only: a design that is identified
    returns the coefficient names it always did."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.feols(
            "l_homicide ~ post + i(year, ref=2000) | sid", castle, backend="native"
        )
    assert list(fit.params.index) == ["post"] + [
        f"year::{y}" for y in range(2001, 2011)
    ]
