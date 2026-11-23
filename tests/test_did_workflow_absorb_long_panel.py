"""``sp.did_workflow`` on a long, multi-level panel.

The design is that of county-by-species-by-year studies of a staggered
local treatment: many periods, treatment assigned at a level above the
unit, and shocks common to a region in a year. Four behaviours are pinned.

* ``absorb=`` adds fixed effects beyond unit and period. The workflow adds
  no estimator, so its numbers are those of ``sp.did_imputation(fe=...)``,
  bit for bit, and on noiseless data with region-by-year shocks the
  absorbed fit returns the true effect while the two-way fit does not.
* With no ``event_window`` a long panel is reported over the five event
  times nearest to treatment on each side; a short panel and an explicit
  window are left alone.
* The staggered-adoption guardrail of ``sp.feols`` reads the calendar
  through a group-by-period fixed effect.
* The pre-test power integral uses SciPy's default tolerance above five
  pre-periods (1e-7 is out of reach there) and still agrees with a
  tighter integral to that tolerance.
"""

from __future__ import annotations

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

TAU = -0.4


def _panel(noise: float, seed: int = 0) -> pd.DataFrame:
    """48 units in 4 regions over 20 years; later regions adopt later and
    carry their own year shocks, so adoption timing is confounded by
    region-by-year effects and by nothing else."""
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2020)
    shock = rng.normal(0.0, 1.0, (4, len(years)))
    rows = []
    for unit in range(48):
        region = unit % 4
        # units 0-23 never treated; the rest adopt in a region-specific year
        first = 0 if unit < 24 else 2006 + 3 * region
        level = rng.normal()
        for k, year in enumerate(years):
            d = int(first > 0 and year >= first)
            y = level + shock[region, k] + TAU * d + noise * rng.normal()
            rows.append((unit, region, year, first, d, y))
    df = pd.DataFrame(rows, columns=["unit", "region", "year", "first", "d", "y"])
    df["region_year"] = df["region"] * 10000 + df["year"]
    return df


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


KW = dict(y="y", id="unit", time="year", cohort="first", cluster="unit")


# --------------------------------------------------------------------- #
# absorb=
# --------------------------------------------------------------------- #
def test_absorb_recovers_the_effect_that_two_way_effects_miss():
    df = _panel(noise=0.0)
    wf = _quiet(sp.did_workflow, df, absorb=["region#year"], through="estimate", **KW)
    assert wf.estimate == pytest.approx(TAU, abs=1e-8)
    plain = _quiet(sp.did_workflow, df, estimator="bjs", through="estimate", **KW)
    # region-by-year shocks do not difference out against other regions
    assert abs(plain.estimate - TAU) > 1e-3


def test_absorb_is_the_imputation_estimator_with_those_fixed_effects():
    df = _panel(noise=0.5)
    wf = _quiet(
        sp.did_workflow,
        df,
        absorb=["region_year"],
        event_window=(-3, 3),
        skip=["sensitivity"],
        **KW,
    )
    direct = _quiet(
        sp.did_imputation,
        df,
        y="y",
        group="unit",
        time="year",
        first_treat="first",
        cluster="unit",
        fe=["unit", "year", "region_year"],
        horizon=list(range(-3, 4)),
        pretrend_method="in-sample",
    )
    assert (wf.estimate, wf.se) == (direct.estimate, direct.se)
    table = wf.robustness
    alt = table[table["check"] == "fixed effects"].iloc[0]
    two_way = _quiet(
        sp.did_imputation,
        df,
        y="y",
        group="unit",
        time="year",
        first_treat="first",
        cluster="unit",
        horizon=list(range(-3, 4)),
    )
    assert (alt["estimate"], alt["se"]) == (two_way.estimate, two_way.se)
    # the other estimators cannot absorb the cells and are not run
    skipped = table[table["check"] == "estimator"].iloc[0]
    assert skipped["variant"] == "not run" and np.isnan(skipped["estimate"])


def test_absorb_is_declared_and_selects_the_estimator():
    df = _panel(noise=0.5)
    wf = _quiet(sp.did_workflow, df, absorb="region#year", through="declare", **KW)
    choices = wf.step("declare").evidence["choices"].set_index("choice")
    assert choices.loc["absorb", "value"] == "region#year"
    assert choices.loc["absorb", "source"] == "declared"
    assert choices.loc["estimator", "value"] == "bjs"
    assert choices.loc["estimator", "source"] == "implied by absorb"
    none = _quiet(sp.did_workflow, df, through="declare", **KW)
    assert (
        none.step("declare")
        .evidence["choices"]
        .set_index("choice")
        .loc["absorb", "value"]
        == "none"
    )


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(absorb=["region#year"], estimator="cs"), "takes no fixed effects"),
        (dict(absorb=["region#year"], estimator="sa"), "takes no fixed effects"),
        (dict(absorb=["no_such_column"]), "not in the data"),
        (dict(absorb=["region#nope"]), "not in the data"),
    ],
)
def test_absorb_refuses_what_it_cannot_honour(kw, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.did_workflow(_panel(noise=0.5), through="declare", **KW, **kw)


# --------------------------------------------------------------------- #
# default event window
# --------------------------------------------------------------------- #
def test_long_panel_is_reported_over_five_event_times_a_side():
    df = _panel(noise=0.5)
    wf = _quiet(sp.did_workflow, df, through="event_study", **KW)
    note = wf.step("design").evidence["event_window_default"]
    # cohorts adopt in 2006..2015 of 2000..2019: up to 15 leads, 13 lags
    assert note == {"window": (-5, 5), "available": (-15, 13)}
    times = wf.event_study["relative_time"]
    assert (times.min(), times.max()) == (-5, 5)
    assert "steps 5 to 7 report -5 to 5" in wf.step("design").finding


def test_explicit_window_and_short_panels_are_left_alone():
    df = _panel(noise=0.5)
    wide = _quiet(
        sp.did_workflow, df, event_window=(-8, 9), through="event_study", **KW
    )
    assert "event_window_default" not in wide.step("design").evidence
    times = wide.event_study["relative_time"]
    assert (times.min(), times.max()) == (-8, 9)

    # 2009 cohort: 5 leads; 2006 cohort: lags 0..5. Nothing to cut.
    short = df[df["year"].between(2004, 2011)]
    short = short[short["first"].isin([0, 2006, 2009])]
    wf = _quiet(sp.did_workflow, short, through="event_study", **KW)
    assert "event_window_default" not in wf.step("design").evidence
    times = wf.event_study["relative_time"]
    assert (times.min(), times.max()) == (-5, 5)


# --------------------------------------------------------------------- #
# guardrail: calendar nested in a group-by-period effect
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "formula", ["y ~ d | unit + region_year", "y ~ d | unit + region^year"]
)
def test_feols_guardrail_counts_adoption_dates_through_region_by_year(formula):
    df = _panel(noise=0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.feols(formula, df, vcov={"CRV1": "unit"})
    said = [str(w.message) for w in caught if "switches on at" in str(w.message)]
    assert said, [str(w.message)[:80] for w in caught]
    # four regions adopt in four different years, not in region x year cells
    assert "switches on at 4 different dates" in said[0]
    bacon = _quiet(sp.bacon_decomposition, df, y="y", treat="d", time="year", id="unit")
    share = float(bacon["already_treated_control_weight_share"])
    assert f"{100 * share:.1f}%" in said[0]


# --------------------------------------------------------------------- #
# pre-test power integral
# --------------------------------------------------------------------- #
def test_rectangle_probability_tolerance_by_dimension():
    pre = importlib.import_module("statspai.did.pretrends")
    assert pre._rectangle_tol(5) == 1e-7
    assert pre._rectangle_tol(6) == 1e-5
    rng = np.random.default_rng(0)
    k = 8
    a = rng.normal(size=(k, k))
    cov = a @ a.T / k + 0.5 * np.eye(k)
    sd = np.sqrt(np.diag(cov))
    mean = 0.3 * sd
    got = pre._normal_rectangle_probability(mean, cov, -1.96 * sd, 1.96 * sd)
    tight = float(
        stats.multivariate_normal.cdf(
            1.96 * sd,
            mean=mean,
            cov=cov,
            lower_limit=-1.96 * sd,
            abseps=1e-7,
            releps=1e-7,
        )
    )
    # the integrator's stated accuracy at the looser setting
    assert got == pytest.approx(tight, abs=5e-5)


def test_etwfe_builds_its_design_without_fragmenting_the_frame():
    df = _panel(noise=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        fit = sp.etwfe(
            df, y="y", group="unit", time="year", first_treat="first", cluster="unit"
        )
    assert np.isfinite(fit.estimate)


# --------------------------------------------------------------------- #
# placebo dates under the staggered estimators' never-treated coding
# --------------------------------------------------------------------- #
def test_placebo_dates_read_zero_and_infinity_as_never_treated():
    df = _panel(noise=0.5)
    as_nan = df.assign(first=df["first"].replace(0, np.nan))
    as_inf = df.assign(first=df["first"].replace(0, np.inf))
    kw = dict(id="unit", event_time="first", time="year", seed=0)
    ref = sp.assign_placebo_dates(as_nan, **kw)
    for coded in (df, as_inf):
        got = sp.assign_placebo_dates(coded, **kw)
        pd.testing.assert_series_equal(
            got["placebo_event_time"], ref["placebo_event_time"]
        )
        pd.testing.assert_series_equal(got["placebo"], ref["placebo"])
    per_unit = ref.drop_duplicates("unit")
    # the 24 never-treated units get a date; none of them is "period 0"
    assert int(per_unit["placebo"].sum()) == 24
    assert set(per_unit["placebo_event_time"]) <= {2006.0, 2009.0, 2012.0, 2015.0}


def test_placebo_dates_keep_zero_when_zero_is_a_period():
    df = _panel(noise=0.5)
    shifted = df.assign(year=df["year"] - 2006, first=df["first"].replace(0, np.nan))
    shifted["first"] = shifted["first"] - 2006  # first cohort adopts in period 0
    out = sp.assign_placebo_dates(
        shifted, id="unit", event_time="first", time="year", seed=0
    )
    per_unit = out.drop_duplicates("unit")
    treated = per_unit[~per_unit["placebo"]]
    assert (treated["placebo_event_time"] == 0).sum() == 6
