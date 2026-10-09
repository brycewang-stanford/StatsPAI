"""Coverage gaps in ``sp.stacked_did``: argument contracts, cohorts that
cannot form a sub-experiment, the pre-built-stack entry, weights and
incomplete rows.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

KW = dict(y="y", group="unit", time="time", first_treat="first_treat", window=(-2, 2))


@pytest.fixture(scope="module")
def panel():
    return sp.dgp_did(n_units=60, n_periods=8, staggered=True, seed=3)


@pytest.fixture(scope="module")
def base(panel):
    return sp.stacked_did(panel, **KW)


@pytest.fixture(scope="module")
def stack():
    """Three sub-experiments, 4 treated and 8 control units each, effect 1."""
    rng = np.random.default_rng(1)
    rows = []
    for e in range(3):
        for u in range(12):
            tr = int(u < 4)
            for k in range(-3, 3):
                rows.append(
                    {
                        "ev": e,
                        "u": f"{e}-{u}",
                        "tr": tr,
                        "k": k,
                        "per": k + 10 * e,
                        "y": rng.normal() + tr * (k >= 0) * 1.0,
                    }
                )
    return pd.DataFrame(rows)


PK = dict(
    y="y",
    group="u",
    time="per",
    event_id="ev",
    treated="tr",
    event_time="k",
    window=(-3, 2),
)


# ---------------------------------------------------------------------------
#  Argument contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"family": "binomial"}, "family must be 'gaussian' or 'poisson'"),
        ({"spec": "dynamic"}, "spec must be 'event_study' or 'pooled'"),
        ({"control_group": "all"}, "control_group must be 'nevertreated'"),
        ({"weights": "nope"}, "Weight column 'nope' not found"),
        ({"cluster": "nope"}, "Cluster column 'nope' not found"),
    ],
)
def test_bad_arguments_are_named(panel, extra, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.stacked_did(panel, **KW, **extra)


def test_negative_weights_are_refused(panel):
    with pytest.raises(MethodIncompatibility, match="finite and non-negative"):
        sp.stacked_did(panel.assign(w=-1.0), weights="w", **KW)


def test_events_argument_contracts(panel):
    kw = dict(y="y", group="unit", time="time", window=(-2, 2))
    df = panel.assign(ev=(panel["time"] == panel["first_treat"]).astype(int))
    with pytest.raises(MethodIncompatibility, match="own_overlap must be"):
        sp.stacked_did(df, events="ev", own_overlap="maybe", **kw)
    with pytest.raises(MethodIncompatibility, match="Column 'nope' not found"):
        sp.stacked_did(df, events="nope", **kw)


def test_events_all_dropped_for_overlap(panel):
    # The only unit with events has two of them one period apart, so each
    # sits inside the other's window and own_overlap='drop' leaves nothing.
    kw = dict(y="y", group="unit", time="time", window=(-2, 2))
    df = panel.assign(
        ev=((panel["unit"] == 0) & panel["time"].isin([3, 4])).astype(int)
    )
    with pytest.raises(DataInsufficient, match="no event has a usable window"):
        sp.stacked_did(df, events="ev", **kw)
    kept = sp.stacked_did(df, events="ev", own_overlap="keep", **kw)
    assert kept.model_info["n_events_dropped_overlap"] == 0
    assert kept.model_info["n_cohorts"] == 2


# ---------------------------------------------------------------------------
#  Cohorts that cannot form a sub-experiment
# ---------------------------------------------------------------------------


def test_no_controls_for_any_cohort(panel):
    treated_only = panel[panel["first_treat"].notna()]
    with pytest.raises(ValueError, match="No valid sub-experiments"):
        sp.stacked_did(treated_only, **KW)


def test_last_cohort_without_not_yet_treated_rows_is_skipped(panel):
    treated_only = panel[panel["first_treat"].notna()]
    res = sp.stacked_did(treated_only, control_group="notyettreated_rows", **KW)
    # Four cohorts, but nobody is still untreated when the last one adopts.
    assert len(res.model_info["cohorts"]) == res.model_info["n_cohorts"] == 3
    last = treated_only["first_treat"].max()
    assert res.model_info["skipped_cohorts"] == [last]
    assert np.isfinite(res.estimate) and res.se > 0


def test_cohort_whose_window_misses_the_panel_is_skipped(panel, base):
    # Unit u adopts long after the panel ends: its cohort has no row in the
    # window, and under never-treated controls it is not a control either,
    # so the fit is the one without that unit.
    u = panel["unit"].iloc[0]
    late = panel.copy()
    late.loc[late["unit"] == u, "first_treat"] = 100
    with_late = sp.stacked_did(late, **KW)
    without = sp.stacked_did(panel[panel["unit"] != u], **KW)
    assert with_late.estimate == pytest.approx(without.estimate, abs=1e-12)
    assert with_late.se == pytest.approx(without.se, abs=1e-12)
    assert with_late.model_info["n_cohorts"] == without.model_info["n_cohorts"]


# ---------------------------------------------------------------------------
#  Pre-built stack
# ---------------------------------------------------------------------------


def test_prebuilt_stack_contracts(stack):
    with pytest.raises(MethodIncompatibility, match="Column 'zz' not found"):
        sp.stacked_did(stack, **{**PK, "treated": "zz"})
    with pytest.raises(MethodIncompatibility, match="Column 'tr' must be 0/1"):
        sp.stacked_did(stack.assign(tr=stack["tr"] * 2), **PK)
    with pytest.raises(DataInsufficient, match="fall inside window"):
        sp.stacked_did(stack.assign(k=stack["k"] + 50), **PK)


def test_prebuilt_stack_without_post_periods(stack):
    with pytest.raises(DataInsufficient, match="no post-treatment") as err:
        sp.stacked_did(stack[stack["k"] < 0], **PK)
    assert err.value.diagnostics == {"window": [-3, 2]}
    with pytest.raises(ValueError, match="Not enough relative time periods"):
        sp.stacked_did(stack[stack["k"] == -1], **PK)


def test_prebuilt_stack_recovers_the_effect(stack):
    res = sp.stacked_did(stack, **PK)
    assert res.model_info["prebuilt_stack"] is True
    assert res.model_info["n_cohorts"] == 3
    assert abs(res.estimate - 1.0) < 3 * res.se


# ---------------------------------------------------------------------------
#  Weights and incomplete rows
# ---------------------------------------------------------------------------


def test_missing_outcome_rows_are_dropped_with_their_weights(panel, base):
    df = panel.assign(w=1.0)
    df.loc[df.index[5], "y"] = np.nan
    res = sp.stacked_did(df, weights="w", **KW)
    ref = sp.stacked_did(df.dropna(subset=["y"]), **KW)
    assert res.estimate == pytest.approx(ref.estimate, abs=1e-10)
    assert res.se == pytest.approx(ref.se, abs=1e-10)
    # The dropped row was in the stack: the fit moved.
    assert abs(res.estimate - base.estimate) > 1e-6
    assert res.model_info["n_stacked_obs"] < base.model_info["n_stacked_obs"]


def test_poisson_with_unit_weights_is_the_unweighted_fit(panel):
    rng = np.random.default_rng(0)
    df = panel.assign(w=1.0)
    df["c"] = rng.poisson(np.exp(0.2 * df["y"].clip(-3, 3)) + 1.0)
    kw = {**KW, "y": "c", "family": "poisson"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = sp.stacked_did(df, weights="w", **kw)
        plain = sp.stacked_did(df, **kw)
    assert weighted.model_info["family"] == "poisson"
    assert weighted.estimate == pytest.approx(plain.estimate, abs=1e-9)
    assert weighted.se == pytest.approx(plain.se, abs=1e-9)
