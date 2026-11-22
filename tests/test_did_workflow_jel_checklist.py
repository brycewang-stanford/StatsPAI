"""``sp.did_workflow`` against the checklist of Baker et al. (2026).

The practitioner's guide closes with eight steps. Three of them were not
covered by the workflow and are pinned here.

* Step 2 asks for the evidence behind the identifying assumption, which
  includes the covariate balance table in levels and in changes. The
  design step now carries it when covariates are declared; it is the
  output of ``sp.did_balance``.
* Step 2 also asks that the parallel-trends assumption actually imposed be
  stated. The imputation estimator restricts every pre-treatment period,
  which is not the assumption of a comparison with the not-yet-treated,
  and step 4 asks for the source of uncertainty to be named.
* Step 7 is the heterogeneity analysis: split the units into the cells of
  a characteristic and run the same DiD inside each. ``wf.heterogeneity``
  adds no estimator, so each row equals the workflow on that cell's rows,
  and on noiseless data with cell-specific trends it returns each cell's
  effect, which the pooled average hides.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

EFFECT = {0: 1.0, 1: 3.0}


def _panel(noise: float, seed: int = 0) -> pd.DataFrame:
    """80 units in two cells over 8 periods. The cells have different
    effects and different time trends, so a comparison across cells is not
    a valid DiD but a comparison inside a cell is."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(80):
        cell = unit % 2
        # 10 never treated per cell; the rest adopt in period 4 or 6
        slot = unit // 2
        first = 0 if slot < 10 else (4 if slot % 2 == 0 else 6)
        level = rng.normal()
        size = 1.0 + (unit % 5)
        for t in range(1, 9):
            d = int(first > 0 and t >= first)
            y = level + (0.5 + cell) * t + EFFECT[cell] * d + noise * rng.normal()
            rows.append((unit, cell, t, first, size, unit // 4, y))
    return pd.DataFrame(
        rows, columns=["unit", "cell", "t", "first", "size", "pair", "y"]
    )


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


KW = dict(
    y="y",
    id="unit",
    time="t",
    cohort="first",
    cluster="unit",
    target="overall",
    control_group="nevertreated",
)


# --------------------------------------------------------------------- #
# heterogeneity
# --------------------------------------------------------------------- #
def test_cells_recover_their_own_effects_which_the_average_hides():
    df = _panel(noise=0.0)
    wf = _quiet(sp.did_workflow, df, through="estimate", **KW)
    table = wf.heterogeneity("cell").set_index("cell")
    assert table.loc[0, "estimate"] == pytest.approx(EFFECT[0], abs=1e-10)
    assert table.loc[1, "estimate"] == pytest.approx(EFFECT[1], abs=1e-10)
    # both cells hold the same number of treated units
    assert table["share_of_treated"].tolist() == [0.5, 0.5]
    assert table.attrs["share_weighted_average"] == pytest.approx(2.0, abs=1e-10)
    assert table.attrs["headline"] == wf.estimate
    # the pooled number is neither cell's effect
    assert abs(wf.estimate - EFFECT[0]) > 0.5 and abs(wf.estimate - EFFECT[1]) > 0.5


def test_each_row_is_the_workflow_on_that_cell():
    df = _panel(noise=0.5)
    wf = _quiet(sp.did_workflow, df, through="estimate", **KW)
    table = wf.heterogeneity("cell")
    for _, row in table.iterrows():
        cell = _quiet(
            sp.did_workflow, df[df["cell"] == row["cell"]], through="estimate", **KW
        )
        assert (row["estimate"], row["se"]) == (cell.estimate, cell.se)
        assert (row["ci_lower"], row["ci_upper"]) == cell.ci
        assert row["n_units"] == 40


def test_shares_are_weighted_shares_of_the_treated_units():
    df = _panel(noise=0.5)
    wf = _quiet(sp.did_workflow, df, weights="size", through="estimate", **KW)
    table = wf.heterogeneity("cell").set_index("cell")
    units = df.drop_duplicates("unit")
    treated = units[units["first"] > 0]
    by_cell = treated.groupby("cell")["size"].sum() / treated["size"].sum()
    assert table["share_of_treated"].to_numpy() == pytest.approx(by_cell.to_numpy())
    assert table["share_of_treated"].sum() == pytest.approx(1.0)


def test_equality_test_only_when_cells_share_no_cluster():
    df = _panel(noise=0.5)
    wf = _quiet(sp.did_workflow, df, through="estimate", **KW)
    table = wf.heterogeneity("cell")
    test = table.attrs["equality_test"]
    assert test["available"] and test["df"] == 1
    prec = 1.0 / table["se"].to_numpy() ** 2
    est = table["estimate"].to_numpy()
    pooled = (prec * est).sum() / prec.sum()
    stat = float((prec * (est - pooled) ** 2).sum())
    assert test["statistic"] == pytest.approx(stat, rel=1e-12)
    assert test["pvalue"] == pytest.approx(stats.chi2.sf(stat, 1), rel=1e-12)

    # 'pair' groups units 0-3, 4-7, ...: every pair holds both cells
    shared = _quiet(
        sp.did_workflow, df, through="estimate", **{**KW, "cluster": "pair"}
    )
    other = shared.heterogeneity("cell").attrs["equality_test"]
    assert other["available"] is False
    assert "share clusters" in other["reason"]


def test_a_cell_without_treated_units_is_reported_not_estimated():
    df = _panel(noise=0.5)
    df["three"] = df["cell"]
    df.loc[(df["cell"] == 0) & (df["first"] == 0), "three"] = 2  # controls only
    wf = _quiet(sp.did_workflow, df, through="estimate", **KW)
    table = wf.heterogeneity("three").set_index("three")
    assert np.isnan(table.loc[2, "estimate"])
    assert table.loc[2, "share_of_treated"] == 0.0
    assert table.loc[2, "note"] != ""
    # cell 0 lost its comparison units with them
    assert np.isnan(table.loc[0, "estimate"]) and table.loc[0, "note"] != ""
    assert np.isfinite(table.loc[1, "estimate"])
    assert table.attrs["share_of_treated_covered"] == pytest.approx(0.5)


def test_heterogeneity_refuses_partitions_it_cannot_use():
    df = _panel(noise=0.5)
    wf = _quiet(sp.did_workflow, df, through="estimate", **KW)
    df2 = df.assign(moving=(df["t"] > 4).astype(int), one=1, many=df["unit"])
    wf2 = _quiet(sp.did_workflow, df2, through="estimate", **KW)
    for by, match in [
        ("moving", "changes within"),
        ("one", "takes one value"),
        ("many", "takes 80 values"),
        ("unit", "unit, period, cohort or outcome"),
        ("nope", "not a column"),
    ]:
        with pytest.raises(MethodIncompatibility, match=match):
            wf2.heterogeneity(by)
    plan = _quiet(sp.did_workflow, df, through="design", **KW)
    with pytest.raises(Exception, match="estimate"):
        plan.heterogeneity("cell")
    assert wf.heterogeneity("cell").attrs["by"] == "cell"


# --------------------------------------------------------------------- #
# the assumption and the source of uncertainty are stated
# --------------------------------------------------------------------- #
def test_imputation_states_the_all_periods_assumption():
    df = _panel(noise=0.5)
    skip = ["sensitivity", "robustness"]
    bjs = _quiet(
        sp.did_workflow, df, estimator="bjs", skip=skip, **{**KW, "control_group": None}
    )
    cs = _quiet(sp.did_workflow, df, skip=skip, **KW)
    said = bjs.checklist.set_index("item")["statement"]
    assert "in every period" in said["Identifying assumption"]
    assert "pre-treatment periods included" in bjs.step("declare").finding
    other = cs.checklist.set_index("item")["statement"]
    assert "never-treated units" in other["Identifying assumption"]
    assert "in every period" not in other["Identifying assumption"]
    for items in (said, other):
        assert "independent draws" in items["Source of uncertainty"]
        assert "unit clusters" in items["Source of uncertainty"]


# --------------------------------------------------------------------- #
# covariate balance in the design step
# --------------------------------------------------------------------- #
def test_design_step_carries_the_balance_table_of_the_covariates():
    df = _panel(noise=0.5)
    units = df["unit"]
    df["x"] = (units % 7) / 7.0 + 1.5 * (df["first"] > 0)  # imbalanced level
    df["z"] = (units % 3) / 3.0  # balanced by construction
    wf = _quiet(sp.did_workflow, df, covariates=["x", "z"], through="design", **KW)
    step = wf.step("design")
    direct = _quiet(
        sp.did_balance,
        df,
        ["x", "z"],
        g="first",
        t="t",
        i="unit",
        control_group="nevertreated",
    )
    pd.testing.assert_frame_equal(step.evidence["balance"], direct.table)
    assert step.evidence["balance_flagged"] == ["x"]
    assert "x differs" in step.finding and "0.25" in step.finding
    # reported, not graded: the covariates are already conditioned on
    assert step.status == "pass"

    plain = _quiet(sp.did_workflow, df, through="design", **KW)
    assert "balance" not in plain.step("design").evidence
