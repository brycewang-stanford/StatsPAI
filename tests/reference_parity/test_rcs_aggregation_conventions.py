"""Repeated cross-sections: whose aggregation weights does ``sp.aggte`` use?

With repeated cross-sections there are no units to count, so an aggregation
of ``ATT(g, t)`` has to weight cells by *observations*, and the two
reference implementations weight them differently. The cells themselves
agree to machine precision on all three sides; only the aggregates move,
by 0.02 to 1.4 percent on this fixture. That is the kind of gap a parity
table records as "unexplained" unless the other side's number can be
rebuilt, so this file rebuilds it.

StatsPAI follows R ``did`` (the authors' implementation): a cohort's weight
is its share of treated observations, constant across ``t`` within the
cohort. Every aggregate matches R to 1e-12.

Stata ``csdid`` weights each *cell* by its own treated-observation count.
Reading ``csdid_estat.ado`` (``csdid_group`` in Mata):

    bw[g, t]  = N_treated(g, t) / sum over ALL cells of N_treated
    simple    = sum_{t >= g} bw * ATT / sum_{t >= g} bw
    ATT(g)    = sum_{t >= g, cohort g} bw * ATT / sum_{t >= g, cohort g} bw
    GAverage  = sum_g rx1_g * ATT(g),
                rx1_g proportional to (sum_{t >= g} bw[g, t]) / #post cells of g

The last line is the one that surprises: a cohort enters the group average
with its *mean* cell weight, not its total, so a cohort observed for fewer
post-treatment periods is not down-weighted for it. All three are rebuilt
here from StatsPAI's own cells and reproduce Stata exactly.

Reference values, run on the committed fixture (``rcs_panel.csv``)::

    # R 4.5.2, did 2.3.0
    a <- att_gt(yname="y", tname="year", idname="id", gname="gvar",
                data=df, panel=FALSE, control_group="nevertreated",
                est_method="reg", bstrap=FALSE, base_period="universal")
    aggte(a, type=<simple|dynamic|group|calendar>, bstrap=FALSE)

    * Stata 18, csdid 1.81
    csdid y, ivar() time(year) gvar(gvar) method(reg) agg(simple)
    estat simple ; estat group ; estat calendar ; estat event
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "rcs_aggregation"

# R did 2.3.0, overall aggregates (17 significant digits).
R_SIMPLE = 0.64439373661999222
R_DYNAMIC = 0.73116010075941595
R_GROUP = 0.62549774356507404
R_CALENDAR = 0.57150316663482115
R_GROUP_CELLS = {  # aggte(type="group")$att.egt
    2004: 0.70880195001627577,
    2005: 0.62158926246672508,
    2006: 0.54454152773106235,
}

# Stata csdid 1.81 (e(b) and r(table) of each estat).
STATA_SIMPLE = 0.6400463979150735
STATA_GROUP_AVG = 0.62560262242805553
STATA_GROUP_CELLS = {
    2004: 0.68646073095114757,
    2005: 0.63753441711530789,
    2006: 0.55290258727296449,
}


@pytest.fixture(scope="module")
def rcs():
    return pd.read_csv(_FIX / "rcs_panel.csv")


@pytest.fixture(scope="module")
def fit(rcs):
    return sp.callaway_santanna(
        rcs,
        y="y",
        g="gvar",
        t="year",
        i="id",
        panel=False,
        estimator="reg",
        control_group="nevertreated",
        base_period="universal",
    )


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("simple", R_SIMPLE),
        ("dynamic", R_DYNAMIC),
        ("group", R_GROUP),
        ("calendar", R_CALENDAR),
    ],
)
def test_aggregates_match_r_did(fit, kind, expected):
    got = sp.aggte(fit, type=kind, bstrap=False, cband=False)
    assert got.estimate == pytest.approx(expected, rel=1e-12)


def test_group_cells_match_r_did(fit):
    grp = sp.aggte(fit, type="group", bstrap=False, cband=False)
    got = dict(zip(grp.detail["group"].astype(int), grp.detail["att"]))
    for g, expected in R_GROUP_CELLS.items():
        assert got[g] == pytest.approx(expected, rel=1e-12), g


def _stata_weights(fit, rcs):
    """csdid's cell weights.

    ``e(gtt)`` reports, for each cell, ``N_trt`` = the cohort's observations
    in *both* periods of the 2x2 comparison: the base period ``t0`` and the
    comparison period ``t1``. With a universal base a post-treatment cell
    uses ``t0 = g - 1``; the one pre-treatment cell each cohort keeps uses
    the period before it (csdid's own long-difference bookkeeping), which is
    why the post-cell weights are what the aggregates renormalise over.
    """
    cells = fit.detail[["group", "time", "att"]].copy()
    n_trt = []
    for _, row in cells.iterrows():
        g, t = int(row["group"]), int(row["time"])
        base = g - 1 if t >= g else t - 1
        in_cohort = rcs["gvar"] == g
        n_trt.append(
            int((in_cohort & (rcs["year"] == base)).sum())
            + int((in_cohort & (rcs["year"] == t)).sum())
        )
    cells["n_trt"] = n_trt
    cells["w"] = cells["n_trt"] / cells["n_trt"].sum()
    return cells


def test_stata_csdid_aggregates_are_rebuilt_from_our_cells(fit, rcs):
    """The convention is earned: csdid's three numbers, to machine precision."""
    cells = _stata_weights(fit, rcs)
    post = cells["time"] >= cells["group"]

    simple = float(
        (cells.loc[post, "w"] * cells.loc[post, "att"]).sum()
        / cells.loc[post, "w"].sum()
    )
    assert simple == pytest.approx(STATA_SIMPLE, rel=1e-12)

    att_g, rx1 = {}, {}
    for g in sorted(cells["group"].unique()):
        m = post & (cells["group"] == g)
        att_g[int(g)] = float(
            (cells.loc[m, "w"] * cells.loc[m, "att"]).sum() / cells.loc[m, "w"].sum()
        )
        rx1[int(g)] = float(cells.loc[m, "w"].sum() / m.sum())
    for g, expected in STATA_GROUP_CELLS.items():
        assert att_g[g] == pytest.approx(expected, rel=1e-12), g

    total = sum(rx1.values())
    gavg = sum(rx1[g] / total * att_g[g] for g in att_g)
    assert gavg == pytest.approx(STATA_GROUP_AVG, rel=1e-12)


def test_agg_weights_csdid_reproduces_stata_directly(fit):
    """The convention is not merely rebuildable, it is selectable.

    The reconstruction above earns the label; this is the same arithmetic
    inside ``sp.aggte``, so a user who wants csdid's number asks for it
    rather than deriving it. The standard errors follow the same
    convention: csdid treats the aggregation weights as fixed (finding F21
    of the reconciliation study), and so does this path.
    """
    simple = sp.aggte(
        fit, type="simple", bstrap=False, cband=False, agg_weights="csdid"
    )
    assert float(simple.estimate) == pytest.approx(STATA_SIMPLE, rel=1e-12)

    group = sp.aggte(fit, type="group", bstrap=False, cband=False, agg_weights="csdid")
    assert float(group.estimate) == pytest.approx(STATA_GROUP_AVG, rel=1e-12)
    got = dict(zip(group.detail["group"].astype(int), group.detail["att"]))
    for g, expected in STATA_GROUP_CELLS.items():
        assert float(got[g]) == pytest.approx(expected, rel=1e-12), g

    # The default is unchanged: R did's convention is still what you get.
    assert float(
        sp.aggte(fit, type="simple", bstrap=False, cband=False).estimate
    ) == pytest.approx(R_SIMPLE, rel=1e-12)


@pytest.mark.parametrize("kind", ["dynamic", "calendar"])
def test_agg_weights_csdid_refuses_the_aggregates_it_was_not_read_off(fit, kind):
    with pytest.raises(sp.MethodIncompatibility, match="type='simple'"):
        sp.aggte(fit, type=kind, bstrap=False, cband=False, agg_weights="csdid")


def test_agg_weights_csdid_needs_the_cell_counts():
    """A panel fit does not record them -- and does not need them."""
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        [
            dict(id=u, year=t, gvar=(0, 3, 4)[u % 3], y=rng.normal())
            for u in range(90)
            for t in range(1, 6)
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        panel_fit = sp.callaway_santanna(
            panel,
            y="y",
            g="gvar",
            t="year",
            i="id",
            control_group="nevertreated",
            estimator="reg",
            base_period="universal",
        )
    with pytest.raises(sp.MethodIncompatibility, match="panel=False"):
        sp.aggte(
            panel_fit, type="simple", bstrap=False, cband=False, agg_weights="csdid"
        )


def test_agg_weights_rejects_an_unknown_convention(fit):
    with pytest.raises(sp.MethodIncompatibility, match="agg_weights"):
        sp.aggte(
            fit, type="simple", bstrap=False, cband=False, agg_weights="hdidregress"
        )


def test_the_two_conventions_disagree_by_more_than_rounding(fit):
    """Guard the premise: if they ever coincide, this file is pointless."""
    simple = sp.aggte(fit, type="simple", bstrap=False, cband=False).estimate
    group = sp.aggte(fit, type="group", bstrap=False, cband=False).estimate
    assert abs(simple / STATA_SIMPLE - 1) > 1e-4
    assert abs(group / STATA_GROUP_AVG - 1) > 1e-5
