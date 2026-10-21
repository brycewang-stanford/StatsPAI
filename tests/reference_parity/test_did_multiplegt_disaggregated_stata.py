"""sp.did_multiplegt on data below the group level, and with controls=,
against Stata 18 ``did_multiplegt_old`` (point estimates, ``breps(0)``).

Fixture: ``_fixtures/dcdh_disaggregated_panel.csv`` (1-5 individuals per
group x period cell, from ``_generate_dcdh_disaggregated_data.py``) and
``_fixtures/dcdh_disaggregated_Stata.csv`` (``_generate_dcdh_disaggregated_Stata.do``).

Stata averages such data to (group, period) cells and weights each cell by
its row count: its results on the individual rows equal its results on the
collapsed cells with ``weight(n)`` to every digit (rows ``indiv`` vs
``cell_w`` of the fixture), and that is the rule implemented here. Before
2026-10-21 StatsPAI paired the rows of a group across periods (cell weight
n_t * n_{t-1}; DID_M 0.6149 against 0.5600) and the placebo / dynamic
paths kept the first row of each cell.

``controls=``: Stata nets out dX @ theta_d, with theta_d from a regression
of dY on dX and period effects over the stayer cells of baseline d. The
old code fitted dY on dX within each period's switchers and stayers
(0.7015 against 0.6425 on the collapsed data).

Tolerance 1e-10 relative: the same arithmetic on the same CSV bytes
(observed differences are below 3e-16).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
RTOL = 1e-10


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(FIX / "dcdh_disaggregated_panel.csv")


@pytest.fixture(scope="module")
def ref():
    raw = pd.read_csv(FIX / "dcdh_disaggregated_Stata.csv", skipinitialspace=True)
    return {(t, n): float(v) for t, n, v in raw.itertuples(index=False)}


def _fit(data, y, d, **kw):
    kw.setdefault("n_boot", 0)
    return sp.did_multiplegt(data, y=y, group="g", time="t", treatment=d, **kw)


def _close(a, b):
    assert np.isclose(a, b, rtol=RTOL, atol=1e-14), (a, b)


@pytest.mark.parametrize(
    "tag, y, d", [("indiv", "y", "d"), ("indiv_count", "y2", "d2")]
)
def test_individual_rows_match_stata(panel, ref, tag, y, d):
    r = _fit(panel, y, d)
    _close(r.estimate, ref[(tag, "effect_0")])
    assert r.model_info["n_switchers"] == ref[(tag, "n_switchers_0")]
    assert r.model_info["collapsed_to_cells"] is True
    assert r.n_obs == len(panel)
    # placebo of the run without robust_dynamic: placebo_sign='r' (did.md)
    p = _fit(panel, y, d, placebo=1, placebo_sign="r")
    _close(p.model_info["placebo"][0]["estimate"], ref[(tag, "placebo_1")])


def test_absorbing_dynamic_and_placebo_match_stata(panel, ref):
    """robust_dynamic run; StatsPAI's dynamic effects equal it on absorbing
    treatments (on/off panels are a documented difference, did.md)."""
    r = _fit(panel, "ya", "da", placebo=1, dynamic=1)
    dyn = [x["estimate"] for x in r.model_info["dynamic"]]
    _close(dyn[0], ref[("indiv_abs", "dyn_effect_0")])
    _close(dyn[1], ref[("indiv_abs", "dyn_effect_1")])
    _close(r.model_info["placebo"][0]["estimate"], ref[("indiv_abs", "dyn_placebo_1")])


def test_collapsed_unweighted_cells_match_stata(panel, ref):
    """One row per cell: the code path that existed before, unchanged."""
    cells = panel.groupby(["g", "t"], as_index=False)[["ya", "da"]].mean()
    r = _fit(cells, "ya", "da", placebo=1, dynamic=1)
    assert r.model_info["collapsed_to_cells"] is False
    dyn = [x["estimate"] for x in r.model_info["dynamic"]]
    _close(dyn[0], ref[("cell_abs_unw", "dyn_effect_0")])
    _close(dyn[1], ref[("cell_abs_unw", "dyn_effect_1")])


@pytest.mark.parametrize(
    "tag, collapse", [("indiv_ctrl", False), ("cell_unw_ctrl", True)]
)
def test_controls_match_stata(panel, ref, tag, collapse):
    data = (
        panel.groupby(["g", "t"], as_index=False)[["y", "d", "x"]].mean()
        if collapse
        else panel
    )
    r = _fit(data, "y", "d", controls=["x"])
    _close(r.estimate, ref[(tag, "effect_0")])
    assert r.model_info["controls_scope"] == "all"


def test_controls_with_placebo_or_dynamic_warn(panel):
    with pytest.warns(UserWarning, match="main DID_M estimate only"):
        r = _fit(panel, "y", "d", controls=["x"], placebo=1)
    assert r.model_info["controls_scope"] == "main"


def test_cluster_inside_a_cell_is_refused(panel):
    bad = panel.assign(c=np.arange(len(panel)) % 2)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="varies"):
        _fit(bad, "y", "d", cluster="c")


def test_bootstrap_resamples_groups_of_cells(panel):
    a = _fit(panel, "y", "d", n_boot=20, seed=3)
    b = _fit(panel, "y", "d", n_boot=20, seed=3)
    assert np.isfinite(a.se) and a.se > 0 and a.se == b.se
