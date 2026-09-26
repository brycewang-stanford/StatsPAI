"""sp.did_summary passes cluster= to the CS row too.

Before 1.32 every method except CS received ``cluster=``; the CS row kept
unit-clustered analytic SEs, understating them by ~2x on this design.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def pn():
    rng = np.random.default_rng(1)
    n_units, n_t = 300, 8
    ids = np.repeat(np.arange(n_units), n_t)
    tt = np.tile(np.arange(1, n_t + 1), n_units)
    coh = np.array([0, 4, 6])[np.arange(n_units) % 3][ids]
    st = np.repeat(np.arange(n_units) % 15, n_t)
    y = (
        ((coh > 0) & (tt >= coh)) * 1.0
        + 0.3 * rng.normal(size=15)[st] * tt
        + rng.normal(size=ids.size)
    )
    return pd.DataFrame({"id": ids, "time": tt, "ft": coh, "st": st, "y": y})


def _cs_row(pn, cluster):
    res = sp.did_summary(
        pn,
        y="y",
        time="time",
        first_treat="ft",
        group="id",
        methods=["cs"],
        cluster=cluster,
    )
    return res.detail.set_index("method").loc["cs"]


def test_cs_row_clusters_on_requested_variable(pn):
    unit = _cs_row(pn, None)
    state = _cs_row(pn, "st")
    assert state["estimate"] == pytest.approx(unit["estimate"], rel=1e-12)
    # Same data as a direct CS multiplier-bootstrap call clustered by state.
    cs = sp.callaway_santanna(
        pn,
        y="y",
        g="ft",
        t="time",
        i="id",
        clustervars=["id", "st"],
        bstrap=True,
        random_state=0,
    )
    ref = sp.aggte(cs, type="simple", bstrap=True, random_state=0)
    assert state["se"] == pytest.approx(ref.se, rel=1e-12)
    assert state["se"] > 1.5 * unit["se"]


def test_unit_cluster_keeps_analytic_se(pn):
    assert _cs_row(pn, "id")["se"] == pytest.approx(_cs_row(pn, None)["se"], rel=1e-12)
