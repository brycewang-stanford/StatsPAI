"""Analytical parity: efficiency / productivity frontier estimators.

* ``sp.malmquist`` satisfies the exact decomposition
  ``M = EC * TC`` (efficiency change times technical change), reports technical
  progress (``TC > 1``) when the frontier advances, and a strictly larger
  ``TC`` under growth than under a stagnant technology.
* ``sp.metafrontier`` produces a metafrontier that envelops the group frontiers
  (``TE_meta <= TE_group``), a technology-gap ratio ``TGR = TE_meta / TE_group``
  in ``(0, 1]``, and a smaller ``TGR`` for the technologically disadvantaged
  group.

Analytical evidence tier (structural identities and known-truth ordering on
deterministic DGPs).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def _malmquist_panel(g, seed=0, N=60, T=3):
    """Cobb-Douglas frontier growing by factor ``g`` per period."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        eff = rng.uniform(0.85, 1.0)
        for t in range(T):
            x1 = rng.uniform(1, 5)
            x2 = rng.uniform(1, 5)
            y = (g ** t) * eff * (x1 ** 0.4) * (x2 ** 0.5)
            rows.append((i, t, y, x1, x2))
    df = pd.DataFrame(rows, columns=["id", "time", "y", "x1", "x2"])
    return sp.malmquist(df, y="y", x=["x1", "x2"], id="id", time="time")


def test_malmquist_decomposition_identity():
    it = _malmquist_panel(1.10).index_table
    np.testing.assert_allclose(
        it["m_index"].to_numpy(), (it["ec"] * it["tc"]).to_numpy(), rtol=1e-9
    )


def test_malmquist_detects_technical_progress():
    it = _malmquist_panel(1.10).index_table
    assert it["tc"].mean() > 1.0


def test_malmquist_more_progress_under_growth_than_stagnation():
    grow = _malmquist_panel(1.10).index_table["tc"].mean()
    flat = _malmquist_panel(1.00).index_table["tc"].mean()
    assert grow > flat


def _metafrontier_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    rows = []
    for grp, gap in [(0, 0.0), (1, 0.3)]:  # group 1 frontier sits below group 0
        for _ in range(n):
            x = rng.uniform(1, 5)
            u = rng.exponential(0.15)  # one-sided inefficiency
            y = np.exp(0.5 * np.log(x) + 1.0 - gap - u)
            rows.append((grp, np.log(y), np.log(x)))
    return pd.DataFrame(rows, columns=["grp", "ly", "lx"])


def test_metafrontier_envelops_group_frontiers():
    df = _metafrontier_data()
    res = sp.metafrontier(df, y="ly", x=["lx"], group="grp")
    te_group = np.asarray(res.te_group, dtype=float)
    te_meta = np.asarray(res.te_meta, dtype=float)
    # Metafrontier is at least as demanding: meta efficiency <= group efficiency.
    assert np.all(te_meta <= te_group + 1e-9)


def test_metafrontier_tgr_is_ratio_in_unit_interval():
    df = _metafrontier_data()
    res = sp.metafrontier(df, y="ly", x=["lx"], group="grp")
    te_group = np.asarray(res.te_group, dtype=float)
    te_meta = np.asarray(res.te_meta, dtype=float)
    tgr = np.asarray(res.tgr, dtype=float)
    np.testing.assert_allclose(tgr, te_meta / te_group, atol=1e-6)
    assert np.all((tgr > 0) & (tgr <= 1 + 1e-9))


def test_metafrontier_disadvantaged_group_has_larger_gap():
    df = _metafrontier_data()
    res = sp.metafrontier(df, y="ly", x=["lx"], group="grp")
    tgr = np.asarray(res.tgr, dtype=float)
    mean_by_grp = pd.DataFrame({"grp": df["grp"].to_numpy(), "tgr": tgr}) \
        .groupby("grp")["tgr"].mean()
    # Group 1 sits below group 0's technology -> smaller technology-gap ratio.
    assert mean_by_grp[1] < mean_by_grp[0]
