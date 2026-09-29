"""``sp.bacon_decomposition(balance='drop_units')``.

The decomposition is defined on a balanced panel, so an unbalanced one is an
error by default (as in R/Stata ``bacondecomp``); ``drop_units`` keeps the
units observed in every period -- Stata ``xtbalance`` first -- and decomposes
that subpanel. The check is that it is exactly the decomposition of the
hand-balanced panel, that its TWFE coefficient is the two-way FE OLS on that
subpanel, and that the dropped units are reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel():
    rng = np.random.default_rng(3)
    cohorts = {
        u: g for u, g in zip(range(1, 13), [4, 4, 4, 6, 6, 6, 8, 8, 99, 99, 99, 99])
    }
    rows = []
    for u, g in cohorts.items():
        for t in range(1, 11):
            d = int(t >= g)
            rows.append(
                {
                    "i": u,
                    "t": t,
                    "d": d,
                    "y": u * 0.2
                    + 0.3 * t
                    + (1.0 + 0.2 * (t - g)) * d
                    + rng.normal(0, 0.5),
                }
            )
    df = pd.DataFrame(rows)
    # unit 2 misses period 5, unit 10 misses its outcome in period 9
    df = df[~((df.i == 2) & (df.t == 5))].copy()
    df.loc[(df.i == 10) & (df.t == 9), "y"] = np.nan
    return df


def test_unbalanced_still_errors_by_default():
    with pytest.raises(sp.MethodIncompatibility, match="Unbalanced"):
        sp.bacon_decomposition(_panel(), y="y", treat="d", time="t", id="i")


def test_drop_units_is_the_balanced_subpanel_decomposition():
    df = _panel()
    r = sp.bacon_decomposition(
        df, y="y", treat="d", time="t", id="i", balance="drop_units"
    )
    assert r["dropped_units"] == [2, 10]
    assert r["n_units"] == 10
    bal = df[~df.i.isin([2, 10])]
    ref = sp.bacon_decomposition(bal, y="y", treat="d", time="t", id="i")
    assert r["beta_twfe"] == pytest.approx(ref["beta_twfe"], abs=1e-12)
    pd.testing.assert_frame_equal(r["decomposition"], ref["decomposition"])
    assert r["weighted_sum"] == pytest.approx(r["beta_twfe"], abs=1e-10)
    twfe = sp.feols("y ~ d | i + t", data=bal)
    assert r["beta_twfe"] == pytest.approx(float(twfe.params["d"]), abs=1e-10)


def test_balanced_panel_unchanged():
    df = _panel()
    bal = df[~df.i.isin([2, 10])]
    a = sp.bacon_decomposition(bal, y="y", treat="d", time="t", id="i")
    b = sp.bacon_decomposition(
        bal, y="y", treat="d", time="t", id="i", balance="drop_units"
    )
    assert b["dropped_units"] == [] and a["beta_twfe"] == b["beta_twfe"]
