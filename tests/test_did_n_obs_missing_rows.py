"""``n_obs`` of sun_abraham / did_imputation counts the rows they use.

2026-10-20 probe: both warned that they dropped the 50 rows with a missing
outcome and then reported ``n_obs = 3000`` (``len(data)``) instead of 2950,
unlike callaway_santanna. The fit on the data with those rows dropped is
the reference: same estimate, same ``n_obs``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panels():
    df = sp.dgp_did(n_units=300, n_periods=10, staggered=True, seed=3)
    df["first_treat"] = df["first_treat"].fillna(0)
    holed = df.copy()
    rng = np.random.default_rng(0)
    holed.loc[rng.choice(len(df), 50, replace=False), "y"] = np.nan
    return holed, holed.dropna(subset=["y"])


@pytest.mark.parametrize(
    "call",
    [
        lambda d: sp.sun_abraham(d, y="y", g="first_treat", t="time", i="unit"),
        lambda d: sp.did_imputation(
            d, y="y", group="unit", time="time", first_treat="first_treat"
        ),
    ],
    ids=["sun_abraham", "did_imputation"],
)
def test_n_obs_excludes_dropped_rows(panels, call):
    holed, clean = panels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b = call(holed), call(clean)
    assert a.n_obs == b.n_obs == len(clean)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12)
