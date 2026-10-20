"""Behaviour pinned by the 2026-10-20 clean-room pass over the CS code.

See ``docs/dev/cleanroom/cs_rcs.md``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did._core import multiplier_bootstrap


def _rcs(seed: int = 1, n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = rng.choice([1, 2, 3, 4, 5], n)
    g = rng.choice([0, 3, 4, 5], n)
    x = rng.normal(size=n) + 0.3 * (g > 0)
    eff = np.where((g > 0) & (t >= g), (t - g + 1) * 0.5, 0.0)
    y = 1 + 0.4 * t + 0.6 * x + 0.04 * g + eff + rng.normal(size=n)
    return pd.DataFrame({"id": np.arange(n), "t": t, "g": g, "x": x, "y": y})


def _panel(seed: int = 2, n_units: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.choice([0, 3, 4, 5], n_units)
    a = rng.normal(size=n_units)
    rows = []
    for i in range(n_units):
        for t in range(1, 6):
            eff = (t - g[i] + 1) * 0.5 if (g[i] > 0 and t >= g[i]) else 0.0
            rows.append((i, t, g[i], a[i] + 0.3 * t + eff + rng.normal()))
    return pd.DataFrame(rows, columns=["id", "t", "g", "y"])


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(df, y="y", g="g", t="t", i="id", **kw)


@pytest.mark.parametrize("panel", [True, False])
def test_cohort_after_the_sample_is_never_treated(panel):
    """Recoding never-treated units to a first-treatment date after the last
    period changes nothing: those units are untreated throughout (R ``did``
    gives the same ATT and SE, black box to 1e-15)."""
    df = _panel() if panel else _rcs()
    kw = {} if panel else {"panel": False}
    base = _fit(df, **kw)
    moved = df.copy()
    never = moved["g"] == 0
    moved.loc[never & (moved["id"] % 2 == 1), "g"] = 9
    alt = _fit(moved, **kw)
    assert alt.estimate == pytest.approx(base.estimate, rel=1e-12)
    assert alt.se == pytest.approx(base.se, rel=1e-12)


def test_cohort_after_the_sample_matches_r_did():
    """R did 2.5.1, ``att_gt(panel=FALSE, est_method='reg',
    base_period='universal', bstrap=FALSE)`` then ``aggte(type='simple')``
    on this data with half the never-treated coded 9: 0.945521495902184
    (SE 0.176948966650982)."""
    df = _rcs()
    never = np.flatnonzero(df["g"].to_numpy() == 0)
    df.loc[never[1::2], "g"] = 9
    res = _fit(df, panel=False, estimator="reg")
    agg = sp.aggte(res, type="simple", bstrap=False)
    assert agg.estimate == pytest.approx(0.945521495902184, rel=1e-9)
    assert agg.se == pytest.approx(0.176948966650982, rel=1e-9)


def test_multiplier_bootstrap_nan_column_is_not_hidden():
    rng = np.random.default_rng(0)
    psi = np.column_stack([rng.normal(size=50), np.full(50, np.nan)])
    se, crit = multiplier_bootstrap(psi, 50, 0.05, 199, 3)
    assert np.isfinite(se[0]) and np.isnan(se[1]) and np.isnan(crit)


def test_multiplier_bootstrap_singleton_clusters_identity():
    """With one row per cluster, clustering changes nothing."""
    rng = np.random.default_rng(1)
    psi = rng.normal(size=(80, 2))
    plain, c1 = multiplier_bootstrap(psi, 80, 0.05, 499, 5)
    clustered, c2 = multiplier_bootstrap(
        psi, 80, 0.05, 499, 5, cluster_ids=np.arange(80)
    )
    np.testing.assert_allclose(clustered, plain, rtol=1e-12)
    assert c1 == pytest.approx(c2, rel=1e-12)
