"""``sp.anderson_rubin_ci(cluster=)`` against Stata ``ivreg2``.

The Anderson-Rubin confidence set had no clustered version (AI-tocracy,
QJE 2023), so the weak-IV-robust interval of a clustered IV could not be
reported. It now inverts the cluster-robust AR statistic ``ivreg2`` reports:
the Wald test of the instruments in the regression of ``y - b0 x`` on them
and the controls, CR0 meat, scaled to an F by ``(n - k_W - k)/(n - 1) *
(G - 1)/G``, referred to ``F(k, G - 1)``.

Reference: Stata 18, ``ivreg2 (y - b0 x) w (x = z), cluster(cl) ffirst`` at
seven null values on ``_fixtures/iv_wild_data.csv`` (10 clusters), from
``_generate_ar_ci_cluster_Stata.do``: ``e(arf)`` and ``e(arfp)`` agree to
1e-12, and the set's bounds are where the statistic crosses the critical
value.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "ar_ci_cluster_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "iv_wild_data.csv")


def test_statistic_matches_ivreg2_on_the_grid(data):
    grid = np.array([r["b0"] for r in R])
    ci = sp.anderson_rubin_ci(
        "y", "x", ["z"], exog=["w"], data=data, cluster="cl", beta_grid=grid
    )
    np.testing.assert_allclose(ci.statistic, [r["arf"] for r in R], rtol=1e-12)
    G = data["cl"].nunique()
    np.testing.assert_allclose(
        stats.f.sf(ci.statistic, 1, G - 1),
        [r["arfp"] for r in R],
        rtol=1e-10,
        atol=1e-15,
    )


def test_bounds_sit_at_the_critical_value(data):
    ci = sp.anderson_rubin_ci("y", "x", ["z"], exog=["w"], data=data, cluster="cl")
    assert ci.is_connected and not ci.is_unbounded
    crit = stats.f.ppf(0.95, 1, data["cl"].nunique() - 1)
    at = sp.anderson_rubin_ci(
        "y",
        "x",
        ["z"],
        exog=["w"],
        data=data,
        cluster="cl",
        beta_grid=np.array([ci.lower, ci.upper]),
    )
    np.testing.assert_allclose(at.statistic, crit, rtol=1e-6)
    assert ci.extra["vcov"] == "cluster"


def test_unclustered_path_is_unchanged(data):
    a = sp.anderson_rubin_ci("y", "x", ["z"], exog=["w"], data=data)
    assert a.extra["vcov"] == "homoskedastic"
    assert a.lower < a.upper
