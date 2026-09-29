"""``sp.hdfe_ols(..., wild=True)`` against Stata ``boottest``.

Reference: Stata 18 ``areg y d [x2] i.year, absorb(unit) cluster(cl)`` then
``boottest <var>, reps(10000) weight(rademacher)`` on
``_fixtures/hdfe_wild_panel.csv`` (60 units in 12 clusters, 10 years), from
``_fixtures/_generate_hdfe_wild_Stata.do``. With 12 clusters ``boottest``
enumerates all 2^12 Rademacher sign vectors, so its p-value is exact and
StatsPAI must reproduce it to the last draw.

The year effect is not nested in the clusters. The previous implementation
ran the bootstrap on already-demeaned data without re-absorbing the fixed
effects, which is only valid when every absorbed effect is nested in the
cluster; on the Web of Power replication (QJE 2023, Table 5 col. 1) it gave
p = 0.045 against ``boottest``'s 0.218. Now:

* p-values equal ``boottest`` exactly (``|t*| = |t|`` draws count as ties);
* the confidence set comes from test inversion, as in ``boottest``; its
  bounds are held to 0.3% of the interval width, since ``boottest`` locates
  them only to its ``ptol`` (1e-3).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "hdfe_wild_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(_FIX / "hdfe_wild_panel.csv")


@pytest.mark.parametrize(
    "key, rhs, var",
    [("one_d", "d", "d"), ("two_d", "d + x2", "d"), ("two_x2", "d + x2", "x2")],
)
def test_wild_matches_boottest(ref, panel, key, rhs, var):
    R = ref[key]
    r = sp.hdfe_ols(
        f"y ~ {rhs} | unit + year",
        data=panel,
        cluster="cl",
        wild=True,
        wild_n_boot=10000,
        wild_weight_type="rademacher",
    )
    ci = r.cluster_info["wild_ci"][var]
    assert r.cluster_info["wild_enumerated"] is True
    assert r.cluster_info["wild_n_boot"] == R["reps"]
    assert r.cluster_info["wild_p"][var] == pytest.approx(R["p"], abs=1e-12)
    width = R["hi"] - R["lo"]
    assert ci[0] == pytest.approx(R["lo"], abs=3e-3 * width)
    assert ci[1] == pytest.approx(R["hi"], abs=3e-3 * width)


def test_fe_not_nested_is_reabsorbed(panel):
    """Demeaning the bootstrap outcome matters: with the year effect ignored
    in the bootstrap the p-value would be far smaller."""
    r = sp.hdfe_ols(
        "y ~ d | unit + year",
        data=panel,
        cluster="cl",
        wild=True,
        wild_n_boot=4096,
        wild_weight_type="rademacher",
    )
    assert r.cluster_info["wild_method"].startswith("WCR")
    assert 0.5 < r.cluster_info["wild_p"]["d"] < 0.7


def test_webb_is_reproducible_and_close(ref, panel):
    kw = dict(
        data=panel, cluster="cl", wild=True, wild_n_boot=9999, wild_weight_type="webb"
    )
    a = sp.hdfe_ols("y ~ d | unit + year", wild_seed=1, **kw)
    b = sp.hdfe_ols("y ~ d | unit + year", wild_seed=1, **kw)
    assert a.cluster_info["wild_p"]["d"] == b.cluster_info["wild_p"]["d"]
    # Monte Carlo draws, not boottest's: within a few Monte Carlo SEs
    assert a.cluster_info["wild_p"]["d"] == pytest.approx(ref["one_d"]["p"], abs=0.03)


def test_wild_with_weights_raises(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(sp.MethodIncompatibility, match="weights"):
            sp.hdfe_ols(
                "y ~ d | unit + year",
                data=panel.assign(w=np.linspace(1, 2, len(panel))),
                weights="w",
                cluster="cl",
                wild=True,
            )
