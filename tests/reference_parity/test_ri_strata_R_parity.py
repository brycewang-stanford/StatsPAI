"""``sp.ri_test(strata=, stat='ols', covariates=)`` against R ``ri2``.

``ri_test`` permuted treatment across the whole sample, and its statistic
only saw ``(Y, D)``. A stratified experiment (UCT, QJE 2016: treatment
assigned within villages) needs within-stratum re-randomization and a
regression statistic with controls and stratum effects; permuting across
strata gave an education p of 0.70 against 0.17.

Reference: R 4.5.2, ``ri2`` 0.5.0 ``conduct_ri`` with
``declare_ra(blocks=, [clusters=], block_m=)`` on the two designs of
``_fixtures/ri_strata_{units,clusters}.csv``, both small enough to enumerate
every assignment (8,000 and 1,296), from ``_generate_ri_strata_R.R``. The
observed statistics and the two-sided p-values agree exactly.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "ri_strata_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def units():
    return pd.read_csv(_FIX / "ri_strata_units.csv")


@pytest.fixture(scope="module")
def clusters():
    return pd.read_csv(_FIX / "ri_strata_clusters.csv")


@pytest.mark.parametrize("spec", ["dim", "ols"])
def test_unit_strata_match_ri2(units, spec):
    ref = R[f"units_{spec}"]
    kw = {} if spec == "dim" else dict(stat="ols", covariates=["x"])
    r = sp.ri_test(units, y="y", treat="Z", strata="block", n_perms=10000, **kw)
    assert r["exact"] and r["n_perms"] == ref["n"]
    assert r["observed"] == pytest.approx(ref["obs"], abs=1e-12)
    assert r["p_value"] == pytest.approx(ref["p"], abs=1e-12)


def test_clusters_within_strata_match_ri2(clusters):
    ref = R["clusters_ols"]
    r = sp.ri_test(
        clusters,
        y="y",
        treat="Z",
        strata="block",
        cluster="clust",
        stat="ols",
        covariates=["x"],
        n_perms=10000,
    )
    assert r["exact"] and r["n_perms"] == ref["n"]
    assert r["observed"] == pytest.approx(ref["obs"], abs=1e-12)
    assert r["p_value"] == pytest.approx(ref["p"], abs=1e-12)


def test_random_permutations_keep_stratum_counts(units):
    """Above the enumeration limit the draws still re-randomize within
    strata: the treated count of every stratum is preserved."""
    seen = []

    def stat(y, d):
        seen.append(pd.Series(d).groupby(units["block"].to_numpy()).sum().tolist())
        return float(np.mean(y[d == 1]) - np.mean(y[d == 0]))

    sp.ri_test(units, y="y", treat="Z", strata="block", stat=stat, n_perms=50, seed=1)
    assert all(s == [3, 3, 3] for s in seen)


def test_ols_t_runs_and_covariates_need_ols(units):
    r = sp.ri_test(
        units,
        y="y",
        treat="Z",
        strata="block",
        stat="ols_t",
        covariates=["x"],
        n_perms=10000,
    )
    assert 0 < r["p_value"] <= 1
    with pytest.raises(ValueError, match="covariates"):
        sp.ri_test(units, y="y", treat="Z", covariates=["x"])
