"""``sp.match(method='cem', k2k=True)`` vs R ``MatchIt`` 4.7.2 and the
``sp.l1_imbalance`` statistic.

Fixture: ``_fixtures/cem_k2k_R.json`` from ``_fixtures/_generate_cem_k2k_R.R``
on ``matchit_constraints.csv`` (MatchIt is GPL; black box). With four
equal-width bins per covariate both sides form the same cells; the k2k
pairs (Mahalanobis nearest neighbour without replacement inside each cell,
data order) must then agree row by row, and the ATT of the pairs to 1e-10.
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
REF = json.loads((_FIX / "cem_k2k_R.json").read_text(encoding="utf-8"))
CASES = [k for k in REF if k != "versions"]


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "matchit_constraints.csv")


@pytest.mark.parametrize("key", CASES)
def test_k2k_pairs_match_matchit(data, key):
    c = REF[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(data, "y", "d", c["covariates"], method="cem", n_bins=4, k2k=True)
    md = r.model_info["matched_data"]
    row_of_id = dict(zip(md["_id"].to_numpy(), md.index.to_numpy() + 1))
    tr = md[(md["_treated"] == 1) & md["_n1"].notna()]
    ours = {int(i + 1): int(row_of_id[v]) for i, v in zip(tr.index, tr["_n1"])}
    theirs = {
        int(t): int(cr)
        for t, cr in zip(c["treated_row"], c["control_row"])
        if cr is not None and cr != "NA"
    }
    assert ours == theirs
    assert abs(r.estimate - c["att"]) < 1e-10


def test_l1_imbalance_bounds_and_identities(data):
    df = data.copy()
    # identical distributions in both arms: 0
    dup = pd.concat([df.assign(d=1), df.assign(d=0)], ignore_index=True)
    assert sp.l1_imbalance(dup, "d", ["x1", "x2"], n_bins=5) == pytest.approx(
        0.0, abs=1e-15
    )
    # arms in disjoint cells (split at an internal bin edge): 1
    edge = np.linspace(df["x1"].min(), df["x1"].max(), 5)[2]
    sep = df.assign(d=(df["x1"] > edge).astype(int))
    assert sp.l1_imbalance(sep, "d", ["x1"], n_bins=4) == pytest.approx(1.0)
    # hand computation on one coarsened variable
    t = df["d"].to_numpy() == 1
    edges = np.linspace(df["x1"].min(), df["x1"].max(), 4)
    cell = np.clip(np.searchsorted(edges[1:], df["x1"], side="left"), 0, 2)
    f = np.bincount(cell[t], minlength=3) / t.sum()
    g = np.bincount(cell[~t], minlength=3) / (~t).sum()
    assert sp.l1_imbalance(df, "d", ["x1"], n_bins=3) == pytest.approx(
        0.5 * np.abs(f - g).sum()
    )


def test_matching_reduces_l1(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(data, "y", "d", ["x1", "x2"], method="cem", n_bins=4, k2k=True)
    md = r.model_info["matched_data"]
    before = sp.l1_imbalance(data, "d", ["x1", "x2"], n_bins=4)
    after = sp.l1_imbalance(md, "d", ["x1", "x2"], weights="_weight", n_bins=4)
    assert after < before
