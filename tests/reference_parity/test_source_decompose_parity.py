"""Analytical parity: sp.source_decompose exact Gini source identity.

Lerman-Yitzhaki (1985) Gini source decomposition attributes total Gini to
income sources via  contribution_k = S_k * R_k * G_k  (share * Gini-correlation
* own Gini). Two exact identities hold to machine precision:

    sum_k contribution_k == total_gini
    contribution_k == share_k * gini_corr_k * gini_k_k   (per source)
    sum_k pct_of_gini_k == 100

Closed-form identity (no external fixture).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "labor": rng.lognormal(9.5, 0.4, 400),
            "capital": rng.lognormal(8.0, 0.7, 400),
        }
    )
    return sp.source_decompose(df, sources=["labor", "capital"])


def test_contributions_sum_to_total_gini(result):
    sd = result.sources
    assert float(sd["contribution"].sum()) == pytest.approx(
        float(result.total_gini), abs=1e-12
    )


def test_lerman_yitzhaki_per_source_identity(result):
    sd = result.sources
    lhs = sd["share"] * sd["gini_corr"] * sd["gini_k"]
    assert float((lhs - sd["contribution"]).abs().max()) == pytest.approx(
        0.0, abs=1e-12
    )


def test_percentages_sum_to_100(result):
    sd = result.sources
    assert float(sd["pct_of_gini"].sum()) == pytest.approx(100.0, abs=1e-9)
