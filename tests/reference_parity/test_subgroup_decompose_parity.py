"""Analytical parity: sp.subgroup_decompose within/between identity (exact).

The additive decomposition of a generalized-entropy inequality index (Theil-T,
Theil-L) into within-group and between-group components is an exact algebraic
identity: ``total == within + between``. This is the defining property of the
Theil family (Shorrocks 1980); the match is machine-precision.

Closed-form identity (no external fixture needed).
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
def data():
    rng = np.random.default_rng(5)
    n = 1200
    return pd.DataFrame(
        {
            "y": np.abs(rng.normal(5, 2, n)) + 1.0,
            "by": rng.choice(["g1", "g2", "g3"], n),
        }
    )


@pytest.mark.parametrize("index", ["theil_t", "theil_l"])
def test_within_plus_between_equals_total(data, index):
    r = sp.subgroup_decompose(data, y="y", by="by", index=index)
    assert r.total == pytest.approx(r.within + r.between, abs=1e-12)
    assert r.within >= 0.0
    assert r.between >= 0.0
