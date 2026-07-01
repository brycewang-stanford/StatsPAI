"""Analytical parity: sp.kitagawa_decompose exact additive identity.

Kitagawa (1955) two-factor rate decomposition splits the group rate gap into a
*rate effect* and a *composition effect* (plus a symmetric interaction term).
By construction the split is exact:

    gap == rate_a - rate_b
    gap == rate_effect + composition_effect + interaction

Both hold to machine precision. Closed-form identity (no external fixture).
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
    rng = np.random.default_rng(7)
    n = 2000
    grp = rng.integers(0, 2, n)
    cat = rng.integers(0, 3, n)
    rate = 0.2 + 0.1 * grp + 0.05 * cat + rng.normal(0, 0.02, n)
    df = pd.DataFrame({"y": rate, "g": grp, "c": cat})
    return sp.kitagawa_decompose(df, rate="y", group="g", by="c")


def test_gap_equals_rate_difference(result):
    assert float(result.gap) == pytest.approx(
        float(result.rate_a) - float(result.rate_b), abs=1e-12
    )


def test_gap_equals_effect_sum(result):
    interaction = float(getattr(result, "interaction", 0.0))
    total = float(result.rate_effect) + float(result.composition_effect) + interaction
    assert float(result.gap) == pytest.approx(total, abs=1e-12)
