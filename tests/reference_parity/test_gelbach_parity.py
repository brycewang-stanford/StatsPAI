"""Analytical parity: sp.gelbach conditional decomposition (closed-form identity).

The Gelbach (2016) decomposition attributes the change in a focal coefficient
when adding covariates to each added covariate. Two exact algebraic identities
must hold (Gelbach 2016, JOLE 34(2)):

  1. ``total_change == base_coef - full_coef`` (the focal coefficient moves by
     exactly the total attributed to the added covariates).
  2. The per-covariate ``delta`` contributions sum to ``total_change`` (the
     decomposition is exact and order-invariant — the key property that
     distinguishes Gelbach from sequential/Oaxaca decompositions).

Closed-form identity (no external fixture needed); match is machine-precision.

References
----------
- Gelbach, J.B. (2016). When Do Covariates Matter? *JOLE* 34(2), 509-543.
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
    rng = np.random.default_rng(13)
    n = 800
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    d = 0.5 * x1 + 0.3 * x2 + rng.normal(0, 1, n)
    y = 1.0 + 2.0 * d + 0.8 * x1 - 0.5 * x2 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    return sp.gelbach(df, y="y", base_x=["d"], added_x=["x1", "x2"])


def test_total_change_equals_coefficient_move(result):
    assert result.total_change == pytest.approx(
        result.base_coef - result.full_coef, abs=1e-12
    )


def test_decomposition_sums_to_total_change(result):
    delta_sum = float(result.decomposition["delta"].sum())
    assert delta_sum == pytest.approx(result.total_change, abs=1e-12)


def test_percentages_sum_to_one_hundred(result):
    pct_sum = float(result.decomposition["pct_of_change"].sum())
    assert pct_sum == pytest.approx(100.0, abs=1e-9)
