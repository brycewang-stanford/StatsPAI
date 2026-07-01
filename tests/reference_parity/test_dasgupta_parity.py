"""Analytical parity: sp.das_gupta standardization decomposition (exact identity).

The Das Gupta (1993) standardization decomposition attributes the difference in
a crude rate between two populations to each factor. The defining exact
algebraic identity is that the per-factor effects sum to the total gap, and the
percentage contributions sum to 100%.

Closed-form identity (no external fixture needed); machine-precision.

References
----------
- Das Gupta, P. (1993). Standardization and Decomposition of Rates.
  *US Census Bureau*, Current Population Reports P23-186.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


@pytest.fixture(scope="module")
def result():
    a = pd.DataFrame({"rate": [0.10, 0.20, 0.30], "weight": [0.5, 0.3, 0.2]})
    b = pd.DataFrame({"rate": [0.15, 0.25, 0.35], "weight": [0.4, 0.4, 0.2]})
    return sp.das_gupta(a, b, factor_names=["rate", "weight"])


def test_factor_effects_sum_to_gap(result):
    effects = result.factor_effects
    effect_sum = float(effects["effect"].sum())
    assert effect_sum == pytest.approx(result.gap, abs=1e-12)


def test_percentages_sum_to_one_hundred(result):
    effects = result.factor_effects
    pct_sum = float(effects["pct_of_gap"].sum())
    assert pct_sum == pytest.approx(100.0, abs=1e-9)
