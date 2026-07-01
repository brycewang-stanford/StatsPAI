"""Analytical parity: sp.mediation_decompose total-effect identity (exact).

Natural-effects mediation decomposition satisfies the exact algebraic identity
``total effect = NDE + NIE`` (natural direct + natural indirect effect;
Pearl 2001, VanderWeele 2015), and the proportion mediated equals
``NIE / total``. The match is machine-precision.

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
def result():
    rng = np.random.default_rng(31)
    n = 1000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    tr = rng.integers(0, 2, n)
    med = 0.4 * tr + 0.3 * x1 + rng.normal(0, 1, n)
    y = 1.0 * tr + 0.5 * med + 0.4 * x1 - 0.2 * x2 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "treatment": tr, "mediator": med, "x1": x1, "x2": x2})
    return sp.mediation_decompose(
        df,
        y="y",
        treatment="treatment",
        mediator="mediator",
        covariates=["x1", "x2"],
    )


def test_total_effect_equals_nde_plus_nie(result):
    total = getattr(result, "total_effect", None)
    assert total is not None
    assert float(total) == pytest.approx(
        float(result.nde) + float(result.nie), abs=1e-12
    )


def test_proportion_mediated_identity(result):
    total = float(getattr(result, "total_effect"))
    if abs(total) > 1e-8:
        assert float(result.propn_mediated) == pytest.approx(
            float(result.nie) / total, abs=1e-10
        )
