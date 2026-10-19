"""sp.mi_estimate: the complete-data df of Barnard-Rubin is the fit's own.

Stata's ``mi estimate`` reads the complete-data df from ``e(df_r)`` of each
completed-data fit: ``n - k`` after ``regress``, ``G - 1`` after
``regress, vce(cluster g)`` and none (large-sample df) after ``logit``.
The expected per-coefficient df below are ``e(df_mi)`` from Stata 18 MP
(2026-10-19) on the three completed datasets built here, imported with
``mi import flong``.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.imputation.mice import MICEResult


@pytest.fixture(scope="module")
def completed():
    rng = np.random.default_rng(11)
    n, G = 150, 12
    g = rng.integers(1, G + 1, n)
    x1 = rng.normal(size=n) + 0.3 * g / G
    x2 = 0.5 * x1 + rng.normal(size=n)
    y = 1 + x1 + x2 + rng.normal(size=n) + rng.normal(size=G + 1)[g]
    miss = rng.choice(n, 25, replace=False)
    base = pd.DataFrame({"g": g, "x1": x1, "x2": x2, "y": y})
    sets = []
    for _ in range(3):
        c = base.copy()
        c.loc[miss, "x2"] = 0.5 * x1[miss] + rng.normal(size=len(miss))
        c["yb"] = (c["y"] > 1.5).astype(int)
        sets.append(c)
    return MICEResult(sets, 3, n, {"x2": 25}, ["x2"], {"x2": "pmm"}, {})


def _df(result):
    # Stata orders x1, x2, _cons; the formula puts the intercept first
    return np.asarray(result["df"])[[1, 2, 0]]


def test_clustered_regress_uses_g_minus_one(completed):
    r = sp.mi_estimate(
        completed, sp.regress, formula="y ~ x1 + x2", vce="cluster", cluster="g"
    )
    assert r["dfcom"] == 11.0  # Stata: Complete DF = 11
    np.testing.assert_allclose(
        _df(r), [8.104490064600705, 5.659912934198975, 9.275016989956601], rtol=1e-9
    )


def test_ols_regress_uses_n_minus_k(completed):
    r = sp.mi_estimate(completed, sp.regress, formula="y ~ x1 + x2")
    assert r["dfcom"] == 147.0
    np.testing.assert_allclose(
        _df(r), [77.32346494915595, 17.630060408717938, 112.7457956136777], rtol=1e-9
    )


def test_logit_uses_the_large_sample_df(completed):
    r = sp.mi_estimate(completed, sp.logit, formula="yb ~ x1 + x2")
    assert np.isinf(r["dfcom"])  # Stata: DF adjustment: Large sample
    # rtol 1e-5: the logit optimisers stop at different tolerances
    np.testing.assert_allclose(
        _df(r), [691.3877044720206, 1451.9771408954634, 24305.99300308955], rtol=1e-5
    )
