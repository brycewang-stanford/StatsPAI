"""Analytical parity: sp.romano_wolf multiple-testing identities.

Romano-Wolf stepdown FWER control. The bootstrap-resampled RW column is
Monte-Carlo, but the surrounding table carries exact identities:

    coef[k] == OLS slope of y_k ~ x       (controls cancel)  -> machine-exact
    t       == coef / se                                     -> machine-exact
    p_bonf  == min(1, p_value * n_outcomes)  (Bonferroni)    -> machine-exact
    p_rw    >= p_value                        (adjustment is monotone)

Analytical evidence tier (exact adjustment identities + MC RW column).
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
def fitted():
    rng = np.random.default_rng(1)
    n = 800
    d = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "y1": 0.5 * d + rng.normal(0, 1, n),
            "y2": rng.normal(0, 1, n),
            "y3": 0.3 * d + rng.normal(0, 1, n),
            "d": d,
        }
    )
    return df, sp.romano_wolf(df, y=["y1", "y2", "y3"], x="d", n_boot=500, seed=0)


def test_coef_and_t_are_exact_ols(fitted):
    df, r = fitted
    t = r.table.set_index("outcome")
    d = df["d"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(d)), d])
    for outcome in ("y1", "y2", "y3"):
        y = df[outcome].to_numpy(dtype=float)
        beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
        assert float(t.loc[outcome, "coef"]) == pytest.approx(beta, abs=1e-10)
        assert float(t.loc[outcome, "t"]) == pytest.approx(
            float(t.loc[outcome, "coef"]) / float(t.loc[outcome, "se"]), abs=1e-10
        )


def test_bonferroni_identity_and_rw_monotone(fitted):
    _, r = fitted
    t = r.table
    k = len(t)
    for _, row in t.iterrows():
        assert float(row["p_bonf"]) == pytest.approx(
            min(1.0, float(row["p_value"]) * k), abs=1e-12
        )
        # RW-adjusted p never falls below the raw p (MC resolution 1/n_boot)
        assert float(row["p_rw"]) >= float(row["p_value"]) - 1.0 / 500
