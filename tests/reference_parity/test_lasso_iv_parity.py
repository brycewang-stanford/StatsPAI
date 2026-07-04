"""Analytical parity: sp.lasso_iv known-truth effect recovery.

Angrist-Pischke / Belloni-Chen-Chernozhukov-Hansen LASSO-iv selects
instruments from a large candidate set (BIC / aic / pvalue) and reports
the resulting 2SLS estimate. On a 30-instrument DGP with two true
instruments, the recovered 2SLS coefficient tracks the planted effect
analytically. Analytical evidence tier (known-truth recovery; the LASSO
selection path is greedy and not bit-exact vs a single R reference).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

TRUE_BETA = 0.8
TRUE_CONST = 1.0
N_INSTR = 30


def _dgp(seed, n=1500, K=N_INSTR):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, K))
    X = 0.6 * Z[:, 0] - 0.4 * Z[:, 1] + rng.standard_normal(n) * 0.5
    y = TRUE_CONST + TRUE_BETA * X + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"y": y, "x": X, **{f"z{i}": Z[:, i] for i in range(K)}})


def test_lasso_iv_recovers_true_effect_bic():
    df = _dgp(0)
    r = sp.lasso_iv(
        df,
        "y",
        x_endog=["x"],
        z=[f"z{i}" for i in range(N_INSTR)],
        penalty="bic",
    )
    assert float(r.params["x"]) == pytest.approx(TRUE_BETA, abs=0.1)
    assert float(r.params["Intercept"]) == pytest.approx(TRUE_CONST, abs=0.1)


def test_lasso_iv_recovers_across_seeds():
    for seed in range(4):
        df = _dgp(seed)
        r = sp.lasso_iv(
            df,
            "y",
            x_endog=["x"],
            z=[f"z{i}" for i in range(N_INSTR)],
            penalty="bic",
        )
        assert float(r.params["x"]) == pytest.approx(TRUE_BETA, abs=0.15)
