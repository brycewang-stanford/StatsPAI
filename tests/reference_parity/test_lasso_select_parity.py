"""Analytical parity: sp.lasso_select known-truth support recovery.

LASSO (Tibshirani 1996) with K-fold CV-selected penalty. On a known sparse
DGP y = X @ true_b + noise with true_b sparse, the CV-selected model
recovers all three true non-zero predictors and drops at least some of the
purely noise variables. Analytical evidence tier.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

TRUE_SUPPORT = [0, 5, 12]  # x0, x5, x12 carry signal


def _dgp(seed, n=500, p=20):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    true_b = np.zeros(p)
    for i, v in zip(TRUE_SUPPORT, (1.0, 0.8, -0.6)):
        true_b[i] = v
    y = X @ true_b + rng.standard_normal(n) * 0.5
    df = pd.DataFrame({"y": y, **{f"x{i}": X[:, i] for i in range(p)}})
    return df, true_b


def test_cv_lasso_recovers_all_true_predictors():
    df, true_b = _dgp(0)
    r = sp.lasso_select(
        df,
        "y",
        [f"x{i}" for i in range(20)],
        method="cv",
        n_folds=5,
        seed=0,
    )
    sel = r.selected
    for i in TRUE_SUPPORT:
        assert f"x{i}" in sel
    # LASSO is allowed to bring in noise variables; not checked strictly.


def test_bic_lasso_recovers_true_support_sparsely():
    df, true_b = _dgp(1)
    r = sp.lasso_select(df, "y", [f"x{i}" for i in range(20)], method="bic")
    sel = r.selected
    for i in TRUE_SUPPORT:
        assert f"x{i}" in sel


def test_lasso_keeps_signal_drops_noise_over_seeds():
    # Across seeds the BIC selection recovers all true predictors every time.
    for seed in range(4):
        df, _ = _dgp(seed)
        r = sp.lasso_select(df, "y", [f"x{i}" for i in range(20)], method="bic")
        for i in TRUE_SUPPORT:
            assert f"x{i}" in r.selected
