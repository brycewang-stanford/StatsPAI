"""
FOCaL — Doubly Robust Functional CATE (arXiv 2602.11118, 2026).

Estimates a CATE when the outcome is itself a function (e.g. blood
glucose curve, daily order series, fMRI time series). For each
"function evaluation point" t, run a separate DR-Learner CATE; the
joint surface τ(x, t) is the functional CATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class FunctionalCATEResult:
    """Output of FOCaL functional CATE estimator."""
    cate_grid: np.ndarray            # (n_test, n_t)
    function_grid: np.ndarray        # (n_t,)
    se_grid: np.ndarray              # (n_test, n_t)
    n_train: int
    n_test: int

    def summary(self) -> str:
        return (
            "FOCaL Functional CATE\n"
            "=" * 42 + "\n"
            f"  Train / Test : {self.n_train} / {self.n_test}\n"
            f"  Function pts : {len(self.function_grid)}\n"
            f"  CATE shape   : {self.cate_grid.shape}\n"
            f"  Mean over t  : {self.cate_grid.mean():+.4f}\n"
            f"  Min / Max τ  : {self.cate_grid.min():+.4f} /"
            f" {self.cate_grid.max():+.4f}\n"
        )


def focal_cate(
    data: pd.DataFrame,
    y_columns: List[str],
    treat: str,
    covariates: List[str],
    test_data: Optional[pd.DataFrame] = None,
    seed: int = 0,
) -> FunctionalCATEResult:
    """
    Functional doubly-robust CATE estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Training data with outcome columns y_columns (one per
        function evaluation point t).
    y_columns : list of str
        Outcome columns; len = number of function points.
    treat : str
    covariates : list of str
    test_data : pd.DataFrame, optional
        Defaults to ``data``.
    seed : int

    Returns
    -------
    FunctionalCATEResult
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression
    df = data[y_columns + [treat] + list(covariates)].dropna() \
        .reset_index(drop=True)
    if df[treat].nunique() != 2:
        raise ValueError("FOCaL requires binary treatment.")
    n_train = len(df)
    test_df = test_data if test_data is not None else df
    test_df = test_df[list(covariates)].dropna().reset_index(drop=True)
    n_test = len(test_df)

    X = df[list(covariates)].to_numpy(float)
    Xt = test_df.to_numpy(float)
    D = df[treat].to_numpy(int)
    rng = np.random.default_rng(seed)

    # Single propensity for all t (D is the same)
    try:
        ps = LogisticRegression(max_iter=1000).fit(X, D).predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.02, 0.98)
    except Exception:
        ps = np.full(n_train, 0.5)

    cate_grid = np.zeros((n_test, len(y_columns)))
    se_grid = np.zeros((n_test, len(y_columns)))
    for j, ycol in enumerate(y_columns):
        Y = df[ycol].to_numpy(float)
        m1 = LinearRegression().fit(X[D == 1], Y[D == 1])
        m0 = LinearRegression().fit(X[D == 0], Y[D == 0])
        # DR pseudo-outcome
        mu1 = m1.predict(X)
        mu0 = m0.predict(X)
        psi = (mu1 - mu0
               + D * (Y - mu1) / ps
               - (1 - D) * (Y - mu0) / (1 - ps))
        # Regress psi on X to obtain τ̂(x); then predict on Xt
        try:
            cate_model = LinearRegression().fit(X, psi)
            cate_grid[:, j] = cate_model.predict(Xt)
            # Approx SE: std of psi residuals
            se_grid[:, j] = float(np.std(psi - cate_model.predict(X), ddof=1)) \
                            * np.ones(n_test)
        except Exception:
            cate_grid[:, j] = float(np.mean(psi))
            se_grid[:, j] = float(np.std(psi, ddof=1) / np.sqrt(n_train))

    return FunctionalCATEResult(
        cate_grid=cate_grid,
        function_grid=np.arange(len(y_columns)),
        se_grid=se_grid,
        n_train=n_train,
        n_test=n_test,
    )
