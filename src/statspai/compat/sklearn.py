"""
Scikit-learn compatible wrappers for StatsPAI estimators.

Each wrapper implements the sklearn ``BaseEstimator`` / ``RegressorMixin``
interface so that StatsPAI models can be used in pipelines, cross-validation,
and grid search while retaining full econometric diagnostics via
``.results_``.

Design principles
-----------------
- ``fit(X, y)`` follows sklearn convention (numpy arrays, no formula).
- After fitting, ``.results_`` holds the native ``EconometricResults``.
- ``get_params`` / ``set_params`` work correctly for ``GridSearchCV``.
- ``predict(X)`` returns point predictions as a 1-d numpy array.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore[import-untyped]
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Provide stubs so the module can be imported even without sklearn
    class BaseEstimator:  # type: ignore[no-redef]
        pass
    class RegressorMixin:  # type: ignore[no-redef]
        pass


def _check_sklearn():
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for the compat module.\n"
            "Install it with: pip install scikit-learn"
        )


class SklearnOLS(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible OLS with robust/clustered standard errors.

    Parameters
    ----------
    robust : str
        'nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac'.
    add_constant : bool
        If True, prepend a column of ones.

    Attributes (after fit)
    ----------------------
    results_ : EconometricResults
    coef_ : ndarray
    intercept_ : float
    """

    def __init__(self, robust: str = "nonrobust", add_constant: bool = True):
        self.robust = robust
        self.add_constant = add_constant

    def fit(self, X, y, **fit_params):
        _check_sklearn()
        from ..regression.ols import OLSEstimator
        from ..core.results import EconometricResults

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if self.add_constant:
            X = np.column_stack([np.ones(len(X)), X])

        est = OLSEstimator()
        res = est.estimate(y, X, robust=self.robust, **fit_params)

        self.coef_ = res["params"][1:] if self.add_constant else res["params"]
        self.intercept_ = res["params"][0] if self.add_constant else 0.0

        names = [f"x{i}" for i in range(len(res["params"]))]
        if self.add_constant:
            names[0] = "const"
        self.results_ = EconometricResults(
            params=pd.Series(res["params"], index=names),
            std_errors=pd.Series(res["std_errors"], index=names),
            model_info={"model_type": "OLS", "robust": self.robust},
            data_info={
                "nobs": res["nobs"],
                "df_model": res["df_model"],
                "df_resid": res["df_resid"],
                "fitted_values": res["fitted_values"],
                "residuals": res["residuals"],
            },
            diagnostics={
                "R-squared": res["r_squared"],
                "Adj. R-squared": res["adj_r_squared"],
            },
        )
        self._X_fit = X
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.add_constant:
            X = np.column_stack([np.ones(len(X)), X])
        return X @ np.concatenate([[self.intercept_], self.coef_]) if self.add_constant else X @ self.coef_

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.add_constant:
            X = np.column_stack([np.ones(len(X)), X])
            params = np.concatenate([[self.intercept_], self.coef_])
        else:
            params = self.coef_
        return X @ params


class SklearnIV(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible 2SLS / IV estimator.

    Parameters
    ----------
    n_instruments : int
        Number of columns at the *end* of X that are instruments
        (excluded from the second stage).
    add_constant : bool
        Prepend constant column.

    Notes
    -----
    In ``fit(X, y)`` the last ``n_instruments`` columns of *X* are
    treated as excluded instruments.
    """

    def __init__(self, n_instruments: int = 1, add_constant: bool = True):
        self.n_instruments = n_instruments
        self.add_constant = add_constant

    def fit(self, X, y, **fit_params):
        _check_sklearn()
        from ..regression.iv import IVRegression

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_endog = 1  # simplification: first non-instrument column is endogenous
        k = X.shape[1]
        n_exog = k - self.n_instruments - n_endog

        # Build DataFrame for the IV machinery
        cols = [f"exog_{i}" for i in range(n_exog)]
        cols += [f"endog_{i}" for i in range(n_endog)]
        cols += [f"instr_{i}" for i in range(self.n_instruments)]
        df = pd.DataFrame(X, columns=cols)
        df["__y__"] = y

        endog_cols = [c for c in cols if c.startswith("endog_")]
        instr_cols = [c for c in cols if c.startswith("instr_")]
        exog_cols = [c for c in cols if c.startswith("exog_")]

        # Build formula
        exog_str = " + ".join(exog_cols) if exog_cols else ""
        endog_str = " + ".join(endog_cols)
        instr_str = " + ".join(instr_cols)
        formula = f"__y__ ~ ({endog_str} ~ {instr_str})"
        if exog_str:
            formula += f" + {exog_str}"

        model = IVRegression(formula=formula, data=df)
        self.results_ = model.fit()
        self.coef_ = self.results_.params.values
        self.intercept_ = 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Use only the non-instrument columns
        k_used = X.shape[1] - self.n_instruments
        return X[:, :k_used] @ self.coef_[:k_used]


class SklearnDML(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible Double/Debiased ML wrapper.

    Parameters
    ----------
    n_folds : int
        Number of cross-fitting folds.
    model_y : object
        sklearn estimator for the outcome model.
    model_t : object
        sklearn estimator for the treatment model.

    Notes
    -----
    ``fit(X, y)`` expects the *first* column of X to be the treatment
    variable, remaining columns are controls/confounders.
    """

    def __init__(self, n_folds: int = 5, model_y=None, model_t=None):
        self.n_folds = n_folds
        self.model_y = model_y
        self.model_t = model_t

    def fit(self, X, y, **fit_params):
        _check_sklearn()
        from ..dml.double_ml import DoubleML

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        treatment = X[:, 0]
        controls = X[:, 1:]
        col_names = [f"control_{i}" for i in range(controls.shape[1])]

        df = pd.DataFrame(controls, columns=col_names)
        df["__y__"] = y
        df["__treat__"] = treatment

        model = DoubleML(
            data=df,
            y="__y__",
            treat="__treat__",
            covariates=col_names,
            n_folds=self.n_folds,
        )
        if self.model_y is not None:
            model.model_y = self.model_y
        if self.model_t is not None:
            model.model_t = self.model_t

        self.results_ = model.fit()
        self.coef_ = np.array([self.results_.params.iloc[0]])
        self.intercept_ = 0.0
        self.ate_ = self.coef_[0]
        return self

    def predict(self, X):
        # DML estimates a constant treatment effect
        return np.full(X.shape[0], self.ate_)


class SklearnCausalForest(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible Causal Forest wrapper.

    ``fit(X, y)`` expects the *first* column of X to be the binary
    treatment indicator.  ``predict(X)`` returns CATE estimates.

    Parameters
    ----------
    n_trees : int
    min_leaf_size : int
    """

    def __init__(self, n_trees: int = 100, min_leaf_size: int = 5):
        self.n_trees = n_trees
        self.min_leaf_size = min_leaf_size

    def fit(self, X, y, **fit_params):
        _check_sklearn()
        from ..causal.causal_forest import CausalForest

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        treatment = X[:, 0]
        covariates = X[:, 1:]
        col_names = [f"x{i}" for i in range(covariates.shape[1])]

        df = pd.DataFrame(covariates, columns=col_names)
        df["__y__"] = y
        df["__treat__"] = treatment

        formula = f"__y__ ~ __treat__ | {' + '.join(col_names)}"
        self.cf_ = CausalForest(
            formula=formula,
            data=df,
            n_trees=self.n_trees,
            min_leaf_size=self.min_leaf_size,
        )
        self.results_ = self.cf_.fit()
        self.coef_ = np.array([self.results_.params.get("ATE", 0.0)])
        self.intercept_ = 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        covariates = X[:, 1:]
        if hasattr(self.cf_, "predict_cate"):
            return self.cf_.predict_cate(covariates)
        return np.full(X.shape[0], self.coef_[0])
