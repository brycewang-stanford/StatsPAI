"""
OLS regression implementation with comprehensive features
"""

from typing import Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
import warnings

from ..core.base import BaseModel, BaseEstimator
from ..core.results import EconometricResults
from ..core.utils import parse_formula, create_design_matrices, prepare_data
from ..exceptions import NumericalInstability


def _detect_perfect_collinearity(X: np.ndarray, var_names: List[str]) -> None:
    """Raise :class:`NumericalInstability` on an exactly rank-deficient design.

    Perfect collinearity leaves coefficients unidentified; without a guard the
    least-squares solve returns enormous garbage (e.g. ``1e14``) with no signal
    — a silent-failure violation of the "fail loudly" rule.

    Detection is deliberately **structural** (duplicate / proportional columns
    and zero-variance regressors) rather than conditioning-based. A singular-
    value / rank tolerance loose enough to catch real collinearity also flags
    legitimately ill-conditioned *full-rank* designs: the NIST StRD Filippelli
    benchmark has ``s_min/s_max ~ 6e-16`` — numerically *more* singular than an
    exactly duplicated column — yet it is full rank and must fit. Structural
    detection separates the two cleanly: the worst off-diagonal |correlation|
    across every NIST ill-conditioned design is ~0.999, far under the
    ``1 - 1e-8`` duplicate threshold here. The trade-off is that a general
    exact dependence among 3+ columns (not reducible to a pairwise duplicate or
    a constant column) is intentionally *not* auto-detected, because a detector
    that caught it could not also pass Filippelli.
    """
    n, k = X.shape
    names = list(var_names) if var_names is not None else [f"x{i}" for i in range(k)]

    # 1) Zero-variance non-intercept regressor: no identifying variation, and
    #    collinear with the intercept when one is present.
    for j in range(k):
        if names[j] == "Intercept":
            continue
        col = X[:, j]
        if np.ptp(col) <= 1e-12 * max(1.0, float(np.max(np.abs(col)))):
            raise NumericalInstability(
                f"Regressor '{names[j]}' is constant (no variation); its "
                f"coefficient is not identified — perfectly collinear with "
                f"the intercept.",
                recovery_hint=(
                    f"Drop '{names[j]}', or remove the intercept if it is the "
                    f"only regressor."
                ),
                diagnostics={"zero_variance_regressor": names[j]},
            )

    # 2) Duplicate / proportional columns (|corr| == 1), including
    #    complementary 0/1 dummies (the dummy-variable trap). Needs >=3 rows for
    #    a meaningful correlation; smaller-n degeneracy is caught elsewhere.
    if k >= 2 and n >= 3:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        for i in range(k):
            for j in range(i + 1, k):
                c = corr[i, j]
                if np.isfinite(c) and abs(c) >= 1.0 - 1e-8:
                    raise NumericalInstability(
                        f"Regressors '{names[i]}' and '{names[j]}' are "
                        f"perfectly collinear (|correlation| = {abs(c):.10f}); "
                        f"the design matrix is rank-deficient and their "
                        f"coefficients are not separately identified.",
                        recovery_hint=(f"Drop one of '{names[i]}' or '{names[j]}'."),
                        diagnostics={
                            "collinear_pair": [names[i], names[j]],
                            "abs_correlation": float(abs(c)),
                        },
                    )


def _numba_kernels():
    """Load accelerated kernels only when OLS is actually estimated."""
    from ..core._numba_kernels import (
        cluster_meat,
        hac_meat,
        ols_fit,
        sandwich_hc,
    )

    return ols_fit, sandwich_hc, cluster_meat, hac_meat


class OLSEstimator(BaseEstimator):
    """
    Ordinary Least Squares estimator with robust standard errors
    """

    def estimate(
        self,
        y: np.ndarray,
        X: np.ndarray,
        robust: str = "nonrobust",
        cluster: Optional[pd.Series] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Estimate OLS parameters

        Parameters
        ----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Independent variables (including constant if desired)
        robust : str, default 'nonrobust'
            Type of standard errors ('nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac')
        cluster : pd.Series, optional
            Cluster variable for clustered standard errors
        **kwargs
            Additional options

        Returns
        -------
        Dict[str, Any]
            Estimation results
        """
        n, k = X.shape

        # Fast OLS via Numba-accelerated kernel (graceful fallback)
        (
            _fast_ols,
            _fast_sandwich_hc,
            _fast_cluster_meat,
            _fast_hac_meat,
        ) = _numba_kernels()
        params, fitted_values, residuals = _fast_ols(X, y)

        # (X'X)^{-1} via the QR factor R (X = QR  =>  X'X = R'R), which keeps
        # the covariance accuracy tracking cond(X) rather than cond(X)**2.
        # Forming inv(X'X) directly squares the condition number and collapses
        # on ill-conditioned designs (see NIST StRD Filippelli/Wampler).
        try:
            _Q, R = np.linalg.qr(X)
            R_inv = np.linalg.solve(R, np.eye(R.shape[0]))
            XtX_inv = R_inv @ R_inv.T
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)
            warnings.warn("X'X matrix is singular, using pseudo-inverse")

        # Variance-covariance via accelerated sandwich kernels
        if cluster is not None:
            cluster_arr = np.asarray(cluster)
            meat = _fast_cluster_meat(X, residuals, cluster_arr)
            n_clusters = len(np.unique(cluster_arr))
            correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
            var_cov = correction * XtX_inv @ meat @ XtX_inv
        elif robust == "nonrobust":
            sigma2 = np.sum(residuals**2) / (n - k)
            var_cov = sigma2 * XtX_inv
        elif robust.lower() in ["hc0", "hc1", "hc2", "hc3"]:
            var_cov = _fast_sandwich_hc(X, residuals, XtX_inv, robust.lower())
        elif robust.lower() == "hac":
            lags = kwargs.get("lags", None)
            meat = _fast_hac_meat(X, residuals, lags)
            var_cov = XtX_inv @ meat @ XtX_inv
        else:
            raise ValueError(f"Unknown robust option: {robust}")

        std_errors = np.sqrt(np.diag(var_cov))

        # Model diagnostics
        tss = np.sum((y - np.mean(y)) ** 2)
        rss = np.sum(residuals**2)
        r_squared = 1 - rss / tss
        adj_r_squared = 1 - (rss / (n - k)) / (tss / (n - 1))

        # F-statistic (assuming constant in first column)
        if k > 1:
            r_squared_restricted = 0  # R² from constant-only model
            denom = (1 - r_squared) / (n - k)
            if denom <= 0:
                # Exact fit (R² == 1): F diverges. NIST StRD certifies this as
                # "Infinity" (e.g. Wampler1/2); report it without tripping a
                # divide-by-zero warning.
                f_stat = np.inf
                f_pvalue = 0.0
            else:
                f_stat = ((r_squared - r_squared_restricted) / (k - 1)) / denom
                f_pvalue = 1 - stats.f.cdf(f_stat, k - 1, n - k)
        else:
            f_stat = f_pvalue = np.nan

        return {
            "params": params,
            "std_errors": std_errors,
            "var_cov": var_cov,
            "fitted_values": fitted_values,
            "residuals": residuals,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "f_statistic": f_stat,
            "f_pvalue": f_pvalue,
            "nobs": n,
            "df_model": k - 1,
            "df_resid": n - k,
            "rss": rss,
            "tss": tss,
        }

    def _robust_cov_matrix(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        XtX_inv: np.ndarray,
        robust_type: str,
    ) -> np.ndarray:
        """Calculate heteroskedasticity-robust covariance matrix"""
        n, k = X.shape

        if robust_type == "hc0":
            # White (1980)
            weights = residuals**2
        elif robust_type == "hc1":
            # Degree of freedom correction
            weights = (n / (n - k)) * residuals**2
        elif robust_type == "hc2":
            # MacKinnon and White (1985)
            h = np.diag(X @ XtX_inv @ X.T)
            weights = residuals**2 / (1 - h)
        elif robust_type == "hc3":
            # Davidson and MacKinnon (1993)
            h = np.diag(X @ XtX_inv @ X.T)
            weights = residuals**2 / (1 - h) ** 2

        # Sandwich estimator
        meat = X.T @ np.diag(weights) @ X
        return XtX_inv @ meat @ XtX_inv

    def _hac_cov_matrix(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        XtX_inv: np.ndarray,
        lags: Optional[int] = None,
    ) -> np.ndarray:
        """Calculate HAC (Newey-West) covariance matrix"""
        n, k = X.shape

        if lags is None:
            # Automatic lag selection (Newey-West rule)
            lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

        # Calculate centered moments.  The HAC meat is intentionally
        # unnormalised so ``XtX_inv @ meat @ XtX_inv`` has the same scale as
        # HC and clustered covariance estimators.
        moments = X * residuals[:, np.newaxis]

        # Gamma_0 (contemporaneous covariance)
        gamma_0 = moments.T @ moments

        # Gamma_j for j = 1, ..., lags
        gamma_sum = gamma_0.copy()
        for j in range(1, lags + 1):
            gamma_j = moments[j:].T @ moments[:-j]
            weight = 1 - j / (lags + 1)  # Bartlett kernel
            gamma_sum += weight * (gamma_j + gamma_j.T)

        return XtX_inv @ gamma_sum @ XtX_inv

    def _cluster_cov_matrix(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        XtX_inv: np.ndarray,
        cluster: pd.Series,
    ) -> np.ndarray:
        """Calculate clustered standard errors"""
        n, k = X.shape

        # Get unique clusters
        clusters = cluster.unique()
        n_clusters = len(clusters)

        # Calculate cluster sum of moments
        meat = np.zeros((k, k))
        for cluster_id in clusters:
            cluster_idx = cluster == cluster_id
            X_c = X[cluster_idx]
            resid_c = residuals[cluster_idx]
            moments_c = (X_c * resid_c[:, np.newaxis]).sum(axis=0)
            meat += np.outer(moments_c, moments_c)

        # Finite sample correction
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))

        return correction * XtX_inv @ meat @ XtX_inv


class OLSRegression(BaseModel):
    """
    OLS regression model with comprehensive functionality
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        var_names: Optional[List[str]] = None,
    ):
        """
        Initialize OLS regression

        Parameters
        ----------
        formula : str, optional
            Regression formula (e.g., "y ~ x1 + x2")
        data : pd.DataFrame, optional
            Data containing variables
        y : np.ndarray, optional
            Dependent variable (alternative to formula)
        X : np.ndarray, optional
            Independent variables (alternative to formula)
        var_names : List[str], optional
            Variable names when using y, X directly
        """
        super().__init__()

        self.formula = formula
        self.data = data
        self.y = y
        self.X = X
        self.var_names = var_names
        self.estimator = OLSEstimator()

    def fit(
        self, robust: str = "nonrobust", cluster: Optional[str] = None, **kwargs
    ) -> EconometricResults:
        """
        Fit the OLS model

        Parameters
        ----------
        robust : str, default 'nonrobust'
            Type of standard errors
        cluster : str, optional
            Variable name for clustering
        **kwargs
            Additional options

        Returns
        -------
        EconometricResults
            Fitted model results
        """
        # Prepare data
        if self.formula is not None and self.data is not None:
            y_df, X_df = create_design_matrices(self.formula, self.data)
            self.y = y_df.values.ravel()
            self.X = X_df.values
            self.var_names = list(X_df.columns)
            self.dependent_var = y_df.columns[0]
        elif self.y is not None and self.X is not None:
            if self.var_names is None:
                self.var_names = [f"x{i}" for i in range(self.X.shape[1])]
            self.dependent_var = "y"
        else:
            raise ValueError("Must provide either (formula, data) or (y, X)")

        # Fail loudly on an exactly rank-deficient design rather than returning
        # unidentified garbage coefficients.
        _detect_perfect_collinearity(self.X, self.var_names)

        # Handle clustering
        cluster_var = None
        if cluster and self.data is not None:
            cluster_var = self.data[cluster]

        # Estimate model
        results = self.estimator.estimate(
            self.y, self.X, robust=robust, cluster=cluster_var, **kwargs
        )

        # Create results object
        params = pd.Series(results["params"], index=self.var_names)
        std_errors = pd.Series(results["std_errors"], index=self.var_names)

        model_info = {
            "model_type": "OLS",
            "method": "Least Squares",
            "robust": robust,
            "cluster": cluster,
        }

        data_info = {
            "nobs": results["nobs"],
            "df_model": results["df_model"],
            "df_resid": results["df_resid"],
            "dependent_var": self.dependent_var,
            "fitted_values": results["fitted_values"],
            "residuals": results["residuals"],
            "X": self.X,
            "y": self.y,
            "var_cov": results.get("var_cov"),
            "var_names": self.var_names,
        }

        diagnostics = {
            "R-squared": results["r_squared"],
            "Adj. R-squared": results["adj_r_squared"],
            "F-statistic": results["f_statistic"],
            "Prob (F-statistic)": results["f_pvalue"],
            "Log-Likelihood": -0.5
            * results["nobs"]
            * (np.log(2 * np.pi * results["rss"] / results["nobs"]) + 1),
            "AIC": results["nobs"] * np.log(results["rss"] / results["nobs"])
            + 2 * (results["df_model"] + 1),
            "BIC": results["nobs"] * np.log(results["rss"] / results["nobs"])
            + np.log(results["nobs"]) * (results["df_model"] + 1),
        }

        self._results = EconometricResults(
            params=params,
            std_errors=std_errors,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )

        self.is_fitted = True
        return self._results

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        what: str = "mean",
        alpha: float = 0.05,
        return_df: bool = False,
    ) -> "np.ndarray | pd.DataFrame":
        """Generate predictions from the fitted OLS model.

        Parameters
        ----------
        data : pd.DataFrame, optional
            New data at which to predict. If ``None``, returns the
            in-sample fitted values.
        what : {"mean", "confidence", "prediction"}, default "mean"
            - ``"mean"`` — point predictions only (default).
            - ``"confidence"`` — point + ``(1-alpha)`` confidence interval
              for the conditional mean ``E[y | x]``.
            - ``"prediction"`` — point + ``(1-alpha)`` prediction interval
              for a new observation (wider than the CI by ``sqrt(sigma^2)``).
        alpha : float, default 0.05
            Significance level for the interval.
        return_df : bool, default False
            Return a DataFrame with columns ``["yhat", "lower", "upper"]``.
            Ignored (forces True) when ``what != "mean"``.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Point predictions, optionally with interval columns.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # In-sample path
        if data is None:
            yhat = np.asarray(self._results.fitted_values()).ravel()
            if what == "mean" and not return_df:
                return yhat
            # Fall through to interval machinery using the training design X.
            X_new = self.X
        else:
            if self.formula is None:
                raise ValueError(
                    "Out-of-sample prediction requires the model to have been fit "
                    "with a formula (not raw y, X arrays)."
                )
            # Build X from the RHS of the formula. patsy's dmatrices() wants
            # the LHS variable present in `data`; at prediction time we only
            # have the regressors, so use dmatrix on the RHS only.
            from patsy import dmatrix

            rhs = self.formula.split("~", 1)[1].strip()
            X_df = dmatrix(rhs, data, return_type="dataframe")
            missing = [nm for nm in self.var_names if nm not in X_df.columns]
            if missing:
                raise ValueError(
                    f"New data is missing columns produced by the formula: {missing}"
                )
            X_new = X_df[self.var_names].values
            params = np.asarray(self._results.params)
            yhat = X_new @ params

        if what == "mean" and not return_df:
            return yhat

        params = np.asarray(self._results.params)
        # Covariance of the estimated coefficients
        cov = None
        diag = (
            self._results.data_info.get("cov_params", None)
            if hasattr(self._results, "data_info")
            else None
        )
        if diag is not None:
            cov = np.asarray(diag)
        else:
            # Reconstruct from std_errors (diagonal approximation if full cov missing)
            se = np.asarray(self._results.std_errors)
            cov = np.diag(se**2)

        # var(x' beta) = x' Σ x
        var_mean = np.einsum("ij,jk,ik->i", X_new, cov, X_new)
        var_mean = np.maximum(var_mean, 0.0)
        se_mean = np.sqrt(var_mean)

        df_resid = self._results.data_info.get("df_resid", np.inf)
        t_crit = stats.t.ppf(1 - alpha / 2, df_resid)

        if what == "confidence":
            lower = yhat - t_crit * se_mean
            upper = yhat + t_crit * se_mean
        elif what == "prediction":
            sigma2 = self._results.diagnostics.get("sigma2", None)
            if sigma2 is None:
                # fall back to residual variance
                e = np.asarray(self._results.data_info.get("residuals", []))
                sigma2 = float(e @ e) / df_resid if len(e) else 0.0
            se_pred = np.sqrt(var_mean + float(sigma2))
            lower = yhat - t_crit * se_pred
            upper = yhat + t_crit * se_pred
        else:
            raise ValueError(
                f"`what` must be 'mean', 'confidence', or 'prediction'; got {what!r}"
            )

        out = pd.DataFrame({"yhat": yhat, "lower": lower, "upper": upper})
        return out


def regress(
    formula: str,
    data: pd.DataFrame,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    **kwargs,
) -> EconometricResults:
    """
    Convenient function for OLS regression

    Parameters
    ----------
    formula : str
        Regression formula
    data : pd.DataFrame
        Data containing variables
    robust : str, default 'nonrobust'
        Type of standard errors
    cluster : str, optional
        Variable name for clustering
    **kwargs
        Additional options

    Returns
    -------
    EconometricResults
        Fitted model results

    Examples
    --------
    >>> results = regress("wage ~ education + experience", data=df)
    >>> print(results.summary())

    >>> results = regress("wage ~ education + experience", data=df,
    ...                   robust='hc1', cluster='firm_id')
    """
    # --- Input validation (Stata-quality error messages) ---
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"'data' must be a pandas DataFrame, got {type(data).__name__}. "
            f"Example: sp.regress('y ~ x', data=df)"
        )
    if data.empty:
        raise ValueError("DataFrame is empty — no observations to regress.")
    # Check formula variables exist in data
    if "~" in formula:
        import re

        lhs, rhs = formula.split("~", 1)
        # Strip function calls: C(...), I(...), np.log(...), bs(...), etc.
        rhs_stripped = re.sub(r"[A-Za-z_][\w.]*\s*\([^)]*\)", "", rhs)
        # Split on operators
        rhs_stripped = re.sub(r"[+*:\-]", " ", rhs_stripped)
        tokens = rhs_stripped.split()
        # Keep only bare column identifiers (no digits, no '1'/'0')
        bare_vars = [
            v for v in tokens if re.match(r"^[A-Za-z_]\w*$", v) and v not in ("1", "0")
        ]
        # Include LHS dep var
        dep_check = lhs.strip()
        all_vars = (
            [dep_check] if re.match(r"^[A-Za-z_]\w*$", dep_check) else []
        ) + bare_vars
        missing = [v for v in all_vars if v not in data.columns]
        if missing:
            available = ", ".join(sorted(data.columns)[:10])
            raise ValueError(
                f"Variable(s) not found in data: {missing}. "
                f"Available columns: {available}"
                + (" ..." if len(data.columns) > 10 else "")
            )
    # Check for all-NaN outcome
    dep_var = formula.split("~")[0].strip()
    if dep_var in data.columns and data[dep_var].isna().all():
        raise ValueError(
            f"Outcome variable '{dep_var}' is entirely NaN — "
            f"cannot estimate regression."
        )

    model = OLSRegression(formula=formula, data=data)
    _result = model.fit(robust=robust, cluster=cluster, **kwargs)
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.regress",
            params={
                "formula": formula,
                "robust": robust,
                "cluster": cluster,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("weights", "vcov", "se_type")
                },
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover — provenance must never break fit
        pass
    return _result
