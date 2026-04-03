"""
Instrumental Variables (2SLS) regression implementation

Implements Two-Stage Least Squares with proper standard error correction,
first-stage diagnostics, and overidentification tests.

References
----------
- Wooldridge, J. M. (2010). Econometric Analysis of Cross Section and Panel Data.
- Stock, J. H., & Yogo, M. (2005). Testing for Weak Instruments.
- Sargan, J. D. (1958). The Estimation of Economic Relationships Using
  Instrumental Variables.
"""

from typing import Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np
from scipy import stats
import warnings

from ..core.base import BaseModel, BaseEstimator
from ..core.results import EconometricResults
from ..core.utils import parse_formula


class IVEstimator(BaseEstimator):
    """
    Two-Stage Least Squares (2SLS) estimator

    Implements the standard 2SLS procedure with corrected standard errors
    and support for heteroskedasticity-robust and clustered inference.
    """

    def estimate(
        self,
        y: np.ndarray,
        X_exog: np.ndarray,
        X_endog: np.ndarray,
        Z: np.ndarray,
        robust: str = 'nonrobust',
        cluster: Optional[pd.Series] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Estimate IV parameters via 2SLS

        Parameters
        ----------
        y : np.ndarray, shape (n,)
            Dependent variable.
        X_exog : np.ndarray, shape (n, k1)
            Exogenous regressors (including constant if desired).
        X_endog : np.ndarray, shape (n, k2)
            Endogenous regressors.
        Z : np.ndarray, shape (n, m)
            Excluded instruments (must satisfy m >= k2).
        robust : str, default 'nonrobust'
            Standard-error type ('nonrobust', 'hc0', 'hc1', 'hc2', 'hc3').
        cluster : pd.Series, optional
            Cluster variable for clustered standard errors.

        Returns
        -------
        Dict[str, Any]
            Estimation results.
        """
        n = len(y)
        k2 = X_endog.shape[1]
        m = Z.shape[1]

        if m < k2:
            raise ValueError(
                f"Under-identified: {m} instruments for {k2} endogenous "
                f"variables. Need at least {k2} instruments."
            )

        # Full instrument matrix: [X_exog, Z]
        W = np.column_stack([X_exog, Z])

        # --- First stage ---
        # For each endogenous variable, regress on W
        X_endog_hat = np.empty_like(X_endog)
        first_stage_results = []

        for j in range(k2):
            WtW_inv = np.linalg.inv(W.T @ W)
            gamma_j = WtW_inv @ W.T @ X_endog[:, j]
            X_endog_hat[:, j] = W @ gamma_j

            # First-stage F-statistic for excluded instruments
            # Test H0: coefficients on excluded instruments = 0
            resid_full = X_endog[:, j] - W @ gamma_j
            # Restricted model: only exogenous regressors
            XeXe_inv = np.linalg.inv(X_exog.T @ X_exog)
            gamma_r = XeXe_inv @ X_exog.T @ X_endog[:, j]
            resid_restricted = X_endog[:, j] - X_exog @ gamma_r

            rss_full = resid_full @ resid_full
            rss_restricted = resid_restricted @ resid_restricted
            df_num = m  # number of excluded instruments
            df_denom = n - W.shape[1]

            if rss_full > 0 and df_denom > 0:
                f_stat = ((rss_restricted - rss_full) / df_num) / (rss_full / df_denom)
                f_pvalue = 1 - stats.f.cdf(f_stat, df_num, df_denom)
            else:
                f_stat = f_pvalue = np.nan

            first_stage_results.append({
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue,
                'partial_r_squared': 1 - rss_full / rss_restricted if rss_restricted > 0 else np.nan,
            })

        # --- Second stage ---
        # Regress y on [X_exog, X_endog_hat]
        X_hat = np.column_stack([X_exog, X_endog_hat])
        X_actual = np.column_stack([X_exog, X_endog])
        k = X_actual.shape[1]

        XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
        params = XhXh_inv @ X_hat.T @ y

        # CRITICAL: residuals use ACTUAL endogenous regressors, not fitted
        fitted_values = X_actual @ params
        residuals = y - fitted_values

        # --- Standard errors (corrected for 2SLS) ---
        # Variance uses X_hat in the bread but actual residuals in the meat
        if cluster is not None:
            var_cov = self._cluster_cov_2sls(
                X_hat, X_actual, residuals, XhXh_inv, cluster
            )
        elif robust != 'nonrobust':
            var_cov = self._robust_cov_2sls(
                X_hat, X_actual, residuals, XhXh_inv, robust
            )
        else:
            # Classical 2SLS standard errors
            sigma2 = np.sum(residuals**2) / (n - k)
            var_cov = sigma2 * XhXh_inv

        std_errors = np.sqrt(np.diag(var_cov))

        # --- Model diagnostics ---
        y_bar = np.mean(y)
        tss = np.sum((y - y_bar)**2)
        rss = np.sum(residuals**2)

        # R-squared can be negative for IV
        r_squared = 1 - rss / tss

        # --- Sargan overidentification test (if over-identified) ---
        sargan = self._sargan_test(residuals, W, m, k2) if m > k2 else None

        # --- Durbin-Wu-Hausman endogeneity test ---
        hausman = self._hausman_test(y, X_exog, X_endog, W, residuals)

        return {
            'params': params,
            'std_errors': std_errors,
            'var_cov': var_cov,
            'fitted_values': fitted_values,
            'residuals': residuals,
            'r_squared': r_squared,
            'nobs': n,
            'df_model': k - 1,
            'df_resid': n - k,
            'rss': rss,
            'tss': tss,
            'first_stage': first_stage_results,
            'sargan': sargan,
            'hausman': hausman,
            'n_instruments': m,
            'n_endogenous': k2,
        }

    def _robust_cov_2sls(
        self,
        X_hat: np.ndarray,
        X_actual: np.ndarray,
        residuals: np.ndarray,
        XhXh_inv: np.ndarray,
        robust_type: str,
    ) -> np.ndarray:
        """Heteroskedasticity-robust covariance for 2SLS."""
        n, k = X_actual.shape

        if robust_type == 'hc0':
            weights = residuals**2
        elif robust_type == 'hc1':
            weights = (n / (n - k)) * residuals**2
        elif robust_type == 'hc2':
            h = np.diag(X_hat @ XhXh_inv @ X_hat.T)
            weights = residuals**2 / (1 - h)
        elif robust_type == 'hc3':
            h = np.diag(X_hat @ XhXh_inv @ X_hat.T)
            weights = residuals**2 / (1 - h)**2
        else:
            raise ValueError(f"Unknown robust type: {robust_type}")

        meat = X_hat.T @ np.diag(weights) @ X_hat
        return XhXh_inv @ meat @ XhXh_inv

    def _cluster_cov_2sls(
        self,
        X_hat: np.ndarray,
        X_actual: np.ndarray,
        residuals: np.ndarray,
        XhXh_inv: np.ndarray,
        cluster: pd.Series,
    ) -> np.ndarray:
        """Clustered standard errors for 2SLS."""
        n, k = X_actual.shape
        clusters = cluster.unique()
        n_clusters = len(clusters)

        meat = np.zeros((k, k))
        for cid in clusters:
            idx = cluster == cid
            Xh_c = X_hat[idx]
            resid_c = residuals[idx]
            moments_c = (Xh_c * resid_c[:, np.newaxis]).sum(axis=0)
            meat += np.outer(moments_c, moments_c)

        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
        return correction * XhXh_inv @ meat @ XhXh_inv

    def _sargan_test(
        self,
        residuals: np.ndarray,
        W: np.ndarray,
        n_excluded: int,
        n_endog: int,
    ) -> Dict[str, float]:
        """
        Sargan test for overidentifying restrictions.

        H0: instruments are valid (uncorrelated with error term).
        """
        n = len(residuals)
        WtW_inv = np.linalg.inv(W.T @ W)
        P_W = W @ WtW_inv @ W.T

        stat = (residuals @ P_W @ residuals) / (residuals @ residuals / n)
        df = n_excluded - n_endog
        pvalue = 1 - stats.chi2.cdf(stat, df) if df > 0 else np.nan

        return {
            'statistic': stat,
            'pvalue': pvalue,
            'df': df,
        }

    def _hausman_test(
        self,
        y: np.ndarray,
        X_exog: np.ndarray,
        X_endog: np.ndarray,
        W: np.ndarray,
        iv_residuals: np.ndarray,
    ) -> Dict[str, float]:
        """
        Durbin-Wu-Hausman test for endogeneity.

        H0: OLS is consistent (endogenous regressors are actually exogenous).
        Regression-based version: include first-stage residuals in OLS,
        test their joint significance.
        """
        n = len(y)
        k2 = X_endog.shape[1]

        # First-stage residuals
        WtW_inv = np.linalg.inv(W.T @ W)
        v_hat = np.empty_like(X_endog)
        for j in range(k2):
            gamma_j = WtW_inv @ W.T @ X_endog[:, j]
            v_hat[:, j] = X_endog[:, j] - W @ gamma_j

        # Augmented regression: y ~ X_exog + X_endog + v_hat
        X_aug = np.column_stack([X_exog, X_endog, v_hat])
        X_orig = np.column_stack([X_exog, X_endog])

        try:
            # Full model
            XaXa_inv = np.linalg.inv(X_aug.T @ X_aug)
            beta_aug = XaXa_inv @ X_aug.T @ y
            resid_aug = y - X_aug @ beta_aug
            rss_aug = resid_aug @ resid_aug

            # Restricted model (OLS)
            XoXo_inv = np.linalg.inv(X_orig.T @ X_orig)
            beta_orig = XoXo_inv @ X_orig.T @ y
            resid_orig = y - X_orig @ beta_orig
            rss_orig = resid_orig @ resid_orig

            df_num = k2
            df_denom = n - X_aug.shape[1]

            if rss_aug > 0 and df_denom > 0:
                f_stat = ((rss_orig - rss_aug) / df_num) / (rss_aug / df_denom)
                f_pvalue = 1 - stats.f.cdf(f_stat, df_num, df_denom)
            else:
                f_stat = f_pvalue = np.nan

        except np.linalg.LinAlgError:
            f_stat = f_pvalue = np.nan

        return {
            'statistic': f_stat,
            'pvalue': f_pvalue,
            'df': k2,
        }


class IVRegression(BaseModel):
    """
    Instrumental Variables regression model (2SLS)

    Parameters
    ----------
    formula : str, optional
        Formula with IV syntax: ``"y ~ (endog ~ z1 + z2) + exog1 + exog2"``
    data : pd.DataFrame, optional
        Data containing all variables.
    y : np.ndarray, optional
        Dependent variable (alternative to formula).
    X_exog : np.ndarray, optional
        Exogenous regressors.
    X_endog : np.ndarray, optional
        Endogenous regressors.
    Z : np.ndarray, optional
        Excluded instruments.
    var_names : dict, optional
        Variable names: ``{'exog': [...], 'endog': [...], 'instruments': [...]}``.
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[np.ndarray] = None,
        X_exog: Optional[np.ndarray] = None,
        X_endog: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
        var_names: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.formula = formula
        self.data = data
        self.y = y
        self.X_exog = X_exog
        self.X_endog = X_endog
        self.Z = Z
        self.var_names = var_names
        self.estimator = IVEstimator()

    def _prepare_from_formula(self):
        """Parse formula and build matrices from data."""
        parsed = parse_formula(self.formula)

        if not parsed['endogenous'] or not parsed['instruments']:
            raise ValueError(
                "IV formula must specify endogenous variables and instruments. "
                "Use syntax: \"y ~ (endog ~ z1 + z2) + exog\""
            )

        self.dependent_var = parsed['dependent']
        exog_names = parsed['exogenous']
        endog_names = parsed['endogenous']
        instrument_names = parsed['instruments']

        # Check all variables exist
        all_vars = [self.dependent_var] + exog_names + endog_names + instrument_names
        missing = [v for v in all_vars if v not in self.data.columns]
        if missing:
            raise ValueError(f"Variables not found in data: {missing}")

        # Include extra columns (e.g. cluster) in clean data
        extra_cols = [c for c in self.data.columns if c not in all_vars]
        clean = self.data[all_vars + extra_cols].dropna(subset=all_vars)

        self.y = clean[self.dependent_var].values

        # Exogenous: always include constant + named exogenous
        if parsed['has_constant']:
            const = np.ones((len(clean), 1))
            if exog_names:
                self.X_exog = np.column_stack([const, clean[exog_names].values])
            else:
                self.X_exog = const
            self._exog_names = ['Intercept'] + exog_names
        else:
            self.X_exog = clean[exog_names].values
            self._exog_names = exog_names

        self.X_endog = clean[endog_names].values
        self.Z = clean[instrument_names].values

        self._endog_names = endog_names
        self._instrument_names = instrument_names
        self._clean_data = clean

    def fit(
        self,
        robust: str = 'nonrobust',
        cluster: Optional[str] = None,
        **kwargs,
    ) -> EconometricResults:
        """
        Fit the IV model via 2SLS

        Parameters
        ----------
        robust : str, default 'nonrobust'
            Standard-error type ('nonrobust', 'hc0', 'hc1', 'hc2', 'hc3').
        cluster : str, optional
            Variable name for clustering.

        Returns
        -------
        EconometricResults
        """
        if self.formula is not None and self.data is not None:
            self._prepare_from_formula()
        elif not (self.y is not None and self.X_exog is not None
                  and self.X_endog is not None and self.Z is not None):
            raise ValueError(
                "Provide either (formula, data) or (y, X_exog, X_endog, Z)"
            )
        else:
            self._exog_names = (
                self.var_names.get('exog', [f'exog{i}' for i in range(self.X_exog.shape[1])])
                if self.var_names else [f'exog{i}' for i in range(self.X_exog.shape[1])]
            )
            self._endog_names = (
                self.var_names.get('endog', [f'endog{i}' for i in range(self.X_endog.shape[1])])
                if self.var_names else [f'endog{i}' for i in range(self.X_endog.shape[1])]
            )
            self._instrument_names = (
                self.var_names.get('instruments', [f'z{i}' for i in range(self.Z.shape[1])])
                if self.var_names else [f'z{i}' for i in range(self.Z.shape[1])]
            )
            self.dependent_var = (
                self.var_names.get('dependent', 'y')
                if self.var_names else 'y'
            )

        # Cluster variable
        cluster_var = None
        if cluster and self.data is not None:
            if hasattr(self, '_clean_data'):
                cluster_var = self._clean_data[cluster]
            else:
                cluster_var = self.data[cluster]

        # Estimate
        results = self.estimator.estimate(
            self.y, self.X_exog, self.X_endog, self.Z,
            robust=robust, cluster=cluster_var, **kwargs,
        )

        # Build results object
        all_names = self._exog_names + self._endog_names
        params = pd.Series(results['params'], index=all_names)
        std_errors = pd.Series(results['std_errors'], index=all_names)

        model_info = {
            'model_type': 'IV-2SLS',
            'method': 'Two-Stage Least Squares',
            'robust': robust,
            'cluster': cluster,
        }

        data_info = {
            'nobs': results['nobs'],
            'df_model': results['df_model'],
            'df_resid': results['df_resid'],
            'dependent_var': self.dependent_var,
            'fitted_values': results['fitted_values'],
            'residuals': results['residuals'],
        }

        # Build diagnostics dict
        diagnostics = {
            'R-squared': results['r_squared'],
            'N instruments': results['n_instruments'],
            'N endogenous': results['n_endogenous'],
        }

        # First-stage F-statistics
        for j, fs in enumerate(results['first_stage']):
            endog_name = self._endog_names[j]
            diagnostics[f'First-stage F ({endog_name})'] = fs['f_statistic']
            diagnostics[f'First-stage F p-value ({endog_name})'] = fs['f_pvalue']
            diagnostics[f'Partial R² ({endog_name})'] = fs['partial_r_squared']

        # Weak instrument warning (Stock-Yogo rule of thumb: F < 10)
        for j, fs in enumerate(results['first_stage']):
            if fs['f_statistic'] < 10:
                endog_name = self._endog_names[j]
                warnings.warn(
                    f"Weak instrument warning: First-stage F-statistic for "
                    f"'{endog_name}' is {fs['f_statistic']:.2f} (< 10). "
                    f"See Stock & Yogo (2005).",
                    UserWarning,
                    stacklevel=2,
                )

        # Sargan test (over-identification)
        if results['sargan'] is not None:
            diagnostics['Sargan statistic'] = results['sargan']['statistic']
            diagnostics['Sargan p-value'] = results['sargan']['pvalue']
            diagnostics['Sargan df'] = results['sargan']['df']

        # Hausman test (endogeneity)
        if results['hausman'] is not None:
            diagnostics['Hausman F-stat'] = results['hausman']['statistic']
            diagnostics['Hausman p-value'] = results['hausman']['pvalue']

        # Store extra info for programmatic access
        self._first_stage = results['first_stage']
        self._sargan = results['sargan']
        self._hausman = results['hausman']
        self._instruments = self._instrument_names

        self._results = EconometricResults(
            params=params,
            std_errors=std_errors,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )

        self.is_fitted = True
        return self._results

    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Generate predictions from fitted IV model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if data is None:
            return self._results.fitted_values()
        raise NotImplementedError("Out-of-sample prediction not yet implemented")

    @property
    def first_stage(self) -> List[Dict[str, float]]:
        """First-stage diagnostics for each endogenous variable."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self._first_stage

    @property
    def sargan_test(self) -> Optional[Dict[str, float]]:
        """Sargan overidentification test results."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self._sargan

    @property
    def hausman_test(self) -> Dict[str, float]:
        """Durbin-Wu-Hausman endogeneity test results."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self._hausman


def ivreg(
    formula: str,
    data: pd.DataFrame,
    robust: str = 'nonrobust',
    cluster: Optional[str] = None,
    **kwargs,
) -> EconometricResults:
    """
    Instrumental variables regression (2SLS)

    Parameters
    ----------
    formula : str
        IV formula: ``"y ~ (endog ~ z1 + z2) + exog1 + exog2"``

        - Variables in parentheses before ``~``: endogenous regressors
        - Variables in parentheses after ``~``: excluded instruments
        - Variables outside parentheses: exogenous controls
    data : pd.DataFrame
        Data containing all variables.
    robust : str, default 'nonrobust'
        Standard-error type ('nonrobust', 'hc0', 'hc1', 'hc2', 'hc3').
    cluster : str, optional
        Variable name for clustered standard errors.

    Returns
    -------
    EconometricResults
        Fitted model results with IV diagnostics.

    Examples
    --------
    >>> # Classic returns to schooling (Card 1995)
    >>> result = ivreg("lwage ~ (educ ~ nearc4) + exper + expersq", data=df)
    >>> print(result.summary())

    >>> # Multiple endogenous variables
    >>> result = ivreg("y ~ (x1 + x2 ~ z1 + z2 + z3) + w1", data=df,
    ...               robust='hc1')

    Notes
    -----
    - First-stage F < 10 triggers a weak instrument warning (Stock & Yogo 2005).
    - Sargan test is reported when over-identified (more instruments than
      endogenous variables).
    - Hausman test compares IV and OLS to test for endogeneity.
    """
    model = IVRegression(formula=formula, data=data)
    return model.fit(robust=robust, cluster=cluster, **kwargs)
