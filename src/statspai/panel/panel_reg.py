"""
Panel regression with Stata-style API, wrapping linearmodels.

Provides a unified interface for fixed effects, random effects, between,
and first-difference estimators with flexible standard error options.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from ..core.results import EconometricResults


def panel(
    data: pd.DataFrame,
    formula: str,
    entity: str,
    time: str,
    method: str = 'fe',
    robust: str = 'nonrobust',
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Panel regression with Stata-style syntax.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data (long format).
    formula : str
        Regression formula: ``"y ~ x1 + x2"``.
    entity : str
        Entity (individual/unit) identifier column.
    time : str
        Time period column.
    method : str, default 'fe'
        Estimation method:
        - 'fe' : Fixed Effects (within estimator)
        - 're' : Random Effects (GLS)
        - 'be' : Between estimator
        - 'fd' : First Differences
        - 'pooled' : Pooled OLS
    robust : str, default 'nonrobust'
        Standard errors: 'nonrobust', 'robust' (HC1), or 'kernel'.
    cluster : str, optional
        Cluster variable for standard errors. If 'entity', clusters by
        entity. If 'time', clusters by time period.
    weights : str, optional
        Weight variable name.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> # Fixed Effects
    >>> result = panel(df, "wage ~ experience + tenure",
    ...               entity='worker', time='year', method='fe')
    >>> print(result.summary())

    >>> # Random Effects with clustered SE
    >>> result = panel(df, "wage ~ experience + tenure",
    ...               entity='worker', time='year', method='re',
    ...               cluster='entity')

    >>> # Pooled OLS with robust SE
    >>> result = panel(df, "wage ~ experience + tenure",
    ...               entity='worker', time='year', method='pooled',
    ...               robust='robust')
    """
    model = PanelRegression(
        data=data, formula=formula, entity=entity, time=time,
        method=method, robust=robust, cluster=cluster,
        weights=weights, alpha=alpha,
    )
    return model.fit()


class PanelRegression:
    """
    Panel regression estimator wrapping linearmodels.
    """

    _METHOD_MAP = {
        'fe': 'PanelOLS',
        'fixed_effects': 'PanelOLS',
        're': 'RandomEffects',
        'random_effects': 'RandomEffects',
        'be': 'BetweenOLS',
        'between': 'BetweenOLS',
        'fd': 'FirstDifferenceOLS',
        'first_difference': 'FirstDifferenceOLS',
        'pooled': 'PooledOLS',
    }

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        entity: str,
        time: str,
        method: str = 'fe',
        robust: str = 'nonrobust',
        cluster: Optional[str] = None,
        weights: Optional[str] = None,
        alpha: float = 0.05,
    ):
        self.data = data.copy()
        self.formula = formula
        self.entity = entity
        self.time = time
        self.method = method.lower()
        self.robust = robust
        self.cluster = cluster
        self.weights = weights
        self.alpha = alpha

        self._validate()

    def _validate(self):
        if self.method not in self._METHOD_MAP:
            valid = list(self._METHOD_MAP.keys())
            raise ValueError(f"method must be one of {valid}, got '{self.method}'")

        # Parse formula to check variables
        if '~' not in self.formula:
            raise ValueError("Formula must contain '~'")

        dep, indep = self.formula.split('~', 1)
        self._dep_var = dep.strip()
        self._indep_vars = [v.strip() for v in indep.split('+') if v.strip()]

        all_cols = [self._dep_var] + self._indep_vars + [self.entity, self.time]
        for col in all_cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")

    def fit(self) -> EconometricResults:
        """Fit the panel model and return results."""
        try:
            import linearmodels.panel as lp
        except ImportError:
            raise ImportError(
                "linearmodels required for panel regression. "
                "Install: pip install linearmodels"
            )

        # Set MultiIndex for linearmodels
        panel_data = self.data.set_index([self.entity, self.time])

        # Build linearmodels formula (uses ~ syntax)
        dep = panel_data[self._dep_var]
        exog = panel_data[self._indep_vars]

        # Add constant for models that need it (not FE)
        from linearmodels.panel import PanelOLS, RandomEffects, BetweenOLS
        from linearmodels.panel import FirstDifferenceOLS, PooledOLS
        from statsmodels.tools import add_constant

        model_class_name = self._METHOD_MAP[self.method]
        model_classes = {
            'PanelOLS': PanelOLS,
            'RandomEffects': RandomEffects,
            'BetweenOLS': BetweenOLS,
            'FirstDifferenceOLS': FirstDifferenceOLS,
            'PooledOLS': PooledOLS,
        }
        ModelClass = model_classes[model_class_name]

        # FE and FD don't use constant; others do
        if self.method in ('fe', 'fixed_effects', 'fd', 'first_difference'):
            exog_final = exog
            fit_kwargs = {}
            if self.method in ('fe', 'fixed_effects'):
                fit_kwargs['entity_effects'] = True
        else:
            exog_final = add_constant(exog)

        # Create and fit model
        if self.method in ('fe', 'fixed_effects'):
            lm_model = ModelClass(dep, exog_final, entity_effects=True)
        else:
            lm_model = ModelClass(dep, exog_final)

        # Covariance options
        cov_kwargs = self._build_cov_kwargs()
        lm_result = lm_model.fit(**cov_kwargs)

        # Convert to StatsPAI EconometricResults
        return self._convert_results(lm_result)

    def _build_cov_kwargs(self) -> Dict[str, Any]:
        """Build covariance estimation keyword arguments."""
        if self.cluster == 'entity':
            return {'cov_type': 'clustered', 'cluster_entity': True}
        elif self.cluster == 'time':
            return {'cov_type': 'clustered', 'cluster_time': True}
        elif self.cluster:
            return {'cov_type': 'clustered', 'cluster_entity': True}
        elif self.robust == 'robust':
            return {'cov_type': 'robust'}
        elif self.robust == 'kernel':
            return {'cov_type': 'kernel'}
        return {'cov_type': 'unadjusted'}

    def _convert_results(self, lm_result) -> EconometricResults:
        """Convert linearmodels result to EconometricResults."""
        params = lm_result.params
        std_errors = lm_result.std_errors

        method_names = {
            'fe': 'Panel FE', 'fixed_effects': 'Panel FE',
            're': 'Panel RE', 'random_effects': 'Panel RE',
            'be': 'Panel Between', 'between': 'Panel Between',
            'fd': 'Panel FD', 'first_difference': 'Panel FD',
            'pooled': 'Pooled OLS',
        }

        model_info = {
            'model_type': method_names.get(self.method, self.method),
            'method': self._METHOD_MAP[self.method],
            'robust': self.robust,
            'cluster': self.cluster,
        }

        data_info = {
            'nobs': int(lm_result.nobs),
            'df_model': int(lm_result.df_model) if hasattr(lm_result, 'df_model') and not isinstance(lm_result.df_model, tuple) else len(params) - 1,
            'df_resid': int(lm_result.df_resid),
            'dependent_var': self._dep_var,
            'fitted_values': lm_result.fitted_values.values.ravel(),
            'residuals': lm_result.resids.values.ravel(),
        }

        diagnostics = {
            'R-squared': float(lm_result.rsquared),
        }

        # Add entity/time info
        if hasattr(lm_result, 'entity_info'):
            diagnostics['N entities'] = lm_result.entity_info.total
        if hasattr(lm_result, 'time_info'):
            diagnostics['N time periods'] = lm_result.time_info.total

        # F-statistic
        if hasattr(lm_result, 'f_statistic') and lm_result.f_statistic is not None:
            diagnostics['F-statistic'] = float(lm_result.f_statistic.stat)
            diagnostics['F p-value'] = float(lm_result.f_statistic.pval)

        return EconometricResults(
            params=params,
            std_errors=std_errors,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )
