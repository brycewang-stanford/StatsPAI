"""
Unified panel regression with Stata-style API.

Provides a single entry point ``panel()`` for all static and dynamic
panel estimators, with panel-specific diagnostics on the result object.

Estimators
----------
- ``'fe'``         Fixed Effects (within estimator)
- ``'re'``         Random Effects (GLS)
- ``'be'``         Between estimator
- ``'fd'``         First Differences
- ``'pooled'``     Pooled OLS
- ``'twoway'``     Two-way FE (entity + time)
- ``'mundlak'``    Correlated Random Effects (Mundlak 1978)
- ``'chamberlain'``  Chamberlain (1982) CRE
- ``'ab'``         Arellano-Bond difference GMM
- ``'system'``     Blundell-Bond system GMM

References
----------
Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data.
Mundlak, Y. (1978). "On the Pooling of Time Series and Cross Section Data."
Chamberlain, G. (1982). "Multivariate Regression Models for Panel Data."
Arellano, M. and Bond, S. (1991). "Some Tests of Specification for Panel Data."
Blundell, R. and Bond, S. (1998). "Initial Conditions and Moment Restrictions."
"""

from typing import Optional, List, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import EconometricResults


# ======================================================================
# PanelResults — extends EconometricResults with panel diagnostics
# ======================================================================

class PanelResults(EconometricResults):
    """
    Panel regression results with built-in diagnostics.

    Extends EconometricResults with panel-specific tests that can be
    called directly on the result object:

    >>> result = sp.panel(df, "y ~ x1 + x2", entity='id', time='t')
    >>> result.hausman_test()        # FE vs RE
    >>> result.bp_lm_test()          # Pooled vs RE (Breusch-Pagan LM)
    >>> result.f_test_effects()      # Joint significance of entity FE
    >>> result.compare('re')         # Compare with RE side by side
    """

    def __init__(
        self,
        params: pd.Series,
        std_errors: pd.Series,
        model_info: Dict[str, Any],
        data_info: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        *,
        _panel_data: Optional[pd.DataFrame] = None,
        _formula: Optional[str] = None,
        _entity: Optional[str] = None,
        _time: Optional[str] = None,
        _dep_var: Optional[str] = None,
        _indep_vars: Optional[List[str]] = None,
        _method: Optional[str] = None,
        _lm_result: Optional[Any] = None,
    ):
        super().__init__(params, std_errors, model_info, data_info, diagnostics)
        self._panel_data = _panel_data
        self._formula = _formula
        self._entity = _entity
        self._time = _time
        self._dep_var = _dep_var
        self._indep_vars = _indep_vars
        self._method = _method
        self._lm_result = _lm_result

    # ------------------------------------------------------------------
    # Hausman test: FE vs RE
    # ------------------------------------------------------------------

    def hausman_test(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Hausman (1978) specification test: FE vs RE.

        Under H0 (RE consistent), both FE and RE are consistent but RE
        is efficient. Under H1, only FE is consistent.

        Returns
        -------
        dict
            'statistic', 'df', 'pvalue', 'recommendation', 'interpretation'
        """
        if self._panel_data is None:
            raise ValueError("Panel data not stored — cannot run Hausman test.")
        from .panel_diagnostics import _hausman_from_data
        return _hausman_from_data(
            self._panel_data, self._dep_var, self._indep_vars,
            self._entity, self._time, alpha,
        )

    # ------------------------------------------------------------------
    # Breusch-Pagan LM test: Pooled OLS vs RE
    # ------------------------------------------------------------------

    def bp_lm_test(self) -> Dict[str, Any]:
        """
        Breusch-Pagan (1980) Lagrange Multiplier test for random effects.

        Tests H0: Var(alpha_i) = 0 (Pooled OLS is appropriate)
        vs   H1: Var(alpha_i) > 0 (Random Effects needed).

        Returns
        -------
        dict
            'statistic', 'df', 'pvalue', 'recommendation', 'interpretation'
        """
        if self._panel_data is None:
            raise ValueError("Panel data not stored — cannot run BP-LM test.")
        from .panel_diagnostics import _bp_lm_test
        return _bp_lm_test(
            self._panel_data, self._dep_var, self._indep_vars,
            self._entity, self._time,
        )

    # ------------------------------------------------------------------
    # F-test for entity effects
    # ------------------------------------------------------------------

    def f_test_effects(self) -> Dict[str, Any]:
        """
        F-test for joint significance of entity fixed effects.

        Tests H0: all alpha_i = 0 (entity effects not needed).

        Returns
        -------
        dict
            'statistic', 'df1', 'df2', 'pvalue', 'interpretation'
        """
        if self._panel_data is None:
            raise ValueError("Panel data not stored — cannot run F-test.")
        from .panel_diagnostics import _f_test_effects
        return _f_test_effects(
            self._panel_data, self._dep_var, self._indep_vars,
            self._entity, self._time,
        )

    # ------------------------------------------------------------------
    # Pesaran CD test for cross-sectional dependence
    # ------------------------------------------------------------------

    def pesaran_cd_test(self) -> Dict[str, Any]:
        """
        Pesaran (2004) CD test for cross-sectional dependence in residuals.

        Returns
        -------
        dict
            'statistic', 'pvalue', 'interpretation'
        """
        if self._lm_result is None:
            raise ValueError("linearmodels result not stored — cannot run CD test.")
        from .panel_diagnostics import _pesaran_cd
        resids = self._lm_result.resids
        return _pesaran_cd(resids, self._entity, self._time, self._panel_data)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, type: str = 'coef', **kwargs):
        """
        Generate panel-specific plots.

        Parameters
        ----------
        type : str
            ``'coef'``      — Coefficient forest plot (default)
            ``'effects'``   — Distribution of entity fixed effects
            ``'residuals'`` — Residual diagnostics (2x2 grid)
            ``'hausman'``   — Visual FE vs RE comparison
        **kwargs
            Passed to the underlying plot function.

        Returns
        -------
        (fig, ax)
        """
        from .panel_plots import plot_coef, plot_effects, plot_residuals, plot_hausman
        if type == 'coef':
            return plot_coef(self, **kwargs)
        elif type == 'effects':
            return plot_effects(self, **kwargs)
        elif type == 'residuals':
            return plot_residuals(self, **kwargs)
        elif type == 'hausman':
            return plot_hausman(self, **kwargs)
        else:
            raise ValueError(
                f"Unknown plot type '{type}'. "
                f"Choose from: coef, effects, residuals, hausman"
            )

    def plot_effects(self, **kwargs):
        """Shortcut for ``.plot(type='effects')``. Distribution of entity FE."""
        return self.plot(type='effects', **kwargs)

    def plot_residuals(self, **kwargs):
        """Shortcut for ``.plot(type='residuals')``. Residual diagnostics (2x2)."""
        return self.plot(type='residuals', **kwargs)

    def plot_hausman(self, **kwargs):
        """Shortcut for ``.plot(type='hausman')``. Visual FE vs RE comparison."""
        return self.plot(type='hausman', **kwargs)

    # ------------------------------------------------------------------
    # Compare with another method
    # ------------------------------------------------------------------

    def compare(self, method: str, **kwargs) -> 'PanelCompareResults':
        """
        Re-estimate with a different method and compare side by side.

        Parameters
        ----------
        method : str
            Alternative method to compare against.

        Returns
        -------
        PanelCompareResults
            Side-by-side comparison with diagnostics.
        """
        other = panel(
            data=self._panel_data, formula=self._formula,
            entity=self._entity, time=self._time,
            method=method, **kwargs,
        )
        return PanelCompareResults(self, other)


class PanelCompareResults:
    """Side-by-side comparison of two panel models."""

    def __init__(self, model_a: PanelResults, model_b: PanelResults):
        self.model_a = model_a
        self.model_b = model_b

    def summary(self) -> str:
        name_a = self.model_a.model_info.get('model_type', 'Model A')
        name_b = self.model_b.model_info.get('model_type', 'Model B')

        all_vars = list(dict.fromkeys(
            list(self.model_a.params.index) + list(self.model_b.params.index)
        ))

        rows = []
        for var in all_vars:
            coef_a = self.model_a.params.get(var, np.nan)
            se_a = self.model_a.std_errors.get(var, np.nan)
            coef_b = self.model_b.params.get(var, np.nan)
            se_b = self.model_b.std_errors.get(var, np.nan)
            rows.append({
                'Variable': var,
                f'{name_a} coef': coef_a, f'{name_a} SE': se_a,
                f'{name_b} coef': coef_b, f'{name_b} SE': se_b,
            })

        df_cmp = pd.DataFrame(rows).set_index('Variable')

        lines = ["=" * 78, f"  Panel Comparison: {name_a} vs {name_b}", "=" * 78, ""]
        lines.append(df_cmp.to_string(float_format='%.4f'))
        lines.append("")

        # Diagnostics
        for label, model in [(name_a, self.model_a), (name_b, self.model_b)]:
            r2 = model.diagnostics.get('R-squared', np.nan)
            nobs = model.data_info.get('nobs', '?')
            lines.append(f"  {label}: R² = {r2:.4f}, N = {nobs}")
        lines.append("=" * 78)
        return "\n".join(lines)

    def plot(self, variables: Optional[List[str]] = None, **kwargs):
        """
        Side-by-side coefficient comparison plot.

        Returns
        -------
        (fig, ax)
        """
        from .panel_plots import plot_compare
        name_a = self.model_a.model_info.get('model_type', 'Model A')
        name_b = self.model_b.model_info.get('model_type', 'Model B')
        return plot_compare(
            {name_a: self.model_a, name_b: self.model_b},
            variables=variables, **kwargs,
        )

    def __repr__(self):
        name_a = self.model_a.model_info.get('model_type', 'A')
        name_b = self.model_b.model_info.get('model_type', 'B')
        return f"<PanelCompareResults: {name_a} vs {name_b}>"

    def __str__(self):
        return self.summary()


# ======================================================================
# Main entry point
# ======================================================================

_METHOD_ALIASES = {
    'fe': 'fe', 'fixed_effects': 'fe', 'within': 'fe',
    're': 're', 'random_effects': 're',
    'be': 'be', 'between': 'be',
    'fd': 'fd', 'first_difference': 'fd',
    'pooled': 'pooled', 'pols': 'pooled',
    'twoway': 'twoway', 'two_way': 'twoway', 'twfe': 'twoway',
    'mundlak': 'mundlak', 'cre': 'mundlak', 'correlated_re': 'mundlak',
    'chamberlain': 'chamberlain',
    'ab': 'ab', 'arellano_bond': 'ab', 'diff_gmm': 'ab',
    'system': 'system', 'blundell_bond': 'system', 'sys_gmm': 'system',
}

_LINEARMODELS_METHODS = {'fe', 're', 'be', 'fd', 'pooled', 'twoway'}
_GMM_METHODS = {'ab', 'system'}
_CRE_METHODS = {'mundlak', 'chamberlain'}


# ======================================================================
# balance_panel — keep only units observed in all time periods
# ======================================================================

def balance_panel(
    data: pd.DataFrame,
    entity: str,
    time: str,
) -> pd.DataFrame:
    """
    Balance a panel by keeping only units observed in every time period.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    entity : str
        Entity (unit) identifier column.
    time : str
        Time period column.

    Returns
    -------
    pd.DataFrame
        Balanced panel (same column order, sorted by entity then time).

    Examples
    --------
    >>> import statspai as sp
    >>> balanced = sp.balance_panel(df, entity='id', time='year')
    >>> balanced.groupby('id')['year'].count().nunique()  # all same count
    1
    """
    all_periods = data[time].nunique()
    counts = data.groupby(entity)[time].transform('nunique')
    balanced = data.loc[counts == all_periods].sort_values([entity, time])
    return balanced.reset_index(drop=True)


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
    balance: bool = False,
    # Dynamic panel (AB/System GMM) options
    lags: int = 1,
    gmm_lags: Tuple[int, int] = (2, 5),
    twostep: bool = False,
) -> PanelResults:
    """
    Unified panel regression with Stata-style syntax.

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

        **Static models** (via linearmodels):

        - ``'fe'``          Fixed Effects (within estimator)
        - ``'re'``          Random Effects (GLS)
        - ``'be'``          Between estimator
        - ``'fd'``          First Differences
        - ``'pooled'``      Pooled OLS
        - ``'twoway'``      Two-way FE (entity + time effects)

        **Correlated Random Effects**:

        - ``'mundlak'``     Mundlak (1978): RE + group means of X
        - ``'chamberlain'`` Chamberlain (1982): RE + time-specific group means

        **Dynamic panel GMM**:

        - ``'ab'``          Arellano-Bond (difference GMM)
        - ``'system'``      Blundell-Bond (system GMM)

    robust : str, default 'nonrobust'
        Standard errors: ``'nonrobust'``, ``'robust'`` (HC1), ``'kernel'``,
        ``'driscoll-kraay'``.
    cluster : str, optional
        Cluster variable: ``'entity'``, ``'time'``, or ``'twoway'``
        (two-way clustering by entity and time).
    weights : str, optional
        Weight variable name.
    alpha : float, default 0.05
        Significance level.
    balance : bool, default False
        If True, drop units not observed in every time period before
        estimation (equivalent to R's ``make.pbalanced()``).
    lags : int, default 1
        Number of AR lags (for dynamic panel methods ``'ab'``/``'system'``).
    gmm_lags : tuple, default (2, 5)
        GMM instrument lag range (for ``'ab'``/``'system'``).
    twostep : bool, default False
        Two-step GMM (for ``'ab'``/``'system'``).

    Returns
    -------
    PanelResults
        Results with built-in panel diagnostics:
        ``.hausman_test()``, ``.bp_lm_test()``, ``.f_test_effects()``,
        ``.pesaran_cd_test()``, ``.compare(method)``.

    Examples
    --------
    >>> import statspai as sp
    >>>
    >>> # Fixed Effects
    >>> r = sp.panel(df, "wage ~ edu + exp", entity='id', time='year')
    >>> print(r.summary())
    >>>
    >>> # Two-way FE (entity + time)
    >>> r = sp.panel(df, "wage ~ edu + exp", entity='id', time='year',
    ...              method='twoway')
    >>>
    >>> # Mundlak / Correlated RE
    >>> r = sp.panel(df, "wage ~ edu + exp", entity='id', time='year',
    ...              method='mundlak')
    >>>
    >>> # Arellano-Bond dynamic panel
    >>> r = sp.panel(df, "y ~ x1 + x2", entity='id', time='year',
    ...              method='ab', lags=1)
    >>>
    >>> # System GMM (Blundell-Bond)
    >>> r = sp.panel(df, "y ~ x1 + x2", entity='id', time='year',
    ...              method='system', lags=1, twostep=True)
    >>>
    >>> # Two-way clustered SE
    >>> r = sp.panel(df, "wage ~ edu + exp", entity='id', time='year',
    ...              method='fe', cluster='twoway')
    >>>
    >>> # Diagnostics on result
    >>> r.hausman_test()       # FE vs RE
    >>> r.bp_lm_test()         # Pooled vs RE
    >>> r.f_test_effects()     # Joint significance of FE
    >>> r.compare('re')        # Side-by-side comparison

    See Also
    --------
    xtabond : Standalone Arellano-Bond / Blundell-Bond estimator.
    hausman_test : Standalone Hausman test.
    """
    # --- Resolve method alias ---
    method_key = method.lower().replace('-', '_').replace(' ', '_')
    if method_key not in _METHOD_ALIASES:
        valid = sorted(set(_METHOD_ALIASES.keys()))
        raise ValueError(f"method must be one of {valid}, got '{method}'")
    canonical = _METHOD_ALIASES[method_key]

    # --- Parse formula ---
    if '~' not in formula:
        raise ValueError("Formula must contain '~'")
    dep, indep = formula.split('~', 1)
    dep_var = dep.strip()
    indep_vars = [v.strip() for v in indep.split('+') if v.strip()]

    all_cols = [dep_var] + indep_vars + [entity, time]
    for col in all_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    # --- Balance panel if requested ---
    if balance:
        n_units = data[entity].nunique()
        n_periods = data[time].nunique()
        data = balance_panel(data, entity=entity, time=time)
        if len(data) == 0:
            raise ValueError(
                f"balance=True dropped all units: none of the {n_units} "
                f"entities appear in all {n_periods} time periods. "
                "Check data or set balance=False."
            )

    # --- Route to estimator ---
    if canonical in _GMM_METHODS:
        return _fit_gmm(
            data=data, dep_var=dep_var, indep_vars=indep_vars,
            entity=entity, time=time, formula=formula,
            gmm_method='difference' if canonical == 'ab' else 'system',
            lags=lags, gmm_lags=gmm_lags, twostep=twostep,
            robust=(robust != 'nonrobust'), alpha=alpha,
        )
    elif canonical in _CRE_METHODS:
        return _fit_cre(
            data=data, dep_var=dep_var, indep_vars=indep_vars,
            entity=entity, time=time, formula=formula,
            cre_method=canonical, robust=robust, cluster=cluster,
            weights=weights, alpha=alpha,
        )
    else:
        return _fit_linearmodels(
            data=data, dep_var=dep_var, indep_vars=indep_vars,
            entity=entity, time=time, formula=formula,
            method=canonical, robust=robust, cluster=cluster,
            weights=weights, alpha=alpha,
        )


# ======================================================================
# linearmodels-based estimators (fe, re, be, fd, pooled, twoway)
# ======================================================================

def _fit_linearmodels(
    data, dep_var, indep_vars, entity, time, formula,
    method, robust, cluster, weights, alpha,
) -> PanelResults:
    try:
        from linearmodels.panel import (
            PanelOLS, RandomEffects, BetweenOLS,
            FirstDifferenceOLS, PooledOLS,
        )
        from statsmodels.tools import add_constant
    except ImportError:
        raise ImportError(
            "linearmodels required for panel regression. "
            "Install: pip install linearmodels"
        )

    panel_data = data.set_index([entity, time])
    dep = panel_data[dep_var]
    exog = panel_data[indep_vars]

    if method == 'twoway':
        # Two-way FE: entity + time effects via PanelOLS
        lm_model = PanelOLS(dep, exog, entity_effects=True, time_effects=True)
    elif method == 'fe':
        lm_model = PanelOLS(dep, exog, entity_effects=True)
    elif method == 'fd':
        lm_model = FirstDifferenceOLS(dep, exog)
    elif method == 're':
        lm_model = RandomEffects(dep, add_constant(exog))
    elif method == 'be':
        lm_model = BetweenOLS(dep, add_constant(exog))
    elif method == 'pooled':
        lm_model = PooledOLS(dep, add_constant(exog))
    else:
        raise ValueError(f"Unknown linearmodels method: {method}")

    cov_kwargs = _build_cov_kwargs(robust, cluster)
    lm_result = lm_model.fit(**cov_kwargs)

    return _convert_lm_result(
        lm_result, method, dep_var, indep_vars, entity, time,
        formula, robust, cluster, data,
    )


def _build_cov_kwargs(robust: str, cluster: Optional[str]) -> Dict[str, Any]:
    if cluster == 'twoway':
        return {'cov_type': 'clustered', 'cluster_entity': True, 'cluster_time': True}
    elif cluster == 'entity':
        return {'cov_type': 'clustered', 'cluster_entity': True}
    elif cluster == 'time':
        return {'cov_type': 'clustered', 'cluster_time': True}
    elif cluster:
        return {'cov_type': 'clustered', 'cluster_entity': True}
    elif robust == 'robust':
        return {'cov_type': 'robust'}
    elif robust == 'kernel' or robust == 'driscoll-kraay':
        return {'cov_type': 'kernel'}
    return {'cov_type': 'unadjusted'}


_METHOD_NAMES = {
    'fe': 'Panel FE (Within)',
    'twoway': 'Panel Two-way FE',
    're': 'Panel RE (GLS)',
    'be': 'Panel Between',
    'fd': 'Panel First Difference',
    'pooled': 'Pooled OLS',
    'mundlak': 'Mundlak CRE',
    'chamberlain': 'Chamberlain CRE',
    'ab': 'Arellano-Bond GMM',
    'system': 'Blundell-Bond System GMM',
}


def _convert_lm_result(
    lm_result, method, dep_var, indep_vars, entity, time,
    formula, robust, cluster, raw_data,
) -> PanelResults:
    params = lm_result.params
    std_errors = lm_result.std_errors

    model_info = {
        'model_type': _METHOD_NAMES.get(method, method),
        'method': method,
        'robust': robust,
        'cluster': cluster,
    }

    data_info = {
        'nobs': int(lm_result.nobs),
        'df_model': int(lm_result.df_model) if hasattr(lm_result, 'df_model') and not isinstance(lm_result.df_model, tuple) else len(params) - 1,
        'df_resid': int(lm_result.df_resid),
        'dependent_var': dep_var,
        'fitted_values': lm_result.fitted_values.values.ravel(),
        'residuals': lm_result.resids.values.ravel(),
    }

    diagnostics = {
        'R-squared': float(lm_result.rsquared),
    }
    if hasattr(lm_result, 'rsquared_within') and lm_result.rsquared_within is not None:
        diagnostics['R-squared (within)'] = float(lm_result.rsquared_within)
    if hasattr(lm_result, 'rsquared_between') and lm_result.rsquared_between is not None:
        diagnostics['R-squared (between)'] = float(lm_result.rsquared_between)
    if hasattr(lm_result, 'entity_info'):
        diagnostics['N entities'] = lm_result.entity_info.total
    if hasattr(lm_result, 'time_info'):
        diagnostics['N time periods'] = lm_result.time_info.total
    if hasattr(lm_result, 'f_statistic') and lm_result.f_statistic is not None:
        diagnostics['F-statistic'] = float(lm_result.f_statistic.stat)
        diagnostics['F p-value'] = float(lm_result.f_statistic.pval)

    return PanelResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
        _panel_data=raw_data,
        _formula=formula,
        _entity=entity,
        _time=time,
        _dep_var=dep_var,
        _indep_vars=indep_vars,
        _method=method,
        _lm_result=lm_result,
    )


# ======================================================================
# Correlated Random Effects (Mundlak / Chamberlain)
# ======================================================================

def _fit_cre(
    data, dep_var, indep_vars, entity, time, formula,
    cre_method, robust, cluster, weights, alpha,
) -> PanelResults:
    """
    Correlated Random Effects: adds group means to a RE model.

    Mundlak (1978): adds entity-level means of all X variables.
    Chamberlain (1982): adds entity-level means separately for each
    time period (more flexible, uses more degrees of freedom).
    """
    try:
        from linearmodels.panel import RandomEffects
        from statsmodels.tools import add_constant
    except ImportError:
        raise ImportError(
            "linearmodels required for CRE estimation. "
            "Install: pip install linearmodels"
        )

    df = data.copy()

    if cre_method == 'mundlak':
        # Mundlak: add entity-level means of each X
        mundlak_vars = []
        for var in indep_vars:
            mean_col = f'_mean_{var}'
            df[mean_col] = df.groupby(entity)[var].transform('mean')
            mundlak_vars.append(mean_col)
        all_exog = indep_vars + mundlak_vars
    else:
        # Chamberlain: add entity means for each time period interaction
        # This creates T-1 additional variables per X
        chamberlain_vars = []
        time_vals = sorted(df[time].unique())
        for var in indep_vars:
            mean_col = f'_mean_{var}'
            df[mean_col] = df.groupby(entity)[var].transform('mean')
            chamberlain_vars.append(mean_col)
            # Add time-specific deviations from the entity mean
            for t_val in time_vals[1:]:  # skip first to avoid collinearity
                t_col = f'_cham_{var}_t{t_val}'
                t_mean = df.loc[df[time] == t_val].groupby(entity)[var].transform('mean')
                df[t_col] = 0.0
                df.loc[df[time] == t_val, t_col] = (
                    df.loc[df[time] == t_val, var] -
                    df.loc[df[time] == t_val, mean_col]
                )
                chamberlain_vars.append(t_col)
        all_exog = indep_vars + chamberlain_vars

    panel_data = df.set_index([entity, time])
    dep = panel_data[dep_var]
    exog = add_constant(panel_data[all_exog])

    lm_model = RandomEffects(dep, exog)
    cov_kwargs = _build_cov_kwargs(robust, cluster)
    lm_result = lm_model.fit(**cov_kwargs)

    result = _convert_lm_result(
        lm_result, cre_method, dep_var, indep_vars, entity, time,
        formula, robust, cluster, data,
    )

    # Test: are the Mundlak terms jointly significant?
    # (equivalent to Hausman test)
    if cre_method == 'mundlak':
        mundlak_params = {k: v for k, v in lm_result.params.items()
                          if k.startswith('_mean_')}
        if mundlak_params:
            result.diagnostics['Mundlak terms'] = len(mundlak_params)
            # Wald test for joint significance of means
            try:
                mean_coefs = np.array(list(mundlak_params.values()))
                mean_idx = [i for i, name in enumerate(lm_result.params.index)
                            if name.startswith('_mean_')]
                vcov = lm_result.cov
                V_sub = vcov.values[np.ix_(mean_idx, mean_idx)]
                wald = float(mean_coefs @ np.linalg.pinv(V_sub) @ mean_coefs)
                wald_df = len(mean_coefs)
                wald_p = float(1 - stats.chi2.cdf(wald, wald_df))
                result.diagnostics['CRE Wald chi2'] = wald
                result.diagnostics['CRE Wald df'] = wald_df
                result.diagnostics['CRE Wald p-value'] = wald_p
                result.diagnostics['CRE interpretation'] = (
                    'Reject H0: use FE' if wald_p < 0.05
                    else 'Cannot reject H0: RE is efficient'
                )
            except Exception:
                pass

    return result


# ======================================================================
# Dynamic panel GMM (Arellano-Bond / Blundell-Bond)
# ======================================================================

def _fit_gmm(
    data, dep_var, indep_vars, entity, time, formula,
    gmm_method, lags, gmm_lags, twostep, robust, alpha,
) -> PanelResults:
    """Route to existing xtabond implementation and wrap as PanelResults."""
    from ..gmm.arellano_bond import xtabond

    causal_result = xtabond(
        data=data, y=dep_var, x=indep_vars if indep_vars else None,
        id=entity, time=time, lags=lags, gmm_lags=gmm_lags,
        method=gmm_method, twostep=twostep, robust=robust, alpha=alpha,
    )

    # Convert CausalResult detail into PanelResults format
    if causal_result.detail is not None and 'coefficient' in causal_result.detail.columns:
        params = pd.Series(
            causal_result.detail['coefficient'].values,
            index=causal_result.detail['variable'].values,
        )
        std_errors = pd.Series(
            causal_result.detail['se'].values,
            index=causal_result.detail['variable'].values,
        )
    else:
        params = causal_result.params
        std_errors = causal_result.std_errors

    method_key = 'ab' if gmm_method == 'difference' else 'system'

    model_info = {
        'model_type': _METHOD_NAMES.get(method_key, gmm_method),
        'method': method_key,
        'robust': 'robust' if robust else 'nonrobust',
        'twostep': twostep,
        'gmm_lags': gmm_lags,
    }
    # Merge in the GMM-specific diagnostics
    model_info.update({
        k: v for k, v in causal_result.model_info.items()
        if k not in model_info
    })

    data_info = {
        'nobs': causal_result.n_obs,
        'dependent_var': dep_var,
        'df_resid': max(causal_result.n_obs - len(params), 1),
    }

    diagnostics = {}
    mi = causal_result.model_info
    if 'ar1_z' in mi:
        diagnostics['AR(1) z'] = mi['ar1_z']
        diagnostics['AR(1) p-value'] = mi['ar1_p']
    if 'ar2_z' in mi:
        diagnostics['AR(2) z'] = mi['ar2_z']
        diagnostics['AR(2) p-value'] = mi['ar2_p']
    if 'hansen_stat' in mi:
        diagnostics['Hansen J'] = mi['hansen_stat']
        diagnostics['Hansen df'] = mi['hansen_df']
        diagnostics['Hansen p-value'] = mi['hansen_p']
    if 'n_units' in mi:
        diagnostics['N entities'] = mi['n_units']
    if 'n_instruments' in mi:
        diagnostics['N instruments'] = mi['n_instruments']

    return PanelResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
        _panel_data=data,
        _formula=formula,
        _entity=entity,
        _time=time,
        _dep_var=dep_var,
        _indep_vars=indep_vars,
        _method=method_key,
    )


# ======================================================================
# panel_compare — multi-method comparison
# ======================================================================

def panel_compare(
    data: pd.DataFrame,
    formula: str,
    entity: str,
    time: str,
    methods: Optional[List[str]] = None,
    robust: str = 'nonrobust',
    cluster: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Estimate the same model with multiple methods and compare.

    Parameters
    ----------
    data : pd.DataFrame
    formula : str
    entity, time : str
    methods : list of str, optional
        Methods to compare. Default: ['pooled', 'fe', 're', 'twoway', 'mundlak'].
    robust, cluster : str
        Passed to each ``panel()`` call.

    Returns
    -------
    pd.DataFrame
        Comparison table with coefficients, SEs, and diagnostics.

    Examples
    --------
    >>> comparison = sp.panel_compare(
    ...     df, "wage ~ edu + exp", entity='id', time='year',
    ...     methods=['pooled', 'fe', 're', 'twoway', 'mundlak']
    ... )
    >>> print(comparison)
    """
    if methods is None:
        methods = ['pooled', 'fe', 're', 'twoway', 'mundlak']

    results = {}
    for m in methods:
        try:
            r = panel(data, formula, entity, time, method=m,
                      robust=robust, cluster=cluster, **kwargs)
            results[_METHOD_NAMES.get(m, m)] = r
        except Exception as e:
            results[_METHOD_NAMES.get(m, m)] = str(e)

    # Build comparison DataFrame
    # Gather all variable names
    all_vars = []
    for name, r in results.items():
        if isinstance(r, PanelResults):
            for v in r.params.index:
                if v not in all_vars:
                    all_vars.append(v)

    rows = []
    for var in all_vars:
        row = {'Variable': var}
        for name, r in results.items():
            if isinstance(r, PanelResults):
                coef = r.params.get(var, np.nan)
                se = r.std_errors.get(var, np.nan)
                pvals = r.pvalues
                if isinstance(pvals, pd.Series):
                    pv = pvals.get(var, np.nan)
                elif hasattr(pvals, '__getitem__') and var in r.params.index:
                    idx = list(r.params.index).index(var)
                    pv = float(pvals[idx]) if idx < len(pvals) else np.nan
                else:
                    pv = np.nan
                stars = '***' if pv < 0.01 else '**' if pv < 0.05 else '*' if pv < 0.1 else ''
                row[name] = f"{coef:.4f}{stars}" if not np.isnan(coef) else ''
                row[f"{name} (SE)"] = f"({se:.4f})" if not np.isnan(se) else ''
            else:
                row[name] = 'error'
                row[f"{name} (SE)"] = ''
        rows.append(row)

    # Add diagnostics rows
    for diag_key in ['R-squared', 'N entities', 'N time periods']:
        row = {'Variable': diag_key}
        for name, r in results.items():
            if isinstance(r, PanelResults):
                val = r.diagnostics.get(diag_key, np.nan)
                if isinstance(val, float):
                    row[name] = f"{val:.4f}"
                elif isinstance(val, int):
                    row[name] = str(val)
                else:
                    row[name] = ''
            else:
                row[name] = ''
            row[f"{name} (SE)"] = ''
        rows.append(row)

    # N obs
    row = {'Variable': 'N obs'}
    for name, r in results.items():
        if isinstance(r, PanelResults):
            row[name] = str(r.data_info.get('nobs', ''))
        else:
            row[name] = ''
        row[f"{name} (SE)"] = ''
    rows.append(row)

    df_out = pd.DataFrame(rows).set_index('Variable')
    # Interleave coef and SE columns
    ordered_cols = []
    for name in results.keys():
        ordered_cols.append(name)
        se_col = f"{name} (SE)"
        if se_col in df_out.columns:
            ordered_cols.append(se_col)

    return df_out[[c for c in ordered_cols if c in df_out.columns]]


# ======================================================================
# Keep old class name for backward compatibility
# ======================================================================

class PanelRegression:
    """Deprecated: use ``panel()`` directly. Kept for backward compatibility."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def fit(self) -> PanelResults:
        return panel(**self._kwargs)
