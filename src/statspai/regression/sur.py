"""
Seemingly Unrelated Regression (SUR) and system estimation.

Zellner's (1962) SUR estimator for a system of equations with
correlated error terms, plus Three-Stage Least Squares (3SLS).

Equivalent to Stata's ``sureg`` / ``reg3`` and R's ``systemfit``.

References
----------
Zellner, A. (1962).
"An Efficient Method of Estimating Seemingly Unrelated Regressions
and Tests for Aggregation Bias."
*JASA*, 57(298), 348-368. [@zellner1962efficient]

Zellner, A. & Theil, H. (1962).
"Three-Stage Least Squares: Simultaneous Estimation of
Simultaneous Equations."
*Econometrica*, 30(1), 54-78. [@zellner1962three]
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import EconometricResults


class SURResult:
    """Results from SUR estimation."""

    def __init__(self, equations, sigma, params_all, se_all, n_obs,
                 n_equations, method, breusch_pagan):
        self.equations = equations  # dict: eq_name -> {params, se, r2, ...}
        self.sigma = sigma  # cross-equation covariance matrix
        self.params_all = params_all
        self.se_all = se_all
        self.n_obs = n_obs
        self.n_equations = n_equations
        self.method = method
        self.breusch_pagan = breusch_pagan  # test of diagonal Sigma

    def summary(self) -> str:
        lines = [
            f"Seemingly Unrelated Regression ({self.method})",
            "=" * 65,
            f"Equations: {self.n_equations}   N obs: {self.n_obs}",
        ]
        if self.breusch_pagan is not None:
            lines.append(f"Breusch-Pagan χ²({self.breusch_pagan['df']})"
                         f" = {self.breusch_pagan['chi2']:.3f}"
                         f" (p = {self.breusch_pagan['p_value']:.4f})")
        lines.append("")

        for eq_name, eq_info in self.equations.items():
            lines.append(f"Equation: {eq_name}  (R² = {eq_info.get('r2', 0):.4f})")
            lines.append(f"{'Variable':<20s} {'Coef':>10s} {'SE':>10s} "
                         f"{'t':>8s} {'P>|t|':>8s}")
            lines.append("-" * 60)
            params = eq_info['params']
            se = eq_info['se']
            for var in params.index:
                t_val = params[var] / se[var] if se[var] > 0 else np.nan
                p_val = 2 * (1 - stats.t.cdf(abs(t_val), self.n_obs)) if np.isfinite(t_val) else np.nan
                lines.append(f"{var:<20s} {params[var]:>10.4f} {se[var]:>10.4f} "
                             f"{t_val:>8.3f} {p_val:>8.4f}")
            lines.append("")

        lines.append("=" * 65)
        return "\n".join(lines)


def sureg(
    equations: Dict[str, Tuple[str, List[str]]],
    data: pd.DataFrame,
    method: str = "fgls",
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> SURResult:
    """
    Seemingly Unrelated Regression (SUR).

    Estimates a system of equations with correlated errors using
    Zellner's (1962) FGLS estimator.

    Equivalent to Stata's ``sureg`` and R's ``systemfit("SUR")``.

    Parameters
    ----------
    equations : dict
        Mapping from equation name to (dep_var, list_of_regressors).
        Example: {'eq1': ('y1', ['x1', 'x2']), 'eq2': ('y2', ['x1', 'x3'])}
    data : pd.DataFrame
    method : str, default 'fgls'
        'ols' (equation-by-equation), 'fgls' (feasible GLS / SUR),
        'iterative' (iterated SUR).
    maxiter : int, default 100
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    SURResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.sureg(
    ...     equations={
    ...         'demand': ('quantity', ['price', 'income']),
    ...         'supply': ('quantity', ['price', 'cost']),
    ...     },
    ...     data=df,
    ... )
    >>> print(result.summary())
    """
    eq_names = list(equations.keys())
    M = len(eq_names)

    # Build equation-specific data
    all_vars = set()
    for eq_name, (dep, indep) in equations.items():
        all_vars.add(dep)
        all_vars.update(indep)

    df = data.dropna(subset=list(all_vars))
    n = len(df)

    Y_list = []
    X_list = []
    k_list = []
    var_names_list = []

    for eq_name in eq_names:
        dep, indep = equations[eq_name]
        y_eq = df[dep].values.astype(float)
        X_eq = np.column_stack([np.ones(n), df[indep].values.astype(float)])
        Y_list.append(y_eq)
        X_list.append(X_eq)
        k_list.append(X_eq.shape[1])
        var_names_list.append(['_cons'] + list(indep))

    k_total = sum(k_list)

    # Step 1: OLS for each equation
    beta_ols = []
    resid_ols = []
    for m in range(M):
        b = np.linalg.lstsq(X_list[m], Y_list[m], rcond=None)[0]
        beta_ols.append(b)
        resid_ols.append(Y_list[m] - X_list[m] @ b)

    # Estimate cross-equation covariance
    E = np.column_stack(resid_ols)  # n x M
    Sigma = E.T @ E / n  # M x M

    if method == 'ols':
        # Just return OLS results
        beta_all = np.concatenate(beta_ols)
        # SE from OLS
        se_list = []
        for m in range(M):
            sigma2 = np.sum(resid_ols[m]**2) / (n - k_list[m])
            se = np.sqrt(sigma2 * np.diag(np.linalg.inv(X_list[m].T @ X_list[m])))
            se_list.append(se)
        se_all = np.concatenate(se_list)
    else:
        # FGLS / SUR
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            Sigma_inv = np.linalg.pinv(Sigma)

        for iteration in range(maxiter if method == 'iterative' else 1):
            # Build block-diagonal X and stacked Y
            # GLS: β = (X' (Σ^{-1} ⊗ I) X)^{-1} X' (Σ^{-1} ⊗ I) Y
            XtSX = np.zeros((k_total, k_total))
            XtSY = np.zeros(k_total)

            row_offset = 0
            for i in range(M):
                col_offset = 0
                for j in range(M):
                    w = Sigma_inv[i, j]
                    XiXj = w * (X_list[i].T @ X_list[j])
                    XtSX[row_offset:row_offset + k_list[i],
                          col_offset:col_offset + k_list[j]] = XiXj
                    col_offset += k_list[j]

                XtSY_i = np.zeros(k_list[i])
                for j in range(M):
                    XtSY_i += Sigma_inv[i, j] * (X_list[i].T @ Y_list[j])
                XtSY[row_offset:row_offset + k_list[i]] = XtSY_i
                row_offset += k_list[i]

            try:
                beta_all = np.linalg.solve(XtSX, XtSY)
            except np.linalg.LinAlgError:
                beta_all = np.concatenate(beta_ols)
                break

            # Update residuals
            offset = 0
            resid_new = []
            for m in range(M):
                b_m = beta_all[offset:offset + k_list[m]]
                resid_new.append(Y_list[m] - X_list[m] @ b_m)
                offset += k_list[m]

            if method == 'iterative':
                E_new = np.column_stack(resid_new)
                Sigma_new = E_new.T @ E_new / n
                if np.max(np.abs(Sigma_new - Sigma)) < tol:
                    Sigma = Sigma_new
                    Sigma_inv = np.linalg.inv(Sigma)
                    break
                Sigma = Sigma_new
                Sigma_inv = np.linalg.inv(Sigma)

        # SE from GLS
        try:
            var_cov = np.linalg.inv(XtSX)
            se_all = np.sqrt(np.abs(np.diag(var_cov)))
        except np.linalg.LinAlgError:
            se_all = np.full(k_total, np.nan)

    # Build equation-level results
    eq_results = {}
    offset = 0
    for m, eq_name in enumerate(eq_names):
        b_m = beta_all[offset:offset + k_list[m]]
        se_m = se_all[offset:offset + k_list[m]]
        resid_m = Y_list[m] - X_list[m] @ b_m
        tss = np.sum((Y_list[m] - Y_list[m].mean())**2)
        rss = np.sum(resid_m**2)
        r2 = 1 - rss / tss if tss > 0 else 0

        eq_results[eq_name] = {
            'params': pd.Series(b_m, index=var_names_list[m]),
            'se': pd.Series(se_m, index=var_names_list[m]),
            'r2': r2,
            'dep_var': equations[eq_name][0],
        }
        offset += k_list[m]

    # Breusch-Pagan test of diagonal Sigma
    R = np.corrcoef(np.column_stack([Y_list[m] - X_list[m] @ eq_results[eq_names[m]]['params'].values
                                      for m in range(M)]).T)
    bp_stat = n * np.sum(np.triu(R, k=1)**2)
    bp_df = M * (M - 1) // 2
    bp_p = 1 - stats.chi2.cdf(bp_stat, bp_df) if bp_df > 0 else np.nan

    return SURResult(
        equations=eq_results,
        sigma=pd.DataFrame(Sigma, index=eq_names, columns=eq_names),
        params_all=beta_all,
        se_all=se_all,
        n_obs=n,
        n_equations=M,
        method=method.upper(),
        breusch_pagan={'chi2': bp_stat, 'df': bp_df, 'p_value': bp_p},
    )


def three_sls(
    equations: Dict[str, Tuple[str, List[str], List[str]]],
    data: pd.DataFrame,
    instruments: List[str] = None,
    maxiter: int = 100,
    alpha: float = 0.05,
) -> SURResult:
    """
    Three-Stage Least Squares (3SLS).

    System estimator that accounts for both endogeneity and
    cross-equation error correlation.

    Equivalent to Stata's ``reg3`` and R's ``systemfit("3SLS")``.

    Parameters
    ----------
    equations : dict
        Mapping: eq_name -> (dep_var, exog_vars, endog_vars).
    data : pd.DataFrame
    instruments : list of str
        Full set of instruments (all exogenous variables in the system).
    maxiter : int, default 100
    alpha : float, default 0.05

    Returns
    -------
    SURResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.three_sls(
    ...     equations={
    ...         'demand': ('q', ['p', 'income'], []),
    ...         'supply': ('q', ['p', 'cost'], []),
    ...     },
    ...     data=df, instruments=['income', 'cost', 'weather'],
    ... )
    """
    eq_names = list(equations.keys())
    M = len(eq_names)

    # Collect all variables
    all_vars = set()
    for _, (dep, exog, endog) in equations.items():
        all_vars.add(dep)
        all_vars.update(exog)
        all_vars.update(endog)
    if instruments:
        all_vars.update(instruments)

    df = data.dropna(subset=list(all_vars))
    n = len(df)

    # Build instrument matrix
    if instruments is None:
        # Use all exogenous variables as instruments
        instruments = list(set().union(*[set(exog) for _, (_, exog, _) in equations.items()]))

    Z = np.column_stack([np.ones(n), df[instruments].values.astype(float)])
    Pz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T  # projection onto instruments

    # Stage 1: 2SLS for each equation
    Y_list, X_list, k_list, var_names_list = [], [], [], []
    beta_2sls = []
    resid_2sls = []

    for eq_name in eq_names:
        dep, exog, endog = equations[eq_name]
        y_eq = df[dep].values.astype(float)
        X_eq = np.column_stack([np.ones(n)] +
                                [df[v].values.astype(float) for v in exog] +
                                [df[v].values.astype(float) for v in endog])

        # Instrument endogenous variables
        X_hat = Pz @ X_eq

        b = np.linalg.lstsq(X_hat, y_eq, rcond=None)[0]
        beta_2sls.append(b)
        resid_2sls.append(y_eq - X_eq @ b)

        Y_list.append(y_eq)
        X_list.append(X_eq)
        k_list.append(X_eq.shape[1])
        var_names_list.append(['_cons'] + exog + endog)

    # Stage 2: Estimate Sigma from 2SLS residuals
    E = np.column_stack(resid_2sls)
    Sigma = E.T @ E / n
    Sigma_inv = np.linalg.inv(Sigma)

    # Stage 3: GLS with instruments
    k_total = sum(k_list)
    X_hat_list = [Pz @ X for X in X_list]

    XtSX = np.zeros((k_total, k_total))
    XtSY = np.zeros(k_total)

    row_offset = 0
    for i in range(M):
        col_offset = 0
        for j in range(M):
            w = Sigma_inv[i, j]
            XtSX[row_offset:row_offset + k_list[i],
                  col_offset:col_offset + k_list[j]] = w * (X_hat_list[i].T @ X_list[j])
            col_offset += k_list[j]

        XtSY_i = np.zeros(k_list[i])
        for j in range(M):
            XtSY_i += Sigma_inv[i, j] * (X_hat_list[i].T @ Y_list[j])
        XtSY[row_offset:row_offset + k_list[i]] = XtSY_i
        row_offset += k_list[i]

    try:
        beta_all = np.linalg.solve(XtSX, XtSY)
        var_cov = np.linalg.inv(XtSX)
        se_all = np.sqrt(np.abs(np.diag(var_cov)))
    except np.linalg.LinAlgError:
        beta_all = np.concatenate(beta_2sls)
        se_all = np.full(k_total, np.nan)

    # Build equation results
    eq_results = {}
    offset = 0
    for m, eq_name in enumerate(eq_names):
        b_m = beta_all[offset:offset + k_list[m]]
        se_m = se_all[offset:offset + k_list[m]]
        resid_m = Y_list[m] - X_list[m] @ b_m
        tss = np.sum((Y_list[m] - Y_list[m].mean())**2)
        rss = np.sum(resid_m**2)
        r2 = 1 - rss / tss if tss > 0 else 0

        eq_results[eq_name] = {
            'params': pd.Series(b_m, index=var_names_list[m]),
            'se': pd.Series(se_m, index=var_names_list[m]),
            'r2': r2,
            'dep_var': equations[eq_name][0],
        }
        offset += k_list[m]

    return SURResult(
        equations=eq_results,
        sigma=pd.DataFrame(Sigma, index=eq_names, columns=eq_names),
        params_all=beta_all,
        se_all=se_all,
        n_obs=n,
        n_equations=M,
        method='3SLS',
        breusch_pagan=None,
    )
