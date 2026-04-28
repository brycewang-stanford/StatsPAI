"""
Matrix Completion for Causal Panel Data.

Estimates treatment effects by imputing the counterfactual outcomes
matrix using nuclear-norm regularisation (soft-thresholded SVD):

    min_{L}  sum_{(i,t) in Omega} (Y_it - L_it)^2 + lambda * ||L||_*

where Omega is the set of control (untreated) observations and
||L||_* is the nuclear norm (sum of singular values).

The treatment effect for treated unit i at time t is:
    tau_it = Y_it - L*_it

This approach subsumes both synthetic control (low-rank across units)
and interactive fixed effects (low-rank across time).

References
----------
Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021).
"Matrix Completion Methods for Causal Panel Data Models."
JASA, 116(536), 1716-1730. [@athey2021matrix]
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def mc_panel(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treat: str,
    lambda_reg: Optional[float] = None,
    max_rank: Optional[int] = None,
    max_iter: int = 1000,
    tol: float = 1e-5,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CausalResult:
    """
    Estimate treatment effects using matrix completion.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable.
    unit : str
        Unit identifier variable.
    time : str
        Time period variable.
    treat : str
        Binary treatment indicator (0/1). Can be staggered.
    lambda_reg : float, optional
        Nuclear norm regularisation parameter. If None,
        estimated via the universal threshold: lambda = sigma * sqrt(n).
    max_rank : int, optional
        Maximum rank for the completed matrix. If None, no constraint.
    max_iter : int, default 1000
        Maximum iterations for the proximal gradient algorithm.
    tol : float, default 1e-5
        Convergence tolerance.
    n_bootstrap : int, default 200
        Bootstrap iterations for standard errors.
    alpha : float, default 0.05
        Significance level.
    random_state : int, default 42

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.mc_panel(df, y='gdp', unit='country',
    ...                      time='year', treat='treated')
    >>> print(result.summary())
    """
    est = MCPanel(
        data=data, y=y, unit=unit, time=time, treat=treat,
        lambda_reg=lambda_reg, max_rank=max_rank,
        max_iter=max_iter, tol=tol, n_bootstrap=n_bootstrap,
        alpha=alpha, random_state=random_state,
    )
    _result = est.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.matrix_completion.mc_panel",
            params={
                "y": y, "unit": unit, "time": time, "treat": treat,
                "lambda_reg": lambda_reg, "max_rank": max_rank,
                "max_iter": max_iter, "tol": tol,
                "n_bootstrap": n_bootstrap,
                "alpha": alpha, "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ======================================================================
# MCPanel class
# ======================================================================

class MCPanel:
    """
    Matrix Completion for Causal Panels.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    unit : str
    time : str
    treat : str
    lambda_reg : float, optional
    max_rank : int, optional
    max_iter : int
    tol : float
    n_bootstrap : int
    alpha : float
    random_state : int
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        unit: str,
        time: str,
        treat: str,
        lambda_reg: Optional[float] = None,
        max_rank: Optional[int] = None,
        max_iter: int = 1000,
        tol: float = 1e-5,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        self.data = data
        self.y = y
        self.unit = unit
        self.time = time
        self.treat = treat
        self.lambda_reg = lambda_reg
        self.max_rank = max_rank
        self.max_iter = max_iter
        self.tol = tol
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Run matrix completion and return treatment effect estimates."""
        cols = [self.y, self.unit, self.time, self.treat]
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        df = self.data[cols].dropna().copy()

        # Pivot to matrix form
        Y_mat = df.pivot_table(
            index=self.unit, columns=self.time,
            values=self.y, aggfunc='first'
        )
        W_mat = df.pivot_table(
            index=self.unit, columns=self.time,
            values=self.treat, aggfunc='first'
        ).fillna(0)

        units = Y_mat.index.tolist()
        times = Y_mat.columns.tolist()
        N, T = Y_mat.shape

        Y = Y_mat.values.astype(np.float64)
        W = W_mat.values.astype(np.float64)

        # Mask: 1 where we observe untreated outcome (control obs)
        Omega = (W == 0) & (~np.isnan(Y))

        # Handle NaN in Y
        Y_filled = np.nan_to_num(Y, nan=0.0)

        # Determine lambda
        if self.lambda_reg is None:
            # Universal threshold: sigma * sqrt(max(N, T))
            # Estimate sigma from control observations
            control_vals = Y_filled[Omega]
            if len(control_vals) > 1:
                sigma = float(np.std(control_vals, ddof=1))
            else:
                sigma = 1.0
            self.lambda_reg = sigma * np.sqrt(max(N, T)) / 10

        # Solve via soft-impute (proximal gradient)
        L = self._soft_impute(Y_filled, Omega, N, T)

        # Treatment effects: tau_it = Y_it - L_it for treated obs
        treated_mask = W == 1
        if treated_mask.sum() == 0:
            raise ValueError("No treated observations found.")

        tau_matrix = np.where(treated_mask, Y - L, np.nan)

        # Average treatment effect on treated
        tau_values = tau_matrix[treated_mask]
        att = float(np.mean(tau_values))

        # Bootstrap SE
        rng = np.random.RandomState(self.random_state)
        boot_atts = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Resample units
            unit_idx = rng.choice(N, size=N, replace=True)
            Y_b = Y[unit_idx]
            W_b = W[unit_idx]
            Omega_b = (W_b == 0) & (~np.isnan(Y_b))
            Y_b_filled = np.nan_to_num(Y_b, nan=0.0)

            L_b = self._soft_impute(Y_b_filled, Omega_b, N, T)
            treated_b = W_b == 1
            if treated_b.sum() > 0:
                boot_atts[b] = np.mean(Y_b[treated_b] - L_b[treated_b])
            else:
                boot_atts[b] = att

        se = float(np.std(boot_atts, ddof=1))

        if se > 0:
            z_stat = att / se
            pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z_stat))))
        else:
            pvalue = 0.0

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        ci = (att - z_crit * se, att + z_crit * se)

        # Build detail: per-unit ATT
        unit_atts = []
        for i, u in enumerate(units):
            treated_times = np.where(treated_mask[i])[0]
            if len(treated_times) > 0:
                tau_u = np.mean(tau_matrix[i, treated_times])
                unit_atts.append({
                    'unit': u,
                    'att': float(tau_u),
                    'n_treated_periods': len(treated_times),
                })

        detail = pd.DataFrame(unit_atts) if unit_atts else None

        # Completed matrix rank
        U, s, Vt = np.linalg.svd(L, full_matrices=False)
        effective_rank = int(np.sum(s > 1e-6))

        model_info = {
            'lambda_reg': self.lambda_reg,
            'effective_rank': effective_rank,
            'n_units': N,
            'n_periods': T,
            'n_treated_cells': int(treated_mask.sum()),
            'n_control_cells': int(Omega.sum()),
            'completed_matrix': L,
            'treatment_effects_matrix': tau_matrix,
        }

        self._L = L
        self._tau = tau_matrix

        return CausalResult(
            method='Matrix Completion (Athey et al. 2021)',
            estimand='ATT',
            estimate=att,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=int(Omega.sum() + treated_mask.sum()),
            detail=detail,
            model_info=model_info,
            _citation_key='mc_panel',
        )

    def _soft_impute(self, Y, Omega, N, T):
        """
        Soft-impute algorithm for nuclear norm regularised completion.

        Iteratively:
        1. Replace treated entries with current estimate.
        2. Compute SVD.
        3. Soft-threshold singular values.
        """
        L = np.zeros((N, T))

        for iteration in range(self.max_iter):
            # Fill in: use observed controls, impute treated
            Z = np.where(Omega, Y, L)

            # SVD
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)

            # Soft-threshold
            s_thresh = np.maximum(s - self.lambda_reg, 0)

            # Max rank constraint
            if self.max_rank is not None:
                s_thresh[self.max_rank:] = 0

            # Reconstruct
            L_new = U * s_thresh @ Vt

            # Check convergence
            diff = np.linalg.norm(L_new - L, 'fro')
            norm_L = np.linalg.norm(L_new, 'fro') + 1e-10

            L = L_new

            if diff / norm_L < self.tol:
                break

        return L


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS['mc_panel'] = (
    "@article{athey2021matrix,\n"
    "  title={Matrix Completion Methods for Causal Panel Data Models},\n"
    "  author={Athey, Susan and Bayati, Mohsen and Doudchenko, Nikolay "
    "and Imbens, Guido and Khosravi, Khashayar},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={116},\n"
    "  number={536},\n"
    "  pages={1716--1730},\n"
    "  year={2021},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
