"""
Matrix Completion for Causal Panel Data.

Estimates treatment effects by imputing the counterfactual outcomes
matrix using nuclear-norm regularisation (soft-thresholded SVD):

    min_{a, b, L}  1/2 sum_{(i,t) in Omega} (Y_it - a_i - b_t - L_it)^2
                   + lambda * ||L||_*

where Omega is the set of control (untreated) observations, ||L||_* is
the nuclear norm (sum of singular values) and the unit / time effects
``a_i`` / ``b_t`` are unpenalised (``fixed_effects='two-way'``, the
default, as in R ``MCPanel::mcnnm`` and ``fect(method="mc")``).

The treatment effect for treated unit i at time t is:
    tau_it = Y_it - (a_i + b_t + L_it)

This approach subsumes both synthetic control (low-rank across units)
and interactive fixed effects (low-rank across time).

References
----------
Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021).
"Matrix Completion Methods for Causal Panel Data Models."
JASA, 116(536), 1716-1730. [@athey2021matrix]
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult
from ._core import _check_fixed_effects, mc_nnm_fit

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
    max_iter: int = 5000,
    tol: float = 1e-10,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
    fixed_effects: str = "two-way",
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
        Singular-value threshold ``lambda`` on the objective above (the
        ``1/2``-scaled squared loss). The same minimiser is obtained from
        R ``MCPanel::mcnnm_fit(lambda_L = 2 * lambda / |Omega|)`` and from
        ``fect(method = "mc", CV = FALSE, lambda = lambda / (N * T))``;
        both conversions are reported in ``model_info``. If None, a
        heuristic (no reference implementation uses it) is applied:
        ``lambda = sd(Y[Omega]) * sqrt(max(N, T)) / 10``.
    max_rank : int, optional
        Hard cap on the rank of the low-rank component (not part of the
        reference problem). If None, no constraint.
    max_iter : int, default 5000
        Maximum soft-impute iterations. A ``RuntimeWarning`` is raised if
        the point-estimate fit hits the cap before ``tol``.
    tol : float, default 1e-10
        Convergence tolerance on the relative Frobenius change of the
        fitted counterfactual matrix.
    n_bootstrap : int, default 200
        Unit-bootstrap replications for the standard error.
    alpha : float, default 0.05
        Significance level.
    random_state : int, default 42
    fixed_effects : {"two-way", "unit", "time", "none"}, default "two-way"
        Unpenalised additive effects fitted alongside the low-rank
        component. ``"two-way"`` is the Athey et al. (2021) estimator
        and the default of R ``MCPanel`` (``to_estimate_u = to_estimate_v
        = TRUE``) and ``fect`` (``force = "two-way"``). ``"none"`` is pure
        soft-impute (``MCPanel`` with both switches off; no intercept, so
        the outcome level is shrunk towards zero); it is the estimator this
        function computed in StatsPAI <= 1.28.0.

    Returns
    -------
    CausalResult
        ``model_info['completed_matrix']`` is the imputed untreated
        outcome matrix (fixed effects + low-rank part);
        ``model_info['low_rank_matrix']`` is the low-rank part alone.

    Notes
    -----
    ``se`` is a unit (row) bootstrap of the whole fit at the same
    ``lambda``; neither ``MCPanel`` nor ``fect`` reports this exact
    quantity, so only the point estimate is reference-aligned.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for u in range(8):
    ...     fe = rng.normal()
    ...     for t in range(6):
    ...         treated = int(u >= 6 and t >= 4)
    ...         val = fe + 0.3 * t + rng.normal(0, 0.2) + (1.5 if treated else 0.0)
    ...         rows.append((u, t, val, treated))
    >>> panel = pd.DataFrame(rows, columns=['country', 'year', 'gdp', 'treated'])
    >>> result = sp.mc_panel(panel, y='gdp', unit='country',
    ...                      time='year', treat='treated', n_bootstrap=50)
    >>> bool(result.estimate > 0)
    True
    """
    est = MCPanel(
        data=data,
        y=y,
        unit=unit,
        time=time,
        treat=treat,
        lambda_reg=lambda_reg,
        max_rank=max_rank,
        max_iter=max_iter,
        tol=tol,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
        fixed_effects=fixed_effects,
    )
    _result = est.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.matrix_completion.mc_panel",
            params={
                "y": y,
                "unit": unit,
                "time": time,
                "treat": treat,
                "lambda_reg": lambda_reg,
                "max_rank": max_rank,
                "max_iter": max_iter,
                "tol": tol,
                "n_bootstrap": n_bootstrap,
                "alpha": alpha,
                "random_state": random_state,
                "fixed_effects": fixed_effects,
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
    fixed_effects : {"two-way", "unit", "time", "none"}

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> from statspai.matrix_completion.mc_panel import MCPanel
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for u in range(8):
    ...     fe = rng.normal()
    ...     for t in range(6):
    ...         treated = int(u >= 6 and t >= 4)
    ...         y = fe + 0.3 * t + rng.normal(0, 0.2) + (1.5 if treated else 0.0)
    ...         rows.append((u, t, y, treated))
    >>> panel = pd.DataFrame(rows, columns=['unit', 'period', 'y', 'treated'])
    >>> est = MCPanel(data=panel, y='y', unit='unit', time='period',
    ...               treat='treated', n_bootstrap=50)
    >>> res = est.fit()
    >>> bool(res.estimate > 0)
    True

    References
    ----------
    [@athey2021matrix]
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
        max_iter: int = 5000,
        tol: float = 1e-10,
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        random_state: int = 42,
        fixed_effects: str = "two-way",
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
        self.fixed_effects = _check_fixed_effects(fixed_effects)

    def fit(self) -> CausalResult:
        """Run matrix completion and return treatment effect estimates."""
        cols = [self.y, self.unit, self.time, self.treat]
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ValueError(f"Columns not found in data: {missing}")

        df = self.data[cols].dropna().copy()

        # Pivot to matrix form
        Y_mat = df.pivot_table(
            index=self.unit, columns=self.time, values=self.y, aggfunc="first"
        )
        W_mat = df.pivot_table(
            index=self.unit, columns=self.time, values=self.treat, aggfunc="first"
        ).fillna(0)

        units = Y_mat.index.tolist()
        N, T = Y_mat.shape

        Y = Y_mat.values.astype(np.float64)
        W = W_mat.values.astype(np.float64)

        # Mask: 1 where we observe untreated outcome (control obs)
        Omega = (W == 0) & (~np.isnan(Y))

        # Handle NaN in Y
        Y_filled = np.nan_to_num(Y, nan=0.0)

        # Determine lambda
        lam = self.lambda_reg
        if lam is None:
            # Heuristic: sd(control Y) * sqrt(max(N, T)) / 10
            control_vals = Y_filled[Omega]
            if len(control_vals) > 1:
                sigma = float(np.std(control_vals, ddof=1))
            else:
                sigma = 1.0
            lam = sigma * np.sqrt(max(N, T)) / 10
        lam = float(lam)

        # Solve via soft-impute with unpenalised fixed effects
        sol = self._solve(Y_filled, Omega, lam, warn=True)
        L = sol["fit"]

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
        n_boot_unconverged = 0

        for b in range(self.n_bootstrap):
            # Resample units
            unit_idx = rng.choice(N, size=N, replace=True)
            Y_b = Y[unit_idx]
            W_b = W[unit_idx]
            Omega_b = (W_b == 0) & (~np.isnan(Y_b))
            Y_b_filled = np.nan_to_num(Y_b, nan=0.0)

            sol_b = self._solve(Y_b_filled, Omega_b, lam, warn=False)
            n_boot_unconverged += int(not sol_b["converged"])
            L_b = sol_b["fit"]
            treated_b = W_b == 1
            if treated_b.sum() > 0:
                boot_atts[b] = np.mean(Y_b[treated_b] - L_b[treated_b])
            else:
                boot_atts[b] = att

        # A handful of slow resamples barely moves the SE; many means the
        # SE is computed from unconverged fits -> say so.
        if n_boot_unconverged > 0.05 * self.n_bootstrap:
            import warnings

            warnings.warn(
                f"{n_boot_unconverged} of {self.n_bootstrap} bootstrap fits "
                f"hit max_iter={self.max_iter} before tol={self.tol:g}; "
                "increase max_iter.",
                RuntimeWarning,
                stacklevel=2,
            )

        se = float(np.std(boot_atts, ddof=1)) if self.n_bootstrap > 1 else np.nan

        if se > 0:
            z_stat = att / se
            pvalue = float(2 * sp_stats.norm.sf(abs(z_stat)))
        else:
            pvalue = np.nan

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        ci = (att - z_crit * se, att + z_crit * se)

        # Build detail: per-unit ATT
        unit_atts = []
        for i, u in enumerate(units):
            treated_times = np.where(treated_mask[i])[0]
            if len(treated_times) > 0:
                tau_u = np.mean(tau_matrix[i, treated_times])
                unit_atts.append(
                    {
                        "unit": u,
                        "att": float(tau_u),
                        "n_treated_periods": len(treated_times),
                    }
                )

        detail = pd.DataFrame(unit_atts) if unit_atts else None

        # Rank of the (penalised) low-rank component
        effective_rank = int(np.sum(sol["singular_values"] > 1e-6))

        periods = Y_mat.columns.tolist()
        treated_units = [u for i, u in enumerate(units) if treated_mask[i].any()]
        completed_df = pd.DataFrame(L, index=units, columns=periods)
        completed_df.index.name = self.unit
        completed_df.columns.name = self.time

        n_omega = int(Omega.sum())
        model_info = {
            "lambda_reg": lam,
            # Same minimiser, reference parameterisations:
            "lambda_mcpanel": 2.0 * lam / n_omega,
            "lambda_fect": lam / (N * T),
            "fixed_effects": self.fixed_effects,
            "converged": bool(sol["converged"]),
            "n_iter": int(sol["n_iter"]),
            "n_bootstrap_unconverged": n_boot_unconverged,
            "effective_rank": effective_rank,
            "n_units": N,
            "n_periods": T,
            "n_treated_cells": int(treated_mask.sum()),
            "n_control_cells": int(Omega.sum()),
            "completed_matrix": L,
            "low_rank_matrix": sol["L"],
            "fixed_effects_matrix": sol["fe"],
            "treatment_effects_matrix": tau_matrix,
            # Labeled views — completed_matrix rows follow this exact
            # unit order; use these instead of guessing row positions.
            "units": list(units),
            "periods": list(periods),
            "treated_units": treated_units,
            "completed_df": completed_df,
            "observed_df": Y_mat,
            # Counterfactual (untreated potential outcome) series for
            # each treated unit: rows = treated units, columns = periods.
            "counterfactual": completed_df.loc[treated_units],
        }

        self._L = L
        self._tau = tau_matrix

        return CausalResult(
            method="Matrix Completion (Athey et al. 2021)",
            estimand="ATT",
            estimate=att,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=int(Omega.sum() + treated_mask.sum()),
            detail=detail,
            model_info=model_info,
            _citation_key="mc_panel",
        )

    def _solve(
        self,
        Y: np.ndarray,
        Omega: np.ndarray,
        lam: float,
        warn: bool,
    ) -> dict:
        """Nuclear-norm completion on the control cells (shared solver)."""
        return mc_nnm_fit(
            Y,
            Omega,
            lam,
            fixed_effects=self.fixed_effects,
            max_iter=self.max_iter,
            tol=self.tol,
            max_rank=self.max_rank,
            warn=warn,
        )


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS["mc_panel"] = (
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
