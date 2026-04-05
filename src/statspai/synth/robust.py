"""
Robust / Unconstrained Synthetic Control Methods.

Relaxes the standard SCM constraints (non-negativity & sum-to-one)
and supports regularisation via ridge, lasso, or elastic net.

Variants
--------
* **unconstrained** — allow negative weights and an intercept
  (Doudchenko & Imbens 2016). Useful when the treated unit lies
  outside the convex hull of the donors.
* **elastic_net** — L1 + L2 penalty to produce sparse but
  regularised donor weights (no sum / sign constraints).
* **penalized** — classic SCM constraints (>= 0, sum = 1)
  with elastic-net penalty inside the feasible set.

References
----------
Doudchenko, N. and Imbens, G.W. (2016).
"Balancing, Regression, Difference-in-Differences and Synthetic
Control Methods: A Synthesis." NBER Working Paper 22791.

Abadie, A. and L'Hour, J. (2021).
"A Penalized Synthetic Control Estimator for Disaggregated Data."
*Journal of the American Statistical Association*, 116(536), 1817-1834.
"""

from __future__ import annotations

from typing import Any, List, Optional, Literal

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult


def robust_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    covariates: Optional[List[str]] = None,
    variant: Literal["unconstrained", "elastic_net", "penalized"] = "unconstrained",
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.01,
    intercept: bool = True,
    placebo: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Robust / unconstrained Synthetic Control.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period.
    covariates : list of str, optional
        Additional covariates to match on.
    variant : {'unconstrained', 'elastic_net', 'penalized'}, default 'unconstrained'
        * ``'unconstrained'`` — no sign / sum constraints; optional intercept.
        * ``'elastic_net'`` — L1 + L2 penalty, no sign constraints.
        * ``'penalized'`` — classic SCM constraints + elastic-net penalty.
    l1_penalty : float, default 0.0
        Lasso (L1) penalty strength.
    l2_penalty : float, default 0.01
        Ridge (L2) penalty strength.
    intercept : bool, default True
        Fit an intercept (level shift). Only for unconstrained / elastic_net.
    placebo : bool, default True
        Run in-space placebo inference.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> result = sp.robust_synth(df, outcome='gdp', unit='state',
    ...     time='year', treated_unit='California', treatment_time=1989,
    ...     variant='unconstrained')
    """
    # --- Build panel ---
    pivot = data.pivot_table(index=time, columns=unit, values=outcome)
    times = pivot.index.values
    pre_mask = times < treatment_time
    post_mask = times >= treatment_time

    if pre_mask.sum() < 2:
        raise ValueError("Need at least 2 pre-treatment periods")
    if post_mask.sum() < 1:
        raise ValueError("Need at least 1 post-treatment period")

    Y_treated = pivot[treated_unit].values.astype(np.float64)
    donor_cols = [c for c in pivot.columns if c != treated_unit]
    Y_donors = pivot[donor_cols].values.astype(np.float64)

    # Drop donors with NaN
    pre_donors = Y_donors[pre_mask]
    valid = ~np.any(np.isnan(pre_donors), axis=0)
    if valid.sum() == 0:
        raise ValueError("No valid donor units")
    Y_donors = Y_donors[:, valid]
    donor_cols = [donor_cols[i] for i in range(len(donor_cols)) if valid[i]]
    J = Y_donors.shape[1]

    # --- Solve weights ---
    weights, intercept_val = _solve_robust_weights(
        Y_treated[pre_mask], Y_donors[pre_mask],
        variant=variant,
        l1_penalty=l1_penalty,
        l2_penalty=l2_penalty,
        fit_intercept=(intercept and variant != "penalized"),
    )

    # --- Synthetic control ---
    Y_synth = Y_donors @ weights + intercept_val
    gap = Y_treated - Y_synth
    gap_post = gap[post_mask]
    gap_pre = gap[pre_mask]
    att = float(np.mean(gap_post))
    pre_mspe = float(np.mean(gap_pre ** 2))

    # --- Placebo ---
    placebo_atts = []
    placebo_pre_mspes = []
    if placebo and J >= 2:
        all_Y = np.column_stack([Y_treated[:, np.newaxis], Y_donors])
        for i in range(J):
            idx_p = i + 1
            Y_p = all_Y[:, idx_p]
            didx = [j for j in range(all_Y.shape[1]) if j != idx_p]
            Y_d = all_Y[:, didx]
            try:
                w_p, int_p = _solve_robust_weights(
                    Y_p[pre_mask], Y_d[pre_mask],
                    variant=variant, l1_penalty=l1_penalty,
                    l2_penalty=l2_penalty,
                    fit_intercept=(intercept and variant != "penalized"),
                )
                synth_p = Y_d @ w_p + int_p
                gap_p = Y_p - synth_p
                placebo_atts.append(float(np.mean(gap_p[post_mask])))
                placebo_pre_mspes.append(float(np.mean(gap_p[pre_mask] ** 2)))
            except Exception:
                continue

    if len(placebo_atts) > 0:
        post_mspe = float(np.mean(gap_post ** 2))
        ratio_treated = post_mspe / pre_mspe if pre_mspe > 1e-10 else np.inf
        placebo_ratios = [
            a ** 2 / m if m > 1e-10 else 0
            for a, m in zip(placebo_atts, placebo_pre_mspes)
        ]
        pvalue = float(np.mean(np.array(placebo_ratios) >= ratio_treated))
        pvalue = max(pvalue, 1 / (len(placebo_ratios) + 1))
        se = float(np.std(placebo_atts)) if len(placebo_atts) > 1 else 0.0
    else:
        pvalue = np.nan
        se = float(np.std(gap_post)) / max(np.sqrt(len(gap_post)), 1)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att - z_crit * se, att + z_crit * se)

    weight_df = pd.DataFrame({
        "unit": donor_cols, "weight": weights,
    }).sort_values("weight", ascending=False, key=abs).reset_index(drop=True)

    gap_df = pd.DataFrame({
        "time": times, "treated": Y_treated, "synthetic": Y_synth,
        "gap": gap, "post_treatment": post_mask,
    })

    variant_labels = {
        "unconstrained": "Unconstrained SCM (Doudchenko & Imbens 2016)",
        "elastic_net": "Elastic-Net SCM",
        "penalized": "Penalized SCM (Abadie & L'Hour 2021)",
    }

    model_info = {
        "variant": variant,
        "n_donors": J,
        "n_pre_periods": int(pre_mask.sum()),
        "n_post_periods": int(post_mask.sum()),
        "pre_treatment_mspe": round(pre_mspe, 6),
        "pre_treatment_rmse": round(np.sqrt(pre_mspe), 6),
        "l1_penalty": l1_penalty,
        "l2_penalty": l2_penalty,
        "intercept": intercept_val,
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "weights": weight_df,
        "gap_table": gap_df,
        "Y_synth": Y_synth,
        "Y_treated": Y_treated,
        "times": times,
        "n_nonzero_weights": int(np.sum(np.abs(weights) > 1e-6)),
    }

    if placebo_atts:
        model_info["placebo_atts"] = placebo_atts
        model_info["n_placebos"] = len(placebo_atts)

    return CausalResult(
        method=variant_labels.get(variant, f"Robust SCM ({variant})"),
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(Y_treated),
        detail=weight_df,
        model_info=model_info,
        _citation_key="robust_synth",
    )


def _solve_robust_weights(
    y: np.ndarray,
    X: np.ndarray,
    variant: str = "unconstrained",
    l1_penalty: float = 0.0,
    l2_penalty: float = 0.01,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Solve for donor weights under various constraint regimes.

    Returns (weights, intercept).
    """
    T, J = X.shape

    if variant == "penalized":
        # Classic constraints + elastic-net penalty
        return _solve_penalized_constrained(y, X, l1_penalty, l2_penalty)

    # --- Unconstrained / elastic net: OLS with regularisation ---
    if fit_intercept:
        X_aug = np.column_stack([np.ones(T), X])
    else:
        X_aug = X

    p = X_aug.shape[1]

    # Elastic net: min ||y - X_aug β||^2 + l2 ||β||^2 + l1 ||β||_1
    if l1_penalty > 0:
        # Coordinate descent for elastic net
        beta = _elastic_net_cd(y, X_aug, l1_penalty, l2_penalty, max_iter=1000)
    else:
        # Ridge closed form
        XtX = X_aug.T @ X_aug
        reg = l2_penalty * np.eye(p)
        if fit_intercept:
            reg[0, 0] = 0  # don't penalise intercept
        beta = np.linalg.solve(XtX + reg, X_aug.T @ y)

    if fit_intercept:
        return beta[1:], float(beta[0])
    return beta, 0.0


def _solve_penalized_constrained(
    y: np.ndarray,
    X: np.ndarray,
    l1_penalty: float,
    l2_penalty: float,
) -> tuple[np.ndarray, float]:
    """SCM constraints (w >= 0, sum = 1) + elastic-net penalty."""
    J = X.shape[1]

    def objective(w):
        r = y - X @ w
        return (r @ r
                + l2_penalty * (w @ w)
                + l1_penalty * np.sum(np.abs(w)))

    res = optimize.minimize(
        objective, np.ones(J) / J, method="SLSQP",
        bounds=[(0, None)] * J,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return res.x, 0.0


def _elastic_net_cd(
    y: np.ndarray,
    X: np.ndarray,
    l1: float,
    l2: float,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """Coordinate descent for elastic net (no constraints)."""
    n, p = X.shape
    beta = np.zeros(p)
    XtX_diag = np.sum(X ** 2, axis=0)

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r = y - X @ beta + X[:, j] * beta[j]
            rho = X[:, j] @ r
            # Soft threshold
            if abs(rho) <= l1:
                beta[j] = 0.0
            else:
                beta[j] = (np.sign(rho) * (abs(rho) - l1)) / (XtX_diag[j] + l2)
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


# Citation
CausalResult._CITATIONS["robust_synth"] = (
    "@techreport{doudchenko2016balancing,\n"
    "  title={Balancing, Regression, Difference-in-Differences and "
    "Synthetic Control Methods: A Synthesis},\n"
    "  author={Doudchenko, Nikolay and Imbens, Guido W.},\n"
    "  institution={NBER},\n"
    "  type={Working Paper},\n"
    "  number={22791},\n"
    "  year={2016}\n"
    "}"
)
