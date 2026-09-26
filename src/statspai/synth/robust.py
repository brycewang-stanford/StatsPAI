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
* **penalized** — classic SCM constraints (>= 0, sum = 1) with a
  ridge penalty ``l2 * ||w||^2`` inside the feasible set (on the simplex
  ``||w||_1 == 1``, so ``l1_penalty`` is a constant there and has no
  effect). This is *not* the Abadie & L'Hour (2021) penalized estimator,
  whose penalty is ``sum_j w_j ||X_1 - X_j||^2``; use
  ``sp.synth(method='penalized')`` for that.

Objective of the regression variants (intercept ``mu`` never penalised)::

    min_{mu, w}  ||y - mu - X w||^2 + l2_penalty ||w||_2^2 + l1_penalty ||w||_1

References
----------
Doudchenko, N. and Imbens, G.W. (2016).
"Balancing, Regression, Difference-in-Differences and Synthetic
Control Methods: A Synthesis." NBER Working Paper 22791. [@doudchenko2016balancing]
"""

from __future__ import annotations

import warnings
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility
from ._core import placebo_rank_pvalue, solve_simplex_weights


@accepts_aliases(id="unit", y="outcome")
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
        Not supported: the weights are fitted on pre-treatment outcomes
        only. Passing covariates raises ``NotImplementedError`` (it used
        to be silently ignored).
    variant : {'unconstrained', 'elastic_net', 'penalized'}, default 'unconstrained'
        * ``'unconstrained'`` — no sign / sum constraints; optional intercept.
        * ``'elastic_net'`` — L1 + L2 penalty, no sign constraints.
        * ``'penalized'`` — simplex constraints + ridge penalty (see the
          module docstring; not Abadie & L'Hour 2021).
    l1_penalty : float, default 0.0
        L1 penalty ``l1_penalty * ||w||_1`` in the objective
        ``||y - mu - X w||^2 + l2 ||w||^2 + l1 ||w||_1``. glmnet's
        ``(lambda, alpha)`` on the same data maps to
        ``l1 = 2 n lambda alpha`` (``n`` = pre-periods) once ``y`` is
        scaled to unit (1/n) standard deviation.
    l2_penalty : float, default 0.01
        Ridge penalty ``l2_penalty * ||w||^2`` (glmnet:
        ``l2 = n lambda (1 - alpha)``). ``0`` with ``l1_penalty=0`` is
        ordinary least squares with an intercept — Doudchenko & Imbens'
        unconstrained estimator, identical to
        ``scpi::scest(w.constr = list(name = "ols"))`` on
        ``scdata(constant = TRUE)``.
    intercept : bool, default True
        Fit an unpenalised intercept (level shift). Only for
        unconstrained / elastic_net.
    placebo : bool, default True
        Run in-space placebo inference.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> result = sp.robust_synth(df, outcome='packspercapita', unit='state',
    ...     time='year', treated_unit='California', treatment_time=1989,
    ...     variant='unconstrained')
    >>> bool(result.estimate < 0)  # Prop 99 reduced cigarette sales
    True
    """
    if covariates:
        raise NotImplementedError(
            "robust_synth fits donor weights on pre-treatment outcomes only; "
            "`covariates` is not supported (it was previously ignored "
            "silently). Use sp.synth(method='classic', covariates=...) or "
            "sp.scpi for covariate matching."
        )
    if variant not in ("unconstrained", "elastic_net", "penalized"):
        raise MethodIncompatibility(
            "variant must be 'unconstrained', 'elastic_net' or 'penalized', "
            f"got {variant!r}"
        )

    # --- Build panel ---
    pivot = data.pivot_table(index=time, columns=unit, values=outcome)
    times = pivot.index.values
    pre_mask = times < treatment_time
    post_mask = times >= treatment_time

    if pre_mask.sum() < 2:
        raise ValueError("Need at least 2 pre-treatment periods")  # pragma: no cover
    if post_mask.sum() < 1:
        raise ValueError("Need at least 1 post-treatment period")  # pragma: no cover

    Y_treated = pivot[treated_unit].values.astype(np.float64)
    donor_cols = [c for c in pivot.columns if c != treated_unit]
    Y_donors = pivot[donor_cols].values.astype(np.float64)

    # Drop donors with NaN
    pre_donors = Y_donors[pre_mask]
    valid = ~np.any(np.isnan(pre_donors), axis=0)
    if valid.sum() == 0:
        raise ValueError("No valid donor units")  # pragma: no cover
    Y_donors = Y_donors[:, valid]
    donor_cols = [donor_cols[i] for i in range(len(donor_cols)) if valid[i]]
    J = Y_donors.shape[1]

    # --- Solve weights ---
    weights, intercept_val = _solve_robust_weights(
        Y_treated[pre_mask],
        Y_donors[pre_mask],
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
    pre_mspe = float(np.mean(gap_pre**2))

    # --- Placebo ---
    placebo_atts = []
    placebo_pre_mspes = []
    placebo_post_mspes = []
    if placebo and J >= 2:
        all_Y = np.column_stack([Y_treated[:, np.newaxis], Y_donors])
        for i in range(J):
            idx_p = i + 1
            Y_p = all_Y[:, idx_p]
            didx = [j for j in range(all_Y.shape[1]) if j != idx_p]
            Y_d = all_Y[:, didx]
            try:
                w_p, int_p = _solve_robust_weights(
                    Y_p[pre_mask],
                    Y_d[pre_mask],
                    variant=variant,
                    l1_penalty=l1_penalty,
                    l2_penalty=l2_penalty,
                    fit_intercept=(intercept and variant != "penalized"),
                )
                synth_p = Y_d @ w_p + int_p
                gap_p = Y_p - synth_p
                placebo_atts.append(float(np.mean(gap_p[post_mask])))
                placebo_pre_mspes.append(float(np.mean(gap_p[pre_mask] ** 2)))
                placebo_post_mspes.append(float(np.mean(gap_p[post_mask] ** 2)))
            except np.linalg.LinAlgError:  # pragma: no cover
                continue  # pragma: no cover

    if len(placebo_atts) > 0:
        # Abadie-Diamond-Hainmueller post/pre MSPE ratio, computed the SAME
        # way for the treated unit and every placebo.
        post_mspe = float(np.mean(gap_post**2))
        ratio_treated = _mspe_ratio(post_mspe, pre_mspe)
        placebo_ratios = [
            _mspe_ratio(post_m, pre_m)
            for post_m, pre_m in zip(placebo_post_mspes, placebo_pre_mspes)
        ]
        pvalue = placebo_rank_pvalue(ratio_treated, placebo_ratios)
        se = float(np.std(placebo_atts)) if len(placebo_atts) > 1 else 0.0
    else:
        pvalue = np.nan
        se = float(np.std(gap_post)) / max(np.sqrt(len(gap_post)), 1)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att - z_crit * se, att + z_crit * se)

    weight_df = (
        pd.DataFrame(
            {
                "unit": donor_cols,
                "weight": weights,
            }
        )
        .sort_values("weight", ascending=False, key=abs)
        .reset_index(drop=True)
    )

    gap_df = pd.DataFrame(
        {
            "time": times,
            "treated": Y_treated,
            "synthetic": Y_synth,
            "gap": gap,
            "post_treatment": post_mask,
        }
    )

    variant_labels = {
        "unconstrained": "Unconstrained SCM (Doudchenko & Imbens 2016)",
        "elastic_net": "Elastic-Net SCM",
        "penalized": "Ridge-penalized simplex SCM",
    }

    model_info = {
        "variant": variant,
        "n_donors": J,
        "n_pre_periods": int(pre_mask.sum()),
        "n_post_periods": int(post_mask.sum()),
        "pre_treatment_mspe": pre_mspe,
        "pre_treatment_rmse": float(np.sqrt(pre_mspe)),
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
        model_info["mspe_ratio"] = ratio_treated
        model_info["placebo_mspe_ratios"] = placebo_ratios

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


def _mspe_ratio(post_mspe: float, pre_mspe: float) -> float:
    """Post/pre MSPE ratio; a perfect pre-fit gives ``+inf`` for any unit."""
    if pre_mspe > 1e-10:
        return post_mspe / pre_mspe
    return np.inf if post_mspe > 0 else 0.0


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

    Regression variants minimise
    ``||y - mu - X w||^2 + l2 ||w||^2 + l1 ||w||_1`` with the intercept
    ``mu`` unpenalised (profiled out by centring when ``fit_intercept``).

    Returns (weights, intercept).
    """
    if variant == "penalized":
        # Simplex constraints + ridge; ||w||_1 == 1 on the simplex, so the
        # L1 term is a constant and does not move the minimiser.
        return solve_simplex_weights(y, X, penalization=l2_penalty), 0.0

    if fit_intercept:
        x_bar = X.mean(axis=0)
        y_bar = float(y.mean())
        Xc = X - x_bar
        yc = y - y_bar
    else:
        Xc, yc = X, y

    if l1_penalty > 0:
        beta = _elastic_net_cd(yc, Xc, l1_penalty, l2_penalty)
    else:
        # Ridge / OLS closed form (lstsq so that l2 = 0 is plain OLS).
        J = Xc.shape[1]
        if l2_penalty > 0:
            A = np.vstack([Xc, np.sqrt(l2_penalty) * np.eye(J)])
            b = np.concatenate([yc, np.zeros(J)])
        else:
            A, b = Xc, yc
        beta = np.linalg.lstsq(A, b, rcond=None)[0]

    if fit_intercept:
        return beta, float(y_bar - x_bar @ beta)
    return beta, 0.0


def _elastic_net_cd(
    y: np.ndarray,
    X: np.ndarray,
    l1: float,
    l2: float,
    max_iter: int = 100_000,
    tol: float = 1e-13,
) -> np.ndarray:
    """
    Cyclic coordinate descent for
    ``min_b ||y - X b||^2 + l2 ||b||^2 + l1 ||b||_1`` (no constraints).

    The coordinate update is ``S(x_j' r_j, l1 / 2) / (x_j' x_j + l2)`` with
    ``S`` the soft-threshold operator and ``r_j`` the partial residual.
    Convergence: largest coordinate change below ``tol`` times the
    largest coefficient; a non-converged run raises a ``RuntimeWarning``.
    """
    n, p = X.shape
    beta = np.zeros(p)
    xtx = np.sum(X**2, axis=0)
    r = y.copy()
    half_l1 = 0.5 * l1
    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            if xtx[j] == 0.0 and l2 == 0.0:
                continue
            bj_old = beta[j]
            rho = X[:, j] @ r + xtx[j] * bj_old
            if abs(rho) <= half_l1:
                bj = 0.0
            else:
                bj = np.sign(rho) * (abs(rho) - half_l1) / (xtx[j] + l2)
            if bj != bj_old:
                r -= X[:, j] * (bj - bj_old)
                beta[j] = bj
                max_delta = max(max_delta, abs(bj - bj_old))
        if max_delta <= tol * max(1.0, float(np.max(np.abs(beta)))):
            return beta
    warnings.warn(
        f"elastic-net coordinate descent did not converge in {max_iter} "
        "sweeps; weights may be inaccurate.",
        RuntimeWarning,
        stacklevel=3,
    )
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
