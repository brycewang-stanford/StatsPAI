"""
Stable Balancing Weights (Zubizarreta 2015, JASA).

Finds weights that minimise dispersion (e.g. variance, or KL divergence
from the uniform distribution) while imposing user-specified covariate
balance tolerances. Unlike entropy balancing, SBW allows *approximate*
balance via per-covariate tolerance ``δ_j``, which is essential when
exact balance is infeasible or would blow up variance.

Formulation
-----------
For ATT estimation with treated group :math:`\\mathcal{T}` and control
group :math:`\\mathcal{C}`, solve

.. math::

    \\min_{w} \\; \\frac{1}{|\\mathcal{C}|}\\sum_{i \\in \\mathcal{C}} w_i^2

    \\text{s.t.} \\quad
    \\left| \\frac{1}{|\\mathcal{T}|}\\sum_{i \\in \\mathcal{T}} X_{ij}
          - \\sum_{i \\in \\mathcal{C}} w_i X_{ij} \\right|
          \\;\\le\\; \\delta_j \\sigma_j  \\; \\forall j,

    \\sum_{i \\in \\mathcal{C}} w_i = 1, \\; w_i \\ge 0.

The variance-minimising objective is equivalent to maximising effective
sample size :math:`\\mathrm{ESS}(w) = (\\sum w_i)^2 / \\sum w_i^2`.

This complements :func:`ebalance` (exact balance, KL objective) and
:func:`cbps` (covariate-balancing propensity score) — together forming
the 2026 triumvirate of modern weighting estimators.

References
----------
Zubizarreta, J.R. (2015).
"Stable Weights that Balance Covariates for Estimation with Incomplete
Outcome Data." *Journal of the American Statistical Association*,
110(511), 910-922.

Wang, Y. and Zubizarreta, J.R. (2020).
"Minimal dispersion approximately balancing weights: asymptotic
properties and practical considerations." *Biometrika*, 107(1), 93-105.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize

from ..core.results import CausalResult


class SBWResult(CausalResult):
    """Stable balancing weights with a diagnostic panel.

    Thin subclass of :class:`CausalResult` that attaches the weight
    vector, effective sample size, and covariate balance table.
    """

    def __init__(
        self,
        *,
        method: str,
        estimand: str,
        estimate: float,
        se: float,
        pvalue: float,
        ci: tuple,
        alpha: float,
        n_obs: int,
        weights: np.ndarray,
        effective_sample_size: float,
        balance: pd.DataFrame,
        solver_status: str,
    ):
        super().__init__(
            method=method, estimand=estimand, estimate=estimate,
            se=se, pvalue=pvalue, ci=ci, alpha=alpha, n_obs=n_obs,
            detail=balance,
            model_info={
                "weights": np.asarray(weights, dtype=float),
                "balance": balance,
                "effective_sample_size": float(effective_sample_size),
                "solver_status": str(solver_status),
            },
            _citation_key="zubizarreta_2015_sbw",
        )
        self.weights = np.asarray(weights, dtype=float)
        self.effective_sample_size = float(effective_sample_size)
        self.balance = balance
        self.solver_status = str(solver_status)


def sbw(
    data: pd.DataFrame,
    treat: str,
    covariates: List[str],
    y: Optional[str] = None,
    *,
    estimand: str = "att",
    delta: Union[float, Sequence[float]] = 0.02,
    objective: str = "variance",
    tolerance_scale: str = "sd",
    include_squares: bool = False,
    alpha: float = 0.05,
    solver_options: Optional[dict] = None,
) -> SBWResult:
    """
    Stable Balancing Weights (Zubizarreta 2015) with optional ATT/ATE
    treatment-effect estimation.

    Parameters
    ----------
    data : DataFrame
    treat : str
        Binary 0/1 treatment indicator column.
    covariates : list of str
        Columns whose means must be balanced.
    y : str, optional
        Outcome column. If provided, a weighted ATT/ATE estimate with
        HC-robust SE is attached to the returned :class:`SBWResult`.
    estimand : {'att', 'ate', 'atc'}, default 'att'
        ``'att'`` reweights controls to match treated means (standard);
        ``'atc'`` reweights treated to match control means;
        ``'ate'`` reweights each group to match the pooled means.
    delta : float or sequence, default 0.02
        Balance tolerance. With ``tolerance_scale='sd'`` the constraint
        is ``|mean_T(X_j) - weighted mean_C(X_j)| ≤ δ_j · sd(X_j)``.
    objective : {'variance', 'entropy'}, default 'variance'
        Dispersion objective. ``'variance'`` minimises Σ w_i²;
        ``'entropy'`` minimises Σ w_i log(n · w_i) (KL from uniform).
    tolerance_scale : {'sd', 'raw'}, default 'sd'
        Whether ``delta`` is in SD units (standard) or raw units.
    include_squares : bool, default False
        Also balance second-moments (w_j² columns).
    alpha : float, default 0.05
        Significance level for inference on the outcome.
    solver_options : dict, optional
        Passed to ``scipy.optimize.minimize``.

    Returns
    -------
    SBWResult

    Examples
    --------
    >>> res = sp.sbw(df, treat='D', covariates=['age', 'educ', 'race'],
    ...              y='wage', delta=0.02)
    >>> print(res.summary())
    >>> res.balance                        # per-covariate SMD before/after
    """
    if estimand not in ("att", "ate", "atc"):
        raise ValueError("estimand must be one of 'att', 'atc', 'ate'")
    if objective not in ("variance", "entropy"):
        raise ValueError("objective must be one of 'variance', 'entropy'")

    req_cols = [treat] + list(covariates) + ([y] if y else [])
    df = data[req_cols].dropna().copy()
    T = df[treat].values.astype(int)
    if not set(np.unique(T)).issubset({0, 1}):
        raise ValueError(f"`{treat}` must be 0/1 binary.")
    X = df[covariates].values.astype(float)
    n = X.shape[0]
    n_t = int(T.sum())
    n_c = int(n - n_t)
    if n_t == 0 or n_c == 0:
        raise ValueError("Both treatment groups must be non-empty.")

    if include_squares:
        X = np.column_stack([X, X ** 2])
        cov_names = list(covariates) + [f"{c}^2" for c in covariates]
    else:
        cov_names = list(covariates)

    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)

    # ── Broadcast delta ───────────────────────────────────────────────
    if np.isscalar(delta):
        delta_vec = np.full(X.shape[1], float(delta))
    else:
        delta_vec = np.asarray(delta, dtype=float)
        if delta_vec.size != X.shape[1]:
            raise ValueError(
                f"delta must be scalar or length {X.shape[1]}"
            )
    tol_vec = delta_vec * (sd if tolerance_scale == "sd" else np.ones_like(sd))

    # ── Solve weights per estimand ────────────────────────────────────
    if estimand == "att":
        target = X[T == 1].mean(axis=0)
        w = _solve_sbw(X[T == 0], target, tol_vec, objective, solver_options)
        weights_full = np.zeros(n)
        weights_full[T == 0] = w
        weights_full[T == 1] = 1.0 / n_t
        solver_status = "att"
    elif estimand == "atc":
        target = X[T == 0].mean(axis=0)
        w = _solve_sbw(X[T == 1], target, tol_vec, objective, solver_options)
        weights_full = np.zeros(n)
        weights_full[T == 1] = w
        weights_full[T == 0] = 1.0 / n_c
        solver_status = "atc"
    else:  # ate
        target = X.mean(axis=0)
        w_t = _solve_sbw(X[T == 1], target, tol_vec, objective, solver_options)
        w_c = _solve_sbw(X[T == 0], target, tol_vec, objective, solver_options)
        weights_full = np.zeros(n)
        weights_full[T == 1] = w_t
        weights_full[T == 0] = w_c
        solver_status = "ate"

    # ── Balance diagnostics ───────────────────────────────────────────
    bal = _balance_table(X, T, weights_full, cov_names, estimand)

    # ESS only meaningful for the reweighted arm(s)
    if estimand == "att":
        w_arm = weights_full[T == 0]
    elif estimand == "atc":
        w_arm = weights_full[T == 1]
    else:
        w_arm = np.concatenate([weights_full[T == 1], weights_full[T == 0]])
    ess = float((w_arm.sum() ** 2) / np.sum(w_arm ** 2))

    # ── Optional outcome inference ────────────────────────────────────
    if y is not None:
        Y = df[y].values.astype(float)
        estimate, se, low, high, pval = _weighted_treatment_effect(
            Y, T, weights_full, estimand, alpha,
        )
    else:
        estimate = se = pval = low = high = np.nan

    return SBWResult(
        method=f"SBW-{estimand.upper()} ({objective})",
        estimand=estimand.upper(),
        estimate=float(estimate),
        se=float(se),
        pvalue=float(pval),
        ci=(float(low), float(high)),
        alpha=float(alpha),
        n_obs=int(n),
        weights=weights_full,
        effective_sample_size=ess,
        balance=bal,
        solver_status=solver_status,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Solver
# ═══════════════════════════════════════════════════════════════════════

def _solve_sbw(
    X_arm: np.ndarray,
    target: np.ndarray,
    tol: np.ndarray,
    objective: str,
    solver_options: Optional[dict],
) -> np.ndarray:
    """
    Minimise dispersion s.t. |X_arm' w - target| ≤ tol, 1'w = 1, w ≥ 0.
    """
    m = X_arm.shape[0]

    if objective == "variance":
        # Quadratic objective with linear constraints → SLSQP is fine.
        def fn(w):
            return float(np.sum(w * w))

        def grad(w):
            return 2.0 * w

    else:  # entropy (KL from uniform 1/m)
        eps_log = 1e-12

        def fn(w):
            w_safe = np.maximum(w, eps_log)
            return float(np.sum(w_safe * np.log(m * w_safe)))

        def grad(w):
            w_safe = np.maximum(w, eps_log)
            return np.log(m * w_safe) + 1.0

    # Constraints as a single vector function (SLSQP handles vector inequalities)
    def ineq_pos(w):
        # tol_j + (X_arm' w)_j - target_j ≥ 0
        return tol + (X_arm.T @ w) - target

    def ineq_neg(w):
        # tol_j - (X_arm' w)_j + target_j ≥ 0
        return tol - (X_arm.T @ w) + target

    def eq_sum(w):
        return float(np.sum(w)) - 1.0

    w0 = np.full(m, 1.0 / m)
    bounds = [(0.0, None)] * m
    constraints = [
        {"type": "eq", "fun": eq_sum},
        {"type": "ineq", "fun": ineq_pos},
        {"type": "ineq", "fun": ineq_neg},
    ]
    opts = {"maxiter": 500, "ftol": 1e-10}
    if solver_options:
        opts.update(solver_options)

    res = optimize.minimize(
        fn, w0, jac=grad, method="SLSQP",
        bounds=bounds, constraints=constraints, options=opts,
    )
    if not res.success:
        # Retry with loosened ftol (SLSQP sometimes reports failure when
        # the KKT conditions are effectively satisfied numerically)
        retry_opts = dict(opts)
        retry_opts.update({"ftol": 1e-6, "maxiter": 1000})
        res = optimize.minimize(
            fn, res.x if res.x is not None else w0, jac=grad, method="SLSQP",
            bounds=bounds, constraints=constraints, options=retry_opts,
        )
    w = np.asarray(res.x, dtype=float)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s > 0:
        w = w / s
    # Verify feasibility; if delta too tight, raise with a helpful message.
    viol = np.max(np.abs(X_arm.T @ w - target) - tol)
    if viol > 1e-5:
        raise ValueError(
            f"SBW infeasible at delta; max constraint violation = {viol:.4g}. "
            "Loosen `delta` or check for perfect separation in covariates."
        )
    return w


# ═══════════════════════════════════════════════════════════════════════
#  Diagnostics
# ═══════════════════════════════════════════════════════════════════════

def _balance_table(
    X: np.ndarray,
    T: np.ndarray,
    w: np.ndarray,
    cov_names: List[str],
    estimand: str,
) -> pd.DataFrame:
    """Per-covariate standardised mean differences before / after."""
    X_t, X_c = X[T == 1], X[T == 0]
    w_t, w_c = w[T == 1], w[T == 0]
    sd_pooled = np.sqrt(0.5 * (X_t.var(axis=0, ddof=0) + X_c.var(axis=0, ddof=0)))
    sd_pooled = np.where(sd_pooled < 1e-12, 1.0, sd_pooled)

    def _mean(x, ww):
        s = ww.sum()
        return (x * ww[:, None]).sum(axis=0) / s if s > 0 else x.mean(axis=0)

    m_t_raw = X_t.mean(axis=0)
    m_c_raw = X_c.mean(axis=0)
    m_t_w = _mean(X_t, w_t)
    m_c_w = _mean(X_c, w_c)

    smd_before = (m_t_raw - m_c_raw) / sd_pooled
    smd_after = (m_t_w - m_c_w) / sd_pooled

    return pd.DataFrame({
        "mean_treated": m_t_w,
        "mean_control": m_c_w,
        "SMD_before": smd_before,
        "SMD_after": smd_after,
    }, index=cov_names)


def _weighted_treatment_effect(
    Y: np.ndarray, T: np.ndarray, w: np.ndarray,
    estimand: str, alpha: float,
):
    """Point estimate + *conditional-on-weights* SE for the target estimand.

    Uses a Horvitz-Thompson variance that treats the SBW weights as
    fixed — i.e. does not propagate uncertainty from weight
    estimation. For most reasonable δ and sample sizes this is within
    5–10% of a Bayesian-bootstrap SE (Wang & Zubizarreta 2020), but
    users who need a fully correct variance should call :func:`sbw` in
    a bootstrap loop over ``data``.
    """
    from scipy import stats as _stats

    n = len(Y)
    w = np.asarray(w, dtype=float)

    if estimand == "att":
        # ATT = mean_T(Y) - Σ w_i Y_i over controls
        w_t = w[T == 1]
        w_c = w[T == 0]
        mu_t = float(np.sum(w_t * Y[T == 1]) / w_t.sum())
        mu_c = float(np.sum(w_c * Y[T == 0]) / w_c.sum())
        ate = mu_t - mu_c
        # Sandwich-type SE: Var(mu_t) via iid + Var(mu_c) via Σ w_i² · (Y_i - mu_c)²
        var_t = float(np.var(Y[T == 1], ddof=1) / (T == 1).sum())
        resid_c = Y[T == 0] - mu_c
        var_c = float(np.sum((w_c ** 2) * (resid_c ** 2)))
    elif estimand == "atc":
        w_t = w[T == 1]
        w_c = w[T == 0]
        mu_t = float(np.sum(w_t * Y[T == 1]) / w_t.sum())
        mu_c = float(np.sum(w_c * Y[T == 0]) / w_c.sum())
        ate = mu_t - mu_c
        resid_t = Y[T == 1] - mu_t
        var_t = float(np.sum((w_t ** 2) * (resid_t ** 2)))
        var_c = float(np.var(Y[T == 0], ddof=1) / (T == 0).sum())
    else:  # ate
        w_t = w[T == 1]
        w_c = w[T == 0]
        mu_t = float(np.sum(w_t * Y[T == 1]) / w_t.sum())
        mu_c = float(np.sum(w_c * Y[T == 0]) / w_c.sum())
        ate = mu_t - mu_c
        resid_t = Y[T == 1] - mu_t
        resid_c = Y[T == 0] - mu_c
        var_t = float(np.sum((w_t ** 2) * (resid_t ** 2)))
        var_c = float(np.sum((w_c ** 2) * (resid_c ** 2)))

    se = float(np.sqrt(var_t + var_c))
    z = float(_stats.norm.ppf(1 - alpha / 2))
    low, high = ate - z * se, ate + z * se
    t_stat = ate / se if se > 0 else np.nan
    pval = float(2 * (1 - _stats.norm.cdf(abs(t_stat)))) if se > 0 else np.nan
    return ate, se, low, high, pval


__all__ = ["sbw", "SBWResult"]
