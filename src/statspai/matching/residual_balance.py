"""
Approximate residual balancing for average treatment effects.

With many covariates, weights that balance every covariate mean exactly
may not exist. Approximate balancing weights trade the size of the
weights against the worst remaining imbalance::

    minimise   (1 - zeta) * sum(gamma_i^2) + zeta * max_j |imbalance_j|^2
    subject to sum(gamma_i) = 1,  gamma_i >= 0

where ``imbalance = X_arm' gamma - target`` and the target is the
covariate mean of the population of interest. The weights are then
applied to the *residuals* of a regularised linear outcome model, so
that the regression removes most of the confounding and the weights
remove what the regression missed::

    mu_hat = target' beta_hat + sum_i gamma_i (Y_i - X_i' beta_hat)

The estimator is root-n consistent when the outcome model is linear and
sparse, without a model for the propensity score.

The weights are found from the dual problem, which has one variable
per covariate, so the cost grows with the number of covariates rather
than with the number of units.

References
----------
[@athey2018approximate]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core._validate import require_binary_treatment
from ..core.results import CausalResult
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility

# --------------------------------------------------------------------
# Balancing weights
# --------------------------------------------------------------------


def approx_balance_weights(
    M: np.ndarray,
    target: np.ndarray,
    zeta: float = 0.5,
    allow_negative_weights: bool = False,
    tol: float = 1e-12,
    max_iter: int = 20000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Weights on the rows of ``M`` whose weighted mean approximates ``target``.

    Solves ``min (1 - zeta) ||g||^2 + zeta ||M'g - target||_inf^2`` subject
    to ``sum(g) = 1`` (and ``g >= 0`` unless negative weights are allowed)
    through its dual.

    Returns the weights and a dict with the attained imbalance, the
    sum of squared weights and the solver status.
    """
    M = np.asarray(M, dtype=float)
    target = np.asarray(target, dtype=float).ravel()
    n, p = M.shape
    if target.shape[0] != p:
        raise MethodIncompatibility("target must have one entry per column of M")
    if not 0 < zeta < 1:
        raise MethodIncompatibility("zeta must be strictly between 0 and 1")
    a, b = 1.0 - zeta, zeta
    # Centring at the target leaves the problem unchanged (weights sum to
    # one) and conditions the dual.
    D = M - target

    # Dual variables are of order a / n; working in units of 2a / n keeps
    # the problem well scaled as zeta approaches 0 or 1.
    unit = 2 * a / n

    def weights_from(theta: np.ndarray, nu: float) -> np.ndarray:
        r = nu - D @ theta
        if not allow_negative_weights:
            r = np.maximum(r, 0.0)
        return np.asarray(r / (2 * a))

    def neg_dual(zs: np.ndarray) -> Tuple[float, np.ndarray]:
        z = zs * unit
        u, v, nu = z[:p], z[p : 2 * p], z[2 * p]
        theta = u - v
        s = float(u.sum() + v.sum())
        g = weights_from(theta, nu)
        imb = D.T @ g
        val = a * float(g @ g) + s * s / (4 * b) - nu
        grad = np.empty(2 * p + 1)
        grad[:p] = -imb + s / (2 * b)
        grad[p : 2 * p] = imb + s / (2 * b)
        grad[2 * p] = float(g.sum()) - 1.0
        return val / unit, grad

    z0 = np.zeros(2 * p + 1)
    z0[2 * p] = 1.0  # uniform weights
    bounds = [(0.0, None)] * (2 * p) + [(None, None)]
    res = optimize.minimize(
        neg_dual,
        z0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": max_iter,
            "maxfun": 10 * max_iter,
            "ftol": tol,
            "gtol": 1e-11,
            "maxcor": 30,
        },
    )
    theta = (res.x[:p] - res.x[p : 2 * p]) * unit
    nu = float(res.x[2 * p]) * unit
    theta, nu = _polish(D, theta, nu, a, b, allow_negative_weights)
    g = weights_from(theta, nu)
    total = float(g.sum())
    if not np.isfinite(total) or abs(total - 1.0) > 1e-4:
        raise MethodIncompatibility(
            "the balancing problem did not converge to weights summing to one "
            f"(sum = {total:.6g}); check the covariates for constant or "
            "duplicated columns"
        )
    g = g / total
    info = {
        "imbalance": float(np.max(np.abs(D.T @ g))) if p else 0.0,
        "sum_sq_weights": float(g @ g),
        "converged": bool(res.success),
        "iterations": int(res.nit),
    }
    if not res.success:
        warnings.warn(
            f"approximate balancing did not report convergence: {res.message}",
            ConvergenceWarning,
            stacklevel=2,
        )
    return g, info


def _polish(
    D: np.ndarray,
    theta: np.ndarray,
    nu: float,
    a: float,
    b: float,
    allow_negative_weights: bool,
    rounds: int = 500,
) -> Tuple[np.ndarray, float]:
    """Solve the optimality conditions exactly by an active-set iteration.

    The quasi-Newton iterate identifies which weights are positive and
    which covariates attain the worst imbalance; given those sets the
    solution is a linear system. Sets are then corrected one element at
    a time until every optimality condition holds. If that does not
    happen within ``rounds`` steps the quasi-Newton point is returned.
    """
    n, p = D.shape
    start = (theta, nu)
    r = nu - D @ theta
    pos = np.ones(n, dtype=bool) if allow_negative_weights else r > 0
    g = np.where(pos, r, 0.0) / (2 * a)
    imb = D.T @ g
    t = float(np.abs(imb).max()) if p else 0.0
    sign = np.where(np.abs(imb) >= t * (1 - 1e-4), np.sign(imb), 0.0)
    for _ in range(rounds):
        J = np.flatnonzero(sign)
        k = J.size
        if k == 0 or pos.sum() == 0:
            return start
        sg = sign[J]
        Dp = D[pos][:, J]
        # Unknowns (theta_J, nu). Rows: D_J' g = sign * t, then sum(g) = 1,
        # with g = (nu - D theta) / (2a) on the positive set and
        # t = sign' theta_J / (2b).
        K = np.zeros((k + 1, k + 1))
        K[:k, :k] = -Dp.T @ Dp / (2 * a) - np.outer(sg, sg) / (2 * b)
        K[:k, k] = Dp.sum(axis=0) / (2 * a)
        K[k, :k] = -Dp.sum(axis=0) / (2 * a)
        K[k, k] = pos.sum() / (2 * a)
        rhs = np.zeros(k + 1)
        rhs[k] = 1.0
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            return start
        th = np.zeros(p)
        th[J] = sol[:k]
        nu_new = float(sol[k])
        r = nu_new - D @ th
        g = np.where(pos, r, 0.0) / (2 * a)
        imb = D.T @ g
        t = float(sg @ sol[:k] / (2 * b))
        tol = 1e-10 * max(abs(t), 1e-12)
        # 1. A multiplier with the wrong sign: that constraint is not binding.
        wrong = sg * sol[:k]
        if wrong.min() < 0:
            sign[J[int(np.argmin(wrong))]] = 0.0
            continue
        if not allow_negative_weights:
            # 2. A unit in the positive set with a negative weight.
            neg = np.where(pos, r, 0.0)
            if neg.min() < -1e-14:
                pos[int(np.argmin(neg))] = False
                continue
            # 3. A unit outside the set that should carry weight.
            out = np.where(~pos, r, 0.0)
            if out.max() > 1e-14:
                pos[int(np.argmax(out))] = True
                continue
        # 4. An imbalance beyond the bound: that constraint must bind.
        excess = np.abs(imb) - t
        excess[J] = 0.0
        if excess.max() > tol:
            jmax = int(np.argmax(excess))
            sign[jmax] = np.sign(imb[jmax])
            continue
        return th, nu_new
    return start


# --------------------------------------------------------------------
# Outcome model
# --------------------------------------------------------------------


def _fit_outcome(
    X: np.ndarray,
    y: np.ndarray,
    outcome_model: Any,
    l1_ratio: float,
    cv: int,
    random_state: Optional[int],
) -> Tuple[np.ndarray, float, int]:
    """Linear fit on one arm; returns (coefficients, intercept, non-zeros)."""
    if isinstance(outcome_model, str) and outcome_model == "none":
        return np.zeros(X.shape[1]), 0.0, 0
    if isinstance(outcome_model, str):
        from sklearn.linear_model import ElasticNetCV, LassoCV
        from sklearn.model_selection import KFold

        folds = KFold(
            n_splits=min(cv, max(2, X.shape[0] // 2)),
            shuffle=True,
            random_state=random_state,
        )
        if outcome_model == "lasso":
            model = LassoCV(cv=folds, max_iter=20000)
        else:
            model = ElasticNetCV(l1_ratio=l1_ratio, cv=folds, max_iter=20000)
    else:
        from sklearn.base import clone

        model = clone(outcome_model)
    model.fit(X, y)
    if not hasattr(model, "coef_"):
        raise MethodIncompatibility(
            "outcome_model must be a linear model exposing coef_ and "
            "intercept_ after fitting"
        )
    coef = np.asarray(model.coef_, dtype=float).ravel()
    intercept = float(np.ravel(getattr(model, "intercept_", 0.0))[0])
    return coef, intercept, int(np.count_nonzero(coef))


def _arm_mean(
    X: np.ndarray,
    y: np.ndarray,
    target: np.ndarray,
    *,
    zeta: float,
    allow_negative_weights: bool,
    outcome_model: Any,
    l1_ratio: float,
    cv: int,
    random_state: Optional[int],
) -> Dict[str, Any]:
    """Residual-balanced estimate of the mean outcome at ``target``."""
    gamma, info = approx_balance_weights(
        X, target, zeta=zeta, allow_negative_weights=allow_negative_weights
    )
    coef, intercept, nnz = _fit_outcome(X, y, outcome_model, l1_ratio, cv, random_state)
    resid = y - X @ coef - intercept
    mean = float(target @ coef + intercept + gamma @ resid)
    n = X.shape[0]
    # Degrees-of-freedom correction for the selected regressors.
    dof = max(n - nnz - 1, 1)
    var = float(np.sum(gamma**2 * resid**2) * n / dof)
    return {
        "mean": mean,
        "var": var,
        "gamma": gamma,
        "coef": coef,
        "intercept": intercept,
        "nonzero": nnz,
        **info,
    }


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


@_expands_categorical("covariates")
def residual_balance(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    *,
    estimand: str = "ATE",
    zeta: float = 0.5,
    outcome_model: Any = "elnet",
    l1_ratio: float = 0.9,
    standardize: bool = True,
    allow_negative_weights: bool = False,
    cv: int = 10,
    random_state: Optional[int] = 0,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Approximate residual balancing estimate of an average treatment effect.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    treat : str
        Binary (0/1) treatment column.
    covariates : list of str
        Covariates to balance; there may be more of them than units.
    estimand : {"ATE", "ATT", "ATC"}, default "ATE"
        Population whose covariate mean is the balancing target: all
        units, the treated or the controls.
    zeta : float in (0, 1), default 0.5
        Weight on imbalance relative to the sum of squared weights.
        Values near 1 push towards exact balance, values near 0 towards
        uniform weights.
    outcome_model : {"elnet", "lasso", "none"} or estimator, default "elnet"
        Linear outcome model fitted separately in each arm.
        ``"elnet"`` is a cross-validated elastic net with mixing
        ``l1_ratio``; ``"lasso"`` a cross-validated lasso; ``"none"``
        skips the regression, leaving a pure weighting estimator. A
        scikit-learn linear estimator exposing ``coef_`` is also
        accepted.
    l1_ratio : float, default 0.9
        Elastic-net mixing parameter (1 is the lasso).
    standardize : bool, default True
        Divide each covariate by its standard deviation before
        balancing, so that the worst imbalance is measured in standard
        deviations.
    allow_negative_weights : bool, default False
        Drop the non-negativity constraint on the weights.
    cv : int, default 10
        Folds for the cross-validated outcome model.
    random_state : int, optional
        Seed for the cross-validation folds.
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``model_info`` holds the weights (``weights``), the attained
        imbalance in each reweighted arm, the number of selected
        regressors and the effective sample size.

    Notes
    -----
    The standard error is ``sqrt(sum_i gamma_i^2 * resid_i^2)`` summed
    over the reweighted arms, with a degrees-of-freedom correction for
    the selected regressors. It treats the covariates as fixed, so the
    interval is for the average effect in the sample at hand (the
    average of the conditional effects at the observed covariates). It
    is not available with ``outcome_model="none"``, where residuals
    still contain the covariate signal.

    With ``estimand="ATT"`` the treated mean is the sample mean of the
    treated outcomes and only the controls are reweighted (and
    symmetrically for ``"ATC"``).

    References
    ----------
    [@athey2018approximate]

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n, p = 300, 40
    >>> X = rng.normal(size=(n, p))
    >>> w = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
    >>> yv = X[:, 0] + 2 * X[:, 1] + w * 1.0 + rng.normal(size=n)
    >>> df = pd.DataFrame(X, columns=[f"x{j}" for j in range(p)])
    >>> df["y"], df["w"] = yv, w
    >>> r = sp.residual_balance(df, "y", "w", [f"x{j}" for j in range(p)])
    >>> bool(abs(r.estimate - 1.0) < 0.6)
    True
    """
    estimand_u = str(estimand).upper()
    if estimand_u not in ("ATE", "ATT", "ATC"):
        raise MethodIncompatibility("estimand must be 'ATE', 'ATT' or 'ATC'")
    if not 0 < alpha < 1:
        raise MethodIncompatibility("alpha must be in (0, 1)")
    if isinstance(outcome_model, str) and outcome_model not in (
        "elnet",
        "lasso",
        "none",
    ):
        raise MethodIncompatibility(
            "outcome_model must be 'elnet', 'lasso', 'none' or a linear estimator"
        )
    cols = [y, treat] + list(covariates)
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"columns not found in data: {missing}")
    if not covariates:
        raise MethodIncompatibility("at least one covariate is required")
    df = data[cols].dropna()
    require_binary_treatment(df[treat], function="residual_balance")
    W = df[treat].to_numpy(dtype=float).astype(int)
    Y = df[y].to_numpy(dtype=float)
    X = df[list(covariates)].to_numpy(dtype=float)
    n = X.shape[0]
    n1, n0 = int(W.sum()), int(n - W.sum())
    if n1 < 2 or n0 < 2:
        raise DataInsufficient("each treatment arm needs at least two units")

    if standardize:
        sd = X.std(axis=0, ddof=1)
        sd = np.where(sd > 1e-12, sd, 1.0)
        X = (X - X.mean(axis=0)) / sd
    if estimand_u == "ATE":
        target = X.mean(axis=0)
    elif estimand_u == "ATT":
        target = X[W == 1].mean(axis=0)
    else:
        target = X[W == 0].mean(axis=0)

    kw = dict(
        zeta=zeta,
        allow_negative_weights=allow_negative_weights,
        outcome_model=outcome_model,
        l1_ratio=l1_ratio,
        cv=cv,
        random_state=random_state,
    )
    arms: Dict[int, Optional[Dict[str, Any]]] = {0: None, 1: None}
    if estimand_u in ("ATE", "ATC"):
        arms[1] = _arm_mean(X[W == 1], Y[W == 1], target, **kw)
    if estimand_u in ("ATE", "ATT"):
        arms[0] = _arm_mean(X[W == 0], Y[W == 0], target, **kw)

    weights = np.zeros(n)
    means: Dict[int, float] = {}
    var = 0.0
    for w_val, n_arm in ((1, n1), (0, n0)):
        fit = arms[w_val]
        idx = W == w_val
        if fit is None:
            means[w_val] = float(Y[idx].mean())
            weights[idx] = 1.0 / n_arm
            if not (isinstance(outcome_model, str) and outcome_model == "none"):
                # The arm's own mean is known up to its outcome noise, so
                # its variance comes from regression residuals as well.
                coef, intercept, nnz = _fit_outcome(
                    X[idx], Y[idx], outcome_model, l1_ratio, cv, random_state
                )
                resid = Y[idx] - X[idx] @ coef - intercept
                dof = max(n_arm - nnz - 1, 1)
                var += float(np.sum(resid**2) / n_arm**2 * n_arm / dof)
        else:
            means[w_val] = fit["mean"]
            weights[idx] = fit["gamma"]
            var += fit["var"]
    estimate = means[1] - means[0]
    no_model = isinstance(outcome_model, str) and outcome_model == "none"
    se = np.nan if no_model else float(np.sqrt(var))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    if np.isfinite(se) and se > 0:
        pvalue = float(2 * stats.norm.sf(abs(estimate / se)))
        ci = (estimate - crit * se, estimate + crit * se)
    else:
        pvalue, ci = np.nan, (np.nan, np.nan)

    fitted = [fit for fit in (arms[1], arms[0]) if fit is not None]
    w_re = np.concatenate([fit["gamma"] for fit in fitted])
    model_info: Dict[str, Any] = {
        "zeta": zeta,
        "outcome_model": (
            outcome_model
            if isinstance(outcome_model, str)
            else type(outcome_model).__name__
        ),
        "standardize": standardize,
        "weights": weights,
        "mean_treated": means[1],
        "mean_control": means[0],
        "n_treated": n1,
        "n_control": n0,
        "n_covariates": X.shape[1],
        "effective_sample_size": float(
            sum(1.0 / np.sum(fit["gamma"] ** 2) for fit in fitted)
        ),
        "max_weight": float(w_re.max()),
        "se_conditional_on_covariates": True,
    }
    for w_val, label in ((1, "treated"), (0, "control")):
        fit = arms[w_val]
        if fit is not None:
            model_info[f"imbalance_{label}"] = fit["imbalance"]
            model_info[f"nonzero_{label}"] = fit["nonzero"]
            model_info[f"converged_{label}"] = fit["converged"]

    result = CausalResult(
        method="Approximate residual balancing",
        estimand=estimand_u,
        estimate=float(estimate),
        se=float(se),
        pvalue=float(pvalue),
        ci=(float(ci[0]), float(ci[1])),
        alpha=float(alpha),
        n_obs=int(n),
        model_info=model_info,
    )
    return result


__all__ = ["residual_balance", "approx_balance_weights"]
