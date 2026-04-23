"""
Covariate-Balancing Propensity Score (Imai & Ratkovic 2014).

CBPS estimates the propensity score by solving a moment condition that
*jointly* enforces:

    (a) the logit score equation (standard MLE first-order condition);
    (b) exact mean-balance of covariates under the implied IPW weights.

The "just-identified" (exact) variant uses ``K`` moment conditions where
``K`` equals the covariate dimension (drops the score equation).
The "over-identified" variant stacks both sets and solves via GMM.
This module implements both.

Mathematically, denote ``π(X; β) = 1 / (1 + exp(-X'β))``. The
over-identified moment vector for ATE is

    g_i(β) = [ (T_i - π_i) * X_i ,                (MLE)
               (T_i - π_i) / (π_i (1 - π_i)) X_i ] (Balance)

CBPS minimises ``ḡ' W ḡ`` with W = identity for the exact case
(K equations, K unknowns → method of moments) or with the efficient GMM
weighting matrix for the over-identified case.

Treatment-effect point estimate uses the resulting weights in the
standard (normalised Hajek) IPW formula; SEs come from a paired
bootstrap re-estimation by default.

References
----------
Imai, K., Ratkovic, M. (2014). "Covariate Balancing Propensity Score."
JRSS-B, 76(1), 243-263. [@imai2014covariate]

Fong, C., Ratkovic, M., Imai, K. (2022). ``CBPS`` R package documentation.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import stats as sp_stats

from ..core.results import CausalResult


def cbps(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    estimand: Literal["ATE", "ATT"] = "ATE",
    variant: Literal["exact", "over"] = "over",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    add_intercept: bool = True,
    trim: float = 0.0,
) -> CausalResult:
    """Covariate-Balancing Propensity Score estimator (Imai-Ratkovic 2014).

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome column.
    treat : str
        Binary 0/1 treatment column.
    covariates : list of str
        Covariates entering the logit score.
    estimand : {'ATE', 'ATT'}
    variant : {'exact', 'over'}
        'exact': just-identified CBPS (only balance moments). 'over':
        over-identified CBPS (MLE + balance, solved via two-step GMM).
    n_bootstrap : int
    alpha : float
    seed : int, optional
    add_intercept : bool, default True
        Prepend a constant to the covariate matrix.
    trim : float
        Optional pscore clip for stability.

    Returns
    -------
    CausalResult
        ``estimate`` is the CBPS weighted treatment effect; ``model_info``
        contains the estimated coefficients, balance diagnostics and
        effective sample size.
    """
    if estimand not in ("ATE", "ATT"):
        raise ValueError(f"estimand must be 'ATE' or 'ATT', got {estimand!r}")
    if variant not in ("exact", "over"):
        raise ValueError(f"variant must be 'exact' or 'over', got {variant!r}")

    rng = np.random.default_rng(seed)

    df = data[[y, treat] + list(covariates)].dropna().copy()
    Y = df[y].to_numpy(dtype=np.float64)
    T = df[treat].to_numpy(dtype=np.float64)
    X = df[covariates].to_numpy(dtype=np.float64)
    if add_intercept:
        X = np.column_stack([np.ones(len(df)), X])
    n, p = X.shape

    def _solve(X_, T_):
        return _fit_cbps(X_, T_, estimand=estimand, variant=variant)

    beta_hat, ps, converged_pt, obj_pt = _solve(X, T)
    if trim > 0:
        ps = np.clip(ps, trim, 1 - trim)
    w1, w0 = _cbps_weights(T, ps, estimand)
    est = float(np.sum(w1 * Y) - np.sum(w0 * Y))

    # Bootstrap with draw-until-success. Pathological resamples (all
    # treated or all controls, optimizer failure, singular Hessian) are
    # discarded and re-drawn so that we return exactly ``n_bootstrap``
    # successful reps. A hard ceiling on retries guards against
    # degenerate DGPs.
    boot = np.empty(n_bootstrap)
    boot_converged = np.empty(n_bootstrap, dtype=bool)
    max_retries = 10 * n_bootstrap
    retries = 0
    b = 0
    while b < n_bootstrap and retries < max_retries:
        idx = rng.integers(0, n, size=n)
        T_b = T[idx]
        # Skip degenerate resamples where the treatment or control arm
        # disappeared — CBPS has no solution in that subsample.
        if T_b.sum() < 2 or (1 - T_b).sum() < 2:
            retries += 1
            continue
        X_b, Y_b = X[idx], Y[idx]
        try:
            _, ps_b, conv_b, _ = _solve(X_b, T_b)
            if trim > 0:
                ps_b = np.clip(ps_b, trim, 1 - trim)
            w1b, w0b = _cbps_weights(T_b, ps_b, estimand)
            boot[b] = float(np.sum(w1b * Y_b) - np.sum(w0b * Y_b))
            boot_converged[b] = conv_b
            b += 1
        except Exception:
            retries += 1
            continue
    boot = boot[:b]
    boot_converged = boot_converged[:b]
    n_boot_success = b
    n_boot_nonconv = int((~boot_converged).sum())
    se = float(np.std(boot, ddof=1)) if boot.size > 1 else np.nan
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (est - z * se, est + z * se) if np.isfinite(se) else (np.nan, np.nan)
    pval = float(2 * (1 - sp_stats.norm.cdf(abs(est) / se))) if se and se > 0 else np.nan

    # Balance diagnostics: std mean difference after weighting
    w_overall = np.where(T == 1, w1, w0)
    mean_t = (X[T == 1] * w1[T == 1, None]).sum(axis=0) / max(w1[T == 1].sum(), 1e-12)
    mean_c = (X[T == 0] * w0[T == 0, None]).sum(axis=0) / max(w0[T == 0].sum(), 1e-12)
    pooled_sd = np.sqrt(0.5 * (X[T == 1].var(axis=0) + X[T == 0].var(axis=0)) + 1e-12)
    smd = (mean_t - mean_c) / pooled_sd

    model_info = {
        "model_type": f"CBPS ({variant})",
        "estimand": estimand,
        "beta": beta_hat,
        "n_treated": int(T.sum()),
        "n_control": int((1 - T).sum()),
        "pscore_min": float(ps.min()),
        "pscore_max": float(ps.max()),
        "std_mean_diff_after": dict(
            zip(["_intercept"] + list(covariates) if add_intercept else list(covariates),
                smd.tolist())
        ),
        "converged": converged_pt,
        "gmm_objective": obj_pt,
        "n_bootstrap": n_bootstrap,
        "n_bootstrap_success": n_boot_success,
        "n_bootstrap_nonconverged": n_boot_nonconv,
        "n_bootstrap_retries": retries,
    }

    return CausalResult(
        method=f"CBPS ({variant}, {estimand})",
        estimand=estimand,
        estimate=est,
        se=se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
    )


# ======================================================================
# Core solver
# ======================================================================


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _score_and_balance(
    beta: np.ndarray, X: np.ndarray, T: np.ndarray, estimand: str
) -> np.ndarray:
    """Return the stacked 2K moment vector (score + balance) for ATE."""
    ps = _sigmoid(X @ beta)
    eps = 1e-8
    ps = np.clip(ps, eps, 1 - eps)
    if estimand == "ATE":
        # MLE score
        g1 = (T - ps)[:, None] * X
        # Balance moment: weighted covariate difference = 0
        w = (T - ps) / (ps * (1 - ps))
        g2 = w[:, None] * X
    else:  # ATT
        # Balance the treated-group mean and IPW-weighted control-group mean
        g1 = (T - ps)[:, None] * X
        w = T - (1 - T) * ps / (1 - ps)
        g2 = w[:, None] * X
    return np.concatenate([g1.mean(axis=0), g2.mean(axis=0)])


def _balance_only(
    beta: np.ndarray, X: np.ndarray, T: np.ndarray, estimand: str
) -> np.ndarray:
    ps = _sigmoid(X @ beta)
    eps = 1e-8
    ps = np.clip(ps, eps, 1 - eps)
    if estimand == "ATE":
        w = (T - ps) / (ps * (1 - ps))
    else:
        w = T - (1 - T) * ps / (1 - ps)
    return (w[:, None] * X).mean(axis=0)


def _fit_cbps(
    X: np.ndarray,
    T: np.ndarray,
    estimand: str,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Fit CBPS by minimising ``||g(β)||²`` (quadratic in moments).

    For ``variant == 'exact'``, solves the K balance equations directly.
    For ``variant == 'over'``, two-step GMM: step 1 uses identity
    weighting, step 2 uses the efficient weighting
    ``(E[g g'])^{-1}``.

    Returns
    -------
    beta : ndarray
        Final logit coefficients.
    ps : ndarray
        Final propensity scores, clipped to (ε, 1-ε) for stability.
    converged : bool
        Whether the final optimiser call reported convergence.
    obj_value : float
        Final GMM objective. Useful as a warm-start quality gauge.
    """
    # Warm start: prefer statsmodels Logit (Newton-Raphson, exact Hessian
    # — more numerically stable and closer to R `glm` than sklearn's L2-
    # regularised LBFGS solver). Fallback to sklearn or to zeros if
    # neither is available / convergent on this resample.
    beta0 = _warm_start_logit(X, T)

    if variant == "exact":
        def obj(b: np.ndarray) -> float:
            g = _balance_only(b, X, T, estimand)
            return float(g @ g)
        res = optimize.minimize(obj, beta0, method="BFGS", options={"maxiter": 200})
        beta = res.x
        converged = bool(res.success)
        obj_value = float(res.fun)
    else:
        # Step 1: identity-weighted GMM on stacked moments
        def obj1(b: np.ndarray) -> float:
            g = _score_and_balance(b, X, T, estimand)
            return float(g @ g)
        res1 = optimize.minimize(obj1, beta0, method="BFGS", options={"maxiter": 200})
        beta1 = res1.x

        # Step 2: efficient weighting W = (E[g g'])^{-1}
        ps1 = _sigmoid(X @ beta1)
        ps1 = np.clip(ps1, 1e-8, 1 - 1e-8)
        if estimand == "ATE":
            g1 = (T - ps1)[:, None] * X
            w_all = (T - ps1) / (ps1 * (1 - ps1))
            g2 = w_all[:, None] * X
        else:
            g1 = (T - ps1)[:, None] * X
            w_all = T - (1 - T) * ps1 / (1 - ps1)
            g2 = w_all[:, None] * X
        G = np.concatenate([g1, g2], axis=1)
        S = G.T @ G / X.shape[0]
        # Ridge-regularised pinv for singular S
        ridge = 1e-8 * (np.trace(S) / S.shape[0] + 1.0)
        try:
            W = np.linalg.pinv(S + ridge * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            W = np.eye(S.shape[0])

        def obj2(b: np.ndarray) -> float:
            g = _score_and_balance(b, X, T, estimand)
            return float(g @ W @ g)
        res2 = optimize.minimize(obj2, beta1, method="BFGS", options={"maxiter": 200})
        beta = res2.x
        converged = bool(res2.success)
        obj_value = float(res2.fun)

    ps_final = _sigmoid(X @ beta)
    ps_final = np.clip(ps_final, 1e-8, 1 - 1e-8)
    return beta, ps_final, converged, obj_value


def _warm_start_logit(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Warm-start coefficients for CBPS: Newton-Raphson logit via
    statsmodels if available; else sklearn LogisticRegression; else
    zeros. Silent warnings from a non-converging fit are swallowed — a
    mediocre warm start is still useful to BFGS on the GMM objective.
    """
    import warnings

    try:
        import statsmodels.api as sm  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.Logit(T, X).fit(disp=False, maxiter=100)
        return np.asarray(model.params, dtype=np.float64)
    except Exception:
        pass

    try:
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(max_iter=1000, solver="lbfgs", C=1e6, fit_intercept=False)
        m.fit(X, T)
        return m.coef_[0].astype(np.float64)
    except Exception:
        return np.zeros(X.shape[1], dtype=np.float64)


def _cbps_weights(
    T: np.ndarray, ps: np.ndarray, estimand: str
) -> tuple[np.ndarray, np.ndarray]:
    """Hajek-normalised weights implied by CBPS."""
    if estimand == "ATE":
        w1 = T / ps
        w0 = (1 - T) / (1 - ps)
    else:  # ATT
        w1 = T.copy()
        w0 = (1 - T) * ps / (1 - ps)
    s1 = w1.sum()
    s0 = w0.sum()
    if s1 > 0:
        w1 = w1 / s1
    if s0 > 0:
        w0 = w0 / s0
    return w1, w0


__all__ = ["cbps"]
