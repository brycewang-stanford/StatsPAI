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

Mathematically, denote ``π(X; β) = 1 / (1 + exp(-X'β))``. The stacked
moment vector, scaled by ``1/n`` as in the reference implementation, is

    ḡ(β) = [ X'(T - π) / n ,     (MLE score)
             X'w(β)      / n ]   (Balance)

with ``w = T/π - (1-T)/(1-π)`` for the ATE and
``w = (n/n_1)·(T - π)/(1 - π)`` for the ATT. CBPS minimises
``ḡ' V(β₀)⁻¹ ḡ`` for the over-identified variant, where ``V`` is the
*model-implied* moment covariance frozen at the starting value, and
minimises the balance block alone for the just-identified variant. Both
problems are solved in a standardised, orthonormalised basis; see the
solver note above ``_fit_cbps`` for why that basis is load-bearing.

Treatment-effect point estimate uses the resulting weights in the
standard (normalised Hajek) IPW formula; SEs come from a paired
bootstrap re-estimation by default.

Relationship to the R package
-----------------------------
``sp.cbps`` reproduces ``CBPS::CBPS`` for ``estimand='ATE'`` (both
variants) and for ``estimand='ATT', variant='exact'`` to ~1e-3 relative,
the residual being the R optimiser's own convergence slack — StatsPAI
drives the just-identified balance loss to ~1e-20 where CBPS stops
around 1e-10.

For ``estimand='ATT', variant='over'`` the two implementations land on
different points. CBPS's analytic ATT gradient scales the balance block
by ``1/n_1`` where the moment's Jacobian carries ``1/n``, overstating
that block by ``n/n_1``, and its ``optim`` call stops at a
non-stationary point as a result. StatsPAI uses the correct Jacobian and
attains both a lower GMM objective and better covariate balance (on
``MatchIt::lalonde``: max |SMD| 0.037 vs 0.106, mean 0.016 vs 0.034).
This is asserted in ``tests/reference_parity/test_matching_r_parity.py``.

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

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> result = sp.cbps(df, y='log_wage', treat='union',
    ...                  covariates=['education', 'experience', 'tenure'],
    ...                  estimand='ATT', n_bootstrap=50, seed=42)
    >>> result.summary()
    >>> result.model_info['std_mean_diff_after']  # balance after weighting

    >>> # Just-identified CBPS (balance moments only)
    >>> result = sp.cbps(df, y='log_wage', treat='union',
    ...                  covariates=['education', 'experience', 'tenure'],
    ...                  variant='exact', n_bootstrap=50, seed=42)
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

    def _solve(
        X_: np.ndarray,
        T_: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, float]:
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
    boot_used = boot[:b]
    boot_converged_used = boot_converged[:b]
    n_boot_success = b
    n_boot_nonconv = int((~boot_converged_used).sum())
    se = float(np.std(boot_used, ddof=1)) if boot_used.size > 1 else np.nan
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci = (est - z * se, est + z * se) if np.isfinite(se) else (np.nan, np.nan)
    pval = (
        float(2 * (1 - sp_stats.norm.cdf(abs(est) / se))) if se and se > 0 else np.nan
    )

    # Balance diagnostics: std mean difference after weighting
    mean_t = (X[T == 1] * w1[T == 1, None]).sum(axis=0) / max(w1[T == 1].sum(), 1e-12)
    mean_c = (X[T == 0] * w0[T == 0, None]).sum(axis=0) / max(w0[T == 0].sum(), 1e-12)
    pooled_sd = np.sqrt(0.5 * (X[T == 1].var(axis=0) + X[T == 0].var(axis=0)) + 1e-12)
    smd = (mean_t - mean_c) / pooled_sd
    balance_labels = (
        ["_intercept"] + list(covariates) if add_intercept else list(covariates)
    )

    model_info = {
        "model_type": f"CBPS ({variant})",
        "estimand": estimand,
        "beta": beta_hat,
        "n_treated": int(T.sum()),
        "n_control": int((1 - T).sum()),
        "pscore_min": float(ps.min()),
        "pscore_max": float(ps.max()),
        "std_mean_diff_after": dict(zip(balance_labels, smd.tolist())),
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
#
# This is a faithful port of ``CBPS:::CBPS.2Treat`` (CBPS 0.24). Every
# convention that affects the numerical answer is reproduced:
#
# 1. The design matrix is standardised (non-intercept columns centred and
#    scaled by their sample sd) and then replaced by the left singular
#    vectors ``U`` of the standardised matrix. The GMM problem is solved
#    in that orthonormal basis and the coefficients are transformed back
#    at the end. This is not cosmetic: the GMM weighting matrix and the
#    just-identified quadratic form are basis dependent, so solving in the
#    raw basis gives a *different* estimator.
# 2. ``XprimeX.inv`` (the metric of the balance-only objective) is
#    ``ginv(X'X)``, which equals the identity in the ``U`` basis.
# 3. The GMM weighting matrix is the *model-implied* moment covariance
#    evaluated at the starting value (``twostep=TRUE``), not the empirical
#    outer-product of the moments.
# 4. The optimiser sequence is: logit MLE -> scalar rescaling on
#    [0.8, 1.1] -> BFGS on the balance loss -> (for ``method='over'``) two
#    BFGS runs on the GMM loss started from the MLE and from the balance
#    solution, keeping whichever attains the lower objective.
#
# Reference: Imai & Ratkovic (2014), JRSS-B 76(1), 243-263
# [@imai2014covariate]; Fong, Hazlett & Imai (2018) implementation.
# ======================================================================

_PROBS_MIN = 1e-6


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(z, -35, 35))), dtype=float)


def _probs(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    ps = _sigmoid(X @ beta)
    return np.clip(ps, _PROBS_MIN, 1.0 - _PROBS_MIN)


def _balance_weights(
    ps: np.ndarray, T: np.ndarray, att: bool, n_t: float
) -> np.ndarray:
    """The un-normalised balance-moment weights ``w`` of CBPS.2Treat.

    ATT: ``(n / n_t) * (T - p) / (1 - p)``  (``ATT.wt.func``)
    ATE: ``(p - 1 + T)^{-1}``               (``T/p - (1-T)/(1-p)``)
    """
    n = float(T.size)
    if att:
        return np.asarray((n / n_t) * (T - ps) / (1.0 - ps), dtype=float)
    return np.asarray(1.0 / (ps - 1.0 + T), dtype=float)


def _gbar(
    beta: np.ndarray, X: np.ndarray, T: np.ndarray, att: bool, n_t: float
) -> np.ndarray:
    """Stacked ``[score, balance]`` sample moments, both scaled by ``1/n``."""
    n = float(T.size)
    ps = _probs(beta, X)
    w = _balance_weights(ps, T, att, n_t)
    return np.concatenate([X.T @ (T - ps) / n, X.T @ w / n])


def _inv_v(beta: np.ndarray, X: np.ndarray, T: np.ndarray, att: bool) -> np.ndarray:
    """Model-implied inverse moment covariance (CBPS.2Treat ``invV``)."""
    n = float(T.size)
    n_t = float(T.sum())
    ps = _probs(beta, X)
    if att:
        X1 = X * np.sqrt(ps * (1.0 - ps))[:, None]
        X2 = X * np.sqrt(ps / (1.0 - ps))[:, None]
        X11 = X * np.sqrt(ps)[:, None]
        A = X1.T @ X1
        B = (X11.T @ X11) * (n / n_t)
        C = (X2.T @ X2) * (n**2 / n_t**2)
    else:
        X1 = X * np.sqrt(ps * (1.0 - ps))[:, None]
        X2 = X / np.sqrt(ps * (1.0 - ps))[:, None]
        X11 = X
        A = X1.T @ X1
        B = X11.T @ X11
        C = X2.T @ X2
    V = np.block([[A, B], [B, C]]) / n
    return np.asarray(np.linalg.pinv(V), dtype=float)


def _fit_cbps(
    X: np.ndarray,
    T: np.ndarray,
    estimand: str,
    variant: str,
    twostep: bool = True,
    iterations: int = 1000,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Fit CBPS following ``CBPS:::CBPS.2Treat`` (CBPS 0.24).

    ``X`` is the *raw* design matrix whose first column is the intercept.
    The solver internally standardises and orthonormalises it (see the
    module-level note), then maps the coefficients back to the raw scale
    so the returned ``beta`` is directly comparable to ``coef(CBPS(...))``.

    Parameters
    ----------
    X, T : ndarray
        Design matrix (intercept first) and 0/1 treatment.
    estimand : {'ATE', 'ATT'}
    variant : {'exact', 'over'}
        ``'exact'`` minimises the balance loss only (just-identified);
        ``'over'`` minimises the over-identified GMM loss.
    twostep : bool, default True
        Hold the GMM weighting matrix fixed at its value under the logit
        starting value, as ``CBPS(..., twostep = TRUE)`` does.
    iterations : int, default 1000
        Maximum BFGS iterations, mirroring CBPS's ``iterations``.

    Returns
    -------
    beta : ndarray
        Coefficients on the raw ``X`` scale.
    ps : ndarray
        Fitted propensity scores, clipped to ``(1e-6, 1 - 1e-6)``.
    converged : bool
    obj_value : float
        Final objective (balance loss or GMM J).
    """
    att = estimand == "ATT"
    bal_only = variant == "exact"
    n = float(T.size)
    n_t = float(T.sum())

    # --- Standardise + orthonormalise (CBPS.fit) -------------------------
    # CBPS always carries an intercept and standardises every *other*
    # column. Locate the constant column rather than assuming position 0,
    # so a caller-supplied design matrix cannot be silently mis-scaled.
    X_in = np.asarray(X, dtype=float)
    col_sd = X_in.std(axis=0, ddof=1)
    const_cols = np.flatnonzero(col_sd <= 0)
    if const_cols.size != 1:
        raise ValueError(
            "CBPS requires a design matrix with exactly one constant "
            f"(intercept) column; found {const_cols.size}. Pass "
            "add_intercept=True and drop any constant covariates."
        )
    icept = int(const_cols[0])
    order = np.concatenate([[icept], np.delete(np.arange(X_in.shape[1]), icept)])
    inverse_order = np.argsort(order)
    X_raw = X_in[:, order]
    x_mean = X_raw[:, 1:].mean(axis=0)
    x_sd = X_raw[:, 1:].std(axis=0, ddof=1)
    X_std = X_raw.copy()
    X_std[:, 1:] = (X_raw[:, 1:] - x_mean) / x_sd
    U, d_sv, Vt = np.linalg.svd(X_std, full_matrices=False)
    if np.linalg.matrix_rank(U) < U.shape[1]:
        raise ValueError("CBPS design matrix is not full rank.")
    Xs = U  # solve in the orthonormal basis, as CBPS does
    XpX_inv = np.linalg.pinv(Xs.T @ Xs)

    # --- Objectives ------------------------------------------------------
    def bal_loss(beta: np.ndarray) -> float:
        ps = _probs(beta, Xs)
        w = _balance_weights(ps, T, att, n_t) / n
        xw = Xs.T @ w
        return float(abs(xw @ XpX_inv @ xw))

    def bal_gradient(beta: np.ndarray) -> np.ndarray:
        ps = _probs(beta, Xs)
        w = _balance_weights(ps, T, att, n_t) / n
        if att:
            dw2 = -(n / n_t) * ps / (1.0 - ps)
            dw2 = np.where(T == 1, 0.0, dw2)
        else:
            dw2 = -((T - ps) ** 2) / (ps * (1.0 - ps))
        dw = (Xs * dw2[:, None]).T / n
        xw = Xs.T @ w
        loss1 = xw @ XpX_inv @ xw
        raw = 2.0 * (dw @ Xs @ XpX_inv @ xw)
        # CBPS's sign convention on the balance gradient.
        sign = np.where(
            ((raw > 0) & (loss1 > 0)) | ((raw < 0) & (loss1 < 0)), 1.0, -1.0
        )
        return np.asarray(sign * np.abs(raw), dtype=float)

    def gmm_loss(beta: np.ndarray, invV: np.ndarray) -> float:
        g = _gbar(beta, Xs, T, att, n_t)
        return float(g @ invV @ g)

    def gmm_gradient(beta: np.ndarray, invV: np.ndarray) -> np.ndarray:
        ps = _probs(beta, Xs)
        g = _gbar(beta, Xs, T, att, n_t)
        block1 = (Xs * (-ps * (1.0 - ps))[:, None]).T @ Xs / n
        if att:
            dw = -(n / n_t) * ps / (1.0 - ps)
            dw = np.where(T == 1, 0.0, dw)
        else:
            dw = -((T - ps) ** 2) / (ps * (1.0 - ps))
        # NOTE: the balance moment is (1/n) X'w for both estimands, so its
        # Jacobian carries 1/n. CBPS's own ATT gradient divides by n_t
        # instead, which overstates that block by a factor n/n_t and
        # leaves optim() stopping at a non-stationary point. We use the
        # correct Jacobian and verify convergence on the objective itself.
        block2 = (Xs * dw[:, None]).T @ Xs / n
        dgbar = np.concatenate([block1, block2], axis=1)
        return np.asarray(2.0 * (dgbar @ invV @ g), dtype=float)

    # --- Starting values: logit MLE, then a scalar rescaling -------------
    # CBPS rescales the MLE by the alpha in [0.8, 1.1] that minimises the
    # *continuously updated* GMM loss (invV recomputed at each candidate),
    # then freezes invV at the resulting point for the twostep objective.
    beta_glm = _warm_start_logit(Xs, T)

    def _cu_loss(a: float) -> float:
        b = beta_glm * a
        return gmm_loss(b, _inv_v(b, Xs, T, att))

    scal = optimize.minimize_scalar(
        _cu_loss, bounds=(0.8, 1.1), method="bounded", options={"xatol": 1e-10}
    )
    gmm_init = beta_glm * float(scal.x)
    this_invV = _inv_v(gmm_init, Xs, T, att)

    opts = {"maxiter": iterations, "gtol": 1e-12}
    opt_bal = optimize.minimize(
        bal_loss, gmm_init, method="BFGS", jac=bal_gradient, options=opts
    )

    if bal_only:
        best = opt_bal
    else:
        invV = this_invV if twostep else None

        def _loss(b: np.ndarray) -> float:
            return gmm_loss(b, invV if invV is not None else _inv_v(b, Xs, T, att))

        def _grad(b: np.ndarray) -> np.ndarray:
            return gmm_gradient(b, invV if invV is not None else _inv_v(b, Xs, T, att))

        # The over-identified CBPS objective is not convex: CBPS itself
        # hedges by starting BFGS from both the MLE and the balance
        # solution and keeping the better one. We start from those two
        # plus the unscaled MLE, then restart BFGS from the incumbent
        # until it stops improving (a fresh restart discards the inverse-
        # Hessian approximation and reliably clears the stalls that leave
        # optim() at a non-stationary point).
        best = None
        for start in (gmm_init, opt_bal.x, beta_glm):
            cand = optimize.minimize(
                _loss, start, method="BFGS", jac=_grad, options=opts
            )
            if best is None or cand.fun < best.fun:
                best = cand
        for _ in range(5):
            polished = optimize.minimize(
                _loss, best.x, method="BFGS", jac=_grad, options=opts
            )
            if not polished.fun < best.fun * (1 - 1e-12):
                break
            best = polished

    beta_svd = np.asarray(best.x, dtype=float)
    ps_final = _probs(beta_svd, Xs)

    # Convergence is judged on the *problem*, not on the optimiser's exit
    # flag. BFGS is run with a deliberately tight gtol so it keeps
    # descending, which means it almost always terminates on precision
    # loss and reports success=False even at an excellent solution --
    # propagating that flag would mark nearly every fit non-converged.
    # What matters is whether the first-order condition actually holds:
    # for the just-identified variant, whether the balance moments are
    # zero; for the over-identified one, whether the GMM gradient is.
    if bal_only:
        converged = bool(bal_loss(beta_svd) < 1e-12)
    else:
        invV_final = this_invV if twostep else _inv_v(beta_svd, Xs, T, att)
        grad_norm = float(np.max(np.abs(gmm_gradient(beta_svd, invV_final))))
        scale = max(1.0, abs(float(best.fun)))
        converged = bool(grad_norm < 1e-6 * scale)

    # --- Map coefficients back to the raw scale (CBPS.fit) ---------------
    d_inv = np.where(d_sv > 1e-5, 1.0 / np.where(d_sv > 1e-5, d_sv, 1.0), 0.0)
    beta_std = Vt.T @ (d_inv * beta_svd)
    beta_raw = beta_std.copy()
    beta_raw[1:] = beta_std[1:] / x_sd
    beta_raw[0] = beta_std[0] - x_mean @ beta_raw[1:]
    beta_raw = beta_raw[inverse_order]  # restore the caller's column order

    return (
        beta_raw,
        np.asarray(ps_final, dtype=float),
        converged,
        float(best.fun),
    )


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

        m = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1e6,
            fit_intercept=False,
        )
        m.fit(X, T)
        return np.asarray(m.coef_[0], dtype=np.float64)
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
    return np.asarray(w1, dtype=float), np.asarray(w0, dtype=float)


__all__ = ["cbps"]
