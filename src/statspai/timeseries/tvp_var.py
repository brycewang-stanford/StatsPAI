"""Vector autoregression with time-varying coefficients.

    y_t = A_{1,t} y_{t-1} + ... + A_{p,t} y_{t-p} + c_t + u_t,
    beta_t = beta_{t-1} + eta_t,

``beta_t`` stacking every coefficient. Two deterministic estimators, no
simulation:

``method='kalman'``
    Each equation is a dynamic linear model with a constant error
    variance; the variances of the coefficient innovations are estimated
    by maximum likelihood, the paths by the Kalman filter and smoother.
    Equation by equation the numbers are those of :func:`statspai.dlm`.

``method='forgetting'``
    The forgetting-factor filter of Koop and Korobilis: the state
    covariance is inflated by ``1 / lam`` each period and the error
    covariance is an exponentially weighted moving average with decay
    ``kappa``. Nothing is estimated by search. With ``lam = 1`` and
    ``kappa = 1`` it is recursive least squares, so the last filtered
    coefficients are the ordinary VAR estimates.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    StatsPAIWarning,
)
from ._tvp_var_core import (
    forgetting_filter,
    kalman_equation,
    lag_design,
    mle_common,
    mle_free,
)
from ._tvp_var_results import TVPVARResult

__all__ = ["tvp_var", "TVPVARResult"]


def _prior(
    K: int,
    lags: int,
    prior: str,
    m0: Any,
    C0: Any,
    tightness: float,
    own_lag: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prior means and variances of the coefficients, both (K, K p + 1)."""
    k = K * lags + 1
    mean = np.zeros((K, k))
    var = np.empty((K, k))
    c0 = np.asarray(C0, dtype=float)
    if c0.ndim == 0:
        var[:] = float(c0)
    elif c0.shape in ((k,), (K, k)):
        var[:] = c0
    else:
        raise MethodIncompatibility(
            f"tvp_var: C0 must be a scalar, {k} variances or a {K} x {k} array."
        )
    if np.any(var <= 0) or not np.all(np.isfinite(var)):
        raise MethodIncompatibility("tvp_var: C0 must be positive and finite.")
    if prior == "minnesota":
        if not tightness > 0:
            raise MethodIncompatibility("tvp_var: prior_tightness must be positive.")
        for lag in range(1, lags + 1):
            var[:, (lag - 1) * K : lag * K] = tightness / lag**2
        for i in range(K):
            mean[i, i] = own_lag
    elif prior != "diffuse":
        raise MethodIncompatibility(
            f"tvp_var: prior={prior!r} is not 'diffuse' or 'minnesota'."
        )
    if m0 is not None:
        m = np.asarray(m0, dtype=float)
        if m.ndim == 0:
            mean[:] = float(m)
        elif m.shape in ((k,), (K, k)):
            mean[:] = m
        else:
            raise MethodIncompatibility(
                f"tvp_var: m0 must be a scalar, {k} values or a {K} x {k} array."
            )
    return mean, var


def tvp_var(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    lags: int = 1,
    method: str = "kalman",
    time: Optional[str] = None,
    common: bool = False,
    obs_var: Optional[Any] = None,
    state_var: Optional[Any] = None,
    lam: float = 0.99,
    kappa: float = 0.96,
    sigma0: Optional[Any] = None,
    sigma_update: str = "filtered",
    prior: str = "diffuse",
    m0: Optional[Any] = None,
    C0: Any = 1e7,
    prior_tightness: float = 0.1,
    prior_own_lag: float = 0.0,
    alpha: float = 0.05,
) -> TVPVARResult:
    """VAR whose intercepts and lag coefficients follow random walks.

    ``y_t = A_{1,t} y_{t-1} + ... + A_{p,t} y_{t-p} + c_t + u_t`` with
    every element of ``A_{l,t}`` and ``c_t`` a random walk. Returns the
    coefficient paths with standard errors, the error covariance, and
    impulse responses, companion roots and forecasts of the VAR frozen at
    a date.

    Parameters
    ----------
    data : DataFrame
        Rows in time order (or sorted by ``time=``), no missing values.
    variables : list of str, optional
        Default: every numeric column (except ``time``). Their order is
        the Cholesky order of ``.irf()``.
    lags : int, default 1
    method : {'kalman', 'forgetting'}
        ``'kalman'``: each equation is ``y_it = x_t' b_it + u_it`` with
        ``u_it ~ N(0, V_i)`` and ``b_it`` a random walk with diagonal
        innovation variance; the variances maximise the equation's
        Kalman-filter likelihood, and filter and smoother give the paths.
        ``'forgetting'``: no variances are estimated. The state
        covariance is divided by ``lam`` each period and the error
        covariance is ``S_t = kappa S_{t-1} + (1 - kappa) e_t e_t'``.
        Filtered paths only.
    time : str, optional
        Column to sort by; its values label the dates.
    common : bool, default False
        ``'kalman'``: one innovation variance shared by all coefficients
        of an equation instead of one per coefficient. Two parameters per
        equation instead of ``K p + 2``; a common variance only makes
        sense when the regressors are on comparable scales.
    obs_var : array of K, optional
        ``'kalman'``: fix the error variances.
    state_var : float, array of ``K p + 1``, or ``K`` x ``(K p + 1)``
        ``'kalman'``: fix the innovation variances (0 for a constant
        coefficient). With ``obs_var`` too, nothing is estimated.
    lam : float, default 0.99
        ``'forgetting'``: forgetting factor in (0, 1]. Observations
        ``h`` periods back have weight ``lam ** h``; 1 means constant
        coefficients (recursive least squares when ``kappa = 1`` too).
    kappa : float, default 0.96
        ``'forgetting'``: decay of the error covariance in (0, 1]; 1 keeps
        it at ``sigma0``.
    sigma0 : K x K array, optional
        ``'forgetting'``: error covariance at the start. Default: the
        residual covariance (divisor T) of the constant-coefficient VAR on
        the whole sample. That uses later data; its weight in ``S_t`` is
        ``kappa ** t``.
    sigma_update : {'filtered', 'predicted'}
        ``'forgetting'``: ``e_t`` is the residual at the updated
        coefficients, or the one-step prediction error. Under a diffuse
        start prediction errors of the first dates are huge; keep
        ``'filtered'`` there.
    prior : {'diffuse', 'minnesota'}
        Coefficients before the first date. ``'diffuse'``: mean 0,
        variance ``C0``. ``'minnesota'``: mean ``prior_own_lag`` on a
        variable's own first lag and 0 elsewhere, variance
        ``prior_tightness / l ** 2`` at lag ``l``, ``C0`` for the
        intercept. The Minnesota variances are not rescaled by the
        variables' scales: put the variables on comparable scales first.
    m0, C0 : scalar, array of ``K p + 1``, or ``K`` x ``(K p + 1)``
        Prior means (overriding ``prior``) and prior variances. The
        default variance 1e7 is the diffuse value of :func:`statspai.dlm`.
    prior_tightness, prior_own_lag : float
    alpha : float, default 0.05
        Bands have pointwise coverage ``1 - alpha``.

    Returns
    -------
    TVPVARResult

    Raises
    ------
    DataInsufficient
        Fewer than ``K p + 4`` dates after the lags.
    MethodIncompatibility
        Bad arguments, missing values.

    Notes
    -----
    *Kalman.* The equations are estimated one at a time, which is the
    full-system estimator when the coefficient innovations are
    independent across equations and the error covariance does not enter
    the coefficient update. ``sigma``, constant over time, is the
    covariance of the one-step forecast errors ``y_t - B_{t-1} x_t``,
    which is what impulse responses and forecasts need. Its correlations
    are those of the prediction errors standardised by the filter's
    forecast standard deviations, and its diagonal is the median over
    dates of the filter's forecast variance of each equation; the first
    ``K p + 1`` dates (dominated by the diffuse prior) are left out of
    both. The diagonal is a little above the ``V_i`` in
    ``variances['obs']`` because a forecast error also carries the
    uncertainty about the coefficients. For a very persistent variable
    the likelihood can put ``V_i`` at zero and let a random-walk
    intercept explain the series (the equation becomes a local level
    model); a warning says so, and ``sigma`` remains usable. An
    innovation variance estimated at zero means that coefficient is
    constant; a warning lists them. The likelihood in ``K p + 1``
    separate variances is often flat: most are estimated at zero and the
    rest are imprecise, so read the paths, not the variances. Under the
    diffuse prior the likelihood includes the first ``K p + 1`` dates, as
    in ``sp.dlm``.

    *Forgetting.* With ``kappa = 1`` the filtered coefficients at date
    ``t`` are discounted least squares with weights ``lam ** (t - s)``,
    and ordinary least squares when ``lam = 1`` too. With ``kappa < 1``
    they are the generalised least squares estimate with weights
    ``lam ** (t - s) S_{s-1}^{-1}``, which differs from least squares
    equation by equation because the error covariance moves.
    Under the diffuse prior the first ``K p + 1`` dates fit exactly and
    carry no information. The standard errors are the square roots of the
    filter's state covariance and depend on ``lam``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((160, 2))
    >>> for t in range(1, 160):
    ...     a = 0.2 + 0.6 * t / 160
    ...     y[t, 0] = a * y[t - 1, 0] + rng.normal()
    ...     y[t, 1] = 0.3 * y[t - 1, 1] + rng.normal()
    >>> df = pd.DataFrame(y, columns=["x", "z"])
    >>> fit = sp.tvp_var(df, lags=1, common=True)
    >>> fit.coef_smoothed.shape
    (159, 2, 3)
    >>> paths = fit.coefficients()
    >>> roots = fit.stability()
    >>> fc = fit.forecast(4)
    >>> ff = sp.tvp_var(df, lags=1, method="forgetting", lam=0.98)

    References
    ----------
    [@koop2013large],
    [@primiceri2005time],
    [@petris2009dynamic],
    [@lutkepohl2005new]
    """
    method = str(method).lower()
    if method not in ("kalman", "forgetting"):
        raise MethodIncompatibility(
            f"tvp_var: method={method!r} is not 'kalman' or 'forgetting'.",
            recovery_hint="Use method='kalman' or method='forgetting'.",
        )
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("tvp_var: data must be a pandas DataFrame.")
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(f"tvp_var: alpha={alpha} is not in (0, 1).")
    if not isinstance(lags, (int, np.integer)) or isinstance(lags, bool) or lags < 1:
        raise MethodIncompatibility("tvp_var: lags must be a positive integer.")
    work = data
    if time is not None:
        if time not in data.columns:
            raise MethodIncompatibility(f"tvp_var: time column {time!r} not found.")
        work = data.sort_values(time, kind="stable")
    if variables is None:
        names = [
            str(c) for c in work.select_dtypes(include=[np.number]).columns if c != time
        ]
    else:
        names = [str(v) for v in variables]
        missing = [v for v in names if v not in work.columns]
        if missing:
            raise MethodIncompatibility(
                f"tvp_var: columns {missing} are not in data.",
                recovery_hint="Check the names passed to variables=.",
            )
    if not names or len(set(names)) != len(names):
        raise MethodIncompatibility(
            "tvp_var: variables must name distinct numeric columns."
        )
    Yall = work[names].to_numpy(dtype=float)
    if not np.all(np.isfinite(Yall)):
        raise MethodIncompatibility(
            "tvp_var: the variables contain missing or infinite values.",
            recovery_hint="Dropping rows would join dates that are not "
            "adjacent; fill or trim the sample first.",
        )
    K, p = len(names), int(lags)
    k = K * p + 1
    n_eff = Yall.shape[0] - p
    if n_eff < k + 3:
        raise DataInsufficient(
            f"tvp_var: {max(n_eff, 0)} dates after {p} lags for {k} "
            f"time-varying coefficients per equation; at least {k + 3} are "
            "needed.",
            recovery_hint="Use fewer variables or lags, or a longer sample.",
        )
    Y, X = lag_design(Yall, p)
    labels = work[time] if time is not None else work.index
    index = pd.Index(labels[p:])
    terms = [f"L{lag}.{v}" for lag in range(1, p + 1) for v in names] + ["_cons"]
    mean0, var0 = _prior(
        K, p, str(prior).lower(), m0, C0, float(prior_tightness), float(prior_own_lag)
    )
    notes: List[str] = []
    info: Dict[str, Any] = {"prior": str(prior).lower(), "notes": notes}
    state = {"last_obs": Yall[-p:].copy()}

    if method == "forgetting":
        if not 0.0 < lam <= 1.0 or not 0.0 < kappa <= 1.0:
            raise MethodIncompatibility(
                f"tvp_var: lam={lam} and kappa={kappa} must be in (0, 1].",
                recovery_hint="Typical values are lam in [0.95, 1] and "
                "kappa in [0.9, 1].",
            )
        sigma_update = str(sigma_update).lower()
        if sigma_update not in ("filtered", "predicted"):
            raise MethodIncompatibility(
                "tvp_var: sigma_update must be 'filtered' or 'predicted'."
            )
        if obs_var is not None or state_var is not None or common:
            raise MethodIncompatibility(
                "tvp_var: obs_var, state_var and common belong to " "method='kalman'.",
                recovery_hint="method='forgetting' is governed by lam and kappa.",
            )
        if sigma0 is None:
            resid = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
            S0 = resid.T @ resid / n_eff
        else:
            S0 = np.asarray(sigma0, dtype=float)
            if S0.shape != (K, K) or not np.allclose(S0, S0.T):
                raise MethodIncompatibility(
                    f"tvp_var: sigma0 must be a symmetric {K} x {K} matrix."
                )
        if np.min(np.linalg.eigvalsh(S0)) <= 0:
            raise MethodIncompatibility(
                "tvp_var: the starting error covariance is not positive " "definite."
            )
        out = forgetting_filter(
            Y,
            X,
            float(lam),
            float(kappa),
            mean0,
            np.diag(var0.reshape(-1)),
            S0,
            update=sigma_update,
        )
        info.update(
            lam=float(lam),
            kappa=float(kappa),
            sigma_update=sigma_update,
            sigma0=S0,
            pred_error=out["pred_error"],
            pred_cov=out["pred_cov"],
            constant_coefficients=[],
        )
        state["cov_last"] = out["cov_last"]
        return TVPVARResult(
            coef_filtered=out["coef"],
            se_filtered=out["se"],
            coef_smoothed=None,
            se_smoothed=None,
            sigma=pd.DataFrame(out["sigma"][-1], index=names, columns=names),
            sigma_t=out["sigma"],
            variances=None,
            loglik=float(out["loglik"]),
            index=index,
            var_names=names,
            terms=terms,
            lags=p,
            n_obs=int(n_eff),
            method=method,
            alpha=float(alpha),
            model_info=info,
            _state=state,
        )

    # ---------------------------- kalman ------------------------------ #
    V_fix: Optional[np.ndarray] = None
    if obs_var is not None:
        V_fix = np.asarray(obs_var, dtype=float).reshape(-1)
        if V_fix.size == 1:
            V_fix = np.full(K, float(V_fix[0]))
        if V_fix.shape != (K,) or np.any(V_fix <= 0):
            raise MethodIncompatibility(
                f"tvp_var: obs_var must be {K} positive variances."
            )
    W_fix: Optional[np.ndarray] = None
    if state_var is not None:
        sv = np.asarray(state_var, dtype=float)
        if sv.ndim == 0 or sv.shape in ((k,), (K, k)):
            W_fix = np.empty((K, k))
            W_fix[:] = sv
        else:
            raise MethodIncompatibility(
                f"tvp_var: state_var must be a scalar, {k} variances or a "
                f"{K} x {k} array."
            )
        if np.any(W_fix < 0):
            raise MethodIncompatibility("tvp_var: state_var must be non-negative.")
        if common:
            raise MethodIncompatibility(
                "tvp_var: common=True estimates the innovation variance; "
                "do not pass state_var with it."
            )
    Vs = np.empty(K)
    Ws = np.empty((K, k))
    filt = np.empty((n_eff, K, k))
    filt_se = np.empty((n_eff, K, k))
    smo = np.empty((n_eff, K, k))
    smo_se = np.empty((n_eff, K, k))
    pred_err = np.empty((n_eff, K))
    pred_var = np.empty((n_eff, K))
    logliks = np.empty(K)
    fixed = []
    for i in range(K):
        y_i = np.ascontiguousarray(Y[:, i])
        C0_i = np.diag(var0[i])
        v_i = None if V_fix is None else float(V_fix[i])
        if W_fix is not None and v_i is not None:
            Vs[i], Ws[i] = v_i, W_fix[i]
        elif common:
            Vs[i], Ws[i], ok, msg = mle_common(y_i, X, mean0[i], C0_i, v_i)
            if not ok:
                warnings.warn(
                    f"tvp_var: the likelihood search of equation {names[i]} "
                    f"stopped early: {msg}",
                    ConvergenceWarning,
                    stacklevel=2,
                )
        else:
            w_i = None if W_fix is None else W_fix[i]
            Vs[i], Ws[i] = mle_free(y_i, X, mean0[i], C0_i, v_i, w_i)
        eq = kalman_equation(y_i, X, Vs[i], Ws[i], mean0[i], C0_i)
        filt[:, i], filt_se[:, i] = eq["filtered"], np.sqrt(eq["filtered_var"])
        smo[:, i], smo_se[:, i] = eq["smoothed"], np.sqrt(eq["smoothed_var"])
        pred_err[:, i] = y_i - eq["forecast"]
        pred_var[:, i] = eq["forecast_var"]
        logliks[i] = eq["loglik"]
        if W_fix is None:
            scale = float(np.var(y_i)) or 1.0
            fixed += [
                f"{names[i]}: {terms[j]}" for j in range(k) if Ws[i, j] < 1e-10 * scale
            ]
    std = pred_err[k:] / np.sqrt(pred_var[k:])
    corr = np.atleast_2d(np.corrcoef(std, rowvar=False)) if K > 1 else np.ones((1, 1))
    sd = np.sqrt(np.median(pred_var[k:], axis=0))
    sigma = corr * np.outer(sd, sd)
    degenerate = [
        names[i]
        for i in range(K)
        if V_fix is None and Vs[i] < 1e-8 * (float(np.var(Y[:, i])) or 1.0)
    ]
    if degenerate:
        text = (
            "the error variance of "
            + ", ".join(degenerate)
            + " is estimated at zero: the coefficients (typically a "
            "random-walk intercept) absorb all the variation of the "
            "series, and their bands at each date are degenerate"
        )
        notes.append(text)
        warnings.warn(
            "tvp_var: " + text,
            StatsPAIWarning,
            stacklevel=2,
        )
    variances = pd.DataFrame(
        np.column_stack([Vs, Ws]), index=names, columns=["obs"] + terms
    )
    if fixed:
        n_all = K * k
        text = (
            f"{len(fixed)} of {n_all} coefficient innovation variances are "
            "estimated at zero, so these coefficients are constant: " + "; ".join(fixed)
        )
        notes.append(text)
        warnings.warn(
            "tvp_var: " + text,
            StatsPAIWarning,
            stacklevel=2,
        )
    info.update(
        common=bool(common),
        equation_loglik=dict(zip(names, (float(v) for v in logliks))),
        constant_coefficients=fixed,
        sigma_from="one-step prediction errors",
        pred_error=pred_err,
        pred_var=pred_var,
        degenerate_equations=degenerate,
    )
    return TVPVARResult(
        coef_filtered=filt,
        se_filtered=filt_se,
        coef_smoothed=smo,
        se_smoothed=smo_se,
        sigma=pd.DataFrame(sigma, index=names, columns=names),
        sigma_t=None,
        variances=variances,
        loglik=float(logliks.sum()),
        index=index,
        var_names=names,
        terms=terms,
        lags=p,
        n_obs=int(n_eff),
        method=method,
        alpha=float(alpha),
        model_info=info,
        _state=state,
    )
