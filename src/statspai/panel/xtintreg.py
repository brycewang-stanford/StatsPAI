"""Random-effects interval regression and tobit (Stata ``xtintreg`` /
``xttobit``).

``y*_it = x_it'b + nu_i + e_it`` with ``nu_i ~ N(0, s_u^2)`` and
``e_it ~ N(0, s_e^2)``; ``y*`` is observed exactly, censored on one side, or
known to lie in an interval (the ``intreg`` coding). The panel likelihood

    l_i = int phi(nu; 0, s_u) prod_t F(y_it | x_it'b + nu, s_e) d nu

is integrated by adaptive Gauss-Hermite quadrature centred at the mode of
each panel's integrand with its curvature as scale (Stata's default
``intmethod(mvaghermite)`` adapts to the posterior mean and standard
deviation instead; both converge to the same integral). Parameters are
``(b, ln s_u, ln s_e)``; the fit is BFGS followed by Newton on complex-step
scores, the shared ML path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility
from ..regression._optim_helpers import (
    complex_step_scores,
    inverse_information,
    ml_newton_polish,
    robust_convergence,
    se_from_vcov,
)
from ..regression.intreg import _log_prob_between

__all__ = ["xtintreg", "xttobit"]

_HALF_LOG_2PI = 0.5 * np.log(2 * np.pi)


class _Panel:
    """Row layout of an interval-coded panel."""

    def __init__(
        self, lo: np.ndarray, hi: np.ndarray, X: np.ndarray, codes: np.ndarray
    ) -> None:
        self.X, self.codes = X, codes
        self.n = int(codes.max()) + 1
        self.point = lo == hi
        self.left = np.isnan(lo) & ~np.isnan(hi)
        self.right = ~np.isnan(lo) & np.isnan(hi)
        self.inter = ~self.point & ~self.left & ~self.right
        self.lo = np.where(np.isnan(lo), 0.0, lo)
        self.hi = np.where(np.isnan(hi), 0.0, hi)

    def row_loglik(self, idx: Any, ln_se: Any) -> Any:
        """``ln F(y_it | idx_it, s_e)`` for an array of indices (any shape
        whose first axis is the rows)."""
        s = np.exp(ln_se)
        shape = (-1,) + (1,) * (np.ndim(idx) - 1)
        lo, hi = self.lo.reshape(shape), self.hi.reshape(shape)
        out = np.zeros(np.shape(idx), dtype=np.result_type(idx, ln_se, float))
        p, lft, rgt, itv = self.point, self.left, self.right, self.inter
        if p.any():
            r = (lo[p] - idx[p]) / s
            out[p] = -_HALF_LOG_2PI - ln_se - 0.5 * r * r
        if lft.any():
            out[lft] = special.log_ndtr((hi[lft] - idx[lft]) / s)
        if rgt.any():
            out[rgt] = special.log_ndtr((idx[rgt] - lo[rgt]) / s)
        if itv.any():
            out[itv] = _log_prob_between(
                (lo[itv] - idx[itv]) / s, (hi[itv] - idx[itv]) / s
            )
        return out

    def panel_sum(self, v: Any) -> Any:
        out = np.zeros((self.n,) + np.shape(v)[1:], dtype=np.result_type(v, float))
        np.add.at(out, self.codes, v)
        return out


def _adapt(panel: _Panel, theta: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Mode and curvature scale of each panel's integrand in ``nu``."""
    xb = panel.X @ theta[:k]
    s_u = np.exp(theta[k])
    ln_se = theta[k + 1]
    mu = np.zeros(panel.n)
    h = 1e-4
    for _ in range(50):
        nu = mu[panel.codes]
        f0 = panel.panel_sum(panel.row_loglik(xb + nu, ln_se))
        fp = panel.panel_sum(panel.row_loglik(xb + nu + h, ln_se))
        fm = panel.panel_sum(panel.row_loglik(xb + nu - h, ln_se))
        g = (fp - fm) / (2 * h) - mu / s_u**2
        H = (fp - 2 * f0 + fm) / (h * h) - 1.0 / s_u**2
        H = np.minimum(H, -1e-8)
        step = g / H
        mu = mu - step
        if np.max(np.abs(step)) < 1e-10:
            break
    nu = mu[panel.codes]
    f0 = panel.panel_sum(panel.row_loglik(xb + nu, ln_se))
    fp = panel.panel_sum(panel.row_loglik(xb + nu + h, ln_se))
    fm = panel.panel_sum(panel.row_loglik(xb + nu - h, ln_se))
    H = np.minimum((fp - 2 * f0 + fm) / (h * h) - 1.0 / s_u**2, -1e-8)
    return mu, 1.0 / np.sqrt(-H)


def _fit(panel: _Panel, k: int, theta0: np.ndarray, intpoints: int) -> Dict[str, Any]:
    nodes, weights = np.polynomial.hermite.hermgauss(intpoints)
    log_w = np.log(weights) + nodes**2  # e^{a^2} w
    state: Dict[str, Any] = {}

    def refresh(theta: np.ndarray) -> None:
        state["mu"], state["sig"] = _adapt(panel, np.real(theta), k)

    def obs_panel_loglik(theta: Any) -> Any:
        mu, sig = state["mu"], state["sig"]
        xb = panel.X @ theta[:k]
        ln_su, ln_se = theta[k], theta[k + 1]
        s_u = np.exp(ln_su)
        nu = np.sqrt(2.0) * sig[:, None] * nodes[None, :] + mu[:, None]  # (n, M)
        rows = panel.row_loglik(xb[:, None] + nu[panel.codes], ln_se)  # (N, M)
        g = panel.panel_sum(rows)  # (n, M)
        prior = -_HALF_LOG_2PI - ln_su - 0.5 * (nu / s_u) ** 2
        terms = g + prior + log_w[None, :]
        top = np.max(np.real(terms), axis=1, keepdims=True)
        return (
            np.log(np.sqrt(2.0) * sig)
            + top[:, 0]
            + np.log(np.sum(np.exp(terms - top), axis=1))
        )

    def neg_total(theta: np.ndarray) -> float:
        refresh(theta)
        with np.errstate(all="ignore"):
            v = -float(np.sum(obs_panel_loglik(theta)))
        return v if np.isfinite(v) else 1e300

    def neg_grad(theta: np.ndarray) -> np.ndarray:
        refresh(theta)
        grad: np.ndarray = -complex_step_scores(obs_panel_loglik, theta).sum(axis=0)
        return grad

    opt = optimize.minimize(
        neg_total, theta0, jac=neg_grad, method="BFGS", options={"maxiter": 3000}
    )
    converged, _ = robust_convergence(opt)
    theta = np.asarray(opt.x, dtype=float)
    # Newton with the quadrature re-centred at each step
    for _ in range(3):
        refresh(theta)
        theta, scores, H, _ = ml_newton_polish(obs_panel_loglik, theta, maxiter=10)
    refresh(theta)
    theta, scores, H, _ = ml_newton_polish(obs_panel_loglik, theta, maxiter=10)
    grad_norm = float(np.max(np.abs(scores.sum(axis=0))))
    return {
        "theta": theta,
        "scores": scores,
        "H": H,
        "loglik": float(np.sum(np.real(obs_panel_loglik(theta)))),
        "converged": bool(converged or grad_norm < 1e-6),
        "gradient_norm": grad_norm,
    }


@records_call("xtintreg")
@accepts_aliases(entity="id", covariates="x")
def xtintreg(
    data: pd.DataFrame,
    y_lower: str,
    y_upper: str,
    x: Union[str, Sequence[str]],
    id: str,
    intpoints: int = 30,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Random-effects interval regression by maximum likelihood.

    Equivalent to Stata's ``xtintreg lo hi x``: the ``intreg`` coding of the
    outcome (equal bounds exact, a missing bound open) with a normal panel
    effect. ``sp.xttobit`` is the censored-at-limits special case.

    Parameters
    ----------
    data : pd.DataFrame
    y_lower, y_upper : str
        Bound columns (``NaN`` = unbounded on that side).
    x : str or list of str
        Regressors (a constant is added).
    id : str
        Panel identifier.
    intpoints : int, default 30
        Gauss-Hermite nodes per panel (adaptive). Stata's default is 12;
        more nodes only bring the integral closer to its exact value.
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params``: the coefficients, ``sigma_u`` and ``sigma_e``
        (delta-method standard errors from ``ln sigma``); ``model_info``:
        ``rho = s_u^2 / (s_u^2 + s_e^2)``, the log-likelihood, the
        likelihood-ratio test of ``s_u = 0`` against the pooled model and
        the counts of each kind of observation.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> ids = np.repeat(np.arange(100), 5)
    >>> df = pd.DataFrame({"id": ids, "x": rng.normal(size=500)})
    >>> y = 1 + df["x"] + rng.normal(size=100)[ids] + rng.normal(size=500)
    >>> df["lo"], df["hi"] = np.floor(y), np.floor(y) + 1
    >>> fit = sp.xtintreg(df, y_lower="lo", y_upper="hi", x="x", id="id")
    >>> round(float(fit.params["x"]), 1)  # doctest: +SKIP
    1.0
    """
    xs = [x] if isinstance(x, str) else list(x)
    check_columns(
        data,
        {"y_lower": y_lower, "y_upper": y_upper, "x": xs, "id": id},
        function="xtintreg",
    )
    if int(intpoints) < 2:
        raise MethodIncompatibility("xtintreg: intpoints must be at least 2.")
    df = data.loc[data[xs + [id]].notna().all(axis=1)]
    lo = df[y_lower].to_numpy(dtype=float)
    hi = df[y_upper].to_numpy(dtype=float)
    keep = ~(np.isnan(lo) & np.isnan(hi))
    df, lo, hi = df.loc[keep], lo[keep], hi[keep]
    bad = ~np.isnan(lo) & ~np.isnan(hi) & (lo > hi)
    if bad.any():
        raise MethodIncompatibility(
            f"xtintreg: {int(bad.sum())} row(s) have {y_lower} > {y_upper}.",
            diagnostics={"n_inverted": int(bad.sum())},
        )
    codes, uniq = pd.factorize(df[id], sort=True)
    if len(uniq) < 2:
        raise DataInsufficient("xtintreg: needs at least two panels.")
    N = len(df)
    X = np.column_stack([df[v].to_numpy(dtype=float) for v in xs] + [np.ones(N)])
    k = X.shape[1]
    names = xs + ["_cons"]
    panel = _Panel(lo, hi, X, codes)

    # pooled intreg as the start and the rho = 0 likelihood
    from ..regression.intreg import intreg

    pooled_df = pd.DataFrame(X[:, :-1], columns=[f"_x{j}" for j in range(k - 1)])
    pooled_df["_lo"], pooled_df["_hi"] = lo, hi
    pooled = intreg(
        pooled_df, y_lower="_lo", y_upper="_hi", x=[f"_x{j}" for j in range(k - 1)]
    )
    pth = pooled._ml_state["theta"]  # const first, then x, ln s
    b0 = np.r_[pth[1:k], pth[0]]
    s0 = float(np.exp(pth[k]))
    theta0 = np.r_[b0, np.log(s0 / np.sqrt(2)), np.log(s0 / np.sqrt(2))]
    fit = _fit(panel, k, theta0, int(intpoints))
    theta = fit["theta"]
    V = inverse_information(fit["H"])
    se = se_from_vcov(V)
    s_u, s_e = float(np.exp(theta[k])), float(np.exp(theta[k + 1]))
    params = np.r_[theta[:k], s_u, s_e]
    ses = np.r_[se[:k], se[k] * s_u, se[k + 1] * s_e]
    # covariance of (b, s_u, s_e) by the delta method
    J = np.eye(k + 2)
    J[k, k], J[k + 1, k + 1] = s_u, s_e
    V_out = J @ V @ J.T
    ll_pooled = float(pooled.model_info["log_likelihood"])
    lr = max(2.0 * (fit["loglik"] - ll_pooled), 0.0)
    info: Dict[str, Any] = {
        "model_type": "Random-effects interval regression",
        "rho": s_u**2 / (s_u**2 + s_e**2),
        "log_likelihood": fit["loglik"],
        "lr_sigma_u0": lr,
        # the boundary test: half chi2(0) + half chi2(1)
        "lr_sigma_u0_pvalue": float(0.5 * stats.chi2.sf(lr, 1)),
        "n_groups": len(uniq),
        "n_uncensored": int(panel.point.sum()),
        "n_left_censored": int(panel.left.sum()),
        "n_right_censored": int(panel.right.sum()),
        "n_interval": int(panel.inter.sum()),
        "intpoints": int(intpoints),
        "converged": fit["converged"],
        "gradient_norm": fit["gradient_norm"],
        "ln_sigma_u": float(theta[k]),
        "ln_sigma_e": float(theta[k + 1]),
        "id": id,
        "alpha": alpha,
    }
    out_names = names + ["sigma_u", "sigma_e"]
    return EconometricResults(
        params=pd.Series(params, index=out_names),
        std_errors=pd.Series(ses, index=out_names),
        model_info=info,
        data_info={
            "nobs": N,
            "n_obs": N,
            "dependent_var": f"{y_lower}/{y_upper}",
            "var_cov": V_out,
            "var_names": out_names,
            "df_resid": N - k - 2,
            "inference": "z",
        },
    )


@records_call("xttobit")
@accepts_aliases(entity="id", covariates="x")
def xttobit(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str]],
    id: str,
    ll: Optional[float] = None,
    ul: Optional[float] = None,
    intpoints: int = 30,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Random-effects tobit by maximum likelihood.

    Equivalent to Stata's ``xttobit y x, ll(#) ul(#)``: observations at or
    below ``ll`` are left-censored, at or above ``ul`` right-censored, with a
    normal panel effect. It is :func:`xtintreg` with that coding.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Censored outcome.
    x : str or list of str
        Regressors (a constant is added).
    id : str
        Panel identifier.
    ll, ul : float, optional
        Censoring limits; at least one is needed.
    intpoints : int, default 30
        Adaptive Gauss-Hermite nodes per panel (Stata's default is 12).
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        As :func:`xtintreg`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> ids = np.repeat(np.arange(100), 5)
    >>> df = pd.DataFrame({"id": ids, "x": rng.normal(size=500)})
    >>> ystar = df["x"] + rng.normal(size=100)[ids] + rng.normal(size=500)
    >>> df["y"] = np.maximum(ystar, 0)
    >>> fit = sp.xttobit(df, y="y", x="x", id="id", ll=0)
    >>> round(float(fit.params["x"]), 1)  # doctest: +SKIP
    1.0
    """
    if ll is None and ul is None:
        raise MethodIncompatibility(
            "xttobit: give a censoring limit, ll= and/or ul=.",
            recovery_hint="Without limits the model is a random-effects "
            "linear regression: use sp.panel(method='re') or sp.mixed.",
        )
    check_columns(data, {"y": y}, function="xttobit")
    yv = data[y].to_numpy(dtype=float)
    lo, hi = yv.copy(), yv.copy()
    if ll is not None:
        cens = yv <= ll
        lo = np.where(cens, np.nan, lo)
        hi = np.where(cens, float(ll), hi)
    if ul is not None:
        cens = yv >= ul
        lo = np.where(cens, float(ul), lo)
        hi = np.where(cens, np.nan, hi)
    lo = np.where(np.isnan(yv), np.nan, lo)
    hi = np.where(np.isnan(yv), np.nan, hi)
    tmp = data.assign(_xt_lo=lo, _xt_hi=hi)
    res = xtintreg(
        tmp,
        y_lower="_xt_lo",
        y_upper="_xt_hi",
        x=x,
        id=id,
        intpoints=intpoints,
        alpha=alpha,
    )
    res.model_info["model_type"] = "Random-effects tobit"
    res.model_info["lower_limit"] = ll
    res.model_info["upper_limit"] = ul
    res.data_info["dependent_var"] = y
    return res
