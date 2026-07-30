"""Repeated cross-section DiD primitives (Sant'Anna & Zhao 2020).

The panel path in :mod:`statspai.did.callaway_santanna` builds ATT(g, t) from
within-unit differences ``Y_t - Y_base``.  Repeated cross-sections have no
unit pairing, so they need a different estimator entirely: one that pools the
two periods and identifies the ATT from period-specific outcome regressions
plus a propensity score.

This module ports the three estimators R ``did::att_gt(panel = FALSE)``
dispatches to, keeping their influence functions so downstream aggregation
(``sp.aggte``) and the multiplier bootstrap work unchanged:

============================  ==============================
``est_method`` in R ``did``   function here
============================  ==============================
``"dr"``                      :func:`drdid_rc`
``"ipw"``                     :func:`std_ipw_did_rc`
``"reg"``                     :func:`reg_did_rc`
============================  ==============================

Every convention that moves the answer is reproduced from the reference
implementation rather than re-derived:

1. The propensity score is a logit of ``D`` on the covariates over the
   **pooled** sample, with fitted values capped at ``1 - 1e-6``.
2. Control units are trimmed at ``ps < trim_level`` (default 0.995); treated
   units are never trimmed (their bound is ``ps < 1.01``).
3. Four **separate** outcome regressions are fit — control/pre, control/post,
   treated/pre, treated/post — each on its own cell and then predicted for
   every observation.
4. The influence function carries the estimation error of the propensity
   score *and* of all four outcome regressions; dropping any of those terms
   changes the standard error materially.

Verified against R ``DRDID`` 1.2.3: ``drdid_rc`` reproduces both the ATT and
the analytic standard error to 10 decimal places on the module's parity
fixture.

References
----------
Sant'Anna, P.H.C. and Zhao, J. (2020). "Doubly robust
difference-in-differences estimators." *Journal of Econometrics*, 219(1),
101-122. [@santanna2020doubly]
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["RCSResult", "drdid_rc", "reg_did_rc", "std_ipw_did_rc"]

#: Control units with a propensity score above this are dropped.  Matches
#: ``DRDID``'s ``trim.level`` default.
_TRIM_LEVEL = 0.995


class RCSResult(NamedTuple):
    """ATT, analytic SE, and the per-observation influence function."""

    att: float
    se: float
    influence: np.ndarray


def _prepare(
    y: np.ndarray,
    post: np.ndarray,
    d: np.ndarray,
    covariates: Optional[np.ndarray],
    weights: Optional[np.ndarray],
) -> tuple:
    y = np.asarray(y, dtype=float).ravel()
    post = np.asarray(post, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    n = y.size

    if not (post.size == d.size == n):
        raise MethodIncompatibility(
            "y, post and D must have the same length.",
            recovery_hint="Check the (g, t) cell construction.",
            diagnostics={"n_y": n, "n_post": post.size, "n_d": d.size},
        )

    if covariates is None:
        x = np.ones((n, 1))
    else:
        x = np.asarray(covariates, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if not np.allclose(x[:, 0], 1.0):
            x = np.column_stack([np.ones(n), x])

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    if np.min(w) < 0:
        raise MethodIncompatibility(
            "weights must be non-negative.",
            recovery_hint="Drop or repair the negative weights.",
            diagnostics={"min_weight": float(np.min(w))},
        )
    w = w / w.mean()

    # All four cells must be populated or the outcome regressions are not
    # identified — fail loudly rather than return a silently degenerate ATT.
    cells = {
        "control/pre": (d == 0) & (post == 0),
        "control/post": (d == 0) & (post == 1),
        "treated/pre": (d == 1) & (post == 0),
        "treated/post": (d == 1) & (post == 1),
    }
    empty = [name for name, mask in cells.items() if int(mask.sum()) < x.shape[1]]
    if empty:
        raise DataInsufficient(
            "repeated cross-section DiD needs all four "
            f"treatment x period cells populated; too few observations in: "
            f"{', '.join(empty)}.",
            recovery_hint="Widen the comparison window, drop covariates, or "
            "use a different control group.",
            diagnostics={name: int(mask.sum()) for name, mask in cells.items()},
        )

    return y, post, d, x, w, n


def _pscore(d: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted logit propensity score, capped like ``DRDID``."""
    import statsmodels.api as sm

    fit = sm.GLM(d, x, family=sm.families.Binomial(), freq_weights=w).fit()
    return np.minimum(np.asarray(fit.fittedvalues, dtype=float), 1 - 1e-6)


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Weighted OLS on a cell, returned as coefficients."""
    sw = np.sqrt(w[mask])
    coef, *_ = np.linalg.lstsq(x[mask] * sw[:, None], y[mask] * sw, rcond=None)
    return coef


def _asy_lin_ols(
    x: np.ndarray, weights: np.ndarray, resid: np.ndarray, n: int
) -> np.ndarray:
    """Asymptotic linear representation of a weighted OLS coefficient."""
    wx = weights[:, None] * x
    xpx = wx.T @ x / n
    if np.linalg.cond(xpx) > 1 / np.finfo(float).eps:
        raise DataInsufficient(
            "outcome-regression design matrix is singular in a "
            "treatment x period cell.",
            recovery_hint="Drop collinear covariates.",
            diagnostics={"cond": float(np.linalg.cond(xpx))},
        )
    return ((weights * resid)[:, None] * x) @ np.linalg.inv(xpx)


def drdid_rc(
    y: np.ndarray,
    post: np.ndarray,
    d: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    trim_level: float = _TRIM_LEVEL,
) -> RCSResult:
    """Locally efficient DR DiD for repeated cross-sections.

    Port of ``DRDID::drdid_rc`` — the estimator R ``did::att_gt`` uses for
    ``panel = FALSE, est_method = "dr"``.  Verified to 10 decimal places on
    both the ATT and the analytic SE.
    """
    y, post, d, x, w, n = _prepare(y, post, d, covariates, weights)

    ps = _pscore(d, x, w)
    trim = np.where(d == 0, ps < trim_level, ps < 1.01).astype(float)

    b_c_pre = _wls(x, y, w, (d == 0) & (post == 0))
    b_c_post = _wls(x, y, w, (d == 0) & (post == 1))
    b_t_pre = _wls(x, y, w, (d == 1) & (post == 0))
    b_t_post = _wls(x, y, w, (d == 1) & (post == 1))

    oc_pre, oc_post = x @ b_c_pre, x @ b_c_post
    ot_pre, ot_post = x @ b_t_pre, x @ b_t_post
    oc = post * oc_post + (1 - post) * oc_pre

    w_tr_pre = trim * w * d * (1 - post)
    w_tr_post = trim * w * d * post
    w_co_pre = trim * w * ps * (1 - d) * (1 - post) / (1 - ps)
    w_co_post = trim * w * ps * (1 - d) * post / (1 - ps)
    w_d = trim * w * d
    w_dt1 = trim * w * d * post
    w_dt0 = trim * w * d * (1 - post)

    mean = np.mean
    eta_tr_pre = w_tr_pre * (y - oc) / mean(w_tr_pre)
    eta_tr_post = w_tr_post * (y - oc) / mean(w_tr_post)
    eta_co_pre = w_co_pre * (y - oc) / mean(w_co_pre)
    eta_co_post = w_co_post * (y - oc) / mean(w_co_post)
    eta_d_post = w_d * (ot_post - oc_post) / mean(w_d)
    eta_dt1_post = w_dt1 * (ot_post - oc_post) / mean(w_dt1)
    eta_d_pre = w_d * (ot_pre - oc_pre) / mean(w_d)
    eta_dt0_pre = w_dt0 * (ot_pre - oc_pre) / mean(w_dt0)

    a_tr_pre, a_tr_post = mean(eta_tr_pre), mean(eta_tr_post)
    a_co_pre, a_co_post = mean(eta_co_pre), mean(eta_co_post)
    a_d_post, a_dt1_post = mean(eta_d_post), mean(eta_dt1_post)
    a_d_pre, a_dt0_pre = mean(eta_d_pre), mean(eta_dt0_pre)

    att = float(
        (a_tr_post - a_tr_pre)
        - (a_co_post - a_co_pre)
        + (a_d_post - a_dt1_post)
        - (a_d_pre - a_dt0_pre)
    )

    # --- influence function -------------------------------------------------
    alr_c_pre = _asy_lin_ols(x, w * (1 - d) * (1 - post), y - oc_pre, n)
    alr_c_post = _asy_lin_ols(x, w * (1 - d) * post, y - oc_post, n)
    alr_t_pre = _asy_lin_ols(x, w * d * (1 - post), y - ot_pre, n)
    alr_t_post = _asy_lin_ols(x, w * d * post, y - ot_post, n)

    hess = np.linalg.inv(x.T @ ((ps * (1 - ps) * w)[:, None] * x)) * n
    alr_ps = ((w * (d - ps))[:, None] * x) @ hess

    inf_tr_pre = eta_tr_pre - w_tr_pre * a_tr_pre / mean(w_tr_pre)
    inf_tr_post = eta_tr_post - w_tr_post * a_tr_post / mean(w_tr_post)
    m1_post = -np.mean((w_tr_post * post)[:, None] * x, axis=0) / mean(w_tr_post)
    m1_pre = -np.mean((w_tr_pre * (1 - post))[:, None] * x, axis=0) / mean(w_tr_pre)
    inf_tr_or = alr_c_post @ m1_post + alr_c_pre @ m1_pre

    inf_co_pre = eta_co_pre - w_co_pre * a_co_pre / mean(w_co_pre)
    inf_co_post = eta_co_post - w_co_post * a_co_post / mean(w_co_post)
    m2_pre = np.mean((w_co_pre * (y - oc - a_co_pre))[:, None] * x, axis=0) / mean(
        w_co_pre
    )
    m2_post = np.mean((w_co_post * (y - oc - a_co_post))[:, None] * x, axis=0) / mean(
        w_co_post
    )
    inf_co_ps = alr_ps @ (m2_post - m2_pre)
    m3_post = -np.mean((w_co_post * post)[:, None] * x, axis=0) / mean(w_co_post)
    m3_pre = -np.mean((w_co_pre * (1 - post))[:, None] * x, axis=0) / mean(w_co_pre)
    inf_co_or = alr_c_post @ m3_post + alr_c_pre @ m3_pre

    inf_eff = (
        (eta_d_post - w_d * a_d_post / mean(w_d))
        - (eta_dt1_post - w_dt1 * a_dt1_post / mean(w_dt1))
    ) - (
        (eta_d_pre - w_d * a_d_pre / mean(w_d))
        - (eta_dt0_pre - w_dt0 * a_dt0_pre / mean(w_dt0))
    )

    mom_post = np.mean((w_d / mean(w_d) - w_dt1 / mean(w_dt1))[:, None] * x, axis=0)
    mom_pre = np.mean((w_d / mean(w_d) - w_dt0 / mean(w_dt0))[:, None] * x, axis=0)
    inf_or = (alr_t_post - alr_c_post) @ mom_post - (alr_t_pre - alr_c_pre) @ mom_pre

    inf_treat = inf_tr_post - inf_tr_pre + inf_tr_or
    inf_cont = inf_co_post - inf_co_pre + inf_co_ps + inf_co_or
    influence = (inf_treat - inf_cont) + inf_eff + inf_or

    se = float(np.std(influence, ddof=1) * np.sqrt(n - 1) / n)
    return RCSResult(att=att, se=se, influence=np.asarray(influence, dtype=float))


def reg_did_rc(
    y: np.ndarray,
    post: np.ndarray,
    d: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> RCSResult:
    """Outcome-regression DiD for repeated cross-sections.

    Port of ``DRDID::reg_did_rc`` (R ``did``'s ``est_method = "reg"``).
    """
    y, post, d, x, w, n = _prepare(y, post, d, covariates, weights)

    b_c_pre = _wls(x, y, w, (d == 0) & (post == 0))
    b_c_post = _wls(x, y, w, (d == 0) & (post == 1))
    oc_pre, oc_post = x @ b_c_pre, x @ b_c_post

    w_treat = w * d
    w_tr_pre = w * d * (1 - post)
    w_tr_post = w * d * post
    mean = np.mean

    eta_treat = w_treat * (oc_post - oc_pre) / mean(w_treat)
    eta_tr_pre = w_tr_pre * y / mean(w_tr_pre)
    eta_tr_post = w_tr_post * y / mean(w_tr_post)

    a_treat = mean(eta_treat)
    a_tr_pre, a_tr_post = mean(eta_tr_pre), mean(eta_tr_post)
    att = float((a_tr_post - a_tr_pre) - a_treat)

    alr_c_pre = _asy_lin_ols(x, w * (1 - d) * (1 - post), y - oc_pre, n)
    alr_c_post = _asy_lin_ols(x, w * (1 - d) * post, y - oc_post, n)

    inf_tr_pre = eta_tr_pre - w_tr_pre * a_tr_pre / mean(w_tr_pre)
    inf_tr_post = eta_tr_post - w_tr_post * a_tr_post / mean(w_tr_post)
    inf_treat_obs = eta_treat - w_treat * a_treat / mean(w_treat)

    m_post = np.mean(w_treat[:, None] * x, axis=0) / mean(w_treat)
    m_pre = np.mean(w_treat[:, None] * x, axis=0) / mean(w_treat)
    inf_or = alr_c_post @ m_post - alr_c_pre @ m_pre

    influence = (inf_tr_post - inf_tr_pre) - (inf_treat_obs + inf_or)
    se = float(np.std(influence, ddof=1) * np.sqrt(n - 1) / n)
    return RCSResult(att=att, se=se, influence=np.asarray(influence, dtype=float))


def std_ipw_did_rc(
    y: np.ndarray,
    post: np.ndarray,
    d: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    trim_level: float = _TRIM_LEVEL,
) -> RCSResult:
    """Standardised IPW DiD for repeated cross-sections.

    Port of ``DRDID::std_ipw_did_rc`` (R ``did``'s ``est_method = "ipw"``).
    """
    y, post, d, x, w, n = _prepare(y, post, d, covariates, weights)

    ps = _pscore(d, x, w)
    trim = np.where(d == 0, ps < trim_level, ps < 1.01).astype(float)

    w_tr_pre = trim * w * d * (1 - post)
    w_tr_post = trim * w * d * post
    w_co_pre = trim * w * ps * (1 - d) * (1 - post) / (1 - ps)
    w_co_post = trim * w * ps * (1 - d) * post / (1 - ps)
    mean = np.mean

    eta_tr_pre = w_tr_pre * y / mean(w_tr_pre)
    eta_tr_post = w_tr_post * y / mean(w_tr_post)
    eta_co_pre = w_co_pre * y / mean(w_co_pre)
    eta_co_post = w_co_post * y / mean(w_co_post)

    a_tr_pre, a_tr_post = mean(eta_tr_pre), mean(eta_tr_post)
    a_co_pre, a_co_post = mean(eta_co_pre), mean(eta_co_post)
    att = float((a_tr_post - a_tr_pre) - (a_co_post - a_co_pre))

    hess = np.linalg.inv(x.T @ ((ps * (1 - ps) * w)[:, None] * x)) * n
    alr_ps = ((w * (d - ps))[:, None] * x) @ hess

    inf_tr_pre = eta_tr_pre - w_tr_pre * a_tr_pre / mean(w_tr_pre)
    inf_tr_post = eta_tr_post - w_tr_post * a_tr_post / mean(w_tr_post)
    inf_co_pre = eta_co_pre - w_co_pre * a_co_pre / mean(w_co_pre)
    inf_co_post = eta_co_post - w_co_post * a_co_post / mean(w_co_post)

    m_pre = np.mean((w_co_pre * (y - a_co_pre))[:, None] * x, axis=0) / mean(w_co_pre)
    m_post = np.mean((w_co_post * (y - a_co_post))[:, None] * x, axis=0) / mean(
        w_co_post
    )
    inf_co_ps = alr_ps @ (m_post - m_pre)

    influence = (inf_tr_post - inf_tr_pre) - (inf_co_post - inf_co_pre + inf_co_ps)
    se = float(np.std(influence, ddof=1) * np.sqrt(n - 1) / n)
    return RCSResult(att=att, se=se, influence=np.asarray(influence, dtype=float))
