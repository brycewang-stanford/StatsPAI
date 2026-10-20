"""
RD inference with a discrete running variable (Kolesar & Rothe 2018).

``sp.rd_discrete`` offers the two honest confidence intervals of
``kolesar2018inference``:

1. ``method='bsd'`` -- bounded second derivative. The Armstrong-Kolesar
   honest local linear interval (the engine of :func:`sp.rd_honest`). Its
   construction makes no assumption on the support of the running variable;
   with mass points the nearest-neighbour variance keeps tied units together.
2. ``method='bme'`` (alias ``'bm'``) -- bounded misspecification error
   (section 5.2 of the paper): a polynomial of order ``order`` on each side
   inside ``[c - h, c + h]``, with the interval taken over the worst
   combination of specification errors at one support point on each side,
   estimated by the deviations of the cell means from the fitted polynomial.

Clean-room implementation from the paper and the documentation (help pages
and vignette) of R ``RDHonest`` (GPL-3), whose outputs were used only as a
black box acceptance target. Specification:
``docs/dev/cleanroom/rd_honest.md``.

.. versionchanged:: 1.29.0
   ⚠️ Rebuilt to reproduce ``RDHonest``. Through 1.28.0 the function fitted
   a local linear regression on bin means whose variance omitted a factor of
   the bin size, used an ad-hoc second-difference ``M``, and a
   ``K * sum|w|`` bias bound that is not the Kolesar-Rothe BME interval.
   See MIGRATION.md.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._rdhonest import honest_rd

__all__ = ["rd_discrete"]

_METHODS = {"bsd": "bsd", "bme": "bme", "bm": "bme"}


def rd_discrete(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0.0,
    M: Optional[float] = None,
    K: Optional[float] = None,
    method: str = "bsd",
    h: Optional[float] = None,
    alpha: float = 0.05,
    *,
    order: int = 0,
    kernel: str = "triangular",
    opt_criterion: str = "MSE",
) -> CausalResult:
    """Honest CI for RD with a discrete running variable (Kolesar-Rothe 2018).

    Parameters
    ----------
    data : pandas.DataFrame
    y, x : str
        Outcome and (discrete) running variable. Units with ``x >= c`` are
        treated.
    c : float, default 0
        RD cutoff.
    M : float, optional
        ``'bsd'``: bound on the second derivative of the conditional mean on
        either side. If omitted, the rule of thumb of Armstrong and Kolesar
        (global quartic on each side) is used.
    K : float, optional
        Kolesar and Rothe's name for the same bound; used when ``M`` is not
        given.
    method : {'bsd', 'bme', 'bm'}, default 'bsd'
        Smoothness class: bounded second derivative, or bounded
        misspecification error (``'bm'`` is an alias of ``'bme'``).
    h : float, optional
        Bandwidth. ``'bsd'``: chosen by ``opt_criterion`` when omitted.
        ``'bme'``: the window ``|x - c| <= h``; all data when omitted.
    alpha : float, default 0.05
        One minus the confidence level.
    order : int, default 0
        ``'bme'``: order of the polynomial fitted on each side (0 compares
        side means).
    kernel : {'triangular', 'uniform', 'epanechnikov'}
        Kernel of the ``'bsd'`` local linear fit.
    opt_criterion : {'MSE', 'FLCI', 'OCI'}
        ``'bsd'`` bandwidth criterion when ``h`` is omitted.

    Returns
    -------
    CausalResult
        ``model_info['discrete']`` holds the method's ingredients: for
        ``'bsd'`` the bound ``M``, bandwidth, effective observations and
        worst-case bias; for ``'bme'`` the one-sided limits, leverage, the
        number of observations in the window and the bias bound at the
        binding support points. The headline p-value tests ``tau = 0``
        against the honest interval (``P(|N(b, 1)| > |t|)`` with ``b`` the
        bias bound over the standard error).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> age = rng.integers(8, 29, 1500)
    >>> yy = 0.05 * age + 0.8 * (age >= 18) + rng.normal(0, 1, 1500)
    >>> df = pd.DataFrame({"y": yy, "age": age})
    >>> r = sp.rd_discrete(df, y="y", x="age", c=18, method="bme", h=4)

    References
    ----------
    kolesar2018inference, armstrong2020simple
    """
    meth = _METHODS.get(str(method).lower())
    if meth is None:
        raise MethodIncompatibility(
            f"method must be 'bsd' or 'bme' (alias 'bm'), got {method!r}",
            recovery_hint="Use method='bsd' (bounded second derivative) or "
            "method='bme' (bounded misspecification error).",
        )
    check_columns(data, {"y": y, "x": x}, function="sp.rd_discrete")
    df = data[[y, x]].dropna()
    xv = df[x].to_numpy(dtype=float)
    yv = df[y].to_numpy(dtype=float)
    n_obs = int(xv.size)
    n_support = int(np.unique(xv).size)
    for side, lab in ((xv >= c, "above"), (xv < c, "below")):
        if np.unique(xv[side]).size < 1:
            raise DataInsufficient(
                f"no observations {lab} the cutoff",
                recovery_hint="Check the cutoff c against the range of the "
                "running variable.",
            )
    z2 = float(stats.norm.ppf(1 - alpha / 2))

    if meth == "bsd":
        if M is None and K is not None:
            M = K
        sides = (np.unique(xv[xv >= c]).size, np.unique(xv[xv < c]).size)
        if min(sides) < 3:
            raise DataInsufficient(
                f"the running variable has {min(sides)} support point(s) on "
                "one side of the cutoff; a local line with a curvature bound "
                "needs at least three",
                recovery_hint="Use method='bme' with order=0, or a parametric "
                "fit, and report the restriction.",
            )
        try:
            fit, m_est = honest_rd(
                xv,
                yv,
                c=c,
                M=M,
                h=h,
                kernel=kernel,
                alpha=alpha,
                opt_criterion=str(opt_criterion).upper(),
            )
        except ValueError as exc:
            raise DataInsufficient(
                str(exc),
                recovery_hint="Supply M, widen h, or use method='bme'.",
            ) from exc
        est, se, bias = fit["estimate"], fit["se"], fit["bias"]
        ci = (fit["ci_lower"], fit["ci_upper"])
        pval = fit["p_value"]
        info = {
            "method": "bsd",
            "reference": "RDHonest::RDHonest",
            "M": fit["M"],
            "M_rule_of_thumb": bool(m_est),
            "bandwidth": fit["bandwidth"],
            "eff_obs": fit["eff_obs"],
            "kernel": kernel,
            "cv": fit["cv"],
            "maximum_bias": bias,
        }
        label = "BSD"
    else:
        if M is not None or K is not None:
            warnings.warn(
                "rd_discrete(method='bme') does not use M or K: the BME class "
                "bounds the specification error by the deviations estimated "
                "from the data",
                UserWarning,
                stacklevel=2,
            )
        hh = np.inf if h is None else float(h)
        b = _bme(xv - c, yv, hh, alpha, int(order))
        est, se, bias = b["estimate"], b["se"], b["maximum_bias"]
        ci = (b["conf_low"], b["conf_high"])
        t = abs(est) / se
        r = bias / se
        pval = float(stats.norm.sf(t - r) + stats.norm.cdf(-t - r))
        info = {
            "method": "bme",
            "reference": "RDHonest::RDHonestBME",
            "order": int(order),
            "bandwidth": hh,
            "eff_obs": b["eff_obs"],
            "leverage": b["leverage"],
            "conf_low_onesided": b["conf_low_onesided"],
            "conf_high_onesided": b["conf_high_onesided"],
            "p_value_rdhonest": b["p_value_rdhonest"],
            "maximum_bias": bias,
        }
        label = "BME"

    info["n_support_points"] = n_support
    info["naive_ci"] = (est - z2 * se, est + z2 * se)
    return CausalResult(
        method=f"Honest RD, discrete running variable ({label})",
        estimand="Sharp RD effect at the cutoff",
        estimate=float(est),
        se=float(se),
        pvalue=float(pval),
        ci=(float(ci[0]), float(ci[1])),
        alpha=alpha,
        n_obs=n_obs,
        model_info={"discrete": info},
    )


def _side_poly(d: np.ndarray, order: int) -> np.ndarray:
    """Design ``(1{d >= 0}, 1, d, 1{d >= 0} d, ..., d^p, 1{d >= 0} d^p)``:
    the jump coefficient first."""
    t = (d >= 0).astype(float)
    cols = [t, np.ones_like(d)]
    for p in range(1, order + 1):
        cols += [d**p, t * d**p]
    return np.column_stack(cols)


def _bme(x: np.ndarray, y: np.ndarray, h: float, alpha: float, order: int) -> dict:
    """Kolesar-Rothe BME interval (running variable centred at the cutoff).

    With ``m(x)`` the polynomial design and ``w(x)`` the support-point
    dummies inside ``|x| <= h``: ``theta`` and the cell means ``mu`` are least
    squares fits; their joint covariance is the sandwich ``sum_i T_i T_i'``
    with ``T_i = (u_i m_i' Q^-1, e_i w_i' S^-1)`` (``u``, ``e`` the two sets of
    residuals) scaled by ``n / (n - 1)``. The specification errors are
    ``delta_g = mu_g - m(x_g)' theta``. For each choice ``W`` of one support
    point and a sign on each side, ``tau_hat + b(W) +/- z se(W)`` where
    ``b(W) = s_- delta_{g-} + s_+ delta_{g+}`` and ``se(W)`` is the standard
    error of that sum; the interval is the union over ``W``
    (union-intersection, Proposition 2), at level ``1 - alpha/2`` per side
    for the two-sided interval and ``1 - alpha`` for the one-sided limits.
    """
    keep = np.abs(x) <= h
    x, y = x[keep], y[keep]
    n = int(x.size)
    sup, g = np.unique(x, return_inverse=True)
    left = np.flatnonzero(sup < 0)
    right = np.flatnonzero(sup >= 0)
    if left.size < order + 1 or right.size < order + 1:
        raise DataInsufficient(
            f"the BME interval of order {order} needs at least {order + 1} "
            "support point(s) on each side inside the window; found "
            f"{left.size} below and {right.size} above",
            recovery_hint="Widen h or lower order.",
        )
    Mx = _side_poly(x, order)
    k = Mx.shape[1]
    if n <= k:
        raise DataInsufficient(
            "too few observations in the window",
            recovery_hint="Widen h or lower order.",
        )
    Qi = np.linalg.inv(Mx.T @ Mx)
    theta = Qi @ (Mx.T @ y)
    u = y - Mx @ theta
    G = sup.size
    cnt = np.bincount(g, minlength=G).astype(float)
    mu = np.bincount(g, weights=y, minlength=G) / cnt
    e = y - mu[g]
    Wd = np.zeros((n, G))
    Wd[np.arange(n), g] = 1.0
    T = np.hstack([(u[:, None] * Mx) @ Qi, e[:, None] * Wd / cnt[None, :]])
    Omega = (T.T @ T) * n / (n - 1.0)
    Xg = _side_poly(sup, order)
    A = np.zeros((G + 1, k + G))
    A[:G, :k] = -Xg
    A[:G, k:] = np.eye(G)
    A[G, 0] = 1.0
    Sig = A @ Omega @ A.T
    delta = mu - Xg @ theta
    est = float(theta[0])
    se = float(np.sqrt(Omega[0, 0]))

    z1 = float(stats.norm.ppf(1 - alpha))
    z2 = float(stats.norm.ppf(1 - alpha / 2))
    bs, ss = [], []
    for gm, gp, sm, sp in itertools.product(left, right, (-1.0, 1.0), (-1.0, 1.0)):
        a = np.zeros(G + 1)
        a[gm], a[gp], a[G] = sm, sp, 1.0
        bs.append(sm * delta[gm] + sp * delta[gp])
        ss.append(float(np.sqrt(a @ Sig @ a)))
    bs, ss = np.asarray(bs), np.asarray(ss)
    up2, lo2 = bs + z2 * ss, bs - z2 * ss
    iu, il = int(np.argmax(up2)), int(np.argmin(lo2))
    # the bias reported is the one at the binding end of the two-sided set
    bias = float(abs(bs[iu]) if up2[iu] >= -lo2[il] else abs(bs[il]))
    kw = Qi[0] @ Mx.T
    t = abs(est) / se
    return {
        "estimate": est,
        "se": se,
        "maximum_bias": bias,
        "conf_low": est + float(lo2[il]),
        "conf_high": est + float(up2[iu]),
        "conf_low_onesided": est + float(np.min(bs - z1 * ss)),
        "conf_high_onesided": est + float(np.max(bs + z1 * ss)),
        "eff_obs": n,
        "leverage": float(np.max(kw * kw) / np.sum(kw * kw)),
        # RDHonestBME's own p-value adds the bias bound, in outcome units,
        # to the z statistic; reported for comparison only.
        "p_value_rdhonest": float(stats.norm.sf(t - bias) + stats.norm.cdf(-t - bias)),
    }
