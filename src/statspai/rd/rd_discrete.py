"""
RD inference with a discrete running variable (Kolesar & Rothe 2018).

``sp.rd_discrete`` offers the two honest confidence intervals of Kolesar
and Rothe [@kolesar2018inference], each reproducing the canonical
implementation maintained by Kolesar, R ``RDHonest``:

1. ``method='bsd'`` -- bounded second derivative.  The Armstrong-Kolesar
   honest local-linear interval, ``RDHonest(y ~ x)``: with a discrete
   running variable nothing changes except that the nearest-neighbour
   variance keeps ties whole.  Delegates to the same engine as
   :func:`sp.rd_honest` (bit-exact to ``RDHonest``).
2. ``method='bme'`` (alias ``'bm'``) -- bounded misspecification error,
   ``RDHonestBME(y ~ x)``: a global polynomial of order ``order`` on each
   side inside ``[c - h, c + h]``, with the interval taken over the worst
   combination of the specification errors at one support point on each
   side, estimated by the deviations of the cell means from the fitted
   polynomial.

.. versionchanged:: 1.29.0
   ⚠️ Rebuilt on ``RDHonest``.  Through 1.28.0 the function fitted a
   local linear regression on bin means whose variance omitted a factor of
   the bin size (standard errors understated by roughly the square root of
   the bin count), used an ad-hoc second-difference ``M``, and a
   ``K * sum|w|`` bias bound that is not the Kolesar-Rothe BME interval.
   See MIGRATION.md.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import missing_named_columns
from ..core.results import CausalResult
from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility
from ._rdhonest import honest_rd

__all__ = ["rd_discrete"]


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
    """
    Honest CI for RD with a discrete running variable (Kolesar-Rothe 2018).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    x : str
        Discrete running variable column.
    c : float, default 0.0
        Cutoff.  Treatment assigned for ``x >= c``.
    M : float, optional
        ``method='bsd'``: bound on the second derivative of the conditional
        mean.  ``None`` uses RDHonest's Armstrong-Kolesar rule of thumb.
    K : float, optional
        Deprecated and ignored: the BME interval estimates the
        specification errors itself.
    method : {'bsd', 'bme', 'bm'}, default 'bsd'
        ``'bsd'`` = ``RDHonest``; ``'bme'`` (or ``'bm'``) = ``RDHonestBME``.
    h : float, optional
        ``'bsd'``: bandwidth (``None``: RDHonest's optimal bandwidth).
        ``'bme'``: half-width of the window of support points used
        (``None``: all, RDHonest's ``h = Inf``).
    alpha : float, default 0.05
    order : int, default 0
        ``'bme'``: order of the polynomial on each side (RDHonestBME's
        ``order``; 0 compares side means).
    kernel : {'triangular', 'uniform', 'epanechnikov'}, default 'triangular'
        ``'bsd'`` kernel.
    opt_criterion : {'MSE', 'FLCI', 'OCI'}, default 'MSE'
        ``'bsd'`` bandwidth criterion when ``h`` is None.

    Returns
    -------
    CausalResult
        Estimate, standard error and the honest interval; ``model_info``
        carries RDHonest's ``maximum.bias``, ``bandwidth``, ``eff.obs``
        and, for BME, the one-sided bounds and ``leverage``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> age = rng.integers(10, 26, size=400)
    >>> outcome = 0.2 * age + 2.0 * (age >= 18) + rng.normal(0, 1.0, size=age.shape)
    >>> df = pd.DataFrame({'age_in_years': age, 'outcome': outcome})
    >>> r = sp.rd_discrete(df, y='outcome', x='age_in_years', c=18, method='bme')
    >>> r.ci[0] < r.estimate < r.ci[1]
    True

    References
    ----------
    [@kolesar2018inference]
    """
    missing = missing_named_columns({"y": y, "x": x}, data.columns)
    if missing:
        raise ColumnNotFound.from_missing(
            missing,
            list(data.columns),
            context="sp.rd_discrete",
            extra_diagnostics={"missing": next(iter(missing.values()))},
        )
    if method == "bm":
        method = "bme"
    if method not in ("bsd", "bme"):
        raise MethodIncompatibility(
            "method must be 'bsd' or 'bme' (alias 'bm').",
            recovery_hint="Use method='bsd' (RDHonest) or 'bme' (RDHonestBME).",
            diagnostics={"method": method},
        )
    if K is not None:
        warnings.warn(
            "rd_discrete(K=) is deprecated and ignored since 1.29.0: the "
            "Kolesar-Rothe BME interval estimates the specification errors "
            "from the cell means (RDHonestBME).",
            DeprecationWarning,
            stacklevel=2,
        )
    df = data[[y, x]].dropna()
    Y = df[y].to_numpy(dtype=float)
    X = df[x].to_numpy(dtype=float)
    n_support = int(len(np.unique(X)))
    if method == "bsd":
        fit, m_est = honest_rd(
            X,
            Y,
            c=float(c),
            M=M,
            h=h,
            kernel=kernel,
            alpha=alpha,
            opt_criterion=opt_criterion.upper(),
        )
        est, se, bias = fit["estimate"], fit["se"], fit["bias"]
        ci = (fit["ci_lower"], fit["ci_upper"])
        info = {
            "method": "bsd",
            "reference": "RDHonest::RDHonest",
            "M": fit["M"],
            "M_rule_of_thumb": bool(m_est),
            "bandwidth": fit["bandwidth"],
            "eff_obs": fit["eff_obs"],
            "kernel": kernel,
            "cv": fit["cv"],
        }
    else:
        out = _bme(
            X - float(c), Y, np.inf if h is None else float(h), alpha, int(order)
        )
        est, se, bias = out["estimate"], out["se"], out["maximum_bias"]
        ci = (out["conf_low"], out["conf_high"])
        info = {
            "method": "bme",
            "reference": "RDHonest::RDHonestBME",
            "order": int(order),
            "bandwidth": out["bandwidth"],
            "eff_obs": out["eff_obs"],
            "leverage": out["leverage"],
            "conf_low_onesided": out["conf_low_onesided"],
            "conf_high_onesided": out["conf_high_onesided"],
            "p_value_rdhonest": out["p_value_rdhonest"],
        }
    # Honest p-value: P(|N(b, 1)| >= |t|) with b = worst-case bias / se.
    t = abs(est / se)
    b = bias / se
    pvalue = float(stats.norm.cdf(b - t) + stats.norm.cdf(-b - t))
    info.update(
        {
            "maximum_bias": float(bias),
            "n_support_points": n_support,
            "naive_ci": (
                est - stats.norm.ppf(1 - alpha / 2) * se,
                est + stats.norm.ppf(1 - alpha / 2) * se,
            ),
        }
    )
    return CausalResult(
        method=(
            "Honest RD, discrete running variable "
            f"({'BSD' if method == 'bsd' else 'BME'})"
        ),
        estimand="Sharp RD effect at the cutoff",
        estimate=float(est),
        se=float(se),
        pvalue=pvalue,
        ci=(float(ci[0]), float(ci[1])),
        alpha=alpha,
        n_obs=int(len(Y)),
        model_info={"discrete": info},
    )


def _bme(x: np.ndarray, y: np.ndarray, h: float, alpha: float, order: int) -> dict:
    """Port of ``RDHonest::RDHonestBME`` (running variable already centred)."""
    keep = (x <= h) & (x >= -h)
    x, y = x[keep], y[keep]
    n = len(y)
    support = np.unique(x)
    G = len(support)
    Gm = int((support < 0).sum())
    if Gm < 1 or G - Gm < 1:
        raise DataInsufficient(
            "BME needs at least one support point on each side of the cutoff.",
            recovery_hint="Widen h.",
            diagnostics={"G_left": Gm, "G_right": G - Gm},
        )

    def design(v: np.ndarray) -> np.ndarray:
        d = (v >= 0).astype(float)
        pw = [v**p for p in range(1, order + 1)]
        return np.column_stack([np.ones(len(v))] + pw + [d] + [q * d for q in pw])

    X1 = design(x)
    k = X1.shape[1]
    if n <= k + G:
        raise DataInsufficient(
            "Too few observations for the BME regressions.",
            recovery_hint="Widen h or lower order.",
            diagnostics={"n": n, "k": k, "G": G},
        )
    Q1 = np.linalg.inv(X1.T @ X1)
    b1 = Q1 @ (X1.T @ y)
    r1 = y - X1 @ b1
    cell = np.searchsorted(support, x)
    cnt = np.bincount(cell, minlength=G).astype(float)
    means = np.bincount(cell, weights=y, minlength=G) / cnt
    r2 = y - means[cell]
    X2 = np.zeros((n, G))
    X2[np.arange(n), cell] = 1.0
    delta = means - design(support) @ b1
    S = np.hstack([(X1 * r1[:, None]) @ Q1, (X2 * r2[:, None]) / cnt[None, :]])
    V = n * np.cov(S, rowvar=False, ddof=1)
    jump = order + 1  # 0-based index of I(x >= 0)
    e2 = np.zeros(G + k)
    e2[jump] = 1.0
    aa = np.vstack([np.hstack([-design(support), np.eye(G)]), e2])
    vdt = aa @ V @ aa.T
    # expand.grid(1:Gm, (Gm+1):G, c(-1,1), c(-1,1)): first index fastest
    L = np.arange(Gm)
    U = np.arange(Gm, G)
    rows = []
    for s2 in (-1.0, 1.0):
        for s1 in (-1.0, 1.0):
            for u in U:
                for l_ in L:
                    rows.append((l_, u, s1, s2))
    sel = np.zeros((len(rows), G + 1))
    for i, (l_, u, s1, s2) in enumerate(rows):
        sel[i, l_] = s1
        sel[i, u] = s2
    sel[:, -1] = 1.0
    se_all = np.sqrt(np.sum((sel @ vdt) * sel, axis=1))
    dev = sel[:, :-1] @ delta
    est = float(b1[jump])
    z2, z1 = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(1 - alpha)
    ci_l, ci_u = est + dev - z2 * se_all, est + dev + z2 * se_all
    li, ui = int(np.argmin(ci_l)), int(np.argmax(ci_u))
    wt = Q1 @ X1.T
    wt = wt[jump]
    se = float(np.sqrt(vdt[G, G]))
    maxb = float(max(abs(dev[ui]), abs(dev[li])))
    tt = abs(est / se)
    return {
        "estimate": est,
        "se": se,
        "maximum_bias": maxb,
        "conf_low": float(ci_l[li]),
        "conf_high": float(ci_u[ui]),
        "conf_low_onesided": float(np.min(est + dev - z1 * se_all)),
        "conf_high_onesided": float(np.max(est + dev + z1 * se_all)),
        "bandwidth": float(h),
        "eff_obs": int(n),
        "leverage": float(np.max(wt**2) / np.sum(wt**2)),
        # RDHonestBME's own p-value adds the bias in outcome units to a z
        # statistic; kept for reference, not used as the headline.
        "p_value_rdhonest": float(
            stats.norm.cdf(maxb - tt) + stats.norm.cdf(-maxb - tt)
        ),
    }
