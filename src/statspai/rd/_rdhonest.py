"""Armstrong-Kolesar honest RD inference, ported from R ``RDHonest``.

Why this file exists
--------------------
``sp.rd_honest`` produced the right *point estimate* but built its interval
out of two approximations that are not the Armstrong-Kolesar construction:

1. **The worst-case bias was a closed form, ``M h^2 / 6``**, rather than the
   bias of the estimator actually computed. The honest bias depends on the
   realised kernel weights, so no formula in ``h`` alone can be it. Measured
   against ``RDHonest``, the closed form was 1.58-1.62x too large -- and the
   ratio *moved with* ``h``, which is the signature of a quantity that is
   design-dependent on one side and not the other.

2. **The interval was ``estimate +/- (2 * bias + z * se)``.** The honest
   interval is ``estimate +/- cv_{1-alpha}(bias / se) * se``, where ``cv``
   is the ``1-alpha`` quantile of ``|N(t, 1)|``. The two are not the same
   and the difference is not conservative in a useful way: at ``bias/se =
   1.3`` the correct half-length is ``2.95 * se`` while the old form gave
   ``6.24 * se``, more than twice as wide.

Together these made intervals up to ~2x wider than they should be. A CI that
is needlessly wide is not "safe" -- it is a different, less informative
procedure being reported under Armstrong-Kolesar's name.

The formulas
------------
Write the estimator as a linear functional ``tau_hat = sum_i w_i Y_i``,
where ``w`` comes from the two one-sided weighted local linear fits.

* **Taylor class** (``sclass='T'``), ``|f(x) - f(0) - f'(0) x| <= M x^2/2``::

      bias = M/2 * sum_i |w_i x_i^2|

* **Holder class** (``sclass='H'``, the default: ``f'`` is ``M``-Lipschitz)::

      bias = M/2 * |sum_{x<0} w_i x_i^2  -  sum_{x>=0} w_i x_i^2|

  The Holder bound is the smaller of the two -- cancellation between the
  sides is allowed because a Lipschitz ``f'`` cannot bend in opposite
  directions at will.

* **Standard error**: ``sqrt(sum_i w_i^2 sigma2_i)`` with ``sigma2`` from
  ``J``-nearest-neighbour differences (``J = 3``).

* **Interval**: ``tau_hat +/- cv_{1-alpha}(bias/se) * se``.

References
----------
armstrong2018optimal, armstrong2020simple
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import optimize, stats

__all__ = [
    "cv_bias",
    "honest_weights",
    "honest_bias",
    "sigma_nn",
    "honest_fit",
    "honest_bandwidth",
    "m_rule_of_thumb",
]

_KERNELS = ("triangular", "uniform", "epanechnikov")


def _kern(u: np.ndarray, kernel: str) -> np.ndarray:
    a = np.abs(u)
    inside = a <= 1
    if kernel == "uniform":
        return inside.astype(float)
    if kernel == "epanechnikov":
        return 0.75 * (1 - u**2) * inside
    return (1 - a) * inside


def cv_bias(t: float, alpha: float = 0.05) -> float:
    """``RDHonest::CVb``: the ``1-alpha`` quantile of ``|N(t, 1)|``.

    Solves ``P(-cv <= Z + t <= cv) = 1 - alpha``. At ``t = 0`` this is the
    usual ``1.96``; it grows to roughly ``t + z_{1-alpha}`` for large ``t``,
    which is why using ``bias + z * se`` overstates the interval at every
    finite ``t``.

    Examples
    --------
    >>> from statspai.rd._rdhonest import cv_bias
    >>> round(cv_bias(0.0), 6)
    1.959964
    >>> round(cv_bias(1.0), 6)
    2.646146
    >>> bool(cv_bias(8.0) < 8.0 + 1.959964)  # strictly below bias + z*se
    True
    """
    t = float(abs(t))
    if not np.isfinite(t):
        return float("inf")

    def f(cv: float) -> float:
        return stats.norm.cdf(cv - t) - stats.norm.cdf(-cv - t) - (1 - alpha)

    hi = t + stats.norm.ppf(1 - alpha / 2) + 10.0
    return float(optimize.brentq(f, 0.0, hi, xtol=1e-14, rtol=1e-15))


def honest_weights(
    x: np.ndarray, c: float, h: float, kernel: str = "triangular"
) -> np.ndarray:
    """Weights ``w`` with ``tau_hat = sum_i w_i Y_i`` for local linear RD.

    Two one-sided weighted least squares fits; the RD estimate is the
    difference of their intercepts, so the right side enters with ``+`` and
    the left with ``-``.
    """
    u = (x - c) / h
    k = _kern(u, kernel)
    w = np.zeros_like(x, dtype=float)
    for sign, mask in ((1.0, x >= c), (-1.0, x < c)):
        idx = np.flatnonzero(mask & (k > 0))
        if idx.size < 2:
            continue
        xs = np.column_stack([np.ones(idx.size), x[idx] - c])
        kw = k[idx]
        xtw = (xs * kw[:, None]).T
        try:
            a = np.linalg.solve(xtw @ xs, xtw)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate window
            continue
        w[idx] += sign * a[0]
    return w


def honest_bias(
    w: np.ndarray, x: np.ndarray, c: float, M: float, sclass: str = "H"
) -> float:
    """Worst-case bias of the linear estimator over the smoothness class.

    ``sclass='H'`` (Holder, the RDHonest default) allows the two sides'
    curvature contributions to cancel; ``'T'`` (Taylor) does not, and is
    therefore always at least as large.
    """
    if sclass not in ("H", "T"):
        raise ValueError(f"sclass must be 'H' or 'T', got {sclass!r}")
    nz = w != 0
    wt, xx = w[nz], x[nz] - c
    if wt.size == 0:  # pragma: no cover - empty window
        return float("inf")
    if sclass == "T":
        return float(M / 2 * np.sum(np.abs(wt * xx**2)))
    left = np.sum(wt[xx < 0] * xx[xx < 0] ** 2)
    right = np.sum(wt[xx >= 0] * xx[xx >= 0] ** 2)
    return float(M / 2 * abs(left - right))


def sigma_nn(x: np.ndarray, y: np.ndarray, J: int = 3) -> np.ndarray:
    """``RDHonest::sigmaNN``: J-nearest-neighbour conditional variances.

    ``x`` must be sorted ascending. Ties are kept whole -- the neighbour set
    is everything within the J-th smallest distance -- so the count can
    exceed ``J``, which matters on running variables with mass points.
    """
    n = len(x)
    if n < 2:  # pragma: no cover - guarded upstream
        return np.zeros(n)
    J = min(J, n - 1)
    out = np.empty(n)
    for k in range(n):
        lo = max(k - J, 0)
        cand = np.concatenate([x[lo:k], x[k + 1 : min(k + J + 1, n)]])
        d = np.sort(np.abs(cand - x[k]))[J - 1]
        ind = np.abs(x - x[k]) <= d
        ind[k] = False
        jk = float(ind.sum())
        if jk == 0:  # pragma: no cover - only if all points coincide
            out[k] = 0.0
            continue
        out[k] = jk / (jk + 1.0) * (y[k] - y[ind].mean()) ** 2
    return out


def honest_fit(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    h: float,
    M: float,
    kernel: str = "triangular",
    alpha: float = 0.05,
    sclass: str = "H",
    J: int = 3,
    sigma2: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """One honest RD fit at a given bandwidth and smoothness bound."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    w = honest_weights(xs, c, h, kernel)
    nz = w != 0
    if nz.sum() < 4:
        raise ValueError(
            f"bandwidth h={h:g} leaves only {int(nz.sum())} observations "
            "with non-zero weight; too few for local linear RD"
        )
    est = float(np.sum(w * ys))
    bias = honest_bias(w, xs, c, M, sclass)
    if sigma2 is None:
        s2 = np.zeros_like(xs)
        # Each SIDE gets its own nearest-neighbour variance. Pooling across
        # the cutoff lets a point just left of c difference against one just
        # right of it, straddling the discontinuity the whole design is
        # about; that alone was worth ~1% on the standard error.
        for side in (xs < c, xs >= c):
            idx = np.flatnonzero(side & nz)
            if idx.size >= 2:
                s2[idx] = sigma_nn(xs[idx], ys[idx], J)
    else:
        s2 = sigma2[order]
    se = float(np.sqrt(np.sum(w[nz] ** 2 * s2[nz])))
    cv = cv_bias(bias / se, alpha) if se > 0 else float("inf")
    half = cv * se
    eff_obs = _eff_obs(xs, c, h, w)
    return {
        "estimate": est,
        "se": se,
        "bias": bias,
        "cv": cv,
        "ci_lower": est - half,
        "ci_upper": est + half,
        "half_length": half,
        "bandwidth": float(h),
        "M": float(M),
        "eff_obs": eff_obs,
        "n_effective": int(nz.sum()),
    }


def _eff_obs(x: np.ndarray, c: float, h: float, w: np.ndarray) -> float:
    """``NPReg``'s effective observations.

    The count of observations a *uniform*-kernel fit on the same window
    would need to reach this fit's variance: ``sum(Wu) * sum(wu^2) /
    sum(w^2)``, where ``wu`` are the uniform-kernel estimator weights. It is
    not ``sum`` of the kernel weights, which is what a first guess produces
    and which is off by four orders of magnitude.
    """
    wu = honest_weights(x, c, h, "uniform")
    denom = float(np.sum(w**2))
    if denom <= 0:  # pragma: no cover - guarded upstream
        return 0.0
    n_u = float(np.sum(np.abs(x - c) <= h))
    return n_u * float(np.sum(wu**2)) / denom


def _hmin(x: np.ndarray, c: float) -> float:
    """``PrelimVar``'s floor: enough distinct AND enough raw points per side."""
    xp = np.sort(x[x >= c] - c)
    xm = np.sort(np.abs(x[x < c] - c))
    up, um = np.unique(xp), np.unique(xm)
    cands = []
    for arr, k in ((up, 2), (um, 2), (xp, 3), (xm, 3)):
        if arr.size > k:
            cands.append(float(arr[k]))
    if not cands:  # pragma: no cover - degenerate design
        raise ValueError("too few observations on one side of the cutoff")
    return max(cands)


def _sidewise_mean(
    resid2: np.ndarray, x: np.ndarray, c: float, inwin: np.ndarray
) -> np.ndarray:
    """Collapse residual variances to one value per side.

    ``PrelimVar`` returns a homoskedastic-within-side variance -- the mean
    over the observations that actually entered the preliminary fit. This
    is why ``IKBW`` can index it with ``[1]``.
    """
    out = np.zeros_like(x, dtype=float)
    for mask in (x >= c, x < c):
        sel = mask & inwin
        if sel.any():
            out[mask] = float(resid2[sel].mean())
    return out


def _prelim_var_silverman(x: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    """``PrelimVar(se.initial='Silverman')``: order-0 uniform fit.

    Two group means inside a Silverman-rule window, with the usual
    ``l/(l-1)`` correction for having estimated each mean.
    """
    n = len(x)
    h1 = max(1.84 * float(np.std(x, ddof=1)) / n ** (1 / 5), _hmin(x, c))
    inwin = np.abs(x - c) <= h1
    resid = np.zeros_like(x, dtype=float)
    counts = {}
    for key, mask in (("p", x >= c), ("m", x < c)):
        sel = mask & inwin
        counts[key] = int(sel.sum())
        if sel.any():
            resid[sel] = y[sel] - y[sel].mean()
    r2 = resid**2
    for key, mask in (("p", x >= c), ("m", x < c)):
        lk = counts[key]
        if lk > 1:
            r2[mask & inwin] *= lk / (lk - 1.0)
    return _sidewise_mean(r2, x, c, inwin)


def _npreg_resid2(
    x: np.ndarray, y: np.ndarray, c: float, h: float, kernel: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Residuals from ``NPReg``'s JOINT order-1 fit, and the window mask.

    The design is ``[I(x>=c), I(x>=c)*x, 1, x]`` -- both sides in one
    weighted regression, not two separate ones. Residuals are raw (not
    weighted), and are defined outside the window too.
    """
    w = _kern((x - c) / h, kernel)
    xc = x - c
    right = (x >= c).astype(float)
    z = np.column_stack([right, right * xc, np.ones_like(xc), xc])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(z * sw[:, None], y * sw, rcond=None)
    return (y - z @ beta) ** 2, w != 0


def _ik_bandwidth(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """``RDHonest::IKBW``: Imbens-Kalyanaraman pilot bandwidth.

    Only used to pick the window for the preliminary variance, so its own
    constants are the boundary triangular-kernel ones (``nu0 = 4.8``,
    ``mu2 = -0.1``) regardless of the kernel the user asked for.
    """
    xc = x - c
    n = len(x)
    n_m = int((xc < 0).sum())
    n_p = int((xc >= 0).sum())
    const = (4.8 / 0.1**2) ** (1 / 5)
    sigma2 = _prelim_var_silverman(x, y, c)
    h1 = 1.84 * float(np.std(x, ddof=1)) / n ** (1 / 5)
    f0 = float(np.sum(np.abs(xc) <= h1)) / (2 * n * h1)
    var_m = float(sigma2[xc < 0][0])
    var_p = float(sigma2[xc >= 0][0])

    design = np.column_stack(
        [np.ones(n), (xc >= 0).astype(float), xc, xc**2, xc**3]
    )
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    m3 = 6 * beta[4]
    if m3 == 0 or f0 <= 0:  # pragma: no cover - degenerate design
        return float("nan")

    h2m = 7200 ** (1 / 7) * (var_m / (f0 * m3**2)) ** (1 / 7) * n_m ** (-1 / 7)
    h2p = 7200 ** (1 / 7) * (var_p / (f0 * m3**2)) ** (1 / 7) * n_p ** (-1 / 7)

    def quad(mask: np.ndarray) -> float:
        xi, yi = xc[mask], y[mask]
        d = np.column_stack([np.ones(xi.size), xi, xi**2])
        b, *_ = np.linalg.lstsq(d, yi, rcond=None)
        return 2 * float(b[2])

    sel_m = (xc >= -h2m) & (xc < 0)
    sel_p = (xc <= h2p) & (xc >= 0)
    if sel_m.sum() < 3 or sel_p.sum() < 3:  # pragma: no cover
        return float("nan")
    m2m, m2p = quad(sel_m), quad(sel_p)
    r_m = 2160 * var_m / (int(sel_m.sum()) * h2m**4)
    r_p = 2160 * var_p / (int(sel_p.sum()) * h2p**4)
    denom = f0 * n * ((m2p - m2m) ** 2 + r_m + r_p)
    return float(const * ((var_p + var_m) / denom) ** (1 / 5))


def _prelim_var_ehw(x: np.ndarray, y: np.ndarray, c: float, kernel: str) -> np.ndarray:
    """``PrelimVar(se.initial='EHW')``: the variance ``OptBW`` searches under.

    Freezing the variance before the bandwidth search is deliberate: it
    makes the objective a smooth function of ``h`` rather than one whose
    noise term is re-estimated at each trial value.
    """
    h1 = _ik_bandwidth(x, y, c)
    if not np.isfinite(h1):  # pragma: no cover - degenerate design
        h1 = float("inf")
    # Deliberately NOT clamped to the data range. The IK pilot bandwidth can
    # exceed it -- on a design with x in [-1, 1] it comes out at 2.62 -- and
    # RDHonest lets it, which flattens the triangular weighting rather than
    # truncating it. Clamping here changed the preliminary variance by ~5%
    # and moved the selected bandwidth by 0.9%.
    h = max(h1, _hmin(x, c))
    r2, inwin = _npreg_resid2(x, y, c, h, "triangular")
    return _sidewise_mean(r2, x, c, inwin)


def honest_bandwidth(
    x: np.ndarray,
    y: np.ndarray,
    c: float,
    M: float,
    kernel: str = "triangular",
    opt_criterion: str = "MSE",
    alpha: float = 0.05,
    beta: float = 0.8,
    sclass: str = "H",
) -> float:
    """``RDHonest::OptBW``: the bandwidth minimising the chosen criterion.

    ``MSE`` minimises ``bias^2 + se^2``; ``FLCI`` minimises the honest
    interval's length directly; ``OCI`` minimises the one-sided
    ``beta``-quantile criterion.
    """
    crit = opt_criterion.upper()
    if crit not in ("MSE", "FLCI", "OCI"):
        raise ValueError(
            f"opt_criterion must be 'MSE', 'FLCI' or 'OCI', got {opt_criterion!r}"
        )
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]
    s2 = _prelim_var_ehw(xs, ys, c, kernel)

    def obj(h: float) -> float:
        try:
            r = honest_fit(xs, ys, c, float(h), M, kernel, alpha, sclass, sigma2=s2)
        except (ValueError, np.linalg.LinAlgError):
            return float("inf")
        if crit == "MSE":
            return r["bias"] ** 2 + r["se"] ** 2
        if crit == "FLCI":
            return r["ci_upper"] - r["ci_lower"]
        return 2 * r["bias"] + r["se"] * (
            stats.norm.ppf(1 - alpha) + stats.norm.ppf(beta)
        )

    xr = xs[xs >= c] - c
    xl = np.abs(xs[xs < c] - c)
    ur, ul = np.unique(xr), np.unique(xl)
    if ur.size < 2 or ul.size < 2:  # pragma: no cover - degenerate design
        raise ValueError("each side of the cutoff needs >= 2 distinct x values")
    hmin = max(ur[1], ul[1])
    hmax = float(np.max(np.abs(xs - c)))
    if kernel == "uniform":
        # The objective is a step function in h under a uniform kernel, so a
        # continuous optimiser can stall between support points.
        supp = np.unique(np.abs(xs - c))
        supp = supp[supp >= hmin]
        vals = [obj(float(hh)) for hh in supp]
        return float(supp[int(np.argmin(vals))])
    res = optimize.minimize_scalar(
        obj,
        bounds=(hmin, hmax),
        method="bounded",
        options={"xatol": np.finfo(float).eps ** 0.75},
    )
    return float(abs(res.x))


def m_rule_of_thumb(x: np.ndarray, y: np.ndarray, c: float) -> float:
    """``RDHonest::MROT``: Armstrong-Kolesar rule of thumb for ``M``.

    A global quartic is fitted on each side; ``M`` is the largest ``|f''|``
    over that side's support, checking the interior stationary point of
    ``f''`` as well as the two endpoints. The two sides' values are maxed.
    """
    out = []
    for mask in (x >= c, x < c):
        xi, yi = x[mask], y[mask]
        if len(np.unique(xi)) < 4:
            raise ValueError(
                "insufficient unique values of the running variable on one "
                "side of the cutoff to compute the rule of thumb for M"
            )
        design = np.column_stack([xi**k for k in range(5)])
        beta, *_ = np.linalg.lstsq(design, yi, rcond=None)
        b2, b3, b4 = beta[2], beta[3], beta[4]

        def f2(v: float) -> float:
            return float(abs(2 * b2 + 6 * v * b3 + 12 * v**2 * b4))

        m = max(f2(float(xi.min())), f2(float(xi.max())))
        if abs(b4) > 1e-10:
            stat = -b3 / (4 * b4)
            if xi.min() < stat < xi.max():
                m = max(m, f2(stat))
        out.append(m)
    return float(max(out))


def honest_rd(
    x: np.ndarray,
    y: np.ndarray,
    c: float = 0.0,
    M: Optional[float] = None,
    h: Optional[float] = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
    opt_criterion: str = "MSE",
    sclass: str = "H",
    J: int = 3,
) -> Tuple[Dict[str, float], bool]:
    """Full honest RD: pick ``M`` and ``h`` if not supplied, then fit.

    Returns the fit dict and whether ``M`` came from the rule of thumb.
    """
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}, got {kernel!r}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    m_estimated = M is None
    if M is None:
        M = m_rule_of_thumb(x, y, c)
    if h is None:
        h = honest_bandwidth(x, y, c, M, kernel, opt_criterion, alpha, sclass=sclass)
    fit = honest_fit(x, y, c, float(h), float(M), kernel, alpha, sclass, J)
    return fit, m_estimated
