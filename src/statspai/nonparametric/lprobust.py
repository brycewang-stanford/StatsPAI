"""Local polynomial regression with robust bias-corrected inference.

Reproduces ``nprobust``'s ``lprobust`` (Calonico, Cattaneo & Farrell
2019, JSS 91(8)) at a supplied evaluation point: the conventional local
polynomial fit, the bias-corrected fit, and the robust bias-corrected
standard error of Calonico, Cattaneo & Farrell (2018).

Why this exists separately from :func:`statspai.lpoly`
------------------------------------------------------
``lpoly`` fits a smooth curve over a grid for description. This answers a
different question: what is the regression function *at one point*, with
an interval whose coverage survives the bias that smoothing introduces?
Undersmoothing is the usual fix and it wastes data; bias correction plus
the matching variance is the CCF alternative, and it is what
``sp.did_had`` needs at dose zero.

The estimator
-------------
With ``R_p`` the polynomial basis in ``x - x0``, ``W_h`` the kernel
weights at bandwidth ``h``, and ``G_p = R_p' W_h R_p``:

    beta_p  = G_p^-1 R_p' W_h Y                       conventional
    Q       = R_p' W_h - h^(p+1) (L e') (G_q^-1 R_q')' W_b
    beta_bc = G_p^-1 Q Y                              bias-corrected

where the second term estimates the leading bias with a ``q = p+1`` order
fit at bandwidth ``b``. Both are linear in ``Y``, so each gets a sandwich
variance built from its own weights — the robust interval uses ``Q``, not
``R_p' W_h``, which is precisely what makes it robust to the bias
correction rather than merely centred by it.

Residuals default to the heteroskedasticity-robust nearest-neighbour
estimator (``nprobust``'s ``vce(nn)`` with 3 neighbours), which needs no
variance model.

References
----------
- Calonico, S., Cattaneo, M. D. and Farrell, M. H. (2019). "nprobust:
  Nonparametric Kernel-Based Estimation and Robust Bias-Corrected
  Inference." *Journal of Statistical Software*, 91(8), 1-33.
  [@calonico2019nprobust]
- Calonico, S., Cattaneo, M. D. and Farrell, M. H. (2018). "On the Effect
  of Bias Estimation on Coverage Accuracy in Nonparametric Inference."
  *Journal of the American Statistical Association*, 113(522), 767-779.
  [@calonico2018effect]
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Optional

import numpy as np

__all__ = ["LProbustPoint", "lprobust_at_point"]

_KERNELS = ("epanechnikov", "triangular", "uniform", "gaussian")


@dataclass
class LProbustPoint:
    """One evaluation point of :func:`lprobust_at_point`.

    Attributes mirror ``lprobust``'s ``e(Result)`` columns so a reader can
    line the two up: ``tau_us`` is column 5, ``tau_bc`` column 6,
    ``se_us`` column 7, ``se_rb`` column 8.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 500)
    >>> y = np.sin(2 * x) + rng.normal(0, 0.3, 500)
    >>> point = sp.lprobust_at_point(x=x, y=y, eval_point=0.0, h=0.5)
    >>> type(point).__name__
    'LProbustPoint'
    >>> point.n_eff                      # observations inside the bandwidth
    248

    The robust bias-corrected standard error is wider than the conventional
    one, because it also carries the variance of the estimated bias:

    >>> bool(point.se_rb > point.se_us)
    True
    """

    eval_point: float
    h: float
    b: float
    n_eff: int
    tau_us: float
    tau_bc: float
    se_us: float
    se_rb: float

    @property
    def bias(self) -> float:
        """Estimated leading bias, ``tau_us - tau_bc``.

        ``did_had`` uses this directly rather than the conventional
        standard error when forming its interval.
        """
        return self.tau_us - self.tau_bc


def _kernel_weights(u: np.ndarray, kernel: str) -> np.ndarray:
    """Kernel weights on the scaled distance ``u = (x - x0) / h``.

    Compact kernels return exactly zero outside ``|u| <= 1``; the caller
    relies on that to define the effective window, so the boundary must
    not leak small positive weights.
    """
    a = np.abs(u)
    if kernel == "epanechnikov":
        return np.where(a <= 1.0, 0.75 * (1.0 - u**2), 0.0)
    if kernel == "triangular":
        return np.where(a <= 1.0, 1.0 - a, 0.0)
    if kernel == "uniform":
        return np.where(a <= 1.0, 0.5, 0.0)
    if kernel == "gaussian":
        return np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)
    raise ValueError(f"kernel must be one of {_KERNELS}, got {kernel!r}")


def _nn_residuals(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Abadie-Imbens nearest-neighbour residuals.

    ``r_i = sqrt(J/(J+1)) * (y_i - mean of y over i's J nearest x)``.

    This estimates the conditional variance without fitting a mean
    function, so the variance does not inherit the smoothing bias it is
    meant to describe. The ``sqrt(J/(J+1))`` factor makes ``r_i^2``
    unbiased for ``sigma^2(x_i)`` under local constancy.

    Ties are broken by ``argpartition``'s ordering, matching how the
    reference walks equally distant neighbours; with duplicated ``x`` the
    choice among tied neighbours does not change the mean when their
    outcomes are exchangeable, and does change it otherwise — see the
    masspoints caveat in the reference's documentation.
    """
    n = len(x)
    j = min(n_neighbors, n - 1)
    if j < 1:
        raise ValueError(
            "nearest-neighbour residuals need at least two observations "
            f"inside the bandwidth; got {n}."
        )
    out = np.empty(n, dtype=float)
    scale = np.sqrt(j / (j + 1.0))
    for i in range(n):
        dist = np.abs(x - x[i])
        dist[i] = np.inf
        nb = np.argpartition(dist, j - 1)[:j]
        out[i] = scale * (y[i] - y[nb].mean())
    return out


def _meat(weights: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """Sandwich meat ``sum_i (w_i r_i)(w_i r_i)'``."""
    m = weights * resid[:, None]
    return m.T @ m


def lprobust_at_point(
    x: np.ndarray,
    y: np.ndarray,
    eval_point: float,
    h: float,
    b: Optional[float] = None,
    *,
    kernel: str = "epanechnikov",
    p: int = 1,
    deriv: int = 0,
    n_neighbors: int = 3,
) -> LProbustPoint:
    """Local polynomial fit at ``eval_point`` with robust bias correction.

    Parameters
    ----------
    x, y : ndarray
        Running variable and outcome, same length.
    eval_point : float
        Where to evaluate the regression function.
    h : float
        Main bandwidth, for the order-``p`` fit.
    b : float, optional
        Bias bandwidth, for the order-``p+1`` fit that estimates the
        leading bias. Defaults to ``h`` (``rho = 1``), which is
        ``lprobust``'s behaviour when only ``h`` is supplied.
    kernel : {'epanechnikov', 'triangular', 'uniform', 'gaussian'}
        Default epanechnikov, matching ``lprobust``.
    p : int, default 1
        Polynomial order — 1 is local linear.
    deriv : int, default 0
        Order of the derivative to report; 0 is the level.
    n_neighbors : int, default 3
        Neighbours for the nearest-neighbour variance, matching
        ``lprobust``'s ``vce(nn)`` default of 3.

    Returns
    -------
    LProbustPoint

    Notes
    -----
    The fit uses the **union** of the ``h`` and ``b`` windows, so
    observations only reachable by the larger bandwidth still enter the
    bias-correction step. Restricting to the ``h`` window alone silently
    drops them and moves ``tau_bc`` whenever ``b > h``.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = np.abs(rng.gamma(1.4, 0.6, 400))
    >>> y = 0.8 + 1.3 * x + rng.normal(0, 0.5, 400)
    >>> fit = sp.lprobust_at_point(x, y, 0.0, h=0.8)
    >>> bool(fit.se_rb > fit.se_us)  # robust interval is wider
    True

    References
    ----------
    calonico2019nprobust, calonico2018effect
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same length, got {x.shape} and {y.shape}"
        )
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}, got {kernel!r}")
    if p < 0 or deriv < 0 or deriv > p:
        raise ValueError(f"need 0 <= deriv <= p, got deriv={deriv}, p={p}")
    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"h must be a positive finite bandwidth, got {h}")
    b = float(h) if b is None else float(b)
    if not np.isfinite(b) or b <= 0:
        raise ValueError(f"b must be a positive finite bandwidth, got {b}")

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    q = p + 1
    w_h_all = _kernel_weights((x - eval_point) / h, kernel)
    w_b_all = _kernel_weights((x - eval_point) / b, kernel)
    # Union of the two windows -- see Notes.
    inside = (w_h_all > 0) | (w_b_all > 0)
    n_eff = int(inside.sum())
    if n_eff <= q + 1:
        raise ValueError(
            f"only {n_eff} observations fall inside the bandwidth at "
            f"eval_point={eval_point}; a p={p} fit with bias correction "
            f"needs more than {q + 1}. Widen h/b or lower p."
        )

    ex, ey = x[inside], y[inside]
    w_h, w_b = w_h_all[inside], w_b_all[inside]
    dx = ex - eval_point

    r_q = np.column_stack([dx**j for j in range(q + 1)])
    r_p = r_q[:, : p + 1]

    g_p = r_p.T @ (r_p * w_h[:, None])
    g_q = r_q.T @ (r_q * w_b[:, None])
    try:
        inv_g_p = np.linalg.inv(g_p)
        inv_g_q = np.linalg.inv(g_q)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the local polynomial design is singular at "
            f"eval_point={eval_point} -- the running variable has too "
            "little variation inside the bandwidth."
        ) from exc

    u = dx / h
    lvec = (r_p * w_h[:, None]).T @ (u ** (p + 1))
    e_p1 = np.zeros(q + 1)
    e_p1[p + 1] = 1.0

    rx_h = r_p * w_h[:, None]
    correction = np.outer(lvec, e_p1) @ ((inv_g_q @ r_q.T).T * w_b[:, None]).T
    q_mat = (rx_h.T - h ** (p + 1) * correction).T

    beta_p = inv_g_p @ (rx_h.T @ ey)
    beta_bc = inv_g_p @ (q_mat.T @ ey)

    scale = float(factorial(deriv))
    tau_us = scale * beta_p[deriv]
    tau_bc = scale * beta_bc[deriv]

    resid = _nn_residuals(ex, ey, n_neighbors)
    v_us = inv_g_p @ _meat(rx_h, resid) @ inv_g_p
    v_rb = inv_g_p @ _meat(q_mat, resid) @ inv_g_p

    return LProbustPoint(
        eval_point=float(eval_point),
        h=float(h),
        b=float(b),
        # The UNION window, matching lprobust's e(Result)[1,4]. With b > h
        # this exceeds the h-window count (317 vs 241 on the parity
        # fixture), because the bias step genuinely uses those rows.
        n_eff=n_eff,
        tau_us=float(tau_us),
        tau_bc=float(tau_bc),
        se_us=float(scale * np.sqrt(max(v_us[deriv, deriv], 0.0))),
        se_rb=float(scale * np.sqrt(max(v_rb[deriv, deriv], 0.0))),
    )
