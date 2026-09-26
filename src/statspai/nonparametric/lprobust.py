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
from typing import Dict, Optional

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

__all__ = [
    "LProbustPoint",
    "lprobust_at_point",
    "lpbwselect_mse_dpi",
    "lpbwselect_mse_rot",
    "lpbwselect_imse_dpi",
    "lpbwselect_imse_rot",
    "lpbwselect_ce_rot",
]

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
    raise MethodIncompatibility(f"kernel must be one of {_KERNELS}, got {kernel!r}")


def _nn_residuals(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Abadie-Imbens nearest-neighbour residuals, ties consumed whole.

    ``r_i = sqrt(J/(J+1)) * (y_i - mean of y over i's J nearest x)``,
    estimating the conditional variance without fitting a mean function
    so the variance does not inherit the smoothing bias it describes.

    **Tied x values are not a corner case here.** In a
    heterogeneous-adoption design the true stayers all sit at dose
    exactly 0, so the mass point at the evaluation point is typically the
    largest group in the sample. A plain k-NN search picks an arbitrary
    ``J`` of them and gets a different variance than the reference: on
    the ``did_had`` fixture, 25 tied groups out of 300 moved the standard
    error by 3-6% while the point estimate was still exact to 2e-9.

    ``rdrobust``/``nprobust`` instead consume a whole tie group at a
    time. That algorithm already lives in ``rd/_cct_bandwidth.py``, is
    pinned against ``rdrobust`` there, and is reused rather than
    reimplemented (CLAUDE.md §4).
    """
    n = len(x)
    if min(n_neighbors, n - 1) < 1:
        raise MethodIncompatibility(
            "nearest-neighbour residuals need at least two observations "
            f"inside the bandwidth; got {n}."
        )
    from ..rd._cct_bandwidth import _nn_residuals as _rd_nn_residuals

    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order]

    # Run lengths of tied x, and the 1-based position within each run --
    # the two arrays rdrobust's residual walker needs.
    dups = np.empty(n, dtype=int)
    dupsid = np.empty(n, dtype=int)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        run = j - i + 1
        dups[i : j + 1] = run
        dupsid[i : j + 1] = np.arange(1, run + 1)
        i = j + 1

    res_sorted = _rd_nn_residuals(xs, ys, dups, dupsid, n_neighbors)
    out = np.empty(n, dtype=float)
    out[order] = res_sorted
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
        raise MethodIncompatibility(
            f"x and y must have the same length, got {x.shape} and {y.shape}"
        )
    if kernel not in _KERNELS:
        raise MethodIncompatibility(f"kernel must be one of {_KERNELS}, got {kernel!r}")
    if p < 0 or deriv < 0 or deriv > p:
        raise MethodIncompatibility(f"need 0 <= deriv <= p, got deriv={deriv}, p={p}")
    if not np.isfinite(h) or h <= 0:
        raise MethodIncompatibility(f"h must be a positive finite bandwidth, got {h}")
    b = float(h) if b is None else float(b)
    if not np.isfinite(b) or b <= 0:
        raise MethodIncompatibility(f"b must be a positive finite bandwidth, got {b}")

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    q = p + 1
    w_h_all = _kernel_weights((x - eval_point) / h, kernel)
    w_b_all = _kernel_weights((x - eval_point) / b, kernel)
    # Union of the two windows -- see Notes.
    inside = (w_h_all > 0) | (w_b_all > 0)
    n_eff = int(inside.sum())
    if n_eff <= q + 1:
        raise MethodIncompatibility(
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
        raise MethodIncompatibility(
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


# ======================================================================
# MSE-optimal bandwidth selection (direct plug-in)
# ======================================================================

#: Rule-of-thumb pilot constant per kernel, from ``nprobust``'s
#: ``lpbwselect.mse.dpi``. Stata's ``lpbwselect.ado`` carries the same
#: four numbers.
_PILOT_CONST = {
    "epanechnikov": 2.34,
    "uniform": 1.843,
    "triangular": 2.576,
    "gaussian": 1.06,
}


def _bw_pieces(
    x: np.ndarray,
    y: np.ndarray,
    eval_point: float,
    *,
    o: int,
    nu: int,
    o_b: int,
    h_v: float,
    h_b1: float,
    h_b2: float,
    scale: float,
    kernel: str,
    n_neighbors: int,
) -> Dict[str, float]:
    """Port of ``nprobust``'s ``lprobust.bw``.

    Returns the variance constant ``V``, the two bias constants ``B1``
    and ``B2``, the regularization term ``R``, and the bandwidth ``bw``
    they imply. Three separate fits are involved and they do NOT share a
    bandwidth: ``h_v`` drives the variance, ``h_b1``/``h_b2`` the two
    bias terms.

    Note the variance fit divides its kernel weights by ``h_v`` while the
    bias fits do not — that scaling is what makes ``V`` comparable across
    bandwidths, and dropping it silently rescales every selected value.
    """
    n = len(x)

    # --- variance piece, weights scaled by 1/h_v ---
    w = _kernel_weights((x - eval_point) / h_v, kernel) / h_v
    ind = w > 0
    ex, ey, ew = x[ind], y[ind], w[ind]
    dx = ex - eval_point
    r_v = np.column_stack([dx**j for j in range(o + 1)])
    inv_g_v = np.linalg.inv(r_v.T @ (r_v * ew[:, None]))

    res_v = _nn_residuals(ex, ey, n_neighbors)
    rx = r_v * ew[:, None]
    v_v = (inv_g_v @ _meat(rx, res_v) @ inv_g_v)[nu, nu]

    hp = np.array([h_v**j for j in range(o + 1)])
    u = dx / h_v
    v1 = rx.T @ (u ** (o + 1))
    v2 = rx.T @ (u ** (o + 2))
    bconst1 = float((hp * (inv_g_v @ v1))[nu])
    bconst2 = float((hp * (inv_g_v @ v2))[nu])

    # --- first bias piece, unscaled weights ---
    w = _kernel_weights((x - eval_point) / h_b1, kernel)
    ind = w > 0
    ex, ey, ew = x[ind], y[ind], w[ind]
    dx = ex - eval_point
    r_b1 = np.column_stack([dx**j for j in range(o_b + 1)])
    inv_g_b1 = np.linalg.inv(r_b1.T @ (r_b1 * ew[:, None]))
    beta_b1 = inv_g_b1 @ ((r_b1 * ew[:, None]).T @ ey)

    bwreg = 0.0
    if scale > 0:
        res_b = _nn_residuals(ex, ey, n_neighbors)
        v_b = (inv_g_b1 @ _meat(r_b1 * ew[:, None], res_b) @ inv_g_b1)[o + 1, o + 1]
        bwreg = 3.0 * bconst1**2 * v_b

    # --- second bias piece ---
    w = _kernel_weights((x - eval_point) / h_b2, kernel)
    ind = w > 0
    ex, ey, ew = x[ind], y[ind], w[ind]
    dx = ex - eval_point
    r_b2 = np.column_stack([dx**j for j in range(o_b + 2)])
    inv_g_b2 = np.linalg.inv(r_b2.T @ (r_b2 * ew[:, None]))
    beta_b2 = inv_g_b2 @ ((r_b2 * ew[:, None]).T @ ey)

    b1 = bconst1 * beta_b1[o + 1]
    b2 = bconst2 * beta_b2[o + 2]
    v = n * h_v ** (2 * nu + 1) * v_v
    r_exp = 1.0 / (2 * o + 3)
    r_b = 2 * (o + 1 - nu)
    r_v_pow = 2 * nu + 1
    denom = n * r_b * (b1**2 + scale * bwreg)
    bw = float(((r_v_pow * v) / denom) ** r_exp) if denom > 0 else np.inf
    return {
        "V": float(v),
        "B1": float(b1),
        "B2": float(b2),
        "R": float(bwreg),
        "bw": bw,
        # Rate constants, exported because the IMSE selectors need to
        # average rV*V and rB*B1^2 over a grid. Note B.h there is
        # rB*B1^2 with NO regularization term, unlike `bw` above, which
        # carries `scale * R` -- copying `bw`'s denominator into an IMSE
        # average would be wrong in a way no smoke test would catch.
        "rV": float(r_v_pow),
        "rB": float(r_b),
    }


def lpbwselect_mse_dpi(
    x: np.ndarray,
    y: np.ndarray,
    eval_point: float,
    *,
    kernel: str = "epanechnikov",
    p: int = 1,
    deriv: int = 0,
    n_neighbors: int = 3,
    bwcheck: int = 21,
    bwregul: float = 1.0,
) -> Dict[str, float]:
    """MSE-optimal direct-plug-in bandwidths, as ``bwselect('mse-dpi')``.

    Port of ``nprobust``'s ``lpbwselect.mse.dpi``. Returns ``{'h', 'b'}``
    — the main and bias bandwidths that :func:`lprobust_at_point` takes.

    The selector is a ladder: two pilot fits at increasing polynomial
    order give the bandwidths that a third fit uses to estimate the bias
    of the fourth, which is the one you asked for. Each rung is clamped
    into ``[bw_min, bw_max]``, where ``bw_min`` is the distance to the
    ``bwcheck``-th nearest observation — without it the ladder can select
    a bandwidth containing too few points to fit.

    Only implemented for the ``(p - deriv)`` odd case, which covers the
    default local-linear level fit (``p=1, deriv=0``) and everything
    ``sp.did_had`` needs. The even case additionally requires a 1-D
    numerical optimization of the MSE expansion, and is rejected rather
    than approximated.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = np.abs(rng.gamma(1.4, 0.6, 400))
    >>> y = 0.8 + 1.3 * x + rng.normal(0, 0.5, 400)
    >>> bw = sp.lpbwselect_mse_dpi(x, y, 0.0)
    >>> bw['h'] > 0 and bw['b'] > 0
    True

    References
    ----------
    calonico2019nprobust, calonico2018effect
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    if kernel not in _PILOT_CONST:
        raise MethodIncompatibility(
            f"kernel must be one of {tuple(_PILOT_CONST)}, got {kernel!r}"
        )
    if (p - deriv) % 2 == 0:
        raise MethodIncompatibility(
            f"mse-dpi is implemented for odd (p - deriv); got p={p}, "
            f"deriv={deriv}. The even case needs a numerical optimization "
            "of the MSE expansion that is not ported yet, and is rejected "
            "rather than approximated."
        )

    n = len(x)
    q = p + 1
    x_iqr = float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
    rng_x = float(x.max() - x.min())
    bw_max = float(max(abs(eval_point - x.min()), abs(eval_point - x.max())))

    c_bw = _PILOT_CONST[kernel] * min(float(np.std(x, ddof=1)), x_iqr / 1.349)
    c_bw = min(c_bw * n ** (-1 / 5), bw_max)

    bw_min = None
    if bwcheck is not None:
        if n < bwcheck:
            raise MethodIncompatibility(
                f"bwcheck={bwcheck} needs at least that many observations, " f"got {n}."
            )
        bw_min = float(np.sort(np.abs(x - eval_point))[bwcheck - 1])
        c_bw = max(c_bw, bw_min)

    def _clamp(v: float) -> float:
        v = min(v, bw_max)
        return max(v, bw_min) if bw_min is not None else v

    common = dict(kernel=kernel, n_neighbors=n_neighbors)
    d1 = _bw_pieces(
        x,
        y,
        eval_point,
        o=q + 1,
        nu=q + 1,
        o_b=q + 2,
        h_v=c_bw,
        h_b1=rng_x,
        h_b2=rng_x,
        scale=0.0,
        **common,
    )
    bw_mp2 = _clamp(d1["bw"])
    d2 = _bw_pieces(
        x,
        y,
        eval_point,
        o=q + 2,
        nu=q + 2,
        o_b=q + 3,
        h_v=c_bw,
        h_b1=rng_x,
        h_b2=rng_x,
        scale=0.0,
        **common,
    )
    bw_mp3 = _clamp(d2["bw"])

    cb = _bw_pieces(
        x,
        y,
        eval_point,
        o=q,
        nu=p + 1,
        o_b=q + 1,
        h_v=c_bw,
        h_b1=bw_mp2,
        h_b2=bw_mp3,
        scale=bwregul,
        **common,
    )
    b_mse = _clamp(cb["bw"])

    ch = _bw_pieces(
        x,
        y,
        eval_point,
        o=p,
        nu=deriv,
        o_b=q,
        h_v=c_bw,
        h_b1=b_mse,
        h_b2=bw_mp2,
        scale=bwregul,
        **common,
    )
    h_mse = _clamp(ch["bw"])

    return {
        "h": float(h_mse),
        "b": float(b_mse),
        # Variance/bias pieces at this evaluation point, in the form
        # lpbwselect.imse.dpi averages. Odd (p - deriv) only, which is
        # the branch guarded above.
        "V_h": float(ch["rV"] * ch["V"]),
        "B_h": float(ch["rB"] * ch["B1"] ** 2),
        "V_b": float(cb["rV"] * cb["V"]),
        "B_b": float(cb["rB"] * cb["B1"] ** 2),
    }


def _kernel_moment_constants(p: int, deriv: int, kernel: str) -> Dict[str, float]:
    """Asymptotic bias and variance constants of the local polynomial fit.

    Port of ``nprobust``'s ``lp.bw.fun``. Dose zero is a **boundary**
    point of the support, so the moment integrals run over ``[0, inf)``
    rather than the whole line — and since every kernel here is
    supported on ``[-1, 1]``, that reduces to ``[0, 1]``, which is what
    the helpers below evaluate::

        Gamma_ij = int u^(i+j) K(u) du
        nu_i     = int u^(i+p+1) K(u) du
        Psi_ij   = int u^(i+j) K(u)^2 du

    ``C1 = (Gamma^-1 nu)_deriv`` scales the leading bias, ``C2 =
    (Gamma^-1 Psi Gamma^-1)_{deriv,deriv}`` the variance.

    Evaluated in closed form. Stata's ``lpbwselect.ado`` integrates these
    numerically instead, which is the whole of its 5.1e-8 disagreement.
    """
    if kernel == "gaussian":
        raise MethodIncompatibility(
            "the rule-of-thumb moment constants are implemented for the "
            "compact kernels only; got 'gaussian'."
        )

    def m1(power: int) -> float:
        """int_0^1 u^power K(u) du."""
        if kernel == "epanechnikov":
            return 0.75 * (1.0 / (power + 1) - 1.0 / (power + 3))
        if kernel == "triangular":
            return 1.0 / (power + 1) - 1.0 / (power + 2)
        if kernel == "uniform":
            return 0.5 / (power + 1)
        raise MethodIncompatibility(f"kernel must be one of {_KERNELS}, got {kernel!r}")

    def m2(power: int) -> float:
        """int_0^1 u^power K(u)^2 du."""
        if kernel == "epanechnikov":
            # (3/4)^2 (1 - u^2)^2 = 9/16 (1 - 2u^2 + u^4)
            return (9.0 / 16.0) * (
                1.0 / (power + 1) - 2.0 / (power + 3) + 1.0 / (power + 5)
            )
        if kernel == "triangular":
            # (1 - u)^2 = 1 - 2u + u^2
            return 1.0 / (power + 1) - 2.0 / (power + 2) + 1.0 / (power + 3)
        if kernel == "uniform":
            return 0.25 / (power + 1)
        raise MethodIncompatibility(f"kernel must be one of {_KERNELS}, got {kernel!r}")

    gamma = np.array(
        [[m1(i + j) for j in range(p + 1)] for i in range(p + 1)], dtype=float
    )
    nu = np.array([m1(i + p + 1) for i in range(p + 1)], dtype=float)
    psi = np.array(
        [[m2(i + j) for j in range(p + 1)] for i in range(p + 1)], dtype=float
    )

    g_inv = np.linalg.inv(gamma)
    c1 = float((g_inv @ nu)[deriv])
    c2 = float((g_inv @ psi @ g_inv)[deriv, deriv])
    return {"C1": c1, "C2": c2}


def lpbwselect_mse_rot(
    x: np.ndarray,
    y: np.ndarray,
    eval_point: float,
    *,
    kernel: str = "epanechnikov",
    p: int = 1,
    deriv: int = 0,
) -> Dict[str, float]:
    """MSE-optimal rule-of-thumb bandwidth, as ``bwselect('mse-rot')``.

    Port of ``nprobust``'s ``lpbwselect.mse.rot``. Cheaper and cruder than
    :func:`lpbwselect_mse_dpi`: the density at the evaluation point comes
    from a box count, the residual variance from a global degree-``p+3``
    polynomial, and the curvature from two high-order local fits at the
    full data range.

    Returns ``{'h', 'C1', 'C2'}`` — the bandwidth and the two kernel
    moment constants behind it.

    **No ``b``.** R reports one, but it gets it by re-entering this same
    selector at ``p = q = deriv + 1``, which makes ``(p - deriv)`` even —
    the branch that needs a 1-D optimization and is not ported. Neither
    consumer needs it: Stata's ``lprobust`` and :func:`~statspai.did_had`
    both run at ``rho = 1``, so their bias bandwidth is ``h``. Returning
    a made-up ``b`` would be worse than omitting it.

    Only the ``(p - deriv)`` odd case is implemented, matching
    :func:`lpbwselect_mse_dpi`; the even case is rejected rather than
    approximated.

    Parameters
    ----------
    x, y : array-like
        Running variable and outcome. Non-finite rows are dropped.
    eval_point : float
        Where the bandwidth is optimal for.
    kernel : {'epanechnikov', 'triangular', 'uniform'}
    p : int, default 1
        Polynomial order.
    deriv : int, default 0
        Derivative estimated; 0 is the intercept ``did_had`` needs.

    Returns
    -------
    dict with ``h``, ``C1``, ``C2``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = np.abs(rng.gamma(1.4, 0.6, 400))
    >>> y = 1.0 + 0.5 * x + rng.normal(0, 0.3, 400)
    >>> bw = sp.lpbwselect_mse_rot(x, y, 0.0)
    >>> bool(bw['h'] > 0)
    True

    References
    ----------
    calonico2019nprobust
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise MethodIncompatibility(
            f"x and y must be the same length, got {x.size} and {y.size}."
        )
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    if kernel not in _PILOT_CONST:
        raise MethodIncompatibility(
            f"kernel must be one of {tuple(_PILOT_CONST)}, got {kernel!r}"
        )
    if (p - deriv) % 2 == 0:
        raise MethodIncompatibility(
            f"mse-rot is implemented for odd (p - deriv); got p={p}, deriv={deriv}."
        )

    n = len(x)
    if n < p + 4:
        raise DataInsufficient(
            f"mse-rot fits a degree-{p + 3} global polynomial and needs at "
            f"least {p + 4} observations, got {n}."
        )

    x_iqr = float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
    rng_x = float(x.max() - x.min())
    c_bw = _PILOT_CONST[kernel] * min(float(np.std(x, ddof=1)), x_iqr / 1.349)
    c_bw *= n ** (-1 / 5)

    # Density at the evaluation point by box count, and the residual
    # variance from a global polynomial -- both deliberately crude.
    f_hat = float(np.sum(np.abs(x - eval_point) <= c_bw) / (2.0 * n * c_bw))
    if f_hat <= 0:
        raise MethodIncompatibility(
            "the rule-of-thumb density estimate at the evaluation point is "
            "zero -- no observation falls within the pilot window, so no "
            "bandwidth is defined there."
        )
    k_ord = p + 3
    design = np.column_stack([x**j for j in range(k_ord + 1)])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    s2_hat = float(resid @ resid / (n - (k_ord + 1)))

    # Curvature: high-order local fits over the whole range.
    m1 = lprobust_at_point(
        x, y, eval_point, h=rng_x, b=rng_x, kernel=kernel, p=p + 3, deriv=p + 1
    ).tau_us

    const = _kernel_moment_constants(p, deriv, kernel)
    bsq = (m1 / factorial(p + 1)) ** 2
    num = (2 * deriv + 1) * const["C2"] * (s2_hat / f_hat)
    den = 2 * (p + 1 - deriv) * const["C1"] ** 2 * bsq * n
    if den <= 0:
        raise NumericalInstability(
            "the rule-of-thumb curvature estimate is zero, so the MSE-optimal "
            "bandwidth is unbounded; supply a bandwidth explicitly."
        )
    h = float((num / den) ** (1.0 / (2 * p + 3)))
    return {
        "h": h,
        "C1": const["C1"],
        "C2": const["C2"],
        # The variance and bias pieces, in the form lpbwselect.imse.rot
        # averages over its grid: h = ((1+2v) mean(V) / (N mean(2(p+1-v)
        # B^2)))^(1/(2p+3)) reduces to the expression above at a single
        # evaluation point.
        "V": float(const["C2"] * (s2_hat / f_hat)),
        "B": float(const["C1"] * m1 / factorial(p + 1)),
    }


def lpbwselect_imse_rot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = "epanechnikov",
    p: int = 1,
    deriv: int = 0,
) -> Dict[str, float]:
    """IMSE-optimal rule-of-thumb bandwidth, as ``bwselect('imse-rot')``.

    Port of ``nprobust``'s ``lpbwselect.imse.rot``. Where ``mse-rot``
    optimizes the MSE **at one point**, this optimizes the *integrated*
    MSE, so it takes no ``eval_point``: it averages :func:`lpbwselect_mse_rot`'s
    variance and bias pieces over an interior grid of ``x``-quantiles
    (5th to 95th percentile in steps of 2.5, i.e. 37 points) and solves
    the same first-order condition on the averages::

        h = ( (1+2v) mean(V) / (N mean(2(p+1-v) B^2)) )^(1/(2p+3))

    The grid is quantile-spaced, deliberately: ``lpbwselect.imse.rot``
    ignores the ``imsegrid`` argument that the DPI variant honours, and
    uses quantiles where :func:`lpbwselect_imse_dpi` uses equal spacing.
    Copying one grid to the other silently moves the answer.

    Returns ``{'h'}``. As with :func:`lpbwselect_mse_rot`, no ``b``:
    R derives it by re-entering at ``p = q, deriv = p + 1``, an even
    ``(p - deriv)`` case that is not ported.

    Parameters
    ----------
    x, y : array-like
        Running variable and outcome. Non-finite rows are dropped.
    kernel : {'epanechnikov', 'triangular', 'uniform'}
    p : int, default 1
        Polynomial order.
    deriv : int, default 0

    Returns
    -------
    dict with ``h``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = np.abs(rng.gamma(1.4, 0.6, 400))
    >>> y = 1.0 + 0.5 * x + rng.normal(0, 0.3, 400)
    >>> bool(sp.lpbwselect_imse_rot(x, y)['h'] > 0)
    True

    References
    ----------
    calonico2019nprobust
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise MethodIncompatibility(
            f"x and y must be the same length, got {x.size} and {y.size}."
        )
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = len(x)

    if (p - deriv) % 2 == 0:
        raise MethodIncompatibility(
            f"imse-rot is implemented for odd (p - deriv); got p={p}, "
            f"deriv={deriv}. The even case needs a 1-D optimization of the "
            "IMSE expansion that is not ported."
        )

    # R: quantile(x, probs = seq(0.05, 0.95, 0.025)) -- 37 points.
    # linspace rather than arange: a float step accumulates error and can
    # silently drop or add the last probability. numpy's default quantile
    # method is linear interpolation, which is R's type 7 default.
    grid = np.quantile(x, np.linspace(0.05, 0.95, 37))
    v_vals, b_vals = [], []
    for point in grid:
        piece = lpbwselect_mse_rot(x, y, float(point), kernel=kernel, p=p, deriv=deriv)
        v_vals.append(piece["V"])
        b_vals.append(piece["B"])

    num = (1 + 2 * deriv) * float(np.mean(v_vals))
    den = n * float(np.mean(2 * (p + 1 - deriv) * np.asarray(b_vals) ** 2))
    if den <= 0:
        raise NumericalInstability(
            "the averaged rule-of-thumb curvature is zero, so the IMSE-optimal "
            "bandwidth is unbounded; supply a bandwidth explicitly."
        )
    return {"h": float((num / den) ** (1.0 / (2 * p + 3)))}


def lpbwselect_imse_dpi(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = "epanechnikov",
    p: int = 1,
    deriv: int = 0,
    imsegrid: int = 30,
    n_neighbors: int = 3,
    bwcheck: int = 21,
    bwregul: float = 1.0,
) -> Dict[str, float]:
    """IMSE-optimal direct-plug-in bandwidths, as ``bwselect('imse-dpi')``.

    Port of ``nprobust``'s ``lpbwselect.imse.dpi``: run
    :func:`lpbwselect_mse_dpi` at each of ``imsegrid`` **equally spaced**
    points across the range of ``x``, then solve on the averaged pieces::

        h = ( mean(V_h) / (N mean(B_h)) )^(1/(2p+3))
        b = ( mean(V_b) / (N mean(B_b)) )^(1/(2q+3)),   q = p + 1

    ``B_h`` here is ``rB * B1^2`` with **no** regularization term, unlike
    the pointwise ``mse-dpi`` bandwidth, which carries ``bwregul * R``.
    The two denominators genuinely differ and substituting one for the
    other moves the answer without failing anything obvious.

    Equally spaced, not quantiles — :func:`lpbwselect_imse_rot` uses
    quantiles. The reference is inconsistent here and this port follows
    it rather than harmonizing.

    Parameters
    ----------
    x, y : array-like
    kernel : {'epanechnikov', 'triangular', 'uniform', 'gaussian'}
    p, deriv : int
    imsegrid : int, default 30
        Number of equally spaced evaluation points.
    n_neighbors, bwcheck, bwregul
        Forwarded to :func:`lpbwselect_mse_dpi` at every grid point.

    Returns
    -------
    dict with ``h`` and ``b``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = np.abs(rng.gamma(1.4, 0.6, 400))
    >>> y = 1.0 + 0.5 * x + rng.normal(0, 0.3, 400)
    >>> bool(sp.lpbwselect_imse_dpi(x, y)['h'] > 0)
    True

    References
    ----------
    calonico2019nprobust
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise MethodIncompatibility(
            f"x and y must be the same length, got {x.size} and {y.size}."
        )
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    n = len(x)

    if (p - deriv) % 2 == 0:
        raise MethodIncompatibility(
            f"imse-dpi is implemented for odd (p - deriv); got p={p}, "
            f"deriv={deriv}. The even case needs a 1-D optimization of the "
            "IMSE expansion that is not ported."
        )
    if imsegrid < 2:
        raise MethodIncompatibility(f"imsegrid must be at least 2, got {imsegrid}.")

    grid = np.linspace(float(x.min()), float(x.max()), imsegrid)
    v_h, b_h, v_b, b_b = [], [], [], []
    for point in grid:
        piece = lpbwselect_mse_dpi(
            x,
            y,
            float(point),
            kernel=kernel,
            p=p,
            deriv=deriv,
            n_neighbors=n_neighbors,
            bwcheck=bwcheck,
            bwregul=bwregul,
        )
        v_h.append(piece["V_h"])
        b_h.append(piece["B_h"])
        v_b.append(piece["V_b"])
        b_b.append(piece["B_b"])

    q = p + 1
    mean_b_h, mean_b_b = float(np.mean(b_h)), float(np.mean(b_b))
    if mean_b_h <= 0 or mean_b_b <= 0:
        raise NumericalInstability(
            "the averaged plug-in curvature is zero, so the IMSE-optimal "
            "bandwidth is unbounded; supply a bandwidth explicitly."
        )
    h = (float(np.mean(v_h)) / (n * mean_b_h)) ** (1.0 / (2 * p + 3))
    b = (float(np.mean(v_b)) / (n * mean_b_b)) ** (1.0 / (2 * q + 3))
    return {"h": float(h), "b": float(b)}


def lpbwselect_ce_rot(
    x: np.ndarray,
    y: np.ndarray,
    eval_point: float,
    *,
    kernel: str = "epanechnikov",
    p: int = 1,
    deriv: int = 0,
    n_neighbors: int = 3,
    bwcheck: int = 21,
    bwregul: float = 1.0,
) -> Dict[str, float]:
    """Coverage-error-optimal rule of thumb, as ``bwselect('ce-rot')``.

    Port of ``nprobust``'s ``ce-rot`` branch. MSE-optimal bandwidths are
    too wide for *coverage*: they trade bias for variance at the rate
    that minimizes point-estimate error, which leaves enough bias to
    distort a confidence interval. The rule of thumb undersmooths the
    MSE-DPI choice by a power of ``N``::

        h = h_mse_dpi * N^( -p     / ((2p+3)(p+3)) )
        b = b_mse_dpi * N^( -(q+2) / ((2q+5)(q+3)) ),   q = p + 1

    for the odd ``(p - deriv)`` case implemented here. Cheap, because it
    reuses :func:`lpbwselect_mse_dpi` rather than deriving anything new.

    Note ``ce-dpi`` shares this **exact** ``b`` and differs only in ``h``,
    which needs a separate ~150-line routine that is not ported; see
    :func:`~statspai.did_had`.

    Parameters
    ----------
    x, y : array-like
    eval_point : float
    kernel : {'epanechnikov', 'triangular', 'uniform', 'gaussian'}
    p, deriv : int
    n_neighbors, bwcheck, bwregul
        Forwarded to :func:`lpbwselect_mse_dpi`.

    Returns
    -------
    dict with ``h`` and ``b``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = np.abs(rng.gamma(1.4, 0.6, 400))
    >>> y = 1.0 + 0.5 * x + rng.normal(0, 0.3, 400)
    >>> bw = sp.lpbwselect_ce_rot(x, y, 0.0)
    >>> bool(bw['h'] < sp.lpbwselect_mse_dpi(x, y, 0.0)['h'])   # undersmoothed
    True

    References
    ----------
    calonico2019nprobust, calonico2018effect
    """
    if (p - deriv) % 2 == 0:
        raise MethodIncompatibility(
            f"ce-rot is implemented for odd (p - deriv); got p={p}, "
            f"deriv={deriv}; the even case uses different exponents and is "
            "not ported."
        )
    mse = lpbwselect_mse_dpi(
        x,
        y,
        eval_point,
        kernel=kernel,
        p=p,
        deriv=deriv,
        n_neighbors=n_neighbors,
        bwcheck=bwcheck,
        bwregul=bwregul,
    )
    # N must be the sample mse-dpi actually used, which drops a row when
    # EITHER x or y is non-finite. Counting finite x alone would rescale
    # by the wrong N whenever y carries the missing value -- a silent few
    # percent, in the direction of undersmoothing.
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = int(np.sum(np.isfinite(xa) & np.isfinite(ya)))
    q = p + 1
    h = mse["h"] * n ** (-(p / ((2 * p + 3) * (p + 3))))
    b = mse["b"] * n ** (-((q + 2) / ((2 * q + 5) * (q + 3))))
    return {"h": float(h), "b": float(b)}
