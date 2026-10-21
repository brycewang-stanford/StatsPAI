"""Building blocks for :mod:`statspai.nonparametric.lprobust`.

Clean-room rewrite; the reference (R ``nprobust`` 1.0.0, GPL-3) was used as
a black box only. The specification these functions implement, and the
black-box evidence behind every convention below, is in
``docs/dev/cleanroom/lprobust.md``.

Contents
--------
* kernel weights and the weighted-least-squares operator of a local
  polynomial fit (computed by QR on the square-root-weighted design, so
  the normal equations are never formed);
* the nearest-neighbour variance estimate of each observation
  (Calonico, Cattaneo and Farrell 2018, supplement S.II.2.3);
* the pre-asymptotic variance and bias constants of the MSE expansion in
  Calonico, Cattaneo and Farrell (2019, Section 2.3), evaluated at a pilot
  bandwidth, and the closed-form bandwidth they imply;
* the kernel moment constants and data summaries of the rule-of-thumb
  selectors.

References
----------
[@calonico2019nprobust] [@calonico2018effect] [@calonico2014robust]
"""

from __future__ import annotations

from math import factorial
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility

KERNELS = ("epanechnikov", "triangular", "uniform", "gaussian")

#: Normal-reference (Silverman) pilot constants
#: ``(8 sqrt(pi) R(K) / (3 kappa_2(K)^2))^(1/5)``, with ``R(K) = int K^2``
#: and ``kappa_2(K) = int u^2 K``; the derivation is in the supplement of
#: Calonico, Cattaneo and Titiunik (2014). They evaluate to 2.3449,
#: 2.5760, 1.8431 and 1.0592; the reference rounds them to the values
#: below (black-box: the pilot bandwidth reproduces the reference's
#: selected bandwidths to 1e-14 only with these roundings).
NORMAL_REFERENCE_CONSTANT = {
    "epanechnikov": 2.34,
    "triangular": 2.576,
    "uniform": 1.843,
    "gaussian": 1.06,
}


def kernel_weights(u: np.ndarray, kernel: str) -> np.ndarray:
    """Kernel ``K(u)`` on the scaled distance ``u = (x - x0) / h``.

    The compact kernels are supported on ``|u| <= 1``; Epanechnikov and
    triangular are zero at ``|u| = 1`` and the uniform kernel is not, which
    is why the estimation window is defined as ``K(u) > 0``.
    """
    a = np.abs(u)
    if kernel == "epanechnikov":
        return np.where(a <= 1.0, 0.75 * (1.0 - u * u), 0.0)
    if kernel == "triangular":
        return np.where(a <= 1.0, 1.0 - a, 0.0)
    if kernel == "uniform":
        return np.where(a <= 1.0, 0.5, 0.0)
    if kernel == "gaussian":
        g: np.ndarray = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
        return g
    raise ValueError(f"kernel must be one of {KERNELS}; got {kernel!r}")


def wls_operator(u: np.ndarray, w: np.ndarray, order: int) -> np.ndarray:
    """Rows of ``(R' W R)^{-1} R' W`` for the basis ``R = [1, u, ..., u^order]``.

    Row ``j`` maps the outcome vector to the ``j``-th coefficient of the
    weighted polynomial fit, in the units of ``u``. Computed from the QR
    factorization of ``sqrt(W) R``; zero-weight rows contribute nothing.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the weighted design is rank deficient.
    """
    sw = np.sqrt(w)
    design = np.vander(u, order + 1, increasing=True) * sw[:, None]
    q_mat, r_mat = np.linalg.qr(design, mode="reduced")
    diag = np.abs(np.diag(r_mat))
    if diag.size == 0 or diag.min() <= 1e-13 * max(diag.max(), 1e-300):
        raise np.linalg.LinAlgError("rank-deficient local polynomial design")
    rows: np.ndarray = np.linalg.solve(r_mat, q_mat.T) * sw[None, :]
    return rows


def nn_variance(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Nearest-neighbour estimate of ``Var(Y_i | X_i)`` for every row.

    ``v_i = J_i / (J_i + 1) * (y_i - mean of the matched y)^2``, where the
    match set holds the ``n_neighbors`` observations closest to ``x_i``
    (the row itself excluded) plus every observation tied with the last
    of them, and ``J_i`` is its size (supplement S.II.2.3 of Calonico,
    Cattaneo and Farrell 2018; tie handling is black-box: the reference's
    standard errors are reproduced only when ties are matched whole).
    """
    n = x.size
    if n < 2:
        return np.zeros(n)
    k = min(int(n_neighbors), n - 1)
    out = np.empty(n)
    # Blocks keep the pairwise distance matrix small for large windows.
    step = max(1, int(4_000_000 // max(n, 1)))
    for start in range(0, n, step):
        stop = min(n, start + step)
        dist = np.abs(x[start:stop, None] - x[None, :])
        dist[np.arange(stop - start), np.arange(start, stop)] = np.inf
        kth = np.partition(dist, k - 1, axis=1)[:, k - 1]
        match = dist <= kth[:, None]
        size = match.sum(axis=1)
        mean = (match @ y) / size
        out[start:stop] = size / (size + 1.0) * (y[start:stop] - mean) ** 2
    return out


def pilot_bandwidth(x: np.ndarray, kernel: str) -> float:
    """Normal-reference pilot ``C_K * min(sd, IQR / 1.349) * n^(-1/5)``.

    ``sd`` uses the ``n - 1`` divisor and the quartiles are the linear
    interpolation quantiles (R's default type 7).
    """
    n = x.size
    sd = float(np.std(x, ddof=1))
    q75, q25 = np.quantile(x, [0.75, 0.25])
    spread = min(sd, float(q75 - q25) / 1.349)
    return float(NORMAL_REFERENCE_CONSTANT[kernel] * spread * n ** (-0.2))


def kth_distance(x: np.ndarray, x0: float, k: int) -> float:
    """Distance from ``x0`` to its ``k``-th nearest observation (``k >= 1``)."""
    return float(np.partition(np.abs(x - x0), k - 1)[k - 1])


def local_fit(
    x: np.ndarray, y: np.ndarray, x0: float, h: float, order: int, kernel: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Operator of an order-``order`` local fit on its kernel window.

    Returns ``(window_mask, scaled_distance_in_window, operator)`` where
    ``operator`` is :func:`wls_operator` on the window rows.

    Raises
    ------
    DataInsufficient
        Unless the window holds more distinct values of ``x`` than the fit
        has coefficients. With exactly ``order + 1`` distinct values the fit
        interpolates the cell means: the derivative it reports is the
        interpolating polynomial's, and its bias and variance constants are
        not the pre-asymptotic quantities the plug-in needs (the reference
        refuses these windows too, black-box).
    """
    u = (x - x0) / h
    w = kernel_weights(u, kernel) / h
    mask = w > 0
    distinct = np.unique(x[mask]).size
    if distinct <= order + 1:
        raise DataInsufficient(
            f"a pilot local polynomial fit of order {order} at {x0!r} has "
            f"{distinct} distinct value(s) of x in its window; it needs more "
            f"than {order + 1}, otherwise it interpolates the data exactly.",
            recovery_hint="Raise bwcheck so every pilot window holds more "
            "distinct values of x, lower p, or supply the bandwidth directly.",
            diagnostics={"distinct_x": int(distinct), "order": int(order)},
        )
    return mask, u[mask], wls_operator(u[mask], w[mask], order)


def derivative_estimate(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    h: float,
    order: int,
    k: int,
    kernel: str,
    n_neighbors: int,
) -> Tuple[float, float]:
    """``m^(k)(x0)`` from an order-``order`` local fit, and its variance.

    The variance is the sandwich ``sum_i l_i^2 v_i`` with the
    nearest-neighbour ``v_i`` computed inside the fit's own window.
    """
    mask, _, op = local_fit(x, y, x0, h, order, kernel)
    scale = factorial(k) / h**k
    row = op[k]
    est = scale * float(row @ y[mask])
    var = scale**2 * float(np.sum(row**2 * nn_variance(x[mask], y[mask], n_neighbors)))
    return est, var


def mse_constants(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    *,
    order: int,
    deriv: int,
    pilot: float,
    n_neighbors: int,
    kernel: str,
) -> Tuple[float, float]:
    """Variance constant and bias multiplier of an order-``order`` fit.

    With the fit evaluated at the pilot bandwidth ``v`` (Calonico,
    Cattaneo and Farrell 2019, Section 2.3):

    * ``variance = n v nu!^2 sum_i l_i^2 v_i``, the pre-asymptotic
      variance ``V`` scaled so that ``Var = V / (n h^(1 + 2 nu))``;
    * ``multiplier = nu! / (order + 1)! * e_nu' Gamma^{-1} Lambda``, so
      that the leading bias is ``h^(order + 1 - nu) * multiplier *
      m^(order + 1)(x0)``.
    """
    n = x.size
    mask, u, op = local_fit(x, y, x0, pilot, order, kernel)
    row = op[deriv]
    variance = (
        n
        * pilot
        * factorial(deriv) ** 2
        * float(np.sum(row**2 * nn_variance(x[mask], y[mask], n_neighbors)))
    )
    multiplier = factorial(deriv) / factorial(order + 1) * float(row @ u ** (order + 1))
    return variance, multiplier


def closed_form_bandwidth(
    variance: float, denominator: float, n: int, order: int, deriv: int
) -> float:
    """``((1 + 2 nu) V / (2 (order + 1 - nu) n D))^(1 / (2 order + 3))``.

    The minimizer of ``h^(2(order+1-nu)) D + V / (n h^(1+2nu))``. A zero
    (or negative) bias term ``D`` means the expansion puts no upper bound
    on ``h``; the result is then ``inf`` and the caller caps it.
    """
    if not denominator > 0:
        return float("inf")
    ratio = (1 + 2 * deriv) * variance / (2 * (order + 1 - deriv) * n * denominator)
    return float(ratio ** (1.0 / (2 * order + 3)))


def stage_terms(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    *,
    order: int,
    deriv: int,
    pilot: float,
    derivative_bw: float,
    regularization: float,
    kernel: str,
    n_neighbors: int,
) -> Dict[str, float]:
    """One stage of the plug-in: variance, squared bias and bandwidth.

    The unknown ``m^(order+1)(x0)`` is estimated by an order-``order + 1``
    local fit at ``derivative_bw``. The bias term in the denominator is
    ``multiplier^2 * (d^2 + 3 * regularization * Var(d))``: the ``3 Var``
    regularization term of the supplement of Calonico, Cattaneo and
    Titiunik (2014), scaled by ``regularization`` (``bwregul``). The
    unregularized term is returned separately as ``bias_sq``.
    """
    variance, multiplier = mse_constants(
        x,
        y,
        x0,
        order=order,
        deriv=deriv,
        pilot=pilot,
        n_neighbors=n_neighbors,
        kernel=kernel,
    )
    d, var_d = derivative_estimate(
        x, y, x0, derivative_bw, order + 1, order + 1, kernel, n_neighbors
    )
    bias_sq = multiplier**2 * d**2
    denominator = bias_sq + multiplier**2 * 3.0 * regularization * var_d
    return {
        "variance": variance,
        "bias_sq": bias_sq,
        "denominator": denominator,
        "bandwidth": closed_form_bandwidth(variance, denominator, x.size, order, deriv),
    }


def plug_in_point(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    *,
    p: int,
    deriv: int,
    kernel: str,
    n_neighbors: int,
    bwcheck: int,
    bwregul: float,
    with_h: bool = True,
) -> Dict[str, Any]:
    """The three-stage MSE direct plug-in at one evaluation point.

    ``x`` must be sorted. Each selected bandwidth is capped at the largest
    distance from ``x0`` to the data and floored at the distance to the
    ``bwcheck``-th nearest observation; the pilot is floored too.

    Stage 1: bandwidth for ``m^(p+2)`` with an order-``p+2`` fit, its bias
    taken from an order-``p+3`` fit over the whole sample (bandwidth = the
    range of ``x``), no regularization.
    Stage 2: ``b``, the bandwidth for ``m^(p+1)`` with an order-``p+1``
    fit, its bias derivative estimated at the stage-1 bandwidth.
    Stage 3: ``h``, its bias derivative estimated at ``b``.
    """
    q = p + 1
    cap = float(np.max(np.abs(x - x0)))
    floor = kth_distance(x, x0, bwcheck) if bwcheck > 0 else 0.0
    pilot = max(pilot_bandwidth(x, kernel), floor)
    common: Dict[str, Any] = dict(pilot=pilot, kernel=kernel, n_neighbors=n_neighbors)

    stage1 = stage_terms(
        x,
        y,
        x0,
        order=q + 1,
        deriv=q + 1,
        derivative_bw=float(np.ptp(x)),
        regularization=0.0,
        **common,
    )
    c = max(min(stage1["bandwidth"], cap), floor)
    stage2 = stage_terms(
        x,
        y,
        x0,
        order=q,
        deriv=p + 1,
        derivative_bw=c,
        regularization=bwregul,
        **common,
    )
    b = max(min(stage2["bandwidth"], cap), floor)
    if not with_h:
        return {"b": b, "c": c, "pilot": pilot, "stage_b": stage2}
    stage3 = stage_terms(
        x,
        y,
        x0,
        order=p,
        deriv=deriv,
        derivative_bw=b,
        regularization=bwregul,
        **common,
    )
    h = max(min(stage3["bandwidth"], cap), floor)
    return {
        "h": h,
        "b": b,
        "c": c,
        "pilot": pilot,
        "stage_b": stage2,
        "stage_h": stage3,
    }


# ----------------------------------------------------------------------
# Rule-of-thumb pieces
# ----------------------------------------------------------------------


def _one_sided_moment(k: int, kernel: str, squared: bool) -> float:
    """``int_0^1 u^k K(u) du`` (or with ``K^2``) in closed form."""
    if kernel == "uniform":
        return (0.25 if squared else 0.5) / (k + 1)
    if kernel == "triangular":
        if squared:
            return 1.0 / (k + 1) - 2.0 / (k + 2) + 1.0 / (k + 3)
        return 1.0 / (k + 1) - 1.0 / (k + 2)
    if kernel == "epanechnikov":
        if squared:
            return 0.5625 * (1.0 / (k + 1) - 2.0 / (k + 3) + 1.0 / (k + 5))
        return 0.75 * (1.0 / (k + 1) - 1.0 / (k + 3))
    raise ValueError(
        "the rule-of-thumb kernel constants exist in closed form for the "
        f"compact kernels {KERNELS[:3]}; got {kernel!r}"
    )


def one_sided_kernel_constants(p: int, deriv: int, kernel: str) -> Tuple[float, float]:
    """Equivalent-kernel bias and variance constants on ``[0, 1]``.

    With ``S = int_0^1 r(u) r(u)' K``, ``c = int_0^1 r(u) u^(p+1) K`` and
    ``T = int_0^1 r(u) r(u)' K^2`` for ``r(u) = (1, u, ..., u^p)``:
    ``bias = e_nu' S^{-1} c`` and ``variance = e_nu' S^{-1} T S^{-1} e_nu``.
    The one-sided (boundary) version is used at every evaluation point
    (black-box: the reference's rule-of-thumb bandwidths are reproduced at
    interior points only with these).
    """
    idx = range(p + 1)
    s_mat = np.array(
        [[_one_sided_moment(i + j, kernel, False) for j in idx] for i in idx]
    )
    c_vec = np.array([_one_sided_moment(i + p + 1, kernel, False) for i in idx])
    t_mat = np.array(
        [[_one_sided_moment(i + j, kernel, True) for j in idx] for i in idx]
    )
    s_inv = np.linalg.inv(s_mat)
    return float((s_inv @ c_vec)[deriv]), float((s_inv @ t_mat @ s_inv)[deriv, deriv])


def residual_variance_global(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    """``SSR / (n - degree - 1)`` of an unweighted global polynomial fit."""
    mid = 0.5 * (x.max() + x.min())
    half = 0.5 * float(np.ptp(x)) or 1.0
    design = np.vander((x - mid) / half, degree + 1, increasing=True)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(resid @ resid) / (x.size - degree - 1)


def rot_point_terms(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    *,
    p: int,
    deriv: int,
    kernel: str,
    sigma2: float,
    constants: Tuple[float, float],
) -> Tuple[float, float]:
    """Rule-of-thumb bias ``B`` and variance ``V`` constants at ``x0``.

    ``B = C_bias * m^(p+1)(x0) / (p+1)!`` with the derivative from an
    order-``p+3`` kernel-weighted fit over the whole sample (bandwidth =
    range of ``x``); ``V = C_var * sigma2 / f(x0)`` with ``f`` the
    uniform-kernel density estimate at the normal-reference pilot of the
    chosen kernel.
    """
    n = x.size
    span = float(np.ptp(x))
    mask, _, op = local_fit(x, y, x0, span, p + 3, kernel)
    m_coef = float(op[p + 1] @ y[mask]) / span ** (p + 1)
    pilot = pilot_bandwidth(x, kernel)
    density = 0.5 * float(np.sum(np.abs(x - x0) <= pilot)) / (n * pilot)
    c_bias, c_var = constants
    if density <= 0:
        return c_bias * m_coef, float("inf")
    return c_bias * m_coef, c_var * sigma2 / density


# ----------------------------------------------------------------------
# Input handling
# ----------------------------------------------------------------------


def clean_xy(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Float arrays sorted by ``x``, rows with a non-finite value dropped."""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if xa.shape != ya.shape:
        raise MethodIncompatibility(
            f"x and y must have the same length; got {xa.size} and {ya.size}.",
            recovery_hint="Pass the running variable and the outcome row by row.",
        )
    keep = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[keep], ya[keep]
    order = np.argsort(xa, kind="stable")
    return xa[order], ya[order]


def check_kernel(kernel: str) -> None:
    if kernel not in KERNELS:
        raise MethodIncompatibility(
            f"kernel must be one of {KERNELS}; got {kernel!r}.",
            recovery_hint="Use 'epanechnikov' (the default), 'triangular', "
            "'uniform' or 'gaussian'.",
        )


def check_orders(p: int, deriv: int, *, odd_only: Optional[str] = None) -> None:
    if not (int(p) == p and int(deriv) == deriv and 0 <= deriv <= p):
        raise MethodIncompatibility(
            f"need integers with 0 <= deriv <= p; got p={p}, deriv={deriv}.",
            recovery_hint="Raise p or lower deriv.",
        )
    if odd_only is not None and (p - deriv) % 2 == 0:
        raise MethodIncompatibility(
            f"{odd_only} is implemented for odd (p - deriv); got p={p}, "
            f"deriv={deriv}. With (p - deriv) even the bias expansion has a "
            "second leading term and the bandwidth has no closed form; that "
            "branch is not implemented.",
            recovery_hint="Use p = deriv + 1 (the default local-linear fit for "
            "deriv = 0), or supply the bandwidth directly.",
        )


def check_bwcheck(n: int, bwcheck: int) -> None:
    if int(bwcheck) != bwcheck or bwcheck < 1:
        raise MethodIncompatibility(
            f"bwcheck must be a positive integer; got {bwcheck!r}.",
            recovery_hint="bwcheck=1 imposes no minimum on the window size.",
        )
    if n < bwcheck:
        raise DataInsufficient(
            f"bwcheck={bwcheck} asks for at least {bwcheck} observations in "
            f"every window, but only {n} usable rows are available.",
            recovery_hint="Lower bwcheck or supply more data.",
            diagnostics={"n": n, "bwcheck": int(bwcheck)},
        )


def check_point_inputs(
    x: Any, y: Any, kernel: str, p: int, deriv: int, n_neighbors: int
) -> Tuple[np.ndarray, np.ndarray]:
    check_kernel(kernel)
    check_orders(p, deriv)
    if int(n_neighbors) != n_neighbors or n_neighbors < 1:
        raise MethodIncompatibility(
            f"n_neighbors must be a positive integer; got {n_neighbors!r}.",
            recovery_hint="The reference default is 3.",
        )
    return clean_xy(x, y)
