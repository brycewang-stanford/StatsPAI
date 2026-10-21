"""Local polynomial regression with robust bias-corrected inference.

Clean-room rewrite; the reference (R ``nprobust`` 1.0.0, GPL-3) was used as
a black box only. The specification, with the black-box evidence for each
convention that the papers leave open, is ``docs/dev/cleanroom/lprobust.md``.

What it computes
----------------
At one evaluation point ``x0``, the local polynomial estimate of
``m^(nu)(x0)``, the bias-corrected estimate and the two standard errors of
Calonico, Cattaneo and Farrell (2018, 2019): the conventional one and the
robust one that also carries the variance of the estimated bias.

With ``R_p`` the polynomial basis in ``(x - x0) / h``, ``W_h`` the kernel
weights at ``h`` and ``Gamma_p = R_p' W_h R_p``:

    beta_p   = Gamma_p^-1 R_p' W_h Y                         conventional
    Xi       = R_p' W_h - rho^(p+1) Lambda_p e_(p+1)' Gamma_q^-1 R_q' W_b
    beta_bc  = Gamma_p^-1 Xi Y                                bias-corrected

with ``q = p + 1``, ``rho = h / b`` and ``Lambda_p = R_p' W_h ((x-x0)/h)^(p+1)``
(Calonico, Cattaneo and Farrell 2019, Section 2.5). Both are linear in ``Y``,
so each standard error is the sandwich of its own weights with a
nearest-neighbour estimate of ``Var(Y_i | X_i)``.

Why it exists separately from :func:`statspai.lpoly`: ``lpoly`` draws a
curve; this answers "what is the regression function at one point, with an
interval whose coverage survives smoothing bias". It is what
:func:`statspai.did_had` needs at dose zero.

The five bandwidth selectors (``mse-dpi``, ``mse-rot``, ``imse-dpi``,
``imse-rot``, ``ce-rot``) follow Section 2.3 and 2.6 of Calonico,
Cattaneo and Farrell (2019); their building blocks are in
:mod:`statspai.nonparametric._lprobust_bw`.

References
----------
[@calonico2019nprobust], [@calonico2018effect], [@calonico2014robust]
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Dict, Optional

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability
from . import _lprobust_bw as _bw
from . import _lprobust_select as _sel
from ._lprobust_select import (  # noqa: F401  (private name kept)
    kernel_moment_constants as _kernel_moment_constants,
)

__all__ = [
    "LProbustPoint",
    "lprobust_at_point",
    "lpbwselect_mse_dpi",
    "lpbwselect_mse_rot",
    "lpbwselect_imse_dpi",
    "lpbwselect_imse_rot",
    "lpbwselect_ce_rot",
]

_KERNELS = _bw.KERNELS


@dataclass
class LProbustPoint:
    """One evaluation point of :func:`lprobust_at_point`.

    Attributes
    ----------
    eval_point : float
        The evaluation point ``x0``.
    h, b : float
        Main and bias bandwidths.
    n_eff : int
        Observations with positive kernel weight in the ``h`` window or the
        ``b`` window (the rows that enter either fit).
    tau_us : float
        Conventional estimate of ``m^(deriv)(x0)``.
    tau_bc : float
        Bias-corrected estimate.
    se_us : float
        Standard error of ``tau_us``.
    se_rb : float
        Robust standard error of ``tau_bc``.

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
        """Estimated bias of the conventional estimate, ``tau_us - tau_bc``."""
        return self.tau_us - self.tau_bc


# ----------------------------------------------------------------------
# Estimation at a supplied bandwidth
# ----------------------------------------------------------------------


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

    Estimates ``m^(deriv)(eval_point)`` with an order-``p`` local polynomial
    at bandwidth ``h``, removes the estimated leading bias with an
    order-``p+1`` fit at bandwidth ``b``, and reports both standard errors
    of Calonico, Cattaneo and Farrell (2018). Rows with positive kernel
    weight in either window form the estimation sample; the
    nearest-neighbour variance estimate is computed within that sample.

    No minimum window size is imposed: the bandwidths are used as given.

    Parameters
    ----------
    x : array-like
        Running variable (the regressor).
    y : array-like
        Outcome, same length as ``x``. Rows where either is non-finite are
        dropped.
    eval_point : float
        Point at which the regression function (or its derivative) is
        estimated.
    h : float
        Main bandwidth, a positive finite number.
    b : float, optional
        Bias bandwidth, for the order-``p+1`` fit that estimates the
        leading bias. Defaults to ``h`` (``rho = 1``), the reference's
        behaviour when only ``h`` is supplied.
    kernel : {'epanechnikov', 'triangular', 'uniform', 'gaussian'}
        Kernel function. Default epanechnikov, as in the reference.
    p : int
        Order of the local polynomial for the point estimate. Default 1.
    deriv : int
        Order of the derivative estimated, ``0 <= deriv <= p``. Default 0.
    n_neighbors : int
        Minimum number of nearest neighbours in the variance estimate;
        ties with the last neighbour are all included. Default 3.

    Returns
    -------
    LProbustPoint
        Bandwidths, effective sample size, the two estimates and the two
        standard errors.

    Raises
    ------
    MethodIncompatibility
        For an invalid bandwidth, kernel or order, or when the window holds
        too few distinct values of ``x`` for the polynomial fits.
    DataInsufficient
        When ``p + 2`` or fewer observations fall inside the bandwidths.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> x = rng.uniform(0, 2, 800)
    >>> y = 1 + x - 0.5 * x**2 + rng.normal(0, 0.2, 800)
    >>> fit = sp.lprobust_at_point(x, y, eval_point=0.0, h=0.6)
    >>> bool(abs(fit.tau_bc - 1.0) < 3 * fit.se_rb)
    True

    References
    ----------
    [@calonico2018effect], [@calonico2019nprobust]
    """
    x0 = float(eval_point)
    b_val = h if b is None else b
    for name, val in (("h", h), ("b", b_val)):
        if not (
            isinstance(val, (int, float, np.floating, np.integer))
            and np.isfinite(val)
            and val > 0
        ):
            raise MethodIncompatibility(
                f"{name} must be a positive finite bandwidth; got {val!r}.",
                recovery_hint="Pass a bandwidth on the scale of x, or select "
                "one with sp.lpbwselect_mse_dpi.",
            )
    h_val, b_val = float(h), float(b_val)
    xs, ys = _bw.check_point_inputs(x, y, kernel, p, deriv, n_neighbors)
    q = p + 1

    u_h = (xs - x0) / h_val
    u_b = (xs - x0) / b_val
    w_h = _bw.kernel_weights(u_h, kernel) / h_val
    w_b = _bw.kernel_weights(u_b, kernel) / b_val
    window = (w_h > 0) | (w_b > 0)
    n_eff = int(window.sum())
    if n_eff <= q + 1:
        raise DataInsufficient(
            f"only {n_eff} observations fall inside the bandwidth at "
            f"eval_point={x0!r}; a p={p} fit with bias correction needs more "
            f"than {q + 1}.",
            recovery_hint="Widen h/b or lower p.",
            diagnostics={"n_eff": n_eff, "p": int(p)},
        )
    distinct_h = np.unique(xs[w_h > 0]).size
    distinct_b = np.unique(xs[w_b > 0]).size
    if distinct_h < p + 1 or distinct_b < q + 1:
        raise MethodIncompatibility(
            f"the local polynomial design is singular at eval_point={x0!r}: "
            f"{distinct_h} distinct x values in the h window (order {p} needs "
            f"{p + 1}) and {distinct_b} in the b window (order {q} needs "
            f"{q + 1}).",
            recovery_hint="Widen h/b or lower p; the running variable has too "
            "little variation inside the bandwidth.",
            diagnostics={"distinct_h": int(distinct_h), "distinct_b": int(distinct_b)},
        )

    xw, yw = xs[window], ys[window]
    u_h, u_b, w_h, w_b = u_h[window], u_b[window], w_h[window], w_b[window]
    try:
        op_p = _bw.wls_operator(u_h, w_h, p)
        op_q = _bw.wls_operator(u_b, w_b, q)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            f"the local polynomial design is numerically singular at "
            f"eval_point={x0!r}.",
            recovery_hint="Widen h/b or lower p.",
        ) from exc
    rho = h_val / b_val
    # Gamma_p^-1 Lambda_p, the coefficient-space image of the (p+1)-th power.
    lam = op_p @ u_h ** (p + 1)
    op_bc = op_p - rho ** (p + 1) * np.outer(lam, op_q[p + 1])

    scale = factorial(deriv) / h_val**deriv
    row_us, row_bc = op_p[deriv], op_bc[deriv]
    sigma2 = _bw.nn_variance(xw, yw, int(n_neighbors))
    return LProbustPoint(
        eval_point=x0,
        h=h_val,
        b=b_val,
        n_eff=n_eff,
        tau_us=scale * float(row_us @ yw),
        tau_bc=scale * float(row_bc @ yw),
        se_us=scale * float(np.sqrt(np.sum(row_us**2 * sigma2))),
        se_rb=scale * float(np.sqrt(np.sum(row_bc**2 * sigma2))),
    )


# ----------------------------------------------------------------------
# Bandwidth selection
# ----------------------------------------------------------------------


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

    Three closed-form stages (Calonico, Cattaneo and Farrell 2019,
    Section 2.3, case ``p - deriv`` odd), all with variance and bias
    constants evaluated at a normal-reference pilot bandwidth:

    1. a bandwidth for ``m^(p+2)`` (order-``p+2`` fit), whose bias uses an
       order-``p+3`` fit over the whole sample;
    2. ``b``, the MSE-optimal bandwidth for ``m^(p+1)`` with an
       order-``p+1`` fit, its bias derivative estimated at stage 1;
    3. ``h``, the MSE-optimal bandwidth for ``m^(deriv)`` with an
       order-``p`` fit, its bias derivative estimated at ``b``.

    Stages 2 and 3 add ``bwregul`` times three times the variance of the
    estimated bias to the squared bias. Every bandwidth is capped at the
    largest distance from ``eval_point`` to the data and floored at the
    distance to the ``bwcheck``-th nearest observation.

    Parameters
    ----------
    x, y : array-like
        Running variable and outcome; rows with a non-finite value are
        dropped.
    eval_point : float
        Evaluation point.
    kernel : {'epanechnikov', 'triangular', 'uniform', 'gaussian'}
        Kernel function. Default epanechnikov.
    p : int
        Order of the local polynomial for the point estimate. Default 1.
    deriv : int
        Derivative order; ``p - deriv`` must be odd. Default 0.
    n_neighbors : int
        Nearest neighbours in the variance estimates. Default 3.
    bwcheck : int
        Minimum number of observations every window must contain. Default
        21. ``1`` imposes no minimum.
    bwregul : float
        Scale of the regularization term; ``0`` removes it. Default 1.

    Returns
    -------
    dict
        ``h`` and ``b``, plus the pieces of the closed forms in coefficient
        units: ``V_h`` / ``B_h`` (stage 3) and ``V_b`` / ``B_b`` (stage 2),
        where ``bw^(2 order + 3) = V / (n (B + regularization))``.

    Raises
    ------
    MethodIncompatibility
        For ``p - deriv`` even or invalid options.
    DataInsufficient
        When fewer than ``bwcheck`` rows are available.
    NumericalInstability
        When a pilot fit is singular.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 2, 400)
    >>> y = np.sin(2 * x) + rng.normal(0, 0.3, 400)
    >>> bw = sp.lpbwselect_mse_dpi(x, y, eval_point=1.0)
    >>> bool(0 < bw["h"] <= 1.0 + 1e-12)
    True

    References
    ----------
    [@calonico2019nprobust], [@calonico2014robust]
    """
    _bw.check_kernel(kernel)
    xs, ys = _sel.selector_inputs(x, y, kernel, p, deriv, n_neighbors, "mse-dpi")
    _bw.check_bwcheck(xs.size, bwcheck)
    res = _sel.run_plug_in(
        xs,
        ys,
        float(eval_point),
        p=p,
        deriv=deriv,
        kernel=kernel,
        n_neighbors=int(n_neighbors),
        bwcheck=int(bwcheck),
        bwregul=float(bwregul),
    )
    v_h, b_h = _sel.coefficient_units(res["stage_h"], deriv, p)
    v_b, b_b = _sel.coefficient_units(res["stage_b"], p + 1, p + 1)
    return {
        "h": res["h"],
        "b": res["b"],
        "V_h": v_h,
        "B_h": b_h,
        "V_b": v_b,
        "B_b": b_b,
    }


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

    ``h = ((1 + 2 deriv) V / (2 (p + 1 - deriv) n B^2))^(1 / (2p + 3))``
    with ``B = C1 * m^(p+1)(x0) / (p+1)!`` and ``V = C2 * sigma^2 / f(x0)``:
    ``C1``, ``C2`` are the equivalent-kernel bias and variance constants on
    ``[0, 1]``; the derivative comes from an order-``p+3`` kernel-weighted
    polynomial fit over the whole sample; ``sigma^2`` is the residual
    variance of a global polynomial of degree ``p + 3``; ``f`` is a
    uniform-kernel density estimate at the normal-reference pilot
    bandwidth. No cap or ``bwcheck`` floor is applied.

    Parameters
    ----------
    x, y : array-like
        Running variable and outcome; rows with a non-finite value are
        dropped.
    eval_point : float
        Evaluation point.
    kernel : {'epanechnikov', 'triangular', 'uniform'}
        Kernel function (compact kernels only). Default epanechnikov.
    p : int
        Polynomial order. Default 1.
    deriv : int
        Derivative order; ``p - deriv`` must be odd. Default 0.

    Returns
    -------
    dict
        ``h`` and the pieces ``B``, ``V``, ``C1``, ``C2``. The selector's
        bias bandwidth is not reported (it would need the ``p - deriv``
        even branch); consumers that run ``rho = 1`` use ``h`` for both.

    Raises
    ------
    MethodIncompatibility
        For the gaussian kernel, ``p - deriv`` even or invalid options.
    DataInsufficient
        When the global polynomial has too few observations.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 2, 400)
    >>> y = np.sin(2 * x) + rng.normal(0, 0.3, 400)
    >>> bool(sp.lpbwselect_mse_rot(x, y, eval_point=1.0)["h"] > 0)
    True

    References
    ----------
    [@calonico2019nprobust]
    """
    xs, ys, consts, sigma2 = _sel.rot_inputs(x, y, kernel, p, deriv, "mse-rot")
    b_val, v_val = _bw.rot_point_terms(
        xs,
        ys,
        float(eval_point),
        p=p,
        deriv=deriv,
        kernel=kernel,
        sigma2=sigma2,
        constants=consts,
    )
    h = _sel.rot_bandwidth(v_val, b_val**2, xs.size, p, deriv)
    return {"h": h, "B": b_val, "V": v_val, "C1": consts[0], "C2": consts[1]}


def lpbwselect_imse_rot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = "epanechnikov",
    p: int = 1,
    deriv: int = 0,
) -> Dict[str, float]:
    """IMSE-optimal rule-of-thumb bandwidth, as ``bwselect('imse-rot')``.

    The pointwise ``mse-rot`` constants ``V`` and ``B^2`` are averaged over
    the ``x`` quantiles at probabilities ``0.05, 0.075, ..., 0.95`` and put
    into the same closed form. The result is global, so there is no
    evaluation point.

    Parameters
    ----------
    x, y : array-like
        Running variable and outcome; rows with a non-finite value are
        dropped.
    kernel : {'epanechnikov', 'triangular', 'uniform'}
        Kernel function (compact kernels only). Default epanechnikov.
    p : int
        Polynomial order. Default 1.
    deriv : int
        Derivative order; ``p - deriv`` must be odd. Default 0.

    Returns
    -------
    dict
        ``h``. The bias bandwidth is not reported, as for ``mse-rot``.

    Raises
    ------
    MethodIncompatibility
        For the gaussian kernel, ``p - deriv`` even or invalid options.
    DataInsufficient
        When the global polynomial has too few observations.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 2, 400)
    >>> y = np.sin(2 * x) + rng.normal(0, 0.3, 400)
    >>> bool(sp.lpbwselect_imse_rot(x, y)["h"] > 0)
    True

    References
    ----------
    [@calonico2019nprobust]
    """
    xs, ys, consts, sigma2 = _sel.rot_inputs(x, y, kernel, p, deriv, "imse-rot")
    grid = np.quantile(xs, _sel.IMSE_ROT_PROBS)
    terms = [
        _bw.rot_point_terms(
            xs,
            ys,
            float(g),
            p=p,
            deriv=deriv,
            kernel=kernel,
            sigma2=sigma2,
            constants=consts,
        )
        for g in grid
    ]
    v_mean = float(np.mean([v for _, v in terms]))
    b_mean = float(np.mean([b**2 for b, _ in terms]))
    return {"h": _sel.rot_bandwidth(v_mean, b_mean, xs.size, p, deriv)}


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

    On ``imsegrid`` equally spaced points from ``min(x)`` to ``max(x)``:

    * ``b`` puts the grid averages of the stage-2 variance constant and
      unregularized squared bias of :func:`lpbwselect_mse_dpi` into the
      stage-2 closed form;
    * ``h`` does the same with stage 3, where the bias derivative at each
      grid point is estimated at that point's own ``mse-dpi`` ``b``
      (regularized by ``bwregul``) and the squared bias is unregularized.

    No cap or floor is applied to the two averages; ``bwcheck`` acts on the
    pointwise pilots.

    Parameters
    ----------
    x, y : array-like
        Running variable and outcome; rows with a non-finite value are
        dropped.
    kernel : {'epanechnikov', 'triangular', 'uniform', 'gaussian'}
        Kernel function. Default epanechnikov.
    p : int
        Polynomial order. Default 1.
    deriv : int
        Derivative order; ``p - deriv`` must be odd. Default 0.
    imsegrid : int
        Number of grid points, at least 2. Default 30.
    n_neighbors : int
        Nearest neighbours in the variance estimates. Default 3.
    bwcheck : int
        Minimum number of observations in every pilot window. Default 21.
    bwregul : float
        Scale of the regularization term of the pointwise ``b``. Default 1.

    Returns
    -------
    dict
        ``h`` and ``b``.

    Raises
    ------
    MethodIncompatibility
        For ``p - deriv`` even or invalid options.
    DataInsufficient
        When fewer than ``bwcheck`` rows are available.
    NumericalInstability
        When a pilot fit at a grid point is singular.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 2, 400)
    >>> y = np.sin(2 * x) + rng.normal(0, 0.3, 400)
    >>> bw = sp.lpbwselect_imse_dpi(x, y)
    >>> bool(bw["h"] > 0 and bw["b"] > 0)
    True

    References
    ----------
    [@calonico2019nprobust]
    """
    if int(imsegrid) != imsegrid or imsegrid < 2:
        raise MethodIncompatibility(
            f"imsegrid must be an integer of at least 2; got {imsegrid!r}.",
            recovery_hint="The reference default is 30.",
        )
    _bw.check_kernel(kernel)
    xs, ys = _sel.selector_inputs(x, y, kernel, p, deriv, n_neighbors, "imse-dpi")
    _bw.check_bwcheck(xs.size, bwcheck)
    return _sel.imse_dpi(
        xs,
        ys,
        p=p,
        deriv=deriv,
        kernel=kernel,
        imsegrid=int(imsegrid),
        n_neighbors=int(n_neighbors),
        bwcheck=int(bwcheck),
        bwregul=float(bwregul),
    )


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

    The :func:`lpbwselect_mse_dpi` bandwidths shrunk by the sample-size
    factors of Calonico, Cattaneo and Farrell (2019, Section 2.6):
    ``h * n^(-p/((2p+3)(p+3)))`` and ``b * n^(-(q+2)/((2q+5)(q+3)))`` with
    ``q = p + 1`` (``b`` estimates ``m^(p+1)`` with an order-``q`` fit, an
    even case). ``n`` counts the rows actually used. The floor ``bwcheck``
    acts before the rescaling, not after.

    Parameters
    ----------
    x, y : array-like
        Running variable and outcome; rows with a non-finite value are
        dropped.
    eval_point : float
        Evaluation point.
    kernel : {'epanechnikov', 'triangular', 'uniform', 'gaussian'}
        Kernel function. Default epanechnikov.
    p : int
        Polynomial order. Default 1.
    deriv : int
        Derivative order; ``p - deriv`` must be odd. Default 0.
    n_neighbors : int
        Nearest neighbours in the variance estimates. Default 3.
    bwcheck : int
        Minimum number of observations in every window. Default 21.
    bwregul : float
        Scale of the regularization term. Default 1.

    Returns
    -------
    dict
        ``h`` and ``b``.

    Raises
    ------
    MethodIncompatibility
        For ``p - deriv`` even or invalid options.
    DataInsufficient
        When fewer than ``bwcheck`` rows are available.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 2, 400)
    >>> y = np.sin(2 * x) + rng.normal(0, 0.3, 400)
    >>> ce = sp.lpbwselect_ce_rot(x, y, eval_point=1.0)
    >>> mse = sp.lpbwselect_mse_dpi(x, y, eval_point=1.0)
    >>> bool(ce["h"] < mse["h"])
    True

    References
    ----------
    [@calonico2019nprobust], [@calonico2018effect]
    """
    _bw.check_kernel(kernel)
    _bw.check_orders(p, deriv, odd_only="ce-rot")
    xs, _ = _bw.clean_xy(x, y)
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
    n = xs.size
    return {
        "h": mse["h"] * _sel.ce_rescale(p, deriv, n),
        "b": mse["b"] * _sel.ce_rescale(p + 1, p + 1, n),
    }
