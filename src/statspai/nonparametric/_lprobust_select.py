"""Selector plumbing for :mod:`statspai.nonparametric.lprobust`.

Clean-room rewrite; the reference (R ``nprobust`` 1.0.0, GPL-3) was used as
a black box only. Specification: ``docs/dev/cleanroom/lprobust.md``.
"""

from __future__ import annotations

from math import factorial
from typing import Any, Dict, List, Tuple

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability
from . import _lprobust_bw as _bw

#: Probabilities of the fixed quantile grid the IMSE rule of thumb averages
#: over: 0.05, 0.075, ..., 0.95 (black-box: the reference's ``imse-rot``
#: does not depend on ``imsegrid`` or ``eval``, and is reproduced to 1e-12
#: only with this grid).
IMSE_ROT_PROBS = np.linspace(0.05, 0.95, 37)


def selector_inputs(
    x: Any, y: Any, kernel: str, p: int, deriv: int, n_neighbors: int, name: str
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = _bw.check_point_inputs(x, y, kernel, p, deriv, n_neighbors)
    _bw.check_orders(p, deriv, odd_only=name)
    return xs, ys


def run_plug_in(xs: np.ndarray, ys: np.ndarray, x0: float, **kw: Any) -> Dict[str, Any]:
    try:
        return _bw.plug_in_point(xs, ys, x0, **kw)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            f"a pilot local polynomial fit is singular at eval_point={x0!r}; "
            "the window holds too few distinct values of x.",
            recovery_hint="Raise bwcheck so every pilot window holds more "
            "observations, or supply the bandwidth directly.",
        ) from exc


def coefficient_units(
    stage: Dict[str, float], deriv: int, order: int
) -> Tuple[float, float]:
    """``(V, B)`` so that ``bw^(2 order + 3) = V / (n (B + regularization))``.

    Both are expressed for the ``deriv``-th polynomial coefficient rather
    than the derivative (the ``deriv!`` factor cancels in the bandwidth).
    """
    f2 = float(factorial(deriv) ** 2)
    v_num = (1 + 2 * deriv) * stage["variance"] / f2
    b_den = 2 * (order + 1 - deriv) * stage["bias_sq"] / f2
    return v_num, b_den


def kernel_moment_constants(p: int, deriv: int, kernel: str) -> Dict[str, float]:
    """Asymptotic bias and variance constants of the local polynomial fit.

    The equivalent-kernel constants on the one-sided support ``[0, 1]``
    (see :func:`statspai.nonparametric._lprobust_bw.one_sided_kernel_constants`),
    in closed form for the compact kernels.

    Returns
    -------
    dict
        ``bias`` (``e_nu' S^-1 c``) and ``variance`` (``e_nu' S^-1 T S^-1 e_nu``).
    """
    if kernel not in _bw.KERNELS[:3]:
        raise MethodIncompatibility(
            "the rule-of-thumb moment constants are implemented for the compact "
            f"kernels {_bw.KERNELS[:3]}; got {kernel!r}.",
            recovery_hint="Use a compact kernel, or the direct plug-in selector.",
        )
    c_bias, c_var = _bw.one_sided_kernel_constants(p, deriv, kernel)
    return {"bias": c_bias, "variance": c_var}


def rot_inputs(
    x: Any, y: Any, kernel: str, p: int, deriv: int, name: str
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float], float]:
    _bw.check_kernel(kernel)
    xs, ys = _bw.clean_xy(x, y)
    _bw.check_orders(p, deriv, odd_only=name)
    if xs.size <= p + 4:
        raise DataInsufficient(
            f"the rule of thumb fits a global polynomial of degree {p + 3}, "
            f"which needs more than {p + 4} observations; got {xs.size}.",
            recovery_hint="Supply more data or lower p.",
        )
    consts = kernel_moment_constants(p, deriv, kernel)
    sigma2 = _bw.residual_variance_global(xs, ys, p + 3)
    return xs, ys, (consts["bias"], consts["variance"]), sigma2


def rot_bandwidth(v_sum: float, b_sum: float, n: int, p: int, deriv: int) -> float:
    if not (np.isfinite(v_sum) and b_sum > 0):
        raise NumericalInstability(
            "the rule-of-thumb bias or density estimate is degenerate "
            f"(V={v_sum!r}, B^2={b_sum!r}); the bandwidth is not defined.",
            recovery_hint="Use the direct plug-in selector or supply h.",
        )
    return _bw.closed_form_bandwidth(v_sum, b_sum, n, p, deriv)


def ce_rescale(order: int, deriv: int, n: int) -> float:
    """Factor turning an MSE-optimal bandwidth into the CE rule of thumb.

    ``n^(-o/((2o+3)(o+3)))`` when ``o - deriv`` is odd and
    ``n^(-(o+2)/((2o+5)(o+3)))`` when it is even (Calonico, Cattaneo and
    Farrell 2019, Section 2.6, with the parity taken on ``o - deriv``).
    """
    if (order - deriv) % 2 == 1:
        return float(n ** (-order / ((2 * order + 3) * (order + 3))))
    return float(n ** (-(order + 2) / ((2 * order + 5) * (order + 3))))


def imse_dpi(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    p: int,
    deriv: int,
    kernel: str,
    imsegrid: int,
    n_neighbors: int,
    bwcheck: int,
    bwregul: float,
) -> Dict[str, float]:
    """Grid averages of the plug-in stages; see :func:`lpbwselect_imse_dpi`."""
    n, q = xs.size, p + 1
    stages_b, stages_h = [], []
    for g in np.linspace(xs[0], xs[-1], imsegrid):
        point = run_plug_in(
            xs,
            ys,
            float(g),
            p=p,
            deriv=deriv,
            kernel=kernel,
            n_neighbors=int(n_neighbors),
            bwcheck=int(bwcheck),
            bwregul=float(bwregul),
            with_h=False,
        )
        stages_b.append(point["stage_b"])
        try:
            stages_h.append(
                _bw.stage_terms(
                    xs,
                    ys,
                    float(g),
                    order=p,
                    deriv=deriv,
                    pilot=point["pilot"],
                    derivative_bw=point["b"],
                    regularization=0.0,
                    kernel=kernel,
                    n_neighbors=int(n_neighbors),
                )
            )
        except np.linalg.LinAlgError as exc:
            raise NumericalInstability(
                f"a local polynomial fit at grid point {g!r} is singular.",
                recovery_hint="Raise bwcheck or lower imsegrid.",
            ) from exc

    def _average(stages: List[Dict[str, Any]], order: int, d: int) -> float:
        return _bw.closed_form_bandwidth(
            float(np.mean([s["variance"] for s in stages])),
            float(np.mean([s["bias_sq"] for s in stages])),
            n,
            order,
            d,
        )

    return {"h": _average(stages_h, p, deriv), "b": _average(stages_b, q, p + 1)}
