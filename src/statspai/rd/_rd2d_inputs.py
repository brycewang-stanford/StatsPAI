"""Input handling of :func:`statspai.rd.rd2d` / :func:`statspai.rd.rd2d_bw`.

Validation, evaluation points, distances to boundary points or curves, and
the shared option checks. Part of the clean-room boundary-RD implementation
(2026-10-21); see ``docs/dev/cleanroom/rd2d.md``.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._rd2d_common import KERNEL_NAMES

_APPROACHES = ("location", "distance", "pooled")
_KERNEL_TYPES = ("prod", "rad")
_VCES = ("hc0", "hc1", "hc2", "hc3")
_BWSELECTS = (
    "mserd",
    "cerrd",
    "imserd",
    "icerrd",
    "msetwo",
    "certwo",
    "imsetwo",
    "icertwo",
)
_MASSPOINTS = ("check", "adjust", "off")
_FITMETHODS = ("joint", "separate")


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _bad(msg: str, hint: str, **diag: Any) -> MethodIncompatibility:
    return MethodIncompatibility(msg, recovery_hint=hint, diagnostics=dict(diag))


def _require_frame(
    data: Any, cols: Sequence[Tuple[str, Any]], function: str = "sp.rd2d"
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise _bad("`data` must be a pandas DataFrame.", "Pass a DataFrame as data=.")
    for name, value in cols:
        if not isinstance(value, str) or not value:
            raise _bad(
                f"`{name}` must be a non-empty column-name string.",
                f"Pass {name}='column_name'.",
            )
    check_columns(data, {name: value for name, value in cols}, function=function)


def _choice(value: Any, allowed: Sequence[str], name: str) -> str:
    v = str(value).lower() if isinstance(value, str) else value
    if v not in allowed:
        opts = ", ".join(repr(a) for a in allowed)
        raise _bad(
            f"{name} must be one of {opts}; got {value!r}", f"Use one of {opts}."
        )
    return v


def _kernel(kernel: Any) -> str:
    try:
        return KERNEL_NAMES[str(kernel).lower()]
    except KeyError:
        raise _bad(
            "kernel must be 'triangular', 'epanechnikov', 'uniform' or "
            f"'gaussian' (or tri/epa/uni/gau); got {kernel!r}",
            "Use kernel='triangular'.",
        ) from None


def _nonneg_int(v: Any, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0:
        raise _bad(
            f"{name} must be a non-negative integer", f"Pass an integer {name} >= 0."
        )
    return int(v)


def _numeric(
    data: pd.DataFrame, cols: Dict[str, Optional[str]]
) -> Dict[str, np.ndarray]:
    out = {}
    for key, col in cols.items():
        if col is None:
            continue
        try:
            out[key] = pd.to_numeric(data[col], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataInsufficient(
                "rd2d columns must be numeric",
                recovery_hint=f"Convert column {col!r} to numbers.",
                diagnostics={"column": col},
            ) from exc
    return out


def _eval_points(
    eval_points: Any, X1: Any, X2: Any, boundary: Any, n_eval: Any
) -> np.ndarray:
    """Evaluation points: user ``(k, 2)``; else on the boundary.

    Default boundary ``x1 = 0``: ``x2`` at the median (one point) or evenly
    spaced between the 10th and 90th percentiles of ``x2``. A callable
    boundary ``x2 = f(x1)``: ``x1`` evenly spaced between the 10th and 90th
    percentiles of ``x1`` (one point: their midpoint).
    """
    if eval_points is not None:
        b = np.asarray(eval_points, dtype=float)
        if b.ndim == 1 and b.size == 2:
            b = b[None, :]
        if (
            b.ndim != 2
            or b.shape[1] != 2
            or b.shape[0] == 0
            or not np.all(np.isfinite(b))
        ):
            raise _bad(
                "eval_points must be a finite array with shape (k, 2)",
                "Pass boundary points as [[x1, x2], ...].",
            )
        return b
    k = int(n_eval)
    if boundary is None:
        lo, hi = np.percentile(np.asarray(X2, dtype=float), [10, 90])
        x2 = np.array([np.median(X2)]) if k == 1 else np.linspace(lo, hi, k)
        return np.column_stack([np.zeros_like(x2), x2])
    lo, hi = np.percentile(np.asarray(X1, dtype=float), [10, 90])
    x1 = np.array([(lo + hi) / 2.0]) if k == 1 else np.linspace(lo, hi, k)
    x2 = np.array([float(boundary(v)) for v in x1])
    return np.column_stack([x1, x2])


def _bwcheck_value(bwcheck: Any, default: int) -> Optional[int]:
    if bwcheck is None:
        return None
    if isinstance(bwcheck, str):
        if bwcheck == "auto":
            return default
    elif (
        not isinstance(bwcheck, bool)
        and isinstance(bwcheck, (int, np.integer))
        and bwcheck > 0
    ):
        return int(bwcheck)
    raise _bad(
        "bwcheck must be 'auto', None or a positive integer",
        "Use bwcheck='auto' (the reference default).",
        bwcheck=bwcheck,
    )


def _kink_position(kink_position: Any, neval: int) -> np.ndarray:
    if kink_position is None:
        return np.zeros(neval, dtype=bool)
    kp = list(np.atleast_1d(kink_position))
    if kp and all(isinstance(v, (bool, np.bool_)) for v in kp):
        if len(kp) != neval:
            raise _bad(
                "kink_position must have one True/False value per boundary point.",
                "Pass a mask the length of eval_points, or 0-based indices.",
            )
        return np.asarray(kp, dtype=bool)
    mask = np.zeros(neval, dtype=bool)
    for v in kp:
        if (
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer))
            or not 0 <= v < neval
        ):
            raise _bad(
                "kink_position indices must index eval_points (0-based).",
                "Pass 0-based integer indices of the kink points.",
            )
        mask[int(v)] = True
    return mask


def _kink_unknown(kink_unknown: Any) -> Tuple[bool, bool]:
    if isinstance(kink_unknown, (bool, np.bool_)):
        return bool(kink_unknown), bool(kink_unknown)
    ku = tuple(kink_unknown) if isinstance(kink_unknown, (tuple, list)) else None
    if (
        ku is None
        or len(ku) != 2
        or not all(isinstance(v, (bool, np.bool_)) for v in ku)
    ):
        raise _bad(
            "kink_unknown must be a bool or a pair of bools.",
            "Pass kink_unknown=True or kink_unknown=(True, False).",
        )
    if ku[1] and not ku[0]:
        raise _bad(
            "kink_unknown[1] can be True only when kink_unknown[0] is True.",
            "Use kink_unknown=(True, True).",
        )
    return bool(ku[0]), bool(ku[1])


def _distance_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    T: np.ndarray,
    b: np.ndarray,
    distance: Any,
    data: Any,
    valid: np.ndarray,
) -> np.ndarray:
    """Signed distances: user-supplied, or Euclidean to each boundary point."""
    J = b.shape[0]
    if distance is None:
        sign = 2.0 * T - 1.0
        return np.column_stack(
            [np.hypot(X1 - b[j, 0], X2 - b[j, 1]) * sign for j in range(J)]
        )
    if (
        isinstance(distance, (list, tuple))
        and distance
        and all(isinstance(c, str) for c in distance)
    ):
        check_columns(data, {"distance": list(distance)}, function="sp.rd2d")
        D: np.ndarray = data[list(distance)].to_numpy(dtype=float)[valid]
    else:
        D = np.asarray(distance, dtype=float)
        if D.ndim == 1:
            D = D[:, None]
        if D.shape[0] != valid.size:
            raise _bad(
                "distance must have one row per row of data.",
                "Pass an (n, k) array aligned with data.",
            )
        D = D[valid]
    if D.shape[1] != J:
        raise _bad(
            "distance must have one column per evaluation point.",
            "Pass an (n, k) array with k = len(eval_points).",
        )
    return D


def _signed_boundary_distance(X1: Any, X2: Any, T: Any, boundary: Any) -> np.ndarray:
    """Distance to the boundary curve, positive for treated units.

    ``boundary=None`` is the line ``x1 = 0``. A callable ``x2 = f(x1)``: the
    nearest curve point is located on a grid and refined by golden-section
    search (exact for lines up to rounding).
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    sign = 2.0 * np.asarray(T, dtype=float) - 1.0
    if boundary is None:
        signed: np.ndarray = np.abs(X1) * sign
        return signed
    lo, hi = X1.min(), X1.max()
    span = hi - lo if hi > lo else 1.0
    grid = np.linspace(lo - span, hi + span, 2001)
    fg = np.asarray([boundary(v) for v in grid], dtype=float)
    if not np.all(np.isfinite(fg)):
        raise _bad(
            "boundary returned non-finite values over the range of x1.",
            "Pass a boundary function defined on the whole range of x1.",
        )
    step = grid[1] - grid[0]
    d2 = (X1[:, None] - grid[None, :]) ** 2 + (X2[:, None] - fg[None, :]) ** 2
    s0 = grid[np.argmin(d2, axis=1)]
    a, c = s0 - step, s0 + step
    f = np.vectorize(lambda v: float(boundary(v)))

    def dist2(s: np.ndarray) -> np.ndarray:
        d2s: np.ndarray = (X1 - s) ** 2 + (X2 - f(s)) ** 2
        return d2s

    g = (math.sqrt(5.0) - 1.0) / 2.0
    x_lo, x_hi = c - g * (c - a), a + g * (c - a)
    f_lo, f_hi = dist2(x_lo), dist2(x_hi)
    for _ in range(80):
        left = f_lo < f_hi
        c = np.where(left, x_hi, c)
        a = np.where(left, a, x_lo)
        x_new_lo = c - g * (c - a)
        x_new_hi = a + g * (c - a)
        x_lo, x_hi = x_new_lo, x_new_hi
        f_lo, f_hi = dist2(x_lo), dist2(x_hi)
    best = np.minimum(np.sqrt(dist2((a + c) / 2.0)), np.sqrt(d2.min(axis=1)))
    out: np.ndarray = best * sign
    return out


def _inference(est: np.ndarray, se: np.ndarray, alpha: float, side: str) -> Any:
    """t statistic, two-sided p-value and the (one- or two-sided) interval."""
    est = np.asarray(est, dtype=float)
    se = np.asarray(se, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = est / se
    pv = 2.0 * stats.norm.sf(np.abs(t))
    if side == "two":
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        return t, pv, est - z * se, est + z * se
    z = stats.norm.ppf(1.0 - alpha)
    if side == "left":
        return t, pv, np.full_like(est, -np.inf), est + z * se
    return t, pv, est - z * se, np.full_like(est, np.inf)


# --------------------------------------------------------------------------
# shared preparation
# --------------------------------------------------------------------------


def _prepare(
    data: Any,
    y: str,
    x1: str,
    x2: str,
    treatment: str,
    cluster: Optional[str],
    fuzzy: Optional[str],
    function: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    cols = [("y", y), ("x1", x1), ("x2", x2), ("treatment", treatment)]
    _require_frame(data, cols, function)
    extra = {}
    if cluster is not None:
        check_columns(data, {"cluster": cluster}, function=function)
        extra["cluster"] = cluster
    if fuzzy is not None:
        check_columns(data, {"fuzzy": fuzzy}, function=function)
    arrs = _numeric(data, {"y": y, "x1": x1, "x2": x2, "t": treatment, "fuzzy": fuzzy})
    valid = np.ones(len(data), dtype=bool)
    for v in arrs.values():
        valid &= np.isfinite(v)
    if cluster is not None:
        valid &= data[cluster].notna().to_numpy()
    arrs = {k: v[valid] for k, v in arrs.items()}
    if cluster is not None:
        arrs["cluster"] = data[cluster].to_numpy()[valid]
    if valid.sum() < 20:
        raise DataInsufficient(
            f"Too few valid observations ({int(valid.sum())}). Need at least 20.",
            recovery_hint="Provide more complete rows.",
        )
    t = arrs["t"]
    if not np.all(np.isin(t, (0.0, 1.0))):
        raise _bad("treatment must be a 0/1 indicator", "Recode treatment to 0/1.")
    n1, n0 = int((t == 1).sum()), int((t == 0).sum())
    if min(n0, n1) < 5:
        raise DataInsufficient(
            f"Too few treated ({n1}) or control ({n0}) units.",
            recovery_hint="Both sides of the boundary need observations.",
        )
    return arrs, valid


def _cluster_vce(vce: str, cluster: Any) -> str:
    if cluster is not None and vce in ("hc2", "hc3"):
        warnings.warn(
            "When cluster is specified, vce must be 'hc0' or 'hc1'. "
            "Resetting vce to 'hc1'.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "hc1"
    return vce


def _mass_points(
    arrs: Dict[str, np.ndarray], masspoints: str, bwcheck: Optional[int], p: int
) -> Optional[int]:
    """Warn on mass points (distinct score locations < 80% of n); with mass
    points and no ``bwcheck``, the floor ``50 + p + 1`` is switched on."""
    if masspoints == "off":
        return bwcheck
    X = np.column_stack([arrs["x1"], arrs["x2"]])
    share = np.unique(X, axis=0).shape[0] / X.shape[0]
    if share < 0.8:
        warnings.warn(
            "Mass points detected in the running variables.",
            RuntimeWarning,
            stacklevel=3,
        )
        if masspoints == "check":
            warnings.warn(
                "Try using option masspoints=adjust.", RuntimeWarning, stacklevel=3
            )
        if bwcheck is None:
            bwcheck = 50 + p + 1
    return bwcheck


def _check_bwcheck_size(bwcheck: Optional[int], t: np.ndarray) -> None:
    if bwcheck is None:
        return
    n0, n1 = int((t == 0).sum()), int((t == 1).sum())
    if bwcheck > min(n0, n1):
        raise DataInsufficient(
            f"bwcheck={bwcheck} requires at least {bwcheck} observations on each side "
            f"of the cutoff; found {n0} on the control side and {n1} on the treatment "
            "side. Decrease bwcheck or set bwcheck=None.",
            recovery_hint="Decrease bwcheck or set bwcheck=None.",
            diagnostics={"bwcheck": bwcheck, "n_control": n0, "n_treated": n1},
        )


def _common_options(
    kernel: Any,
    kernel_type: Any,
    vce: Any,
    bwselect: Any,
    method: Any,
    masspoints: Any,
    fitmethod: Any,
    bwparam: Any,
    p: Any,
    cqt: Any,
    scaleregul: Any,
) -> Tuple[str, str, str, str, str, str, str, str, int]:
    kernel = _kernel(kernel)
    kernel_type = _choice(kernel_type, _KERNEL_TYPES, "kernel_type")
    vce = _choice(vce, _VCES, "vce")
    bwselect = _choice(bwselect, _BWSELECTS, "bwselect")
    method = _choice(method, ("dpi", "rot"), "method")
    masspoints = _choice(masspoints, _MASSPOINTS, "masspoints")
    fitmethod = _choice(fitmethod, _FITMETHODS, "fitmethod")
    bwparam = _choice(bwparam, ("main", "itt"), "bwparam")
    p = _nonneg_int(p, "p")
    if not (isinstance(cqt, (int, float)) and 0.0 < float(cqt) <= 1.0):
        raise _bad("cqt must lie in (0, 1]", "Use cqt=0.5.", cqt=cqt)
    if scaleregul is not None and not (
        isinstance(scaleregul, (int, float))
        and math.isfinite(scaleregul)
        and scaleregul >= 0
    ):
        raise _bad(
            "scaleregul must be a non-negative number",
            "Use scaleregul=3 (rd2d) or 1 (rd2d_bw).",
        )
    return kernel, kernel_type, vce, bwselect, method, masspoints, fitmethod, bwparam, p


def _deriv_options(
    deriv: Any, tangvec: Any, p: int, J: int
) -> Tuple[Tuple[int, int], Optional[np.ndarray]]:
    d = tuple(deriv) if isinstance(deriv, (tuple, list, np.ndarray)) else None
    if (
        d is None
        or len(d) != 2
        or any(
            isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0
            for v in d
        )
        or sum(d) > p
    ):
        raise _bad(
            "deriv must be two non-negative integers summing to at most p",
            "Use deriv=(0, 0) for the level effect.",
        )
    tv = None
    if tangvec is not None:
        if p < 1:
            raise _bad(
                "tangvec requires p >= 1",
                "Use p=1 or more for directional derivatives.",
            )
        tv = np.asarray(tangvec, dtype=float)
        if tv.ndim == 1:
            tv = np.tile(tv, (J, 1))
        if tv.shape != (J, 2):
            raise _bad(
                "tangvec must have shape (k, 2), one row per evaluation point",
                "Pass one direction per point.",
            )
    return (int(d[0]), int(d[1])), tv


def _with_first_stage_guard(
    run: Callable[[str], Any], bwparam: str, is_fuzzy: bool
) -> Any:
    if not is_fuzzy:
        return run(bwparam)
    try:
        return run(bwparam)
    except DataInsufficient as exc:
        if "first-stage" not in str(exc):
            raise
        warnings.warn(
            "rd2d: zero first-stage fuzzy RD estimate detected in bandwidth "
            "selection; selecting the bandwidth for the reduced form instead.",
            RuntimeWarning,
            stacklevel=3,
        )
        return run("itt")


def _wbate_weights(weights: Any, J: int) -> np.ndarray:
    if weights is None:
        return np.full(J, 1.0 / J)
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != J or not np.all(np.isfinite(w)) or w.sum() == 0:
        raise _bad(
            "weights must be finite, one per evaluation point, with a nonzero sum",
            "Pass one weight per evaluation point.",
        )
    wn: np.ndarray = w / w.sum()
    return wn


def _location_h(h: Any, J: int) -> Optional[np.ndarray]:
    if h is None:
        return None
    a = np.asarray(h, dtype=float)
    if a.ndim == 0:
        return np.full((J, 4), float(a))
    if a.ndim == 1 and a.size == 4:
        a = np.tile(a, (J, 1))
    if a.shape != (J, 4) or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise _bad(
            "h must be a scalar or an array of shape (k, 4)",
            "One row per evaluation point.",
        )
    return a


def _distance_h(h: Any, J: int) -> Optional[np.ndarray]:
    if h is None:
        return None
    a = np.asarray(h, dtype=float)
    if a.ndim == 0:
        return np.full((J, 2), float(a))
    if a.ndim == 1 and a.size == 2:
        a = np.tile(a, (J, 1))
    if a.shape != (J, 2) or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise _bad(
            "h must be a scalar or an array of shape (k, 2)",
            "One row per evaluation point.",
        )
    return a
