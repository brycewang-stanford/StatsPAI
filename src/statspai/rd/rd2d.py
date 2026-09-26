"""
Boundary discontinuity designs with a bivariate score (R ``rd2d``).

Treatment is assigned by position relative to a boundary in the plane of
two running variables ``(x1, x2)``.  The parameter of interest is the
*pointwise* boundary effect ``tau(b) = E[Y(1) - Y(0) | X = b]`` at chosen
points ``b`` on the boundary.  Two estimators are provided, both ports of
the R package ``rd2d`` 1.0.0 by Cattaneo, Titiunik and Yu
[@cattaneo2025boundary] and reproduced to machine precision on the same
data bytes (``tests/reference_parity/test_rd_open_R_parity.py``):

- ``approach="location"`` (R ``rd2d``): a bivariate local polynomial in
  ``(x1, x2)`` fitted on each side at every boundary point, with the
  ``rdbw2d`` MSE/CER-optimal bandwidths and robust bias-corrected
  inference (the q = p + 1 fit supplies the bias-corrected estimate and
  its standard error).
- ``approach="distance"`` (R ``rd2d.distance``): a univariate local
  polynomial in the Euclidean distance to each boundary point, signed by
  assignment, with the ``rdbw2d.distance`` bandwidths.

A third, ``approach="pooled"``, is the older one-score design of Keele and
Titiunik [@keele2015geographic]: it collapses the plane onto the signed
perpendicular distance to the whole boundary and runs :func:`sp.rdrobust`
on it, so it reports a single pooled effect rather than pointwise effects.

.. versionchanged:: 1.29.0
   ⚠️ Rebuilt on R ``rd2d``.  Through 1.28.0 ``approach="distance"`` (then
   the default) ran a home-grown local linear fit on the signed distance to
   the boundary *line* with a Silverman-type bandwidth and reported one
   pooled number, and ``approach="location"`` pooled pointwise fits by
   inverse-variance weights with a rule-of-thumb bandwidth; neither is the
   estimator of the reference.  See MIGRATION.md.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._rd2d_distance import rd2d_distance_bw, rd2d_distance_estimate
from ._rd2d_location import _KERNEL_CANON, Rd2dData, rd2d_location_bw, rd2d_location_fit
from ._rd2d_plot import rd2d_plot

__all__ = ["rd2d", "rd2d_bw", "rd2d_plot"]

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

CausalResult._CITATIONS["rd2d"] = (
    "@article{cattaneo2025boundary,\n"
    "  title={rd2d: Boundary Regression Discontinuity Designs},\n"
    "  author={Cattaneo, Matias D. and Titiunik, Rocio and Yu, Ruiqi Rae},\n"
    "  journal={CRAN: Contributed Packages},\n"
    "  year={2025},\n"
    "  doi={10.32614/cran.package.rd2d}\n"
    "}"
)


# ======================================================================
# Validation
# ======================================================================


def _bad(msg: str, hint: str, **diag: Any) -> MethodIncompatibility:
    return MethodIncompatibility(msg, recovery_hint=hint, diagnostics=diag)


def _require_frame(data: Any, cols: Sequence[Tuple[str, Any]]) -> None:
    if not isinstance(data, pd.DataFrame):
        raise _bad(
            "`data` must be a pandas DataFrame.",
            "Pass a DataFrame containing y, x1, x2, and treatment.",
            type=type(data).__name__,
        )
    for label, name in cols:
        if not isinstance(name, str) or not name:
            raise _bad(
                f"`{label}` must be a non-empty column-name string.",
                f"Pass an existing DataFrame column name for `{label}`.",
                argument=label,
                value=repr(name),
            )
    missing = [name for _, name in cols if name not in data.columns]
    if missing:
        raise _bad(
            f"Column '{missing[0]}' not found in data",
            "Check y/x1/x2/treatment names against data.columns.",
            missing_columns=missing,
            available_columns=list(data.columns),
        )


def _choice(value: Any, allowed: Sequence[str], name: str) -> str:
    if value not in allowed:
        raise _bad(
            f"{name} must be one of {', '.join(repr(a) for a in allowed)}; got "
            f"{value!r}",
            f"Use {name}=" + " or ".join(repr(a) for a in allowed) + ".",
            **{name: value},
        )
    return value


def _kernel(kernel: Any) -> str:
    if kernel not in _KERNEL_CANON:
        raise _bad(
            "kernel must be 'triangular', 'epanechnikov', 'uniform' or 'gaussian' "
            f"(or tri/epa/uni/gau); got {kernel!r}",
            "Use kernel='triangular' (default).",
            kernel=kernel,
        )
    return _KERNEL_CANON[kernel]


def _nonneg_int(v: Any, name: str) -> int:
    if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 0:
        raise _bad(
            f"{name} must be a non-negative integer",
            f"Use {name}=1 (local linear) or another non-negative integer.",
            **{name: v},
        )
    return int(v)


def _numeric(
    data: pd.DataFrame, cols: Dict[str, Optional[str]]
) -> Dict[str, np.ndarray]:
    out = {}
    for k, c in cols.items():
        if c is None:
            continue
        try:
            out[k] = data[c].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise DataInsufficient(
                "rd2d columns must be numeric",
                recovery_hint=(
                    "Convert y, x1, x2, treatment (and fuzzy) to numeric values."
                ),
                diagnostics={"column": c},
            ) from exc
    return out


def _eval_points(eval_points, X1, X2, boundary, n_eval) -> np.ndarray:
    if eval_points is not None:
        b = np.asarray(eval_points, dtype=float)
        if b.ndim == 1 and b.size == 2:
            b = b.reshape(1, 2)
        if b.ndim != 2 or b.shape[1] != 2 or not np.isfinite(b).all():
            raise _bad(
                "eval_points must be a finite array with shape (k, 2)",
                "Pass boundary points as [[x1, x2], ...] (R's `b`).",
                shape=tuple(np.shape(b)),
            )
        return b
    if boundary is None:
        if n_eval == 1:
            return np.array([[0.0, float(np.median(X2))]])
        grid = np.linspace(np.percentile(X2, 10), np.percentile(X2, 90), n_eval)
        return np.column_stack([np.zeros(n_eval), grid])
    lo, hi = np.percentile(X1, 10), np.percentile(X1, 90)
    grid = np.array([(lo + hi) / 2]) if n_eval == 1 else np.linspace(lo, hi, n_eval)
    return np.column_stack([grid, [float(boundary(v)) for v in grid]])


def _bwcheck_value(bwcheck: Any, default: int) -> Optional[int]:
    if isinstance(bwcheck, str):
        if bwcheck != "auto":
            raise _bad(
                "bwcheck must be 'auto', None or a positive integer",
                "Use bwcheck='auto' (R's default).",
                bwcheck=bwcheck,
            )
        return default
    if bwcheck is None:
        return None
    if (
        not isinstance(bwcheck, (int, np.integer))
        or isinstance(bwcheck, bool)
        or bwcheck < 1
    ):
        raise _bad(
            "bwcheck must be 'auto', None or a positive integer",
            "Use bwcheck='auto' (R's default).",
            bwcheck=bwcheck,
        )
    return int(bwcheck)


def _kink_position(kink_position, neval: int) -> np.ndarray:
    if kink_position is None:
        return np.zeros(neval, dtype=bool)
    k = np.asarray(kink_position)
    if k.dtype == bool:
        if k.size != neval:
            raise _bad(
                "kink_position must have one True/False value per boundary point.",
                "Pass a boolean vector of length len(eval_points).",
            )
        return k.astype(bool)
    out = np.zeros(neval, dtype=bool)
    idx = k.astype(int)
    if np.any(idx < 0) or np.any(idx >= neval):
        raise _bad(
            "kink_position indices must index eval_points (0-based).",
            "Pass 0-based integer indices into eval_points.",
        )
    out[idx] = True
    return out


def _kink_unknown(kink_unknown) -> Tuple[bool, bool]:
    if isinstance(kink_unknown, (bool, np.bool_)):
        return bool(kink_unknown), bool(kink_unknown)
    k = tuple(bool(v) for v in kink_unknown)
    if len(k) != 2:
        raise _bad(
            "kink_unknown must be a bool or a pair of bools.",
            "Use kink_unknown=True or (True, False).",
        )
    if not k[0] and k[1]:
        raise _bad(
            "kink_unknown[1] can be True only when kink_unknown[0] is True.",
            "Use kink_unknown=(True, True) or (True, False).",
        )
    return k


def _distance_matrix(X1, X2, T, b, distance, data, valid) -> np.ndarray:
    """Signed distances: user-supplied, or Euclidean to each boundary point."""
    if distance is not None:
        if isinstance(distance, (list, tuple)) and all(
            isinstance(c, str) for c in distance
        ):
            D = data[list(distance)].to_numpy(dtype=float)[valid]
        else:
            D = np.asarray(distance, dtype=float)
            if D.ndim == 1:
                D = D[:, None]
            D = D[valid]
        if D.shape[1] != len(b):
            raise _bad(
                "distance must have one column per evaluation point.",
                "Pass an (n, len(eval_points)) matrix of signed distances.",
                shape=D.shape,
                neval=len(b),
            )
        return D
    sign = 2.0 * T - 1.0
    return np.column_stack(
        [
            np.sqrt((X1 - b[j, 0]) ** 2 + (X2 - b[j, 1]) ** 2) * sign
            for j in range(len(b))
        ]
    )


# ======================================================================
# Pooled (one-score) approach
# ======================================================================


def _signed_boundary_distance(X1, X2, T, boundary) -> np.ndarray:
    """Distance to the boundary curve, positive for treated units."""
    if boundary is None:
        dist = np.abs(X1)
    else:
        lo, hi = float(X1.min()), float(X1.max())
        pad = 0.1 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 4001)
        by = np.array([float(boundary(v)) for v in grid])
        if not np.all(np.isfinite(by)):
            raise _bad(
                "boundary(x1) returned non-finite values on the data range.",
                "Pass a boundary function defined on the range of x1.",
            )
        dist = np.empty(len(X1))
        step = grid[1] - grid[0]
        for i in range(len(X1)):
            d2 = (grid - X1[i]) ** 2 + (by - X2[i]) ** 2
            j = int(np.argmin(d2))
            # refine on the bracketing segment by golden-section search
            a, c = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
            f = (
                lambda t: (t - X1[i]) ** 2 + (float(boundary(t)) - X2[i]) ** 2
            )  # noqa: E731
            gr = (math.sqrt(5) - 1) / 2
            for _ in range(60):
                x1_, x2_ = c - gr * (c - a), a + gr * (c - a)
                if f(x1_) < f(x2_):
                    c = x2_
                else:
                    a = x1_
                if c - a < 1e-12 * max(1.0, step):
                    break
            dist[i] = math.sqrt(min(d2[j], f((a + c) / 2)))
    return dist * np.where(T == 1, 1.0, -1.0)


def _rd2d_pooled(
    data, arrs, valid, boundary, p, kernel, h, bwselect, alpha, fuzzy, cluster
):
    from .rdrobust import rdrobust

    X1, X2, T, Y = arrs["x1"], arrs["x2"], arrs["t"], arrs["y"]
    dist = _signed_boundary_distance(X1, X2, T, boundary)
    frame = pd.DataFrame({"y": Y, "dist": dist})
    if fuzzy is not None:
        frame["fz"] = arrs["fuzzy"]
    if cluster is not None:
        frame["cl"] = data[cluster].to_numpy()[valid]
    res = rdrobust(
        frame,
        y="y",
        x="dist",
        c=0.0,
        fuzzy="fz" if fuzzy is not None else None,
        p=p,
        kernel=kernel,
        bwselect=bwselect,
        h=h,
        alpha=alpha,
        cluster="cl" if cluster is not None else None,
        manipulation_test=False,
    )
    res.method = "2D Boundary RD (pooled distance to boundary, rdrobust)"
    res.model_info = dict(res.model_info or {})
    res.model_info.update(
        {"approach": "pooled", "boundary": "x1=0" if boundary is None else "custom"}
    )
    return res


# ======================================================================
# Public API
# ======================================================================


@accepts_aliases(treat="treatment")
def rd2d(
    data: pd.DataFrame,
    y: str,
    x1: str,
    x2: str,
    treatment: str,
    boundary: Optional[Callable] = None,
    approach: str = "location",
    p: int = 1,
    kernel: str = "triangular",
    h: Optional[Union[float, np.ndarray]] = None,
    bwselect: str = "mserd",
    eval_points: Optional[np.ndarray] = None,
    n_eval: int = 1,
    alpha: float = 0.05,
    *,
    q: Optional[int] = None,
    deriv: Tuple[int, int] = (0, 0),
    tangvec: Optional[np.ndarray] = None,
    kernel_type: str = "prod",
    vce: str = "hc1",
    cluster: Optional[str] = None,
    fuzzy: Optional[str] = None,
    fitmethod: str = "joint",
    bwparam: str = "main",
    method: str = "dpi",
    masspoints: str = "check",
    bwcheck: Union[int, str, None] = "auto",
    scaleregul: Optional[float] = None,
    scalebiascrct: float = 1.0,
    stdvars: bool = True,
    kink_unknown: Union[bool, Tuple[bool, bool]] = False,
    kink_position: Optional[Sequence] = None,
    cqt: float = 0.5,
    distance: Optional[Union[np.ndarray, List[str]]] = None,
    side: str = "two",
    weights: Optional[Sequence[float]] = None,
) -> CausalResult:
    """
    Boundary discontinuity design: pointwise effects along a 2-D boundary.

    Ports R ``rd2d::rd2d`` (``approach="location"``) and
    ``rd2d::rd2d.distance`` (``approach="distance"``) [@cattaneo2025boundary].
    Estimates ``tau(b)`` at every row ``b`` of ``eval_points``.  Following
    the reference, inference is robust bias-corrected: the point estimate
    ``estimate_q`` and standard error ``std_err_q`` come from the order
    ``q = p + 1`` fit, and the interval and p-value are built from them;
    the order-``p`` estimate is reported as ``estimate_p``.

    The headline ``estimate`` is ``estimate_q`` at the single evaluation
    point, or -- when several are given -- R's weighted boundary average
    treatment effect (``summary(fit, WBATE = weights)``): the weighted
    average of the pointwise ``estimate_q`` with the standard error implied
    by their full covariance matrix (R's ``params.cov = "main"``).  Equal
    weights by default; the conventional ``sum(w * estimate_p)`` and the
    full WBATE row are in ``model_info["wbate"]``, the per-point table in
    ``result.detail``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome.
    x1, x2 : str
        The two running variables.
    treatment : str
        0/1 assignment (R's ``assignment``); defines the two sides of the
        boundary.
    boundary : callable, optional
        ``f(x1) -> x2`` describing the boundary; used only to place default
        evaluation points (``n_eval`` of them) and by ``approach="pooled"``.
        ``None`` means the line ``x1 = 0``.
    approach : {'location', 'distance', 'pooled'}, default 'location'
        See the module docstring.
    p : int, default 1
        Polynomial order of the point estimator.
    kernel : {'triangular', 'epanechnikov', 'uniform', 'gaussian'}
    h : float or array, optional
        Bandwidth. ``location``: scalar or ``(k, 4)`` array
        ``(h01, h02, h11, h12)``; ``distance``: scalar or ``(k, 2)`` array
        ``(h0, h1)``.  ``None`` selects it (R ``rdbw2d`` /
        ``rdbw2d.distance``).
    bwselect : str, default 'mserd'
        ``mserd``, ``cerrd``, ``imserd``, ``icerrd``, ``msetwo``,
        ``certwo``, ``imsetwo`` or ``icertwo``.
    eval_points : array (k, 2), optional
        Boundary points (R's ``b``).  Default: ``n_eval`` points along
        ``boundary`` between the 10th and 90th percentiles of the data.
    n_eval : int, default 1
    alpha : float, default 0.05
        ``1 - alpha`` is R's ``level / 100``.
    q : int, optional
        Order of the bias-correction fit; default ``p + 1`` (``p`` for the
        distance approach with ``kink_unknown``).
    deriv : (int, int), default (0, 0)
        Location approach: partial derivative of the effect to estimate.
    tangvec : array (k, 2), optional
        Location approach: tangential direction for a directional
        derivative (overrides ``deriv``).
    kernel_type : {'prod', 'rad'}, default 'prod'
        Location approach: product or radial kernel.
    vce : {'hc1', 'hc0', 'hc2', 'hc3'}, default 'hc1'
    cluster : str, optional
        Cluster identifier (vce must then be hc0 or hc1).
    fuzzy : str, optional
        Treatment take-up for a fuzzy design; the effect is the ratio of
        the outcome and take-up jumps.
    fitmethod : {'joint', 'separate'}, default 'joint'
        ``joint`` applies the degrees-of-freedom correction to both sides
        together (R's default).
    bwparam : {'main', 'itt'}, default 'main'
        Fuzzy designs: select the bandwidth for the ratio or the ITT.
    method : {'dpi', 'rot'}, default 'dpi'
        Location approach: pilot bandwidth rule for the bias constants.
    masspoints : {'check', 'adjust', 'off'}, default 'check'
    bwcheck : int, None or 'auto'
        Minimum number of observations (or mass points) inside every
        bandwidth; ``'auto'`` is R's default ``50 + p + 1``.
    scaleregul : float, optional
        Regularisation weight of the bandwidth selector; default is R's
        (3 for location, 1 for distance).
    scalebiascrct : float, default 1
        Location approach: weight of the higher-order bias correction in
        the bandwidth selector.
    stdvars : bool, default True
        Location approach: standardise x1, x2 before selecting bandwidths.
    kink_unknown : bool or (bool, bool), default False
        Distance approach: boundary may have kinks at unknown places;
        undersmooth the bandwidth (first) and the bias-correction bandwidth
        (second).
    kink_position : sequence, optional
        Distance approach: 0-based indices (or a boolean mask) of
        evaluation points that are known kinks of the boundary.
    cqt : float, default 0.5
        Distance approach: quantile of the distance below which the pilot
        polynomial for the bias constant is fitted.
    distance : array (n, k) or list of column names, optional
        Distance approach: signed distances (non-negative on the treated
        side) to each evaluation point.  Default: Euclidean distance to
        ``eval_points`` signed by ``treatment``.
    side : {'two', 'left', 'right'}, default 'two'
        Two-sided or one-sided intervals.
    weights : sequence of float, optional
        WBATE weights over the evaluation points (R ``summary(..., WBATE=)``);
        normalised to sum to one.  Default: equal weights.

    Returns
    -------
    CausalResult
        ``detail`` holds one row per evaluation point with R's ``main``
        columns (``b1, b2, estimate_p, std_err_p, estimate_q, std_err_q,
        t_value, p_value, ci_lower, ci_upper``, bandwidths, effective
        sample sizes).  ``model_info`` holds the covariance matrices of the
        ``p`` and ``q`` estimates across points (``cov_p``, ``cov_q``), the
        side-specific intercepts and, for fuzzy designs, the ITT and first
        stage.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 3000
    >>> x1 = rng.uniform(-1, 1, n)
    >>> x2 = rng.uniform(-1, 1, n)
    >>> treat = (x1 >= 0).astype(int)
    >>> y = 1.5 * treat + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "treat": treat})
    >>> b = [[0.0, -0.5], [0.0, 0.0], [0.0, 0.5]]
    >>> res = sp.rd2d(df, y="y", x1="x1", x2="x2", treatment="treat",
    ...               eval_points=b)
    >>> res.detail[["b2", "estimate_q", "std_err_q"]].shape
    (3, 3)

    References
    ----------
    [@cattaneo2025boundary] (R package ``rd2d``); [@keele2015geographic]
    for ``approach="pooled"``.
    """
    approach = _choice(approach, _APPROACHES, "approach")
    cols = [("y", y), ("x1", x1), ("x2", x2), ("treatment", treatment)]
    if fuzzy is not None:
        cols.append(("fuzzy", fuzzy))
    if cluster is not None:
        cols.append(("cluster", cluster))
    _require_frame(data, cols)
    kernel = _kernel(kernel)
    p = _nonneg_int(p, "p")
    if q is not None:
        q = _nonneg_int(q, "q")
        if q < p:
            raise _bad(
                "q must be no smaller than p", "Use q=None (p + 1) or q >= p.", p=p, q=q
            )
    if h is not None:
        harr = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(harr)) or np.any(harr <= 0):
            raise _bad(
                "h must be positive and finite",
                "Pass a positive bandwidth or h=None.",
                h=repr(h),
            )
    if not (isinstance(alpha, (float, int)) and 0 < alpha < 1):
        raise _bad(
            "alpha must be between 0 and 1",
            "Pass a significance level such as alpha=0.05.",
            alpha=alpha,
        )
    if boundary is not None and not callable(boundary):
        raise _bad(
            "boundary must be callable or None",
            "Pass boundary=lambda x1: f(x1), or leave boundary=None.",
            boundary_type=type(boundary).__name__,
        )
    if (
        not isinstance(n_eval, (int, np.integer))
        or isinstance(n_eval, bool)
        or n_eval < 1
    ):
        raise _bad(
            "n_eval must be a positive integer",
            "Pass n_eval=1 or a larger integer.",
            n_eval=n_eval,
        )
    kernel_type = _choice(kernel_type, _KERNEL_TYPES, "kernel_type")
    vce = _choice(vce, _VCES, "vce")
    bwselect = _choice(bwselect, _BWSELECTS, "bwselect")
    masspoints = _choice(masspoints, _MASSPOINTS, "masspoints")
    fitmethod = _choice(fitmethod, _FITMETHODS, "fitmethod")
    method = _choice(method, ("dpi", "rot"), "method")
    bwparam = _choice(bwparam, ("main", "itt"), "bwparam")
    side = _choice(side, ("two", "left", "right"), "side")
    if cluster is not None and vce not in ("hc0", "hc1"):
        warnings.warn(
            "When cluster is specified, vce must be 'hc0' or 'hc1'. Resetting vce "
            "to 'hc1'.",
            RuntimeWarning,
            stacklevel=2,
        )
        vce = "hc1"

    arrs = _numeric(data, {"y": y, "x1": x1, "x2": x2, "t": treatment, "fuzzy": fuzzy})
    valid = (
        np.isfinite(arrs["y"])
        & np.isfinite(arrs["x1"])
        & np.isfinite(arrs["x2"])
        & np.isfinite(arrs["t"])
    )
    if fuzzy is not None:
        valid &= np.isfinite(arrs["fuzzy"])
    if cluster is not None:
        valid &= data[cluster].notna().to_numpy()
    arrs = {k: v[valid] for k, v in arrs.items()}
    n = int(valid.sum())
    if n < 20:
        raise DataInsufficient(
            f"Too few valid observations ({n}). Need at least 20.",
            recovery_hint="Provide at least 20 complete finite rows.",
            diagnostics={"n_valid": n, "min_required": 20},
        )
    T = arrs["t"]
    if not np.all(np.isin(T, (0.0, 1.0))):
        raise _bad("treatment must be a 0/1 indicator", "Recode the assignment as 0/1.")
    n1, n0 = int((T == 1).sum()), int((T == 0).sum())
    if n1 < 5 or n0 < 5:
        raise DataInsufficient(
            f"Too few treated ({n1}) or control ({n0}) units.",
            recovery_hint="Provide at least 5 treated and 5 control units.",
            diagnostics={"n_treated": n1, "n_control": n0},
        )
    cl = data[cluster].to_numpy()[valid] if cluster is not None else None

    if approach == "pooled":
        return _rd2d_pooled(
            data, arrs, valid, boundary, p, kernel, h, bwselect, alpha, fuzzy, cluster
        )

    b = _eval_points(eval_points, arrs["x1"], arrs["x2"], boundary, n_eval)
    neval = len(b)
    level_z = alpha
    if weights is not None:
        wv = np.asarray(weights, dtype=float)
        if wv.shape != (neval,) or not np.all(np.isfinite(wv)) or wv.sum() == 0:
            raise _bad(
                "weights must be finite, one per evaluation point, with a nonzero sum",
                "Pass weights of length len(eval_points), or None for equal weights.",
                n_weights=int(wv.size),
                neval=neval,
            )
    bwcheck_v = _bwcheck_value(bwcheck, 50 + p + 1)
    try:
        if approach == "location":
            if tangvec is not None:
                tangvec = np.asarray(tangvec, dtype=float).reshape(neval, 2)
                if p < 1:
                    raise _bad("tangvec requires p >= 1", "Use p=1 or larger.")
            dv = tuple(int(v) for v in deriv)
            if len(dv) != 2 or min(dv) < 0 or sum(dv) > p:
                raise _bad(
                    "deriv must be two non-negative integers summing to at most p",
                    "Use deriv=(0, 0) for the effect itself.",
                    deriv=deriv,
                )
            if h is not None:
                harr = np.asarray(h, dtype=float)
                if harr.size != 1 and harr.shape != (neval, 4):
                    raise _bad(
                        "h must be a scalar or an array of shape (k, 4)",
                        "Pass h=(h01, h02, h11, h12) per evaluation point.",
                        shape=harr.shape,
                    )
            Y = (
                arrs["y"]
                if fuzzy is None
                else np.column_stack([arrs["y"], arrs["fuzzy"]])
            )
            D = Rd2dData(arrs["x1"], arrs["x2"], T, Y, cl)
            res = rd2d_location_fit(
                D,
                b,
                h,
                p,
                p + 1 if q is None else q,
                dv,
                tangvec,
                kernel,
                kernel_type,
                vce,
                masspoints,
                bwcheck_v,
                fitmethod,
                bwselect,
                method,
                3.0 if scaleregul is None else float(scaleregul),
                float(scalebiascrct),
                bool(stdvars),
                bwparam,
            )
        else:
            Dm = _distance_matrix(arrs["x1"], arrs["x2"], T, b, distance, data, valid)
            if h is not None:
                harr = np.asarray(h, dtype=float)
                if harr.size != 1 and harr.shape != (neval, 2):
                    raise _bad(
                        "h must be a scalar or an array of shape (k, 2)",
                        "Pass h=(h0, h1) per evaluation point.",
                        shape=harr.shape,
                    )
            ku = _kink_unknown(kink_unknown)
            kp = _kink_position(kink_position, neval)
            if kp.any() and ku[0]:
                raise _bad(
                    "Use either kink_position or kink_unknown, not both.",
                    "Drop one of the two kink options.",
                )
            if h is not None and (kp.any() or ku[0]):
                raise _bad(
                    "kink options apply only to automatic bandwidth selection.",
                    "Omit h to use kink_position / kink_unknown.",
                )
            res = rd2d_distance_estimate(
                arrs["y"],
                Dm,
                b,
                h,
                p,
                q,
                ku,
                kp,
                kernel,
                bwselect,
                vce,
                bwcheck_v,
                masspoints,
                cl,
                fitmethod,
                1.0 if scaleregul is None else float(scaleregul),
                float(cqt),
                None if fuzzy is None else arrs["fuzzy"],
                bwparam,
            )
    except ValueError as exc:
        if "bwcheck" in str(exc):
            raise DataInsufficient(
                str(exc),
                recovery_hint="Decrease bwcheck or set bwcheck=None.",
                diagnostics={"bwcheck": bwcheck_v},
            ) from exc
        raise
    return _build_result(
        res,
        b,
        approach,
        p,
        kernel,
        kernel_type,
        vce,
        fitmethod,
        level_z,
        side,
        n,
        fuzzy is not None,
        cluster,
        masspoints,
        weights,
    )


def _inference(est: np.ndarray, se: np.ndarray, alpha: float, side: str):
    t = est / se
    pv = 2 * stats.norm.sf(np.abs(t))
    if side == "two":
        z = stats.norm.ppf(1 - alpha / 2)
        return t, pv, est - z * se, est + z * se
    z = stats.norm.ppf(1 - alpha)
    if side == "left":
        return t, pv, np.full_like(est, -np.inf), est + z * se
    return t, pv, est - z * se, np.full_like(est, np.inf)


def _build_result(
    res,
    b,
    approach,
    p,
    kernel,
    kernel_type,
    vce,
    fitmethod,
    alpha,
    side,
    n,
    is_fuzzy,
    cluster,
    masspoints,
    weights=None,
) -> CausalResult:
    est_p, se_p = res["tau_p"], res["se_p"]
    est_q, se_q = res["tau_q"], res["se_q"]
    t, pv, lo, hi = _inference(est_q, se_q, alpha, side)
    detail = pd.DataFrame(
        {
            "b1": b[:, 0],
            "b2": b[:, 1],
            "estimate_p": est_p,
            "std_err_p": se_p,
            "estimate_q": est_q,
            "std_err_q": se_q,
            "t_value": t,
            "p_value": pv,
            "ci_lower": lo,
            "ci_upper": hi,
        }
    )
    if approach == "location":
        fp = res["fit_p"]
        detail["h01"], detail["h02"] = res["hgrid0"][:, 0], res["hgrid0"][:, 1]
        detail["h11"], detail["h12"] = res["hgrid1"][:, 0], res["hgrid1"][:, 1]
        detail["n_co"], detail["n_tr"] = fp["eN0"].astype(int), fp["eN1"].astype(int)
        if kernel_type == "rad":
            # radius of the radial kernel actually used: sqrt(hx^2 + hy^2)
            detail["radius_co"] = np.hypot(fp["h0x"], fp["h0y"])
            detail["radius_tr"] = np.hypot(fp["h1x"], fp["h1y"])
    else:
        f = res["fit_p"][0]
        detail["h0"], detail["h1"] = res["hfull"][:, 0], res["hfull"][:, 1]
        detail["h0_rbc"], detail["h1_rbc"] = res["hrbc"][:, 0], res["hrbc"][:, 1]
        detail["n_co"], detail["n_tr"] = f["N0"].astype(int), f["N1"].astype(int)
    k = len(b)
    if weights is None:
        w = np.full(k, 1.0 / k)
    else:
        w = np.asarray(weights, dtype=float) / float(np.sum(weights))
    # R summary.rd2d(WBATE = w): conventional sum(w * estimate.p); inference
    # centred at sum(w * estimate.q) with se sqrt(w' V_q w).
    est = float(w @ est_q)
    se = float(math.sqrt(max(float(w @ res["cov_q"] @ w), 0.0)))
    tt, pp, l0, h0 = _inference(np.array([est]), np.array([se]), alpha, side)
    wbate = {
        "weights": w,
        "estimate_p": float(w @ est_p),
        "estimate_q": est,
        "std_err_q": se,
        "t_value": float(tt[0]),
        "p_value": float(pp[0]),
        "ci_lower": float(l0[0]),
        "ci_upper": float(h0[0]),
    }
    if k == 1:
        estimand = "Boundary effect tau(b) at the evaluation point"
    elif weights is None:
        estimand = "WBATE: equally weighted average of the pointwise boundary effects"
    else:
        estimand = "WBATE: weighted average of the pointwise boundary effects"
    info: Dict[str, Any] = {
        "approach": approach,
        "reference": "R rd2d 1.0.0 "
        + ("rd2d()" if approach == "location" else "rd2d.distance()"),
        "polynomial_p": p,
        "kernel": kernel,
        "vce": vce,
        "fitmethod": fitmethod,
        "masspoints": masspoints,
        "bwcheck": res["bwcheck"],
        "bwselect": res["bwselect"],
        "cluster": cluster,
        "fuzzy": is_fuzzy,
        "side": side,
        "n_eval_points": k,
        "eval_points": b.tolist(),
        "cov_p": res["cov_p"],
        "cov_q": res["cov_q"],
        "headline": "estimate_q" if k == 1 else "WBATE (bias-corrected centre)",
        "wbate": wbate,
    }
    if approach == "location":
        info["kernel_type"] = kernel_type
        info["deriv"] = res["deriv"]
        for tag in ("p", "q"):
            f = res[f"fit_{tag}"]
            # bandwidths the fits actually used (after the bwcheck clamp)
            info[f"h_used_{tag}"] = np.column_stack(
                [f["h0x"], f["h0y"], f["h1x"], f["h1y"]]
            )
    else:
        info["q"] = res["q"]
    if is_fuzzy:
        for tag in ("p", "q"):
            info[f"itt_{tag}"] = res[f"itt_{tag}"]
            info[f"fs_{tag}"] = res[f"fs_{tag}"]
            info[f"se_itt_{tag}"] = res[f"se_itt_{tag}"]
            info[f"se_fs_{tag}"] = res[f"se_fs_{tag}"]
    else:
        for tag in ("p", "q"):
            for part in ("mu0", "mu1", "se0", "se1"):
                info[f"{part}_{tag}"] = res[f"{part}_{tag}"]
    return CausalResult(
        method=f"2D Boundary RD ({approach}-based, rd2d)",
        estimand=estimand,
        estimate=est,
        se=se,
        pvalue=float(pp[0]),
        ci=(float(l0[0]), float(h0[0])),
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=info,
        _citation_key="rd2d",
    )


@accepts_aliases(treat="treatment")
def rd2d_bw(
    data: pd.DataFrame,
    y: str,
    x1: str,
    x2: str,
    treatment: str,
    boundary: Optional[Callable] = None,
    approach: str = "location",
    p: int = 1,
    kernel: str = "triangular",
    *,
    eval_points: Optional[np.ndarray] = None,
    n_eval: int = 1,
    deriv: Tuple[int, int] = (0, 0),
    tangvec: Optional[np.ndarray] = None,
    kernel_type: str = "prod",
    bwselect: str = "mserd",
    method: str = "dpi",
    vce: str = "hc1",
    cluster: Optional[str] = None,
    fuzzy: Optional[str] = None,
    bwparam: str = "main",
    fitmethod: str = "joint",
    masspoints: str = "check",
    bwcheck: Union[int, str, None] = "auto",
    scaleregul: float = 1.0,
    scalebiascrct: float = 1.0,
    stdvars: bool = True,
    kink_unknown: Union[bool, Tuple[bool, bool]] = False,
    kink_position: Optional[Sequence] = None,
    cqt: float = 0.5,
    distance: Optional[Union[np.ndarray, List[str]]] = None,
) -> pd.DataFrame:
    """
    Bandwidth selection for boundary discontinuity designs.

    Ports R ``rd2d::rdbw2d`` (``approach="location"``) and
    ``rd2d::rdbw2d.distance`` (``approach="distance"``)
    [@cattaneo2025boundary], with *their* defaults -- ``bwcheck = 20``
    (location) / ``20 + p + 1`` (distance) and ``scaleregul = 1`` -- which
    differ from the values :func:`sp.rd2d` passes when it selects the
    bandwidth itself (``50 + p + 1`` and 3), exactly as in R.

    .. versionchanged:: 1.29.0
       Returns a DataFrame of per-point bandwidths (R's ``$bws``) instead
       of one float; see MIGRATION.md.

    Parameters
    ----------
    data : pandas.DataFrame
        Long frame holding the outcome, both running variables and the
        treatment indicator.
    y, x1, x2, treatment : str
        Column names, as in :func:`sp.rd2d`.
    boundary : callable, optional
        ``f(x1) -> x2`` tracing the boundary; ``None`` uses ``x1 = 0``.
    approach : {'location', 'distance'}, default 'location'
        Bivariate fit at boundary points, or a univariate fit on the
        signed distance. ``sp.rd2d``'s ``'pooled'`` has no bandwidth
        selector of its own -- call :func:`sp.rdbwselect` on the signed
        distance instead.
    p : int, default 1
        Polynomial order of the point estimator.
    kernel : str, default 'triangular'
        ``'triangular'``, ``'uniform'`` or ``'epanechnikov'``.

    Returns
    -------
    pd.DataFrame
        ``b1, b2, h01, h02, h11, h12`` (location) or ``b1, b2, h0, h1``
        (distance), one row per evaluation point.

    Notes
    -----
    The remaining keyword arguments -- ``eval_points``, ``n_eval``,
    ``deriv``, ``tangvec``, ``kernel_type``, ``bwselect``, ``method``,
    ``vce``, ``cluster``, ``fuzzy``, ``bwparam``, ``fitmethod``,
    ``masspoints``, ``bwcheck``, ``scaleregul``, ``scalebiascrct``,
    ``stdvars``, ``kink_unknown``, ``kink_position``, ``cqt`` and
    ``distance`` -- carry the meaning and the defaults they have in
    ``rd2d::rdbw2d``; :func:`sp.rd2d` documents the ones the two
    functions share.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> x1 = rng.uniform(-1, 1, n)
    >>> x2 = rng.uniform(-1, 1, n)
    >>> treat = (x1 >= 0).astype(int)
    >>> y = 1.5 * treat + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "treat": treat})
    >>> bw = sp.rd2d_bw(df, y="y", x1="x1", x2="x2", treatment="treat")
    >>> list(bw.columns)
    ['b1', 'b2', 'h01', 'h02', 'h11', 'h12']
    """
    approach = _choice(approach, ("location", "distance"), "approach")
    cols = [("y", y), ("x1", x1), ("x2", x2), ("treatment", treatment)]
    if fuzzy is not None:
        cols.append(("fuzzy", fuzzy))
    if cluster is not None:
        cols.append(("cluster", cluster))
    _require_frame(data, cols)
    kernel = _kernel(kernel)
    p = _nonneg_int(p, "p")
    if boundary is not None and not callable(boundary):
        raise _bad(
            "boundary must be callable or None",
            "Pass boundary=lambda x1: f(x1), or leave boundary=None.",
            boundary_type=type(boundary).__name__,
        )
    kernel_type = _choice(kernel_type, _KERNEL_TYPES, "kernel_type")
    vce = _choice(vce, _VCES, "vce")
    bwselect = _choice(bwselect, _BWSELECTS, "bwselect")
    masspoints = _choice(masspoints, _MASSPOINTS, "masspoints")
    fitmethod = _choice(fitmethod, _FITMETHODS, "fitmethod")
    method = _choice(method, ("dpi", "rot"), "method")
    bwparam = _choice(bwparam, ("main", "itt"), "bwparam")
    if cluster is not None and vce not in ("hc0", "hc1"):
        warnings.warn(
            "When cluster is specified, vce must be 'hc0' or 'hc1'. Resetting vce "
            "to 'hc1'.",
            RuntimeWarning,
            stacklevel=2,
        )
        vce = "hc1"
    arrs = _numeric(data, {"y": y, "x1": x1, "x2": x2, "t": treatment, "fuzzy": fuzzy})
    valid = (
        np.isfinite(arrs["y"])
        & np.isfinite(arrs["x1"])
        & np.isfinite(arrs["x2"])
        & np.isfinite(arrs["t"])
    )
    if fuzzy is not None:
        valid &= np.isfinite(arrs["fuzzy"])
    if cluster is not None:
        valid &= data[cluster].notna().to_numpy()
    arrs = {k: v[valid] for k, v in arrs.items()}
    T = arrs["t"]
    if not np.all(np.isin(T, (0.0, 1.0))):
        raise _bad("treatment must be a 0/1 indicator", "Recode the assignment as 0/1.")
    cl = data[cluster].to_numpy()[valid] if cluster is not None else None
    b = _eval_points(eval_points, arrs["x1"], arrs["x2"], boundary, n_eval)
    try:
        if approach == "location":
            if tangvec is not None:
                tangvec = np.asarray(tangvec, dtype=float).reshape(len(b), 2)
            Y = (
                arrs["y"]
                if fuzzy is None
                else np.column_stack([arrs["y"], arrs["fuzzy"]])
            )
            D = Rd2dData(arrs["x1"], arrs["x2"], T, Y, cl)
            out = rd2d_location_bw(
                D,
                b,
                p,
                tuple(int(v) for v in deriv),
                tangvec,
                kernel,
                kernel_type,
                bwselect,
                method,
                vce,
                _bwcheck_value(bwcheck, 20),
                masspoints,
                fitmethod,
                float(scaleregul),
                float(scalebiascrct),
                bool(stdvars),
                bwparam,
            )
            h = out["bws"]
            return pd.DataFrame(
                {
                    "b1": b[:, 0],
                    "b2": b[:, 1],
                    "h01": h[:, 0],
                    "h02": h[:, 1],
                    "h11": h[:, 2],
                    "h12": h[:, 3],
                }
            )
        Dm = _distance_matrix(arrs["x1"], arrs["x2"], T, b, distance, data, valid)
        ku = _kink_unknown(kink_unknown)
        kp = _kink_position(kink_position, len(b))
        if kp.any() and ku[0]:
            raise _bad(
                "Use either kink_position or kink_unknown, not both.",
                "Drop one of the two kink options.",
            )
        out = rd2d_distance_bw(
            arrs["y"],
            Dm,
            b,
            p,
            ku,
            kp,
            kernel,
            bwselect,
            vce,
            _bwcheck_value(bwcheck, 20 + p + 1),
            masspoints,
            cl,
            float(scaleregul),
            float(cqt),
            fitmethod,
            None if fuzzy is None else arrs["fuzzy"],
            bwparam,
        )
    except ValueError as exc:
        if "bwcheck" in str(exc):
            raise DataInsufficient(
                str(exc),
                recovery_hint="Decrease bwcheck or set bwcheck=None.",
                diagnostics={"bwcheck": bwcheck},
            ) from exc
        raise
    return pd.DataFrame(
        {"b1": b[:, 0], "b2": b[:, 1], "h0": out["h0"], "h1": out["h1"]}
    )
