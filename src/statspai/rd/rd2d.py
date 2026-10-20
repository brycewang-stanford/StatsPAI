"""Boundary discontinuity designs with a bivariate score.

Treatment is assigned on one side of a boundary curve in the plane of two
running variables ``(x1, x2)``; effects are estimated pointwise at
evaluation points ``b`` on the boundary. Clean-room implementation
(2026-10-21) of Cattaneo, Titiunik and Yu (location-based methods, arXiv
2505.05670; distance-based methods, arXiv 2510.26051; software article
arXiv 2505.07989; package ``cattaneo2025boundary``). R ``rd2d`` 1.0.0
(GPL-3) was used only as a black box; the conventions it fixed are recorded
in ``docs/dev/cleanroom/rd2d.md`` and pinned by
``tests/reference_parity/test_rd_open_R_parity.py``.

Three approaches:

- ``"location"``: a bivariate local polynomial in ``x - b`` on each side;
- ``"distance"``: a univariate local polynomial in the signed distance to
  ``b``, with unknown or known boundary kinks handled by bandwidth rates;
- ``"pooled"``: ``sp.rdrobust`` on the signed distance to the boundary
  curve, one pooled effect (not a method of the papers; kept for
  continuity).

The headline of a multi-point call is the equally weighted (or
``weights=``-weighted) average of the bias-corrected pointwise effects with
the standard error from their covariance.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..core._call_record import records_call
from ..core.results import CausalResult
from ._rd2d_distance import distance_assemble, distance_bandwidths, distance_estimates
from ._rd2d_inputs import (  # noqa: F401  (re-exported: tests reach them here)
    _APPROACHES,
    _BWSELECTS,
    _FITMETHODS,
    _KERNEL_TYPES,
    _MASSPOINTS,
    _VCES,
    _bad,
    _bwcheck_value,
    _check_bwcheck_size,
    _choice,
    _cluster_vce,
    _common_options,
    _deriv_options,
    _distance_h,
    _distance_matrix,
    _eval_points,
    _inference,
    _kernel,
    _kink_position,
    _kink_unknown,
    _location_h,
    _mass_points,
    _nonneg_int,
    _prepare,
    _require_frame,
    _signed_boundary_distance,
    _wbate_weights,
    _with_first_stage_guard,
)
from ._rd2d_location import (
    BoundaryData,
    assemble,
    location_bandwidths,
    location_estimates,
)
from ._rd2d_plot import rd2d_plot

__all__ = ["rd2d", "rd2d_bw", "rd2d_plot"]

CausalResult._CITATIONS["rd2d"] = (
    "@article{cattaneo2025boundary,\n"
    "  title={rd2d: Boundary Regression Discontinuity Designs},\n"
    "  author={Cattaneo, Matias D. and Titiunik, Rocio and Yu, Ruiqi Rae},\n"
    "  journal={CRAN: Contributed Packages},\n"
    "  year={2025},\n"
    "  doi={10.32614/cran.package.rd2d}\n"
    "}"
)


# --------------------------------------------------------------------------
# rd2d
# --------------------------------------------------------------------------


@records_call("rd2d")
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
    """Boundary discontinuity design: pointwise effects along a 2-D boundary.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome column.
    x1, x2 : str
        The two running variables.
    treatment : str
        0/1 column: the side of the boundary each unit is on.
    boundary : callable, optional
        Boundary curve ``x2 = f(x1)``, used for default evaluation points
        and by ``approach='pooled'``; default the line ``x1 = 0``.
    approach : {'location', 'distance', 'pooled'}, default 'location'
        Bivariate local polynomial in ``x - b``; univariate in the signed
        distance to ``b``; or ``sp.rdrobust`` on the signed distance to the
        boundary curve (one pooled effect).
    p : int, default 1
        Polynomial order of the point estimate.
    kernel : {'triangular', 'epanechnikov', 'uniform', 'gaussian'}
    h : float or array, optional
        Bandwidth: a scalar, or per point ``(k, 4)`` (location: control x1,
        control x2, treated x1, treated x2) or ``(k, 2)`` (distance:
        control, treated). Default: selected by ``bwselect``.
    bwselect : str, default 'mserd'
        ``mserd``/``cerrd`` (common), ``msetwo``/``certwo`` (per side),
        ``i``-prefixed forms integrate over the evaluation points.
    eval_points : array (k, 2), optional
        Boundary points; default on the boundary (see ``n_eval``).
    n_eval : int, default 1
        Number of default evaluation points.
    alpha : float, default 0.05
    q : int, optional
        Order of the bias-correction fit (default ``p + 1``; ``p`` with an
        unknown kink in the distance approach).
    deriv : (int, int), default (0, 0)
        Partial derivative order of the target (location approach).
    tangvec : array (k, 2), optional
        Direction of a directional derivative at each point (location).
    kernel_type : {'prod', 'rad'}, default 'prod'
        Product or radial kernel (location approach).
    vce : {'hc0', 'hc1', 'hc2', 'hc3'}, default 'hc1'
        Variance estimator; with ``cluster`` only hc0/hc1 (hc2/hc3 are reset
        to hc1 with a warning).
    cluster : str, optional
        Cluster column.
    fuzzy : str, optional
        Treatment-receipt column (fuzzy design: effect = ITT / first stage).
    fitmethod : {'joint', 'separate'}, default 'joint'
        Treatment-interacted fit (joint small-sample factors and cluster
        sums) or side-by-side fits.
    bwparam : {'main', 'itt'}, default 'main'
        Fuzzy bandwidth target: the Wald ratio or the reduced form.
    method : {'dpi', 'rot'}, default 'dpi'
        Bias-bandwidth rule of the location selector.
    masspoints : {'check', 'adjust', 'off'}, default 'check'
    bwcheck : int, 'auto' or None, default 'auto'
        Minimum number of observations (distinct locations with
        ``masspoints='adjust'``) in every window; 'auto' is ``50 + p + 1``.
    scaleregul : float, optional
        Weight of the regularisation term of the selector (default 3 for
        the location approach, 1 for the distance approach).
    scalebiascrct : float, default 1.0
        Weight of the higher-order bias term of the location selector.
    stdvars : bool, default True
        Standardise the scores before selecting bandwidths (location).
    kink_unknown : bool or (bool, bool), default False
        Distance approach: shrink the bandwidth to the kink rate
        ``n^(-1/4)`` (first) and the inference bandwidth to ``n^(-1/3)``
        (second).
    kink_position : sequence, optional
        Distance approach: 0-based indices (or a mask) of kink points.
    cqt : float, default 0.5
        Distance approach: quantile of ``|D|`` defining the subsample of the
        bias polynomial.
    distance : array (n, k) or list of str, optional
        Distance approach: user signed distances (negative on the control
        side), one column per evaluation point.
    side : {'two', 'left', 'right'}, default 'two'
        Two-sided or one-sided intervals.
    weights : sequence, optional
        Weights of the headline average over evaluation points.

    Returns
    -------
    CausalResult
        ``detail``: one row per evaluation point (``estimate_p``,
        ``std_err_p``, ``estimate_q``, ``std_err_q``, ``t_value``,
        ``p_value``, ``ci_lower``, ``ci_upper``, bandwidths, ``n_co``,
        ``n_tr``); headline: the weighted average of ``estimate_q``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x1, x2 = rng.uniform(-1, 1, 2000), rng.uniform(-1, 1, 2000)
    >>> t = (x1 >= 0).astype(int)
    >>> y = 0.5 * x1 + 0.3 * x2 + 2.0 * t + rng.normal(0, 0.3, 2000)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "t": t})
    >>> r = sp.rd2d(df, y="y", x1="x1", x2="x2", treatment="t", n_eval=3)
    >>> len(r.detail)
    3

    References
    ----------
    cattaneo2025boundary
    """
    _require_frame(data, [("y", y), ("x1", x1), ("x2", x2), ("treatment", treatment)])
    approach = _choice(approach, _APPROACHES, "approach")
    kernel, kernel_type, vce, bwselect, method, masspoints, fitmethod, bwparam, p = (
        _common_options(
            kernel,
            kernel_type,
            vce,
            bwselect,
            method,
            masspoints,
            fitmethod,
            bwparam,
            p,
            cqt,
            scaleregul,
        )
    )
    if not (isinstance(alpha, (int, float)) and 0.0 < alpha < 1.0):
        raise _bad("alpha must be between 0 and 1", "Use alpha=0.05.")
    if boundary is not None and not callable(boundary):
        raise _bad("boundary must be callable or None", "Pass a function x2 = f(x1).")
    if (
        isinstance(n_eval, bool)
        or not isinstance(n_eval, (int, np.integer))
        or n_eval < 1
    ):
        raise _bad("n_eval must be a positive integer", "Use n_eval=1 or more.")
    side = _choice(side, ("two", "left", "right"), "side")
    if (
        h is not None
        and np.ndim(h) == 0
        and not (np.isfinite(float(h)) and float(h) > 0)
    ):
        raise _bad("h must be positive and finite", "Pass a positive bandwidth.")
    vce = _cluster_vce(vce, cluster)
    arrs, valid = _prepare(data, y, x1, x2, treatment, cluster, fuzzy, "sp.rd2d")
    if approach == "pooled":
        return _rd2d_pooled(
            data, arrs, valid, boundary, p, kernel, h, bwselect, alpha, fuzzy, cluster
        )
    b = _eval_points(eval_points, arrs["x1"], arrs["x2"], boundary, n_eval)
    J = b.shape[0]
    bwcheck = _bwcheck_value(bwcheck, 50 + p + 1)
    if q is None:
        q_val = None
    else:
        q_val = _nonneg_int(q, "q")
        if q_val < p:
            raise _bad("q must be no smaller than p", "Use q = p + 1 (default).")
    deriv_t, tv = _deriv_options(deriv, tangvec, p, J)
    w = _wbate_weights(weights, J)
    if scaleregul is not None:
        sreg = float(scaleregul)
    else:
        sreg = 3.0 if approach == "location" else 1.0
    _check_bwcheck_size(bwcheck, arrs["t"])
    bwcheck = _mass_points(arrs, masspoints, bwcheck, p)
    Y = arrs["y"] if fuzzy is None else np.column_stack([arrs["y"], arrs["fuzzy"]])
    cl = arrs.get("cluster")
    if approach == "location":
        q_val = p + 1 if q_val is None else q_val
        H = _location_h(h, J)
        data2 = BoundaryData(arrs["x1"], arrs["x2"], arrs["t"], Y, cl)
        bws = "user provided"
        if H is None:
            sel = _with_first_stage_guard(
                lambda bp: location_bandwidths(
                    data2,
                    b,
                    p,
                    deriv_t,
                    tv,
                    kernel,
                    kernel_type,
                    bwselect,
                    method,
                    vce,
                    bwcheck,
                    masspoints,
                    fitmethod,
                    sreg,
                    scalebiascrct,
                    stdvars,
                    bp,
                ),
                bwparam,
                fuzzy is not None,
            )
            H = sel["h"]
            bws = bwselect
        scale = (
            tuple(np.std(np.column_stack([arrs["x1"], arrs["x2"]]), axis=0, ddof=1))
            if stdvars
            else (1.0, 1.0)
        )
        raw = location_estimates(
            data2,
            b,
            H,
            p,
            q_val,
            deriv_t,
            tv,
            kernel,
            kernel_type,
            vce,
            fitmethod,
            bwcheck,
            masspoints,
            scale,
        )
        res = assemble(data2, raw, vce, fitmethod)
        bw_cols = {"h01": H[:, 0], "h02": H[:, 1], "h11": H[:, 2], "h12": H[:, 3]}
        extra_cols = {}
        if kernel_type == "rad":
            extra_cols = {
                "radius_co": np.hypot(res["h_used"][:, 0], res["h_used"][:, 1]),
                "radius_tr": np.hypot(res["h_used"][:, 2], res["h_used"][:, 3]),
            }
        info_extra = {
            "kernel_type": kernel_type,
            "deriv": deriv_t,
            "h_used_p": res["h_used"],
            "h_used_q": res["h_used"],
        }
        reference = "R rd2d 1.0.0 rd2d()"
    else:
        ku = _kink_unknown(kink_unknown)
        kp = _kink_position(kink_position, J)
        if kp.any() and any(ku):
            raise _bad(
                "Use either kink_position or kink_unknown, not both.",
                "Pick one kink option.",
            )
        if h is not None and (kp.any() or any(ku)):
            raise _bad(
                "kink options apply only to automatic bandwidth selection.",
                "Drop h= or the kink options.",
            )
        q_val = (p if ku[0] else p + 1) if q_val is None else q_val
        D = _distance_matrix(
            arrs["x1"], arrs["x2"], arrs["t"], b, distance, data, valid
        )
        H = _distance_h(h, J)
        H_rbc = H
        bws = "user provided"
        if H is None:
            sel = _with_first_stage_guard(
                lambda bp: distance_bandwidths(
                    arrs["y"],
                    D,
                    b,
                    p,
                    ku,
                    kp,
                    kernel,
                    bwselect,
                    vce,
                    bwcheck,
                    masspoints,
                    cl,
                    sreg,
                    float(cqt),
                    fitmethod,
                    arrs.get("fuzzy"),
                    bp,
                ),
                bwparam,
                fuzzy is not None,
            )
            H, H_rbc = sel["h"], sel["h_rbc"]
            bws = bwselect
        raw = distance_estimates(
            Y, D, H, H_rbc, p, q_val, kernel, bwcheck, masspoints, cl, vce
        )
        res = distance_assemble(raw, D.shape[0], cl, vce, fitmethod)
        bw_cols = {
            "h0": H[:, 0],
            "h1": H[:, 1],
            "h0_rbc": H_rbc[:, 0],
            "h1_rbc": H_rbc[:, 1],
        }
        extra_cols = {}
        info_extra = {"q": q_val}
        reference = "R rd2d 1.0.0 rd2d.distance()"
    return _build_result(
        res,
        b,
        approach,
        p,
        kernel,
        vce,
        fitmethod,
        alpha,
        side,
        int(valid.sum()),
        fuzzy is not None,
        cluster,
        masspoints,
        w,
        bw_cols,
        extra_cols,
        info_extra,
        reference,
        bwcheck,
        bws,
    )


def _rd2d_pooled(
    data, arrs, valid, boundary, p, kernel, h, bwselect, alpha, fuzzy, cluster
) -> Any:
    from .rdrobust import rdrobust

    dist = _signed_boundary_distance(arrs["x1"], arrs["x2"], arrs["t"], boundary)
    frame = pd.DataFrame({"__y": arrs["y"], "__d": dist})
    kw: Dict[str, Any] = {}
    if fuzzy is not None:
        frame["__f"] = arrs["fuzzy"]
        kw["fuzzy"] = "__f"
    if cluster is not None:
        frame["__c"] = arrs["cluster"]
        kw["cluster"] = "__c"
    if h is not None:
        kw["h"] = float(h)
    bw = bwselect if bwselect in ("mserd", "msetwo", "cerrd", "certwo") else "mserd"
    res = rdrobust(
        frame,
        y="__y",
        x="__d",
        c=0.0,
        p=p,
        kernel=kernel if kernel != "gaussian" else "triangular",
        bwselect=bw,
        alpha=alpha,
        manipulation_test=False,
        **kw,
    )
    res.method = "2D Boundary RD (pooled distance to boundary, rdrobust)"
    res.model_info["approach"] = "pooled"
    res.model_info["boundary"] = "x1=0" if boundary is None else "custom"
    return res


def _build_result(
    res,
    b,
    approach,
    p,
    kernel,
    vce,
    fitmethod,
    alpha,
    side,
    n,
    is_fuzzy,
    cluster,
    masspoints,
    w,
    bw_cols,
    extra_cols,
    info_extra,
    reference,
    bwcheck,
    bwselect,
) -> CausalResult:
    t_q, pv_q, lo_q, hi_q = _inference(res["tau_q"], res["se_q"], alpha, side)
    detail = pd.DataFrame(
        {
            "b1": b[:, 0],
            "b2": b[:, 1],
            "estimate_p": res["tau_p"],
            "std_err_p": res["se_p"],
            "estimate_q": res["tau_q"],
            "std_err_q": res["se_q"],
            "t_value": t_q,
            "p_value": pv_q,
            "ci_lower": lo_q,
            "ci_upper": hi_q,
            **bw_cols,
            "n_co": res["n_co"].astype(int),
            "n_tr": res["n_tr"].astype(int),
            **extra_cols,
        }
    )
    est_q = float(w @ res["tau_q"])
    est_p = float(w @ res["tau_p"])
    se_h = (
        float(np.sqrt(max(w @ res["cov_q"] @ w, 0.0)))
        if np.all(np.isfinite(res["cov_q"]))
        else float("nan")
    )
    th, ph, loh, hih = _inference(np.array([est_q]), np.array([se_h]), alpha, side)
    wb = {
        "weights": w,
        "estimate_p": est_p,
        "estimate_q": est_q,
        "std_err_q": se_h,
        "t_value": float(th[0]),
        "p_value": float(ph[0]),
        "ci_lower": float(loh[0]),
        "ci_upper": float(hih[0]),
    }
    info: Dict[str, Any] = {
        "approach": approach,
        "reference": reference,
        "polynomial_p": p,
        "kernel": kernel,
        "vce": vce,
        "fitmethod": fitmethod,
        "masspoints": masspoints,
        "bwcheck": bwcheck,
        "bwselect": bwselect,
        "cluster": cluster,
        "fuzzy": is_fuzzy,
        "side": side,
        "n_eval_points": int(b.shape[0]),
        "eval_points": b.tolist(),
        "cov_p": res["cov_p"],
        "cov_q": res["cov_q"],
        "headline": "WBATE (bias-corrected centre)",
        "wbate": wb,
    }
    info.update(info_extra)
    keys = ("itt", "fs") if is_fuzzy else ("mu0", "mu1")
    for tag in ("p", "q"):
        if is_fuzzy:
            for k in ("itt", "fs"):
                info[f"{k}_{tag}"] = res[f"{k}_{tag}"]
                info[f"se_{k}_{tag}"] = res[f"se_{k}_{tag}"]
        else:
            for k in keys:
                info[f"{k}_{tag}"] = res[f"{k}_{tag}"]
            info[f"se0_{tag}"] = res[f"se0_{tag}"]
            info[f"se1_{tag}"] = res[f"se1_{tag}"]
    if is_fuzzy and np.any(~np.isfinite(res["tau_p"])):
        warnings.warn(
            "rd2d: zero first-stage fuzzy RD estimate detected in bandwidth window; "
            "returning NaN for the fuzzy effect at those points.",
            RuntimeWarning,
            stacklevel=3,
        )
    label = "location-based" if approach == "location" else "distance-based"
    return CausalResult(
        method=f"2D Boundary RD ({label}, rd2d)",
        estimand="WBATE: equally weighted average of the pointwise boundary effects",
        estimate=est_q,
        se=se_h,
        pvalue=float(ph[0]),
        ci=(float(loh[0]), float(hih[0])),
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=info,
        _citation_key="rd2d",
    )


# --------------------------------------------------------------------------
# rd2d_bw
# --------------------------------------------------------------------------


@records_call("rd2d_bw")
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
    """Bandwidth selection for boundary discontinuity designs.

    Parameters
    ----------
    data : DataFrame
    y, x1, x2, treatment : str
        Outcome, the two running variables and the 0/1 side indicator.
    boundary : callable, optional
        Boundary ``x2 = f(x1)`` for default evaluation points.
    approach : {'location', 'distance'}, default 'location'
    p : int, default 1
    kernel : {'triangular', 'epanechnikov', 'uniform', 'gaussian'}
    eval_points : array (k, 2), optional
    n_eval : int, default 1
    deriv : (int, int), default (0, 0)
    tangvec : array (k, 2), optional
    kernel_type : {'prod', 'rad'}, default 'prod'
    bwselect : str, default 'mserd'
        One of mserd, cerrd, imserd, icerrd, msetwo, certwo, imsetwo, icertwo.
    method : {'dpi', 'rot'}, default 'dpi'
        Bias-bandwidth rule (location approach).
    vce : {'hc0', 'hc1', 'hc2', 'hc3'}, default 'hc1'
    cluster : str, optional
    fuzzy : str, optional
    bwparam : {'main', 'itt'}, default 'main'
    fitmethod : {'joint', 'separate'}, default 'joint'
    masspoints : {'check', 'adjust', 'off'}, default 'check'
    bwcheck : int, 'auto' or None, default 'auto'
        'auto' is 20 (location) or ``20 + p + 1`` (distance).
    scaleregul : float, default 1.0
    scalebiascrct : float, default 1.0
    stdvars : bool, default True
    kink_unknown : bool or (bool, bool), default False
    kink_position : sequence, optional
    cqt : float, default 0.5
    distance : array (n, k) or list of str, optional

    Returns
    -------
    DataFrame
        One row per evaluation point: ``b1``, ``b2`` and ``h01``, ``h02``,
        ``h11``, ``h12`` (location) or ``h0``, ``h1`` (distance).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x1, x2 = rng.uniform(-1, 1, 2000), rng.uniform(-1, 1, 2000)
    >>> t = (x1 >= 0).astype(int)
    >>> y = x1 + 2 * t + rng.normal(0, 0.3, 2000)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "t": t})
    >>> list(sp.rd2d_bw(df, y="y", x1="x1", x2="x2", treatment="t").columns)
    ['b1', 'b2', 'h01', 'h02', 'h11', 'h12']

    References
    ----------
    cattaneo2025boundary
    """
    _require_frame(
        data, [("y", y), ("x1", x1), ("x2", x2), ("treatment", treatment)], "sp.rd2d_bw"
    )
    approach = _choice(approach, ("location", "distance"), "approach")
    method = _choice(method, ("dpi", "rot"), "method")
    kernel, kernel_type, vce, bwselect, method, masspoints, fitmethod, bwparam, p = (
        _common_options(
            kernel,
            kernel_type,
            vce,
            bwselect,
            method,
            masspoints,
            fitmethod,
            bwparam,
            p,
            cqt,
            scaleregul,
        )
    )
    if boundary is not None and not callable(boundary):
        raise _bad("boundary must be callable or None", "Pass a function x2 = f(x1).")
    vce = _cluster_vce(vce, cluster)
    arrs, valid = _prepare(data, y, x1, x2, treatment, cluster, fuzzy, "sp.rd2d_bw")
    b = _eval_points(eval_points, arrs["x1"], arrs["x2"], boundary, n_eval)
    J = b.shape[0]
    bwcheck = _bwcheck_value(bwcheck, 20 if approach == "location" else 20 + p + 1)
    deriv_t, tv = _deriv_options(deriv, tangvec, p, J)
    _check_bwcheck_size(bwcheck, arrs["t"])
    bwcheck = _mass_points(arrs, masspoints, bwcheck, p)
    cl = arrs.get("cluster")
    if approach == "location":
        Y = arrs["y"] if fuzzy is None else np.column_stack([arrs["y"], arrs["fuzzy"]])
        data2 = BoundaryData(arrs["x1"], arrs["x2"], arrs["t"], Y, cl)
        sel = _with_first_stage_guard(
            lambda bp: location_bandwidths(
                data2,
                b,
                p,
                deriv_t,
                tv,
                kernel,
                kernel_type,
                bwselect,
                method,
                vce,
                bwcheck,
                masspoints,
                fitmethod,
                float(scaleregul),
                scalebiascrct,
                stdvars,
                bp,
            ),
            bwparam,
            fuzzy is not None,
        )
        H = sel["h"]
        return pd.DataFrame(
            {
                "b1": b[:, 0],
                "b2": b[:, 1],
                "h01": H[:, 0],
                "h02": H[:, 1],
                "h11": H[:, 2],
                "h12": H[:, 3],
            }
        )
    ku = _kink_unknown(kink_unknown)
    kp = _kink_position(kink_position, J)
    if kp.any() and any(ku):
        raise _bad(
            "Use either kink_position or kink_unknown, not both.",
            "Pick one kink option.",
        )
    D = _distance_matrix(arrs["x1"], arrs["x2"], arrs["t"], b, distance, data, valid)
    sel = _with_first_stage_guard(
        lambda bp: distance_bandwidths(
            arrs["y"],
            D,
            b,
            p,
            ku,
            kp,
            kernel,
            bwselect,
            vce,
            bwcheck,
            masspoints,
            cl,
            float(scaleregul),
            float(cqt),
            fitmethod,
            arrs.get("fuzzy"),
            bp,
        ),
        bwparam,
        fuzzy is not None,
    )
    H = sel["h"]
    return pd.DataFrame({"b1": b[:, 0], "b2": b[:, 1], "h0": H[:, 0], "h1": H[:, 1]})
