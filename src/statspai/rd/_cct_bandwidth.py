"""MSE- and CER-optimal bandwidths for local polynomial RD estimators.

Clean-room rewrite (2026-10-20). The selector follows the direct plug-in
procedure of Calonico, Cattaneo and Titiunik (2014, bib key
``calonico2014robust``, Supplemental Appendix S.2.6) as extended in
Calonico, Cattaneo, Farrell and Titiunik (2019, Review of Economics and
Statistics 101(3), Supplemental Appendix Part IV: covariates, clustering,
one bandwidth per side) and, for the CER rules, Calonico, Cattaneo and
Farrell (2020, bib key ``calonico2020optimal``). R ``rdrobust`` 4.0.0
(GPL-3) was used only as a black box: its outputs on committed data fixed
the implementation choices the papers leave open. The specification, with
the black-box evidence for each convention, is
``docs/dev/cleanroom/rd_cct.md``.

The procedure, for a target of derivative order ``nu`` estimated with a
local polynomial of order ``o`` (CCFT 2019 SA §12):

* a pilot bandwidth ``c = C_K * min(sd(x), IQR(x)/1.349) * N^(-1/5)``
  (§12.1) gives the variance ``V`` of the target and the factor
  multiplying the unknown ``(o+1)``-th derivative in its bias;
* that derivative is estimated by a fit of order ``o_B`` on a preliminary
  bandwidth, which gives the bias ``B`` and the variance ``R`` of the bias
  estimate (the regularisation term of Imbens and Kalyanaraman);
* the bandwidth is ``c * [(1+2nu) V / (2(o+1-nu) (B^2 + 3 s^2 R))]^(1/(2o+3))``
  where ``s`` is ``scaleregul``.

Three such stages run in sequence: the bias bandwidth's own preliminary
bandwidth ``d`` (order ``q+1``, bias from a global order-``q+2`` fit), the
bias bandwidth ``b`` (order ``q``, target ``p+1``, bias on ``d``) and the
main bandwidth ``h`` (order ``p``, target ``deriv``, bias on ``b``). Each
stage is combined across the sides as the RD difference (``rd``), the sum
(``sum``) or side by side (``two``).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import (
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
    StatsPAIWarning,
)
from ._cct_side import (
    Side,
    complete_rows,
    kernel_name,
    partial_out_covariates,
    residual_rows,
    variance_of,
)

__all__ = ["cct_bandwidth", "cct_bias_corrected", "BW_SELECTORS", "masspoint_shares"]

#: The ten selectors of CCFT (2017, Stata Journal) / rdrobust.
BW_SELECTORS = (
    "mserd",
    "msetwo",
    "msesum",
    "msecomb1",
    "msecomb2",
    "cerrd",
    "certwo",
    "cersum",
    "cercomb1",
    "cercomb2",
)

MASSPOINTS = ("adjust", "check", "off")

_VCE_KINDS = frozenset({"nn", "hc0", "hc1", "hc2", "hc3", "cr1", "cr2", "cr3"})

#: Pilot-bandwidth constants C_K (CCT 2014 SA S.2.6 / CCFT 2019 SA §12.1:
#: 2.576 triangular, 1.843 uniform); 2.34 for the Epanechnikov kernel was
#: fixed by black-box comparison with rdrobust 4.0.0.
_PILOT_CONSTANT = {"triangular": 2.576, "uniform": 1.843, "epanechnikov": 2.34}

#: Share of tied running-variable values above which a side counts as having
#: mass points (black box: no adjustment at a tied share of 0.19996, the
#: ``bwcheck`` floor at 0.2002).
_MASS_SHARE = 0.2

#: Unique values the pilot window must contain when mass points are
#: detected under ``masspoints='adjust'`` (rdrobust documentation).
_DEFAULT_BWCHECK = 10

#: Added to the distance of the ``bwcheck``-th unique value so that value
#: gets positive kernel weight (black box).
_FLOOR_EPS = 1e-8


# --------------------------------------------------------------------------- #
#  Option handling
# --------------------------------------------------------------------------- #


def masspoint_shares(x_c: np.ndarray, masspoints: str, where: str) -> Dict[str, Any]:
    """Validate ``masspoints`` and report the tied share on each side.

    The tied share of a side is ``1 - (unique values) / (observations)``.
    ``detected`` is True when either share reaches 0.2. Under
    ``masspoints='check'`` a detection is reported as a ``UserWarning``.
    """
    if masspoints not in MASSPOINTS:
        raise MethodIncompatibility(
            f"{where}: masspoints={masspoints!r}; use 'adjust', 'check' or 'off'.",
            recovery_hint="Pass masspoints='adjust' (default), 'check' or 'off'.",
            diagnostics={"masspoints": masspoints},
        )
    x_c = np.asarray(x_c, dtype=float)

    def share(v: np.ndarray) -> float:
        return 1.0 - np.unique(v).size / v.size if v.size else 0.0

    left, right = share(x_c[x_c < 0]), share(x_c[x_c >= 0])
    detected = max(left, right) >= _MASS_SHARE
    if detected and masspoints == "check":
        warnings.warn(
            f"{where}: mass points detected in the running variable (tied share "
            f"{left:.2f} left, {right:.2f} right); masspoints='adjust' applies "
            "R rdrobust's correction.",
            UserWarning,
            stacklevel=3,
        )
    return {
        "mass_left": float(left),
        "mass_right": float(right),
        "detected": bool(detected),
        "masspoints": masspoints,
    }


def _split_rd_vce(vce: Any, cluster: Any, function: str) -> Tuple[Any, Any]:
    """Read Stata's ``vce(cluster v)`` / ``vce(nncluster v)`` spellings.

    ``'cluster v'`` means plug-in residuals clustered on ``v`` (``hc1`` +
    ``cluster='v'``); ``'nncluster v'`` means nearest-neighbour residuals
    clustered on ``v``. Anything else is returned unchanged for the caller
    to validate.
    """
    if not isinstance(vce, str):
        return vce, cluster
    text = vce.strip()
    if text.lower().startswith("vce(") and text.endswith(")"):
        text = text[4:-1]
    words = text.split()
    if len(words) != 2 or words[0].lower() not in ("cluster", "nncluster"):
        return vce, cluster
    kind = "hc1" if words[0].lower() == "cluster" else "nn"
    var = words[1]
    if cluster is not None and cluster != var:
        raise MethodIncompatibility(
            f"{function}: vce={vce!r} clusters on {var!r} but cluster={cluster!r} "
            "was also passed.",
            recovery_hint="Name the cluster variable once, in vce= or cluster=.",
            diagnostics={"vce": vce, "cluster": cluster},
        )
    return kind, var


def _cluster_vce_note(vce: str, cluster: Any, function: str) -> str:
    """R rdrobust 4.0.0's cluster mapping of ``vce``, with its warnings.

    With clusters ``hc2`` / ``hc3`` are CR2 / CR3 (and ``nn`` / ``hc0`` /
    ``hc1`` CR1, silently, as before). ``cr1`` / ``cr2`` / ``cr3`` without
    clusters fall back to ``hc1`` / ``hc2`` / ``hc3``, as R does.
    """
    has_cluster = cluster is not None and not (isinstance(cluster, str) and not cluster)
    if has_cluster and vce in ("hc2", "hc3"):
        warnings.warn(
            f"{function}: with cluster=, vce={vce!r} is the cluster-robust "
            f"CR{vce[2]} variance (R rdrobust 4.0.0 maps it the same way; "
            "earlier StatsPAI releases used HC-scaled residuals with the CR1 "
            "factor).",
            StatsPAIWarning,
            stacklevel=3,
        )
    if not has_cluster and vce in ("cr1", "cr2", "cr3"):
        warnings.warn(
            f"{function}: vce={vce!r} needs cluster=; using vce='hc{vce[2]}' "
            "(as R rdrobust does).",
            StatsPAIWarning,
            stacklevel=3,
        )
        return "hc" + vce[2]
    return vce


# --------------------------------------------------------------------------- #
#  One stage of the plug-in cascade
# --------------------------------------------------------------------------- #


def _combination(
    gamma: np.ndarray, core_coef: np.ndarray, nu: int, fuzzy: bool
) -> np.ndarray:
    """Weights on the columns that define one side's target statistic.

    Sharp: ``(1, -gamma_Y)``. Fuzzy: the delta-method weights of
    ``Y_nu / T_nu`` on that side, where ``(Y_nu, T_nu)`` are the
    covariate-adjusted ``nu``-th coefficients of the stage's own fit
    (black box: rdrobust's fuzzy cascade linearises a per-side ratio).
    """
    if not fuzzy:
        return np.r_[1.0, -gamma[:, 0]]
    y_nu, t_nu = core_coef[nu, 0], core_coef[nu, 1]
    head = np.r_[1.0 / t_nu, -y_nu / t_nu**2]
    if gamma.shape[0] == 0:
        return head
    return np.r_[head, -(gamma[:, 0] / t_nu - y_nu * gamma[:, 1] / t_nu**2)]


def _stage(
    side: Side,
    order: int,
    nu: int,
    bias_order: int,
    pilot: float,
    bias_bw: float,
    kernel: str,
    vce: str,
    nnmatch: int,
    n_core: int,
    fuzzy: bool,
) -> Dict[str, float]:
    """``V``, signed ``B`` and regularisation ``R`` for one side and stage."""
    fit = side.fit(pilot, order, kernel)
    gamma, core = partial_out_covariates(side, fit, n_core)
    s = _combination(gamma, core, nu, fuzzy)
    ell = np.zeros(order + 1)
    ell[nu] = math.factorial(nu)
    rho = residual_rows(side, fit, vce, nnmatch)
    var = variance_of(side, fit.mask, fit.weights_for(ell), rho, s, order + 1, vce)
    factor = fit.bias_factor(ell)

    bias_fit = side.fit(bias_bw, bias_order, kernel)
    pick = np.zeros(bias_order + 1)
    pick[order + 1] = 1.0
    deriv_coef = float(bias_fit.coef[order + 1] @ s)
    rho_b = residual_rows(side, bias_fit, vce, nnmatch)
    var_coef = variance_of(
        side, bias_fit.mask, bias_fit.weights_for(pick), rho_b, s, bias_order + 1, vce
    )
    if fit.singular or bias_fit.singular:
        bad = fit if fit.singular else bias_fit
        raise DataInsufficient(
            "rdbwselect: a preliminary local polynomial of order "
            f"{bad.order} has fewer than {bad.order + 1} distinct values of the "
            f"running variable inside its window (bandwidth {bad.h:.4g}); the "
            "plug-in bandwidth is not identified.",
            recovery_hint=(
                "Pass h= yourself, lower p / q, or use a method built for a "
                "discrete running variable (sp.rd_discrete)."
            ),
            diagnostics={"order": bad.order, "bandwidth": bad.h},
            alternative_functions=["sp.rd_discrete", "sp.rdrandinf"],
        )
    return {"V": var, "B": factor * deriv_coef, "R": factor**2 * var_coef}


def _plug_in(
    V: float, B2: float, R: float, order: int, nu: int, pilot: float, scale: float
) -> float:
    """``pilot * [(1+2nu) V / (2(o+1-nu)(B^2 + 3 scale^2 R))]^(1/(2o+3))``.

    ``scale`` enters squared (black box; ``scaleregul=2`` quadruples the
    regularisation term in rdrobust 4.0.0).
    """
    denom = 2.0 * (order + 1 - nu) * (B2 + 3.0 * scale**2 * R)
    ratio = (1.0 + 2.0 * nu) * V / denom if denom > 0 else math.inf
    if not (np.isfinite(ratio) and ratio > 0):
        raise NumericalInstability(
            "rdbwselect: the plug-in bandwidth is not defined on this sample "
            f"(variance {V:.3g}, squared bias plus regularisation "
            f"{denom / (2.0 * (order + 1 - nu)):.3g}).",
            recovery_hint=(
                "The running variable is too coarse for the selector: pass h= "
                "yourself, use masspoints='adjust', or use sp.rd_discrete."
            ),
            diagnostics={"variance": V, "denominator": denom, "order": order},
            alternative_functions=["sp.rd_discrete", "sp.rdrandinf"],
        )
    return pilot * ratio ** (1.0 / (2 * order + 3))


def _combine(
    left: Dict[str, float],
    right: Dict[str, float],
    form: str,
    order: int,
    nu: int,
    pilot: float,
    scale: float,
) -> Tuple[float, float]:
    if form == "two":
        return (
            _plug_in(left["V"], left["B"] ** 2, left["R"], order, nu, pilot, scale),
            _plug_in(right["V"], right["B"] ** 2, right["R"], order, nu, pilot, scale),
        )
    sign = -1.0 if form == "rd" else 1.0
    bw = _plug_in(
        left["V"] + right["V"],
        (right["B"] + sign * left["B"]) ** 2,
        left["R"] + right["R"],
        order,
        nu,
        pilot,
        scale,
    )
    return bw, bw


# --------------------------------------------------------------------------- #
#  Public selector
# --------------------------------------------------------------------------- #


def _pilot_bandwidth(
    x: np.ndarray, kernel: str, masspoints: str, bwcheck: Optional[int]
) -> float:
    """Rule-of-thumb pilot ``c`` (CCFT 2019 SA §12.1) with the mass-point rules.

    Under ``masspoints='adjust'`` the sample size is the number of distinct
    values of ``x``, and when a side's tied share reaches 0.2 the window is
    widened to contain ``bwcheck`` (default 10) distinct values on each
    side. An explicit ``bwcheck`` always applies. The IQR uses the
    inverse-CDF sample quantile.
    """
    sd = float(np.std(x, ddof=1))
    iqr = float(
        np.quantile(x, 0.75, method="inverted_cdf")
        - np.quantile(x, 0.25, method="inverted_cdf")
    )
    n_eff = x.size
    left, right = x[x < 0], x[x >= 0]
    if masspoints == "adjust":
        n_eff = np.unique(x).size
        shares = [1.0 - np.unique(v).size / v.size for v in (left, right) if v.size]
        if bwcheck is None and shares and max(shares) >= _MASS_SHARE:
            bwcheck = _DEFAULT_BWCHECK
    pilot = _PILOT_CONSTANT[kernel] * min(sd, iqr / 1.349) * n_eff ** (-0.2)
    if bwcheck is not None and bwcheck > 0:
        for dist in (np.unique(-left), np.unique(right)):
            if dist.size:
                floor = dist[min(int(bwcheck), dist.size) - 1] + _FLOOR_EPS
                pilot = max(pilot, float(floor))
    return pilot


def _build_sides(
    y: np.ndarray,
    x: np.ndarray,
    covs: Optional[np.ndarray],
    fuzzy: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    cluster: Optional[np.ndarray],
) -> Tuple[Side, Side, int]:
    cols: List[np.ndarray] = [np.asarray(y, dtype=float)]
    if fuzzy is not None:
        cols.append(np.asarray(fuzzy, dtype=float))
    n_core = len(cols)
    if covs is not None:
        z = np.asarray(covs, dtype=float)
        z = z[:, None] if z.ndim == 1 else z
        cols.extend(z[:, j] for j in range(z.shape[1]))
    mat = np.column_stack(cols)
    w = None if weights is None else np.asarray(weights, dtype=float)
    cl = None if cluster is None else np.asarray(cluster)
    left = x < 0
    sides = []
    for m in (left, ~left):
        sides.append(
            Side(
                x[m], mat[m], None if w is None else w[m], None if cl is None else cl[m]
            )
        )
    return sides[0], sides[1], n_core


def _drop_incomplete(y, x, covs, fuzzy, weights, cluster):
    """Keep the rows where the outcome, score, covariates, treatment and
    weights are all finite (the same complete-case rule as the callers)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    keep = complete_rows(y, x, covs, fuzzy, weights)
    if keep.all():
        return y, x, covs, fuzzy, weights, cluster

    def sub(a):
        return None if a is None else np.asarray(a)[keep]

    return y[keep], x[keep], sub(covs), sub(fuzzy), sub(weights), sub(cluster)


def _perfect_compliance(fuzzy: np.ndarray, x: np.ndarray) -> bool:
    """True when the treatment is constant on either side of the cutoff."""
    t = np.asarray(fuzzy, dtype=float)
    return any(np.unique(t[m]).size <= 1 for m in (x < 0, x >= 0) if m.any())


def cct_bandwidth(
    y: np.ndarray,
    x: np.ndarray,
    c: float = 0.0,
    p: int = 1,
    q: Optional[int] = None,
    deriv: int = 0,
    kernel: str = "triangular",
    bwselect: str = "mserd",
    scaleregul: float = 1.0,
    nnmatch: int = 3,
    masspoints: str = "adjust",
    bwrestrict: bool = True,
    covs: Optional[np.ndarray] = None,
    vce: str = "nn",
    cluster: Optional[np.ndarray] = None,
    fuzzy: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    bwcheck: Optional[int] = None,
) -> Dict[str, float]:
    """MSE- or CER-optimal ``(h, b)`` on each side of the cutoff.

    Parameters
    ----------
    y, x : array
        Outcome and running variable.
    c : float
        Cutoff.
    p, q, deriv : int
        Order of the point estimator, of the bias correction (default
        ``p + 1``) and of the derivative of interest.
    kernel : {'triangular', 'uniform', 'epanechnikov'}
    bwselect : one of :data:`BW_SELECTORS`
    scaleregul : float
        Scale of the regularisation term (0 removes it).
    nnmatch : int
        Neighbours for ``vce='nn'``.
    masspoints : {'adjust', 'check', 'off'}
    bwrestrict : bool
        Cap every bandwidth at the range of the running variable (common
        bandwidths: the wider side's range; per-side bandwidths: that side's).
    covs, cluster, fuzzy, weights : array, optional
        Covariates (adjusted per side, CCFT 2019), cluster ids (CR1-type
        variance), treatment indicator (fuzzy design) and observation weights
        (multiplying the kernel).
    vce : {'nn', 'hc0', 'hc1', 'hc2', 'hc3', 'cr1', 'cr2', 'cr3'}
    bwcheck : int, optional
        Minimum number of distinct running-variable values in the pilot
        window on each side.

    Returns
    -------
    dict
        ``h_left``, ``h_right``, ``b_left``, ``b_right``.

    References
    ----------
    calonico2014robust, calonico2020optimal
    """
    if bwselect not in BW_SELECTORS:
        raise ValueError(f"bwselect must be one of {BW_SELECTORS}, got {bwselect!r}")
    vce = str(vce).lower()
    if vce not in _VCE_KINDS:
        raise ValueError(f"vce must be one of {sorted(_VCE_KINDS)}, got {vce!r}")
    if cluster is None and vce in ("cr1", "cr2", "cr3"):
        vce = "hc" + vce[2]
    kernel = kernel_name(kernel)
    q = p + 1 if q is None else int(q)
    y, x, covs, fuzzy, weights, cluster = _drop_incomplete(
        y, x, covs, fuzzy, weights, cluster
    )
    xc = x - c
    if fuzzy is not None and _perfect_compliance(fuzzy, xc):
        fuzzy = None  # sharp bandwidth under one-sided compliance (CCFT 2017)
    left, right, n_core = _build_sides(y, xc, covs, fuzzy, weights, cluster)
    if min(left.n, right.n) < q + 2:
        raise ValueError("rdbwselect: too few observations on a side of the cutoff.")
    is_fuzzy = fuzzy is not None
    yv = np.asarray(y, dtype=float)
    if np.ptp(yv) <= 1e-12 * max(1.0, float(np.max(np.abs(yv)))):
        raise DataInsufficient(
            "rdbwselect: the outcome is constant, so the plug-in bandwidth "
            "(variance over squared bias) is not defined.",
            recovery_hint="Pass h= yourself; the RD effect of a constant is zero.",
            diagnostics={"outcome_range": float(np.ptp(yv))},
        )

    pilot = _pilot_bandwidth(xc, kernel, masspoints, bwcheck)
    range_l, range_r = -float(xc.min()), float(xc.max())
    cap = max(range_l, range_r)

    def restrict(pair: Tuple[float, float], form: str) -> Tuple[float, float]:
        if not bwrestrict:
            return pair
        if form == "two":
            return min(pair[0], range_l), min(pair[1], range_r)
        common = min(pair[0], cap)
        return common, common

    def stage(order, nu, bias_order, bias_l, bias_r):
        args = (kernel, vce, nnmatch, n_core, is_fuzzy)
        return (
            _stage(left, order, nu, bias_order, pilot, bias_l, *args),
            _stage(right, order, nu, bias_order, pilot, bias_r, *args),
        )

    prefix, rule = bwselect[:3], bwselect[3:]
    forms = {
        "rd": ("rd",),
        "two": ("two",),
        "sum": ("sum",),
        "comb1": ("rd", "sum"),
        "comb2": ("rd", "sum", "two"),
    }[rule]

    d_left, d_right = stage(q + 1, q + 1, q + 2, range_l, range_r)
    results: Dict[str, Tuple[float, float, float, float]] = {}
    for form in forms:
        d = restrict(_combine(d_left, d_right, form, q + 1, q + 1, pilot, 0.0), form)
        b_left, b_right = stage(q, p + 1, q + 1, d[0], d[1])
        b = restrict(_combine(b_left, b_right, form, q, p + 1, pilot, scaleregul), form)
        h_left, h_right = stage(p, deriv, q, b[0], b[1])
        h = restrict(_combine(h_left, h_right, form, p, deriv, pilot, scaleregul), form)
        results[form] = (h[0], h[1], b[0], b[1])

    if rule == "comb1":
        chosen = tuple(min(a, b) for a, b in zip(results["rd"], results["sum"]))
    elif rule == "comb2":
        chosen = tuple(
            float(np.median(v))
            for v in zip(results["rd"], results["sum"], results["two"])
        )
    else:
        chosen = results[forms[0]]
    h_l, h_r, b_l, b_r = chosen

    if prefix == "cer":
        # CCF (2020): h_cer = h_mse * m^(-p/((3+p)(3+2p))), with m the sample
        # size, or the clusters counted on each side and added (black box).
        if left.cluster is not None and right.cluster is not None:
            m = np.unique(left.cluster).size + np.unique(right.cluster).size
        else:
            m = xc.size
        shrink = float(m) ** (-p / ((3.0 + p) * (3.0 + 2.0 * p)))
        h_l, h_r = h_l * shrink, h_r * shrink

    return {
        "h_left": float(h_l),
        "h_right": float(h_r),
        "b_left": float(b_l),
        "b_right": float(b_r),
    }


def cct_bias_corrected(*args: Any, **kwargs: Any) -> Tuple[float, float, float, float]:
    """Conventional and robust bias-corrected RD estimate and SEs.

    Thin re-export of :func:`statspai.rd._cct_inference.cct_bias_corrected`,
    kept here because callers and tests reach it through this module.
    """
    from ._cct_inference import cct_bias_corrected as _impl

    return _impl(*args, **kwargs)
