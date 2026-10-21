"""Power and sample-size calculations for RD designs.

Clean-room implementation (2026-10-21) of the power function and the
sample-size rule of Cattaneo, Titiunik and Vazquez-Bare (2019, bib key
``cattaneo2019power``), section 2. R ``rdpower`` 3.0 (GPL-3) was used only
as a black box; the conventions it fixed are recorded in
``docs/dev/cleanroom/rdpower.md`` and pinned by
``tests/reference_parity/test_rdpower_R_parity.py``.

Two modes:

* **data mode** (``data=``, ``y=``, ``x=``): the variances come from a
  robust bias-corrected local polynomial fit, ``sp.rdrobust``; this is the
  paper's ex-post / ex-ante calculation.
* **design mode** (no data): a planning calculator from postulated outcome
  variances, side sample sizes and bandwidths.

The two-sided power of the robust bias-corrected test with standard error
``s`` against the alternative ``tau`` is
``1 - Phi(tau/s + z) + Phi(tau/s - z)``, ``z = Phi^-1(1 - alpha/2)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import brentq

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["RDPowerResult", "RDSampSiResult", "rdpower", "rdsampsi"]

# Design-mode variance constant: s^2 = 0.35 * sum(var / (n h)). A planning
# heuristic kept from the previous StatsPAI calculator (tests pin it); it is
# not a property of any reference implementation.
_DESIGN_KAPPA = 0.35

# rdrobust options that the per-side variance primitive reproduces.
_SAMPSI_DATA_OPTIONS = frozenset(
    {
        "fuzzy",
        "deriv",
        "p",
        "q",
        "kernel",
        "bwselect",
        "h",
        "b",
        "rho",
        "covs",
        "cluster",
        "vce",
        "weights",
        "masspoints",
        "warn_mass_points",
        "warn_weak_first_stage",
        "manipulation_test",
    }
)


@dataclass
class RDPowerResult(ResultProtocolMixin):
    """Power of an RD design against one alternative."""

    power: float
    mde: float
    alpha: float
    n_left: int
    n_right: int
    se: float
    tau: float

    _citation_keys: ClassVar[Tuple[str, ...]] = ("cattaneo2019power",)

    def summary(self) -> str:
        """Plain-text table of the calculation."""
        lines = [
            "RD Power Calculation",
            "-" * 40,
            f"Hypothesised τ : {self.tau: .4f}",
            f"SE(τ̂)         : {self.se: .4f}",
            f"Alpha          : {self.alpha}",
            f"Power          : {self.power: .4f}",
            f"MDE (at 80%)   : {self.mde: .4f}",
            f"n (left)       : {self.n_left}",
            f"n (right)      : {self.n_right}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


@dataclass
class RDSampSiResult(ResultProtocolMixin):
    """Sample size needed to reach a target power."""

    n_left: int
    n_right: int
    n_total: int
    tau: float
    target_power: float
    alpha: float

    _citation_keys: ClassVar[Tuple[str, ...]] = ("cattaneo2019power",)

    def summary(self) -> str:
        """One-line statement of the required sample."""
        return (
            f"RD Sample Size: N = {self.n_total} ({self.n_left} left + "
            f"{self.n_right} right) for τ = {self.tau:.4f} at power = "
            f"{self.target_power:.0%}, α = {self.alpha}"
        )

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------------
# shared pieces
# --------------------------------------------------------------------------


def _bad(msg: str, hint: str, **diag: object) -> MethodIncompatibility:
    return MethodIncompatibility(msg, recovery_hint=hint, diagnostics=dict(diag))


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise _bad(
            f"alpha must lie strictly between 0 and 1; got {alpha!r}.",
            "Pass the significance level as a fraction, e.g. alpha=0.05.",
            alpha=alpha,
        )
    return a


def _check_target(target_power: float, alpha: float) -> float:
    t = float(target_power)
    if not (alpha < t < 1.0):
        raise _bad(
            f"target_power must lie strictly between alpha ({alpha}) and 1; "
            f"got {target_power!r}.",
            "Power of a level-alpha test ranges over (alpha, 1); pick a target "
            "such as 0.8.",
            target_power=target_power,
            alpha=alpha,
        )
    return t


def _check_positive(**values: float) -> None:
    for name, v in values.items():
        if not (np.isfinite(v) and v > 0):
            raise _bad(
                f"{name} must be a positive finite number; got {v!r}.",
                f"Pass a positive value for {name}.",
                **{name: v},
            )


def _power(tau: float, se: float, alpha: float) -> float:
    """Two-sided power of the z-test with standard error ``se`` at ``tau``."""
    z = sp_stats.norm.ppf(1.0 - alpha / 2.0)
    shift = tau / se
    return float(1.0 - sp_stats.norm.cdf(shift + z) + sp_stats.norm.cdf(shift - z))


def _mde(se: float, alpha: float, target_power: float) -> float:
    """Minimum detectable effect, one-tail approximation ``(z_a + z_b) se``."""
    return float(
        (sp_stats.norm.ppf(1.0 - alpha / 2.0) + sp_stats.norm.ppf(target_power)) * se
    )


def _root_of_power(
    variance_numerator: float, tau: float, alpha: float, beta: float
) -> float:
    """Sample size ``m`` with ``power(tau; sqrt(A / m)) = beta`` (continuous).

    Power increases in ``m`` from ``alpha`` (``m -> 0``) to 1, so the root is
    bracketed by growing the upper end.
    """

    def gap(m: float) -> float:
        return _power(tau, math.sqrt(variance_numerator / m), alpha) - beta

    lo = 1e-12
    hi = 1.0
    while gap(hi) < 0.0:
        lo, hi = hi, hi * 2.0
        if hi > 1e300:  # pragma: no cover - tau is checked to be non-zero
            raise _bad(
                "rdsampsi: no finite sample size reaches the target power.",
                "Use a larger |tau| or a lower target_power.",
                tau=tau,
            )
    return float(brentq(gap, lo, hi, xtol=1e-12, rtol=4 * np.finfo(float).eps))


def _data_columns(
    data: Optional["pd.DataFrame"], y: Optional[str], x: Optional[str], fn: str
) -> bool:
    """Whether data mode applies; refuse half-specified calls."""
    if data is None:
        if y is not None or x is not None:
            raise _bad(
                f"{fn}: y=/x= only mean something with data=; pass data= as "
                "well, or use design mode (var_left/var_right/n_left/n_right/"
                "h_left/h_right).",
                "Pass data= together with y= and x=.",
            )
        return False
    if y is None or x is None:
        raise _bad(
            f"{fn}: data mode needs both y= and x= (the outcome and running "
            "variable column names) together with data=.",
            "Pass y= and x= naming columns of data=.",
        )
    return True


def _fit_rdrobust(data: Any, y: str, x: str, c: float, kwargs: Dict[str, Any]) -> Any:
    from .rdrobust import rdrobust

    return rdrobust(data, y=y, x=x, c=c, **kwargs)


def _as_pair(v: Any) -> Tuple[float, float]:
    if isinstance(v, (tuple, list, np.ndarray)):
        return float(v[0]), float(v[1])
    return float(v), float(v)


# --------------------------------------------------------------------------
# rdpower
# --------------------------------------------------------------------------


def rdpower(
    tau: float,
    var_left: float = 1.0,
    var_right: float = 1.0,
    n_left: int = 500,
    n_right: int = 500,
    h_left: float = 0.5,
    h_right: float = 0.5,
    alpha: float = 0.05,
    target_power: Optional[float] = None,
    data: Optional["pd.DataFrame"] = None,
    y: Optional[str] = None,
    x: Optional[str] = None,
    c: float = 0.0,
    **rdrobust_kwargs: object,
) -> RDPowerResult:
    """Power of an RD design given sample size and effect size.

    Parameters
    ----------
    tau : float
        Treatment effect under the alternative.
    var_left, var_right : float, default 1.0
        Design mode: outcome variance on each side of the cutoff.
    n_left, n_right : int, default 500
        Design mode: sample size on each side.
    h_left, h_right : float, default 0.5
        Design mode: bandwidth on each side.
    alpha : float, default 0.05
        Significance level of the two-sided test.
    target_power : float, optional
        Power at which the minimum detectable effect ``mde`` is reported
        (default 0.8).
    data : DataFrame, optional
        Data mode: the sample. Requires ``y`` and ``x``; the standard error is
        the robust bias-corrected one of ``sp.rdrobust`` on these data.
    y, x : str, optional
        Data mode: outcome and running-variable columns.
    c : float, default 0.0
        Data mode: cutoff.
    **rdrobust_kwargs
        Data mode: options passed to ``sp.rdrobust`` (``p``, ``kernel``,
        ``bwselect``, ``vce``, ``h``, ``covs``, ``cluster``, ...).

    Returns
    -------
    RDPowerResult
        ``power``, ``se``, ``mde``, ``alpha``, ``tau`` and the side sample
        sizes (data mode: the observed ``N-`` / ``N+``).

    Notes
    -----
    Power is ``1 - Phi(tau/se + z) + Phi(tau/se - z)`` with
    ``z = Phi^-1(1 - alpha/2)``. Design mode uses
    ``se^2 = 0.35 (var_left/(n_left h_left) + var_right/(n_right h_right))``,
    a planning heuristic. ``mde = (z + Phi^-1(target_power)) se``. In data
    mode the design-stage arguments are not used.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.rdpower(tau=0.15, n_left=500, n_right=500)
    >>> round(r.power, 3)
    0.809

    References
    ----------
    cattaneo2019power
    """
    alpha = _check_alpha(alpha)
    target = 0.8 if target_power is None else _check_target(target_power, alpha)
    tau = float(tau)
    if not np.isfinite(tau):
        raise _bad(f"tau must be finite; got {tau!r}.", "Pass a numeric effect size.")

    if _data_columns(data, y, x, "rdpower"):
        assert y is not None and x is not None  # _data_columns checked them
        fit = _fit_rdrobust(data, y, x, c, rdrobust_kwargs)
        se = float(fit.se)
        mi = fit.model_info
        nl, nr = int(mi["n_left"]), int(mi["n_right"])
    else:
        _check_positive(
            var_left=var_left,
            var_right=var_right,
            n_left=n_left,
            n_right=n_right,
            h_left=h_left,
            h_right=h_right,
        )
        se = math.sqrt(
            _DESIGN_KAPPA
            * (var_left / (n_left * h_left) + var_right / (n_right * h_right))
        )
        nl, nr = int(n_left), int(n_right)

    return RDPowerResult(
        power=_power(tau, se, alpha),
        mde=_mde(se, alpha, target),
        alpha=alpha,
        n_left=nl,
        n_right=nr,
        se=se,
        tau=tau,
    )


# --------------------------------------------------------------------------
# rdsampsi
# --------------------------------------------------------------------------


def rdsampsi(
    tau: float,
    var_left: float = 1.0,
    var_right: float = 1.0,
    h_left: float = 0.5,
    h_right: float = 0.5,
    alpha: float = 0.05,
    target_power: float = 0.80,
    ratio: float = 1.0,
    data: Optional["pd.DataFrame"] = None,
    y: Optional[str] = None,
    x: Optional[str] = None,
    c: float = 0.0,
    **rdrobust_kwargs: object,
) -> RDSampSiResult:
    """Minimum sample size for a given power in an RD design.

    Parameters
    ----------
    tau : float
        Treatment effect under the alternative (non-zero).
    var_left, var_right : float, default 1.0
        Design mode: outcome variance on each side.
    h_left, h_right : float, default 0.5
        Design mode: bandwidth on each side.
    alpha : float, default 0.05
        Significance level of the two-sided test.
    target_power : float, default 0.80
        Power to reach.
    ratio : float, default 1.0
        Design mode: ``n_right / n_left``.
    data : DataFrame, optional
        Data mode: a pilot sample. Requires ``y`` and ``x``.
    y, x : str, optional
        Data mode: outcome and running-variable columns.
    c : float, default 0.0
        Data mode: cutoff.
    **rdrobust_kwargs
        Data mode: options passed to ``sp.rdrobust``.

    Returns
    -------
    RDSampSiResult
        ``n_left``, ``n_right``, ``n_total`` (data mode: observations inside
        the bandwidth window on each side), ``tau``, ``target_power``,
        ``alpha``.

    Notes
    -----
    Data mode (Cattaneo, Titiunik and Vazquez-Bare 2019, section 2.2): with
    the robust bias-corrected side variances ``V-``, ``V+`` scaled by
    ``n h^(1+2 deriv)``, the total sample ``n*`` is the smallest integer at
    least the root of ``power(tau; sqrt(sum V/h^(1+2 deriv) / n)) =
    target_power``. The treated share of the window sample is
    ``rho = sqrt(V+)/(sqrt(V-) + sqrt(V+))``; the window sample is
    ``M = n* / (rho N+/Nh+ + (1-rho) N-/Nh-)`` and each side is
    ``ceil(rho M)`` / ``ceil((1-rho) M)``. Design mode solves the same power
    equation with ``n_right = ratio n_left`` and the design-mode standard
    error of :func:`rdpower`; each side is rounded up.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.rdsampsi(tau=0.15, target_power=0.80)
    >>> (r.n_left, r.n_right, r.n_total)
    (489, 489, 978)

    References
    ----------
    cattaneo2019power
    """
    alpha = _check_alpha(alpha)
    target = _check_target(target_power, alpha)
    tau = float(tau)
    if not (np.isfinite(tau) and tau != 0.0):
        raise _bad(
            f"rdsampsi: tau must be a non-zero finite number; got {tau!r} "
            "(no sample size detects a zero effect).",
            "Pass the effect size you want to detect, e.g. tau=0.2.",
            tau=tau,
        )

    if _data_columns(data, y, x, "rdsampsi"):
        assert y is not None and x is not None  # _data_columns checked them
        return _sampsi_data(data, y, x, c, tau, alpha, target, rdrobust_kwargs)

    _check_positive(
        var_left=var_left,
        var_right=var_right,
        h_left=h_left,
        h_right=h_right,
        ratio=ratio,
    )
    # se^2(n_left) = kappa (vl/hl + vr/(ratio hr)) / n_left
    numerator = _DESIGN_KAPPA * (var_left / h_left + var_right / (ratio * h_right))
    root = _root_of_power(numerator, abs(tau), alpha, target)
    nl = max(1, math.ceil(root))
    nr = max(1, math.ceil(ratio * nl))
    return RDSampSiResult(
        n_left=nl,
        n_right=nr,
        n_total=nl + nr,
        tau=tau,
        target_power=target,
        alpha=alpha,
    )


def _side_variances(
    data: Any,
    y: str,
    x: str,
    c: float,
    fit: Any,
    kwargs: Dict[str, Any],
) -> Dict[str, float]:
    """Robust bias-corrected variance of each side's contribution.

    Recomputed with the shared CCT primitive at the fit's bandwidths and
    checked against the fit's robust variance.
    """
    from ._cct_inference import cct_bias_corrected

    unsupported = sorted(set(kwargs) - _SAMPSI_DATA_OPTIONS)
    if unsupported:
        raise _bad(
            "rdsampsi(data=...): the per-side variance split is not available "
            f"with the rdrobust option(s) {unsupported}.",
            "Drop these options, or use sp.rdpower(data=...) for the power "
            "at the observed sample size.",
            unsupported=unsupported,
        )
    mi = fit.model_info
    h_l, h_r = _as_pair(mi["bandwidth_h"])
    b_l, b_r = _as_pair(mi["bandwidth_b"])

    def col(name: Optional[str]) -> Optional[np.ndarray]:
        if name is None:
            return None
        return np.asarray(data[name], dtype=float)

    covs = kwargs.get("covs")
    covs_arr = None
    if covs:
        covs_list = [covs] if isinstance(covs, str) else list(covs)
        covs_arr = np.asarray(data[covs_list], dtype=float)
    cluster = kwargs.get("cluster")
    comp: Dict[str, float] = {}
    cct_bias_corrected(
        np.asarray(data[y], dtype=float),
        np.asarray(data[x], dtype=float),
        float(c),
        h_l,
        h_r,
        b_l,
        b_r,
        int(mi["polynomial_p"]),
        int(mi["polynomial_q"]),
        int(mi.get("deriv", 0)),
        str(mi["kernel"]),
        covs=covs_arr,
        vce=str(mi.get("vce", kwargs.get("vce", "nn"))),
        cluster=None if cluster is None else np.asarray(data[cluster]),
        components=comp,
        fuzzy=col(kwargs.get("fuzzy")),
        weights=col(kwargs.get("weights")),
    )
    total = comp["V_rb_left"] + comp["V_rb_right"]
    if not math.isclose(total, float(fit.se) ** 2, rel_tol=1e-6):
        raise _bad(
            "rdsampsi(data=...): the per-side variances do not add up to "
            f"rdrobust's robust variance ({total:.6g} vs {float(fit.se) ** 2:.6g}).",
            "Drop non-default rdrobust options, or use sp.rdpower(data=...).",
            per_side_total=total,
            robust_variance=float(fit.se) ** 2,
        )
    comp.update(h_left=h_l, h_right=h_r)
    return comp


def _sampsi_data(
    data: Any,
    y: str,
    x: str,
    c: float,
    tau: float,
    alpha: float,
    target: float,
    kwargs: Dict[str, Any],
) -> RDSampSiResult:
    if float(kwargs.get("scalepar", 1.0)) != 1.0:
        raise _bad(
            "rdsampsi(data=...) does not support scalepar.",
            "Rescale tau instead of the estimator.",
        )
    fit = _fit_rdrobust(data, y, x, c, kwargs)
    mi = fit.model_info
    comp = _side_variances(data, y, x, c, fit, kwargs)
    nu = int(mi.get("deriv", 0))
    n_l, n_r = int(mi["n_left"]), int(mi["n_right"])
    nh_l, nh_r = int(mi["n_effective_left"]), int(mi["n_effective_right"])
    n = n_l + n_r
    if min(nh_l, nh_r) <= 0:
        raise _bad(
            "rdsampsi(data=...): a side has no observation inside the bandwidth.",
            "Use a larger bandwidth (h=) or more data near the cutoff.",
            n_effective_left=nh_l,
            n_effective_right=nh_r,
        )
    h_l, h_r = comp["h_left"], comp["h_right"]
    # Scaled variances of the paper: V = v * n * h^(1 + 2 nu).
    v_l = comp["V_rb_left"] * n * h_l ** (1 + 2 * nu)
    v_r = comp["V_rb_right"] * n * h_r ** (1 + 2 * nu)
    numerator = v_r / h_r ** (1 + 2 * nu) + v_l / h_l ** (1 + 2 * nu)
    n_star = math.ceil(_root_of_power(numerator, abs(tau), alpha, target))
    rho = math.sqrt(v_r) / (math.sqrt(v_l) + math.sqrt(v_r))
    window = n_star / (rho * n_r / nh_r + (1.0 - rho) * n_l / nh_l)
    m_r = math.ceil(rho * window)
    m_l = math.ceil((1.0 - rho) * window)
    return RDSampSiResult(
        n_left=m_l,
        n_right=m_r,
        n_total=m_l + m_r,
        tau=tau,
        target_power=target,
        alpha=alpha,
    )
