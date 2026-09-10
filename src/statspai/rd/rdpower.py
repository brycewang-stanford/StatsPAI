"""RD Power and Sample-Size Calculations (Cattaneo, Titiunik & Vazquez-Bare, 2019).

Provides two entry points:

- :func:`rdpower`  — compute power of an RD design given a sample size
  (or a MDE given target power).
- :func:`rdsampsi` — compute minimum sample size for a target power.

Both are based on the asymptotic distribution of the local-polynomial
point estimator (as in ``rdrobust``). They assume MSE-optimal bandwidth
for the variance calibration and a normal-approximation critical region.

References
----------
Cattaneo, M.D., Titiunik, R. & Vazquez-Bare, G. (2019).
  "Power calculations for regression-discontinuity designs."
  *Stata Journal*, 19(1), 210–245. [@cattaneo2019power]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy import stats as sp_stats

from .._result_serialize import ResultProtocolMixin

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


@dataclass
class RDPowerResult(ResultProtocolMixin):
    power: float
    mde: float
    alpha: float
    n_left: int
    n_right: int
    se: float
    tau: float  # assumed or hypothesised effect

    def summary(self) -> str:
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
    n_left: int
    n_right: int
    n_total: int
    tau: float
    target_power: float
    alpha: float

    def summary(self) -> str:
        return (
            f"RD Sample Size: N = {self.n_total} "
            f"({self.n_left} left + {self.n_right} right) "
            f"for τ = {self.tau:.4f} at power = {self.target_power:.0%}, "
            f"α = {self.alpha}"
        )

    def __repr__(self) -> str:
        return self.summary()


def _rd_se(
    n_left: int,
    n_right: int,
    var_left: float,
    var_right: float,
    h_left: float,
    h_right: float,
) -> float:
    """Approximate SE for the RD point estimator (local-linear).

    SE ≈ sqrt(C_K * [σ²_L / (n_L h_L) + σ²_R / (n_R h_R)])

    where C_K = ∫K²(u)du is the kernel constant (~0.35 for triangular).
    """
    Ck = 0.35
    return float(
        np.sqrt(
            Ck
            * (
                var_left / max(n_left * h_left, 1.0)
                + var_right / max(n_right * h_right, 1.0)
            )
        )
    )


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

    Two modes, matching R ``rdpower``:

    * **Design mode** (the default): supply ``var_*``, ``n_*`` and ``h_*``
      and the standard error is built from them. Useful before data exist.
    * **Data mode**: supply ``data``, ``y`` and ``x``. The standard error is
      then the robust bias-corrected SE from :func:`statspai.rdrobust` on
      that data, which is what R's ``rdpower(data = ...)`` does and the only
      mode that reproduces its numbers. Extra keyword arguments (``p``,
      ``kernel``, ``vce``, ``cluster``, ``covs``, ...) are forwarded to
      ``rdrobust``.

    Parameters
    ----------
    tau : float
        Hypothesised treatment effect at the cutoff.
    var_left, var_right : float
        Outcome variance on each side of the cutoff. Design mode only.
    n_left, n_right : int
        Available sample size on each side. Design mode only.
    h_left, h_right : float
        Bandwidth fractions (proportion of running-variable support used).
        Design mode only.
    alpha : float
        Significance level.
    target_power : float, optional
        If set, compute MDE for this target power instead.
    data : pandas.DataFrame, optional
        Enables data mode. Requires ``y`` and ``x``.
    y, x : str, optional
        Outcome and running-variable column names, for data mode.
    c : float, default 0.0
        Cutoff, for data mode.
    **rdrobust_kwargs
        Forwarded to :func:`statspai.rdrobust` in data mode.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.rdpower(tau=0.15, n_left=500, n_right=500)
    >>> round(res.power, 3)
    0.809
    >>> round(res.mde, 3)
    0.148
    >>> res90 = sp.rdpower(tau=0.15, n_left=500, n_right=500,
    ...                    target_power=0.90)
    >>> round(res90.mde, 3)
    0.172
    """
    if data is not None:
        if y is None or x is None:
            raise ValueError(
                "data mode needs both y= and x= (the outcome and running "
                "variable column names)"
            )
        from .rdrobust import rdrobust as _rdrobust

        _fit = _rdrobust(data, y=y, x=x, c=c, **rdrobust_kwargs)
        # The robust bias-corrected SE, which is the one R's rdpower uses.
        se = float(_fit.detail["se"][1])
        _mi = _fit.model_info
        n_left = int(_mi.get("n_left", n_left))
        n_right = int(_mi.get("n_right", n_right))
    elif y is not None or x is not None:
        raise ValueError(
            "y=/x= only mean something with data=; pass data= as well, or "
            "use design mode (var_left/var_right/n_left/n_right/h_left/h_right)"
        )
    else:
        se = _rd_se(n_left, n_right, var_left, var_right, h_left, h_right)
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)

    # Power = P(reject | tau)
    power = float(
        1.0
        - sp_stats.norm.cdf(z_alpha - tau / se)
        + sp_stats.norm.cdf(-z_alpha - tau / se)
    )

    # MDE at 80% power
    z80 = sp_stats.norm.ppf(0.80)
    mde_80 = float((z_alpha + z80) * se)

    if target_power is not None:
        z_pow = sp_stats.norm.ppf(target_power)
        mde = float((z_alpha + z_pow) * se)
    else:
        mde = mde_80

    return RDPowerResult(
        power=power,
        mde=mde,
        alpha=alpha,
        n_left=n_left,
        n_right=n_right,
        se=se,
        tau=tau,
    )


def _power_newton_raphson(
    x0: float, tau: float, stilde: float, z: float, beta: float
) -> int:
    """Smallest sample size reaching power ``beta``, as ``rdpower`` solves it.

    A transcription of ``rdpower:::rdpower.powerNR``. Two details are load
    bearing and neither is guessable from the power formula alone:

    * the loop converges on the **power**, not on ``x`` --
      ``tol = |power(x1) - beta|`` against machine epsilon;
    * the return is ``ceiling(x1)``, and the ceiling happens *here*, before
      the total is divided by the design factor. Rounding at the end
      instead moves the answer: on the rdlocrand senate fixture at
      ``tau = 3`` it lands 1015 where ``rdpower`` reports 1016.

    The guard loops are R's: nudge ``x0`` by 1.2x / 0.8x while the
    derivative is numerically flat, and halve the Newton step while it
    would put ``x1`` below 2.
    """

    def power(m: float) -> float:
        a = math.sqrt(m) * tau / stilde
        return float(1 - sp_stats.norm.cdf(z - a) + sp_stats.norm.cdf(-z - a))

    def power_dot(m: float) -> float:
        a = math.sqrt(m) * tau / stilde
        d = tau / (2 * stilde * math.sqrt(m))
        return float(d * (sp_stats.norm.pdf(z - a) + sp_stats.norm.pdf(-z - a)))

    tol = 1.0
    x1 = x0
    guard = 0
    while tol > np.finfo(float).eps:
        guard += 1
        if guard > 10_000:  # pragma: no cover - defensive
            raise RuntimeError(
                "rdsampsi: the sample-size solve did not converge in 10000 "
                "iterations; report the inputs rather than trusting the "
                "last iterate."
            )
        k = 1.0
        dot0 = power_dot(x0)
        power0 = power(x0)
        while dot0 < 1e-5:
            x0 = 1.2 * x0 if power0 <= beta else 0.8 * x0
            dot0 = power_dot(x0)
            power0 = power(x0)
        x1 = x0 - (power0 - beta) / dot0
        while x1 < 2:
            x1 = x0 - k * (power0 - beta) / dot0
            k /= 2
        tol = abs(power(x1) - beta)
        x0 = x1
    return int(math.ceil(x1))


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

    Two modes, mirroring ``sp.rdpower`` and R's ``rdsampsi``:

    * **Design mode** (the default): supply ``var_*``, ``h_*`` and
      ``ratio``. Useful before data exist.
    * **Data mode**: supply ``data``, ``y`` and ``x``. The variances,
      bandwidths and effective sample sizes are estimated from that data
      by ``sp.rdrobust``, which is what R's ``rdsampsi(data = ...)`` does
      and the only mode comparable against it.

    .. versionadded:: 1.27.0
       Data mode. ``sp.rdpower`` has had it since the CCT cascade work;
       its sibling did not, so R's reference call had no StatsPAI
       counterpart and the ``rdsampsi`` rows in
       ``tests/reference_parity/_fixtures/rdlocrand_R.json`` could not be
       asserted against anything.

    Parameters
    ----------
    ratio : float, default 1.0
        ``n_right / n_left``. Default 1.0 assumes equal allocation.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.rdsampsi(tau=0.15, target_power=0.80)
    >>> res.n_total
    978
    >>> (res.n_left, res.n_right)
    (489, 489)
    """
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_pow = sp_stats.norm.ppf(target_power)
    Ck = 0.35

    if data is not None:
        return _rdsampsi_from_data(
            tau=tau,
            data=data,
            y=y,
            x=x,
            c=c,
            alpha=alpha,
            target_power=target_power,
            z_alpha=z_alpha,
            **rdrobust_kwargs,
        )
    if y is not None or x is not None:
        raise ValueError(
            "y=/x= only mean something with data=; pass data= as well, or "
            "use design mode (var_left/var_right/h_left/h_right/ratio)"
        )

    # SE = sqrt(Ck * [σ²_L / (n_L h_L) + σ²_R / (r n_L h_R)])
    # Solve for n_L: SE ≤ τ / (z_α + z_β)
    se_target = abs(tau) / (z_alpha + z_pow)
    # SE² = Ck * (var_L/(n_L h_L) + var_R/(ratio n_L h_R))
    # n_L = Ck * (var_L/h_L + var_R/(ratio h_R)) / SE²_target
    n_left = int(
        np.ceil(
            Ck * (var_left / h_left + var_right / (ratio * h_right)) / se_target**2
        )
    )
    n_right = int(np.ceil(ratio * n_left))

    return RDSampSiResult(
        n_left=n_left,
        n_right=n_right,
        n_total=n_left + n_right,
        tau=tau,
        target_power=target_power,
        alpha=alpha,
    )


def _rdsampsi_from_data(
    *,
    tau: float,
    data: "pd.DataFrame",
    y: Optional[str],
    x: Optional[str],
    c: float,
    alpha: float,
    target_power: float,
    z_alpha: float,
    **rdrobust_kwargs: object,
) -> RDSampSiResult:
    """Data mode: the required N implied by a fitted RD design.

    A transcription of R ``rdpower::rdsampsi(data = ...)``. The chain is

    ``stilde = sqrt(V_rbc)``  with  ``V_rbc = v_l / h_l + v_r / h_r``
    and ``v_side = N * h_side * V_rb_side`` -- which for the default
    ``deriv = 0`` and no rescaled bandwidth collapses to
    ``stilde = se_robust * sqrt(N)``, i.e. the standard error carried to
    the full-sample scale;

    ``m``  the Newton-Raphson sample size at which power reaches the
    target, **ceilinged there** (see ``_power_newton_raphson``);

    ``nratio = sqrt(v_r) / (sqrt(v_r) + sqrt(v_l))`` -- allocation between
    the sides by the square root of their variances, *not* by their
    observed counts. Allocating by counts instead reproduces the total to
    about 1% while splitting the sides visibly wrong (8567/7687 against
    R's 9135/7256 on the senate fixture), which is the failure mode a
    total-only check would have missed;

    ``denom = nratio * n_right / n_h_right + (1 - nratio) * n_left / n_h_left``
    rescales from the in-bandwidth sample to the whole sample, and
    ``M = m / denom`` splits by ``nratio`` with each side ceilinged.
    """
    if y is None or x is None:
        raise ValueError(
            "data mode needs both y= and x= (the outcome and running "
            "variable column names)"
        )

    from ._cct_bandwidth import cct_bias_corrected
    from .rdrobust import rdrobust as _rdrobust

    fit = _rdrobust(data, y=y, x=x, c=c, **rdrobust_kwargs)
    mi = fit.model_info

    yv = np.asarray(data[y], dtype=float)
    xv = np.asarray(data[x], dtype=float)
    keep = np.isfinite(yv) & np.isfinite(xv)
    yv, xv = yv[keep], xv[keep]
    n_total_obs = int(keep.sum())

    h_l = float(
        mi["bandwidth_h"]["left"]
        if isinstance(mi["bandwidth_h"], dict)
        else mi["bandwidth_h"]
    )
    h_r = float(
        mi["bandwidth_h"]["right"]
        if isinstance(mi["bandwidth_h"], dict)
        else mi["bandwidth_h"]
    )
    b_l = float(
        mi["bandwidth_b"]["left"]
        if isinstance(mi["bandwidth_b"], dict)
        else mi["bandwidth_b"]
    )
    b_r = float(
        mi["bandwidth_b"]["right"]
        if isinstance(mi["bandwidth_b"], dict)
        else mi["bandwidth_b"]
    )

    components: dict = {}
    cct_bias_corrected(
        yv,
        xv,
        c,
        h_l,
        h_r,
        b_l,
        b_r,
        int(mi["polynomial_p"]),
        int(mi["polynomial_q"]),
        int(mi.get("deriv", 0) or 0),
        str(mi["kernel"]),
        components=components,
    )

    v_l = n_total_obs * h_l * components["V_rb_left"]
    v_r = n_total_obs * h_r * components["V_rb_right"]
    stilde = math.sqrt(v_l / h_l + v_r / h_r)

    n_plus = int((xv >= c).sum())
    n_minus = int((xv < c).sum())
    n_h_right = int(((xv >= c) & (xv <= c + h_r)).sum())
    n_h_left = int(((xv < c) & (xv >= c - h_l)).sum())
    if min(n_h_left, n_h_right) == 0:  # pragma: no cover - defensive
        raise ValueError(
            "rdsampsi: the selected bandwidth contains no observations on "
            "one side of the cutoff; the required sample size is undefined."
        )

    root_l, root_r = math.sqrt(v_l), math.sqrt(v_r)
    nratio = root_r / (root_r + root_l)

    m = _power_newton_raphson(
        float(n_total_obs), abs(tau), stilde, z_alpha, target_power
    )
    denom = nratio * n_plus / n_h_right + (1 - nratio) * n_minus / n_h_left
    total_scaled = m / denom
    n_right = int(math.ceil(total_scaled * nratio))
    n_left = int(math.ceil(total_scaled * (1 - nratio)))

    return RDSampSiResult(
        n_left=n_left,
        n_right=n_right,
        n_total=n_left + n_right,
        tau=tau,
        target_power=target_power,
        alpha=alpha,
    )
