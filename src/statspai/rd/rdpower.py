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

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp_stats


@dataclass
class RDPowerResult:
    power: float
    mde: float
    alpha: float
    n_left: int
    n_right: int
    se: float
    tau: float                       # assumed or hypothesised effect

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
class RDSampSiResult:
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


def _rd_se(n_left: int, n_right: int,
           var_left: float, var_right: float,
           h_left: float, h_right: float) -> float:
    """Approximate SE for the RD point estimator (local-linear).

    SE ≈ sqrt(C_K * [σ²_L / (n_L h_L) + σ²_R / (n_R h_R)])

    where C_K = ∫K²(u)du is the kernel constant (~0.35 for triangular).
    """
    Ck = 0.35
    return float(np.sqrt(Ck * (var_left / max(n_left * h_left, 1.0) +
                                var_right / max(n_right * h_right, 1.0))))


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
) -> RDPowerResult:
    """Power of an RD design given sample size and effect size.

    Parameters
    ----------
    tau : float
        Hypothesised treatment effect at the cutoff.
    var_left, var_right : float
        Outcome variance on each side of the cutoff.
    n_left, n_right : int
        Available sample size on each side.
    h_left, h_right : float
        Bandwidth fractions (proportion of running-variable support used).
    alpha : float
        Significance level.
    target_power : float, optional
        If set, compute MDE for this target power instead.
    """
    se = _rd_se(n_left, n_right, var_left, var_right, h_left, h_right)
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)

    # Power = P(reject | tau)
    power = float(1.0 - sp_stats.norm.cdf(z_alpha - tau / se)
                  + sp_stats.norm.cdf(-z_alpha - tau / se))

    # MDE at 80% power
    z80 = sp_stats.norm.ppf(0.80)
    mde_80 = float((z_alpha + z80) * se)

    if target_power is not None:
        z_pow = sp_stats.norm.ppf(target_power)
        mde = float((z_alpha + z_pow) * se)
    else:
        mde = mde_80

    return RDPowerResult(
        power=power, mde=mde, alpha=alpha,
        n_left=n_left, n_right=n_right, se=se, tau=tau,
    )


def rdsampsi(
    tau: float,
    var_left: float = 1.0,
    var_right: float = 1.0,
    h_left: float = 0.5,
    h_right: float = 0.5,
    alpha: float = 0.05,
    target_power: float = 0.80,
    ratio: float = 1.0,
) -> RDSampSiResult:
    """Minimum sample size for a given power in an RD design.

    Parameters
    ----------
    ratio : float, default 1.0
        ``n_right / n_left``. Default 1.0 assumes equal allocation.
    """
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_pow = sp_stats.norm.ppf(target_power)
    Ck = 0.35

    # SE = sqrt(Ck * [σ²_L / (n_L h_L) + σ²_R / (r n_L h_R)])
    # Solve for n_L: SE ≤ τ / (z_α + z_β)
    se_target = abs(tau) / (z_alpha + z_pow)
    # SE² = Ck * (var_L/(n_L h_L) + var_R/(ratio n_L h_R))
    # n_L = Ck * (var_L/h_L + var_R/(ratio h_R)) / SE²_target
    n_left = int(np.ceil(
        Ck * (var_left / h_left + var_right / (ratio * h_right)) / se_target ** 2
    ))
    n_right = int(np.ceil(ratio * n_left))

    return RDSampSiResult(
        n_left=n_left, n_right=n_right,
        n_total=n_left + n_right,
        tau=tau, target_power=target_power, alpha=alpha,
    )
