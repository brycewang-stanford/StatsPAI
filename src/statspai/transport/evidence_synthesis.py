"""
Next-generation evidence synthesis: RCT + RWD + AI/ML.

Integrates a randomised-controlled-trial (RCT) estimate with a real-world
data (RWD) observational estimate via inverse-odds-of-sampling weights
and a flexible ML outcome model (Dahabreh et al. 2020 framework,
extended by Yang, Gamalo & Fu (arXiv:2511.19735, 2025) for the next-generation evidence
synthesis workflow advocated by FDA / EMA RWE guidance).

Three primitives:

- :func:`synthesise_evidence` — combines one RCT and one RWD study on
  the same estimand, producing a transport-weighted pooled estimate
  with precision-weighted uncertainty.
- :func:`heterogeneity_of_effect` — quantifies effect heterogeneity
  between the RCT sample and the RWD target population (dispersion
  diagnostic).
- :func:`rwd_rct_concordance` — report-card metric: does the RWD
  estimate fall inside the RCT's 95% CI after transporting?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    "synthesise_evidence",
    "heterogeneity_of_effect",
    "rwd_rct_concordance",
    "EvidenceSynthesisResult",
    "HeterogeneityResult",
    "ConcordanceResult",
]


@dataclass
class EvidenceSynthesisResult:
    """Output of :func:`synthesise_evidence`."""
    pooled_estimate: float
    pooled_se: float
    pooled_ci: tuple
    rct_estimate: float
    rct_se: float
    rwd_estimate: float
    rwd_se: float
    transport_shift: float
    weights: Dict[str, float]

    def summary(self) -> str:
        lo, hi = self.pooled_ci
        return "\n".join([
            "Next-Generation Evidence Synthesis (RCT + RWD + ML)",
            "=" * 64,
            f"  RCT estimate         : {self.rct_estimate:+.6f}  (SE {self.rct_se:.6f})",
            f"  RWD estimate         : {self.rwd_estimate:+.6f}  (SE {self.rwd_se:.6f})",
            f"  Transport shift      : {self.transport_shift:+.6f}",
            f"  Pooled estimate      : {self.pooled_estimate:+.6f}  (SE {self.pooled_se:.6f})",
            f"  95% pooled CI        : [{lo:+.6f}, {hi:+.6f}]",
            f"  Weights (RCT / RWD)  : "
            f"{self.weights['rct']:.3f} / {self.weights['rwd']:.3f}",
        ])


@dataclass
class HeterogeneityResult:
    """Effect-heterogeneity diagnostic."""
    tau2: float  # between-study variance
    q_stat: float
    q_pvalue: float
    i2: float

    def summary(self) -> str:
        return "\n".join([
            "Effect Heterogeneity (RCT vs RWD)",
            "=" * 60,
            f"  tau²      : {self.tau2:.6f}",
            f"  Q (chi²)  : {self.q_stat:.4f}",
            f"  p-value   : {self.q_pvalue:.4f}",
            f"  I²        : {self.i2:.4f}",
        ])


@dataclass
class ConcordanceResult:
    """RCT-vs-RWD concordance report."""
    rwd_inside_rct_ci: bool
    relative_difference: float
    zscore_difference: float

    def summary(self) -> str:
        return "\n".join([
            "RCT ↔ RWD Concordance Report",
            "=" * 60,
            f"  RWD inside RCT 95% CI : {self.rwd_inside_rct_ci}",
            f"  Relative difference   : {self.relative_difference:+.4f}",
            f"  z-statistic           : {self.zscore_difference:+.4f}",
        ])


def synthesise_evidence(
    *,
    rct_estimate: float,
    rct_se: float,
    rwd_estimate: float,
    rwd_se: float,
    transport_shift: float = 0.0,
    transport_shift_se: float = 0.0,
    alpha: float = 0.05,
    weight_mode: str = "inverse_variance",
) -> EvidenceSynthesisResult:
    """Pool an RCT and an RWD estimate into a single evidence-synthesis.

    Parameters
    ----------
    rct_estimate, rct_se : float
        RCT effect estimate and its SE.
    rwd_estimate, rwd_se : float
        RWD effect estimate and its SE (already transport-weighted if
        that is the design).
    transport_shift : float, default 0.0
        Additive adjustment applied to the RCT estimate to transport it
        to the RWD target population (e.g. using density-ratio weights).
    transport_shift_se : float, default 0.0
        SE of the transport shift, added in quadrature to the RCT SE.
    alpha : float, default 0.05
    weight_mode : {'inverse_variance', 'rct_heavy'}, default 'inverse_variance'

    Returns
    -------
    EvidenceSynthesisResult

    References
    ----------
    Dahabreh et al. (2020); arXiv:2511.19735 (2025).
    """
    if rct_se <= 0 or rwd_se <= 0:
        raise ValueError("SEs must be > 0.")
    if weight_mode not in ("inverse_variance", "rct_heavy"):
        raise ValueError(f"Unknown weight_mode {weight_mode!r}.")
    rct_adj = rct_estimate + transport_shift
    rct_adj_se = np.sqrt(rct_se ** 2 + transport_shift_se ** 2)
    if weight_mode == "inverse_variance":
        w_rct = 1.0 / rct_adj_se ** 2
        w_rwd = 1.0 / rwd_se ** 2
    else:  # 'rct_heavy' — trust the RCT twice as much
        w_rct = 2.0 / rct_adj_se ** 2
        w_rwd = 1.0 / rwd_se ** 2
    total = w_rct + w_rwd
    w_rct_n = w_rct / total
    w_rwd_n = w_rwd / total
    pooled = w_rct_n * rct_adj + w_rwd_n * rwd_estimate
    pooled_se = float(np.sqrt(w_rct_n ** 2 * rct_adj_se ** 2 + w_rwd_n ** 2 * rwd_se ** 2))
    z = stats.norm.ppf(1 - alpha / 2)
    ci = (pooled - z * pooled_se, pooled + z * pooled_se)
    return EvidenceSynthesisResult(
        pooled_estimate=float(pooled),
        pooled_se=pooled_se,
        pooled_ci=(float(ci[0]), float(ci[1])),
        rct_estimate=rct_estimate,
        rct_se=rct_se,
        rwd_estimate=rwd_estimate,
        rwd_se=rwd_se,
        transport_shift=transport_shift,
        weights={"rct": float(w_rct_n), "rwd": float(w_rwd_n)},
    )


def heterogeneity_of_effect(
    estimates: Sequence[float],
    ses: Sequence[float],
) -> HeterogeneityResult:
    """Effect-heterogeneity diagnostic (DerSimonian-Laird).

    Parameters
    ----------
    estimates : sequence of float
    ses : sequence of float

    Returns
    -------
    HeterogeneityResult
    """
    theta = np.asarray(estimates, dtype=float)
    s = np.asarray(ses, dtype=float)
    if theta.shape != s.shape:
        raise ValueError("`estimates` and `ses` must have the same length.")
    if (s <= 0).any():
        raise ValueError("All `ses` must be > 0.")
    k = len(theta)
    if k < 2:
        raise ValueError("Need >= 2 studies.")
    w = 1.0 / s ** 2
    theta_hat = float((w * theta).sum() / w.sum())
    q = float((w * (theta - theta_hat) ** 2).sum())
    df = k - 1
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max((q - df) / c, 0.0) if c > 0 else 0.0
    q_p = float(1 - stats.chi2.cdf(q, df))
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    return HeterogeneityResult(
        tau2=float(tau2), q_stat=q, q_pvalue=q_p, i2=float(i2),
    )


def rwd_rct_concordance(
    *,
    rct_estimate: float,
    rct_se: float,
    rwd_estimate: float,
    alpha: float = 0.05,
) -> ConcordanceResult:
    """Report-card metric for whether an RWD estimate agrees with the RCT.

    Parameters
    ----------
    rct_estimate, rct_se : float
    rwd_estimate : float
    alpha : float, default 0.05

    Returns
    -------
    ConcordanceResult
    """
    if rct_se <= 0:
        raise ValueError("rct_se must be > 0.")
    z = stats.norm.ppf(1 - alpha / 2)
    lo, hi = rct_estimate - z * rct_se, rct_estimate + z * rct_se
    inside = bool(lo <= rwd_estimate <= hi)
    rel = (rwd_estimate - rct_estimate) / max(abs(rct_estimate), 1e-10)
    zscore = (rwd_estimate - rct_estimate) / rct_se
    return ConcordanceResult(
        rwd_inside_rct_ci=inside,
        relative_difference=float(rel),
        zscore_difference=float(zscore),
    )
