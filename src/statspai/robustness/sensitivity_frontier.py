"""
Frontier sensitivity analysis extensions (v0.9.17).

Three new sensitivity-analysis primitives complementing the E-value /
Oster / Rosenbaum / Sensemakr dashboard in
:mod:`statspai.robustness.unified_sensitivity`:

1. :func:`copula_sensitivity` — Copula-based normalising-flow-style
   sensitivity (Balgi, Braun, Peña & Daoud arXiv:2508.08752, 2025). Bounds the
   treatment-effect bias under a Gaussian-copula dependence between the
   unobserved ``U`` and the outcome ``Y``, parametrised by a correlation
   ``rho`` that the user varies on a grid.
2. :func:`survival_sensitivity` — Nonparametric sensitivity for survival
   outcomes (Hu & Westling arXiv:2511.01412, 2025). Converts
   hazard-ratio bounds into shifted Kaplan-Meier differences.
3. :func:`calibrate_confounding_strength` — E-value-style calibration of
   the confounding strength required to explain the observed effect
   (Baitairian et al. arXiv:2510.16560, 2025 update of E-value to
   ML-estimated effects).

All three share a simple ``(estimate, se)`` interface: they take a
point estimate and its standard error, plus domain-specific parameters,
and return a tidy :class:`FrontierSensitivityResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    "copula_sensitivity",
    "survival_sensitivity",
    "calibrate_confounding_strength",
    "FrontierSensitivityResult",
]


@dataclass
class FrontierSensitivityResult:
    """Container for frontier sensitivity analysis."""
    method: str
    estimate: float
    se: float
    curve: pd.DataFrame      # columns depend on method
    breakpoint: Optional[float]
    interpretation: str

    def summary(self) -> str:
        lines = [
            f"Frontier Sensitivity Analysis — {self.method}",
            "=" * 64,
            f"  Observed estimate : {self.estimate:+.6f}  (SE {self.se:.6f})",
        ]
        if self.breakpoint is not None:
            lines.append(f"  Breakpoint        : {self.breakpoint:.4f}")
        lines += [
            f"  Interpretation    : {self.interpretation}",
            "",
            "Sensitivity curve:",
            self.curve.to_string(index=False, float_format="%.4f"),
        ]
        return "\n".join(lines)


def copula_sensitivity(
    estimate: float,
    se: float,
    *,
    sigma_u: float = 1.0,
    sigma_y: float = 1.0,
    rho_grid: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
) -> FrontierSensitivityResult:
    """Gaussian-copula sensitivity to unobserved confounding.

    Under a Gaussian copula with correlation ``rho`` between U (one
    latent unit-level confounder) and Y (outcome), the bias in an OLS /
    DML point estimate scales linearly with ``rho``:

        bias(rho) = rho * sigma_u * sigma_y / sigma_D²  ≈ rho * sigma_u * sigma_y

    under the normalisation sigma_D = 1. The adjusted estimate is
    ``estimate - bias(rho)``; we sweep ``rho`` on a grid to find the
    breakpoint ``rho*`` that zeros the effect.

    Parameters
    ----------
    estimate, se : float
    sigma_u, sigma_y : float, default 1.0
        Standard deviations of the latent confounder and the outcome.
        With default values the bias coefficient is numerically equal to
        ``rho``, matching Chernozhukov-Cinelli-Hazlett's "percentile
        scaling."
    rho_grid : sequence of float, optional
        Correlation grid. Defaults to ``np.linspace(-0.5, 0.5, 21)``.
    alpha : float, default 0.05

    Returns
    -------
    FrontierSensitivityResult

    References
    ----------
    Balgi, Braun, Peña & Daoud (arXiv:2508.08752, 2025). [@balgi2025sensitivity]
    """
    if rho_grid is None:
        rho_grid = np.linspace(-0.5, 0.5, 21)
    rho_grid = np.array(rho_grid, dtype=float)
    bias = rho_grid * sigma_u * sigma_y
    adjusted = estimate - bias
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low = adjusted - z * se
    ci_high = adjusted + z * se
    # Breakpoint: smallest |rho| such that the adjusted estimate's CI
    # includes zero.
    covers_zero = (ci_low <= 0) & (0 <= ci_high)
    try:
        bp = float(rho_grid[covers_zero][np.argmin(np.abs(rho_grid[covers_zero]))])
    except ValueError:
        bp = None
    curve = pd.DataFrame({
        "rho": rho_grid,
        "bias": bias,
        "adjusted_estimate": adjusted,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": (ci_low > 0) | (ci_high < 0),
    })
    if bp is None:
        interpretation = (
            "No correlation in the grid makes the adjusted effect cross zero; "
            "effect is robust."
        )
    else:
        interpretation = (
            f"A Gaussian-copula correlation of |rho| = {abs(bp):.3f} suffices "
            "to explain the observed effect away."
        )
    return FrontierSensitivityResult(
        method="copula_gaussian",
        estimate=estimate,
        se=se,
        curve=curve,
        breakpoint=bp,
        interpretation=interpretation,
    )


def survival_sensitivity(
    log_hr: float,
    se_log_hr: float,
    *,
    gamma_grid: Optional[Sequence[float]] = None,
    baseline_survival_t: float = 0.5,
    alpha: float = 0.05,
) -> FrontierSensitivityResult:
    """Nonparametric sensitivity for survival / hazard-ratio outcomes.

    Extends Rosenbaum's Gamma bounds to hazard ratios and converts them
    into shifted survival differences at a chosen time ``t``.

    Given an observed log hazard ratio ``log_hr`` with SE ``se_log_hr``,
    bound the worst-case log-HR at sensitivity parameter ``Γ``:

        log_hr_worst(Γ) = log_hr − log(Γ),  log_hr_best(Γ) = log_hr + log(Γ)

    and translate the worst case into a survival shift at time ``t``
    using the proportional-hazards identity
    ``S_1(t) = S_0(t) ^ exp(log_hr)``.

    Parameters
    ----------
    log_hr, se_log_hr : float
    gamma_grid : sequence of float, optional
        Gamma (≥ 1) values. Defaults to ``np.linspace(1.0, 3.0, 21)``.
    baseline_survival_t : float, default 0.5
        Baseline S_0(t) used to report Δ survival at time t.
    alpha : float, default 0.05

    References
    ----------
    Hu & Westling (arXiv:2511.01412, 2025). [@hu2025nonparametric]
    """
    if se_log_hr <= 0:
        raise ValueError("`se_log_hr` must be > 0.")
    if not (0 < baseline_survival_t < 1):
        raise ValueError("`baseline_survival_t` must be in (0,1).")
    if gamma_grid is None:
        gamma_grid = np.linspace(1.0, 3.0, 21)
    gamma_grid = np.array(gamma_grid, dtype=float)
    log_gamma = np.log(gamma_grid)
    log_hr_worst = log_hr - log_gamma
    log_hr_best = log_hr + log_gamma
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low_worst = log_hr_worst - z * se_log_hr
    ci_high_worst = log_hr_worst + z * se_log_hr
    # Δ survival = S_0(t)^exp(log_hr) - S_0(t)
    delta_worst = baseline_survival_t ** np.exp(log_hr_worst) - baseline_survival_t
    delta_best = baseline_survival_t ** np.exp(log_hr_best) - baseline_survival_t
    covers_zero_worst = (ci_low_worst <= 0) & (0 <= ci_high_worst)
    try:
        bp = float(gamma_grid[covers_zero_worst][np.argmin(gamma_grid[covers_zero_worst])])
    except ValueError:
        bp = None
    curve = pd.DataFrame({
        "gamma": gamma_grid,
        "log_hr_worst": log_hr_worst,
        "log_hr_best": log_hr_best,
        "delta_survival_worst": delta_worst,
        "delta_survival_best": delta_best,
        "worst_ci_low": ci_low_worst,
        "worst_ci_high": ci_high_worst,
    })
    if bp is None:
        interpretation = (
            "No sensitivity parameter Γ in the grid overturns the effect — "
            "hazard ratio is robust."
        )
    else:
        interpretation = (
            f"Γ = {bp:.3f} suffices to drive the worst-case log-HR to zero; "
            "effects below that Γ remain significant."
        )
    return FrontierSensitivityResult(
        method="survival_gamma",
        estimate=log_hr,
        se=se_log_hr,
        curve=curve,
        breakpoint=bp,
        interpretation=interpretation,
    )


def calibrate_confounding_strength(
    estimate: float,
    se: float,
    *,
    observed_r2_outcome: float,
    observed_r2_treatment: float,
    alpha: float = 0.05,
    target_estimate: float = 0.0,
) -> FrontierSensitivityResult:
    """Calibrate the strength of an unobserved confounder required to
    explain the observed effect to a target value.

    Follows the Cinelli-Hazlett (2020) and Zhang et al. (2025)
    "ml-calibrated E-value" generalisation: given observed-covariate
    partial-R² with the outcome and treatment, the amount of residual
    variation an unobserved ``U`` would need to share with ``Y`` and
    ``D`` to shift the effect to ``target_estimate``.

    Parameters
    ----------
    estimate, se : float
    observed_r2_outcome, observed_r2_treatment : float in (0, 1)
        Partial-R² of the observed covariate(s) with Y (resp. D). Used
        to benchmark "1x as confounding as observed" / "2x" etc.
    alpha : float, default 0.05
    target_estimate : float, default 0.0
        Effect value to explain away.

    References
    ----------
    Baitairian et al. (arXiv:2510.16560, 2025). [@baitairian2025calibrating]
    Cinelli & Hazlett (JRSS-B 2020).
    """
    if not (0 < observed_r2_outcome < 1):
        raise ValueError("`observed_r2_outcome` must be in (0,1).")
    if not (0 < observed_r2_treatment < 1):
        raise ValueError("`observed_r2_treatment` must be in (0,1).")
    delta = estimate - target_estimate
    if delta == 0:
        raise ValueError(
            "`estimate` already equals `target_estimate`; nothing to calibrate."
        )
    # Maximum bias from a hidden U with partial-R² (r_y, r_d):
    # |bias| ≤ sqrt( r_y * r_d / (1 - r_d) ) * se  (approx, from Cinelli-Hazlett)
    multipliers = np.linspace(0.5, 5.0, 19)
    rows = []
    for k in multipliers:
        ry = min(k * observed_r2_outcome, 0.99)
        rd = min(k * observed_r2_treatment, 0.99)
        max_bias = np.sqrt(ry * rd / max(1.0 - rd, 1e-6)) * se
        adjusted = estimate - np.sign(delta) * max_bias
        rows.append({
            "multiplier": k,
            "r2_outcome": ry,
            "r2_treatment": rd,
            "max_bias": max_bias,
            "adjusted_estimate": adjusted,
            "explains_away": bool(
                (delta > 0 and adjusted <= target_estimate)
                or (delta < 0 and adjusted >= target_estimate)
            ),
        })
    curve = pd.DataFrame(rows)
    survivors = curve.loc[curve["explains_away"]]
    bp = float(survivors["multiplier"].min()) if not survivors.empty else None
    if bp is None:
        interpretation = (
            "Even 5x as strong as the observed covariate bundle cannot "
            "explain the effect away — robust."
        )
    else:
        interpretation = (
            f"An unobserved U roughly {bp:.1f}x as strong as the observed "
            "covariate bundle (on both Y and D) is required to explain "
            f"the effect down to {target_estimate:g}."
        )
    return FrontierSensitivityResult(
        method="calibrate_confounding_strength",
        estimate=estimate,
        se=se,
        curve=curve,
        breakpoint=bp,
        interpretation=interpretation,
    )
