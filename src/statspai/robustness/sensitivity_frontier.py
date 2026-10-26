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

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility

__all__ = [
    "copula_sensitivity",
    "survival_sensitivity",
    "calibrate_confounding_strength",
    "FrontierSensitivityResult",
]


@dataclass
class FrontierSensitivityResult(ResultProtocolMixin):
    """Container for frontier sensitivity analysis.

    Returned by :func:`copula_sensitivity`, :func:`survival_sensitivity`,
    and :func:`calibrate_confounding_strength`.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.copula_sensitivity(0.3, 0.1)
    >>> isinstance(res, sp.FrontierSensitivityResult)
    True
    >>> res.method
    'copula_gaussian'
    """

    _citation_keys = (
        "balgi2025sensitivity",
        "hu2025nonparametric",
        "baitairian2025calibrating",
    )
    method: str
    estimate: float
    se: float
    curve: pd.DataFrame  # columns depend on method
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


def _overturned(estimate: float, ci_low: np.ndarray, ci_high: np.ndarray) -> np.ndarray:
    """Grid rows whose interval is not entirely on the estimate's side of 0."""
    if estimate > 0:
        return np.asarray(ci_low <= 0)
    if estimate < 0:
        return np.asarray(ci_high >= 0)
    return np.asarray((ci_low <= 0) & (0 <= ci_high))


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
    breakpoint ``rho*`` that zeros the effect: the grid value of smallest
    ``|rho|`` whose adjusted interval covers zero or lies on the other side
    of it.

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

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.copula_sensitivity(0.3, 0.1)
    >>> res.method
    'copula_gaussian'
    >>> int(len(res.curve))
    21

    References
    ----------
    Balgi, Braun, Peña & Daoud (arXiv:2508.08752, 2025). [@balgi2025sensitivity]
    """
    if rho_grid is None:
        grid: np.ndarray = np.linspace(-0.5, 0.5, 21)
    else:
        grid = np.array(rho_grid, dtype=float)
    bias = grid * sigma_u * sigma_y
    adjusted = estimate - bias
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low = adjusted - z * se
    ci_high = adjusted + z * se
    # Breakpoint: smallest |rho| at which the adjusted estimate's CI is no
    # longer entirely on the side of zero the estimate is on -- it covers
    # zero or lies beyond it. Testing only "covers zero" missed a CI that
    # jumps across zero between two grid values (a small SE) and called
    # a reversed effect robust.
    covers_zero = _overturned(float(estimate), ci_low, ci_high)
    try:
        bp = float(grid[covers_zero][np.argmin(np.abs(grid[covers_zero]))])
    except ValueError:
        bp = None
    curve = pd.DataFrame(
        {
            "rho": grid,
            "bias": bias,
            "adjusted_estimate": adjusted,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significant": (ci_low > 0) | (ci_high < 0),
        }
    )
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
    bound the log-HR at sensitivity parameter ``Γ``; the worst case moves
    it toward the null (no effect), the best case away from it:

        log_hr_worst(Γ) = log_hr − sign(log_hr) log(Γ)
        log_hr_best(Γ)  = log_hr + sign(log_hr) log(Γ)

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

    Notes
    -----
    ``breakpoint`` is the smallest grid ``Γ`` whose worst-case confidence
    interval contains 0 or lies on the other side of it (with a small SE
    the interval can cross 0 between two grid values). The continuous
    crossing is the confidence limit
    nearest the null on the hazard-ratio scale,
    ``Γ* = exp(|log_hr| - z se)`` (when it exceeds 1) -- the bias factor
    whose E-value ``Γ* + sqrt(Γ*(Γ* - 1))`` R ``EValue::evalues.HR(rare =
    TRUE)`` reports for the confidence limit.

    Before 1.30 the worst case was always ``log_hr - log(Γ)``: for a
    protective effect (``log_hr < 0``) it moved *away* from the null, so no
    ``Γ`` ever overturned the effect and every protective hazard ratio was
    reported as robust.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.survival_sensitivity(0.4, 0.15)
    >>> res.method
    'survival_gamma'
    >>> int(len(res.curve))
    21

    References
    ----------
    Hu & Westling (arXiv:2511.01412, 2025). [@hu2025nonparametric]
    """
    if se_log_hr <= 0:
        raise ValueError("`se_log_hr` must be > 0.")
    if not (0 < baseline_survival_t < 1):
        raise ValueError("`baseline_survival_t` must be in (0,1).")
    if gamma_grid is None:
        grid2: np.ndarray = np.linspace(1.0, 3.0, 21)
    else:
        grid2 = np.array(gamma_grid, dtype=float)
    if np.any(grid2 < 1):
        raise MethodIncompatibility("`gamma_grid` values must be >= 1.")
    log_gamma = np.log(grid2)
    direction = float(np.sign(log_hr))
    log_hr_worst = log_hr - direction * log_gamma
    log_hr_best = log_hr + direction * log_gamma
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low_worst = log_hr_worst - z * se_log_hr
    ci_high_worst = log_hr_worst + z * se_log_hr
    # Δ survival = S_0(t)^exp(log_hr) - S_0(t)
    delta_worst = baseline_survival_t ** np.exp(log_hr_worst) - baseline_survival_t
    delta_best = baseline_survival_t ** np.exp(log_hr_best) - baseline_survival_t
    # covers the null or lies beyond it (see copula_sensitivity)
    covers_zero_worst = _overturned(float(log_hr), ci_low_worst, ci_high_worst)
    try:
        bp = float(grid2[covers_zero_worst][np.argmin(grid2[covers_zero_worst])])
    except ValueError:
        bp = None
    curve = pd.DataFrame(
        {
            "gamma": grid2,
            "log_hr_worst": log_hr_worst,
            "log_hr_best": log_hr_best,
            "delta_survival_worst": delta_worst,
            "delta_survival_best": delta_best,
            "worst_ci_low": ci_low_worst,
            "worst_ci_high": ci_high_worst,
        }
    )
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
    dof: Optional[float] = None,
    alpha: float = 0.05,
    target_estimate: float = 0.0,
    multipliers: Optional[Sequence[float]] = None,
) -> FrontierSensitivityResult:
    """Calibrate the strength of an unobserved confounder required to
    explain the observed effect to a target value.

    Cinelli & Hazlett's (2020) benchmark bounds: a confounder ``Z`` that
    is ``k`` times as strong as an observed benchmark covariate ``X_j``
    (``k`` times its partial R² with the treatment and with the outcome)
    has at most

    * ``R²_{D~Z|X} = k R²_{D~Xj|X-j} / (1 - R²_{D~Xj|X-j})``,
    * ``R²_{Y~Z|D,X} = ((√k + √r) / √(1 - r))² R²_{Y~Xj|D,X-j} / (1 - R²_{Y~Xj|D,X-j})``
      with ``r = k R²_{D~Xj}² / ((1 - k R²_{D~Xj})(1 - R²_{D~Xj}))``,

    and biases the OLS coefficient by at most
    ``se · √dof · √(R²_{Y~Z|D,X} R²_{D~Z|X} / (1 - R²_{D~Z|X}))``. This is
    R ``sensemakr::ovb_bounds`` / ``ovb_partial_r2_bound`` /
    ``adjusted_estimate`` with ``kd = ky = k``, generalised to an arbitrary
    ``target_estimate`` (the estimate is moved toward the target).

    Parameters
    ----------
    estimate, se : float
        OLS coefficient of the treatment and its (classical) standard error.
    observed_r2_outcome : float in (0, 1)
        Partial R² of the benchmark covariate with the outcome, given the
        treatment and the other covariates (sensemakr ``r2yxj.dx``).
    observed_r2_treatment : float in (0, 1)
        Partial R² of the benchmark covariate with the treatment, given the
        other covariates (sensemakr ``r2dxj.x``).
    dof : float
        Residual degrees of freedom of the regression. Required: the bias
        bound scales with ``√dof``.
    alpha : float, default 0.05
        Level of the adjusted confidence interval, referred to t with
        ``dof`` degrees of freedom as R ``sensemakr::ovb_bounds`` reports
        it (sensemakr's stand-alone ``adjusted_ci`` uses ``dof - 1``).
    target_estimate : float, default 0.0
        Effect value to explain away.
    multipliers : sequence of float, optional
        Values of ``k``; default ``np.linspace(0.5, 5.0, 19)``. A ``k`` that
        implies ``R²_{D~Z|X} >= 1`` is impossible and its row is NaN with
        ``feasible = False`` (sensemakr stops with an error instead).

    Notes
    -----
    Before 1.30 the bias bound omitted the ``√dof`` factor and scaled both
    partial R² linearly (``k R²``), so it was smaller than Cinelli and
    Hazlett's by roughly ``√dof`` -- a factor of ~20 at 400 observations --
    and the function reported effects as robust that a confounder as strong
    as the benchmark explains away.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.calibrate_confounding_strength(
    ...     0.3, 0.1, observed_r2_outcome=0.1, observed_r2_treatment=0.1,
    ...     dof=500)
    >>> res.method
    'calibrate_confounding_strength'
    >>> list(res.curve.columns)[:3]
    ['multiplier', 'r2_outcome', 'r2_treatment']

    References
    ----------
    Baitairian et al. (arXiv:2510.16560, 2025). [@baitairian2025calibrating]
    Cinelli & Hazlett (JRSS-B 2020). [@cinelli2020making]
    """
    if not (0 < observed_r2_outcome < 1):
        raise ValueError("`observed_r2_outcome` must be in (0,1).")
    if not (0 < observed_r2_treatment < 1):
        raise ValueError("`observed_r2_treatment` must be in (0,1).")
    if dof is None:
        raise MethodIncompatibility(
            "`dof` (the regression's residual degrees of freedom) is required: "
            "Cinelli-Hazlett's bias bound is se * sqrt(dof) * sqrt(R2yz R2dz / "
            "(1 - R2dz)). Before 1.30 it was silently computed without "
            "sqrt(dof), understating the bias by that factor.",
            recovery_hint="Pass dof= (the regression's residual degrees of freedom).",
        )
    if not dof > 1:
        raise MethodIncompatibility(f"`dof` must exceed 1; got {dof}.")
    if not se > 0:
        raise MethodIncompatibility(f"`se` must be positive; got {se}.")
    delta = estimate - target_estimate
    if delta == 0:
        raise ValueError(
            "`estimate` already equals `target_estimate`; nothing to calibrate."
        )
    grid = (
        np.linspace(0.5, 5.0, 19)
        if multipliers is None
        else np.asarray(multipliers, dtype=float)
    )
    r2d, r2y = float(observed_r2_treatment), float(observed_r2_outcome)
    # t(dof), as sensemakr::ovb_bounds reports adjusted_lower_CI /
    # adjusted_upper_CI (tests/reference_parity/test_misc_sens_R_parity.py).
    # sensemakr 0.1.6 is not consistent here: its stand-alone adjusted_ci()
    # refers the same adjusted SE to t(dof - 1).
    z = stats.t.ppf(1 - alpha / 2, dof)
    rows = []
    for k in grid:
        r2dz = k * r2d / (1.0 - r2d)
        r2zxj = k * r2d**2 / ((1.0 - k * r2d) * (1.0 - r2d))
        feasible = bool(r2dz < 1 and 0 <= r2zxj < 1 and k * r2d < 1)
        if feasible:
            r2yz = min(
                ((np.sqrt(k) + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)) ** 2
                * (r2y / (1 - r2y)),
                1.0,
            )
            max_bias = float(se * np.sqrt(dof) * np.sqrt(r2yz * r2dz / (1 - r2dz)))
            adjusted = float(estimate - np.sign(delta) * max_bias)
            adj_se = float(
                se * np.sqrt((1 - r2yz) / (1 - r2dz)) * np.sqrt(dof / (dof - 1))
            )
            explains = bool(
                (delta > 0 and adjusted <= target_estimate)
                or (delta < 0 and adjusted >= target_estimate)
            )
        else:
            r2dz = r2yz = max_bias = adjusted = adj_se = float("nan")
            explains = False
        rows.append(
            {
                "multiplier": float(k),
                "r2_outcome": r2yz,
                "r2_treatment": r2dz,
                "max_bias": max_bias,
                "adjusted_estimate": adjusted,
                "adjusted_se": adj_se,
                "adjusted_ci_low": adjusted - z * adj_se,
                "adjusted_ci_high": adjusted + z * adj_se,
                "explains_away": explains,
                "feasible": feasible,
            }
        )
    curve = pd.DataFrame(rows)
    survivors = curve.loc[curve["explains_away"]]
    bp = float(survivors["multiplier"].min()) if not survivors.empty else None
    if bp is None:
        interpretation = (
            f"Even {grid.max():g}x as strong as the benchmark covariate cannot "
            "explain the effect away -- robust on this grid."
        )
    else:
        interpretation = (
            f"An unobserved confounder {bp:g}x as strong as the benchmark "
            "covariate (on both Y and D) is enough to move the estimate to "
            f"{target_estimate:g}."
        )
    return FrontierSensitivityResult(
        method="calibrate_confounding_strength",
        estimate=estimate,
        se=se,
        curve=curve,
        breakpoint=bp,
        interpretation=interpretation,
    )
