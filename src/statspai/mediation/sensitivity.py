"""Mediation sensitivity analysis (Imai, Keele & Tingley, 2010).

Under sequential ignorability (SI), the average causal mediation
effect (ACME) is identified. But SI is untestable. The sensitivity
parameter ``ρ`` quantifies the degree of unobserved confounding
between the mediator and the outcome (the correlation between their
error terms). When ρ = 0, SI holds; as |ρ| increases, the ACME may
shrink to zero and even flip sign.

- :func:`mediate_sensitivity` — over a grid of ρ values, re-estimate
  the ACME with an imputed correlation between mediator and outcome
  errors, and return the ACME as a function of ρ.
- The ρ at which ACME = 0 is the *sensitivity point* (analogous to
  the E-value in observational studies).

References
----------
Imai, K., Keele, L. & Tingley, D. (2010). "A General Approach to
  Causal Mediation Analysis." *Psych Methods*, 15(4), 309–334.
Imai, K., Keele, L. & Yamamoto, T. (2010). "Identification, Inference
  and Sensitivity Analysis for Causal Mediation Effects." *Stat Sci*,
  25(1), 51–71.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MediateSensitivityResult:
    rho_grid: np.ndarray             # (n_grid,)
    acme_at_rho: np.ndarray          # (n_grid,)
    rho_at_zero: Optional[float]     # ρ where ACME crosses zero
    acme_at_zero: float              # ACME at ρ=0 (the baseline)

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.rho_grid, self.acme_at_rho, "-", color="C0")
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")
        if self.rho_at_zero is not None:
            ax.axvline(self.rho_at_zero, color="C3", linewidth=1, linestyle="--",
                       label=f"ACME=0 at ρ={self.rho_at_zero:.2f}")
            ax.legend()
        ax.set_xlabel("ρ (mediator-outcome error correlation)")
        ax.set_ylabel("ACME(ρ)")
        ax.set_title("Mediation sensitivity analysis")
        return ax

    def summary(self) -> str:
        lines = [
            "Mediation Sensitivity (Imai et al. 2010)",
            "-" * 45,
            f"Baseline ACME (ρ=0) : {self.acme_at_zero: .4f}",
        ]
        if self.rho_at_zero is not None:
            lines.append(f"ρ at which ACME = 0  : {self.rho_at_zero: .4f}")
            lines.append(
                f"Interpretation: unobserved confounding with |ρ| > "
                f"{abs(self.rho_at_zero):.2f} would explain away the "
                f"estimated mediation effect."
            )
        else:
            lines.append("ACME does not cross zero in [-0.9, 0.9].")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def mediate_sensitivity(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    covariates: Optional[list] = None,
    rho_range: tuple = (-0.9, 0.9),
    n_grid: int = 41,
) -> MediateSensitivityResult:
    """Sensitivity analysis for causal mediation.

    For each candidate ρ (correlation between mediator and outcome
    errors), compute a bias-adjusted ACME. The method follows Imai,
    Keele & Yamamoto (2010):

    1. Fit the mediator model: ``M = α₀ + α₁ T + α₂ X + ε_M``.
    2. Fit the outcome model: ``Y = β₀ + β₁ T + β₂ M + β₃ X + ε_Y``.
    3. For each ρ, the bias in the ACME estimate is approximately
       ``ρ · σ_M · σ_Y / σ²_M`` (from the omitted-variable formula).
       Subtract this bias from the naïve ACME.

    Parameters
    ----------
    rho_range : (lo, hi)
    n_grid : int, default 41
    """
    if covariates is None:
        covariates = []
    df = data[[y, treat, mediator] + covariates].dropna()
    Y = df[y].to_numpy(float)
    T = df[treat].to_numpy(float)
    M = df[mediator].to_numpy(float)
    n = len(df)

    # Design matrices
    X_cov = df[covariates].to_numpy(float) if covariates else np.empty((n, 0))
    ones = np.ones((n, 1))

    # Mediator model: M ~ T + X
    Xm = np.column_stack([ones, T, X_cov]) if X_cov.shape[1] else np.column_stack([ones, T])
    beta_m = np.linalg.lstsq(Xm, M, rcond=None)[0]
    resid_m = M - Xm @ beta_m
    sigma_m = float(resid_m.std())

    # Outcome model: Y ~ T + M + X
    Xy = np.column_stack([ones, T, M, X_cov]) if X_cov.shape[1] else np.column_stack([ones, T, M])
    beta_y = np.linalg.lstsq(Xy, Y, rcond=None)[0]
    resid_y = Y - Xy @ beta_y
    sigma_y = float(resid_y.std())

    # Naive ACME = alpha_1 * beta_2 (effect of T on M × effect of M on Y)
    alpha_1 = float(beta_m[1])         # T → M
    beta_2 = float(beta_y[2])          # M → Y (controlling for T)
    acme_naive = alpha_1 * beta_2

    # Bias at ρ: Δ(ρ) ≈ ρ · σ_Y · sign(alpha_1)  (simplified linear formula)
    # More precisely: if ρ ≠ 0, the unobserved path contributes
    # ρ * sigma_y * sigma_m to the covariance of (ε_M, ε_Y), which biases
    # β₂ by ρ * sigma_y / sigma_m per omitted-variable algebra.
    # ACME bias = alpha_1 * Δβ₂ = alpha_1 * ρ * sigma_y / sigma_m
    var_m = float(resid_m.var())
    rho_grid = np.linspace(rho_range[0], rho_range[1], n_grid)
    acme_at_rho = np.empty(n_grid)
    for k, rho in enumerate(rho_grid):
        bias = alpha_1 * rho * sigma_y / max(sigma_m, 1e-12)
        acme_at_rho[k] = acme_naive - bias

    # Find ρ where ACME = 0 (linear interpolation)
    rho_at_zero = None
    for k in range(len(rho_grid) - 1):
        if acme_at_rho[k] * acme_at_rho[k + 1] <= 0:
            frac = acme_at_rho[k] / (acme_at_rho[k] - acme_at_rho[k + 1] + 1e-15)
            rho_at_zero = float(rho_grid[k] + frac * (rho_grid[k + 1] - rho_grid[k]))
            break

    return MediateSensitivityResult(
        rho_grid=rho_grid,
        acme_at_rho=acme_at_rho,
        rho_at_zero=rho_at_zero,
        acme_at_zero=float(acme_naive),
    )
