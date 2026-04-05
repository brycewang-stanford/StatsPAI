"""
Kitagawa (2015) Test for LATE Validity
=======================================

Tests the necessary conditions for a valid LATE (Local Average Treatment
Effect) interpretation in IV / Wald estimands.

Under the standard LATE assumptions (independence, exclusion, monotonicity),
the CDFs of potential outcomes for compliers must be proper (non-decreasing,
bounded in [0, 1]).  Kitagawa (2015) derives testable implications from this
requirement and constructs a specification test.

The test checks two inequality conditions over a grid of outcome values:

1. P(Y <= y | Z=1) - P(Y <= y | Z=0) >= 0   (treated complier CDF non-neg)
2. P(Y <= y | Z=0)*P(D=1|Z=1) - P(Y <= y | Z=1)*P(D=1|Z=0) >= 0
   (untreated complier CDF non-neg)

The test statistic is the maximum violation across both conditions. Under
H0 (valid LATE), this statistic should be zero.  A bootstrap procedure
provides the p-value.

Reference
---------
Kitagawa, T. (2015). A test for instrument validity. Econometrica, 83(5),
    2043-2063.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Result class
# --------------------------------------------------------------------------- #

@dataclass
class KitagawaResult:
    """
    Result of the Kitagawa (2015) LATE validity test.

    Attributes
    ----------
    statistic : float
        Test statistic (maximum violation of the inequality conditions).
    p_value : float
        Bootstrap p-value.
    first_stage : float
        First-stage effect: P(D=1|Z=1) - P(D=1|Z=0).
    n_boot : int
        Number of bootstrap replications used.
    n_obs : int
        Number of observations.
    max_violation_treated : float
        Maximum violation of the treated-complier CDF condition.
    max_violation_untreated : float
        Maximum violation of the untreated-complier CDF condition.

    Methods
    -------
    summary()
        Print a formatted summary.
    """

    statistic: float
    p_value: float
    first_stage: float
    n_boot: int
    n_obs: int
    max_violation_treated: float
    max_violation_untreated: float

    def summary(self) -> str:
        """Return a formatted summary string."""
        s = self._format()
        print(s)
        return s

    def _format(self) -> str:
        reject = "Yes" if self.p_value < 0.05 else "No"
        return (
            "\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Kitagawa (2015) Test for LATE Validity\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  H0: LATE assumptions hold (monotonicity + exclusion)\n"
            f"  H1: Assumptions violated\n"
            f"\n"
            f"  Test statistic:       {self.statistic:.6f}\n"
            f"  Bootstrap p-value:    {self.p_value:.4f}\n"
            f"  Reject at 5%:         {reject}\n"
            f"\n"
            f"  First stage:          {self.first_stage:.4f}\n"
            f"  Max violation (treated CDF):   {self.max_violation_treated:.6f}\n"
            f"  Max violation (untreated CDF): {self.max_violation_untreated:.6f}\n"
            f"\n"
            f"  N obs:                {self.n_obs}\n"
            f"  Bootstrap reps:       {self.n_boot}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

    def _repr_html_(self) -> str:
        reject = "Yes" if self.p_value < 0.05 else "No"
        colour = "#d32f2f" if self.p_value < 0.05 else "#388e3c"
        return (
            "<div style='font-family:monospace; padding:12px; "
            "border:1px solid #ccc; border-radius:6px; max-width:520px;'>"
            "<h3 style='margin:0 0 8px 0;'>Kitagawa (2015) Test for LATE Validity</h3>"
            "<table style='border-collapse:collapse;'>"
            f"<tr><td style='padding:2px 12px 2px 0;'>Test statistic</td>"
            f"<td>{self.statistic:.6f}</td></tr>"
            f"<tr><td style='padding:2px 12px 2px 0;'>Bootstrap p-value</td>"
            f"<td style='color:{colour};font-weight:bold;'>{self.p_value:.4f}</td></tr>"
            f"<tr><td style='padding:2px 12px 2px 0;'>Reject at 5%</td>"
            f"<td style='color:{colour};font-weight:bold;'>{reject}</td></tr>"
            f"<tr><td style='padding:2px 12px 2px 0;'>First stage</td>"
            f"<td>{self.first_stage:.4f}</td></tr>"
            f"<tr><td style='padding:2px 12px 2px 0;'>N obs</td>"
            f"<td>{self.n_obs}</td></tr>"
            f"<tr><td style='padding:2px 12px 2px 0;'>Bootstrap reps</td>"
            f"<td>{self.n_boot}</td></tr>"
            "</table></div>"
        )

    def __repr__(self) -> str:
        return (
            f"KitagawaResult(statistic={self.statistic:.6f}, "
            f"p_value={self.p_value:.4f}, first_stage={self.first_stage:.4f})"
        )


# --------------------------------------------------------------------------- #
# Core test statistic computation
# --------------------------------------------------------------------------- #

def _compute_test_statistic(
    y: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
    grid: np.ndarray,
) -> tuple:
    """
    Compute the Kitagawa test statistic for a single sample.

    Returns (statistic, max_violation_treated, max_violation_untreated).
    """
    z1 = z == 1
    z0 = z == 0
    n1 = z1.sum()
    n0 = z0.sum()

    if n1 == 0 or n0 == 0:
        return 0.0, 0.0, 0.0

    # First stage probabilities
    p_d1_z1 = d[z1].mean()  # P(D=1 | Z=1)
    p_d1_z0 = d[z0].mean()  # P(D=1 | Z=0)

    max_viol_t = 0.0
    max_viol_u = 0.0

    for yval in grid:
        # CDF estimates
        F_y_z1 = np.mean(y[z1] <= yval)  # P(Y <= y | Z=1)
        F_y_z0 = np.mean(y[z0] <= yval)  # P(Y <= y | Z=0)

        # Condition 1: treated complier CDF must be non-negative
        # P(Y <= y | Z=1) - P(Y <= y | Z=0) >= 0 is NOT the right condition;
        # The correct conditions from Kitagawa (2015) Theorem 1:
        #   f_1(y) = [P(Y<=y, D=1|Z=1) - P(Y<=y, D=1|Z=0)] / [P(D=1|Z=1) - P(D=1|Z=0)]
        #   f_0(y) = [P(Y<=y, D=0|Z=0) - P(Y<=y, D=0|Z=1)] / [P(D=1|Z=1) - P(D=1|Z=0)]
        # Both must be in [0, 1] (proper CDFs), so the numerators must be >= 0.

        # P(Y<=y, D=1 | Z=z)
        F_yd1_z1 = np.mean((y[z1] <= yval) & (d[z1] == 1))
        F_yd1_z0 = np.mean((y[z0] <= yval) & (d[z0] == 1))

        # P(Y<=y, D=0 | Z=z)
        F_yd0_z1 = np.mean((y[z1] <= yval) & (d[z1] == 0))
        F_yd0_z0 = np.mean((y[z0] <= yval) & (d[z0] == 1))

        # Condition 1 (treated complier CDF numerator >= 0):
        # P(Y<=y, D=1|Z=1) - P(Y<=y, D=1|Z=0) >= 0
        viol_t = max(0.0, -(F_yd1_z1 - F_yd1_z0))

        # Condition 2 (untreated complier CDF numerator >= 0):
        # P(Y<=y, D=0|Z=0) - P(Y<=y, D=0|Z=1) >= 0
        F_yd0_z0_correct = np.mean((y[z0] <= yval) & (d[z0] == 0))
        viol_u = max(0.0, -(F_yd0_z0_correct - F_yd0_z1))

        max_viol_t = max(max_viol_t, viol_t)
        max_viol_u = max(max_viol_u, viol_u)

    statistic = max(max_viol_t, max_viol_u)
    return statistic, max_viol_t, max_viol_u


# --------------------------------------------------------------------------- #
# Main function
# --------------------------------------------------------------------------- #

def kitagawa_test(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    instrument: str,
    n_boot: int = 1000,
    n_grid: int = 100,
    seed: Optional[int] = None,
) -> KitagawaResult:
    """
    Kitagawa (2015) specification test for the validity of LATE.

    Tests whether the data are consistent with the LATE assumptions
    (independence, exclusion restriction, monotonicity) by checking that
    the implied complier potential-outcome CDFs are proper distribution
    functions.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable name.
    treatment : str
        Endogenous binary treatment variable (D).
    instrument : str
        Binary instrument variable (Z).
    n_boot : int, default 1000
        Number of bootstrap replications for the p-value.
    n_grid : int, default 100
        Number of grid points for evaluating the CDF conditions.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    KitagawaResult
        Test results including statistic, p-value, and first-stage effect.

    Raises
    ------
    ValueError
        If instrument or treatment are not binary, or first stage is zero.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.kitagawa_test(
    ...     data=df,
    ...     y="outcome",
    ...     treatment="treated",
    ...     instrument="assigned",
    ...     n_boot=1000,
    ...     seed=42,
    ... )
    >>> result.summary()
    """
    df = data.dropna(subset=[y, treatment, instrument])
    y_arr = df[y].values.astype(float)
    d_arr = df[treatment].values.astype(float)
    z_arr = df[instrument].values.astype(float)
    n_obs = len(y_arr)

    # Validate binary variables
    if not set(np.unique(z_arr)).issubset({0.0, 1.0}):
        raise ValueError(
            f"Instrument '{instrument}' must be binary (0/1). "
            f"Found values: {np.unique(z_arr)}"
        )
    if not set(np.unique(d_arr)).issubset({0.0, 1.0}):
        raise ValueError(
            f"Treatment '{treatment}' must be binary (0/1). "
            f"Found values: {np.unique(d_arr)}"
        )

    # First stage
    first_stage = d_arr[z_arr == 1].mean() - d_arr[z_arr == 0].mean()
    if abs(first_stage) < 1e-10:
        raise ValueError(
            "First stage is essentially zero — the instrument has no effect "
            "on the treatment. Cannot test LATE validity."
        )

    # Evaluation grid
    grid = np.linspace(np.min(y_arr), np.max(y_arr), n_grid)

    # Observed test statistic
    stat_obs, viol_t, viol_u = _compute_test_statistic(y_arr, d_arr, z_arr, grid)

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        boot_stat, _, _ = _compute_test_statistic(
            y_arr[idx], d_arr[idx], z_arr[idx], grid
        )
        boot_stats[b] = boot_stat

    # P-value: fraction of bootstrap stats >= observed
    # Under H0 we re-centre: the test statistic should be 0.
    # Use the bootstrap distribution of the statistic itself.
    p_value = np.mean(boot_stats >= stat_obs)

    return KitagawaResult(
        statistic=stat_obs,
        p_value=p_value,
        first_stage=first_stage,
        n_boot=n_boot,
        n_obs=n_obs,
        max_violation_treated=viol_t,
        max_violation_untreated=viol_u,
    )
