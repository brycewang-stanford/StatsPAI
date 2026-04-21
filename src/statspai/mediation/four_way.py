"""
VanderWeele (2014) four-way decomposition of the total effect.

The total effect (TE) of a binary exposure :math:`A` on outcome
:math:`Y` (for a unit with covariates :math:`C`) decomposes into four
non-overlapping components:

.. math::

    TE = CDE(0) + INT_{ref} + INT_{med} + PIE.

* **Controlled Direct Effect (CDE(0))** — direct effect at :math:`M = 0`.
* **Reference Interaction (INT_ref)** — requires only mediator-exposure
  interaction.
* **Mediated Interaction (INT_med)** — requires both mediation AND
  interaction.
* **Pure Indirect Effect (PIE)** — pure mediation with no interaction.

Under no unmeasured confounding of A-Y, A-M, M-Y and no
exposure-induced mediator-outcome confounder, all four are identified
and estimated from simple regressions of ``M`` and ``Y`` on ``A, M,
A*M, C``. This is the standard parametric VanderWeele (2014)
four-way decomposition.

References
----------
VanderWeele, T. J. (2014). "A unification of mediation and
interaction: a four-way decomposition." *Epidemiology*, 25(5),
749-761.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import LinearRegression


@dataclass
class FourWayResult:
    cde: float
    int_ref: float
    int_med: float
    pie: float
    total_effect: float
    proportions: Dict[str, float]
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        prop = self.proportions
        return (
            "VanderWeele (2014) Four-Way Decomposition\n"
            "-----------------------------------------\n"
            f"  N               : {self.n_obs}\n"
            f"  Total Effect    : {self.total_effect:+.4f}\n"
            f"  CDE(0)          : {self.cde:+.4f}  ({prop.get('cde', 0):.1%})\n"
            f"  INT_ref         : {self.int_ref:+.4f}  ({prop.get('int_ref', 0):.1%})\n"
            f"  INT_med         : {self.int_med:+.4f}  ({prop.get('int_med', 0):.1%})\n"
            f"  PIE             : {self.pie:+.4f}  ({prop.get('pie', 0):.1%})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"FourWayResult(TE={self.total_effect:+.4f})"


def four_way_decomposition(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    covariates: Optional[Sequence[str]] = None,
    a0: float = 0.0,
    a1: float = 1.0,
    m0: float = 0.0,
) -> FourWayResult:
    """
    Parametric four-way decomposition of TE = CDE + INT_ref + INT_med + PIE.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, mediator : str
    covariates : sequence of str, optional
    a0, a1 : float
        Reference and comparison levels of the treatment (default 0, 1).
    m0 : float
        Mediator reference level at which CDE is evaluated.

    Returns
    -------
    FourWayResult
    """
    cov = list(covariates or [])
    df = data[[y, treat, mediator] + cov].dropna().reset_index(drop=True)
    n = len(df)

    A = df[treat].to_numpy(dtype=float)
    M = df[mediator].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    Xc = df[cov].to_numpy(dtype=float) if cov else np.zeros((n, 0))

    # Outcome model: Y ~ A + M + A*M + C
    X_out = np.column_stack([np.ones(n), A, M, A * M, Xc])
    lr = LinearRegression(fit_intercept=False).fit(X_out, Y)
    theta0 = float(lr.coef_[0])
    theta1 = float(lr.coef_[1])  # coeff on A
    theta2 = float(lr.coef_[2])  # coeff on M
    theta3 = float(lr.coef_[3])  # coeff on A*M

    # Mediator model: M ~ A + C
    X_med = np.column_stack([np.ones(n), A, Xc])
    mlm = LinearRegression(fit_intercept=False).fit(X_med, M)
    beta0 = float(mlm.coef_[0])
    beta1 = float(mlm.coef_[1])

    # E[M | A=a, C=Cbar]
    Cbar = Xc.mean(axis=0) if Xc.size else np.array([])
    EM_a0 = beta0 + beta1 * a0 + (Cbar @ mlm.coef_[2:]) if Xc.size else beta0 + beta1 * a0
    EM_a1 = beta0 + beta1 * a1 + (Cbar @ mlm.coef_[2:]) if Xc.size else beta0 + beta1 * a1

    # Closed-form from VanderWeele (2014) Table 1:
    cde = (theta1 + theta3 * m0) * (a1 - a0)
    int_ref = theta3 * (EM_a0 - m0) * (a1 - a0)
    int_med = theta3 * beta1 * (a1 - a0) ** 2
    pie = (theta2 + theta3 * a0) * beta1 * (a1 - a0)

    te = cde + int_ref + int_med + pie
    if abs(te) > 1e-10:
        prop = {
            "cde": cde / te,
            "int_ref": int_ref / te,
            "int_med": int_med / te,
            "pie": pie / te,
        }
    else:
        prop = {"cde": 0.0, "int_ref": 0.0, "int_med": 0.0, "pie": 0.0}

    return FourWayResult(
        cde=float(cde),
        int_ref=float(int_ref),
        int_med=float(int_med),
        pie=float(pie),
        total_effect=float(te),
        proportions=prop,
        n_obs=n,
        detail={
            "theta": [theta0, theta1, theta2, theta3],
            "beta": [beta0, beta1],
        },
    )


__all__ = ["four_way_decomposition", "FourWayResult"]
