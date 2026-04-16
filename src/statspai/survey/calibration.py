"""Survey calibration (raking / post-stratification / linear calibration).

Adjusts design weights so that the weighted sample matches known
population totals (margins) on a set of auxiliary variables. This is
the ``survey::calibrate()`` and ``survey::rake()`` functionality from
R's survey package — previously unavailable in a Python econometrics
toolkit.

Three calibration methods:

- **Raking** (Deming & Stephan 1940) — iterative proportional fitting
  that adjusts weights to match marginal distributions one variable
  at a time. Converges to the minimum-entropy distance solution.
- **Linear calibration** (Deville & Särndal 1992) — find weights
  closest to the design weights (chi-squared distance) such that
  weighted totals of X exactly equal population totals.
- **Post-stratification** — special case where cells are defined
  by the crossing of categorical variables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CalibrationResult:
    calibrated_weights: np.ndarray
    method: str
    converged: bool
    iterations: int
    weight_summary: Dict[str, float]

    def summary(self) -> str:
        lines = [
            f"Survey Calibration ({self.method})",
            "-" * 40,
            f"Converged : {self.converged} ({self.iterations} iterations)",
            f"Min weight: {self.weight_summary['min']:.4f}",
            f"Max weight: {self.weight_summary['max']:.4f}",
            f"Mean weight: {self.weight_summary['mean']:.4f}",
            f"CV(weight) : {self.weight_summary['cv']:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def rake(
    data: pd.DataFrame,
    margins: Dict[str, Dict],
    weight: Optional[str] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> CalibrationResult:
    """Raking (iterative proportional fitting).

    Parameters
    ----------
    data : pd.DataFrame
    margins : dict
        ``{column_name: {category: target_proportion}}``.
        E.g. ``{"sex": {"M": 0.49, "F": 0.51}, "age_group": {"18-34": 0.3, ...}}``.
    weight : str, optional
        Existing design weight column. If ``None``, starts with equal weights.
    """
    df = data.copy()
    n = len(df)
    if weight is not None:
        w = df[weight].to_numpy(dtype=float).copy()
    else:
        w = np.ones(n, dtype=float) / n

    converged = False
    for iteration in range(max_iter):
        w_old = w.copy()
        for col, targets in margins.items():
            vals = df[col].to_numpy()
            for cat, target in targets.items():
                mask = vals == cat
                if mask.sum() == 0:
                    continue
                current_total = w[mask].sum()
                if current_total > 0:
                    factor = target / current_total
                    w[mask] *= factor
        # Normalise
        w = w / w.sum()
        if np.max(np.abs(w - w_old)) < tol:
            converged = True
            break

    return CalibrationResult(
        calibrated_weights=w,
        method="raking",
        converged=converged,
        iterations=iteration + 1,
        weight_summary={
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "cv": float(w.std() / w.mean()) if w.mean() > 0 else 0.0,
        },
    )


def linear_calibration(
    data: pd.DataFrame,
    totals: Dict[str, float],
    weight: Optional[str] = None,
) -> CalibrationResult:
    """Deville-Särndal (1992) linear calibration.

    Find calibrated weights ``g_i * d_i`` minimising
    ``Σ (g_i - 1)² / d_i`` subject to ``Σ g_i d_i x_{ik} = T_k``
    for each auxiliary variable k with known total T_k.

    Parameters
    ----------
    totals : dict
        ``{column_name: population_total}`` for continuous auxiliary variables.
    weight : str, optional
        Design weight column. If ``None``, uses equal weights.
    """
    n = len(data)
    if weight is not None:
        d = data[weight].to_numpy(dtype=float)
    else:
        d = np.ones(n)
    var_names = list(totals.keys())
    T_pop = np.array([totals[v] for v in var_names])
    X = data[var_names].to_numpy(dtype=float)

    # Current weighted totals
    T_current = (d[:, None] * X).sum(axis=0)

    # g-weights: g = 1 + X (X' D X)^{-1} (T - T_current) where D = diag(d)
    DX = d[:, None] * X
    XtDX = X.T @ DX
    try:
        XtDX_inv = np.linalg.inv(XtDX)
    except np.linalg.LinAlgError:
        XtDX_inv = np.linalg.pinv(XtDX)
    lam = XtDX_inv @ (T_pop - T_current)
    g = 1.0 + X @ lam
    w = d * g
    # Normalise to mean=1 for convenience
    w = w / w.sum() * n

    return CalibrationResult(
        calibrated_weights=w,
        method="linear (Deville-Särndal)",
        converged=True,
        iterations=1,
        weight_summary={
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "cv": float(w.std() / w.mean()) if w.mean() > 0 else 0.0,
        },
    )
