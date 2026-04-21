"""
Counterfactual-Fair Conformal Prediction (arXiv 2510.08724, 2025).

Adds counterfactual fairness constraints to conformal ITE intervals:
the interval coverage rate should be the same across protected
subgroups (e.g. race, sex). Achieved by stratified conformalization —
compute calibration quantiles separately per subgroup and report
both per-group and pooled intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FairConformalResult:
    """Fairness-aware conformal ITE intervals."""
    intervals: np.ndarray
    point_estimate: np.ndarray
    group_assignment: np.ndarray
    group_widths: Dict[str, float]
    group_coverage_targets: Dict[str, float]
    coverage_target: float

    def summary(self) -> str:
        rows = [
            "Counterfactual-Fair Conformal ITE",
            "=" * 42,
            f"  Target coverage : {1 - self.coverage_target:.2f}",
            "  Per-group widths:",
        ]
        for g, w in self.group_widths.items():
            rows.append(f"    Group {g!r:>10s}: width = {w:.4f}")
        return "\n".join(rows)


def conformal_fair_ite(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    protected: str,
    test_data: Optional[pd.DataFrame] = None,
    alpha: float = 0.1,
    seed: int = 0,
) -> FairConformalResult:
    """
    Counterfactual-fair conformal ITE intervals.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat : str
    covariates : list of str
        Predictive features. ``protected`` is **excluded** from the
        outcome regression to preserve counterfactual fairness, but
        used downstream for stratified calibration.
    protected : str
        Protected attribute column (categorical).
    test_data : pd.DataFrame, optional
    alpha : float, default 0.1
    seed : int

    Returns
    -------
    FairConformalResult
    """
    from sklearn.linear_model import LinearRegression

    cov_no_protected = [c for c in covariates if c != protected]
    df = data[[y, treat, protected] + cov_no_protected] \
        .dropna().reset_index(drop=True)
    if df[treat].nunique() != 2:
        raise ValueError("Fair conformal requires binary treatment.")
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(int)
    X = df[cov_no_protected].to_numpy(float)
    G = df[protected].to_numpy()
    n = len(df)
    rng = np.random.default_rng(seed)

    # 50/50 split
    perm = rng.permutation(n)
    train = perm[: n // 2]
    cal = perm[n // 2:]

    m1 = LinearRegression().fit(X[train][D[train] == 1], Y[train][D[train] == 1])
    m0 = LinearRegression().fit(X[train][D[train] == 0], Y[train][D[train] == 0])

    # Per-group calibration quantile
    group_q = {}
    group_widths = {}
    for g in np.unique(G):
        mask_cal = (G[cal] == g)
        if mask_cal.sum() < 5:
            # fall back to pooled
            resid_g = np.concatenate([
                np.abs(Y[cal] - m1.predict(X[cal])),
                np.abs(Y[cal] - m0.predict(X[cal])),
            ])
        else:
            resid_g = np.concatenate([
                np.abs(Y[cal][mask_cal & (D[cal] == 1)]
                       - m1.predict(X[cal][mask_cal & (D[cal] == 1)])),
                np.abs(Y[cal][mask_cal & (D[cal] == 0)]
                       - m0.predict(X[cal][mask_cal & (D[cal] == 0)])),
            ])
        if len(resid_g) < 5:
            q_g = float(np.std(resid_g)) if len(resid_g) else 1.0
        else:
            idx = min(int(np.ceil((len(resid_g) + 1) * (1 - alpha))),
                      len(resid_g)) - 1
            q_g = float(np.sort(resid_g)[idx])
        group_q[g] = q_g
        group_widths[str(g)] = 2 * q_g

    test_df = test_data if test_data is not None else df
    test_df = test_df[cov_no_protected + [protected]] \
        .dropna().reset_index(drop=True)
    Xt = test_df[cov_no_protected].to_numpy(float)
    Gt = test_df[protected].to_numpy()
    point = m1.predict(Xt) - m0.predict(Xt)

    intervals = np.zeros((len(test_df), 2))
    for i, g in enumerate(Gt):
        q = group_q.get(g, np.median(list(group_q.values())))
        intervals[i] = [point[i] - q, point[i] + q]

    group_targets = {str(g): 1 - alpha for g in np.unique(G)}

    return FairConformalResult(
        intervals=intervals,
        point_estimate=point,
        group_assignment=Gt,
        group_widths=group_widths,
        group_coverage_targets=group_targets,
        coverage_target=alpha,
    )
