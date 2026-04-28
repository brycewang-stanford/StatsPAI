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

    # Per-group calibration quantile.  When a group has too few
    # calibration points to support the per-group quantile, we fall
    # back to the *conservative* pooled quantile — the max across
    # groups with sufficient data — rather than blending arms.  Blending
    # would destroy the group-specific coverage guarantee exactly for
    # the small groups that fair conformal is designed to protect.
    group_q: Dict = {}
    group_widths: Dict[str, float] = {}
    group_fallback: Dict[str, bool] = {}

    # First pass: compute per-group quantiles where we have enough data.
    for g in np.unique(G):
        mask_cal = (G[cal] == g)
        if mask_cal.sum() < 5:
            group_q[g] = None  # placeholder, fill in pass 2
            group_fallback[str(g)] = True
            continue
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
        group_fallback[str(g)] = False

    # Second pass: for small groups, use the max quantile across
    # well-covered groups (conservative upper bound that maintains the
    # marginal 1 - alpha coverage guarantee).  If *all* groups are
    # small, fall back to a pooled quantile and emit a warning.
    valid_qs = [q for q in group_q.values() if q is not None]
    if valid_qs:
        fallback_q = float(max(valid_qs))
    else:
        import warnings as _warnings
        _warnings.warn(
            "conformal_fair_ite: every group has fewer than 5 "
            "calibration points; falling back to pooled quantile. "
            "Per-group coverage guarantee does NOT hold.",
            stacklevel=2,
        )
        pooled = np.concatenate([
            np.abs(Y[cal] - m1.predict(X[cal])),
            np.abs(Y[cal] - m0.predict(X[cal])),
        ])
        if len(pooled) < 5:
            fallback_q = float(np.std(pooled)) if len(pooled) else 1.0
        else:
            idx = min(int(np.ceil((len(pooled) + 1) * (1 - alpha))),
                      len(pooled)) - 1
            fallback_q = float(np.sort(pooled)[idx])
    for g, q in list(group_q.items()):
        if q is None:
            group_q[g] = fallback_q
            group_widths[str(g)] = 2 * fallback_q

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

    _result = FairConformalResult(
        intervals=intervals,
        point_estimate=point,
        group_assignment=Gt,
        group_widths=group_widths,
        group_coverage_targets=group_targets,
        coverage_target=alpha,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.conformal_causal.conformal_fair_ite",
            params={
                "y": y, "treat": treat,
                "covariates": list(covariates),
                "protected": protected,
                "alpha": alpha, "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
