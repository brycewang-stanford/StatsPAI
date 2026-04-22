"""
Multiple Decision Point Conformal ITE (Bose & Dempsey 2025, arXiv 2512.08828).

Extends conformal ITE to multi-stage / sequential decision settings
(DTRs, dynamic treatment regimes), where the treatment effect at
stage k+1 depends on history through stage k. Uses a Bonferroni
adjustment across stages (or sequential calibration) to maintain
joint coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MultiDPConformalResult:
    """Multi-decision-point conformal ITE intervals."""
    intervals_per_stage: List[np.ndarray]   # K elements of (n_test, 2)
    cumulative_interval: np.ndarray         # (n_test, 2) for sum over stages
    coverage_target: float
    n_stages: int
    n_test: int

    def summary(self) -> str:
        rows = [
            "Multi-Decision-Point Conformal ITE",
            "=" * 42,
            f"  Stages   : {self.n_stages}",
            f"  Test N   : {self.n_test}",
            f"  Target  α: {self.coverage_target:.3f}"
            f"  (Bonferroni-adjusted per stage:"
            f" {self.coverage_target / max(self.n_stages, 1):.3f})",
            f"  Mean cum width: "
            f"{(self.cumulative_interval[:, 1] - self.cumulative_interval[:, 0]).mean():.4f}",
        ]
        return "\n".join(rows)


def conformal_ite_multidp(
    data: pd.DataFrame,
    y_per_stage: List[str],
    treat_per_stage: List[str],
    history_per_stage: List[List[str]],
    test_data: Optional[pd.DataFrame] = None,
    alpha: float = 0.1,
    seed: int = 0,
) -> MultiDPConformalResult:
    """
    Multi-decision-point conformal ITE.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format wide data: each row = subject; columns include
        stage-specific outcomes / treatments / histories.
    y_per_stage : list of str
        Outcome column at each stage k = 1, ..., K.
    treat_per_stage : list of str
        Binary treatment at each stage.
    history_per_stage : list of list of str
        History (covariates available at decision time k).
    test_data : pd.DataFrame, optional
    alpha : float, default 0.1
    seed : int

    Returns
    -------
    MultiDPConformalResult
    """
    K = len(y_per_stage)
    if K != len(treat_per_stage) or K != len(history_per_stage):
        raise ValueError(
            "y_per_stage / treat_per_stage / history_per_stage must "
            "all be the same length (K stages)."
        )
    df = data.dropna().reset_index(drop=True)
    test_df = test_data if test_data is not None else df
    n_test = len(test_df)
    rng = np.random.default_rng(seed)

    # Bonferroni per-stage α
    alpha_k = alpha / K

    intervals_per_stage = []
    cum_lower = np.zeros(n_test)
    cum_upper = np.zeros(n_test)
    for k in range(K):
        from sklearn.linear_model import LinearRegression
        Y = df[y_per_stage[k]].to_numpy(float)
        D = df[treat_per_stage[k]].to_numpy(int)
        Xcols = list(history_per_stage[k])
        if not Xcols:
            raise ValueError(f"Stage {k}: history must be non-empty.")
        X = df[Xcols].to_numpy(float)
        Xt = test_df[Xcols].to_numpy(float)

        # Split-conformal: 50/50 split
        n = len(df)
        perm = rng.permutation(n)
        train = perm[: n // 2]
        cal = perm[n // 2:]
        m1 = LinearRegression().fit(X[train][D[train] == 1],
                                     Y[train][D[train] == 1])
        m0 = LinearRegression().fit(X[train][D[train] == 0],
                                     Y[train][D[train] == 0])
        # Calibration residuals
        resid = np.concatenate([
            np.abs(Y[cal][D[cal] == 1] - m1.predict(X[cal][D[cal] == 1])),
            np.abs(Y[cal][D[cal] == 0] - m0.predict(X[cal][D[cal] == 0])),
        ])
        if len(resid) < 5:
            q = float(np.std(resid)) if len(resid) else 1.0
        else:
            idx = min(int(np.ceil((len(resid) + 1) * (1 - alpha_k))),
                      len(resid)) - 1
            q = float(np.sort(resid)[idx])
        ite_k = m1.predict(Xt) - m0.predict(Xt)
        interval_k = np.column_stack([ite_k - q, ite_k + q])
        intervals_per_stage.append(interval_k)
        cum_lower += interval_k[:, 0]
        cum_upper += interval_k[:, 1]

    return MultiDPConformalResult(
        intervals_per_stage=intervals_per_stage,
        cumulative_interval=np.column_stack([cum_lower, cum_upper]),
        coverage_target=alpha,
        n_stages=K,
        n_test=n_test,
    )
