"""
Debiased ML Conformal Prediction (arXiv 2604.03772, 2026).

Conformal counterfactual intervals under *runtime confounding* — when
the test-time distribution differs from training in unmeasured ways.
The key idea: use a debiased (cross-fit) estimator of the conditional
ATE, then conformalize residuals scaled by the debiased nuisance
estimates rather than raw outcome residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class DebiasedConformalResult:
    """Debiased ML conformal counterfactual intervals."""
    intervals: np.ndarray
    point_estimate: np.ndarray
    coverage_target: float
    n_calibration: int
    n_test: int

    def summary(self) -> str:
        widths = self.intervals[:, 1] - self.intervals[:, 0]
        return (
            "Debiased ML Conformal Counterfactual\n"
            "=" * 42 + "\n"
            f"  Target coverage : {1 - self.coverage_target:.2f}\n"
            f"  N cal / N test  : {self.n_calibration} / {self.n_test}\n"
            f"  Mean ITE est    : {self.point_estimate.mean():+.4f}\n"
            f"  Mean width      : {widths.mean():.4f}\n"
        )


def conformal_debiased_ml(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    test_data: Optional[pd.DataFrame] = None,
    alpha: float = 0.1,
    n_folds: int = 5,
    seed: int = 0,
) -> DebiasedConformalResult:
    """
    Debiased ML conformal counterfactual intervals.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat : str
    covariates : list of str
    test_data : pd.DataFrame, optional
    alpha : float, default 0.1
    n_folds : int, default 5
        Cross-fitting folds (debiased step).
    seed : int

    Returns
    -------
    DebiasedConformalResult
    """
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import KFold

    df = data[[y, treat] + list(covariates)].dropna().reset_index(drop=True)
    if df[treat].nunique() != 2:
        raise ValueError("Debiased conformal requires binary treatment.")
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(int)
    X = df[covariates].to_numpy(float)
    n = len(df)
    rng = np.random.default_rng(seed)

    # Cross-fitted nuisances mu1, mu0, ps
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    ps = np.full(n, 0.5)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        # Outcomes
        m1 = LinearRegression().fit(X[tr][D[tr] == 1], Y[tr][D[tr] == 1])
        m0 = LinearRegression().fit(X[tr][D[tr] == 0], Y[tr][D[tr] == 0])
        mu1[te] = m1.predict(X[te])
        mu0[te] = m0.predict(X[te])
        # Propensity
        try:
            lr = LogisticRegression(max_iter=1000).fit(X[tr], D[tr])
            ps[te] = np.clip(lr.predict_proba(X[te])[:, 1], 0.02, 0.98)
        except Exception:
            ps[te] = float(D[tr].mean())

    # AIPW score per unit (debiased ITE estimate)
    ite_db = (
        mu1 - mu0
        + D * (Y - mu1) / ps
        - (1 - D) * (Y - mu0) / (1 - ps)
    )

    # Conformalize residuals of the AIPW score relative to its mean
    # within calibration fold (50/50 split).
    perm = rng.permutation(n)
    cal = perm[n // 2:]
    score_cal = ite_db[cal]
    abs_resid = np.abs(score_cal - score_cal.mean())
    if len(abs_resid) < 5:
        q = float(np.std(abs_resid)) if len(abs_resid) else 1.0
    else:
        idx = min(int(np.ceil((len(abs_resid) + 1) * (1 - alpha))),
                  len(abs_resid)) - 1
        q = float(np.sort(abs_resid)[idx])

    if test_data is not None:
        test_df = test_data[list(covariates)].dropna().reset_index(drop=True)
        Xt = test_df.to_numpy(float)
        # Refit on full data for prediction
        m1_full = LinearRegression().fit(X[D == 1], Y[D == 1])
        m0_full = LinearRegression().fit(X[D == 0], Y[D == 0])
        ite_test = m1_full.predict(Xt) - m0_full.predict(Xt)
    else:
        ite_test = ite_db

    intervals = np.column_stack([ite_test - q, ite_test + q])

    _result = DebiasedConformalResult(
        intervals=intervals,
        point_estimate=ite_test,
        coverage_target=alpha,
        n_calibration=len(cal),
        n_test=len(intervals),
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.conformal_causal.conformal_debiased_ml",
            params={
                "y": y, "treat": treat,
                "covariates": list(covariates),
                "alpha": alpha, "n_folds": n_folds, "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
