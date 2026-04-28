"""
Continuous-treatment and interference-aware conformal causal inference.

Extends :func:`conformal_cate` (Lei-Candes 2021) along two axes
highlighted by the 2025 systematic review of conformal treatment-effect
procedures (arXiv:2509.21660):

- :func:`conformal_continuous` — conformal prediction bands for a
  continuous-treatment dose-response function ``E[Y | T=t, X]``. Based
  on Schröder et al. (arXiv:2407.03094, 2024).
- :func:`conformal_interference` — cluster-exchangeable conformal
  intervals for individual treatment effects in networks where the
  spillover from neighbours violates standard exchangeability. Implements
  the cluster-level split-conformal approach (Lei, Sesia, Candes 2021
  extended to HTE on networks).

Both are *split-conformal* — no distributional assumptions — and rely
on a user-supplied scoring function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


__all__ = [
    "conformal_continuous",
    "conformal_interference",
    "ContinuousConformalResult",
    "InterferenceConformalResult",
]


@dataclass
class ContinuousConformalResult:
    """Output of :func:`conformal_continuous`."""
    alpha: float
    quantile: float
    predictions: pd.DataFrame  # columns: prediction, lo, hi
    model: Any
    dose_grid: Optional[np.ndarray] = None
    dose_curves: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        lines = [
            "Continuous-Treatment Conformal Bands",
            "=" * 60,
            f"  alpha          : {self.alpha}",
            f"  calibration q  : {self.quantile:.6f}",
            f"  # test points  : {len(self.predictions)}",
        ]
        if self.dose_grid is not None:
            lines.append(f"  dose grid      : [{self.dose_grid[0]:.3f}, {self.dose_grid[-1]:.3f}] ({len(self.dose_grid)} pts)")
        return "\n".join(lines)


@dataclass
class InterferenceConformalResult:
    """Output of :func:`conformal_interference`."""
    alpha: float
    quantile: float
    predictions: pd.DataFrame  # columns: cluster, prediction, lo, hi
    cluster_scores: pd.Series

    def summary(self) -> str:
        return "\n".join([
            "Cluster-Exchangeable Conformal Inference on Networks",
            "=" * 60,
            f"  alpha            : {self.alpha}",
            f"  # clusters (cal) : {len(self.cluster_scores)}",
            f"  # test clusters  : {len(self.predictions)}",
            f"  calibration q    : {self.quantile:.6f}",
        ])


# -------------------------------------------------------------------------
# Continuous-treatment conformal (arXiv:2407.03094)
# -------------------------------------------------------------------------


def conformal_continuous(
    data: pd.DataFrame,
    *,
    y: str,
    treatment: str,
    covariates: Sequence[str],
    test_data: pd.DataFrame,
    dose_grid: Optional[Sequence[float]] = None,
    alpha: float = 0.1,
    estimator: Optional[Any] = None,
    calibration_frac: float = 0.5,
    random_state: int = 0,
) -> ContinuousConformalResult:
    """Split-conformal bands for a continuous-treatment dose response.

    Parameters
    ----------
    data : DataFrame
        Training sample with continuous ``treatment`` and outcome ``y``.
    y, treatment : str
    covariates : sequence of str
    test_data : DataFrame
        Test rows to predict on.
    dose_grid : sequence of float, optional
        If provided, return conformal bands for the *entire curve*
        ``E[Y | T=t, X]`` at each ``t`` on the grid (one curve per test
        row).
    alpha : float, default 0.1
        Target miscoverage rate (``1 - alpha`` is the nominal coverage).
    estimator : sklearn-style regressor, optional
        Model for ``E[Y | T, X]``. Must accept ``fit(X, y)`` and
        ``predict(X)``. Defaults to a gradient-boosting regressor.
    calibration_frac : float, default 0.5
    random_state : int, default 0

    Returns
    -------
    ContinuousConformalResult

    References
    ----------
    Schröder et al. (arXiv:2407.03094, 2024).
    """
    if estimator is None:
        from sklearn.ensemble import GradientBoostingRegressor
        estimator = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=random_state,
        )
    if not (0 < calibration_frac < 1):
        raise ValueError("`calibration_frac` must be in (0,1).")
    required = {y, treatment, *covariates}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Training data missing columns: {missing}")
    required_test = {treatment, *covariates}
    missing_t = required_test - set(test_data.columns)
    if missing_t:
        raise ValueError(f"Test data missing columns: {missing_t}")

    rng = np.random.default_rng(random_state)
    idx = np.arange(len(data))
    rng.shuffle(idx)
    n_cal = int(calibration_frac * len(data))
    cal_idx = idx[:n_cal]
    train_idx = idx[n_cal:]
    X_train = data.iloc[train_idx][[treatment] + list(covariates)].to_numpy(dtype=float)
    y_train = data.iloc[train_idx][y].to_numpy(dtype=float)
    X_cal = data.iloc[cal_idx][[treatment] + list(covariates)].to_numpy(dtype=float)
    y_cal = data.iloc[cal_idx][y].to_numpy(dtype=float)
    estimator.fit(X_train, y_train)
    cal_pred = estimator.predict(X_cal)
    cal_scores = np.abs(y_cal - cal_pred)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q = float(np.quantile(cal_scores, q_level))

    # Point + bands for test rows
    X_test = test_data[[treatment] + list(covariates)].to_numpy(dtype=float)
    preds = estimator.predict(X_test)
    test_out = pd.DataFrame({
        "prediction": preds,
        "lo": preds - q,
        "hi": preds + q,
    })

    curves = None
    dose = np.array(dose_grid, dtype=float) if dose_grid is not None else None
    if dose is not None:
        curve_rows = []
        for i in range(len(test_data)):
            cov_row = X_test[i, 1:]  # drop treatment column
            for t in dose:
                X_design = np.concatenate([[t], cov_row])
                p = estimator.predict(X_design.reshape(1, -1))[0]
                curve_rows.append({
                    "test_idx": i, "dose": t,
                    "prediction": p, "lo": p - q, "hi": p + q,
                })
        curves = pd.DataFrame(curve_rows)

    _result = ContinuousConformalResult(
        alpha=alpha, quantile=q, predictions=test_out,
        model=estimator, dose_grid=dose, dose_curves=curves,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.conformal_causal.conformal_continuous",
            params={
                "y": y, "treatment": treatment,
                "covariates": list(covariates),
                "alpha": alpha,
                "calibration_frac": calibration_frac,
                "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# -------------------------------------------------------------------------
# Interference-aware conformal (cluster-exchangeable split)
# -------------------------------------------------------------------------


def conformal_interference(
    data: pd.DataFrame,
    *,
    y: str,
    treatment: str,
    cluster: str,
    covariates: Sequence[str],
    test_clusters: Sequence,
    alpha: float = 0.1,
    estimator: Optional[Any] = None,
    calibration_frac: float = 0.5,
    random_state: int = 0,
) -> InterferenceConformalResult:
    """Cluster-exchangeable split-conformal prediction under interference.

    When units within a cluster interfere (spillover, networks) but
    clusters are exchangeable, the exchangeable-data guarantee of split
    conformal survives at the **cluster** level. We compute a
    cluster-level absolute-residual score by averaging per-unit
    residuals inside each cluster, then build the usual split-conformal
    quantile over clusters.

    Parameters
    ----------
    data : DataFrame
        Full sample with cluster identifier ``cluster``.
    y, treatment, cluster : str
    covariates : sequence of str
    test_clusters : sequence
        Cluster IDs to predict on. Must appear in ``data``.
    alpha : float, default 0.1
    estimator : sklearn-style regressor, optional
    calibration_frac : float, default 0.5
    random_state : int, default 0

    Returns
    -------
    InterferenceConformalResult

    Notes
    -----
    This is the cluster-exchangeable variant used in the Memmesheimer,
    Heuveline & Hesser (arXiv:2509.21660, 2025) systematic review as
    the recommended default when SUTVA is violated within clusters.
    """
    if estimator is None:
        from sklearn.ensemble import GradientBoostingRegressor
        estimator = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, random_state=random_state,
        )
    required = {y, treatment, cluster, *covariates}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    clusters_all = data[cluster].unique()
    clusters_train_cal = [c for c in clusters_all if c not in set(test_clusters)]
    if len(clusters_train_cal) < 4:
        raise ValueError(
            "Need >= 4 non-test clusters for cluster split-conformal; got "
            f"{len(clusters_train_cal)}."
        )

    rng = np.random.default_rng(random_state)
    order = np.array(clusters_train_cal, dtype=object)
    rng.shuffle(order)
    n_cal = max(2, int(calibration_frac * len(order)))
    cal_clusters = set(order[:n_cal].tolist())
    train_clusters = set(order[n_cal:].tolist())
    if not train_clusters:
        raise ValueError("Training cluster set is empty; reduce `calibration_frac`.")

    train_mask = data[cluster].isin(train_clusters)
    cal_mask = data[cluster].isin(cal_clusters)
    X_train = data.loc[train_mask, [treatment] + list(covariates)].to_numpy(dtype=float)
    y_train = data.loc[train_mask, y].to_numpy(dtype=float)
    X_cal = data.loc[cal_mask, [treatment] + list(covariates)].to_numpy(dtype=float)
    y_cal = data.loc[cal_mask, y].to_numpy(dtype=float)
    cal_cluster_series = data.loc[cal_mask, cluster].to_numpy()

    estimator.fit(X_train, y_train)
    cal_pred = estimator.predict(X_cal)
    cal_abs = np.abs(y_cal - cal_pred)
    cal_scores = (
        pd.Series(cal_abs).groupby(cal_cluster_series).mean()
    )
    n = len(cal_scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(cal_scores.to_numpy(), q_level))

    test_rows = []
    for cid in test_clusters:
        sub = data.loc[data[cluster] == cid]
        if sub.empty:
            raise ValueError(f"Test cluster {cid!r} not found in data.")
        Xc = sub[[treatment] + list(covariates)].to_numpy(dtype=float)
        pred_c = float(estimator.predict(Xc).mean())
        test_rows.append({
            "cluster": cid,
            "prediction": pred_c,
            "lo": pred_c - q,
            "hi": pred_c + q,
        })

    return InterferenceConformalResult(
        alpha=alpha, quantile=q,
        predictions=pd.DataFrame(test_rows),
        cluster_scores=cal_scores,
    )
