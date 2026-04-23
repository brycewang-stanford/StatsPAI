"""
Long-panel Double/Debiased ML (Semenova-Chernozhukov 2023, simplified).

Estimates the causal effect of a (continuous or binary) treatment on an
outcome from panel data while (i) absorbing unit and optional time
fixed effects, (ii) debiasing high-dimensional covariate controls via
cross-fit ML nuisance learners, and (iii) reporting cluster-robust
standard errors at the unit level.

Use this when:

- Your dataset is panel (repeated observations per unit).
- You want unit (and optionally time) FE to absorb time-invariant
  unobservables — but also have high-dimensional covariates X_it whose
  confounding you would not trust a linear control to remove.
- Assumption: *homogeneous* causal effect β (PLR) within the FE
  demeaned outcome.

This does NOT do:

- Time-varying confounders in the Robins 1986 sense (those need MSM or
  g-formula; see :func:`sp.dml_msm` (v1.7) or :func:`sp.msm`).
- Heterogeneous CATE in panels — see :func:`sp.causal_forest` with unit
  FE pre-residualisation for that.

Model
-----
.. math::

   Y_{it} &= \\alpha_i + \\lambda_t + \\beta D_{it} + g(X_{it}) + \\varepsilon_{it} \\\\
   D_{it} &= \\alpha_i^D + \\lambda_t^D + m(X_{it}) + v_{it}

Within-transform :math:`\\tilde Y, \\tilde D, \\tilde X` (subtract unit
means; optionally also time means), then run cross-fit PLR on the
within-transformed data with folds that **split units** (not
observations) so that no unit appears in both a nuisance-training set
and the corresponding scoring set.

Standard error: cluster-robust at the unit level (Liang-Zeger 1986)
using the DML score residuals.

References
----------
Semenova, V. & Chernozhukov, V. (2023).
"Debiased Machine Learning of Conditional Average Treatment Effects
and Other Causal Functions." *Econometrics Journal*, 26(2).

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W. & Robins, J. (2018).
"Double/Debiased Machine Learning for Treatment and Structural
Parameters." *Econometrics Journal*, 21(1), C1-C68. [@chernozhukov2018double]

Cameron, A.C. & Miller, D.L. (2015).
"A Practitioner's Guide to Cluster-Robust Inference."
*Journal of Human Resources*, 50(2), 317-372. [@cameron2015practitioner]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


__all__ = ["DMLPanelResult", "dml_panel"]


@dataclass
class DMLPanelResult:
    """Output of :func:`dml_panel`.

    Attributes
    ----------
    estimate : float
        Debiased treatment effect β̂.
    se : float
        Cluster-robust SE at the unit level.
    ci_lower, ci_upper : float
    p_value : float
    t_stat : float
    n_units : int
    n_obs : int
    n_folds : int
    include_time_fe : bool
    ml_g_name : str
        Short name of the outcome nuisance learner.
    ml_m_name : str
        Short name of the treatment nuisance learner.
    method : str
        Always ``"dml_panel"``.
    diagnostics : dict
        Populated with ``{'y_resid_std', 'd_resid_std', 'corr_yd_resid',
        'within_r2', 'omega_cluster'}``.
    """
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    t_stat: float
    n_units: int
    n_obs: int
    n_folds: int
    include_time_fe: bool
    ml_g_name: str
    ml_m_name: str
    method: str = "dml_panel"
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        ci = f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
        tfe = "Y" if self.include_time_fe else "N"
        return (
            "Long-panel Double/Debiased ML\n"
            + "=" * 62 + "\n"
            f"  n units      : {self.n_units}\n"
            f"  n obs        : {self.n_obs}\n"
            f"  n folds      : {self.n_folds}\n"
            f"  unit FE      : Y        time FE: {tfe}\n"
            f"  ml_g (outcome): {self.ml_g_name}\n"
            f"  ml_m (treat)  : {self.ml_m_name}\n"
            "\n"
            f"  β (causal)   : {self.estimate:+.4f}   "
            f"cluster-SE = {self.se:.4f}\n"
            f"  t-stat       : {self.t_stat:+.3f}\n"
            f"  95% CI       : {ci}\n"
            f"  p-value      : {self.p_value:.4g}"
        )


def _default_outcome_learner():
    """Gradient boosting for g(X) — same convention as sp.dml PLR."""
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
    )


def _default_treatment_learner(binary: bool):
    """Gradient boosting for m(X)."""
    from sklearn.ensemble import (
        GradientBoostingClassifier, GradientBoostingRegressor,
    )
    if binary:
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
        )
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
    )


def _within_transform(values: np.ndarray, unit_idx: np.ndarray,
                      time_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Subtract unit means (and optionally time means) from a vector.

    When ``time_idx`` is supplied this is the two-way within transform
    (sum(y - y_i. - y_.t + y_..)).  Otherwise it is the one-way within
    transform (y - y_i.).
    """
    v = values.astype(float).copy()
    # Unit demean
    unit_means = pd.Series(v).groupby(unit_idx).transform("mean").to_numpy()
    v = v - unit_means
    if time_idx is not None:
        # Add back the time-demean correction
        time_means = pd.Series(v).groupby(time_idx).transform("mean").to_numpy()
        v = v - time_means
    return v


def _cluster_se_from_psi(psi: np.ndarray, J: float, unit_ids: np.ndarray) -> tuple:
    """Cluster-robust SE for the DML orthogonal score.

    psi_i = (Y_tilde - θ D_tilde) * D_tilde  —  stacked over all i,t.
    Sandwich variance at the unit level:

        Var(θ̂) ≈ (1/n) * J^{-1} Ω_cluster J^{-1}

    where Ω_cluster = (1/n) Σ_g (Σ_{i∈g} psi_i) (Σ_{i∈g} psi_i)'.
    """
    n = len(psi)
    s = pd.Series(psi).groupby(unit_ids).sum().to_numpy()
    omega = float(np.sum(s ** 2) / n)
    if abs(J) < 1e-12:
        return float("nan"), omega
    var_theta = omega / (n * J ** 2)
    return float(np.sqrt(var_theta)), omega


def dml_panel(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    *,
    unit: str,
    time: Optional[str] = None,
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
    n_folds: int = 5,
    alpha: float = 0.05,
    include_time_fe: bool = False,
    binary_treatment: bool = False,
    seed: int = 0,
) -> DMLPanelResult:
    """Long-panel Double/Debiased ML with unit FE and cluster-robust SE.

    Estimates β in

    .. math::

       Y_{it} = \\alpha_i + \\lambda_t + \\beta D_{it} + g(X_{it}) + \\varepsilon_{it}

    by (1) within-transforming ``y``, ``treat`` and ``covariates`` to
    absorb unit (and optionally time) fixed effects; (2) running
    cross-fit PLR on the demeaned data with folds that split *units*;
    (3) computing the Neyman-orthogonal score and cluster-robust SE at
    the unit level.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel; must contain ``y``, ``treat``, all
        ``covariates``, ``unit``, and ``time`` (if given).
    y, treat : str
        Column names.
    covariates : list of str
        High-dimensional controls X_it.  Pass ``[]`` to fit a pure
        FE model with ML-free residuals.
    unit : str
        Unit identifier column.  Used both for FE absorption and
        cross-fit fold assignment.
    time : str, optional
        Time identifier column; required when ``include_time_fe=True``.
    ml_g, ml_m : sklearn-style estimators, optional
        Nuisance learners.  Default: GradientBoosting with 200 trees,
        depth 3, lr 0.05 — same convention as :func:`sp.dml` PLR.
    n_folds : int, default 5
        Cross-fit folds over units.  Must be >= 2 and <= n_units.
    alpha : float, default 0.05
    include_time_fe : bool, default False
        If True, also subtract time means (two-way within transform).
    binary_treatment : bool, default False
        Use a classifier for m(X) and predict propensity scores.
    seed : int, default 0
        RNG seed for fold assignment.

    Returns
    -------
    :class:`DMLPanelResult`

    Notes
    -----
    The cluster-robust SE follows Liang-Zeger 1986 at the unit level
    (the coarser of the two dimensions): stacked scores are summed
    within unit before squaring.  This is the appropriate clustering
    level when shocks at higher frequencies than the unit are
    plausibly correlated (cf. Cameron-Miller 2015 §3.2).

    Identification requires *no* unobserved time-varying confounders —
    only time-invariant unit heterogeneity + high-dim observed X_it.
    Violations of strict exogeneity of D (e.g. dynamic-feedback) are
    not handled here; use :func:`sp.msm` or :func:`sp.gformula_ice`.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.dml_panel(
    ...     df, y='log_wage', treat='union',
    ...     covariates=['exper', 'educ', 'married', 'south'],
    ...     unit='pid', time='year', include_time_fe=True,
    ... )
    >>> print(res.summary())
    """
    # ---- Input validation & bookkeeping --------------------------------
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")
    required = [y, treat, unit] + list(covariates)
    if include_time_fe and time is None:
        raise ValueError("time must be provided when include_time_fe=True")
    if time is not None:
        required.append(time)
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"missing columns in data: {missing}")

    df = data[required].dropna().reset_index(drop=True)
    n = len(df)
    unit_ids = df[unit].to_numpy()
    time_ids = df[time].to_numpy() if time is not None else None
    Y = df[y].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=float)
    if covariates:
        X = df[list(covariates)].to_numpy(dtype=float)
    else:
        # No covariates: X is a column of zeros so nuisance learners
        # return the mean; equivalent to pure FE-OLS within-transform.
        X = np.zeros((n, 1))

    unique_units = pd.unique(unit_ids)
    n_units = len(unique_units)
    if n_folds > n_units:
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed n_units ({n_units})"
        )

    # ---- Within transform (absorb FE) ----------------------------------
    Y_tilde = _within_transform(Y, unit_ids,
                                time_ids if include_time_fe else None)
    D_tilde = _within_transform(D, unit_ids,
                                time_ids if include_time_fe else None)
    # Covariates demeaned the same way so the nuisance learners work on
    # within-variation only — matches Semenova-Chernozhukov 2023 §4.
    if covariates:
        X_tilde = np.column_stack([
            _within_transform(X[:, j], unit_ids,
                              time_ids if include_time_fe else None)
            for j in range(X.shape[1])
        ])
    else:
        X_tilde = X

    # ---- Cross-fit at the unit level -----------------------------------
    if ml_g is None:
        ml_g = _default_outcome_learner()
    if ml_m is None:
        ml_m = _default_treatment_learner(binary=binary_treatment)

    rng = np.random.default_rng(seed)
    unit_perm = rng.permutation(unique_units)
    unit_folds = np.array_split(unit_perm, n_folds)
    # Map each observation to its fold via its unit
    obs_fold = np.empty(n, dtype=int)
    for k, fold_units in enumerate(unit_folds):
        mask = np.isin(unit_ids, fold_units)
        obs_fold[mask] = k

    y_resid = np.zeros(n)
    d_resid = np.zeros(n)

    # Track per-nuisance within-R² for diagnostics
    within_r2 = 0.0

    from sklearn.base import clone

    for k in range(n_folds):
        train = obs_fold != k
        test = obs_fold == k
        if not test.any() or not train.any():
            continue

        g_k = clone(ml_g)
        g_k.fit(X_tilde[train], Y_tilde[train])
        y_resid[test] = Y_tilde[test] - g_k.predict(X_tilde[test])

        m_k = clone(ml_m)
        if binary_treatment:
            m_k.fit(X_tilde[train], D[train].astype(int))
            # For binary D use propensity on the RAW treatment scale;
            # residualise D_tilde against its predicted conditional mean.
            if hasattr(m_k, "predict_proba"):
                p_hat = m_k.predict_proba(X_tilde[test])[:, 1]
            else:  # pragma: no cover
                p_hat = m_k.predict(X_tilde[test])
            # Transform propensity back to the within-scale: subtract
            # unit (and time) mean of p_hat over training units so the
            # residual is centered.
            d_resid[test] = D_tilde[test] - (
                p_hat - np.mean(p_hat)
            )
        else:
            m_k.fit(X_tilde[train], D_tilde[train])
            d_resid[test] = D_tilde[test] - m_k.predict(X_tilde[test])

    # ---- PLR moment equation -------------------------------------------
    denom = float(np.sum(d_resid * d_resid))
    if denom < 1e-12:
        raise RuntimeError(
            "dml_panel: Σ d_tilde² ≈ 0 after within + nuisance residualisation. "
            "Treatment has no residual within-variation — try a lower-"
            "capacity ml_m, drop time FE, or check for multicollinearity."
        )
    theta = float(np.sum(d_resid * y_resid) / denom)

    # ---- Cluster-robust SE ---------------------------------------------
    psi = (y_resid - theta * d_resid) * d_resid  # Neyman-orthogonal score
    J = -float(np.mean(d_resid ** 2))
    se, omega = _cluster_se_from_psi(psi, J, unit_ids)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se) and se > 0:
        t_stat = theta / se
        p_value = float(2.0 * stats.norm.sf(abs(t_stat)))
        lo = theta - z_crit * se
        hi = theta + z_crit * se
    else:
        t_stat = float("nan")
        p_value = float("nan")
        lo = hi = float("nan")

    # Within-R² of the outcome nuisance
    y_var = float(np.var(Y_tilde))
    if y_var > 0:
        within_r2 = 1.0 - float(np.var(y_resid) / y_var)

    diagnostics = {
        "y_resid_std": float(np.std(y_resid)),
        "d_resid_std": float(np.std(d_resid)),
        "corr_yd_resid": float(
            np.corrcoef(y_resid, d_resid)[0, 1]
        ) if np.std(y_resid) > 0 and np.std(d_resid) > 0 else 0.0,
        "within_r2_outcome": within_r2,
        "omega_cluster": omega,
    }

    return DMLPanelResult(
        estimate=theta,
        se=se,
        ci_lower=lo,
        ci_upper=hi,
        p_value=p_value,
        t_stat=t_stat if np.isfinite(t_stat) else 0.0,
        n_units=n_units,
        n_obs=n,
        n_folds=n_folds,
        include_time_fe=include_time_fe,
        ml_g_name=type(ml_g).__name__,
        ml_m_name=type(ml_m).__name__,
        diagnostics=diagnostics,
    )
