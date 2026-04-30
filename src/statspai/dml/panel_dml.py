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


def _default_treatment_learner():
    """Gradient boosting regressor for m(X̃) ≈ E[D̃ | X̃].

    PLR-with-FE residualises D against the within-transformed covariates,
    so the nuisance is **always** a regressor — even when raw D is
    binary, the within transform produces a continuous D̃ ∈ ℝ. Fitting
    a classifier on (X̃, raw-D) mixes scales (classifier on demeaned
    features, raw {0,1} target) and produces a propensity that has no
    well-defined relationship to E[D̃ | X̃]; this used to be the
    ``binary_treatment=True`` path and was incorrect.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
    )


def _within_transform(values: np.ndarray, unit_idx: np.ndarray,
                      time_idx: Optional[np.ndarray] = None,
                      sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
    """Subtract unit (and optionally time) means from a vector.

    When ``sample_weight`` is supplied the unit and time means are the
    *weighted* means under those weights — this is the natural FE
    absorption for survey-weighted estimation: minimising
    :math:`\\sum_{i,t} w_{it} (y_{it} - \\alpha_i - \\beta D_{it} -
    g(X_{it}))^2` w.r.t. :math:`\\alpha_i` yields
    :math:`\\hat\\alpha_i = (\\sum_t w_{it} \\cdot \\text{rest}_{it}) /
    \\sum_t w_{it}`.

    When ``time_idx`` is supplied this is the two-way within transform
    (``y - y_i. - y_.t + y_..``).  Otherwise it is the one-way within
    transform (``y - y_i.``).
    """
    v = values.astype(float).copy()
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        s_v = pd.Series(v * w)
        s_w = pd.Series(w)
        unit_means = (
            s_v.groupby(unit_idx).transform("sum").to_numpy() /
            s_w.groupby(unit_idx).transform("sum").to_numpy()
        )
    else:
        unit_means = pd.Series(v).groupby(unit_idx).transform("mean").to_numpy()
    v = v - unit_means
    if time_idx is not None:
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float)
            s_v = pd.Series(v * w)
            s_w = pd.Series(w)
            time_means = (
                s_v.groupby(time_idx).transform("sum").to_numpy() /
                s_w.groupby(time_idx).transform("sum").to_numpy()
            )
        else:
            time_means = pd.Series(v).groupby(time_idx).transform("mean").to_numpy()
        v = v - time_means
    return v


def _cluster_se_from_psi(
    psi: np.ndarray, J: float, unit_ids: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> tuple:
    """Cluster-robust SE for the (possibly weighted) DML orthogonal score.

    Unweighted formulation:
        ψ_{it} = (Ỹ_{it} - θ D̃_{it}) D̃_{it}
        Var(θ̂) ≈ J^{-2} · (1/n) · Σ_g (Σ_{i∈g} ψ_{it})²

    Weighted formulation (Liang-Zeger with weights w_{it}):
        Z-equation: Σ_{it} w_{it} ψ_{it}(θ̂) = 0
        J = Σ w_{it} ψ̇_{it} = -Σ w_{it} D̃²_{it}
        Var(θ̂) ≈ Σ_g (Σ_{i∈g} w_{it} ψ_{it})² / J²
    """
    n = len(psi)
    if sample_weight is None:
        s = pd.Series(psi).groupby(unit_ids).sum().to_numpy()
        omega = float(np.sum(s ** 2) / n)
        if abs(J) < 1e-12:
            return float("nan"), omega
        var_theta = omega / (n * J ** 2)
        return float(np.sqrt(var_theta)), omega
    # Weighted version: J is now the *sum* (not mean) of w·d², and
    # the cluster sum aggregates w·ψ within unit.
    w = np.asarray(sample_weight, dtype=float)
    s = pd.Series(w * psi).groupby(unit_ids).sum().to_numpy()
    omega = float(np.sum(s ** 2))
    if abs(J) < 1e-12:
        return float("nan"), omega
    var_theta = omega / (J ** 2)
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
    sample_weight: Optional[Any] = None,
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
        **Deprecated.** Previously routed binary D through a classifier
        path that mixed within-demeaned features with raw {0,1} labels —
        the resulting propensity had no clean interpretation as
        :math:`E[\\tilde D_{it} \\mid \\tilde X_{it}]`. The flag is now
        ignored: :func:`dml_panel` always residualises the
        within-transformed D̃ with a regressor (PLR is agnostic to D's
        type, and within-transformed binary D is no longer binary). For
        DR-style ATE on binary D in panels, use
        :func:`sp.dml(..., model='irm')` with unit dummies in the
        covariate set, :func:`sp.etwfe`, or :func:`sp.callaway_santanna`.
        A :class:`DeprecationWarning` is emitted when this argument is
        passed; passing ``D ∈ {0,1}`` while ``binary_treatment=True`` is
        validated to catch incidental misuse.
    seed : int, default 0
        RNG seed for fold assignment.
    sample_weight : np.ndarray | pd.Series | str, optional
        Per-observation weights (e.g., survey/probability weights). Pass
        either a 1-D array of length ``len(data)`` or a column name in
        ``data``. The within transform becomes a *weighted* within
        transform (subtract weighted unit / time means), and the PLR
        moment + cluster-robust SE use weighted sums. Required if your
        survey design carries informative sampling probabilities.

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
    if binary_treatment:
        import warnings
        d_unique = pd.unique(data[treat].dropna())
        if not set(d_unique.tolist()).issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                "binary_treatment=True requires D ∈ {0, 1}; "
                f"saw {len(d_unique)} unique values: "
                f"{sorted(map(float, d_unique))[:10]}"
            )
        warnings.warn(
            "dml_panel(binary_treatment=True) is deprecated and now a "
            "no-op: the previous classifier path on within-demeaned "
            "covariates was incorrect. The estimator now always uses a "
            "regressor on D̃ (PLR-with-FE; agnostic to D type). For "
            "DR-style ATE on binary D in panels, prefer "
            "sp.dml(..., model='irm') with unit dummies, sp.etwfe, or "
            "sp.callaway_santanna.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Build the working frame including ``sample_weight`` so the dropna
    # mask aligns across (Y, D, X, unit, time, w).
    work = data[required].copy()
    if sample_weight is not None:
        if isinstance(sample_weight, str):
            if sample_weight not in data.columns:
                raise ValueError(
                    f"sample_weight column '{sample_weight}' not in data"
                )
            work["__sw__"] = data[sample_weight].astype(float).values
        else:
            arr = np.asarray(sample_weight, dtype=float)
            if arr.ndim != 1 or len(arr) != len(data):
                raise ValueError(
                    f"sample_weight must be 1-D of length {len(data)}; "
                    f"got shape {arr.shape}"
                )
            work["__sw__"] = arr
    df = work.dropna().reset_index(drop=True)
    n = len(df)
    unit_ids = df[unit].to_numpy()
    time_ids = df[time].to_numpy() if time is not None else None
    Y = df[y].to_numpy(dtype=float)
    D = df[treat].to_numpy(dtype=float)
    if "__sw__" in df.columns:
        w_full = df["__sw__"].to_numpy(dtype=float)
        if np.any(w_full < 0):
            raise ValueError("sample_weight must be non-negative")
        if not np.isfinite(w_full).all():
            raise ValueError("sample_weight contains non-finite values")
        if w_full.sum() <= 0:
            raise ValueError("sample_weight has zero total mass")
    else:
        w_full = None
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
    # When ``sample_weight`` is supplied the within transform absorbs FE
    # via *weighted* unit (and time) means — see ``_within_transform``.
    time_idx_for_within = time_ids if include_time_fe else None
    Y_tilde = _within_transform(Y, unit_ids, time_idx_for_within, sample_weight=w_full)
    D_tilde = _within_transform(D, unit_ids, time_idx_for_within, sample_weight=w_full)
    # Covariates demeaned the same way so the nuisance learners work on
    # within-variation only — matches Semenova-Chernozhukov 2023 §4.
    if covariates:
        X_tilde = np.column_stack([
            _within_transform(
                X[:, j], unit_ids, time_idx_for_within, sample_weight=w_full,
            )
            for j in range(X.shape[1])
        ])
    else:
        X_tilde = X

    # ---- Cross-fit at the unit level -----------------------------------
    if ml_g is None:
        ml_g = _default_outcome_learner()
    if ml_m is None:
        ml_m = _default_treatment_learner()

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

    def _maybe_weighted_fit(learner, Xfit, yfit, wfit):
        clf = clone(learner)
        if wfit is None:
            clf.fit(Xfit, yfit)
            return clf
        try:
            clf.fit(Xfit, yfit, sample_weight=wfit)
        except TypeError:
            import warnings
            warnings.warn(
                f"{type(learner).__name__}.fit does not accept "
                f"sample_weight; falling back to unweighted nuisance "
                f"fit. The weighted moment + cluster SE still apply.",
                RuntimeWarning,
                stacklevel=3,
            )
            clf.fit(Xfit, yfit)
        return clf

    for k in range(n_folds):
        train = obs_fold != k
        test = obs_fold == k
        if not test.any() or not train.any():
            continue
        w_train = w_full[train] if w_full is not None else None

        g_k = _maybe_weighted_fit(ml_g, X_tilde[train], Y_tilde[train], w_train)
        y_resid[test] = Y_tilde[test] - g_k.predict(X_tilde[test])

        m_k = _maybe_weighted_fit(ml_m, X_tilde[train], D_tilde[train], w_train)
        d_resid[test] = D_tilde[test] - m_k.predict(X_tilde[test])

    # ---- PLR moment equation -------------------------------------------
    if w_full is None:
        denom = float(np.sum(d_resid * d_resid))
    else:
        denom = float(np.sum(w_full * d_resid * d_resid))
    if denom < 1e-12:
        raise RuntimeError(
            "dml_panel: Σ d_tilde² ≈ 0 after within + nuisance residualisation. "
            "Treatment has no residual within-variation — try a lower-"
            "capacity ml_m, drop time FE, or check for multicollinearity."
        )
    if w_full is None:
        theta = float(np.sum(d_resid * y_resid) / denom)
    else:
        theta = float(np.sum(w_full * d_resid * y_resid) / denom)

    # ---- Cluster-robust SE ---------------------------------------------
    psi = (y_resid - theta * d_resid) * d_resid  # Neyman-orthogonal score
    if w_full is None:
        J = -float(np.mean(d_resid ** 2))
    else:
        # Weighted version of J: minus the *sum* of w·d² (matches the
        # weighted Liang-Zeger derivation in ``_cluster_se_from_psi``).
        J = -float(np.sum(w_full * d_resid ** 2))
    se, omega = _cluster_se_from_psi(psi, J, unit_ids, sample_weight=w_full)

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
        "weighted": w_full is not None,
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
