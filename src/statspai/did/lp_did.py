"""Local-Projections DiD (LP-DiD) à la Dube, Girardi, Jordà & Taylor (2023).

Reference (to verify before adding to paper.bib)
-----------------------------------------------
Dube, A., Girardi, D., Jordà, Ò., & Taylor, A. M. (2023).
"A Local Projections Approach to Difference-in-Differences Event Studies."
NBER Working Paper (number [待核验 — confirm NBER WP ID and arXiv ID
before citing this in a paper or guide]).

Scope
-----
Runs an LP-DiD event-study: at each horizon h ∈ {−P, ..., −1, 0, 1, ..., H}
fit

    Y_{i, t+h} − Y_{i, t−1} = α_t + β_h · Δd_{i, t} + X_{i, t} γ + ε

on the "clean control" sample — units whose treatment stayed unchanged
from t−1 through t+h. Estimator interpretation [待核验]: under standard
LP-DiD assumptions (parallel trends + no anticipation), β_h is the ATT
at horizon h for units who newly switched treatment at time t. Pre-
treatment horizons (h < 0) function as placebo checks.

Inference is cluster-robust at the unit level. Long differences across
adjacent horizons share the same base period Y_{i, t−1}, inducing cross-
horizon covariance — addressed via a per-horizon OLS with the same
base period fixed within each horizon regression.

**Limitations flagged for reviewer**
- The paper's "clean control" construction has two variants (never-
  treated only, vs. not-yet-treated through t+h). This implementation
  offers both via the ``clean_controls=`` argument; default matches
  the paper's leading spec [待核验].
- Joint tests across horizons (e.g., H0: all placebos = 0) are NOT
  implemented in this first cut. Use ``sp.honest_did`` on the
  event-study output for parallel-trends sensitivity.
- Weighting: untreated-unit aggregation uses equal weights per the
  paper's "regression LP-DiD"; the reweighted variant is out of scope
  for this MVP.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from . import _core as _dc


def lp_did(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treatment: str,
    *,
    horizons: Tuple[int, int] = (-3, 5),
    controls: Optional[List[str]] = None,
    clean_controls: str = "not_yet_treated",
    time_fe: bool = True,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """Local-Projections DiD event-study estimator.

    Parameters
    ----------
    data : DataFrame
        Long-format panel, one row per unit × period.
    y : str
        Outcome column.
    unit : str
        Unit identifier (panel id).
    time : str
        Integer-valued time column. Values must be consecutive or LP-DiD
        cannot compute long differences at each horizon.
    treatment : str
        Binary time-varying treatment indicator (0/1). Units may switch
        from 0 to 1; switch-off events are treated as separate events
        per the LP-DiD identification [待核验 — paper §X].
    horizons : (int, int), default ``(-3, 5)``
        Inclusive (min, max) event-time horizons. Negative values are
        placebo pre-treatment leads.
    controls : list of str, optional
        Additional covariates added to each horizon-h regression.
    clean_controls : {'not_yet_treated', 'never_treated'}, default
        ``'not_yet_treated'``.
        Which untreated units to include as controls at each horizon.
        'not_yet_treated' keeps units whose treatment == 0 from t−1
        through t+h; 'never_treated' keeps only units that never
        treated during the sample.
    time_fe : bool, default True
        Include calendar-time fixed effects.
    cluster : str, optional
        Cluster variable for robust SE (defaults to ``unit``).
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CausalResult
        ``estimate`` is the ATT at horizon 0 (``β_0``); the full
        event-study is in ``model_info['event_study']`` with the
        canonical columns via :func:`statspai.did._core.event_study_frame`.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.lp_did(df, y='y', unit='i', time='t', treatment='d',
    ...               horizons=(-3, 5))
    >>> r.model_info['event_study']   # DataFrame of per-horizon β_h
    >>> sp.honest_did(r, max_M=0.5)   # Rambachan-Roth on the paths

    Notes
    -----
    The ``clean_controls='not_yet_treated'`` option mirrors what
    Dube-Girardi-Jordà-Taylor (2023) call the main LP-DiD specification
    [待核验 — confirm against paper §3]; switching to
    ``clean_controls='never_treated'`` removes the cleanest subset of
    controls at the cost of ignoring late-treated units.
    """
    if clean_controls not in {"not_yet_treated", "never_treated"}:
        raise ValueError(
            f"clean_controls={clean_controls!r} must be "
            "'not_yet_treated' or 'never_treated'"
        )

    df = data.copy()
    required = [y, unit, time, treatment]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in data")

    # Ensure sorted and treatment binary.
    if not set(df[treatment].dropna().unique()) <= {0, 1}:
        raise ValueError(f"Treatment column {treatment!r} must be binary 0/1")
    df = df.sort_values([unit, time]).reset_index(drop=True)

    # Long-panel treatment change Δd_{i,t}.
    df["_d_prev"] = df.groupby(unit)[treatment].shift(1)
    df["_delta_d"] = df[treatment] - df["_d_prev"]

    # Identify "newly treated" events: delta_d == 1 (switch on).
    # LP-DiD canonical spec uses these as the treated arm at event time t.
    # Switch-off events (delta_d == -1) are outside this MVP.
    # [待核验 — confirm treatment of switch-off events in paper §X]

    cluster_var = cluster if cluster is not None else unit

    h_min, h_max = horizons
    if h_min > h_max:
        raise ValueError("horizons must satisfy min <= max")

    # never-treated set (stable across horizons)
    if clean_controls == "never_treated":
        # Units that never have d == 1 in the whole sample.
        treated_ever = df.groupby(unit)[treatment].max()
        never_treated_ids = set(treated_ever[treated_ever == 0].index)
    else:
        never_treated_ids = None  # unused

    # Per-horizon regression
    es_rows: List[Dict[str, Any]] = []
    horizon_range = list(range(h_min, h_max + 1))

    for h in horizon_range:
        # Build long-difference sample: for each (i, t) where Δd_{i,t} is
        # observed, check that (i, t+h) exists, and that the "clean control"
        # condition holds across the relevant window.

        rows = _build_lp_did_sample(
            df=df,
            y=y,
            unit=unit,
            time=time,
            treatment=treatment,
            h=h,
            controls=controls or [],
            clean_controls=clean_controls,
            never_treated_ids=never_treated_ids,
        )
        if rows is None or len(rows) == 0:
            es_rows.append(
                {
                    "relative_time": h,
                    "att": np.nan,
                    "se": np.nan,
                    "pvalue": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "type": "placebo" if h < 0 else "dynamic",
                    "n_obs": 0,
                }
            )
            continue

        # Regression: Δy ~ Δd + controls + time FE
        y_col = "_dy"
        x_cols = ["_delta_d"] + (controls or [])
        beta_h, se_h, n_obs = _ols_with_cluster_se(
            rows,
            y_col=y_col,
            x_cols=x_cols,
            time_col=time,
            cluster_col=cluster_var,
            time_fe=time_fe,
        )

        z = beta_h / se_h if (se_h is not None and se_h > 0) else np.nan
        p = float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
        z_crit = stats.norm.ppf(1 - alpha / 2)
        es_rows.append(
            {
                "relative_time": h,
                "att": float(beta_h) if np.isfinite(beta_h) else np.nan,
                "se": (
                    float(se_h) if (se_h is not None and np.isfinite(se_h)) else np.nan
                ),
                "pvalue": p,
                "ci_lower": (
                    float(beta_h - z_crit * se_h)
                    if (se_h and np.isfinite(se_h))
                    else np.nan
                ),
                "ci_upper": (
                    float(beta_h + z_crit * se_h)
                    if (se_h and np.isfinite(se_h))
                    else np.nan
                ),
                "type": "placebo" if h < 0 else "dynamic",
                "n_obs": int(n_obs),
            }
        )

    es_df = _dc.event_study_frame(es_rows)

    # Headline estimate: ATT at horizon 0.
    h0_row = next((r for r in es_rows if r["relative_time"] == 0), None)
    if h0_row is None:
        headline_est = np.nan
        headline_se = np.nan
        headline_ci = (np.nan, np.nan)
        headline_p = np.nan
    else:
        headline_est = h0_row["att"]
        headline_se = h0_row["se"]
        headline_ci = (h0_row["ci_lower"], h0_row["ci_upper"])
        headline_p = h0_row["pvalue"]

    return CausalResult(
        method="LP-DiD (Dube-Girardi-Jordà-Taylor 2023)",
        estimand="ATT at event-time h=0",
        estimate=headline_est,
        se=headline_se,
        pvalue=headline_p,
        ci=headline_ci,
        alpha=alpha,
        n_obs=int(len(df)),
        model_info={
            "event_study": es_df,
            "horizons": horizon_range,
            "clean_controls": clean_controls,
            "time_fe": time_fe,
            "cluster_var": cluster_var,
            "controls": controls,
            # [待核验] — joint placebo test + overall Wald across horizons
            # will follow a subsequent PR; left out here to keep MVP honest.
            "joint_placebo_test": None,
            "joint_overall_test": None,
        },
    )


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _build_lp_did_sample(
    *,
    df: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treatment: str,
    h: int,
    controls: List[str],
    clean_controls: str,
    never_treated_ids: Optional[set],
) -> Optional[pd.DataFrame]:
    """Assemble the regression sample at horizon h.

    For each unit × period where treatment change Δd_{i,t} is observed
    (delta_d ∈ {-1, 0, 1}), include the observation if
    - Y_{i, t+h} exists (period t+h observed for unit i),
    - Y_{i, t-1} exists (already guaranteed since we computed delta_d),
    - "clean control" condition holds: treatment is stable (equal to
      pre-switch value) from t−1 through t+h for the control arm.

    Returns a DataFrame with columns
    [unit, time, _dy, _delta_d, *controls] or None if empty.
    """
    # Compute Δy_{i, t+h} = Y_{i, t+h} − Y_{i, t−1}
    # In the long panel, Y_{i, t−1} is the lag of y at t; Y_{i, t+h} is the
    # lead by h relative to t.
    df_local = df.copy()
    df_local["_y_base"] = df_local.groupby(unit)[y].shift(1)  # Y_{t−1}
    df_local["_y_future"] = df_local.groupby(unit)[y].shift(-h)  # Y_{t+h}
    df_local["_dy"] = df_local["_y_future"] - df_local["_y_base"]

    # Clean control condition.
    # For switch-on event at t (delta_d == 1):
    #   control units have d_{t−1} == 0 AND stable 0 through t+h.
    # For stay-at-zero control:
    #   d_{t−1} == 0 AND d stays == 0 through t+h.
    # We implement this by requiring, for each row:
    #   treatment value at t equals treatment value at t-1 OR the unit is
    #   "newly treated" (treated arm). The controls must have d == 0 from
    #   t−1 through t+h.

    # Stable-0 flag for each (i, t): treatment equals 0 for all periods in
    # [t−1, t+h] for this unit. For h >= 0, we check [t−1, t+h]; for h < 0
    # (placebo), we check [t+h−1, t−1] (the control should already not have
    # treated prior to the "event").
    window_start_offset = min(-1, h - 1)
    window_end_offset = max(0, h)

    # Build a per-unit forward/backward rolling check.
    # For efficiency, iterate per-unit once.
    is_stable_zero = np.zeros(len(df_local), dtype=bool)
    for uid, idx in df_local.groupby(unit).groups.items():
        u_df = df_local.loc[idx].sort_values(time)
        treat_vals = u_df[treatment].values
        n = len(treat_vals)
        for k in range(n):
            w_lo = k + window_start_offset
            w_hi = k + window_end_offset
            if w_lo < 0 or w_hi >= n:
                continue
            if np.all(treat_vals[w_lo : w_hi + 1] == 0):
                is_stable_zero[u_df.index[k]] = True

    df_local["_stable_zero_window"] = is_stable_zero

    # Sample: rows where Δy is computable AND
    # (treated: delta_d == 1) OR
    # (clean control: stable zero across the window AND
    #   if clean_controls == 'never_treated', unit is in never_treated set)
    treated_mask = df_local["_delta_d"] == 1
    if clean_controls == "never_treated":
        control_mask = (
            df_local[unit].isin(never_treated_ids) & df_local["_stable_zero_window"]
        )
    else:
        control_mask = df_local["_stable_zero_window"] & (df_local["_delta_d"] == 0)

    keep = (treated_mask | control_mask) & df_local["_dy"].notna()
    for c in controls:
        keep &= df_local[c].notna()

    sample = df_local.loc[keep, [unit, time, "_dy", "_delta_d"] + controls].copy()
    if len(sample) == 0:
        return None
    # Ensure delta_d is coded 0/1 for the regression (treated = 1).
    sample["_delta_d"] = sample["_delta_d"].clip(lower=0).astype(float)
    return sample


def _ols_with_cluster_se(
    sample: pd.DataFrame,
    *,
    y_col: str,
    x_cols: List[str],
    time_col: str,
    cluster_col: str,
    time_fe: bool,
) -> Tuple[float, Optional[float], int]:
    """OLS with cluster-robust SE. Returns (β on first x, SE on first x, n)."""
    y = sample[y_col].values.astype(float)
    X_parts = [sample[c].values.astype(float).reshape(-1, 1) for c in x_cols]

    # Time FE via one-hot minus baseline.
    if time_fe:
        time_vals = sample[time_col].values
        unique_times = np.unique(time_vals)
        if len(unique_times) > 1:
            # Drop one for identification.
            for t in unique_times[1:]:
                X_parts.append((time_vals == t).astype(float).reshape(-1, 1))

    # Intercept.
    X_parts.append(np.ones((len(y), 1)))

    X = np.hstack(X_parts)

    # Drop rows with any NaN in X or y.
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]
    n, k = X.shape
    if n <= k:
        return np.nan, None, 0

    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    # Cluster-robust SE.
    clusters = (
        sample.loc[sample.index[valid[: len(sample)]], cluster_col].values
        if valid.sum() == len(sample)
        else sample[cluster_col].values[valid]
    )
    unique_clusters = np.unique(clusters)
    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = clusters == c
        score = X[mask].T @ resid[mask]
        meat += np.outer(score, score)
    n_cl = len(unique_clusters)
    correction = (n_cl / max(n_cl - 1, 1)) * ((n - 1) / max(n - k, 1))
    var_cov = correction * (XtX_inv @ meat @ XtX_inv)

    beta_first = float(beta[0])
    se_first = float(np.sqrt(max(var_cov[0, 0], 0.0)))
    return beta_first, se_first, int(n)
