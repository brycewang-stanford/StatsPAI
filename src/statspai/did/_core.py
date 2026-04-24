"""Shared primitives for the sp.did family.

Parallel to ``rd/_core.py`` and ``decomposition/_common.py``. Hosts the
low-level helpers that multiple DiD estimators need — cluster-bootstrap
resampling, event-study DataFrame construction, influence-function →
variance plumbing, and joint-Wald tests.

**Scope discipline**: this module is additive. Existing estimators
(``callaway_santanna``, ``did_multiplegt``, ``sun_abraham``,
``did_imputation``, ...) have their own in-file copies of some of these
routines. Do NOT refactor them onto ``_core.py`` in the same commit that
introduces ``_core.py`` — that collapses two risks (new API + numerical
shift) into one. The refactor is a separate, test-guarded pass.

New estimators (e.g., ``sp.did_multiplegt_dyn``, ``sp.lp_did``) should
import from ``_core.py`` from day one.

Public helpers
--------------
- ``cluster_bootstrap_draw``: resample cluster IDs with collision-safe
  relabeling. Mirrors the pattern in ``did_multiplegt.did_multiplegt``.
- ``event_study_frame``: build the canonical ``model_info['event_study']``
  DataFrame shape so ``sp.did_plot`` works uniformly across estimators.
- ``influence_function_se``: cluster-robust SE from an influence-function
  matrix, following the standard ``Var(IF) / n`` form.
- ``joint_wald``: joint Wald statistic with regularised covariance, used
  for placebo / overall tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------
# Cluster-bootstrap draw
# ----------------------------------------------------------------------


def cluster_bootstrap_draw(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    rng: np.random.Generator,
    relabel_cols: Optional[Sequence[str]] = None,
    sep: str = "_b",
) -> pd.DataFrame:
    """Resample clusters with replacement and relabel to avoid collisions.

    Parameters
    ----------
    df : DataFrame
        Long-format panel. One row per observation.
    cluster_col : str
        Column identifying the resampling unit (often the panel unit id).
    rng : numpy.random.Generator
        Pre-seeded generator. Callers own reproducibility.
    relabel_cols : sequence of str, optional
        Columns whose values must be re-mapped so that identical clusters
        drawn twice don't collide (e.g., if ``cluster_col`` is the panel
        unit id, the draw must keep each copy independent). Defaults to
        ``[cluster_col]``.
    sep : str
        Separator used when building the relabel suffix.

    Returns
    -------
    DataFrame
        A bootstrap sample of the same row count as ``df``, with the
        relabel columns cast to ``str`` and suffixed by a per-draw index.

    Notes
    -----
    This mirrors the idiom in ``did_multiplegt.did_multiplegt`` that
    prepends an index suffix to each re-sampled cluster so that
    downstream groupby ops don't merge independent draws.
    """
    if cluster_col not in df.columns:
        raise ValueError(f"cluster_col={cluster_col!r} not in DataFrame")
    if relabel_cols is None:
        relabel_cols = [cluster_col]

    clusters = df[cluster_col].unique()
    sampled = rng.choice(clusters, size=len(clusters), replace=True)

    frames = []
    for j, c in enumerate(sampled):
        chunk = df[df[cluster_col] == c].copy()
        suffix = f"{sep}{j}"
        for col in relabel_cols:
            chunk[col] = chunk[col].astype(str) + suffix
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# Event-study DataFrame shape
# ----------------------------------------------------------------------

EVENT_STUDY_COLUMNS: Tuple[str, ...] = (
    "relative_time",
    "att",
    "se",
    "pvalue",
    "ci_lower",
    "ci_upper",
    "type",
)


def event_study_frame(
    rows: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    """Build a canonical event-study DataFrame for ``model_info['event_study']``.

    Ensures every estimator in the family exposes the same columns so
    ``sp.did_plot`` and ``sp.cs_report`` work uniformly.

    Parameters
    ----------
    rows : sequence of dict
        Each dict must contain at minimum ``relative_time``, ``att``, ``se``;
        optional keys ``pvalue``, ``ci_lower``, ``ci_upper``, ``type``
        (``'placebo'`` / ``'dynamic'``). Missing optional keys are filled
        with NaN / empty string.
    """
    if not rows:
        return pd.DataFrame(columns=list(EVENT_STUDY_COLUMNS))

    out = pd.DataFrame(rows)
    for col in EVENT_STUDY_COLUMNS:
        if col == "type":
            if col not in out.columns:
                out[col] = ""
        elif col not in out.columns:
            out[col] = np.nan

    # Keep only canonical columns (order) + any extras after.
    extras = [c for c in out.columns if c not in EVENT_STUDY_COLUMNS]
    return out[list(EVENT_STUDY_COLUMNS) + extras]


# ----------------------------------------------------------------------
# Influence-function → SE
# ----------------------------------------------------------------------


def influence_function_se(
    if_matrix: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None,
) -> "float | np.ndarray":
    """Standard error(s) from an influence-function matrix.

    Parameters
    ----------
    if_matrix : ndarray, shape (n, k) or (n,)
        Influence function per observation per estimand. For a scalar
        estimand, pass a 1-D array.
    cluster_ids : ndarray, shape (n,), optional
        If provided, sum IFs within cluster before computing Var. This
        gives a cluster-robust variance. If omitted, uses the
        observation-level Var(IF)/n formula.

    Returns
    -------
    float or ndarray
        Scalar SE when ``if_matrix`` is 1-D; otherwise an ndarray of
        length k (one SE per estimand column).
    """
    if_matrix = np.asarray(if_matrix, dtype=float)
    scalar = if_matrix.ndim == 1
    if scalar:
        if_matrix = if_matrix[:, None]

    n = if_matrix.shape[0]
    if n == 0:
        return np.nan if scalar else np.full(if_matrix.shape[1], np.nan)

    if cluster_ids is None:
        var = np.nanvar(if_matrix, axis=0, ddof=1) / n
    else:
        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.shape[0] != n:
            raise ValueError(
                f"cluster_ids length {cluster_ids.shape[0]} ≠ " f"if_matrix length {n}"
            )
        uniq = np.unique(cluster_ids)
        scores = np.vstack([if_matrix[cluster_ids == c].sum(axis=0) for c in uniq])
        n_clust = scores.shape[0]
        if n_clust < 2:
            var = np.full(if_matrix.shape[1], np.nan)
        else:
            var = np.nanvar(scores, axis=0, ddof=1) / n_clust

    se = np.sqrt(np.maximum(var, 0.0))
    return float(se[0]) if scalar else se


# ----------------------------------------------------------------------
# Joint Wald test
# ----------------------------------------------------------------------


def joint_wald(
    estimates: np.ndarray,
    covariance: np.ndarray,
    *,
    ridge: float = 1e-10,
) -> Dict[str, float]:
    """Joint Wald statistic for H0: all entries of ``estimates`` == 0.

    Returns ``{'statistic', 'df', 'pvalue'}``. Regularises a singular
    covariance with ``ridge`` before inverting, falling back to pseudo-
    inverse if the regularised matrix is still not solvable.
    """
    est = np.asarray(estimates, dtype=float).ravel()
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    if cov.shape != (est.size, est.size):
        raise ValueError(
            f"covariance shape {cov.shape} inconsistent with "
            f"estimates size {est.size}"
        )
    k = est.size
    cov_reg = cov + np.eye(k) * ridge
    try:
        w = float(est @ np.linalg.solve(cov_reg, est))
    except np.linalg.LinAlgError:
        w = float(est @ np.linalg.pinv(cov_reg) @ est)
    pval = float(1 - stats.chi2.cdf(w, k)) if k > 0 else np.nan
    return {"statistic": w, "df": int(k), "pvalue": pval}


# ----------------------------------------------------------------------
# Misc utilities
# ----------------------------------------------------------------------


def sorted_periods(time: pd.Series) -> List[Any]:
    """Sorted unique period values; hoisted so estimators share one idiom."""
    return sorted(pd.Series(time).dropna().unique())


def long_difference(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    y_col: str,
    t_base: Any,
    t_future: Any,
) -> pd.DataFrame:
    """Compute ``y(t_future) - y(t_base)`` per unit from a long panel.

    Returns
    -------
    DataFrame
        Columns ``[id_col, 'ldy']`` with one row per unit that appears in
        both periods.
    """
    base = df.loc[df[time_col] == t_base, [id_col, y_col]].rename(
        columns={y_col: "_y_base"}
    )
    fut = df.loc[df[time_col] == t_future, [id_col, y_col]].rename(
        columns={y_col: "_y_future"}
    )
    merged = fut.merge(base, on=id_col, how="inner")
    merged["ldy"] = merged["_y_future"] - merged["_y_base"]
    return merged[[id_col, "ldy"]]
