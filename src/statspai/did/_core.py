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

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core._validate import (  # noqa: F401  (re-exported for the DiD family)
    require_bool_flag as require_bool,
)
from ..exceptions import DataInsufficient

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

    # Pre-group the row positions of each cluster once, then build the draw
    # with a single fancy-index instead of a boolean scan + copy per cluster.
    # Equivalent to the old per-cluster ``df[df[cluster_col] == c]`` + concat,
    # but O(n) per draw rather than O(n_clusters * n).
    group_pos = df.groupby(cluster_col, sort=False).indices
    pos_chunks = []
    draw_suffix_chunks = []
    for j, c in enumerate(sampled):
        idx = group_pos[c]
        pos_chunks.append(idx)
        draw_suffix_chunks.append(np.full(len(idx), j, dtype=np.int64))
    positions = np.concatenate(pos_chunks)
    draw_suffix = np.concatenate(draw_suffix_chunks)

    out = df.iloc[positions].reset_index(drop=True)
    suffixes = np.char.add(sep, draw_suffix.astype(str))
    for col in relabel_cols:
        out[col] = out[col].astype(str).to_numpy() + suffixes
    return out


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

    Ensures each DID estimator in the family exposes the same columns so
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
# Multiplier bootstrap (Callaway–Sant'Anna / R did::mboot convention)
# ----------------------------------------------------------------------


def multiplier_bootstrap(
    psi: np.ndarray,
    n_units: int,
    alpha: float,
    n_boot: int,
    random_state: Optional[int] = None,
    *,
    weight_type: str = "rademacher",
    cluster_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Multiplier bootstrap on influence functions, following R ``did::mboot``.

    Parameters
    ----------
    psi : ndarray, shape (n, K)
        Influence functions of the K estimands, one row per unit
        (or per observation for repeated cross-sections).
    n_units : int
        Number of rows the influence functions are scaled to (the ``n``
        in ``Var(θ̂) = E[ψ²]/n``). Usually ``psi.shape[0]``.
    alpha : float
        Nominal level for the uniform (sup-t) critical value.
    n_boot : int
        Number of bootstrap replications.
    random_state : int, optional
        Seed for the multiplier draws.
    weight_type : {'rademacher', 'mammen'}, default 'rademacher'
        Multiplier distribution. ``'rademacher'`` (±1) matches what the
        R ``did`` package actually draws (``BMisc::multiplier_bootstrap``
        — its docs cite Mammen 1993, but the implementation is
        Rademacher; verified empirically against BMisc 1.4.x).
        ``'mammen'`` is the Mammen (1993) two-point distribution of
        CS2021 §4.2, available in Stata ``csdid`` via ``wbtype(mammen)``.
    cluster_ids : ndarray of shape (n,), optional
        Cluster membership per row. When given, rows are collapsed to
        cluster *means* and the bootstrap runs over ``n_clusters`` rows —
        exactly the R ``did::mboot`` clustering convention.

    Returns
    -------
    se : ndarray of shape (K,)
        Pointwise bootstrap standard errors (IQR-rescaled, robust to
        heavy-tailed multiplier draws — same rescaling as R ``did``).
    crit_unif : float
        Uniform (sup-t) critical value at level ``1 - alpha``. Never
        smaller than the pointwise Normal quantile.

    References
    ----------
    Callaway, B. and Sant'Anna, P.H.C. (2021). "Difference-in-Differences
    with Multiple Time Periods." *Journal of Econometrics*, 225(2),
    200-230, Section 4.2. [@callaway2021difference]

    Mammen, E. (1993). "Bootstrap and Wild Bootstrap for High Dimensional
    Linear Models." *Annals of Statistics*, 21(1), 255-285.
    [@mammen1993bootstrap]
    """
    psi = np.asarray(psi, dtype=float)
    if psi.ndim == 1:
        psi = psi[:, None]
    if weight_type not in ("mammen", "rademacher"):
        raise ValueError(
            f"weight_type must be 'mammen' or 'rademacher', got {weight_type!r}"
        )

    # Collapse to cluster means (R did::mboot: rowsum(if)/cluster_n) so the
    # bootstrap resamples clusters, not units.
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.shape[0] != psi.shape[0]:
            raise ValueError(
                f"cluster_ids length {cluster_ids.shape[0]} ≠ psi rows "
                f"{psi.shape[0]}"
            )
        codes, uniq = pd.factorize(cluster_ids, sort=True)
        n_clusters = len(uniq)
        sums = np.zeros((n_clusters, psi.shape[1]))
        np.add.at(sums, codes, psi)
        counts = np.bincount(codes, minlength=n_clusters).astype(float)
        psi_boot = sums / counts[:, None]
        n_eff = n_clusters
    else:
        psi_boot = psi
        n_eff = int(n_units)

    rng = np.random.default_rng(random_state)
    n_rows = psi_boot.shape[0]

    if weight_type == "mammen":
        # Two-point Mammen weights with mean 0, variance 1.
        # P(V = (1-√5)/2) = (√5+1)/(2√5); P(V = (1+√5)/2) = (√5-1)/(2√5).
        sqrt5 = np.sqrt(5.0)
        a, b = (1 - sqrt5) / 2.0, (1 + sqrt5) / 2.0
        pa = (sqrt5 + 1.0) / (2.0 * sqrt5)
        u = rng.random((n_boot, n_rows))
        V = np.where(u < pa, a, b)
    else:  # rademacher
        V = rng.integers(0, 2, size=(n_boot, n_rows)) * 2.0 - 1.0

    # Bootstrap draws of the K linear combinations, centered under
    # H0: θ_true = θ̂ (influence functions are asymptotically mean-zero;
    # centering removes the finite-sample mean).
    psi_centered = psi_boot - psi_boot.mean(axis=0, keepdims=True)
    boot = V @ psi_centered / n_eff

    # Pointwise SEs from bootstrap, IQR-rescaled (R did convention —
    # robust to heavy tails in multiplier weights compared to raw std).
    # method='inverted_cdf' matches R's quantile(type = 1): with few
    # clusters the multiplier distribution is discrete and the quantile
    # convention shifts SEs by several percent — parity requires R's.
    q75 = np.quantile(boot, 0.75, axis=0, method="inverted_cdf")
    q25 = np.quantile(boot, 0.25, axis=0, method="inverted_cdf")
    iqr_norm = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
    se = (q75 - q25) / iqr_norm
    # Guard against degenerate columns (e.g. a singleton cell).
    fallback_std = boot.std(axis=0, ddof=1)
    se = np.where(se > 0, se, fallback_std)
    se = np.where(se > 0, se, 1e-12)

    # Uniform (sup-t) critical value (R: quantile type = 1).
    max_t = np.max(np.abs(boot) / se, axis=1)
    crit_unif = float(np.quantile(max_t, 1 - alpha, method="inverted_cdf"))
    # Never shrink below the pointwise Normal quantile.
    crit_unif = max(crit_unif, float(stats.norm.ppf(1 - alpha / 2)))

    return np.asarray(se, dtype=float), crit_unif


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


# ----------------------------------------------------------------------
# Missing-data guard
# ----------------------------------------------------------------------


def drop_unusable_rows(
    data: pd.DataFrame,
    *,
    columns: Sequence[str],
    function: str,
    stacklevel: int = 3,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Drop rows with NaN in any estimation column, failing loudly if none survive.

    Missingness must not be left for the estimator to trip over downstream.
    Reshaping a long panel to wide (``pivot_table``) drops entirely-NaN rows
    and columns outright, so a cohort or a period whose outcome was wiped — a
    failed merge, say — leaves a panel that *looks* perfectly balanced: there
    is no NaN cell left for an unbalanced-panel check to count. Every
    ATT(g, t) then loses its cell and contributes 0.0, aggregating to a
    headline ATT of exactly 0.0 with SE 0.0 and p = 1.0. That reads as a
    precisely-estimated null rather than as an error, which is the most
    expensive way to be wrong. A fully-NaN covariate is the quieter twin: it
    drops out of the regression, returning the *unadjusted* estimate while the
    caller believes they adjusted for it.

    Call this before any other validation so that the estimator's existing
    checks (empty never-treated group, no treatment cohorts, no valid
    ``(g, t)`` pairs) all see the real estimation sample. (§7: fail loudly.)

    Parameters
    ----------
    data : pd.DataFrame
        Long-format input.
    columns : Sequence[str]
        Estimation columns that must be non-NaN for a row to be usable —
        typically outcome, time, unit id, and covariates. Group/cohort columns
        are normally excluded, since NaN there encodes "never treated".
    function : str
        Caller name, used in the error and warning text.
    stacklevel : int, default 3
        ``warnings.warn`` stacklevel, so the warning points at user code.
    reset_index : bool, default True
        Reset the index of the returned frame. Pass ``False`` when the caller
        hands back per-row artefacts (imputation weights, residuals) keyed by
        the caller's original index — resetting would silently break that
        alignment.

    Returns
    -------
    pd.DataFrame
        The usable rows.

    Raises
    ------
    DataInsufficient
        If no row has complete data across ``columns``.
    """
    required = list(dict.fromkeys(columns))
    clean = data.dropna(subset=required)
    n_dropped = len(data) - len(clean)

    if n_dropped:
        na_counts = {
            col: int(data[col].isna().sum())
            for col in required
            if data[col].isna().any()
        }
        if clean.empty:
            raise DataInsufficient(
                f"No observations remain after dropping NaNs: all {len(data)} "
                f"rows have NaN in at least one estimation column. NaN counts "
                f"by column: {na_counts}.",
                recovery_hint=(
                    "Check that the outcome, time, unit id, and covariate "
                    "columns survived any upstream merge — an all-NaN column "
                    "here is usually a failed join or a misspelled key."
                ),
                diagnostics={
                    "function": function,
                    "n_rows": len(data),
                    "na_counts": na_counts,
                    "required_columns": required,
                },
            )
        warnings.warn(
            f"{function}: dropped {n_dropped} of {len(data)} rows with NaN in "
            f"an estimation column (NaN counts by column: {na_counts}). The "
            "estimate is computed on the remaining rows, so the effective "
            "sample — and the estimand, if missingness is related to "
            "treatment — differs from the full panel.",
            UserWarning,
            stacklevel=stacklevel,
        )

    return clean.reset_index(drop=True) if reset_index else clean
