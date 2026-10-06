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

import functools
import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, cast

import numpy as np
import pandas as pd
from scipy import stats

from ..core._validate import (  # noqa: F401  (re-exported for the DiD family)
    require_bool_flag as require_bool,
)
from ..exceptions import DataInsufficient, MethodIncompatibility

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


def influence_se_did(
    psi: np.ndarray,
    n: int,
    cluster_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """R ``did`` / Stata ``csdid`` analytic SE from influence functions.

    Without clusters: ``sqrt(mean(psi^2) / n)`` per column (R ``did``'s
    ``getSE``).  With ``cluster_ids`` (row-aligned with ``psi``): the
    cluster sums replace the rows, ``sqrt(sum_c (sum_{i in c} psi_i)^2) /
    n`` -- no ``G/(G-1)`` factor, which is Stata ``csdid, cluster()``'s
    analytic variance (and ``csdid2``'s clustering step).  With every row
    its own cluster the two coincide.
    """
    psi = np.asarray(psi, dtype=float)
    squeeze = psi.ndim == 1
    if squeeze:
        psi = psi[:, None]
    if cluster_ids is None:
        out = np.sqrt(np.mean(psi**2, axis=0) / n)
    else:
        codes = pd.factorize(np.asarray(cluster_ids))[0]
        if codes.shape[0] != psi.shape[0]:
            raise MethodIncompatibility(
                f"cluster_ids has {codes.shape[0]} entries for {psi.shape[0]} "
                "influence-function rows."
            )
        sums = np.zeros((codes.max() + 1, psi.shape[1]))
        np.add.at(sums, codes, psi)
        out = np.sqrt(np.sum(sums**2, axis=0)) / n
    return np.asarray(out[0] if squeeze else out)


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
        cluster **sums** (``rowsum(inf.func, cluster)``) and one multiplier
        is drawn per cluster, keeping the ``1/n`` scaling of the estimator
        — the R ``did::mboot`` convention (CS2021, Remark 10).

        .. warning::
           Collapsing to cluster *means* instead — which StatsPAI did
           before 1.23.0, mirroring an older R ``did`` — coincides with
           this only when every cluster has the same size. With unbalanced
           clusters it silently rescales each cluster's contribution by
           ``1/|c|`` and can inflate the SEs several-fold.

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

    # Collapse to cluster *sums* (R did::mboot: `rowsum(inf.func, cluster)`)
    # so the bootstrap resamples clusters, not units.
    #
    # The divisor stays `n` (rows), not `n_clusters`: the estimator is the
    # 1/n empirical average of ψ, so the bootstrap analogue of θ̂ − θ is
    # (1/n) Σ_c V_c Σ_{i∈c} ψ_i.  R writes the same thing as
    # `bres = sqrt(n_clusters) * mean_c(V_c · S_c)` rescaled by
    # `sqrt(n_clusters)/n`, which reduces to dividing the cluster-sum
    # bootstrap draws by n.  Collapsing to cluster means and dividing by
    # n_clusters instead — the pre-1.23.0 path — only agrees when all
    # clusters are the same size; with unbalanced clusters (the usual case,
    # e.g. counties within states) it reweights each cluster by 1/|c| and
    # blows the SEs up.
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
        psi_boot = sums
        n_eff = int(n_units)
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


def fe_dof_not_nested(
    df: pd.DataFrame, fe_cols: Sequence[str], cluster_col: str
) -> int:
    """Fixed-effect parameters counted in ``K`` under the "nested" rule.

    ``fixest::ssc(fixef.K = "nested")`` and ``reghdfe``'s default both
    count, in the small-sample factor ``(N-1)/(N-K)``, the levels of every
    absorbed fixed effect that is *not* nested inside the cluster
    variable (a fixed effect is nested when each of its levels maps to a
    single cluster, e.g. unit effects under ``cluster=unit``), and remove
    one collinear level per additional non-nested effect. Nested effects
    contribute nothing because the cluster-robust meat already absorbs
    them. Returns the number of such parameters; add it to the slope
    count to reproduce the reference degrees of freedom.
    """
    levels: List[int] = []
    for col in fe_cols:
        per_level = df.groupby(col, sort=False)[cluster_col].nunique()
        if per_level.max() <= 1:
            continue
        levels.append(int(per_level.shape[0]))
    if not levels:
        return 0
    return int(sum(levels) - (len(levels) - 1))


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
    pval = float(stats.chi2.sf(w, k)) if k > 0 else np.nan
    return {"statistic": w, "df": int(k), "pvalue": pval}


# ----------------------------------------------------------------------
# Misc utilities
# ----------------------------------------------------------------------


def sorted_periods(time: pd.Series) -> List[Any]:
    """Sorted unique period values; hoisted so estimators share one idiom."""
    return sorted(pd.Series(time).dropna().unique())


_F = TypeVar("_F", bound=Callable[..., Any])


def _is_calendar(col: pd.Series) -> bool:
    return bool(
        pd.api.types.is_datetime64_any_dtype(col.dtype)
        or isinstance(col.dtype, pd.PeriodDtype)
    )


def index_calendar_time(
    data: pd.DataFrame,
    time: str,
    cohort: Optional[str] = None,
    *,
    function: str,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """Replace a date-typed time column (and cohort column) by period numbers.

    The staggered estimators count periods: event time is ``t - g`` and a
    never-treated unit has ``g = 0``. A ``datetime64`` or ``Period`` column
    has neither property, and daily or weekly panels keep their dates in
    exactly that form. When ``time`` is date-typed the observed dates are
    numbered ``1, ..., P`` in order, and a date-typed ``cohort`` becomes the
    number of the first observed period on or after it. A missing cohort
    date, or one after the last observed period (the ``2100-01-01`` habit),
    is never treated in the sample and becomes ``0``.

    A numeric ``time`` is returned untouched with ``info = None``, so the
    numeric code paths do not change.

    Returns
    -------
    data : pd.DataFrame
        The input frame, or a copy with the two columns replaced.
    info : dict or None
        ``{'periods': [...], 'time': name, 'cohort': name, 'regular': bool}``;
        ``periods[k - 1]`` is the date of period ``k``.
    """
    if time not in data.columns or not _is_calendar(data[time]):
        if (
            cohort is not None
            and cohort in data.columns
            and time in data.columns
            and _is_calendar(data[cohort])
        ):
            raise MethodIncompatibility(
                f"{function}: {cohort!r} holds dates but {time!r} does not, so "
                "the two cannot be compared.",
                recovery_hint=(
                    f"Give {time!r} and {cohort!r} the same type: both dates, "
                    "or both period numbers."
                ),
            )
        return data, None

    periods = pd.Index(sorted(data[time].dropna().unique()))
    out = data.copy()
    t_idx = pd.Series(periods.get_indexer(out[time]) + 1, index=out.index)
    out[time] = t_idx.where(out[time].notna())

    if cohort is not None and cohort in out.columns:
        g_raw = out[cohort]
        if not _is_calendar(g_raw):
            if pd.api.types.is_numeric_dtype(g_raw.dtype) and bool(
                (g_raw.fillna(0) == 0).all()
            ):
                out[cohort] = 0
            else:
                raise MethodIncompatibility(
                    f"{function}: {time!r} holds dates but {cohort!r} does "
                    "not, so the two cannot be compared.",
                    recovery_hint=(
                        f"Store the first treated date in {cohort!r} (missing "
                        "for never-treated units)."
                    ),
                )
        else:
            g_pos = np.searchsorted(periods.values, g_raw.values, side="left") + 1
            never = g_raw.isna().to_numpy() | (g_pos > len(periods))
            out[cohort] = np.where(never, 0, g_pos).astype(int)

    regular = True
    if len(periods) > 2:
        steps = np.diff(periods.values.astype("datetime64[ns]").astype("int64"))
        if isinstance(data[time].dtype, pd.PeriodDtype):
            steps = np.diff(np.asarray([p.ordinal for p in periods]))
        # Calendar months and quarters differ in length by a few days; a gap
        # is a step at least half again as long as the usual one.
        regular = bool(steps.max() < 1.5 * np.median(steps))
    if not regular:
        warnings.warn(
            f"{function}: the dates in {time!r} are not evenly spaced. "
            "Periods are numbered in the order they are observed, so event "
            "time counts observed periods, not calendar time.",
            UserWarning,
            stacklevel=3,
        )
    info = {
        "periods": list(periods),
        "time": time,
        "cohort": cohort,
        "regular": regular,
    }
    return out, info


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


# ---------------------------------------------------------------------------
# Weight-estimation influence (Callaway-Sant'Anna aggregation)
# ---------------------------------------------------------------------------


def cohort_share_context(
    cell_groups: "np.ndarray",
    unit_cohorts: "np.ndarray",
    unit_weights: "Optional[np.ndarray]" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Build ``(pg, ind)`` for a vector of per-cell cohort labels.

    ``pg[k]`` is the share of **all** units (never-treated included) whose
    cohort equals cell ``k``'s group, matching R ``did``'s
    ``pg <- mean(weights * (G == g))``.  ``ind[i, k]`` is
    ``ω_i · 1{G_i == g_k}``, matching ``did:::wif``'s
    ``weights.ind * 1*(glist == group[k])``.

    Parameters
    ----------
    unit_weights : ndarray, optional
        Unit weights ω, normalised to mean 1. ``None`` (or an all-ones
        vector) reproduces the unweighted head-count shares exactly.
    """
    g_units = np.asarray(unit_cohorts, dtype=float)
    g_cells = np.asarray(cell_groups, dtype=float)
    ind = (g_units[:, None] == g_cells[None, :]).astype(float)
    if unit_weights is not None:
        w = np.asarray(unit_weights, dtype=float)
        if w.shape[0] != ind.shape[0]:
            raise ValueError(
                f"unit_weights has length {w.shape[0]}, expected "
                f"{ind.shape[0]} to align with unit_cohorts"
            )
        if not np.allclose(w, 1.0):
            ind = ind * w[:, None]
    return ind.mean(axis=0), ind


def weight_influence(pg: "np.ndarray", ind: "np.ndarray") -> "np.ndarray":
    """Influence function of the *estimated* aggregation weights.

    Port of R ``did:::wif``.  Callaway-Sant'Anna aggregations weight each
    cell by its cohort share ``pg[k] = P(G = g_k)``, which is itself
    estimated.  Treating those weights as fixed drops a term from the
    variance and makes the reported standard error too small
    (anti-conservative).

    Parameters
    ----------
    pg : ndarray, shape (K,)
        Estimated cohort share behind each kept cell.
    ind : ndarray, shape (n_units, K)
        ``1{G_i == g_k}`` membership indicators.

    Returns
    -------
    ndarray, shape (n_units, K)
        Multiply by the cell ATTs and add to ``Psi[:, keep] @ w``.

    Notes
    -----
    When every kept cell comes from the same cohort the ``pg`` factors
    cancel and this is identically zero — which is why single-cohort
    aggregates were never affected by the omission.
    """
    total = float(np.sum(pg))
    if total <= 0:
        return np.zeros_like(ind, dtype=float)
    centered = ind - pg[None, :]
    if1 = centered / total
    if2 = np.outer(centered.sum(axis=1), pg / (total**2))
    return np.asarray(if1 - if2, dtype=float)


# ======================================================================
# Standard-error method vocabulary
# ======================================================================

#: Cameron, Gelbach & Miller (2008) report that standard cluster-robust
#: asymptotics over-reject with "few (five to thirty) clusters", and that
#: bootstrap procedures with asymptotic refinement restore the nominal
#: size there. That is the empirical basis for the ``'auto'`` switchover
#: below. [@cameron2008bootstrap]
FEW_CLUSTERS = 30

#: Canonical DiD inference *procedures*, with the spellings users arrive
#: with. This is a different axis from ``core/_vcov_spec.py``, which
#: normalizes *which sandwich* (HC0/HC1/CR1/CR2/CR3) a regression uses.
#: Here the question is which procedure produces the variance at all.
_SE_METHOD_ALIASES = {
    # closed-form / influence-function
    "analytic": "analytic",
    "asymptotic": "analytic",
    "influence": "analytic",
    "if": "analytic",
    # pairs / cluster bootstrap
    "bootstrap": "bootstrap",
    "cluster": "bootstrap",
    "cluster_bootstrap": "bootstrap",
    "pairs": "bootstrap",
    # multiplier / wild bootstrap
    "multiplier": "multiplier",
    "wild": "multiplier",
    "wild-bootstrap": "multiplier",
    "wild_bootstrap": "multiplier",
    "wboot": "multiplier",
    # design-based variants used by sdid
    "jackknife": "jackknife",
    "placebo": "placebo",
    "auto": "auto",
}


def normalize_se_method(
    value: Any,
    *,
    supported: Sequence[str],
    function: str,
    n_clusters: Optional[int] = None,
) -> str:
    """Resolve a user ``se_method=`` to one this estimator implements.

    StatsPAI grew four spellings for the same question — ``vce=``,
    ``se_method=``, ``bstrap=``, ``robust=`` — because each estimator
    copied the reference package it was aligned against. This maps them
    onto one vocabulary without moving any default: callers who never
    pass ``se_method`` keep the behaviour they had.

    ``'auto'`` picks the bootstrap when the design has few clusters and
    the estimator offers one, otherwise the analytic variance. "Few" is
    ``FEW_CLUSTERS`` = 30, the top of the range over which Cameron,
    Gelbach & Miller (2008) document over-rejection by cluster-robust
    asymptotics.

    ``'auto'`` with an unknown cluster count resolves to the analytic
    variance rather than guessing, and the resolved choice is always
    recorded by the caller in ``model_info['se_method']`` so the decision
    is auditable rather than implicit.

    Raises on a spelling this estimator cannot honour, listing what it
    can — silently downgrading a requested wild bootstrap to analytic
    standard errors would be exactly the kind of quiet degradation that
    understates uncertainty.
    """
    if not isinstance(value, str):
        raise MethodIncompatibility(
            f"{function}: se_method must be a string, got " f"{type(value).__name__}.",
            recovery_hint=f"Use one of {sorted(set(supported))}.",
            diagnostics={"se_method": repr(value)},
        )

    key = value.strip().lower().replace(" ", "_")
    canonical = _SE_METHOD_ALIASES.get(key)
    if canonical is None:
        raise MethodIncompatibility(
            f"{function}: unknown se_method {value!r}.",
            recovery_hint=(
                f"Supported here: {sorted(set(supported))}. Accepted "
                f"spellings: {sorted(_SE_METHOD_ALIASES)}."
            ),
            diagnostics={"se_method": value, "supported": list(supported)},
        )

    if canonical == "auto":
        few = n_clusters is not None and n_clusters <= FEW_CLUSTERS
        for candidate in ("multiplier", "bootstrap"):
            if few and candidate in supported:
                return candidate
        if "analytic" in supported:
            return "analytic"
        return supported[0]

    if canonical not in supported:
        raise MethodIncompatibility(
            f"{function}: se_method={value!r} resolves to {canonical!r}, "
            f"which this estimator does not implement.",
            recovery_hint=(
                f"{function} supports {sorted(set(supported))}. Requesting "
                "an unavailable procedure is rejected rather than silently "
                "downgraded, because the fallback would usually be the "
                "narrower interval."
            ),
            diagnostics={"requested": canonical, "supported": list(supported)},
        )
    return canonical


# ----------------------------------------------------------------------
# Parallel-trends vocabulary
# ----------------------------------------------------------------------
#
# Baker, Callaway, Cunningham, Goodman-Bacon and Sant'Anna (2026) name the
# distinct parallel-trends assumptions a staggered design can impose, and
# ask practitioners to say which one they used:
#
#   "At the very least, we strongly recommend that researchers clearly
#    state the specific parallel trends assumption they are actually
#    imposing in their analysis to allow readers to discuss its
#    plausibility in a scientifically grounded manner."  (§5.2.2)
#
# An estimator choice *is* an assumption choice, but users read
# ``control_group='notyettreated'`` as a knob rather than as a commitment.
# Every staggered estimator therefore stamps the assumption it imposes
# into ``model_info['parallel_trends']`` so that ``.summary()``, the
# report writers, and agent consumers can surface it verbatim.

PT_ASSUMPTIONS: Dict[str, Dict[str, str]] = {
    "PT-GT-NEV": {
        "label": "PT-GT-NEV",
        "name": "Parallel trends based on never-treated groups",
        "comparison": "never-treated units only",
        "restricts_pretrends": "no",
        "statement": (
            "For every eventually-treated group g and post-treatment "
            "period t >= g, the average change in untreated potential "
            "outcomes is the same for group g and the never-treated group."
        ),
        "tradeoff": (
            "Avoids compositional change in the comparison group and "
            "leaves pre-trends unrestricted, but discards the "
            "not-yet-treated units and can be imprecise when few units "
            "are never treated."
        ),
    },
    "PT-GT-NYT": {
        "label": "PT-GT-NYT",
        "name": "Parallel trends based on not-yet-treated groups",
        "comparison": "all units not yet treated by t",
        "restricts_pretrends": "no",
        "statement": (
            "For every eventually-treated group g, every not-yet-treated "
            "group g' > t and every t >= g, the average change in "
            "untreated potential outcomes is the same for g and g'."
        ),
        "tradeoff": (
            "Uses more information than PT-GT-NEV without restricting "
            "pre-trends; the comparison group's composition changes with "
            "t, and later-treated units may have selected on outcomes."
        ),
    },
    "PT-GT-ALL": {
        "label": "PT-GT-ALL",
        "name": "Parallel trends for every period and group",
        "comparison": "all groups, all periods (pre and post)",
        "restricts_pretrends": "yes",
        "statement": (
            "For every pair of groups g, g' and every time period t "
            "(pre-treatment included), the average change in untreated "
            "potential outcomes is the same."
        ),
        "tradeoff": (
            "Most precise, and makes pre-trends directly testable because "
            "the model is over-identified; but if pre-trends are not "
            "parallel the ATT(g, t) estimates are biased."
        ),
    },
    "PT": {
        "label": "PT",
        "name": "2x2 parallel trends",
        "comparison": "the single untreated comparison group",
        "restricts_pretrends": "n/a (two periods)",
        "statement": (
            "The average change in untreated potential outcomes between "
            "the pre- and post-period is the same in the treated and "
            "comparison groups."
        ),
        "tradeoff": "The canonical two-group two-period assumption.",
    },
    "PT-ES": {
        "label": "PT-ES",
        "name": "Parallel trends event study (2xT)",
        "comparison": "the single untreated comparison group",
        "restricts_pretrends": "no",
        "statement": (
            "For every post-treatment period t >= g, the average change "
            "in untreated potential outcomes from g-1 to t is the same "
            "in the treated and comparison groups."
        ),
        "tradeoff": (
            "Longer horizons need the assumption to hold in more periods, "
            "so long-run effects rest on strictly stronger assumptions "
            "than short-run ones."
        ),
    },
}


def parallel_trends_block(
    label: str,
    *,
    conditional: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the ``model_info['parallel_trends']`` record for an estimator.

    Parameters
    ----------
    label : str
        One of the keys of :data:`PT_ASSUMPTIONS`.
    conditional : bool, default False
        Whether covariates were used, i.e. whether the assumption is the
        conditional variant (``CPT-GT-NYT`` rather than ``PT-GT-NYT``).
    extra : dict, optional
        Estimator-specific annotations merged into the record.

    Returns
    -------
    dict
        ``label`` / ``name`` / ``statement`` / ``comparison`` /
        ``restricts_pretrends`` / ``tradeoff`` / ``conditional``, plus
        ``reference``.
    """
    if label not in PT_ASSUMPTIONS:
        raise ValueError(
            f"unknown parallel-trends label {label!r}; "
            f"expected one of {sorted(PT_ASSUMPTIONS)}"
        )
    block: Dict[str, Any] = dict(PT_ASSUMPTIONS[label])
    block["conditional"] = bool(conditional)
    if conditional:
        block["label"] = "C" + block["label"]
        block["name"] = block["name"].replace(
            "Parallel trends", "Conditional parallel trends"
        )
        block["statement"] = (
            block["statement"].rstrip(".") + ", conditional on the covariates X."
        )
        block["also_requires"] = (
            "Strong overlap (SO): the conditional probability of treatment "
            "given X is bounded away from 0 and 1."
        )
    block["reference"] = "baker2026difference"
    if extra:
        block.update(extra)
    return block


# ----------------------------------------------------------------------
# R-style covariate formulas
# ----------------------------------------------------------------------


def covariates_from_formula(
    data: pd.DataFrame,
    formula: str,
    *,
    function: str = "callaway_santanna",
) -> Tuple[pd.DataFrame, List[str]]:
    """Materialise an R ``xformla`` into concrete covariate columns.

    R's DiD packages take covariates as a one-sided formula —
    ``att_gt(xformla = ~ lpop + I(lpop^2))`` — while StatsPAI takes a list
    of column names. Migrating a script therefore meant hand-expanding
    every transformation into a new column first, which is exactly the
    kind of busywork that produces silent mistakes.

    This evaluates the right-hand side with patsy and returns the frame
    extended with the resulting columns, plus their names.

    Two conventions are worth stating because they bite:

    - The intercept is dropped. Every DiD estimator here adds its own,
      and passing a constant column would make the design singular.
    - Inside ``I(...)``, patsy evaluates **Python**, so a power is
      ``I(x**2)``, not R's ``I(x^2)``. In Python ``^`` is bitwise XOR, so
      an unconverted R formula would either raise deep inside patsy or —
      on integer columns — quietly compute something else entirely. That
      case is detected and rejected up front.

    Rows with missing values in a referenced variable are kept, with NaN
    in the materialised columns, so the caller's own missing-data policy
    still applies rather than this function silently shrinking the sample.

    Parameters
    ----------
    data : pd.DataFrame
        Source frame.
    formula : str
        ``"~ x1 + x2"``, or ``"y ~ x1 + x2"`` (the left-hand side is
        ignored — R's ``xformla`` carries one for historical reasons).
    function : str
        Caller name, for error diagnostics.

    Returns
    -------
    (pd.DataFrame, list of str)
        The frame with materialised columns appended, and their names.
        ``~ 1`` (or an empty right-hand side) yields ``(data, [])``.
    """
    rhs = formula.split("~", 1)[1] if "~" in formula else formula
    rhs = rhs.strip()
    if rhs in ("", "1"):
        return data, []

    if "^" in rhs:
        raise MethodIncompatibility(
            f"{function}: '^' in a covariate formula is ambiguous. patsy "
            "evaluates the inside of I(...) as Python, where '^' is bitwise "
            "XOR, not exponentiation — so an R formula copied verbatim would "
            f"not mean what it says. Got: {formula!r}.",
            recovery_hint=(
                "Write powers as I(x**2) rather than I(x^2); for an "
                "interaction use x1:x2 or x1*x2."
            ),
            diagnostics={"formula": formula},
        )

    try:
        import patsy
    except ImportError as exc:  # pragma: no cover - patsy is a core dep
        raise MethodIncompatibility(
            f"{function}: covariate formulas need patsy, which is a core "
            "StatsPAI dependency but is not importable here.",
            recovery_hint="Reinstall statspai, or pass x=['col', ...].",
            diagnostics={"formula": formula},
        ) from exc

    from ..core.utils import _coerce_string_extension_dtypes

    try:
        from ..core.utils import formula_eval_env, r_formula_idioms

        design = patsy.dmatrix(
            r_formula_idioms(rhs),
            # int16 / int8 columns wrap around inside I(x**2)
            _coerce_string_extension_dtypes(data),
            eval_env=formula_eval_env(),
            return_type="dataframe",
            NA_action="drop",
        )
    except Exception as exc:
        raise MethodIncompatibility(
            f"{function}: could not evaluate the covariate formula "
            f"{formula!r}: {type(exc).__name__}: {exc}",
            recovery_hint=(
                "Check that every name exists in the data and that "
                "transformations use Python syntax (I(x**2), np.log(x))."
            ),
            diagnostics={"formula": formula},
        ) from exc

    cols = [c for c in design.columns if c != "Intercept"]
    if not cols:
        return data, []

    # patsy drops incomplete rows; reindex so the caller sees the original
    # row set with NaN where a covariate could not be built.
    design = design[cols].reindex(data.index)

    # A plain term like `~ lpop` yields a column named `lpop` that already
    # exists and holds the same values — that is the ordinary case, not a
    # clash, and the existing column is reused untouched. A same-named
    # column holding *different* values would silently redefine the
    # caller's data, so that is rejected.
    out = data.copy()
    clashes = []
    for c in cols:
        new_vals = design[c].to_numpy(dtype=float)
        if c in data.columns:
            try:
                old_vals = data[c].to_numpy(dtype=float)
            except (TypeError, ValueError):
                clashes.append(c)
                continue
            if np.allclose(old_vals, new_vals, equal_nan=True):
                continue  # identical passthrough term; leave the column be
            clashes.append(c)
            continue
        out[c] = new_vals
    if clashes:
        raise MethodIncompatibility(
            f"{function}: covariate formula {formula!r} builds column(s) "
            f"{clashes} that already exist in the data with different "
            "values, so materialising them would redefine the caller's "
            "own columns.",
            recovery_hint=(
                "Rename the existing column, or pass x=['col', ...] "
                "directly instead of a formula."
            ),
            diagnostics={"formula": formula, "clashes": clashes},
        )
    return out, cols


def calendar_time_aware(
    *, time: str, cohort: Optional[str] = None
) -> Callable[[_F], _F]:
    """Let an estimator take date-typed time and cohort columns.

    ``time`` and ``cohort`` name the estimator's parameters that hold the
    two column names. When the time column is date-typed the call is made
    on the period-numbered copy from :func:`index_calendar_time`, and the
    numbering is stored in ``result.model_info['calendar_time']``. Any other
    call passes straight through.
    """

    def decorate(fn: _F) -> _F:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = sig.bind(*args, **kwargs)
            except TypeError:
                return fn(*args, **kwargs)
            a = bound.arguments
            data = a.get("data")
            t_col = a.get(time)
            g_col = a.get(cohort) if cohort is not None else None
            if not isinstance(data, pd.DataFrame) or not isinstance(t_col, str):
                return fn(*args, **kwargs)
            cols = [c for c in (t_col, g_col) if isinstance(c, str)]
            if not any(c in data.columns and _is_calendar(data[c]) for c in cols):
                return fn(*args, **kwargs)
            a["data"], info = index_calendar_time(
                data,
                t_col,
                g_col if isinstance(g_col, str) else None,
                function=fn.__name__,
            )
            result = fn(*bound.args, **bound.kwargs)
            model_info = getattr(result, "model_info", None)
            if info is not None and isinstance(model_info, dict):
                model_info["calendar_time"] = info
            return result

        return cast(_F, wrapper)

    return decorate
