"""
Aggregated group-time ATTs for staggered DID with multiplier bootstrap.

Implements the four aggregation schemes of Callaway & Sant'Anna (2021):

- ``simple``   — cohort-size-weighted average over all post-treatment (g, t)
- ``dynamic``  — event-study: average ATT by relative time e = t − g
- ``group``    — average ATT per cohort g across its post-treatment periods
- ``calendar`` — average ATT per calendar time t across already-treated cohorts

Inference is by Mammen (1993) multiplier bootstrap applied to the influence
functions of the underlying ATT(g, t) estimates.  This reproduces the
uniform (simultaneous) confidence bands that are the signature of the R
package ``did`` / Python package ``csdid``.

References
----------
Callaway, B. and Sant'Anna, P.H.C. (2021).
    "Difference-in-Differences with Multiple Time Periods."
    *Journal of Econometrics*, 225(2), 200-230.
    Section 4 (aggregated parameters) and Section 4.2
    (uniform inference via multiplier bootstrap).

Mammen, E. (1993).
    "Bootstrap and Wild Bootstrap for High Dimensional Linear Models."
    *Annals of Statistics*, 21(1), 255-285.
    Two-point multiplier distribution used for the wild weights.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ======================================================================
# Public API
# ======================================================================

def aggte(
    result: CausalResult,
    type: str = 'simple',
    balance_e: Optional[int] = None,
    min_e: float = -np.inf,
    max_e: float = np.inf,
    na_rm: bool = True,
    bstrap: bool = True,
    boot_type: str = 'multiplier',
    n_boot: int = 1000,
    cband: bool = True,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> CausalResult:
    """
    Aggregate group-time ATT(g, t) estimates from ``callaway_santanna``.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`callaway_santanna`. Must contain influence
        functions in ``result._influence_funcs``.
    type : {'simple', 'dynamic', 'group', 'calendar'}, default 'simple'
        Aggregation scheme. ``'dynamic'`` is the event study.
    balance_e : int, optional
        Only used when ``type='dynamic'``. If given, the event study
        restricts attention to cohorts observed over the *balanced* set of
        event times ``{-∞ ≤ e ≤ balance_e}``. This is the ``balance_e``
        option of R's :func:`did::aggte`.
    min_e, max_e : float, default (-inf, inf)
        Truncate the reported event-time window.
    na_rm : bool, default True
        Drop ATT(g, t) with missing / infinite SE before aggregating.
    bstrap : bool, default True
        If ``True``, compute SE / CI by Mammen multiplier bootstrap on the
        influence functions. If ``False``, fall back to the analytic
        (delta-method) SE already attached to each ATT(g, t).
    boot_type : {'multiplier'}, default 'multiplier'
        Only ``'multiplier'`` is supported; kept for ``csdid`` parity.
    n_boot : int, default 1000
        Number of bootstrap replications.
    cband : bool, default True
        If ``True`` and ``type != 'simple'``, report a *uniform* confidence
        band (sup-t critical value) across the aggregation dimension.
        Otherwise pointwise intervals.
    alpha : float, default 0.05
        Nominal level for confidence intervals.
    random_state : int, optional
        Seed for the multiplier bootstrap.

    Returns
    -------
    CausalResult
        ``.estimate`` / ``.se`` hold the overall aggregated ATT,
        matching R's ``did::aggte`` print convention:

        - ``'simple'``   — the single cohort-share-weighted overall ATT
        - ``'dynamic'``  — simple average of *post-treatment* event times
          (e ≥ 0); pre-treatment cells are placebos and excluded
        - ``'group'``    — simple average of the per-cohort θ(g) estimates
        - ``'calendar'`` — simple average of the per-calendar-time θ(t)
          estimates

        ``.detail`` is a tidy frame with one row per aggregation cell and
        both pointwise and (if requested) uniform bands.

    Examples
    --------
    >>> from statspai.did import callaway_santanna, aggte
    >>> cs  = callaway_santanna(df, y='y', g='g', t='t', i='id')
    >>> es  = aggte(cs, type='dynamic', cband=True, random_state=42)
    >>> grp = aggte(cs, type='group')
    >>> cal = aggte(cs, type='calendar')
    """
    if type not in ('simple', 'dynamic', 'group', 'calendar'):
        raise ValueError(
            f"type must be one of 'simple', 'dynamic', 'group', 'calendar', "
            f"got {type!r}"
        )
    if boot_type != 'multiplier':
        raise NotImplementedError(
            "only boot_type='multiplier' is currently supported "
            "(csdid parity)"
        )

    detail = result.detail
    inf_matrix = result._influence_funcs
    model_info = result.model_info or {}
    cohort_sizes = model_info.get('cohort_sizes')
    n_units = model_info.get('n_units', result.n_obs)

    if detail is None or len(detail) == 0:
        raise ValueError(
            "result has no ATT(g,t) detail to aggregate — was this produced "
            "by callaway_santanna()?"
        )
    if inf_matrix is None:
        # Analytic fallback still works but only gives pointwise intervals.
        bstrap = False

    # Optional pre-filter: drop NA ATT(g, t) before aggregating.
    if na_rm:
        finite = np.isfinite(detail['att'].values) & np.isfinite(detail['se'].values)
        if not finite.all():
            detail = detail.loc[finite].reset_index(drop=True)
            if inf_matrix is not None:
                inf_matrix = inf_matrix[:, finite]

    # Optional balancing for event study (Callaway-Sant'Anna 2021, eq. 3.8).
    if type == 'dynamic' and balance_e is not None:
        detail, inf_matrix = _apply_balance_e(detail, inf_matrix, balance_e)

    # Build the weight matrix W: rows = reported cells, cols = ATT(g, t).
    if type == 'simple':
        labels, W = _weights_simple(detail, cohort_sizes)
        dim_name = 'overall'
    elif type == 'dynamic':
        labels, W = _weights_dynamic(detail, cohort_sizes, min_e, max_e)
        dim_name = 'relative_time'
    elif type == 'group':
        labels, W = _weights_group(detail, cohort_sizes)
        dim_name = 'group'
    else:  # 'calendar'
        labels, W = _weights_calendar(detail, cohort_sizes)
        dim_name = 'time'

    if W.shape[0] == 0:
        raise ValueError(
            f"no aggregation cells available for type={type!r} after "
            "filtering — check min_e / max_e / balance_e"
        )

    att_vec = detail['att'].values
    est_cells = W @ att_vec  # shape (K,)

    # SE + CI per cell, plus uniform band if requested.
    if bstrap and inf_matrix is not None:
        se_cells, crit_unif = _multiplier_bootstrap(
            W, inf_matrix, n_units, alpha, n_boot, random_state,
        )
    else:
        se_cells = _analytic_se(W, detail)
        crit_unif = stats.norm.ppf(1 - alpha / 2)

    z_point = stats.norm.ppf(1 - alpha / 2)
    pval = 2 * (1 - stats.norm.cdf(np.abs(est_cells / np.where(se_cells > 0, se_cells, np.nan))))

    out = pd.DataFrame({
        dim_name: labels,
        'att': est_cells,
        'se': se_cells,
        'ci_lower': est_cells - z_point * se_cells,
        'ci_upper': est_cells + z_point * se_cells,
        'pvalue': pval,
    })
    if cband and type != 'simple':
        out['cband_lower'] = est_cells - crit_unif * se_cells
        out['cband_upper'] = est_cells + crit_unif * se_cells
        out['crit_val_uniform'] = crit_unif

    # "Overall" summary — matches R's did::aggte print() convention:
    #   simple   : the single cohort-share-weighted overall ATT
    #   dynamic  : simple average of POST-treatment event times only
    #              (pre-treatment cells are placebos, not part of the
    #              overall causal summary)
    #   group    : simple average across cohorts (all post-treatment by
    #              construction of the θ(g) weights)
    #   calendar : simple average across calendar times (all post-treatment
    #              by construction)
    if type == 'simple':
        overall_est = float(est_cells[0])
        overall_se = float(se_cells[0])
    else:
        if type == 'dynamic':
            post_mask_agg = np.asarray(labels, dtype=float) >= 0
            if not post_mask_agg.any():
                # No post-treatment cells survived the min_e / max_e filter
                # — fall back to the legacy "mean of all reported cells"
                # behaviour so the caller still gets a number.
                post_mask_agg = np.ones(W.shape[0], dtype=bool)
            idx = np.where(post_mask_agg)[0]
            w_overall = np.zeros(W.shape[0])
            w_overall[idx] = 1.0 / idx.size
        else:
            w_overall = np.full(W.shape[0], 1.0 / W.shape[0])
        overall_est = float(w_overall @ est_cells)
        if bstrap and inf_matrix is not None:
            agg_inf = (w_overall @ W) @ inf_matrix.T
            overall_se = float(np.sqrt(np.mean(agg_inf ** 2) / n_units))
        else:
            overall_se = float(
                np.sqrt(np.sum((w_overall ** 2) * se_cells ** 2))
            )

    overall_z = overall_est / overall_se if overall_se > 0 else 0.0
    overall_pval = float(2 * (1 - stats.norm.cdf(abs(overall_z))))
    overall_ci = (
        overall_est - z_point * overall_se,
        overall_est + z_point * overall_se,
    )

    agg_info = {
        'aggregation': type,
        'balance_e': balance_e,
        'min_e': min_e,
        'max_e': max_e,
        'bstrap': bstrap,
        'n_boot': n_boot if bstrap else 0,
        'cband': cband and type != 'simple',
        'crit_val_uniform': float(crit_unif),
        'n_units': n_units,
        'source_method': result.method,
    }

    return CausalResult(
        method=f"Callaway and Sant'Anna (2021) — aggte[{type}]",
        estimand='ATT',
        estimate=overall_est,
        se=overall_se,
        pvalue=overall_pval,
        ci=overall_ci,
        alpha=alpha,
        n_obs=result.n_obs,
        detail=out,
        model_info=agg_info,
        _influence_funcs=inf_matrix,
        _citation_key='callaway_santanna',
    )


# ======================================================================
# Weight builders
# ======================================================================

def _cohort_weight_series(
    detail: pd.DataFrame,
    cohort_sizes: Optional[pd.Series],
) -> pd.Series:
    """Return a Series mapping cohort g -> share used as aggregation weight.

    Uses the empirical cohort sizes attached to the CausalResult.  Falls
    back to equal weights if unavailable.  Keys are aligned to the unique
    cohorts present in ``detail``.
    """
    cohorts = sorted(detail['group'].unique())
    if cohort_sizes is None:
        sizes = pd.Series({g: 1.0 for g in cohorts})
    else:
        sizes = pd.Series({g: float(cohort_sizes.get(g, 0.0)) for g in cohorts})
    total = sizes.sum()
    if total <= 0:
        sizes = pd.Series({g: 1.0 for g in cohorts})
        total = float(len(cohorts))
    return sizes / total


def _weights_simple(
    detail: pd.DataFrame,
    cohort_sizes: Optional[pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Weights for the 'simple' aggregation (CS2021 eq. 3.6)."""
    post = (detail['relative_time'] >= 0).values
    if not post.any():
        return np.array(['overall']), np.zeros((0, len(detail)))
    shares = _cohort_weight_series(detail, cohort_sizes)
    raw = np.where(post, detail['group'].map(shares).values.astype(float), 0.0)
    s = raw.sum()
    if s <= 0:
        return np.array(['overall']), np.zeros((0, len(detail)))
    W = (raw / s).reshape(1, -1)
    return np.array(['overall']), W


def _weights_dynamic(
    detail: pd.DataFrame,
    cohort_sizes: Optional[pd.Series],
    min_e: float,
    max_e: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Event-study weights (CS2021 eq. 3.7)."""
    shares = _cohort_weight_series(detail, cohort_sizes)
    rel = detail['relative_time'].values
    keep = (rel >= min_e) & (rel <= max_e)
    labels = sorted({int(e) for e, ok in zip(rel, keep) if ok})
    W = np.zeros((len(labels), len(detail)))
    for row, e in enumerate(labels):
        mask = (rel == e) & keep
        if not mask.any():
            continue
        w_raw = np.where(mask, detail['group'].map(shares).values.astype(float), 0.0)
        s = w_raw.sum()
        if s > 0:
            W[row] = w_raw / s
    return np.array(labels), W


def _weights_group(
    detail: pd.DataFrame,
    cohort_sizes: Optional[pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cohort aggregation θ(g) (CS2021 eq. 3.9 applied within g)."""
    cohorts = sorted(detail['group'].unique())
    post = (detail['relative_time'] >= 0).values
    rows = []
    labels = []
    for g in cohorts:
        mask = (detail['group'].values == g) & post
        if not mask.any():
            continue
        w_raw = mask.astype(float)
        s = w_raw.sum()
        w = w_raw / s
        rows.append(w)
        labels.append(int(g))
    W = np.vstack(rows) if rows else np.zeros((0, len(detail)))
    return np.array(labels), W


def _weights_calendar(
    detail: pd.DataFrame,
    cohort_sizes: Optional[pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-calendar-time aggregation θ(t) (CS2021 eq. 3.10)."""
    shares = _cohort_weight_series(detail, cohort_sizes)
    times = sorted(detail['time'].unique())
    post = (detail['relative_time'] >= 0).values
    rows = []
    labels = []
    for t in times:
        mask = (detail['time'].values == t) & post
        if not mask.any():
            continue
        w_raw = np.where(mask, detail['group'].map(shares).values.astype(float), 0.0)
        s = w_raw.sum()
        if s <= 0:
            continue
        rows.append(w_raw / s)
        labels.append(int(t))
    W = np.vstack(rows) if rows else np.zeros((0, len(detail)))
    return np.array(labels), W


def _apply_balance_e(
    detail: pd.DataFrame,
    inf_matrix: Optional[np.ndarray],
    balance_e: int,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Restrict to cohorts observed for all e ∈ [0, balance_e] (eq. 3.8).

    Keeps only cohorts g such that every e in {0, …, balance_e} has a
    corresponding ATT(g, g+e) in ``detail``, then also drops rows with
    e > balance_e so the reported window is balanced across cohorts.
    """
    required = set(range(0, balance_e + 1))
    good = []
    for g, sub in detail.groupby('group'):
        present = set(sub.loc[sub['relative_time'] >= 0, 'relative_time'].astype(int))
        if required.issubset(present):
            good.append(g)
    keep = (
        detail['group'].isin(good).values
        & (detail['relative_time'] <= balance_e).values
    )
    new_detail = detail.loc[keep].reset_index(drop=True)
    new_inf = inf_matrix[:, keep] if inf_matrix is not None else None
    return new_detail, new_inf


# ======================================================================
# Inference
# ======================================================================

def _analytic_se(W: np.ndarray, detail: pd.DataFrame) -> np.ndarray:
    """Conservative SE assuming independence across (g, t)."""
    v = detail['se'].values ** 2
    return np.sqrt((W ** 2) @ v)


def _multiplier_bootstrap(
    W: np.ndarray,
    inf_matrix: np.ndarray,
    n_units: int,
    alpha: float,
    n_boot: int,
    random_state: Optional[int],
) -> Tuple[np.ndarray, float]:
    """Mammen (1993) multiplier bootstrap on the influence functions.

    Returns
    -------
    se_cells : ndarray of shape (K,)
        Pointwise standard errors.
    crit_unif : float
        Uniform (sup-t) critical value at level ``1 - alpha``.
    """
    rng = np.random.default_rng(random_state)
    # Influence functions of the K linear combinations: shape (n, K)
    psi = inf_matrix @ W.T  # (n_units, K)
    n = psi.shape[0]
    K = psi.shape[1]

    # Two-point Mammen weights with mean 0, variance 1.
    # P(V = (1-√5)/2) = (√5+1)/(2√5); P(V = (1+√5)/2) = (√5-1)/(2√5).
    sqrt5 = np.sqrt(5.0)
    a, b = (1 - sqrt5) / 2.0, (1 + sqrt5) / 2.0
    pa = (sqrt5 + 1.0) / (2.0 * sqrt5)

    # shape: (n_boot, n_units)
    u = rng.random((n_boot, n))
    V = np.where(u < pa, a, b)

    # Bootstrap linear-combo draws: (n_boot, K) = V @ psi / n_units
    # Subtract the sample mean (centered around 0 under H0: θ_true = θ̂).
    psi_centered = psi - psi.mean(axis=0, keepdims=True)
    boot = V @ psi_centered / n

    # Pointwise SEs from bootstrap.  Use the IQR-based rescaling that
    # the R package `did` uses (Callaway & Sant'Anna 2021, Sec. 4.2,
    # implementation details) — robust to heavy tails in multiplier
    # weights compared to raw std.
    q75 = np.quantile(boot, 0.75, axis=0)
    q25 = np.quantile(boot, 0.25, axis=0)
    iqr_norm = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
    se_cells = (q75 - q25) / iqr_norm
    # Guard against degenerate columns (e.g. a singleton cell).
    fallback_std = boot.std(axis=0, ddof=1)
    se_cells = np.where(se_cells > 0, se_cells, fallback_std)
    se_cells = np.where(se_cells > 0, se_cells, 1e-12)

    # Uniform (sup-t) critical value.
    max_t = np.max(np.abs(boot) / se_cells, axis=1)
    crit_unif = float(np.quantile(max_t, 1 - alpha))
    # Never shrink below pointwise Normal quantile.
    crit_unif = max(crit_unif, stats.norm.ppf(1 - alpha / 2))

    return se_cells, crit_unif
