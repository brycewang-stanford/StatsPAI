"""
Aggregated group-time ATTs for staggered DID with multiplier bootstrap.

Implements the four aggregation schemes of Callaway & Sant'Anna (2021):

- ``simple``   — cohort-size-weighted average over all post-treatment (g, t)
- ``dynamic``  — event-study: average ATT by relative time e = t − g
- ``group``    — average ATT per cohort g across its post-treatment periods
  (the reported overall weights those θ(g) by treated cohort size)
- ``calendar`` — average ATT per calendar time t across already-treated cohorts

Inference is by multiplier bootstrap applied to the influence functions
of the underlying ATT(g, t) estimates, with Rademacher (±1) weights —
matching what the R package ``did`` actually draws via
``BMisc::multiplier_bootstrap`` (its docs cite Mammen 1993, but the
implementation is Rademacher).  This reproduces the uniform
(simultaneous) confidence bands that are the signature of R ``did`` /
Stata ``csdid``.

References
----------
Callaway, B. and Sant'Anna, P.H.C. (2021).
    "Difference-in-Differences with Multiple Time Periods."
    *Journal of Econometrics*, 225(2), 200-230.
    Section 4 (aggregated parameters) and Section 4.2
    (uniform inference via multiplier bootstrap). [@callaway2021difference]

Mammen, E. (1993).
    "Bootstrap and Wild Bootstrap for High Dimensional Linear Models."
    *Annals of Statistics*, 21(1), 255-285.
    Two-point multiplier distribution used for the wild weights. [@mammen1993bootstrap]
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ._core import multiplier_bootstrap as _core_multiplier_bootstrap

# ======================================================================
# Public API
# ======================================================================


def aggte(
    result: CausalResult,
    type: str = "simple",
    balance_e: Optional[int] = None,
    min_e: float = -np.inf,
    max_e: float = np.inf,
    na_rm: bool = True,
    bstrap: bool = True,
    boot_type: str = "multiplier",
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
        If ``True``, compute SE / CI by multiplier bootstrap (Rademacher
        weights, matching the R ``did`` implementation) on the influence
        functions — required for the uniform ``cband``. If ``False``, use
        the closed-form influence-function SE instead, which carries the
        same cross-cell covariances but gives pointwise intervals only.
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
        - ``'group'``    — treated-cohort-size weighted average of the
          per-cohort θ(g) estimates
        - ``'calendar'`` — simple average of the per-calendar-time θ(t)
          estimates

        ``.detail`` is a tidy frame with one row per aggregation cell and
        both pointwise and (if requested) uniform bands.

    References
    ----------
    Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences
    with multiple time periods. *Journal of Econometrics*, 225(2),
    200-230. [@callaway2021difference]

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=8, staggered=True, seed=42)
    >>> df['first_treat'] = df['first_treat'].fillna(0)  # 0 = never-treated
    >>> cs = sp.callaway_santanna(df, y='y', g='first_treat', t='time', i='unit')
    >>> es = sp.aggte(cs, type='dynamic', cband=True, n_boot=200, random_state=42)
    >>> es.estimand
    'ATT'
    >>> grp = sp.aggte(cs, type='group', n_boot=200, random_state=42)
    >>> bool('group' in grp.detail.columns)  # one row per cohort
    True
    """
    if type not in ("simple", "dynamic", "group", "calendar"):
        raise ValueError(
            f"type must be one of 'simple', 'dynamic', 'group', 'calendar', "
            f"got {type!r}"
        )
    if boot_type != "multiplier":
        raise NotImplementedError(
            "only boot_type='multiplier' is currently supported " "(csdid parity)"
        )

    detail = result.detail
    inf_matrix = result._influence_funcs
    model_info = result.model_info or {}
    cohort_sizes = model_info.get("cohort_sizes")
    n_units = model_info.get("n_units", result.n_obs)

    if detail is None or len(detail) == 0:
        raise ValueError(
            "result has no ATT(g,t) detail to aggregate — was this produced "
            "by callaway_santanna()?"
        )
    # aggte needs the CS (g, t) grid; SA / BJS / dCDH event studies don't
    # carry it and would otherwise crash deep inside the weight builders.
    _required_cols = {"group", "time", "att", "se", "relative_time"}
    _have = set(detail.columns)
    if not _required_cols.issubset(_have):
        raise ValueError(
            "aggte() requires a Callaway–Sant'Anna result (detail frame "
            f"with columns {sorted(_required_cols)}).  Got a result of "
            f"method {result.method!r} with columns {sorted(_have)}.  For "
            "Sun–Abraham or BJS event studies use honest_did() directly."
        )
    if inf_matrix is None:
        # Analytic fallback still works but only gives pointwise intervals.
        bstrap = False

    # Optional pre-filter: drop NA ATT(g, t) before aggregating.
    if na_rm:
        finite = np.isfinite(detail["att"].values) & np.isfinite(detail["se"].values)
        if not finite.all():
            detail = detail.loc[finite].reset_index(drop=True)
            if inf_matrix is not None:
                inf_matrix = inf_matrix[:, finite]

    # Optional balancing for event study (Callaway-Sant'Anna 2021, eq. 3.8).
    if type == "dynamic" and balance_e is not None:
        detail, inf_matrix = _apply_balance_e(detail, inf_matrix, balance_e)

    # Build the weight matrix W: rows = reported cells, cols = ATT(g, t).
    if type == "simple":
        labels, W = _weights_simple(detail, cohort_sizes)
        dim_name = "overall"
    elif type == "dynamic":
        labels, W = _weights_dynamic(detail, cohort_sizes, min_e, max_e)
        dim_name = "relative_time"
    elif type == "group":
        labels, W = _weights_group(detail, cohort_sizes)
        dim_name = "group"
    else:  # 'calendar'
        labels, W = _weights_calendar(detail, cohort_sizes)
        dim_name = "time"

    if W.shape[0] == 0:
        raise ValueError(
            f"no aggregation cells available for type={type!r} after "
            "filtering — check min_e / max_e / balance_e"
        )

    att_vec = detail["att"].values
    est_cells = W @ att_vec  # shape (K,)

    # Cluster ids attached by callaway_santanna(clustervars=...) — the
    # bootstrap then resamples clusters instead of units (R did::mboot).
    cluster_ids = model_info.get("_cluster_ids")

    # SE + CI per cell, plus uniform band if requested.
    if bstrap and inf_matrix is not None:
        se_cells, crit_unif = _multiplier_bootstrap(
            W,
            inf_matrix,
            n_units,
            alpha,
            n_boot,
            random_state,
            cluster_ids=cluster_ids,
        )
    elif inf_matrix is not None:
        # ⚠️ correctness fix: the non-bootstrap path used to sum the
        # per-cell variances as if the ATT(g, t) cells were independent.
        # They are not — cells share control units, so the omitted
        # covariances are large and positive and the SE came out ~0.64x
        # the truth.  Aggregate through the influence functions instead,
        # which is the same estimator R ``did`` uses and matches the
        # ``bstrap=True`` branch below.
        se_cells = _analytic_se_influence(W, inf_matrix, n_units)
        crit_unif = stats.norm.ppf(1 - alpha / 2)
    else:
        se_cells = _analytic_se(W, detail)
        crit_unif = stats.norm.ppf(1 - alpha / 2)

    z_point = stats.norm.ppf(1 - alpha / 2)
    denom = np.where(se_cells > 0, se_cells, np.nan)
    pval = 2 * (1 - stats.norm.cdf(np.abs(est_cells / denom)))

    out = pd.DataFrame(
        {
            dim_name: labels,
            "att": est_cells,
            "se": se_cells,
            "ci_lower": est_cells - z_point * se_cells,
            "ci_upper": est_cells + z_point * se_cells,
            "pvalue": pval,
        }
    )
    if cband and type != "simple":
        out["cband_lower"] = est_cells - crit_unif * se_cells
        out["cband_upper"] = est_cells + crit_unif * se_cells
        out["crit_val_uniform"] = crit_unif

    # "Overall" summary — matches R's did::aggte print() convention:
    #   simple   : the single cohort-share-weighted overall ATT
    #   dynamic  : simple average of POST-treatment event times only
    #              (pre-treatment cells are placebos, not part of the
    #              overall causal summary)
    #   group    : treated-cohort-size weighted average of the θ(g)
    #              (R did::aggte weights each cohort by its share of
    #              treated units, not equally — ⚠️ corrected)
    #   calendar : simple average across calendar times (all post-treatment
    #              by construction)
    if type == "simple":
        overall_est = float(est_cells[0])
        overall_se = float(se_cells[0])
    else:
        if type == "dynamic":
            post_mask_agg = np.asarray(labels, dtype=float) >= 0
            if not post_mask_agg.any():
                # No post-treatment cells survived the min_e / max_e filter
                # — fall back to the legacy "mean of all reported cells"
                # behaviour so the caller still gets a number.
                post_mask_agg = np.ones(W.shape[0], dtype=bool)
            idx = np.where(post_mask_agg)[0]
            w_overall = np.zeros(W.shape[0])
            w_overall[idx] = 1.0 / idx.size
        elif type == "group":
            # ⚠️ correctness fix: R ``did::aggte(type="group")`` reports
            # the overall as sum_g (p_g / sum p_g) * theta(g), i.e. weighted
            # by each cohort's share of treated units.  We used 1/K.
            shares = _cohort_weight_series(detail, cohort_sizes)
            w_overall = np.array(
                [float(shares.get(int(g), 0.0)) for g in labels], dtype=float
            )
            s_overall = w_overall.sum()
            if s_overall <= 0:
                # No usable cohort sizes — fall back to the equal-weight mean
                # rather than emitting a zero-weight (and hence zero) ATT.
                w_overall = np.full(W.shape[0], 1.0 / W.shape[0])
            else:
                w_overall = w_overall / s_overall
        else:
            w_overall = np.full(W.shape[0], 1.0 / W.shape[0])
        overall_est = float(w_overall @ est_cells)
        if bstrap and inf_matrix is not None:
            if cluster_ids is not None:
                # Cluster-aware: bootstrap the single overall combination
                # so the SE resamples clusters like the per-cell SEs do.
                se_overall_arr, _ = _multiplier_bootstrap(
                    (w_overall @ W).reshape(1, -1),
                    inf_matrix,
                    n_units,
                    alpha,
                    n_boot,
                    random_state,
                    cluster_ids=cluster_ids,
                )
                overall_se = float(se_overall_arr[0])
            else:
                agg_inf = (w_overall @ W) @ inf_matrix.T
                overall_se = float(np.sqrt(np.mean(agg_inf**2) / n_units))
        elif inf_matrix is not None:
            # ⚠️ correctness fix: same independence bug as the per-cell
            # path above — aggregate through the influence functions so the
            # cross-cell covariances are carried.
            overall_se = float(
                _analytic_se_influence(
                    (w_overall @ W).reshape(1, -1), inf_matrix, n_units
                )[0]
            )
        else:
            overall_se = float(np.sqrt(np.sum((w_overall**2) * se_cells**2)))

    overall_z = overall_est / overall_se if overall_se > 0 else 0.0
    overall_pval = float(2 * (1 - stats.norm.cdf(abs(overall_z))))
    overall_ci = (
        overall_est - z_point * overall_se,
        overall_est + z_point * overall_se,
    )

    agg_info = {
        "aggregation": type,
        "balance_e": balance_e,
        "min_e": min_e,
        "max_e": max_e,
        "bstrap": bstrap,
        "n_boot": n_boot if bstrap else 0,
        "cband": cband and type != "simple",
        "crit_val_uniform": float(crit_unif),
        "n_units": n_units,
        "source_method": result.method,
    }

    _result = CausalResult(
        method=f"Callaway and Sant'Anna (2021) — aggte[{type}]",
        estimand="ATT",
        estimate=overall_est,
        se=overall_se,
        pvalue=overall_pval,
        ci=overall_ci,
        alpha=alpha,
        n_obs=result.n_obs,
        detail=out,
        model_info=agg_info,
        _influence_funcs=inf_matrix,
        _citation_key="callaway_santanna",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        from ..output._lineage import get_provenance

        upstream = get_provenance(result)
        _attach_prov(
            _result,
            function="sp.did.aggte",
            params={
                "type": type,
                "balance_e": balance_e,
                "min_e": min_e if not np.isinf(min_e) else None,
                "max_e": max_e if not np.isinf(max_e) else None,
                "na_rm": na_rm,
                "bstrap": bstrap,
                "boot_type": boot_type,
                "n_boot": n_boot,
                "cband": cband,
                "alpha": alpha,
                "random_state": random_state,
                "upstream_run_id": upstream.run_id if upstream else None,
                "upstream_function": upstream.function if upstream else None,
            },
            # aggte's input is the upstream CausalResult, not a frame —
            # data_hash flows in via upstream_run_id.
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


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
    cohorts = sorted(detail["group"].unique())
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
    post = (detail["relative_time"] >= 0).values
    if not post.any():
        return np.array(["overall"]), np.zeros((0, len(detail)))
    shares = _cohort_weight_series(detail, cohort_sizes)
    raw = np.where(post, detail["group"].map(shares).values.astype(float), 0.0)
    s = raw.sum()
    if s <= 0:
        return np.array(["overall"]), np.zeros((0, len(detail)))
    W = (raw / s).reshape(1, -1)
    return np.array(["overall"]), W


def _weights_dynamic(
    detail: pd.DataFrame,
    cohort_sizes: Optional[pd.Series],
    min_e: float,
    max_e: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Event-study weights (CS2021 eq. 3.7)."""
    shares = _cohort_weight_series(detail, cohort_sizes)
    rel = detail["relative_time"].values
    keep = (rel >= min_e) & (rel <= max_e)
    labels = sorted({int(e) for e, ok in zip(rel, keep) if ok})
    W = np.zeros((len(labels), len(detail)))
    for row, e in enumerate(labels):
        mask = (rel == e) & keep
        if not mask.any():
            continue
        w_raw = np.where(mask, detail["group"].map(shares).values.astype(float), 0.0)
        s = w_raw.sum()
        if s > 0:
            W[row] = w_raw / s
    return np.array(labels), W


def _weights_group(
    detail: pd.DataFrame,
    cohort_sizes: Optional[pd.Series],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cohort aggregation θ(g) (CS2021 eq. 3.9 applied within g)."""
    cohorts = sorted(detail["group"].unique())
    post = (detail["relative_time"] >= 0).values
    rows = []
    labels = []
    for g in cohorts:
        mask = (detail["group"].values == g) & post
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
    times = sorted(detail["time"].unique())
    post = (detail["relative_time"] >= 0).values
    rows = []
    labels = []
    for t in times:
        mask = (detail["time"].values == t) & post
        if not mask.any():
            continue
        w_raw = np.where(mask, detail["group"].map(shares).values.astype(float), 0.0)
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
    for g, sub in detail.groupby("group"):
        present = set(sub.loc[sub["relative_time"] >= 0, "relative_time"].astype(int))
        if required.issubset(present):
            good.append(g)
    keep = (
        detail["group"].isin(good).values
        & (detail["relative_time"] <= balance_e).values
    )
    new_detail = detail.loc[keep].reset_index(drop=True)
    new_inf = inf_matrix[:, keep] if inf_matrix is not None else None
    return new_detail, new_inf


# ======================================================================
# Inference
# ======================================================================


def _analytic_se_influence(
    W: np.ndarray,
    inf_matrix: np.ndarray,
    n_units: int,
) -> np.ndarray:
    """Covariance-aware analytic SE for each row of ``W``.

    Forms ψ = Ψ W′ (one column per aggregation cell) and returns
    ``sqrt(mean(ψ²) / n)`` — the influence-function standard error R
    ``did`` reports, and the closed-form limit of the multiplier
    bootstrap in :func:`_multiplier_bootstrap`.  Because it works on the
    influence functions rather than the per-cell variances, it carries
    the covariance between ATT(g, t) cells that share control units.
    """
    psi = inf_matrix @ W.T  # (n_units, K)
    return np.asarray(np.sqrt(np.mean(psi**2, axis=0) / n_units), dtype=float)


def _analytic_se(W: np.ndarray, detail: pd.DataFrame) -> np.ndarray:
    """Last-resort SE that *assumes independence* across (g, t).

    Only used when no influence-function matrix is available.  ATT(g, t)
    cells share control units, so the omitted covariances are positive
    and this understates the SE — it is **anti-conservative**, not
    conservative.  Prefer :func:`_analytic_se_influence`.
    """
    v = detail["se"].values ** 2
    return np.asarray(np.sqrt((W**2) @ v), dtype=float)


def _multiplier_bootstrap(
    W: np.ndarray,
    inf_matrix: np.ndarray,
    n_units: int,
    alpha: float,
    n_boot: int,
    random_state: Optional[int],
    cluster_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Multiplier bootstrap (Rademacher weights) on the influence functions.

    Thin wrapper over :func:`statspai.did._core.multiplier_bootstrap`
    that first forms the K linear combinations ``ψ = IF · W'``. When
    ``cluster_ids`` is given the bootstrap resamples clusters (R
    ``did::mboot`` convention).

    Returns
    -------
    se_cells : ndarray of shape (K,)
        Pointwise standard errors.
    crit_unif : float
        Uniform (sup-t) critical value at level ``1 - alpha``.
    """
    psi = inf_matrix @ W.T  # (n_units, K)
    return _core_multiplier_bootstrap(
        psi,
        n_units,
        alpha,
        n_boot,
        random_state,
        weight_type="rademacher",
        cluster_ids=cluster_ids,
    )
