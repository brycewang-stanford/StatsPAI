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

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility
from . import _cs_conventions as _conv
from ._core import cohort_share_context as _cohort_share_context
from ._core import multiplier_bootstrap as _core_multiplier_bootstrap
from ._core import weight_influence as _weight_influence
from ._validate import inference_note as _inference_note

# ======================================================================
# Public API
# ======================================================================


@accepts_aliases(seed="random_state", alp="alpha", biters="n_boot")
@_conv.track_explicit
def aggte(
    result: CausalResult,
    type: str = "simple",
    balance_e: Optional[int] = None,
    min_e: float = -np.inf,
    max_e: float = np.inf,
    na_rm: bool = True,
    bstrap: Optional[bool] = None,
    boot_type: str = "multiplier",
    n_boot: Optional[int] = None,
    cband: Optional[bool] = None,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    share_variance: bool = True,
    agg_weights: str = "did",
    parity: Optional[Literal["r", "stata"]] = None,
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
    bstrap : bool, optional
        If ``True``, compute SE / CI by multiplier bootstrap (Rademacher
        weights, matching the R ``did`` implementation) on the influence
        functions — required for the uniform ``cband``. If ``False``, use
        the closed-form influence-function SE instead, which carries the
        same cross-cell covariances but gives pointwise intervals only.
        ``None`` (default) bootstraps when ``cband=True`` or ``n_boot`` is
        given, and otherwise inherits the setting of the fit, as R
        ``did::aggte(bstrap = NULL)`` inherits ``att_gt``'s: a
        ``callaway_santanna`` fit with its default ``bstrap=False`` is
        aggregated analytically, so repeated calls return the same SE.

        .. versionchanged:: 1.33.0
           The default was ``True`` with no seed, so ``sp.aggte(fit)``
           returned a different SE on every call and disagreed with the
           analytic SE of the fit it aggregated.
    boot_type : {'multiplier'}, default 'multiplier'
        Only ``'multiplier'`` is supported; kept for ``csdid`` parity.
    n_boot : int, optional
        Number of bootstrap replications; defaults to the fit's ``biters``
        (else 1000).
    cband : bool, optional
        If ``True`` and ``type != 'simple'``, report a *uniform* confidence
        band (sup-t critical value) across the aggregation dimension.
        Otherwise pointwise intervals.  ``None`` inherits the fit's
        ``cband`` when ``bstrap`` is inherited, and otherwise follows
        ``bstrap``.
    alpha : float, default 0.05
        Nominal level for confidence intervals.
    agg_weights : {'did', 'csdid'}, default 'did'
        Which implementation's aggregation weights to use. ``'did'`` gives
        a cohort one weight -- its share of the treated -- constant across
        ``t``, which is R ``did``'s rule and what this function has always
        done. ``'csdid'`` weights each *cell* by its own treated count,
        which is Stata ``csdid``'s rule in every ``estat`` aggregate: on
        repeated cross-sections the cohort's rows in the cell's two periods,
        on an unbalanced panel (``balance='pair'``) the cohort's units seen
        in both. The standard errors carry the estimation of those counts,
        as csdid's do, in place of R's cohort-share term; the group average
        enters each cohort at its *mean* post-treatment count, held fixed,
        so a cohort observed for fewer post-treatment periods is not
        down-weighted for it (``share_variance`` does not apply). With
        sampling weights (``weights=`` of the fit) a cell's count is the
        weight mass of those treated observations, as csdid ``[iw=]``
        uses it (its ``e(gtt)`` still lists the head count). The two
        rules coincide on a balanced panel, which carries no counts and
        raises. Established against csdid 1.81 as a black box, unweighted
        and weighted (``tests/reference_parity/test_cs_parity_rc_unbal.py``).

        .. versionadded:: 1.31.0
        .. versionchanged:: 1.40.1
           All four aggregation types, unbalanced panels, and csdid's
           count term in the SEs of every aggregate.
    share_variance : bool, default True
        Carry the sampling variability of the *estimated* cohort shares
        into the aggregated variance (the Callaway & Sant'Anna reference
        convention, also R ``did``'s). ``False`` holds the cohort shares
        fixed, which is the convention of Stata ``csdid``'s
        ``estat group`` aggregate (``GAverage``); point estimates are
        identical either way and only the ``'group'`` overall (and any
        multi-cohort cell) standard error changes. Use ``False`` to
        reproduce ``csdid`` digit for digit.

        .. versionadded:: 1.29.0
    random_state : int, optional
        Seed for the multiplier bootstrap.  When the bootstrap runs without
        one, a seed is drawn and recorded in ``model_info['random_state']``
        so the reported SE can be reproduced exactly.
    parity : {'r', 'stata'}, optional
        Aggregate with the conventions of R ``did::aggte`` or Stata
        ``csdid``'s ``estat``. ``None`` (default) inherits the parity of
        the fit (``sp.callaway_santanna(parity=...)``); passing a parity
        that differs from the fit's raises, because the base period and
        comparison group are fixed when ATT(g, t) is estimated.

        - ``'r'``: cohort-share term everywhere (``share_variance=True``)
          and R ``did``'s cohort-share weights; the bootstrap follows the
          fit, as ``aggte(bstrap = NULL)`` does.
        - ``'stata'``: analytic standard errors even if the fit
          bootstrapped (``estat`` always reports them); the group
          average holds the cohort shares fixed while every other
          aggregate and every cell carries the share term; on repeated
          cross-sections and unbalanced panels every aggregate uses csdid's
          per-cell counts (``agg_weights='csdid'``).

        An explicit ``share_variance`` / ``agg_weights`` / ``bstrap`` /
        ``cband`` / ``n_boot`` that contradicts the parity raises. The
        conventions used are in ``model_info['conventions']``.

        .. versionadded:: 1.40.0

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
    from ..survey.linearization import refuse_design_result

    refuse_design_result(result, "aggte")

    # parity= (or the fit's parity) fixes the aggregation conventions of R
    # did / Stata csdid; contradicting arguments raise in there.
    parity = _conv.require_parity(parity, function="aggte")
    _p = _conv.resolve_aggte_parity(
        parity,
        fit_info=model_info,
        explicit=_conv.explicit_args(),
        type=type,
        bstrap=bstrap,
        cband=cband,
        n_boot=n_boot,
        share_variance=share_variance,
        agg_weights=agg_weights,
        has_cell_counts=_has_csdid_cells(detail, model_info),
    )
    parity = _p["parity"]
    bstrap, cband, agg_weights = _p["bstrap"], _p["cband"], _p["agg_weights"]
    share_cells, share_overall = _p["share_cells"], _p["share_overall"]

    # Inference settings follow the fit unless overridden (R did::aggte's
    # bstrap = NULL / cband = NULL / biters from att_gt).
    if bstrap is None:
        if cband is True or n_boot is not None:
            bstrap = True
        else:
            bstrap = bool(model_info.get("bstrap", False))
            if cband is None:
                cband = bool(model_info.get("cband", False)) and bstrap
    if cband is None:
        cband = bool(bstrap)
    bstrap = bool(bstrap)
    cband = bool(cband)
    if n_boot is None:
        n_boot = int(model_info.get("biters") or 1000)
    if bstrap and random_state is None:
        # Draw a seed and report it: an unseeded bootstrap SE could not be
        # reproduced, not even by the caller who just printed it.
        random_state = int(np.random.SeedSequence().entropy % (2**31 - 1))

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

    if agg_weights not in ("did", "csdid"):
        raise MethodIncompatibility(
            f"agg_weights must be 'did' or 'csdid'; got {agg_weights!r}.",
            diagnostics={"agg_weights": agg_weights},
        )
    csdid_ind: Optional[np.ndarray] = None
    csdid_rx1 = np.zeros(0)
    if agg_weights == "csdid":
        if not _has_csdid_cells(detail, model_info):
            raise MethodIncompatibility(
                "agg_weights='csdid' weights each cell by its own count of "
                "treated observations (repeated cross-sections) or of "
                "treated units seen in both of its periods (unbalanced "
                "panel), which this fit did not record. On a balanced panel "
                "the counts are the cohort sizes and the rule is R did's; "
                "the unbalanced-panel estimator of allow_unbalanced_panel="
                "True has no csdid counterpart.",
                recovery_hint=(
                    "Drop agg_weights=, or refit with "
                    "sp.callaway_santanna(panel=False) for repeated "
                    "cross-sections or the default balance='pair' for an "
                    "unbalanced panel."
                ),
                diagnostics={"columns": list(detail.columns)},
            )
        csdid_ind = _csdid_cell_indicators(detail, model_info)
        # With sampling weights csdid weights a cell by the weight mass of
        # its treated observations, not their head count (black box,
        # csdid 1.81 [iw=]; test_cs_parity_rc_unbal.py).
        if model_info.get("weighted"):
            omega = model_info.get("_unit_weights")
            if omega is None or len(omega) != csdid_ind.shape[0]:
                raise MethodIncompatibility(
                    "agg_weights='csdid' on a weighted fit needs the "
                    "sampling weight of every influence-function row, "
                    "which this result does not carry.",
                    recovery_hint="Refit with sp.callaway_santanna(weights=...).",
                    diagnostics={"weighted": True},
                )
            csdid_ind = csdid_ind * np.asarray(omega, dtype=float)[:, None]

    # Build the weight matrix W: rows = reported cells, cols = ATT(g, t).
    if csdid_ind is not None:
        labels, W, csdid_rx1 = _weights_csdid(
            detail, csdid_ind.sum(axis=0, dtype=float), type, min_e, max_e
        )
        dim_name = {
            "simple": "overall",
            "dynamic": "relative_time",
            "group": "group",
            "calendar": "time",
        }[type]
    elif type == "simple":
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

    # Per-cell aggregated influence functions.
    #
    # ⚠️ correctness fix: the aggregation weights are *estimated* cohort
    # shares, not constants.  Treating them as fixed drops their
    # influence function from the variance and the reported SE came out up to
    # ~8% too small (anti-conservative).  Build the corrected functions
    # once here; every SE below is derived from them so the analytic and
    # bootstrap paths cannot drift apart.
    unit_cohorts = model_info.get("_unit_cohorts")
    # ω from the underlying fit. The aggregation weights are cohort
    # *shares* of the target population, so under ω they must be built
    # from ω-mass, pg = mean(ω · 1{G == g}) (as R did does).
    unit_weights = model_info.get("_unit_weights")
    psi_cells: Optional[np.ndarray] = None
    if inf_matrix is not None and csdid_ind is not None:
        # Stata csdid: the cell weights are estimated treated counts, and
        # csdid carries their influence function in every aggregate. This
        # replaces R did's cohort-share term (the two coincide on a balanced
        # panel). The group average below holds its own weights fixed.
        pg_cells = ind_cells = None
        psi_cells = inf_matrix @ W.T + _csdid_count_weight_influence(
            W, att_vec, csdid_ind
        )
    elif inf_matrix is not None:
        if (
            share_cells
            and unit_cohorts is not None
            and len(unit_cohorts) == inf_matrix.shape[0]
        ):
            pg_cells, ind_cells = _cohort_share_context(
                detail["group"].values, unit_cohorts, unit_weights
            )
            psi_cells = _aggregated_influence(
                W, inf_matrix, att_vec, pg_cells, ind_cells
            )
        else:
            # Either the caller asked for fixed cohort shares
            # (``share_variance=False``, the Stata ``csdid`` convention) or
            # the result predates ``_unit_cohorts`` / is misaligned — use
            # fixed weights rather than guessing.  ``ind_cells`` stays
            # ``None`` so the group-overall block below skips the share
            # term as well.
            pg_cells = ind_cells = None
            psi_cells = inf_matrix @ W.T

    # SE + CI per cell, plus uniform band if requested.
    if bstrap and psi_cells is not None:
        se_cells, crit_unif = _bootstrap_from_influence(
            psi_cells,
            n_units,
            alpha,
            n_boot,
            random_state,
            cluster_ids=cluster_ids,
        )
    elif psi_cells is not None:
        # Aggregating through the influence functions carries the
        # covariance between ATT(g, t) cells that share control units;
        # summing per-cell variances would treat them as independent.
        se_cells = _se_from_influence(psi_cells, n_units, cluster_ids)
        crit_unif = stats.norm.ppf(1 - alpha / 2)
    else:
        se_cells = _analytic_se(W, detail)
        crit_unif = stats.norm.ppf(1 - alpha / 2)

    z_point = stats.norm.ppf(1 - alpha / 2)
    denom = np.where(se_cells > 0, se_cells, np.nan)
    pval = 2 * stats.norm.sf(np.abs(est_cells / denom))

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
    overall_inf = None
    if type == "simple":
        overall_est = float(est_cells[0])
        overall_se = float(se_cells[0])
        if psi_cells is not None:
            overall_inf = np.asarray(psi_cells[:, 0], dtype=float)
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
        elif type == "group" and csdid_ind is not None:
            # csdid's GAverage: each cohort enters with its MEAN cell
            # count, so a cohort observed for fewer post periods is not
            # down-weighted for it.
            w_overall = csdid_rx1 / csdid_rx1.sum()
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
        # The unit-level influence function of the overall aggregate. Kept
        # regardless of the SE path so downstream tests that need the whole
        # function -- not just its second moment -- can reach it; the
        # Roth-Sant'Anna functional-form test stacks one of these per
        # outcome bin to get their joint covariance.
        #
        # sqrt(mean(psi**2) / n_units) off this vector reproduces the
        # reported SE exactly for 'group' / 'dynamic' / 'calendar' (the
        # unclustered overall SE is that formula even under bstrap=True).
        # For type='simple' the reported SE is a genuine multiplier
        # bootstrap, so the two agree only up to bootstrap noise.
        if psi_cells is not None:
            # Compose the OVERALL from the already-corrected per-cell
            # functions.  The estimated-share term enters once, at the level where the
            # estimated cohort shares actually enter:
            #
            #   dynamic / calendar — the overall is an equal-weight mean
            #     over event times / calendar periods, so those weights
            #     are constants and carry no extra term.
            #   group — the overall re-weights theta(g) by each cohort's
            #     estimated share, so the term applies here instead (the
            #     per-cohort cells are single-cohort, so their share term is zero).
            overall_inf = np.asarray(psi_cells @ w_overall, dtype=float)
            if type == "group" and ind_cells is not None and share_overall:
                pg_g, ind_g = _cohort_share_context(
                    np.asarray(labels, dtype=float), unit_cohorts, unit_weights
                )
                wif_g = _weight_influence(pg_g, ind_g)
                overall_inf = overall_inf + wif_g @ est_cells

            if bstrap:
                se_overall_arr, _ = _bootstrap_from_influence(
                    overall_inf.reshape(-1, 1),
                    n_units,
                    alpha,
                    n_boot,
                    random_state,
                    cluster_ids=cluster_ids,
                )
                overall_se = float(se_overall_arr[0])
            else:
                overall_se = float(
                    _se_from_influence(overall_inf[:, None], n_units, cluster_ids)[0]
                )
        else:
            overall_se = float(np.sqrt(np.sum((w_overall**2) * se_cells**2)))

    overall_z = overall_est / overall_se if overall_se > 0 else 0.0
    overall_pval = float(2 * stats.norm.sf(abs(overall_z)))
    overall_ci = (
        overall_est - z_point * overall_se,
        overall_est + z_point * overall_se,
    )

    # Joint covariance of the reported cells. Built from the same corrected
    # per-cell influence functions as every standard error above, so a
    # downstream joint test (pre-trend Wald), sensitivity analysis
    # (honest DiD FLCI) or power calculation (Roth 2022) reads the object the
    # reported SEs came from rather than a diagonal approximation of it.
    # R ``did`` exposes the same thing as ``aggte(...)$inf.function``.
    labels_arr = np.asarray(labels)
    if psi_cells is not None:
        vcov_cells = psi_cells.T @ psi_cells / float(n_units) ** 2
        vcov_source = "influence_functions"
    else:
        vcov_cells = np.diag(np.asarray(se_cells, dtype=float) ** 2)
        vcov_source = "diagonal_fallback"
    vcv_pre = None
    pre_event_times = None
    if type == "dynamic":
        pre_idx = np.where(labels_arr.astype(float) < 0)[0]
        if pre_idx.size:
            vcv_pre = vcov_cells[np.ix_(pre_idx, pre_idx)]
            pre_event_times = [int(e) for e in labels_arr[pre_idx]]

    agg_info = {
        "aggregation": type,
        "balance_e": balance_e,
        "min_e": min_e,
        "max_e": max_e,
        "bstrap": bstrap,
        "n_boot": n_boot if bstrap else 0,
        "random_state": random_state if bstrap else None,
        "cband": cband and type != "simple",
        # Same comparison with R did::aggte's inherited defaults as on the
        # att_gt result (a single overall number has no band to widen).
        "inference_note": _inference_note(bstrap, cband or type == "simple"),
        "crit_val_uniform": float(crit_unif),
        "n_units": n_units,
        "overall_influence_function": overall_inf,
        "influence_functions": psi_cells,
        "vcov": vcov_cells,
        "vcov_source": vcov_source,
        "cell_labels": _plain_labels(labels_arr),
        "vcv_pre": vcv_pre,
        "pre_event_times": pre_event_times,
        "source_method": result.method,
        "share_variance": bool(share_overall if type == "group" else share_cells),
        "parity": parity,
        "conventions": _conv.aggte_conventions(
            model_info,
            parity=parity,
            type=type,
            bstrap=bstrap,
            share_cells=share_cells,
            share_overall=share_overall,
            agg_weights=agg_weights,
        ),
    }
    if "base_period" in (result.model_info or {}):
        # sp.honest_did reads it: its bounds need the universal-base path
        agg_info["base_period"] = result.model_info["base_period"]
    if type == "dynamic":
        # Where Sun-Abraham, stacked and BJS keep theirs, so one accessor
        # (``model_info['event_study']`` / ``sp.event_study_table``) serves
        # every event-study estimator; ``detail`` is the same frame.
        agg_info["event_study"] = out

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
                "share_variance": share_variance,
                "parity": parity,
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


def _has_csdid_cells(
    detail: Optional[pd.DataFrame], model_info: Dict[str, Any]
) -> bool:
    """Does the fit carry what csdid's cell-count weights need?

    Repeated cross-sections record each row's cohort and period; an
    unbalanced panel (``balance='pair'``) records which periods each unit is
    observed in. Balanced panels record neither: there every cell's count is
    its cohort size and csdid's weights are the cohort shares.
    """
    if detail is None:
        return False
    rcs = model_info.get("_row_cohorts") is not None and "n_treated_obs" in detail
    return bool(rcs or model_info.get("_unit_observed") is not None)


def _csdid_cell_indicators(
    detail: pd.DataFrame, model_info: Dict[str, Any]
) -> np.ndarray:
    """Which influence-function rows are treated observations of each cell.

    Boolean, one row per influence-function row and one column per row of
    ``detail``.

    Column ``k`` flags, for cell ``(g, t)`` with base period ``b``:

    - repeated cross-sections: the rows of cohort ``g`` observed at ``b`` or
      ``t`` (csdid's ``N_trt`` in ``e(gtt)``);
    - an unbalanced panel: the units of cohort ``g`` observed at both ``b``
      and ``t``, the pairs the cell is estimated from (``N_trt`` / 2).

    Stata ``csdid`` weights cell ``k`` by the column sum in every aggregate
    (black box, ``tests/reference_parity/test_cs_parity_rc_unbal.py``);
    with sampling weights by the weight mass of the flagged rows, which the
    caller applies.
    """
    bases = model_info.get("_cell_bases") or {}
    groups = detail["group"].to_numpy(dtype=float)
    times = detail["time"].to_numpy(dtype=float)
    missing = [
        (g, t) for g, t in zip(groups, times) if (float(g), float(t)) not in bases
    ]
    if missing:
        raise MethodIncompatibility(
            "agg_weights='csdid' needs the base period of every ATT(g, t), "
            f"which this result does not carry (e.g. cell {missing[0]}).",
            recovery_hint="Refit with sp.callaway_santanna and aggregate that.",
            diagnostics={"n_cells_missing": len(missing)},
        )
    base = np.array([bases[(float(g), float(t))] for g, t in zip(groups, times)])
    rg = model_info.get("_row_cohorts")
    if rg is not None:
        rg = np.asarray(rg, dtype=float)
        rt = np.asarray(model_info.get("_row_times"), dtype=float)
        return (rg[:, None] == groups[None, :]) & (
            (rt[:, None] == times[None, :]) | (rt[:, None] == base[None, :])
        )
    observed = np.asarray(model_info["_unit_observed"], dtype=bool)
    periods = [float(p) for p in model_info["_time_periods"]]
    col = {p: j for j, p in enumerate(periods)}
    cohorts = np.asarray(model_info["_unit_cohorts"], dtype=float)
    ind = np.zeros((observed.shape[0], len(groups)), dtype=bool)
    for k, (g, t, b) in enumerate(zip(groups, times, base)):
        ind[:, k] = (
            (cohorts == g) & observed[:, col[float(t)]] & observed[:, col[float(b)]]
        )
    return ind


def _weights_csdid(
    detail: pd.DataFrame,
    counts: np.ndarray,
    type: str,
    min_e: float,
    max_e: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stata ``csdid``'s aggregation weights: each cell by its treated count.

    Every aggregate is a count-weighted mean of the cells it covers: the
    post-treatment cells (``simple``), a cohort's post-treatment cells
    (``group``), the post-treatment cells of one period (``calendar``) or
    the cells of one event time, pre-treatment ones included
    (``dynamic``). R ``did`` weights by cohort shares instead; on a balanced
    panel the two coincide. Established as a black box: every ``estat``
    estimate of csdid 1.81 is rebuilt from these weights to 1e-15
    (``tests/reference_parity/test_cs_parity_rc_unbal.py`` and
    ``test_rcs_aggregation_conventions.py``).

    Returns the labels, the weight matrix and, for ``type='group'``, the
    weights of csdid's group average: proportional to the *mean* count of a
    cohort's post-treatment cells, so a cohort observed for fewer periods is
    not down-weighted for it (empty otherwise).
    """
    rel = detail["relative_time"].to_numpy(dtype=float)
    post = rel >= 0
    counts = np.asarray(counts, dtype=float)
    if type == "simple":
        masks, labels = [post], ["overall"]
    elif type == "group":
        labels = sorted(detail.loc[post, "group"].unique())
        masks = [post & (detail["group"].to_numpy() == g) for g in labels]
    elif type == "calendar":
        labels = sorted(detail.loc[post, "time"].unique())
        masks = [post & (detail["time"].to_numpy() == t) for t in labels]
    else:  # dynamic
        keep = (rel >= min_e) & (rel <= max_e)
        labels = sorted({int(e) for e in rel[keep]})
        masks = [keep & (rel == e) for e in labels]
    rows, labs, mean_count = [], [], []
    for lab, m in zip(labels, masks):
        w = np.where(m, counts, 0.0)
        if w.sum() <= 0:
            continue
        rows.append(w / w.sum())
        labs.append(lab if type == "simple" else int(lab))
        mean_count.append(w.sum() / int(m.sum()))
    if not rows:
        return np.array([], dtype=object), np.zeros((0, len(detail))), np.zeros(0)
    rx1 = np.asarray(mean_count, dtype=float) if type == "group" else np.zeros(0)
    return np.array(labs), np.vstack(rows), rx1


def _csdid_count_weight_influence(
    W: np.ndarray,
    att_vec: np.ndarray,
    indicators: np.ndarray,
) -> np.ndarray:
    """Influence of csdid's estimated cell counts on each aggregate.

    Row ``r`` of ``W`` weights cell ``k`` by ``p_k / sum p`` with ``p_k``
    the mean of indicator column ``k`` (a share of the rows on a repeated
    cross-section, of the units on a panel; under sampling weights the
    columns are ``omega_i`` times the indicator, a share of weight mass). Those shares are estimated,
    and csdid carries them: by the delta method each aggregate gains
    ``IF(w) @ ATT`` over its cells, :func:`statspai.did._core.weight_influence`
    with the cell indicators in place of cohort indicators. Within one
    cohort of a balanced panel the term is zero; across cohorts it is R
    ``did``'s cohort-share term.
    """
    out = np.zeros((indicators.shape[0], W.shape[0]))
    for r in range(W.shape[0]):
        keep = np.flatnonzero(W[r] != 0.0)
        if keep.size == 0:
            continue
        ind = indicators[:, keep].astype(float)
        out[:, r] = _weight_influence(ind.mean(axis=0), ind) @ att_vec[keep]
    return out


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

    Note this treats ``W`` as **fixed**.  When the weights are estimated
    cohort shares the caller must add the weight-estimation term first —
    see :func:`_aggregated_influence`.
    """
    psi = inf_matrix @ W.T  # (n_units, K)
    return _se_from_influence(psi, n_units)


def _plain_labels(labels: np.ndarray) -> List[Any]:
    """Cell labels as JSON-friendly Python scalars (``'overall'`` stays a str)."""
    out: List[Any] = []
    for x in labels:
        try:
            f = float(x)
        except (TypeError, ValueError):
            out.append(str(x))
            continue
        out.append(int(f) if f.is_integer() else f)
    return out


def _se_from_influence(
    psi: np.ndarray, n_units: int, cluster_ids: Optional[np.ndarray] = None
) -> np.ndarray:
    """``sqrt(mean(ψ²) / n)`` per column (the analytic SE R ``did`` reports).

    With ``cluster_ids`` the cluster sums replace the rows (Stata
    ``csdid, cluster()``); see :func:`statspai.did._core.influence_se_did`.
    """
    from ._core import influence_se_did

    return np.atleast_1d(
        np.asarray(influence_se_did(psi, n_units, cluster_ids), dtype=float)
    )


def _aggregated_influence(
    W: np.ndarray,
    inf_matrix: np.ndarray,
    att_vec: np.ndarray,
    pg: np.ndarray,
    ind: np.ndarray,
) -> np.ndarray:
    """Per-cell aggregated influence functions, weight estimation included.

    Column ``r`` is ``inf_matrix @ W[r]`` plus the estimated-share term
    ``weight_influence(pg[keep], ind[:, keep]) @ att_vec[keep]`` over the
    cells ``keep`` that row ``r`` gives non-zero weight. The share term is
    exactly zero when those cells belong to a single cohort, so per-cohort
    averages are unaffected. Clean-room implementation; specification in
    ``docs/dev/cleanroom/did_drdid.md``.

    Returns
    -------
    ndarray of shape (n_units, K)
    """
    psi = np.asarray(inf_matrix @ W.T, dtype=float)
    for r in range(W.shape[0]):
        keep = np.flatnonzero(W[r] != 0.0)
        if keep.size == 0:
            continue
        share_if = _weight_influence(pg[keep], ind[:, keep])
        psi[:, r] = psi[:, r] + share_if @ np.asarray(att_vec, dtype=float)[keep]
    return psi


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
    return _bootstrap_from_influence(
        psi, n_units, alpha, n_boot, random_state, cluster_ids=cluster_ids
    )


def _bootstrap_from_influence(
    psi: np.ndarray,
    n_units: int,
    alpha: float,
    n_boot: int,
    random_state: Optional[int],
    cluster_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Multiplier bootstrap on already-aggregated influence functions.

    Separate from :func:`_multiplier_bootstrap` so callers that have
    applied the weight-estimation correction can bootstrap the corrected
    functions instead of re-deriving them from ``W``.
    """
    return _core_multiplier_bootstrap(
        psi,
        n_units,
        alpha,
        n_boot,
        random_state,
        weight_type="rademacher",
        cluster_ids=cluster_ids,
    )
