"""
Borusyak, Jaravel & Spiess (2024) imputation estimator for staggered DID.

The imputation approach estimates unit and time fixed effects using only
untreated observations, imputes counterfactual outcomes for treated
observations, and computes treatment effects as the difference between
observed and imputed outcomes. This avoids the negative-weighting problem
of TWFE regressions under heterogeneous treatment effects.

References
----------
Borusyak, K., Jaravel, X. and Spiess, J. (2024).
"Revisiting Event-Study Designs: Robust and Efficient Estimation."
*Review of Economic Studies*, 91(6), 3253-3285. [@borusyak2024revisiting]
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import lsqr

from ..core._bootstrap import bootstrap_se as _bootstrap_se
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility
from ._bjs_pretrends import EVENT_STUDY_CONVENTION as _EVENT_STUDY_CONVENTION
from ._bjs_pretrends import (
    PRETREND_METHODS,
    bjs_pretrend_path,
    symmetric_pretrend_scale,
)
from ._bjs_variance import bjs_se_for_target as _bjs_se
from ._core import drop_unusable_rows as _drop_unusable_rows
from ._core import normalize_se_method as _normalize_se_method


def _didimp_cluster_bootstrap(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]],
    cluster: str,
    n_boot: int,
    seed: int,
) -> float:
    """Pairs-cluster bootstrap of the overall ATT for ``did_imputation``.

    The analytic influence-function SE under-counts the variance from
    estimating the unit/time fixed effects on the untreated sample. Resampling
    whole clusters and re-running the full imputation estimator gives a valid
    standard error for the headline ATT. Returns the bootstrap SD of the ATT.
    """
    rng = np.random.default_rng(seed)
    clusters = pd.unique(data[cluster])
    n_g = len(clusters)
    rows_by_cluster = {c: data[data[cluster] == c] for c in clusters}
    ctrl = controls if controls else None

    n_boot_int = int(n_boot)
    boot = np.full(n_boot_int, np.nan, dtype=float)
    for b in range(n_boot_int):
        drawn = rng.choice(n_g, size=n_g, replace=True)
        parts = []
        for j, ci in enumerate(drawn):
            sub = rows_by_cluster[clusters[ci]].copy()
            sub[group] = sub[group].astype(str) + f"__b{j}"
            sub["__bcl"] = j
            parts.append(sub)
        bd = pd.concat(parts, ignore_index=True)
        try:
            r = did_imputation(
                bd,
                y=y,
                group=group,
                time=time,
                first_treat=first_treat,
                controls=ctrl,
                horizon=None,
                cluster="__bcl",
                vce="none",
            )
        except Exception:
            continue  # replicate stays NaN; bootstrap_se tracks the failure
        if np.isfinite(r.estimate):
            boot[b] = float(r.estimate)
    return _bootstrap_se(boot, label="did.imputation")


def did_imputation(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]] = None,
    unit_covariates: Optional[List[str]] = None,
    time_covariates: Optional[List[str]] = None,
    fe: Optional[Sequence[str]] = None,
    horizon: Optional[List[int]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    vce: str = "analytic",
    se_method: Optional[str] = None,
    n_boot: int = 199,
    boot_seed: int = 0,
    pretrends: Optional[int] = None,
    pretrend_method: str = "bjs",
    balanced: bool = False,
    min_n: Optional[int] = None,
    hetby: Optional[str] = None,
    project: Optional[List[str]] = None,
    save_weights: bool = False,
    save_residuals: bool = False,
) -> CausalResult:
    """
    Borusyak, Jaravel & Spiess (2024) imputation DID estimator.

    Estimates ATT by imputing counterfactual outcomes for treated
    observations using a TWFE model fit only on untreated data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable name.
    group : str
        Unit identifier column.
    time : str
        Time period column.
    first_treat : str
        Column indicating the period of first treatment.
        Use ``np.inf``, ``np.nan``, or ``0`` for never-treated units.
    controls : list of str, optional
        Continuous time-varying controls, entering the Y(0) model
        additively. Stata ``did_imputation``'s ``controls()``.
    unit_covariates : list of str, optional
        Controls interacted with the **unit** fixed effects, i.e. one
        slope per unit. Stata's ``unitcontrols()``. The canonical use is
        ``unit_covariates=[time_col]``, which makes Y(0) carry
        unit-specific linear trends.

        Identification bites hard here: a unit-specific slope needs at
        least two untreated periods for that unit, so early-treated
        cohorts can lose their imputation entirely.
    time_covariates : list of str, optional
        Controls interacted with the **period** fixed effects, i.e. one
        coefficient per period. Stata's ``timecontrols()``. Typically a
        time-invariant unit characteristic whose effect is allowed to
        move over calendar time.
    fe : sequence of str, optional
        Which fixed effects the Y(0) model carries, replacing the default
        two-way ``unit + time``. Mirrors Stata ``did_imputation``'s
        ``fe()``. Each entry is one fixed effect, written either as a bare
        column or Stata-style with ``#`` for the interacted cell::

            fe=['time']                  # period FE only
            fe=['unit', 'state#year']    # unit FE + state-by-year FE
            fe=[]                        # no fixed effects at all

        Levels are factorized over the whole panel, so a cell seen only
        among treated rows still gets a column and is imputed rather than
        silently absorbing the reference level.
    horizon : list of int, optional
        Relative time periods for event study estimates,
        e.g. ``list(range(-5, 6))``. If ``None``, reports only the
        overall ATT (no event study disaggregation).
    cluster : str, optional
        Variable for cluster-robust standard errors.
        Defaults to ``group`` (unit-level clustering).
    alpha : float, default 0.05
        Significance level for confidence intervals.
    se_method : str, optional
        Shared DiD spelling for ``vce=`` — ``'analytic'``,
        ``'bootstrap'`` (also ``'cluster'``, ``'pairs'``) or ``'auto'``.
        Passing both raises. ``'auto'`` picks the cluster bootstrap when
        the design has at most 30 clusters (Cameron, Gelbach & Miller
        2008) and the analytic BJS variance otherwise: with few clusters
        the cluster-score sum behind any sandwich is itself noisy, which
        is a small-cluster problem rather than a defect in the formula.
    vce : {'analytic', 'bootstrap'}, default 'analytic'
        Standard-error mode for the overall ATT. ``'analytic'`` is the
        exact Borusyak--Jaravel--Spiess variance: the estimator is linear
        in the outcome, so its weights are computed rather than
        approximated, and the result reproduces Stata ``did_imputation``
        and R ``didimputation`` to ~5e-8. Before v1.23.0 this path used an
        approximation that was materially anti-conservative (18-36% too
        small on the harness fixtures); see MIGRATION.md.

        Measured, not asserted. On a homogeneous-effect design with a
        never-treated half, 400-800 replications per cell:

        ===========  ==========  ====================
        clusters     coverage    mean SE / sd(est)
        ===========  ==========  ====================
        30           0.925       0.985
        60           0.938       0.936
        120          0.948       0.978
        240          0.935       1.007
        480          **0.950**   0.983
        ===========  ==========  ====================

        Two things to read off it. The ratio sits at 1 throughout, so the
        variance formula carries no systematic bias — the approximation it
        replaced ran 18-36% low. And coverage reaches nominal by 480
        clusters, so the shortfall at 60 is a small-cluster effect rather
        than a missing term: there the standard error is simply *noisy*
        (dispersion 9.2% against 3.2% at 480), which fattens the
        studentised distribution (sd(t) = 1.04 against 1.007).

        A ``t(G-1)`` critical value barely helps (0.938 to 0.941 at
        G = 60) because the problem is the variability of the standard
        error, not the degrees of freedom. ``vce='bootstrap'`` is the
        remedy in small designs, and ``se_method='auto'`` selects it
        below 30 clusters.
        ``'bootstrap'`` resamples whole clusters and re-runs
        the full imputation estimator. Point estimates are identical either
        way; per-horizon event-study SEs are unaffected.
    n_boot : int, default 199
        Number of cluster-bootstrap replications when ``vce='bootstrap'``.
    boot_seed : int, default 0
        Seed for the cluster bootstrap (deterministic results).
    pretrends : int, optional
        Stata ``did_imputation, pretrends(k)``: estimate the ``k``
        pre-treatment placebo coefficients (horizons ``-k .. -1``, added
        to ``horizon`` if not already requested) and report their joint
        Wald test in ``model_info['pretrend_test']``.  Under the default
        ``pretrend_method='bjs'`` the test uses the full cluster-robust
        covariance of the auxiliary lead regression; under the other
        conventions it assumes the lead estimates are uncorrelated (valid
        but conservative), and :func:`statspai.bjs_pretrend_joint` gives
        the covariance-aware cluster-bootstrap version.
    pretrend_method : {'bjs', 'in-sample', 'symmetric'}, default 'bjs'
        How the **pre-treatment** event-study coefficients are built.  The
        post-treatment coefficients are imputation residuals either way;
        only the leads differ, and the difference is visible in the plot
        rather than in the ATT.

        - ``'bjs'`` — the convention of Stata ``did_imputation,
          pretrends(k)``: an auxiliary dynamic TWFE regression on the
          untreated observations, with all relative times earlier than the
          requested leads pooled into the omitted category.  Reproduces
          Stata's coefficients and standard errors.  Not available with
          ``fe=``, ``unit_covariates=`` or ``time_covariates=``.
        - ``'in-sample'`` — average the imputation residuals at
          pre-treatment relative times, as ``fect`` and ``did2s`` do.
          These are in-sample prediction errors: in a non-staggered design
          they equal the symmetric benchmark times the untreated unit
          share ``N0/N``, so they understate pre-trends, severely when
          most units are treated.  This was StatsPAI's behaviour before
          v1.23.0.
        - ``'symmetric'`` — Roth's (2026) repair, which uses the average
          of the pre-treatment periods as the reference for both halves of
          the path, so the plot matches a dynamic TWFE event study up to a
          common vertical shift and the usual visual heuristics apply.
          Non-staggered balanced designs without covariates only; raises
          otherwise rather than applying an unverified factor.

        The chosen convention and its caveat are recorded in
        ``model_info['pretrend_method']`` and
        ``model_info['event_study_convention']``.
    balanced : bool, default False
        Stata ``did_imputation, hbalance``: keep only eventually-treated
        units observed at *every* requested horizon, so the event-study
        composition is stable across ``k`` (no cohort churn).
        Never-treated units are always kept.  Requires ``horizon`` (or
        ``pretrends``).  Warns with the number of units dropped.
    min_n : int, optional
        Stata ``did_imputation, minn(#)``: drop event-study horizons
        with fewer than ``min_n`` treated observations (they are noisy
        and dominated by a single cohort).  Dropped horizons are listed
        in a warning and excluded from the pre-trend test.
    hetby : str, optional
        Stata ``did_imputation, hetby(varname)``: report heterogeneous
        overall ATTs by the levels of a **time-invariant** unit-level
        variable.  Results land in ``model_info['hetby']`` (one row per
        level: att, se, ci, pvalue, n_obs).  SEs use the same
        influence-function machinery as the overall ATT.
    save_weights : bool, default False
        Stata ``did_imputation, saveweights()``: store the exact
        estimation weights ``w`` such that ``ATT = w'y`` in
        ``model_info['estimation_weights']`` (aligned with the rows of
        ``data``).  Treated rows get ``1/N1``; untreated rows get the
        (negative) imputation weights implied by the FE projection —
        useful for diagnosing which comparisons drive the estimate.
    save_residuals : bool, default False
        Stata ``did_imputation, saveresid()``: store the untreated-fit
        residuals ``y - ŷ⁰`` in ``model_info['residuals']`` (aligned
        with the rows of ``data``; ``NaN`` on treated rows, whose
        ``y - ŷ⁰`` is the treatment effect, not a residual).

    Returns
    -------
    CausalResult
        Result object with ``.summary()``, ``.plot()`` (event study),
        and ``.cite()`` methods. Event study coefficients are stored
        in ``model_info['event_study']``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for unit in range(40):
    ...     first = [4, 7, 0][unit % 3]  # cohorts 4, 7, never-treated (0)
    ...     for year in range(1, 9):
    ...         treated = first != 0 and year >= first
    ...         te = 2.0 * (year - first + 1) if treated else 0.0
    ...         rows.append({'county': unit, 'year': year,
    ...                      'wage': unit * 0.1 + year + te + rng.normal(),
    ...                      'first_treat': first})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.did_imputation(
    ...     data=df, y='wage', group='county', time='year',
    ...     first_treat='first_treat', horizon=list(range(-5, 6)),
    ... )
    >>> bool(result.estimate > 0)
    True
    >>> fig, ax = result.event_study_plot()

    Notes
    -----
    The algorithm proceeds in four steps:

    1. **Classify** observations as treated (t >= first_treat_i) or
       untreated (t < first_treat_i, or never-treated unit).
    2. **Estimate TWFE** on untreated observations only:
       Y_it = alpha_i + lambda_t + X_it'beta + eps_it.
    3. **Impute** counterfactual outcomes for treated observations:
       tau_hat_it = Y_it - (alpha_hat_i + lambda_hat_t + X_it'beta_hat).
    4. **Aggregate** into ATT or event-study ATT(k) and compute
       cluster-robust standard errors with a two-step adjustment.

    References
    ----------
    Borusyak, K., Jaravel, X. and Spiess, J. (2024). Revisiting event-study
    designs: Robust and efficient estimation. *Review of Economic Studies*.
    [@borusyak2024revisiting]
    """
    # ── Input validation ─────────────────────────────────────────── #
    df = data.copy()
    control_names = list(controls or [])
    for col in [y, group, time, first_treat]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    for col in control_names:
        if col not in df.columns:
            raise ValueError(f"Control column '{col}' not found in data.")

    if se_method is not None:
        if vce != "analytic":
            raise ValueError(
                "did_imputation: pass either se_method= or vce=, not both — "
                "they set the same thing and disagreeing values would "
                "silently resolve one way."
            )
        vce = _normalize_se_method(
            se_method,
            supported=("analytic", "bootstrap"),
            function="did_imputation",
            n_clusters=int(data[cluster or group].nunique()),
        )

    unit_cov_names = list(unit_covariates or [])
    time_cov_names = list(time_covariates or [])
    for label, names in (
        ("unit_covariates", unit_cov_names),
        ("time_covariates", time_cov_names),
    ):
        for col in names:
            if col not in df.columns:
                raise ValueError(f"{label} column '{col}' not found in data.")
        overlap = sorted(set(names) & set(control_names))
        if overlap:
            raise ValueError(
                f"{overlap} appear in both `controls` and `{label}`. A "
                "variable enters Y(0) either additively (controls) or "
                "interacted with the fixed effect ({label}), not both — "
                "including it twice makes the design collinear."
            )
    if fe is not None and isinstance(fe, str):
        raise ValueError(
            f"fe must be a sequence of specs, not a bare string {fe!r}. "
            f"Pass fe=['{fe}'] for one fixed effect, or e.g. "
            "fe=['unit', 'state#year']."
        )

    # Drop rows the estimator cannot use before the cohort/horizon logic runs,
    # so a wiped outcome surfaces as an error instead of a bare NaN ATT. Keep
    # the caller's index: save_weights / save_residuals key their output on it.
    df = _drop_unusable_rows(
        df,
        columns=[y, time, group, *control_names],
        function="did_imputation",
        reset_index=False,
    )

    if vce not in ("analytic", "bootstrap", "none"):
        raise ValueError(f"vce must be 'analytic', 'bootstrap', or 'none'; got {vce!r}")
    if cluster is None:
        cluster = group

    if pretrend_method not in PRETREND_METHODS:
        raise ValueError(
            f"pretrend_method must be one of {list(PRETREND_METHODS)}; got "
            f"{pretrend_method!r}. 'bjs' reproduces Stata did_imputation, "
            "'in-sample' is the fect/did2s residual-average convention, and "
            "'symmetric' is Roth's (2026) repair for non-staggered designs."
        )

    if pretrends is not None:
        if isinstance(pretrends, bool) or int(pretrends) < 1:
            raise ValueError(f"pretrends must be a positive integer, got {pretrends!r}")
        pretrends = int(pretrends)
        # Placebo horizons -k..-1 join whatever the caller asked for.
        horizon = sorted(set(horizon or []) | set(range(-pretrends, 0)))
    if min_n is not None:
        if isinstance(min_n, bool) or int(min_n) < 1:
            raise ValueError(f"min_n must be a positive integer, got {min_n!r}")
        min_n = int(min_n)
        if horizon is None:
            raise ValueError(
                "min_n filters event-study horizons — pass horizon= (or "
                "pretrends=) so there is an event study to filter."
            )
    if balanced and horizon is None:
        raise ValueError(
            "balanced=True balances units across the requested horizons — "
            "pass horizon= (or pretrends=) to define the horizon window."
        )
    if hetby is not None:
        if hetby not in df.columns:
            raise ValueError(f"hetby column '{hetby}' not found in data.")
        het_nuniq = df.groupby(group)[hetby].nunique(dropna=False)
        if (het_nuniq > 1).any():
            bad = het_nuniq[het_nuniq > 1].index.tolist()[:5]
            raise ValueError(
                f"hetby column '{hetby}' is time-varying within unit "
                f"(e.g. units {bad}); heterogeneity splits require a "
                "time-invariant unit-level variable."
            )

    # ── Step 1: Identify treated / untreated observations ──────── #
    # Normalize first_treat: inf, NaN, 0 → never treated (use np.inf)
    ft = df[first_treat].copy().astype(float)
    ft = ft.replace(0, np.inf)
    ft = ft.fillna(np.inf)
    df["_ft"] = ft

    df["_treated_obs"] = df[time].astype(float) >= df["_ft"]
    df["_never_treated"] = np.isinf(df["_ft"])
    df["_untreated_obs"] = ~df["_treated_obs"]  # includes never-treated

    # ── Optional hbalance filter (Stata: hbalance) ──────────────── #
    # Keep only eventually-treated units observed at every requested
    # horizon so the event-study composition is stable across k.
    n_units_balanced_out = 0
    if balanced:
        # Stata hbalance balances on the post-treatment leads only —
        # negative horizons are placebos and must not knock out cohorts
        # that simply lack long pre-histories.
        needed = {int(k) for k in horizon if k >= 0}
        if not needed:
            raise ValueError(
                "balanced=True needs at least one non-negative horizon to "
                "balance on — horizons < 0 are placebos and do not define "
                "the post-treatment window."
            )
        ev_mask = ~np.isinf(df["_ft"].values)
        rel_all = np.round(df[time].astype(float).values - df["_ft"].values)
        rel_frame = pd.DataFrame(
            {"_u": df.loc[ev_mask, group].values, "_rel": rel_all[ev_mask]}
        )
        present = rel_frame.groupby("_u")["_rel"].agg(set)
        keep_units = {u for u, s in present.items() if needed <= s}
        drop_units = [u for u in present.index if u not in keep_units]
        if drop_units:
            n_units_balanced_out = len(drop_units)
            preview = ", ".join(map(str, drop_units[:5]))
            warnings.warn(
                f"did_imputation: balanced=True dropped "
                f"{n_units_balanced_out} eventually-treated unit(s) not "
                f"observed at every requested horizon ({preview}"
                + (" ..." if n_units_balanced_out > 5 else "")
                + "). The event-study composition is now stable across "
                "horizons; the overall ATT refers to the balanced sample.",
                UserWarning,
                stacklevel=2,
            )
            keep_mask = (~ev_mask) | df[group].isin(keep_units).values
            df = df[keep_mask].copy()

    n_treated = df["_treated_obs"].sum()
    n_untreated = df["_untreated_obs"].sum()

    if n_treated == 0:
        raise ValueError("No treated observations found. Check 'first_treat' column.")
    if n_untreated == 0:
        raise ValueError("No untreated observations found. Need control observations.")

    # ── Step 2: Estimate TWFE on untreated observations ─────────── #
    # Encode unit and time as integer indices for FE
    unit_ids = df[group].unique()
    time_ids = sorted(df[time].unique())
    unit_map = {u: idx for idx, u in enumerate(unit_ids)}
    time_map = {t_val: idx for idx, t_val in enumerate(time_ids)}
    n_units = len(unit_ids)
    n_times = len(time_ids)

    df["_uid"] = df[group].map(unit_map)
    df["_tid"] = df[time].map(time_map)

    untreated = df[df["_untreated_obs"]].copy()

    has_controls = len(control_names) > 0
    uid_u = untreated["_uid"].values
    tid_u = untreated["_tid"].values

    unit_adj_count = np.bincount(uid_u, minlength=n_units).astype(float)
    time_resid_count = np.bincount(tid_u, minlength=n_times).astype(float)

    treated_mask = df["_treated_obs"].values
    treated_uids = np.unique(df.loc[treated_mask, "_uid"].values)
    treated_tids = np.unique(df.loc[treated_mask, "_tid"].values)
    missing_units = [
        unit_ids[int(ui)] for ui in treated_uids if unit_adj_count[int(ui)] <= 0
    ]
    missing_times = [
        time_ids[int(ti)] for ti in treated_tids if time_resid_count[int(ti)] <= 0
    ]
    if missing_units:
        preview = ", ".join(map(str, missing_units[:5]))
        raise ValueError(
            "BJS imputation needs at least one untreated observation for "
            "every treated unit to estimate its unit fixed effect. "
            f"Missing untreated history for unit(s): {preview}"
            + (" ..." if len(missing_units) > 5 else "")
        )
    if missing_times:
        preview = ", ".join(map(str, missing_times[:5]))
        raise ValueError(
            "BJS imputation needs at least one untreated observation in "
            "every treated time period to estimate its time fixed effect. "
            f"Missing untreated comparison period(s): {preview}"
            + (" ..." if len(missing_times) > 5 else "")
        )

    # Interacted covariates raise the bar: a unit carrying its own slope on
    # k variables needs k+1 untreated observations, not 1, before its Y(0)
    # is identified. lsqr would otherwise return a minimum-norm solution
    # and hand back a confident-looking number built on an unidentified
    # fit. Stata's did_imputation refuses outright here (rc 481, "collinear
    # in the D==0 subsample but not in the full sample"); so do we (§7).
    for label, names, counts, ids, treated_idx in (
        ("unit_covariates", unit_cov_names, unit_adj_count, unit_ids, treated_uids),
        ("time_covariates", time_cov_names, time_resid_count, time_ids, treated_tids),
    ):
        if not names:
            continue
        need = len(names) + 1
        short = [
            (ids[int(k)], int(counts[int(k)]))
            for k in treated_idx
            if counts[int(k)] < need
        ]
        if short:
            preview = ", ".join(f"{lab} ({n} untreated)" for lab, n in short[:5])
            axis = "unit" if label == "unit_covariates" else "period"
            raise ValueError(
                f"{label}={names} gives every {axis} its own slope on "
                f"{len(names)} variable(s), so each treated {axis} needs at "
                f"least {need} untreated observations to identify the "
                f"intercept plus slope(s). {len(short)} fall short: {preview}"
                + (" ..." if len(short) > 5 else "")
                + f". Drop {label}, shorten the covariate list, or restrict "
                f"to {axis}s with enough untreated history."
            )

    y0_hat, beta, X_u_design, X_all = _fit_untreated_twfe_sparse(
        df=df,
        untreated=untreated,
        y=y,
        controls=control_names if has_controls else None,
        uid_col="_uid",
        tid_col="_tid",
        n_units=n_units,
        n_times=n_times,
        unit_covariates=unit_cov_names,
        time_covariates=time_cov_names,
        fe=fe,
    )

    # These arguments are retained for the existing SE helper API.  The
    # helper only needs the untreated counts and residuals; fitted values
    # now come from the exact sparse TWFE solve above.

    # ── Step 3: Impute counterfactual for treated observations ── #
    y_all = df[y].values.astype(float)

    # Individual treatment effects for treated obs
    tau_hat = y_all - y0_hat  # defined for all obs; meaningful for treated

    df["_tau_hat"] = tau_hat
    df["_y0_hat"] = y0_hat

    # ── Relative time ──────────────────────────────────────────── #
    df["_rel_time"] = df[time].astype(float) - df["_ft"]
    # For never-treated, _rel_time will be -inf; that's fine

    # ── Step 4: Aggregate treatment effects ────────────────────── #
    treated_df = df[treated_mask].copy()

    # Overall ATT
    att = float(treated_df["_tau_hat"].mean())

    # ── Step 5: Standard errors ────────────────────────────────── #
    # Cluster-robust SEs with influence-function approach
    # Compute residuals on untreated for the FE model
    resid_u = np.zeros(len(df))
    resid_u[~treated_mask] = y_all[~treated_mask] - y0_hat[~treated_mask]

    # Exact BJS variance. The estimator is linear in y, so the weights
    # v with tau_hat = v'y are computable rather than approximable, and
    # both reference implementations use them. See did/_bjs_variance.py
    # for what the previous approximation got wrong.
    _cluster_vals = df[cluster].to_numpy()
    _cohort_vals = df["_ft"].to_numpy()
    _rel_vals = np.round(
        df[time].to_numpy(dtype=float) - df["_ft"].to_numpy(dtype=float)
    )
    _n_treated_obs = int(treated_mask.sum())
    _w_overall = np.zeros(len(df), dtype=float)
    if _n_treated_obs > 0:
        _w_overall[treated_mask] = 1.0 / _n_treated_obs
    se_att = _bjs_se(
        design_all=X_all,
        design_untreated=X_u_design,
        treated_mask=treated_mask,
        target_weights=_w_overall,
        adjusted=tau_hat,
        cluster=_cluster_vals,
        cohort=_cohort_vals,
        relative_time=_rel_vals,
    )

    # Standard-error mode for the overall ATT. The analytic route is now
    # the exact BJS variance and reproduces Stata did_imputation and R
    # didimputation to ~5e-8, so the anti-conservatism warning that used
    # to fire here has been removed along with the approximation that
    # justified it. 'bootstrap' resamples whole clusters and re-runs the
    # estimator; 'none' is internal (used by the bootstrap to avoid
    # recursion).
    if vce == "bootstrap":
        # Resample the estimation sample (post-hbalance if balanced=True),
        # not the raw input — otherwise replicates would refit on units
        # the point estimate excluded.
        boot_data = df.drop(columns=[c for c in df.columns if c.startswith("_")])
        se_boot = _didimp_cluster_bootstrap(
            boot_data, y, group, time, first_treat, controls, cluster, n_boot, boot_seed
        )
        if np.isfinite(se_boot):
            se_att = se_boot

    z_crit = stats.norm.ppf(1 - alpha / 2)
    pvalue_att = (
        float(2 * (1 - stats.norm.cdf(abs(att / se_att)))) if se_att > 0 else 1.0
    )
    ci_att = (att - z_crit * se_att, att + z_crit * se_att)

    # ── Event study (if horizon requested) ─────────────────────── #
    event_study_df = None
    pretrend_test = None

    if horizon is not None:
        es_rows = []

        # For event study, we need all obs of eventually-treated units
        # (including pre-treatment periods for placebo/pre-trend checks)
        eventually_treated = ~np.isinf(df["_ft"].values)
        rel_time_rounded = np.round(df["_rel_time"].values)

        # Which relative times does the residual-average loop own?  Under
        # pretrend_method='bjs' the leads come from a separate auxiliary
        # regression instead, so they are skipped here rather than computed
        # twice; 'in-sample' and 'symmetric' both start from the residual
        # averages and 'symmetric' rescales them afterwards.
        residual_horizons = (
            [k for k in sorted(horizon) if k >= 0]
            if pretrend_method == "bjs"
            else sorted(horizon)
        )

        for k in residual_horizons:
            # Observations of eventually-treated units at relative time k
            mask_k = eventually_treated & (rel_time_rounded == k)
            n_k = int(mask_k.sum())
            if n_k == 0:
                continue

            att_k = float(tau_hat[mask_k].mean())

            # Cluster SE for this horizon
            _w_k = np.zeros(len(df), dtype=float)
            _w_k[mask_k] = 1.0 / n_k
            se_k = _bjs_se(
                design_all=X_all,
                design_untreated=X_u_design,
                treated_mask=treated_mask,
                target_weights=_w_k,
                adjusted=tau_hat,
                cluster=_cluster_vals,
                cohort=_cohort_vals,
                relative_time=_rel_vals,
            )

            pval_k = (
                float(2 * (1 - stats.norm.cdf(abs(att_k / se_k)))) if se_k > 0 else 1.0
            )

            es_rows.append(
                {
                    "relative_time": k,
                    "att": att_k,
                    "se": se_k,
                    "ci_lower": att_k - z_crit * se_k,
                    "ci_upper": att_k + z_crit * se_k,
                    "pvalue": pval_k,
                    "n_obs": n_k,
                }
            )

        event_study_df = pd.DataFrame(es_rows)

        # Optional minn() filter: drop horizons with too few treated obs.
        if min_n is not None and len(event_study_df) > 0:
            thin = event_study_df["n_obs"] < min_n
            if thin.any():
                dropped = event_study_df.loc[thin, "relative_time"].tolist()
                warnings.warn(
                    f"did_imputation: min_n={min_n} dropped horizon(s) "
                    f"{dropped} with fewer treated observations; they are "
                    "excluded from the event study and the pre-trend test.",
                    UserWarning,
                    stacklevel=2,
                )
                event_study_df = event_study_df.loc[~thin].reset_index(drop=True)

        # ── Pre-treatment reference convention ─────────────────── #
        # Post-treatment coefficients are imputation residuals under every
        # convention; only the leads differ.  See did/_bjs_pretrends.py
        # for what each option constructs and why the choice is visible
        # in the plotted event study (Roth 2026).
        requested_leads = [int(k) for k in sorted(horizon) if k < 0]
        bjs_joint: Optional[Dict[str, float]] = None

        if requested_leads and pretrend_method == "bjs":
            if fe is not None or unit_cov_names or time_cov_names:
                raise MethodIncompatibility(
                    "pretrend_method='bjs' runs the lead regression on the "
                    "same Y(0) design the imputation step uses, and the "
                    "degrees-of-freedom convention it inherits from Stata "
                    "did_imputation is only pinned for the default "
                    "unit+time model. It is not pinned with fe=, "
                    "unit_covariates= or time_covariates=.",
                    remedy=(
                        "Pass pretrend_method='in-sample' to keep the "
                        "residual-average leads, or drop the non-default "
                        "fixed-effect structure for the pre-trend run."
                    ),
                )
            rel_untreated = np.round(
                untreated[time].to_numpy(dtype=float)
                - untreated["_ft"].to_numpy(dtype=float)
            )
            pre_frame, bjs_joint = bjs_pretrend_path(
                design_untreated=X_u_design,
                y_untreated=untreated[y].to_numpy(dtype=float),
                rel_time_untreated=rel_untreated,
                cluster_untreated=untreated[cluster].to_numpy(),
                n_unit_columns=int(len(np.unique(uid_u)) - 1),
                leads=requested_leads,
                alpha=alpha,
            )
            event_study_df = (
                pd.concat([pre_frame, event_study_df], ignore_index=True)
                .sort_values("relative_time")
                .reset_index(drop=True)
            )
        elif requested_leads and pretrend_method == "symmetric":
            ft_by_unit = df.groupby("_uid")["_ft"].first()
            treated_cohorts = sorted(
                {float(v) for v in ft_by_unit.to_numpy() if np.isfinite(v)}
            )
            obs_per_unit = df.groupby("_uid").size().to_numpy()
            factor = symmetric_pretrend_scale(
                n_units_total=int(n_units),
                n_units_untreated=int((~np.isfinite(ft_by_unit.to_numpy())).sum()),
                n_cohorts_treated=len(treated_cohorts),
                balanced=bool(np.all(obs_per_unit == obs_per_unit[0])),
                has_covariates=bool(
                    control_names or unit_cov_names or time_cov_names or fe is not None
                ),
            )
            pre_mask = event_study_df["relative_time"] < 0
            for col in ("att", "se", "ci_lower", "ci_upper"):
                event_study_df.loc[pre_mask, col] = (
                    event_study_df.loc[pre_mask, col] * factor
                )
            # The z-statistic is scale-invariant, so p-values are unchanged.

        # Pre-trend joint test (Wald chi-squared, independence
        # approximation — conservative; see sp.bjs_pretrend_joint for the
        # covariance-aware cluster-bootstrap version). With pretrends=k
        # the test uses exactly the k requested placebo horizons.
        # Under pretrend_method='bjs' the auxiliary regression supplies the
        # full cluster-robust covariance, so that test is used instead.
        pre_rows = event_study_df[
            (event_study_df["relative_time"] < 0) & (event_study_df["se"] > 0)
        ]
        if pretrends is not None:
            pre_rows = pre_rows[pre_rows["relative_time"] >= -pretrends]
        if bjs_joint and pretrends is None:
            pretrend_test = dict(bjs_joint)
            pretrend_test["periods"] = requested_leads
        elif bjs_joint and pretrends is not None and len(requested_leads) == pretrends:
            pretrend_test = dict(bjs_joint)
            pretrend_test["periods"] = requested_leads
        elif len(pre_rows) > 0:
            pre_atts = pre_rows["att"].to_numpy(dtype=float)
            pre_ses = pre_rows["se"].to_numpy(dtype=float)
            chi2_stat = float(np.sum((pre_atts / pre_ses) ** 2))
            df_chi2 = len(pre_rows)
            chi2_pval = float(1 - stats.chi2.cdf(chi2_stat, df_chi2))
            pretrend_test = {
                "statistic": chi2_stat,
                "df": df_chi2,
                "pvalue": chi2_pval,
                "periods": pre_rows["relative_time"].astype(int).tolist(),
                "method": "wald-independent (conservative); "
                "see sp.bjs_pretrend_joint",
            }

    # ── Heterogeneity by group (Stata: hetby) ──────────────────── #
    project_df = None
    if project is not None:
        project_names = list(project)
        for col in project_names:
            if col not in df.columns:
                raise ValueError(f"project column '{col}' not found in data.")
        if hetby is not None:
            raise ValueError(
                "project= and hetby= cannot be combined: they are two "
                "different heterogeneity summaries of the same effects "
                "(a projection onto covariates versus a split into cells). "
                "Stata's did_imputation rejects the combination too. Run "
                "the estimator twice if you want both."
            )
        project_df = _project_treatment_effects(
            df=df,
            tau_hat=tau_hat,
            treated_mask=treated_mask,
            resid_untreated=resid_u,
            project_vars=project_names,
            cluster_col=cluster,
            time_col=time,
            design_all=X_all,
            design_untreated=X_u_design,
            alpha=alpha,
        )

    hetby_df = None
    if hetby is not None:
        het_rows = []
        het_vals = df.loc[treated_mask, hetby]
        for level in pd.unique(het_vals):
            if pd.isna(level):
                level_mask = treated_mask & df[hetby].isna().values
            else:
                level_mask = treated_mask & (df[hetby] == level).values
            n_level = int(level_mask.sum())
            if n_level == 0:
                continue
            att_h = float(tau_hat[level_mask].mean())
            _w_h = np.zeros(len(df), dtype=float)
            _w_h[level_mask] = 1.0 / n_level
            se_h = _bjs_se(
                design_all=X_all,
                design_untreated=X_u_design,
                treated_mask=treated_mask,
                target_weights=_w_h,
                adjusted=tau_hat,
                cluster=_cluster_vals,
                cohort=_cohort_vals,
                relative_time=_rel_vals,
            )
            pval_h = (
                float(2 * (1 - stats.norm.cdf(abs(att_h / se_h)))) if se_h > 0 else 1.0
            )
            het_rows.append(
                {
                    hetby: level,
                    "att": att_h,
                    "se": se_h,
                    "ci_lower": att_h - z_crit * se_h,
                    "ci_upper": att_h + z_crit * se_h,
                    "pvalue": pval_h,
                    "n_obs": n_level,
                }
            )
        hetby_df = pd.DataFrame(het_rows)

    # ── Estimation weights / residual export ───────────────────── #
    # Weights: the BJS ATT is linear in y — ATT = w'y with w = 1/N1 on
    # treated rows and minus the FE-projection weights on untreated rows
    # (Stata: saveweights()).  Solving (X_u'X_u) z = X̄_treated recovers
    # them exactly; `w @ y == ATT` is verified in the test suite.
    if save_weights:
        n1 = float(treated_mask.sum())
        x_bar_treated = np.asarray(X_all[treated_mask].sum(axis=0)).ravel() / n1
        normal_mat = (X_u_design.T @ X_u_design).tocsr()
        z_sol = lsqr(
            normal_mat,
            x_bar_treated,
            atol=1e-12,
            btol=1e-12,
            iter_lim=max(1000, 4 * normal_mat.shape[1]),
        )[0]
        weights_vec = np.zeros(len(df))
        weights_vec[treated_mask] = 1.0 / n1
        untreated_rows = ~treated_mask
        weights_vec[untreated_rows] = -np.asarray(X_all[untreated_rows] @ z_sol).ravel()

    if save_residuals:
        residuals_vec = np.where(~treated_mask, y_all - y0_hat, np.nan)

    # ── Build model_info ───────────────────────────────────────── #
    model_info: Dict[str, Any] = {
        "estimator": "BJS Imputation",
        "n_treated_obs": int(n_treated),
        "n_control_obs": int(n_untreated),
        "n_units": int(n_units),
        "n_time_periods": int(n_times),
        "n_never_treated": int(df["_never_treated"].sum() // max(n_times, 1)),
        "cluster_var": cluster,
        "vce": vce,
    }

    if has_controls:
        model_info["controls"] = control_names
        model_info["beta_controls"] = dict(zip(control_names, beta.tolist()))

    if event_study_df is not None and len(event_study_df) > 0:
        model_info["event_study"] = event_study_df
        # The reference convention is part of what the event study *is*,
        # not a tuning note: two packages can agree on every post-treatment
        # coefficient and still plot different pre-trends (Roth 2026).
        model_info["pretrend_method"] = pretrend_method
        model_info["event_study_convention"] = _EVENT_STUDY_CONVENTION[pretrend_method]

    if pretrend_test is not None:
        model_info["pretrend_test"] = pretrend_test

    if pretrends is not None:
        model_info["pretrends"] = pretrends
    if balanced:
        model_info["balanced"] = True
        model_info["n_units_dropped_balance"] = n_units_balanced_out
    if min_n is not None:
        model_info["min_n"] = min_n
    if hetby_df is not None:
        model_info["hetby"] = hetby_df
        model_info["hetby_var"] = hetby
    if project_df is not None:
        model_info["project"] = project_df
        model_info["project_vars"] = list(project)
    if save_weights:
        # Indexed by the (possibly balanced-filtered) rows of `data`, so
        # the weights map back to the caller's frame even after filtering.
        model_info["estimation_weights"] = pd.Series(
            weights_vec, index=df.index, name="weight"
        )
    if save_residuals:
        model_info["residuals"] = pd.Series(
            residuals_vec, index=df.index, name="residual"
        )

    # ── Return CausalResult ────────────────────────────────────── #
    _result = CausalResult(
        method="Borusyak, Jaravel & Spiess (2024) Imputation Estimator",
        estimand="ATT",
        estimate=att,
        se=se_att,
        pvalue=pvalue_att,
        ci=ci_att,
        alpha=alpha,
        n_obs=len(data),
        detail=event_study_df,
        model_info=model_info,
        _citation_key="did_imputation",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.did_imputation",
            params={
                "y": y,
                "group": group,
                "time": time,
                "first_treat": first_treat,
                "controls": controls,
                "horizon": horizon,
                "cluster": cluster,
                "alpha": alpha,
                "pretrends": pretrends,
                "balanced": balanced,
                "min_n": min_n,
                "hetby": hetby,
                "save_weights": save_weights,
                "save_residuals": save_residuals,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ══════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════


def _ols_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients via least squares. Returns empty array if no regressors."""
    if X.shape[1] == 0:
        return np.array([])
    try:
        return np.asarray(np.linalg.lstsq(X, y, rcond=None)[0], dtype=float)
    except np.linalg.LinAlgError:
        return np.zeros(X.shape[1])


def _build_fe_blocks(
    df: pd.DataFrame,
    untreated: pd.DataFrame,
    fe: Sequence[str],
) -> Tuple[List[Tuple[str, np.ndarray, int]], List[np.ndarray]]:
    """Parse ``fe=`` into level codes, mirroring did_imputation's ``fe()``.

    Each entry names one fixed effect and is either a bare column
    (``'state'``) or an interacted cell written Stata-style with ``#``
    (``'state#year'`` for state-by-year effects). ``fe=[]`` means no fixed
    effects at all — the intercept-only Y(0) model, which ``fe(none)``
    requests in Stata.

    Levels are factorized over the **full** panel, not the untreated
    subsample, so that a level appearing only among treated rows still
    receives a column and its Y(0) is imputed rather than silently taking
    the reference level's value. Levels never seen untreated end up
    unidentified; ``lsqr`` returns the minimum-norm solution there, and
    the caller's untreated-coverage check is what rejects the genuinely
    unimputable cases.

    Returns ``(blocks, codes_over_full_panel)`` where each block is
    ``(name, codes_on_untreated, n_levels)``.
    """
    blocks: List[Tuple[str, np.ndarray, int]] = []
    codes_all: List[np.ndarray] = []
    for spec in fe:
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError(
                f"fe entries must be non-empty strings, got {spec!r}. Use "
                "'state' for a single factor or 'state#year' for the "
                "interacted cell."
            )
        parts = [p.strip() for p in spec.split("#") if p.strip()]
        missing = [p for p in parts if p not in df.columns]
        if missing:
            raise ValueError(
                f"fe='{spec}' names column(s) {missing} that are not in the "
                f"data. Available: {sorted(df.columns)[:12]}..."
            )
        # Factorize the tuple of levels over the full panel so codes are
        # comparable between the untreated fit and the all-rows prediction.
        key_all = pd.MultiIndex.from_arrays([df[p].values for p in parts])
        codes_full, uniques = pd.factorize(key_all, sort=True)
        n_lev = len(uniques)
        key_u = pd.MultiIndex.from_arrays([untreated[p].values for p in parts])
        codes_untr = pd.Index(uniques).get_indexer(key_u)
        if (codes_untr < 0).any():
            raise ValueError(
                f"fe='{spec}': some untreated rows carry a level absent from "
                "the full-panel factorization — this should be impossible "
                "and indicates the untreated frame is not a subset of `data`."
            )
        blocks.append((spec, codes_untr.astype(int), n_lev))
        codes_all.append(codes_full.astype(int))
    return blocks, codes_all


def _fit_untreated_twfe_sparse(
    df: pd.DataFrame,
    untreated: pd.DataFrame,
    y: str,
    controls: Optional[List[str]],
    uid_col: str,
    tid_col: str,
    n_units: int,
    n_times: int,
    unit_covariates: Optional[List[str]] = None,
    time_covariates: Optional[List[str]] = None,
    fe: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, sparse.csr_matrix, sparse.csr_matrix]:
    """Fit untreated-only TWFE by sparse least squares and predict all rows.

    The untreated sample in staggered DID is usually unbalanced: early
    cohorts contribute fewer untreated periods than late or never-treated
    cohorts.  A one-pass "unit mean + time mean - grand mean" transform is
    exact only on balanced panels.  BJS needs the actual least-squares
    projection on unit and time fixed effects, so we solve the dummy
    regression directly with a sparse design matrix.
    """
    controls = list(controls or [])
    unit_covariates = list(unit_covariates or [])
    time_covariates = list(time_covariates or [])
    y_u = untreated[y].values.astype(float)

    # ---- Fixed-effect blocks -------------------------------------------
    # Default is the two-way unit + time model. ``fe`` replaces it wholesale,
    # mirroring did_imputation's fe() option: each entry is one FE, written
    # either as a single variable or as ``a#b`` for the interacted cell.
    if fe is None:
        blocks = [
            ("unit", untreated[uid_col].values.astype(int), n_units),
            ("time", untreated[tid_col].values.astype(int), n_times),
        ]
        block_codes_all = [
            df[uid_col].values.astype(int),
            df[tid_col].values.astype(int),
        ]
    else:
        blocks, block_codes_all = _build_fe_blocks(df, untreated, fe)

    if not blocks:
        # fe=[] is the legitimate "no fixed effects at all" request; the
        # intercept alone still identifies a level.
        pass
    else:
        for name, codes_u, n_lev in blocks:
            if not (np.bincount(codes_u, minlength=n_lev) > 0).any():
                raise ValueError(
                    f"No untreated observations available to identify the "
                    f"'{name}' fixed effect in the BJS Y(0) model."
                )

    # Column layout: [intercept | FE blocks (one level dropped each) |
    #                 unit-interacted covs | time-interacted covs | controls]
    col_maps: List[np.ndarray] = []
    next_col = 1  # intercept occupies column 0
    for name, codes_u, n_lev in blocks:
        seen = np.bincount(codes_u, minlength=n_lev) > 0
        ref = int(np.flatnonzero(seen)[0])
        cmap = np.full(n_lev, -1, dtype=int)
        for lev in range(n_lev):
            if lev != ref and seen[lev]:
                cmap[lev] = next_col
                next_col += 1
        col_maps.append(cmap)

    # Interaction blocks. These are FE × continuous, so they are NOT
    # collinear with the parent FE and no level is dropped: with unit
    # dummies plus unit#x we get a separate slope on x for every unit,
    # which is exactly did_imputation's "unit-specific trends" reading of
    # unitcontrols(t).
    inter_specs: List[Tuple[str, str, int, int]] = []  # (kind, var, base_col, n_lev)
    for var in unit_covariates:
        inter_specs.append(("unit", var, next_col, n_units))
        next_col += n_units
    for var in time_covariates:
        inter_specs.append(("time", var, next_col, n_times))
        next_col += n_times

    n_fe_cols = next_col

    def _design(frame: pd.DataFrame, codes_list: List[np.ndarray]) -> sparse.csr_matrix:
        n = len(frame)
        rows_parts: List[np.ndarray] = [np.arange(n, dtype=int)]
        cols_parts: List[np.ndarray] = [np.zeros(n, dtype=int)]
        data_parts: List[np.ndarray] = [np.ones(n, dtype=float)]

        for cmap, codes in zip(col_maps, codes_list):
            cols = cmap[codes]
            mask = cols >= 0
            if mask.any():
                rows_parts.append(np.flatnonzero(mask))
                cols_parts.append(cols[mask])
                data_parts.append(np.ones(int(mask.sum()), dtype=float))

        uid = frame[uid_col].values.astype(int)
        tid = frame[tid_col].values.astype(int)
        for kind, var, base_col, _n_lev in inter_specs:
            codes = uid if kind == "unit" else tid
            vals = frame[var].values.astype(float)
            nz = vals != 0.0
            if nz.any():
                rows_parts.append(np.flatnonzero(nz))
                cols_parts.append(base_col + codes[nz])
                data_parts.append(vals[nz])

        fixed = sparse.coo_matrix(
            (
                np.concatenate(data_parts),
                (np.concatenate(rows_parts), np.concatenate(cols_parts)),
            ),
            shape=(n, n_fe_cols),
        ).tocsr()

        if not controls:
            return fixed

        x = sparse.csr_matrix(frame[controls].values.astype(float))
        return sparse.hstack([fixed, x], format="csr")

    codes_u = [b[1] for b in blocks]
    X_u = _design(untreated, codes_u)
    n_cols = X_u.shape[1]

    # Column equilibration before the iterative solve. The FE dummies are
    # 0/1 while an interacted covariate can be O(1e3) (unit_covariates=
    # ['year'] on calendar years is the standard case), and lsqr on that
    # mix stalls far from the least-squares solution: on mpdta it landed
    # 1.6e-4 from Stata, three orders worse than every other variant.
    #
    # Rescaling columns is exact — solving (X D) z = y and returning D z
    # recovers the same coefficients in exact arithmetic for any invertible
    # diagonal D — so unlike centring the covariate it cannot change the
    # column span. That matters because centring is only span-preserving
    # when the parent fixed effect is also in the model, which fe=[] breaks.
    col_norms = np.sqrt(np.asarray(X_u.multiply(X_u).sum(axis=0)).ravel())
    col_norms[col_norms <= 0] = 1.0  # unused levels: leave them alone
    scale = sparse.diags(1.0 / col_norms)
    fit = lsqr(
        X_u @ scale,
        y_u,
        atol=1e-12,
        btol=1e-12,
        iter_lim=max(2000, 8 * n_cols),
    )
    coef = fit[0] / col_norms
    X_all = _design(df, block_codes_all)
    y0_hat = np.asarray(X_all @ coef, dtype=float)
    beta = coef[-len(controls) :] if controls else np.array([])
    # The design matrices ride along for the saveweights() path — the
    # exact estimation weights need (X_u'X_u)⁻¹ X̄_treated.
    return y0_hat, np.asarray(beta, dtype=float), X_u, X_all


def _project_treatment_effects(
    df: pd.DataFrame,
    tau_hat: np.ndarray,
    treated_mask: np.ndarray,
    resid_untreated: np.ndarray,
    project_vars: List[str],
    cluster_col: str,
    time_col: str,
    design_all: "sparse.csr_matrix",
    design_untreated: "sparse.csr_matrix",
    alpha: float,
) -> pd.DataFrame:
    """Regress the imputed treatment effects on covariates.

    Stata ``did_imputation, project(varlist)``: instead of averaging the
    per-observation effects τ̂_it into one ATT, project them onto
    ``[1, Z_it]`` and report the constant and slopes. This is the
    continuous counterpart of ``hetby``, which splits into cells.

    Each coefficient is linear in the outcome, exactly as the ATT is:
    θ̂_j = Σ_i a_ji τ̂_i over treated observations, where ``a_ji`` is the
    row of (Z'Z)⁻¹Z'. Inference therefore reuses the estimator's exact
    BJS variance (:mod:`statspai.did._bjs_variance`) with ``a_j`` as the
    treated-row weight vector. The ATT is the special case ``a_j = 1/N``,
    and the test suite pins that the projection constant's standard error
    reduces to the ATT's.
    """
    z_crit = stats.norm.ppf(1 - alpha / 2)
    t_idx = np.flatnonzero(treated_mask)
    n_t = len(t_idx)

    z_cols = df.loc[treated_mask, project_vars].to_numpy(dtype=float)
    Z = np.column_stack([np.ones(n_t), z_cols])
    names = ["_cons"] + list(project_vars)
    p = Z.shape[1]

    zz = Z.T @ Z
    if np.linalg.matrix_rank(zz) < p:
        raise ValueError(
            f"project={project_vars} is collinear on the treated sample "
            "(after adding the constant), so the projection coefficients "
            "are not identified. Drop a redundant variable — note that a "
            "covariate constant among treated observations is collinear "
            "with the constant."
        )
    zz_inv = np.linalg.inv(zz)
    theta = zz_inv @ (Z.T @ tau_hat[t_idx])

    # A[j, i]: weight of treated observation i in coefficient j.
    A = zz_inv @ Z.T

    cluster_vals = df[cluster_col].to_numpy()
    cohort_vals = df["_ft"].to_numpy()
    rel_vals = np.round(
        df[time_col].to_numpy(dtype=float) - df["_ft"].to_numpy(dtype=float)
    )

    se = np.empty(p, dtype=float)
    for j in range(p):
        w_j = np.zeros(len(df), dtype=float)
        w_j[t_idx] = A[j]
        se[j] = _bjs_se(
            design_all=design_all,
            design_untreated=design_untreated,
            treated_mask=treated_mask,
            target_weights=w_j,
            adjusted=tau_hat,
            cluster=cluster_vals,
            cohort=cohort_vals,
            relative_time=rel_vals,
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = np.where(se > 0, theta / se, 0.0)
    pval = 2 * (1 - stats.norm.cdf(np.abs(zstat)))

    return pd.DataFrame(
        {
            "term": names,
            "coef": theta,
            "se": se,
            "ci_lower": theta - z_crit * se,
            "ci_upper": theta + z_crit * se,
            "pvalue": np.where(se > 0, pval, 1.0),
        }
    )


# Register citation
CausalResult._CITATIONS["did_imputation"] = (
    "@article{borusyak2024revisiting,\n"
    "  title={Revisiting Event-Study Designs: Robust and Efficient Estimation},\n"
    "  author={Borusyak, Kirill and Jaravel, Xavier and Spiess, Jann},\n"
    "  journal={Review of Economic Studies},\n"
    "  volume={91},\n"
    "  number={6},\n"
    "  pages={3253--3285},\n"
    "  year={2024},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)
