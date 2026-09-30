"""
Stacked DID estimator (Cengiz, Dube, Lindner & Zipperer, 2019).

Creates "stacked" datasets — one sub-experiment for each treatment cohort —
then estimates DID on the stacked data with cohort-specific unit and time
fixed effects. This avoids the negative-weighting problem of TWFE under
staggered treatment timing and heterogeneous effects.

References
----------
Cengiz, D., Dube, A., Lindner, A. and Zipperer, B. (2019).
"The Effect of Minimum Wages on Low-Wage Jobs."
*Quarterly Journal of Economics*, 134(3), 1405-1454. [@cengiz2019effect]
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility


@accepts_aliases(_strict=True, id="group", unit="group", covariates="controls")
def stacked_did(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: Optional[str] = None,
    window: Tuple[int, int] = (-5, 5),
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    never_treated_only: bool = True,
    alpha: float = 0.05,
    weights: Optional[str] = None,
    event_id: Optional[str] = None,
    treated: Optional[str] = None,
    event_time: Optional[str] = None,
    family: str = "gaussian",
    spec: str = "event_study",
    absorb: Optional[str] = None,
    control_group: Optional[str] = None,
    events: Optional[str] = None,
    own_overlap: str = "drop",
) -> CausalResult:
    """
    Stacked DID estimator (Cengiz, Dube, Lindner & Zipperer, 2019).

    Constructs a stacked dataset with one sub-experiment per treatment
    cohort, then estimates event-study coefficients via TWFE on the
    stacked data with cohort-specific unit and time fixed effects.

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
    window : tuple of (int, int), default (-5, 5)
        Event window (inclusive) around treatment.
        E.g. ``(-5, 5)`` keeps relative times -5 through 5.
    controls : list of str, optional
        Additional control covariates.
    cluster : str, optional
        Variable for cluster-robust standard errors.
        Defaults to ``group`` (unit-level clustering).
    never_treated_only : bool, default True
        If True, use only never-treated units as controls.
        If False, also include not-yet-treated units as controls.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    weights : str, optional
        Column of non-negative observation weights (e.g. population, as in
        Cengiz et al.). The stacked regression becomes weighted least
        squares with Stata ``reghdfe [aw=]`` semantics.
    event_id, treated, event_time : str, optional
        Pass all three when ``data`` is **already stacked** -- one row per
        (sub-experiment, unit, period), e.g. a CDLZ stack whose controls were
        chosen with a clean-control rule (no other event within the window).
        ``event_id`` names the sub-experiment, ``treated`` is 1 for the
        sub-experiment's treated units, ``event_time`` is the period relative
        to its event. No stack is built; ``first_treat`` and
        ``never_treated_only`` are then ignored, and rows outside ``window``
        are dropped. Passing a pre-built stack as an ordinary panel used to
        rebuild it, reusing the never-treated controls in every cohort.
    family : {'gaussian', 'poisson'}, default 'gaussian'
        ``'poisson'`` fits the stacked regression by PPML with
        :func:`statspai.ppmlhdfe` (Stata ``ppmlhdfe``'s singleton and
        separation rules included), for count outcomes; coefficients are
        on the log scale.
    spec : {'event_study', 'pooled'}, default 'event_study'
        ``'event_study'`` estimates one coefficient per event time (the
        reference period -1 omitted) and reports the mean of the
        post-treatment ones as the ATT. ``'pooled'`` estimates a single
        treated x post coefficient -- the specification most stacked
        regressions in applied papers report.
    absorb : str, optional
        Fixed effects absorbed on top of unit x sub-experiment and period x
        sub-experiment, e.g. ``"id + ind^year + city^year"`` (``#`` also
        accepted).
    control_group : {'nevertreated', 'notyettreated', 'notyettreated_rows'}, optional
        Which rows serve as controls in a cohort's sub-experiment.
        ``'nevertreated'`` (``never_treated_only=True``) and
        ``'notyettreated'`` (units first treated after the window,
        ``never_treated_only=False``) take whole units; ``'notyettreated_rows'``
        takes every later-treated unit but only in the periods before its
        own treatment, as the stacking of Cunningham's *Mixtape* and many
        applied papers does. Overrides ``never_treated_only`` when given.
    events : str, optional
        A 0/1 column marking the periods in which a unit has an event, for
        treatments that happen more than once (several minimum-wage
        increases in one state). Every event becomes its own sub-experiment
        over ``window``; its controls are the units with **no** event inside
        that window (Cengiz et al.'s clean-control rule), in the window's
        periods. Replaces ``first_treat``.
    own_overlap : {'drop', 'keep'}, default 'drop'
        With ``events``: an event whose own unit has another event inside
        its window has a contaminated comparison; ``'drop'`` leaves it out
        (counted in ``model_info['n_events_dropped_overlap']``), ``'keep'``
        keeps it.

    Notes
    -----
    The regression is fitted by :func:`statspai.absorb_ols` with unit x
    sub-experiment and period x sub-experiment effects, clustered by
    ``cluster`` (default ``group``), so the standard errors follow
    ``reghdfe``'s small-sample conventions (effects nested in the cluster are
    not charged).

    Returns
    -------
    CausalResult
        Result object with ``.summary()``, ``.plot()`` (event study),
        and ``.cite()`` methods. Event study coefficients are stored
        in ``model_info['event_study']``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=120, n_periods=8, staggered=True, seed=0)
    >>> result = sp.stacked_did(
    ...     data=df, y='y', group='unit', time='time',
    ...     first_treat='first_treat', window=(-3, 3),
    ... )
    >>> bool(result.estimate is not None)
    True
    >>> _ = result.summary()
    >>> fig, ax = result.plot()  # event-study plot
    """
    # ── Input validation ─────────────────────────────────────────── #
    df = data.copy()
    required_cols = [y, group, time] + (
        [] if (event_id is not None or events is not None) else [first_treat]
    )
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if controls:
        for col in controls:
            if col not in df.columns:
                raise ValueError(f"Control column '{col}' not found in data.")

    if cluster is None:
        cluster = group

    family = str(family).lower()
    if family not in ("gaussian", "poisson"):
        raise MethodIncompatibility(
            f"family must be 'gaussian' or 'poisson', got {family!r}."
        )
    spec = str(spec).lower()
    if spec not in ("event_study", "pooled"):
        raise MethodIncompatibility(
            f"spec must be 'event_study' or 'pooled', got {spec!r}."
        )
    if control_group is None:
        control_group = "nevertreated" if never_treated_only else "notyettreated"
    control_group = str(control_group).lower()
    if control_group not in ("nevertreated", "notyettreated", "notyettreated_rows"):
        raise MethodIncompatibility(
            "control_group must be 'nevertreated', 'notyettreated' or "
            f"'notyettreated_rows', got {control_group!r}."
        )
    never_treated_only = control_group == "nevertreated"

    if window[0] >= 0:
        raise ValueError("window[0] must be negative (pre-treatment periods).")
    if window[1] < 0:
        raise ValueError("window[1] must be non-negative (post-treatment periods).")

    if events is not None:
        if event_id is not None or first_treat is not None:
            raise MethodIncompatibility(
                "stacked_did: pass one of events=, first_treat= or a pre-built "
                "stack (event_id=), not several.",
            )
        if own_overlap not in ("drop", "keep"):
            raise MethodIncompatibility("own_overlap must be 'drop' or 'keep'.")
        if events not in df.columns:
            raise MethodIncompatibility(f"Column '{events}' not found in data.")
        ev_col = pd.to_numeric(df[events], errors="coerce")
        if not ev_col.dropna().isin([0, 1]).all():
            raise MethodIncompatibility(f"Column '{events}' must be 0/1.")
        df, n_events_dropped = _stack_events(
            df, group, time, ev_col, window, own_overlap
        )
        event_id, treated, event_time = "_ev_id", "_ev_treated", "_ev_time"
    else:
        n_events_dropped = None

    prebuilt = event_id is not None
    if prebuilt:
        missing = [
            nm
            for nm, v in (("treated", treated), ("event_time", event_time))
            if v is None
        ]
        if missing:
            raise MethodIncompatibility(
                "A pre-built stack needs event_id, treated and event_time; "
                f"missing {missing}.",
                recovery_hint="Pass all three columns of the stack.",
            )
        for col in (event_id, treated, event_time):
            if col not in df.columns:
                raise MethodIncompatibility(f"Column '{col}' not found in data.")
        tr = pd.to_numeric(df[treated], errors="coerce")
        if not tr.dropna().isin([0, 1]).all():
            raise MethodIncompatibility(f"Column '{treated}' must be 0/1.")
        rel = pd.to_numeric(df[event_time], errors="coerce").astype(float)
        keep = (rel >= window[0]) & (rel <= window[1]) & tr.notna()
        stacked = df.loc[keep].copy()
        stacked["_cohort"] = stacked[event_id]
        stacked["_rel_time"] = rel[keep].to_numpy()
        stacked["_treated_unit"] = tr[keep].astype(int).to_numpy()
        stacked["_post"] = (stacked["_rel_time"] >= 0).astype(int)
        cohort_values = sorted(stacked["_cohort"].unique().tolist())
        if stacked.empty:
            raise DataInsufficient(
                "No rows of the pre-built stack fall inside window.",
                recovery_hint="Widen window or check event_time.",
            )
    else:
        if first_treat is None:
            raise MethodIncompatibility(
                "first_treat is required unless event_id is given.",
                recovery_hint="Pass first_treat=, or a pre-built stack via event_id=.",
            )
        # ── Normalize first_treat ────────────────────────────────────── #
        ft = df[first_treat].copy().astype(float)
        ft = ft.replace(0, np.inf)
        ft = ft.fillna(np.inf)
        df["_ft"] = ft

        # ── Step 1: Identify cohorts ─────────────────────────────────── #
        cohort_values = sorted(df.loc[np.isfinite(df["_ft"]), "_ft"].unique())
        if len(cohort_values) == 0:
            raise ValueError("No treated cohorts found. Check 'first_treat' column.")

        never_mask = np.isinf(df["_ft"])
        never_units = set(df.loc[never_mask, group].unique())

        # ── Step 2 & 3: Build sub-experiments and stack ──────────────── #
        stacked_frames = []

        for g in cohort_values:
            t_lo = g + window[0]
            t_hi = g + window[1]

            # Units in this cohort (treated at time g)
            cohort_units = set(df.loc[df["_ft"] == g, group].unique())

            in_window = (df[time].astype(float) >= t_lo) & (
                df[time].astype(float) <= t_hi
            )
            if control_group == "notyettreated_rows":
                # Later-treated units are controls only before their own
                # treatment; never-treated units throughout.
                ctrl_rows = (df["_ft"] > g) & (df[time].astype(float) < df["_ft"])
                if not (ctrl_rows & in_window).any():
                    continue
                sub = df[in_window & (df[group].isin(cohort_units) | ctrl_rows)].copy()
            else:
                # Control units
                if never_treated_only:
                    ctrl_units = never_units
                else:
                    # Not-yet-treated: units whose first_treat > t_hi
                    nyt_mask = df["_ft"] > t_hi
                    ctrl_units = never_units | set(df.loc[nyt_mask, group].unique())

                all_units = cohort_units | ctrl_units
                if len(ctrl_units) == 0:
                    continue  # skip cohort if no controls available

                # Restrict to units and time window
                sub = df[df[group].isin(all_units) & in_window].copy()

            if len(sub) == 0:
                continue

            sub["_cohort"] = g
            sub["_rel_time"] = sub[time].astype(float) - g
            sub["_treated_unit"] = sub[group].isin(cohort_units).astype(int)
            sub["_post"] = (sub["_rel_time"] >= 0).astype(int)

            stacked_frames.append(sub)

        if len(stacked_frames) == 0:
            raise ValueError(
                "No valid sub-experiments could be constructed. "
                "Check data coverage and window size."
            )

        stacked = pd.concat(stacked_frames, ignore_index=True)
    n_cohorts = len(stacked["_cohort"].unique())
    n_units = stacked[group].nunique()
    n_stacked = len(stacked)

    # ── Step 4: Estimate on stacked data ─────────────────────────── #
    # Create event-study dummies: D_k = 1(rel_time == k & treated_unit)
    # Exclude k = -1 as reference period
    rel_times = sorted(stacked["_rel_time"].unique())
    rel_times_est = [k for k in rel_times if k != -1]

    if len(rel_times_est) == 0:
        raise ValueError("Not enough relative time periods for estimation.")

    if spec == "pooled":
        rel_times_est = []
        stacked["_treat_post"] = (
            (stacked["_treated_unit"] == 1) & (stacked["_post"] == 1)
        ).astype(float)
        D_cols = ["_treat_post"]
    else:
        # Build treatment interaction dummies (names safe for formulas)
        D_cols = []
        for k in rel_times_est:
            col_name = f"_D_{'m' if k < 0 else ''}{abs(int(k))}"
            stacked[col_name] = (
                (stacked["_rel_time"] == k) & (stacked["_treated_unit"] == 1)
            ).astype(float)
            D_cols.append(col_name)

    # Add controls if specified
    x_cols = list(D_cols)
    if controls:
        x_cols = x_cols + controls

    # Create cohort-specific FE groups
    stacked["_unit_cohort"] = (
        stacked[group].astype(str) + "_" + stacked["_cohort"].astype(str)
    )
    stacked["_time_cohort"] = (
        stacked[time].astype(str) + "_" + stacked["_cohort"].astype(str)
    )
    extra_fe: List[str] = []
    if absorb:
        from ..core._group_terms import resolve_group_terms

        stacked, extra_fe = resolve_group_terms(
            stacked, [t.strip() for t in str(absorb).split("+") if t.strip()]
        )
    fe_cols = ["_unit_cohort", "_time_cohort"] + extra_fe

    w_arr = None
    if weights is not None:
        if weights not in stacked.columns:
            raise MethodIncompatibility(f"Weight column '{weights}' not found in data.")
        w_arr = pd.to_numeric(stacked[weights], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(w_arr)) or (w_arr < 0).any():
            raise MethodIncompatibility("weights must be finite and non-negative.")
        pos = w_arr > 0
        stacked = stacked.loc[pos].reset_index(drop=True)
        w_arr = w_arr[pos]
    if cluster not in stacked.columns:
        raise MethodIncompatibility(f"Cluster column '{cluster}' not found in data.")
    complete = (
        stacked[[y] + x_cols + [cluster] + fe_cols].notna().all(axis=1).to_numpy()
    )
    if not complete.all():
        stacked = stacked.loc[complete].reset_index(drop=True)
        if w_arr is not None:
            w_arr = w_arr[complete]

    if family == "poisson":
        # PPML on the stack, Stata ppmlhdfe conventions (singletons and
        # separated rows dropped, regressors absorbed by the effects
        # omitted).
        from ..regression.count import ppmlhdfe

        if w_arr is not None:
            stacked["_stack_w"] = w_arr
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ppmlhdfe: dropped")
            pf = ppmlhdfe(
                data=stacked,
                y=y,
                x=x_cols,
                absorb=" + ".join(fe_cols),
                cluster=cluster,
                weights="_stack_w" if w_arr is not None else None,
            )
        names = list(pf.params.index)
        beta_s = pf.params.reindex(x_cols)
        V_full = np.asarray(pf.data_info["var_cov"], dtype=float)
        pos_of = {nm: i for i, nm in enumerate(names)}
        idx = [pos_of.get(c) for c in x_cols]
        beta = beta_s.to_numpy(dtype=float)
        V = np.full((len(x_cols), len(x_cols)), np.nan)
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                if ia is not None and ib is not None:
                    V[a, b] = V_full[ia, ib]
        n_stacked = int(pf.data_info["nobs"])
        omitted_names = list(pf.model_info.get("omitted", []))
        fit_info = {
            "estimator": "PPML (ppmlhdfe)",
            "n_singletons": pf.model_info.get("n_singletons"),
            "n_separated": pf.model_info.get("n_separated"),
            "pseudo_r2": pf.model_info.get("pseudo_r2"),
        }
    else:
        # Two-way FE regression by the HDFE kernel: unit x sub-experiment and
        # period x sub-experiment effects, reghdfe's weighting and CRV1
        # conventions (effects nested in the cluster are not charged).
        from ..panel.hdfe import absorb_ols

        fit = absorb_ols(
            y=stacked[y].to_numpy(dtype=float),
            X=stacked[x_cols].to_numpy(dtype=float),
            fe=stacked[fe_cols],
            weights=w_arr,
            cluster=stacked[cluster].to_numpy(),
            drop_singletons=False,
        )
        beta = np.asarray(fit["coef"], dtype=float)
        V = np.asarray(fit["vcov"], dtype=float)
        n_stacked = int(fit["n"])
        omitted_names = [x_cols[j] for j in fit.get("omitted", [])]
        fit_info = {"estimator": "OLS (HDFE)"}
    dropped_terms = [
        int(rel_times_est[j])
        for j, c in enumerate(D_cols)
        if c in omitted_names and j < len(rel_times_est)
    ]

    # Map coefficients to event-study names
    es_betas = {}
    for idx, k in enumerate(rel_times_est):
        es_betas[k] = beta[idx]

    # ── Step 5: Cluster-robust standard errors ───────────────────── #
    se_vec = np.sqrt(np.maximum(np.diag(V), 0.0))
    es_se = {}
    for idx, k in enumerate(rel_times_est):
        es_se[k] = se_vec[idx]

    # Joint covariance of the event-study coefficients (one stacked
    # regression, so the cross-horizon terms are available exactly).
    es_vcov = pd.DataFrame(
        V[: len(rel_times_est), : len(rel_times_est)],
        index=[int(k) for k in rel_times_est],
        columns=[int(k) for k in rel_times_est],
    )

    # ── Step 6: Aggregate ATT (post-treatment periods) ───────────── #
    post_ks = [k for k in rel_times_est if k >= 0]
    if spec == "pooled":
        att = float(beta[0])
        att_se = float(np.sqrt(max(V[0, 0], 0.0)))
    elif len(post_ks) == 0:
        raise DataInsufficient(
            f"stacked_did: window={window} contains no post-treatment "
            "relative time (k >= 0) with data, so the ATT is not defined. "
            "Widen the window's upper end.",
            diagnostics={"window": list(window)},
        )
    if spec != "pooled" and len(post_ks) > 0:
        att = np.mean([es_betas[k] for k in post_ks])
        # Delta method: ATT = mean of post betas → se = sqrt(w' V w)
        post_indices = [rel_times_est.index(k) for k in post_ks]
        n_post = len(post_indices)
        w = np.zeros(len(rel_times_est))
        for pi in post_indices:
            w[pi] = 1.0 / n_post
        # Event-time block only: V also carries the controls.
        k_es = len(rel_times_est)
        att_var = w @ V[:k_es, :k_es] @ w
        att_se = np.sqrt(max(att_var, 0.0))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    att_pval = float(2 * stats.norm.sf(abs(att) / att_se)) if att_se > 0 else np.nan
    att_ci = (att - z_crit * att_se, att + z_crit * att_se)

    # ── Build event study detail DataFrame ───────────────────────── #
    all_ks = sorted(set(rel_times_est) | {-1}) if spec != "pooled" else []
    rows = []
    for k in all_ks:
        if k == -1:
            rows.append(
                {
                    "relative_time": int(k),
                    "att": 0.0,
                    "se": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "pvalue": np.nan,
                }
            )
        else:
            b = es_betas[k]
            s = es_se[k]
            p = float(2 * stats.norm.sf(abs(b) / s)) if s > 0 else np.nan
            rows.append(
                {
                    "relative_time": int(k),
                    "att": b,
                    "se": s,
                    "ci_lower": b - z_crit * s,
                    "ci_upper": b + z_crit * s,
                    "pvalue": p,
                }
            )

    detail = pd.DataFrame(
        rows,
        columns=["relative_time", "att", "se", "ci_lower", "ci_upper", "pvalue"],
    )

    # ── Build model_info ─────────────────────────────────────────── #
    model_info = {
        "method_full": "Stacked DID (Cengiz, Dube, Lindner & Zipperer, 2019)",
        "n_cohorts": n_cohorts,
        "cohorts": sorted(cohort_values),
        "n_units": n_units,
        "n_stacked_obs": n_stacked,
        "events": events,
        "n_events_dropped_overlap": n_events_dropped,
        "window": window,
        "never_treated_only": never_treated_only,
        "control_group": control_group,
        "family": family,
        "spec": spec,
        "absorb": absorb,
        "fit": fit_info,
        "prebuilt_stack": prebuilt,
        "weights": weights,
        "omitted_event_times": dropped_terms,
        "cluster_var": cluster,
        "event_study": detail,
        "event_study_betas": es_betas,
        "event_study_se": es_se,
        "event_study_vcov": es_vcov,
    }

    _result = CausalResult(
        method="Stacked DID (Cengiz et al. 2019)",
        estimand="ATT",
        estimate=float(att),
        se=float(att_se),
        pvalue=float(att_pval),
        ci=att_ci,
        alpha=alpha,
        n_obs=n_stacked,
        detail=detail,
        model_info=model_info,
        _citation_key="stacked_did",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.stacked_did",
            params={
                "y": y,
                "group": group,
                "time": time,
                "first_treat": first_treat,
                "window": list(window),
                "controls": controls,
                "cluster": cluster,
                "never_treated_only": never_treated_only,
                "alpha": alpha,
                "weights": weights,
                "event_id": event_id,
                "family": family,
                "spec": spec,
                "absorb": absorb,
                "control_group": control_group,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ──────────────────────────────────────────────────────────────────── #
#  Private helpers
# ──────────────────────────────────────────────────────────────────── #


def _twoway_demean(
    y: np.ndarray,
    X: np.ndarray,
    group1: np.ndarray,
    group2: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterative two-way demeaning (alternating projection) for Y and X.

    Returns demeaned (y_dm, X_dm).
    """
    # Build group index arrays for fast lookup
    g1_map: Dict[object, List[int]] = {}
    for i, g in enumerate(group1):
        g1_map.setdefault(g, []).append(i)
    g2_map: Dict[object, List[int]] = {}
    for i, g in enumerate(group2):
        g2_map.setdefault(g, []).append(i)

    # Stack y and X for simultaneous demeaning
    Z = np.column_stack([y, X])  # (n, 1+k)

    for _ in range(max_iter):
        Z_old = Z.copy()

        # Demean by group1
        for indices in g1_map.values():
            idx = np.array(indices)
            Z[idx] -= Z[idx].mean(axis=0)

        # Demean by group2
        for indices in g2_map.values():
            idx = np.array(indices)
            Z[idx] -= Z[idx].mean(axis=0)

        # Check convergence
        if np.max(np.abs(Z - Z_old)) < tol:
            break

    return Z[:, 0], Z[:, 1:]


def _ols(X: np.ndarray, y: np.ndarray) -> tuple:
    """OLS regression. Returns (coefficients, residuals)."""
    if X.shape[1] == 0:
        return np.array([]), y.copy()

    # Use lstsq for numerical stability
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    return beta, residuals


def _cluster_robust_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """
    Cluster-robust variance-covariance matrix.

    V = c * (X'X)^{-1} B (X'X)^{-1}, B = sum_g (X_g' e_g)(X_g' e_g)',
    small-sample correction c = (G/(G-1)) * ((n-1)/(n-k)) (G>1 else 1).

    Delegates to the canonical ``core._vcov.cluster_robust_vcov`` (CLAUDE.md
    §4); keeps the k==0 guard and the inv->pinv bread fallback. Verified
    byte-identical to the prior hand-rolled implementation (incl. the singular
    pinv path).
    """
    from ..core._vcov import cluster_robust_vcov

    n, k = X.shape
    if k == 0:
        return np.empty((0, 0))

    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    return cluster_robust_vcov(
        X,
        residuals,
        cluster_ids,
        correction="liang_zeger",
        XtX_inv=XtX_inv,
    )


def _cluster_robust_se(
    X: np.ndarray,
    residuals: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    """Cluster-robust standard errors."""
    V = _cluster_robust_vcov(X, residuals, cluster_ids)
    if V.size == 0:
        return np.array([])
    return np.asarray(np.sqrt(np.maximum(np.diag(V), 0.0)), dtype=float)


def _stack_events(
    df: pd.DataFrame,
    group: str,
    time: str,
    ev: pd.Series,
    window: Tuple[int, int],
    own_overlap: str,
) -> Tuple[pd.DataFrame, int]:
    """One sub-experiment per event (unit, g) with clean controls.

    The window is ``[g + window[0], g + window[1]]``, truncated at the edges
    of the panel. Controls are the units with no event inside it; the
    treated unit contributes its own rows. Returns the stack (columns
    ``_ev_id``, ``_ev_treated``, ``_ev_time``) and the number of events
    dropped for another event of their own unit inside the window.
    """
    lo, hi = window
    t = df[time].to_numpy(dtype=float)
    u = df[group].to_numpy()
    is_ev = ev.fillna(0).to_numpy() == 1
    ev_rows = df.loc[is_ev, [group, time]]
    ev_times = {
        unit: np.sort(ts.to_numpy(dtype=float))
        for unit, ts in ev_rows.groupby(group)[time]
    }
    order = ev_rows.assign(_t=ev_rows[time].astype(float)).sort_values(["_t", group])
    frames: List[pd.DataFrame] = []
    dropped = 0
    for eu, g in zip(order[group].to_numpy(), order["_t"].to_numpy()):
        w_lo, w_hi = g + lo, g + hi
        own = ev_times[eu]
        if own_overlap == "drop" and np.any((own != g) & (own >= w_lo) & (own <= w_hi)):
            dropped += 1
            continue
        dirty = [
            unit
            for unit, ts in ev_times.items()
            if unit != eu and np.any((ts >= w_lo) & (ts <= w_hi))
        ]
        keep = (t >= w_lo) & (t <= w_hi) & ((u == eu) | ~np.isin(u, dirty))
        sub = df.loc[keep].copy()
        sub["_ev_id"] = f"{eu}:{g:g}"
        sub["_ev_treated"] = (sub[group].to_numpy() == eu).astype(int)
        sub["_ev_time"] = sub[time].astype(float) - g
        frames.append(sub)
    if not frames:
        raise DataInsufficient(
            "stacked_did(events=): no event has a usable window.",
            recovery_hint="Check the events column, or use own_overlap='keep'.",
        )
    if dropped:
        warnings.warn(
            f"stacked_did: dropped {dropped} event(s) whose own unit has another "
            "event inside the window (own_overlap='drop').",
            UserWarning,
            stacklevel=3,
        )
    return pd.concat(frames, ignore_index=True), dropped
