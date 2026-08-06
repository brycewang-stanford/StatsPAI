"""
Sun & Abraham (2021) interaction-weighted event-study estimator.

Fits a *saturated* regression that interacts every cohort dummy with
every relative-time dummy, then aggregates the interaction coefficients
across cohorts using empirical cohort shares to deliver the IW
estimator δ̂^IW_ℓ that is robust to heterogeneous treatment effects
(Sun & Abraham 2021, Theorem 1 / Corollary 1).

Standard errors are computed from the classical OLS sandwich:

    Var(β̂) = (X'X)⁻¹  ( Σ_c  X_c' u_c u_c' X_c )  (X'X)⁻¹

clustered at the unit (or a user-supplied) level.  δ̂^IW_ℓ is a product
of two estimated objects — the interaction coefficients and the cohort
shares — so its variance carries two terms (SA 2021, Prop. 3):

    Var(δ̂^IW_ℓ) = w_ℓ' Var(β̂) w_ℓ  +  β_ℓ' Var(ŵ_ℓ) β_ℓ

The second term is the cost of estimating the shares. It is degenerate
whenever a single cohort is eligible at ℓ (then ŵ ≡ 1), which is why
omitting it is easy to miss: on ``mpdta`` it changes nothing at
single-cohort event times and understates the SE by up to 2% where two
cohorts contribute.

.. warning::
   The two reference implementations disagree here and StatsPAI cannot
   match both. Stata ``eventstudyinteract`` (Liyang Sun's own package)
   carries the share term; R ``fixest::sunab`` treats the shares as
   fixed and reports the first term only. StatsPAI follows
   ``eventstudyinteract``, since Prop. 3 derives the share term and
   dropping it is anti-conservative. Expect StatsPAI SEs to sit slightly
   *above* ``fixest``'s at multi-cohort event times and to agree with it
   exactly at single-cohort ones.

References
----------
Sun, L. and Abraham, S. (2021).
    "Estimating Dynamic Treatment Effects in Event Studies with
     Heterogeneous Treatment Effects."
    *Journal of Econometrics*, 225(2), 175-199. [@sun2021estimating]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ._core import drop_unusable_rows as _drop_unusable_rows

# ======================================================================
# Public API
# ======================================================================


def _cohort_share_vcov(shares: np.ndarray, n_obs: int) -> np.ndarray:
    """Covariance matrix of the estimated cohort shares at one relative time.

    Stata ``eventstudyinteract`` obtains this by regressing each cohort
    indicator on the full set of relative-time dummies (no constant) and
    sandwiching the stack with ``avar``. Because those dummies are
    *mutually exclusive indicators*, the design matrix is orthogonal with
    ``X'X = diag(N_ℓ)``, and that whole sandwich collapses to the
    multinomial covariance

        Var(ŵ_ℓ) = (diag(ŵ_ℓ) − ŵ_ℓ ŵ_ℓ') / N_ℓ

    which is what is computed here — same estimator, no SUR machinery.
    Derivation: the coefficient at ℓ is the within-ℓ mean of the cohort
    indicator, so its residual is ``1{g_i = g} − ŵ_{g,ℓ}``; the robust
    meat at ℓ is then ``Σ_i u_ig u_ih / N_ℓ²``, which equals
    ``ŵ_g(1 − ŵ_g)/N_ℓ`` on the diagonal and ``−ŵ_g ŵ_h / N_ℓ`` off it.

    ``n_obs`` counts *observations* at the relative time, not units,
    matching eventstudyinteract's panel-level normalization.

    Returns a zero matrix when the shares are degenerate (a single
    eligible cohort, so ŵ ≡ 1) or when ``n_obs`` is unusable — in both
    cases there is no share-estimation uncertainty to add.
    """
    k = len(shares)
    if k <= 1 or n_obs <= 0:
        return np.zeros((k, k))
    return (np.diag(shares) - np.outer(shares, shares)) / float(n_obs)


def _sunab_pretrend_test(
    event_study: pd.DataFrame,
    combos: "dict[int, Tuple[np.ndarray, float]]",
    v_int: np.ndarray,
    *,
    pretest: str,
    pretest_periods: Optional[int],
) -> Optional[dict]:
    """Joint test that the pre-treatment IW effects are all zero.

    Sun & Abraham's estimator produced no pre-trend test at all before
    this: callers had to read the event-study table by eye, which invites
    the classic error of declaring parallel trends because no single
    pre-period coefficient reached significance. Individually
    insignificant leads are routinely jointly significant.

    The test is a Wald statistic on the pre-period IW estimates using
    their **joint** covariance, not the diagonal:

        Cov(δ̂_ℓ, δ̂_m) = w_ℓ' Var(β̂) w_m          for ℓ ≠ m
        Var(δ̂_ℓ)       = w_ℓ' Var(β̂) w_ℓ + β_ℓ' Var(ŵ_ℓ) β_ℓ

    The share-variance term appears only on the diagonal because the
    relative-time dummies are mutually exclusive, so the share estimates
    at different ℓ are built from disjoint observations and their
    covariance block is zero off-diagonal (same algebra as
    :func:`_cohort_share_vcov`).

    ``pretest_periods=k`` keeps the ``k`` **estimated** leads closest to
    treatment — counted over the leads that exist, since ℓ = −1 is the
    omitted reference and a literal ``ℓ >= -k`` cutoff would quietly
    return one fewer than asked for. Distant leads are often estimated on
    few cohorts and drag the test toward non-rejection.

    Returns ``None`` when disabled or when there are no pre-periods.
    """
    if pretest == "none":
        return None

    pre = sorted(e for e in combos if e < 0)
    if pretest_periods is not None:
        # The k nearest *estimated* leads, not literally ℓ >= -k. ℓ = -1 is
        # the omitted reference here, so the estimated leads start at -2 and
        # a literal cutoff would silently return k-1 of them (or none).
        pre = pre[-pretest_periods:]
    if not pre:
        return None

    est = np.array(
        [
            float(event_study.loc[event_study["relative_time"] == e, "att"].iloc[0])
            for e in pre
        ]
    )
    k = len(pre)
    cov = np.empty((k, k), dtype=float)
    for a, ea in enumerate(pre):
        wa, share_a = combos[ea]
        for b, eb in enumerate(pre):
            wb, _ = combos[eb]
            cov[a, b] = float(wa @ v_int @ wb)
        cov[a, a] += share_a

    from ._core import joint_wald as _joint_wald

    out = _joint_wald(est, cov)
    out["relative_times"] = pre
    return out


def _resolve_control_cohort(
    df: pd.DataFrame,
    control_cohort: Any,
    g: str,
) -> Tuple[pd.Series, str]:
    """Turn ``control_cohort=`` into a unit-level boolean reference mask.

    Mirrors Stata ``eventstudyinteract``'s ``control_cohort(varname)``,
    which takes a *binary variable* marking the control cohort — allowing
    either never-treated or last-treated units, chosen by the analyst
    rather than inferred. Two spellings are accepted here:

    - a column name holding a 0/1 (or boolean) indicator, matching the
      Stata option exactly;
    - a cohort value, or a sequence of cohort values, from ``g`` — the
      shorthand that avoids constructing an indicator column by hand.

    Returns ``(mask, label)`` where ``label`` describes the resolved
    reference group for diagnostics and error messages.
    """
    # Column-name spelling. Checked before scalars so that a legitimately
    # numeric column name still resolves as a column.
    if isinstance(control_cohort, str):
        if control_cohort not in df.columns:
            raise ValueError(
                f"control_cohort='{control_cohort}' is not a column in the "
                f"data. Pass the name of a 0/1 indicator column, or a "
                f"cohort value from '{g}'."
            )
        col = df[control_cohort]
        vals = set(pd.unique(col.dropna()))
        if not vals <= {0, 1, True, False, 0.0, 1.0}:
            raise ValueError(
                f"control_cohort column '{control_cohort}' must be a binary "
                f"0/1 indicator (Stata eventstudyinteract convention), but "
                f"it takes values {sorted(vals, key=repr)[:6]}. To select by "
                f"cohort value instead, pass the value itself, e.g. "
                f"control_cohort={sorted(vals, key=repr)[0]!r}."
            )
        mask = col.fillna(0).astype(bool)
        return mask, f"column '{control_cohort}'"

    # Cohort-value spelling (scalar or sequence).
    if isinstance(control_cohort, (list, tuple, set, np.ndarray, pd.Series)):
        wanted = [int(v) for v in control_cohort]
    else:
        wanted = [int(control_cohort)]

    present = set(df[g].unique())
    missing = [v for v in wanted if v not in present]
    if missing:
        raise ValueError(
            f"control_cohort={control_cohort!r} names cohort value(s) "
            f"{missing} that do not occur in '{g}'. Available cohorts: "
            f"{sorted(present)}."
        )
    mask = df[g].isin(wanted)
    return mask, f"{g} in {wanted}"


def sun_abraham(
    data: pd.DataFrame,
    y: str,
    g: str,
    t: str,
    i: str,
    event_window: Optional[Tuple[int, int]] = None,
    control_group: str = "nevertreated",
    control_cohort: Optional[Any] = None,
    covariates: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    aggregation: str = "event_time",
    alpha: float = 0.05,
    pretest: str = "joint",
    pretest_periods: Optional[int] = None,
) -> CausalResult:
    """
    Sun & Abraham (2021) interaction-weighted event-study estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    y : str
        Outcome variable.
    g : str
        Cohort variable: first treatment period (0 or inf = never treated).
    t : str
        Time period variable.
    i : str
        Unit identifier.
    event_window : tuple of (int, int), optional
        (min_relative_time, max_relative_time).
        Default: observed range in the data.
    control_group : str, default 'nevertreated'
        ``'nevertreated'`` or ``'lastcohort'``.  When ``'lastcohort'``,
        the latest treated cohort is used as the reference and dropped
        from the IW aggregation. Ignored when ``control_cohort`` is given.
    control_cohort : str, scalar or sequence, optional
        Nominate the reference cohort explicitly instead of inferring it,
        mirroring Stata ``eventstudyinteract``'s ``control_cohort(varname)``.
        Accepts either

        - a **column name** holding a 0/1 indicator of control units (the
          Stata spelling), or
        - a **cohort value** — or list of values — drawn from ``g``.

        Whatever it selects becomes the reference group and is removed from
        the set of estimated cohorts, so no unit sits on both sides of its
        own comparison. Useful when the never-treated group is unsuitable
        (contaminated, or absent) and a specific late cohort is the
        credible control.

        .. note::
           Sun & Abraham (2021) require the control cohort to be untreated
           over the estimation window. When using a last-treated cohort,
           drop the periods in which it turns on — StatsPAI will not do
           that for you, exactly as ``eventstudyinteract`` will not.
    covariates : list of str, optional
        Additional controls (time-varying; added linearly).
    cluster : str, optional
        Cluster variable for SEs. Default: clusters on ``i``.
    aggregation : {'event_time', 'fixest_att'}, default 'event_time'
        Overall post-treatment summary convention. ``'event_time'`` is the
        historical StatsPAI default: equal-weight the post-treatment
        relative-time IW effects. ``'fixest_att'`` weights each post
        cohort-time cell by its treated cohort size, matching
        ``fixest::summary(..., agg='att')`` and Stata/R default ATT
        parity on balanced staggered panels.
    alpha : float, default 0.05
        Significance level.
    pretest : {'joint', 'none'}, default 'joint'
        Report a joint Wald test that all pre-treatment IW effects are
        zero, in ``model_info['pretrend_test']``.

        Reading the event-study table by eye is the standard way to get
        this wrong: leads that are individually insignificant are often
        jointly significant, and "no star on any lead" is not evidence of
        parallel trends. The test uses the full covariance across leads,
        not the diagonal.

        Failing to reject is still weak evidence — it is a statement about
        power as much as about trends. Pair it with
        :func:`statspai.honest_did` or :func:`statspai.pretrends_power`.
    pretest_periods : int, optional
        Restrict the joint test to the ``pretest_periods`` **estimated**
        leads nearest treatment, in the spirit of Stata's ``pretrends(k)``.
        Default uses every estimated lead. Distant leads often rest on few
        cohorts and pull the statistic toward non-rejection.

        Counted over the leads that exist, not by literal event time:
        ℓ = −1 is the omitted reference, so on a panel whose leads are
        −4, −3, −2 the value ``2`` selects ``[-3, -2]``.

    Returns
    -------
    CausalResult
        ``.detail`` is the event-study table (IW ATT by relative time
        with cluster-robust SE and 1−α CI).  ``.estimate`` / ``.se``
        are the simple post-treatment average and its delta-method SE.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=120, n_periods=8, staggered=True, seed=42)
    >>> result = sp.sun_abraham(df, y='y', g='first_treat', t='time', i='unit')
    >>> result.estimate > 0  # post-treatment IW average (true effect 0.5)
    True
    >>> list(result.detail.columns[:3])  # event-study table by relative time
    ['relative_time', 'att', 'se']

    References
    ----------
    Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects in
    event studies with heterogeneous treatment effects. *Journal of
    Econometrics*. [@sun2021estimating]
    """
    df = data.copy()

    for col in [y, g, t, i]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
    if covariates:
        for c in covariates:
            if c not in df.columns:
                raise ValueError(f"Covariate '{c}' not found")
    # Drop rows the estimator cannot use before the cohort checks below, so a
    # wiped outcome surfaces as an error instead of a headline ATT of 0.0.
    df = _drop_unusable_rows(
        df,
        columns=[y, t, i, *(covariates or [])],
        function="sun_abraham",
    )
    if pretest not in ("joint", "none"):
        raise ValueError(f"pretest must be 'joint' or 'none', got {pretest!r}.")
    if pretest_periods is not None:
        if isinstance(pretest_periods, bool) or not isinstance(pretest_periods, int):
            raise ValueError(
                f"pretest_periods must be a positive int or None, got "
                f"{pretest_periods!r}."
            )
        if pretest_periods < 1:
            raise ValueError(f"pretest_periods must be >= 1, got {pretest_periods}.")
    if control_group not in ("nevertreated", "lastcohort"):
        raise ValueError(
            f"control_group must be 'nevertreated' or 'lastcohort', "
            f"got {control_group!r}"
        )
    aggregation_aliases = {
        "event_time": "event_time",
        "event_time_equal": "event_time",
        "equal_event_time": "event_time",
        "fixest": "fixest_att",
        "fixest_att": "fixest_att",
        "treated_cell": "fixest_att",
        "treated_cell_weighted": "fixest_att",
    }
    if aggregation not in aggregation_aliases:
        raise ValueError(
            "aggregation must be one of 'event_time' or 'fixest_att', "
            f"got {aggregation!r}"
        )
    aggregation_key = aggregation_aliases[aggregation]

    df[g] = df[g].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
    time_periods = sorted(df[t].unique())
    t_max = max(time_periods)
    cohorts_all = sorted([v for v in df[g].unique() if v > 0 and v <= t_max])

    if not cohorts_all:
        raise ValueError("No treated cohorts found in the data.")

    # Reference cohort: explicit (control_cohort=), else never-treated
    # (g=0) or the last cohort.
    if control_cohort is not None:
        ref_mask, ref_label = _resolve_control_cohort(df, control_cohort, g)
        # Anything flagged as control is a reference unit, never an
        # estimated cohort — otherwise the same unit would appear on both
        # sides of its own comparison.
        ref_cohort_vals = set(df.loc[ref_mask, g].unique())
        cohorts = [c for c in cohorts_all if c not in ref_cohort_vals]
        control_cohort_label = ref_label
    elif control_group == "lastcohort":
        ref_cohort = max(cohorts_all)
        cohorts = [c for c in cohorts_all if c != ref_cohort]
        ref_mask = df[g] == ref_cohort
        control_cohort_label = f"lastcohort={ref_cohort}"
    else:
        cohorts = cohorts_all
        ref_mask = df[g] == 0
        control_cohort_label = "nevertreated"

    if not cohorts:
        raise ValueError(
            "No non-reference cohorts available for estimation"
            + (
                f" — control_cohort ({control_cohort_label}) matched every "
                "treated cohort, leaving nothing to compare against it."
                if control_cohort is not None
                else "."
            )
        )
    has_ref = bool(ref_mask.any())
    if not has_ref:
        raise ValueError(
            f"Reference group is empty (control_cohort={control_cohort_label})."
            if control_cohort is not None
            else f"Reference group is empty (control_group={control_group!r})."
        )

    cluster_col = cluster or i

    # Relative time (NaN for reference observations)
    df["_rel_time"] = np.where(df[g] > 0, df[t] - df[g], np.nan)

    if event_window is None:
        rel_obs = df.loc[df[g] > 0, "_rel_time"].dropna()
        e_min = int(rel_obs.min())
        e_max = int(rel_obs.max())
    else:
        e_min, e_max = int(event_window[0]), int(event_window[1])

    # Reference relative time = -1 (CS-SA standard).
    rel_times = [e for e in range(e_min, e_max + 1) if e != -1]

    # ----- Saturated design matrix: 1(G=g) × 1(e=ℓ) -----
    interact_meta: List[Tuple[int, int]] = []  # (g, e) per column
    X_cols: List[np.ndarray] = []
    for g_val in cohorts:
        in_cohort = (df[g] == g_val).values
        for e in rel_times:
            X_cols.append((in_cohort & (df["_rel_time"].values == e)).astype(float))
            interact_meta.append((g_val, e))

    X_int = np.column_stack(X_cols)

    # Unit + time FE via two-way within transformation ("within" projection).
    # Build the panel of y and X, demean, then flatten.
    unit_idx = pd.Categorical(df[i])
    time_idx = pd.Categorical(df[t])
    y_dm = _two_way_demean(df[y].values.astype(float), unit_idx, time_idx)
    X_dm = np.column_stack(
        [
            _two_way_demean(X_int[:, k], unit_idx, time_idx)
            for k in range(X_int.shape[1])
        ]
    )

    if covariates:
        for c in covariates:
            X_dm = np.column_stack(
                [
                    X_dm,
                    _two_way_demean(df[c].values.astype(float), unit_idx, time_idx),
                ]
            )

    valid = np.isfinite(y_dm) & np.all(np.isfinite(X_dm), axis=1)
    y_v = y_dm[valid]
    X_v = X_dm[valid]
    cluster_v = df.loc[valid, cluster_col].values
    k_int = len(interact_meta)

    # ----- OLS with ridge safety -----
    XtX = X_v.T @ X_v
    try:
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(X_v.shape[1]))
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X_v.T @ y_v)

    # ----- Cluster-robust sandwich SE (Liang-Zeger) -----
    u = y_v - X_v @ beta
    Xu = X_v * u[:, None]
    clusters = pd.Series(cluster_v)
    Xu_sum = np.zeros_like(XtX)
    for _, idx in clusters.groupby(clusters).indices.items():
        s = Xu[idx].sum(axis=0)
        Xu_sum += np.outer(s, s)
    n_clust = clusters.nunique()
    n, k = X_v.shape
    df_adj = (n_clust / max(n_clust - 1, 1)) * ((n - 1) / max(n - k, 1))
    V_beta = df_adj * XtX_inv @ Xu_sum @ XtX_inv

    # Slice to interaction block (drop covariate rows/cols for IW weights).
    V_int = V_beta[:k_int, :k_int]
    beta_int = beta[:k_int]

    # ----- IW aggregation at each relative time -----
    unit_cohorts = df.groupby(i)[g].first()
    cohort_counts = unit_cohorts[unit_cohorts > 0].value_counts()
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Observation counts per relative time over the *estimated* cohorts
    # only — eventstudyinteract restricts the share regression to
    # `control_cohort == 0`, so reference units must not inflate N_ℓ.
    _est_rows = df[df[g].isin(cohorts)]
    n_obs_at_rel = _est_rows["_rel_time"].value_counts().to_dict()

    es_rows = []
    combos: Dict[int, Tuple[np.ndarray, float]] = {}
    for e in sorted(set(rel_times)):
        eligible = [
            g_val
            for g_val in cohorts
            if (g_val, e) in {m for m in interact_meta} and (g_val + e) in time_periods
        ]
        if not eligible:
            continue

        shares = np.array(
            [cohort_counts.get(g_val, 0) for g_val in eligible], dtype=float
        )
        if shares.sum() <= 0:
            continue
        shares = shares / shares.sum()

        # Selection vector w of length k_int picking out (g, e) positions.
        w = np.zeros(k_int)
        for share, g_val in zip(shares, eligible):
            idx = interact_meta.index((g_val, e))
            w[idx] = share

        est_e = float(w @ beta_int)

        # Var(δ̂_ℓ) has TWO terms, because δ̂_ℓ = Σ_g ŵ_{g,ℓ} β̂_{g,ℓ} is a
        # product of two estimated objects (SA 2021, Prop. 3):
        #
        #   (1) w' Var(β̂) w          — the interacted-regression term
        #   (2) β' Var(ŵ) β          — the cohort-share estimation term
        #
        # Stata's eventstudyinteract carries both and treats them as
        # independent (no cross term). Term (2) vanishes when a single
        # cohort is eligible, since then ŵ ≡ 1 is degenerate — which is
        # exactly why omitting it looked harmless: on mpdta the SEs agreed
        # to 0.02% at single-cohort relative times and drifted up to 2%
        # wherever two cohorts contributed, always downward.
        beta_e = np.array(
            [beta_int[interact_meta.index((g_val, e))] for g_val in eligible],
            dtype=float,
        )
        var_share = _cohort_share_vcov(shares, n_obs_at_rel.get(e, 0))
        var_e = float(w @ V_int @ w) + float(beta_e @ var_share @ beta_e)
        se_e = float(np.sqrt(max(var_e, 0.0)))
        pval = float(2 * (1 - stats.norm.cdf(abs(est_e / se_e)))) if se_e > 0 else 1.0

        # Keep the linear combination and the share-variance block so the
        # pre-trend test below can build the JOINT covariance rather than
        # pretending the event-time estimates are independent.
        combos[e] = (w, float(beta_e @ var_share @ beta_e))

        es_rows.append(
            {
                "relative_time": e,
                "att": est_e,
                "se": se_e,
                "ci_lower": est_e - z_crit * se_e,
                "ci_upper": est_e + z_crit * se_e,
                "pvalue": pval,
                "n_cohorts": len(eligible),
            }
        )

    event_study = pd.DataFrame(es_rows)

    pretrend = _sunab_pretrend_test(
        event_study, combos, V_int, pretest=pretest, pretest_periods=pretest_periods
    )

    # ----- Overall post-treatment ATT via single linear combinations -----
    post = event_study[event_study["relative_time"] >= 0]
    summary_stats = {
        "event_time": (0.0, np.inf, np.zeros(k_int)),
        "fixest_att": (0.0, np.inf, np.zeros(k_int)),
    }
    if len(post) > 0:
        # Historical StatsPAI summary: equal-weight each post relative-time
        # IW coefficient after cohort-share aggregation within that event time.
        W_event = np.zeros(k_int)
        event_total = 0.0
        for e in post["relative_time"]:
            eligible = [
                g_val
                for g_val in cohorts
                if (g_val, e) in set(interact_meta) and (g_val + e) in time_periods
            ]
            if not eligible:
                continue
            shares = np.array(
                [cohort_counts.get(g_val, 0) for g_val in eligible],
                dtype=float,
            )
            if shares.sum() <= 0:
                continue
            shares = shares / shares.sum()
            for share, g_val in zip(shares, eligible):
                W_event[interact_meta.index((g_val, e))] += share
            event_total += 1.0
        if event_total > 0:
            W_event /= event_total
        att_event = float(W_event @ beta_int)
        se_event = float(np.sqrt(max(W_event @ V_int @ W_event, 0.0)))
        summary_stats["event_time"] = (att_event, se_event, W_event)

        # fixest::summary(..., agg='att') convention: weight every observed
        # post-treatment cohort-time cell by treated cohort size.
        W_fixest = np.zeros(k_int)
        cell_total = 0.0
        for e in post["relative_time"]:
            eligible = [
                g_val
                for g_val in cohorts
                if (g_val, e) in set(interact_meta) and (g_val + e) in time_periods
            ]
            for g_val in eligible:
                count = float(cohort_counts.get(g_val, 0))
                if count <= 0:
                    continue
                W_fixest[interact_meta.index((g_val, e))] += count
                cell_total += count
        if cell_total > 0:
            W_fixest /= cell_total
        att_fixest = float(W_fixest @ beta_int)
        se_fixest = float(np.sqrt(max(W_fixest @ V_int @ W_fixest, 0.0)))
        summary_stats["fixest_att"] = (att_fixest, se_fixest, W_fixest)

    att, se_att, _ = summary_stats[aggregation_key]

    z = att / se_att if se_att > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))
    ci = (att - z_crit * se_att, att + z_crit * se_att)

    model_info = {
        "estimator": "Sun-Abraham IW",
        "control_group": control_group,
        "control_cohort": control_cohort_label,
        "pretrend_test": pretrend,
        "pretest": pretest,
        "pretest_periods": pretest_periods,
        "event_window": (e_min, e_max),
        "n_cohorts": len(cohorts),
        "cohorts": cohorts,
        "event_study": event_study,
        "summary_aggregation": aggregation_key,
        "att_event_time": float(summary_stats["event_time"][0]),
        "se_event_time": float(summary_stats["event_time"][1]),
        "att_fixest_att": float(summary_stats["fixest_att"][0]),
        "se_fixest_att": float(summary_stats["fixest_att"][1]),
        "se_type": f"cluster-robust on {cluster_col}",
        "n_clusters": int(n_clust),
        "n_coeffs": int(k_int),
    }

    _result = CausalResult(
        method="Sun and Abraham (2021)",
        estimand="ATT",
        estimate=att,
        se=se_att,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=event_study,
        model_info=model_info,
        _citation_key="sun_abraham",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.sun_abraham",
            params={
                "y": y,
                "g": g,
                "t": t,
                "i": i,
                "event_window": list(event_window) if event_window else None,
                "control_group": control_group,
                "covariates": list(covariates) if covariates else None,
                "cluster": cluster,
                "aggregation": aggregation,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ======================================================================
# Helpers
# ======================================================================


def _two_way_demean(
    x: np.ndarray,
    unit_idx: pd.Categorical,
    time_idx: pd.Categorical,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> np.ndarray:
    """Iterative within-transformation for unbalanced two-way FE.

    Falls back to the identity transformation on a single-unit / single-period
    sample.  Converges in a handful of passes on well-behaved panels.
    """
    x = x.astype(float).copy()
    n_units = len(unit_idx.categories)
    n_times = len(time_idx.categories)
    if n_units <= 1 or n_times <= 1:
        return np.asarray(x - np.nanmean(x), dtype=float)

    u_codes = unit_idx.codes
    t_codes = time_idx.codes

    for _ in range(max_iter):
        u_count = np.bincount(u_codes, minlength=n_units).clip(min=1)
        u_mean = np.bincount(u_codes, weights=x, minlength=n_units) / u_count
        x = x - u_mean[u_codes]
        t_count = np.bincount(t_codes, minlength=n_times).clip(min=1)
        t_mean = np.bincount(t_codes, weights=x, minlength=n_times) / t_count
        x = x - t_mean[t_codes]
        if np.nanmax(np.abs(u_mean)) < tol and np.nanmax(np.abs(t_mean)) < tol:
            break
    return np.asarray(x, dtype=float)


# ----------------------------------------------------------------------
# Citation (redundant-safe registration)
# ----------------------------------------------------------------------
CausalResult._CITATIONS["sun_abraham"] = (
    "@article{sun2021estimating,\n"
    "  title={Estimating Dynamic Treatment Effects in Event Studies "
    "with Heterogeneous Treatment Effects},\n"
    "  author={Sun, Liyang and Abraham, Sarah},\n"
    "  journal={Journal of Econometrics},\n"
    "  volume={225},\n"
    "  number={2},\n"
    "  pages={175--199},\n"
    "  year={2021},\n"
    "  publisher={Elsevier}\n"
    "}"
)
