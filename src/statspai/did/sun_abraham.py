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

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility
from ._core import calendar_time_aware as _calendar_time_aware
from ._core import drop_unusable_rows as _drop_unusable_rows
from ._core import fe_dof_not_nested as _fe_dof_not_nested

# ======================================================================
# Public API
# ======================================================================


def _cohort_share_vcov_weighted(
    shares: np.ndarray,
    eligible: list,
    omega: np.ndarray,
    cohort_at_rows: np.ndarray,
) -> np.ndarray:
    """Weighted analogue of :func:`_cohort_share_vcov`.

    With observation weights the share of cohort ``g`` at relative time
    ``l`` is the omega-weighted mean of the cohort indicator over the rows
    at ``l``, and the robust sandwich of that weighted regression is

        Var(w_hat_l) = sum_i omega_i^2 u_i u_i' / (sum_i omega_i)^2,
        u_ig = 1{g_i = g} - w_hat_g,

    with ``omega`` rescaled to mean one over the estimation sample. This is
    exactly Stata ``eventstudyinteract``'s ``regress ... [aw] ; avar ...,
    robust`` construction (``Sxxi S Sxxi / N`` with ``Sxx = X'WX/N``); at
    omega == 1 it collapses to the multinomial form.
    """
    u = np.column_stack(
        [
            (cohort_at_rows == g_val).astype(float) - share
            for share, g_val in zip(shares, eligible)
        ]
    )
    uw = u * omega[:, None]
    return np.asarray((uw.T @ uw) / float(omega.sum()) ** 2, dtype=float)


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


def _joint_event_time_vcov(
    combos: "dict[int, Tuple[np.ndarray, float]]", v_int: np.ndarray
) -> np.ndarray:
    """``Cov(δ̂_ℓ, δ̂_m) = w_ℓ' Var(β̂) w_m`` plus the share term on the diagonal.

    This is the matrix Stata ``eventstudyinteract`` stores as ``e(V_iw)``:
    its ``avar`` on mutually exclusive relative-time dummies has no
    cross-relative-time share covariance, so the share term is diagonal.
    """
    es = sorted(combos)
    k = len(es)
    cov = np.empty((k, k), dtype=float)
    for a, ea in enumerate(es):
        wa, share_a = combos[ea]
        for b, eb in enumerate(es):
            wb, _ = combos[eb]
            cov[a, b] = float(wa @ v_int @ wb)
        cov[a, a] += share_a
    return cov


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


@accepts_aliases(
    _strict=True,
    id="i",
    unit="i",
    time="t",
    first_treat="g",
    cohort="g",
    controls="covariates",
)
@_calendar_time_aware(time="t", cohort="g")
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
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    aggregation: str = "event_time",
    share_variance: bool = True,
    alpha: float = 0.05,
    pretest: str = "joint",
    pretest_periods: Optional[int] = None,
    window_rule: str = "report",
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
        Default: observed range in the data. What happens to the treated
        observations outside the window is set by ``window_rule``.
    window_rule : {'report', 'bin', 'reference'}, default 'report'
        Only matters when ``event_window`` leaves out relative times that
        are in the data.

        - ``'report'``: the regression is saturated in every observed
          relative time and the window only selects what is reported and
          aggregated. The estimates inside the window are those of the
          unrestricted fit.
        - ``'bin'``: relative times at or beyond an end of the window are
          pooled into that end, so the first and last coefficients are
          "``e_min`` or earlier" and "``e_max`` or later". This is the
          specification one writes for Stata ``eventstudyinteract`` with
          binned end dummies.
        - ``'reference'``: the left-out relative times join the omitted
          category, next to period -1 and the control cohort. Every
          coefficient is then measured against a reference that contains
          treated observations, and is biased by the effects they carry
          unless those are zero. Kept to reproduce earlier output.

        .. versionchanged:: 1.39.0
           ⚠️ The default was what is now ``'reference'``. On the Wolfers
           (2006) divorce panel, restricting the window to (-14, 12) moved
           the effect at relative time 0 from 0.246 to 0.502, because
           relative times 13 to 19, where the effect is negative, had
           become part of the reference.
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
    share_variance : bool, default True
        Whether Var(δ̂_ℓ) carries the cohort-share estimation term
        ``β' Var(ŵ) β`` of Sun & Abraham (2021, Prop. 3). ``True`` is the
        authors' own Stata ``eventstudyinteract`` convention and is what
        the Stata parity module pins at machine precision. ``False``
        treats the interaction weights as fixed, which is what
        ``fixest::sunab`` does; the two coincide wherever a single cohort
        is eligible at a relative time and differ by the (positive
        semi-definite) share term wherever several are. Point estimates
        do not depend on this switch.

        Degrees of freedom follow the ``fixest`` / ``reghdfe`` "nested"
        rule on both settings: the small-sample factor is
        ``(N-1)/(N-K) * G/(G-1)`` with ``K`` counting the estimated
        cohort-by-relative-time cells, the covariates, and the levels of
        every fixed effect that is *not* nested in the cluster variable
        (with the default ``cluster=i`` that is the time effects only).
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
    if window_rule not in ("report", "bin", "reference"):
        raise MethodIncompatibility(
            "window_rule must be 'report', 'bin' or 'reference', "
            f"got {window_rule!r}",
            diagnostics={"window_rule": repr(window_rule)},
        )

    df[g] = df[g].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
    time_periods = sorted(df[t].unique())
    time_set = set(time_periods)
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

    rel_obs = df.loc[df[g] > 0, "_rel_time"].dropna()
    obs_min, obs_max = int(rel_obs.min()), int(rel_obs.max())
    if event_window is None:
        e_min, e_max = obs_min, obs_max
    else:
        e_min, e_max = int(event_window[0]), int(event_window[1])
        if e_min > e_max:
            raise MethodIncompatibility(
                f"event_window={event_window!r} has its bounds reversed.",
                diagnostics={"event_window": repr(event_window)},
            )

    # Reference relative time = -1 (CS-SA standard).
    binned = (
        window_rule == "bin"
        and event_window is not None
        and (obs_min < e_min or obs_max > e_max)
    )
    if binned:
        if e_min > -2 or e_max < 0:
            raise MethodIncompatibility(
                "window_rule='bin' needs a window that keeps the reference "
                "period out of the end bins: event_window[0] <= -2 and "
                f"event_window[1] >= 0, got {event_window!r}.",
            )
        # "e_min or earlier" and "e_max or later"
        df["_rel_time"] = df["_rel_time"].clip(lower=e_min, upper=e_max)
        rel_times = [e for e in range(e_min, e_max + 1) if e != -1]
    elif window_rule == "report":
        # Saturate in every observed relative time; the window selects what
        # is reported. Leaving relative times out of the regression would
        # put their treated observations into the reference.
        rel_times = [e for e in range(obs_min, obs_max + 1) if e != -1]
    else:
        rel_times = [e for e in range(e_min, e_max + 1) if e != -1]

    # ----- Saturated design: 1(G=g) × 1(e=ℓ), one column per observed cell -----
    # Only cells that exist in the data are parameters. A cohort first
    # treated late has no observations at the most negative relative times
    # (and vice versa); an all-zero column is not an estimated coefficient,
    # and counting it in K would inflate the small-sample degrees-of-freedom
    # factor relative to fixest and reghdfe, which only ever see the
    # observed cells. The design is kept as a cell index per row (one-hot):
    # with ~100 cohorts the dense columns cost ~35 s and several GB
    # (``_sa_within``).
    g_vals = df[g].to_numpy()
    rel_vals = df["_rel_time"].to_numpy()
    rel_set = set(rel_times)
    cohort_set = set(cohorts)
    present = (
        pd.DataFrame({"g": g_vals, "e": rel_vals})[
            pd.Series(g_vals).isin(cohort_set).to_numpy()
            & pd.Series(rel_vals).isin(rel_set).to_numpy()
        ]
        .drop_duplicates()
        .itertuples(index=False)
    )
    present_cells = {(pg, int(pe)) for pg, pe in present}
    interact_meta: List[Tuple[int, int]] = [
        (g_val, e)
        for g_val in cohorts
        for e in rel_times
        if (g_val, e) in present_cells
    ]
    cell_index = {m: j for j, m in enumerate(interact_meta)}
    keys = pd.DataFrame(
        {
            "_g": [float(m[0]) for m in interact_meta],
            "_e": [float(m[1]) for m in interact_meta],
            "_j": np.arange(len(interact_meta)),
        }
    )
    rows = pd.DataFrame(
        {"_g": g_vals.astype(float), "_e": rel_vals.astype(float)}
    ).merge(keys, on=["_g", "_e"], how="left")
    cell = rows["_j"].fillna(-1).to_numpy(dtype=np.int64)

    unit_idx = pd.Categorical(df[i])
    time_idx = pd.Categorical(df[t])
    # Observation weights omega. Unit weights are constant within unit, but
    # the projection and the solve both work at observation level.
    w_all = (
        np.ones(len(df), dtype=float)
        if weights is None
        else df[weights].to_numpy(dtype=float)
    )
    k_int = len(interact_meta)
    y_raw = df[y].to_numpy(dtype=float)
    Z_raw = (
        np.column_stack([df[c].to_numpy(dtype=float) for c in covariates])
        if covariates
        else None
    )
    cluster_codes = pd.factorize(df[cluster_col])[0]
    if not (
        np.all(np.isfinite(y_raw))
        and (Z_raw is None or np.all(np.isfinite(Z_raw)))
        and np.all(cluster_codes >= 0)
    ):
        raise MethodIncompatibility(
            "sun_abraham: the outcome, covariates and cluster variable must "
            "be non-missing on the estimation sample; drop incomplete rows "
            "first.",
        )
    from ._sa_within import within_fit

    fit = within_fit(
        y_raw,
        cell,
        k_int,
        Z_raw,
        unit_idx.codes.astype(np.int64),
        time_idx.codes.astype(np.int64),
        len(unit_idx.categories),
        len(time_idx.categories),
        w_all,
        cluster_codes,
    )
    beta, XtX_inv, Xu_sum = fit["beta"], fit["XtX_inv"], fit["meat"]
    n, k = int(fit["n"]), int(fit["k"])
    valid = np.ones(len(df), dtype=bool)
    n_clust = int(cluster_codes.max()) + 1
    # Small-sample factor: fixest / reghdfe "nested" convention. K counts
    # the slope parameters plus the levels of every fixed effect that is
    # not nested inside the cluster variable; a fixed effect nested in the
    # cluster (unit FE under cluster=unit) contributes nothing, because
    # the cluster-robust meat already absorbs it. With several non-nested
    # fixed effects, one level per additional effect is collinear and is
    # not counted (fixest's "levels minus (Q - 1)" rule).
    k_fe = _fe_dof_not_nested(df.loc[valid], [i, t], cluster_col)
    K = k + k_fe
    df_adj = (n_clust / max(n_clust - 1, 1)) * ((n - 1) / max(n - K, 1))
    V_beta = df_adj * XtX_inv @ Xu_sum @ XtX_inv

    # Slice to interaction block (drop covariate rows/cols for IW weights).
    V_int = V_beta[:k_int, :k_int]
    beta_int = beta[:k_int]

    # ----- IW aggregation at each relative time -----
    # Interaction weights: the share of each cohort among the observations
    # at relative time l (their omega-mass under weights), as in SA (2021)
    # and eventstudyinteract's share regression. On a balanced panel this is
    # the cohort's share of units; where rows are missing, or relative times
    # are pooled into an end bin, only the observation count is right.
    _cell_rows = df[df[g].isin(cohorts) & df["_rel_time"].notna()]
    _cell_w = (
        pd.Series(1.0, index=_cell_rows.index)
        if weights is None
        else _cell_rows[weights].astype(float)
    )
    cell_mass: Dict[Tuple[Any, int], float] = {
        (gv, int(ev)): float(m)
        for (gv, ev), m in _cell_w.groupby([_cell_rows[g], _cell_rows["_rel_time"]])
        .sum()
        .items()
    }
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Observation counts per relative time over the *estimated* cohorts
    # only — eventstudyinteract restricts the share regression to
    # `control_cohort == 0`, so reference units must not inflate N_ℓ.
    _est_rows = df[df[g].isin(cohorts)]
    n_obs_at_rel = _est_rows["_rel_time"].value_counts().to_dict()
    # Under omega the share regression is weighted least squares and its
    # robust sandwich is built from omega_i^2 u_i u_i' with the weights
    # normalised to mean one over the estimation sample, which is what
    # eventstudyinteract's ``regress ... [aw]`` + ``avar ... robust``
    # computes (Stata rescales aweights to sum to N). Keep the per-row
    # omega and cohort at every relative time for that construction.
    _omega_rows: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if weights is not None:
        _w_est = _est_rows[weights].to_numpy(dtype=float)
        _w_est = _w_est / _w_est.mean()
        _g_est = _est_rows[g].to_numpy()
        _rel_est = _est_rows["_rel_time"].to_numpy()
        for _e in np.unique(_rel_est):
            _m = _rel_est == _e
            _omega_rows[int(_e)] = (_w_est[_m], _g_est[_m])

    es_rows = []
    combos: Dict[int, Tuple[np.ndarray, float]] = {}
    for e in sorted(set(rel_times)):
        if not e_min <= e <= e_max:
            continue  # window_rule='report': estimated, not reported
        eligible = [
            g_val
            for g_val in cohorts
            if (g_val, e) in cell_index
            and ((g_val + e) in time_set or (binned and e in (e_min, e_max)))
        ]
        if not eligible:
            continue

        shares = np.array(
            [cell_mass.get((g_val, e), 0.0) for g_val in eligible], dtype=float
        )
        if shares.sum() <= 0:
            continue
        shares = shares / shares.sum()

        # Selection vector w of length k_int picking out (g, e) positions.
        w = np.zeros(k_int)
        for share, g_val in zip(shares, eligible):
            idx = cell_index[(g_val, e)]
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
            [beta_int[cell_index[(g_val, e)]] for g_val in eligible],
            dtype=float,
        )
        if not share_variance:
            var_share = np.zeros((len(eligible), len(eligible)))
        elif weights is None:
            var_share = _cohort_share_vcov(shares, n_obs_at_rel.get(e, 0))
        else:
            var_share = _cohort_share_vcov_weighted(
                shares, eligible, *_omega_rows[int(e)]
            )
        var_e = float(w @ V_int @ w) + float(beta_e @ var_share @ beta_e)
        se_e = float(np.sqrt(max(var_e, 0.0)))
        pval = float(2 * stats.norm.sf(abs(est_e / se_e))) if se_e > 0 else 1.0

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
                if (g_val, e) in cell_index
                and ((g_val + e) in time_set or (binned and e in (e_min, e_max)))
            ]
            if not eligible:
                continue
            shares = np.array(
                [cell_mass.get((g_val, e), 0.0) for g_val in eligible],
                dtype=float,
            )
            if shares.sum() <= 0:
                continue
            shares = shares / shares.sum()
            for share, g_val in zip(shares, eligible):
                W_event[cell_index[(g_val, e)]] += share
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
                if (g_val, e) in cell_index
                and ((g_val + e) in time_set or (binned and e in (e_min, e_max)))
            ]
            for g_val in eligible:
                count = float(cell_mass.get((g_val, e), 0.0))
                if count <= 0:
                    continue
                W_fixest[cell_index[(g_val, e)]] += count
                cell_total += count
        if cell_total > 0:
            W_fixest /= cell_total
        att_fixest = float(W_fixest @ beta_int)
        se_fixest = float(np.sqrt(max(W_fixest @ V_int @ W_fixest, 0.0)))
        summary_stats["fixest_att"] = (att_fixest, se_fixest, W_fixest)

        # The same aggregate with the cohort-share estimation term carried,
        # which is what a ``lincom`` on Stata ``eventstudyinteract``'s
        # ``e(V_iw)`` returns.  The share terms of different relative times
        # are independent (the share regression's dummies are mutually
        # exclusive), so the aggregate's extra variance is the sum of
        # omega_e^2 times each relative time's share term, with omega_e the
        # weight the aggregate puts on relative time e.
        meta_set = set(interact_meta)
        share_extra = 0.0
        for e in post["relative_time"]:
            if e not in combos:
                continue
            omega_e = float(
                sum(
                    W_fixest[cell_index[(g_val, e)]]
                    for g_val in cohorts
                    if (g_val, e) in meta_set
                )
            )
            share_extra += omega_e**2 * combos[e][1]
        se_fixest_share = float(
            np.sqrt(max(W_fixest @ V_int @ W_fixest + share_extra, 0.0))
        )
        summary_stats["fixest_att_share"] = (att_fixest, se_fixest_share, W_fixest)

    att, se_att, _ = summary_stats[aggregation_key]

    z = att / se_att if se_att > 0 else 0.0
    pvalue = float(2 * stats.norm.sf(abs(z)))
    ci = (att - z_crit * se_att, att + z_crit * se_att)

    model_info = {
        "estimator": "Sun-Abraham IW",
        "control_group": control_group,
        "control_cohort": control_cohort_label,
        "pretrend_test": pretrend,
        "pretest": pretest,
        "pretest_periods": pretest_periods,
        "event_window": (e_min, e_max),
        "window_rule": window_rule,
        "n_cohorts": len(cohorts),
        "cohorts": cohorts,
        "event_study": event_study,
        "summary_aggregation": aggregation_key,
        "att_event_time": float(summary_stats["event_time"][0]),
        "se_event_time": float(summary_stats["event_time"][1]),
        "att_fixest_att": float(summary_stats["fixest_att"][0]),
        "se_fixest_att": float(summary_stats["fixest_att"][1]),
        # eventstudyinteract e(V_iw) convention (cohort-share term carried in the
        # aggregate); equals se_fixest_att when share_variance=False
        "se_fixest_att_share": float(
            summary_stats.get("fixest_att_share", summary_stats["fixest_att"])[1]
        ),
        # joint covariance of the IW event-time coefficients: regression cross
        # terms plus the share term on the diagonal (the matrix e(V_iw))
        "vcov_event_time": _joint_event_time_vcov(combos, V_int),
        "event_times": sorted(combos),
        "se_type": f"cluster-robust on {cluster_col}",
        "n_clusters": int(n_clust),
        "n_coeffs": int(k_int),
        "share_variance": bool(share_variance),
        "dof_K": int(K),
        "dof_fe_not_nested": int(k_fe),
        "dof_convention": "fixest/reghdfe nested: (N-1)/(N-K) * G/(G-1)",
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
                "window_rule": window_rule,
                "control_group": control_group,
                "covariates": list(covariates) if covariates else None,
                "cluster": cluster,
                "aggregation": aggregation,
                "share_variance": share_variance,
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
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Iterative within-transformation for unbalanced two-way FE.

    Falls back to the identity transformation on a single-unit / single-period
    sample.  Converges in a handful of passes on well-behaved panels.

    ``w`` carries observation weights. The weighted projection subtracts
    the *weighted* group mean at each pass, which is what makes the
    residual orthogonal to the fixed effects **in the weighted inner
    product** -- the orthogonality WLS actually needs. Demeaning by the
    unweighted mean and then running WLS would leave the fixed effects
    correlated with the regressors and bias the coefficients, so the
    weight has to enter the projection, not just the final solve.
    ``w=None`` reproduces the unweighted path exactly.
    """
    x = x.astype(float).copy()
    n_units = len(unit_idx.categories)
    n_times = len(time_idx.categories)
    if n_units <= 1 or n_times <= 1:
        return np.asarray(x - np.nanmean(x), dtype=float)

    u_codes = unit_idx.codes
    t_codes = time_idx.codes
    weighted = w is not None and not np.allclose(w, 1.0)
    if weighted:
        w = np.asarray(w, dtype=float)
        u_wsum = np.bincount(u_codes, weights=w, minlength=n_units)
        u_wsum = np.where(u_wsum > 0, u_wsum, 1.0)
        t_wsum = np.bincount(t_codes, weights=w, minlength=n_times)
        t_wsum = np.where(t_wsum > 0, t_wsum, 1.0)

    for _ in range(max_iter):
        if weighted:
            u_mean = np.bincount(u_codes, weights=w * x, minlength=n_units) / u_wsum
            x = x - u_mean[u_codes]
            t_mean = np.bincount(t_codes, weights=w * x, minlength=n_times) / t_wsum
            x = x - t_mean[t_codes]
        else:
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
