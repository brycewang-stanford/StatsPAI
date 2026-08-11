"""
Callaway & Sant'Anna (2021) estimator for staggered DID.

Estimates group-time average treatment effects ATT(g,t) under staggered
treatment adoption, with proper handling of heterogeneous treatment effects
that invalidate standard TWFE estimators.

Supports three estimation approaches:
- Doubly Robust (DR) — default, combines outcome regression and IPW
- Inverse Probability Weighting (IPW)
- Outcome Regression (REG)

References
----------
Callaway, B. and Sant'Anna, P.H.C. (2021).
"Difference-in-Differences with Multiple Time Periods."
*Journal of Econometrics*, 225(2), 200-230. [@callaway2021difference]

Sant'Anna, P.H.C. and Zhao, J. (2020).
"Doubly Robust Difference-in-Differences Estimators."
*Journal of Econometrics*, 219(1), 101-122.
"""

from __future__ import annotations

import warnings
from numbers import Real
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._core import cohort_share_context as _cohort_share_context
from ._core import drop_unusable_rows as _drop_unusable_rows
from ._core import multiplier_bootstrap as _core_multiplier_bootstrap
from ._core import normalize_se_method as _normalize_se_method
from ._core import parallel_trends_block as _pt_block
from ._core import require_bool as _require_bool
from ._core import weight_influence as _weight_influence


class CallawayNotImplemented(MethodIncompatibility, NotImplementedError):
    """Unsupported CS branch that preserves historical NotImplementedError."""


#: Propensity-score trimming cutoff for *control* units, matching
#: ``DRDID``'s ``trim.level`` — the value R ``did`` and Stata ``csdid``
#: both inherit without overriding it.
_DEFAULT_PSCORE_TRIM = 0.995

#: ATT(g,t) estimators, mapped to the names the reference packages use.
#: ``'ipw'``/``'stdipw'`` are the SAME estimator: R ``did``'s
#: ``est_method='ipw'`` dispatches to ``DRDID::std_ipw_did_panel``, which is
#: Stata's ``method(stdipw)``. Stata's ``method(ipw)`` is Abadie (2005) and
#: is reached here as ``'ipw_abadie'``.
_ESTIMATORS = ("dr", "ipw", "stdipw", "ipw_abadie", "reg")


def _require_dataframe(data: Any, *, function: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            f"`data` must be a pandas DataFrame, got {type(data).__name__}.",
            recovery_hint=f"Pass `{function}` data as a pandas DataFrame.",
            diagnostics={"function": function, "type": type(data).__name__},
        )
    if data.empty:
        raise DataInsufficient(
            "`data` must contain at least one row.",
            recovery_hint=f"Provide non-empty data before calling `{function}`.",
            diagnostics={"function": function, "n_rows": 0},
        )
    return data


def _require_column_name(name: Any, *, argument: str) -> str:
    if not isinstance(name, str) or not name:
        raise MethodIncompatibility(
            f"`{argument}` must be a non-empty column name string.",
            recovery_hint=(
                f"Pass the name of an existing DataFrame column for `{argument}`."
            ),
            diagnostics={"argument": argument, "type": type(name).__name__},
        )
    return name


def _coerce_optional_columns(
    columns: Optional[Sequence[str] | str],
    *,
    argument: str,
) -> Optional[List[str]]:
    if columns is None:
        return None
    if isinstance(columns, str):
        out = [columns]
    else:
        try:
            out = list(columns)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"`{argument}` must be a column name or sequence of column names.",
                recovery_hint=(f"Pass `{argument}` as 'x' or ['x1', 'x2']."),
                diagnostics={"argument": argument, "type": type(columns).__name__},
            ) from exc
    return [_require_column_name(col, argument=argument) for col in out]


def _require_string_option(value: Any, *, argument: str, valid: Sequence[str]) -> str:
    if not isinstance(value, str):
        raise MethodIncompatibility(
            f"`{argument}` must be one of {tuple(valid)}, got "
            f"{type(value).__name__}.",
            recovery_hint=f"Pass a supported string value for `{argument}`.",
            diagnostics={"argument": argument, "type": type(value).__name__},
        )
    if value not in valid:
        raise MethodIncompatibility(
            f"{argument} must be one of {tuple(valid)}, got {value!r}",
            recovery_hint=f"Choose a supported `{argument}` value.",
            diagnostics={"argument": argument, "value": value, "valid": list(valid)},
        )
    return value


def _require_alpha(alpha: Any) -> float:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise MethodIncompatibility(
            "`alpha` must be a finite number in (0, 1).",
            recovery_hint="Pass a significance level such as alpha=0.05.",
            diagnostics={"argument": "alpha", "value": alpha},
        )
    out = float(alpha)
    if not np.isfinite(out) or not (0.0 < out < 1.0):
        raise MethodIncompatibility(
            "`alpha` must be a finite number in (0, 1).",
            recovery_hint="Pass a significance level such as alpha=0.05.",
            diagnostics={"argument": "alpha", "value": alpha},
        )
    return out


def _require_int_at_least(value: Any, *, argument: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"{argument} must be an integer >= {minimum}.",
            recovery_hint=f"Pass an integer value for `{argument}`.",
            diagnostics={"argument": argument, "value": value},
        )
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{argument} must be an integer >= {minimum}.",
            recovery_hint=f"Pass an integer value for `{argument}`.",
            diagnostics={"argument": argument, "type": type(value).__name__},
        ) from exc
    if out < minimum:
        raise MethodIncompatibility(
            f"{argument} must be an integer >= {minimum}.",
            recovery_hint=f"Increase `{argument}` to at least {minimum}.",
            diagnostics={"argument": argument, "value": value},
        )
    return out


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    function: str,
) -> None:
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"Column(s)/Covariate(s) not found in data: {missing}",
            recovery_hint=f"Check column names before calling `{function}`.",
            diagnostics={
                "function": function,
                "missing_columns": missing,
                "available_columns": list(data.columns),
            },
        )


# ======================================================================
# Public API
# ======================================================================


def callaway_santanna(
    data: pd.DataFrame,
    y: str,
    g: str,
    t: str,
    i: str,
    x: Optional[List[str]] = None,
    weights: Optional[str] = None,
    estimator: str = "dr",
    control_group: str = "nevertreated",
    notyet_cutoff: str = "period",
    pscore_trim: float = _DEFAULT_PSCORE_TRIM,
    base_period: str = "universal",
    anticipation: int = 0,
    alpha: float = 0.05,
    panel: bool = True,
    allow_unbalanced_panel: bool = False,
    clustervars: Optional[Sequence[str] | str] = None,
    bstrap: bool = False,
    biters: int = 1000,
    cband: bool = False,
    boot_weight_type: str = "rademacher",
    random_state: Optional[int] = None,
    pretest: str = "joint",
    pretest_periods: Optional[int] = None,
    se_method: Optional[str] = None,
) -> CausalResult:
    """
    Callaway & Sant'Anna (2021) estimator for staggered DID.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    y : str
        Outcome variable name.
    g : str
        Group variable: first period of treatment (0 = never treated).
    t : str
        Time period variable.
    i : str
        Unit identifier variable.
    x : list of str, optional
        Covariate names for conditional parallel trends.
    weights : str, optional
        Column holding **unit-level sampling / population weights** ω
        (R ``did::att_gt(weightsname=)``, Stata ``csdid [aweight=]``).

        Weights are *not* a precision knob. They enter the definition of
        the target parameter itself: the unweighted ATT answers "what was
        the average effect across treated **counties**", the
        population-weighted ATT answers "across treated **people**".
        These are different estimands and can differ in sign
        [@baker2026difference, §3.1; @solon2015weighting]. Comparing a
        weighted with an unweighted estimate is therefore *not* a
        robustness check.

        Must be constant within unit; a time-varying column raises
        :class:`~statspai.exceptions.MethodIncompatibility`. Weights are
        renormalised to mean 1 over the estimation sample, matching
        ``DRDID``/``did``, so the reported ATT is scale-invariant.
    estimator : str, default 'dr'
        Estimation method, one of:

        - ``'dr'`` — doubly robust (Sant'Anna & Zhao 2020).
        - ``'ipw'`` / ``'stdipw'`` — **the same estimator**: stabilized
          (Hájek-normalized) IPW, where the control weights are divided by
          their own mean. This is what R ``did``'s ``est_method='ipw'``
          dispatches to (``DRDID::std_ipw_did_panel``) and what Stata
          ``csdid`` calls ``method(stdipw)``.
        - ``'ipw_abadie'`` — Abadie (2005) IPW, where both arms share the
          single denominator E[D] and the control weights need not average
          to one. This is Stata ``csdid``'s ``method(ipw)``.
        - ``'reg'`` — outcome regression.

        .. warning::
           ``estimator='ipw'`` here is **not** Stata ``csdid``'s
           ``method(ipw)``. StatsPAI follows R ``did``'s naming, in which
           ``'ipw'`` means the stabilized estimator. Migrating a
           ``method(ipw)`` script means writing ``estimator='ipw_abadie'``;
           the two differ by O(1e-4) on ``mpdta``.
    control_group : str, default 'nevertreated'
        Comparison group: 'nevertreated' or 'notyettreated'.
    notyet_cutoff : str, default 'period'
        Only consulted when ``control_group='notyettreated'``. Which date
        must a unit still be untreated at to serve as a control for
        ATT(g, t)?

        - ``'period'`` — untreated as of ``t``. R ``did``, and Stata
          ``csdid, asinr``.
        - ``'cohort'`` — untreated as of ``g``, a strictly smaller control
          set that also excludes cohorts switching on between ``t`` and
          ``g``. Stata ``csdid``'s own default.

        The two coincide for post-treatment cells and differ only on
        pre-treatment placebos.
    pscore_trim : float, default 0.995
        Drop *control* units whose estimated propensity score is at or
        above this cutoff, matching ``DRDID``'s ``trim.level`` (inherited
        unchanged by both R ``did`` and Stata ``csdid``). A control with
        p(X) → 1 carries an exploding odds weight p/(1−p) and can dominate
        the control arm on its own. Pass ``1.0`` to disable. When trimming
        actually binds it is recorded in ``diagnostics['n_pscore_trimmed']``
        rather than applied silently.
    base_period : str, default 'universal'
        Base period: 'universal' (always g-1) or 'varying'.

        Maps onto Stata ``csdid``'s pre-treatment gap options:
        ``'universal'`` is ``long2``, ``'varying'`` is the csdid default
        (short gaps). ``csdid``'s ``long`` is ``long2`` with the sign of
        the pre-treatment cells flipped, which StatsPAI does not mirror —
        its pre-treatment cells always carry the ``long2`` sign.
    anticipation : int, default 0
        Number of pre-treatment periods over which units may anticipate
        the treatment.  The cohort base period moves back from ``g − 1``
        to ``g − 1 − anticipation``, and a cohort with no period
        satisfying ``t + anticipation < g`` is dropped with a warning.
        See Callaway & Sant'Anna (2021), Section 3.2.

        Following R ``did``, the shift applies to *post-treatment* cells
        (and to every cell under ``base_period='universal'``). A
        pre-treatment placebo under ``base_period='varying'`` keeps the
        period immediately before it as its base, so its value does not
        move when ``anticipation`` changes.

        .. versionchanged:: 1.23.0
           ``base_period='varying'`` previously shifted the pre-treatment
           base by ``anticipation`` too, which moved the placebo cells off
           R ``did`` / Stata ``csdid`` and dropped the earliest ones.
           Post-treatment ATT(g, t) are unaffected.
    alpha : float, default 0.05
        Significance level.
    clustervars : str or list of str, optional
        Cluster variable(s) for the multiplier bootstrap, mirroring R
        ``did::att_gt(clustervars=...)`` (Stata: ``csdid, cluster()``).
        The unit id ``i`` is always implied and may be included or
        omitted; at most **one** additional variable is allowed, and it
        must be time-invariant within unit.  Clustering beyond the unit
        level requires ``bstrap=True`` — analytic SEs do not account for
        it, so passing ``clustervars`` with ``bstrap=False`` raises.
    bstrap : bool, default False
        If ``True``, replace the analytic (delta-method) SEs of each
        ATT(g, t) with multiplier-bootstrap SEs on the influence
        functions — the R ``did`` / Stata ``csdid wboot`` inference
        path.  Default ``False`` preserves the analytic SEs (note R's
        ``att_gt`` defaults to ``bstrap=TRUE``; set ``bstrap=True`` for
        exact R-default parity).
    biters : int, default 1000
        Number of multiplier-bootstrap replications (R: ``biters``,
        Stata: ``reps()``).
    cband : bool, default False
        If ``True``, additionally report *uniform* (simultaneous)
        confidence bands across all ATT(g, t) via the sup-t bootstrap
        critical value — columns ``cband_lower`` / ``cband_upper`` in
        ``.detail``.  Requires ``bstrap=True``.
    boot_weight_type : {'rademacher', 'mammen'}, default 'rademacher'
        Multiplier weight distribution.  ``'rademacher'`` (±1) matches
        what the R ``did`` package actually draws
        (``BMisc::multiplier_bootstrap``; its docs cite Mammen 1993 but
        the implementation is Rademacher).  ``'mammen'`` is the Mammen
        (1993) two-point distribution of CS2021 §4.2 (Stata:
        ``csdid, wbtype(mammen)``).
    random_state : int, optional
        Seed for the multiplier bootstrap.
    panel : bool, default True
        If ``True`` (default), treat the data as a balanced panel and
        estimate ATT(g, t) via within-unit first differences.
        If ``False``, treat the data as *repeated cross-sections* —
        observations are not matched across time.  In RCS mode the
        estimator is the unconditional 2×2 cell-mean DID per (g, t)
        pair (CS2021 §3.2, eqn 2.4, RCS version) for covariate-free
        ``estimator='reg'``; every other combination routes to the
        Sant'Anna-Zhao repeated-cross-section estimators
        (``drdid_rc`` / ``std_ipw_did_rc`` / ``reg_did_rc``), matching R
        ``did``'s ``panel=FALSE`` path.
    allow_unbalanced_panel : bool, default False
        What to do when ``panel=True`` but some units are missing some
        periods.

        - ``False`` (default) — estimate ATT(g, t) from within-unit
          differences anyway. A unit missing either the base or the
          comparison period simply drops out of that cell, so the
          effective sample varies across cells; a warning names how many
          units are affected.
        - ``True`` — switch to the repeated-cross-section estimators,
          which never difference within unit and so keep every observed
          row, then fold the influence functions back to the unit level
          so the SEs still account for within-unit correlation. This is R
          ``did::att_gt(allow_unbalanced_panel = TRUE)``. If the panel
          turns out to be balanced the flag is inert and the ordinary
          panel estimator runs, again matching R.

        The two produce genuinely different estimates on the same
        unbalanced data — this is an estimator choice, not a tuning knob.

        .. versionadded:: 1.23.0

    Returns
    -------
    CausalResult
        Results with group-time ATTs, event study coefficients,
        pre-trend test, and all standard CausalResult methods.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd, numpy as np
    >>> # Create staggered panel data
    >>> rng = np.random.default_rng(42)
    >>> rows = []
    >>> for unit in range(90):
    ...     g_val = [4, 6, 0][unit // 30]  # 3 cohorts
    ...     for period in range(1, 9):
    ...         te = max(0, period - g_val + 1) if g_val > 0 else 0
    ...         rows.append({'i': unit, 't': period, 'y': te + rng.normal(),
    ...                      'g': g_val})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.callaway_santanna(df, y='y', g='g', t='t', i='i')
    >>> bool(result.estimate > 0)
    True

    References
    ----------
    Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences
    with multiple time periods. *Journal of Econometrics*. [@callaway2021difference]
    """
    data = _require_dataframe(data, function="callaway_santanna")
    y = _require_column_name(y, argument="y")
    g = _require_column_name(g, argument="g")
    t = _require_column_name(t, argument="t")
    i = _require_column_name(i, argument="i")
    x = _coerce_optional_columns(x, argument="x")
    estimator = _require_string_option(
        estimator,
        argument="estimator",
        valid=_ESTIMATORS,
    )
    control_group = _require_string_option(
        control_group,
        argument="control_group",
        valid=("nevertreated", "notyettreated"),
    )
    notyet_cutoff = _require_string_option(
        notyet_cutoff,
        argument="notyet_cutoff",
        valid=("period", "cohort"),
    )
    pscore_trim = _require_pscore_trim(pscore_trim)
    # A non-default cutoff with never-treated controls is silently inert —
    # say so rather than let the caller believe it took effect (§7).
    if notyet_cutoff != "period" and control_group == "nevertreated":
        warnings.warn(
            "callaway_santanna: notyet_cutoff='cohort' only affects the "
            "'notyettreated' control group, but control_group="
            "'nevertreated' was requested, so it has no effect. Pass "
            "control_group='notyettreated' to use it.",
            UserWarning,
            stacklevel=2,
        )
    base_period = _require_string_option(
        base_period,
        argument="base_period",
        valid=("universal", "varying"),
    )
    anticipation = _require_int_at_least(
        anticipation,
        argument="anticipation",
        minimum=0,
    )
    alpha = _require_alpha(alpha)
    panel = _require_bool(panel, argument="panel")
    allow_unbalanced_panel = _require_bool(
        allow_unbalanced_panel, argument="allow_unbalanced_panel"
    )
    if allow_unbalanced_panel and not panel:
        warnings.warn(
            "callaway_santanna: allow_unbalanced_panel=True has no effect "
            "when panel=False — the data are already treated as repeated "
            "cross-sections, with observation-level (not unit-level) "
            "influence functions.",
            UserWarning,
            stacklevel=2,
        )
    bstrap = _require_bool(bstrap, argument="bstrap")
    cband = _require_bool(cband, argument="cband")

    # se_method= is the shared DiD vocabulary; bstrap= is this estimator's
    # historical spelling and stays the source of truth when se_method is
    # not passed, so no existing call changes behaviour.
    if se_method is not None:
        if bstrap:
            raise MethodIncompatibility(
                "callaway_santanna: pass either se_method= or bstrap=, not "
                "both — they set the same thing and disagreeing values "
                "would silently resolve one way.",
                recovery_hint="Drop bstrap= and use se_method='multiplier'.",
                diagnostics={"se_method": se_method, "bstrap": bstrap},
            )
        _n_clusters = int(data[i].nunique()) if i in data.columns else None
        _resolved = _normalize_se_method(
            se_method,
            supported=("analytic", "multiplier"),
            function="callaway_santanna",
            n_clusters=_n_clusters,
        )
        bstrap = _resolved == "multiplier"
    biters = _require_int_at_least(biters, argument="biters", minimum=1)
    boot_weight_type = _require_string_option(
        boot_weight_type,
        argument="boot_weight_type",
        valid=("mammen", "rademacher"),
    )
    clustervars = _coerce_optional_columns(clustervars, argument="clustervars")
    _require_columns(
        data,
        (y, g, t, i, *(x or []), *(clustervars or [])),
        function="callaway_santanna",
    )
    # Drop unusable rows before any other validation so the never-treated,
    # cohort, and (g, t)-pair checks below all see the real estimation sample
    # rather than rows the estimator would silently discard.
    data = _drop_unusable_rows(
        data,
        columns=[y, t, i, *(x or [])],
        function="callaway_santanna",
    )

    if cband and not bstrap:
        raise MethodIncompatibility(
            "cband=True (uniform confidence bands) requires bstrap=True — "
            "the sup-t critical value comes from the multiplier bootstrap.",
            recovery_hint="Pass bstrap=True together with cband=True.",
            diagnostics={"bstrap": bstrap, "cband": cband},
        )

    # Resolve extra cluster variable (beyond the unit id), R did::mboot
    # convention: idname is always implied; at most one extra variable.
    extra_cluster: Optional[str] = None
    if clustervars:
        extras = [c for c in clustervars if c != i]
        if len(extras) > 1:
            raise MethodIncompatibility(
                f"clustervars supports at most one variable beyond the unit "
                f"id {i!r}, got {extras}.",
                recovery_hint="Pass clustervars=[i, one_cluster_var].",
                diagnostics={"clustervars": clustervars, "extras": extras},
            )
        extra_cluster = extras[0] if extras else None
    if extra_cluster is not None and not bstrap:
        raise MethodIncompatibility(
            "Clustering beyond the unit level requires bstrap=True — the "
            "analytic SEs do not account for within-cluster dependence, so "
            "reporting them under clustervars would silently understate "
            "uncertainty.",
            recovery_hint="Pass bstrap=True together with clustervars.",
            diagnostics={"clustervars": clustervars, "bstrap": bstrap},
        )
    if extra_cluster is not None:
        nuniq = data.groupby(i)[extra_cluster].nunique(dropna=False)
        if (nuniq > 1).any():
            bad = nuniq[nuniq > 1].index.tolist()[:5]
            raise MethodIncompatibility(
                f"Cluster variable {extra_cluster!r} is time-varying within "
                f"unit (e.g. units {bad}) — the CS multiplier bootstrap "
                "requires a time-invariant cluster membership.",
                recovery_hint=(
                    "Use a time-invariant cluster variable (R did::mboot "
                    "imposes the same restriction)."
                ),
                diagnostics={
                    "clustervar": extra_cluster,
                    "n_time_varying_units": int((nuniq > 1).sum()),
                },
            )

    # Fail loudly when the requested never-treated comparison group is empty.
    # With no never-treated units every ATT(g,t) loses its control cell and
    # returns 0.0, which silently aggregates to a headline ATT of 0.0 — a wrong
    # number that reads as "no effect" rather than an error. The internal group
    # encoding treats NaN / inf `g` as never-treated (0), so mirror that here.
    if control_group == "nevertreated":
        g_clean = data[g].fillna(0).replace([np.inf, -np.inf], 0)
        if not (g_clean == 0).any():
            raise MethodIncompatibility(
                "control_group='nevertreated' but the panel has no "
                "never-treated units (every unit is eventually treated), so "
                "there is no valid comparison group.",
                recovery_hint=(
                    "Use control_group='notyettreated', or add never-treated "
                    "units (g=0) to the panel."
                ),
                diagnostics={
                    "function": "callaway_santanna",
                    "control_group": control_group,
                    "n_never_treated": 0,
                },
            )

    # ---- Repeated cross-sections / unbalanced-panel branch --------------
    #
    # `allow_unbalanced_panel=True` only bites when the panel really is
    # unbalanced; on a balanced one R resets the flag and runs the ordinary
    # panel estimator, so the option never silently changes the estimand on
    # data that does not need it.
    use_unbalanced = False
    if panel and allow_unbalanced_panel:
        _n_units_seen = data[i].nunique()
        _n_periods_seen = data[t].nunique()
        use_unbalanced = len(data.dropna(subset=[y, t, i])) != (
            _n_units_seen * _n_periods_seen
        )

    if (not panel) or use_unbalanced:
        if clustervars:
            raise CallawayNotImplemented(
                (
                    "allow_unbalanced_panel=True"
                    if use_unbalanced
                    else "panel=False (repeated cross-sections)"
                )
                + " does not yet support clustervars — the influence "
                "functions are built cell by cell over unmatched "
                "observations, so the cluster bootstrap needs a separate "
                "design.",
                recovery_hint=(
                    "Drop clustervars, or aggregate via "
                    "sp.aggte(result, bstrap=True) which bootstraps the "
                    "influence functions directly."
                ),
                diagnostics={
                    "panel": panel,
                    "allow_unbalanced_panel": allow_unbalanced_panel,
                    "clustervars": clustervars,
                },
            )
        if use_unbalanced and weights is not None:
            raise CallawayNotImplemented(
                "allow_unbalanced_panel=True does not yet support weights= — "
                "the repeated-cross-section estimators it routes to take "
                "unweighted cell moments.",
                recovery_hint=(
                    "Balance the panel first (sp.balance_panel) and keep "
                    "weights, or drop weights."
                ),
                diagnostics={"allow_unbalanced_panel": True, "weights": weights},
            )
        return _callaway_santanna_rcs(
            data=data,
            y=y,
            g=g,
            t=t,
            x=x,
            base_period=base_period,
            anticipation=anticipation,
            alpha=alpha,
            estimator=estimator,
            control_group=control_group,
            pretest=pretest,
            pretest_periods=pretest_periods,
            unit_col=i if use_unbalanced else None,
        )

    # 1. Prepare panel data
    y_wide, unit_info, time_periods, cohorts, n_units, unit_weights = _prepare_panel(
        data, y, g, t, i, x, weights
    )

    if not cohorts:
        raise DataInsufficient(
            "No treatment cohorts found. Check group variable encoding.",
            recovery_hint=(
                "Encode first treatment periods in `g`, using 0 for "
                "never-treated units."
            ),
            diagnostics={"function": "callaway_santanna"},
        )

    # 2. Determine (g, t, base) estimation triples
    gt_pairs = _get_gt_pairs(cohorts, time_periods, base_period, anticipation)
    if not gt_pairs:
        raise DataInsufficient(
            "No valid (group, time) pairs to estimate.",
            recovery_hint=(
                "Check treatment timing, base_period, anticipation, and "
                "available periods."
            ),
            diagnostics={
                "function": "callaway_santanna",
                "cohorts": cohorts,
                "time_periods": time_periods,
                "base_period": base_period,
                "anticipation": anticipation,
            },
        )

    # 3. Estimate ATT(g,t) for each pair
    gt_results: List[Dict[str, Any]] = []
    inf_funcs_list: List[np.ndarray] = []
    z_crit = stats.norm.ppf(1 - alpha / 2)
    _TRIM_TALLY.reset()

    for g_val, t_val, base_val in gt_pairs:
        att, se, inf_func = _estimate_single_att(
            y_wide,
            unit_info,
            g_val,
            t_val,
            base_val,
            g,
            x,
            estimator,
            control_group,
            n_units,
            notyet_cutoff,
            pscore_trim,
            unit_weights,
        )

        pval = 2 * (1 - stats.norm.cdf(abs(att / se))) if se > 0 else 1.0

        gt_results.append(
            {
                "group": g_val,
                "time": t_val,
                "att": att,
                "se": se,
                "ci_lower": att - z_crit * se,
                "ci_upper": att + z_crit * se,
                "pvalue": pval,
                "relative_time": t_val - g_val,
            }
        )
        inf_funcs_list.append(inf_func)

    detail = pd.DataFrame(gt_results)

    # Stack influence functions: (n_units, n_gt_pairs)
    inf_matrix = np.column_stack(inf_funcs_list) if inf_funcs_list else None

    # 3.5 Optional multiplier-bootstrap inference (R did::att_gt bstrap
    # path).  Replaces the analytic SEs of each ATT(g,t) with bootstrap
    # SEs and, when cband=True, adds uniform (sup-t) confidence bands.
    cluster_ids: Optional[np.ndarray] = None
    if extra_cluster is not None:
        cluster_ids = (
            data.groupby(i)[extra_cluster].first().reindex(unit_info.index).to_numpy()
        )
    boot_cfg: Optional[Dict[str, Any]] = None
    crit_val_uniform: Optional[float] = None
    if bstrap and inf_matrix is not None:
        boot_cfg = {
            "n_boot": biters,
            "random_state": random_state,
            "weight_type": boot_weight_type,
            "cluster_ids": cluster_ids,
        }
        se_boot, crit = _core_multiplier_bootstrap(
            inf_matrix,
            n_units,
            alpha,
            biters,
            random_state,
            weight_type=boot_weight_type,
            cluster_ids=cluster_ids,
        )
        # Keep degenerate cells (empty (g,t) → zero IF, analytic se=inf)
        # flagged as inf rather than reporting the bootstrap's ~0 SE.
        valid_se = np.isfinite(detail["se"].values)
        se_new = np.where(valid_se, se_boot, np.inf)
        att_vals = detail["att"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            z_stat = np.where(
                np.isfinite(se_new) & (se_new > 0), att_vals / se_new, 0.0
            )
        detail["se"] = se_new
        detail["pvalue"] = np.where(
            np.isfinite(se_new) & (se_new > 0),
            2 * (1 - stats.norm.cdf(np.abs(z_stat))),
            1.0,
        )
        detail["ci_lower"] = att_vals - z_crit * se_new
        detail["ci_upper"] = att_vals + z_crit * se_new
        if cband:
            crit_val_uniform = crit
            detail["cband_lower"] = att_vals - crit * se_new
            detail["cband_upper"] = att_vals + crit * se_new

    # 4. Cohort sizes (for weighting).
    #
    # Under ω these must be ω-weighted cohort *mass*, not head counts:
    # the event-study/simple aggregations weight each cohort by its share
    # of the treated population, P_ω(G = g | ...), so a head count would
    # silently mix a weighted ATT(g,t) with unweighted cohort shares and
    # target neither estimand [@baker2026difference, §5.2.4].
    if weights is None:
        cohort_sizes = unit_info[g].value_counts()
    else:
        cohort_sizes = (
            pd.Series(unit_weights, index=unit_info.index)
            .groupby(unit_info[g])
            .sum()
            .sort_values(ascending=False)
        )

    # 5. Simple aggregation (post-treatment)
    post_mask = detail["relative_time"] >= 0
    post_inf = (
        np.asarray(inf_matrix[:, post_mask.values], dtype=float)
        if inf_matrix is not None
        else None
    )
    agg_est, agg_se, agg_pval, agg_ci = _aggregate_simple(
        detail[post_mask],
        post_inf,
        cohort_sizes,
        n_units,
        alpha,
        boot_cfg=boot_cfg,
        unit_cohorts=unit_info[g].to_numpy(),
        unit_weights=unit_weights,
    )

    # 6. Event study aggregation
    event_study = _aggregate_event_study(
        detail,
        inf_matrix,
        cohort_sizes,
        n_units,
        alpha,
        boot_cfg=boot_cfg,
    )

    # 7. Pre-trend test
    pretrend = _pretrend_test(
        detail, inf_matrix, n_units, pretest=pretest, pretest_periods=pretest_periods
    )

    # Trimming that binds changes the estimand — it silently redefines the
    # control arm. Never let that pass unannounced (§7).
    if _TRIM_TALLY.n_trimmed:
        warnings.warn(
            f"callaway_santanna: propensity trimming removed "
            f"{_TRIM_TALLY.n_trimmed} of {_TRIM_TALLY.n_controls} control "
            f"unit-cells (summed over the ATT(g,t) cells) whose estimated "
            f"p(X) reached pscore_trim={pscore_trim}. Those controls are "
            "too treated-looking to reweight stably, so the effective "
            "control group is smaller than the nominal one. Inspect "
            "overlap before reading the estimate; pass pscore_trim=1.0 to "
            "disable trimming.",
            UserWarning,
            stacklevel=2,
        )

    # 8. Build result
    model_info: Dict[str, Any] = {
        "estimator": estimator.upper(),
        "control_group": control_group,
        "notyet_cutoff": notyet_cutoff,
        "pscore_trim": pscore_trim,
        "n_pscore_trimmed": _TRIM_TALLY.n_trimmed,
        "base_period": base_period,
        "anticipation": anticipation,
        # Which parallel-trends assumption this fit actually imposes.
        # CS identifies ATT(g, t) off post-treatment periods only, so the
        # assumption is NEV or NYT according to the comparison group;
        # base_period changes which pre-period contrasts get *reported*,
        # not what is assumed for identification.
        "parallel_trends": _pt_block(
            "PT-GT-NEV" if control_group == "nevertreated" else "PT-GT-NYT",
            conditional=bool(x),
            extra={"base_period": base_period, "anticipation": anticipation},
        ),
        "weights": weights,
        "weighted": weights is not None,
        "n_units": n_units,
        "n_periods": len(time_periods),
        "n_cohorts": len(cohorts),
        "cohorts": cohorts,
        "event_study": event_study,
        "pretrend_test": pretrend,
        "cohort_sizes": cohort_sizes,
        "clustervars": clustervars,
        "bstrap": bstrap,
        "se_method": "multiplier" if bstrap else "analytic",
        "biters": biters if bstrap else 0,
        "cband": cband,
        "boot_weight_type": boot_weight_type if bstrap else None,
        "crit_val_uniform": crit_val_uniform,
        # Private plumbing: aligned with the influence-function rows.
        # Consumed by sp.aggte (cluster-aware bootstrap) and
        # sp.influence_functions (unit-labelled export).
        "_cluster_ids": cluster_ids,
        "_unit_ids": unit_info.index.to_numpy(),
        "_unit_cohorts": unit_info[g].to_numpy(),
        # Aggregation weights for sp.aggte: cohort shares must be built
        # from ω-mass, not head counts, once ω is in play.
        "_unit_weights": unit_weights,
    }

    _result = CausalResult(
        method="Callaway and Sant'Anna (2021)",
        estimand="ATT",
        estimate=agg_est,
        se=agg_se,
        pvalue=agg_pval,
        ci=agg_ci,
        alpha=alpha,
        n_obs=len(data),
        detail=detail,
        model_info=model_info,
        _influence_funcs=inf_matrix,
        _citation_key="callaway_santanna",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.callaway_santanna",
            params={
                "y": y,
                "g": g,
                "t": t,
                "i": i,
                "x": x,
                "estimator": estimator,
                "control_group": control_group,
                "base_period": base_period,
                "anticipation": anticipation,
                "alpha": alpha,
                "panel": panel,
                "clustervars": clustervars,
                "bstrap": bstrap,
                "biters": biters,
                "cband": cband,
                "boot_weight_type": boot_weight_type,
                "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover — provenance must never break fit
        pass
    return _result


# ======================================================================
# Data preparation
# ======================================================================


def _prepare_panel(
    data: pd.DataFrame,
    y: str,
    g: str,
    t: str,
    i: str,
    x: Optional[List[str]],
    weights_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, list, list, int, np.ndarray]:
    """Validate and reshape panel data to wide format.

    Returns
    -------
    y_wide : pd.DataFrame
        Outcomes pivoted to (units × time periods).
    unit_info : pd.DataFrame
        Unit-level info indexed by unit id (group, covariates).
    time_periods : list
        Sorted unique time periods.
    cohorts : list
        Sorted treatment cohorts (excluding never-treated).
    n_units : int
    """
    _require_columns(data, (y, g, t, i, *(x or [])), function="_prepare_panel")

    # Pivot outcome to wide: rows = units, columns = time periods.
    # dropna=False keeps the missingness count below honest: the default would
    # silently delete any entirely-NaN unit or period, so the panel would look
    # balanced precisely in the cases where it is most broken. Callers reach
    # here past `_drop_unusable_rows`, so this is a no-op on the happy path —
    # it is here to keep the invariant from regressing.
    y_wide = data.pivot_table(
        index=i, columns=t, values=y, aggfunc="first", dropna=False
    )

    # An unbalanced panel silently changes the estimand here: ATT(g, t) is
    # built from within-unit differences Y_t - Y_base, so any unit missing
    # either period drops out of that cell.  R ``did`` takes a different
    # route — ``allow_unbalanced_panel=TRUE`` switches to the repeated
    # cross-section estimator ("Proceeding as such") — so the two disagree
    # substantially on the same data.  Warn rather than let the divergence
    # pass as a parity failure.  (§7: fail loudly.)
    _n_missing = int(y_wide.isna().to_numpy().sum())
    if _n_missing:
        _n_cells = int(y_wide.size)
        _n_incomplete = int(y_wide.isna().any(axis=1).sum())
        warnings.warn(
            f"callaway_santanna: unbalanced panel — {_n_incomplete} of "
            f"{y_wide.shape[0]} units are missing at least one period "
            f"({_n_missing}/{_n_cells} unit-period cells absent). ATT(g, t) "
            "is formed from within-unit differences, so units missing "
            "either the base or the comparison period drop out of that "
            "cell; the effective sample varies across cells. Pass "
            "allow_unbalanced_panel=True to switch to the "
            "repeated-cross-section estimators instead, which keep every "
            "observed row (R `did`'s allow_unbalanced_panel=TRUE), or "
            "balance the panel first (sp.balance_panel) if you need a "
            "single fixed sample. The three give different numbers on the "
            "same data — that is an estimator choice, not a bug.",
            UserWarning,
            stacklevel=3,
        )

    # Unit-level info (first occurrence per unit)
    info_cols = [g] + (x or [])
    unit_info = data.groupby(i)[info_cols].first()

    # Unit-level weights ω.  Normalised to mean 1 over the unit universe so
    # that the ATT is scale-invariant and every ``np.mean(w * z)`` below
    # reduces *exactly* to ``np.mean(z)`` in the unweighted path.
    unit_weights = _prepare_unit_weights(data, i, weights_col, unit_info.index)

    # Replace NaN / inf in group variable with 0 (never-treated)
    unit_info[g] = unit_info[g].fillna(0).replace([np.inf, -np.inf], 0)
    # Ensure integer type for group
    unit_info[g] = unit_info[g].astype(int)

    time_periods = sorted(data[t].unique())
    max_time = max(time_periods)

    g_values = sorted(unit_info[g].unique())
    # Cohorts: groups that actually get treated within the sample period
    cohorts = [v for v in g_values if v > 0 and v <= max_time]

    n_units = len(unit_info)
    return y_wide, unit_info, time_periods, cohorts, n_units, unit_weights


def _prepare_unit_weights(
    data: pd.DataFrame,
    i: str,
    weights_col: Optional[str],
    unit_index: pd.Index,
) -> np.ndarray:
    """Collapse ω to one weight per unit and renormalise to mean 1.

    ``None`` returns an exact vector of ones, so the whole weighted code
    path below is bit-identical to the historical unweighted one.

    Weights must be time-invariant: a unit whose ω changes across periods
    has no well-defined share in the target population, and silently
    taking the first value would answer a question nobody asked
    (CLAUDE.md §3.7).
    """
    n = len(unit_index)
    if weights_col is None:
        return np.ones(n, dtype=float)

    if weights_col not in data.columns:
        raise MethodIncompatibility(
            f"weights column {weights_col!r} is not in the data.",
            recovery_hint="Pass the name of an existing numeric column.",
            diagnostics={"weights": weights_col},
        )

    grouped = data.groupby(i)[weights_col]
    spread = grouped.nunique(dropna=False)
    varying = spread[spread > 1]
    if len(varying):
        raise MethodIncompatibility(
            f"weights column {weights_col!r} varies within unit for "
            f"{len(varying)} of {n} units (e.g. "
            f"{list(varying.index[:3])}). Unit weights define each unit's "
            "share of the target population, so they must be constant "
            "within unit.",
            recovery_hint=(
                "Use a fixed baseline weight (e.g. base-year population) "
                "rather than a time-varying one."
            ),
            diagnostics={"weights": weights_col, "n_varying": int(len(varying))},
        )

    w = grouped.first().reindex(unit_index).to_numpy(dtype=float)

    if not np.all(np.isfinite(w)):
        raise MethodIncompatibility(
            f"weights column {weights_col!r} contains NaN or infinite "
            f"values for {int((~np.isfinite(w)).sum())} units.",
            recovery_hint="Drop or impute those units before estimating.",
            diagnostics={"weights": weights_col},
        )
    if np.any(w < 0):
        raise MethodIncompatibility(
            f"weights column {weights_col!r} contains negative values.",
            recovery_hint="Sampling/population weights must be >= 0.",
            diagnostics={"weights": weights_col, "min": float(w.min())},
        )
    total = float(w.sum())
    if total <= 0:
        raise MethodIncompatibility(
            f"weights column {weights_col!r} sums to {total}; at least one "
            "unit must carry positive weight.",
            recovery_hint="Check the weights column for an all-zero slice.",
            diagnostics={"weights": weights_col},
        )
    return w * (n / total)


def _get_gt_pairs(
    cohorts: list,
    time_periods: list,
    base_period: str,
    anticipation: int = 0,
) -> List[Tuple[int, int, int]]:
    """Determine all (g, t, base) triples to estimate.

    Mirrors the ``pret`` block of R ``did::compute.att_gt``:

    - the cohort-level base is the last period with ``t + δ < g``
      (``g − 1 − δ`` on a unit-spaced grid), used for every
      post-treatment cell and for *all* cells under ``'universal'``;
    - under ``'varying'`` a pre-treatment cell (``t < g``) instead uses
      the period immediately before ``t``, **unshifted by δ**. R only
      applies the anticipation shift once a cell is post-treatment, so
      shifting the pre-cell base here too would move every placebo off
      the reference implementation (this is what StatsPAI did before
      1.23.0).

    Comparisons are strict inequalities on the observed period grid
    rather than ``t − 1`` arithmetic, so irregularly spaced periods
    (1990, 1995, 2000, …) resolve to the neighbouring observed period
    instead of silently dropping the cell.
    """
    pairs = []
    for g_val in cohorts:
        # R: pret_g <- tail(which((tlist + anticipation) < glist[g]), 1)
        available_pre = [tp for tp in time_periods if tp + anticipation < g_val]
        if not available_pre:
            # R warns and drops the cohort ("There are no pre-treatment
            # periods for the group first treated at ..."); say so rather
            # than silently returning a smaller grid (§7).
            warnings.warn(
                f"callaway_santanna: cohort g={g_val} has no pre-treatment "
                f"period once anticipation={anticipation} is accounted for "
                "(no observed period t with t + anticipation < g). Every "
                "ATT(g, t) for this cohort is unidentified, so the cohort is "
                "dropped — matching R `did`. Its units still serve as "
                "controls where the control group allows.",
                UserWarning,
                stacklevel=3,
            )
            continue
        universal_base = max(available_pre)

        for t_val in time_periods:
            if base_period == "varying" and t_val < g_val:
                # R: varying base period => pret <- t (the period right
                # before the cell), regardless of anticipation.
                pre_of_t = [tp for tp in time_periods if tp < t_val]
                if not pre_of_t:
                    continue
                this_base = max(pre_of_t)
            else:
                this_base = universal_base

            # R: `if (tlist[pret] == tlist[t + tfac]) next` — a cell whose
            # base *is* itself is the normalised reference (ATT ≡ 0 by
            # construction), not an estimable placebo. Under 'varying'
            # this never fires for a pre-treatment cell, so event time
            # e = −1 stays a genuine placebo, as R / Stata csdid report.
            if this_base == t_val:
                continue

            pairs.append((g_val, t_val, this_base))

    return pairs


# ======================================================================
# ATT(g,t) estimators
# ======================================================================


def _estimate_single_att(
    y_wide: pd.DataFrame,
    unit_info: pd.DataFrame,
    g_val: int,
    t_val: int,
    base_val: int,
    g_col: str,
    x_cols: Optional[List[str]],
    estimator: str,
    control_group: str,
    n_total: int,
    notyet_cutoff: str = "period",
    pscore_trim: float = _DEFAULT_PSCORE_TRIM,
    unit_weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Estimate a single ATT(g,t) and return (att, se, influence_func)."""

    g_series = unit_info[g_col]

    # Treatment indicator: units first treated at g_val
    is_treated = g_series == g_val

    # Control indicator.
    #
    # For 'notyettreated' there are two defensible cutoffs, and the two
    # reference implementations disagree on which is the default:
    #
    #   notyet_cutoff='period' — untreated as of t. R ``did`` and Stata
    #       ``csdid, asinr``. This is StatsPAI's default and matches R.
    #   notyet_cutoff='cohort' — untreated as of g, i.e. a stricter set that
    #       excludes anyone treated between t and g. Stata ``csdid``'s
    #       own default.
    #
    # They coincide for post-treatment cells (t >= g) and differ only on
    # pre-treatment placebos, where 'cohort' drops cohorts that switch on
    # between t and g. Verified against csdid on mpdta: 'period' reproduces
    # ``asinr`` and 'cohort' reproduces the csdid default, both to 1e-9.
    if control_group == "nevertreated":
        is_control = g_series == 0
    elif notyet_cutoff == "cohort":
        # max(t, g), not g: csdid's rule is scoped to PRE-treatment cells
        # ("pre-treatment ATTGT's ... are estimated using ..."). For t >= g
        # the cutoff is already t, so max() reproduces the shared
        # post-treatment behaviour instead of shrinking those cells too.
        is_control = (g_series == 0) | (g_series > max(t_val, g_val))
    else:  # notyettreated, cutoff at the comparison period t
        is_control = (g_series == 0) | (g_series > t_val)

    # Outcome change ΔY = Y_t - Y_base
    if t_val not in y_wide.columns or base_val not in y_wide.columns:
        return 0.0, np.inf, np.zeros(n_total)

    dy = y_wide[t_val] - y_wide[base_val]

    # Valid units: in a relevant group AND observed in both periods
    relevant = (is_treated | is_control) & dy.notna()
    n_rel = relevant.sum()
    if n_rel < 5:
        return 0.0, np.inf, np.zeros(n_total)

    dy_sub = dy[relevant].values
    d_sub = is_treated[relevant].values.astype(float)
    n1 = d_sub.sum()
    n0 = n_rel - n1

    # ω on this (g, t) cell, renormalised to mean 1 *within the cell* so
    # that every weighted mean below collapses to its unweighted form when
    # ω ≡ 1 and the estimating equations stay Hájek-normalised.
    if unit_weights is None:
        w_sub = np.ones(int(n_rel), dtype=float)
    else:
        w_sub = np.asarray(unit_weights, dtype=float)[relevant.values]
        w_mean = w_sub.mean()
        if w_mean <= 0:
            return 0.0, np.inf, np.zeros(n_total)
        w_sub = w_sub / w_mean

    if n1 < 1 or n0 < 1:
        return 0.0, np.inf, np.zeros(n_total)

    # Covariates
    x_sub = None
    if x_cols:
        x_sub = unit_info.loc[relevant, x_cols].values.astype(float)
        # Drop covariates with zero variance
        var = np.var(x_sub, axis=0)
        if np.any(var < 1e-12):
            keep = var >= 1e-12
            if keep.sum() == 0:
                x_sub = None
            else:
                x_sub = x_sub[:, keep]

    # Dispatch. 'ipw' and 'stdipw' are deliberately the same estimator —
    # see _ESTIMATORS.
    if estimator == "dr":
        att, se, inf_local = _dr_att(dy_sub, d_sub, x_sub, pscore_trim, w_sub)
    elif estimator in ("ipw", "stdipw"):
        att, se, inf_local = _ipw_att(dy_sub, d_sub, x_sub, pscore_trim, w_sub)
    elif estimator == "ipw_abadie":
        att, se, inf_local = _ipw_abadie_att(dy_sub, d_sub, x_sub, pscore_trim, w_sub)
    else:  # reg
        att, se, inf_local = _reg_att(dy_sub, d_sub, x_sub, w_sub)

    # Map the local influence function to the full unit universe.  The
    # ATT(g,t) estimator is computed on the relevant treated/control
    # subset, so the subset-level IF must be rescaled when embedded in
    # the n_total-vector used for cross-(g,t) aggregation.  Without this
    # n_total / n_rel factor, simple-ATT aggregation treats the shared
    # control influence too weakly and systematically understates SEs.
    inf_full = np.zeros(n_total)
    relevant_idx = np.where(relevant.values)[0]
    inf_full[relevant_idx] = inf_local * (n_total / n_rel)

    return att, se, inf_full


# ------------------------------------------------------------------
# Doubly Robust
# ------------------------------------------------------------------


def _dr_att(
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
    pscore_trim: float = _DEFAULT_PSCORE_TRIM,
    w: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Doubly robust ATT(g,t) estimator (Sant'Anna & Zhao 2020).

    ``w`` carries the unit weights ω (mean 1). Every ``np.mean(w * z)``
    reduces exactly to ``np.mean(z)`` at ω ≡ 1, so the unweighted path is
    numerically untouched.
    """
    n = len(dy)
    c = 1 - d
    w = np.ones(n, dtype=float) if w is None else np.asarray(w, dtype=float)

    # --- Propensity score ---
    pscore = _estimate_pscore(d, x, n, w)
    keep = _pscore_trim_mask(pscore, d, pscore_trim)
    _TRIM_TALLY.record(keep, d)

    # --- Outcome regression ---
    m_hat = _estimate_outcome_reg(dy, c, x, n, w)

    # --- DR weights ---
    p_d = np.mean(w * d)
    w1 = keep * w * d / p_d if p_d > 0 else np.zeros(n)

    ipw_raw = keep * w * pscore * c / (1 - pscore)
    ipw_denom = np.mean(ipw_raw)
    w0 = ipw_raw / ipw_denom if ipw_denom > 1e-12 else np.zeros(n)

    # ATT, as the difference of two Hajek-normalised arm means.
    resid = dy - m_hat
    eta_t = float(np.mean(w1 * resid))
    eta_c = float(np.mean(w0 * resid))
    att = eta_t - eta_c

    # ⚠️ correctness fix (2): de-mean each arm by its own mean.
    #
    # The previous form ``(w1 - w0) resid - att w1`` substitutes ``w1``
    # for ``w0`` in the control arm's centring term. Both weights are
    # Hajek-normalised to mean one, so the ATT is unaffected, but the
    # variance is not. See the fuller note in :func:`_ipw_att`.
    #
    inf_func = w1 * (resid - eta_t) - w0 * (resid - eta_c)

    # ⚠️ correctness fix (3): propagate both nuisance estimation effects.
    #
    # DR is Neyman-orthogonal in each nuisance separately, so these terms
    # are second-order and the omission cost far less than in
    # :func:`_ipw_att` — but "second-order" is not "zero", and the gap to
    # ``DRDID::drdid_panel`` was up to 0.9% on the reference grid. Port
    # the three remaining pieces (DRDID 1.2.3, read from source):
    #
    #     inf.treat.2 = alr_wols @ M1 / E[w_treat]
    #     inf.cont.2  = alr_ps   @ M2 / E[w_cont]
    #     inf.cont.3  = alr_wols @ M3 / E[w_cont]
    #     att.inf = inf.treat.1 - inf.treat.2 - (inf.cont.1 + inf.cont.2 - inf.cont.3)
    adj = _dr_estimation_effect(
        dy=dy,
        d=d,
        x=x,
        w=w,
        keep=keep,
        pscore=pscore,
        resid=resid,
        eta_c=eta_c,
        p_d=p_d,
        ipw_denom=ipw_denom,
    )
    if adj is not None:
        inf_func = inf_func + adj

    se = float(np.sqrt(np.mean(inf_func**2) / n))

    return float(att), se, inf_func


def _dr_estimation_effect(
    *,
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
    w: np.ndarray,
    keep: np.ndarray,
    pscore: np.ndarray,
    resid: np.ndarray,
    eta_c: float,
    p_d: float,
    ipw_denom: float,
) -> Optional[np.ndarray]:
    """Nuisance-estimation correction for the DR influence function.

    Port of ``DRDID::drdid_panel``'s ``inf.treat.2``, ``inf.cont.2`` and
    ``inf.cont.3``. Returns the signed sum to be *added* to the
    first-order influence function, or ``None`` when either first-stage
    design is singular — in which case the caller keeps the uncorrected
    (slightly conservative) variance rather than emitting a nonsense one.

    ``x`` is the covariate block *without* an intercept; DRDID's
    ``int.cov`` is the model matrix including one, and the correction is
    non-trivial even in the no-covariate case, where ``int.cov`` is a
    single column of ones.
    """
    n = len(dy)
    try:
        import statsmodels.api as sm

        xc = (
            np.ones((n, 1), dtype=float)
            if x is None or x.shape[1] == 0
            else sm.add_constant(np.asarray(x, dtype=float))
        )
    except ImportError:  # pragma: no cover - statsmodels is a core dep
        return None

    if p_d <= 0 or ipw_denom <= 1e-12:
        return None

    # DRDID's un-normalised arm weights.
    w_treat = keep * w * d
    w_cont = keep * w * pscore * (1.0 - d) / (1.0 - pscore)
    mw_t, mw_c = float(np.mean(w_treat)), float(np.mean(w_cont))
    if mw_t <= 0 or mw_c <= 0:
        return None

    # Asymptotic linear representation of the control-arm WLS.
    weights_ols = w * (1.0 - d)
    wols_x = weights_ols[:, None] * xc
    xpx = (wols_x.T @ xc) / n
    try:
        xpx_inv = np.linalg.inv(xpx)
    except np.linalg.LinAlgError:
        return None
    alr_wols = ((weights_ols * resid)[:, None] * xc) @ xpx_inv

    # Asymptotic linear representation of the weighted logit.
    hess_w = pscore * (1.0 - pscore) * w
    info = (xc.T * hess_w) @ xc / n
    try:
        info_inv = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return None
    alr_ps = ((w * (d - pscore))[:, None] * xc) @ info_inv

    m1 = np.mean(w_treat[:, None] * xc, axis=0)
    m2 = np.mean((w_cont * (resid - eta_c))[:, None] * xc, axis=0)
    m3 = np.mean(w_cont[:, None] * xc, axis=0)

    out = -(alr_wols @ m1) / mw_t - (alr_ps @ m2) / mw_c + (alr_wols @ m3) / mw_c
    return out if np.all(np.isfinite(out)) else None


# ------------------------------------------------------------------
# IPW
# ------------------------------------------------------------------


def _ipw_att(
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
    pscore_trim: float = _DEFAULT_PSCORE_TRIM,
    w: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Stabilized (Hájek-normalized) IPW ATT(g,t) estimator.

    This is what R ``did``/``DRDID`` call ``std_ipw_did_panel`` and what
    Stata ``csdid`` calls ``method(stdipw)`` — *not* Abadie (2005), which
    normalizes both arms by the same E[D]. See :func:`_ipw_abadie_att`.
    """
    n = len(dy)
    c = 1 - d
    w = np.ones(n, dtype=float) if w is None else np.asarray(w, dtype=float)

    pscore = _estimate_pscore(d, x, n, w)
    keep = _pscore_trim_mask(pscore, d, pscore_trim)
    _TRIM_TALLY.record(keep, d)

    p_d = np.mean(w * d)
    w1 = keep * w * d / p_d if p_d > 0 else np.zeros(n)

    ipw_raw = keep * w * pscore * c / (1 - pscore)
    ipw_denom = np.mean(ipw_raw)
    w0 = ipw_raw / ipw_denom if ipw_denom > 1e-12 else np.zeros(n)

    # ⚠️ correctness fix (two parts), validated against R
    # ``DRDID::std_ipw_did_panel`` via ``did::att_gt(est_method="ipw")``.
    #
    # (1) The influence function must de-mean each arm by *its own* mean,
    #     not both by the ATT. The old form was
    #
    #         (w1 - w0) dY - att w1
    #       = w1 (dY - eta_t) - w0 dY + w1 eta_c
    #
    #     which carries ``+w1 eta_c`` where the correct term is
    #     ``+w0 eta_c``. Both weight vectors are Hajek-normalised to mean
    #     one, so the two agree in expectation and the ATT was never
    #     affected — but they have different variances, so every reported
    #     IPW standard error was wrong.
    #
    # (2) IPW is not Neyman-orthogonal in the propensity score, so the
    #     score's estimation effect belongs in the influence function.
    #     Omitting it inflates the SE further.
    eta_t = float(np.mean(w1 * dy))
    eta_c = float(np.mean(w0 * dy))
    att = eta_t - eta_c

    inf_func = w1 * (dy - eta_t) - w0 * (dy - eta_c)
    if x is not None and x.shape[1] > 0 and ipw_denom > 1e-12:
        adj = _pscore_estimation_effect(
            dy=dy,
            x=x,
            w=w,
            d=d,
            pscore=pscore,
            arm_weights=ipw_raw,
            arm_denom=ipw_denom,
        )
        if adj is not None:
            inf_func = inf_func - adj

    se = float(np.sqrt(np.mean(inf_func**2) / n))

    return float(att), se, inf_func


def _pscore_estimation_effect(
    *,
    dy: np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    pscore: np.ndarray,
    arm_weights: np.ndarray,
    arm_denom: float,
    center: bool = True,
) -> Optional[np.ndarray]:
    """Asymptotic-linear correction for having *estimated* the logit.

    Port of the ``asy.lin.rep.ps %*% M2`` term in ``DRDID``. Returns
    ``None`` when the logit information matrix is singular, in which case
    the caller keeps the uncorrected (conservative) influence function
    rather than producing a nonsense variance.

    Parameters mirror :func:`_ipw_att`'s locals; ``arm_weights`` is the
    un-normalised control-arm weight vector and ``arm_denom`` its mean.

    ``center`` selects which of DRDID's two ``mom.logit`` forms to build,
    and the choice is dictated by what the estimator divides by:

    - ``True`` — ``colMeans(w_cont (ΔY − η_c) X) / E[w_cont]``, the
      Hájek-normalised form used by ``std_ipw_did_panel``. The estimator
      divides the control arm by its *own* weight mass, so that mass is
      itself estimated and the ``η_c`` term is its derivative.
    - ``False`` — ``colMeans(w_cont ΔY X)``, the form used by
      ``ipw_did_panel`` (Abadie). Both arms share the fixed denominator
      ``E[D]``, so there is no control-mass derivative to subtract and
      de-meaning here would remove a term that belongs in the variance.
    """
    try:
        import statsmodels.api as sm

        xc = sm.add_constant(x)
    except ImportError:  # pragma: no cover - statsmodels is a core dep
        return None

    # Score of the weighted logit: w (D - p) X.
    score = (w * (d - pscore))[:, None] * xc
    # Information matrix: E[w p (1-p) X X'].
    hess_w = w * pscore * (1.0 - pscore)
    info = (xc.T * hess_w) @ xc / len(dy)
    try:
        info_inv = np.linalg.pinv(info)
    except np.linalg.LinAlgError:  # pragma: no cover - pinv rarely raises
        return None
    if not np.all(np.isfinite(info_inv)):
        return None
    asy_lin_ps = score @ info_inv

    if center:
        eta = float(np.mean(arm_weights * dy) / arm_denom)
        m2 = np.mean((arm_weights * (dy - eta))[:, None] * xc, axis=0) / arm_denom
    else:
        m2 = np.mean((arm_weights * dy)[:, None] * xc, axis=0) / arm_denom
    out = asy_lin_ps @ m2
    return out if np.all(np.isfinite(out)) else None


def _ipw_abadie_att(
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
    pscore_trim: float = _DEFAULT_PSCORE_TRIM,
    w: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Abadie (2005) IPW ATT(g,t) — Horvitz-Thompson, *not* normalized.

    Reference: ``abadie2005semiparametric``.

    Both arms are scaled by the same E[D], so the control weights need not
    average to one in finite samples. This is Stata ``csdid``'s
    ``method(ipw)``; it is a genuinely different estimator from
    :func:`_ipw_att`, differing at O(1e-4) on ``mpdta``.
    """
    n = len(dy)
    c = 1 - d
    w = np.ones(n, dtype=float) if w is None else np.asarray(w, dtype=float)

    pscore = _estimate_pscore(d, x, n, w)
    keep = _pscore_trim_mask(pscore, d, pscore_trim)
    _TRIM_TALLY.record(keep, d)

    p_d = np.mean(w * d)
    if p_d <= 0:
        return 0.0, np.inf, np.zeros(n)

    # Abadie's single common denominator E[ωD] for both arms.
    w1 = keep * w * d / p_d
    w0 = keep * w * pscore * c / ((1 - pscore) * p_d)

    att = float(np.mean((w1 - w0) * dy))

    # ⚠️ correctness fix: propagate the propensity-score estimation effect.
    #
    # IPW is not Neyman-orthogonal in the score, so treating p̂(X) as known
    # is not a second-order omission — it inflated this SE by ~34% against
    # ``DRDID::ipw_did_panel`` on the reference fixture. The sibling
    # :func:`_ipw_att` (Hájek-normalised) already carried this term; the
    # Abadie variant did not.
    #
    # DRDID builds it un-centred here (``mom.logit <- att.cont * int.cov``)
    # because both arms are divided by the fixed E[ωD] rather than by the
    # control weight mass — see :func:`_pscore_estimation_effect`.
    inf_func = (w1 - w0) * dy - att * w1
    if x is not None and x.shape[1] > 0:
        adj = _pscore_estimation_effect(
            dy=dy,
            x=x,
            w=w,
            d=d,
            pscore=pscore,
            arm_weights=w0,
            arm_denom=1.0,
            center=False,
        )
        if adj is not None:
            inf_func = inf_func - adj

    se = float(np.sqrt(np.mean(inf_func**2) / n))

    return att, se, inf_func


# ------------------------------------------------------------------
# Outcome regression
# ------------------------------------------------------------------


def _reg_att(
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
    w: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Outcome regression ATT(g,t) estimator."""
    n = len(dy)
    c = 1 - d
    w = np.ones(n, dtype=float) if w is None else np.asarray(w, dtype=float)

    p_d = np.mean(w * d)
    w1 = w * d / p_d if p_d > 0 else np.zeros(n)

    c_mask = c.astype(bool)
    c_count = int(c_mask.sum())

    x_arr: Optional[np.ndarray]
    if x is None or x.shape[1] == 0:
        x_arr = None
        use_constant_outcome = True
    else:
        x_arr = x
        use_constant_outcome = c_count < 2

    if x_arr is not None and not use_constant_outcome:
        k = x_arr.shape[1]
        use_constant_outcome = c_count <= k + 1

    if use_constant_outcome:
        m0 = _weighted_control_mean(dy, c_mask, w) if c_count > 0 else 0.0
        m_hat_const = np.full(n, m0, dtype=float)
        resid = dy - m_hat_const
        att = float(np.mean(w1 * resid))

        p_c = np.mean(w * c)
        control_adjust = w * c * resid / p_c if p_c > 0 else np.zeros(n)
    else:
        assert x_arr is not None
        try:
            import statsmodels.api as sm

            x_const_control = sm.add_constant(x_arr[c_mask])
            # WLS with ω; at ω ≡ 1 the sqrt(w) scaling is exactly 1.0, so
            # this is bit-identical to the historical OLS call.
            ols = sm.WLS(dy[c_mask], x_const_control, weights=w[c_mask])
            result = ols.fit()
            x_const = sm.add_constant(x_arr)
            m_hat = np.asarray(result.predict(x_const), dtype=float).reshape(n)
            resid = dy - m_hat
            att = float(np.mean(w1 * resid))

            # Outcome-regression inference must include the uncertainty in
            # the control regression used to estimate m0(X).  For the WLS
            # first stage this is the delta-method term:
            #   - E[ω D X / p]' (E[ω C X X'])^{-1} ω C X_i u_i.
            # ω ≡ 1 keeps the original ``A'A`` contraction verbatim: the
            # weighted form ``(A' diag(ω)) A`` is algebraically identical
            # but reassociates the BLAS call and shifts the SE by ~2 ULP,
            # which is enough to move a pinned parity fixture.
            if np.allclose(w[c_mask], 1.0):
                a_mat = (x_const_control.T @ x_const_control) / n
            else:
                a_mat = (x_const_control.T * w[c_mask]) @ x_const_control / n
            xbar_treat = np.mean(w1[:, None] * x_const, axis=0)
            lever = x_const @ (np.linalg.pinv(a_mat).T @ xbar_treat)
            control_adjust = w * c * resid * lever
        except Exception as exc:
            # The user requested a covariate-adjusted outcome regression but
            # the control OLS failed (collinear / degenerate design). Fall
            # back to the control-mean estimator, but do NOT do so silently:
            # this changes the point estimate from covariate-adjusted to
            # unconditional (CLAUDE.md §3.7 — fail loudly).
            warnings.warn(
                "callaway_santanna: the covariate-adjusted outcome "
                f"regression failed ({type(exc).__name__}: {exc}); this "
                "ATT(g,t) falls back to the UNADJUSTED control-mean "
                "estimator, so it is no longer covariate-adjusted. Check "
                "for collinear or degenerate covariates.",
                ConvergenceWarning,
                stacklevel=2,
            )
            m0 = _weighted_control_mean(dy, c_mask, w) if c_count > 0 else 0.0
            m_hat = np.full(n, m0)
            resid = dy - m_hat
            att = float(np.mean(w1 * resid))
            p_c = np.mean(w * c)
            control_adjust = w * c * resid / p_c if p_c > 0 else np.zeros(n)

    inf_func = w1 * (resid - att) - control_adjust
    se = float(np.sqrt(np.mean(inf_func**2) / n))

    return att, se, inf_func


# ======================================================================
# Nuisance estimators
# ======================================================================


class _PscoreTrimTally:
    """Count control units zeroed by propensity trimming, across all (g,t).

    Trimming that silently removes a third of the control arm is a very
    different object from trimming that removes nobody, and the caller
    cannot tell the two apart from the point estimate. The tally is
    surfaced in ``diagnostics['n_pscore_trimmed']`` (§7: no silent
    degradation).

    Not thread-safe by construction, and does not need to be: it is reset
    at the top of each ``callaway_santanna`` call and read at the bottom,
    both on the calling thread.
    """

    def __init__(self) -> None:
        self.n_trimmed = 0
        self.n_controls = 0

    def reset(self) -> None:
        self.n_trimmed = 0
        self.n_controls = 0

    def record(self, keep: np.ndarray, d: np.ndarray) -> None:
        is_control = d == 0
        self.n_controls += int(is_control.sum())
        self.n_trimmed += int((is_control & (keep == 0.0)).sum())


_TRIM_TALLY = _PscoreTrimTally()


def _require_pscore_trim(value: Any) -> float:
    """Validate ``pscore_trim`` — a probability cutoff in (0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise MethodIncompatibility(
            f"`pscore_trim` must be a float in (0, 1], got " f"{type(value).__name__}.",
            recovery_hint="Pass pscore_trim=0.995 (the default) or 1.0 to "
            "disable control trimming.",
            diagnostics={"pscore_trim": repr(value)},
        )
    value = float(value)
    if not np.isfinite(value) or value <= 0.0 or value > 1.0:
        raise MethodIncompatibility(
            f"`pscore_trim` must lie in (0, 1], got {value}.",
            recovery_hint="0.995 matches R `did`/Stata `csdid`; 1.0 disables "
            "control trimming.",
            diagnostics={"pscore_trim": value},
        )
    return value


def _pscore_trim_mask(
    pscore: np.ndarray,
    d: np.ndarray,
    trim: float,
) -> np.ndarray:
    """Zero-weight control units whose propensity score is at/above ``trim``.

    Follows ``DRDID``'s convention exactly (and hence R ``did`` and Stata
    ``csdid``, both of which inherit it)::

        trim.ps       <- (ps < 1.01)                 # treated: never trimmed
        trim.ps[D==0] <- (ps[D==0] < trim.level)     # controls: ps >= level out

    A control with p(X) → 1 looks almost exactly like a treated unit, so its
    odds weight p/(1−p) explodes and a single observation can dominate the
    control arm. Trimming bounds that leverage.

    ``trim=1.0`` disables control trimming (nothing is ``>= 1.0`` after the
    ``1 − 1e-6`` clip in :func:`_estimate_pscore`).
    """
    keep = np.ones_like(pscore, dtype=float)
    is_control = d == 0
    keep[is_control] = (pscore[is_control] < trim).astype(float)
    return keep


def _weighted_control_mean(
    dy: np.ndarray,
    c_mask: np.ndarray,
    w: np.ndarray,
) -> float:
    """ω-weighted mean of ``dy`` over the control arm (Hájek form).

    At ω ≡ 1 this defers to ``np.mean``. The two are algebraically the
    same, but ``np.dot`` and ``np.mean`` sum in different orders and the
    result differs by ~1 ULP — enough to move pinned parity fixtures, so
    the unweighted path keeps the original call.
    """
    w_c = w[c_mask]
    if np.allclose(w_c, 1.0):
        return float(np.mean(dy[c_mask]))
    denom = float(w_c.sum())
    if denom <= 0:
        return 0.0
    return float(np.dot(w_c, dy[c_mask]) / denom)


def _estimate_pscore(
    d: np.ndarray,
    x: Optional[np.ndarray],
    n: int,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate propensity score P(D=1 | X) via (weighted) logit.

    Uses statsmodels (core dep) — no sklearn needed.
    Falls back to unconditional probability if logit fails or no covariates.

    With ω supplied the logit is fit by weighted maximum likelihood, which
    is what R ``DRDID`` does via ``glm(..., weights = i.weights)``. The
    unweighted branch keeps the original ``sm.Logit`` call verbatim so the
    existing R/Stata parity fixtures stay bit-identical.
    """
    weighted = w is not None and not np.allclose(w, 1.0)
    p_d = np.mean(d) if not weighted else float(np.mean(w * d))
    if x is None or x.shape[1] == 0:
        return np.full(n, p_d)

    try:
        import statsmodels.api as sm

        x_const = sm.add_constant(x)
        if weighted:
            glm = sm.GLM(
                d,
                x_const,
                family=sm.families.Binomial(),
                freq_weights=np.asarray(w, dtype=float),
            )
            result = glm.fit(maxiter=500)
        else:
            logit = sm.Logit(d, x_const)
            result = logit.fit(disp=0, maxiter=500, warn_convergence=False)
        pscore = np.asarray(result.predict(x_const), dtype=float)
    except Exception as exc:
        # Covariates were supplied (checked above) but the logit failed:
        # falling back to a constant propensity silently turns IPW/DR into a
        # plain reweighting. Warn (CLAUDE.md §3.7).
        warnings.warn(
            "callaway_santanna: the covariate propensity-score logit failed "
            f"({type(exc).__name__}: {exc}); this ATT(g,t) falls back to a "
            "CONSTANT (unconditional) propensity, so IPW/DR reweighting is "
            "no longer covariate-adjusted. Check for separated or collinear "
            "covariates.",
            ConvergenceWarning,
            stacklevel=2,
        )
        pscore = np.full(n, p_d)

    return np.clip(pscore, 1e-6, 1 - 1e-6)


def _estimate_outcome_reg(
    dy: np.ndarray,
    c: np.ndarray,
    x: Optional[np.ndarray],
    n: int,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate E[ΔY | X, D=0] via (weighted) least squares on the controls."""
    c_mask = c.astype(bool)
    c_count = c_mask.sum()
    w = np.ones(n, dtype=float) if w is None else np.asarray(w, dtype=float)

    if x is None or x.shape[1] == 0 or c_count < 2:
        m0 = _weighted_control_mean(dy, c_mask, w) if c_count > 0 else 0.0
        return np.full(n, m0)

    k = x.shape[1]
    if c_count <= k + 1:
        # Not enough control obs for regression
        return np.full(n, _weighted_control_mean(dy, c_mask, w))

    try:
        import statsmodels.api as sm

        x_const = sm.add_constant(x[c_mask])
        ols = sm.WLS(dy[c_mask], x_const, weights=w[c_mask])
        result = ols.fit()
        m_hat = np.asarray(result.predict(sm.add_constant(x)), dtype=float)
    except Exception as exc:
        # Covariates supplied but the control OLS failed: the control-mean
        # fallback drops the covariate adjustment silently. Warn (§3.7).
        warnings.warn(
            "callaway_santanna: the covariate outcome regression E[dY|X,D=0] "
            f"failed ({type(exc).__name__}: {exc}); this ATT(g,t) falls back "
            "to the UNADJUSTED control mean and is no longer covariate-"
            "adjusted. Check for collinear or degenerate covariates.",
            ConvergenceWarning,
            stacklevel=2,
        )
        m_hat = np.full(n, _weighted_control_mean(dy, c_mask, w))

    return m_hat


# ======================================================================
# Aggregation
# ======================================================================


def _aggregate_simple(
    post_detail: pd.DataFrame,
    post_inf: Optional[np.ndarray],
    cohort_sizes: pd.Series,
    n_total: int,
    alpha: float,
    boot_cfg: Optional[Dict[str, Any]] = None,
    unit_cohorts: Optional[np.ndarray] = None,
    unit_weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, Tuple[float, float]]:
    """Simple aggregation: group-size-weighted average of post-treatment ATTs.

    When ``boot_cfg`` is given (bstrap=True path), the SE of the
    aggregated ATT comes from the multiplier bootstrap on the aggregated
    influence function instead of the analytic plug-in.

    ``unit_cohorts`` (row-aligned with ``post_inf``) enables the
    weight-estimation influence term — the aggregation weights are
    estimated cohort shares, and treating them as fixed understates the
    SE.  See :func:`statspai.did._core.weight_influence`.
    """
    if len(post_detail) == 0:
        return 0.0, np.inf, 1.0, (np.nan, np.nan)

    weights = post_detail["group"].map(cohort_sizes).values.astype(float)
    weights = weights / weights.sum()

    att_agg = float(np.average(post_detail["att"].values, weights=weights))

    if post_inf is not None and post_inf.shape[1] > 0:
        inf_agg = post_inf @ weights
        if unit_cohorts is not None and len(unit_cohorts) == post_inf.shape[0]:
            pg, ind = _cohort_share_context(
                post_detail["group"].values, unit_cohorts, unit_weights
            )
            inf_agg = inf_agg + _weight_influence(pg, ind) @ (
                post_detail["att"].values.astype(float)
            )
        if boot_cfg is not None:
            se_arr, _ = _core_multiplier_bootstrap(
                inf_agg,
                n_total,
                alpha,
                boot_cfg["n_boot"],
                boot_cfg["random_state"],
                weight_type=boot_cfg["weight_type"],
                cluster_ids=boot_cfg["cluster_ids"],
            )
            se_agg = float(se_arr[0])
        else:
            se_agg = float(np.sqrt(np.mean(inf_agg**2) / n_total))
    else:
        se_agg = float(
            np.sqrt(np.average(post_detail["se"].values ** 2, weights=weights))
        )

    z = att_agg / se_agg if se_agg > 0 else 0
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (att_agg - z_crit * se_agg, att_agg + z_crit * se_agg)

    return att_agg, se_agg, pval, ci


def _aggregate_event_study(
    detail: pd.DataFrame,
    inf_matrix: Optional[np.ndarray],
    cohort_sizes: pd.Series,
    n_total: int,
    alpha: float,
    boot_cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Event study aggregation: average ATT by relative time e = t − g.

    When ``boot_cfg`` is given (bstrap=True path), the per-event-time SEs
    come from a joint multiplier bootstrap over all event-time
    combinations instead of the analytic plug-in.
    """
    relative_times = sorted(detail["relative_time"].unique())
    z_crit = stats.norm.ppf(1 - alpha / 2)

    rows = []
    psi_cols: List[np.ndarray] = []
    for e in relative_times:
        mask = detail["relative_time"] == e
        sub = detail[mask]
        if len(sub) == 0:
            continue

        weights = sub["group"].map(cohort_sizes).values.astype(float)
        w_sum = weights.sum()
        if w_sum == 0:
            continue
        weights = weights / w_sum

        att_e = float(np.average(sub["att"].values, weights=weights))

        if inf_matrix is not None:
            col_idx = np.where(mask.values)[0]
            inf_e = inf_matrix[:, col_idx] @ weights
            se_e = float(np.sqrt(np.mean(inf_e**2) / n_total))
            psi_cols.append(inf_e)
        else:
            se_e = float(np.sqrt(np.average(sub["se"].values ** 2, weights=weights)))

        pval = float(2 * (1 - stats.norm.cdf(abs(att_e / se_e)))) if se_e > 0 else 1.0

        rows.append(
            {
                "relative_time": e,
                "att": att_e,
                "se": se_e,
                "ci_lower": att_e - z_crit * se_e,
                "ci_upper": att_e + z_crit * se_e,
                "pvalue": pval,
            }
        )

    es = pd.DataFrame(rows)

    # Bootstrap SEs (one joint multiplier pass over all event times) —
    # keeps the convenience event study consistent with the bstrap=True
    # detail SEs.  The canonical event study remains sp.aggte('dynamic').
    if boot_cfg is not None and psi_cols and len(es) == len(psi_cols):
        se_vec, _ = _core_multiplier_bootstrap(
            np.column_stack(psi_cols),
            n_total,
            alpha,
            boot_cfg["n_boot"],
            boot_cfg["random_state"],
            weight_type=boot_cfg["weight_type"],
            cluster_ids=boot_cfg["cluster_ids"],
        )
        att_vals = es["att"].values
        es["se"] = se_vec
        with np.errstate(divide="ignore", invalid="ignore"):
            z_stat = np.where(se_vec > 0, att_vals / se_vec, 0.0)
        es["pvalue"] = np.where(
            se_vec > 0, 2 * (1 - stats.norm.cdf(np.abs(z_stat))), 1.0
        )
        es["ci_lower"] = att_vals - z_crit * se_vec
        es["ci_upper"] = att_vals + z_crit * se_vec

    return es


# ======================================================================
# Pre-trend test
# ======================================================================


def _pretrend_test(
    detail: pd.DataFrame,
    inf_matrix: Optional[np.ndarray],
    n_total: int,
    pretest: str = "joint",
    pretest_periods: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Joint Wald test for H0: all pre-treatment ATT(g,t) = 0.

    ``pretest_periods=k`` restricts the test to the ``k`` event times
    closest to treatment. Deep leads typically rest on few cohorts, so
    including them adds degrees of freedom faster than signal and pushes
    the test toward non-rejection — the direction that flatters the
    design.
    """
    if pretest == "none":
        return None

    pre_mask = (detail["relative_time"] < 0).values
    if pretest_periods is not None:
        rel = detail["relative_time"].values
        pre_times = sorted({int(e) for e in rel[pre_mask]})
        keep = set(pre_times[-pretest_periods:])
        pre_mask = pre_mask & np.array([int(e) in keep for e in rel])
    pre_mask = pd.Series(pre_mask, index=detail.index)
    pre = detail[pre_mask]

    if len(pre) == 0:
        return {"statistic": np.nan, "df": 0, "pvalue": np.nan}

    theta = pre["att"].values
    k = len(theta)

    G = None
    if inf_matrix is not None:
        col_idx = np.where(pre_mask.values)[0]
        inf_pre = inf_matrix[:, col_idx]
        # Variance-covariance: V = (1/n²) IF' IF
        V = inf_pre.T @ inf_pre / (n_total**2)
        G = int(inf_pre.shape[0])  # number of IF contributions (units)
    else:
        V = np.diag(pre["se"].values ** 2)

    # Regularise for numerical stability
    V += np.eye(k) * 1e-10

    try:
        V_inv = np.linalg.inv(V)
        W = float(theta @ V_inv @ theta)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)
        W = float(theta @ V_inv @ theta)

    # The pre-period ATT(g,t) are strongly correlated (shared base period and
    # control group), and V is *estimated*, so the plug-in chi²(k) Wald
    # over-rejects in finite samples (≈0.14 at a nominal 5% level for ~60
    # units, vs the Callaway–Sant'Anna multiplier-bootstrap pre-test which is
    # correctly sized). Apply the Hotelling T² finite-sample correction:
    #   F = W · (G − k) / (k · (G − 1))  ~  F(k, G − k)
    # which is exact under (asymptotically) normal influence functions and
    # converges to chi²(k)/k as G → ∞. Falls back to chi² when G is unknown
    # or too small relative to k.
    if G is not None and G > k + 1:
        f_stat = W * (G - k) / (k * (G - 1))
        pvalue = float(stats.f.sf(f_stat, k, G - k))
    else:
        pvalue = float(stats.chi2.sf(W, k))

    return {"statistic": W, "df": k, "pvalue": pvalue}


# ======================================================================
# Repeated cross-sections (panel=False) branch
# ======================================================================


def _estimate_single_att_rcs_sz(
    y_arr: np.ndarray,
    g_arr: np.ndarray,
    t_arr: np.ndarray,
    x_mat: Optional[np.ndarray],
    *,
    g_val: int,
    t_val: int,
    base_val: int,
    estimator: str,
    control_group: str,
    n_obs: int,
) -> Tuple[float, float, np.ndarray]:
    """Sant'Anna-Zhao ATT(g, t) for one (g, t) cell of a repeated cross-section.

    Builds the two-period sub-sample — observations in period ``base_val`` or
    ``t_val`` belonging to cohort ``g_val`` or the control group — and hands it
    to the matching :mod:`statspai.did._rcs` estimator.  The cell-level
    influence function is then embedded in the full ``n_obs`` universe with the
    ``n_obs / n_rel`` rescaling the panel path uses, so cross-(g, t)
    aggregation in :func:`sp.aggte` sees influence functions on a common scale.
    """
    from ._rcs import drdid_rc, reg_did_rc, std_ipw_did_rc

    is_treated = g_arr == g_val
    if control_group == "nevertreated":
        is_control = g_arr == 0
    else:  # notyettreated
        is_control = (g_arr == 0) | (g_arr > t_val)

    in_period = (t_arr == t_val) | (t_arr == base_val)
    relevant = (is_treated | is_control) & in_period
    n_rel = int(relevant.sum())
    if n_rel < 5:
        return 0.0, np.inf, np.zeros(n_obs)

    idx = np.where(relevant)[0]
    d_sub = is_treated[relevant].astype(float)
    post_sub = (t_arr[relevant] == t_val).astype(float)
    y_sub = y_arr[relevant]
    x_sub = None if x_mat is None else x_mat[relevant]

    # All four treatment x period cells must be populated; the RCS estimators
    # raise otherwise, which for a single (g, t) cell should degrade to "no
    # estimate" rather than kill the whole fit.
    fn = {"dr": drdid_rc, "ipw": std_ipw_did_rc, "reg": reg_did_rc}[estimator]
    try:
        res = fn(y_sub, post_sub, d_sub, x_sub)
    except (DataInsufficient, MethodIncompatibility, np.linalg.LinAlgError):
        return 0.0, np.inf, np.zeros(n_obs)
    if not np.isfinite(res.att) or not np.isfinite(res.se):
        return 0.0, np.inf, np.zeros(n_obs)

    inf_full = np.zeros(n_obs)
    inf_full[idx] = res.influence * (n_obs / n_rel)
    return float(res.att), float(res.se), inf_full


def _callaway_santanna_rcs(
    data: pd.DataFrame,
    y: str,
    g: str,
    t: str,
    base_period: str,
    anticipation: int,
    alpha: float,
    x: Optional[List[str]] = None,
    estimator: str = "reg",
    control_group: str = "nevertreated",
    pretest: str = "joint",
    pretest_periods: Optional[int] = None,
    unit_col: Optional[str] = None,
) -> CausalResult:
    """Unconditional (or regression-adjusted) 2×2 cell-mean DID for RCS.

    For each (g, t) pair with base period b:

        ATT(g, t) = (Ȳ_{g,t} - Ȳ_{g,b}) - (Ȳ_{c,t} - Ȳ_{c,b})

    where c = never-treated cohort.  Observation-level influence
    functions are assembled as

        ψ_i =  1{G_i=g, T_i=t}  (Y_i - Ȳ_{g,t}) / p_{g,t}
            -  1{G_i=g, T_i=b}  (Y_i - Ȳ_{g,b}) / p_{g,b}
            -  1{G_i=c, T_i=t}  (Y_i - Ȳ_{c,t}) / p_{c,t}
            +  1{G_i=c, T_i=b}  (Y_i - Ȳ_{c,b}) / p_{c,b}

    with ``p_{g,t} = #{i: G_i=g, T_i=t} / n``.  SE(ATT) is the sample
    variance of ψ divided by ``n``, matching CS2021 eqn (2.4) for RCS.

    ``unit_col`` drives the unbalanced-panel route
    (``allow_unbalanced_panel=True``). The estimators are unchanged —
    observations are still not matched across time — but the influence
    functions are folded to the unit level by summation and ``n`` becomes
    the number of units rather than the number of rows, mirroring R
    ``did``'s ``.rowid <- idname`` assignment for this case. That fold is
    what lets the SEs pick up the within-unit correlation that a true
    repeated cross-section does not have. With ``unit_col=None`` (a true
    repeated cross-section) the fold is the identity.
    """
    df = data.copy()
    _require_columns(df, (y, g, t, *(x or [])), function="_callaway_santanna_rcs")
    if unit_col is not None:
        _require_columns(df, (unit_col,), function="_callaway_santanna_rcs")

    df[g] = df[g].fillna(0).replace([np.inf, -np.inf], 0).astype(int)
    drop_cols = [y, t] + (list(x) if x else [])
    df = df.dropna(subset=drop_cols).reset_index(drop=True)
    n_obs = len(df)
    if n_obs == 0:
        raise DataInsufficient(
            "No observations after dropping NaNs.",
            recovery_hint=(
                "Drop fewer rows or impute outcome/time/covariate data before "
                "RCS DID."
            ),
            diagnostics={"function": "_callaway_santanna_rcs"},
        )

    # Unbalanced-panel fold. `unit_codes` maps each row to its unit's slot
    # in the influence-function matrix; `n_scale` is the `n` that scales
    # every influence function and every SE (R: `n <- length(unique(
    # data[, idname]))`, with idname = .rowid = the unit id here).
    if unit_col is not None:
        unit_codes, unit_uniques = pd.factorize(df[unit_col], sort=True)
        n_scale = int(len(unit_uniques))
        if n_scale < 2:
            raise DataInsufficient(
                f"allow_unbalanced_panel=True needs at least 2 units, "
                f"found {n_scale}.",
                recovery_hint=(
                    "Check the unit identifier `i`, or pass panel=False to "
                    "treat the data as true repeated cross-sections."
                ),
                diagnostics={"function": "_callaway_santanna_rcs"},
            )
    else:
        unit_codes, unit_uniques = None, None
        n_scale = n_obs

    def _fold(inf_obs: np.ndarray) -> np.ndarray:
        """Rows → influence-function slots, summing within unit.

        The per-cell influence function arrives scaled by ``n_obs / n_cell``
        (the RCS convention); rescaling by ``n_scale / n_obs`` restores R's
        ``n / n1`` before the ``rowsum`` fold.
        """
        if unit_codes is None:
            return inf_obs
        out = np.zeros(n_scale, dtype=float)
        np.add.at(out, unit_codes, inf_obs)
        return out * (n_scale / n_obs)

    # Covariate adjustment: residualise Y on X using the never-treated
    # pool with period fixed effects. Plug-in influence functions treat
    # β̂ as known (asymptotically valid; see Sant'Anna & Zhao 2020).
    y_series = df[y].astype(float).to_numpy().copy()
    covariate_info: Optional[Dict[str, Any]] = None
    if x:
        y_series = _rcs_residualise_on_controls(y_series, df, g, t, x)
        covariate_info = {
            "covariates": list(x),
            "approach": "residualisation on never-treated with period FEs",
        }

    time_periods = sorted(df[t].unique())
    t_max = max(time_periods)
    cohorts = sorted([v for v in df[g].unique() if v > 0 and v <= t_max])
    if not cohorts:
        raise DataInsufficient(
            "No treatment cohorts found.",
            recovery_hint=(
                "Encode first treatment periods in `g`, using 0 for "
                "never-treated observations."
            ),
            diagnostics={"function": "_callaway_santanna_rcs"},
        )

    gt_pairs = _get_gt_pairs(cohorts, time_periods, base_period, anticipation)
    if not gt_pairs:
        raise DataInsufficient(
            "No valid (group, time) pairs to estimate.",
            recovery_hint=(
                "Check treatment timing, base_period, anticipation, and "
                "available periods."
            ),
            diagnostics={
                "function": "_callaway_santanna_rcs",
                "cohorts": cohorts,
                "time_periods": time_periods,
                "base_period": base_period,
                "anticipation": anticipation,
            },
        )

    y_arr = y_series  # possibly residualised (legacy cell-mean path)
    y_raw_arr = df[y].astype(float).to_numpy()  # Sant'Anna-Zhao path
    g_arr = df[g].values
    t_arr = df[t].values

    gt_results: List[Dict[str, Any]] = []
    inf_funcs_list: List[np.ndarray] = []
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # The Sant'Anna-Zhao RCS estimators (matching R did's panel=FALSE path)
    # consume raw covariates directly; the legacy cell-mean path residualises
    # first and then differences means, so the two must not be mixed.
    x_mat = df[list(x)].to_numpy(dtype=float) if x else None
    use_sz = estimator in {"dr", "ipw"} or (estimator == "reg" and bool(x))

    for g_val, t_val, base_val in gt_pairs:
        if use_sz:
            att, se, inf_func = _estimate_single_att_rcs_sz(
                y_raw_arr,
                g_arr,
                t_arr,
                x_mat,
                g_val=g_val,
                t_val=t_val,
                base_val=base_val,
                estimator=estimator,
                control_group=control_group,
                n_obs=n_obs,
            )
        else:
            att, se, inf_func = _estimate_single_att_rcs(
                y_arr,
                g_arr,
                t_arr,
                g_val=g_val,
                t_val=t_val,
                base_val=base_val,
                n_obs=n_obs,
            )
        if unit_codes is not None:
            # Fold to units, then re-derive the SE from the folded function.
            # Taking the per-cell RCS SE at face value here would treat a
            # unit's pre and post rows as independent draws and understate
            # the variance.
            inf_func = _fold(inf_func)
            if np.isfinite(se):
                se = float(np.sqrt(np.mean(inf_func**2) / n_scale))
        pval = float(2 * (1 - stats.norm.cdf(abs(att / se)))) if se > 0 else 1.0
        gt_results.append(
            {
                "group": g_val,
                "time": t_val,
                "att": att,
                "se": se,
                "ci_lower": att - z_crit * se,
                "ci_upper": att + z_crit * se,
                "pvalue": pval,
                "relative_time": t_val - g_val,
            }
        )
        inf_funcs_list.append(inf_func)

    detail = pd.DataFrame(gt_results)
    inf_matrix = np.column_stack(inf_funcs_list) if inf_funcs_list else None

    # Cohort sizes for aggregation weights, and the cohort label attached to
    # each influence-function row. Both live in the same space as the
    # influence functions: units under the unbalanced-panel fold, rows for a
    # true repeated cross-section (where a "unit" is one observation).
    if unit_codes is not None:
        cohort_by_slot = np.zeros(n_scale, dtype=g_arr.dtype)
        cohort_by_slot[unit_codes] = g_arr
    else:
        cohort_by_slot = g_arr
    cohort_sizes = pd.Series(
        {g_val: int((cohort_by_slot == g_val).sum()) for g_val in cohorts}
    )

    post_mask = detail["relative_time"] >= 0
    agg_est, agg_se, agg_pval, agg_ci = _aggregate_simple(
        detail[post_mask],
        inf_matrix[:, post_mask.values] if inf_matrix is not None else None,
        cohort_sizes,
        n_scale,
        alpha,
        unit_cohorts=cohort_by_slot,
    )
    event_study = _aggregate_event_study(
        detail,
        inf_matrix,
        cohort_sizes,
        n_scale,
        alpha,
    )
    pretrend = _pretrend_test(
        detail, inf_matrix, n_scale, pretest=pretest, pretest_periods=pretest_periods
    )

    unbalanced = unit_codes is not None
    model_info: Dict[str, Any] = {
        "estimator": (
            f"{estimator.upper()} (RCS, Sant'Anna-Zhao)"
            if use_sz
            else "REG (RCS, cell-mean)" + (" + covariates" if x else "")
        )
        + (" — unbalanced panel" if unbalanced else ""),
        "control_group": control_group,
        "base_period": base_period,
        "anticipation": anticipation,
        # An unbalanced panel keeps its panel identity: the estimator is the
        # RCS one, but the influence functions are indexed by unit.
        "panel": bool(unbalanced),
        "allow_unbalanced_panel": unbalanced,
        "n_units": n_scale,  # the "n" for aggte / the bootstrap
        "n_obs": n_obs,
        "n_periods": len(time_periods),
        "n_cohorts": len(cohorts),
        "cohorts": cohorts,
        "event_study": event_study,
        "pretrend_test": pretrend,
        "cohort_sizes": cohort_sizes,
        # Private plumbing for sp.influence_functions. Under the unbalanced
        # fold a row is a unit; for a true RCS it is an observation.
        "_cluster_ids": None,
        "_unit_ids": (np.asarray(unit_uniques) if unbalanced else df.index.to_numpy()),
        "_unit_cohorts": cohort_by_slot,
    }
    if covariate_info is not None:
        model_info.update(covariate_info)

    return CausalResult(
        method=(
            "Callaway and Sant'Anna (2021) — unbalanced panel"
            if unbalanced
            else "Callaway and Sant'Anna (2021) — repeated cross-sections"
        ),
        estimand="ATT",
        estimate=agg_est,
        se=agg_se,
        pvalue=agg_pval,
        ci=agg_ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info=model_info,
        _influence_funcs=inf_matrix,
        _citation_key="callaway_santanna",
    )


def _estimate_single_att_rcs(
    y_arr: np.ndarray,
    g_arr: np.ndarray,
    t_arr: np.ndarray,
    g_val: int,
    t_val: int,
    base_val: int,
    n_obs: int,
) -> Tuple[float, float, np.ndarray]:
    """Observation-level 2×2 cell-mean DID + influence function."""
    m_gt = (g_arr == g_val) & (t_arr == t_val)
    m_gb = (g_arr == g_val) & (t_arr == base_val)
    m_ct = (g_arr == 0) & (t_arr == t_val)
    m_cb = (g_arr == 0) & (t_arr == base_val)

    # Any empty cell kills the estimator for this (g, t).
    for m in (m_gt, m_gb, m_ct, m_cb):
        if m.sum() < 2:
            return 0.0, np.inf, np.zeros(n_obs)

    mu_gt = y_arr[m_gt].mean()
    mu_gb = y_arr[m_gb].mean()
    mu_ct = y_arr[m_ct].mean()
    mu_cb = y_arr[m_cb].mean()

    att = float((mu_gt - mu_gb) - (mu_ct - mu_cb))

    p_gt = m_gt.sum() / n_obs
    p_gb = m_gb.sum() / n_obs
    p_ct = m_ct.sum() / n_obs
    p_cb = m_cb.sum() / n_obs

    inf = np.zeros(n_obs)
    inf[m_gt] += (y_arr[m_gt] - mu_gt) / p_gt
    inf[m_gb] += -(y_arr[m_gb] - mu_gb) / p_gb
    inf[m_ct] += -(y_arr[m_ct] - mu_ct) / p_ct
    inf[m_cb] += (y_arr[m_cb] - mu_cb) / p_cb

    # SE from sample variance of the influence function.
    se = float(np.sqrt(np.mean(inf**2) / n_obs))
    return att, se, inf


def _rcs_residualise_on_controls(
    y_arr: np.ndarray,
    df: pd.DataFrame,
    g_col: str,
    t_col: str,
    x_cols: List[str],
) -> np.ndarray:
    """Fit Y = Xβ + period-FE on never-treated observations; return
    Y − X'β̂ for every observation (treated + control).

    The period FEs absorb the unconditional time pattern in the control
    group, so after residualisation the remaining cross-period mean
    movement in the control cells is zero and the RCS DID reduces to a
    covariate-adjusted comparison, matching the "outcome regression"
    flavour of Sant'Anna & Zhao (2020) adapted to repeated cross-sections.
    Influence functions downstream treat β̂ as known; asymptotically
    negligible at √n.
    """
    y_arr = np.asarray(y_arr, dtype=float)
    g_arr = df[g_col].values
    t_arr = df[t_col].values
    x_mat = df[x_cols].to_numpy(dtype=float)

    # Keep only observations with finite covariates; the Y slot is
    # already clean from the upstream dropna.
    control = g_arr == 0
    if control.sum() < x_mat.shape[1] + 2:
        # Not enough controls to fit; return untouched Y.
        return y_arr

    # Build the control design: covariates + period dummies.
    periods = sorted(np.unique(t_arr[control]))
    X_ctrl = x_mat[control]
    t_ctrl = t_arr[control]
    period_dummies_ctrl = np.column_stack(
        [(t_ctrl == p).astype(float) for p in periods]
    )
    design_ctrl = np.column_stack([X_ctrl, period_dummies_ctrl])

    y_ctrl = y_arr[control]
    try:
        beta, *_ = np.linalg.lstsq(design_ctrl, y_ctrl, rcond=None)
    except np.linalg.LinAlgError:
        return y_arr

    beta_x = beta[: x_mat.shape[1]]
    # For residualisation of every observation (including treated), we
    # only subtract the X contribution.  The period FE absorbs only the
    # control group's period mean, which is exactly what we want to
    # leave inside Y for the treated cell.
    return np.asarray(y_arr - x_mat @ beta_x, dtype=float)
