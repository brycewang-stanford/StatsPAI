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
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core._call_record import records_call
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core._vcov_spec import accepts_vce
from ..core.results import CausalResult
from ..exceptions import (
    AssumptionWarning,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from . import _dr_common as _drc
from ._core import calendar_time_aware as _calendar_time_aware
from ._core import cohort_share_context as _cohort_share_context
from ._core import covariates_from_formula as _covariates_from_formula
from ._core import drop_unusable_rows as _drop_unusable_rows
from ._core import influence_se_did
from ._core import multiplier_bootstrap as _core_multiplier_bootstrap
from ._core import normalize_se_method as _normalize_se_method
from ._core import parallel_trends_block as _pt_block
from ._core import require_bool as _require_bool
from ._core import weight_influence as _weight_influence
from ._cs_rcs import cs_repeated_cross_sections as _callaway_santanna_rcs
from ._cs_rcs import gt_cells as _get_gt_pairs
from ._validate import inference_note as _inference_note
from ._validate import require_constant_cohort as _require_constant_cohort

#: Below this many treated clusters the cluster-robust and multiplier-
#: bootstrap variances over-reject whatever the total cluster count
#: (Conley-Taber 2011; Ferman-Pinto 2019). ``sp.audit`` uses the same cutoff.
_FEW_TREATED_UNITS = 10

#: A never-treated comparison group with fewer than this many units, plus one
#: per covariate, is the size below which R ``did`` (>= 2.5.0) refuses
#: ``control_group = "nevertreated"``. StatsPAI estimates and warns.
_FEW_NEVER_TREATED_UNITS = 5


def few_never_treated_cutoff(x: Optional[Sequence[str]]) -> int:
    """Smallest never-treated group R ``did`` accepts as the comparison."""
    return _FEW_NEVER_TREATED_UNITS + (len(x) if x else 0)


class CallawayNotImplemented(MethodIncompatibility, NotImplementedError):
    """Unsupported CS branch that preserves historical NotImplementedError."""


#: Propensity-score trimming cutoff for *control* units, matching
#: ``DRDID``'s ``trim.level``, which R ``did`` inherits without overriding
#: it. Stata ``csdid`` (1.81) does override it: its default is
#: ``pscoretrim(1)``, no trimming, so a script ported from ``csdid`` needs
#: ``pscore_trim=1.0`` to give the same cells when a control unit's
#: propensity score exceeds 0.995.
_DEFAULT_PSCORE_TRIM = 0.995

#: ATT(g,t) estimators, mapped to the names the reference packages use.
#: ``'ipw'``/``'stdipw'`` are the SAME estimator: R ``did``'s
#: ``est_method='ipw'`` dispatches to ``DRDID::std_ipw_did_panel``, which is
#: Stata's ``method(stdipw)``. Stata's ``method(ipw)`` is Abadie (2005) and
#: is reached here as ``'ipw_abadie'``. ``'drimp'`` is the improved doubly
#: robust estimator (``DRDID::drdid_imp_panel``, Stata's ``method(drimp)``).
_ESTIMATORS = ("dr", "drimp", "ipw", "stdipw", "ipw_abadie", "reg")


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


@records_call("callaway_santanna")
@accepts_aliases(
    _strict=True,
    id="i",
    unit="i",
    time="t",
    first_treat="g",
    cohort="g",
    covariates="x",
    controls="x",
    # House-style cluster= (every other DiD estimator); clustervars= is
    # R did::att_gt's spelling and stays the parameter name.
    cluster="clustervars",
    seed="random_state",
    # R did::att_gt spellings, so a call copied from R runs unchanged.
    yname="y",
    tname="t",
    idname="i",
    gname="g",
    xformla="x",
    weightsname="weights",
    est_method="estimator",
    alp="alpha",
)
@accepts_vce(
    default=None, cluster="clustervars", native={"bootstrap": {"bstrap": True}}
)
@_calendar_time_aware(time="t", cohort="g")
@_expands_categorical("x")
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
    balance: Optional[str] = None,
    design: Optional[Any] = None,
    variance_convention: Literal["delta", "reference"] = "delta",
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
    x : list of str or str, optional
        Covariates for conditional parallel trends, either as column
        names (``["lpop"]``) or as an R-style one-sided formula
        (``"~ lpop + I(lpop**2)"``), which is materialised into columns
        before estimation. A left-hand side is accepted and ignored, so
        an ``xformla`` copied straight out of R ``did`` / ``csdid``
        works: ``x="lemp ~ lpop"``.

        A plain string without ``~`` is still a single column name.

        Inside ``I(...)`` the expression is Python, so powers are
        ``I(x**2)``; an R ``I(x^2)`` is rejected rather than silently
        evaluated as bitwise XOR.

        A covariate that varies within unit is read, for each ATT(g, t),
        in the earlier of the cell's two periods (the base period of a
        post-treatment cell), as R ``did`` and Stata ``csdid`` do. A unit
        whose covariate is missing in that period leaves the cell. The
        estimator still assumes the covariate is not itself moved by
        treatment; :func:`did_timevarying_covariates` is for when it is.

        .. versionchanged:: 1.39.0
           ⚠️ Time-varying covariates used to be read from the unit's
           first row for every cell, which matched neither package.

        .. versionadded:: 1.23.0
           The formula spelling.
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
        - ``'drimp'`` — the *improved* doubly robust estimator of the same
          paper: the propensity score is fitted by inverse probability
          tilting and the outcome model by least squares weighted with the
          fitted odds, so the estimator is also doubly robust for
          inference and its influence function has no nuisance-estimation
          terms. ``DRDID::drdid_imp_panel`` (``drdid_imp_rc`` on repeated
          cross-sections); Stata ``csdid``'s ``method(drimp)``. Without
          covariates it is the same number as ``'dr'``.
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
        Comparison group: 'nevertreated' or 'notyettreated'. With
        'nevertreated' and fewer than five never-treated units plus one
        per covariate, an ``AssumptionWarning`` is raised and the
        estimate is still returned; R ``did`` (>= 2.5.0) stops there.
    notyet_cutoff : str, default 'period'
        Only consulted when ``control_group='notyettreated'``. Which date
        must a unit still be untreated at to serve as a control for
        ATT(g, t)?

        - ``'period'`` — untreated through both periods of the comparison,
          ``G > max(t, base) + anticipation``. R ``did`` (the authors'
          implementation). Under ``base_period='universal'`` a
          pre-treatment cell's base is ``g - 1``, so cohorts first treated
          between ``t`` and the base are excluded: their base-period
          outcome is already treated.
        - ``'asinr'`` — untreated as of ``t`` only, ``G > t``. Stata
          ``csdid, asinr``. Identical to ``'period'`` under
          ``base_period='varying'`` with no anticipation; under
          ``'universal'`` it keeps already-treated cohorts in pre-treatment
          controls.
        - ``'cohort'`` — ``G > max(t, g)``. Stata ``csdid``'s own default.

        All three coincide for post-treatment cells without anticipation.
        On ``mpdta`` (universal base) ATT(2007, 2004) is 0.033813 under
        ``'period'`` and ``'cohort'`` (R ``did``, Stata default) and
        0.032971 under ``'asinr'``.
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
        Cluster variable(s), mirroring R ``did::att_gt(clustervars=...)``
        and Stata ``csdid, cluster()``. The unit id ``i`` is always implied
        and may be included or omitted; at most **one** additional variable
        is allowed. The analytic SEs sum the influence functions within
        cluster (``sqrt(sum_c (sum_i psi_i)^2) / n``, csdid's analytic
        variance; equal to Stata ``csdid`` v1 to 1e-9) and the bootstrap
        draws one multiplier per cluster. On a panel the influence
        functions are per unit, so the cluster must be time-invariant; with
        ``panel=False`` (repeated cross-sections, as ``csdid`` / ``csdid2``
        without ``ivar()``) any row-level variable works, e.g. city x year.

        .. versionchanged:: 1.33.0
           Analytic SEs honour ``clustervars`` (it used to require
           ``bstrap=True``).
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

        .. versionchanged:: 1.23.0
           ``weights``, ``clustervars``, ``bstrap``, ``cband`` and
           ``boot_weight_type`` now work on this path (and on the
           ``allow_unbalanced_panel=True`` route); they previously
           raised. As on the panel path, ``clustervars`` requires
           ``bstrap=True``.
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
        They also rest on different assumptions about the missing cells.
        The default compares each unit with itself, so cells may go
        missing in any way that is unrelated to the unit's own change
        over time. ``True`` compares group means across periods, so it
        also needs the *composition* of each group to be stable: if
        units leave according to their level (high earners drop out of
        the treated group), the group mean moves for a reason that is
        not the treatment. In ``tests/reliability/unbalanced_panel.py``
        both cover at the nominal rate when cells are missing at random;
        when treated units with a high unit effect leave, the default
        still covers 94% and ``True`` is biased by 40% of the effect and
        covers 50% (100 units) and 3% (400 units).

        .. versionadded:: 1.23.0

    design : SurveyDesign, optional
        A complex survey design from :func:`sp.svydesign` declared on the
        long ``data`` (same rows), with the panel unit as the sampled
        element: its weight, stratum and PSU must be the same in every
        period. The design weights are the unit weights ``omega`` (as
        ``weights=``), and every analytic standard error -- ATT(g, t),
        the simple aggregate, the event study and the pre-trend Wald test
        -- is the survey linearisation of the unit-level influence
        functions over strata, PSUs and the finite population correction
        (R ``survey::svyrecvar`` on ``psi / n``), with t intervals on the
        design degrees of freedom (#PSU - #strata). Balanced panels only
        (no ``panel=False`` / unbalanced path), analytic SEs only (no
        multiplier bootstrap), and not with ``weights=`` or
        ``clustervars=``. ``sp.aggte`` / ``sp.honest_did`` on such a fit
        are refused rather than rebuilt without the design.
        A replicate-weight design (:func:`sp.svrepdesign`) is accepted
        too: the estimator is refitted with every replicate's weights
        and the replicate variance replaces the linearisation (R
        ``svrepdesign`` / Stata ``svy brr`` / ``svy jackknife``).

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

    balance : {'pair', 'full', 'none'}, optional
        How an unbalanced panel is used when ``panel=True``. The three
        rules are the ``bal()`` settings of Stata ``csdid`` 2.0.0 and give
        different numbers on the same data. On a balanced panel all three
        are the same estimator.

        - ``'pair'`` -- each ATT(g, t) uses the units observed in both its
          base and its comparison period, so the sample varies across
          cells. This is the default (``balance=None`` with
          ``allow_unbalanced_panel=False``) and the rule of Stata
          ``csdid`` 1.8x.
        - ``'full'`` -- drop every unit that is not observed in all
          periods, once, before estimation, and warn with the count. This
          is the default of R ``did`` (``allow_unbalanced_panel = FALSE``)
          and of Stata ``csdid`` 2.0.0; pass it to reproduce their numbers.
        - ``'none'`` -- keep every row and use the repeated-cross-section
          estimators, the same as ``allow_unbalanced_panel=True``.

        ``'pair'`` and ``'full'`` compare a unit with itself; ``'full'``
        does so on one fixed sample at the price of discarding the
        incomplete units (with 30% of cells missing at random over eight
        periods, about 6% of units are complete).
    variance_convention : {'delta', 'reference'}, default 'delta'
        Only for ``estimator='dr'`` on the repeated-cross-section path
        (``panel=False``, or an unbalanced panel with
        ``allow_unbalanced_panel=True``). ``'delta'`` uses the influence
        function of Sant'Anna and Zhao (2020, Appendix B). ``'reference'``
        reproduces R ``did`` / ``DRDID``, whose influence function carries
        the estimation effect of the pre-period comparison regression
        with the opposite sign; its standard errors are too small when the
        propensity model is misspecified (95% intervals covered 81% of the
        time in a Monte Carlo where ``'delta'`` covered 95%). The ATT(g, t)
        are the same under both.

        .. versionchanged:: 1.40.0
           ⚠️ The default was the R convention (``'reference'``).

    References
    ----------
    Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences
    with multiple time periods. *Journal of Econometrics*. [@callaway2021difference]
    """
    if design is not None and getattr(design, "is_replicate_design", False):
        _args = dict(locals())
        from ..survey._replicate_refit import replicate_refit

        return replicate_refit(
            callaway_santanna, _args, design, function="sp.callaway_santanna"
        )
    data = _require_dataframe(data, function="callaway_santanna")
    variance_convention = _require_string_option(
        variance_convention,
        argument="variance_convention",
        valid=("delta", "reference"),
    )  # type: ignore[assignment]
    y = _require_column_name(y, argument="y")
    g = _require_column_name(g, argument="g")
    t = _require_column_name(t, argument="t")
    i = _require_column_name(i, argument="i")
    svy: Any = None
    svy_data: Optional[pd.DataFrame] = None
    if design is not None:
        from ..survey.linearization import check_design

        svy = check_design(
            design,
            data,
            function="callaway_santanna",
            weights=weights,
            cluster=clustervars or None,
            allow_calibrated=False,
        )
        if not panel or bstrap or cband or se_method not in (None, "analytic"):
            raise MethodIncompatibility(
                "callaway_santanna: design= is implemented for the panel "
                "estimator with analytic standard errors (survey "
                "linearisation of the unit influence functions); not with "
                "panel=False or the multiplier bootstrap.",
                recovery_hint="Use panel=True and drop bstrap= / cband= / "
                "se_method=.",
                diagnostics={"panel": panel, "bstrap": bstrap},
            )
        from ..survey.linearization import unit_design_rows

        _require_columns(data, (i,), function="callaway_santanna")
        # the unit must be the sampled element (raises otherwise)
        unit_design_rows(
            svy, data, i, list(pd.unique(data[i])), function="callaway_santanna"
        )
        svy_data = data
        data = data.assign(__svy_weight__=svy.weights)
        weights = "__svy_weight__"
    # R-style one-sided covariate formula, for `did`/`csdid` migrations:
    # x="~ lpop + I(lpop**2)" is materialised into real columns here so the
    # rest of the estimator keeps seeing a plain list of column names. A
    # bare string without "~" stays a single column name, as before.
    if isinstance(x, str) and "~" in x:
        data, x = _covariates_from_formula(data, x, function="callaway_santanna")
        x = x or None
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
        valid=("period", "asinr", "cohort"),
    )
    pscore_trim = _require_pscore_trim(pscore_trim)
    # A non-default cutoff with never-treated controls is silently inert —
    # say so rather than let the caller believe it took effect (§7).
    if notyet_cutoff != "period" and control_group == "nevertreated":
        warnings.warn(
            f"callaway_santanna: notyet_cutoff={notyet_cutoff!r} only affects the "
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
    if balance is not None:
        balance = _require_string_option(
            balance, argument="balance", valid=("pair", "full", "none")
        )
        if allow_unbalanced_panel and balance != "none":
            raise MethodIncompatibility(
                f"callaway_santanna: balance={balance!r} and "
                "allow_unbalanced_panel=True ask for different estimators.",
                recovery_hint=(
                    "Pass one of them. allow_unbalanced_panel=True is "
                    "balance='none'."
                ),
                diagnostics={
                    "balance": balance,
                    "allow_unbalanced_panel": allow_unbalanced_panel,
                },
            )
        if not panel:
            raise MethodIncompatibility(
                "callaway_santanna: balance= applies to panel=True; with "
                "panel=False the rows are repeated cross-sections and there "
                "is nothing to balance.",
                recovery_hint="Drop balance=, or pass panel=True.",
                diagnostics={"balance": balance, "panel": panel},
            )
        allow_unbalanced_panel = balance == "none"
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
    data = _never_treated_beyond_horizon(data, g, t)
    if panel:
        # R `did` stops on a cohort that varies within a unit; the reshape
        # below would otherwise keep each unit's first row only.
        _require_constant_cohort(data, unit=i, cohort=g, function="callaway_santanna")
    balance_info: Optional[Dict[str, Any]] = None
    if balance == "full":
        data, balance_info = _balance_full(data, i, t)

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
    if extra_cluster is not None:
        n_clusters = int(data[extra_cluster].nunique(dropna=True))
        if n_clusters < 2:
            # The influence functions summed within one cluster are identically
            # zero: before this check the call returned SE ~1e-17 (gardner_did
            # and did_imputation already refused the same input).
            raise DataInsufficient(
                f"callaway_santanna: cluster variable {extra_cluster!r} takes "
                f"{n_clusters} value(s); a cluster-robust variance needs at "
                "least two clusters.",
                recovery_hint="Cluster on a variable with several values, or "
                "drop clustervars= to cluster on the unit.",
                diagnostics={"clustervar": extra_cluster, "n_clusters": n_clusters},
            )
    # Analytic SEs honour the cluster too: the influence functions are
    # summed within cluster (Stata csdid, cluster()).  On a panel the
    # influence functions live at the unit level, so the cluster must be
    # time-invariant; a repeated cross-section (panel=False) can cluster on
    # any row-level variable, e.g. city x year.
    if extra_cluster is not None and panel:
        nuniq = data.groupby(i)[extra_cluster].nunique(dropna=False)
        if (nuniq > 1).any():
            bad = nuniq[nuniq > 1].index.tolist()[:5]
            raise MethodIncompatibility(
                f"Cluster variable {extra_cluster!r} is time-varying within "
                f"unit (e.g. units {bad}) — the CS multiplier bootstrap "
                "requires a time-invariant cluster membership.",
                recovery_hint=(
                    "Use a time-invariant cluster variable (R did::mboot "
                    "imposes the same restriction), or treat the rows as "
                    "repeated cross-sections with panel=False (Stata "
                    "csdid / csdid2 without ivar()), where any row-level "
                    "cluster is allowed."
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
        # A handful of never-treated units carries every comparison. R `did`
        # stops below five units plus one per covariate (did >= 2.5.0; the
        # same guard never fired on did 2.3.0's default path). The estimate
        # is still defined, so it is reported, with the warning.
        _never_rows = (g_clean == 0).to_numpy()
        if panel:
            _n_never = int(data.loc[_never_rows, i].nunique())
        else:
            _n_never = int(_never_rows.sum() // max(int(data[t].nunique()), 1))
        _never_cutoff = few_never_treated_cutoff(x)
        if _n_never < _never_cutoff:
            warnings.warn(
                f"callaway_santanna: only {_n_never} never-treated "
                f"{'units' if panel else 'observations per period'} serve as "
                "the comparison group for every ATT(g, t). R `did` refuses "
                f"control_group='nevertreated' below {_never_cutoff} (five "
                "plus one per covariate). Each estimate rests on that many "
                "comparison units; consider control_group='notyettreated' "
                "and report sp.cs_jackknife (CV3) alongside.",
                AssumptionWarning,
                stacklevel=2,
            )

    # No never-treated units under 'notyettreated': R `did` (pre_process_did)
    # keeps only the periods before the last cohort's first treatment date
    # (minus anticipation) and drops that cohort from the estimated groups,
    # so it serves purely as a not-yet-treated comparison. Without this the
    # late cells have no comparison units at all.
    excluded_cohorts: List[Any] = []
    if control_group == "notyettreated":
        _g_num = (
            pd.to_numeric(data[g], errors="coerce")
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
        )
        _t_max = data[t].max()
        if not ((_g_num == 0) | (_g_num > _t_max)).any():
            latest_g = _g_num.max()
            cutoff_t = latest_g - anticipation
            n_rows = len(data)
            data = data.loc[data[t] < cutoff_t].copy()
            excluded_cohorts = [latest_g]
            warnings.warn(
                "callaway_santanna: no never-treated units, so the last "
                f"treated cohort (g={latest_g:g}) serves as the not-yet-treated "
                f"comparison group and periods t >= {cutoff_t:g} are dropped "
                f"({n_rows - len(data)} rows): after that date no unit is "
                "untreated. Its own ATT(g, t) are not estimated. This is R "
                "`did`'s rule.",
                UserWarning,
                stacklevel=2,
            )
            if data.empty:
                raise DataInsufficient(
                    "No periods remain before the last cohort is treated.",
                    recovery_hint="Add never-treated units or earlier periods.",
                    diagnostics={"latest_cohort": float(latest_g)},
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

    if svy is not None and use_unbalanced:
        raise MethodIncompatibility(
            "callaway_santanna: design= needs a balanced panel; this one is "
            "unbalanced and allow_unbalanced_panel / balance='none' would "
            "use the observation-level estimator.",
            recovery_hint="Pass balance='full' to keep the complete units "
            "(the others then contribute zero but keep their PSUs).",
        )
    if (not panel) or use_unbalanced:
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
            notyet_cutoff=notyet_cutoff,
            pretest=pretest,
            pretest_periods=pretest_periods,
            unit_col=i if use_unbalanced else None,
            weights_col=weights,
            clustervar=extra_cluster,
            bstrap=bstrap,
            biters=biters,
            cband=cband,
            boot_weight_type=boot_weight_type,
            random_state=random_state,
            exclude_cohorts=excluded_cohorts,
            variance_convention=variance_convention,
        )

    # A panel has one row per unit and period. Repeated (unit, period) rows
    # used to be collapsed to the first one by the reshape below, so the
    # estimate came from an arbitrary subset of the data without a word.
    _dup = data.duplicated(subset=[i, t], keep=False)
    if _dup.any():
        raise MethodIncompatibility(
            f"callaway_santanna: {int(_dup.sum())} rows share a (unit, period) "
            f"pair ({i!r}, {t!r}); a panel needs one row per unit and period.",
            recovery_hint=(
                f"If {t!r} is a pre/post flag over several dates, pass the "
                "date column as `t` (and the first treated date as `g`), or "
                "average the outcome within unit and period first. If the "
                "rows are different individuals sampled in each period, pass "
                "panel=False."
            ),
            diagnostics={
                "function": "callaway_santanna",
                "n_duplicated_rows": int(_dup.sum()),
                "n_periods": int(data[t].nunique()),
            },
        )

    # 1. Prepare panel data
    y_wide, unit_info, time_periods, cohorts, n_units, unit_weights = _prepare_panel(
        data, y, g, t, i, x, weights
    )
    cohorts = [c for c in cohorts if not any(c == e for e in excluded_cohorts)]

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
    x_wide = _covariates_by_period(data, i, t, x, unit_info.index)

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
            anticipation=anticipation,
            x_wide=x_wide,
        )

        pval = 2 * stats.norm.sf(abs(att / se)) if se > 0 else 1.0

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
    detail, inf_funcs_list, dropped_cells = _drop_unidentified_cells(
        detail, inf_funcs_list
    )

    # Stack influence functions: (n_units, n_gt_pairs)
    inf_matrix = np.column_stack(inf_funcs_list) if inf_funcs_list else None

    # 3.5 Optional multiplier-bootstrap inference (R did::att_gt bstrap
    # path).  Replaces the analytic SEs of each ATT(g,t) with bootstrap
    # SEs and, when cband=True, adds uniform (sup-t) confidence bands.
    cluster_ids: Any = None
    if extra_cluster is not None:
        cluster_ids = (
            data.groupby(i)[extra_cluster].first().reindex(unit_info.index).to_numpy()
        )
    if svy is not None:
        from ..survey.linearization import DesignVariance, unit_design_rows

        assert svy_data is not None
        cluster_ids = DesignVariance(
            svy,
            unit_design_rows(
                svy, svy_data, i, unit_info.index, function="callaway_santanna"
            ).rows,
        )
    boot_cfg: Optional[Dict[str, Any]] = None
    crit_val_uniform: Optional[float] = None
    analytic_cluster = cluster_ids if (not bstrap) else None
    if analytic_cluster is not None and inf_matrix is not None:
        detail = _cluster_detail_se(detail, inf_matrix, n_units, cluster_ids, alpha)
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
            2 * stats.norm.sf(np.abs(z_stat)),
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
        cluster_ids=analytic_cluster,
    )

    # 6. Event study aggregation
    event_study = _aggregate_event_study(
        detail,
        inf_matrix,
        cohort_sizes,
        n_units,
        alpha,
        boot_cfg=boot_cfg,
        unit_cohorts=unit_info[g].to_numpy(),
        unit_weights=unit_weights,
        cluster_ids=analytic_cluster,
    )

    # 7. Pre-trend test
    pretrend = _pretrend_test(
        detail,
        inf_matrix,
        n_units,
        pretest=pretest,
        pretest_periods=pretest_periods,
        cluster_ids=cluster_ids,
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

    # Few *treated* clusters: the cluster-robust / multiplier-bootstrap
    # variance estimates the treated side's contribution from as many draws
    # as there are treated clusters, so it over-rejects whatever the total
    # cluster count (Conley-Taber 2011; Ferman-Pinto 2019).
    _n_treated_units = int((unit_info[g] > 0).sum())
    if _n_treated_units < _FEW_TREATED_UNITS:
        warnings.warn(
            f"callaway_santanna: only {_n_treated_units} treated units. The "
            "analytic and multiplier-bootstrap standard errors estimate the "
            "treated side's variance from that many draws and over-reject "
            "however many control units there are. Report "
            "sp.did_few_treated (Conley-Taber / Ferman-Pinto) or "
            "sp.cs_jackknife (CV3) alongside.",
            AssumptionWarning,
            stacklevel=2,
        )

    if svy is not None:
        from ..survey.linearization import design_inference, t_inference_frame

        _svy_df = float(cluster_ids.df)
        detail = t_inference_frame(detail, alpha, _svy_df)
        event_study = t_inference_frame(event_study, alpha, _svy_df)
        agg_ci, agg_pval = design_inference(agg_est, agg_se, _svy_df, alpha)

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
        "covariates": list(x) if x else None,
        "n_units": n_units,
        # Treated *clusters*, not cohorts: what the few-treated-cluster
        # literature counts (sp.did_few_treated, sp.cs_jackknife).
        "n_treated_units": _n_treated_units,
        "n_periods": len(time_periods),
        "n_cohorts": len(cohorts),
        "cohorts": cohorts,
        "comparison_only_cohorts": [float(c) for c in excluded_cohorts],
        "dropped_gt_cells": dropped_cells,
        "event_study": event_study,
        "pretrend_test": pretrend,
        "cohort_sizes": cohort_sizes,
        "clustervars": clustervars,
        "bstrap": bstrap,
        "se_method": "multiplier" if bstrap else "analytic",
        "inference_note": _inference_note(bstrap, cband),
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
    if svy is not None:
        model_info["weights"] = None
        model_info["_cluster_ids"] = None
        model_info["se_method"] = "survey_linearization"
        model_info["inference_note"] = (
            "Survey linearisation of the unit influence functions "
            "(strata / PSU / fpc); t intervals on the design df."
        )
        model_info["survey_design"] = cluster_ids.info()

    # Which unbalanced-panel rule produced the sample (csdid 2.0.0's bal()).
    model_info["balance"] = "full" if balance_info is not None else "pair"
    if balance_info is not None:
        model_info["balance_n_units_dropped"] = balance_info["n_units_dropped"]
        model_info["balance_n_obs_dropped"] = balance_info["n_obs_dropped"]

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


def _never_treated_beyond_horizon(data: pd.DataFrame, g: str, t: str) -> pd.DataFrame:
    """Recode cohorts first treated after the last observed period to 0.

    Units whose treatment starts after the sample ends are untreated in
    every observed period, so they belong to the never-treated comparison
    group. R ``did`` treats them that way (black box: same ATT and SE to
    1e-15 on a panel and on repeated cross-sections). Until 1.40 they were
    left out of ``control_group='nevertreated'``.
    """
    g_num = pd.to_numeric(data[g], errors="coerce")
    beyond = (g_num > data[t].max()).to_numpy()
    if not beyond.any():
        return data
    data = data.copy()
    data.loc[beyond, g] = 0
    return data


def _balance_full(
    data: pd.DataFrame, i: str, t: str
) -> "tuple[pd.DataFrame, Dict[str, Any]]":
    """Keep the units observed in every period (R ``did``'s default).

    R ``did::att_gt(allow_unbalanced_panel = FALSE)`` and Stata ``csdid``
    2.0.0 ``bal(full)`` balance an unbalanced panel this way before they
    estimate anything, and say how much they removed. So does this.
    """
    n_periods = int(data[t].nunique())
    seen = data.groupby(i)[t].nunique()
    complete = seen.index[seen == n_periods]
    n_units = int(len(seen))
    info: Dict[str, Any] = {
        "rule": "full",
        "n_units_dropped": n_units - int(len(complete)),
        "n_obs_dropped": 0,
    }
    if len(complete) == n_units:
        return data, info
    if len(complete) == 0:
        raise MethodIncompatibility(
            "callaway_santanna: balance='full' leaves no unit, because none "
            f"is observed in all {n_periods} periods.",
            recovery_hint=(
                "Use balance='pair' (the default) or balance='none', or "
                "restrict the data to periods most units are observed in."
            ),
            diagnostics={"n_units": n_units, "n_periods": n_periods},
        )
    kept = data[data[i].isin(complete)]
    info["n_obs_dropped"] = int(len(data) - len(kept))
    warnings.warn(
        f"callaway_santanna: balance='full' dropped {info['n_units_dropped']} "
        f"of {n_units} units ({info['n_obs_dropped']} observations) that are "
        f"not observed in all {n_periods} periods, as R `did` and Stata "
        "`csdid` 2.0.0 do by default.",
        UserWarning,
        stacklevel=3,
    )
    return kept, info


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
            "cell; the effective sample varies across cells "
            "(balance='pair', the rule of Stata `csdid` 1.8x). Pass "
            "allow_unbalanced_panel=True (balance='none') to switch to the "
            "repeated-cross-section estimators instead, which keep every "
            "observed row (R `did`'s allow_unbalanced_panel=TRUE) but "
            "compare group means and so need each group's composition to "
            "be stable over time (biased when units leave according to "
            "their level; see tests/reliability/unbalanced_panel.py), or "
            "balance='full' to drop the incomplete units first, which is "
            "what R `did` and Stata `csdid` 2.0.0 do by default. The three "
            "give different numbers on the same data — that is an "
            "estimator choice, not a bug.",
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


def _covariates_by_period(
    data: pd.DataFrame,
    i: str,
    t: str,
    x: Optional[List[str]],
    units: pd.Index,
) -> Optional[Dict[str, pd.DataFrame]]:
    """Covariates as (unit x period) tables, when any of them moves over time.

    A covariate that is constant within unit is read once per unit and this
    returns ``None``. One that varies has no single unit-level value: each
    ATT(g,t) cell conditions on its value in the earlier of the cell's two
    periods, which is what R ``did`` (``panel2cs2`` keeps the first period
    of the pair) and Stata ``csdid`` / ``drdid`` do.
    """
    if not x:
        return None
    moves = data.groupby(i)[list(x)].nunique(dropna=False).gt(1).any(axis=0)
    if not bool(moves.any()):
        return None
    return {
        col: data.pivot_table(
            index=i, columns=t, values=col, aggfunc="first", dropna=False
        ).reindex(units)
        for col in x
    }


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


def _drop_unidentified_cells(
    detail: pd.DataFrame, inf_funcs_list: List[np.ndarray]
) -> Tuple[pd.DataFrame, List[np.ndarray], List[Tuple[Any, Any]]]:
    """Remove ATT(g, t) cells that could not be estimated.

    A cell without comparison (or treated) observations comes back from the
    cell estimators as ``att = 0, se = inf`` with a zero influence function.
    Kept, it entered every aggregate as a genuine zero effect and pulled the
    headline ATT towards zero without a word. R ``did`` reports such cells
    as NA and ``aggte(na.rm = TRUE)`` skips them; they are dropped here, with
    a warning, and listed in ``model_info['dropped_gt_cells']``.
    """
    if detail.empty:
        return detail, inf_funcs_list, []
    bad = ~np.isfinite(detail["se"].to_numpy(dtype=float))
    if not bad.any():
        return detail, inf_funcs_list, []
    cells = [
        (detail["group"].iloc[k], detail["time"].iloc[k]) for k in np.flatnonzero(bad)
    ]
    if bad.all():
        raise DataInsufficient(
            "No ATT(g, t) cell has both treated and comparison observations.",
            recovery_hint=(
                "Check the comparison group: add never-treated units or use "
                "control_group='notyettreated'."
            ),
            diagnostics={"n_cells": len(cells)},
        )
    shown = ", ".join(f"({a}, {b})" for a, b in cells[:8])
    more = f" and {len(cells) - 8} more" if len(cells) > 8 else ""
    warnings.warn(
        f"callaway_santanna: {len(cells)} ATT(g, t) cell(s) have no treated "
        f"or comparison observations and are left out of the estimates and "
        f"every aggregate: (g, t) = {shown}{more}.",
        UserWarning,
        stacklevel=3,
    )
    keep = np.flatnonzero(~bad)
    return (
        detail.loc[~bad].reset_index(drop=True),
        [inf_funcs_list[k] for k in keep],
        cells,
    )


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
    anticipation: int = 0,
    x_wide: Optional[Dict[str, pd.DataFrame]] = None,
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
    #   notyet_cutoff='period' — untreated as of both periods of the 2x2
    #       comparison, i.e. G > max(t, base) + anticipation. R ``did``
    #       (``get_did_cohort_index`` / ``compute.att_gt``:
    #       ``time_periods[max(t, pret) + tfac] + anticipation``) and Stata
    #       ``csdid, asinr``. This is StatsPAI's default and matches R.
    #       Under base_period='varying' base < t, so the cutoff is t; under
    #       'universal' a pre-treatment cell has base = g - 1 - anticipation
    #       > t, and a cohort treated between t and base must be excluded
    #       because its base-period outcome is already treated.
    #   notyet_cutoff='asinr' — untreated as of t only (G > t), Stata
    #       ``csdid, asinr``; differs from 'period' on universal-base
    #       pre-treatment cells and whenever anticipation > 0.
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
    elif notyet_cutoff == "asinr":
        is_control = (g_series == 0) | (g_series > t_val)
    elif notyet_cutoff == "cohort":
        # max(t, g), not g: csdid's rule is scoped to PRE-treatment cells
        # ("pre-treatment ATTGT's ... are estimated using ..."). For t >= g
        # the cutoff is already t, so max() reproduces the shared
        # post-treatment behaviour instead of shrinking those cells too.
        is_control = (g_series == 0) | (g_series > max(t_val, g_val))
    else:  # notyettreated, untreated through both comparison periods
        is_control = (g_series == 0) | (g_series > max(t_val, base_val) + anticipation)

    # Outcome change ΔY = Y_t - Y_base
    if t_val not in y_wide.columns or base_val not in y_wide.columns:
        return 0.0, np.inf, np.zeros(n_total)

    dy = y_wide[t_val] - y_wide[base_val]

    # Valid units: in a relevant group AND observed in both periods
    relevant = (is_treated | is_control) & dy.notna()
    x_cell = None
    if x_cols and x_wide is not None:
        # time-varying covariates: the value in the earlier period of the pair
        first_period = min(t_val, base_val)
        x_cell = np.column_stack(
            [x_wide[c][first_period].to_numpy(dtype=float) for c in x_cols]
        )
        relevant = relevant & np.isfinite(x_cell).all(axis=1)
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
        if x_cell is not None:
            x_sub = x_cell[relevant.to_numpy()]
        else:
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
    elif estimator == "drimp":
        att, se, inf_local = _dr_imp_att(dy_sub, d_sub, x_sub, pscore_trim, w_sub)
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
# Shared plumbing of the panel 2x2 engines
# ------------------------------------------------------------------
#
# Every engine below works on the long difference ``dy = Y_t - Y_base`` of
# one (g, t) cell and returns ``(att, se, influence)``, the influence
# function being on the scale ``att_hat - att ~ mean(influence)`` so that
# ``se = sqrt(mean(influence**2) / n)``. The algebra (Hajek means, the
# nuisance influence functions and the delta-method gradients) is written
# out in docs/dev/cleanroom/did_drdid.md, section "Panel estimators", after
# Sant'Anna and Zhao (2020) [@santanna2020doubly].


def _not_estimated(n: int) -> Tuple[float, float, np.ndarray]:
    """The "not estimated" triple of a degenerate cell."""
    return 0.0, np.inf, np.zeros(n)


def _cell_arrays(
    dy: np.ndarray,
    d: np.ndarray,
    w: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``dy``, ``d`` and the weights rescaled to mean one.

    ``None`` when the weights have no positive mass (the cell cannot be
    estimated).
    """
    dy = np.asarray(dy, dtype=float)
    d = np.asarray(d, dtype=float)
    if w is None:
        return dy, d, np.ones(dy.shape[0], dtype=float)
    w = np.asarray(w, dtype=float)
    mass = float(w.mean()) if w.size else 0.0
    if not np.isfinite(mass) or mass <= 0:
        return None
    return dy, d, _drc.normalized_weights(w, dy.shape[0])


def _arms_have_mass(d: np.ndarray, w: np.ndarray) -> bool:
    """True when both the treated and the comparison arm carry weight."""
    return float(np.sum(w * d)) > 0 and float(np.sum(w * (1.0 - d))) > 0


def _nuisance_design(x: Optional[np.ndarray], fitted: np.ndarray) -> np.ndarray:
    """Design matrix (with intercept) of the nuisance fit behind ``fitted``.

    The covariate fits of :func:`_estimate_pscore` and
    :func:`_estimate_outcome_reg` fall back to a constant (with a warning)
    when they cannot use the covariates, so a constant ``fitted`` means the
    nuisance parameter is the intercept alone. Constant covariate columns
    are dropped: they span the intercept, and the estimation effect does
    not depend on how the column space is parameterised.
    """
    n = fitted.shape[0]
    if np.ptp(fitted) == 0:
        return np.ones((n, 1), dtype=float)
    return _covariate_design(x, n)


def _covariate_design(x: Optional[np.ndarray], n: int) -> np.ndarray:
    """Intercept plus the non-constant covariate columns."""
    if x is None:
        return np.ones((n, 1), dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    varying = np.ptp(x, axis=0) > 0 if x.shape[1] else np.zeros(0, dtype=bool)
    return _drc.design_matrix(x[:, varying] if varying.any() else None, n)


def _project(influence: Any, gram_and_score: Any, grad: np.ndarray) -> np.ndarray:
    """``l_i' grad`` for a nuisance influence function ``l``.

    ``influence`` computes the ``n x k`` influence matrix with the shared
    primitive; when its Gram / Hessian matrix is singular (collinear
    covariates) ``gram_and_score`` supplies ``(gram, score)`` and the
    minimum-norm solve is used, which gives the same ``l_i' grad`` for any
    ``grad`` in the column space of the design.
    """
    try:
        out = influence() @ grad
        if np.all(np.isfinite(out)):
            return out
    except np.linalg.LinAlgError:
        pass
    gram, score = gram_and_score()
    return score @ np.linalg.lstsq(gram, grad, rcond=None)[0]


def _wls_effect(
    x_design: np.ndarray,
    resid: np.ndarray,
    w: np.ndarray,
    rows: np.ndarray,
    grad: np.ndarray,
) -> np.ndarray:
    """Estimation effect ``l_b' grad`` of a WLS fit with residuals ``resid``."""
    n, k = x_design.shape
    zeros = np.zeros(k)

    def gram_and_score() -> Tuple[np.ndarray, np.ndarray]:
        wr = np.where(rows, w, 0.0)
        gram = (x_design.T * wr) @ x_design / n
        return gram, x_design * (wr * np.where(rows, resid, 0.0))[:, None]

    # ``resid`` already is ``y - x'b`` on the fitted rows, so passing it as
    # the outcome with a zero coefficient vector reproduces the residuals
    # exactly.
    return _project(
        lambda: _drc.wls_influence(x_design, resid, w, rows, zeros),
        gram_and_score,
        grad,
    )


def _logit_effect(
    x_design: np.ndarray,
    d: np.ndarray,
    w: np.ndarray,
    pscore: np.ndarray,
    grad: np.ndarray,
) -> np.ndarray:
    """Estimation effect ``l_g' grad`` of the weighted logit propensity fit."""
    n = x_design.shape[0]

    def gram_and_score() -> Tuple[np.ndarray, np.ndarray]:
        hess = (x_design.T * (w * pscore * (1.0 - pscore))) @ x_design / n
        return hess, x_design * (w * (d - pscore))[:, None]

    return _project(
        lambda: _drc.logit_influence(x_design, d, w, pscore),
        gram_and_score,
        grad,
    )


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
    """Doubly robust 2x2 DiD on a panel (Sant'Anna-Zhao eq. 3.1).

    The difference of two Hajek means of the outcome-regression residual
    ``r = dy - m(X)``: over the treated (weights ``w``) and over the
    comparison units (weights ``w p/(1-p)``, trimmed at ``pscore_trim``).
    ``m`` is the weighted least-squares fit of ``dy`` on the comparison
    units, ``p`` the weighted logit propensity score.

    Parameters
    ----------
    dy : ndarray
        Outcome change ``Y_t - Y_base`` per unit.
    d : ndarray
        1 for the treated cohort, 0 for the comparison units.
    x : ndarray or None
        Covariates without an intercept column.
    pscore_trim : float
        Comparison units with ``p >= pscore_trim`` get weight zero.
    w : ndarray, optional
        Unit weights; rescaled to mean one here.

    Returns
    -------
    tuple
        ``(att, se, influence)``; a degenerate cell gives
        ``(0.0, inf, zeros)``.

    References
    ----------
    [@santanna2020doubly]; docs/dev/cleanroom/did_drdid.md.
    """
    n = len(dy)
    arrays = _cell_arrays(dy, d, w)
    if arrays is None:
        return _not_estimated(n)
    dy, d, w = arrays
    if not _arms_have_mass(d, w):
        return _not_estimated(n)

    pscore = _estimate_pscore(d, x, n, w)
    keep = _pscore_trim_mask(pscore, d, pscore_trim)
    _TRIM_TALLY.record(keep, d)
    m_hat = _estimate_outcome_reg(dy, 1 - d, x, n, w)
    resid = dy - m_hat

    a_1 = w * d
    a_0 = keep * w * (1.0 - d) * pscore / (1.0 - pscore)
    p_d = float(a_1.mean())
    ipw_denom = float(a_0.mean())
    if p_d <= 0 or not np.isfinite(ipw_denom) or ipw_denom <= 0:
        return _not_estimated(n)

    treated = _drc.HajekMean(a_1, resid)
    comparison = _drc.HajekMean(a_0, resid)
    att = treated.theta - comparison.theta

    effect = _dr_estimation_effect(
        dy=dy,
        d=d,
        x=x,
        w=w,
        keep=keep,
        pscore=pscore,
        resid=resid,
        eta_c=comparison.theta,
        p_d=p_d,
        ipw_denom=ipw_denom,
    )
    if effect is None:
        return _not_estimated(n)
    influence = treated.influence - comparison.influence + effect
    return float(att), _drc.se_from_influence(influence), influence


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
    """Nuisance estimation effect in the influence function of :func:`_dr_att`.

    The sum of the outcome-regression term
    ``l_b' (E_n[a_0 X]/E_n[a_0] - E_n[a_1 X]/E_n[a_1])`` and the propensity
    term ``-l_g' E_n[a_0 (r - eta_c) X]/E_n[a_0]``, with ``a_1 = w d``,
    ``a_0 = keep w p/(1-p) (1-d)`` and ``l_b`` / ``l_g`` the influence
    functions of the comparison-arm WLS fit and of the logit fit.

    Parameters
    ----------
    dy, d, x, w : ndarray
        Outcome change, treated dummy, covariates (no intercept, or None)
        and mean-one weights of the cell.
    keep : ndarray
        Trimming mask (0 for trimmed comparison units).
    pscore : ndarray
        Fitted propensity scores.
    resid : ndarray
        ``dy - m(X)``, the outcome-regression residual.
    eta_c : float
        Hajek mean of ``resid`` over the comparison arm.
    p_d, ipw_denom : float
        ``E_n[a_1]`` and ``E_n[a_0]``.

    Returns
    -------
    ndarray or None
        The ``n``-vector, or ``None`` when an arm has no weight.

    References
    ----------
    [@santanna2020doubly]; docs/dev/cleanroom/did_drdid.md.
    """
    dy = np.asarray(dy, dtype=float)
    d = np.asarray(d, dtype=float)
    w = np.asarray(w, dtype=float)
    pscore = np.asarray(pscore, dtype=float)
    resid = np.asarray(resid, dtype=float)
    if not (np.isfinite(p_d) and p_d > 0 and np.isfinite(ipw_denom) and ipw_denom > 0):
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        a_0 = np.asarray(keep, dtype=float) * w * (1.0 - d) * pscore / (1.0 - pscore)
    if not np.all(np.isfinite(a_0)) or float(a_0.mean()) <= 0:
        return None
    a_1 = w * d

    # Outcome regression: the residual moves by -X db in both Hajek means.
    x_or = _nuisance_design(x, dy - resid)
    m_1 = (a_1[:, None] * x_or).mean(axis=0) / p_d
    m_0 = (a_0[:, None] * x_or).mean(axis=0) / ipw_denom
    or_effect = _wls_effect(x_or, resid, w, d == 0, m_0 - m_1)

    # Propensity score: the comparison weights move by a_0 X dg.
    ps_effect = _pscore_estimation_effect(
        dy=resid - eta_c,
        d=d,
        x=x,
        w=w,
        pscore=pscore,
        arm_weights=a_0,
        arm_denom=ipw_denom,
        center=False,
    )
    if ps_effect is None:
        return None
    return or_effect + ps_effect


def _dr_imp_att(
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
    pscore_trim: float = _DEFAULT_PSCORE_TRIM,
    w: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Improved (locally efficient, bias-reduced) DR 2x2 DiD on a panel.

    As :func:`_dr_att`, with the propensity score fitted by inverse
    probability tilting and the comparison outcome regression fitted by
    least squares with weights ``w p/(1-p)`` (Sant'Anna-Zhao Section 3.1).
    Both fits' first-order conditions remove the estimation effect, so the
    influence function is the two Hajek terms alone. Without covariates
    the tilting and the logit scores coincide and :func:`_dr_att` is used.

    Parameters
    ----------
    dy, d, x, pscore_trim, w
        As in :func:`_dr_att`.

    Returns
    -------
    tuple
        ``(att, se, influence)``; a degenerate cell gives
        ``(0.0, inf, zeros)``.

    References
    ----------
    [@santanna2020doubly]; docs/dev/cleanroom/did_drdid.md.
    """
    n = len(dy)
    arrays = _cell_arrays(dy, d, w)
    if arrays is None:
        return _not_estimated(n)
    dy, d, w = arrays
    if not _arms_have_mass(d, w):
        return _not_estimated(n)

    x_design = _covariate_design(x, n)
    if x_design.shape[1] == 1:
        return _dr_att(dy, d, None, pscore_trim, w)

    try:
        _, pscore = _drc.ipt_fit(d, x_design, w)
        if not np.all(np.isfinite(pscore)):
            raise FloatingPointError("non-finite tilted propensity scores")
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        warnings.warn(
            "callaway_santanna: the inverse-probability-tilting propensity "
            f"fit failed ({type(exc).__name__}: {exc}); this ATT(g,t) falls "
            "back to the standard doubly robust estimator (logit propensity "
            "score, unweighted comparison regression). Check for separated "
            "or collinear covariates.",
            ConvergenceWarning,
            stacklevel=2,
        )
        return _dr_att(dy, d, x, pscore_trim, w)
    pscore = np.clip(pscore, 1e-6, 1 - 1e-6)
    keep = _pscore_trim_mask(pscore, d, pscore_trim)
    _TRIM_TALLY.record(keep, d)

    odds = pscore / (1.0 - pscore)
    controls = d == 0
    coef = _drc.wls_coef(x_design, dy, w * odds, controls)
    resid = dy - x_design @ coef

    a_1 = w * d
    a_0 = keep * w * (1.0 - d) * odds
    if float(a_0.mean()) <= 0:
        return _not_estimated(n)
    treated = _drc.HajekMean(a_1, resid)
    comparison = _drc.HajekMean(a_0, resid)
    att = treated.theta - comparison.theta
    influence = treated.influence - comparison.influence
    return float(att), _drc.se_from_influence(influence), influence


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
    """Normalised (Hajek) IPW 2x2 DiD on a panel.

    ``theta(w d, dy) - theta(w p/(1-p) (1-d), dy)``: the treated mean of
    ``dy`` minus the odds-weighted comparison mean, comparison units with
    ``p >= pscore_trim`` trimmed. The influence function carries the
    estimation effect of the logit propensity fit.

    Parameters
    ----------
    dy, d, x, pscore_trim, w
        As in :func:`_dr_att`.

    Returns
    -------
    tuple
        ``(att, se, influence)``; a degenerate cell gives
        ``(0.0, inf, zeros)``.

    References
    ----------
    [@santanna2020doubly]; docs/dev/cleanroom/did_drdid.md.
    """
    n = len(dy)
    arrays = _cell_arrays(dy, d, w)
    if arrays is None:
        return _not_estimated(n)
    dy, d, w = arrays
    if not _arms_have_mass(d, w):
        return _not_estimated(n)

    pscore = _estimate_pscore(d, x, n, w)
    keep = _pscore_trim_mask(pscore, d, pscore_trim)
    _TRIM_TALLY.record(keep, d)
    a_1 = w * d
    a_0 = keep * w * (1.0 - d) * pscore / (1.0 - pscore)
    if float(a_0.mean()) <= 0:
        return _not_estimated(n)

    treated = _drc.HajekMean(a_1, dy)
    comparison = _drc.HajekMean(a_0, dy)
    effect = _pscore_estimation_effect(
        dy=dy,
        d=d,
        x=x,
        w=w,
        pscore=pscore,
        arm_weights=a_0,
        arm_denom=comparison.mass,
        center=True,
    )
    if effect is None:
        return _not_estimated(n)
    att = treated.theta - comparison.theta
    influence = treated.influence - comparison.influence + effect
    return float(att), _drc.se_from_influence(influence), influence


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
    """Logit estimation effect of a comparison-arm term ``E_n[a z]/arm_denom``.

    For comparison weights ``a = arm_weights`` proportional to the
    propensity odds, ``d a / d g = a X``, so the term's gradient is
    ``E_n[a z X] / arm_denom``; it enters an ATT with a minus sign. With
    ``center=True`` (a Hajek mean, ``arm_denom = E_n[a]``) ``z`` is ``dy``
    minus its Hajek mean over the arm; with ``center=False`` ``z = dy``
    (the Abadie estimator divides by the treated mass instead).

    Parameters
    ----------
    dy, d, x, w : ndarray
        Outcome (change), treated dummy, covariates (no intercept, or
        None) and mean-one weights.
    pscore : ndarray
        Fitted logit propensity scores.
    arm_weights : ndarray
        The comparison-arm weights ``a``.
    arm_denom : float
        The denominator of the term.
    center : bool
        Centre ``dy`` at its Hajek mean over the arm.

    Returns
    -------
    ndarray or None
        ``-l_g' E_n[a z X]/arm_denom`` per row, or ``None`` when the arm
        has no weight or ``arm_denom`` is not positive.

    References
    ----------
    [@santanna2020doubly]; docs/dev/cleanroom/did_drdid.md.
    """
    dy = np.asarray(dy, dtype=float)
    a = np.asarray(arm_weights, dtype=float)
    pscore = np.asarray(pscore, dtype=float)
    if not (np.isfinite(arm_denom) and arm_denom > 0):
        return None
    mass = float(a.mean())
    if not np.isfinite(mass) or mass <= 0:
        return None
    z = dy - float(np.mean(a * dy) / mass) if center else dy
    x_ps = _nuisance_design(x, pscore)
    grad = (a[:, None] * z[:, None] * x_ps).mean(axis=0) / arm_denom
    return -_logit_effect(
        x_ps, np.asarray(d, dtype=float), np.asarray(w, dtype=float), pscore, grad
    )


def _ipw_abadie_att(
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
    pscore_trim: float = _DEFAULT_PSCORE_TRIM,
    w: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Abadie (2005) IPW 2x2 DiD on a panel (Sant'Anna-Zhao eq. 2.4).

    ``E_n[(a_1 - a_0) dy] / E_n[a_1]`` with ``a_1 = w d`` and
    ``a_0 = w p/(1-p) (1-d)`` (trimmed at ``pscore_trim``): both arms are
    divided by the treated mass, so the comparison weights are not
    normalised to sum to one.

    Parameters
    ----------
    dy, d, x, pscore_trim, w
        As in :func:`_dr_att`.

    Returns
    -------
    tuple
        ``(att, se, influence)``; a degenerate cell gives
        ``(0.0, inf, zeros)``.

    References
    ----------
    [@santanna2020doubly]; docs/dev/cleanroom/did_drdid.md.
    """
    n = len(dy)
    arrays = _cell_arrays(dy, d, w)
    if arrays is None:
        return _not_estimated(n)
    dy, d, w = arrays
    if not _arms_have_mass(d, w):
        return _not_estimated(n)

    pscore = _estimate_pscore(d, x, n, w)
    keep = _pscore_trim_mask(pscore, d, pscore_trim)
    _TRIM_TALLY.record(keep, d)
    a_1 = w * d
    a_0 = keep * w * (1.0 - d) * pscore / (1.0 - pscore)
    p_d = float(a_1.mean())
    if float(a_0.mean()) <= 0:
        return _not_estimated(n)

    att = float(np.mean((a_1 - a_0) * dy) / p_d)
    effect = _pscore_estimation_effect(
        dy=dy,
        d=d,
        x=x,
        w=w,
        pscore=pscore,
        arm_weights=a_0,
        arm_denom=p_d,
        center=False,
    )
    if effect is None:
        return _not_estimated(n)
    influence = ((a_1 - a_0) * dy - att * a_1) / p_d + effect
    return att, _drc.se_from_influence(influence), influence


# ------------------------------------------------------------------
# Outcome regression
# ------------------------------------------------------------------


def _reg_att(
    dy: np.ndarray,
    d: np.ndarray,
    x: Optional[np.ndarray],
    w: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Outcome-regression 2x2 DiD on a panel (Sant'Anna-Zhao eq. 2.2).

    The treated Hajek mean of ``dy - m(X)``, ``m`` the weighted
    least-squares fit of ``dy`` on the comparison units. The influence
    function carries the estimation effect of that fit. When the
    regression cannot use the covariates (it warns) the estimator is the
    unadjusted difference of the arm means.

    Parameters
    ----------
    dy, d, x, w
        As in :func:`_dr_att`.

    Returns
    -------
    tuple
        ``(att, se, influence)``; a degenerate cell gives
        ``(0.0, inf, zeros)``.

    References
    ----------
    [@santanna2020doubly]; docs/dev/cleanroom/did_drdid.md.
    """
    n = len(dy)
    arrays = _cell_arrays(dy, d, w)
    if arrays is None:
        return _not_estimated(n)
    dy, d, w = arrays
    if not _arms_have_mass(d, w):
        return _not_estimated(n)

    m_hat = _estimate_outcome_reg(dy, 1 - d, x, n, w)
    resid = dy - m_hat
    a_1 = w * d
    treated = _drc.HajekMean(a_1, resid)

    x_or = _nuisance_design(x, m_hat)
    m_1 = (a_1[:, None] * x_or).mean(axis=0) / treated.mass
    effect = _wls_effect(x_or, resid, w, d == 0, -m_1)
    influence = treated.influence + effect
    return float(treated.theta), _drc.se_from_influence(influence), influence


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

    Treated units are never trimmed; a control is kept when its score is
    below ``trim`` (the ``trim.level`` convention documented by R ``DRDID``
    and used by R ``did`` and Stata ``csdid``; confirmed on their outputs,
    see docs/dev/cleanroom/did_drdid.md).

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


def _fit_converged(result: Any) -> Optional[bool]:
    """Return a fitter's convergence flag, or ``None`` when it has none.

    ``sm.Logit`` records it under ``mle_retvals['converged']``; ``sm.GLM``'s
    IRLS exposes a ``converged`` attribute instead. Neither is a contractual
    part of the statsmodels API, so a missing or non-boolean flag is
    reported as *unknown* rather than as *diverged*: this must never
    manufacture a warning on the happy path.
    """
    retvals = getattr(result, "mle_retvals", None)
    if isinstance(retvals, dict) and isinstance(
        retvals.get("converged"), (bool, np.bool_)
    ):
        return bool(retvals["converged"])
    converged = getattr(result, "converged", None)
    if isinstance(converged, (bool, np.bool_)):
        return bool(converged)
    return None


def _estimate_pscore(
    d: np.ndarray,
    x: Optional[np.ndarray],
    n: int,
    w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate propensity score P(D=1 | X) via (weighted) logit.

    Uses statsmodels (core dep) — no sklearn needed.
    Falls back to unconditional probability if logit fails or no covariates,
    and warns when the fit merely fails to converge (see below): either way
    the caller is told that p(X) is not the object they asked for.

    With ω supplied the logit is fit by weighted maximum likelihood (the
    reference's weighted fit agrees on its outputs). The
    unweighted branch keeps the original ``sm.Logit`` call verbatim so the
    existing R/Stata parity fixtures stay bit-identical.
    """
    weighted = w is not None and not np.allclose(w, 1.0)
    p_d = np.mean(d) if not weighted else float(np.mean(w * d))
    if x is None or x.shape[1] == 0:
        return np.full(n, p_d)

    result = None
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
        result = None
        pscore = np.full(n, p_d)

    # A separated or collinear design does not always *raise*: statsmodels
    # >= 0.15 walks the Newton step that older releases turned into a
    # ``LinAlgError`` and returns diverging coefficients instead, and
    # ``warn_convergence=False`` above suppresses its own notice. Without
    # this branch the resulting p(X) — pinned at the clipping bounds by
    # coefficients that never settled — would feed the IPW/DR weights with
    # no signal to the caller, which is exactly the silent degradation
    # CLAUDE.md §7 forbids. The fitted p(X) is kept rather than replaced:
    # swapping in the constant here would move the point estimate of every
    # non-converged fit, and a diverged logit is not evidence that the
    # unconditional propensity is the better answer.
    if result is not None and _fit_converged(result) is False:
        warnings.warn(
            "callaway_santanna: the covariate propensity-score logit did "
            "not converge (typically perfect separation or a collinear "
            "covariate). The fitted p(X) is used as reported, but it rests "
            "on coefficients that never settled, so the IPW/DR weights for "
            "this ATT(g,t) are unreliable. Inspect overlap, drop the "
            "separated covariate, or use estimator='reg'.",
            ConvergenceWarning,
            stacklevel=2,
        )

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


def _cluster_detail_se(
    detail: pd.DataFrame,
    inf_matrix: np.ndarray,
    n_scale: int,
    cluster_ids: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    """Replace the per-cell analytic SEs by their cluster-robust version.

    ``influence_se_did`` with clusters: Stata ``csdid, cluster()``'s
    analytic variance.  Cells whose SE was not finite (unidentified) stay
    as they were.
    """
    detail = detail.copy()
    se_cl = np.asarray(influence_se_did(inf_matrix, n_scale, cluster_ids))
    ok = np.isfinite(detail["se"].to_numpy(dtype=float))
    se_new = np.where(ok, se_cl, detail["se"].to_numpy(dtype=float))
    att = detail["att"].to_numpy(dtype=float)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se_new > 0, att / se_new, 0.0)
    detail["se"] = se_new
    detail["pvalue"] = np.where(se_new > 0, 2 * stats.norm.sf(np.abs(z)), 1.0)
    detail["ci_lower"] = att - z_crit * se_new
    detail["ci_upper"] = att + z_crit * se_new
    return detail


def _aggregate_simple(
    post_detail: pd.DataFrame,
    post_inf: Optional[np.ndarray],
    cohort_sizes: pd.Series,
    n_total: int,
    alpha: float,
    boot_cfg: Optional[Dict[str, Any]] = None,
    unit_cohorts: Optional[np.ndarray] = None,
    unit_weights: Optional[np.ndarray] = None,
    cluster_ids: Any = None,
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
            se_agg = float(influence_se_did(inf_agg, n_total, cluster_ids))
    else:
        se_agg = float(
            np.sqrt(np.average(post_detail["se"].values ** 2, weights=weights))
        )

    z = att_agg / se_agg if se_agg > 0 else 0
    pval = float(2 * stats.norm.sf(abs(z)))
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
    unit_cohorts: Optional[np.ndarray] = None,
    unit_weights: Optional[np.ndarray] = None,
    cluster_ids: Any = None,
) -> pd.DataFrame:
    """Event study aggregation: average ATT by relative time e = t − g.

    When ``boot_cfg`` is given (bstrap=True path), the per-event-time SEs
    come from a joint multiplier bootstrap over all event-time
    combinations instead of the analytic plug-in.

    ``unit_cohorts`` (row-aligned with ``inf_matrix``) adds the
    cohort-share estimation term, as :func:`_aggregate_simple` and
    ``sp.aggte(type='dynamic')`` do; without it an event time that mixes
    cohorts has too small a standard error.
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
            if unit_cohorts is not None and len(unit_cohorts) == inf_matrix.shape[0]:
                pg, ind = _cohort_share_context(
                    sub["group"].values, unit_cohorts, unit_weights
                )
                inf_e = inf_e + _weight_influence(pg, ind) @ (
                    sub["att"].values.astype(float)
                )
            se_e = float(influence_se_did(inf_e, n_total, cluster_ids))
            psi_cols.append(inf_e)
        else:
            se_e = float(np.sqrt(np.average(sub["se"].values ** 2, weights=weights)))

        pval = float(2 * stats.norm.sf(abs(att_e / se_e))) if se_e > 0 else 1.0

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
        es["pvalue"] = np.where(se_vec > 0, 2 * stats.norm.sf(np.abs(z_stat)), 1.0)
        es["ci_lower"] = att_vals - z_crit * se_vec
        es["ci_upper"] = att_vals + z_crit * se_vec

    return es


def _pretrend_test(
    detail: pd.DataFrame,
    inf_matrix: Optional[np.ndarray],
    n_total: int,
    pretest: str = "joint",
    pretest_periods: Optional[int] = None,
    cluster_ids: Any = None,
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
        if getattr(cluster_ids, "is_survey_design", False):
            # design= fits: survey linearisation of psi / n; the design
            # degrees of freedom play the role of G - 1
            V = cluster_ids.vcov(inf_pre, n_total)  # type: ignore[union-attr]
            G = int(cluster_ids.df) + 1  # type: ignore[union-attr]
        else:
            if cluster_ids is not None:
                # Cluster sums replace the rows (csdid, cluster()).
                codes = pd.factorize(np.asarray(cluster_ids))[0]
                sums = np.zeros((codes.max() + 1, inf_pre.shape[1]))
                np.add.at(sums, codes, inf_pre)
                inf_pre = sums
            # Variance-covariance: V = (1/n²) IF' IF
            V = inf_pre.T @ inf_pre / (n_total**2)
            G = int(inf_pre.shape[0])  # number of IF contributions (units)
    else:
        V = np.diag(pre["se"].values ** 2)

    # The pre-treatment ATT(g, t) cells are linearly dependent by
    # construction whenever the panel has more cells than independent
    # long differences: under a universal base period every cell is a
    # difference Y_t - Y_{g-1} of the same control-group means, and a cohort
    # with a single unit contributes no treated-side variation at all.  On
    # the castle-doctrine panel (50 states, 30 pre cells) V has rank 17.
    # Ridge-regularising and inverting such a matrix returned a Wald
    # statistic of order 1e9 -- a number, not a test.  Use the spectral
    # pseudo-inverse instead: the statistic is the quadratic form on the
    # non-null eigenspace and the degrees of freedom are the rank, which is
    # what Stata's ``test`` does when it drops redundant constraints.
    V = 0.5 * (V + V.T)
    eigval, eigvec = np.linalg.eigh(V)
    tol = (
        max(float(np.max(np.abs(eigval))), 0.0) * max(k, 1) * np.finfo(float).eps * 1e3
    )
    nonnull = eigval > tol
    rank = int(nonnull.sum())
    if rank == 0:
        return {"statistic": np.nan, "df": 0, "n_cells": k, "rank": 0, "pvalue": np.nan}
    proj = eigvec[:, nonnull].T @ theta
    W = float(np.sum(proj**2 / eigval[nonnull]))

    # The pre-period ATT(g,t) are strongly correlated (shared base period and
    # control group), and V is *estimated*, so the plug-in chi²(r) Wald
    # over-rejects in finite samples (≈0.14 at a nominal 5% level for ~60
    # units, vs the Callaway–Sant'Anna multiplier-bootstrap pre-test which is
    # correctly sized). Apply the Hotelling T² finite-sample correction:
    #   F = W · (G − r) / (r · (G − 1))  ~  F(r, G − r)
    # which is exact under (asymptotically) normal influence functions and
    # converges to chi²(r)/r as G → ∞. Falls back to chi² when G is unknown
    # or too small relative to r.
    if G is not None and G > rank + 1:
        f_stat = W * (G - rank) / (rank * (G - 1))
        pvalue = float(stats.f.sf(f_stat, rank, G - rank))
    else:
        pvalue = float(stats.chi2.sf(W, rank))

    return {
        "statistic": W,
        "df": rank,
        "n_cells": k,
        "rank": rank,
        "rank_deficient": rank < k,
        "pvalue": pvalue,
    }


# ======================================================================
# Repeated cross-sections (panel=False) branch
# ======================================================================


def _rcs_control_mask(
    g_arr: np.ndarray,
    g_val: int,
    t_val: int,
    base_val: int,
    control_group: str,
    notyet_cutoff: str,
    anticipation: int,
) -> np.ndarray:
    """Comparison-arm rows of one (g, t) cell on the observation-level paths."""
    if control_group == "nevertreated":
        return g_arr == 0
    if notyet_cutoff == "asinr":
        return (g_arr == 0) | (g_arr > t_val)
    if notyet_cutoff == "cohort":
        return (g_arr == 0) | (g_arr > max(t_val, g_val))
    # notyettreated: untreated through both periods (R did's rule)
    return (g_arr == 0) | (g_arr > max(t_val, base_val) + anticipation)


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
    w_arr: Optional[np.ndarray] = None,
    anticipation: int = 0,
    notyet_cutoff: str = "period",
    variance_convention: str = "delta",
) -> Tuple[float, float, np.ndarray]:
    """Sant'Anna-Zhao ATT(g, t) for one (g, t) cell of a repeated cross-section.

    Builds the two-period sub-sample — observations in period ``base_val`` or
    ``t_val`` belonging to cohort ``g_val`` or the control group — and hands it
    to the matching :mod:`statspai.did._rcs` estimator.  The cell-level
    influence function is then embedded in the full ``n_obs`` universe with the
    ``n_obs / n_rel`` rescaling the panel path uses, so cross-(g, t)
    aggregation in :func:`sp.aggte` sees influence functions on a common scale.
    """
    from ._rcs import drdid_imp_rc, drdid_rc, ipw_did_rc, reg_did_rc, std_ipw_did_rc

    is_treated = g_arr == g_val
    is_control = _rcs_control_mask(
        g_arr, g_val, t_val, base_val, control_group, notyet_cutoff, anticipation
    )

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
    if x_sub is not None:
        # Covariates constant or collinear within the cell drop out, as R's
        # lm() aliases them; kept, they made every regression singular and
        # the cell silently empty.
        from ._etwfe_glm_fit import _independent_columns

        live = _independent_columns(np.column_stack([np.ones(len(x_sub)), x_sub]))
        cols = [j - 1 for j in live if j > 0]
        x_sub = x_sub[:, cols] if cols else None
    # ω is renormalised to mean one *within the cell* by the engine, which
    # is what DRDID does — the ATT for this (g, t) is a within-cell
    # quantity, so the cell's own weight mass is the right denominator.
    w_sub = None if w_arr is None else w_arr[relevant]

    # All four treatment x period cells must be populated; the RCS estimators
    # raise otherwise, which for a single (g, t) cell should degrade to "no
    # estimate" rather than kill the whole fit.
    fn = {
        "dr": drdid_rc,
        "drimp": drdid_imp_rc,
        "ipw": std_ipw_did_rc,
        "stdipw": std_ipw_did_rc,
        "ipw_abadie": ipw_did_rc,
        "reg": reg_did_rc,
    }[estimator]
    extra = {"variance_convention": variance_convention} if estimator == "dr" else {}
    try:
        res = fn(y_sub, post_sub, d_sub, x_sub, weights=w_sub, **extra)
    except (DataInsufficient, MethodIncompatibility, np.linalg.LinAlgError):
        return 0.0, np.inf, np.zeros(n_obs)
    if not np.isfinite(res.att) or not np.isfinite(res.se):
        return 0.0, np.inf, np.zeros(n_obs)

    inf_full = np.zeros(n_obs)
    inf_full[idx] = res.influence * (n_obs / n_rel)
    return float(res.att), float(res.se), inf_full


def _estimate_single_att_rcs(
    y_arr: np.ndarray,
    g_arr: np.ndarray,
    t_arr: np.ndarray,
    g_val: int,
    t_val: int,
    base_val: int,
    n_obs: int,
    w_arr: Optional[np.ndarray] = None,
    control_group: str = "nevertreated",
    notyet_cutoff: str = "period",
    anticipation: int = 0,
) -> Tuple[float, float, np.ndarray]:
    """Observation-level 2×2 cell-mean DID + influence function.

    The comparison arm follows ``control_group`` with the same rule as
    :func:`_estimate_single_att_rcs_sz`; until 1.32 it was hard-wired to
    the never-treated units, so ``control_group='notyettreated'`` was
    ignored on this path (unbalanced panels, ``estimator='reg'`` without
    covariates) while ``model_info`` still reported it.

    With ``w_arr`` every cell mean becomes an ω-weighted mean and every
    cell share an ω-mass share, so the estimand is the ω-weighted ATT
    rather than the unweighted one. The influence function picks up the
    same ω factor, which is what keeps the reported SE consistent for the
    quantity actually estimated.
    """
    # The cohort itself satisfies the not-yet rule in its pre-treatment
    # cells; it is the treated arm, never its own comparison.
    is_control = _rcs_control_mask(
        g_arr, g_val, t_val, base_val, control_group, notyet_cutoff, anticipation
    ) & (g_arr != g_val)
    m_gt = (g_arr == g_val) & (t_arr == t_val)
    m_gb = (g_arr == g_val) & (t_arr == base_val)
    m_ct = is_control & (t_arr == t_val)
    m_cb = is_control & (t_arr == base_val)

    # Any empty cell kills the estimator for this (g, t).
    for m in (m_gt, m_gb, m_ct, m_cb):
        if m.sum() < 2:
            return 0.0, np.inf, np.zeros(n_obs)

    w = np.ones(n_obs, dtype=float) if w_arr is None else np.asarray(w_arr, float)

    def _wmean(mask: np.ndarray) -> float:
        mass = w[mask].sum()
        if mass <= 0:
            return float("nan")
        return float((w[mask] * y_arr[mask]).sum() / mass)

    mu_gt, mu_gb = _wmean(m_gt), _wmean(m_gb)
    mu_ct, mu_cb = _wmean(m_ct), _wmean(m_cb)
    if not all(np.isfinite([mu_gt, mu_gb, mu_ct, mu_cb])):
        return 0.0, np.inf, np.zeros(n_obs)

    att = float((mu_gt - mu_gb) - (mu_ct - mu_cb))

    # Cell shares are shares of ω-mass, not head counts.
    total = w.sum()
    p_gt = w[m_gt].sum() / total
    p_gb = w[m_gb].sum() / total
    p_ct = w[m_ct].sum() / total
    p_cb = w[m_cb].sum() / total

    inf = np.zeros(n_obs)
    inf[m_gt] += w[m_gt] * (y_arr[m_gt] - mu_gt) / p_gt
    inf[m_gb] += -w[m_gb] * (y_arr[m_gb] - mu_gb) / p_gb
    inf[m_ct] += -w[m_ct] * (y_arr[m_ct] - mu_ct) / p_ct
    inf[m_cb] += w[m_cb] * (y_arr[m_cb] - mu_cb) / p_cb

    # SE from sample variance of the influence function.
    se = float(np.sqrt(np.mean(inf**2) / n_obs))
    return att, se, inf
