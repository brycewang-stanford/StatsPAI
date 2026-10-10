"""de Chaisemartin & D'Haultfœuille (2024) intertemporal event-study DiD.

Differs from ``sp.did_multiplegt`` (dCDH 2020 DID_M): the 2020 estimator
is a consecutive-period switcher-vs-stayer pair rollup, while this 2024
estimator is a **long-difference event study** — at each horizon l ≥ 0,
compares ``Y_{F+l} − Y_{F−1}`` between units first switching at F and a
"not-yet-treated at F+l" control group held stable across the horizon.

Verified anchor
---------------
- de Chaisemartin & D'Haultfœuille (2024) "Difference-in-Differences
  Estimators of Intertemporal Treatment Effects", DOI
  ``10.1162/rest_a_01414`` (bib key ``dechaisemartin2024difference``).

Identification details
----------------------
Every item below is pinned against the authors' own implementations:
R ``DIDmultiplegtDYN`` 2.3.4 (Track A module ``78_multiplegt_dyn``,
``tests/reference_parity/test_multiplegt_dyn_parity.py``) and Stata
``did_multiplegt_dyn`` (``tests/stata_parity/78_multiplegt_dyn.do``), plus
the castle-doctrine panel in
``tests/test_did_multiplegt_dyn_castle_reference.py``. Effects, placebos,
switcher counts, the switcher-weighted aggregate and -- since 1.25.1 --
the analytic standard errors agree to better than 1e-6 relative.

1. **Switcher definition**: each unit's FIRST treatment change, in either
   direction, at period F. Switch-off events are switchers too -- that is
   the design's whole point -- and they are handled the way the reference
   does: their controls must share the switcher's BASELINE treatment level
   (a unit going 1 -> 0 belongs against units that were at 1 and stayed),
   and the difference is divided by the change in treatment so both
   directions measure the same effect per unit of treatment.

2. **Control group per horizon l**: "not-yet-treated at F+l" = units
   whose d stays at its pre-F value through F+l inclusive, which is
   what reproduces the R package's per-horizon samples (matching
   switcher and observation counts, not just point estimates). The
   never-treated-only variant is exposed as ``control='never_treated'``
   and is not separately pinned.

3. **Per-horizon estimate**:

       δ_l = Σ_F w_F × {E[Y_{F+l} − Y_{F−1} | switchers at F]
                        − E[Y_{F+l} − Y_{F−1} | not-yet-treated at F+l]}

   with weights ``w_F`` proportional to the (weighted) number of
   switchers at F. The heteroskedastic-weights variant (dCDH 2023 EJ
   survey) is not implemented.

4. **Placebo lag l < 0** is the mirror image of effect ``|l| − 1`` about
   the base period: the outcome contrast is ``Y_{F−1−|l|} − Y_{F−1}``,
   but the *sample* is the one behind effect ``|l| − 1`` -- switchers
   observed at ``F − 1 + |l|`` and controls not yet switched at
   ``F − 1 + |l|`` -- with the extra requirement that ``Y_{F−1−|l|}`` be
   observed. This is what the reference calls the placebo being computed
   "on the same switchers and controls as the corresponding effect".

   .. versionchanged:: 1.26.0
      ⚠️ Placebo lags 2 and deeper used to take the controls not yet
      switched at ``F`` (the lag-1 rule) and did not require the
      switcher to be observed at ``F − 1 + |l|``. That is a different
      quantity: on the castle-doctrine panel ``placebo_2`` moved from
      0.062241 to 0.050966 and ``placebo_3`` from 0.015389 (21
      switchers) to 0.023244 (20 switchers). Placebo 1 was already the
      same object under both rules and is unchanged.

   .. versionchanged:: 1.21.0
      ⚠️ Before 1.21.0 this was ``Y_{F−1−|l|} − Y_{F−1−|l|−1}`` -- a
      one-period difference sliding backwards rather than a mirrored
      long difference.

5. **Inference**: ``se_method='analytic'`` is the authors' influence
   function variance (``U_Gg_var`` in the package source): every
   (g, t) contribution is a *cell-demeaned* residual, ``diff_y − Ê``,
   where the cell is the cohort for a switcher and the (baseline, t)
   control set for a control, multiplied by the small-sample factor
   ``sqrt(n_cell / (n_cell − 1))`` with ``n_cell`` the number of distinct
   clusters in the cell. A cell with a single cluster (e.g. a cohort with
   one switcher) is centred on and scaled by the pooled switcher+control
   cell at that period instead -- without this a lone switcher would
   contribute a zero residual and the variance would be understated.
   Contributions are summed within ``cluster`` before squaring, and the
   variance is ``Σ_c (Σ_{g∈c} U_g)² / G²`` with ``G`` the number of
   groups.

   .. versionchanged:: 1.26.0
      ⚠️ The previous analytic variance used the plain two-sample
      influence function (no DOF factor, zero residual for
      single-switcher cohorts, no cluster summation). It ran 4-18 %
      below the reference on the castle-doctrine panel and ~1 % below
      on the Track A fixture. It now reproduces ``DIDmultiplegtDYN`` /
      Stata ``did_multiplegt_dyn`` to 1e-6 relative on every horizon
      and on the switcher-weighted aggregate. The default stays
      ``'bootstrap'``; ``model_info['se_method']`` records the choice.

Options
-------
``controls=``, ``trends_nonparam=``, ``normalized=`` and ``continuous=``
reproduce ``DIDmultiplegtDYN`` 2.3.4 to machine precision (2e-16 to 9e-16
relative on every effect and placebo); see
``tests/reference_parity/test_dcdh_options_parity.py``.

``continuous=k`` is the escape hatch for Design Restriction 1(i): when every
group has a different period-one treatment there is no group to match a
switcher against, so the status-quo outcome *evolution* is modelled as a
degree-``k`` polynomial in the period-one treatment instead, fitted per
period on the not-yet-switched cells and residualised out. The treatment may
then be non-binary, which is the one case where that check is relaxed.

Discrete treatments
-------------------
The treatment may take any number of values (a count of newspapers, a tax
rate on a grid). Three things then follow the reference, all checked against
Stata ``did_multiplegt_dyn`` in
``tests/reference_parity/test_dcdh_textbook_stata_parity.py``:

* periods are ranked, so a panel observed every four years is handled like
  an annual one;
* the ``(g, t)`` cells at which a group has been both above and below its
  period-one treatment are dropped (Design Restriction 2);
* a switcher is compared with the groups that had its period-one treatment
  and have not changed yet, and in the variance it is centred within the
  groups that also switched at the same period to the same treatment.

``aggregation='switchers'`` is the reference's ``Av_tot_eff``: the effects of
all horizons added up and divided by the treatment changes that produced
them, so it is an effect per unit of treatment whether or not
``normalized=True``.

Where the reference is not followed
-----------------------------------
On an unbalanced panel the reference first drops, for each period-one
treatment, the periods at which no group with that treatment is still
unswitched, and then drops the groups whose remaining post-switch treatment
averages to their period-one treatment. A period without controls in the
middle of the panel (controls exist again later) therefore removes the switch
period of a group that switched then; if that group's treatment is back at
its starting level when controls reappear, the second rule discards the
group altogether, including the periods *before* its switch in which it was a
valid not-yet-switched control. The source comments say the second rule "can
only arise if dont_drop_larger_lower specified", so this is not intended.
Here such a group stays a control until it switches. On the Gentzkow,
Shapiro and Sinkinson (2011) newspaper panel this is one county out of
1,195: three switchers keep their only control, 1,122 switchers contribute to
the first effect instead of 1,119, and the effect is 0.014548 against
0.014424. Removing that county from the input reproduces the reference on
every effect, placebo, standard error and test.

With ``controls=`` the covariate slopes are estimated, and the analytic
variance carries the term for that (``U^{var,X}`` of the companion paper):
each group's influence loses ``M_d' b_g``, with ``M_d`` the derivative of
the effect with respect to the slopes of period-one treatment ``d`` and
``b_g`` the group's contribution to their estimation error. Estimates and
standard errors agree with the reference weighted and clustered, within
``trends_nonparam`` cells, normalized and on unbalanced panels.

Not implemented
---------------
``trends_lin``, ``predict_het`` and the
heteroskedastic-weights variant. See ``docs/rfc/multiplegt_dyn.md``.

``trends_lin`` was implemented as the reference documents it -- an event
study on the outcome's first difference, summed over horizons -- and does
*not* reproduce the reference, so it is deliberately absent rather than
wrong: on that panel the reference returns
``[1.443025, 3.141753, 4.781375, 7.004474]`` while summing this estimator's
first-difference effects gives ``[1.443025, 2.998724, 4.659879, 6.514063]``.
The two agree at horizon 0 and diverge after. What is known: the
reference's estimates do not depend on how many effects are requested,
while the switcher counts it reports do (2, 3 or 4 requested effects give
39, 29 or 16 for every horizon), so the counts are a reporting artefact and
each effect has its own sample; the gap is not the pre-trend slope times
the horizon, and is not explained by holding the switcher composition
fixed. Naming an argument after the reference's option while computing a
different number is the one outcome worth avoiding, so the argument is not
offered until the arithmetic is located.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core._bootstrap import bootstrap_se as _bootstrap_se
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility, StatsPAIError
from . import _core as _dc
from . import _dcdh_arrays as _arr

# What a resampled panel can raise when an event has no comparison cell or
# a cell is empty; the replicate is then recorded as missing.
_REPLICATE_ERRORS = (
    StatsPAIError,
    ArithmeticError,
    LookupError,
    ValueError,
    np.linalg.LinAlgError,
)


@accepts_aliases(
    covariates="controls",
    _strict=True,
    id="group",
    unit="group",
    treat="treatment",
    weight="weights",
)
def did_multiplegt_dyn(
    data: pd.DataFrame,
    y: str,
    *,
    group: str,
    time: str,
    treatment: str,
    placebo: int = 0,
    dynamic: int = 3,
    control: str = "not_yet_treated",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    aggregation: str = "simple",
    se_method: str = "bootstrap",
    switchers: Optional[str] = None,
    same_switchers: bool = False,
    effects_equal: Any = False,
    controls: Optional[List[str]] = None,
    trends_nonparam: Optional[List[str]] = None,
    normalized: bool = False,
    continuous: Optional[int] = None,
    design: Optional[float] = None,
    by_path: Optional[int] = None,
    normalized_weights: bool = False,
) -> CausalResult:
    """dCDH (2024) intertemporal event-study DiD estimator.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome column.
    group : str
        Unit identifier.
    time : str
        Integer-valued period column.
    treatment : str
        Time-varying treatment: binary, or discrete (a count, a level).
        Groups are switchers from the first period their treatment differs
        from its period-one value, in either direction (``switchers=``
        separates the two), and are compared with groups that started at
        the same treatment and have not changed yet.

        With a non-binary treatment the effect at horizon ``l`` is the
        effect of having been on the group's own treatment path for
        ``l + 1`` periods rather than at its period-one treatment, so it
        mixes paths of different sizes; ``normalized=True`` divides by the
        treatment received and ``design=`` / ``by_path=`` show and separate
        the paths. The ``(g, t)`` cells at which a group has been both
        above and below its period-one treatment are dropped (Design
        Restriction 2; their count is in
        ``model_info['n_dropped_bidirectional']``).

        .. versionchanged:: 1.39.0
           Non-binary treatments used to be refused.
    placebo : int, default 0
        Number of pre-treatment placebo horizons (l = -1, ..., -placebo).
    dynamic : int, default 3
        Number of post-treatment dynamic horizons (l = 0, ..., dynamic),
        i.e. ``dynamic + 1`` effects. R / Stata ``effects=k`` is
        ``dynamic=k-1`` here: their ``Effect_k`` is horizon ``k-1``.
    control : {'not_yet_treated', 'never_treated'}, default
        ``'not_yet_treated'``.
    cluster : str, optional
        Cluster column (defaults to ``group``). Used by the bootstrap and,
        since 1.25.1, by the analytic variance: influence contributions
        are summed within cluster before squaring, and the small-sample
        cell factor counts distinct clusters, exactly as the reference
        does. ``group`` must be nested in ``cluster``.
    weights : str, optional
        Observation weight column (Stata ``weight()`` / R ``weight=``, whose
        spelling is accepted here as the alias ``weight=``).
        The weight is read at the row a unit contributes from -- period
        ``F+l`` for an effect and ``F-1+|l|`` for a placebo, mirroring the
        reference -- so time-varying weights behave identically. Every
        switcher mean, control mean, cohort weight and the switcher counts
        used by ``aggregation='switchers'`` become weighted; ``detail``
        keeps the raw count in ``n_switchers`` and the weighted one in
        ``w_switchers``. Must be non-negative; a zero or missing weight
        drops the row.
    n_boot : int, default 500
        Bootstrap replications.
    alpha : float, default 0.05
    seed : int, optional
    se_method : {"bootstrap", "analytic"}, default "bootstrap"
        ``"bootstrap"`` resamples clusters; ``"analytic"`` is the authors'
        influence-function variance and needs no draws, which makes it
        roughly a hundred times faster. It is pinned against
        ``DIDmultiplegtDYN`` and Stata ``did_multiplegt_dyn`` to 1e-6
        relative on every horizon and on the switcher-weighted aggregate
        (module docstring, item 5). The default stays on the bootstrap so
        existing callers' numbers do not move; ``model_info["se_method"]``
        records the choice.

        With ``"analytic"`` the joint tests (``joint_placebo_test``,
        ``joint_effects_test``, ``joint_overall_test``,
        ``effects_equal_test``) are Wald tests on the analytic joint
        covariance of the horizons, which is how the reference computes its
        "joint nullity" and "equality of the effects" p-values, and no
        bootstrap replicate is drawn (``n_boot`` is ignored). With
        ``"bootstrap"`` they use the covariance of the replicates.

        .. versionchanged:: 1.39.0
           The joint tests of an analytic fit used to come from ``n_boot``
           bootstrap replicates, which were drawn even though the standard
           errors did not use them.
    aggregation : {"simple", "switchers"}, default "simple"
        How the dynamic horizons are combined into the headline
        ``estimate``. ``"simple"`` gives each horizon equal weight;
        ``"switchers"`` is ``DIDmultiplegtDYN``'s ``Av_tot_eff`` and its
        standard error: the non-normalized effects weighted by the
        (weighted) number of switchers behind each, divided by the
        treatment changes in place at those horizons. For a binary
        treatment that stays switched this is the switcher-weighted mean
        of the effects; when the treatment can come back, or is not
        binary, it is an effect per unit of treatment. The two differ
        whenever later horizons rest on fewer cohorts, which is the normal
        case in staggered designs. The default is left on ``"simple"``
        because changing it would move the number existing callers get
        back; ``model_info["aggregation"]`` records which was used.

        .. versionchanged:: 1.39.0
           ⚠️ ``"switchers"`` did not divide by the treatment change, so
           on a panel where switchers return to their starting treatment
           within the horizons it was not ``Av_tot_eff``. Unchanged for a
           binary treatment that stays switched.
    switchers : {None, 'in', 'out'}, optional
        Estimate on switch-**in** events (treatment rises above its
        period-one level) or switch-**out** events (falls below) only.
        Stata ``did_multiplegt_dyn, switchers()``. Default ``None`` pools
        both, which is what StatsPAI has always done.

        dCDH recommend running the two separately: pooling is only
        meaningful if a switch up and a switch down move the outcome by
        the same amount per unit of treatment, which is an assumption, not
        a fact. Splitting is the way to check it.
    same_switchers : bool, default False
        Restrict the treated arm to switchers whose effect can be
        estimated at *every* requested horizon, so the composition is
        held fixed across ℓ. Stata ``did_multiplegt_dyn, same_switchers``.

        .. versionchanged:: 1.39.0
           ⚠️ Being observed at every horizon was the only requirement, so
           a switcher that ran out of not-yet-switched controls at a long
           horizon still entered the short ones and the composition was
           not fixed. On a staggered panel without never-treated groups
           that is every late cohort.

        Without it, later horizons rest on fewer — and differently
        selected — switchers, so a rising or falling ℓ-profile confounds
        the true dynamic path with a moving sample. With it, the profile
        is comparable across ℓ but rests on a smaller sample. Availability
        is judged per unit against the periods that unit is actually
        observed in, so this is correct on unbalanced panels.
    effects_equal : bool or (int, int), default False
        Test H0 that the dynamic effects are all equal. ``True`` tests
        every estimated effect; a ``(lower, upper)`` pair tests the closed
        horizon range, matching Stata ``did_multiplegt_dyn,
        effects_equal()``.

        Reported in ``model_info['effects_equal_test']`` as
        ``{'statistic', 'df', 'pvalue', 'horizons'}``. The statistic is
        χ² on ``k−1`` degrees of freedom for ``k`` effects — one fewer
        than the all-zero joint test, since equality leaves the common
        level unrestricted. Rejecting says the effect moves with exposure
        length; failing to reject does **not** establish a constant effect.
    controls : list of str, optional
        Covariates. These do not enter a regression of the outcome: the
        option replaces the outcome's first difference with the residual
        from a regression of that first difference on the first differences
        of the covariates and time fixed effects, fitted on the (g, t)s
        whose treatment has not changed yet and separately for each value
        of the baseline treatment. The resulting estimators are unbiased
        under differential trends that a linear model in covariate changes
        explains. To adjust for a time-invariant covariate, interact it
        with the time variable first. The regression uses ``weights=``
        and is fitted within ``trends_nonparam`` cells when both are
        given, and the analytic standard errors account for the
        estimation of its slopes, as the reference's do.

        .. versionchanged:: 1.39.0
           ⚠️ The regression is now fitted on the groups that never
           switch as well; it used only the pre-switch periods of the
           groups that do, and left the never-switchers' outcomes
           unadjusted. On a panel with holes a covariate change between
           two rows that are not consecutive periods is no longer used as
           a one-period change. The regression is weighted when
           ``weights=`` is given and fitted within ``trends_nonparam``
           cells, and the analytic variance has the slope-estimation
           term (it was 0.1 to 0.6% off without it). Unweighted panels
           where every group switches and no period is missing keep
           their estimates; every analytic standard error with
           ``controls=`` changes slightly.

        .. versionadded:: 1.31.0
    trends_nonparam : list of str, optional
        Time-invariant variables whose values a control must share with the
        switcher, on top of sharing its baseline treatment. The estimators
        are then unbiased even when groups trend differently, provided
        groups with the same value of these variables trend in parallel.

        .. versionadded:: 1.31.0
    continuous : int, optional
        Degree of the polynomial in the period-one treatment to model the
        status-quo outcome evolution with. Use it when groups' period-one
        treatments are continuous, so that no two groups share one and the
        baseline match the other estimators rely on is impossible (Design
        Restriction 1(i) of dCDH 2024). The polynomial is fitted per period
        on the (g, t)s that have not switched yet and residualised out of the
        outcome's first difference; with it, ``treatment`` need not be
        binary, and controls are no longer required to share the switcher's
        baseline.

        .. versionadded:: 1.31.0
    normalized : bool, default False
        Report effects per unit of treatment: each horizon's effect is
        divided by the average cumulative treatment change its switchers
        received between the base period and that horizon. For a binary
        absorbing switch the divisor is the number of periods of exposure,
        so a flat normalized series is a constant per-period effect.

        .. versionadded:: 1.31.0
    design : float, optional
        Describe the treatment paths behind the last requested effect:
        ``model_info['design']`` lists, most frequent first, the paths
        ``(D_{F-1}, D_F, ..., D_{F+dynamic})`` followed by at least this
        share (between 0 and 1) of the switchers behind the last requested
        effect, with the number of groups on each. Stata ``design(p,
        console)`` counts every switcher with an observed path, estimable
        effect or not, so its totals are a little larger and its shares a
        little smaller. With a non-binary or non-absorbing
        treatment this is what tells which treatment trajectories an
        event-study effect averages over.
    by_path : int, optional
        Estimate the effects separately for the ``by_path`` most frequent
        treatment paths (Stata ``by_path()``). Each path's switchers are
        compared with the same not-yet-switched controls as in the pooled
        estimation, and only switchers with every requested effect
        estimable enter. ``model_info['by_path']`` is a list of
        ``{'path', 'n_switchers', 'event_study', 'estimate', 'se',
        'joint_effects_test'}``. Needs ``se_method='analytic'``.
    normalized_weights : bool, default False
        With ``normalized=True``: ``model_info['normalized_weights']``, the
        weight the normalized effect at each horizon puts on the effect of
        the current treatment and of each of its lags (rows ``k`` = lag,
        columns = horizon; each column sums to one). Stata
        ``normalized_weights``.

    Returns
    -------
    CausalResult with ``detail`` = per-event decomposition and
    ``model_info['event_study']`` = horizon-level DataFrame matching the
    canonical event-study schema (so ``sp.did_plot`` works).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(20):
    ...     g = int(rng.choice([5, 8, 0]))  # cohort; 0 = never treated
    ...     for t in range(1, 13):
    ...         d = 1 if (g != 0 and t >= g) else 0
    ...         rows.append({'i': i, 't': t, 'd': d,
    ...                      'y': i + 0.1 * t + 1.0 * d + rng.normal(0, 0.4)})
    >>> df = pd.DataFrame(rows)
    >>> r = sp.did_multiplegt_dyn(
    ...     df, y='y', group='i', time='t', treatment='d',
    ...     placebo=2, dynamic=4, n_boot=50, seed=0,
    ... )
    >>> es = r.model_info['event_study']  # horizon-level event study
    >>> bool(len(es) > 0)
    True
    >>> sens = sp.honest_did(r, m_grid=[0.5])  # Rambachan-Roth sensitivity
    """
    if switchers not in (None, "in", "out"):
        raise ValueError(
            f"switchers must be None, 'in' or 'out', got {switchers!r}. "
            "dCDH recommend estimating switch-in and switch-out effects "
            "separately rather than pooling them."
        )
    if control not in {"not_yet_treated", "never_treated"}:
        raise ValueError(
            f"control={control!r} must be 'not_yet_treated' or 'never_treated'"
        )
    if dynamic < 0 or placebo < 0:
        raise ValueError("dynamic and placebo must be non-negative")
    if design is not None and not 0 < float(design) <= 1:
        raise MethodIncompatibility(
            f"design must be a share in (0, 1], got {design!r}.",
            diagnostics={"design": repr(design)},
        )
    if by_path is not None and (isinstance(by_path, bool) or int(by_path) < 1):
        raise MethodIncompatibility(
            f"by_path must be a positive integer, got {by_path!r}.",
            diagnostics={"by_path": repr(by_path)},
        )
    if (design is not None or by_path is not None) and continuous is not None:
        raise MethodIncompatibility(
            "design= and by_path= describe discrete treatment paths and "
            "cannot be combined with continuous=."
        )
    if normalized_weights and not normalized:
        raise MethodIncompatibility(
            "normalized_weights=True describes the normalized effects: pass "
            "normalized=True as well."
        )
    if aggregation not in {"simple", "switchers"}:
        raise ValueError(
            f"aggregation must be 'simple' or 'switchers', got {aggregation!r}"
        )
    if cluster is not None and cluster not in data.columns:
        raise MethodIncompatibility(
            f"cluster column {cluster!r} not in data",
            diagnostics={"cluster": cluster},
        )
    se_method = _dc.normalize_se_method(
        se_method,
        supported=("analytic", "bootstrap"),
        function="did_multiplegt_dyn",
        n_clusters=int(data[cluster or group].nunique()),
    )

    df = data.copy()
    for col in (y, group, time, treatment):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in data")
    if not pd.api.types.is_numeric_dtype(df[treatment]):
        raise MethodIncompatibility(
            f"Treatment {treatment!r} must be numeric (binary, or a discrete "
            "level such as a count).",
            diagnostics={"treatment": treatment, "dtype": str(df[treatment].dtype)},
        )
    if weights is not None:
        if weights not in df.columns:
            raise MethodIncompatibility(
                f"weights column {weights!r} not in data",
                diagnostics={"weights": weights},
            )
        wv = pd.to_numeric(df[weights], errors="coerce")
        if (wv.dropna() < 0).any():
            raise MethodIncompatibility(
                f"weights column {weights!r} has negative values",
                diagnostics={"weights": weights},
            )
        df[weights] = wv
    if cluster is not None:
        if df[cluster].isna().any():
            raise MethodIncompatibility(
                f"cluster column {cluster!r} has missing values",
                diagnostics={"cluster": cluster},
            )
        if (df.groupby(group)[cluster].nunique() > 1).any():
            raise MethodIncompatibility(
                f"group {group!r} must be nested within cluster {cluster!r}",
                diagnostics={"group": group, "cluster": cluster},
            )

    # Periods are ranked, as the reference does (``egen time = group(T)``):
    # the estimator only ever speaks of "the period before" and "l periods
    # later", so a panel observed every four years is handled like an
    # annual one. Everything below works on the rank.
    df = df[df[time].notna()].copy()
    df["_tidx"] = pd.factorize(df[time], sort=True)[0] + 1
    time_label = time
    time = "_tidx"
    df = df.sort_values([group, time]).reset_index(drop=True)
    cluster_var = cluster if cluster is not None else group

    # Design Restriction 2 of dCDH (2024): drop the (g, t) cells at which
    # the group has by then had both a strictly higher and a strictly lower
    # treatment than its period-one treatment. Their effect is a mix of an
    # increase and a decrease and has no sign. A binary treatment can never
    # be on both sides of its starting value, so nothing is dropped there.
    n_bidirectional = 0
    if continuous is None:
        d_num = df[treatment].astype(float)
        first_d = d_num.groupby(df[group]).transform("first")
        above = (d_num > first_d).groupby(df[group]).cummax()
        below = (d_num < first_d).groupby(df[group]).cummax()
        both = (above & below).to_numpy()
        n_bidirectional = int(both.sum())
        if n_bidirectional:
            df = df.loc[~both].reset_index(drop=True)

    controls = list(controls) if controls else None
    trends_nonparam = list(trends_nonparam) if trends_nonparam else None
    for name, cols in (("controls", controls), ("trends_nonparam", trends_nonparam)):
        for col in cols or []:
            if col not in df.columns:
                raise MethodIncompatibility(
                    f"{name} column {col!r} not in data",
                    diagnostics={name: cols},
                )
    if trends_nonparam:
        varying = df.groupby(group)[trends_nonparam].nunique().max()
        if int(varying.max()) > 1:
            raise MethodIncompatibility(
                "trends_nonparam variables must be time-invariant within a "
                "group; they define the cells switchers are compared inside.",
                diagnostics={"trends_nonparam": trends_nonparam},
            )
        df["_tcell"] = df[trends_nonparam].astype(str).agg("\u241f".join, axis=1)

    # Identify each unit's FIRST switch, in either direction, and which way
    # it went. A unit that turns treatment off is as much a switcher as one
    # that turns it on -- that is the whole point of the dCDH design, and
    # dropping those events (as this used to) silently changes the estimand.
    first_switch, switch_dir, baseline = _first_switch(
        df, group=group, time=time, treatment=treatment
    )
    df = df.merge(first_switch, on=group, how="left")
    df = df.merge(switch_dir, on=group, how="left")
    df = df.merge(baseline, on=group, how="left")

    # controls= works by changing the outcome the estimator differences,
    # so it is applied once, here, and the rest of the pipeline is
    # untouched.
    y_work = y
    if continuous is not None:
        if (
            isinstance(continuous, bool)
            or not isinstance(continuous, (int, np.integer))
            or continuous < 1
        ):
            raise MethodIncompatibility(
                "continuous= is the degree of the polynomial in the "
                "period-one treatment, so it must be a positive integer; got "
                f"{continuous!r}.",
                diagnostics={"continuous": repr(continuous)},
            )
        df, y_work = _residualise_on_baseline_polynomial(
            df, y=y_work, group=group, time=time, degree=int(continuous)
        )
    controls_info: Optional[Dict[float, Dict[str, Any]]] = None
    if controls:
        df, y_work, controls_info = _residualise_on_controls(
            df,
            y=y,
            group=group,
            time=time,
            treatment=treatment,
            controls=controls,
            weights=weights,
            cluster=cluster_var,
        )
        if continuous is not None:
            # the polynomial in the period-one treatment replaces the
            # baseline match; the slope-estimation term is not derived there
            controls_info = None
    # Horizons list: placebo (negative) + dynamic (0..H).
    horizons = list(range(-placebo, dynamic + 1))
    # l = -1 is a genuine placebo, not a mechanical zero: it is
    # Y_{F-2} - Y_{F-1} differenced against the controls, which is only
    # zero in expectation under parallel trends. It maps to the R
    # package's Placebo_1.

    main = _estimate_all_horizons(
        df=df,
        y=y_work,
        group=group,
        time=time,
        treatment=treatment,
        horizons=horizons,
        control=control,
        switchers=switchers,
        same_switchers=same_switchers,
        weights=weights,
        cluster=cluster_var,
        normalized=normalized,
        match_baseline=continuous is None,
        controls=controls,
        controls_info=controls_info,
    )

    if not any(c["n_events"] for c in main["cell_estimates"]):
        # Every horizon came back empty: there is no (switcher, control)
        # pair anywhere in the design. Returning NaN here would look like
        # an estimate that happens to be missing rather than a design that
        # cannot be estimated.
        raise DataInsufficient(
            "No switcher could be matched with a control at any horizon: "
            "every event needs at least one unit that has not switched by "
            "the horizon's anchor period and shares the switcher's "
            "baseline treatment"
            + (" and its trends_nonparam cell" if trends_nonparam else "")
            + ". If the period-one treatment is continuous, so that no two "
            "groups share it, pass continuous= (the degree of a polynomial "
            "in the period-one treatment).",
            diagnostics={
                "n_groups": int(df[group].nunique()),
                "control": control,
                "trends_nonparam": trends_nonparam,
            },
        )

    # Cluster bootstrap for SE
    # The analytic variance comes with the horizons' joint covariance, so
    # every test below is computed from it and no replicate is drawn.
    if se_method == "analytic":
        n_boot = 0
    rng = np.random.default_rng(seed)
    boot_hist = np.full((n_boot, len(horizons)), np.nan)
    # A replicate copies whole groups, so it is the main panel's matrices with
    # rows repeated: nothing has to be rebuilt from a resampled frame.
    boot_panel = None
    if n_boot > 0:
        boot_panel = _arr.build_panel(
            df,
            y=y_work,
            group=group,
            time=time,
            treatment=treatment,
            weights=weights,
            cluster=cluster_var,
        )
    if boot_panel is not None:
        resampler = _arr.Resampler(boot_panel, df, cluster_var)
        for b in range(n_boot):
            try:
                draw = resampler.draw(rng)
                directions = _switch_directions(switchers)
                if same_switchers and len(horizons) > 1:
                    draw = _arr.common_switchers(
                        draw,
                        horizons=horizons,
                        control=control,
                        directions=directions,
                        match_baseline=continuous is None,
                    )
                deltas = _arr.point_estimates(
                    draw,
                    horizons=horizons,
                    control=control,
                    directions=directions,
                    normalized=normalized,
                    match_baseline=continuous is None,
                )["delta"]
                if deltas is not None:
                    boot_hist[b, :] = deltas
            except _REPLICATE_ERRORS:
                continue  # replicate stays NaN; bootstrap_se tracks the failure
    # the same replicates on resampled frames, for layouts without matrices
    for b in range(n_boot if boot_panel is None else 0):
        try:
            bdf = _dc.cluster_bootstrap_draw(
                df,
                cluster_col=cluster_var,
                rng=rng,
                relabel_cols=[group],
            )
            # Recompute the switch date in the bootstrap sample the SAME
            # way the point estimate does.
            #
            # This used to be `min(time | d == 1)`, i.e. "first period
            # treated". That is the first *switch* only for switch-ON
            # units: a unit going 1 → 0 at F got _F = 1, its own first
            # period, which has no base period F−1 and so silently
            # dropped out of every replicate. The point estimate has
            # handled switch-off events since 1.21.0 via _first_switch;
            # the bootstrap was never updated to match, so on any
            # non-absorbing panel the replicates were estimating a
            # different quantity than the estimate whose variance they
            # were supposed to describe.
            bdf = bdf.drop(
                columns=[c for c in ("_F", "_dir", "_base") if c in bdf.columns]
            )
            fs_b, dir_b, base_b = _first_switch(
                bdf, group=group, time=time, treatment=treatment
            )
            bdf = bdf.merge(fs_b, on=group, how="left")
            bdf = bdf.merge(dir_b, on=group, how="left")
            bdf = bdf.merge(base_b, on=group, how="left")
            best = _estimate_all_horizons(
                df=bdf,
                y=y_work,
                group=group,
                time=time,
                treatment=treatment,
                horizons=horizons,
                control=control,
                switchers=switchers,
                same_switchers=same_switchers,
                weights=weights,
                cluster=cluster_var,
                normalized=normalized,
                match_baseline=continuous is None,
            )
            for j, h in enumerate(horizons):
                # Align by h
                row = next(
                    (r for r in best["cell_estimates"] if r["horizon"] == h), None
                )
                if row is not None:
                    boot_hist[b, j] = row["delta_l"]
        except Exception:
            continue  # replicate stays NaN; bootstrap_se tracks the failure

    # Per-horizon SE + CI
    es_rows: List[Dict[str, Any]] = []
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    for j, h in enumerate(horizons):
        row = next((r for r in main["cell_estimates"] if r["horizon"] == h), None)
        if row is None:
            es_rows.append(
                {
                    "relative_time": h,
                    "att": np.nan,
                    "se": np.nan,
                    "pvalue": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "type": "placebo" if h < 0 else "dynamic",
                    "n_switchers": 0,
                }
            )
            continue
        est = row["delta_l"]
        if se_method == "analytic":
            se = row["_se_analytic"]
        else:
            se = _bootstrap_se(boot_hist[:, j], label=f"did.multiplegt_dyn[h={h}]")
        p = (
            float(2 * stats.norm.sf(abs(est / se)))
            if (se > 0 and np.isfinite(se))
            else np.nan
        )
        ci_lo = est - z_crit * se if (se > 0 and np.isfinite(se)) else np.nan
        ci_hi = est + z_crit * se if (se > 0 and np.isfinite(se)) else np.nan
        es_rows.append(
            {
                "relative_time": h,
                "att": float(est) if np.isfinite(est) else np.nan,
                "se": float(se) if np.isfinite(se) else np.nan,
                "pvalue": p,
                "ci_lower": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
                "ci_upper": float(ci_hi) if np.isfinite(ci_hi) else np.nan,
                "type": "placebo" if h < 0 else "dynamic",
                "n_switchers": int(row["n_switchers"]),
            }
        )

    es_df = _dc.event_study_frame(es_rows)

    # Joint covariance of the reported placebos and effects. Analytic: the
    # horizons' clustered influence sums, cross-multiplied -- the same
    # object whose diagonal is each horizon's SE (the reference's
    # sqrt(var_sq_sum)/G). Bootstrap: the covariance of the replicates.
    _ok = [
        (j, h)
        for j, h in enumerate(horizons)
        if np.isfinite(es_rows[j]["se"]) and es_rows[j]["se"] > 0
    ]
    es_vcov = None
    if _ok:
        _labels = [int(h) for _, h in _ok]
        if se_method == "analytic":
            _rows_h = {r["horizon"]: r for r in main["cell_estimates"]}
            _ncl = int(main["cluster_codes"].max()) + 1
            _S = np.column_stack(
                [
                    np.bincount(
                        main["cluster_codes"],
                        weights=_rows_h[h]["_influence"],
                        minlength=_ncl,
                    )
                    for _, h in _ok
                ]
            )
            _V: Optional[np.ndarray] = (_S.T @ _S) / float(main["n_groups"]) ** 2
        else:
            _B = boot_hist[:, [j for j, _ in _ok]]
            _B = _B[np.all(np.isfinite(_B), axis=1)]
            _V = np.cov(_B, rowvar=False, ddof=1) if len(_B) > 1 else None
        if _V is not None:
            es_vcov = pd.DataFrame(np.atleast_2d(_V), index=_labels, columns=_labels)

    # Joint tests
    placebo_idx = [j for j, h in enumerate(horizons) if h < 0]
    dyn_idx = [j for j, h in enumerate(horizons) if h >= 0]

    if se_method == "analytic":
        joint_placebo = _joint_test_from_vcov(es_rows, es_vcov, horizons, placebo_idx)
        joint_effects = _joint_test_from_vcov(es_rows, es_vcov, horizons, dyn_idx)
        joint_overall = _joint_test_from_vcov(
            es_rows, es_vcov, horizons, placebo_idx + dyn_idx
        )
    else:
        joint_placebo = _joint_test_from_boot(main, horizons, boot_hist, placebo_idx)
        joint_effects = _joint_test_from_boot(main, horizons, boot_hist, dyn_idx)
        joint_overall = _joint_test_from_boot(
            main, horizons, boot_hist, placebo_idx + dyn_idx
        )

    # effects_equal: H0 that the dynamic effects share a common value.
    # False disables it; True tests every estimated effect; (lo, hi) tests
    # the closed range, matching Stata's lower/upper bound form.
    equal_test = None
    equal_range = None
    if effects_equal is not False and effects_equal is not None:
        if effects_equal is True:
            sel = dyn_idx
            equal_range = (
                (horizons[dyn_idx[0]], horizons[dyn_idx[-1]]) if dyn_idx else None
            )
        else:
            try:
                lo, hi = effects_equal
            except (TypeError, ValueError):
                raise ValueError(
                    "effects_equal must be False, True, or a (lower, upper) "
                    f"pair of horizons, got {effects_equal!r}."
                ) from None
            lo, hi = int(lo), int(hi)
            if lo > hi:
                raise ValueError(f"effects_equal=({lo}, {hi}) has its bounds reversed.")
            sel = [j for j in dyn_idx if lo <= horizons[j] <= hi]
            if len(sel) < 2:
                raise ValueError(
                    f"effects_equal=({lo}, {hi}) selects {len(sel)} of the "
                    f"estimated effects {[horizons[j] for j in dyn_idx]}; the "
                    "range must cover at least two of them for an equality "
                    "test to mean anything."
                )
            equal_range = (lo, hi)
        if se_method == "analytic":
            equal_test = _joint_test_from_vcov(
                es_rows, es_vcov, horizons, sel, equal=True
            )
        else:
            equal_test = _effects_equal_test(main, horizons, boot_hist, sel)

    # Headline estimate over the dynamic horizons. "simple" gives each
    # horizon equal weights; "switchers" weights by the (weighted) switchers
    # behind each one, which is DIDmultiplegtDYN's Av_tot_eff.
    dyn_est = np.array(
        [es_rows[j]["att"] for j in dyn_idx],
        dtype=float,
    )
    rows_by_h = {r["horizon"]: r for r in main["cell_estimates"]}

    def _switcher_weight(j: int) -> float:
        r_h = rows_by_h.get(horizons[j])
        return float(r_h["w_switchers"]) if r_h is not None else 0.0

    def _dose_now(j: int) -> float:
        r_h = rows_by_h.get(horizons[j])
        d = float(r_h["dose_now"]) if r_h is not None else np.nan
        return d if np.isfinite(d) else 1.0

    # aggregation="switchers" is the reference's Av_tot_eff: the effects of
    # all horizons added up and divided by the treatment changes that
    # produced them, sum_l N_l delta_l / sum_l N_l delta^D_l, with delta^D_l
    # the switchers' average treatment change in place at horizon l. It is
    # built from the non-normalized effects whatever normalized= says. For
    # a binary treatment that stays switched delta^D_l is 1 and this is the
    # switcher-weighted mean of the effects.
    raw_est = np.array(
        [
            rows_by_h[horizons[j]]["_delta_raw"] if horizons[j] in rows_by_h else np.nan
            for j in dyn_idx
        ],
        dtype=float,
    )
    dose_vec = np.array([_dose_now(j) for j in dyn_idx], dtype=float)

    if not dyn_est.size:
        headline = np.nan
    elif aggregation == "switchers":
        w = np.array([_switcher_weight(j) for j in dyn_idx], dtype=float)
        ok = np.isfinite(raw_est) & (w > 0)
        headline = (
            float(np.sum(w[ok] * raw_est[ok]) / np.sum(w[ok] * dose_vec[ok]))
            if ok.any()
            else np.nan
        )
    else:
        headline = float(np.nanmean(dyn_est))
    # SE: cross-horizon bootstrap of the average. A replicate contributes
    # only when at least one dynamic horizon was estimated; a fully-failed
    # draw stays NaN so bootstrap_se can surface the failure rate.
    if dyn_idx and n_boot > 0:
        with warnings.catch_warnings():
            # nanmean of an all-NaN replicate row is an intended NaN.
            warnings.simplefilter("ignore", RuntimeWarning)
            if aggregation == "switchers":
                wb = np.array([_switcher_weight(j) for j in dyn_idx], dtype=float)
                sub = boot_hist[:, dyn_idx]
                if normalized:
                    # replicates hold normalized effects; undo with the
                    # point estimate's own divisors
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sub = sub * np.where(dyn_est != 0, raw_est / dyn_est, 1.0)
                mask = np.isfinite(sub)
                denom = (mask * wb * dose_vec).sum(axis=1)
                num = np.nansum(np.where(mask, sub, 0.0) * wb, axis=1)
                boot_avg = np.where(
                    denom > 0, num / np.where(denom > 0, denom, 1.0), np.nan
                )
            else:
                boot_avg = np.nanmean(boot_hist[:, dyn_idx], axis=1)
        se_avg = _bootstrap_se(boot_avg, label="did.multiplegt_dyn.headline")
    else:
        se_avg = np.nan
    if se_method == "analytic" and dyn_idx:
        # Combine the horizons' influence functions with the same weights the
        # headline uses, then square once -- the horizons share control units,
        # so adding their variances would understate the spread. This is the
        # reference's U_Gg_var_global: Σ_l w_l U_Gg_var_l with w_l ∝ N_l.
        psis, wts, dens = [], [], []
        for k, j in enumerate(dyn_idx):
            r_h = rows_by_h.get(horizons[j])
            if r_h is None or not np.isfinite(dyn_est[k]):
                continue
            if aggregation == "switchers":
                psis.append(r_h["_influence_raw"])
                wts.append(float(r_h["w_switchers"]))
                dens.append(float(r_h["w_switchers"]) * dose_vec[k])
            else:
                psis.append(r_h["_influence"])
                wts.append(1.0)
                dens.append(1.0)
        if psis:
            wv = np.asarray(wts, dtype=float)
            wv = wv / float(np.sum(dens))
            psi_head = np.sum([w * p for w, p in zip(wv, psis)], axis=0)
            se_avg = _clustered_if_se(psi_head, main["cluster_codes"], main["n_groups"])

    if se_avg and se_avg > 0:
        z = headline / se_avg
        p_h = float(2 * stats.norm.sf(abs(z)))
        ci_h = (headline - z_crit * se_avg, headline + z_crit * se_avg)
    else:
        p_h = np.nan
        ci_h = (np.nan, np.nan)

    design_table = None
    by_path_out = None
    if design is not None or by_path is not None:
        paths = _treatment_paths(
            df,
            group=group,
            time=time,
            treatment=treatment,
            n_effects=dynamic + 1,
            among=list(main["group_effects"][int(dynamic)].index),
        )
        counts = (
            paths.value_counts(sort=False)
            .rename("n_groups")
            .rename_axis("path")
            .reset_index()
        )
        counts = counts.sort_values(
            ["n_groups", "path"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        total_paths = int(counts["n_groups"].sum())
        counts["share"] = counts["n_groups"] / max(total_paths, 1)
        if design is not None:
            cum = counts["share"].cumsum().to_numpy()
            n_keep = int(np.searchsorted(cum, float(design) - 1e-12) + 1)
            design_table = counts.iloc[: min(n_keep, len(counts))].copy()
            design_table.attrs["n_groups"] = total_paths
            design_table.attrs["share_covered"] = float(design_table["share"].sum())
        if by_path is not None:
            if se_method != "analytic":
                raise MethodIncompatibility(
                    "by_path= reports analytic standard errors: pass "
                    "se_method='analytic'."
                )
            by_path_out = []
            effect_h = [h for h in horizons if h >= 0]
            for row in counts.head(int(by_path)).itertuples(index=False):
                on_path = set(paths.index[paths == row.path])
                path_fit = _estimate_all_horizons(
                    df=df,
                    y=y_work,
                    group=group,
                    time=time,
                    treatment=treatment,
                    horizons=effect_h,
                    control=control,
                    switchers=switchers,
                    same_switchers=True,
                    weights=weights,
                    cluster=cluster_var,
                    normalized=normalized,
                    match_baseline=True,
                    eligible=on_path,
                    controls=controls,
                    controls_info=controls_info,
                )
                by_path_out.append(
                    _path_result(path_fit, row.path, effect_h, alpha, aggregation)
                )

    norm_weights = None
    if normalized_weights:
        lag_cols: Dict[int, pd.Series] = {}
        for h in horizons:
            r_h = rows_by_h.get(h)
            if h < 0 or r_h is None:
                continue
            tot = float(np.sum(r_h["_lag_dose"]))
            lag_cols[h + 1] = pd.Series(
                r_h["_lag_dose"] / tot if tot > 0 else np.nan,
                index=pd.RangeIndex(len(r_h["_lag_dose"]), name="lag"),
            )
        norm_weights = pd.DataFrame(lag_cols)
        norm_weights.columns.name = "effect"

    return CausalResult(
        method=(
            "did_multiplegt_dyn (dCDH 2024 ReStat) "
            "[experimental MVP; pinned to DIDmultiplegtDYN / Stata "
            "did_multiplegt_dyn for "
            "effects, placebos, analytic SEs and joint tests, binary and "
            "discrete treatments; "
            "trends_lin, predict_het not implemented]"
        ),
        estimand=(
            "Average dynamic effect across horizons 0..dynamic "
            f"({aggregation}-weighted)"
        ),
        estimate=headline,
        se=se_avg,
        pvalue=p_h,
        ci=ci_h,
        alpha=alpha,
        n_obs=int(len(df)),
        detail=pd.DataFrame(main["cell_estimates"]),
        model_info={
            "event_study": es_df,
            "event_study_vcov": es_vcov,
            "horizons": horizons,
            "control": control,
            "aggregation": aggregation,
            "se_method": se_method,
            "n_boot": n_boot,
            "cluster_var": cluster_var,
            "weights": weights,
            "n_groups": int(main["n_groups"]),
            "joint_placebo_test": joint_placebo,
            "joint_effects_test": joint_effects,
            "joint_overall_test": joint_overall,
            "effects_equal_test": equal_test,
            "effects_equal_range": equal_range,
            "switchers": switchers,
            # One estimated effect per switching group per horizon. These
            # average (weighted by the event they belong to) to the reported
            # delta_l, and are what a heterogeneity regression would use as
            # its dependent variable -- see the note on predict_het in the
            # module docstring.
            "group_effects": main.get("group_effects", {}),
            "same_switchers": same_switchers,
            "time": time_label,
            # (g, t) cells removed by Design Restriction 2: the group had by
            # then been both above and below its period-one treatment.
            "n_dropped_bidirectional": n_bidirectional,
            "design": design_table,
            "by_path": by_path_out,
            "normalized_weights": norm_weights,
            "warning": (
                "controls=, trends_nonparam=, normalized= and continuous= "
                "are available and pinned to DIDmultiplegtDYN; trends_lin, "
                "predict_het and the heteroskedastic-weights variant are "
                "not. See docs/rfc/multiplegt_dyn.md."
            ),
        },
        _citation_key="dechaisemartin2024difference",
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _first_switch(
    df: pd.DataFrame,
    *,
    group: str,
    time: str,
    treatment: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Each unit's first treatment change: when, which way, and from what.

    Returns three frames keyed on ``group``: ``_F`` (the period of the first
    change), ``_dir`` (+1 for a switch on, -1 for a switch off) and ``_base``
    (the treatment level held just before it). Units that never change get
    no row and are therefore eligible as controls.

    The baseline matters as much as the direction. A unit turning treatment
    OFF has to be compared with units that were also ON and stayed ON; using
    the never-treated as its control group compares two different
    counterfactuals and does not give the reference's answer.
    """
    ordered = df.sort_values([group, time])
    labels = ordered[group]
    n = len(ordered)
    codes = pd.factorize(labels)[0] if n else np.zeros(0, dtype=np.int64)
    new_unit = np.r_[True, codes[1:] != codes[:-1]] if n else np.zeros(0, dtype=bool)
    if (
        isinstance(labels.dtype, pd.CategoricalDtype)
        or bool(labels.isna().any())
        or int(new_unit.sum()) != len(np.unique(codes))
    ):
        return _first_switch_by_unit(df, group=group, time=time, treatment=treatment)
    vals = ordered[treatment].to_numpy()
    times = ordered[time].to_numpy()
    changed = np.zeros(n, dtype=bool)
    if n > 1:
        changed[1:] = (np.diff(vals) != 0) & ~new_unit[1:]
    rows = np.flatnonzero(changed)
    # factorize numbers the units in the order the sorted frame meets them,
    # which is the order the per-unit loop visits them in
    k = rows[np.unique(codes[rows], return_index=True)[1]]
    idx = pd.Index(list(labels.iloc[k]), name=group)
    return (
        pd.Series(list(times[k]), index=idx, name="_F").reset_index(),
        pd.Series(
            np.where(vals[k] > vals[k - 1], 1, -1).tolist(), index=idx, name="_dir"
        ).reset_index(),
        pd.Series(
            vals[k - 1].astype(float).tolist(), index=idx, name="_base"
        ).reset_index(),
    )


def _first_switch_by_unit(
    df: pd.DataFrame,
    *,
    group: str,
    time: str,
    treatment: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """:func:`_first_switch`, one unit at a time (its definition)."""
    F: Dict[Any, Any] = {}
    direction: Dict[Any, int] = {}
    base: Dict[Any, float] = {}
    for uid, u_df in df.sort_values([group, time]).groupby(group, sort=False):
        vals = u_df[treatment].to_numpy()
        times = u_df[time].to_numpy()
        if len(vals) < 2:
            continue
        changed = np.nonzero(np.diff(vals) != 0)[0]
        if changed.size == 0:
            continue
        k = int(changed[0]) + 1
        F[uid] = times[k]
        direction[uid] = 1 if vals[k] > vals[k - 1] else -1
        base[uid] = float(vals[k - 1])
    idx = pd.Index(list(F), name=group)
    return (
        pd.Series(list(F.values()), index=idx, name="_F").reset_index(),
        pd.Series(list(direction.values()), index=idx, name="_dir").reset_index(),
        pd.Series(list(base.values()), index=idx, name="_base").reset_index(),
    )


def _treatment_paths(
    df: pd.DataFrame,
    *,
    group: str,
    time: str,
    treatment: str,
    n_effects: int,
    among: Any,
) -> pd.Series:
    """Treatments of the switchers in ``among`` from the period before the switch.

    A path is ``(D_{F-1}, D_F, ..., D_{F-1+n_effects})``. ``among`` are the
    switchers behind the last requested effect, so the paths are the ones
    that effect averages over. A group whose treatment is missing somewhere
    on its path is left out. Returned as a Series of tuples indexed by group.
    """
    f_of = df.groupby(group, sort=False)["_F"].first()
    wide = df.pivot_table(index=group, columns=time, values=treatment, aggfunc="first")
    out: Dict[Any, Tuple[float, ...]] = {}
    for uid in among:
        f_int = int(f_of.loc[uid])
        periods = list(range(f_int - 1, f_int + n_effects))
        if any(q not in wide.columns for q in periods):
            continue
        values = wide.loc[uid, periods].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)):
            continue
        out[uid] = tuple(float(v) for v in values)
    return pd.Series(out, dtype=object)


def _path_result(
    sub: Dict[str, Any],
    path: Tuple[float, ...],
    effect_h: List[int],
    alpha: float,
    aggregation: str,
) -> Dict[str, Any]:
    """Event study, average effect and joint test for one treatment path."""
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    rows_h = {r["horizon"]: r for r in sub["cell_estimates"]}
    es_rows: List[Dict[str, Any]] = []
    for h in effect_h:
        r_h = rows_h[h]
        est, se = r_h["delta_l"], r_h["_se_analytic"]
        ok = np.isfinite(est) and np.isfinite(se) and se > 0
        es_rows.append(
            {
                "relative_time": h,
                "att": est,
                "se": se,
                "pvalue": float(2 * stats.norm.sf(abs(est / se))) if ok else np.nan,
                "ci_lower": est - z_crit * se if ok else np.nan,
                "ci_upper": est + z_crit * se if ok else np.nan,
                "n_switchers": int(r_h["n_switchers"]),
            }
        )
    good = [h for h in effect_h if np.isfinite(rows_h[h]["delta_l"])]
    estimate, se_avg, joint = np.nan, np.nan, None
    if good:
        n_cl = int(sub["cluster_codes"].max()) + 1
        S = np.column_stack(
            [
                np.bincount(
                    sub["cluster_codes"],
                    weights=rows_h[h]["_influence"],
                    minlength=n_cl,
                )
                for h in good
            ]
        )
        V = (S.T @ S) / float(sub["n_groups"]) ** 2
        est_vec = np.array([rows_h[h]["delta_l"] for h in good], dtype=float)
        joint = _dc.joint_wald(est_vec, V, ridge=0.0)
        if aggregation == "switchers":
            w = np.array([rows_h[h]["w_switchers"] for h in good], dtype=float)
            dose = np.array(
                [
                    d if np.isfinite(d) else 1.0
                    for d in (rows_h[h]["dose_now"] for h in good)
                ]
            )
            raw = np.array([rows_h[h]["_delta_raw"] for h in good], dtype=float)
            den = float(np.sum(w * dose))
            if den > 0:
                estimate = float(np.sum(w * raw) / den)
                psi = np.sum(
                    [wi / den * rows_h[h]["_influence_raw"] for wi, h in zip(w, good)],
                    axis=0,
                )
            else:
                psi = np.full_like(rows_h[good[0]]["_influence_raw"], np.nan)
        else:
            estimate = float(np.mean(est_vec))
            psi = np.mean([rows_h[h]["_influence"] for h in good], axis=0)
        se_avg = _clustered_if_se(psi, sub["cluster_codes"], sub["n_groups"])
    return {
        "path": path,
        "n_switchers": int(max((r["n_switchers"] for r in es_rows), default=0)),
        "event_study": pd.DataFrame(es_rows),
        "estimate": estimate,
        "se": se_avg,
        "joint_effects_test": joint,
    }


def _clustered_if_se(
    psi: np.ndarray, cluster_codes: np.ndarray, n_groups: int
) -> float:
    """``sqrt(Σ_c (Σ_{g∈c} ψ_g)²) / G`` -- the reference's ``sqrt(var_sq_sum)/G``.

    ``psi`` is the group-level influence contribution scaled so that its
    mean is the estimate (``G / N`` × raw contribution). With one group per
    cluster this is the plain ``sqrt(mean(ψ²) / G)``.
    """
    sums = np.bincount(
        cluster_codes, weights=psi, minlength=int(cluster_codes.max()) + 1
    )
    return float(np.sqrt(np.sum(sums**2)) / n_groups)


def _residualise_on_baseline_polynomial(
    df: pd.DataFrame,
    *,
    y: str,
    group: str,
    time: str,
    degree: int,
) -> Tuple[pd.DataFrame, str]:
    """Take a polynomial in the period-one treatment out of the outcome path.

    The baseline estimator compares a switcher with groups that had the same
    period-one treatment. With a genuinely continuous period-one treatment no
    two groups share one -- Design Restriction 1(i) of dCDH (2024) fails --
    so the authors replace that comparison with a modelling assumption: the
    status-quo outcome *evolution* is a polynomial of the given degree in the
    period-one treatment. Residualising the outcome's first difference on that
    polynomial, with time fixed effects, on the (g, t)s that have not switched
    yet is what implements it, and the control set then no longer has to match
    on the baseline.
    """
    work = df.sort_values([group, time]).copy()
    dy = work.groupby(group)[y].diff().to_numpy(dtype=float)
    d1 = work["_base"].to_numpy(dtype=float)
    poly = np.column_stack([d1**k for k in range(1, degree + 1)])
    period_codes, periods = pd.factorize(work[time], sort=True)
    not_yet = (work["_F"].isna() | (work[time] < work["_F"])).to_numpy()
    usable = np.isfinite(dy) & np.all(np.isfinite(poly), axis=1)
    fit_rows = not_yet & usable
    if int(fit_rows.sum()) <= poly.shape[1] + len(periods):
        raise DataInsufficient(
            "continuous=: too few not-yet-switched observations to fit a "
            f"degree-{degree} polynomial in the period-one treatment with "
            "time fixed effects.",
            diagnostics={"degree": degree, "n_rows": int(fit_rows.sum())},
        )
    fe = np.eye(len(periods))[period_codes]
    # One polynomial per period: the status-quo *evolution* is allowed to
    # depend on the period-one treatment differently at each date, which a
    # pooled polynomial plus time effects would forbid.
    design = np.column_stack([fe] + [fe * poly[:, [j]] for j in range(poly.shape[1])])
    coef, *_ = np.linalg.lstsq(design[fit_rows], dy[fit_rows], rcond=None)
    resid = np.where(usable, dy - design @ coef, np.nan)
    work["_yres_cont"] = resid
    work["_yadj_cont"] = (
        work.groupby(group)["_yres_cont"]
        .apply(lambda col: col.fillna(0.0).cumsum())
        .reset_index(level=0, drop=True)
    )
    return work, "_yadj_cont"


def _residualise_on_controls(
    df: pd.DataFrame,
    *,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: List[str],
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, Dict[float, Dict[str, Any]]]:
    """Adjust the outcome for the controls, and keep what the variance needs.

    dCDH's ``controls`` option does not add covariates to a regression. It
    replaces the first difference of the outcome with the residual from a
    regression of that first difference on the first differences of the
    controls and period fixed effects (period by ``trends_nonparam`` cell
    when that option is on), weighted, fitted on the *control* (g, t)s --
    those whose treatment has not changed yet -- and separately for each
    value of the period-one treatment. Estimators built on those residuals
    are unbiased even under differential trends, as long as the differential
    trend is a linear function of the covariate changes.

    The adjusted outcome is written in levels, ``Y - X theta_d - sum of the
    period effects so far``, so that every long difference the estimator
    takes is the long difference of the residualised first differences,
    whatever happens to the group in between.

    The third value returned holds, per period-one treatment ``d``: the
    slopes ``theta``; ``T``, the last period with a control; and ``b``, one
    row per group, the reference's term in brackets -- the group's
    contribution to the estimation error of ``theta_d`` (zero outside
    baseline ``d``) minus ``theta_d``. The variance of an effect subtracts
    ``M_d' b_g`` from each group's influence, with ``M_d`` the effect's
    derivative with respect to ``theta_d`` (``U^{var,X}`` of the companion
    paper, as coded in the authors' command).
    """
    work = df.sort_values([group, time]).copy()
    n_x = len(controls)
    cols = [y] + list(controls)
    diffs = work.groupby(group)[cols].diff()
    # A first difference is a change between two CONSECUTIVE periods. On a
    # panel with holes the previous row of a group can be several periods
    # back, and that longer change must not enter the regression as if it
    # were a one-period one.
    consecutive = (work.groupby(group)[time].diff() == 1).to_numpy()
    dy = diffs[y].to_numpy(dtype=float)
    dx = diffs[list(controls)].to_numpy(dtype=float)
    levels_y = work[y].to_numpy(dtype=float)
    levels_x = work[list(controls)].to_numpy(dtype=float)
    d_now = work[treatment].to_numpy(dtype=float)
    n_gt = (
        np.ones(len(work))
        if weights is None
        else work[weights].astype(float).fillna(0.0).to_numpy()
    )
    n_gt = np.where(np.isfinite(levels_y) & np.isfinite(d_now), n_gt, 0.0)

    # Control (g, t): the treatment has not changed by t. `_F` is the first
    # switch period, NaN for never-switchers.
    f_of = work["_F"].to_numpy(dtype=float)
    t_now = work[time].to_numpy(dtype=float)
    not_yet = np.isnan(f_of) | (t_now < f_of)
    dy_ok = consecutive & np.isfinite(dy)
    usable = dy_ok & np.all(np.isfinite(dx), axis=1)

    period_codes, periods = pd.factorize(work[time], sort=True)
    n_periods = len(periods)
    if "_tcell" in work.columns:
        tc_codes = pd.factorize(work["_tcell"], sort=True)[0]
    else:
        tc_codes = np.zeros(len(work), dtype=np.int64)
    n_tc = int(tc_codes.max()) + 1 if len(work) else 1
    cell = tc_codes * n_periods + period_codes
    n_cells = n_tc * n_periods

    units = pd.Index(sorted(work[group].unique()))
    g_codes = units.get_indexer(work[group])
    cl_col = cluster if cluster is not None else group
    cl_codes = pd.factorize(work[cl_col])[0]
    t_max = float(work[time].max())
    f_or_end = np.where(np.isnan(f_of), t_max + 1.0, f_of)

    adjusted = np.full(len(work), np.nan)
    info: Dict[float, Dict[str, Any]] = {}
    # Period-one treatment of EVERY group. `_base` is only filled for the
    # groups that switch, and the never-switchers are the bulk of the
    # control (g, t)s the regression is fitted on.
    base_all = (
        work[treatment].astype(float).groupby(work[group]).transform("first")
    ).to_numpy()
    for base in sorted(pd.unique(base_all[np.isfinite(base_all)])):
        in_base = base_all == base
        if np.unique(f_or_end[in_base]).size < 2:
            # every group with this period-one treatment switches at the
            # same date, or none does: nothing is estimated there
            adjusted[in_base] = levels_y[in_base]
            continue
        fit = in_base & not_yet & usable & (n_gt > 0)
        if fit.sum() <= n_x + 1:
            raise DataInsufficient(
                "controls=: too few not-yet-switched observations at "
                f"baseline treatment {base!r} to fit the first-difference "
                "regression the option is defined by.",
                diagnostics={"baseline": float(base), "n_rows": int(fit.sum())},
            )
        w_fit = np.bincount(cell[fit], weights=n_gt[fit], minlength=n_cells)
        has = w_fit > 0
        safe = np.where(has, w_fit, 1.0)
        avg_dx = np.column_stack(
            [
                np.bincount(
                    cell[fit], weights=n_gt[fit] * dx[fit, k], minlength=n_cells
                )
                / safe
                for k in range(n_x)
            ]
        )
        avg_dy = np.bincount(cell[fit], weights=n_gt[fit] * dy[fit], minlength=n_cells)
        avg_dy = avg_dy / safe
        x_dot = dx - avg_dx[cell]
        xtx = (x_dot[fit] * n_gt[fit, None]).T @ x_dot[fit]
        xty = (x_dot[fit] * n_gt[fit, None]).T @ (dy[fit] - avg_dy[cell[fit]])
        xtx_inv = np.linalg.pinv(xtx)
        theta = xtx_inv @ xty
        lam = np.where(has, avg_dy - avg_dx @ theta, 0.0)
        cum_lam = np.cumsum(lam.reshape(n_tc, n_periods), axis=1).ravel()
        adjusted[in_base] = (
            levels_y[in_base] - levels_x[in_base] @ theta - cum_lam[cell[in_base]]
        )

        # --- what the variance needs
        ctrl_rows = in_base & not_yet & dy_ok
        denom = np.zeros(n_periods)
        if ctrl_rows.any():
            pairs = pd.DataFrame(
                {"t": period_codes[ctrl_rows], "c": cl_codes[ctrl_rows]}
            ).drop_duplicates()
            denom = np.bincount(pairs["t"], minlength=n_periods).astype(float)
        den_row = denom[period_codes]
        dof = np.where(den_row >= 2, np.sqrt(den_row / np.maximum(den_row - 1, 1)), 1.0)
        fitted = np.where(den_row >= 2, lam[cell] + dx @ theta, 0.0)
        fitted = np.where(has[cell], fitted, np.nan)
        rows = in_base & not_yet & usable & np.isfinite(fitted)
        n_c = float(n_gt[ctrl_rows].sum())
        in_sum = np.zeros((len(units), n_x))
        if n_c > 0:
            term = (
                (n_gt[rows] * dof[rows] * (dy[rows] - fitted[rows]))[:, None]
                * x_dot[rows]
                / n_c
            )
            np.add.at(in_sum, g_codes[rows], term)
        inv_denom = xtx_inv * float(n_gt[fit].sum()) * len(units)
        unit_base = pd.Series(base_all, index=work[group].to_numpy())
        unit_base = unit_base[~unit_base.index.duplicated()].reindex(units)
        unit_f = pd.Series(f_or_end, index=work[group].to_numpy())
        unit_f = unit_f[~unit_f.index.duplicated()].reindex(units)
        member = ((unit_base == base) & (unit_f >= 3)).to_numpy(dtype=float)
        b = member[:, None] * (in_sum @ inv_denom.T) - theta[None, :]
        info[float(base)] = {
            "theta": theta,
            "b": pd.DataFrame(b, index=units),
            "T": float(f_or_end[in_base].max() - 1.0),
        }

    work["_yadj"] = adjusted
    return work, "_yadj", info


def _switch_directions(switchers: Optional[str]) -> Tuple[int, ...]:
    if switchers == "in":
        return (1,)
    if switchers == "out":
        return (-1,)
    return (1, -1)


def _estimate_all_horizons(
    *,
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    horizons: List[int],
    control: str,
    switchers: Optional[str] = None,
    same_switchers: bool = False,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    normalized: bool = False,
    match_baseline: bool = True,
    eligible: Optional[set] = None,
    controls: Optional[List[str]] = None,
    controls_info: Optional[Dict[float, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """δ_l, its influence function and its pieces at every horizon.

    Evaluates the events of :func:`_estimate_all_horizons_frame` on unit x
    period matrices (``_dcdh_arrays``); frames those matrices cannot
    represent go to the frame code.
    """
    panel = _arr.build_panel(
        df,
        y=y,
        group=group,
        time=time,
        treatment=treatment,
        weights=weights,
        cluster=cluster,
        controls=controls if controls_info else None,
    )
    if panel is None:
        return _estimate_all_horizons_frame(
            df=df,
            y=y,
            group=group,
            time=time,
            treatment=treatment,
            horizons=horizons,
            control=control,
            switchers=switchers,
            same_switchers=same_switchers,
            weights=weights,
            cluster=cluster,
            normalized=normalized,
            match_baseline=match_baseline,
            eligible=eligible,
            controls=controls,
            controls_info=controls_info,
        )
    assert panel.units is not None and panel.labels is not None
    assert panel.cluster_codes is not None
    all_units = panel.units
    labels = panel.labels
    n_panel = panel.n
    cluster_codes = panel.cluster_codes
    directions = _switch_directions(switchers)

    if same_switchers and len(horizons) > 1:
        panel = _arr.common_switchers(
            panel,
            horizons=horizons,
            control=control,
            directions=directions,
            match_baseline=match_baseline,
        )
    if eligible is not None:
        keep = panel.never | labels.isin(list(eligible))
        panel = _arr.replace(
            panel, elig=keep if panel.elig is None else panel.elig & keep
        )

    grid = _arr._event_grid(panel, directions, match_baseline)
    if grid is None:
        return {
            "cell_estimates": [],
            "cluster_codes": cluster_codes,
            "n_groups": n_panel,
        }

    cells: List[Dict[str, Any]] = []
    group_effects: Dict[int, pd.Series] = {}
    for h in horizons:
        per_group: List[pd.Series] = []
        sum_wdelta = 0.0
        sum_wdose = 0.0
        sum_wdose_now = 0.0
        lag_dose = np.zeros(max(h, 0) + 1)
        slope_grad: Dict[float, np.ndarray] = {}
        w_total = 0.0
        n_sw = 0
        n_events = 0
        psi: np.ndarray = np.zeros(n_panel, dtype=float)

        for F, direction, base, tcode in grid:
            _cell = _arr.one_event(
                panel,
                F=F,
                h=h,
                direction=direction,
                base=base,
                tcode=tcode,
                control=control,
                match_baseline=match_baseline,
                full=True,
                need_dose=True,
            )
            if _cell is None:
                continue
            sum_wdelta += _cell["delta"] * _cell["w_sw"]
            sum_wdose += _cell["dose"] * _cell["w_sw"]
            lag_dose += _cell["lag_dose"]
            if controls_info and base is not None:
                d_info = controls_info.get(float(base))
                ell = h + 1 if h >= 0 else -h
                if d_info is not None and ell <= d_info["T"] - 2:
                    slope_grad[float(base)] = (
                        slope_grad.get(float(base), 0.0) + _cell["m_x"]
                    )
            if np.isfinite(_cell["dose_now"]):
                sum_wdose_now += _cell["dose_now"] * _cell["w_sw"]
            w_total += _cell["w_sw"]
            n_sw += _cell["n_sw"]
            n_events += 1
            # an event touches each unit once, as a switcher or as a control
            psi[_cell["sw"]] += _cell["psi_sw"]
            psi[_cell["c"]] += _cell["psi_c"]
            per_group.append(
                pd.Series(_cell["effects"], index=labels.take(_cell["sw"]))
            )

        delta_raw = np.nan
        psi_raw = psi
        if w_total > 0:
            delta_l = sum_wdelta / w_total
            delta_raw = delta_l
            psi = np.asarray(psi * (n_panel / w_total), dtype=float)
            if controls_info:
                for d_key, grad in slope_grad.items():
                    b = controls_info[d_key]["b"].reindex(all_units).to_numpy()
                    psi = psi - b @ (np.asarray(grad, dtype=float) / w_total)
            psi_raw = psi
            if normalized:
                dose = sum_wdose / w_total
                if np.isfinite(dose) and dose != 0:
                    delta_l = delta_l / dose
                    psi = psi / dose
                else:
                    delta_l = np.nan
                    psi = np.full(n_panel, np.nan)
            se_analytic = _clustered_if_se(psi, cluster_codes, n_panel)
        else:
            delta_l = np.nan
            se_analytic = np.nan

        group_effects[int(h)] = (
            pd.concat(per_group) if per_group else pd.Series(dtype=float)
        )
        cells.append(
            {
                "horizon": h,
                "delta_l": float(delta_l) if np.isfinite(delta_l) else np.nan,
                "n_switchers": n_sw,
                "w_switchers": float(w_total),
                "n_events": n_events,
                "_influence": psi,
                "_se_analytic": se_analytic,
                "_delta_raw": float(delta_raw) if np.isfinite(delta_raw) else np.nan,
                "_influence_raw": psi_raw,
                "dose_now": (sum_wdose_now / w_total) if w_total > 0 else np.nan,
                "_lag_dose": lag_dose,
            }
        )

    return {
        "cell_estimates": cells,
        "cluster_codes": cluster_codes,
        "n_groups": n_panel,
        "group_effects": group_effects,
    }


def _estimate_all_horizons_frame(
    *,
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    horizons: List[int],
    control: str,
    switchers: Optional[str] = None,
    same_switchers: bool = False,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    normalized: bool = False,
    match_baseline: bool = True,
    eligible: Optional[set] = None,
    controls: Optional[List[str]] = None,
    controls_info: Optional[Dict[float, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compute δ_l for each horizon h using long-difference event-study.

    The definition on the long frame; :func:`_estimate_all_horizons` runs the
    same arithmetic on matrices whenever the frame allows it.

    For each unique first-treatment period F in the sample:
      switchers at F = units with _F == F.
      For each horizon l:
        - If l >= 0: compare Y_{F+l} − Y_{F-1} between switchers and controls
          not yet switched at F+l.
        - If l < 0: compare Y_{F-1-|l|} − Y_{F-1} (placebo) on the SAME
          switchers and controls as effect |l|-1, i.e. the anchor period is
          F-1+|l|. Matches DIDmultiplegtDYN's Placebo_|l|.
      Control set depends on `control=`.

    Aggregate per horizon with (weighted) n_switchers weights, and carry
    the reference's influence-function variance (``U_Gg_var``) per group.
    """
    cells: List[Dict[str, Any]] = []

    # Unit-level index for the influence functions. Each horizon's estimate
    # is a weighted sum of two-sample mean differences, so its influence
    # function is the corresponding sum of within-cell deviations -- and
    # summing rather than adding variances is what carries the fact that a
    # control unit can serve several events.
    all_units = pd.Index(sorted(df[group].unique()))
    n_panel = len(all_units)
    unit_pos = pd.Series(np.arange(n_panel), index=all_units)
    cl_col = cluster if cluster is not None else group
    cluster_of = df.groupby(group)[cl_col].first().reindex(all_units)
    cluster_codes = pd.factorize(cluster_of.to_numpy())[0]

    # switchers=: estimate switch-in and switch-out events separately.
    # dCDH recommend running the command twice rather than pooling, because
    # the two need not measure the same effect per unit of treatment.
    directions = _switch_directions(switchers)

    # same_switchers=: hold the switcher composition fixed across horizons.
    # Without it, a longer horizon is estimated on a shrinking, differently
    # composed set of switchers, so movement across ℓ mixes a genuine
    # dynamic path with a change of who is being averaged over.
    if same_switchers and len(horizons) > 1:
        df = _restrict_to_common_switchers(
            df, group=group, time=time, horizons=horizons
        )
        # Being observed at every horizon is not enough: the effect must be
        # *estimable* there, which also takes a control that has not
        # switched yet. Late switchers run out of controls at long horizons
        # (and, with a discrete treatment, so do rare baselines), so the
        # common set is read off a first pass: the switchers that contribute
        # to every requested effect. Controls are picked by switch date and
        # do not depend on who is eligible, so one pass settles it.
        effect_h = [h for h in horizons if h >= 0]
        if len(effect_h) > 1:
            first = _estimate_all_horizons_frame(
                df=df,
                y=y,
                group=group,
                time=time,
                treatment=treatment,
                horizons=effect_h,
                control=control,
                switchers=switchers,
                same_switchers=False,
                weights=weights,
                cluster=cluster,
                normalized=False,
                match_baseline=match_baseline,
            )
            contributing = [set(first["group_effects"][int(h)].index) for h in effect_h]
            common = set.intersection(*contributing) if contributing else set()
            df = df.copy()
            df["_elig"] = df["_elig"] & (df["_F"].isna() | df[group].isin(common))

    if eligible is not None:
        # by_path: only these switchers contribute an effect; everyone else
        # stays in the frame as a control until its own switch.
        df = df.copy()
        keep = df["_F"].isna() | df[group].isin(eligible)
        df["_elig"] = (df["_elig"] & keep) if "_elig" in df.columns else keep

    F_values = sorted(df["_F"].dropna().unique())
    # One event per (switch period, direction, period-one treatment): with a
    # binary treatment the direction fixes the baseline, with a discrete one
    # switchers from different baselines have different controls.
    _ev = df.loc[df["_F"].notna(), ["_F", "_dir", "_base"]].drop_duplicates()
    bases_of: Dict[Tuple[Any, int], List[float]] = {}
    for _f, _d, _b in _ev.itertuples(index=False):
        bases_of.setdefault((_f, int(_d)), []).append(float(_b))
    if not F_values:
        return {
            "cell_estimates": [],
            "cluster_codes": cluster_codes,
            "n_groups": n_panel,
        }

    # trends_nonparam: one event per value of the varlist, so that a
    # switcher is only ever compared with controls that share it.
    if "_tcell" in df.columns:
        trends_cells: Tuple[Any, ...] = tuple(sorted(df["_tcell"].dropna().unique()))
    else:
        trends_cells = (None,)

    # Never-treated set (units with _F NaN)
    never_ids = set(df[df["_F"].isna()][group].unique())
    # Earliest period actually observed -- a horizon whose base period
    # falls before it has no data and is skipped, exactly as the R
    # package drops the cohorts that cannot support a given placebo.
    t_min = float(df[time].min())

    group_effects: Dict[int, pd.Series] = {}
    for h in horizons:
        per_group: List[pd.Series] = []
        sum_wdelta = 0.0
        sum_wdose = 0.0
        sum_wdose_now = 0.0
        lag_dose = np.zeros(max(h, 0) + 1)
        # derivative of the effect with respect to the covariate slopes of
        # each period-one treatment (the reference's M_d), unscaled
        slope_grad: Dict[float, np.ndarray] = {}
        w_total = 0.0
        n_sw = 0
        n_events = 0
        psi: np.ndarray = np.zeros(n_panel, dtype=float)

        for F in F_values:
            for direction in directions:
                _bases: List[Optional[float]] = [None]
                if match_baseline:
                    _bases = list(sorted(bases_of.get((F, direction), [])))
                for base, trends_cell in ((b, c) for b in _bases for c in trends_cells):
                    _cell = _one_event(
                        df=df,
                        y=y,
                        group=group,
                        time=time,
                        treatment=treatment,
                        F=F,
                        h=h,
                        direction=direction,
                        control=control,
                        never_ids=never_ids,
                        t_min=t_min,
                        n_panel=n_panel,
                        unit_pos=unit_pos,
                        weights=weights,
                        cluster_of=cluster_of,
                        trends_cell=trends_cell,
                        match_baseline=match_baseline,
                        base=base,
                        controls=controls if controls_info else None,
                    )
                    if _cell is None:
                        continue
                    sum_wdelta += _cell["delta"] * _cell["w_sw"]
                    sum_wdose += _cell["dose"] * _cell["w_sw"]
                    lag_dose += _cell["lag_dose"]
                    if controls_info and base is not None:
                        d_info = controls_info.get(float(base))
                        ell = h + 1 if h >= 0 else -h
                        if d_info is not None and ell <= d_info["T"] - 2:
                            slope_grad[float(base)] = (
                                slope_grad.get(float(base), 0.0) + _cell["m_x"]
                            )
                    if np.isfinite(_cell["dose_now"]):
                        sum_wdose_now += _cell["dose_now"] * _cell["w_sw"]
                    w_total += _cell["w_sw"]
                    n_sw += _cell["n_sw"]
                    n_events += 1
                    psi += _cell["psi"]
                    per_group.append(_cell["group_effects"])

        delta_raw = np.nan
        psi_raw = psi
        if w_total > 0:
            delta_l = sum_wdelta / w_total
            delta_raw = delta_l
            # Reference scaling: U_Gg = (G / N_l) × Σ_t contribution_gt.
            psi = np.asarray(psi * (n_panel / w_total), dtype=float)
            if controls_info:
                # the covariate slopes are estimated: U^{var,X} of the
                # companion paper
                for d_key, grad in slope_grad.items():
                    b = controls_info[d_key]["b"].reindex(all_units).to_numpy()
                    psi = psi - b @ (np.asarray(grad, dtype=float) / w_total)
            psi_raw = psi
            if normalized:
                # Effect per unit of treatment: divide by the average
                # cumulative treatment the switchers received. The influence
                # function is scaled by the same constant, which treats the
                # denominator as fixed -- the reference does the same.
                dose = sum_wdose / w_total
                if np.isfinite(dose) and dose != 0:
                    delta_l = delta_l / dose
                    psi = psi / dose
                else:
                    delta_l = np.nan
                    psi = np.full(n_panel, np.nan)
            se_analytic = _clustered_if_se(psi, cluster_codes, n_panel)
        else:
            delta_l = np.nan
            se_analytic = np.nan

        group_effects[int(h)] = (
            pd.concat(per_group) if per_group else pd.Series(dtype=float)
        )
        cells.append(
            {
                "horizon": h,
                "delta_l": float(delta_l) if np.isfinite(delta_l) else np.nan,
                "n_switchers": n_sw,
                "w_switchers": float(w_total),
                "n_events": n_events,
                "_influence": psi,
                "_se_analytic": se_analytic,
                # Non-normalized effect and influence function, and the
                # switchers' average treatment change at the horizon: the
                # pieces of the average total effect per unit of treatment.
                "_delta_raw": float(delta_raw) if np.isfinite(delta_raw) else np.nan,
                "_influence_raw": psi_raw,
                "dose_now": (sum_wdose_now / w_total) if w_total > 0 else np.nan,
                "_lag_dose": lag_dose,
            }
        )

    return {
        "cell_estimates": cells,
        "cluster_codes": cluster_codes,
        "n_groups": n_panel,
        "group_effects": group_effects,
    }


def _cell_centre(
    n_own: int, mean_own: float, n_pool: int, mean_pool: float
) -> Tuple[float, float]:
    """(Ê, DOF) for one cell of the reference variance.

    ``compute_E_hat_gt`` / ``compute_DOF_gt`` in ``DIDmultiplegtDYN``: a cell
    with at least two clusters is centred on its own mean and scaled by
    ``sqrt(n/(n-1))``; a single-cluster cell borrows the pooled
    switcher+control cell at the same period; a cell that cannot even do
    that is left uncentred with factor 1.
    """
    if n_own >= 2:
        return mean_own, float(np.sqrt(n_own / (n_own - 1.0)))
    if n_pool >= 2:
        return mean_pool, float(np.sqrt(n_pool / (n_pool - 1.0)))
    return 0.0, 1.0


def _one_event(
    *,
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    F: Any,
    h: int,
    direction: int,
    control: str,
    never_ids: set,
    t_min: float,
    n_panel: int,
    unit_pos: pd.Series,
    weights: Optional[str],
    cluster_of: pd.Series,
    trends_cell: Any = None,
    match_baseline: bool = True,
    base: Optional[float] = None,
    controls: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """One (switch period, direction, baseline) event at horizon ``h``.

    Three things distinguish a switch-off event from a switch-on one, and
    all three matter:

    * Its controls must share its BASELINE treatment level, not merely be
      untreated. A unit going 1 -> 0 belongs against units that were at 1
      and stayed there; comparing it with the never-treated is a different
      counterfactual and gives a different number.
    * The difference is divided by the change in treatment, so a switch-off
      contributes with the opposite sign and both directions measure the
      same "effect per unit of treatment".
    * It is otherwise an ordinary two-sample comparison, so the influence
      function has the same shape.

    The *anchor* period ``F − 1 + i`` (``i = h + 1`` for an effect,
    ``i = |h|`` for a placebo) decides the sample: a unit must be observed
    there, controls must not have switched by then, and ``weights`` is read
    there. For a placebo the outcome contrast is nonetheless the mirrored
    ``Y_{F−1−i} − Y_{F−1}``.
    """
    sw_mask = (df["_F"] == F) & (df["_dir"] == direction)
    if base is not None:
        sw_mask &= df["_base"] == base
    if trends_cell is not None:
        # trends_nonparam: switchers are compared only with controls that
        # share the value of the varlist, so each cell is its own event.
        sw_mask &= df["_tcell"] == trends_cell
    if "_elig" in df.columns:
        # same_switchers: this unit switches at F but cannot support every
        # requested horizon, so it must not contribute an effect. It stays
        # in the frame as a control candidate.
        sw_mask &= df["_elig"]
    switchers = df[sw_mask]
    if switchers.empty:
        return None
    switcher_ids = set(switchers[group].unique())
    base_level = float(switchers["_base"].iloc[0])

    if h >= 0:
        t_pre, t_post = F - 1, F + h
        t_anchor = t_post
    else:
        i = -h
        t_pre, t_post = F - 1 - i, F - 1
        t_anchor = F - 1 + i
    if t_pre < t_min:
        return None

    # Controls: never-switchers plus units that have not switched by the
    # anchor period, restricted to the switcher's baseline level.
    if control == "never_treated":
        candidates = never_ids
    else:
        candidates = set(df[(df["_F"] > t_anchor) | (df["_F"].isna())][group].unique())
    if match_baseline:
        pre_rows = df[df[time] == t_pre]
        same_base = set(pre_rows[pre_rows[treatment] == base_level][group].unique())
        ctrl_ids = (candidates & same_base) - switcher_ids
    else:
        # continuous=: no two groups share a period-one treatment, so the
        # baseline match is impossible and is replaced by the polynomial
        # already residualised out of the outcome.
        ctrl_ids = candidates - switcher_ids
    if trends_cell is not None:
        ctrl_ids &= set(df[df["_tcell"] == trends_cell][group].unique())
    if not ctrl_ids:
        return None

    sw = _event_sample(
        df, group, time, y, weights, switcher_ids, t_pre, t_post, t_anchor
    )
    ct = _event_sample(df, group, time, y, weights, ctrl_ids, t_pre, t_post, t_anchor)
    if sw is None or ct is None:
        return None
    sw_ids, sw_dy, sw_w = sw
    c_ids, c_dy, c_w = ct

    w_s = float(sw_w.sum())
    w_c = float(c_w.sum())
    mean_s = float(np.sum(sw_w * sw_dy) / w_s)
    mean_c = float(np.sum(c_w * c_dy) / w_c)
    scale = (-1.0 if h < 0 else 1.0) / direction
    delta = scale * (mean_s - mean_c)

    # Reference variance cells: the cohort for switchers, the (baseline, t)
    # control set for controls, and the pooled cell as the fallback when a
    # cell holds a single cluster. Cluster counts, weighted means.
    n_c = int(cluster_of.loc[c_ids].nunique())
    n_pool = int(cluster_of.loc[sw_ids.append(c_ids)].nunique())
    mean_pool = float((np.sum(sw_w * sw_dy) + np.sum(c_w * c_dy)) / (w_s + w_c))
    e_c, dof_c = _cell_centre(n_c, mean_c, n_pool, mean_pool)

    # Switchers are centred within their cohort: same baseline, same switch
    # period and same treatment AT the switch period (the reference's
    # ``d_sq, F_g, d_fg``). With a binary treatment that is the whole event.
    d_at_f = (
        df[(df[time] == F) & df[group].isin(set(sw_ids))]
        .set_index(group)[treatment]
        .reindex(sw_ids)
        .to_numpy(dtype=float)
    )
    sw_clusters = cluster_of.loc[sw_ids].to_numpy()
    e_s = np.empty(len(sw_ids), dtype=float)
    dof_s = np.empty(len(sw_ids), dtype=float)
    for level in pd.unique(d_at_f):
        m = d_at_f == level if level == level else np.isnan(d_at_f)
        n_s = int(pd.Series(sw_clusters[m]).nunique())
        mean_cell = float(np.sum(sw_w[m] * sw_dy[m]) / np.sum(sw_w[m]))
        e_s[m], dof_s[m] = _cell_centre(n_s, mean_cell, n_pool, mean_pool)

    psi = np.zeros(n_panel, dtype=float)
    psi[unit_pos.reindex(sw_ids).to_numpy()] += sw_w * dof_s * (sw_dy - e_s)
    psi[unit_pos.reindex(c_ids).to_numpy()] -= (w_s / w_c) * c_w * dof_c * (c_dy - e_c)
    psi *= scale

    m_x = np.zeros(len(controls) if controls else 0)
    if controls:
        # the same weighted difference in differences, taken on each
        # control's long difference instead of the outcome's
        for k, col in enumerate(controls):
            x_pre = _unit_values(df, group, time, col, sw_ids.append(c_ids), t_pre)
            x_post = _unit_values(df, group, time, col, sw_ids.append(c_ids), t_post)
            x_diff = (x_post - x_pre).reindex(sw_ids.append(c_ids))
            xs = x_diff.loc[sw_ids].to_numpy(dtype=float)
            xc = x_diff.loc[c_ids].to_numpy(dtype=float)
            m_x[k] = scale * (
                float(np.sum(sw_w * xs)) - (w_s / w_c) * float(np.sum(c_w * xc))
            )

    dose = _event_dose(
        df,
        group=group,
        time=time,
        treatment=treatment,
        ids=sw_ids,
        weights=sw_w,
        F=F,
        h=h,
        base_level=base_level,
    )
    # Weighted treatment change in place at each of the periods F .. F + h,
    # indexed by lag (0 = the horizon itself): what normalized_weights reports.
    lag_dose = np.zeros(max(h, 0) + 1)
    if h >= 0:
        sub = df[df[group].isin(set(sw_ids)) & (df[time] >= F) & (df[time] <= t_post)]
        gap_all = sub[treatment].to_numpy(dtype=float) - base_level
        w_of = pd.Series(sw_w, index=sw_ids).reindex(sub[group]).to_numpy()
        lags = (t_post - sub[time].to_numpy()).astype(int)
        ok_gap = np.isfinite(gap_all)
        np.add.at(lag_dose, lags[ok_gap], (w_of * np.abs(gap_all))[ok_gap])
    # Treatment change in place AT the horizon (not cumulated): the divisor
    # of the average total effect per unit of treatment.
    if h >= 0:
        d_now = (
            df[(df[time] == t_post) & df[group].isin(set(sw_ids))]
            .set_index(group)[treatment]
            .reindex(sw_ids)
            .to_numpy(dtype=float)
        )
        gap = np.abs(d_now - base_level)
        ok = np.isfinite(gap)
        dose_now = (
            float(np.sum(sw_w[ok] * gap[ok]) / np.sum(sw_w[ok]))
            if ok.any()
            else float("nan")
        )
    else:
        dose_now = float("nan")
    return {
        "delta": float(delta),
        "n_sw": int(len(sw_ids)),
        "w_sw": w_s,
        "psi": psi,
        "dose": dose,
        "dose_now": dose_now,
        "lag_dose": lag_dose,
        "m_x": m_x,
        # Each switcher's own effect: the contrast behind delta with this
        # group's outcome change in place of the switcher mean. predict_het
        # regresses these on group-level covariates.
        "group_effects": pd.Series(scale * (sw_dy - mean_c), index=sw_ids),
    }


def _event_dose(
    df: pd.DataFrame,
    *,
    group: str,
    time: str,
    treatment: str,
    ids: pd.Index,
    weights: np.ndarray,
    F: Any,
    h: int,
    base_level: float,
) -> float:
    """Average cumulative treatment change behind one event's switchers.

    ``normalized`` divides the effect by how much treatment the switchers
    actually received between the base period and the horizon, so that the
    reported number is an effect *per unit of treatment* rather than the
    effect of a path whose length grows with the horizon. For a binary
    absorbing switch this is exactly ``h + 1`` (or ``|h|`` for a placebo),
    which is what makes the normalised series flat when the per-period
    effect is constant.
    """
    span = range(0, h + 1) if h >= 0 else range(0, -h)
    periods = [F + s for s in span]
    sub = df[df[group].isin(set(ids)) & df[time].isin(periods)]
    if sub.empty:
        return float("nan")
    per_unit = (
        sub.assign(_inc=(sub[treatment] - base_level).abs())
        .groupby(group)["_inc"]
        .sum()
        .reindex(ids)
        .to_numpy(dtype=float)
    )
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0 or not np.all(np.isfinite(per_unit)):
        return float("nan")
    return float(np.sum(w * per_unit) / total)


def _unit_values(
    df: pd.DataFrame,
    group: str,
    time: str,
    y: str,
    ids: Any,
    t: Any,
) -> pd.Series:
    """``y`` at period ``t`` for the given units, indexed by unit, NaN dropped."""
    sub = df[(df[time] == t) & df[group].isin(ids)]
    s = sub.set_index(group)[y].astype(float)
    return s[s.notna()]


def _event_sample(
    df, group, time, y, weights, ids, t_pre, t_post, t_anchor
) -> Optional[Tuple[pd.Index, np.ndarray, np.ndarray]]:
    """Units observed at every period the event needs, with their change and weights.

    Only units observed at BOTH ends of the contrast (and at the anchor
    period, which for a placebo is a third period) contribute -- a unit
    missing any of them has no change and cannot enter the difference. The
    weights is the reference's ``N_gt`` at the anchor row; zero or missing
    weights drop the unit.
    """
    pre = _unit_values(df, group, time, y, ids, t_pre)
    post = _unit_values(df, group, time, y, ids, t_post)
    idx = pre.index.intersection(post.index)
    if t_anchor != t_post:
        anchor = _unit_values(df, group, time, y, ids, t_anchor)
        idx = idx.intersection(anchor.index)
    if len(idx) == 0:
        return None
    if weights is None:
        w = np.ones(len(idx), dtype=float)
    else:
        sub = df[(df[time] == t_anchor) & df[group].isin(idx)].set_index(group)[weights]
        w = sub.reindex(idx).astype(float).fillna(0.0).to_numpy()
        keep = w != 0
        if not keep.any():
            return None
        idx = idx[keep]
        w = w[keep]
    dy = (post.loc[idx] - pre.loc[idx]).to_numpy(dtype=float)
    return idx, dy, w


def _restrict_to_common_switchers(
    df: pd.DataFrame,
    *,
    group: str,
    time: str,
    horizons: List[int],
) -> pd.DataFrame:
    """Drop switchers that cannot support every requested horizon.

    Stata ``did_multiplegt_dyn, same_switchers``. A switcher at F
    contributes to horizon ℓ only if it is observed at both the base
    period F−1 and the comparison period F+ℓ. Panels being what they are,
    late switchers drop out of long horizons and early ones drop out of
    deep placebos — so the ℓ-profile silently mixes the dynamic path with
    a moving composition. Restricting to switchers observed at *every*
    requested period removes that confound, at the cost of sample size.

    Availability is checked per unit against the periods that unit is
    actually observed in, not against the global panel range, so the
    restriction is correct on unbalanced panels.

    Control units (``_F`` NaN) are never dropped: the restriction is about
    the composition of the treated arm.
    """
    # Judged on the EFFECTS (h >= 0) plus the base period F-1, not on the
    # placebos: Stata scopes same_switchers to the effects and has a
    # separate same_switchers_pl for extending it to the placebos.
    need = sorted({-1, *(h for h in horizons if h >= 0)})
    obs = df.groupby(group)[time].apply(set)
    f_of = df.groupby(group)["_F"].first()

    eligible = {}
    for uid, f_val in f_of.items():
        if pd.isna(f_val):
            eligible[uid] = True  # never-switchers are controls, not switchers
            continue
        periods = obs.loc[uid]
        eligible[uid] = all((f_val + h) in periods for h in need)

    # Flag rather than filter. A restricted switcher must stay in the frame
    # with its `_F` intact, because control eligibility is decided from
    # `_F` (`not yet treated at F+l`): dropping the rows would promote a
    # unit that really does switch at F=8 out of the data entirely, and
    # silently shrink the control pool available to the F=4 and F=6
    # events. Blanking `_F` would be worse still — it would promote that
    # unit to a *never*-switcher and let it serve as a control at horizons
    # where it is already treated.
    out = df.copy()
    out["_elig"] = out[group].map(eligible).fillna(True).astype(bool)
    return out


def _effects_equal_test(
    main: Dict[str, Any],
    horizons: List[int],
    boot_hist: np.ndarray,
    indices: List[int],
) -> Optional[Dict[str, Any]]:
    """Wald test of H0: every effect in ``indices`` is equal.

    Stata ``did_multiplegt_dyn, effects_equal``. Differencing adjacent
    effects turns "all equal" into "all contrasts zero", so the existing
    :func:`joint_wald` applies once the contrast matrix R has been pushed
    through both the estimates and the bootstrap covariance::

        H0: δ_1 = δ_2 = ... = δ_k   <=>   R δ = 0,  R = [e_j - e_{j+1}]

    Under the null the statistic is χ²(k−1) — one fewer degree of freedom
    than the corresponding all-zero test, because equality leaves the
    common level free.

    Returns ``None`` when fewer than two effects are available (nothing to
    compare) or the bootstrap has too few usable draws to form R V R'.
    """
    if len(indices) < 2:
        return None
    est = np.array(
        [
            next(
                (
                    r["delta_l"]
                    for r in main["cell_estimates"]
                    if r["horizon"] == horizons[j]
                ),
                np.nan,
            )
            for j in indices
        ],
        dtype=float,
    )
    if np.any(np.isnan(est)):
        return None

    sub = boot_hist[:, indices]
    valid = ~np.any(np.isnan(sub), axis=1)
    if valid.sum() < len(indices) + 1:
        return None
    cov = np.cov(sub[valid], rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    k = len(indices)
    contrast = np.zeros((k - 1, k))
    for r in range(k - 1):
        contrast[r, r] = 1.0
        contrast[r, r + 1] = -1.0

    out = _dc.joint_wald(contrast @ est, contrast @ cov @ contrast.T)
    out["horizons"] = [horizons[j] for j in indices]
    return out


def _joint_test_from_vcov(
    es_rows: List[Dict[str, Any]],
    es_vcov: Optional[pd.DataFrame],
    horizons: List[int],
    indices: List[int],
    equal: bool = False,
) -> Optional[Dict[str, Any]]:
    """Wald test from the analytic joint covariance of the horizons.

    ``equal=False`` tests that every estimate in ``indices`` is zero
    (chi-squared, ``k`` degrees of freedom); ``equal=True`` tests that they
    are all equal, through the adjacent contrasts (``k - 1``). This is how
    the reference computes its "joint nullity" and "equality of the
    effects" p-values.
    """
    if es_vcov is None or len(indices) < (2 if equal else 1):
        return None
    hs = [int(horizons[j]) for j in indices]
    if any(h not in es_vcov.index for h in hs):
        return None
    est = np.array([es_rows[j]["att"] for j in indices], dtype=float)
    if np.any(~np.isfinite(est)):
        return None
    cov = es_vcov.loc[hs, hs].to_numpy(dtype=float)
    if equal:
        k = len(hs)
        contrast = np.zeros((k - 1, k))
        for r in range(k - 1):
            contrast[r, r] = 1.0
            contrast[r, r + 1] = -1.0
        out: Dict[str, Any] = dict(
            _dc.joint_wald(contrast @ est, contrast @ cov @ contrast.T, ridge=0.0)
        )
        out["horizons"] = hs
        return out
    return dict(_dc.joint_wald(est, cov, ridge=0.0))


def _joint_test_from_boot(
    main: Dict[str, Any],
    horizons: List[int],
    boot_hist: np.ndarray,
    indices: List[int],
) -> Optional[Dict[str, Any]]:
    if not indices:
        return None
    est = np.array(
        [
            next(
                (
                    r["delta_l"]
                    for r in main["cell_estimates"]
                    if r["horizon"] == horizons[j]
                ),
                np.nan,
            )
            for j in indices
        ],
        dtype=float,
    )
    sub = boot_hist[:, indices]
    valid = ~np.any(np.isnan(sub), axis=1)
    if valid.sum() < len(indices) + 1:
        return None
    cov = np.cov(sub[valid], rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    return _dc.joint_wald(est, cov)
