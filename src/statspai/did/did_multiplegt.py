"""
de Chaisemartin & D'Haultfœuille (2020) DID_M estimator.

Estimates the effect of a binary or discrete treatment that can move in
both directions, robust to heterogeneous treatment effects across groups
and time periods.
Unlike Callaway & Sant'Anna (which assumes staggered adoption / no
treatment reversal), this estimator handles general treatment paths
where units can enter and exit treatment.

The DID_M estimator is a **pair-rollup**: for each consecutive period
pair ``(t-1, t)`` it computes a cell-level DID comparing "switchers"
(units whose treatment changed) to "stayers" (units whose treatment
did not change), then aggregates to an overall estimate using the
switcher count per cell as the weight.

**Scope note** — by default the ``dynamic=`` argument is the pair-rollup
event-study extension (long differences relative to stayers at each
pair) of R ``DIDmultiplegt`` 0.1.4. ``robust_dynamic=True`` switches to
the first-time-switcher DID_l estimators of Stata ``did_multiplegt_old,
robust_dynamic`` (``_dcdh_first_switch.py``). The dCDH (2024) event study
with its influence-function variance is ``sp.did_multiplegt_dyn``.

References
----------
de Chaisemartin, C. and D'Haultfoeuille, X. (2020).
"Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects."
*American Economic Review*, 110(9), 2964-2996. [@dechaisemartin2020two]

de Chaisemartin, C. and D'Haultfoeuille, X. (2022).
"Two-way fixed effects and differences-in-differences with heterogeneous
treatment effects: A survey."  *The Econometrics Journal*, 26(3), C1-C30.
[@dechaisemartin2022fixed]

de Chaisemartin, C. and D'Haultfoeuille, X. (2024).
"Difference-in-Differences Estimators of Intertemporal Treatment Effects."
*Review of Economics and Statistics*, forthcoming.  (Joint placebo test
and average cumulative effect, Section 3.) [@dechaisemartin2024difference]
"""

import warnings
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._input_validation import check_columns, missing_named_columns
from ..core._call_record import records_call
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core._vcov_spec import accepts_vce
from ..core.results import CausalResult
from ..exceptions import ColumnNotFound, MethodIncompatibility
from ._arg_roles import check_unit_is_not_treatment as _check_unit_is_not_treatment
from ._dcdh_cells import _collapse_to_cells, _stayer_slopes
from ._dcdh_first_switch import first_switch_effect
from ._dcdh_inference import (
    _avg_cumulative_effect,
    _effect_average,
    _joint_placebo_test,
)
from ._dcdh_pairwise import _estimate_did_m, _estimate_dynamic, _estimate_placebo

# ======================================================================
# Public API
# ======================================================================


@records_call("did_multiplegt")
@accepts_aliases(
    _strict=True, id="group", unit="group", treat="treatment", covariates="controls"
)
@accepts_vce(default="bootstrap")
@_expands_categorical("controls")
def did_multiplegt(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]] = None,
    placebo: int = 0,
    dynamic: int = 0,
    cluster: Optional[str] = None,
    n_boot: int = 100,
    seed: Optional[int] = None,
    alpha: float = 0.05,
    placebo_sign: str = "stata",
    robust_dynamic: bool = False,
    placebo_controls: Optional[Literal["refit", "main"]] = None,
    firstdiff_placebo: bool = False,
) -> CausalResult:
    """
    de Chaisemartin & D'Haultfœuille (2020) DID_M estimator.

    Estimates the effect of a binary or discrete treatment that can move in
    both directions, robust to heterogeneous treatment effects across groups
    and time.

    **Note** — by default ``dynamic=H`` extends the DID_M pair rollup to H
    horizons, as R ``DIDmultiplegt`` 0.1.4 does. ``robust_dynamic=True``
    computes the first-time-switcher DID_l estimators of Stata
    ``did_multiplegt_old, robust_dynamic`` instead. The two coincide on a
    binary staggered (absorbing) treatment with one row per group and
    period, and differ when treatment switches on and off. For the dCDH
    (2024) event study with its analytic variance and full option set, see
    ``sp.did_multiplegt_dyn``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format: one row per group and period, or several
        (individuals within groups). Several rows are averaged to the
        (group, period) cell and the cell is weighted by its row count, as
        Stata ``did_multiplegt_old`` does.

        .. versionchanged:: 1.40.1
           ⚠️ Correctness fix: several rows per cell used to be paired
           across periods (main estimate) or reduced to the first row
           (placebo and dynamic effects).
    y : str
        Outcome variable name.
    group : str
        Group (unit) identifier.
    time : str
        Time period variable.
    treatment : str
        Treatment in the current period: a 0/1 indicator or a discrete
        (ordered) treatment such as a count. Unlike Callaway-Sant'Anna,
        this is the *current* treatment, not the first-treatment period,
        and units may move in both directions.

        With a non-binary treatment switchers are compared to stayers
        that had the same treatment in the previous period, each switcher
        enters with the sign of its own change, and the sum is divided by
        the total absolute change of treatment, so ``DID_M`` is an effect
        per unit of treatment (the non-binary estimator of dCDH 2020, as
        Stata ``did_multiplegt_old`` computes it). A baseline level with
        switchers but no stayer, or the reverse, contributes nothing.

        .. versionchanged:: 1.39.0
           ⚠️ Correctness fix for non-binary treatments. Every switcher
           from a baseline other than 0 used to be signed as a switch
           *off* whichever way it moved, and nothing was divided by the
           size of the change. On the Gentzkow, Shapiro and Sinkinson
           (2011) newspaper panel that returned -0.00082 where
           ``did_multiplegt_old`` returns 0.0057791. Binary treatments
           are unaffected.
    controls : list of str, optional
        Control variables. The outcome difference of every estimate
        becomes ``dY - dX @ theta_d``, with ``theta_d`` from a regression of
        dY on dX and period effects, per baseline treatment ``d``, over the
        estimate's control cells (weighted like the cells). For DID_M the
        sample is the stayer cells (Stata ``did_multiplegt_old`` and R
        ``DIDmultiplegt``). For a dynamic effect it is that horizon's robust
        stayers with long differences, for a placebo its stayers with
        lagged differences (R; see ``placebo_controls``). With
        ``robust_dynamic=True`` it is the cells not yet switched, with first
        differences, for every estimate (Stata). A control with no
        variation net of the period effects gets a slope of 0, as in both
        references.

        .. versionchanged:: 1.40.1
           ⚠️ Correctness fix: the slope used to be fitted within each
           period's switchers and stayers, and the placebo and dynamic
           effects ignored the controls.
    placebo : int, default 0
        Number of placebo (pre-treatment) tests.  Placebo *l* checks
        whether switchers at *t* already differed from stayers between
        *t-l-1* and *t-l*.
    dynamic : int, default 0
        Number of dynamic (post-treatment) effect horizons to estimate.
        Dynamic effect *l* measures the long difference Y_{t+l} - Y_{t-1}
        for switchers at *t* that keep their new treatment through *t+l*,
        relative to groups that keep the baseline through *t+l* (R
        ``DIDmultiplegt`` 0.1.4, which documents the option for binary
        staggered designs; Stata ``did_multiplegt_old`` computes dynamic
        effects only with ``robust_dynamic``). With several rows per cell a
        cell weighs its size in period *t*.

        .. versionchanged:: 1.40.1
           With several rows per cell the weight is the size at *t* (R).
           The unreleased 2026-10-21 change weighed it at *t+l*, which is
           Stata's first-time-switcher rule, now ``robust_dynamic=True``.
    cluster : str, optional
        Variable for cluster bootstrap.  Defaults to ``group``.
    n_boot : int, default 100
        Number of bootstrap replications for standard errors.
    seed : int, optional
        Random seed for reproducibility.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    placebo_sign : {"stata", "r"}, default "stata"
        Sign convention of the placebos. ``"r"`` is the forward difference
        ``Y_{t-l} - Y_{t-l-1}`` of switchers minus stayers, which is what R
        ``DIDmultiplegt`` 0.1.4 **and** Stata ``did_multiplegt_old`` without
        ``robust_dynamic`` report. ``"stata"`` is its negative, the
        convention of Stata ``did_multiplegt_old, robust_dynamic`` (on
        ``did::mpdta``: ``+0.024269`` there, ``-0.024269`` in R). The
        magnitude and the inference are the same either way; pass ``"r"``
        to reproduce a plain ``did_multiplegt_old Y G T D, placebo()`` run.
        The default is what this function has always returned. Must stay
        ``"stata"`` with ``robust_dynamic=True``, whose long-difference
        placebos have their own fixed sign.
    robust_dynamic : bool, default False
        Compute the first-time-switcher estimators of Stata
        ``did_multiplegt_old, robust_dynamic`` (the DID_l of dCDH, REStat
        2024 [@dechaisemartin2024difference]). A group's baseline is its
        first observed treatment; DID_l compares the outcome change from the
        period before its first switch to *l* periods after it with groups
        of the same baseline that have not switched yet, without dividing
        by the size of the change. The estimate becomes DID_0 (Stata's
        ``e(effect_0)``), ``dynamic`` gives DID_1..DID_L, and placebo *l* is
        the long difference ``Y_{F-1-l} - Y_{F-1}`` over DID_{l-1}'s groups.
        A group enters only if observed in every period of the effect's
        window. Matches Stata 18 to 1e-10 on balanced and unbalanced
        panels, binary and count treatments, with and without controls and
        with several rows per cell.

        With ``dynamic >= 1``, ``model_info['effect_average']`` holds Stata's
        ``e(effect_average)``: the DID_0..DID_L weighted by the number of
        switchers each applies to, divided by the same average with the
        treatment as outcome (the first stage). It estimates an effect per
        unit of treatment of the first switches; its SE is the bootstrap
        standard deviation of the ratio.
    placebo_controls : {"refit", "main"}, optional
        How ``controls`` enter the placebos of the default (pair-by-pair)
        estimator, where the two references differ. ``"refit"`` (used when
        omitted) fits the slopes on the placebo's own stayers and lagged
        differences, as R ``DIDmultiplegt`` 0.1.4 does. ``"main"`` reuses
        the slopes of the main DID_M estimate, as Stata
        ``did_multiplegt_old`` does without ``robust_dynamic``. Not allowed
        with ``robust_dynamic=True``.
    firstdiff_placebo : bool, default False
        With ``robust_dynamic=True``, compute first-difference placebos
        (Stata ``firstdiff_placebo``): placebo *l* compares the outcome
        change ``Y_{F-l} - Y_{F-l-1}`` of the first-time switchers with that
        of the groups not yet switched at *F*, the comparison of DID_0, so
        it tests parallel trends over two consecutive periods *l* periods
        before the switch (useful for anticipation) and does not depend on
        ``dynamic``. Placebo 1 is minus the long-difference placebo 1.
        Requires ``robust_dynamic=True``: the default placebos already are
        first differences (Stata ignores the option without
        ``robust_dynamic``).

    Returns
    -------
    CausalResult
        Result object with ``.summary()`` and ``.plot()`` methods.
        The ``detail`` DataFrame contains all (group, time)-level DID
        estimates; ``model_info`` stores event-study coefficients
        (placebo and dynamic effects) for plotting.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for county in range(12):
    ...     state = county % 3
    ...     g = int(rng.choice([4, 6, 0]))  # cohort; 0 = never treated
    ...     for year in range(1, 9):
    ...         on = 1 if (g != 0 and year >= g) else 0
    ...         rows.append({'county': county, 'state': state, 'year': year,
    ...                      'treated': on,
    ...                      'wage': county + 0.2 * year + 1.5 * on
    ...                              + rng.normal(0, 0.5)})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.did_multiplegt(
    ...     data=df, y="wage", group="county", time="year",
    ...     treatment="treated", placebo=1, dynamic=2,
    ...     cluster="state", n_boot=50, seed=42,
    ... )
    >>> bool(np.isfinite(result.estimate))
    True

    Notes
    -----
    The estimator proceeds as follows for each consecutive pair of
    periods (t-1, t):

    1. **Switchers**: units whose treatment status changed between t-1
       and t.
    2. **Stayers**: units whose treatment status did *not* change.
    3. DID_{g,t} = mean(DeltaY_switchers) - mean(DeltaY_stayers),
       where DeltaY = Y_t - Y_{t-1}.
    4. DID_M = weighted average of DID_{g,t} with weights proportional
       to the number of switchers in each cell.

    Standard errors are computed via cluster bootstrap (resampling
    clusters with replacement).
    """
    # ── Validate inputs ──────────────────────────────────────────── #
    if placebo_sign not in {"stata", "r"}:
        raise ValueError(f"placebo_sign must be 'stata' or 'r', got {placebo_sign!r}")
    if placebo_controls not in (None, "refit", "main"):
        raise ValueError(
            "placebo_controls must be 'refit' or 'main', got " f"{placebo_controls!r}"
        )
    if robust_dynamic and (placebo_sign != "stata" or placebo_controls is not None):
        raise MethodIncompatibility(
            "sp.did_multiplegt: robust_dynamic=True computes Stata "
            "did_multiplegt_old's first-time-switcher estimators, whose "
            "placebos are long differences with a fixed sign and whose "
            "controls are fitted on the not-yet-switched cells; "
            "placebo_sign='r' and placebo_controls= apply to the pair-by-pair "
            "estimators only.",
            recovery_hint=(
                "Drop placebo_sign / placebo_controls, or set "
                "robust_dynamic=False to use them."
            ),
            diagnostics={
                "placebo_sign": placebo_sign,
                "placebo_controls": placebo_controls,
            },
        )
    if firstdiff_placebo and not robust_dynamic:
        raise MethodIncompatibility(
            "sp.did_multiplegt: firstdiff_placebo=True selects the "
            "first-difference placebos of the first-time-switcher estimators "
            "(robust_dynamic=True). The default pair-by-pair placebos already "
            "are first differences.",
            recovery_hint=(
                "Set robust_dynamic=True, or drop firstdiff_placebo: without "
                "robust_dynamic the placebos are first differences already "
                "(Stata did_multiplegt_old ignores the option there)."
            ),
            diagnostics={"firstdiff_placebo": True, "robust_dynamic": False},
        )
    placebo_controls_used = placebo_controls or "refit"
    df = data.copy()
    check_columns(
        df,
        {
            "y": y,
            "group": group,
            "time": time,
            "treatment": treatment,
            "cluster": cluster,
        },
        function="sp.did_multiplegt",
    )
    missing = missing_named_columns(
        {"controls": None if controls is None else list(controls)}, df.columns
    )
    if missing:
        raise ColumnNotFound.from_missing(
            missing,
            list(df.columns),
            message=f"Control column '{next(iter(missing.values()))}' "
            "not found in data.",
        )

    cluster_var = cluster if cluster is not None else group

    # A missing outcome or treatment makes every first difference it enters
    # missing, and one such difference wipes out the whole cell mean. The
    # cell is unusable, so it is removed before the differences are formed.
    _cell_cols = list(dict.fromkeys([y, group, time, treatment, *(controls or [])]))
    _incomplete = df[_cell_cols].isna().any(axis=1).to_numpy()
    n_incomplete = int(_incomplete.sum())
    if n_incomplete:
        # The count is reported by the shared listwise-deletion warning
        # (output/_lineage), which compares n_obs with the input frame.
        df = df.loc[~_incomplete]

    # group= is the unit id. A 0/1 column with several rows per period and
    # a treatment that varies within them is the treatment indicator passed
    # in its place: the period-to-period merge on group would compare rows
    # of different units, so refuse; with a constant treatment, warn.
    _check_unit_is_not_treatment(
        df,
        unit=group,
        time=time,
        cohort=None,
        function="did_multiplegt",
        cell_constant=treatment,
        example_tail=f"treatment={treatment!r}",
    )

    # Data below the group level (individuals within groups): Stata's
    # did_multiplegt_old averages every variable to (group, period) cells
    # and weights each cell by its number of observations, which equals
    # running it on the collapsed data with weight(n). Pairing the rows of
    # a group across periods instead weighted a cell by n_t * n_{t-1}, and
    # the placebo / dynamic paths kept only the first row of each cell.
    wcol: Optional[str] = None
    n_rows_input = len(df)
    if df.duplicated([group, time]).any():
        df, wcol = _collapse_to_cells(
            df, y, group, time, treatment, controls, cluster_var
        )

    # ── Sort and compute main estimate ───────────────────────────── #
    df = df.sort_values([group, time]).reset_index(drop=True)

    def _main(frame: pd.DataFrame) -> Dict[str, Any]:
        if not robust_dynamic:
            return _estimate_did_m(frame, y, group, time, treatment, controls, w=wcol)
        # Stata's e(effect_0) under robust_dynamic is DID_0 of the
        # first-time switchers, not DID_M.
        first = first_switch_effect(
            frame, y, group, time, treatment, controls, "dynamic", 0, w=wcol
        )
        return {
            "did_m": first["estimate"],
            "n_switching_cells": first["n_cells"],
            "n_switchers": first["n_switchers"],
            "cell_estimates": first["cells"],
        }

    def _placebo(frame: pd.DataFrame, lag: int) -> Dict[str, Any]:
        if robust_dynamic:
            kind = "placebo_fd" if firstdiff_placebo else "placebo"
            return first_switch_effect(
                frame, y, group, time, treatment, controls, kind, lag, w=wcol
            )
        main_slopes = (
            _stayer_slopes(frame, y, group, time, treatment, controls, wcol)
            if controls and placebo_controls_used == "main"
            else None
        )
        return _estimate_placebo(
            frame,
            y,
            group,
            time,
            treatment,
            controls,
            lag=lag,
            placebo_sign=placebo_sign,
            w=wcol,
            placebo_controls=placebo_controls_used,
            main_slopes=main_slopes,
        )

    def _dynamic(frame: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        if robust_dynamic:
            return first_switch_effect(
                frame, y, group, time, treatment, controls, "dynamic", horizon, w=wcol
            )
        return _estimate_dynamic(
            frame, y, group, time, treatment, controls, horizon=horizon, w=wcol
        )

    main = _main(df)

    # ── Placebo effects ──────────────────────────────────────────── #
    placebo_results = [_placebo(df, lag_idx) for lag_idx in range(1, placebo + 1)]

    # ── Dynamic effects ──────────────────────────────────────────── #
    dynamic_results = [_dynamic(df, h) for h in range(0, dynamic + 1)]

    # ── Bootstrap standard errors ────────────────────────────────── #
    rng = np.random.default_rng(seed)
    clusters = df[cluster_var].unique()
    n_clusters = len(clusters)

    boot_main = np.full(n_boot, np.nan)
    boot_placebo = np.full((n_boot, max(placebo, 0)), np.nan) if placebo > 0 else None
    boot_dynamic = np.full((n_boot, dynamic + 1), np.nan) if dynamic >= 0 else None
    # Stata reports e(effect_average) with robust_dynamic and dynamic >= 1
    want_effect_average = robust_dynamic and dynamic >= 1
    boot_effect_average = np.full(n_boot, np.nan)

    for b in range(n_boot):
        sampled = rng.choice(clusters, size=n_clusters, replace=True)
        # Build bootstrap sample preserving panel structure
        frames = []
        for j, c in enumerate(sampled):
            chunk = df[df[cluster_var] == c].copy()
            # Relabel group to avoid collisions from sampling with replacement
            chunk[group] = chunk[group].astype(str) + f"_b{j}"
            frames.append(chunk)
        bdf = pd.concat(frames, ignore_index=True)

        boot_main[b] = _main(bdf)["did_m"]

        if placebo > 0:
            assert boot_placebo is not None
            for lag_idx in range(1, placebo + 1):
                boot_placebo[b, lag_idx - 1] = _placebo(bdf, lag_idx)["estimate"]

        if dynamic >= 0:
            assert boot_dynamic is not None
            boot_dyn = [_dynamic(bdf, h) for h in range(0, dynamic + 1)]
            boot_dynamic[b] = [r["estimate"] for r in boot_dyn]
            if want_effect_average:
                boot_effect_average[b] = _effect_average(boot_dyn, None, alpha)[
                    "estimate"
                ]

    # ── Compute SEs, p-values, CIs ───────────────────────────────── #
    z_crit = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(main["did_m"]):
        se_main = np.nanstd(boot_main, ddof=1)
        z_main = main["did_m"] / se_main if se_main > 0 else 0.0
        p_main = 2 * stats.norm.sf(abs(z_main))
        ci_main = (
            main["did_m"] - z_crit * se_main,
            main["did_m"] + z_crit * se_main,
        )
    else:
        se_main = p_main = float("nan")
        ci_main = (float("nan"), float("nan"))
        warnings.warn(
            (
                (
                    "did_multiplegt: no group switches treatment for the first "
                    "time while another group with the same initial treatment has "
                    "not switched yet, so DID_0 is not estimable; reported as "
                    "missing."
                )
                if robust_dynamic
                else (
                    "did_multiplegt: no group changes treatment between two "
                    "consecutive periods while another group with the same "
                    "previous treatment stays put, so DID_M is not estimable; "
                    "reported as missing."
                )
            ),
            UserWarning,
            stacklevel=2,
        )

    def _inference_row(est: float, draws: Optional[np.ndarray]) -> Dict[str, float]:
        """Bootstrap se, p-value and interval; all missing if not estimable."""
        nan = float("nan")
        if not np.isfinite(est):
            return {
                "estimate": nan,
                "se": nan,
                "pvalue": nan,
                "ci_lower": nan,
                "ci_upper": nan,
            }
        se = float(np.nanstd(draws, ddof=1)) if draws is not None else 0.0
        z = est / se if se > 0 else 0.0
        return {
            "estimate": est,
            "se": se,
            "pvalue": float(2 * stats.norm.sf(abs(z))),
            "ci_lower": est - z_crit * se,
            "ci_upper": est + z_crit * se,
        }

    placebo_out = [
        {
            "lag": -(lag_idx + 1),
            **_inference_row(
                placebo_results[lag_idx]["estimate"],
                boot_placebo[:, lag_idx] if boot_placebo is not None else None,
            ),
            # first-time switchers (rows) behind placebo l; robust_dynamic only
            **(
                {"n_switchers": placebo_results[lag_idx]["n_switchers"]}
                if robust_dynamic
                else {}
            ),
        }
        for lag_idx in range(placebo)
    ]
    dynamic_out = [
        {
            "horizon": horizon_idx,
            **_inference_row(
                dynamic_results[horizon_idx]["estimate"],
                boot_dynamic[:, horizon_idx] if boot_dynamic is not None else None,
            ),
            # first-time switchers (rows) behind DID_l; robust_dynamic only
            **(
                {"n_switchers": dynamic_results[horizon_idx]["n_switchers"]}
                if robust_dynamic
                else {}
            ),
        }
        for horizon_idx in range(dynamic + 1)
    ]
    # missing estimate, not missing SE: with n_boot=0 every SE is NaN
    not_estimable = [
        f"placebo {r['lag']}" for r in placebo_out if np.isnan(r["estimate"])
    ]
    not_estimable += [
        f"dynamic {r['horizon']}" for r in dynamic_out if np.isnan(r["estimate"])
    ]
    if not_estimable:
        warnings.warn(
            "did_multiplegt: no switching cell with a comparison cell for "
            f"{', '.join(not_estimable)}; reported as missing. The panel has "
            "too few periods before or after the switches for these.",
            UserWarning,
            stacklevel=2,
        )

    effect_average = (
        _effect_average(dynamic_results, boot_effect_average, alpha)
        if want_effect_average
        else None
    )
    if (
        effect_average is not None
        and effect_average["n_horizons"]
        and not np.isfinite(effect_average["estimate"])
    ):
        warnings.warn(
            "did_multiplegt: the first-time switchers' treatment changes sum "
            "to zero over the dynamic effects, so effect_average (the "
            "switcher-weighted DID_l over the same average with the treatment "
            "as outcome) is not defined; reported as missing.",
            UserWarning,
            stacklevel=2,
        )

    # ── Build detail DataFrame ───────────────────────────────────── #
    detail_rows = main.get("cell_estimates", [])
    detail_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()

    # ── Event-study DataFrame for plotting ───────────────────────── #
    # CausalResult.summary() and .plot() expect columns:
    #   relative_time, att, se, pvalue, ci_lower, ci_upper
    es_rows = []
    for p_row in placebo_out:
        es_rows.append(
            {
                "relative_time": p_row["lag"],
                "att": p_row["estimate"],
                "se": p_row["se"],
                "pvalue": p_row["pvalue"],
                "ci_lower": p_row["ci_lower"],
                "ci_upper": p_row["ci_upper"],
                "type": "placebo",
            }
        )
    for d_row in dynamic_out:
        es_rows.append(
            {
                "relative_time": d_row["horizon"],
                "att": d_row["estimate"],
                "se": d_row["se"],
                "pvalue": d_row["pvalue"],
                "ci_lower": d_row["ci_lower"],
                "ci_upper": d_row["ci_upper"],
                "type": "dynamic",
            }
        )
    es_df = pd.DataFrame(es_rows) if es_rows else pd.DataFrame()

    # ── Joint placebo test + avg cumulative dynamic effect ───────── #
    joint_placebo = _joint_placebo_test(placebo_results, boot_placebo)
    avg_cumulative = _avg_cumulative_effect(
        dynamic_results,
        boot_dynamic,
        alpha,
    )

    # ── Assemble model_info ──────────────────────────────────────── #
    _method = (
        "de Chaisemartin-D'Haultfoeuille DID_l (first-time switchers)"
        if robust_dynamic
        else "de Chaisemartin-D'Haultfoeuille (2020)"
    )
    model_info: Dict[str, Any] = {
        "estimator": _method,
        "n_switching_cells": main["n_switching_cells"],
        "n_switchers": main["n_switchers"],
        "n_boot": n_boot,
        "seed": seed,
        "n_rows_dropped_missing": n_incomplete,
        "cluster_var": cluster_var,
        "controls": controls,
        # the main, placebo and dynamic estimates are all adjusted
        "controls_scope": "all" if controls else None,
        "robust_dynamic": bool(robust_dynamic),
        # how the placebos' control slopes are fitted (pair-by-pair path)
        "placebo_controls": (
            placebo_controls_used
            if controls and placebo and not robust_dynamic
            else None
        ),
        # rows below the group level are averaged to (group, period) cells,
        # each weighted by its row count (Stata did_multiplegt_old)
        "collapsed_to_cells": wcol is not None,
        "n_cells": len(df),
        "placebo": placebo_out,
        "dynamic": dynamic_out,
        "event_study": es_df,
        "joint_placebo_test": joint_placebo,
        "avg_cumulative_effect": avg_cumulative,
        "effect_average": effect_average,
        "firstdiff_placebo": bool(firstdiff_placebo),
    }

    _result = CausalResult(
        method=_method,
        estimand="ATT",
        estimate=float(main["did_m"]),
        se=float(se_main),
        pvalue=float(p_main),
        ci=ci_main,
        alpha=alpha,
        n_obs=n_rows_input,
        detail=detail_df,
        model_info=model_info,
        _citation_key="did_multiplegt",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.did_multiplegt",
            params={
                "y": y,
                "group": group,
                "time": time,
                "treatment": treatment,
                "controls": controls,
                "placebo": placebo,
                "dynamic": dynamic,
                "cluster": cluster,
                "n_boot": n_boot,
                "seed": seed,
                "alpha": alpha,
                "placebo_sign": placebo_sign,
                "robust_dynamic": robust_dynamic,
                "placebo_controls": placebo_controls,
                "firstdiff_placebo": firstdiff_placebo,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
