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
from typing import Any, Dict, List, Literal, Optional, Tuple

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
from ._dcdh_first_switch import first_switch_effect, period_fe_slopes

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

        .. versionchanged:: 1.41.0
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

        .. versionchanged:: 1.41.0
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

        .. versionchanged:: 1.41.0
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
        with several rows per cell. Stata's ``firstdiff_placebo`` and its
        ``effect_average`` ratio are not implemented.
    placebo_controls : {"refit", "main"}, optional
        How ``controls`` enter the placebos of the default (pair-by-pair)
        estimator, where the two references differ. ``"refit"`` (used when
        omitted) fits the slopes on the placebo's own stayers and lagged
        differences, as R ``DIDmultiplegt`` 0.1.4 does. ``"main"`` reuses
        the slopes of the main DID_M estimate, as Stata
        ``did_multiplegt_old`` does without ``robust_dynamic``. Not allowed
        with ``robust_dynamic=True``.

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
            return first_switch_effect(
                frame, y, group, time, treatment, controls, "placebo", lag, w=wcol
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
            for horizon_idx in range(0, dynamic + 1):
                boot_dynamic[b, horizon_idx] = _dynamic(bdf, horizon_idx)["estimate"]

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
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ======================================================================
# Internal helpers
# ======================================================================


_CELL_N = "_dcdh_cell_n"


def _collapse_to_cells(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    cluster_var: str,
) -> Tuple[pd.DataFrame, str]:
    """Average rows below the group level to (group, period) cells.

    Every analysis variable becomes its cell mean and the cell keeps its
    row count in ``_CELL_N``, the weight Stata ``did_multiplegt_old``
    gives it. A treatment that varies within a cell is averaged too, as
    Stata does. A clustering variable must be constant within a cell.
    """
    keys = [group, time]
    if cluster_var not in keys:
        varies = df.groupby(keys, sort=False)[cluster_var].nunique(dropna=False)
        if (varies > 1).any():
            raise MethodIncompatibility(
                f"sp.did_multiplegt: cluster variable {cluster_var!r} varies "
                f"within ({group}, {time}) cells. The estimator works on "
                "(group, period) cell means, so the bootstrap clusters must "
                "contain whole cells.",
                recovery_hint=(
                    f"Cluster on {group!r} or on a variable coarser than the "
                    "(group, period) cell, e.g. a region that contains whole "
                    "groups."
                ),
                diagnostics={"cluster": cluster_var},
                alternative_functions=["sp.did_multiplegt_dyn"],
            )
    cols = list(dict.fromkeys([y, treatment, *(controls or [])]))
    grouped = df.groupby(keys, sort=False)
    out = grouped[cols].mean()
    out[_CELL_N] = grouped.size().astype(float)
    if cluster_var not in keys:
        out[cluster_var] = grouped[cluster_var].first()
    return out.reset_index(), _CELL_N


def _first_differences(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: List[str],
    w: Optional[str],
) -> pd.DataFrame:
    """(group, period) first differences between consecutive periods."""
    periods = sorted(df[time].unique())
    frames = []
    cols = [group, y, treatment, *controls]
    for t_prev, t_curr in zip(periods[:-1], periods[1:]):
        cur = df.loc[df[time] == t_curr, cols + ([w] if w else [])]
        prev = df.loc[df[time] == t_prev, cols]
        m = cur.merge(prev, on=group, how="inner", suffixes=("", "_prev"))
        if m.empty:
            continue
        out = pd.DataFrame(
            {
                "_t": t_curr,
                "_d": m[treatment].to_numpy(dtype=float),
                "_d_prev": m[f"{treatment}_prev"].to_numpy(dtype=float),
                "_dy": m[y].to_numpy(dtype=float)
                - m[f"{y}_prev"].to_numpy(dtype=float),
                "_w": m[w].to_numpy(dtype=float) if w else 1.0,
            }
        )
        for c in controls:
            out[f"_dx_{c}"] = m[c].to_numpy(dtype=float) - m[f"{c}_prev"].to_numpy(
                dtype=float
            )
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _stayer_slopes(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: List[str],
    w: Optional[str],
) -> Dict[float, np.ndarray]:
    """Slopes of dY on dX with period effects, over stayers, per baseline.

    The sample for baseline ``d`` is every (group, period) cell whose
    treatment is ``d`` in both periods. A baseline with no stayer gets no
    slope (its switchers have no comparison cell anyway).
    """
    fd = _first_differences(df, y, group, time, treatment, controls, w)
    if fd.empty:
        return {}
    dx_cols = [f"_dx_{c}" for c in controls]
    stay = fd[fd["_d"] == fd["_d_prev"]]
    slopes: Dict[float, np.ndarray] = {}
    for base, sub in stay.groupby("_d_prev", sort=True):
        slopes[float(base)] = period_fe_slopes(
            sub[dx_cols].to_numpy(dtype=float),
            sub["_t"],
            sub["_dy"].to_numpy(dtype=float),
            sub["_w"].to_numpy(dtype=float),
        )
    return slopes


def _wmean(x: np.ndarray, w: Optional[np.ndarray]) -> float:
    """Mean of ``x``, weighted by ``w`` when given."""
    if w is None:
        return float(np.mean(x))
    return float(np.sum(w * x) / np.sum(w))


def _estimate_did_m(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    w: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Core DID_M estimator: weighted average of cell-level DID estimates.

    ``w`` names a column of (group, period) cell sizes (data collapsed by
    :func:`_collapse_to_cells`); each cell then counts that many times.

    For each period t, identify switchers and stayers, compute DID_gt.

    With ``controls`` the first difference of the outcome is replaced by
    ``dY - dX @ theta_d``, where ``theta_d`` comes from a regression of dY
    on dX and period effects over the stayer cells whose treatment is
    ``d`` in both periods (Stata ``did_multiplegt_old``, one regression per
    baseline value, weighted like the cells).
    """
    periods = sorted(df[time].unique())
    cell_estimates = []
    total_switchers = 0
    slopes = (
        _stayer_slopes(df, y, group, time, treatment, controls, w) if controls else {}
    )

    for idx in range(1, len(periods)):
        t_prev, t_curr = periods[idx - 1], periods[idx]

        df_prev = df[df[time] == t_prev][
            [group, y, treatment] + (controls or [])
        ].copy()
        df_curr = df[df[time] == t_curr][
            [group, y, treatment] + (controls or []) + ([w] if w else [])
        ].copy()

        df_prev = df_prev.rename(
            columns={y: f"{y}_prev", treatment: f"{treatment}_prev"},
        )
        if controls:
            df_prev = df_prev.rename(columns={c: f"{c}_prev" for c in controls})

        merged = df_curr.merge(df_prev, on=group, how="inner")
        if merged.empty:
            continue

        merged["_switched"] = merged[treatment] != merged[f"{treatment}_prev"]
        merged["_dy"] = merged[y] - merged[f"{y}_prev"]

        # de Chaisemartin--D'Haultfoeuille (2020): the cell-level DID must
        # condition on the BASELINE treatment d_{t-1}. Switchers are compared
        # only to stayers that shared the same period-(t-1) treatment value, and
        # switch-OFF cells (baseline 1 -> 0) enter with a flipped sign so the
        # estimand is the effect of *gaining* treatment. Pooling across baselines
        # (i) contaminates the control trend with already-treated stayers and
        # (ii) mixes switch-on / switch-off effects under a single sign -- both
        # bias DID_M (see Paper-DiD-JAE/replication/did_multiplegt/).
        for base in sorted(merged[f"{treatment}_prev"].unique()):
            grp = merged[merged[f"{treatment}_prev"] == base]
            switchers = grp[grp["_switched"]]
            stayers = grp[~grp["_switched"]]

            n_switch = len(switchers)
            if n_switch == 0 or len(stayers) == 0:
                continue
            w_sw = switchers[w].to_numpy(dtype=float) if w else None
            w_st = stayers[w].to_numpy(dtype=float) if w else None

            dy_switch = switchers["_dy"].values.copy()
            dy_stay = stayers["_dy"].values.copy()

            # Net out the controls with the stayer-sample slopes of this
            # baseline value (Stata did_multiplegt_old's controls()).
            if controls:
                theta = slopes.get(float(base))
                if theta is not None:
                    dX_switch = np.column_stack(
                        [
                            switchers[c].to_numpy(dtype=float)
                            - switchers[f"{c}_prev"].to_numpy(dtype=float)
                            for c in controls
                        ]
                    )
                    dX_stay = np.column_stack(
                        [
                            stayers[c].to_numpy(dtype=float)
                            - stayers[f"{c}_prev"].to_numpy(dtype=float)
                            for c in controls
                        ]
                    )
                    dy_switch = dy_switch - dX_switch @ theta
                    dy_stay = dy_stay - dX_stay @ theta

            # Each switcher enters with the sign of its own treatment change
            # and the cell is weighted by the total absolute change, so that
            # DID_M is the effect per unit of treatment (dCDH 2020, the
            # non-binary case). With a binary treatment the change is +1 from
            # baseline 0 and -1 from baseline 1, and this is the switcher-
            # count-weighted average of signed cell DIDs it has always been.
            dd_switch = switchers[treatment].to_numpy(dtype=float) - float(base)
            sgn = np.sign(dd_switch)
            if w_sw is None:
                abs_change = float(np.abs(dd_switch).sum())
                numerator = float(np.sum(sgn * (dy_switch - np.mean(dy_stay))))
            else:
                abs_change = float(np.sum(w_sw * np.abs(dd_switch)))
                numerator = float(
                    np.sum(w_sw * sgn * (dy_switch - _wmean(dy_stay, w_st)))
                )
                n_switch = int(round(w_sw.sum()))
            did_gt = numerator / abs_change
            if np.all(sgn > 0):
                direction = "on"
            elif np.all(sgn < 0):
                direction = "off"
            else:
                direction = "mixed"

            cell_estimates.append(
                {
                    "time": t_curr,
                    "time_prev": t_prev,
                    "baseline_treatment": float(base),
                    "n_switchers": n_switch,
                    "n_stayers": (
                        len(stayers) if w_st is None else int(round(w_st.sum()))
                    ),
                    "did_gt": did_gt,
                    "mean_dy_switch": _wmean(dy_switch, w_sw),
                    "mean_dy_stay": _wmean(dy_stay, w_st),
                    "direction": direction,
                    "treatment_change": abs_change,
                }
            )
            total_switchers += n_switch

    if total_switchers == 0:
        return {
            # Not estimable: missing, not zero (a bootstrap replicate without
            # a switching cell is then left out of the standard error).
            "did_m": float("nan"),
            "n_switching_cells": 0,
            "n_switchers": 0,
            "cell_estimates": [],
        }

    # Weighted average, weights = total absolute treatment change of the cell
    total_change = sum(ce["treatment_change"] for ce in cell_estimates)
    did_m = sum(
        ce["did_gt"] * ce["treatment_change"] / total_change for ce in cell_estimates
    )

    return {
        "did_m": float(did_m),
        "n_switching_cells": len(cell_estimates),
        "n_switchers": total_switchers,
        "cell_estimates": cell_estimates,
    }


def _control_diffs(
    xpivs: List[pd.DataFrame], idx: Any, t_a: Any, t_b: Any
) -> np.ndarray:
    """``X_{t_a} - X_{t_b}`` for the groups ``idx``, one column per control."""
    return np.column_stack(
        [(xp.loc[idx, t_a] - xp.loc[idx, t_b]).to_numpy(dtype=float) for xp in xpivs]
    )


def _refit_slopes(
    stayer_sets: List[Tuple[Any, float, Any, Any, Any]],
    ypiv: pd.DataFrame,
    xpivs: List[pd.DataFrame],
    wpiv: Optional[pd.DataFrame],
) -> Dict[float, np.ndarray]:
    """Slopes refitted on one placebo's or dynamic effect's own stayers.

    ``stayer_sets`` holds ``(t, baseline, stayers, t_a, t_b)`` for every
    switching period ``t``: the regression is of ``Y_{t_a} - Y_{t_b}`` on
    ``X_{t_a} - X_{t_b}`` and period effects over all of them, per baseline,
    each cell weighted by its size at ``t`` (R ``DIDmultiplegt`` 0.1.4, black
    box: it runs the DID_M machinery, controls included, on the lagged or
    long differences).
    """
    by_base: Dict[float, List[Tuple[np.ndarray, np.ndarray, np.ndarray, Any]]] = {}
    for t_c, base, st, t_a, t_b in stayer_sets:
        if len(st) == 0:
            continue
        dy = (ypiv.loc[st, t_a] - ypiv.loc[st, t_b]).to_numpy(dtype=float)
        dx = _control_diffs(xpivs, st, t_a, t_b)
        w = (
            wpiv.loc[st, t_c].to_numpy(dtype=float)
            if wpiv is not None
            else np.ones(len(st))
        )
        by_base.setdefault(float(base), []).append((dy, dx, w, t_c))
    slopes: Dict[float, np.ndarray] = {}
    for base, parts in by_base.items():
        period = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(parts)])
        slopes[base] = period_fe_slopes(
            np.vstack([p[1] for p in parts]),
            period,
            np.concatenate([p[0] for p in parts]),
            np.concatenate([p[2] for p in parts]),
        )
    return slopes


def _controls_complete(xpivs: List[pd.DataFrame], idx: Any, t_a: Any, t_b: Any) -> Any:
    """The groups of ``idx`` whose controls are observed at both dates."""
    keep = np.ones(len(idx), dtype=bool)
    for xp in xpivs:
        keep &= (xp.loc[idx, t_a].notna() & xp.loc[idx, t_b].notna()).to_numpy()
    return idx[keep]


def _estimate_placebo(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    lag: int,
    placebo_sign: str = "stata",
    w: Optional[str] = None,
    placebo_controls: str = "refit",
    main_slopes: Optional[Dict[float, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Placebo test at lag *l* (de Chaisemartin--D'Haultfoeuille 2020).

    For switchers at time t, the first-difference placebo compares the
    pre-treatment outcome change over (t-l-1, t-l) for switchers vs stayers,
    CONDITIONING on the baseline treatment d_{t-1} (switchers are compared only
    to same-baseline stayers; switch-off cells enter with a flipped sign).

    ⚠️ **Fixed**: every unit in the cell must have held its treatment fixed
    over the ``l`` periods BEFORE the switch (``placebo_cond`` in the
    reference). A unit that was already moving is not a clean pre-trend
    observation. This is invisible on absorbing panels — which is why it
    survived — and matters as soon as treatment can switch back.

    ⚠️ **Not fixed, because there is nothing to fix**: the SIGN is a genuine
    disagreement between dCDH's own two implementations. On ``did::mpdta``
    both report ``|placebo_1| = 0.024269`` and their three effects agree to
    six decimals, but Stata's ``did_multiplegt_old`` reports ``+0.024269``
    and R's ``DIDmultiplegt`` 0.1.4 reports ``-0.024269``. ``placebo_sign``
    selects between them; the default preserves the Stata convention this
    function has always used. See MIGRATION.md.

    With ``controls`` the placebo differences become ``dY - dX @ theta_d``.
    ``placebo_controls="refit"`` (R ``DIDmultiplegt`` 0.1.4) fits
    ``theta_d`` on this placebo's own stayers and lagged differences;
    ``"main"`` (Stata ``did_multiplegt_old``) reuses ``main_slopes``, the
    slopes of the main DID_M estimate.
    """
    periods = sorted(df[time].unique())
    dpiv = df.pivot_table(index=group, columns=time, values=treatment, aggfunc="first")
    ypiv = df.pivot_table(index=group, columns=time, values=y, aggfunc="first")
    wpiv = (
        df.pivot_table(index=group, columns=time, values=w, aggfunc="first")
        if w
        else None
    )
    xpivs = [
        df.pivot_table(index=group, columns=time, values=c, aggfunc="first")
        for c in (controls or [])
    ]

    estimates: List[float] = []
    weights: List[float] = []
    cells: List[Tuple[Any, float, Any, Any, Any, Any]] = []

    for i in range(1, len(periods)):
        t_prev, t_curr = periods[i - 1], periods[i]  # switching step (t-1 -> t)
        e_idx = i - lag  # placebo "end" period index
        b_idx = e_idx - 1  # placebo "start" period index
        if b_idx < 0:
            continue
        t_end, t_base = periods[e_idx], periods[b_idx]

        common = dpiv.index[dpiv[t_prev].notna() & dpiv[t_curr].notna()]
        for base in sorted(set(dpiv.loc[common, t_prev].dropna().unique())):
            in_base = common[dpiv.loc[common, t_prev] == base]
            sw = in_base[dpiv.loc[in_base, t_curr] != base]  # switchers from baseline
            st = in_base[dpiv.loc[in_base, t_curr] == base]  # stayers at baseline
            pre_window = periods[b_idx:i]  # t_base .. t_prev
            if lag > 0 and len(pre_window) > 1:
                # Treatment must be flat over the pre-window for both arms.
                sw = sw[
                    (dpiv.loc[sw, pre_window].eq(dpiv.loc[sw, t_prev], axis=0)).all(
                        axis=1
                    )
                ]
                st = st[
                    (dpiv.loc[st, pre_window].eq(dpiv.loc[st, t_prev], axis=0)).all(
                        axis=1
                    )
                ]
            sw = sw[ypiv.loc[sw, t_end].notna() & ypiv.loc[sw, t_base].notna()]
            st = st[ypiv.loc[st, t_end].notna() & ypiv.loc[st, t_base].notna()]
            if xpivs:
                sw = _controls_complete(xpivs, sw, t_end, t_base)
                st = _controls_complete(xpivs, st, t_end, t_base)
            cells.append((t_curr, base, sw, st, t_end, t_base))

    theta: Dict[float, np.ndarray] = {}
    if xpivs:
        theta = (
            dict(main_slopes or {})
            if placebo_controls == "main"
            else _refit_slopes(
                [(c[0], c[1], c[3], c[4], c[5]) for c in cells], ypiv, xpivs, wpiv
            )
        )

    for t_curr, base, sw, st, t_end, t_base in cells:
        if len(sw) == 0 or len(st) == 0:
            continue
        dd = dpiv.loc[sw, t_curr].to_numpy(dtype=float) - float(base)
        pdiff = (ypiv.loc[sw, t_end] - ypiv.loc[sw, t_base]).to_numpy(dtype=float)
        st_diff = ypiv.loc[st, t_end] - ypiv.loc[st, t_base]
        if xpivs:
            th = theta.get(float(base))
            if th is None:
                continue
            pdiff = pdiff - _control_diffs(xpivs, sw, t_end, t_base) @ th
            st_diff = st_diff - _control_diffs(xpivs, st, t_end, t_base) @ th
        if wpiv is None:
            pt = float(st_diff.mean())
            abs_change = float(np.abs(dd).sum())
            signed = float(np.sum(np.sign(dd) * (pdiff - pt))) / abs_change
        else:
            # cells weigh their size in the switching period (Stata and R)
            w_sw = wpiv.loc[sw, t_curr].to_numpy(dtype=float)
            w_st = wpiv.loc[st, t_curr].to_numpy(dtype=float)
            pt = _wmean(st_diff.to_numpy(dtype=float), w_st)
            abs_change = float(np.sum(w_sw * np.abs(dd)))
            signed = float(np.sum(w_sw * np.sign(dd) * (pdiff - pt))) / abs_change
        # ⚠️ dCDH's own Stata and R implementations disagree here. On
        # did::mpdta both return |placebo_1| = 0.024269, and the three
        # effects agree to six decimals, but Stata reports +0.024269 and
        # DIDmultiplegt 0.1.4 reports -0.024269. Neither is "the" answer,
        # so the convention is a parameter and the default preserves what
        # this function has always reported.
        flip = -1.0 if placebo_sign == "stata" else 1.0
        estimates.append(flip * signed)
        weights.append(abs_change)

    if not estimates:
        # No switching cell with a comparison cell: not estimable. Zero
        # would read as an estimated null effect.
        return {"estimate": float("nan"), "n_cells": 0}

    total_w = sum(weights)
    weighted_est = sum(e * w / total_w for e, w in zip(estimates, weights))

    return {"estimate": float(weighted_est), "n_cells": len(estimates)}


def _estimate_dynamic(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    controls: Optional[List[str]],
    horizon: int,
    w: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Dynamic effect at horizon *l*, pair by pair (R ``DIDmultiplegt`` 0.1.4).

    For switchers at time t (relative to baseline d_{t-1}), the long-difference
    Y_{t+l} - Y_{t-1} is compared to that of ROBUST stayers: units that share the
    baseline treatment AND keep it unchanged over the whole window [t, t+l] (so
    the control trend is not contaminated by units treated during the horizon).
    Switch-off cells enter with a flipped sign.

    ⚠️ Switchers must also hold their NEW treatment through the window. A unit
    that switches at t and switches again at t+1 has no well-defined "effect of
    having switched l periods ago", and the reference implementation excludes
    it (``dynamic_cond``). Including such units is what this used to do; see
    MIGRATION.md.

    Cells of several rows weigh their size at the switching period ``t``, and
    ``controls`` are netted out with slopes refitted on the long differences
    of these robust stayers (both R, black box). The first-time-switcher
    estimator of Stata ``did_multiplegt_old, robust_dynamic`` is
    :func:`._dcdh_first_switch.first_switch_effect`.
    """
    periods = sorted(df[time].unique())
    dpiv = df.pivot_table(index=group, columns=time, values=treatment, aggfunc="first")
    ypiv = df.pivot_table(index=group, columns=time, values=y, aggfunc="first")
    wpiv = (
        df.pivot_table(index=group, columns=time, values=w, aggfunc="first")
        if w
        else None
    )
    xpivs = [
        df.pivot_table(index=group, columns=time, values=c, aggfunc="first")
        for c in (controls or [])
    ]

    estimates: List[float] = []
    weights: List[float] = []
    cells: List[Tuple[Any, float, Any, Any, Any, Any]] = []

    for i in range(1, len(periods)):
        t_prev, t_curr = periods[i - 1], periods[i]
        f_idx = i + horizon
        if f_idx >= len(periods):
            continue
        t_future = periods[f_idx]
        window = periods[i : f_idx + 1]  # t_curr .. t_future

        common = dpiv.index[dpiv[t_prev].notna() & dpiv[t_curr].notna()]
        for base in sorted(set(dpiv.loc[common, t_prev].dropna().unique())):
            in_base = common[dpiv.loc[common, t_prev] == base]
            sw = in_base[dpiv.loc[in_base, t_curr] != base]
            if horizon > 0 and len(sw) > 0:
                # Switchers must stay at the level they switched TO for the
                # whole horizon; a second switch inside the window makes the
                # horizon-l effect undefined.
                held = (dpiv.loc[sw, window].eq(dpiv.loc[sw, t_curr], axis=0)).all(
                    axis=1
                )
                sw = sw[held]
            # robust stayers: keep the baseline treatment over the whole window
            stable = (dpiv.loc[in_base, window] == base).all(axis=1)
            st = in_base[stable]
            sw = sw[ypiv.loc[sw, t_future].notna() & ypiv.loc[sw, t_prev].notna()]
            st = st[ypiv.loc[st, t_future].notna() & ypiv.loc[st, t_prev].notna()]
            if xpivs:
                sw = _controls_complete(xpivs, sw, t_future, t_prev)
                st = _controls_complete(xpivs, st, t_future, t_prev)
            cells.append((t_curr, base, sw, st, t_future, t_prev))

    theta: Dict[float, np.ndarray] = {}
    if xpivs:
        theta = _refit_slopes(
            [(c[0], c[1], c[3], c[4], c[5]) for c in cells], ypiv, xpivs, wpiv
        )

    for t_curr, base, sw, st, t_future, t_prev in cells:
        if len(sw) == 0 or len(st) == 0:
            continue
        dd = dpiv.loc[sw, t_curr].to_numpy(dtype=float) - float(base)
        ldiff = (ypiv.loc[sw, t_future] - ypiv.loc[sw, t_prev]).to_numpy(dtype=float)
        st_diff = ypiv.loc[st, t_future] - ypiv.loc[st, t_prev]
        if xpivs:
            th = theta.get(float(base))
            if th is None:
                continue
            ldiff = ldiff - _control_diffs(xpivs, sw, t_future, t_prev) @ th
            st_diff = st_diff - _control_diffs(xpivs, st, t_future, t_prev) @ th
        if wpiv is None:
            ldt = float(st_diff.mean())
            abs_change = float(np.abs(dd).sum())
            estimates.append(float(np.sum(np.sign(dd) * (ldiff - ldt))) / abs_change)
        else:
            # cells weigh their size in the switching period t (R, black
            # box; Stata's first-time-switcher DID_l weighs t + l instead)
            w_sw = wpiv.loc[sw, t_curr].to_numpy(dtype=float)
            w_st = wpiv.loc[st, t_curr].to_numpy(dtype=float)
            ldt = _wmean(st_diff.to_numpy(dtype=float), w_st)
            abs_change = float(np.sum(w_sw * np.abs(dd)))
            estimates.append(
                float(np.sum(w_sw * np.sign(dd) * (ldiff - ldt))) / abs_change
            )
        weights.append(abs_change)

    if not estimates:
        # No switching cell with a comparison cell: not estimable. Zero
        # would read as an estimated null effect.
        return {"estimate": float("nan"), "n_cells": 0}

    total_w = sum(weights)
    weighted_est = sum(e * w / total_w for e, w in zip(estimates, weights))

    return {"estimate": float(weighted_est), "n_cells": len(estimates)}


# ======================================================================
# Joint inference
# ======================================================================


def _joint_placebo_test(
    placebo_results: List[Dict[str, Any]],
    boot_placebo: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    """Joint Wald test of H0: all placebo lags equal zero.

    Uses the bootstrap draws (cluster-level) to estimate the covariance
    across lags, so the test accounts for correlation between consecutive
    pre-periods.  Follows dCDH (2024, §3.3) "joint placebo" recommendation.
    """
    if not placebo_results or boot_placebo is None:
        return None

    est = np.array([r["estimate"] for r in placebo_results], dtype=float)
    boot = np.asarray(boot_placebo, dtype=float)  # (n_boot, k)
    # Drop all-NaN columns / rows before covariance.
    mask_cols = ~np.all(np.isnan(boot), axis=0)
    if not mask_cols.any():
        return None
    est = est[mask_cols]
    boot = boot[:, mask_cols]
    valid_rows = ~np.any(np.isnan(boot), axis=1)
    if valid_rows.sum() < boot.shape[1] + 1:
        return None
    boot = boot[valid_rows]

    cov = np.cov(boot, rowvar=False, ddof=1)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    k = est.shape[0]
    cov_reg = cov + np.eye(k) * 1e-10

    try:
        W = float(est @ np.linalg.solve(cov_reg, est))
    except np.linalg.LinAlgError:
        W = float(est @ np.linalg.pinv(cov_reg) @ est)

    pval = float(stats.chi2.sf(W, k))
    return {"statistic": W, "df": int(k), "pvalue": pval}


def _avg_cumulative_effect(
    dynamic_results: List[Dict[str, Any]],
    boot_dynamic: Optional[np.ndarray],
    alpha: float,
) -> Optional[Dict[str, Any]]:
    """Average cumulative dynamic effect: mean of dynamic[0..L].

    Bootstrap SE is computed over the per-draw average across horizons,
    preserving the cross-horizon covariance (dCDH 2024 §3.4).
    """
    if not dynamic_results or boot_dynamic is None:
        return None
    est_vec = np.array([r["estimate"] for r in dynamic_results], dtype=float)
    if not np.isfinite(est_vec).any():
        return None
    avg_est = float(np.nanmean(est_vec))

    boot = np.asarray(boot_dynamic, dtype=float)
    per_draw_avg = np.nanmean(boot, axis=1)
    per_draw_avg = per_draw_avg[np.isfinite(per_draw_avg)]
    if per_draw_avg.size < 2:
        return {
            "estimate": avg_est,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
            "n_horizons": int(np.isfinite(est_vec).sum()),
        }
    se = float(np.std(per_draw_avg, ddof=1))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z = avg_est / se if se > 0 else 0.0
    return {
        "estimate": avg_est,
        "se": se,
        "ci_lower": avg_est - z_crit * se,
        "ci_upper": avg_est + z_crit * se,
        "pvalue": float(2 * stats.norm.sf(abs(z))),
        "n_horizons": int(np.isfinite(est_vec).sum()),
    }
