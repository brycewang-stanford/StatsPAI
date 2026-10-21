"""
Goodman-Bacon (2021) decomposition for two-way fixed effects DID.

Decomposes the standard TWFE DID estimator into a weighted average of
all possible 2×2 DID comparisons, revealing which comparisons drive
the estimate and how much weight comes from "forbidden" comparisons
that use already-treated units as controls.

This is a **diagnostic** tool — it does not provide a bias-corrected
estimator. Use Callaway-Sant'Anna or Sun-Abraham for estimation.

References
----------
Goodman-Bacon, A. (2021).
"Difference-in-Differences with Variation in Treatment Timing."
*Journal of Econometrics*, 225(2), 254-277. [@goodmanbacon2021difference]

Goodman-Bacon, A., Goldring, T. and Nichols, A. (2019).
"BACONDECOMP: Stata module to perform the Bacon decomposition
of difference-in-differences estimation."
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._dict_result import DictResult
from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._core import calendar_time_aware as _calendar_time_aware


class BaconDecompositionResult(DictResult):
    """Result of :func:`bacon_decomposition`: a ``dict`` (keys unchanged)
    with ``summary()`` / ``tidy()`` / ``to_latex()`` ... ``estimate`` is the
    TWFE coefficient being decomposed."""

    _title = "Goodman-Bacon decomposition of the TWFE estimate"
    _citation_keys = ("goodmanbacon2021difference",)

    @property
    def estimate(self) -> float:
        return float(self["beta_twfe"])


def _summary_by_type(decomp: pd.DataFrame) -> pd.DataFrame:
    """Total weight and weight-averaged estimate of each comparison type."""
    columns = ["type", "n_comparisons", "weight", "estimate", "comparison_class"]
    if len(decomp) == 0:
        return pd.DataFrame(columns=columns)
    labelled = _label_comparisons(decomp)
    rows = []
    for kind, part in labelled.groupby("type", sort=False):
        weight = float(part["weight"].sum())
        # an unidentified 2x2 (NaN estimate, balance='observed_rows') keeps
        # its weight in the type total but not in the type's average
        known = np.isfinite(part["estimate"].to_numpy(dtype=float))
        w_known = float(part.loc[known, "weight"].sum())
        average = (
            float((part.loc[known, "weight"] * part.loc[known, "estimate"]).sum())
            / w_known
            if w_known != 0
            else float("nan")
        )
        classes = set(part["comparison_class"]) - {"covariates_only"}
        label = (
            classes.pop()
            if len(classes) == 1
            else ("mixed" if classes else "covariates_only")
        )
        rows.append((kind, len(part), weight, average, label))
    return pd.DataFrame(rows, columns=columns)


@accepts_aliases(_strict=True, unit="id", i="id", controls="covariates")
@_calendar_time_aware(time="time")
def bacon_decomposition(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: str,
    alpha: float = 0.05,
    balance: str = "error",
    covariates: Optional[List[str]] = None,
    weights: Optional[str] = None,
) -> "BaconDecompositionResult":
    """
    Goodman-Bacon (2021) decomposition of the TWFE DID estimator.

    Decomposes the overall TWFE coefficient into a weighted sum of
    2×2 DID comparisons between different treatment timing groups.

    Parameters
    ----------
    data : pd.DataFrame
        Balanced panel data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment indicator (0 before treatment, 1 after).
    time : str
        Time period variable.
    id : str
        Unit identifier.
    alpha : float, default 0.05
        Significance level.
    balance : {'error', 'drop_units', 'observed_rows'}, default 'error'
        What to do with an unbalanced panel. The decomposition theorem holds
        for a balanced panel only, so by default an unbalanced one is an
        error (as in R ``bacondecomp`` and Stata ``bacondecomp``).
        ``'drop_units'`` keeps the units with a non-missing outcome and
        treatment in every period -- Stata ``xtbalance`` before
        ``bacondecomp`` -- and decomposes the TWFE estimate on that balanced
        subpanel; the dropped units are listed in the output.
        ``'observed_rows'`` applies Theorem 1 to the observed rows of the
        unbalanced panel, the rule of Stata ``ddtiming``: group shares and
        treated shares are shares of observed rows, each 2x2 is the TWFE
        coefficient on the observed rows of its window, and a unit treated
        in its first observed period is an already-treated control. Off a
        balanced panel this is not a decomposition: ``weighted_sum`` (what
        ``ddtiming`` reports as the DD estimate) differs from ``beta_twfe``,
        weights can be negative, and a 2x2 with no treatment variation left
        after its fixed effects has ``estimate`` NaN and ``identified``
        False (``ddtiming`` reports 0 and keeps its weight). It warns with
        the gap. Not with ``covariates`` or ``weights``. On a balanced panel
        it equals the default.

        .. versionadded:: 1.41.0
    covariates : list of str, optional
        Time-varying controls (R ``bacon(y ~ d + x1 + x2, ...)``). The TWFE
        coefficient with controls splits into a within-timing-group part
        and a weighted sum of between-group 2x2 terms:
        ``beta = omega * beta_within + (1 - omega) * sum(weight * estimate)``
        (Goodman-Bacon 2021, Section 5). Pairs are earlier vs later timing
        groups only (each pair once), typed ``'Treated vs Untreated'``,
        ``'Both Treated'`` or ``'Later vs Always Treated'`` as in R
        ``bacondecomp``; the result adds ``omega``, ``beta_within`` and
        ``beta_between``.
    weights : str, optional
        Column of sampling (analytic) weights, e.g. state population. The
        weight must be positive and constant within each unit. The
        decomposition is then that of the weighted TWFE coefficient
        (``sp.feols(..., weights=)``, Stata ``bacondecomp ... [aw=w]``):
        timing-group shares become weight shares and every 2x2 estimate
        and its weight become the weighted ones (see Notes). Works with
        and without ``covariates``.

    Returns
    -------
    BaconDecompositionResult
        A ``dict`` (the keys below, unchanged) that also has
        ``summary()`` / ``glance()`` / ``to_latex()`` / ``to_word()``.
        Keys:
        - ``beta_twfe``: overall TWFE estimate
        - ``decomposition``: pd.DataFrame with columns
          [type, treated, control, estimate, weight, comparison_class,
          is_forbidden]. ``comparison_class`` is ``'clean'`` (treated vs
          never treated, earlier vs later treated), ``'forbidden'``
          (already-treated units as the control: later vs earlier
          treated, later vs always treated) or, with ``covariates``,
          ``'mixed'`` for a ``'Both Treated'`` pair (it pools the clean
          earlier-vs-later and the forbidden later-vs-earlier 2x2) and
          ``'covariates_only'`` for the always-treated vs never-treated
          pair, which has no 2x2 variation. ``is_forbidden`` is True for
          ``'forbidden'`` rows only.
        - ``summary``: pd.DataFrame with one row per comparison type
          [type, n_comparisons, weight, estimate, comparison_class]: the
          total weight of the type and the weight-averaged 2×2 estimate
          within it, the summary rows of Stata ``estat bdecomp`` and
          ``bacondecomp``
        - ``weighted_sum``: Σ(weight × estimate) — equals beta_twfe
          under the same dyad conventions as R ``bacondecomp`` and Stata
          ``bacondecomp``
        - ``n_comparisons``: number of 2×2 sub-comparisons
        - ``negative_weight_share``: fraction of signed weight mass on
          truly negative Bacon weights. This is usually zero; already-treated
          control comparisons are reported separately as
          ``already_treated_control_weight_share``.
        - ``n_units``: units decomposed; ``dropped_units``: the units
          ``balance='drop_units'`` removed (empty otherwise).
        - ``weights``: the weight column, or None.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> cohorts = {1: 4, 2: 4, 3: 7, 4: 7, 5: 99, 6: 99}  # 99 = never treated
    >>> rows = []
    >>> for unit, g in cohorts.items():
    ...     for year in range(1, 11):
    ...         treated = int(year >= g)
    ...         y = unit + 0.3 * year + 2.0 * treated + rng.normal(0, 0.5)
    ...         rows.append({'unit': unit, 'year': year,
    ...                      'outcome': y, 'treated': treated})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.bacon_decomposition(df, y='outcome', treat='treated',
    ...                                 time='year', id='unit')
    >>> sorted(result['decomposition']['type'].unique())
    ['Earlier vs Later Treated', 'Later vs Earlier Treated', 'Treated vs Untreated']
    >>> bool(abs(result['weighted_sum'] - result['beta_twfe']) < 1e-8)
    True

    Notes
    -----
    The decomposition identifies three types of comparisons:

    1. **Earlier vs Later treated**: Units treated at time g₁ vs units
       treated later at g₂ (g₁ < g₂). These are "good" comparisons.
    2. **Later vs Earlier treated**: Units treated at g₂ vs already-treated
       units at g₁. These are "forbidden" — they use treated units as
       controls and can introduce negative weighting bias.
    3. **Treated vs Never treated**: Always valid comparisons.

    A large ``already_treated_control_weight_share`` signals that TWFE is
    relying heavily on comparisons with already-treated controls, so a
    heterogeneity-robust estimator (C&S, Sun-Abraham) should be used.

    **Weights.** With a weight ``w_i`` that is constant within unit ``i``
    on a balanced panel, weighted two-way demeaning is exact:
    ``x_it - xbar_i - xbar^w_t + xbar^w``, where ``xbar_i`` is the unit's
    time mean and ``xbar^w_t`` / ``xbar^w`` are ``w``-weighted means
    across units (the residual is orthogonal to the unit and the period
    dummies in the ``w``-weighted inner product). So the weighted TWFE
    coefficient is ``sum w d~ y~ / sum w d~^2`` and Theorem 1 goes through
    with every group size ``n_k`` replaced by the group's total weight,
    every group mean by the weighted mean, and every 2x2 estimate by the
    weighted 2x2 on the same window. An integer weight is the same as
    repeating the unit that many times, so the weighted table equals the
    unweighted table of the replicated panel; the identity
    ``beta_twfe = sum(weight * estimate)`` holds exactly. With
    ``covariates`` the same replacement applies to every mean, variance
    and least-squares fit of the Section 5 decomposition (the
    Frisch-Waugh-Lovell step is a weighted projection), and
    ``beta_twfe = omega * beta_within + (1 - omega) * beta_between``
    still holds exactly. ``tests/test_bacon_weights.py`` checks both
    identities against ``sp.feols(..., weights=)`` and the replication
    equivalence.

    See Goodman-Bacon (2021, *JEcon*), Theorem 1.
    """
    check_columns(
        data,
        {"y": y, "treat": treat, "time": time, "id": id, "weights": weights},
        function="sp.bacon_decomposition",
    )
    df = data.copy()
    # float32 columns (a .dta float) would keep the means and sums below in
    # single precision: the identity beta_twfe = sum(weight * estimate)
    # then failed by 2e-7 on Goodman-Bacon's divorce data.
    for col in (y, treat, weights):
        if col is not None and pd.api.types.is_float_dtype(df[col]):
            if df[col].dtype != np.float64:
                df[col] = df[col].astype(np.float64)
    if balance not in ("error", "drop_units", "observed_rows"):
        raise MethodIncompatibility(
            f"balance={balance!r}; use 'error', 'drop_units' or 'observed_rows'.",
            recovery_hint="'drop_units' decomposes the balanced subpanel; "
            "'observed_rows' reproduces Stata ddtiming on the unbalanced one.",
            diagnostics={"balance": balance},
        )
    if balance == "observed_rows":
        if covariates or weights is not None:
            raise MethodIncompatibility(
                "balance='observed_rows' is defined without covariates and "
                "weights (Stata ddtiming has neither).",
                recovery_hint="Use balance='drop_units' for a decomposition "
                "with covariates or weights.",
                diagnostics={
                    "covariates": list(covariates or []),
                    "weights": weights,
                },
            )
        return _observed_rows_result(df, y, treat, time, id)
    dropped_units: list = []
    if balance == "drop_units":
        df = df.dropna(subset=[y, treat, time, id] + ([weights] if weights else []))
        n_periods = df[time].nunique()
        per_unit = df.groupby(id)[time].nunique()
        dup = df.duplicated(subset=[id, time])
        if dup.any():
            raise MethodIncompatibility(
                "Duplicate (unit, time) rows.",
                recovery_hint="Aggregate or drop duplicates before decomposing.",
                diagnostics={"n_duplicates": int(dup.sum())},
            )
        dropped_units = sorted(per_unit.index[per_unit < n_periods].tolist())
        df = df[~df[id].isin(dropped_units)]
        if df[id].nunique() == 0:
            raise DataInsufficient(
                "No unit is observed in every period.",
                recovery_hint="Restrict the time window so some units span it.",
                diagnostics={"n_periods": int(n_periods)},
            )

    # Validate balanced panel using the same requirement as bacondecomp:
    # each unit must contribute the same number of time observations.
    counts = df.groupby(id)[time].nunique()
    if counts.nunique() != 1:
        raise MethodIncompatibility(
            "Unbalanced Panel",
            recovery_hint=(
                "Goodman-Bacon decomposition requires the same number of "
                "time observations per unit; pass balance='drop_units' to "
                "decompose the balanced subpanel, or use a "
                "heterogeneity-robust DID estimator directly."
            ),
            diagnostics={"panel_balance_counts": counts.value_counts().to_dict()},
            alternative_functions=["sp.callaway_santanna", "sp.sun_abraham"],
        )

    if weights is not None:
        _validate_unit_weights(df, weights, id)

    time_periods = sorted(df[time].unique())
    time_min = min(time_periods)

    # Get treatment timing for each unit
    # g_i = first period where treat == 1 (inf for never-treated)
    unit_treat = df.groupby(id)[[treat, time]].apply(
        lambda grp: (
            grp.loc[grp[treat] == 1, time].min() if (grp[treat] == 1).any() else np.inf
        )
    )

    _validate_monotone_treatment(df, treat, time, id, unit_treat)
    df = df.merge(unit_treat.rename("_treat_time"), left_on=id, right_index=True)

    # Identify groups by treatment timing
    timing_groups = sorted(unit_treat.unique())
    never_treated = np.inf

    if covariates:
        return _decompose_with_controls(
            df, y, treat, time, id, list(covariates), dropped_units, weights
        )

    # Overall TWFE estimate (for reference)
    beta_twfe = _twfe_estimate(df, y, treat, time, id, weights)

    # Enumerate all 2×2 comparisons
    comparisons = []

    for untreated_group in timing_groups:
        for treated_group in timing_groups:
            if treated_group == untreated_group:
                continue
            if treated_group == never_treated:
                continue
            if treated_group == time_min:
                continue

            data_pair = _subset_bacon_dyad(
                df, treated_group, untreated_group, time, "_treat_time"
            )
            if data_pair.empty:
                continue

            wt = _bacon_weight(
                data_pair,
                treat,
                "_treat_time",
                treated_group,
                untreated_group,
                weights,
            )
            if not np.isfinite(wt) or abs(wt) < 1e-15:
                continue

            est = _twfe_estimate(data_pair, y, treat, time, id, weights)

            if untreated_group == never_treated:
                comp_type = "Treated vs Untreated"
                control = "Never"
            elif untreated_group == time_min:
                comp_type = "Later vs Always Treated"
                control = untreated_group
            elif treated_group > untreated_group:
                comp_type = "Later vs Earlier Treated"
                control = untreated_group
            else:
                comp_type = "Earlier vs Later Treated"
                control = untreated_group

            comparisons.append(
                {
                    "type": comp_type,
                    "treated": treated_group,
                    "control": control,
                    "estimate": est,
                    "weight": wt,
                }
            )

    decomp = pd.DataFrame(comparisons)

    if len(decomp) == 0:
        return BaconDecompositionResult(
            {
                "beta_twfe": beta_twfe,
                "decomposition": _label_comparisons(decomp),
                "summary": _summary_by_type(decomp),
                "weighted_sum": 0.0,
                "n_comparisons": 0,
                "negative_weight_share": 0.0,
                "already_treated_control_weight_share": 0.0,
                "n_units": int(df[id].nunique()),
                "dropped_units": dropped_units,
                "weights": weights,
            }
        )

    # Normalize weights to sum to 1
    total_w = decomp["weight"].sum()
    if total_w > 0:
        decomp["weight"] = decomp["weight"] / total_w

    weighted_sum = float((decomp["estimate"] * decomp["weight"]).sum())

    abs_weight_sum = float(decomp["weight"].abs().sum())
    if abs_weight_sum > 0:
        neg_share = float(
            decomp.loc[decomp["weight"] < 0, "weight"].abs().sum() / abs_weight_sum
        )
    else:
        neg_share = 0.0

    forbidden = decomp["type"].isin(
        {
            "Later vs Earlier Treated",
            "Later vs Always Treated",
        }
    )
    forbidden_share = float(decomp.loc[forbidden, "weight"].sum())

    return BaconDecompositionResult(
        {
            "beta_twfe": beta_twfe,
            "decomposition": _label_comparisons(decomp),
            "summary": _summary_by_type(decomp),
            "weighted_sum": weighted_sum,
            "n_comparisons": len(decomp),
            "negative_weight_share": neg_share,
            "already_treated_control_weight_share": forbidden_share,
            "n_units": int(df[id].nunique()),
            "dropped_units": dropped_units,
            "weights": weights,
        }
    )


def _observed_rows_result(
    df: pd.DataFrame, y: str, treat: str, time: str, id_col: str
) -> "BaconDecompositionResult":
    """``balance='observed_rows'``: Theorem 1 on the observed rows (ddtiming)."""
    from ._bacon_unbalanced import decompose_observed_rows, timing_groups

    df = df.dropna(subset=[y, treat, time, id_col])
    if df.duplicated(subset=[id_col, time]).any():
        raise MethodIncompatibility(
            "Duplicate (unit, time) rows.",
            recovery_hint="Aggregate or drop duplicates before decomposing.",
            diagnostics={"n_duplicates": int(df.duplicated([id_col, time]).sum())},
        )
    groups = timing_groups(df, treat, time, id_col).replace(-np.inf, df[time].min())
    _validate_monotone_treatment(df, treat, time, id_col, groups)
    decomp, beta_twfe = decompose_observed_rows(df, y, treat, time, id_col)
    if len(decomp):
        total = float(decomp["weight"].sum())
        if total != 0:
            decomp["weight"] = decomp["weight"] / total
    known = decomp["identified"] if len(decomp) else pd.Series(dtype=bool)
    weighted_sum = float(
        (decomp.loc[known, "weight"] * decomp.loc[known, "estimate"]).sum()
    )
    abs_w = float(decomp["weight"].abs().sum()) if len(decomp) else 0.0
    neg_share = (
        float(decomp.loc[decomp["weight"] < 0, "weight"].abs().sum() / abs_w)
        if abs_w > 0
        else 0.0
    )
    forbidden = (
        float(decomp.loc[decomp["type"].isin(_FORBIDDEN_TYPES), "weight"].sum())
        if len(decomp)
        else 0.0
    )
    balanced = df.groupby(id_col)[time].nunique().nunique() == 1
    unidentified = decomp.loc[~known] if len(decomp) else decomp
    if not balanced or len(unidentified):
        msg = (
            "bacon_decomposition(balance='observed_rows'): the panel is "
            "unbalanced, so Theorem 1 is applied to the observed rows (Stata "
            f"ddtiming) and is not an identity: sum(weight x estimate) = "
            f"{weighted_sum:.6g}, the TWFE coefficient = {beta_twfe:.6g}."
        )
        if neg_share > 0:
            msg += f" {neg_share:.1%} of the absolute weight is on negative weights."
        if len(unidentified):
            msg += (
                f" {len(unidentified)} 2x2 comparison(s) have no treatment "
                "variation after their fixed effects (estimate NaN, "
                "identified=False; ddtiming reports 0 and keeps the weight): "
                + ", ".join(
                    f"{r.treated:g} vs {r.control}" for r in unidentified.itertuples()
                )
                + "."
            )
        warnings.warn(msg, UserWarning, stacklevel=4)
    return BaconDecompositionResult(
        {
            "beta_twfe": beta_twfe,
            "decomposition": _label_comparisons(decomp),
            "summary": _summary_by_type(decomp),
            "weighted_sum": weighted_sum,
            "n_comparisons": len(decomp),
            "negative_weight_share": neg_share,
            "already_treated_control_weight_share": forbidden,
            "n_units": int(df[id_col].nunique()),
            "dropped_units": [],
            "weights": None,
            "balance": "observed_rows",
        }
    )


def _decompose_with_controls(
    df: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: str,
    covariates: List[str],
    dropped_units: list,
    unit_weights: Optional[str] = None,
) -> "BaconDecompositionResult":
    from ._bacon_controls import bacon_with_controls, twfe_with_controls

    check_columns(df, {"covariates": covariates}, function="sp.bacon_decomposition")
    out = bacon_with_controls(
        df,
        y=y,
        treat=treat,
        time=time,
        id=id,
        covariates=covariates,
        treat_time="_treat_time",
        weights=unit_weights,
    )
    decomp = out["decomposition"]
    weights = decomp["weight"] if len(decomp) else pd.Series(dtype=float)
    always = decomp["type"].eq("Later vs Always Treated") if len(decomp) else weights
    abs_total = float(weights.abs().sum())
    return BaconDecompositionResult(
        {
            "beta_twfe": twfe_with_controls(
                df, y, treat, time, id, covariates, unit_weights
            ),
            "decomposition": _label_comparisons(decomp),
            "summary": _summary_by_type(decomp),
            "weighted_sum": out["weighted_sum"],
            "n_comparisons": len(decomp),
            "omega": out["omega"],
            "beta_within": out["beta_within"],
            "beta_between": out["beta_between"],
            "covariates": covariates,
            "negative_weight_share": (
                float(weights[weights < 0].abs().sum()) / abs_total
                if abs_total > 0
                else 0.0
            ),
            "already_treated_control_weight_share": (
                float(weights[always].sum()) if len(decomp) else 0.0
            ),
            "n_units": int(df[id].nunique()),
            "dropped_units": dropped_units,
            "weights": unit_weights,
        }
    )


def _twfe_estimate(
    df: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id_col: str,
    weights: Optional[str] = None,
) -> float:
    """Standard TWFE DID regression: Y_it = α_i + γ_t + β·D_it + ε_it."""
    if weights is not None:
        return _weighted_twfe_estimate(df, y, treat, time, id_col, weights)
    # Demean by unit and time (within transformation)
    panel = df.set_index([id_col, time])
    Y = panel[y].unstack()
    D = panel[treat].unstack()

    # Double-demean
    y_dm = (
        Y.values
        - Y.values.mean(axis=1, keepdims=True)
        - Y.values.mean(axis=0, keepdims=True)
        + Y.values.mean()
    )
    d_dm = (
        D.values
        - D.values.mean(axis=1, keepdims=True)
        - D.values.mean(axis=0, keepdims=True)
        + D.values.mean()
    )

    y_flat = y_dm.ravel()
    d_flat = d_dm.ravel()
    valid = np.isfinite(y_flat) & np.isfinite(d_flat)

    if valid.sum() < 2 or np.var(d_flat[valid]) < 1e-12:
        return 0.0

    beta = float(np.sum(d_flat[valid] * y_flat[valid]) / np.sum(d_flat[valid] ** 2))
    return beta


def _weighted_twfe_estimate(
    df: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id_col: str,
    weights: str,
) -> float:
    """Weighted TWFE coefficient, unit-constant weights, balanced panel.

    ``x_it - xbar_i - xbar^w_t + xbar^w`` is the exact residual of the
    ``w``-weighted projection on unit and period dummies when ``w`` is
    constant within unit: ``xbar_i`` is the unit's plain time mean and the
    period and grand means are ``w``-weighted across units.
    """
    panel = df.set_index([id_col, time])
    Y = panel[y].unstack().to_numpy(dtype=float)
    D = panel[treat].unstack().to_numpy(dtype=float)
    w = panel[weights].unstack().to_numpy(dtype=float)[:, 0]
    sw = w.sum()

    def demean(M: np.ndarray) -> np.ndarray:
        row = M.mean(axis=1, keepdims=True)
        col = (w[:, None] * M).sum(axis=0, keepdims=True) / sw
        return M - row - col + float(w @ row[:, 0]) / sw

    y_dm = demean(Y)
    d_dm = demean(D)
    den = float(np.sum(w[:, None] * d_dm**2))
    if den < 1e-12 * sw * D.shape[1]:
        return 0.0
    return float(np.sum(w[:, None] * d_dm * y_dm) / den)


def _validate_unit_weights(df: pd.DataFrame, weights: str, id_col: str) -> None:
    """Weights must be positive, finite and constant within unit."""
    if not pd.api.types.is_numeric_dtype(df[weights]):
        raise MethodIncompatibility(
            f"bacon_decomposition: weights column {weights!r} is not numeric.",
            recovery_hint="Pass a numeric column of positive weights.",
            diagnostics={"weights": weights, "dtype": str(df[weights].dtype)},
        )
    w = df[weights].to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise MethodIncompatibility(
            f"bacon_decomposition: weights column {weights!r} must be "
            "positive and finite in every row.",
            recovery_hint=(
                "Drop the units with a zero, negative or missing weight "
                "before decomposing; a zero weight removes the unit from the "
                "weighted TWFE regression."
            ),
            diagnostics={
                "n_nonpositive": int((w <= 0).sum()),
                "n_nonfinite": int((~np.isfinite(w)).sum()),
            },
        )
    spread = df.groupby(id_col)[weights].agg(lambda v: float(v.max() - v.min()))
    scale = df.groupby(id_col)[weights].agg(lambda v: float(v.abs().max()))
    varying = spread.index[spread > 1e-12 * scale]
    if len(varying):
        raise MethodIncompatibility(
            f"bacon_decomposition: weights column {weights!r} varies within "
            f"{len(varying)} unit(s), e.g. {varying[0]!r}. The decomposition "
            "holds for a weight that is constant within each unit (weighted "
            "two-way demeaning is exact only then).",
            recovery_hint=(
                "Fix each unit's weight at one value, e.g. its first-period "
                "value: df['w0'] = df.sort_values(time).groupby(id)[w]"
                ".transform('first'), and pass weights='w0'."
            ),
            diagnostics={
                "n_units_varying": int(len(varying)),
                "example_unit": varying[0],
            },
        )


#: The comparisons that use already-treated units as the control group.
_FORBIDDEN_TYPES = frozenset({"Later vs Earlier Treated", "Later vs Always Treated"})


def _comparison_class(row: pd.Series) -> str:
    kind = row["type"]
    if kind in _FORBIDDEN_TYPES:
        return "forbidden"
    if kind == "Both Treated":
        return "mixed"
    if "beta_2x2" in row.index and not np.isfinite(row["beta_2x2"]):
        return "covariates_only"
    return "clean"


def _label_comparisons(decomp: pd.DataFrame) -> pd.DataFrame:
    """Add ``comparison_class`` and ``is_forbidden`` to the 2x2 table."""
    decomp = decomp.copy()
    if len(decomp) == 0:
        decomp["comparison_class"] = pd.Series(dtype=object)
        decomp["is_forbidden"] = pd.Series(dtype=bool)
        return decomp
    decomp["comparison_class"] = decomp.apply(_comparison_class, axis=1)
    decomp["is_forbidden"] = decomp["comparison_class"].eq("forbidden")
    return decomp


def _validate_monotone_treatment(
    df: pd.DataFrame,
    treat: str,
    time: str,
    id_col: str,
    unit_treat: pd.Series,
) -> None:
    """Require absorbing treatment timing."""
    merged = df.merge(
        unit_treat.rename("_treat_time"),
        left_on=id_col,
        right_index=True,
    )
    expected = np.where(
        np.isinf(merged["_treat_time"].to_numpy(dtype=float)),
        0,
        (merged[time].to_numpy() >= merged["_treat_time"].to_numpy()).astype(int),
    )
    observed = merged[treat].to_numpy()
    if not np.array_equal(observed.astype(int), expected.astype(int)):
        raise MethodIncompatibility(
            "Treatment not weakly increasing with time",
            recovery_hint=(
                "Pass an absorbing 0/1 treatment-status column to "
                "bacon_decomposition, not a cohort/first-treatment column."
            ),
            diagnostics={"treatment_column": treat},
            alternative_functions=["sp.did_analysis", "sp.callaway_santanna"],
        )


def _subset_bacon_dyad(
    df: pd.DataFrame,
    treated_group: float,
    untreated_group: float,
    time: str,
    treat_time: str,
) -> pd.DataFrame:
    """Subset one bacondecomp dyad and its admissible comparison window."""
    data_pair = df[df[treat_time].isin([treated_group, untreated_group])].copy()
    if treated_group < untreated_group:
        data_pair = data_pair[data_pair[time] < untreated_group]
    elif treated_group > untreated_group:
        data_pair = data_pair[data_pair[time] >= untreated_group]
    return data_pair


def _bacon_weight(
    data_pair: pd.DataFrame,
    treat: str,
    treat_time: str,
    treated_group: float,
    untreated_group: float,
    weights: Optional[str] = None,
) -> float:
    """Uncontrolled Goodman-Bacon weight used by R/Stata bacondecomp.

    With ``weights`` the group sizes are the groups' total weights (the
    shares of Theorem 1 become weight shares); the treatment shares
    ``D_k`` are unchanged, since every unit of a timing group has the same
    treatment path.
    """

    def size(group: float) -> float:
        rows = data_pair[treat_time] == group
        if weights is None:
            return float(rows.sum())
        return float(data_pair.loc[rows, weights].sum())

    if untreated_group == np.inf:
        n_u = size(untreated_group)
        n_k = size(treated_group)
        if n_k + n_u == 0:
            return 0.0
        n_ku = n_k / (n_k + n_u)
        D_k = float(data_pair.loc[data_pair[treat_time] == treated_group, treat].mean())
        V_ku = n_ku * (1 - n_ku) * D_k * (1 - D_k)
        return float((n_k + n_u) ** 2 * V_ku)

    if treated_group < untreated_group:
        n_k = size(treated_group)
        n_l = size(untreated_group)
        if n_k + n_l == 0:
            return 0.0
        n_kl = n_k / (n_k + n_l)
        D_k = float(data_pair.loc[data_pair[treat_time] == treated_group, treat].mean())
        D_l = float(
            data_pair.loc[data_pair[treat_time] == untreated_group, treat].mean()
        )
        denom = 1 - D_l
        if abs(denom) < 1e-15:
            return 0.0
        V_kl = n_kl * (1 - n_kl) * (D_k - D_l) / denom * (1 - D_k) / denom
        return float(((n_k + n_l) * denom) ** 2 * V_kl)

    n_k = size(untreated_group)
    n_l = size(treated_group)
    if n_k + n_l == 0:
        return 0.0
    n_kl = n_k / (n_k + n_l)
    D_k = float(data_pair.loc[data_pair[treat_time] == untreated_group, treat].mean())
    D_l = float(data_pair.loc[data_pair[treat_time] == treated_group, treat].mean())
    if abs(D_k) < 1e-15:
        return 0.0
    V_kl = n_kl * (1 - n_kl) * (D_l / D_k) * (D_k - D_l) / D_k
    return float(((n_k + n_l) * D_k) ** 2 * V_kl)


# Citation
CausalResult._CITATIONS["bacon_decomposition"] = (
    "@article{goodmanbacon2021difference,\n"
    "  title={Difference-in-Differences with Variation in Treatment "
    "Timing},\n"
    "  author={Goodman-Bacon, Andrew},\n"
    "  journal={Journal of Econometrics},\n"
    "  volume={225},\n"
    "  number={2},\n"
    "  pages={254--277},\n"
    "  year={2021},\n"
    "  publisher={Elsevier}\n"
    "}"
)
