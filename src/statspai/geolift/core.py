"""Geo-lift: synthetic-control measurement for geo experiments.

A geo experiment switches an intervention on in a set of treated markets
(geographies) and leaves the rest as controls. ``geolift`` aggregates the
treated markets into a single treated series and builds a synthetic
counterfactual from the untreated markets via :func:`statspai.synth.synth`, so
the estimate, inference, and the observed-vs-counterfactual plot all come from
the tested synthetic-control machinery. The result is the unified
:class:`~statspai.core.results.CausalResult`, so ``sp.counterfactual_plot`` and
``sp.effect_summary`` work out of the box.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..core.results import CausalResult

__all__ = ["geolift"]

_TREATED_LABEL = "_treated_geos"


def geolift(
    data: pd.DataFrame,
    outcome: str,
    geo: str,
    time: str,
    treated_geos: Any,
    treatment_time: Any,
    *,
    agg: str = "mean",
    method: str = "classic",
    alpha: float = 0.05,
    **synth_kwargs: Any,
) -> CausalResult:
    """Estimate geo-experiment lift via synthetic control.

    Parameters
    ----------
    data : pandas.DataFrame
        Long panel: one row per (geo, time).
    outcome : str
        Outcome / KPI column (e.g. sales, conversions).
    geo : str
        Geography (market) id column.
    time : str
        Time column.
    treated_geos : scalar or sequence
        The geography (or geographies) the intervention was switched on in.
    treatment_time : any
        First treated period (inclusive). Earlier periods train the synthetic
        control.
    agg : {'mean', 'sum'}, default 'mean'
        How to aggregate multiple treated geos into one treated series.
    method : str, default 'classic'
        Synthetic-control method passed to :func:`sp.synth` (e.g. ``'classic'``,
        ``'augmented'``, ``'sdid'``).
    alpha : float, default 0.05
        Significance level forwarded to :func:`sp.synth`.
    **synth_kwargs
        Extra keyword arguments forwarded to :func:`sp.synth` (e.g.
        ``covariates``, ``inference``).

    Returns
    -------
    CausalResult
        ``estimand='ATT'`` — the average post-period lift of the treated
        markets. ``model_info`` adds ``treated_geos``, ``agg``, and a
        ``relative_lift_pct`` (lift as a share of the synthetic counterfactual).

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> res = sp.geolift(  # doctest: +SKIP
    ...     df, outcome='sales', geo='dma', time='week',
    ...     treated_geos=['NYC', 'LA'], treatment_time=40,
    ... )
    >>> res.estimate, res.model_info['relative_lift_pct']  # doctest: +SKIP
    """
    if agg not in ("mean", "sum"):
        raise ValueError(f"agg must be 'mean' or 'sum'; got {agg!r}.")
    for c in (outcome, geo, time):
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")

    if np.isscalar(treated_geos) or isinstance(treated_geos, str):
        treated_list: List[Any] = [treated_geos]
    else:
        treated_list = list(treated_geos)
    if not treated_list:
        raise ValueError("treated_geos must name at least one geography.")

    all_geos = set(pd.unique(data[geo]))
    missing = [g for g in treated_list if g not in all_geos]
    if missing:
        raise ValueError(f"treated_geos not found in data: {missing}.")
    control_geos = [g for g in all_geos if g not in set(treated_list)]
    if len(control_geos) < 2:
        raise ValueError(
            f"Need >= 2 control geographies; got {len(control_geos)}. "
            "Geo-lift builds the counterfactual from untreated markets."
        )
    if _TREATED_LABEL in all_geos:
        raise ValueError(
            f"data already contains a geo named {_TREATED_LABEL!r}; rename it."
        )

    # Aggregate treated geos into one treated series.
    treated_rows = data[data[geo].isin(treated_list)]
    agg_series = treated_rows.groupby(time)[outcome].agg(agg).reset_index()
    agg_series[geo] = _TREATED_LABEL

    control_rows = data[data[geo].isin(control_geos)][[geo, time, outcome]]
    panel = pd.concat(
        [control_rows, agg_series[[geo, time, outcome]]], ignore_index=True
    )

    from ..synth import synth

    result = synth(
        panel,
        outcome=outcome,
        unit=geo,
        time=time,
        treated_unit=_TREATED_LABEL,
        treatment_time=treatment_time,
        method=method,
        placebo=synth_kwargs.pop("placebo", False),
        alpha=alpha,
        **synth_kwargs,
    )

    # Re-frame as geo-lift and add a relative-lift summary.
    rel_pct = _relative_lift_pct(result)
    result.method = f"Geo-lift (synthetic control, method={method})"
    if isinstance(result.model_info, dict):
        result.model_info.update(
            {
                "design": "geo-lift",
                "treated_geos": treated_list,
                "n_treated_geos": len(treated_list),
                "n_control_geos": len(control_geos),
                "agg": agg,
                "relative_lift_pct": rel_pct,
            }
        )
    return result


def _relative_lift_pct(result: CausalResult) -> Optional[float]:
    """Post-period lift as a percentage of the synthetic counterfactual."""
    try:
        from ..plots.counterfactual import counterfactual_data

        data = counterfactual_data(result)
        post = data[data["post"]] if "post" in data.columns else data
        cf_mean = float(post["counterfactual"].mean())
        eff_mean = float(post["point_effect"].mean())
        if not np.isfinite(cf_mean) or abs(cf_mean) < 1e-12:
            return None
        return 100.0 * eff_mean / cf_mean
    except (TypeError, KeyError, ValueError, ZeroDivisionError):
        # Secondary display metric only; the ATT estimate is unaffected.
        return None
