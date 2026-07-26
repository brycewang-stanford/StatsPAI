"""Influence-function export and post-hoc aggregation for CS DiD.

The Stata ``csdid`` workflow ``saverif(myrif) → custom aggregation`` has
no refit cost because everything downstream of ATT(g, t) is a linear
combination of influence functions.  :func:`influence_functions` exports
the per-unit influence functions of a :func:`callaway_santanna` fit as a
tidy, self-contained DataFrame (optionally written to disk), and
:func:`aggte_from_influence` rebuilds the aggregation inputs from that
frame — so event-study / group / calendar aggregations (with multiplier
bootstrap and uniform bands) can be recomputed later, in another
process, without the original data.

Examples
--------
>>> import statspai as sp                                   # doctest: +SKIP
>>> cs = sp.callaway_santanna(df, y='y', g='g', t='t', i='i')  # doctest: +SKIP
>>> rif = sp.influence_functions(cs, path='cs_rif.csv')     # doctest: +SKIP
>>> # ... later / elsewhere ...
>>> es = sp.aggte_from_influence('cs_rif.csv', type='dynamic',
...                              min_e=-4, max_e=8)         # doctest: +SKIP

References
----------
Callaway, B. and Sant'Anna, P.H.C. (2021). "Difference-in-Differences
with Multiple Time Periods." *Journal of Econometrics*, 225(2),
200-230, Section 4.2. [@callaway2021difference]
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility
from .aggte import aggte

__all__ = ["influence_functions", "aggte_from_influence"]

# Columns a valid export must carry — enough to rebuild detail + IF matrix.
_REQUIRED_COLUMNS = (
    "unit",
    "unit_cohort",
    "group",
    "time",
    "relative_time",
    "att",
    "influence",
)


def influence_functions(
    result: CausalResult,
    path: Optional[Union[str, pathlib.Path]] = None,
) -> pd.DataFrame:
    """Export per-unit influence functions of a Callaway–Sant'Anna fit.

    Equivalent to Stata ``csdid, saverif()``: the returned frame is
    self-contained — it carries everything :func:`aggte_from_influence`
    needs to recompute any aggregation without refitting.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`statspai.callaway_santanna`.
    path : str or Path, optional
        If given, also write the frame to disk — ``.parquet`` via
        ``to_parquet``, anything else via ``to_csv(index=False)``.

    Returns
    -------
    pd.DataFrame
        Long format, one row per unit × (g, t) pair, columns:

        - ``unit`` — unit identifier (observation index for RCS fits)
        - ``unit_cohort`` — the unit's own first-treatment cohort
          (0 = never treated); used to rebuild aggregation weights
        - ``group``, ``time``, ``relative_time`` — the (g, t) cell
        - ``att`` — the cell's point estimate (repeated across units)
        - ``influence`` — the unit's influence-function value ψᵢ(g, t)
        - ``cluster`` — only when the fit used ``clustervars``

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=60, n_periods=6, staggered=True, seed=1)
    >>> df['first_treat'] = df['first_treat'].fillna(0)
    >>> cs = sp.callaway_santanna(df, y='y', g='first_treat', t='time',
    ...                           i='unit')
    >>> rif = sp.influence_functions(cs)
    >>> set(rif.columns) >= {'unit', 'group', 'time', 'influence'}
    True

    References
    ----------
    Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences
    with multiple time periods. *Journal of Econometrics*, 225(2),
    200-230. [@callaway2021difference]
    """
    inf_matrix = getattr(result, "_influence_funcs", None)
    detail = result.detail
    if inf_matrix is None or detail is None or len(detail) == 0:
        raise MethodIncompatibility(
            "result carries no influence functions — influence_functions() "
            "requires the output of sp.callaway_santanna().",
            recovery_hint="Fit with sp.callaway_santanna() first.",
            diagnostics={"method": getattr(result, "method", None)},
        )
    required = {"group", "time", "att", "relative_time"}
    if not required.issubset(detail.columns):
        raise MethodIncompatibility(
            "result.detail lacks the Callaway–Sant'Anna (g, t) grid "
            f"(columns {sorted(required)}); got {sorted(detail.columns)}.",
            recovery_hint="Pass a result produced by sp.callaway_santanna().",
            diagnostics={"method": getattr(result, "method", None)},
        )

    inf_matrix = np.asarray(inf_matrix, dtype=float)
    n_units, n_pairs = inf_matrix.shape
    if n_pairs != len(detail):
        raise MethodIncompatibility(
            f"influence matrix has {n_pairs} columns but detail has "
            f"{len(detail)} (g, t) rows — result was modified after fit.",
            recovery_hint="Re-fit with sp.callaway_santanna().",
            diagnostics={"n_pairs": n_pairs, "n_detail": len(detail)},
        )

    model_info = result.model_info or {}
    unit_ids = model_info.get("_unit_ids")
    if unit_ids is None:
        unit_ids = np.arange(n_units)
    unit_cohorts = model_info.get("_unit_cohorts")
    if unit_cohorts is None:
        unit_cohorts = np.full(n_units, np.nan)
    cluster_ids = model_info.get("_cluster_ids")

    out = pd.DataFrame(
        {
            "unit": np.tile(np.asarray(unit_ids), n_pairs),
            "unit_cohort": np.tile(np.asarray(unit_cohorts), n_pairs),
            "group": np.repeat(detail["group"].values, n_units),
            "time": np.repeat(detail["time"].values, n_units),
            "relative_time": np.repeat(detail["relative_time"].values, n_units),
            "att": np.repeat(detail["att"].values, n_units),
            "influence": inf_matrix.reshape(-1, order="F"),
        }
    )
    if cluster_ids is not None:
        out["cluster"] = np.tile(np.asarray(cluster_ids), n_pairs)

    if path is not None:
        path = pathlib.Path(path)
        if path.suffix == ".parquet":
            out.to_parquet(path, index=False)
        else:
            out.to_csv(path, index=False)

    return out


def aggte_from_influence(
    source: Union[pd.DataFrame, str, pathlib.Path],
    type: str = "simple",
    **aggte_kwargs: Any,
) -> CausalResult:
    """Aggregate group-time ATTs from exported influence functions.

    The post-hoc half of the Stata ``csdid saverif()`` workflow: rebuild
    the ATT(g, t) grid and influence-function matrix from a frame written
    by :func:`influence_functions` and run :func:`statspai.aggte` on it —
    no refit, no original data needed.

    Parameters
    ----------
    source : DataFrame, str, or Path
        A frame produced by :func:`influence_functions`, or a path to one
        (``.parquet`` or CSV).
    type : {'simple', 'dynamic', 'group', 'calendar'}, default 'simple'
        Aggregation scheme, forwarded to :func:`statspai.aggte`.
    **aggte_kwargs
        Any other :func:`statspai.aggte` options — ``min_e`` / ``max_e``,
        ``balance_e``, ``bstrap``, ``n_boot``, ``cband``, ``alpha``,
        ``random_state``.

    Returns
    -------
    CausalResult
        Same shape as ``sp.aggte`` output.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=60, n_periods=6, staggered=True, seed=1)
    >>> df['first_treat'] = df['first_treat'].fillna(0)
    >>> cs = sp.callaway_santanna(df, y='y', g='first_treat', t='time',
    ...                           i='unit')
    >>> rif = sp.influence_functions(cs)
    >>> es = sp.aggte_from_influence(rif, type='dynamic', n_boot=200,
    ...                              random_state=0)
    >>> es.estimand
    'ATT'

    References
    ----------
    Callaway, B. and Sant'Anna, P. H. C. (2021). Difference-in-differences
    with multiple time periods. *Journal of Econometrics*, 225(2),
    200-230. [@callaway2021difference]
    """
    if isinstance(source, (str, pathlib.Path)):
        p = pathlib.Path(source)
        if p.suffix == ".parquet":
            frame = pd.read_parquet(p)
        else:
            frame = pd.read_csv(p)
    elif isinstance(source, pd.DataFrame):
        frame = source
    else:
        raise MethodIncompatibility(
            "source must be a DataFrame or a file path, got "
            f"{source.__class__.__name__}.",
            recovery_hint="Pass the output of sp.influence_functions().",
            diagnostics={"source_type": source.__class__.__name__},
        )

    missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise MethodIncompatibility(
            f"influence frame is missing required column(s) {missing} — "
            "was it produced by sp.influence_functions()?",
            recovery_hint="Export with sp.influence_functions(result, path).",
            diagnostics={
                "missing": missing,
                "columns": list(frame.columns),
            },
        )

    # Rebuild the (g, t) grid in a stable order.
    pair_cols = ["group", "time", "relative_time", "att"]
    pairs = (
        frame[pair_cols]
        .drop_duplicates(subset=["group", "time"])
        .sort_values(["group", "time"])
        .reset_index(drop=True)
    )

    # Rebuild the influence matrix: rows = units, columns = (g, t) pairs.
    units = pd.unique(frame["unit"])
    wide = frame.pivot_table(
        index="unit", columns=["group", "time"], values="influence", aggfunc="first"
    ).reindex(units)
    col_order = list(zip(pairs["group"], pairs["time"]))
    inf_matrix = wide[col_order].to_numpy(dtype=float)
    if np.isnan(inf_matrix).any():
        raise MethodIncompatibility(
            "influence frame has missing unit × (g, t) cells — every unit "
            "must appear for every (g, t) pair.",
            recovery_hint="Re-export with sp.influence_functions().",
            diagnostics={"n_missing": int(np.isnan(inf_matrix).sum())},
        )

    n_units = len(units)

    # Analytic per-cell SEs from the influence functions (same plug-in
    # used at fit time), so the rebuilt detail is self-consistent.
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(np.mean(inf_matrix**2, axis=0) / n_units)
    se = np.where(se > 0, se, np.inf)
    att = pairs["att"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(np.isfinite(se) & (se > 0), att / se, 0.0)
    detail = pd.DataFrame(
        {
            "group": pairs["group"],
            "time": pairs["time"],
            "att": att,
            "se": se,
            "pvalue": np.where(
                np.isfinite(se), 2 * (1 - stats.norm.cdf(np.abs(z))), 1.0
            ),
            "relative_time": pairs["relative_time"],
        }
    )

    # Cohort sizes from the per-unit cohort labels.
    unit_cohorts = frame.drop_duplicates(subset=["unit"]).set_index("unit")[
        "unit_cohort"
    ]
    cohort_sizes = unit_cohorts.value_counts()

    model_info: dict = {
        "cohort_sizes": cohort_sizes,
        "n_units": n_units,
        "source": "influence_functions export",
    }
    if "cluster" in frame.columns:
        model_info["_cluster_ids"] = (
            frame.drop_duplicates(subset=["unit"])
            .set_index("unit")["cluster"]
            .reindex(units)
            .to_numpy()
        )

    shell = CausalResult(
        method="Callaway and Sant'Anna (2021) — from influence functions",
        estimand="ATT",
        estimate=float("nan"),
        se=float("nan"),
        pvalue=1.0,
        ci=(float("nan"), float("nan")),
        alpha=float(aggte_kwargs.get("alpha", 0.05)),
        n_obs=n_units,
        detail=detail,
        model_info=model_info,
        _influence_funcs=inf_matrix,
        _citation_key="callaway_santanna",
    )
    return aggte(shell, type=type, **aggte_kwargs)
