"""Effects by subgroup for ``sp.did_workflow``.

Baker, Callaway, Cunningham, Goodman-Bacon and Sant'Anna (2026, section
4.5) [@baker2026difference] treat heterogeneity as a design of its own.
For a discrete characteristic the units are split into its cells and the
same DiD is run inside each one, so that every cell has its own treated
and comparison units and its own parallel-trends assumption. The average
effect is then the average of the cell effects weighted by each cell's
share of the treated units.

That is what this module does. It adds no estimator: each row is the
headline of the same workflow on the rows of one cell.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility
from ._validate import normalized_cohort

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import DiDWorkflow

#: More cells than this is a continuous variable or an identifier, not a
#: partition a reader can take in; the caller is asked to bin it.
MAX_CELLS = 20

_FN = "did_workflow.heterogeneity"


def _check_partition(wf: "DiDWorkflow", by: str) -> pd.Series:
    """The partition as one value per unit, or a refusal."""
    inp = wf._inputs
    data = inp.data
    if not isinstance(by, str) or by not in data.columns:
        raise MethodIncompatibility(
            f"{_FN}: by={by!r} is not a column of the data.",
            recovery_hint="Pass the name of a column that is constant within "
            "each unit, e.g. a region or a baseline category.",
            diagnostics={"by": repr(by)},
        )
    if by in (inp.id, inp.time, inp.cohort, inp.y):
        raise MethodIncompatibility(
            f"{_FN}: by={by!r} is the unit, period, cohort or outcome column.",
            recovery_hint="Split on a characteristic of the units. Effects "
            "by cohort are target='group'; effects by event time are "
            "wf.event_study.",
            diagnostics={"by": by},
        )
    if data[by].isna().any():
        raise MethodIncompatibility(
            f"{_FN}: {by!r} has missing values.",
            recovery_hint=f"Drop or fill the rows without {by!r} first.",
            diagnostics={"by": by, "n_missing": int(data[by].isna().sum())},
        )
    per_unit = data.groupby(inp.id, sort=False)[by].nunique()
    moving = per_unit[per_unit > 1]
    if len(moving):
        raise MethodIncompatibility(
            f"{_FN}: {by!r} changes within {len(moving)} unit(s). A split on "
            "a characteristic that moves over time conditions on something "
            "the treatment may have changed.",
            recovery_hint="Split on the value before treatment, e.g. "
            f"df.groupby(id)[{by!r}].transform('first').",
            diagnostics={"by": by, "units": [str(u) for u in moving.index[:5]]},
        )
    levels = data.groupby(inp.id, sort=False)[by].first()
    n_cells = int(levels.nunique())
    if n_cells < 2:
        raise MethodIncompatibility(
            f"{_FN}: {by!r} takes one value; there is nothing to compare.",
            recovery_hint="Pass a characteristic with two or more values.",
            diagnostics={"by": by},
        )
    if n_cells > MAX_CELLS:
        raise MethodIncompatibility(
            f"{_FN}: {by!r} takes {n_cells} values. Each cell is estimated on "
            "its own units, and cells this small estimate noise.",
            recovery_hint=f"Bin it into at most {MAX_CELLS} cells first, e.g. "
            "with pd.qcut on the pre-treatment value.",
            diagnostics={"by": by, "n_cells": n_cells, "max_cells": MAX_CELLS},
        )
    return levels


def _treated_weight(wf: "DiDWorkflow", levels: pd.Series) -> pd.Series:
    """Each cell's (weighted) count of treated units."""
    inp = wf._inputs
    first = inp.data.groupby(inp.id, sort=False)
    cohort = normalized_cohort(first[inp.cohort].first())
    periods = inp.data[inp.time]
    in_sample = (cohort > 0) & (cohort <= periods.max())
    w = (
        first[inp.weights].first().astype(float)
        if inp.weights
        else pd.Series(1.0, index=cohort.index)
    )
    return (w * in_sample).groupby(levels.reindex(cohort.index)).sum()


def _equality_test(table: pd.DataFrame, nested: bool) -> Dict[str, Any]:
    """Are the cell effects equal? Only when the cells share no cluster.

    With clusters nested in cells the cell estimates are independent, and
    the precision-weighted sum of squared deviations from their weighted
    mean is chi-squared with ``K - 1`` degrees of freedom under equality.
    Cells that share clusters are correlated; no statistic is reported.
    """
    ok = table[np.isfinite(table["estimate"]) & (table["se"] > 0)]
    if len(ok) < 2:
        return {"available": False, "reason": "fewer than two cells were estimated"}
    if not nested:
        return {
            "available": False,
            "reason": "the cells share clusters, so their estimates are "
            "correlated and a test that treats them as independent would "
            "be wrong",
        }
    prec = 1.0 / ok["se"].to_numpy(dtype=float) ** 2
    est = ok["estimate"].to_numpy(dtype=float)
    pooled = float((prec * est).sum() / prec.sum())
    stat = float((prec * (est - pooled) ** 2).sum())
    df = int(len(ok) - 1)
    return {
        "available": True,
        "statistic": stat,
        "df": df,
        "pvalue": float(stats.chi2.sf(stat, df)),
        "null": "the effect is the same in every cell",
    }


def run_heterogeneity(wf: "DiDWorkflow", by: str) -> pd.DataFrame:
    """The workflow's headline, cell by cell of ``by``."""
    from .workflow import DiDWorkflow

    inp = wf._inputs
    levels = _check_partition(wf, by)
    treated = _treated_weight(wf, levels)
    total = float(treated.sum())
    cluster = inp.cluster or inp.id
    nested = bool((inp.data.groupby(cluster, sort=False)[by].nunique() <= 1).all())

    rows: List[Dict[str, Any]] = []
    for level in sorted(levels.unique(), key=lambda v: (str(type(v)), v)):
        sub = inp.data[inp.data[by] == level]
        cell = DiDWorkflow(replace(inp, data=sub), dict(wf._options), set(wf._explicit))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cell.run("estimate")
        row: Dict[str, Any] = {
            by: level,
            "n_units": int(sub[inp.id].nunique()),
            "share_of_treated": (
                float(treated.get(level, 0.0) / total) if total else np.nan
            ),
            "estimate": np.nan,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "pvalue": np.nan,
            "note": "",
        }
        head: Optional[Dict[str, Any]] = cell._headline
        if head is not None:
            row.update(
                {
                    k: float(head[k])
                    for k in ("estimate", "se", "ci_lower", "ci_upper", "pvalue")
                }
            )
        else:
            failed = next(
                (s for s in cell._steps if s.status == "fail" or s.key == "estimate"),
                None,
            )
            row["note"] = (
                failed.finding
                if failed is not None and failed.finding
                else "not estimated"
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    ok = table[np.isfinite(table["estimate"])]
    covered = float(ok["share_of_treated"].sum())
    table.attrs.update(
        by=by,
        estimand=wf.estimand,
        headline=wf.estimate,
        # Baker et al. (2026, section 4.5): the ATT is the average of the
        # cell ATTs weighted by each cell's share of the treated units.
        # The sample analogue need not equal the headline, which compares
        # every treated unit with every comparison unit.
        share_weighted_average=(
            float((ok["share_of_treated"] * ok["estimate"]).sum() / covered)
            if covered > 0
            else np.nan
        ),
        share_of_treated_covered=covered,
        equality_test=_equality_test(table, nested),
    )
    return table
