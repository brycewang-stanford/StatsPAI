"""
One-call staggered-DID report.

``cs_report()`` composes the full Callaway-Sant'Anna workflow —
ATT(g,t) estimation, all four :func:`aggte` aggregations with Mammen
multiplier-bootstrap uniform bands, the pre-trend Wald test, and a
Rambachan-Roth breakdown-value row for every post-treatment event
time — into a single function call, and pretty-prints the result.

The design mirrors the one-screen summaries that practitioners expect
from ``did::summary()`` + ``HonestDiD`` + ``ggdid`` in R, so that a
user can interpret a staggered-DID study at a glance.

References
----------
Callaway, B. and Sant'Anna, P.H.C. (2021).
    "Difference-in-Differences with Multiple Time Periods."
    *Journal of Econometrics*, 225(2), 200-230.
Rambachan, A. and Roth, J. (2023).
    "A More Credible Approach to Parallel Trends."
    *Review of Economic Studies*, 90(5), 2555-2591.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from .aggte import aggte
from .callaway_santanna import callaway_santanna
from .honest_did import breakdown_m


# ======================================================================
# Container
# ======================================================================

@dataclass
class CSReport:
    """Structured output of :func:`cs_report`.

    Attributes are plain pandas objects so downstream users can export
    to LaTeX, Markdown, or Excel without any custom converters.
    """

    overall: Dict[str, float]
    simple: pd.DataFrame
    dynamic: pd.DataFrame
    group: pd.DataFrame
    calendar: pd.DataFrame
    pretrend: Dict[str, Any]
    breakdown: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self._format()

    def to_text(self) -> str:
        """Return the human-readable report as a single string."""
        return self._format()

    # ------------------------------------------------------------------
    def _format(self) -> str:
        w = 78
        lines: List[str] = []
        title = " Callaway–Sant'Anna Staggered-DID Report "
        lines.append("=" * w)
        lines.append(title.center(w, "="))
        lines.append("=" * w)

        # Header block
        m = self.meta
        lines.append(f"Units: {m.get('n_units', '?')}    "
                     f"Periods: {m.get('n_periods', '?')}    "
                     f"Cohorts: {m.get('n_cohorts', '?')}    "
                     f"α = {m.get('alpha', 0.05)}")
        lines.append(f"Estimator: {m.get('estimator', '?')}    "
                     f"Control group: {m.get('control_group', '?')}    "
                     f"Anticipation: {m.get('anticipation', 0)}")
        lines.append(f"Multiplier bootstrap: B = {m.get('n_boot', '?')}, "
                     f"seed = {m.get('random_state', '—')}")

        # Overall ATT
        lines.append("-" * w)
        o = self.overall
        lines.append(
            f"Overall ATT  =  {o['estimate']:.4f}   "
            f"SE = {o['se']:.4f}   "
            f"{int(100*(1-m.get('alpha', 0.05)))}% CI = "
            f"[{o['ci_lower']:.4f}, {o['ci_upper']:.4f}]   "
            f"p = {o['pvalue']:.4g}"
        )

        # Pre-trend
        pt = self.pretrend or {}
        if pt:
            lines.append(
                f"Pre-trend Wald: χ²({pt.get('df', 0)}) = "
                f"{pt.get('statistic', float('nan')):.3f}, "
                f"p = {pt.get('pvalue', float('nan')):.4g}"
            )

        # Dynamic (event study)
        lines.append("-" * w)
        lines.append(" Event study (dynamic aggregation) ".center(w, "-"))
        lines.append(self._fmt_event_study(self.dynamic))

        # Group / calendar
        lines.append("-" * w)
        lines.append(" θ(g) — per-cohort aggregation ".center(w, "-"))
        lines.append(self._fmt_aggregation(self.group, id_col="group"))
        lines.append("-" * w)
        lines.append(" θ(t) — per-calendar-time aggregation ".center(w, "-"))
        lines.append(self._fmt_aggregation(self.calendar, id_col="time"))

        # Breakdown M*
        lines.append("-" * w)
        lines.append(" Rambachan–Roth breakdown M* (smoothness) ".center(w, "-"))
        if len(self.breakdown):
            lines.append(self.breakdown.to_string(index=False,
                                                  float_format="%.4f"))
        else:
            lines.append("(no post-treatment event times)")

        lines.append("=" * w)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_event_study(df: pd.DataFrame) -> str:
        cols = ["relative_time", "att", "se", "ci_lower", "ci_upper",
                "cband_lower", "cband_upper", "pvalue"]
        present = [c for c in cols if c in df.columns]
        return df[present].to_string(index=False, float_format="%.4f")

    @staticmethod
    def _fmt_aggregation(df: pd.DataFrame, id_col: str) -> str:
        cols = [id_col, "att", "se", "ci_lower", "ci_upper",
                "cband_lower", "cband_upper", "pvalue"]
        present = [c for c in cols if c in df.columns]
        return df[present].to_string(index=False, float_format="%.4f")


# ======================================================================
# Public entry point
# ======================================================================

def cs_report(
    data_or_result,
    y: Optional[str] = None,
    g: Optional[str] = None,
    t: Optional[str] = None,
    i: Optional[str] = None,
    x: Optional[List[str]] = None,
    estimator: str = 'dr',
    control_group: str = 'nevertreated',
    anticipation: int = 0,
    alpha: float = 0.05,
    n_boot: int = 1000,
    random_state: Optional[int] = 0,
    min_e: float = -np.inf,
    max_e: float = np.inf,
    rr_method: str = 'smoothness',
    verbose: bool = True,
) -> CSReport:
    """One-call staggered-DID workflow: estimate → aggregate → sensitivity.

    Parameters
    ----------
    data_or_result : pd.DataFrame | CausalResult
        Either a long-format panel (then ``y, g, t, i`` are required and
        :func:`callaway_santanna` is run first), or an already-fitted
        :func:`callaway_santanna` result.
    y, g, t, i : str, optional
        Outcome / cohort / time / unit id columns (required when
        ``data_or_result`` is a DataFrame).
    x : list of str, optional
        Covariates for conditional parallel trends.
    estimator : {'dr', 'ipw', 'reg'}, default 'dr'
    control_group : {'nevertreated', 'notyettreated'}, default 'nevertreated'
    anticipation : int, default 0
    alpha : float, default 0.05
    n_boot : int, default 1000
        Multiplier-bootstrap replications for :func:`aggte`.
    random_state : int, default 0
        Seed for the bootstrap (set to ``None`` for non-reproducibility).
    min_e, max_e : float, default (-inf, inf)
        Event-time window passed to the dynamic aggregation.
    rr_method : {'smoothness', 'relative_magnitude'}, default 'smoothness'
        Sensitivity restriction handed to :func:`breakdown_m`.
    verbose : bool, default True
        If ``True``, print the report before returning.

    Returns
    -------
    CSReport
        Structured container; call ``.to_text()`` to re-render.

    Examples
    --------
    >>> import statspai as sp
    >>> rpt = sp.did.cs_report(
    ...     df, y='y', g='g', t='t', i='id', random_state=42)
    >>> rpt.dynamic           # event-study DataFrame w/ uniform bands
    >>> rpt.breakdown         # R-R breakdown M* per post event time
    """
    if isinstance(data_or_result, CausalResult):
        cs = data_or_result
    else:
        if not all([y, g, t, i]):
            raise ValueError(
                "When passing raw data, the 'y', 'g', 't', and 'i' column "
                "names must all be specified."
            )
        cs = callaway_santanna(
            data_or_result, y=y, g=g, t=t, i=i, x=x,
            estimator=estimator, control_group=control_group,
            anticipation=anticipation, alpha=alpha,
        )

    # Four aggregations sharing the same seed for internal consistency.
    simple = aggte(cs, type='simple', alpha=alpha,
                   n_boot=n_boot, random_state=random_state)
    dynamic = aggte(cs, type='dynamic', alpha=alpha,
                    n_boot=n_boot, random_state=random_state,
                    min_e=min_e, max_e=max_e)
    group = aggte(cs, type='group', alpha=alpha,
                  n_boot=n_boot, random_state=random_state)
    calendar = aggte(cs, type='calendar', alpha=alpha,
                     n_boot=n_boot, random_state=random_state)

    # Breakdown M* for every post-treatment event time in the dynamic frame.
    post_es = dynamic.detail[dynamic.detail['relative_time'] >= 0]
    rr_rows = []
    for _, row in post_es.iterrows():
        e_int = int(row['relative_time'])
        try:
            m_star = breakdown_m(dynamic, e=e_int, method=rr_method,
                                 alpha=alpha)
        except Exception:  # pragma: no cover - defensive
            m_star = float('nan')
        rr_rows.append({
            'relative_time': e_int,
            'att': float(row['att']),
            'se': float(row['se']),
            'breakdown_M_star': m_star,
            'robust_at_1_SE': m_star >= float(row['se']) if row['se'] > 0 else False,
        })
    breakdown_df = pd.DataFrame(rr_rows)

    overall = {
        'estimate': float(simple.estimate),
        'se': float(simple.se),
        'ci_lower': float(simple.ci[0]),
        'ci_upper': float(simple.ci[1]),
        'pvalue': float(simple.pvalue),
    }

    meta = {
        'n_units': cs.model_info.get('n_units'),
        'n_periods': cs.model_info.get('n_periods'),
        'n_cohorts': cs.model_info.get('n_cohorts'),
        'alpha': alpha,
        'estimator': cs.model_info.get('estimator'),
        'control_group': cs.model_info.get('control_group'),
        'anticipation': cs.model_info.get('anticipation', 0),
        'n_boot': n_boot,
        'random_state': random_state,
        'rr_method': rr_method,
    }

    report = CSReport(
        overall=overall,
        simple=simple.detail,
        dynamic=dynamic.detail,
        group=group.detail,
        calendar=calendar.detail,
        pretrend=cs.model_info.get('pretrend_test') or {},
        breakdown=breakdown_df,
        meta=meta,
    )

    if verbose:
        print(report.to_text())
    return report
