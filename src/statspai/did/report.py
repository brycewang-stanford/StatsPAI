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
    # Plot: 2×2 summary panel
    # ------------------------------------------------------------------
    def plot(self, figsize=(14, 10), suptitle: Optional[str] = None):
        """Render a 2×2 summary figure of the report.

        The four quadrants show:

        - **Top-left**: event study (dynamic) with uniform confidence band
        - **Top-right**: θ(g) per-cohort aggregation with uniform band
        - **Bottom-left**: θ(t) per-calendar-time aggregation
        - **Bottom-right**: Rambachan–Roth breakdown M* across post event times

        Requires matplotlib.  Returns ``(fig, axes)``.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - env check
            raise ImportError(
                "matplotlib is required for CSReport.plot(). "
                "Install: pip install matplotlib"
            ) from exc

        # Local import to avoid a circular statspai.did.__init__ dependency.
        from .plots import ggdid
        from .aggte import aggte

        cs_like = _CSReportLike(self.dynamic, {'aggregation': 'dynamic'},
                                self.overall.get('estimate', 0.0),
                                self.overall.get('se', 0.0), self.meta)
        cs_like_group = _CSReportLike(self.group, {'aggregation': 'group'},
                                      0.0, 0.0, self.meta)
        cs_like_cal = _CSReportLike(self.calendar, {'aggregation': 'calendar'},
                                    0.0, 0.0, self.meta)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ggdid(cs_like, ax=axes[0, 0])
        ggdid(cs_like_group, ax=axes[0, 1])
        ggdid(cs_like_cal, ax=axes[1, 0])
        self._plot_breakdown(axes[1, 1])

        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=14, y=1.00)
        fig.tight_layout()
        return fig, axes

    def _plot_breakdown(self, ax) -> None:
        """Render the R-R breakdown M* panel (horizontal bars)."""
        import matplotlib.pyplot as plt  # noqa: F401 — caller loaded it
        df = self.breakdown
        if len(df) == 0:
            ax.text(0.5, 0.5, "No post-treatment event times",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Rambachan–Roth breakdown M*")
            return
        y = np.arange(len(df))
        ax.barh(y, df['breakdown_M_star'].values, color='#2E86AB',
                alpha=0.85)
        # Reference: one pointwise SE (robust at 1σ threshold).
        ses = df['se'].values
        ax.plot(ses, y, 'o', color='#E74C3C',
                label='1 × SE', markersize=7, zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels([f"e = {int(e)}" for e in df['relative_time']])
        ax.set_xlabel("Breakdown M* (smoothness bound)")
        ax.set_title("Rambachan–Roth breakdown M*")
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.legend(frameon=False, fontsize=9)

    # ------------------------------------------------------------------
    # Export: Markdown
    # ------------------------------------------------------------------
    def to_markdown(self, float_format: str = "%.4f") -> str:
        """Render the report as GitHub-Flavoured Markdown.

        Suitable for pasting directly into a pull request, blog post,
        or Jupyter notebook Markdown cell.
        """
        m = self.meta
        o = self.overall
        alpha = m.get("alpha", 0.05)
        ci_pct = int(100 * (1 - alpha))

        lines: List[str] = []
        lines.append("## Callaway–Sant'Anna Staggered-DID Report")
        lines.append("")
        lines.append(
            f"- **Units / Periods / Cohorts**: "
            f"{m.get('n_units', '?')} / {m.get('n_periods', '?')} / "
            f"{m.get('n_cohorts', '?')}"
        )
        lines.append(
            f"- **Estimator**: {m.get('estimator', '?')} · "
            f"**Control group**: {m.get('control_group', '?')} · "
            f"**Anticipation**: {m.get('anticipation', 0)}"
        )
        lines.append(
            f"- **Multiplier bootstrap**: B = {m.get('n_boot', '?')}, "
            f"seed = {m.get('random_state', '—')}"
        )
        lines.append(
            f"- **Overall ATT** = {o['estimate']:.4f} "
            f"(SE = {o['se']:.4f}) · "
            f"{ci_pct}% CI [{o['ci_lower']:.4f}, {o['ci_upper']:.4f}] · "
            f"p = {o['pvalue']:.4g}"
        )
        pt = self.pretrend or {}
        if pt:
            lines.append(
                f"- **Pre-trend Wald**: χ²({pt.get('df', 0)}) = "
                f"{pt.get('statistic', float('nan')):.3f}, "
                f"p = {pt.get('pvalue', float('nan')):.4g}"
            )
        lines.append("")

        def _md(df: pd.DataFrame, cols: List[str]) -> str:
            present = [c for c in cols if c in df.columns]
            sub = _format_numeric_columns(df[present], float_format)
            try:
                return sub.to_markdown(index=False)
            except ImportError:  # tabulate missing — fall back to plain
                return sub.to_string(index=False)

        lines.append("### Event study (dynamic aggregation)")
        lines.append("")
        lines.append(_md(self.dynamic, [
            "relative_time", "att", "se",
            "ci_lower", "ci_upper",
            "cband_lower", "cband_upper", "pvalue",
        ]))
        lines.append("")
        lines.append("### θ(g) — per-cohort aggregation")
        lines.append("")
        lines.append(_md(self.group, [
            "group", "att", "se",
            "ci_lower", "ci_upper",
            "cband_lower", "cband_upper", "pvalue",
        ]))
        lines.append("")
        lines.append("### θ(t) — per-calendar-time aggregation")
        lines.append("")
        lines.append(_md(self.calendar, [
            "time", "att", "se",
            "ci_lower", "ci_upper",
            "cband_lower", "cband_upper", "pvalue",
        ]))
        lines.append("")
        lines.append("### Rambachan–Roth breakdown M\\*")
        lines.append("")
        if len(self.breakdown):
            lines.append(_md(self.breakdown, [
                "relative_time", "att", "se",
                "breakdown_M_star", "robust_at_1_SE",
            ]))
        else:
            lines.append("_No post-treatment event times._")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export: Excel workbook (multi-sheet)
    # ------------------------------------------------------------------
    def to_excel(
        self,
        path,
        float_format: Optional[str] = "%.6f",
        engine: Optional[str] = None,
    ) -> str:
        """Dump the full report to a multi-sheet Excel workbook.

        Creates one sheet per block — ``Summary``, ``Dynamic``,
        ``Group``, ``Calendar``, ``Breakdown``, ``Meta`` — so
        downstream Excel consumers (policy briefs, regulatory reports)
        can link to or copy from the individual tables directly.

        Parameters
        ----------
        path : str | Path
            Destination ``.xlsx`` path.
        float_format : str, optional
            Passed through to :meth:`pandas.DataFrame.to_excel`.
            Pass ``None`` to preserve full precision.
        engine : str, optional
            Excel writer engine (``'openpyxl'`` or ``'xlsxwriter'``).
            If ``None`` pandas picks an installed one; raises a clear
            ImportError here if none is available.

        Returns
        -------
        str
            The path written.
        """
        try:
            import openpyxl  # noqa: F401 — most common default engine
        except ImportError:
            try:
                import xlsxwriter  # noqa: F401
            except ImportError as exc:  # pragma: no cover - env check
                raise ImportError(
                    "CSReport.to_excel requires either 'openpyxl' or "
                    "'xlsxwriter'. Install one: pip install openpyxl"
                ) from exc

        path = str(path)

        # Header block as a two-column Summary sheet.
        m = self.meta
        o = self.overall
        summary_rows = [
            ("n_units", m.get("n_units")),
            ("n_periods", m.get("n_periods")),
            ("n_cohorts", m.get("n_cohorts")),
            ("estimator", m.get("estimator")),
            ("control_group", m.get("control_group")),
            ("anticipation", m.get("anticipation", 0)),
            ("alpha", m.get("alpha", 0.05)),
            ("n_boot", m.get("n_boot")),
            ("random_state", m.get("random_state")),
            ("overall_att", o["estimate"]),
            ("overall_se", o["se"]),
            ("overall_ci_lower", o["ci_lower"]),
            ("overall_ci_upper", o["ci_upper"]),
            ("overall_pvalue", o["pvalue"]),
        ]
        pt = self.pretrend or {}
        if pt:
            summary_rows.extend([
                ("pretrend_chi2", pt.get("statistic")),
                ("pretrend_df", pt.get("df")),
                ("pretrend_pvalue", pt.get("pvalue")),
            ])
        summary = pd.DataFrame(summary_rows, columns=["key", "value"])

        meta_df = pd.DataFrame(
            list(m.items()), columns=["key", "value"],
        )

        writer_kwargs = {"engine": engine} if engine else {}
        with pd.ExcelWriter(path, **writer_kwargs) as w:
            summary.to_excel(w, sheet_name="Summary",
                             index=False, float_format=float_format)
            self.dynamic.to_excel(w, sheet_name="Dynamic",
                                  index=False, float_format=float_format)
            self.group.to_excel(w, sheet_name="Group",
                                index=False, float_format=float_format)
            self.calendar.to_excel(w, sheet_name="Calendar",
                                   index=False, float_format=float_format)
            self.breakdown.to_excel(w, sheet_name="Breakdown",
                                    index=False, float_format=float_format)
            meta_df.to_excel(w, sheet_name="Meta",
                             index=False, float_format=float_format)
        return path

    # ------------------------------------------------------------------
    # Export: LaTeX (booktabs)
    # ------------------------------------------------------------------
    def to_latex(self, float_format: str = "%.4f",
                 caption: Optional[str] = None,
                 label: Optional[str] = None) -> str:
        """Render the report as a publication-ready LaTeX fragment.

        Uses the ``booktabs`` package for each sub-table and wraps the
        result in a single ``table`` float.  Requires ``\\usepackage{booktabs}``
        in the preamble of the consuming document.
        """
        m = self.meta
        o = self.overall
        alpha = m.get("alpha", 0.05)
        ci_pct = int(100 * (1 - alpha))
        caption = caption or "Callaway--Sant'Anna staggered-DID report."
        label = label or "tab:cs_report"

        def _latex(df: pd.DataFrame, cols: List[str]) -> str:
            present = [c for c in cols if c in df.columns]
            return _df_to_booktabs(df[present], float_format)

        header = (
            f"Units / Periods / Cohorts: "
            f"{m.get('n_units', '?')} / {m.get('n_periods', '?')} / "
            f"{m.get('n_cohorts', '?')}. "
            f"Estimator: {m.get('estimator', '?')}; "
            f"Control group: {m.get('control_group', '?')}; "
            f"Anticipation: {m.get('anticipation', 0)}. "
            f"Multiplier bootstrap: $B = {m.get('n_boot', '?')}$, "
            f"seed $= {m.get('random_state', '—')}$. "
            f"Overall ATT $= {o['estimate']:.4f}$ "
            f"(SE $= {o['se']:.4f}$); "
            f"{ci_pct}\\% CI $[{o['ci_lower']:.4f}, {o['ci_upper']:.4f}]$; "
            f"$p = {o['pvalue']:.4g}$."
        )
        pt = self.pretrend or {}
        if pt:
            header += (
                f" Pre-trend Wald: $\\chi^2({pt.get('df', 0)}) = "
                f"{pt.get('statistic', float('nan')):.3f}$, "
                f"$p = {pt.get('pvalue', float('nan')):.4g}$."
            )

        parts: List[str] = []
        parts.append("\\begin{table}[htbp]\\centering")
        parts.append(f"\\caption{{{caption}}}")
        parts.append(f"\\label{{{label}}}")
        parts.append("\\footnotesize")
        parts.append("\\par\\noindent" + header + "\\par\\medskip")

        parts.append("\\textbf{Event study (dynamic aggregation)}\\par")
        parts.append(_latex(self.dynamic, [
            "relative_time", "att", "se",
            "ci_lower", "ci_upper",
            "cband_lower", "cband_upper", "pvalue",
        ]))

        parts.append("\\textbf{$\\theta(g)$ --- per-cohort}\\par")
        parts.append(_latex(self.group, [
            "group", "att", "se",
            "ci_lower", "ci_upper",
            "cband_lower", "cband_upper", "pvalue",
        ]))

        parts.append("\\textbf{$\\theta(t)$ --- per-calendar-time}\\par")
        parts.append(_latex(self.calendar, [
            "time", "att", "se",
            "ci_lower", "ci_upper",
            "cband_lower", "cband_upper", "pvalue",
        ]))

        parts.append("\\textbf{Rambachan--Roth breakdown $M^{*}$}\\par")
        if len(self.breakdown):
            parts.append(_latex(self.breakdown, [
                "relative_time", "att", "se",
                "breakdown_M_star", "robust_at_1_SE",
            ]))
        else:
            parts.append("\\emph{No post-treatment event times.}")

        parts.append("\\end{table}")
        return "\n".join(parts)

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
# Minimal ggdid-compatible shim
# ======================================================================

class _CSReportLike:
    """Adapter that lets ``ggdid`` draw directly from a CSReport slice.

    ``ggdid`` dispatches on ``result.model_info['aggregation']`` and reads
    ``result.detail``, ``result.alpha``; this shim exposes exactly those
    attributes without recomputing bootstrap draws.
    """

    __slots__ = ('detail', 'model_info', 'estimate', 'se', 'alpha')

    def __init__(
        self,
        detail: pd.DataFrame,
        model_info: Dict[str, Any],
        estimate: float,
        se: float,
        meta: Dict[str, Any],
    ) -> None:
        self.detail = detail
        self.model_info = {**meta, **model_info}
        self.estimate = float(estimate)
        self.se = float(se)
        self.alpha = float(meta.get('alpha', 0.05))


# ======================================================================
# Formatting helpers (Markdown / LaTeX)
# ======================================================================

def _format_numeric_columns(
    df: pd.DataFrame,
    float_format: str = "%.4f",
) -> pd.DataFrame:
    """Return a copy of ``df`` with every numeric column formatted as a string.

    Integer-typed columns (e.g. relative_time, group, time) are preserved as
    plain integers so they don't render as ``3.0000``.  Boolean columns are
    rendered as ``True`` / ``False`` verbatim.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_float_dtype(s):
            # relative_time / group / time come through as float when mixed
            # with NaN; keep them integer-looking if they truly are ints.
            if s.dropna().apply(lambda x: float(x).is_integer()).all() and \
                    col in {"relative_time", "group", "time"}:
                out[col] = s.astype("Int64")
                continue
            out[col] = s.apply(
                lambda v: "" if pd.isna(v) else (float_format % float(v))
            )
    return out


def _df_to_booktabs(df: pd.DataFrame, float_format: str = "%.4f") -> str:
    """Jinja2-free LaTeX booktabs rendering of ``df``.

    Produces a minimal ``tabular`` block suitable for inclusion in a
    ``table`` float — left-aligns string columns, right-aligns numerics.
    """
    formatted = _format_numeric_columns(df, float_format)
    aligns = []
    for col in formatted.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            aligns.append("r")
        else:
            aligns.append("l")
    col_spec = "".join(aligns)

    # Single-pass LaTeX escape.  Sequential replace() calls are unsafe
    # because `\` → `\textbackslash{}` inserts `{` and `}` that later
    # passes would re-escape, producing broken output like
    # `\textbackslash\{\}` instead of `\textbackslash{}`.  Using re.sub
    # with a lookup table guarantees each input character is escaped
    # exactly once.
    import re as _re
    _LATEX_ESCAPES = {
        '\\': r'\textbackslash{}',
        '~':  r'\textasciitilde{}',
        '^':  r'\textasciicircum{}',
        '&':  r'\&',
        '%':  r'\%',
        '$':  r'\$',
        '#':  r'\#',
        '_':  r'\_',
        '{':  r'\{',
        '}':  r'\}',
    }
    _LATEX_RE = _re.compile(r'[\\~^&%$#_{}]')

    def _escape(v: object) -> str:
        text = "" if (isinstance(v, float) and pd.isna(v)) else str(v)
        return _LATEX_RE.sub(lambda m: _LATEX_ESCAPES[m.group(0)], text)

    header = " & ".join(_escape(c) for c in formatted.columns) + " \\\\"
    body_rows = [
        " & ".join(_escape(v) for v in row) + " \\\\"
        for row in formatted.itertuples(index=False, name=None)
    ]
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header,
        "\\midrule",
        *body_rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


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
    save_to: Optional[str] = None,
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
    save_to : str, optional
        When set, treats the value as a *path prefix* and writes the
        report in every supported format in one call:

        - ``<prefix>.txt``   — fixed-width plain-text report
        - ``<prefix>.md``    — GitHub-flavoured Markdown
        - ``<prefix>.tex``   — booktabs LaTeX fragment
        - ``<prefix>.xlsx``  — multi-sheet workbook
        - ``<prefix>.png``   — 2×2 summary figure (only if matplotlib
          is installed; silently skipped otherwise)

        Missing parent directories are created on the fly.

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

    if save_to is not None:
        _save_report_bundle(report, save_to, verbose=verbose)

    return report


def _save_report_bundle(
    report: CSReport,
    prefix: str,
    verbose: bool = False,
) -> Dict[str, str]:
    """Write ``report`` to every supported format under the given prefix.

    Returns a ``{format: path}`` dictionary of the files actually written,
    skipping any format whose optional dependency is missing.
    """
    import os
    import sys

    def _sys_modules():
        return sys.modules

    # Expand a leading "~" / "~user" so `save_to='~/study/cs_v1'` works,
    # otherwise os.makedirs would create a literal "./~/study" directory.
    prefix = os.path.expanduser(str(prefix))
    parent = os.path.dirname(prefix)
    if parent:
        os.makedirs(parent, exist_ok=True)

    written: Dict[str, str] = {}

    txt_path = f"{prefix}.txt"
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(report.to_text())
    written["txt"] = txt_path

    md_path = f"{prefix}.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(report.to_markdown())
    written["md"] = md_path

    tex_path = f"{prefix}.tex"
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(report.to_latex())
    written["tex"] = tex_path

    try:
        xlsx_path = f"{prefix}.xlsx"
        report.to_excel(xlsx_path)
        written["xlsx"] = xlsx_path
    except ImportError:
        pass  # openpyxl / xlsxwriter missing — skip silently

    try:
        import matplotlib
        # Only switch to a non-interactive backend if nothing has been
        # set yet — calling matplotlib.use() after pyplot has been
        # imported is a warning-only no-op in modern matplotlib but
        # can surprise users running inside Jupyter.
        if "matplotlib.pyplot" not in _sys_modules():
            try:
                matplotlib.use("Agg")
            except Exception:
                pass  # already set — honour caller's choice
        fig, _ = report.plot()
        png_path = f"{prefix}.png"
        fig.savefig(png_path, dpi=110, bbox_inches="tight")
        # Free the figure so callers in a long loop don't accumulate memory.
        import matplotlib.pyplot as plt
        plt.close(fig)
        written["png"] = png_path
    except ImportError:
        pass

    if verbose:
        print("\nSaved report bundle:")
        for kind, path in written.items():
            print(f"  [{kind}] {path}")

    return written
