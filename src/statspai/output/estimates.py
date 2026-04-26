"""
Stata-style estimates store / esttab / estout workflow.

Lets users store multiple model results and produce publication-quality
comparison tables in text, LaTeX, HTML, Markdown, or CSV.

Usage
-----
>>> import statspai as sp
>>> r1 = sp.regress("y ~ x1", data=df)
>>> r2 = sp.regress("y ~ x1 + x2", data=df)
>>> sp.eststo(r1, name="(1)")
>>> sp.eststo(r2, name="(2)")
>>> sp.esttab()           # print stored models
>>> sp.esttab(r1, r2)     # or pass models directly
>>> sp.estclear()         # clear the global store
"""

from __future__ import annotations

import re
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Global model store
# ---------------------------------------------------------------------------

_STORE: List[Tuple[str, Any]] = []  # [(name, result), ...]


def eststo(result, *, name: Optional[str] = None) -> None:
    """Store a model result (like Stata's ``estimates store``)."""
    if name is None:
        name = f"({len(_STORE) + 1})"
    _STORE.append((name, result))


def estclear() -> None:
    """Clear all stored model results."""
    _STORE.clear()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STAT_ALIASES = {
    "N": "N",
    "n": "N",
    "nobs": "N",
    "R2": "R-squared",
    "r2": "R-squared",
    "R-squared": "R-squared",
    "adj_R2": "Adj. R-squared",
    "adj_r2": "Adj. R-squared",
    "Adj. R-squared": "Adj. R-squared",
    "F": "F-statistic",
    "f": "F-statistic",
    "F-statistic": "F-statistic",
    "AIC": "AIC",
    "aic": "AIC",
    "BIC": "BIC",
    "bic": "BIC",
    "ll": "Log-Likelihood",
    "Log-Likelihood": "Log-Likelihood",
}

_STAT_DISPLAY = {
    "N": "N",
    "R-squared": "R\u00b2",
    "Adj. R-squared": "Adj. R\u00b2",
    "F-statistic": "F",
    "AIC": "AIC",
    "BIC": "BIC",
    "Log-Likelihood": "Log-Lik.",
}


def _is_econometric(result) -> bool:
    return hasattr(result, "params") and hasattr(result, "std_errors") and not _is_causal(result)


def _is_causal(result) -> bool:
    return hasattr(result, "estimand") and hasattr(result, "estimate") and hasattr(result, "se")


class _ModelData:
    """Normalised extraction from either result type."""

    __slots__ = ("params", "std_errors", "tvalues", "pvalues",
                 "conf_int_lower", "conf_int_upper", "stats", "depvar",
                 "df_resid")

    def __init__(
        self,
        params: pd.Series,
        std_errors: pd.Series,
        tvalues: pd.Series,
        pvalues: pd.Series,
        conf_int_lower: pd.Series,
        conf_int_upper: pd.Series,
        stats: Dict[str, Any],
        depvar: str,
        df_resid: Optional[float] = None,
    ):
        for attr, val in zip(self.__slots__,
                             [params, std_errors, tvalues, pvalues,
                              conf_int_lower, conf_int_upper, stats, depvar,
                              df_resid]):
            object.__setattr__(self, attr, val)


def _ci_bounds(model: "_ModelData", var: str, alpha: float) -> Tuple[float, float]:
    """Return ``(lower, upper)`` CI bounds at level ``1 - alpha``.

    Reuses the result-stored 95% CI when ``alpha == 0.05`` (preserves the
    exact numbers fit-time produced — typically t-based with model df).
    Otherwise recomputes ``b ± crit · se`` using the t-distribution when
    ``df_resid`` is known, falling back to the standard normal.
    """
    if var not in model.params.index:
        return np.nan, np.nan
    if abs(alpha - 0.05) < 1e-12:
        lo = model.conf_int_lower.get(var, np.nan)
        hi = model.conf_int_upper.get(var, np.nan)
        if not (pd.isna(lo) or pd.isna(hi)):
            return float(lo), float(hi)
    se = model.std_errors.get(var, np.nan)
    if pd.isna(se):
        return np.nan, np.nan
    df = getattr(model, "df_resid", None)
    if df is not None and np.isfinite(df) and df > 0:
        crit = sp_stats.t.ppf(1 - alpha / 2, df)
    else:
        crit = sp_stats.norm.ppf(1 - alpha / 2)
    b = float(model.params[var])
    se_f = float(se)
    return b - crit * se_f, b + crit * se_f


def _extract_model_data(result) -> _ModelData:
    """Unified extraction for EconometricResults and CausalResult."""

    if _is_causal(result):
        name = getattr(result, "estimand", "Treatment")
        params = pd.Series({name: result.estimate})
        std_errors = pd.Series({name: result.se})
        t = result.estimate / result.se if result.se > 0 else np.nan
        tvalues = pd.Series({name: t})
        pvalues = pd.Series({name: result.pvalue})
        ci = getattr(result, "ci", (np.nan, np.nan))
        ci_lo = pd.Series({name: ci[0]})
        ci_hi = pd.Series({name: ci[1]})
        n = getattr(result, "n_obs", None)
        mi = getattr(result, "model_info", {}) or {}
        stats: Dict[str, Any] = {"N": n}
        for k in ("R-squared", "Adj. R-squared", "F-statistic", "AIC", "BIC", "Log-Likelihood"):
            if k in mi:
                stats[k] = mi[k]
        depvar = getattr(result, "method", "")
        df_resid = mi.get("df_resid") if isinstance(mi, dict) else None
        return _ModelData(params, std_errors, tvalues, pvalues, ci_lo, ci_hi,
                          stats, depvar, df_resid=df_resid)

    # EconometricResults (or duck-typed equivalent)
    params = result.params
    std_errors = result.std_errors
    tvalues = getattr(result, "tvalues", params / std_errors)
    pvalues_raw = getattr(result, "pvalues", None)
    if pvalues_raw is None:
        pvalues_raw = pd.Series(np.nan, index=params.index)
    elif not isinstance(pvalues_raw, pd.Series):
        pvalues_raw = pd.Series(pvalues_raw, index=params.index)
    pvalues = pvalues_raw

    tvalues_raw = tvalues
    if not isinstance(tvalues_raw, pd.Series):
        tvalues = pd.Series(tvalues_raw, index=params.index)

    ci_lo = getattr(result, "conf_int_lower", None)
    ci_hi = getattr(result, "conf_int_upper", None)
    if ci_lo is None:
        ci_lo = pd.Series(np.nan, index=params.index)
    elif not isinstance(ci_lo, pd.Series):
        ci_lo = pd.Series(ci_lo, index=params.index)
    if ci_hi is None:
        ci_hi = pd.Series(np.nan, index=params.index)
    elif not isinstance(ci_hi, pd.Series):
        ci_hi = pd.Series(ci_hi, index=params.index)

    diag = getattr(result, "diagnostics", {}) or {}
    dinfo = getattr(result, "data_info", {}) or {}
    n = diag.get("N") or dinfo.get("nobs")
    stats = {"N": n}
    for k in ("R-squared", "Adj. R-squared", "F-statistic", "AIC", "BIC", "Log-Likelihood"):
        if k in diag:
            stats[k] = diag[k]
    depvar = dinfo.get("dependent_var", "")
    df_resid = (
        getattr(result, "df_resid", None)
        or diag.get("df_resid")
        or dinfo.get("df_resid")
    )
    if df_resid is None and n is not None:
        try:
            df_resid = float(n) - float(len(params))
        except (TypeError, ValueError):
            df_resid = None
    return _ModelData(params, std_errors, tvalues, pvalues, ci_lo, ci_hi,
                      stats, depvar, df_resid=df_resid)


def _format_stars(pvalue: float, levels: Tuple[float, ...] = (0.10, 0.05, 0.01)) -> str:
    """Return significance stars for *pvalue* given threshold *levels*."""
    if pvalue is None or (isinstance(pvalue, float) and np.isnan(pvalue)):
        return ""
    stars = ""
    for lev in sorted(levels, reverse=True):
        if pvalue < lev:
            stars += "*"
    return stars


def _fmt_auto(value: float) -> str:
    """Magnitude-adaptive numeric formatting.

    Picks decimal precision per |value| so a single table can mix
    dollar-magnitude coefficients (e.g. ``1521``) and elasticity-magnitude
    coefficients (e.g. ``0.288``) without one side being rounded to zero.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    av = abs(float(value))
    if av >= 1000:
        return f"{value:,.0f}"
    if av >= 100:
        return f"{value:.0f}"
    if av >= 10:
        return f"{value:.1f}"
    if av >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _fmt_val(value: float, fmt: str = "%.4f") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if fmt == "auto":
        return _fmt_auto(value)
    return fmt % value


def _fmt_int(value) -> str:
    if value is None:
        return ""
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)


# ---------------------------------------------------------------------------
# Core table builder
# ---------------------------------------------------------------------------

class EstimateTable:
    """Build a comparison table from multiple model results."""

    def __init__(
        self,
        models: Sequence[_ModelData],
        names: Sequence[str],
        *,
        se: bool = True,
        t: bool = False,
        p: bool = False,
        ci: bool = False,
        stars: bool = True,
        star_levels: Tuple[float, ...] = (0.10, 0.05, 0.01),
        keep: Optional[Sequence[str]] = None,
        drop: Optional[Sequence[str]] = None,
        order: Optional[Sequence[str]] = None,
        labels: Optional[Dict[str, str]] = None,
        stats: Optional[Sequence[str]] = None,
        fmt: str = "%.4f",
        title: Optional[str] = None,
        notes: Optional[Sequence[str]] = None,
        alpha: float = 0.05,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        self.models = list(models)
        self.names = list(names)
        self.se = se
        self.t = t
        self.p = p
        self.ci = ci
        self.show_stars = stars
        self.star_levels = star_levels
        self.keep = list(keep) if keep else None
        self.drop = set(drop) if drop else set()
        self.order = list(order) if order else None
        self.labels = labels or {}
        self.requested_stats = list(stats) if stats else ["N", "R2", "adj_R2", "F"]
        self.fmt = fmt
        self.title = title
        self.notes = list(notes) if notes else []
        self.alpha = float(alpha)
        self.n_models = len(self.models)

        # Resolve ordered variable list across all models
        self._vars = self._resolve_vars()
        # Resolve stat keys
        self._stat_keys = self._resolve_stat_keys()

    # --- variable ordering ---------------------------------------------------

    def _resolve_vars(self) -> List[str]:
        seen: OrderedDict[str, None] = OrderedDict()
        for m in self.models:
            for v in m.params.index:
                seen[v] = None
        all_vars = list(seen)

        if self.keep is not None:
            all_vars = [v for v in all_vars if v in set(self.keep)]
        if self.drop:
            all_vars = [v for v in all_vars if v not in self.drop]
        if self.order:
            ordered: List[str] = []
            remaining = list(all_vars)
            for v in self.order:
                if v in remaining:
                    ordered.append(v)
                    remaining.remove(v)
            ordered.extend(remaining)
            all_vars = ordered
        return all_vars

    def _resolve_stat_keys(self) -> List[str]:
        keys: List[str] = []
        for s in self.requested_stats:
            canonical = _STAT_ALIASES.get(s, s)
            if canonical not in keys:
                keys.append(canonical)
        return keys

    # --- per-cell formatting --------------------------------------------------

    def _coef_cell(self, model: _ModelData, var: str) -> str:
        if var not in model.params.index:
            return ""
        val = model.params[var]
        txt = _fmt_val(val, self.fmt)
        if self.show_stars and var in model.pvalues.index:
            txt += _format_stars(model.pvalues[var], self.star_levels)
        return txt

    def _second_row_cell(self, model: _ModelData, var: str) -> str:
        """Return the parenthetical / bracket row beneath the coefficient."""
        if var not in model.params.index:
            return ""
        if self.ci:
            lo_v, hi_v = _ci_bounds(model, var, self.alpha)
            lo = _fmt_val(lo_v, self.fmt)
            hi = _fmt_val(hi_v, self.fmt)
            return f"[{lo}, {hi}]"
        if self.t:
            return f"({_fmt_val(model.tvalues.get(var, np.nan), self.fmt)})"
        if self.p:
            return f"({_fmt_val(model.pvalues.get(var, np.nan), self.fmt)})"
        # default: standard error
        return f"({_fmt_val(model.std_errors.get(var, np.nan), self.fmt)})"

    def _stat_cell(self, model: _ModelData, key: str) -> str:
        val = model.stats.get(key)
        if val is None:
            return ""
        if key == "N":
            return _fmt_int(val)
        return _fmt_val(float(val), "%.3f")

    def _second_row_label(self) -> str:
        if self.ci:
            level = (1.0 - self.alpha) * 100.0
            level_str = f"{level:g}"
            return f"{level_str}% CI"
        if self.t:
            return "t-statistics"
        if self.p:
            return "p-values"
        return "Standard errors"

    # --- renderers ------------------------------------------------------------

    def _depvar_row(self) -> List[str]:
        return [m.depvar for m in self.models]

    def _star_note(self) -> str:
        parts = []
        sorted_levels = sorted(self.star_levels, reverse=True)
        for i, lev in enumerate(sorted_levels):
            stars = "*" * (i + 1)
            parts.append(f"{stars} p<{lev:.2f}")
        return ", ".join(parts)

    # =========================================================================
    # TEXT output
    # =========================================================================

    def to_text(self) -> str:
        col_w = 13
        label_w = max(
            (len(self.labels.get(v, v)) for v in self._vars),
            default=10,
        )
        label_w = max(label_w, max((len(_STAT_DISPLAY.get(k, k)) for k in self._stat_keys), default=5))
        label_w = max(label_w, 16)
        total_w = label_w + col_w * self.n_models + 2

        line_thick = "\u2501" * total_w
        lines: List[str] = []

        if self.title:
            lines.append(self.title)
            lines.append("")

        # Header
        lines.append(line_thick)
        hdr = " " * label_w
        for n in self.names:
            hdr += f"{n:>{col_w}}"
        lines.append(hdr)

        # Dependent variable row
        depvars = self._depvar_row()
        if any(depvars):
            row = " " * label_w
            for dv in depvars:
                row += f"{dv:>{col_w}}"
            lines.append(row)

        lines.append(line_thick)

        # Coefficients
        for var in self._vars:
            label = self.labels.get(var, var)
            row = f"{label:<{label_w}}"
            for m in self.models:
                row += f"{self._coef_cell(m, var):>{col_w}}"
            lines.append(row)

            # Second row
            if self.se or self.t or self.p or self.ci:
                row2 = " " * label_w
                for m in self.models:
                    row2 += f"{self._second_row_cell(m, var):>{col_w}}"
                lines.append(row2)
                lines.append("")  # blank line between variables

        lines.append(line_thick)

        # Stats
        for key in self._stat_keys:
            disp = _STAT_DISPLAY.get(key, key)
            row = f"{disp:<{label_w}}"
            for m in self.models:
                row += f"{self._stat_cell(m, key):>{col_w}}"
            lines.append(row)

        lines.append(line_thick)

        # Notes
        lines.append(f"{self._second_row_label()} in parentheses")
        if self.show_stars:
            lines.append(self._star_note())
        for note in self.notes:
            lines.append(note)

        return "\n".join(lines)

    # =========================================================================
    # LaTeX output
    # =========================================================================

    def to_latex(self) -> str:
        ncols = self.n_models + 1
        col_spec = "l" + "c" * self.n_models
        lines: List[str] = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        if self.title:
            lines.append(f"\\caption{{{_latex_escape(self.title)}}}")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\hline\\hline")

        # Header
        hdr = " & ".join([""] + [_latex_escape(n) for n in self.names]) + " \\\\"
        lines.append(hdr)

        # Depvar
        depvars = self._depvar_row()
        if any(depvars):
            row = " & ".join([""] + [_latex_escape(dv) for dv in depvars]) + " \\\\"
            lines.append(row)

        lines.append("\\hline")

        # Coefficients
        for var in self._vars:
            label = _latex_escape(self.labels.get(var, var))
            cells = [self._coef_cell(m, var) for m in self.models]
            cells = [_latex_escape(c) for c in cells]
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
            if self.se or self.t or self.p or self.ci:
                cells2 = [self._second_row_cell(m, var) for m in self.models]
                cells2 = [_latex_escape(c) for c in cells2]
                lines.append(" & " + " & ".join(cells2) + " \\\\")

        lines.append("\\hline")

        # Stats
        for key in self._stat_keys:
            disp = _STAT_DISPLAY.get(key, key)
            if key == "R-squared":
                disp = "R$^2$"
            elif key == "Adj. R-squared":
                disp = "Adj. R$^2$"
            else:
                disp = _latex_escape(disp)
            cells = [self._stat_cell(m, key) for m in self.models]
            lines.append(f"{disp} & " + " & ".join(cells) + " \\\\")

        lines.append("\\hline\\hline")

        # Notes
        note_line = f"{self._second_row_label()} in parentheses"
        lines.append(
            f"\\multicolumn{{{ncols}}}{{l}}{{\\footnotesize {_latex_escape(note_line)}}} \\\\"
        )
        if self.show_stars:
            star_note = self._star_note()
            lines.append(
                f"\\multicolumn{{{ncols}}}{{l}}{{\\footnotesize {_latex_escape(star_note)}}} \\\\"
            )
        for note in self.notes:
            lines.append(
                f"\\multicolumn{{{ncols}}}{{l}}{{\\footnotesize {_latex_escape(note)}}} \\\\"
            )

        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    # =========================================================================
    # HTML output (also used for _repr_html_)
    # =========================================================================

    def to_html(self) -> str:
        lines: List[str] = []
        lines.append('<table class="esttab" style="border-collapse:collapse; font-family:serif; font-size:13px;">')

        if self.title:
            lines.append(
                f'<caption style="font-weight:bold; font-size:14px; margin-bottom:6px;">'
                f'{_html_escape(self.title)}</caption>'
            )

        # Header
        lines.append("<thead>")
        lines.append("<tr>")
        lines.append('<th style="text-align:left; border-top:2px solid black; border-bottom:1px solid black;"></th>')
        for n in self.names:
            lines.append(
                f'<th style="text-align:center; border-top:2px solid black; '
                f'border-bottom:1px solid black; padding:2px 10px;">{_html_escape(n)}</th>'
            )
        lines.append("</tr>")

        # Depvar
        depvars = self._depvar_row()
        if any(depvars):
            lines.append("<tr>")
            lines.append('<th style="text-align:left;"></th>')
            for dv in depvars:
                lines.append(f'<th style="text-align:center; padding:0 10px; font-style:italic;">{_html_escape(dv)}</th>')
            lines.append("</tr>")

        lines.append("</thead>")
        lines.append("<tbody>")

        # Coefficients
        for var in self._vars:
            label = _html_escape(self.labels.get(var, var))
            lines.append("<tr>")
            lines.append(f'<td style="text-align:left; padding-right:15px;">{label}</td>')
            for m in self.models:
                lines.append(f'<td style="text-align:center; padding:0 10px;">{_html_escape(self._coef_cell(m, var))}</td>')
            lines.append("</tr>")
            if self.se or self.t or self.p or self.ci:
                lines.append("<tr>")
                lines.append("<td></td>")
                for m in self.models:
                    lines.append(
                        f'<td style="text-align:center; padding:0 10px; color:#555;">'
                        f'{_html_escape(self._second_row_cell(m, var))}</td>'
                    )
                lines.append("</tr>")

        # Separator
        lines.append(
            f'<tr><td colspan="{self.n_models + 1}" '
            f'style="border-top:1px solid black;"></td></tr>'
        )

        # Stats
        for key in self._stat_keys:
            disp = _html_escape(_STAT_DISPLAY.get(key, key))
            lines.append("<tr>")
            lines.append(f'<td style="text-align:left; padding-right:15px;">{disp}</td>')
            for m in self.models:
                lines.append(f'<td style="text-align:center; padding:0 10px;">{self._stat_cell(m, key)}</td>')
            lines.append("</tr>")

        # Bottom border
        lines.append(
            f'<tr><td colspan="{self.n_models + 1}" '
            f'style="border-top:2px solid black;"></td></tr>'
        )

        lines.append("</tbody>")

        # Notes
        lines.append("<tfoot>")
        note_text = f"{self._second_row_label()} in parentheses"
        lines.append(
            f'<tr><td colspan="{self.n_models + 1}" style="text-align:left; font-size:11px;">'
            f'{_html_escape(note_text)}</td></tr>'
        )
        if self.show_stars:
            lines.append(
                f'<tr><td colspan="{self.n_models + 1}" style="text-align:left; font-size:11px;">'
                f'{_html_escape(self._star_note())}</td></tr>'
            )
        for note in self.notes:
            lines.append(
                f'<tr><td colspan="{self.n_models + 1}" style="text-align:left; font-size:11px;">'
                f'{_html_escape(note)}</td></tr>'
            )
        lines.append("</tfoot>")
        lines.append("</table>")
        return "\n".join(lines)

    # =========================================================================
    # Markdown output
    # =========================================================================

    def to_markdown(self) -> str:
        lines: List[str] = []
        if self.title:
            lines.append(f"**{self.title}**")
            lines.append("")

        hdr = "| |" + "|".join(f" {n} " for n in self.names) + "|"
        sep = "|---|" + "|".join("---:" for _ in self.names) + "|"
        lines.append(hdr)
        lines.append(sep)

        for var in self._vars:
            label = self.labels.get(var, var)
            cells = [self._coef_cell(m, var) for m in self.models]
            lines.append(f"| {label} |" + "|".join(f" {c} " for c in cells) + "|")
            if self.se or self.t or self.p or self.ci:
                cells2 = [self._second_row_cell(m, var) for m in self.models]
                lines.append("| |" + "|".join(f" {c} " for c in cells2) + "|")

        # Stats
        for key in self._stat_keys:
            disp = _STAT_DISPLAY.get(key, key)
            cells = [self._stat_cell(m, key) for m in self.models]
            lines.append(f"| {disp} |" + "|".join(f" {c} " for c in cells) + "|")

        lines.append("")
        lines.append(f"*{self._second_row_label()} in parentheses*")
        if self.show_stars:
            lines.append(f"*{self._star_note()}*")
        for note in self.notes:
            lines.append(f"*{note}*")

        return "\n".join(lines)

    # =========================================================================
    # CSV output
    # =========================================================================

    def to_csv(self) -> str:
        rows: List[List[str]] = []
        rows.append([""] + self.names)

        for var in self._vars:
            label = self.labels.get(var, var)
            row = [label] + [self._coef_cell(m, var) for m in self.models]
            rows.append(row)
            if self.se or self.t or self.p or self.ci:
                row2 = [""] + [self._second_row_cell(m, var) for m in self.models]
                rows.append(row2)

        for key in self._stat_keys:
            disp = _STAT_DISPLAY.get(key, key)
            row = [disp] + [self._stat_cell(m, key) for m in self.models]
            rows.append(row)

        return "\n".join(",".join(f'"{c}"' for c in r) for r in rows)

    # =========================================================================
    # DataFrame output
    # =========================================================================

    def to_dataframe(self) -> pd.DataFrame:
        """Return the table as a pandas DataFrame for programmatic use."""
        records: List[Dict[str, str]] = []

        for var in self._vars:
            label = self.labels.get(var, var)
            row: Dict[str, str] = {"": label}
            for name, m in zip(self.names, self.models):
                row[name] = self._coef_cell(m, var)
            records.append(row)

            if self.se or self.t or self.p or self.ci:
                row2: Dict[str, str] = {"": ""}
                for name, m in zip(self.names, self.models):
                    row2[name] = self._second_row_cell(m, var)
                records.append(row2)

        for key in self._stat_keys:
            disp = _STAT_DISPLAY.get(key, key)
            row_s: Dict[str, str] = {"": disp}
            for name, m in zip(self.names, self.models):
                row_s[name] = self._stat_cell(m, key)
            records.append(row_s)

        df = pd.DataFrame(records)
        df = df.set_index("")
        df.index.name = None
        return df

    # =========================================================================
    # Dispatch
    # =========================================================================

    def render(self, output: str = "text") -> str:
        renderers = {
            "text": self.to_text,
            "latex": self.to_latex,
            "html": self.to_html,
            "markdown": self.to_markdown,
            "md": self.to_markdown,
            "csv": self.to_csv,
        }
        func = renderers.get(output)
        if func is None:
            raise ValueError(
                f"Unknown output format '{output}'. "
                f"Choose from: {', '.join(renderers)}"
            )
        return func()


# ---------------------------------------------------------------------------
# Escape helpers
# ---------------------------------------------------------------------------

def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters (except $ and *)."""
    if not text:
        return ""
    text = text.replace("\\", "\\textbackslash{}")
    for ch in ("&", "%", "#", "_", "{", "}"):
        text = text.replace(ch, f"\\{ch}")
    text = text.replace("~", "\\textasciitilde{}")
    text = text.replace("^", "\\textasciicircum{}")
    return text


def _html_escape(text: str) -> str:
    if not text:
        return ""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


# ---------------------------------------------------------------------------
# EstimateTableResult  (wrapper with _repr_html_)
# ---------------------------------------------------------------------------

class EstimateTableResult:
    """Thin wrapper that prints nicely in terminals and Jupyter."""

    def __init__(self, table: EstimateTable, output: str = "text"):
        self._table = table
        self._output = output

    def __str__(self) -> str:
        return self._table.render(self._output)

    def __repr__(self) -> str:
        return self.__str__()

    def _repr_html_(self) -> str:
        """Rich display in Jupyter notebooks."""
        return self._table.to_html()

    def to_text(self) -> str:
        return self._table.to_text()

    def to_latex(self) -> str:
        return self._table.to_latex()

    def to_html(self) -> str:
        return self._table.to_html()

    def to_markdown(self) -> str:
        return self._table.to_markdown()

    def to_csv(self) -> str:
        return self._table.to_csv()

    def to_dataframe(self) -> pd.DataFrame:
        return self._table.to_dataframe()


# ---------------------------------------------------------------------------
# Public API: esttab
# ---------------------------------------------------------------------------

def esttab(
    *results,
    names: Optional[Sequence[str]] = None,
    se: bool = True,
    t: bool = False,
    p: bool = False,
    ci: bool = False,
    stars: bool = True,
    star_levels: Tuple[float, ...] = (0.10, 0.05, 0.01),
    keep: Optional[Sequence[str]] = None,
    drop: Optional[Sequence[str]] = None,
    order: Optional[Sequence[str]] = None,
    labels: Optional[Dict[str, str]] = None,
    stats: Optional[Sequence[str]] = None,
    fmt: str = "%.4f",
    output: str = "text",
    filename: Optional[str] = None,
    title: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> EstimateTableResult:
    """
    Produce a publication-quality model comparison table.

    Accepts model results directly as positional arguments **or** reads
    from the global store populated by :func:`eststo`.  If both positional
    arguments and a non-empty store exist, positional arguments take
    precedence.

    Parameters
    ----------
    *results : model result objects
        ``EconometricResults`` or ``CausalResult`` instances.  If omitted
        the function uses models previously stored with :func:`eststo`.
    names : list of str, optional
        Column header labels.  Defaults to ``(1)``, ``(2)``, etc.
    se : bool, default True
        Show standard errors in parentheses beneath coefficients.
    t : bool, default False
        Show t-statistics instead of standard errors.
    p : bool, default False
        Show p-values instead of standard errors.
    ci : bool, default False
        Show ``(1 - alpha) * 100`` % confidence intervals instead of
        standard errors. The CI level is controlled by ``alpha``.
    stars : bool, default True
        Append significance stars to coefficients.
    star_levels : tuple of float, default (0.10, 0.05, 0.01)
        Thresholds for ``*``, ``**``, ``***``.
    keep : list of str, optional
        Only display these variables.
    drop : list of str, optional
        Hide these variables.
    order : list of str, optional
        Reorder variables (unlisted variables appear after).
    labels : dict, optional
        Rename variables in display (e.g. ``{"education": "Years of Education"}``).
    stats : list of str, optional
        Summary statistics shown below coefficients.  Recognised aliases:
        ``N``, ``R2``, ``adj_R2``, ``F``, ``AIC``, ``BIC``, ``ll``.
        Defaults to ``["N", "R2", "adj_R2", "F"]``.
    fmt : str, default ``"%.4f"``
        ``printf``-style format string for numeric values.
    output : str, default ``"text"``
        Output format: ``"text"``, ``"latex"``, ``"html"``, ``"markdown"``/
        ``"md"``, or ``"csv"``.
    filename : str, optional
        Write table to this file path.
    title : str, optional
        Table title/caption.
    notes : list of str, optional
        Additional notes printed beneath the table.
    alpha : float, default 0.05
        Significance level for confidence intervals (only used when
        ``ci=True``). The displayed CI is ``(1 - alpha) * 100``%.

    Returns
    -------
    EstimateTableResult
        Object that prints the table (and renders as HTML in Jupyter).
    """

    # Resolve models: positional args > global store
    if results:
        model_list = list(results)
        name_list = list(names) if names else [f"({i + 1})" for i in range(len(model_list))]
    elif _STORE:
        name_list = [n for n, _ in _STORE]
        model_list = [r for _, r in _STORE]
        if names:
            name_list = list(names)
    else:
        warnings.warn("No models to tabulate. Pass results directly or use eststo() first.")
        return EstimateTableResult(
            EstimateTable([], [], fmt=fmt, title=title, notes=notes), output
        )

    if len(name_list) != len(model_list):
        raise ValueError(
            f"Length mismatch: {len(model_list)} models but {len(name_list)} names."
        )

    # Extract normalised data
    models_data = [_extract_model_data(m) for m in model_list]

    table = EstimateTable(
        models_data,
        name_list,
        se=se,
        t=t,
        p=p,
        ci=ci,
        stars=stars,
        star_levels=star_levels,
        keep=keep,
        drop=drop,
        order=order,
        labels=labels,
        stats=stats,
        fmt=fmt,
        title=title,
        notes=notes,
        alpha=alpha,
    )

    result_obj = EstimateTableResult(table, output)

    # Write to file if requested
    if filename:
        path = Path(filename)
        # Auto-detect output format from extension if user left default
        if output == "text":
            ext_map = {".tex": "latex", ".html": "html", ".htm": "html",
                       ".md": "markdown", ".csv": "csv"}
            detected = ext_map.get(path.suffix.lower())
            if detected:
                result_obj = EstimateTableResult(table, detected)

        content = str(result_obj)
        path.write_text(content, encoding="utf-8")

    # No auto-print: Jupyter uses _repr_html_, REPL uses __repr__.
    # Scripts wanting stdout output should do `print(esttab(...))`.

    return result_obj
