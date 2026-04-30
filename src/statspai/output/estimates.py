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
    # Dependent-variable summary rows. Top-5 econ journals routinely require
    # these so reviewers can sanity-check effect magnitudes against the
    # outcome's scale; ``modelsummary`` exposes them via ``glance``,
    # ``esttab`` via ``stats(ymean)``.
    "depvar_mean": "Mean of Y",
    "depvar_sd": "SD of Y",
    "Mean of Y": "Mean of Y",
    "SD of Y": "SD of Y",
    "ymean": "Mean of Y",
    "ysd": "SD of Y",
}

_STAT_DISPLAY = {
    "N": "N",
    "R-squared": "R\u00b2",
    "Adj. R-squared": "Adj. R\u00b2",
    "F-statistic": "F",
    "AIC": "AIC",
    "BIC": "BIC",
    "Log-Likelihood": "Log-Lik.",
    "Mean of Y": "Mean of Y",
    "SD of Y": "SD of Y",
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

    # Dependent-variable mean / SD (sample, ddof=1). Looked up across the
    # several conventions different StatsPAI estimators use to stash the
    # outcome vector — ols stores ``data_info['y']``, IV uses ``y``, GLM
    # uses ``endog``, advanced_iv uses ``dep_var``. When none are present
    # we silently skip rather than fabricate.
    y_vec = (
        dinfo.get("y")
        if isinstance(dinfo, dict)
        else None
    )
    if y_vec is None and isinstance(dinfo, dict):
        y_vec = dinfo.get("endog") or dinfo.get("dep_var") or dinfo.get("Y")
    if y_vec is not None:
        try:
            y_arr = np.asarray(y_vec, dtype=float).ravel()
            if y_arr.size > 0 and np.all(np.isfinite(y_arr)):
                stats["Mean of Y"] = float(np.mean(y_arr))
                if y_arr.size > 1:
                    stats["SD of Y"] = float(np.std(y_arr, ddof=1))
        except (TypeError, ValueError):
            pass
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


# Canonical formatters live in ``_format`` — re-exported here under
# their legacy underscore names so that ``regression_table`` / ``_inline``
# / external callers that imported them keep working unchanged.
from ._format import (
    format_stars as _format_stars,
    fmt_auto as _fmt_auto,
    fmt_val as _fmt_val,
    fmt_int as _fmt_int,
)
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
# esttab — Stata-style facade over regtable (PR-B/5c)
# ---------------------------------------------------------------------------
#
# Historically this module shipped a ~500-line ``EstimateTable`` class that
# re-implemented coefficient extraction, star formatting, three-line table
# styling and every export format — duplicating logic already maintained by
# ``sp.regtable``.  PR-B/5c collapses that to the thin facade below.
#
# The user-visible API (``esttab`` function, ``EstimateTableResult`` class,
# all keyword args) is preserved.  Rendered output now matches ``regtable``'s
# book-tab three-line style and is *not* byte-identical to the legacy bespoke
# renderer.  A ``DeprecationWarning`` is emitted on first call pointing users
# to ``sp.regtable``.  Plan to remove the facade in two minor releases (per
# CLAUDE.md §3.8).
#
# Migration notes
# ---------------
# - ``se=True`` / ``t=True`` / ``p=True`` / ``ci=True`` map to regtable's
#   ``se_type='se' | 't' | 'p' | 'ci'``.  Exactly one should be true; the
#   priority is ``ci > p > t > se`` if multiple are passed (matches the legacy
#   behaviour).
# - ``output='csv'`` is implemented via ``to_dataframe().to_csv()``.
# - All other ``output=`` strings (``"text"``, ``"latex"``, ``"html"``,
#   ``"markdown"`` / ``"md"``) round-trip to the corresponding ``regtable``
#   renderer.
# - ``filename=`` writes the rendered string with auto-detected extension,
#   matching the legacy behaviour.

_DEPRECATION_MSG_ESTTAB = (
    "esttab() is now a thin wrapper over sp.regtable() and will be "
    "removed in a future minor release.  Migrate to "
    "sp.regtable(*models, ...) for the same output with full control "
    "over labels, journal templates, and SE formats.  "
    "See docs/rfc/output_pr_b_consolidation.md for migration."
)


def _warn_once_esttab() -> None:
    warnings.warn(_DEPRECATION_MSG_ESTTAB, DeprecationWarning, stacklevel=3)


class EstimateTableResult:
    """Stata ``esttab`` result handle — thin wrapper over a :class:`RegtableResult`.

    Preserves the ``EstimateTableResult`` type identity for callers that
    do ``isinstance(x, EstimateTableResult)``.  Forwards every render
    method to the underlying ``regtable`` result; adds ``to_csv()`` for
    parity with the legacy esttab API (regtable does not natively
    expose CSV but the dataframe path is byte-identical to what the
    legacy esttab produced).
    """

    def __init__(self, regtable_result, output: str = "text"):
        self._rt = regtable_result
        self._output = output

    # ── pass-through renderers ─────────────────────────────────────────
    def to_text(self) -> str:
        return self._rt.to_text()

    def to_latex(self) -> str:
        return self._rt.to_latex()

    def to_html(self) -> str:
        return self._rt.to_html()

    def to_markdown(self) -> str:
        return self._rt.to_markdown()

    def to_dataframe(self) -> pd.DataFrame:
        return self._rt.to_dataframe()

    def to_csv(self) -> str:
        # Legacy esttab returned CSV via the table's dataframe view.
        return self._rt.to_dataframe().to_csv()

    # ── dunder ─────────────────────────────────────────────────────────
    def _render(self, fmt: str) -> str:
        renderers = {
            "text": self.to_text,
            "latex": self.to_latex,
            "tex": self.to_latex,
            "html": self.to_html,
            "markdown": self.to_markdown,
            "md": self.to_markdown,
            "csv": self.to_csv,
        }
        return renderers.get(fmt, self.to_text)()

    def __str__(self) -> str:
        return self._render(self._output)

    def __repr__(self) -> str:
        return self.__str__()

    def _repr_html_(self) -> str:
        return self.to_html()


def _resolve_se_type(se: bool, t: bool, p: bool, ci: bool) -> str:
    """Map esttab's four booleans to regtable's ``se_type=`` string.

    Priority ``ci > p > t > se`` matches the legacy ``EstimateTable``
    behaviour where the first true flag wins.
    """
    if ci:
        return "ci"
    if p:
        return "p"
    if t:
        return "t"
    return "se"


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
    Stata-style ``esttab`` — thin facade over :func:`sp.regtable`.

    .. deprecated::
        Now a thin wrapper over :func:`statspai.output.regtable`. Use
        ``sp.regtable(*models, ...)`` directly for full control. See
        the module docstring for the parameter mapping.

    Accepts model results directly as positional arguments **or** reads
    from the global store populated by :func:`eststo`.  If both
    positional arguments and a non-empty store exist, positional
    arguments take precedence.

    Parameters mirror the original ``esttab`` API; see the module
    docstring for the exact mapping to ``regtable``.
    """
    _warn_once_esttab()

    # Resolve models: positional args > global store
    if results:
        model_list: List[Any] = list(results)
        if names:
            name_list = list(names)
        else:
            name_list = [f"({i + 1})" for i in range(len(model_list))]
    elif _STORE:
        name_list = [n for n, _ in _STORE]
        model_list = [r for _, r in _STORE]
        if names:
            name_list = list(names)
    else:
        warnings.warn(
            "No models to tabulate. Pass results directly or use eststo() first."
        )
        # Construct an empty-but-valid wrapper so callers can still
        # call str() / to_text() without crashing.
        from .regression_table import regtable
        try:
            empty = regtable(
                [], title=title, notes=list(notes) if notes else None
            )
            return EstimateTableResult(empty, output)
        except Exception:
            # If regtable rejects empty input, surface a minimal
            # placeholder so the call site still gets a printable object.
            class _Empty:
                def to_text(self):
                    return "(no models)"
                to_latex = to_html = to_markdown = to_text
                def to_dataframe(self):
                    return pd.DataFrame()
            return EstimateTableResult(_Empty(), output)

    if len(name_list) != len(model_list):
        raise ValueError(
            f"Length mismatch: {len(model_list)} models but "
            f"{len(name_list)} names."
        )

    # Build the regtable call ─────────────────────────────────────────
    from .regression_table import regtable

    se_type = _resolve_se_type(se, t, p, ci)

    kwargs: Dict[str, Any] = dict(
        model_labels=name_list,
        title=title,
        notes=list(notes) if notes else None,
        stars=stars,
        star_levels=star_levels,
        se_type=se_type,
        fmt=fmt,
        coef_labels=labels,
        keep=list(keep) if keep else None,
        drop=list(drop) if drop else None,
        order=list(order) if order else None,
        alpha=alpha,
    )
    if stats is not None:
        kwargs["stats"] = list(stats)

    rt = regtable(model_list, **kwargs)
    result_obj = EstimateTableResult(rt, output)

    # Write to file if requested
    if filename:
        path = Path(filename)
        if output == "text":
            ext_map = {
                ".tex": "latex", ".html": "html", ".htm": "html",
                ".md": "markdown", ".csv": "csv",
            }
            detected = ext_map.get(path.suffix.lower())
            if detected:
                result_obj = EstimateTableResult(rt, detected)
        path.write_text(str(result_obj), encoding="utf-8")

    # No auto-print: Jupyter uses _repr_html_, REPL uses __repr__.
    return result_obj
