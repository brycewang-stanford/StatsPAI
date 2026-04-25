"""Inline coefficient reporting — :func:`sp.cite`.

Reproducibility-focused authors want to drop a coefficient straight into the
manuscript prose:

    The treatment effect is **0.234*** (0.041)** on log earnings.

Doing this by hand means re-typing four numbers (estimate, SE, stars,
optional p-value) every time, which decays into stale text the moment the
spec changes. ``sp.cite(result, "treat")`` reads the live result object,
formats the cell exactly like the table renderer would, and returns a
plain string ready for f-strings, Jupyter Markdown cells, Quarto inline
expressions, or copy-paste into a Word document.

Supported result types
----------------------
- :class:`statspai.core.results.EconometricResults` — pass any coefficient
  name available in ``result.params``.
- :class:`statspai.core.results.CausalResult` — the ``term`` argument is
  optional; when omitted the headline ``estimand`` (e.g. ``ATT``) is used.

Output formats
--------------
- ``"text"`` (default) — ``"0.234*** (0.041)"``.
- ``"latex"`` — ``"0.234^{***}~(0.041)"`` (math-mode safe; uses ``~`` for
  a non-breaking thin space between estimate and SE).
- ``"markdown"`` — ``"**0.234**\\* (0.041)"`` with the estimate bolded.
- ``"html"`` — ``"<b>0.234</b><sup>***</sup> (0.041)"``.

Examples
--------
>>> import statspai as sp
>>> m = sp.regress("y ~ x + treat", data=df)
>>> sp.cite(m, "treat")
'0.234*** (0.041)'
>>> sp.cite(m, "treat", fmt="%.4f")
'0.2341*** (0.0413)'
>>> sp.cite(m, "treat", second_row="ci")  # show 95% CI instead of SE
'0.234*** [0.153, 0.315]'
>>> sp.cite(m, "treat", output="latex")
'0.234^{***}~(0.041)'
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as _sp_stats


# Reuse the same star/format helpers that drive the table renderer so the
# inline output and the in-table cell never disagree.
from .estimates import _format_stars, _fmt_val


_VALID_OUTPUTS = {"text", "latex", "markdown", "md", "html"}
_VALID_SECOND = {"se", "t", "p", "ci", "none"}


def _resolve_term(result, term: Optional[str]) -> str:
    """Pick the term to cite from a result object."""
    if term is not None:
        return term
    if hasattr(result, "estimand"):
        return getattr(result, "estimand")
    if hasattr(result, "params"):
        params = getattr(result, "params")
        if hasattr(params, "index") and len(params.index) > 0:
            return str(params.index[0])
    raise ValueError(
        "Could not infer which term to cite. Pass `term=...` explicitly."
    )


def _extract_point(result, term: str) -> Tuple[float, float, float, Optional[Tuple[float, float]], Optional[int]]:
    """Return ``(estimate, se, pvalue, ci, df_resid)`` for *term*.

    Works on both ``EconometricResults`` (pull from ``params`` /
    ``std_errors`` / ``pvalues`` / ``conf_int``) and ``CausalResult``
    (pull from scalar attributes).
    """
    # CausalResult path
    if hasattr(result, "estimand") and hasattr(result, "estimate") and hasattr(result, "se"):
        if term != getattr(result, "estimand", None) and term not in (None, ""):
            warnings.warn(
                f"CausalResult exposes only one term ({result.estimand!r}); "
                f"requested term={term!r} is ignored.",
                stacklevel=3,
            )
        est = float(result.estimate)
        se = float(result.se)
        pv = float(result.pvalue) if getattr(result, "pvalue", None) is not None else np.nan
        ci_attr = getattr(result, "ci", None)
        ci = (float(ci_attr[0]), float(ci_attr[1])) if ci_attr is not None else None
        df_resid = None
        mi = getattr(result, "model_info", None) or {}
        if isinstance(mi, dict):
            df_resid = mi.get("df_resid")
        return est, se, pv, ci, df_resid

    # EconometricResults path
    params = getattr(result, "params", None)
    if params is None:
        raise KeyError("Result has no `params` attribute.")
    if not isinstance(params, pd.Series):
        params = pd.Series(params)
    if term not in params.index:
        raise KeyError(
            f"Term {term!r} not found in result.params. "
            f"Available: {list(params.index)}"
        )
    est = float(params[term])

    se_raw = getattr(result, "std_errors", None)
    if se_raw is None:
        raise KeyError(f"std_errors missing for term {term!r}.")
    if not isinstance(se_raw, pd.Series):
        se_raw = pd.Series(se_raw, index=params.index)
    if term not in se_raw.index:
        raise KeyError(f"std_errors missing for term {term!r}.")
    se = float(se_raw[term])

    # p-values may be Series, ndarray, or absent.
    pv_raw = getattr(result, "pvalues", None)
    if pv_raw is None:
        pv = np.nan
    else:
        if not isinstance(pv_raw, pd.Series):
            pv_raw = pd.Series(pv_raw, index=params.index)
        pv = float(pv_raw[term]) if term in pv_raw.index else np.nan

    # Confidence-interval bounds.
    ci_lo_raw = getattr(result, "conf_int_lower", None)
    ci_hi_raw = getattr(result, "conf_int_upper", None)
    if ci_lo_raw is not None and not isinstance(ci_lo_raw, pd.Series):
        ci_lo_raw = pd.Series(ci_lo_raw, index=params.index)
    if ci_hi_raw is not None and not isinstance(ci_hi_raw, pd.Series):
        ci_hi_raw = pd.Series(ci_hi_raw, index=params.index)
    if (
        ci_lo_raw is not None and ci_hi_raw is not None
        and term in ci_lo_raw.index and term in ci_hi_raw.index
    ):
        ci = (float(ci_lo_raw[term]), float(ci_hi_raw[term]))
    else:
        ci = None

    df_resid = (
        getattr(result, "df_resid", None)
        or (getattr(result, "data_info", {}) or {}).get("df_resid")
        or (getattr(result, "diagnostics", {}) or {}).get("df_resid")
    )
    return est, se, pv, ci, df_resid


def _ci_at_alpha(
    est: float, se: float, ci: Optional[Tuple[float, float]], alpha: float, df_resid
) -> Tuple[float, float]:
    """Get CI bounds at *alpha*. Reuses 95% from result when alpha=0.05."""
    if abs(alpha - 0.05) < 1e-12 and ci is not None and all(np.isfinite(ci)):
        return ci
    if not np.isfinite(se):
        return (np.nan, np.nan)
    if df_resid is not None and np.isfinite(float(df_resid)) and float(df_resid) > 0:
        crit = _sp_stats.t.ppf(1 - alpha / 2, float(df_resid))
    else:
        crit = _sp_stats.norm.ppf(1 - alpha / 2)
    return (est - crit * se, est + crit * se)


def _wrap_stars(stars: str, output: str) -> str:
    """Wrap stars in superscript markup appropriate to *output*.

    For Markdown output the literal ``*`` would collide with the bold
    delimiters surrounding the estimate, so we escape each star as ``\\*``
    so renderers display it as a real asterisk.
    """
    if not stars:
        return ""
    if output == "latex":
        return f"^{{{stars}}}"
    if output == "html":
        return f"<sup>{stars}</sup>"
    if output in ("markdown", "md"):
        return "\\*" * len(stars)
    return stars


def _wrap_estimate(text: str, output: str) -> str:
    if output == "html":
        return f"<b>{text}</b>"
    if output in ("markdown", "md"):
        return f"**{text}**"
    return text


def cite(
    result,
    term: Optional[str] = None,
    *,
    fmt: str = "%.3f",
    output: str = "text",
    star_levels: Tuple[float, ...] = (0.10, 0.05, 0.01),
    second_row: str = "se",
    alpha: float = 0.05,
    bold_estimate: bool = False,
) -> str:
    """Format a single coefficient as an inline citation string.

    Parameters
    ----------
    result : object
        Any StatsPAI result with either ``params``/``std_errors`` (econometric)
        or ``estimate``/``se`` (causal).
    term : str, optional
        Coefficient name. Defaults to the headline ``estimand`` for causal
        results, or the first row of ``params`` for econometric results.
    fmt : str, default ``"%.3f"``
        ``printf``-style format string.
    output : str, default ``"text"``
        One of ``"text"``, ``"latex"``, ``"markdown"``/``"md"``, ``"html"``.
    star_levels : tuple of float, default ``(0.10, 0.05, 0.01)``
        Star thresholds — same convention as :func:`sp.regtable`.
    second_row : str, default ``"se"``
        What to put in parentheses after the estimate. One of:

        - ``"se"`` — standard error in ``(...)``.
        - ``"t"``  — t-statistic in ``(...)``.
        - ``"p"``  — p-value in ``(...)``.
        - ``"ci"`` — confidence interval in ``[lo, hi]``.
        - ``"none"`` — omit the second row entirely.

    alpha : float, default 0.05
        CI level when ``second_row="ci"``.
    bold_estimate : bool, default False
        For ``output="text"`` / ``"latex"``, whether to bold the estimate
        (HTML / Markdown bold the estimate by default for readability).

    Returns
    -------
    str
        The formatted inline citation string.
    """
    if output not in _VALID_OUTPUTS:
        raise ValueError(
            f"output={output!r} invalid. Choose from {sorted(_VALID_OUTPUTS)}."
        )
    if second_row not in _VALID_SECOND:
        raise ValueError(
            f"second_row={second_row!r} invalid. Choose from {sorted(_VALID_SECOND)}."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")

    term_resolved = _resolve_term(result, term)
    est, se, pv, ci, df_resid = _extract_point(result, term_resolved)

    est_txt = _fmt_val(est, fmt)
    stars = _format_stars(pv, star_levels) if pv is not None else ""
    stars_txt = _wrap_stars(stars, output)

    if bold_estimate or output in ("html", "markdown", "md"):
        est_render = _wrap_estimate(est_txt, output)
    else:
        est_render = est_txt

    # Second-row component
    if second_row == "none":
        second = ""
    elif second_row == "ci":
        lo, hi = _ci_at_alpha(est, se, ci, alpha, df_resid)
        second = f" [{_fmt_val(lo, fmt)}, {_fmt_val(hi, fmt)}]"
    elif second_row == "t":
        t = est / se if se and np.isfinite(se) and se != 0 else np.nan
        second = f" ({_fmt_val(t, fmt)})"
    elif second_row == "p":
        second = f" ({_fmt_val(pv, fmt)})"
    else:  # "se"
        second = f" ({_fmt_val(se, fmt)})"

    # Assemble: estimate + stars + (second row)
    if output == "latex":
        # In LaTeX inline math we want "0.234^{***}~(0.041)" — the ~ is a
        # non-breaking thin space which keeps estimate and SE on the same line.
        # When `second` is non-empty we strip the leading ASCII space so the
        # tilde controls spacing.
        sep = "~" if second else ""
        return f"{est_render}{stars_txt}{sep}{second.lstrip()}"

    return f"{est_render}{stars_txt}{second}"
