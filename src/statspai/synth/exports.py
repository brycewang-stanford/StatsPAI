"""
Formatted exports for Synthetic Control results.

Three table-export formats with consistent SCM-specific structure:

* :func:`synth_to_latex` — booktabs LaTeX, ready for JSS / AER / JASA
* :func:`synth_to_markdown` — GitHub-flavoured Markdown / pandoc
* :func:`synth_to_excel` — multi-sheet Excel workbook (estimates, weights,
  gap series, diagnostics)

All three accept either a single :class:`~statspai.core.results.CausalResult`
from any ``sp.synth(method=...)`` variant, or a
:class:`~statspai.synth.compare.SynthComparison` from
:func:`sp.synth_compare`. Output structure adapts automatically.

Notes
-----
The point estimate, standard error, pre-RMSPE, and donor weights all come
from the result's ``model_info`` dict, which every SCM variant in
StatsPAI populates following the schema documented in
``synth/scm.py``. Methods that do not record a particular field (e.g.
non-standard SCM variants without an explicit donor pool) are exported
with ``NaN`` in those columns rather than silently dropping them.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from ..core.results import CausalResult

if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    from .compare import SynthComparison


# ====================================================================== #
#  Field extractors (shared with compare.py / report.py)
# ====================================================================== #


def _model_info(result: CausalResult) -> Dict[str, Any]:
    return getattr(result, "model_info", {}) or {}


def _pre_rmspe(result: CausalResult) -> float:
    """Extract pre-treatment RMSPE from any SCM variant.

    Falls back to recomputing from ``Y_obs`` / ``Y_synth`` (the
    convention used by SDID and the synthdid framework) when the
    estimator does not record an explicit pre-RMSPE field.
    """
    mi = _model_info(result)
    mspe = mi.get("pre_treatment_mspe")
    if mspe is not None:
        return float(np.sqrt(mspe))
    for key in ("pre_treatment_rmspe", "pre_treatment_rmse", "pre_rmspe", "pre_rmse"):
        val = mi.get(key)
        if val is not None:
            return float(val)

    # Fall back to recomputing from gap series
    gap = _gap_table(result)
    if gap is not None and "gap" in gap.columns:
        if "post_treatment" in gap.columns:
            pre = gap.loc[~gap["post_treatment"], "gap"]
        else:
            pre = gap["gap"]
        if len(pre) > 0:
            arr = np.asarray(pre, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) > 0:
                return float(np.sqrt(np.mean(arr**2)))
    return float("nan")


def _post_rmspe(result: CausalResult) -> float:
    mi = _model_info(result)
    for key in ("post_treatment_rmspe", "post_rmspe", "post_treatment_rmse"):
        val = mi.get(key)
        if val is not None:
            return float(val)
    gap_table = _gap_table(result)
    if gap_table is None or "gap" not in gap_table.columns:
        return float("nan")
    if "post_treatment" in gap_table.columns:
        post = gap_table.loc[gap_table["post_treatment"], "gap"]
    else:
        post = gap_table["gap"]
    arr = np.asarray(post, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan")  # pragma: no cover
    return float(np.sqrt(np.mean(arr**2)))


def _fit_quality_pct(result: CausalResult) -> Tuple[float, str]:
    """Return (pct_of_outcome_sd, qualitative_label).

    The "fit quality" rule of thumb (Abadie, Diamond & Hainmueller 2010,
    §III) compares pre-treatment RMSPE to the standard deviation of the
    treated unit's pre-treatment outcome. < 5%: excellent, 5–10% good,
    10–20% acceptable, > 20% poor — pre-fit beyond ~20% of the SD
    indicates the donor pool cannot reproduce the treated trajectory and
    casts doubt on any post-period estimate.
    """
    pre = _pre_rmspe(result)
    if np.isnan(pre):
        return float("nan"), "n/a"
    gap_table = _gap_table(result)
    if gap_table is None or "treated" not in gap_table.columns:
        return float("nan"), "n/a"
    if "post_treatment" in gap_table.columns:
        pre_rows = gap_table[~gap_table["post_treatment"]]
    else:
        pre_rows = gap_table
    if len(pre_rows) == 0:
        return float("nan"), "n/a"  # pragma: no cover
    sd = float(pre_rows["treated"].std())
    if sd <= 0 or np.isnan(sd):
        return float("nan"), "n/a"  # pragma: no cover
    pct = pre / sd * 100.0
    if pct < 5:
        label = "excellent"
    elif pct < 10:
        label = "good"
    elif pct < 20:
        label = "acceptable"
    else:
        label = "poor"
    return pct, label


def _donor_weights(result: CausalResult) -> Dict[str, float]:
    """Extract donor weights as ``{name: weight}`` (may be empty).

    Handles the four storage conventions used across SCM variants:

    * ``dict`` (e.g. scpi)
    * ``pd.Series`` indexed by donor name
    * ``pd.DataFrame`` with two columns (the first being unit / donor
      name, the second the weight) — the convention used by the
      classic dispatcher and most variants
    * ``np.ndarray`` paired with a ``donor_units`` / ``donor_names``
      list in ``model_info``
    """
    mi = _model_info(result)
    for key in ("donor_weights", "weights", "unit_weights", "omega", "omega_weights"):
        w = mi.get(key)
        if isinstance(w, dict):
            return {str(k): float(v) for k, v in w.items()}
        if isinstance(w, pd.Series):
            return {str(k): float(v) for k, v in w.items()}
        if isinstance(w, pd.DataFrame) and len(w.columns) >= 2:
            name_col = w.columns[0]
            weight_col = next(
                (
                    c
                    for c in w.columns
                    if c != name_col and pd.api.types.is_numeric_dtype(w[c])
                ),
                w.columns[1],
            )
            return {str(n): float(v) for n, v in zip(w[name_col], w[weight_col])}
        if isinstance(w, (np.ndarray, list)):
            for names_key in ("donor_units", "donor_names", "control_units", "donors"):
                names = mi.get(names_key)
                if names is not None and len(names) == len(w):
                    return {str(n): float(v) for n, v in zip(names, w)}
    return {}


def _gap_table(result: CausalResult) -> Optional[pd.DataFrame]:
    """Return a per-period DataFrame ``[time, treated, synthetic, gap, post_treatment]``.

    Reconstructs from ``Y_treated`` / ``Y_synth`` (or SDID's ``Y_obs``
    / ``Y_synth``) when an estimator does not pre-build ``gap_table``.
    """
    mi = _model_info(result)
    g = mi.get("gap_table")
    if isinstance(g, pd.DataFrame):
        return g.copy()

    # Try treated/synthetic pair under several conventions
    yt = mi.get("Y_treated")
    if yt is None:
        yt = mi.get("Y_obs")  # SDID convention
    ys = mi.get("Y_synth")
    if yt is None or ys is None:
        return None

    # Normalise to np.array and pull index for time axis
    times: Any
    if isinstance(yt, pd.Series):
        times = list(yt.index)
        yt_arr = yt.to_numpy(dtype=float)
    else:
        times = mi.get("times") or mi.get("all_times")
        yt_arr = np.asarray(yt, dtype=float)
    if isinstance(ys, pd.Series):
        ys_arr = ys.to_numpy(dtype=float)
    else:
        ys_arr = np.asarray(ys, dtype=float)

    if times is None or len(times) != len(yt_arr):
        return None  # pragma: no cover

    df = pd.DataFrame(
        {
            "time": list(times),
            "treated": yt_arr,
            "synthetic": ys_arr,
        }
    )
    df["gap"] = df["treated"] - df["synthetic"]

    treatment_time = mi.get("treatment_time")
    if treatment_time is None:
        treatment_time = mi.get("treat_time")
    if treatment_time is not None:
        df["post_treatment"] = df["time"] >= treatment_time
    return df


def _stars(pvalue: float) -> str:
    if pvalue is None or np.isnan(pvalue):
        return ""
    if pvalue < 0.01:
        return "$^{***}$"
    if pvalue < 0.05:
        return "$^{**}$"
    if pvalue < 0.1:
        return "$^{*}$"
    return ""


def _stars_md(pvalue: float) -> str:
    if pvalue is None or np.isnan(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return ""


def _format_estimate_se(
    estimate: float,
    se: float,
    pvalue: float,
    *,
    latex: bool = True,
    digits: int = 4,
) -> Tuple[str, str]:
    if np.isnan(estimate):
        return "—", "—"
    star = _stars(pvalue) if latex else _stars_md(pvalue)
    if np.isnan(se):
        return f"{estimate:.{digits}f}{star}", "—"
    return f"{estimate:.{digits}f}{star}", f"({se:.{digits}f})"


def _result_summary_row(
    result: CausalResult,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compact one-row dict capturing the most-cited SCM diagnostics."""
    mi = _model_info(result)
    ci = getattr(result, "ci", (np.nan, np.nan)) or (np.nan, np.nan)
    pct, label = _fit_quality_pct(result)
    weights = _donor_weights(result)
    if weights:
        n_eff = int(sum(1 for w in weights.values() if abs(w) > 0.01))
    else:
        n_eff = mi.get("n_active_donors", mi.get("effective_n_donors", np.nan))
        if isinstance(n_eff, float) and not np.isnan(n_eff):
            n_eff = int(round(n_eff))

    # Period / donor counts: try canonical names, then SDID conventions
    n_pre = mi.get("n_pre_periods", mi.get("T_pre", np.nan))
    n_post = mi.get("n_post_periods", mi.get("T_post", np.nan))
    n_donors = mi.get("n_donors", mi.get("n_control", np.nan))

    return {
        "name": name or getattr(result, "method", "SCM"),
        "att": float(getattr(result, "estimate", np.nan)),
        "se": float(getattr(result, "se", np.nan)),
        "pvalue": float(getattr(result, "pvalue", np.nan)),
        "ci_lower": float(ci[0]) if ci is not None else np.nan,
        "ci_upper": float(ci[1]) if ci is not None else np.nan,
        "pre_rmspe": _pre_rmspe(result),
        "post_rmspe": _post_rmspe(result),
        "fit_pct_sd": pct,
        "fit_label": label,
        "n_pre_periods": n_pre,
        "n_post_periods": n_post,
        "n_donors": n_donors,
        "n_effective_donors": n_eff if n_eff is not None else np.nan,
    }


# ====================================================================== #
#  LaTeX export
# ====================================================================== #


def synth_to_latex(
    obj: Union[CausalResult, "SynthComparison", List[CausalResult]],
    *,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    booktabs: bool = True,
    show_ci: bool = True,
    show_weights: bool = False,
    top_n_weights: int = 5,
    digits: int = 4,
    method_names: Optional[Sequence[str]] = None,
) -> str:
    """Formatted LaTeX table for synthetic-control results.

    Single-result mode produces a vertical table with ATT, SE,
    confidence interval, pre-RMSPE, fit quality, and (optionally) the
    top-N donor weights. Comparison mode (``SynthComparison`` or list
    of results) produces a wide table with one column per method, the
    standard textbook layout for empirical applied work.

    Parameters
    ----------
    obj : CausalResult, SynthComparison, or list of CausalResult
        Object to render. ``SynthComparison`` and lists trigger the
        side-by-side multi-method layout.
    caption : str, optional
        Table caption. Defaults to a sensible auto-generated string.
    label : str, optional
        LaTeX label for cross-referencing. Defaults to
        ``"tab:synth"`` (single) or ``"tab:synth_compare"`` (multi).
    booktabs : bool, default True
        If True, use ``\\toprule`` / ``\\midrule`` / ``\\bottomrule``
        (requires ``\\usepackage{booktabs}``). Falls back to
        ``\\hline`` if False.
    show_ci : bool, default True
        Include the confidence-interval row.
    show_weights : bool, default False
        Append a panel listing the top-N donor weights.
    top_n_weights : int, default 5
        How many donors to show per method when ``show_weights=True``.
    digits : int, default 4
        Number of decimal places.
    method_names : list of str, optional
        Override column labels in comparison mode.

    Returns
    -------
    str
        LaTeX source ready to drop into a paper. Stars use the
        standard ``* p<0.1, ** p<0.05, *** p<0.01`` convention.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> units, periods = 12, 10
    >>> unit = np.repeat(np.arange(units), periods)
    >>> time = np.tile(np.arange(1, periods + 1), units)
    >>> d = (unit == 0) & (time >= 7)
    >>> y = (unit * 0.4 + time * 0.2 + 3.0 * d
    ...      + rng.normal(0, 0.3, unit.size))
    >>> df = pd.DataFrame({"y": y, "unit": unit, "time": time})
    >>> res = sp.synth(
    ...     df, outcome="y", unit="unit", time="time",
    ...     treated_unit=0, treatment_time=7, method="classic",
    ...     placebo=False,
    ... )
    >>> tex = sp.synth_to_latex(res, show_weights=True)
    >>> isinstance(tex, str)
    True

    Multi-method comparison:

    >>> comp = sp.synth_compare(
    ...     df, outcome="y", unit="unit", time="time",
    ...     treated_unit=0, treatment_time=7,
    ...     methods=['classic', 'sdid'], placebo=False,
    ... )
    >>> tex = sp.synth_to_latex(comp, caption='SCM benchmark')
    >>> isinstance(tex, str)
    True
    """
    results, names = _normalise(obj, method_names)
    multi = len(results) > 1

    rule_top = "\\toprule" if booktabs else "\\hline\\hline"
    rule_mid = "\\midrule" if booktabs else "\\hline"
    rule_bot = "\\bottomrule" if booktabs else "\\hline\\hline"

    # Header
    if multi:
        cap = caption or "Synthetic Control: multi-method comparison"
        lab = label or "tab:synth_compare"
    else:
        cap = caption or f"Synthetic Control results — {names[0]}"
        lab = label or "tab:synth"

    rows = _result_summary_row
    summaries = [rows(r, n) for r, n in zip(results, names)]

    n_cols = 1 + len(results)
    spec = "l" + "c" * len(results)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{cap}}}",
        f"\\label{{{lab}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        rule_top,
        " & ".join([""] + [_latex_escape(n) for n in names]) + " \\\\",
        rule_mid,
    ]

    # ATT / SE rows
    att_cells = [""]
    se_cells = [""]
    for s in summaries:
        est, se = _format_estimate_se(
            s["att"],
            s["se"],
            s["pvalue"],
            latex=True,
            digits=digits,
        )
        att_cells.append(est)
        se_cells.append(se)
    lines.append("ATT & " + " & ".join(att_cells[1:]) + " \\\\")
    lines.append("    & " + " & ".join(se_cells[1:]) + " \\\\")

    if show_ci:
        ci_cells = []
        for s in summaries:
            lo, hi = s["ci_lower"], s["ci_upper"]
            if np.isnan(lo) or np.isnan(hi):
                ci_cells.append("—")
            else:
                ci_cells.append(f"[{lo:.{digits}f}, {hi:.{digits}f}]")
        lines.append("95\\% CI & " + " & ".join(ci_cells) + " \\\\")

    lines.append(rule_mid)

    # Diagnostics
    def _fmt_int(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return str(int(v))

    def _fmt_float(v: Any, d: int = digits) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{float(v):.{d}f}"

    lines.append(
        "Pre-RMSPE & "
        + " & ".join(_fmt_float(s["pre_rmspe"]) for s in summaries)
        + " \\\\"
    )
    lines.append(
        "Fit (\\% of outcome SD) & "
        + " & ".join(
            (
                "—"
                if np.isnan(s["fit_pct_sd"])
                else f"{s['fit_pct_sd']:.1f}\\% ({s['fit_label']})"
            )
            for s in summaries
        )
        + " \\\\"
    )
    lines.append(
        "Pre-treatment $T_0$ & "
        + " & ".join(_fmt_int(s["n_pre_periods"]) for s in summaries)
        + " \\\\"
    )
    lines.append(
        "Post-treatment $T_1$ & "
        + " & ".join(_fmt_int(s["n_post_periods"]) for s in summaries)
        + " \\\\"
    )
    lines.append(
        "Donor pool $J$ & "
        + " & ".join(_fmt_int(s["n_donors"]) for s in summaries)
        + " \\\\"
    )
    lines.append(
        "Effective donors & "
        + " & ".join(_fmt_int(s["n_effective_donors"]) for s in summaries)
        + " \\\\"
    )

    # Optional weights panel
    if show_weights:
        lines.append(rule_mid)
        lines.append(
            "\\multicolumn{" + str(n_cols) + "}{l}{\\emph{Top donor weights}} \\\\"
        )
        per_method: List[Any] = []
        max_rows = 0
        for r in results:
            wmap = _donor_weights(r)
            if not wmap:
                per_method.append([])
                continue
            top = sorted(
                wmap.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:top_n_weights]
            per_method.append(top)
            max_rows = max(max_rows, len(top))
        for i in range(max_rows):
            cells = [f"  Donor {i + 1}"]
            for top in per_method:
                if i < len(top):
                    name, w = top[i]
                    cells.append(f"{_latex_escape(str(name))} ({w:.{digits}f})")
                else:
                    cells.append("")
            lines.append(" & ".join(cells) + " \\\\")

    lines += [
        rule_bot,
        "\\end{tabular}",
        "\\begin{tablenotes}",
        "\\footnotesize",
        "\\item Standard errors in parentheses.",
        "\\item Significance: $^{*}$ p<0.1, $^{**}$ p<0.05, $^{***}$ p<0.01.",
        "\\end{tablenotes}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    """Minimal LaTeX escape for table cells / labels."""
    repl = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
        "\\": "\\textbackslash{}",
    }
    out = []
    for ch in str(text):
        out.append(repl.get(ch, ch))
    return "".join(out)


# ====================================================================== #
#  Markdown export
# ====================================================================== #


def synth_to_markdown(
    obj: Union[CausalResult, "SynthComparison", List[CausalResult]],
    *,
    title: Optional[str] = None,
    show_ci: bool = True,
    show_weights: bool = False,
    top_n_weights: int = 5,
    digits: int = 4,
    method_names: Optional[Sequence[str]] = None,
) -> str:
    """GitHub-flavoured Markdown table for synthetic-control results.

    Mirrors :func:`synth_to_latex` in scope but emits a pipe-delimited
    Markdown table that renders cleanly on GitHub, in pandoc, and in
    most static-site generators.

    Parameters
    ----------
    obj, title, show_ci, show_weights, top_n_weights, digits, method_names
        See :func:`synth_to_latex`.

    Returns
    -------
    str
        Markdown source.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> units, periods = 12, 10
    >>> unit = np.repeat(np.arange(units), periods)
    >>> time = np.tile(np.arange(1, periods + 1), units)
    >>> d = (unit == 0) & (time >= 7)
    >>> y = (unit * 0.4 + time * 0.2 + 3.0 * d
    ...      + rng.normal(0, 0.3, unit.size))
    >>> df = pd.DataFrame({"y": y, "unit": unit, "time": time})
    >>> res = sp.synth(
    ...     df, outcome="y", unit="unit", time="time",
    ...     treated_unit=0, treatment_time=7, method="classic",
    ...     placebo=False,
    ... )
    >>> md = sp.synth_to_markdown(res)
    >>> isinstance(md, str)
    True
    >>> "**ATT**" in md
    True
    """
    results, names = _normalise(obj, method_names)
    multi = len(results) > 1

    summaries = [_result_summary_row(r, n) for r, n in zip(results, names)]

    head = title or (
        "Synthetic Control: multi-method comparison"
        if multi
        else f"Synthetic Control — {names[0]}"
    )
    lines = [f"### {head}", ""]

    # Header row
    cols = ["Statistic"] + names
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    # ATT / SE row (combine into single cell with "estimate (SE)***")
    cells = ["**ATT**"]
    for s in summaries:
        est, se = _format_estimate_se(
            s["att"],
            s["se"],
            s["pvalue"],
            latex=False,
            digits=digits,
        )
        cells.append(f"{est} {se}")
    lines.append("| " + " | ".join(cells) + " |")

    if show_ci:
        cells = ["95% CI"]
        for s in summaries:
            lo, hi = s["ci_lower"], s["ci_upper"]
            cells.append(
                "—"
                if np.isnan(lo) or np.isnan(hi)
                else f"[{lo:.{digits}f}, {hi:.{digits}f}]"
            )
        lines.append("| " + " | ".join(cells) + " |")

    def _fmt(v: Any, d: int = digits) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.{d}f}"

    rows: List[Tuple[str, str]] = [
        ("Pre-RMSPE", "pre_rmspe"),
        ("Pre-treatment T₀", "n_pre_periods"),
        ("Post-treatment T₁", "n_post_periods"),
        ("Donor pool J", "n_donors"),
        ("Effective donors", "n_effective_donors"),
    ]
    for label, key in rows:
        cells = [label]
        for s in summaries:
            cells.append(_fmt(s[key]))
        lines.append("| " + " | ".join(cells) + " |")

    # Fit-quality row
    cells = ["Fit quality"]
    for s in summaries:
        if np.isnan(s["fit_pct_sd"]):
            cells.append("—")
        else:
            cells.append(f"{s['fit_pct_sd']:.1f}% of SD ({s['fit_label']})")
    lines.append("| " + " | ".join(cells) + " |")

    if show_weights:
        lines.append("")
        lines.append("**Top donor weights:**")
        lines.append("")
        for r, n in zip(results, names):
            wmap = _donor_weights(r)
            if not wmap:
                lines.append(f"- *{n}*: weights not exposed by this method")
                continue
            top = sorted(
                wmap.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:top_n_weights]
            weight_cells = ", ".join(f"{name} ({w:.{digits}f})" for name, w in top)
            lines.append(f"- *{n}*: {weight_cells}")

    lines.append("")
    lines.append("*Significance: \\* p<0.1, \\*\\* p<0.05, \\*\\*\\* p<0.01.*")
    return "\n".join(lines)


# ====================================================================== #
#  Excel export
# ====================================================================== #


def synth_to_excel(
    obj: Union[CausalResult, "SynthComparison", List[CausalResult]],
    path: str,
    *,
    method_names: Optional[Sequence[str]] = None,
    digits: int = 6,
) -> str:
    """Multi-sheet Excel workbook for synthetic-control results.

    Sheets
    ------
    * ``"Summary"`` — one row per method (ATT, SE, CI, pre-RMSPE, fit
      quality, donor counts).
    * ``"Weights"`` — donor weights per method (one column per method;
      missing donors are NaN).
    * ``"Gap_<method>"`` — per-period treated / synthetic / gap for
      each method.
    * ``"Diagnostics"`` — scalar diagnostics (pre-RMSPE, post/pre RMSPE
      ratio, fit quality, n_donors, etc.).

    Requires ``openpyxl`` (already a soft dependency of pandas
    Excel I/O). Will raise ``ModuleNotFoundError`` with an actionable
    hint if it is not installed.

    Parameters
    ----------
    obj : CausalResult, SynthComparison, or list of CausalResult
        Object to export.
    path : str
        Destination ``.xlsx`` file path.
    method_names : list of str, optional
        Override sheet / column labels.
    digits : int, default 6
        Rounding for floating-point values.

    Returns
    -------
    str
        Absolute path of the file that was written.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> units, periods = 12, 10
    >>> unit = np.repeat(np.arange(units), periods)
    >>> time = np.tile(np.arange(1, periods + 1), units)
    >>> d = (unit == 0) & (time >= 7)
    >>> y = (unit * 0.4 + time * 0.2 + 3.0 * d
    ...      + rng.normal(0, 0.3, unit.size))
    >>> df = pd.DataFrame({"y": y, "unit": unit, "time": time})
    >>> res = sp.synth(
    ...     df, outcome="y", unit="unit", time="time",
    ...     treated_unit=0, treatment_time=7, method="classic",
    ...     placebo=False,
    ... )
    >>> path = sp.synth_to_excel(
    ...     res, os.path.join(tempfile.mkdtemp(), "synth.xlsx")
    ... )
    >>> os.path.exists(path)
    True
    """
    try:
        import openpyxl  # noqa: F401  # availability check only
    except ModuleNotFoundError as exc:  # pragma: no cover - import path
        raise ModuleNotFoundError(
            "synth_to_excel requires `openpyxl`. Install it via "
            "`pip install openpyxl` or `pip install statspai[plotting]`."
        ) from exc

    results, names = _normalise(obj, method_names)
    summaries = [_result_summary_row(r, n) for r, n in zip(results, names)]

    summary_df = pd.DataFrame(summaries).round(digits)

    weights_long: List[pd.DataFrame] = []
    for r, n in zip(results, names):
        wmap = _donor_weights(r)
        if not wmap:
            continue
        df = pd.Series(wmap, name=n).rename_axis("donor").reset_index()
        df["method"] = n
        weights_long.append(df)
    if weights_long:
        weights_df = pd.concat(weights_long, ignore_index=True)
        weights_pivot = weights_df.pivot_table(
            index="donor",
            columns="method",
            values=n,
            aggfunc="first",
        ).round(digits)
        # ``aggfunc='first'`` will fall back to last-seen column if name
        # collides; rebuild explicitly to keep ordering.
        weights_pivot = pd.DataFrame(
            {n: pd.Series(_donor_weights(r)) for r, n in zip(results, names)}
        ).round(digits)
        weights_pivot.index.name = "donor"
    else:
        weights_pivot = pd.DataFrame(columns=names)

    diag_rows = []
    for s in summaries:
        diag_rows.append(
            {
                "method": s["name"],
                "pre_rmspe": s["pre_rmspe"],
                "post_rmspe": s["post_rmspe"],
                "post_pre_ratio": (
                    s["post_rmspe"] / s["pre_rmspe"]
                    if s["pre_rmspe"] and s["pre_rmspe"] > 0
                    else np.nan
                ),
                "fit_pct_sd": s["fit_pct_sd"],
                "fit_label": s["fit_label"],
                "n_pre": s["n_pre_periods"],
                "n_post": s["n_post_periods"],
                "n_donors": s["n_donors"],
                "n_effective_donors": s["n_effective_donors"],
            }
        )
    diagnostics_df = pd.DataFrame(diag_rows).round(digits)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        weights_pivot.to_excel(writer, sheet_name="Weights")
        diagnostics_df.to_excel(
            writer,
            sheet_name="Diagnostics",
            index=False,
        )
        for r, n in zip(results, names):
            gap = _gap_table(r)
            if gap is None:
                continue
            sheet = _safe_sheet_name(f"Gap_{n}")
            gap.round(digits).to_excel(writer, sheet_name=sheet, index=False)

    import os

    return os.path.abspath(path)


def _safe_sheet_name(name: str) -> str:
    """Excel sheet names cap at 31 chars and forbid certain characters."""
    bad = set(":\\/?*[]")
    cleaned = "".join("_" if ch in bad else ch for ch in name)
    return cleaned[:31]


# ====================================================================== #
#  Internal: normalise input
# ====================================================================== #


def _normalise(
    obj: Any,
    method_names: Optional[Sequence[str]] = None,
) -> Tuple[List[CausalResult], List[str]]:
    """Coerce input into ``(list_of_results, list_of_names)``."""
    # Avoid circular import by deferring SynthComparison reference
    from .compare import SynthComparison

    if isinstance(obj, CausalResult):
        results: List[CausalResult] = [obj]
        names: List[str] = (
            list(method_names) if method_names else [getattr(obj, "method", "SCM")]
        )
        return results, names

    if isinstance(obj, SynthComparison):
        results = list(obj.results.values())
        names = list(method_names) if method_names else list(obj.results.keys())
        return results, names

    if isinstance(obj, (list, tuple)):
        results = []
        names = []
        for i, item in enumerate(obj):
            if not isinstance(item, CausalResult):
                raise TypeError(
                    "synth_to_* expected CausalResult elements; got "
                    f"{type(item).__name__} at index {i}."
                )
            results.append(item)
            names.append(getattr(item, "method", f"Method {i + 1}"))
        if method_names is not None:
            if len(method_names) != len(results):
                raise ValueError("len(method_names) must match the number of results.")
            names = list(method_names)
        return results, names

    raise TypeError(
        "synth_to_* expected a CausalResult, SynthComparison, or list "
        f"of CausalResult; got {type(obj).__name__}."
    )
