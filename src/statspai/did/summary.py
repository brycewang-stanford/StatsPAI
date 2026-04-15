"""
One-call DID robustness summary across multiple staggered-DID estimators.

``did_summary()`` fits a common set of modern staggered-DID estimators to
the same data and returns a tidy comparison table with overall ATT, SE,
95 % CI, p-value, and a short note for each method. The goal is to make
method-robustness checks a single function call instead of hand-wiring
five different estimators.

Supported methods (all use the shared ``(y, group, time, first_treat)``
interface — ``group`` is the unit identifier, ``first_treat`` the cohort
variable):

================  =====================================================
Key               Underlying estimator
================  =====================================================
``'cs'``          Callaway & Sant'Anna (2021) + ``aggte(type='simple')``
``'sa'``          Sun & Abraham (2021) interaction-weighted
``'bjs'``         Borusyak, Jaravel & Spiess (2024) imputation
``'etwfe'``       Wooldridge (2021) extended TWFE
``'stacked'``     Cengiz et al. (2019) stacked event study
================  =====================================================

Example
-------
>>> import statspai as sp
>>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True, seed=0)
>>> summary = sp.did_summary(df, y='y', time='time',
...                          first_treat='first_treat', group='unit')
>>> print(summary.detail)
"""

from typing import Optional, List, Union

import numpy as np
import pandas as pd

from ..core.results import CausalResult


def _stars(p: float) -> str:
    """Significance stars for a p-value."""
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


class DIDSummaryResult(CausalResult):
    """
    CausalResult subclass returned by :func:`did_summary`.

    Defines ``.summary()`` as a real method (rather than a closure-bound
    instance attribute) so the result object serialises cleanly for
    caching (joblib / multiprocessing).

    Notes
    -----
    The ``se`` attribute stores the **cross-method dispersion** (SD of
    estimates across successfully-fit methods), *not* a sampling SE.
    Read :attr:`model_info` ``['dispersion']`` for the same value.
    """

    _DID_SUMMARY_MARKER: bool = True

    def summary(self, alpha: Optional[float] = None) -> str:  # noqa: ARG002
        mi = self.model_info or {}
        fit = mi.get("methods_fit", [])
        failed = mi.get("methods_failed", {})
        disp = mi.get("dispersion")
        bd = mi.get("breakdown_m")

        lines = ["=" * 78,
                 "  DID Method-Robustness Summary",
                 "=" * 78, ""]
        if fit:
            lines.append(f"  Fitted methods: {', '.join(fit)}")
        if failed:
            lines.append(f"  Failed:         {', '.join(failed.keys())}")
        if len(fit) >= 2 and not np.isnan(self.estimate):
            lines.append(f"  Mean ATT:       {self.estimate:+.4f}")
            if disp is not None and not np.isnan(disp):
                lines.append(f"  Cross-method SD: {disp:+.4f}")
        elif len(fit) == 1 and not np.isnan(self.estimate):
            lines.append(f"  ATT:            {self.estimate:+.4f}")
        if bd is not None:
            lines.append(f"  Breakdown M*:   {bd:.4f} (CS row)")
        lines.append("")
        lines.append(did_summary_to_markdown(self))
        return "\n".join(lines)


_DEFAULT_METHODS: List[str] = ["cs", "sa", "bjs", "etwfe", "stacked"]

_METHOD_LABELS = {
    "cs": "Callaway & Sant'Anna (2021)",
    "sa": "Sun & Abraham (2021)",
    "bjs": "Borusyak, Jaravel & Spiess (2024)",
    "etwfe": "Wooldridge (2021) ETWFE",
    "stacked": "Stacked DID (Cengiz et al. 2019)",
}


def _run_cs(data, y, group, time, first_treat, controls, cluster, alpha):
    from .callaway_santanna import callaway_santanna
    from .aggte import aggte

    cs = callaway_santanna(
        data, y=y, g=first_treat, t=time, i=group,
        x=controls, alpha=alpha,
    )
    return aggte(cs, type="simple", alpha=alpha, bstrap=False)


def _run_sa(data, y, group, time, first_treat, controls, cluster, alpha):
    from .sun_abraham import sun_abraham

    return sun_abraham(
        data, y=y, g=first_treat, t=time, i=group,
        covariates=controls, cluster=cluster, alpha=alpha,
    )


def _run_bjs(data, y, group, time, first_treat, controls, cluster, alpha):
    from .did_imputation import did_imputation

    return did_imputation(
        data, y=y, group=group, time=time, first_treat=first_treat,
        controls=controls, cluster=cluster, alpha=alpha,
    )


def _run_etwfe(data, y, group, time, first_treat, controls, cluster, alpha):
    from .wooldridge_did import etwfe

    return etwfe(
        data, y=y, group=group, time=time, first_treat=first_treat,
        controls=controls, cluster=cluster, alpha=alpha,
    )


def _run_stacked(data, y, group, time, first_treat, controls, cluster, alpha):
    from .stacked_did import stacked_did

    return stacked_did(
        data, y=y, group=group, time=time, first_treat=first_treat,
        controls=controls, cluster=cluster, alpha=alpha,
    )


_DISPATCH = {
    "cs": _run_cs,
    "sa": _run_sa,
    "bjs": _run_bjs,
    "etwfe": _run_etwfe,
    "stacked": _run_stacked,
}


def _extract(res: CausalResult) -> dict:
    """Pull (estimate, se, pvalue, ci_low, ci_high) from a CausalResult."""
    est = float(res.estimate) if res.estimate is not None else np.nan
    se = float(res.se) if res.se is not None else np.nan
    p = float(res.pvalue) if res.pvalue is not None else np.nan
    ci = res.ci if res.ci is not None else (np.nan, np.nan)
    ci_lo = float(ci[0]) if ci[0] is not None else np.nan
    ci_hi = float(ci[1]) if ci[1] is not None else np.nan
    n = int(res.n_obs) if getattr(res, "n_obs", None) else np.nan
    return dict(estimate=est, se=se, pvalue=p,
                ci_low=ci_lo, ci_high=ci_hi, n_obs=n)


def did_summary(
    data: pd.DataFrame,
    y: str,
    time: str,
    first_treat: str,
    group: str,
    methods: Union[str, List[str]] = "auto",
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    include_sensitivity: bool = False,
    verbose: bool = False,
) -> CausalResult:
    """
    One-call method-robustness comparison for staggered DID.

    Fits every requested estimator to the same data and returns a single
    :class:`CausalResult` whose ``detail`` attribute is a tidy comparison
    table — one row per method, columns ``(method, estimator, estimate,
    se, pvalue, ci_low, ci_high, n_obs, note)``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset (long format).
    y : str
        Outcome variable.
    time : str
        Time / period variable (integer-valued).
    first_treat : str
        First-treatment period per unit; NaN (or 0) for never-treated.
    group : str
        Unit identifier.
    methods : str or list of str, default ``'auto'``
        Methods to run. Valid keys: ``'cs'``, ``'sa'``, ``'bjs'``,
        ``'etwfe'``, ``'stacked'``, or ``'all'`` / ``'auto'`` for all.
    controls : list of str, optional
        Time-varying covariates passed to methods that support them.
    cluster : str, optional
        Cluster variable for SE (defaults to ``group`` in each sub-method).
    alpha : float, default 0.05
        Significance level for confidence intervals.
    include_sensitivity : bool, default False
        If ``True`` and ``'cs'`` is among the methods fit, compute the
        Rambachan–Roth (2023) *breakdown M\\** — the largest relative
        violation of parallel trends under which the treatment effect
        is still significantly different from zero. The value is added
        to ``model_info['breakdown_m']`` and to the ``breakdown_m``
        column of ``detail`` (CS row only; other methods leave ``NaN``).
    verbose : bool, default False
        Print progress for each method.

    Returns
    -------
    CausalResult
        ``estimate`` : mean of successfully-fit overall ATTs.
        ``se``       : standard deviation across methods (not a standard
                       error — a crude dispersion measure).
        ``detail``   : comparison DataFrame described above.
        ``model_info`` : ``{'methods_requested': [...], 'methods_fit':
                        [...], 'methods_failed': {name: error_msg, ...}}``.

    Notes
    -----
    Each method's overall ATT has slightly different interpretation:

    - CS ``aggte(type='simple')`` averages ATT(g, t) for post-treatment
      :math:`t \\geq g`, weighted by cohort size × exposure length.
    - SA / ETWFE / BJS / Stacked report cohort-size-weighted averages
      by construction.

    Differences across methods are informative about heterogeneity,
    model specification, and the sensitivity of conclusions to the
    estimator choice. Large disagreement is a red flag that deserves
    further investigation (e.g., via ``sp.bacon_decomposition`` or
    ``sp.honest_did``).

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=200, n_periods=10, staggered=True, seed=0)
    >>> out = sp.did_summary(df, y='y', time='time',
    ...                      first_treat='first_treat', group='unit')
    >>> out.summary()
    >>> print(out.detail[['method', 'estimate', 'se', 'pvalue']])
    """
    if methods in ("auto", "all"):
        methods_list = list(_DEFAULT_METHODS)
    elif isinstance(methods, str):
        methods_list = [methods]
    else:
        methods_list = list(methods)

    unknown = [m for m in methods_list if m not in _DISPATCH]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {unknown}. "
            f"Valid keys: {sorted(_DISPATCH)} (or 'auto' / 'all')."
        )

    # C4: validate column names up front so user typos surface as
    # crisp KeyErrors rather than 5 "method FAILED" rows.
    required = [y, time, first_treat, group]
    optional = list(controls or [])
    if cluster:
        optional.append(cluster)
    missing = [c for c in required + optional if c not in data.columns]
    if missing:
        raise KeyError(
            f"columns not found in data: {missing}. "
            f"Available columns: {list(data.columns)[:15]}"
            + (" ..." if len(data.columns) > 15 else "")
        )

    rows: List[dict] = []
    failed: dict = {}
    fit: List[str] = []
    cs_raw = None  # raw CS result for sensitivity analysis

    # Narrow exceptions that represent "the estimator couldn't fit this
    # data" rather than a bug in user input. Other exceptions propagate.
    _expected_exc = (
        np.linalg.LinAlgError,
        ValueError,
        RuntimeError,
        ZeroDivisionError,
    )

    for name in methods_list:
        label = _METHOD_LABELS[name]
        if verbose:
            print(f"  running {name} ({label})...", flush=True)
        try:
            if name == "cs" and include_sensitivity:
                from .callaway_santanna import callaway_santanna
                from .aggte import aggte as _aggte
                cs_raw = callaway_santanna(
                    data, y=y, g=first_treat, t=time, i=group,
                    x=controls, alpha=alpha,
                )
                res = _aggte(cs_raw, type="simple", alpha=alpha, bstrap=False)
            else:
                res = _DISPATCH[name](
                    data, y=y, group=group, time=time,
                    first_treat=first_treat, controls=controls,
                    cluster=cluster, alpha=alpha,
                )
            vals = _extract(res)
            rows.append(dict(method=name, estimator=label, note="", **vals))
            fit.append(name)
        except _expected_exc as exc:
            failed[name] = type(exc).__name__ + ": " + str(exc)[:160]
            rows.append(dict(
                method=name, estimator=label,
                estimate=np.nan, se=np.nan, pvalue=np.nan,
                ci_low=np.nan, ci_high=np.nan, n_obs=np.nan,
                note=f"FAILED: {type(exc).__name__}",
            ))

    # Optional Rambachan–Roth breakdown M*
    breakdown_m_value: Optional[float] = None
    breakdown_m_col: List[float] = [np.nan] * len(rows)
    if include_sensitivity and cs_raw is not None:
        try:
            from .honest_did import breakdown_m
            breakdown_m_value = float(breakdown_m(cs_raw, e=0, alpha=alpha))
            # Fill the CS row
            for i, r in enumerate(rows):
                if r["method"] == "cs":
                    breakdown_m_col[i] = breakdown_m_value
                    if r["note"] == "":
                        r["note"] = f"breakdown M* = {breakdown_m_value:.3f}"
                    break
        except Exception as exc:
            failed["__sensitivity__"] = (
                f"breakdown_m failed: {type(exc).__name__}: {str(exc)[:120]}"
            )

    for i, r in enumerate(rows):
        r["breakdown_m"] = breakdown_m_col[i]

    detail = pd.DataFrame(rows, columns=[
        "method", "estimator", "estimate", "se", "pvalue",
        "ci_low", "ci_high", "n_obs", "breakdown_m", "note",
    ])

    ests = detail.loc[detail["estimate"].notna(), "estimate"].values
    if len(ests) > 0:
        avg_est = float(np.mean(ests))
        disp = float(np.std(ests, ddof=1)) if len(ests) > 1 else np.nan
    else:
        avg_est, disp = np.nan, np.nan

    result = DIDSummaryResult(
        method="DID Method-Robustness Summary",
        estimand="Overall ATT (mean across methods)",
        estimate=avg_est,
        se=disp,
        pvalue=np.nan,
        ci=(np.nan, np.nan),
        alpha=alpha,
        n_obs=int(len(data)),
        detail=detail,
        model_info={
            "methods_requested": methods_list,
            "methods_fit": fit,
            "methods_failed": failed,
            "dispersion": disp,
            "breakdown_m": breakdown_m_value,
            # H5: stable sentinel — export helpers and plots match on this
            "_did_summary_marker": True,
        },
        _citation_key="did_summary",
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Export helpers: publication-ready Markdown / LaTeX from a did_summary
# ═══════════════════════════════════════════════════════════════════════

def _ensure_did_summary(result: CausalResult) -> pd.DataFrame:
    """Validate that result came from did_summary() and return its detail."""
    mi = getattr(result, "model_info", None) or {}
    if not mi.get("_did_summary_marker", False):
        raise ValueError(
            "did_summary_to_markdown / _to_latex require a CausalResult "
            "produced by sp.did_summary() (missing '_did_summary_marker' "
            "in model_info)."
        )
    det = getattr(result, "detail", None)
    if not isinstance(det, pd.DataFrame) or "estimator" not in det.columns:
        raise ValueError(
            "did_summary result has malformed detail; expected an "
            "'estimator' column."
        )
    return det


def did_summary_to_markdown(
    result: CausalResult,
    digits: int = 4,
    include_ci: bool = True,
    include_breakdown: bool = True,
) -> str:
    """
    Render a :func:`did_summary` result as a GitHub-Flavoured Markdown table.

    Columns shown (in order):
    ``Method``, ``Estimate``, ``SE``, ``95 % CI``, ``p-value``, and
    optionally ``Breakdown M*`` (when sensitivity was requested).

    Parameters
    ----------
    result : CausalResult
        Output of :func:`did_summary`.
    digits : int, default 4
        Decimal precision for numeric columns.
    include_ci : bool, default True
        Include the 95 % CI column.
    include_breakdown : bool, default True
        Include the Rambachan-Roth breakdown M* column (CS row only,
        blank for others). Ignored if sensitivity was not requested.

    Returns
    -------
    str
        Multi-line Markdown table, ready to paste into notebooks or PRs.
    """
    det = _ensure_did_summary(result)

    has_bd = include_breakdown and det["breakdown_m"].notna().any()
    lines = []
    header_cells = ["Method", "Estimate", "SE"]
    if include_ci:
        header_cells.append("95% CI")
    header_cells += ["p"]
    if has_bd:
        header_cells.append("Breakdown M*")
    header_cells.append("Notes")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("|" + "|".join([":---"] + ["---:"] * (len(header_cells) - 2) + [":---"]) + "|")

    fmt = f"{{:.{digits}f}}"
    for _, row in det.iterrows():
        cells = [row["estimator"]]
        if pd.isna(row["estimate"]):
            cells.append("—")
            cells.append("—")
            if include_ci:
                cells.append("—")
            cells.append("—")
            if has_bd:
                cells.append("—")
        else:
            est_str = fmt.format(row["estimate"]) + _stars(row["pvalue"])
            cells.append(est_str)
            cells.append(fmt.format(row["se"]))
            if include_ci:
                cells.append(
                    f"[{fmt.format(row['ci_low'])}, {fmt.format(row['ci_high'])}]"
                )
            cells.append(fmt.format(row["pvalue"]))
            if has_bd:
                bd = row.get("breakdown_m", np.nan)
                cells.append(fmt.format(bd) if pd.notna(bd) else "—")
        cells.append(str(row.get("note", "") or ""))
        lines.append("| " + " | ".join(cells) + " |")

    # Footer
    lines.append("")
    if getattr(result, "estimate", None) is not None and not pd.isna(result.estimate):
        lines.append(
            f"*Mean across methods = {fmt.format(result.estimate)}, "
            f"SD across methods = {fmt.format(result.se)}.*  "
            "\\* p<0.1, \\*\\* p<0.05, \\*\\*\\* p<0.01."
        )
    return "\n".join(lines)


def did_summary_to_latex(
    result: CausalResult,
    digits: int = 4,
    include_ci: bool = True,
    include_breakdown: bool = True,
    label: str = "tab:did_summary",
    caption: str = "DID method-robustness summary.",
) -> str:
    """
    Render a :func:`did_summary` result as a publication-ready LaTeX
    ``booktabs`` table.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`did_summary`.
    digits : int, default 4
        Decimal precision.
    include_ci : bool, default True
        Include the 95 % CI column.
    include_breakdown : bool, default True
        Include the Rambachan-Roth breakdown M* column when sensitivity
        was requested.
    label : str, default ``'tab:did_summary'``
        LaTeX label for the table.
    caption : str, default ``'DID method-robustness summary.'``
        LaTeX caption.

    Returns
    -------
    str
        Full ``\\begin{table} ... \\end{table}`` block using the
        ``booktabs`` package (``\\toprule``, ``\\midrule``, ``\\bottomrule``).

    Notes
    -----
    Requires ``\\usepackage{booktabs}`` in the LaTeX preamble.
    """
    det = _ensure_did_summary(result)
    has_bd = include_breakdown and det["breakdown_m"].notna().any()

    cols = ["l", "c", "c"]
    header = ["Method", "Estimate", "SE"]
    if include_ci:
        cols.append("c")
        header.append("95\\% CI")
    cols.append("c")
    header.append("$p$")
    if has_bd:
        cols.append("c")
        header.append("Breakdown $M^*$")

    fmt = f"{{:.{digits}f}}"
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "".join(cols) + "}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]

    for _, row in det.iterrows():
        cells = [str(row["estimator"]).replace("&", "\\&")]
        if pd.isna(row["estimate"]):
            cells += ["---"] * (len(header) - 1)
        else:
            est_str = fmt.format(row["estimate"]) + _stars(row["pvalue"])
            cells.append(est_str)
            cells.append(fmt.format(row["se"]))
            if include_ci:
                cells.append(
                    f"[{fmt.format(row['ci_low'])}, {fmt.format(row['ci_high'])}]"
                )
            cells.append(fmt.format(row["pvalue"]))
            if has_bd:
                bd = row.get("breakdown_m", np.nan)
                cells.append(fmt.format(bd) if pd.notna(bd) else "---")
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]

    # Footnote row
    note_parts = []
    if getattr(result, "estimate", None) is not None and not pd.isna(result.estimate):
        note_parts.append(
            f"Mean across methods = {fmt.format(result.estimate)}, "
            f"SD across methods = {fmt.format(result.se)}."
        )
    note_parts.append("$^*p<0.1,\\,^{**}p<0.05,\\,^{***}p<0.01$.")
    if note_parts:
        lines.append(
            "\\vspace{0.5ex}\\footnotesize " + " ".join(note_parts)
        )
    lines.append("\\end{table}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  One-call DID bundle report — txt + md + tex + png in a single directory
# ═══════════════════════════════════════════════════════════════════════

def did_report(
    data: pd.DataFrame,
    y: str,
    time: str,
    first_treat: str,
    group: str,
    save_to: str,
    methods: Union[str, List[str]] = "auto",
    controls: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    include_sensitivity: bool = True,
    plot_sort_by: Optional[str] = "estimate",
    verbose: bool = False,
) -> CausalResult:
    """
    One-call comprehensive DID report: fits all methods + exports every format.

    Writes the following files to ``save_to``:

    - ``did_summary.txt`` : text dump of ``result.summary()``.
    - ``did_summary.md``  : GitHub-Flavoured Markdown table.
    - ``did_summary.tex`` : publication-ready LaTeX ``booktabs`` fragment.
    - ``did_summary.png`` : forest plot (requires matplotlib).
    - ``did_summary.json`` : detail table + model_info in JSON.

    Parameters
    ----------
    data, y, time, first_treat, group, methods, controls, cluster, alpha
        Same as :func:`did_summary`.
    save_to : str
        Directory path. Created if it does not exist.
    include_sensitivity : bool, default True
        Whether to run Rambachan-Roth breakdown M*. Defaults to ``True``
        in ``did_report`` (vs ``False`` in ``did_summary``) because a
        report is expected to be comprehensive.
    plot_sort_by : {'estimate', None}, default 'estimate'
        Sort the forest plot by estimate ascending.
    verbose : bool, default False
        Print progress lines.

    Returns
    -------
    CausalResult
        The underlying :func:`did_summary` output. All side-effect files
        are written to ``save_to`` as a bundle.
    """
    import json
    from pathlib import Path

    out_dir = Path(save_to)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[did_report] fitting methods into {out_dir} ...", flush=True)

    result = did_summary(
        data, y=y, time=time, first_treat=first_treat, group=group,
        methods=methods, controls=controls, cluster=cluster, alpha=alpha,
        include_sensitivity=include_sensitivity, verbose=verbose,
    )

    # TXT
    (out_dir / "did_summary.txt").write_text(result.summary(), encoding="utf-8")

    # Markdown
    (out_dir / "did_summary.md").write_text(
        did_summary_to_markdown(result), encoding="utf-8",
    )

    # LaTeX
    (out_dir / "did_summary.tex").write_text(
        did_summary_to_latex(result), encoding="utf-8",
    )

    # PNG (best-effort — skip if matplotlib unavailable)
    try:
        from .plots import did_summary_plot
        fig, _ = did_summary_plot(result, sort_by=plot_sort_by)
        fig.savefig(out_dir / "did_summary.png", dpi=150, bbox_inches="tight")
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
    except ImportError:
        if verbose:
            print("[did_report] matplotlib missing — skipping PNG", flush=True)

    # JSON (detail + model_info)
    payload = {
        "method": result.method,
        "estimate": None if pd.isna(result.estimate) else float(result.estimate),
        "cross_method_sd": None if pd.isna(result.se) else float(result.se),
        "detail": result.detail.replace({np.nan: None}).to_dict(orient="records"),
        "model_info": {
            k: (list(v) if isinstance(v, (set, tuple)) else v)
            for k, v in result.model_info.items()
        },
    }
    (out_dir / "did_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )

    if verbose:
        print(f"[did_report] wrote bundle to {out_dir}", flush=True)

    return result
