"""
Publication-ready reporting for target trial emulation.

Produces a STROBE-compatible narrative describing the seven protocol
components and the analysis results, formatted either as Markdown
(default) or LaTeX.  Intended for direct inclusion in the Methods
section of a manuscript.
"""

from __future__ import annotations

from typing import Literal, Optional

from .protocol import TargetTrialProtocol
from .emulate import TargetTrialResult


__all__ = ["to_paper", "target_checklist", "TARGET_ITEMS"]


# ---------------------------------------------------------------------------
# TARGET Statement 21-item checklist (JAMA / BMJ, September 2025)
# ---------------------------------------------------------------------------

#: The 21 items of the TARGET Statement (Hernán et al., JAMA 2025; BMJ 2025)
#: grouped by section. Labels follow the published supplementary checklist.
TARGET_ITEMS = [
    # Title & abstract
    ("1",  "Title",                 "Identify the study as a target-trial-emulation observational study."),
    ("2",  "Abstract",              "Provide a structured abstract of the causal question and design."),
    # Introduction
    ("3",  "Background / rationale", "State the causal question and the absence of a feasible RCT."),
    ("4",  "Target trial specification", "Describe the target trial being emulated."),
    # Methods
    ("5",  "Study design",          "Observational study emulating the target trial."),
    ("6",  "Data source",           "Describe the data source and its provenance."),
    ("7",  "Eligibility criteria",  "Eligibility criteria identical to the target trial."),
    ("8",  "Treatment strategies",  "Define the treatment strategies being contrasted."),
    ("9",  "Assignment procedures", "Describe how treatment is assigned (observational)."),
    ("10", "Follow-up",             "Time zero, start and end of follow-up."),
    ("11", "Outcome",               "Primary and secondary outcomes with measurement."),
    ("12", "Causal contrast",       "Identify the contrast (ITT / per-protocol)."),
    ("13", "Analysis plan",         "Estimation strategy aligning emulation with the target trial."),
    ("14", "Variables",             "Confounders, effect modifiers, mediators."),
    # Results
    ("15", "Participants",          "Numbers eligible, included, excluded (with reasons)."),
    ("16", "Descriptive data",      "Baseline characteristics by treatment strategy."),
    ("17", "Outcome data",          "Events / outcomes by strategy."),
    ("18", "Main results",          "Primary causal-contrast estimate with 95% CI."),
    ("19", "Other analyses",        "Subgroup, sensitivity, and secondary analyses."),
    # Discussion
    ("20", "Discussion",            "Interpretation, limitations relative to the target trial."),
    # Other information
    ("21", "Additional information", "Funding, registrations, data / code availability."),
]


def target_checklist(
    result: TargetTrialResult,
    *,
    fmt: Literal["markdown", "text"] = "markdown",
) -> str:
    """Render the TARGET-Statement 21-item checklist as a completed table.

    Each item is tagged ``[AUTO]`` if we can fill it from the
    :class:`TargetTrialProtocol` + :class:`TargetTrialResult` pair, or
    ``[TODO]`` if the author still needs to supply text (e.g. discussion
    and funding). Intended for use as manuscript supplementary material
    — the paper itself still needs hand-written narrative.

    Parameters
    ----------
    result : TargetTrialResult
    fmt : {'markdown', 'text'}, default 'markdown'

    References
    ----------
    Hernán et al. (JAMA 2025; BMJ 2025).
    TARGET Statement: Transparent Reporting of Observational Studies
    Emulating a Target Trial.
    """
    if fmt not in ("markdown", "text"):
        raise ValueError("fmt must be 'markdown' or 'text'")
    p = result.protocol
    lo, hi = result.ci
    est = f"{result.estimate:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}], SE {result.se:.4f})"
    # AUTO-filled mapping of item number → value.
    auto = {
        "4":  f"Target trial: {p.assignment or 'observational emulation'}; "
              f"contrast = {p.causal_contrast}.",
        "6":  "(Specify data source — e.g. insurance claims / EHR / registry.)",
        "7":  _stringify(p.eligibility),
        "8":  ", ".join(p.treatment_strategies),
        "9":  p.assignment,
        "10": f"Time zero: {p.time_zero}; follow-up end: {p.followup_end}.",
        "11": p.outcome,
        "12": p.causal_contrast,
        "13": p.analysis_plan,
        "14": (
            "Baseline: " + (", ".join(p.baseline_covariates) if p.baseline_covariates else "—")
            + "; time-varying: "
            + (", ".join(p.time_varying_covariates) if p.time_varying_covariates else "—")
        ),
        "15": (
            f"n eligible = {result.n_eligible}; "
            f"n excluded (immortal-time prevention) = {result.n_excluded_immortal}."
        ),
        "18": est,
    }
    rows = []
    for num, section, description in TARGET_ITEMS:
        val = auto.get(num)
        tag = "AUTO" if val is not None else "TODO"
        rows.append((num, section, description, val or "(supply text)", tag))

    if fmt == "markdown":
        lines = [
            "# TARGET Statement — 21-item Reporting Checklist",
            "",
            "Source: Hernán et al., *JAMA* & *BMJ*, September 2025.",
            "",
            "| # | Section / Item | TARGET description | Your value | Status |",
            "|---|---|---|---|---|",
        ]
        for num, section, description, value, tag in rows:
            safe_val = str(value).replace("|", "\\|")
            safe_desc = str(description).replace("|", "\\|")
            lines.append(
                f"| {num} | **{section}** | {safe_desc} | {safe_val} | `[{tag}]` |"
            )
        return "\n".join(lines)

    # text
    bar = "=" * 72
    out = [bar, "TARGET Statement — 21-item Reporting Checklist", bar]
    for num, section, description, value, tag in rows:
        out.append(f"{num:>3}. [{tag}] {section}: {description}")
        out.append(f"       → {value}")
    out.append(bar)
    return "\n".join(out)


def to_paper(
    result: TargetTrialResult,
    *,
    fmt: Literal["markdown", "latex", "text", "target"] = "markdown",
    title: Optional[str] = None,
) -> str:
    """Render a target trial emulation result as a publication-ready
    Methods/Results block.

    Parameters
    ----------
    result : TargetTrialResult
        Output of :func:`sp.target_trial.emulate`.
    fmt : {'markdown', 'latex', 'text', 'target'}
        ``'target'`` renders the JAMA/BMJ 2025 TARGET 21-item checklist
        as Markdown; other formats render the STROBE-style Methods &
        Results block.
    title : str, optional
        Paper title / section header.

    Returns
    -------
    str
    """
    if fmt == "target":
        return target_checklist(result, fmt="markdown")
    if fmt not in ("markdown", "latex", "text"):
        raise ValueError(
            "fmt must be 'markdown', 'latex', 'text', or 'target'"
        )

    p: TargetTrialProtocol = result.protocol
    lo, hi = result.ci

    proto_rows = [
        ("Eligibility", _stringify(p.eligibility)),
        ("Treatment strategies", ", ".join(p.treatment_strategies)),
        ("Assignment", p.assignment),
        ("Time zero", p.time_zero),
        ("Follow-up end", p.followup_end),
        ("Outcome", p.outcome),
        ("Causal contrast", p.causal_contrast),
        ("Analysis plan", p.analysis_plan),
    ]
    if p.baseline_covariates:
        proto_rows.append(("Baseline covariates", ", ".join(p.baseline_covariates)))
    if p.time_varying_covariates:
        proto_rows.append(
            ("Time-varying covariates", ", ".join(p.time_varying_covariates))
        )
    if p.notes:
        proto_rows.append(("Notes", p.notes))

    if fmt == "markdown":
        return _render_markdown(proto_rows, result, title)
    if fmt == "latex":
        return _render_latex(proto_rows, result, title)
    return _render_text(proto_rows, result, title)


def _stringify(val) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        return ", ".join(str(x) for x in val)
    if callable(val):
        return f"<predicate {getattr(val, '__name__', 'fn')}>"
    return repr(val)


def _render_markdown(proto_rows, result: TargetTrialResult, title) -> str:
    header = f"# {title}\n\n" if title else ""
    lines = [
        header,
        "## Methods: Target Trial Specification",
        "",
        "This analysis emulates a hypothetical target trial following the",
        "framework of Hernan & Robins (2016; JAMA 2022).",
        "The seven protocol components were pre-specified as:",
        "",
        "| Component | Specification |",
        "|---|---|",
    ]
    for label, val in proto_rows:
        val_esc = str(val).replace("|", "\\|")
        lines.append(f"| **{label}** | {val_esc} |")
    lo, hi = result.ci
    lines += [
        "",
        "## Results",
        "",
        f"Of {result.n_eligible + result.n_excluded_immortal} subjects screened, ",
        f"{result.n_eligible} met eligibility at time zero; ",
        f"{result.n_excluded_immortal} were excluded to prevent immortal-time bias.",
        "",
        f"**Causal contrast ({result.protocol.causal_contrast}):**",
        f"estimate = {result.estimate:+.4f}, "
        f"95% CI [{lo:+.4f}, {hi:+.4f}], SE = {result.se:.4f}.",
        "",
        f"*Analysis method:* {result.method}.",
    ]
    return "\n".join(lines)


def _render_latex(proto_rows, result: TargetTrialResult, title) -> str:
    header = f"\\section*{{{title}}}\n\n" if title else ""
    body = [
        header,
        "\\subsection*{Methods: Target Trial Specification}",
        "",
        "This analysis emulates a hypothetical target trial following",
        "Hern\\'an \\& Robins (2016; JAMA 2022).  Protocol:",
        "",
        "\\begin{tabular}{lp{10cm}}",
        "\\hline",
        "Component & Specification \\\\",
        "\\hline",
    ]
    for label, val in proto_rows:
        val_esc = str(val).replace("&", "\\&").replace("_", "\\_")
        body.append(f"{label} & {val_esc} \\\\")
    body.append("\\hline")
    body.append("\\end{tabular}")
    lo, hi = result.ci
    body += [
        "",
        "\\subsection*{Results}",
        f"Of {result.n_eligible + result.n_excluded_immortal} subjects, "
        f"{result.n_eligible} met eligibility; "
        f"{result.n_excluded_immortal} were excluded to prevent immortal time bias.",
        "",
        f"Causal contrast ({result.protocol.causal_contrast}): "
        f"estimate $= {result.estimate:+.4f}$, "
        f"95\\% CI [{lo:+.4f}, {hi:+.4f}], SE $= {result.se:.4f}$.",
        "",
        f"Analysis method: {result.method}.",
    ]
    return "\n".join(body)


def _render_text(proto_rows, result: TargetTrialResult, title) -> str:
    bar = "=" * 72
    lines = [bar]
    if title:
        lines += [title, bar]
    lines += [
        "Target Trial Emulation Report",
        bar,
        "Protocol:",
    ]
    for label, val in proto_rows:
        lines.append(f"  {label:<22s} {val}")
    lo, hi = result.ci
    lines += [
        "",
        "Results:",
        f"  n eligible       = {result.n_eligible}",
        f"  n excluded       = {result.n_excluded_immortal} (immortal-time prevention)",
        f"  Causal contrast  = {result.protocol.causal_contrast}",
        f"  Estimate         = {result.estimate:+.4f}",
        f"  95% CI           = [{lo:+.4f}, {hi:+.4f}]",
        f"  SE               = {result.se:.4f}",
        f"  Method           = {result.method}",
        bar,
    ]
    return "\n".join(lines)
