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
    fmt: Literal["markdown", "latex", "text", "target", "jama", "bmj"] = "markdown",
    title: Optional[str] = None,
    journal: Optional[str] = None,
    authors: Optional[str] = None,
    funding: Optional[str] = None,
    registration: Optional[str] = None,
    data_availability: Optional[str] = None,
    background: Optional[str] = None,
    limitations: Optional[str] = None,
) -> str:
    """Render a target trial emulation result as a publication-ready
    Methods/Results block.

    Parameters
    ----------
    result : TargetTrialResult
        Output of :func:`sp.target_trial.emulate`.
    fmt : {'markdown', 'latex', 'text', 'target', 'jama', 'bmj'}
        ``'target'`` renders the JAMA/BMJ 2025 TARGET 21-item checklist
        as Markdown; ``'jama'`` / ``'bmj'`` renders a structured
        JAMA/BMJ-style manuscript that fills in all 21 TARGET items that
        can be auto-derived from the protocol + result, flagging the
        remaining items for author attention; other formats render the
        shorter STROBE-style Methods & Results block.
    title, journal, authors, funding, registration, data_availability,
    background, limitations : str, optional
        Used by the ``'jama'`` / ``'bmj'`` renderer to populate TARGET
        items that cannot be derived automatically (title, funding,
        registration, data statement, background narrative, and
        limitations).  Missing values are rendered as ``(supply text)``
        placeholders.

    Returns
    -------
    str
    """
    if fmt == "target":
        return target_checklist(result, fmt="markdown")
    if fmt in ("jama", "bmj"):
        return _render_jama(
            result,
            title=title,
            journal=journal or ("JAMA" if fmt == "jama" else "BMJ"),
            authors=authors,
            funding=funding,
            registration=registration,
            data_availability=data_availability,
            background=background,
            limitations=limitations,
        )
    if fmt not in ("markdown", "latex", "text"):
        raise ValueError(
            "fmt must be 'markdown', 'latex', 'text', 'target', 'jama', or 'bmj'"
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


def _placeholder(val: Optional[str], fallback: str = "(supply text)") -> str:
    return val if val else fallback


def _render_jama(
    result: TargetTrialResult,
    *,
    title: Optional[str],
    journal: str,
    authors: Optional[str],
    funding: Optional[str],
    registration: Optional[str],
    data_availability: Optional[str],
    background: Optional[str],
    limitations: Optional[str],
) -> str:
    """JAMA/BMJ-format manuscript.

    Builds a structured-abstract + Methods + Results manuscript block
    that maps 1:1 to the TARGET Statement (Hernán et al., JAMA/BMJ 2025).
    Each section header carries the TARGET item numbers it satisfies so
    reviewers can cross-check.
    """
    p: TargetTrialProtocol = result.protocol
    lo, hi = result.ci
    total = result.n_eligible + result.n_excluded_immortal
    title_str = title or "(Study Title)"
    authors_str = authors or "(Author Names)"

    est_str = (
        f"{result.estimate:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}; "
        f"SE {result.se:.4f})"
    )

    lines: list[str] = []
    lines.append(f"% Manuscript formatted for {journal} — TARGET 2025 compliant")
    lines.append("")
    lines.append(f"# {title_str}")
    lines.append("")
    lines.append(f"**Authors:** {authors_str}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Structured abstract (TARGET item 2)
    # ------------------------------------------------------------------ #
    lines.append("## Structured Abstract  \\[TARGET #2\\]")
    lines.append("")
    abstract = [
        ("Importance",
         _placeholder(background,
                      "State the clinical / policy importance of the causal question.")),
        ("Objective",
         f"To emulate a target trial comparing "
         f"{', '.join(p.treatment_strategies)} on {p.outcome}."),
        ("Design, Setting, and Participants",
         f"Observational emulation via {p.analysis_plan}. "
         f"Eligibility: {_stringify(p.eligibility)}. "
         f"Time zero: {p.time_zero}; follow-up through {p.followup_end}. "
         f"{result.n_eligible} of {total} screened participants were "
         f"included; {result.n_excluded_immortal} excluded to prevent "
         f"immortal-time bias."),
        ("Exposures",
         "; ".join(p.treatment_strategies)),
        ("Main Outcomes and Measures",
         p.outcome),
        ("Results",
         f"The estimated {p.causal_contrast} effect was {est_str}."),
        ("Conclusions and Relevance",
         "(Supply conclusions grounded in the causal estimand and its "
         "sensitivity to protocol assumptions.)"),
    ]
    for label, body in abstract:
        lines.append(f"**{label}.** {body}")
        lines.append("")

    # ------------------------------------------------------------------ #
    # Introduction (items 3 and 4)
    # ------------------------------------------------------------------ #
    lines.append("## Introduction  \\[TARGET #3–4\\]")
    lines.append("")
    lines.append(_placeholder(
        background,
        "Motivate the causal question, explain why a randomized trial is "
        "infeasible, and describe the target trial being emulated.",
    ))
    lines.append("")
    lines.append("### Target Trial Specification")
    lines.append("")
    lines.append(
        f"We emulated a hypothetical target trial in which eligible "
        f"individuals ({_stringify(p.eligibility)}) were randomly "
        f"assigned at {p.time_zero} to one of "
        f"{len(p.treatment_strategies)} treatment strategies "
        f"({', '.join(p.treatment_strategies)}) and followed until "
        f"{p.followup_end} for {p.outcome}."
    )
    lines.append("")

    # ------------------------------------------------------------------ #
    # Methods (items 5–14)
    # ------------------------------------------------------------------ #
    lines.append("## Methods  \\[TARGET #5–14\\]")
    lines.append("")
    method_rows = [
        ("Study design (5)", "Observational cohort emulating the target trial."),
        ("Data source (6)", _placeholder(
            None,
            "Specify the database / registry / EHR, accrual window, and "
            "data linkage procedures.")),
        ("Eligibility (7)", _stringify(p.eligibility)),
        ("Treatment strategies (8)", "; ".join(p.treatment_strategies)),
        ("Assignment procedures (9)", p.assignment),
        ("Time zero & follow-up (10)",
         f"Time zero = {p.time_zero}; follow-up end = {p.followup_end}."),
        ("Outcome (11)", p.outcome),
        ("Causal contrast (12)", p.causal_contrast),
        ("Analysis plan (13)", p.analysis_plan),
        ("Variables (14)",
         "Baseline: "
         + (", ".join(p.baseline_covariates) if p.baseline_covariates else "—")
         + "; time-varying: "
         + (", ".join(p.time_varying_covariates)
            if p.time_varying_covariates else "—")),
    ]
    lines.append("| Component | Specification |")
    lines.append("|---|---|")
    for label, body in method_rows:
        body_esc = str(body).replace("|", "\\|")
        lines.append(f"| **{label}** | {body_esc} |")
    lines.append("")

    if p.notes:
        lines.append(f"*Protocol notes.* {p.notes}")
        lines.append("")

    # ------------------------------------------------------------------ #
    # Results (items 15–19)
    # ------------------------------------------------------------------ #
    lines.append("## Results  \\[TARGET #15–19\\]")
    lines.append("")
    lines.append(
        f"Of {total} individuals screened, {result.n_eligible} met "
        f"eligibility at time zero and {result.n_excluded_immortal} were "
        f"excluded to prevent immortal-time bias (TARGET #15)."
    )
    lines.append("")
    lines.append(
        "Baseline characteristics by treatment strategy are reported in "
        "eTable 1 (TARGET #16)."
    )
    lines.append("")
    lines.append(
        "Outcome counts and person-time by strategy are reported in "
        "eTable 2 (TARGET #17)."
    )
    lines.append("")
    lines.append(
        f"**Primary analysis (TARGET #18).** The estimated "
        f"{p.causal_contrast} contrast was {est_str}. Estimation method: "
        f"{result.method}."
    )
    lines.append("")
    lines.append(
        "Subgroup, sensitivity, and per-protocol-vs-intention-to-treat "
        "analyses are reported in eAppendix (TARGET #19)."
    )
    lines.append("")

    # ------------------------------------------------------------------ #
    # Discussion (item 20)
    # ------------------------------------------------------------------ #
    lines.append("## Discussion  \\[TARGET #20\\]")
    lines.append("")
    lines.append(_placeholder(
        limitations,
        "Interpret the estimate in light of (a) residual confounding, "
        "(b) positivity violations, (c) imperfect measurement of time-"
        "varying covariates, and (d) any known emulation-vs-trial gaps.",
    ))
    lines.append("")

    # ------------------------------------------------------------------ #
    # Other information (item 21)
    # ------------------------------------------------------------------ #
    lines.append("## Additional Information  \\[TARGET #21\\]")
    lines.append("")
    lines.append(f"**Funding.** {_placeholder(funding)}")
    lines.append("")
    lines.append(f"**Registration.** {_placeholder(registration)}")
    lines.append("")
    lines.append(f"**Data & code availability.** {_placeholder(data_availability)}")
    lines.append("")

    # ------------------------------------------------------------------ #
    # TARGET checklist (supplement)
    # ------------------------------------------------------------------ #
    lines.append("---")
    lines.append("")
    lines.append("## Supplement — TARGET 21-Item Checklist")
    lines.append("")
    lines.append(target_checklist(result, fmt="markdown"))

    return "\n".join(lines)


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
