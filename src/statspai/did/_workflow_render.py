"""Text, table and figure renderings of a :class:`DiDWorkflow`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

from ..exceptions import MethodIncompatibility

if TYPE_CHECKING:  # pragma: no cover
    from .workflow import DiDWorkflow

_FN = "sp.did_workflow"


def summary(wf: "DiDWorkflow") -> str:
    """The protocol with each step's status, finding and action."""
    import textwrap

    inp = wf._inputs
    cohort = repr(inp.cohort) if inp.cohort_source == "column" else inp.cohort_source
    width = 78
    lines = [
        "=" * width,
        "Difference-in-differences workflow",
        f"outcome {inp.y!r} | unit {inp.id!r} | time {inp.time!r} | "
        f"cohort {cohort}",
        "=" * width,
    ]

    def wrap(text: str, lead: str) -> List[str]:
        return textwrap.wrap(
            text,
            width=width,
            initial_indent=lead,
            subsequent_indent=" " * len(lead),
        ) or [lead]

    for s in wf._steps:
        lines.append(f"[{s.status.upper():^7}] {s.number}. {s.title}")
        if s.finding:
            lines.extend(wrap(s.finding, "          "))
        if s.action:
            lines.extend(wrap(s.action, "       -> "))
        for w in s.warnings:
            lines.extend(wrap(w if len(w) <= 300 else w[:297] + "...", "        ! "))
    lines.append("-" * width)
    if wf._headline is not None:
        h = wf._headline
        lines.extend(
            wrap(
                f"{h['estimand']}: {h['estimate']:.4g} (SE {h['se']:.4g}), "
                f"CI [{h['ci_lower']:.4g}, {h['ci_upper']:.4g}]",
                "Estimate  ",
            )
        )
    pending = wf.pending
    lines.append(
        f"Verdict   {wf.verdict.upper()}"
        + (f"  (pending: {', '.join(pending)})" if pending else "")
    )
    lines.append("=" * width)
    return "\n".join(lines)


def to_dict(wf: "DiDWorkflow", *, detail: str = "standard") -> Dict[str, Any]:
    """JSON-safe dictionary of the workflow.

    ``detail='minimal'`` keeps the verdict, the estimate and each
    step's status; ``'standard'`` adds findings and actions;
    ``'agent'`` adds each step's evidence tables.
    """
    from ..core.results import _to_jsonable

    if detail not in ("minimal", "standard", "agent"):
        raise MethodIncompatibility(
            f"{_FN}: detail must be 'minimal', 'standard' or 'agent'; "
            f"got {detail!r}.",
            recovery_hint="wf.to_dict(detail='standard').",
        )
    out: Dict[str, Any] = {
        "method": "did_workflow",
        "verdict": wf.verdict,
        "pending": wf.pending,
    }
    if wf._headline is not None:
        h = wf._headline
        out.update(
            estimand=h["estimand"],
            estimate=h["estimate"],
            se=h["se"],
            pvalue=h["pvalue"],
            ci=[h["ci_lower"], h["ci_upper"]],
            alpha=wf._options["alpha"],
        )
    if detail == "minimal":
        out["steps"] = [
            {"step": s.number, "key": s.key, "status": s.status} for s in wf._steps
        ]
        minimal: Dict[str, Any] = _to_jsonable(out)
        return minimal
    public = (
        "target",
        "estimator",
        "control_group",
        "anticipation",
        "event_window",
        "alpha",
        "seed",
    )
    out["spec"] = {k: wf._options[k] for k in public}
    out["spec"].update(
        y=wf._inputs.y,
        id=wf._inputs.id,
        time=wf._inputs.time,
        cohort=wf._inputs.cohort_source,
        covariates=wf._inputs.covariates,
        weights=wf._inputs.weights,
        cluster=wf._inputs.cluster,
        declared=sorted(wf._explicit),
    )
    steps = [s.to_dict() for s in wf._steps]
    if detail == "standard":
        for item in steps:
            item.pop("evidence")
    out["steps"] = steps
    out["next_steps"] = wf.next_steps()
    out["citations"] = wf.cite(format="json")["citation_keys"]
    out["degradations"] = list(wf.degradations)
    payload: Dict[str, Any] = _to_jsonable(out)
    return payload


def to_markdown(
    wf: "DiDWorkflow", digits: Optional[int] = None, fmt: Any = None
) -> str:
    """The protocol as a GitHub-flavoured Markdown section."""
    lines = [
        "### Difference-in-differences workflow",
        "",
        "| Step | Status | Finding | Action |",
        "| --- | --- | --- | --- |",
    ]
    for s in wf._steps:
        cells = [
            f"{s.number}. {s.title}",
            s.status,
            s.finding.replace("|", r"\|"),
            s.action.replace("|", r"\|"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if wf.step("report").status != "pending":
        lines += ["", "**What to report**", ""]
        for _, row in wf.checklist.iterrows():
            statement = str(row["statement"]).replace("|", r"\|")
            lines.append(f"- **{row['item']}**: {statement}")
    return "\n".join(lines)


def to_latex(
    wf: "DiDWorkflow",
    *,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    digits: Optional[int] = None,
    fmt: Any = None,
) -> str:
    """The protocol as a booktabs table (status and finding per step)."""

    def esc(text: str) -> str:
        out = text.replace("\\", "\0")
        for ch in "&%_#$":
            out = out.replace(ch, "\\" + ch)
        out = out.replace("{", r"\{").replace("}", r"\}")
        return out.replace("\0", r"\textbackslash{}")

    body = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{r l l p{0.55\textwidth}}",
        r"\toprule",
        r" & Step & Status & Finding \\",
        r"\midrule",
    ]
    for s in wf._steps:
        body.append(
            f"{s.number} & {esc(s.title)} & {esc(s.status)} & {esc(s.finding)} \\\\"
        )
    body += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        body.append(rf"\caption{{{esc(caption)}}}")
    if label:
        body.append(rf"\label{{{label}}}")
    body.append(r"\end{table}")
    return "\n".join(body)


def sheets(wf: "DiDWorkflow") -> Dict[str, pd.DataFrame]:
    sheets: Dict[str, pd.DataFrame] = {"steps": wf.steps}
    declare = wf.step("declare").evidence.get("choices")
    if declare is not None:
        sheets["declared"] = declare
    cohorts = wf.step("design").evidence.get("cohorts")
    if cohorts is not None:
        sheets["cohorts"] = cohorts
    if wf._path is not None:
        sheets["event_study"] = wf._path
    if wf._sensitivity is not None:
        sheets["sensitivity"] = wf._sensitivity["rm_grid"]
    if wf._robustness is not None:
        sheets["robustness"] = wf._robustness
    check = wf.step("report").evidence.get("checklist")
    if check is not None:
        sheets["report"] = check
    return sheets


def to_excel(wf: "DiDWorkflow", path: str, **kwargs: Any) -> str:
    """Write every table of the workflow to one workbook, a sheet each."""
    with pd.ExcelWriter(path) as writer:
        for name, frame in sheets(wf).items():
            out = frame.copy()
            for col in out.columns:
                if out[col].map(lambda v: isinstance(v, (tuple, list))).any():
                    out[col] = out[col].map(str)
            out.to_excel(writer, sheet_name=name[:31], index=False, **kwargs)
    return path


def to_word(
    wf: "DiDWorkflow",
    filename: str,
    title: Optional[str] = None,
    digits: Optional[int] = None,
    fmt: Any = None,
) -> str:
    """Write the protocol table and the reporting checklist to .docx."""
    from docx import Document

    doc = Document()
    doc.add_heading(title or "Difference-in-differences workflow", level=1)
    steps = wf.steps
    table = doc.add_table(rows=len(steps) + 1, cols=4)
    table.style = "Light Grid Accent 1"
    for j, name in enumerate(("Step", "Status", "Finding", "Action")):
        table.rows[0].cells[j].text = name
    for i, s in enumerate(wf._steps, start=1):
        cells = table.rows[i].cells
        cells[0].text = f"{s.number}. {s.title}"
        cells[1].text = s.status
        cells[2].text = s.finding
        cells[3].text = s.action
    if wf.step("report").status != "pending":
        doc.add_heading("What to report", level=2)
        for _, row in wf.checklist.iterrows():
            doc.add_paragraph(f"{row['item']}: {row['statement']}")
    doc.save(filename)
    return filename


def plot(wf: "DiDWorkflow", ax: Any = None, **kwargs: Any) -> Any:
    """The event-study path with pointwise and simultaneous intervals."""
    import matplotlib.pyplot as plt

    path = wf.event_study
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 4.2)))
    x = path["relative_time"].to_numpy(dtype=float)
    if "cband_lower" in path:
        ax.fill_between(
            x,
            path["cband_lower"],
            path["cband_upper"],
            alpha=0.18,
            color="C0",
            label="simultaneous band",
        )
    ax.errorbar(
        x,
        path["att"],
        yerr=[path["att"] - path["ci_lower"], path["ci_upper"] - path["att"]],
        fmt="o",
        color="C0",
        capsize=3,
        label="estimate, pointwise interval",
    )
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axvline(-0.5, color="0.3", lw=0.8, ls="--")
    ax.set_xlabel("Periods since treatment")
    ax.set_ylabel(f"Effect on {wf._inputs.y}")
    ax.legend(frameon=False, fontsize=8)
    return ax
