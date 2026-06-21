"""Audit an agent empirical-analysis workflow specification.

This is a static, dependency-free gate for the step *before* an agent starts
running estimators.  A good end-to-end empirical-analysis request should state
the data contract, research question, identification strategy, estimators,
diagnostics, robustness checks, export plan, reproducibility/provenance, and
validation gates.  This script turns that expectation into a machine-readable
contract so agent workflows can fail early on missing assumptions instead of
producing polished but under-identified artifacts.

Usage
-----
::

    python scripts/agent_workflow_spec_audit.py
    python scripts/agent_workflow_spec_audit.py path/to/workflow.json --check
    python scripts/agent_workflow_spec_audit.py path/to/workflow.json --json

Only JSON is supported intentionally: the audit is meant to run in lean CI
environments without PyYAML or optional StatsPAI estimator dependencies.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPEC = (
    REPO_ROOT
    / "plans"
    / "2026-06-21-agent-empirical-analysis-uplift"
    / "example_workflow_spec.json"
)

REQUIRED_SECTIONS: dict[str, tuple[str, ...]] = {
    "research_question": ("treatment", "outcome", "estimand", "population"),
    "data": ("sources", "unit_of_observation", "column_map", "sample_construction"),
    "identification": ("design", "assumptions", "threats"),
    "outputs": ("tables", "figures", "export_formats", "replication_bundle"),
    "reproducibility": ("seed", "environment", "data_provenance"),
    "validation": ("gates", "failure_policy", "human_review_required"),
}

REQUIRED_OUTPUT_FORMATS = {"docx", "xlsx", "tex"}

DESIGN_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "did": {
        "aliases": (
            "did",
            "difference-in-differences",
            "difference in differences",
            "event study",
            "callaway",
            "sun_abraham",
        ),
        "assumptions_any": ("parallel", "no anticipation", "common trends"),
        "data_any": ("id", "unit", "time", "cohort", "first_treat"),
        "diagnostics_any": ("parallel", "pretrend", "event", "rollout"),
        "robustness_any": ("placebo", "honest", "alternative", "lead", "lag"),
    },
    "iv": {
        "aliases": ("iv", "instrument", "2sls", "liml"),
        "assumptions_any": ("relevance", "exclusion", "monotonicity", "first stage"),
        "data_any": ("instrument", "endogenous", "treatment", "outcome"),
        "diagnostics_any": ("first stage", "weak", "f statistic", "overid"),
        "robustness_any": ("liml", "weak", "overid", "placebo"),
    },
    "rd": {
        "aliases": ("rd", "regression discontinuity", "rdd", "rdrobust"),
        "assumptions_any": ("continuity", "manipulation", "local random"),
        "data_any": ("running", "cutoff", "outcome", "treatment"),
        "diagnostics_any": ("bandwidth", "density", "mccrary", "rdplot"),
        "robustness_any": ("bandwidth", "donut", "polynomial", "placebo cutoff"),
    },
    "matching": {
        "aliases": ("matching", "psm", "propensity", "psmatch"),
        "assumptions_any": (
            "unconfounded",
            "overlap",
            "common support",
            "ignorability",
        ),
        "data_any": ("treatment", "outcome", "covariate"),
        "diagnostics_any": ("balance", "overlap", "love plot", "common support"),
        "robustness_any": ("caliper", "neighbor", "kernel", "trimming"),
    },
    "dml": {
        "aliases": ("dml", "double machine learning", "metalearner", "causal forest"),
        "assumptions_any": ("orthogonal", "overlap", "unconfounded", "cross-fit"),
        "data_any": ("treatment", "outcome", "covariate", "features"),
        "diagnostics_any": ("cross-fit", "nuisance", "overlap", "holdout"),
        "robustness_any": ("learner", "fold", "seed", "sensitivity"),
    },
    "synthetic": {
        "aliases": ("synthetic", "scm", "synth", "synthdid"),
        "assumptions_any": ("convex hull", "pre-treatment", "no interference"),
        "data_any": ("unit", "time", "treated", "outcome"),
        "diagnostics_any": ("pre-fit", "placebo", "trajectory", "balance"),
        "robustness_any": ("placebo", "donor", "window", "leave-one-out"),
    },
    "epi": {
        "aliases": ("target trial", "tmle", "g-formula", "iptw", "epidemiology"),
        "assumptions_any": ("exchangeability", "positivity", "consistency"),
        "data_any": ("eligibility", "time zero", "exposure", "outcome"),
        "diagnostics_any": ("positivity", "weights", "censoring", "strobe"),
        "robustness_any": ("e-value", "negative control", "tmle", "g-formula"),
    },
}


@dataclass(frozen=True)
class Issue:
    rule: str
    severity: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
        }


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Sequence, Mapping)) and not isinstance(value, str):
        return bool(value)
    return True


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, inner in value.items():
            out.extend(_flatten_strings(key))
            out.extend(_flatten_strings(inner))
        return out
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        out = []
        for inner in value:
            out.extend(_flatten_strings(inner))
        return out
    return [str(value).lower()]


def _contains_any(value: Any, needles: Sequence[str]) -> bool:
    text = "\n".join(_flatten_strings(value))
    return any(needle.lower() in text for needle in needles)


def _add(
    issues: list[Issue],
    rule: str,
    severity: str,
    path: str,
    message: str,
) -> None:
    issues.append(Issue(rule=rule, severity=severity, path=path, message=message))


def _section(spec: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = spec.get(name)
    return value if isinstance(value, Mapping) else {}


def _check_required_sections(spec: Mapping[str, Any], issues: list[Issue]) -> None:
    for section, fields in REQUIRED_SECTIONS.items():
        value = spec.get(section)
        if not isinstance(value, Mapping):
            _add(
                issues,
                "required_section",
                "error",
                section,
                f"missing mapping section {section!r}",
            )
            continue
        for field in fields:
            if not _is_nonempty(value.get(field)):
                _add(
                    issues,
                    "required_field",
                    "error",
                    f"{section}.{field}",
                    f"missing required field {section}.{field}",
                )


def _design_key(design: str) -> Optional[str]:
    raw = (design or "").strip().lower().replace("_", " ")
    if not raw:
        return None
    for key, rule in DESIGN_RULES.items():
        if any(alias in raw for alias in rule["aliases"]):
            return key
    return None


def _check_design_rules(
    spec: Mapping[str, Any],
    issues: list[Issue],
    design_key: Optional[str],
) -> None:
    identification = _section(spec, "identification")
    raw_design = str(identification.get("design", ""))
    if design_key is None:
        _add(
            issues,
            "design_known",
            "warning",
            "identification.design",
            f"design {raw_design!r} is not recognized by the audit rule table",
        )
        return

    rules = DESIGN_RULES[design_key]
    checks = (
        ("assumptions_any", "identification.assumptions", identification),
        ("data_any", "data", _section(spec, "data")),
        ("diagnostics_any", "diagnostics", spec.get("diagnostics")),
        ("robustness_any", "robustness", spec.get("robustness")),
    )
    for rule_name, path, value in checks:
        if not _contains_any(value, rules[rule_name]):
            _add(
                issues,
                f"{design_key}_{rule_name}",
                "error",
                path,
                (
                    f"{design_key} workflow must mention at least one of: "
                    + ", ".join(rules[rule_name])
                ),
            )


def _check_estimators(spec: Mapping[str, Any], issues: list[Issue]) -> None:
    estimators = _as_list(spec.get("estimators"))
    if not estimators:
        _add(
            issues,
            "estimators_present",
            "error",
            "estimators",
            "workflow must declare at least one StatsPAI estimator",
        )
        return
    estimand = str(_section(spec, "research_question").get("estimand", "")).lower()
    for idx, estimator in enumerate(estimators):
        path = f"estimators[{idx}]"
        if not isinstance(estimator, Mapping):
            _add(
                issues,
                "estimator_shape",
                "error",
                path,
                "estimator must be a mapping",
            )
            continue
        function = str(estimator.get("statspai_function", "")).strip()
        if not function.startswith("sp."):
            _add(
                issues,
                "estimator_function",
                "error",
                f"{path}.statspai_function",
                "estimator must name a public sp.* StatsPAI function",
            )
        target = str(estimator.get("target_estimand", "")).lower()
        if estimand and target and target != estimand:
            _add(
                issues,
                "estimator_estimand_match",
                "warning",
                f"{path}.target_estimand",
                (
                    f"estimator target {target!r} differs from research "
                    f"estimand {estimand!r}"
                ),
            )
        if not _is_nonempty(estimator.get("args")) and not _is_nonempty(
            estimator.get("formula")
        ):
            _add(
                issues,
                "estimator_inputs",
                "warning",
                path,
                "estimator should declare args or formula so an agent can reproduce it",
            )


def _check_outputs(spec: Mapping[str, Any], issues: list[Issue]) -> None:
    outputs = _section(spec, "outputs")
    formats = {
        str(item).lower().lstrip(".")
        for item in _as_list(outputs.get("export_formats"))
    }
    missing = sorted(REQUIRED_OUTPUT_FORMATS - formats)
    if missing:
        _add(
            issues,
            "output_formats",
            "error",
            "outputs.export_formats",
            "full empirical workflow must export "
            + ", ".join(sorted(REQUIRED_OUTPUT_FORMATS)),
        )
    tables = outputs.get("tables")
    if not _contains_any(tables, ("table1", "summary", "balance")):
        _add(
            issues,
            "table1_present",
            "error",
            "outputs.tables",
            "outputs should include a Table 1 / summary / balance table",
        )
    if not _contains_any(tables, ("table2", "main", "result", "regression")):
        _add(
            issues,
            "main_table_present",
            "error",
            "outputs.tables",
            "outputs should include a main result / regression table",
        )


def _check_reproducibility(spec: Mapping[str, Any], issues: list[Issue]) -> None:
    data = _section(spec, "data")
    repro = _section(spec, "reproducibility")
    sources = _as_list(data.get("sources"))
    has_source_hash = any(
        isinstance(source, Mapping)
        and _is_nonempty(source.get("sha256") or source.get("hash"))
        for source in sources
    )
    has_provenance = _is_nonempty(repro.get("data_provenance"))
    if not has_source_hash and not has_provenance:
        _add(
            issues,
            "data_provenance",
            "error",
            "data.sources",
            "workflow must include source hashes or a data_provenance plan",
        )
    if not _is_nonempty(repro.get("code_paths")):
        _add(
            issues,
            "code_paths",
            "warning",
            "reproducibility.code_paths",
            "declare analysis scripts/notebooks that produce the artifacts",
        )


def _check_validation(spec: Mapping[str, Any], issues: list[Issue]) -> None:
    validation = _section(spec, "validation")
    gates = validation.get("gates")
    if not _contains_any(gates, ("pytest", "diff --check", "quality_gate", "check")):
        _add(
            issues,
            "validation_gates",
            "error",
            "validation.gates",
            "validation gates should include executable tests/checks",
        )
    if validation.get("human_review_required") is not True:
        _add(
            issues,
            "human_review_required",
            "error",
            "validation.human_review_required",
            "identification-sensitive workflows must require human review",
        )
    identification = _section(spec, "identification")
    if identification.get("agent_decision_only") is True:
        _add(
            issues,
            "no_agent_only_identification",
            "error",
            "identification.agent_decision_only",
            "agent-only identification decisions are not acceptable",
        )


def audit_workflow_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return a structured audit report for a workflow spec mapping."""
    issues: list[Issue] = []
    if not isinstance(spec, Mapping):
        raise TypeError("workflow spec must be a mapping")

    _check_required_sections(spec, issues)
    raw_design = str(_section(spec, "identification").get("design", ""))
    design_key = _design_key(raw_design)
    _check_design_rules(spec, issues, design_key)
    _check_estimators(spec, issues)
    _check_outputs(spec, issues)
    _check_reproducibility(spec, issues)
    _check_validation(spec, issues)

    counts = {
        "error": sum(1 for issue in issues if issue.severity == "error"),
        "warning": sum(1 for issue in issues if issue.severity == "warning"),
    }
    score = max(0, 100 - counts["error"] * 12 - counts["warning"] * 4)
    return {
        "status": "pass" if counts["error"] == 0 and score >= 85 else "fail",
        "score": score,
        "design": design_key or raw_design or None,
        "issue_counts": counts,
        "issues": [issue.to_dict() for issue in issues],
        "required_sections": sorted(REQUIRED_SECTIONS),
    }


def load_spec(path: Path) -> MutableMapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, MutableMapping):
        raise SystemExit(f"{path}: top-level JSON value must be an object")
    return payload


def render(report: Mapping[str, Any], *, path: Path) -> str:
    if path.is_relative_to(REPO_ROOT):
        display_path = path.relative_to(REPO_ROOT)
    else:
        display_path = path
    lines = [
        "StatsPAI agent empirical workflow spec audit",
        "=" * 54,
        f"Spec   : {display_path}",
        f"Status : {str(report['status']).upper()}",
        f"Score  : {report['score']}",
        f"Design : {report['design'] or '(not declared)'}",
        "",
        "Issue counts",
        "-" * 54,
        f"  errors  : {report['issue_counts']['error']}",
        f"  warnings: {report['issue_counts']['warning']}",
    ]
    issues = list(report.get("issues", []))
    if issues:
        lines.extend(["", "Issues", "-" * 54])
        for issue in issues:
            lines.append(
                "  [{severity}] {path}: {message} ({rule})".format(**issue)
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "spec",
        nargs="?",
        type=Path,
        default=DEFAULT_SPEC,
        help="JSON workflow specification to audit.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON report.")
    parser.add_argument("--check", action="store_true", help="Fail on errors.")
    parser.add_argument("--min-score", type=int, default=85, help="Minimum score.")
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    report = audit_workflow_spec(spec)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print(render(report, path=args.spec))
    if args.check and (
        report["issue_counts"]["error"] or int(report["score"]) < args.min_score
    ):
        print("[agent_workflow_spec_audit] REGRESSION", file=sys.stderr)
        return 1
    if args.check:
        print("[agent_workflow_spec_audit] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
