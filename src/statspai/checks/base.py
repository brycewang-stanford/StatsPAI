"""Causal diagnostic check protocol.

StatsPAI already ships many design-specific diagnostics.  This module supplies
the common result shape that lets workflows and agents compose those checks
without knowing whether the underlying evidence came from RD placebos, synth
leave-one-out, DID pre-trends, weak-IV diagnostics, or the consolidated
robustness battery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

import pandas as pd

from .._result_serialize import attach_result_protocol
from ..core.results import _to_jsonable


@dataclass
class CheckContext:
    """Context shared with pluggable diagnostic checks."""

    design: Optional[str] = None
    data: Optional[pd.DataFrame] = None
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    covariates: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@attach_result_protocol
@dataclass
class CheckResult:
    """Result of one diagnostic or sensitivity check."""

    check_name: str
    passed: Optional[bool] = None
    table: Optional[pd.DataFrame] = None
    text: str = ""
    figures: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe payload for agents and reports."""
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "table": _to_jsonable(self.table),
            "text": self.text,
            "figures": [type(fig).__name__ for fig in self.figures],
            "metadata": _to_jsonable(self.metadata),
        }


@runtime_checkable
class Check(Protocol):
    """Protocol implemented by diagnostic checks."""

    name: str

    def validate(self, result: Any, context: CheckContext) -> None:
        """Raise if the check is not applicable to ``result``."""
        ...

    def run(self, result: Any, context: CheckContext) -> CheckResult:
        """Run the check and return a structured result."""
        ...


class RobustnessBatteryCheck:
    """Adapter exposing ``run_robustness_battery`` as a pluggable check."""

    name = "robustness_battery"

    def validate(self, result: Any, context: CheckContext) -> None:
        if result is None:
            raise TypeError("RobustnessBatteryCheck requires a fitted result.")

    def run(self, result: Any, context: CheckContext) -> CheckResult:
        self.validate(result, context)
        from ..workflow._robustness import run_robustness_battery

        report = run_robustness_battery(
            result,
            design=context.design,
            data=context.data,
            treatment=context.treatment,
            outcome=context.outcome,
            covariates=context.covariates,
        )
        return check_result_from_robustness_report(report)


def check_result_from_robustness_report(report: Any) -> CheckResult:
    """Convert a ``RobustnessReport`` into the common ``CheckResult`` shape."""
    rows = []
    findings = list(getattr(report, "findings", []) or [])
    for finding in findings:
        rows.append(finding.to_dict() if hasattr(finding, "to_dict") else dict(finding))
    table = pd.DataFrame(rows) if rows else None
    failed = [row for row in rows if row.get("severity") in {"violation", "warning"}]
    passed = None if not rows else len(failed) == 0
    return CheckResult(
        check_name="RobustnessBattery",
        passed=passed,
        table=table,
        text=report.to_markdown() if hasattr(report, "to_markdown") else str(report),
        metadata={
            "design": getattr(report, "design", None),
            "n_findings": len(rows),
            "n_flagged": len(failed),
            "notes": list(getattr(report, "notes", []) or []),
        },
    )


def run_checks(
    result: Any,
    checks: Iterable[Check],
    *,
    context: Optional[CheckContext] = None,
) -> List[CheckResult]:
    """Run a list of pluggable checks against one result."""
    ctx = context or CheckContext()
    out: List[CheckResult] = []
    for check in checks:
        if not isinstance(check, Check):
            raise TypeError(
                f"{type(check).__name__} does not satisfy the Check protocol."
            )
        check.validate(result, ctx)
        out.append(check.run(result, ctx))
    return out
