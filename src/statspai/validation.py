"""Source-tree validation summaries for the StatsPAI paper workflow.

The functions in this module are deliberately metadata-only by default:
they summarize live registry state and already-materialized validation
artifacts without running R, Stata, LaTeX, or long Monte Carlo jobs.
Use :func:`reproduce_jss_tables` for explicit regeneration.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

_REPO_ENV = "STATSPAI_REPO_ROOT"


@dataclass
class ValidationReport:
    """Structured validation evidence for the current StatsPAI checkout.

    Examples
    --------
    >>> import statspai as sp
    >>> report = sp.ValidationReport(
    ...     generated_at="2026-01-01T00:00:00+00:00",
    ...     version="1.16.1",
    ...     repo_root=None,
    ...     registry={"total_functions": 0, "total_categories": 0},
    ...     evidence={},
    ... )
    >>> report.version
    '1.16.1'
    >>> isinstance(report.to_dict(), dict)
    True
    """

    generated_at: str
    version: str
    repo_root: Optional[str]
    registry: Dict[str, Any]
    evidence: Dict[str, Any]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    def summary(self) -> str:
        """Return a compact one-paragraph summary."""
        reg = self.registry
        ev = self.evidence
        return (
            f"StatsPAI {self.version}: {reg.get('total_functions', 0)} registered "
            f"functions across {reg.get('total_categories', 0)} categories; "
            f"{ev.get('r_parity', {}).get('matched_modules', 0)} R parity modules, "
            f"{ev.get('stata_parity', {}).get('modules', 0)} Stata bridge modules, "
            f"{ev.get('monte_carlo', {}).get('runs', 0)} Monte Carlo coverage rows, "
            f"{ev.get('agent_bench', {}).get('trials', 0)} agent-benchmark trials."
        )

    def to_markdown(self) -> str:
        """Render the report as a concise Markdown audit table."""
        reg = self.registry
        ev = self.evidence
        lines = [
            "# StatsPAI Validation Report",
            "",
            f"- Generated: `{self.generated_at}`",
            f"- Version: `{self.version}`",
            f"- Repo root: `{self.repo_root or 'not found'}`",
            "",
            "## Registry",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Registered functions | {reg.get('total_functions', 0)} |",
            f"| Categories | {reg.get('total_categories', 0)} |",
            f"| Hand-written specs | {reg.get('handwritten_specs', 0)} |",
            f"| Auto specs | {reg.get('auto_specs', 0)} |",
            f"| Agent cards | {reg.get('agent_cards', 0)} |",
            f"| Certified functions | {reg.get('per_validation_status', {}).get('certified', 0)} |",
            f"| Validated functions | {reg.get('per_validation_status', {}).get('validated', 0)} |",
            "",
            "## Validation Evidence",
            "",
            "| Track | Evidence |",
            "| --- | ---: |",
            (
                "| R parity | "
                f"{ev.get('r_parity', {}).get('matched_modules', 0)} matched modules |"
            ),
            (
                "| Stata bridge | "
                f"{ev.get('stata_parity', {}).get('modules', 0)} modules |"
            ),
            (
                "| Monte Carlo coverage | "
                f"{ev.get('monte_carlo', {}).get('runs', 0)} rows |"
            ),
            (
                "| Reference parity tests | "
                f"{ev.get('pytest_inventory', {}).get('reference_parity_files', 0)} files |"
            ),
            (
                "| External parity tests | "
                f"{ev.get('pytest_inventory', {}).get('external_parity_files', 0)} files |"
            ),
            (
                "| Agent benchmark | "
                f"{ev.get('agent_bench', {}).get('trials', 0)} trials |"
            ),
            (
                "| Open parity/convention gap rows | "
                f"{ev.get('parity_gaps', {}).get('rows', 0)} rows |"
            ),
        ]
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {msg}" for msg in self.warnings)
        return "\n".join(lines) + "\n"


@dataclass
class ReproductionStep:
    """One command or file-copy step in a JSS table reproduction run.

    Examples
    --------
    >>> import statspai as sp
    >>> step = sp.ReproductionStep(
    ...     name="generate_inventory",
    ...     action="command",
    ...     command=["python", "generate_inventory.py"],
    ...     cwd=".",
    ...     returncode=0,
    ... )
    >>> step.ok
    True
    >>> step.name
    'generate_inventory'
    """

    name: str
    action: str
    command: List[str]
    cwd: str
    returncode: Optional[int] = None
    elapsed_s: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.skipped or self.returncode == 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReproductionResult:
    """Structured result returned by :func:`reproduce_jss_tables`.

    Examples
    --------
    >>> import statspai as sp
    >>> step = sp.ReproductionStep(
    ...     name="generate_inventory",
    ...     action="command",
    ...     command=["python", "generate_inventory.py"],
    ...     cwd=".",
    ...     returncode=0,
    ... )
    >>> result = sp.ReproductionResult(
    ...     generated_at="2026-01-01T00:00:00+00:00",
    ...     repo_root=".",
    ...     targets=["inventory"],
    ...     dry_run=True,
    ...     success=True,
    ...     steps=[step],
    ...     artifacts={},
    ... )
    >>> result.success
    True
    >>> result.failed_steps()
    []
    """

    generated_at: str
    repo_root: str
    targets: List[str]
    dry_run: bool
    success: bool
    steps: List[ReproductionStep]
    artifacts: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        data["steps"] = [step.to_dict() for step in self.steps]
        return data

    def failed_steps(self) -> List[str]:
        """Names of failed steps."""
        return [step.name for step in self.steps if not step.ok]

    def to_markdown(self) -> str:
        """Render the run as a compact Markdown execution log."""
        lines = [
            "# StatsPAI JSS Table Reproduction",
            "",
            f"- Generated: `{self.generated_at}`",
            f"- Repo root: `{self.repo_root}`",
            f"- Targets: `{', '.join(self.targets)}`",
            f"- Dry run: `{self.dry_run}`",
            f"- Success: `{self.success}`",
            "",
            "| Step | Action | Return code | Seconds |",
            "| --- | --- | ---: | ---: |",
        ]
        for step in self.steps:
            rc = "skipped" if step.skipped else step.returncode
            lines.append(
                f"| {step.name} | {step.action} | {rc} | {step.elapsed_s:.2f} |"
            )
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {msg}" for msg in self.warnings)
        return "\n".join(lines) + "\n"


def validation_report(
    repo_root: Optional[Union[str, Path]] = None,
    *,
    include_files: bool = True,
    collect_tests: bool = False,
    fmt: str = "object",
) -> Union[ValidationReport, Dict[str, Any], str]:
    """Summarize registry and validation evidence for the JSS narrative.

    Parameters
    ----------
    repo_root : str or Path, optional
        StatsPAI repository root. If omitted, the function searches
        ``STATSPAI_REPO_ROOT``, the current working directory, and the
        installed package's parents.
    include_files : bool, default True
        Include artifact file existence and size metadata.
    collect_tests : bool, default False
        Run ``pytest --collect-only`` on the reference-parity,
        external-parity, and coverage Monte Carlo directories and record the
        authoritative collected-test counts (the integers the JSS manuscript
        headlines, e.g. 124 / 52 / 12). Off by default because it spawns
        pytest collection (seconds); the cheap metadata-only path reports
        test *file* counts instead.
    fmt : {"object", "dict", "markdown"}, default "object"
        Return a :class:`ValidationReport`, a JSON-serializable dict, or
        a Markdown string.

    Examples
    --------
    >>> import statspai as sp
    >>> report = sp.validation_report(collect_tests=False)
    >>> isinstance(report, sp.ValidationReport)
    True
    >>> isinstance(report.registry["total_functions"], int)
    True
    >>> report.summary().startswith("StatsPAI")
    True
    """
    root, warnings = _coerce_repo_root(repo_root)

    import statspai as sp  # local import avoids circular import at module load

    registry = _registry_snapshot(sp)
    evidence = _validation_evidence(root, warnings, collect_tests=collect_tests)
    evidence["parity_gaps"] = {
        "rows": len(parity_gap_report(root, fmt="records")) if root is not None else 0,
    }
    artifacts = _artifact_snapshot(root) if include_files else {}
    report = ValidationReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        version=getattr(sp, "__version__", "unknown"),
        repo_root=str(root) if root is not None else None,
        registry=registry,
        evidence=evidence,
        artifacts=artifacts,
        warnings=warnings,
    )
    if fmt == "object":
        return report
    if fmt == "dict":
        return report.to_dict()
    if fmt == "markdown":
        return report.to_markdown()
    raise ValueError("fmt must be one of 'object', 'dict', or 'markdown'")


def reproduce_jss_tables(
    repo_root: Optional[Union[str, Path]] = None,
    *,
    targets: Union[str, Iterable[str]] = "all",
    dry_run: bool = False,
    strict: bool = False,
    python_executable: Optional[str] = None,
    timeout: Optional[float] = 600.0,
) -> ReproductionResult:
    """Regenerate the JSS-facing table artifacts from source scripts.

    Parameters
    ----------
    repo_root : str or Path, optional
        StatsPAI repository root. Required when the source tree cannot be
        auto-detected.
    targets : str or iterable of str, default "all"
        ``"all"`` expands to worked examples, parity, appendices, and
        inventory. ``"core"`` skips worked examples. Individual targets:
        ``"worked_examples"``, ``"parity"``, ``"appendices"``,
        ``"inventory"``, ``"figures"``, and ``"verify"``.
    dry_run : bool, default False
        Return the planned steps without executing or copying files.
    strict : bool, default False
        Raise ``RuntimeError`` if any step fails.
    python_executable : str, optional
        Python executable used for scripts. Defaults to ``sys.executable``.
    timeout : float, optional
        Per-command timeout in seconds. ``None`` disables the timeout.

    Examples
    --------
    >>> import statspai as sp
    >>> # dry_run plans the steps without running R/Stata/LaTeX or
    >>> # copying files (needs the StatsPAI source tree).
    >>> plan = sp.reproduce_jss_tables(targets="inventory", dry_run=True)
    >>> plan.dry_run
    True
    >>> all(step.skipped for step in plan.steps)
    True
    >>> result = sp.reproduce_jss_tables(targets="all")  # doctest: +SKIP
    """
    root, warnings = _coerce_repo_root(repo_root)
    if root is None:
        raise FileNotFoundError(
            "StatsPAI source tree was not found; pass repo_root=..."
        )
    selected = _normalise_targets(targets)
    python_bin = python_executable or sys.executable
    steps = _planned_reproduction_steps(root, selected, python_bin)

    executed: List[ReproductionStep] = []
    for step in steps:
        if dry_run:
            step.skipped = True
            executed.append(step)
            continue
        if step.action == "copy":
            executed.append(_run_copy_step(step))
        else:
            executed.append(_run_command_step(step, timeout=timeout))

    success = all(step.ok for step in executed)
    result = ReproductionResult(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        repo_root=str(root),
        targets=selected,
        dry_run=dry_run,
        success=success,
        steps=executed,
        artifacts=_jss_table_artifacts(root),
        warnings=warnings,
    )
    if strict and not success:
        failed = ", ".join(result.failed_steps())
        raise RuntimeError(f"JSS table reproduction failed: {failed}")
    return result


def coverage_matrix(
    repo_root: Optional[Union[str, Path]] = None,
    *,
    level: str = "category",
    fmt: str = "dataframe",
) -> Any:
    """Return a reproducible coverage matrix for the paper narrative.

    Parameters
    ----------
    repo_root : str or Path, optional
        StatsPAI source tree. Needed for parity-artifact columns.
    level : {"category", "parity"}, default "category"
        ``"category"`` aggregates live registry coverage by registry
        category and joins Track A evidence counts. ``"parity"`` returns
        the 36-module Track A matrix parsed from ``tests/r_parity/README.md``.
    fmt : {"dataframe", "records", "markdown"}, default "dataframe"
        Output format.

    Examples
    --------
    >>> import statspai as sp
    >>> rows = sp.coverage_matrix(level="category", fmt="records")
    >>> isinstance(rows, list) and len(rows) > 0
    True
    >>> "category" in rows[0]
    True
    >>> "registered_functions" in rows[0]
    True
    """
    root, _warnings = _coerce_repo_root(repo_root)

    import pandas as pd
    import statspai as sp

    parity_rows = _parity_module_rows(root, sp) if root is not None else []
    if level == "parity":
        rows = parity_rows
    elif level == "category":
        rows = _category_coverage_rows(sp, parity_rows)
    else:
        raise ValueError("level must be one of 'category' or 'parity'")

    if fmt == "records":
        return rows
    df = pd.DataFrame(rows)
    if fmt == "dataframe":
        return df
    if fmt == "markdown":
        return df.to_markdown(index=False)
    raise ValueError("fmt must be one of 'dataframe', 'records', or 'markdown'")


def parity_gap_report(
    repo_root: Optional[Union[str, Path]] = None,
    *,
    fmt: str = "dataframe",
) -> Any:
    """Return the open parity/convention gaps from Track A artifacts.

    The report is metadata-only: it parses already-generated
    ``tests/r_parity/results/parity_table_3way.md`` plus the R/Stata
    coverage matrix. It does not run Python/R/Stata.

    Examples
    --------
    >>> import statspai as sp
    >>> gaps = sp.parity_gap_report(fmt="records")
    >>> isinstance(gaps, list)
    True
    >>> # Each gap row, when present, names its module and next action.
    >>> all("module_id" in row and "next_action" in row for row in gaps)
    True
    """
    root, _warnings = _coerce_repo_root(repo_root)
    rows: List[Dict[str, Any]] = []
    if root is not None:
        rows.extend(_documented_gap_rows(root))
        stata_skip_reasons = _stata_skip_reasons(root)

        import statspai as sp

        for parity in _parity_module_rows(root, sp):
            if parity.get("has_stata_bridge"):
                continue
            module_id = parity.get("module_id", "")
            skip_reason = stata_skip_reasons.get(module_id)
            status = (
                _stata_gap_status(skip_reason)
                if skip_reason
                else "stata_harness_missing"
            )
            reason_text = _format_stata_skip_reason(skip_reason)
            rows.append(
                {
                    "module_id": module_id,
                    "method": parity.get("method", ""),
                    "kind": status,
                    "gap": status,
                    "description": (
                        (
                            "No authoritative Stata reference was selected: "
                            f"{reason_text}."
                        )
                        if status == "no_canonical_stata_reference"
                        else (
                            (
                                "Canonical Stata reference exists, but the bridge "
                                "artifact is not materialized yet: "
                                f"{reason_text}."
                            )
                            if status == "stata_bridge_not_materialized"
                            else "Feasible Stata sibling has not been built yet."
                        )
                    ),
                    "priority": (
                        "medium" if status != "no_canonical_stata_reference" else "low"
                    ),
                    "next_action": _gap_next_action(status, skip_reason or ""),
                }
            )

    if fmt == "records":
        return rows
    import pandas as pd

    df = pd.DataFrame(rows)
    if fmt == "dataframe":
        return df
    if fmt == "markdown":
        return df.to_markdown(index=False)
    raise ValueError("fmt must be one of 'dataframe', 'records', or 'markdown'")


def _stata_skip_reasons(root: Path) -> Dict[str, str]:
    """Return ``compare.py::STATA_SKIP_REASON`` when the source tree has it."""
    compare = root / "tests" / "r_parity" / "compare.py"
    if not compare.exists():
        return {}
    spec = importlib.util.spec_from_file_location(
        "statspai_validation_r_parity_compare",
        compare,
    )
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except (
        OSError,
        ImportError,
        AttributeError,
        RuntimeError,
        SyntaxError,
        ValueError,
        TypeError,
    ):
        return {}
    reasons = getattr(module, "STATA_SKIP_REASON", {})
    if not isinstance(reasons, dict):
        return {}
    return {str(key): str(value) for key, value in reasons.items()}


def _stata_gap_status(skip_reason: str) -> str:
    """Classify a Stata skip reason without collapsing all skips to no-reference."""
    reason = skip_reason.lower()
    if "bridge artifact not materialized" in reason or "licensed stata rerun" in reason:
        return "stata_bridge_not_materialized"
    return "no_canonical_stata_reference"


def _format_stata_skip_reason(skip_reason: Optional[str]) -> str:
    """Strip status prefixes from Stata skip reasons for readable reports."""
    if not skip_reason:
        return ""
    reason = skip_reason.strip().rstrip(".")
    prefixes = (
        "bridge artifact not materialized:",
        "no canonical Stata reference selected:",
    )
    for prefix in prefixes:
        if reason.lower().startswith(prefix.lower()):
            return reason[len(prefix) :].strip()
    return reason


def _normalise_targets(targets: Union[str, Iterable[str]]) -> List[str]:
    aliases = {
        "all": ["worked_examples", "parity", "appendices", "inventory"],
        "core": ["parity", "appendices", "inventory"],
        "tables": ["worked_examples", "parity", "appendices", "inventory"],
    }
    allowed = {
        "worked_examples",
        "parity",
        "appendices",
        "inventory",
        "figures",
        "verify",
    }
    if isinstance(targets, str):
        raw = aliases.get(targets, [targets])
    else:
        raw = []
        for target in targets:
            raw.extend(aliases.get(target, [target]))

    out: List[str] = []
    for target in raw:
        if target not in allowed:
            raise ValueError(
                f"Unknown target {target!r}; expected one of {sorted(allowed)} "
                "or aliases 'all', 'core', 'tables'"
            )
        if target not in out:
            out.append(target)
    return out


def _planned_reproduction_steps(
    root: Path,
    targets: List[str],
    python_bin: str,
) -> List[ReproductionStep]:
    paper = root / "Paper-JSS"
    steps: List[ReproductionStep] = []

    def script_step(name: str, script: Path) -> ReproductionStep:
        return ReproductionStep(
            name=name,
            action="command",
            command=[python_bin, str(script)],
            cwd=str(root),
        )

    if "worked_examples" in targets:
        for script in sorted((paper / "replication" / "scripts").glob("ex0*.py")):
            steps.append(script_step(script.stem, script))
    if "parity" in targets:
        steps.append(
            script_step("r_parity_compare", root / "tests" / "r_parity" / "compare.py")
        )
        steps.append(
            ReproductionStep(
                name="copy_appendix_b_parity",
                action="copy",
                command=[
                    str(
                        root
                        / "tests"
                        / "r_parity"
                        / "results"
                        / "parity_table_3way.tex"
                    ),
                    str(paper / "manuscript" / "tables" / "appendix_b_parity.tex"),
                ],
                cwd=str(root),
            )
        )
    if "appendices" in targets:
        scripts = paper / "replication" / "scripts"
        steps.append(script_step("gen_appendix_A", scripts / "gen_appendix_A.py"))
        steps.append(script_step("gen_appendix_C", scripts / "gen_appendix_C.py"))
    if "inventory" in targets:
        steps.append(
            script_step(
                "generate_inventory",
                paper / "replication" / "scripts" / "generate_inventory.py",
            )
        )
    if "figures" in targets:
        steps.append(
            script_step(
                "generate_figures",
                paper / "replication" / "scripts" / "generate_figures.py",
            )
        )
    if "verify" in targets:
        steps.append(
            script_step(
                "verify_citations",
                paper / "replication" / "scripts" / "verify_citations.py",
            )
        )
    return steps


def _run_command_step(
    step: ReproductionStep,
    *,
    timeout: Optional[float],
) -> ReproductionStep:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            step.command,
            cwd=step.cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        step.returncode = proc.returncode
        step.stdout_tail = _tail(proc.stdout)
        step.stderr_tail = _tail(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        step.returncode = -9
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        step.stdout_tail = _tail(out)
        step.stderr_tail = _tail(err + "\nTimed out.")
    except OSError as exc:
        step.returncode = -1
        step.stderr_tail = str(exc)
    step.elapsed_s = time.perf_counter() - started
    return step


def _run_copy_step(step: ReproductionStep) -> ReproductionStep:
    started = time.perf_counter()
    src = Path(step.command[0])
    dst = Path(step.command[1])
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        step.returncode = 0
        step.stdout_tail = f"copied {src} -> {dst}"
    except OSError as exc:
        step.returncode = -1
        step.stderr_tail = str(exc)
    step.elapsed_s = time.perf_counter() - started
    return step


def _jss_table_artifacts(root: Path) -> Dict[str, Any]:
    paper = root / "Paper-JSS"
    manuscript_tables = paper / "manuscript" / "tables"
    replication_tables = paper / "replication" / "tables"
    return {
        "manuscript_tables": [
            _file_status(path, root) for path in sorted(manuscript_tables.glob("*.tex"))
        ],
        "replication_tables": [
            _file_status(path, root)
            for path in sorted(replication_tables.glob("*.tex"))
        ],
    }


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _coerce_repo_root(
    repo_root: Optional[Union[str, Path]],
) -> tuple[Optional[Path], List[str]]:
    warnings: List[str] = []
    candidates: List[Path] = []
    if repo_root is not None:
        candidates.append(Path(repo_root))
    env_root = os.environ.get(_REPO_ENV)
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    package_root = Path(__file__).resolve()
    candidates.extend(package_root.parents)

    seen = set()
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if _looks_like_repo_root(path):
            return path, warnings

    warnings.append(
        "StatsPAI source tree was not found; file-backed evidence is unavailable."
    )
    return None, warnings


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (
        path / "src" / "statspai" / "__init__.py"
    ).exists()


def _registry_snapshot(sp: Any) -> Dict[str, Any]:
    from statspai import registry as reg

    names = sp.list_functions()
    category_counts: Counter = Counter()
    stability_counts: Counter = Counter()
    validation_counts: Counter = Counter()
    limitations = 0
    handwritten = 0
    auto = 0
    reg._ensure_full_registry()
    for name in names:
        spec = reg._REGISTRY[name]
        category_counts[spec.category] += 1
        stability_counts[spec.stability] += 1
        validation_counts[getattr(spec, "validation_status", "api_stable")] += 1
        limitations += int(bool(spec.limitations))
        if getattr(spec, "_auto", False):
            auto += 1
        else:
            handwritten += 1

    return {
        "total_functions": len(names),
        "total_categories": len(category_counts),
        "per_category": dict(sorted(category_counts.items())),
        "per_stability": dict(sorted(stability_counts.items())),
        "per_validation_status": dict(sorted(validation_counts.items())),
        "handwritten_specs": handwritten,
        "auto_specs": auto,
        "agent_cards": len(sp.agent_cards()),
        "functions_with_limitations": limitations,
    }


def _category_coverage_rows(
    sp: Any,
    parity_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    from statspai import registry as reg

    sp.list_functions()
    reg._ensure_full_registry()
    rows_by_category: Dict[str, Dict[str, Any]] = {}
    for spec in reg._REGISTRY.values():
        row = rows_by_category.setdefault(
            spec.category,
            {
                "category": spec.category,
                "registered_functions": 0,
                "stable_functions": 0,
                "experimental_functions": 0,
                "deprecated_functions": 0,
                "handwritten_specs": 0,
                "auto_specs": 0,
                "agent_cards": 0,
                "r_parity_modules": 0,
                "stata_bridge_modules": 0,
            },
        )
        row["registered_functions"] += 1
        row[f"{spec.stability}_functions"] += 1
        if getattr(spec, "_auto", False):
            row["auto_specs"] += 1
        else:
            row["handwritten_specs"] += 1
        if (
            spec.assumptions
            or spec.failure_modes
            or spec.alternatives
            or spec.pre_conditions
            or spec.typical_n_min
        ):
            row["agent_cards"] += 1

    for parity in parity_rows:
        category = parity.get("category") or "other"
        row = rows_by_category.setdefault(
            category,
            {
                "category": category,
                "registered_functions": 0,
                "stable_functions": 0,
                "experimental_functions": 0,
                "deprecated_functions": 0,
                "handwritten_specs": 0,
                "auto_specs": 0,
                "agent_cards": 0,
                "r_parity_modules": 0,
                "stata_bridge_modules": 0,
            },
        )
        row["r_parity_modules"] += int(bool(parity.get("has_r_parity")))
        row["stata_bridge_modules"] += int(bool(parity.get("has_stata_bridge")))

    return sorted(
        rows_by_category.values(),
        key=lambda row: (-row["registered_functions"], row["category"]),
    )


def _parity_module_rows(root: Path, sp: Any) -> List[Dict[str, Any]]:
    readme = root / "tests" / "r_parity" / "README.md"
    if not readme.exists():
        return []

    r_dir = root / "tests" / "r_parity" / "results"
    py_modules = _module_stems(r_dir.glob("*_py.json"), "_py")
    r_modules = _module_stems(r_dir.glob("*_R.json"), "_R")
    matched = set(py_modules) & set(r_modules)
    number_to_module = {module.split("_", 1)[0]: module for module in py_modules}

    st_dir = root / "tests" / "stata_parity" / "results"
    stata_modules = set(_module_stems(st_dir.glob("*_Stata.json"), "_Stata"))

    rows: List[Dict[str, Any]] = []
    for line in readme.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        number, method, statspai_api, reference = parts[:4]
        api_name = _extract_api_name(statspai_api)
        category, schema_registered = _api_category(sp, api_name)
        module_id = number_to_module.get(number, number)
        rows.append(
            {
                "module_id": module_id,
                "method": _strip_markdown(method),
                "statspai_api": _strip_markdown(statspai_api),
                "api_name": api_name,
                "category": category,
                "schema_registered": schema_registered,
                "r_reference": _strip_markdown(reference),
                "has_python_artifact": module_id in py_modules,
                "has_r_artifact": module_id in r_modules,
                "has_r_parity": module_id in matched,
                "has_stata_bridge": module_id in stata_modules,
            }
        )
    return rows


def _documented_gap_rows(root: Path) -> List[Dict[str, Any]]:
    path = root / "tests" / "r_parity" / "results" / "parity_table_3way.md"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    module_id = ""
    method = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Module "):
            rest = line.replace("## Module ", "", 1).strip()
            module_id = rest.split()[0]
            method = rest
            continue
        if not line.startswith("- **") or "**:" not in line:
            continue
        key = line.split("**", 2)[1]
        description = line.split("**:", 1)[1].strip()
        if not _is_gap_note(key, description):
            continue
        rows.append(
            {
                "module_id": module_id,
                "method": method,
                "kind": "documented_gap",
                "gap": key,
                "description": description,
                "priority": _gap_priority(key, description),
                "next_action": _gap_next_action(key, description),
            }
        )
    return rows


def _is_gap_note(key: str, description: str) -> bool:
    text = f"{key} {description}".lower()
    if key in {
        "dgp",
        "true_gap",
        "certification_note",
        "method",
        "backend",
        "reference_backend",
        "validation_tier",
        "tier",
        "methodological_disclosure",
        "stata_status",
        "stata_bridge_status",
        "stata_cluster_reference",
        "stata_precision_note",
        "stata_reference_note",
        "stata_algorithm",
        "quantile_convention",
        "reference_backend_note",
    }:
        return False
    if key == "loglik_note" and "backend diagnostic" in text:
        return False
    if key == "note" and "combined monte carlo error" in text:
        return False
    gap_markers = (
        "gap",
        "warning",
        "non-uniqueness",
        "differs",
        "differ",
        "different",
        "convention",
        "overlap",
    )
    if key.endswith("_parity_note") and not any(
        marker in text for marker in gap_markers
    ):
        return False
    if (
        key == "native_note"
        and "parity" in text
        and not any(marker in text for marker in gap_markers)
    ):
        return False
    markers = (
        "gap",
        "warning",
        "non-uniqueness",
        "differs",
        "differ",
        "different",
        "overlap",
        "convention",
        "bandwidth_selector",
    )
    return any(marker in text for marker in markers)


def _gap_priority(key: str, description: str) -> str:
    text = f"{key} {description}".lower()
    if any(tok in text for tok in ("bandwidth", "aggregation", "overlap")):
        return "high"
    if any(tok in text for tok in ("non-uniqueness", "convention", "differs")):
        return "medium"
    return "low"


def _gap_next_action(key: str, description: str = "") -> str:
    text = f"{key} {description}".lower()
    if "bandwidth" in text:
        return (
            "Add an explicit parity bandwidth mode and isolate the selector "
            "regularisation convention in a unit test."
        )
    if "aggregation" in text:
        return (
            "Expose R/Stata-compatible aggregation switches and record the "
            "chosen convention in model_info."
        )
    if "scm" in text or "non-uniqueness" in text:
        if "multi-start diagnostics" in text or "near-best" in text:
            return (
                "Keep the deterministic multi-start and donor-weight-class "
                "diagnostics in the T4 row; promote only with a common "
                "optimizer/reference specification."
            )
        return (
            "Use deterministic multi-start solver diagnostics and report "
            "donor-weight equivalence classes."
        )
    if "overlap" in text:
        return (
            "Default reviewer output to overlap diagnostics and ATO/overlap "
            "target summaries when propensities are extreme."
        )
    if "stata_harness_missing" in text:
        return (
            "Add the feasible .do sibling, write *_Stata.json, then rerun "
            "tests/r_parity/compare.py."
        )
    if "stata_bridge_not_materialized" in text:
        return (
            "Run the canonical Stata command under a licensed Stata runtime, "
            "write *_Stata.json, then rerun tests/r_parity/compare.py."
        )
    if "no_canonical_stata_reference" in text:
        return "Keep documented as no canonical Stata reference unless the literature changes."
    return "Document the convention or close it with a common-specification test."


def _extract_api_name(text: str) -> str:
    clean = _strip_markdown(text)
    clean = re.sub(r"\(.*$", "", clean).strip()
    if clean.startswith("sp."):
        clean = clean[3:]
    return clean.split(".")[-1]


def _api_category(sp: Any, api_name: str) -> tuple[str, bool]:
    if not api_name:
        return "other", False
    try:
        return sp.describe_function(api_name).get("category", "other"), True
    except KeyError:
        return "other", False


def _strip_markdown(text: str) -> str:
    return text.replace("`", "").replace("\\", "").strip()


def _validation_evidence(
    root: Optional[Path],
    warnings: List[str],
    *,
    collect_tests: bool = False,
) -> Dict[str, Any]:
    if root is None:
        return {
            "r_parity": {"matched_modules": 0},
            "stata_parity": {"modules": 0},
            "monte_carlo": {"runs": 0},
            "pytest_inventory": {
                "reference_parity_files": 0,
                "external_parity_files": 0,
            },
            "agent_bench": {"trials": 0},
        }

    r_dir = root / "tests" / "r_parity" / "results"
    py_modules = _module_stems(r_dir.glob("*_py.json"), "_py")
    r_modules = _module_stems(r_dir.glob("*_R.json"), "_R")
    matched = sorted(set(py_modules) & set(r_modules))

    st_dir = root / "tests" / "stata_parity" / "results"
    stata_modules = _module_stems(st_dir.glob("*_Stata.json"), "_Stata")

    mc = _monte_carlo_summary(root)
    agent = _agent_bench_summary(root)
    pytest_inventory: Dict[str, Any] = {
        "reference_parity_files": _count_files(
            root / "tests" / "reference_parity", "test_*.py"
        ),
        "external_parity_files": _count_files(
            root / "tests" / "external_parity", "test_*.py"
        ),
        "coverage_monte_carlo_files": _count_files(
            root / "tests" / "coverage_monte_carlo", "test_*.py"
        ),
    }
    if collect_tests:
        # Authoritative pytest --collect-only counts (parametrize-expanded),
        # i.e. the exact integers the JSS manuscript headlines. These make
        # the "headline counts are not hand-copied" claim script-verifiable.
        collected = {
            "reference_parity": _pytest_collected_count(root, "tests/reference_parity"),
            "external_parity": _pytest_collected_count(root, "tests/external_parity"),
            "coverage_monte_carlo": _pytest_collected_count(
                root, "tests/coverage_monte_carlo"
            ),
        }
        pytest_inventory["collected"] = collected
        missing = [k for k, v in collected.items() if v is None]
        if missing:
            warnings.append(
                "pytest --collect-only could not be run for: "
                + ", ".join(missing)
                + " (pytest unavailable or directory missing)."
            )

    if py_modules and len(matched) != len(py_modules):
        warnings.append(
            "Some R parity Python JSON artifacts do not have matching R JSON files."
        )

    return {
        "r_parity": {
            "python_modules": len(py_modules),
            "r_modules": len(r_modules),
            "matched_modules": len(matched),
            "modules": matched,
        },
        "stata_parity": {
            "modules": len(stata_modules),
            "module_ids": sorted(stata_modules),
        },
        "monte_carlo": mc,
        "pytest_inventory": pytest_inventory,
        "agent_bench": agent,
    }


def _artifact_snapshot(root: Optional[Path]) -> Dict[str, Any]:
    if root is None:
        return {}
    tracked = {
        "r_parity_table": root / "tests" / "r_parity" / "results" / "parity_table.md",
        "r_parity_table_3way": (
            root / "tests" / "r_parity" / "results" / "parity_table_3way.md"
        ),
        "jss_appendix_a": (
            root / "Paper-JSS" / "manuscript" / "tables" / "appendix_a_inventory.tex"
        ),
        "jss_appendix_b": (
            root / "Paper-JSS" / "manuscript" / "tables" / "appendix_b_parity.tex"
        ),
        "jss_appendix_c": (
            root / "Paper-JSS" / "manuscript" / "tables" / "appendix_c_schema.tex"
        ),
        "jss_function_inventory": (
            root / "Paper-JSS" / "manuscript" / "tables" / "function_inventory_full.tex"
        ),
        "agent_bench_headline": (
            root / "tests" / "agent_bench" / "results" / "headline.md"
        ),
        "coverage_b1000": (
            root
            / "tests"
            / "coverage_monte_carlo"
            / "results_b1000"
            / "coverage_b1000.json"
        ),
    }
    return {name: _file_status(path, root) for name, path in tracked.items()}


def _module_stems(paths: Iterable[Path], suffix: str) -> List[str]:
    return sorted(path.stem[: -len(suffix)] for path in paths)


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob(pattern))


_COLLECT_RE = re.compile(r"(\d+)\s*/?\s*(\d+)?\s*tests? collected")


def _pytest_collected_count(
    root: Path,
    rel_dir: str,
    *,
    timeout: float = 180.0,
) -> Optional[int]:
    """Return the authoritative ``pytest --collect-only`` test count for a
    directory, or ``None`` if pytest is unavailable / the dir is missing.

    This is the count the JSS manuscript headlines (e.g. 124 reference-parity
    and 52 external-parity tests): it expands ``@pytest.mark.parametrize``
    cases, unlike a raw ``def test_`` grep, and includes deselected tests
    (``--collect-only`` lists them). Coverage instrumentation and the cache
    plugin are disabled (``-o addopts=`` clears the project's injected
    ``--cov``) so the collection is fast and side-effect free.
    """
    target = root / rel_dir
    if not target.exists():
        return None
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(target),
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
            ],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    # pytest prints either "N tests collected" or, when some are deselected,
    # "N/M tests collected (K deselected)" where M = N + K is the TOTAL number
    # of tests considered. We report that total: M when the slashed form is
    # present, else N.
    for line in reversed(proc.stdout.splitlines()):
        m = _COLLECT_RE.search(line)
        if m:
            return int(m.group(2)) if m.group(2) else int(m.group(1))
    return None


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _monte_carlo_summary(root: Path) -> Dict[str, Any]:
    path = (
        root
        / "tests"
        / "coverage_monte_carlo"
        / "results_b1000"
        / "coverage_b1000.json"
    )
    payload = _read_json(path)
    if not isinstance(payload, list):
        return {"runs": 0, "source": _rel(path, root)}
    rates = [
        row.get("rate") for row in payload if isinstance(row.get("rate"), (int, float))
    ]
    return {
        "runs": len(payload),
        "source": _rel(path, root),
        "min_rate": min(rates) if rates else None,
        "max_rate": max(rates) if rates else None,
        "mean_rate": sum(rates) / len(rates) if rates else None,
        "rows": payload,
    }


def _agent_bench_summary(root: Path) -> Dict[str, Any]:
    bench = root / "tests" / "agent_bench"
    prompts = _read_json(bench / "prompts" / "prompts.json")
    prompt_count = len(prompts) if isinstance(prompts, list) else 0
    scores = bench / "results" / "scores.csv"
    score_rows = 0
    if scores.exists():
        try:
            with scores.open("r", encoding="utf-8", newline="") as fh:
                score_rows = sum(1 for _ in csv.DictReader(fh))
        except OSError:
            score_rows = 0
    trials = bench / "results" / "trials.jsonl"
    trial_count = 0
    if trials.exists():
        try:
            with trials.open("r", encoding="utf-8") as fh:
                trial_count = sum(1 for line in fh if line.strip())
        except OSError:
            trial_count = 0
    return {
        "prompts": prompt_count,
        "score_rows": score_rows,
        "trials": trial_count,
        "headline": _rel(bench / "results" / "headline.md", root),
    }


def _file_status(path: Path, root: Path) -> Dict[str, Any]:
    exists = path.exists()
    info: Dict[str, Any] = {"path": _rel(path, root), "exists": exists}
    if exists:
        stat = path.stat()
        info.update(
            {
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(timespec="seconds"),
            }
        )
    return info


def _rel(path: Path, root: Path) -> str:
    # ``as_posix()`` (not ``str()``) so the artifact paths in the agent-facing
    # ``sp.validation_report()`` payload are byte-identical on Windows and
    # POSIX — the same OS-portability fix applied to the registry's
    # validation-evidence notes. ``str(PurePath)`` would emit backslash
    # separators on Windows, splitting the report value across shards.
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
