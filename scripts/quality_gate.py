"""Gradual quality gates for the existing StatsPAI codebase.

The repository currently has sizeable historical flake8 and mypy debt.  This
script keeps those checks blocking in CI without pretending the debt is already
zero: each gate passes only if the observed count stays at or below the
documented baseline.  Lower the constants as fixes land.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

DEFAULT_FLAKE8_MAX = 1000
DEFAULT_MYPY_MAX = 1058
MYPY_CONFIG_WARNING_RE = re.compile(r"^.*: \[mypy\]: ", re.MULTILINE)
FORBIDDEN_IMPORT_PREFIXES = (
    "numba",
    "sklearn",
    "statsmodels",
    "linearmodels",
    "torch",
    "pymc",
    "jax",
)
FORBIDDEN_IMPORT_MODULES = (
    "statspai.core._numba_kernels",
    "statspai.panel._hdfe_kernels",
    "statspai.plots.interactive",
)


@dataclass
class GateResult:
    name: str
    count: int
    maximum: int
    output: str
    command_failed: bool

    @property
    def passed(self) -> bool:
        return not self.command_failed and self.count <= self.maximum


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _tail(text: str, n: int = 40) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join(lines[-n:])


def run_flake8(max_violations: int) -> GateResult:
    cmd = [
        sys.executable,
        "-m",
        "flake8",
        "src/statspai",
        "--max-line-length=88",
        "--ignore=E203,W503",
        "--count",
        "--statistics",
    ]
    proc = _run(cmd)
    count = 0
    for line in reversed(proc.stdout.strip().splitlines()):
        if re.fullmatch(r"\d+", line.strip()):
            count = int(line.strip())
            break
    command_failed = proc.returncode != 0 and count == 0
    return GateResult(
        name="flake8",
        count=count,
        maximum=max_violations,
        output=proc.stdout,
        command_failed=command_failed,
    )


def run_mypy(max_errors: int) -> GateResult:
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "src/statspai",
        "--no-error-summary",
        "--hide-error-context",
    ]
    proc = _run(cmd)
    count = len(re.findall(r": error:", proc.stdout))
    command_failed = (proc.returncode != 0 and count == 0) or bool(
        MYPY_CONFIG_WARNING_RE.search(proc.stdout)
    )
    return GateResult(
        name="mypy",
        count=count,
        maximum=max_errors,
        output=proc.stdout,
        command_failed=command_failed,
    )


def run_import_budget() -> GateResult:
    code = f"""
import json
import sys

import statspai  # noqa: F401

prefixes = {FORBIDDEN_IMPORT_PREFIXES!r}
modules = {FORBIDDEN_IMPORT_MODULES!r}
loaded_prefixes = {{
    prefix: sorted(
        name for name in sys.modules
        if name == prefix or name.startswith(prefix + ".")
    )
    for prefix in prefixes
}}
loaded_prefixes = {{
    prefix: names for prefix, names in loaded_prefixes.items() if names
}}
loaded_modules = sorted(name for name in modules if name in sys.modules)
print(json.dumps({{
    "loaded_prefixes": loaded_prefixes,
    "loaded_modules": loaded_modules,
}}, sort_keys=True))
"""
    proc = _run([sys.executable, "-c", code])
    count = 0
    if proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            payload = {}
        loaded_prefixes = payload.get("loaded_prefixes", {})
        loaded_modules = payload.get("loaded_modules", [])
        count = len(loaded_prefixes) + len(loaded_modules)
    return GateResult(
        name="import-budget",
        count=count,
        maximum=0,
        output=proc.stdout,
        command_failed=proc.returncode != 0,
    )


def run_script_gate(name: str, script: str) -> GateResult:
    proc = _run([sys.executable, script, "--check"])
    return GateResult(
        name=name,
        count=0 if proc.returncode == 0 else 1,
        maximum=0,
        output=proc.stdout,
        command_failed=False,
    )


def _report(result: GateResult) -> None:
    print(f"{result.name}: observed={result.count} baseline={result.maximum}")
    if result.command_failed:
        print(f"{result.name}: command failed before producing a count")
    if result.count > result.maximum:
        print(
            f"{result.name}: regression detected "
            f"({result.count} > {result.maximum})"
        )
    if result.output:
        print(f"--- {result.name} output tail ---")
        print(_tail(result.output))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_flake8 = sub.add_parser("flake8", help="Run flake8 debt baseline gate.")
    p_flake8.add_argument("--max-violations", type=int, default=DEFAULT_FLAKE8_MAX)

    p_mypy = sub.add_parser("mypy", help="Run mypy debt baseline gate.")
    p_mypy.add_argument("--max-errors", type=int, default=DEFAULT_MYPY_MAX)

    sub.add_parser("import-budget", help="Run cold-import dependency gate.")
    sub.add_parser("agent-cards", help="Run curated agent-card coverage gate.")
    sub.add_parser("agent-workflow", help="Run empirical workflow spec gate.")
    sub.add_parser("result-protocol", help="Run result-object protocol gate.")
    sub.add_parser("contract-inventory", help="Run cross-contract inventory gate.")
    sub.add_parser("error-taxonomy", help="Run exception taxonomy migration gate.")

    p_all = sub.add_parser("all", help="Run every gradual quality gate.")
    p_all.add_argument("--max-flake8", type=int, default=DEFAULT_FLAKE8_MAX)
    p_all.add_argument("--max-mypy", type=int, default=DEFAULT_MYPY_MAX)

    args = parser.parse_args(argv)

    if args.command == "flake8":
        results = [run_flake8(args.max_violations)]
    elif args.command == "mypy":
        results = [run_mypy(args.max_errors)]
    elif args.command == "import-budget":
        results = [run_import_budget()]
    elif args.command == "agent-cards":
        results = [run_script_gate("agent-cards", "scripts/agent_card_coverage.py")]
    elif args.command == "agent-workflow":
        results = [
            run_script_gate("agent-workflow", "scripts/agent_workflow_spec_audit.py")
        ]
    elif args.command == "result-protocol":
        results = [
            run_script_gate("result-protocol", "scripts/result_protocol_audit.py")
        ]
    elif args.command == "contract-inventory":
        results = [
            run_script_gate("contract-inventory", "scripts/check_contract_inventory.py")
        ]
    elif args.command == "error-taxonomy":
        results = [run_script_gate("error-taxonomy", "scripts/error_taxonomy_audit.py")]
    else:
        results = [
            run_flake8(args.max_flake8),
            run_mypy(args.max_mypy),
            run_import_budget(),
            run_script_gate("agent-cards", "scripts/agent_card_coverage.py"),
            run_script_gate("agent-workflow", "scripts/agent_workflow_spec_audit.py"),
            run_script_gate("result-protocol", "scripts/result_protocol_audit.py"),
            run_script_gate(
                "contract-inventory", "scripts/check_contract_inventory.py"
            ),
            run_script_gate("error-taxonomy", "scripts/error_taxonomy_audit.py"),
        ]

    for result in results:
        _report(result)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
