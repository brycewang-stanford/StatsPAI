"""Cold-import budget checks for top-level ``import statspai``."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUALITY_GATE = REPO_ROOT / "scripts" / "quality_gate.py"


def _load_quality_gate_module():
    spec = importlib.util.spec_from_file_location("statspai_quality_gate", QUALITY_GATE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_import_budget_quality_gate_passes() -> None:
    res = subprocess.run(
        [sys.executable, str(QUALITY_GATE), "import-budget"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "import-budget: observed=0 baseline=0" in res.stdout


def test_mypy_config_warning_fails_quality_gate(monkeypatch) -> None:
    # A run that produced ONLY a config warning and no analysis at all is a
    # broken run — the gate must fail rather than silently report ``count=0``.
    quality_gate = _load_quality_gate_module()

    class Proc:
        returncode = 0
        stdout = (
            "pyproject.toml: [mypy]: python_version: Python 3.9 is not "
            "supported (must be 3.10 or higher)\n"
        )

    monkeypatch.setattr(quality_gate, "_run", lambda cmd: Proc())

    result = quality_gate.run_mypy(max_errors=0)

    assert result.count == 0
    assert result.command_failed is True
    assert result.passed is False


def test_mypy_config_note_with_analysis_passes(monkeypatch) -> None:
    # A config *deprecation note* emitted alongside a real analysis (e.g. a
    # newer mypy warning that ``python_version = 3.9`` is below its floor) must
    # NOT fail the gate — otherwise a routine mypy bump turns it into a
    # permanent red. Only StatsPAI-tree errors count toward the baseline.
    quality_gate = _load_quality_gate_module()

    class Proc:
        returncode = 1  # mypy: type errors found (a normal, non-blocking run)
        stdout = (
            "pyproject.toml: [mypy]: python_version: Python 3.9 is not "
            "supported (must be 3.10 or higher)\n"
            "src/statspai/parity.py:63: error: Returning Any [no-any-return]\n"
            "/usr/lib/python3/coverage/debug.py:169: error: match unsupported\n"
        )

    monkeypatch.setattr(quality_gate, "_run", lambda cmd: Proc())

    result = quality_gate.run_mypy(max_errors=25)

    # Only the ``src/statspai`` error counts; the third-party ``coverage`` error
    # is ignored, and the config note does not mark the run as failed.
    assert result.count == 1
    assert result.command_failed is False
    assert result.passed is True


def test_mypy_third_party_parse_error_does_not_fail_gate(monkeypatch) -> None:
    # mypy follows imports into installed third-party packages; a parse error
    # there (``match`` under ``python_version = 3.9``) drives the exit code to 2
    # without any StatsPAI-code problem. The gate must stay green.
    quality_gate = _load_quality_gate_module()

    class Proc:
        returncode = 2  # blocking exit driven purely by the third-party file
        stdout = "/usr/lib/python3/coverage/debug.py:169: error: match syntax\n"

    monkeypatch.setattr(quality_gate, "_run", lambda cmd: Proc())

    result = quality_gate.run_mypy(max_errors=25)

    assert result.count == 0
    assert result.command_failed is False
    assert result.passed is True


def test_plain_import_keeps_heavy_optional_modules_lazy() -> None:
    code = """
import json
import sys

import statspai as sp

prefixes = ("numba", "sklearn", "statsmodels", "linearmodels",
            "torch", "pymc", "jax")
modules = ("statspai.core._numba_kernels",
           "statspai.panel._hdfe_kernels",
           "statspai.plots.interactive")
payload = {
    "loaded_prefixes": {
        prefix: sorted(
            name for name in sys.modules
            if name == prefix or name.startswith(prefix + ".")
        )
        for prefix in prefixes
    },
    "loaded_modules": [name for name in modules if name in sys.modules],
    "interactive_in_all": "interactive" in sp.__all__,
    "interactive_cached": "interactive" in sp.__dict__,
}
payload["loaded_prefixes"] = {
    key: value for key, value in payload["loaded_prefixes"].items() if value
}
print(json.dumps(payload, sort_keys=True))
"""
    res = _run_python(code)
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["loaded_prefixes"] == {}
    assert payload["loaded_modules"] == []
    assert payload["interactive_in_all"] is True
    assert payload["interactive_cached"] is False


def test_top_level_interactive_resolves_lazily_on_access() -> None:
    code = """
import json
import sys

import statspai as sp

before = "statspai.plots.interactive" in sys.modules
obj = sp.interactive
after = "statspai.plots.interactive" in sys.modules
print(json.dumps({
    "before": before,
    "after": after,
    "callable": callable(obj),
    "cached": "interactive" in sp.__dict__,
}, sort_keys=True))
"""
    res = _run_python(code)
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload == {
        "after": True,
        "before": False,
        "cached": True,
        "callable": True,
    }
