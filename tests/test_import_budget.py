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
