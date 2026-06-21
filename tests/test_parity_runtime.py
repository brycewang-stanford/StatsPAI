"""Optional runtime tests for the external parity harness.

These tests run the parity scripts themselves. They are marked ``slow`` so
default pytest runs only exercise the materialized JSON contracts in
``test_parity_harness_contract.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

pytestmark = [
    pytest.mark.external_parity_runtime,
    pytest.mark.slow,
]

ROOT = Path(__file__).resolve().parents[1]
R_PARITY = ROOT / "tests" / "r_parity"
STATA_PARITY = ROOT / "tests" / "stata_parity"
STATA_DEFAULT = "/Applications/Stata/StataMP.app/Contents/MacOS/stata-mp"


def _run(
    cmd: list[str], *, cwd: Path, timeout: int = 120
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"Command failed: {' '.join(cmd)}\n"
        f"stdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )
    return proc


def _require_rscript() -> str:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not installed; skipping R parity runtime test")
    return rscript


def _require_stata() -> str:
    stata = os.environ.get("STATSPAI_STATA_BIN")
    if stata and Path(stata).exists():
        return stata
    if Path(STATA_DEFAULT).exists():
        return STATA_DEFAULT
    stata = shutil.which("stata-mp") or shutil.which("stata")
    if stata is None:
        pytest.skip("Stata is not installed; skipping Stata parity runtime test")
    return stata


def _assert_result(path: Path, module: str, side: str) -> None:
    assert path.exists(), f"{path} was not created"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["module"] == module
    assert payload["side"] == side
    assert payload["rows"]


@contextmanager
def _preserve_artifacts(paths: list[Path]) -> Iterator[None]:
    snapshots = {path: path.read_bytes() if path.exists() else None for path in paths}
    try:
        yield
    finally:
        for path, content in snapshots.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


@pytest.mark.parametrize("module", ["01_ols", "16_bjs", "50_xtabond"])
def test_r_parity_script_runtime(module: str):
    rscript = _require_rscript()
    py_script = R_PARITY / f"{module}.py"
    r_script = R_PARITY / f"{module}.R"
    py_result = R_PARITY / "results" / f"{module}_py.json"
    r_result = R_PARITY / "results" / f"{module}_R.json"

    assert py_script.exists()
    assert r_script.exists()
    with _preserve_artifacts([py_result, r_result]):
        _run([sys.executable, str(py_script)], cwd=R_PARITY)
        _run([rscript, str(r_script)], cwd=R_PARITY)

        _assert_result(py_result, module, "py")
        _assert_result(r_result, module, "R")


@pytest.mark.parametrize("module", ["16_bjs", "29_panel_sfa", "50_xtabond"])
def test_stata_parity_script_runtime(module: str):
    stata = _require_stata()
    py_script = R_PARITY / f"{module}.py"
    do_script = STATA_PARITY / f"{module}.do"
    py_result = R_PARITY / "results" / f"{module}_py.json"
    stata_result = STATA_PARITY / "results" / f"{module}_Stata.json"

    assert py_script.exists()
    assert do_script.exists()
    with _preserve_artifacts([py_result, stata_result]):
        _run([sys.executable, str(py_script)], cwd=R_PARITY)
        _run([stata, "-b", "-q", "do", do_script.name], cwd=STATA_PARITY)

        _assert_result(py_result, module, "py")
        _assert_result(stata_result, module, "Stata")
