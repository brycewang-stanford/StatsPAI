"""Contract tests for the SE/vcov menu coverage matrix gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "se_menu_matrix.py"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_matrix_renders() -> None:
    res = _run([])
    assert res.returncode == 0, res.stderr
    assert "SE / vcov menu coverage matrix" in res.stdout
    assert "stranded" in res.stdout


def test_every_estimator_and_se_function_resolves_in_live_api() -> None:
    """The drift guard must report no problems against the current sp.* API."""
    res = _run(["--json"])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert payload["drift"] == [], f"matrix references stale names: {payload['drift']}"


def test_status_counts_consistent() -> None:
    res = _run(["--json"])
    payload = json.loads(res.stdout)
    counts = payload["status_counts"]
    n_estimators = len(payload["per_estimator"])
    n_se = len(payload["se_types"])
    assert sum(counts.values()) == n_estimators * n_se


def test_native_wild_boot_estimators() -> None:
    """Native + correct wild cluster bootstrap, all externally validated against
    Stata ``boottest``: the panel ``hdfe_ols``, the pyfixest ``feols``
    (``vce="wild"``), and ``ivreg`` (``vce="wild"`` → WRE bootstrap)."""
    from se_menu_matrix import MATRIX, _cell  # type: ignore

    native_wild = sorted(
        est for est in MATRIX if _cell(est, "wild_cluster_boot") == "native"
    )
    assert native_wild == ["feols", "hdfe_ols", "ivreg"], native_wild


def test_ratchet_holds() -> None:
    res = _run(["--check"])
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_markdown_table() -> None:
    res = _run(["--markdown"])
    assert res.returncode == 0, res.stderr
    assert "| estimator |" in res.stdout
    assert "`feols`" in res.stdout
