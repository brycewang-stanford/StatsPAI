"""Contract tests for the public-API signature house-style lint.

These guard two things:

1. ``statspai._house_style`` is internally coherent — no spelling is both a
   canonical name and an alias, and every false-friend entry is well-formed.
2. ``scripts/signature_house_style.py`` runs, emits the expected shape, and the
   per-theme legacy-spelling counts do not regress above the frozen baseline
   (the same ratchet CI enforces).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "signature_house_style.py"

sys.path.insert(0, str(REPO_ROOT / "src"))


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


# --- vocabulary coherence -------------------------------------------------


def test_canonical_and_aliases_are_disjoint() -> None:
    from statspai import _house_style as hs

    canonical = {str(t["canonical"]) for t in hs.THEMES.values()}
    aliases: set[str] = set()
    for theme in hs.THEMES.values():
        aliases.update(str(a) for a in theme["aliases"])
    overlap = canonical & aliases
    assert not overlap, f"spelling is both canonical and alias: {overlap}"


def test_alias_index_round_trips() -> None:
    from statspai import _house_style as hs

    idx = hs.alias_index()
    assert idx["robust"] == "vce"
    assert idx["vcov"] == "vce"
    assert idx["outcome"] == "y"
    assert idx["treatment"] == "treat"
    # canonical names resolve to themselves
    assert hs.canonical_for("vce") == "vce"
    assert hs.canonical_for("y") == "y"
    # unknown spelling resolves to None
    assert hs.canonical_for("totally_unrelated_param") is None


def test_false_friend_matching() -> None:
    from statspai import _house_style as hs

    # spatial weight matrix is not an observation-weight violation
    assert hs.is_false_friend("W", "sar", "statspai.spatial.sar")
    # GMM weighting matrix
    assert hs.is_false_friend("W", "gmm", "statspai.gmm.general_gmm")
    # RD flexible-adjustment covariate vector
    assert hs.is_false_friend("W", "rd_flex", "statspai.rd.rd_flex")
    # mixed-model RE covariance structure is not an SE type
    assert hs.is_false_friend("cov_type", "mixed", "statspai.multilevel.mixed")
    # but a plain regression `vcov` IS a real legacy spelling
    assert not hs.is_false_friend("vcov", "feols", "statspai.fixest.wrapper")


# --- lint behaviour & ratchet --------------------------------------------


def test_lint_summary_renders() -> None:
    res = _run([])
    assert res.returncode == 0, res.stderr
    assert "signature house-style lint" in res.stdout
    assert "Canonical coverage by theme" in res.stdout


def test_lint_json_shape() -> None:
    res = _run(["--json"])
    assert res.returncode == 0, res.stderr
    payload = json.loads(res.stdout)
    assert {"totals", "by_theme", "coverage", "robust_default_kinds"} <= set(payload)
    # the SE theme is the worst offender — guard it stays visible
    assert payload["coverage"]["se"]["total"] > 0


def test_lint_ratchet_holds() -> None:
    """The committed baseline must still pass --check (no regression)."""
    res = _run(["--check"])
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout
