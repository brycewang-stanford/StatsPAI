"""The release gate (scripts/release_gate.py) and the defects it exists for.

1.33.0 / 1.34.0 shipped three things that only a paper's replication archive
found, each costing a patch release: stale registry-census figures in two
documents, a regular expression whose meaning changed under the archive's
ASCII transliteration, and a promotional phrase in a packaged guide.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def _load_gate():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "release_gate", SCRIPTS / "release_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def test_no_pattern_in_the_package_changes_under_ascii_transliteration():
    assert gate.non_ascii_patterns() == []


def test_packaged_text_carries_no_promotional_wording():
    assert gate.forbidden_wording() == []


def test_the_ascii_gate_catches_a_literal_ellipsis_and_accepts_the_escape():
    literal = 'import re\nP = re.compile(r"a|…|b")\n'
    issues = gate._raw_string_issues("x.py", literal, ast.parse(literal))
    assert len(issues) == 1 and "U+2026" in issues[0]

    escaped = 'import re\nP = re.compile(r"a|\\u2026|b")\n'
    assert gate._raw_string_issues("x.py", escaped, ast.parse(escaped)) == []


def test_the_ascii_gate_sees_a_pattern_kept_in_a_table():
    # agent/remediation.py keeps its patterns in dicts, not in re.compile().
    source = 'RULES = [{"match": r"mean.{0,10}≠"}]\n'
    assert gate._raw_string_issues("x.py", source, ast.parse(source))


def test_a_transliterated_ellipsis_would_match_anything():
    # Why the gate exists: U+2026 -> "..." turns one alternative into
    # "any three characters".
    unicode_pattern = re.compile("etc|…")
    ascii_pattern = re.compile(gate.ascii_source_text("etc|…"))
    text = "Estimand: 'RMST' or 'survival_probability'."
    assert unicode_pattern.search(text) is None
    assert ascii_pattern.search(text) is not None


def test_schema_enrichment_open_set_pattern_survives_transliteration():
    import importlib

    enrich = importlib.import_module("statspai._schema_enrich")
    source = Path(enrich.__file__).read_text(encoding="utf-8")
    start = source.index("_OPEN_SET_RE = re.compile(")
    block = source[start : source.index(")\n", start)]
    assert gate.ascii_source_text(block) == block
    assert enrich._OPEN_SET_RE.search("one, two …")
    assert enrich._OPEN_SET_RE.search("Estimand: 'RMST' or 'x'.") is None


def test_remediation_still_recognises_the_not_equal_sign():
    import importlib

    rem = importlib.import_module("statspai.agent.remediation")
    source = Path(rem.__file__).read_text(encoding="utf-8")
    line = next(ln for ln in source.splitlines() if "dml.*bias" in ln)
    pattern = ast.literal_eval(line.split(":", 1)[1].strip().rstrip(","))
    assert gate.ascii_source_text(line) == line
    assert re.search(pattern, "score mean ≠ 0", re.I)
    assert re.search(pattern, "score mean != 0", re.I)


def test_census_fix_rewrites_every_quoted_figure(tmp_path, monkeypatch):
    for rel in (gate.STABILITY, gate.DOSSIER):
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (REPO_ROOT / rel).read_text(encoding="utf-8"), encoding="utf-8"
        )
    monkeypatch.setattr(gate, "REPO_ROOT", tmp_path)
    live = {
        "functions": 4321,
        "certified": 11,
        "validated": 22,
        "api_stable": 33,
        "experimental": 4,
        "certified_validated": 33,
        "unbacked_auto": 55,
        "evidence_files": 66,
    }
    assert gate.census_drift(live)  # the committed figures are not these
    gate.census_drift(live, fix=True)
    assert gate.census_drift(live) == []
    dossier = (tmp_path / gate.DOSSIER).read_text(encoding="utf-8")
    assert "4,321 registered public functions" in dossier
    assert (
        "11 `certified`, 22 `validated`, 33 `api_stable`, and 4 `experimental`"
        in dossier
    )


def test_census_is_current_inside_the_release_window():
    """Between the version bump and the tag the census documents must be current.

    Outside that window ``main`` keeps registering functions and the documents
    describe the last release, so the comparison is skipped.
    """
    if not gate.in_release_window():
        pytest.skip("package version is already tagged: not preparing a release")
    assert (
        gate.census_drift(gate.census()) == []
    ), "run: python scripts/release_gate.py --fix-census"
