"""The packaged ``statspai-analysis`` skill (roadmap W6).

The original ``StatsPAI_full_data_analysis_skill/`` stays as the JOSS-era
artifact. The packaged copy under ``statspai/agent/_skill`` is the one
``statspai skill install`` ships: a valid Claude Code skill (lower-case
hyphenated name, description under the 1,024-character limit, no
non-standard frontmatter keys), a short ``SKILL.md`` that points at
``references/``, no stale version stamps, and every ``sp.*`` claim checked
by the bundled gate against the installed package.
"""

from __future__ import annotations

import re
import runpy
import sys
from pathlib import Path

import pytest

import statspai as sp
from statspai.cli import SKILL_NAME, main

SKILL_DIR = Path(sp.__file__).resolve().parent / "agent" / "_skill"
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCES = SKILL_DIR / "references"


def _frontmatter(text: str) -> dict:
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    end = text.index("\n---\n", 4)
    block = text[4:end]
    out: dict = {}
    key = None
    for line in block.splitlines():
        if line.startswith((" ", "\t")):
            out[key] = out.get(key, "") + " " + line.strip()
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


def test_frontmatter_is_valid():
    fm = _frontmatter(SKILL_MD.read_text(encoding="utf-8"))
    assert set(fm) == {"name", "description"}, sorted(fm)
    assert fm["name"] == SKILL_NAME
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fm["name"]), fm["name"]
    assert 0 < len(fm["description"]) <= 1024, len(fm["description"])
    assert "statspai" in fm["description"].lower()


def test_skill_md_is_short_and_points_at_references():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 300
    linked = set(re.findall(r"references/([a-z\-]+\.md)", text))
    present = {p.name for p in REFERENCES.glob("*.md")}
    assert linked, "SKILL.md links no reference files"
    assert linked <= present, linked - present
    assert (
        present <= linked
    ), f"reference files not linked from SKILL.md: {present - linked}"


def test_no_hard_coded_version_stamp():
    for path in [SKILL_MD, *REFERENCES.glob("*.md")]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(
            r"statspai\s+1\.\d+\.\d+", text
        ), f"{path.name} carries a version stamp; the validator is the stamp"


def test_every_sp_reference_resolves():
    placeholder = {"power_"}
    for path in [SKILL_MD, *REFERENCES.glob("*.md")]:
        text = path.read_text(encoding="utf-8")
        refs = set(re.findall(r"\bsp\.([A-Za-z_][A-Za-z0-9_]*)", text)) - placeholder
        missing = sorted(r for r in refs if not hasattr(sp, r))
        assert not missing, (path.name, missing)


def test_bundled_gate_quick_passes(capsys):
    ns = runpy.run_path(str(SKILL_DIR / "validate_api_claims.py"), run_name="_gate")
    argv = sys.argv
    sys.argv = ["validate_api_claims.py", "--quick"]
    try:
        rc = ns["main"]()
    finally:
        sys.argv = argv
    out, _ = capsys.readouterr()
    assert rc == 0, out[-3000:]


def _gate():
    return runpy.run_path(str(SKILL_DIR / "validate_api_claims.py"), run_name="_gate")


def test_bundled_gate_full_passes(capsys):
    """The smoke fits behind the skill's attribute / return-shape claims.

    Until 2026-10 only the JOSS-era copy under
    ``StatsPAI_full_data_analysis_skill/`` had its full gate wired into
    pytest, and that test is marked ``slow``; the packaged skill -- the one
    ``statspai skill install`` ships -- ran ``--quick`` only.
    """
    pytest.importorskip("matplotlib")
    ns = _gate()
    failures: list = []
    ns["check_attributes"](failures)
    out, _ = capsys.readouterr()
    assert not failures, out[-3000:]


_BAD_BLOCKS = {
    "unknown keyword": (
        "```python\nr = sp.callaway_santanna(df, y='y', g='g', t='t', i='i', clustr='s')\n```",
        "has no argument ['clustr']",
    ),
    "too many positionals": (
        "```python\nsp.validation_scope(fit, 'iv', 'extra')\n```",
        "positional",
    ),
    "unresolved name": (
        "```python\nsp.did.no_such_estimator(df)\n```",
        "does not resolve",
    ),
    "not python": (
        "```python\nr = sp.regress('y ~ x', data=df\n```",
        "not valid Python",
    ),
    "kwargs function, misspelled keyword": (
        "```python\nf = sp.causal_forest('y ~ t | x', df, n_estimators=50, honset=True)\n```",
        "passes ['honset']",
    ),
    "route, question the guide does not ask": (
        "```python\nsp.route('did', design='staggered', timing='random')\n```",
        "not a question of the 'did' guide",
    ),
    "route, answer the guide does not offer": (
        "```python\nsp.route('did', design='staggered', covariates='some')\n```",
        "the guide offers ['none', 'yes']",
    ),
    "power, argument of another design": (
        "```python\nsp.power('did', n=100, effect_size=0.2, icc=0.1)\n```",
        "passes ['icc']",
    ),
    "inside a blockquote": (
        "> ```python\n> sp.rdrobust(df, y='y', x='x', cutof=0)\n> ```",
        "has no argument ['cutof']",
    ),
}


@pytest.mark.parametrize("case", sorted(_BAD_BLOCKS))
def test_call_check_catches_a_wrong_call_in_a_reference(case, tmp_path, capsys):
    """A resolvable name with a wrong argument must fail, not only a typo'd name."""
    block, expected = _BAD_BLOCKS[case]
    doc = tmp_path / "injected.md"
    doc.write_text("# Injected\n\n" + block + "\n", encoding="utf-8")
    ns = _gate()
    check = ns["check_call_keywords"]
    check.__globals__["SKILL_FILES"] = [doc]
    failures: list = []
    check(failures)
    capsys.readouterr()
    assert len(failures) == 1 and expected in failures[0], failures


def test_call_check_reads_every_block_of_the_shipped_skill(capsys):
    ns = _gate()
    blocks = [b for f in ns["SKILL_FILES"] for b in ns["_python_blocks"](f)]
    fenced = sum(
        len(
            re.findall(r"^\s*(?:>\s?)*```(?:python|py)\s*$", f.read_text("utf-8"), re.M)
        )
        for f in ns["SKILL_FILES"]
    )
    assert len(blocks) == fenced > 50
    failures: list = []
    ns["check_call_keywords"](failures)
    out, _ = capsys.readouterr()
    assert not failures, out[-3000:]


def test_cli_skill_install_and_path(tmp_path, capsys):
    assert main(["skill", "path"]) == 0
    out, _ = capsys.readouterr()
    assert out.strip() == str(SKILL_DIR)

    assert main(["skill", "install", "--target", str(tmp_path)]) == 0
    dest = tmp_path / SKILL_NAME
    assert (dest / "SKILL.md").read_bytes() == SKILL_MD.read_bytes()
    assert {p.name for p in (dest / "references").glob("*.md")} == {
        p.name for p in REFERENCES.glob("*.md")
    }
    assert (dest / "validate_api_claims.py").exists()
    # Second install without --force refuses; with --force replaces.
    assert main(["skill", "install", "--target", str(tmp_path)]) == 2
    assert main(["skill", "install", "--target", str(tmp_path), "--force"]) == 0


def test_references_cover_the_original_playbook():
    """The split must not lose sections of the JOSS-era SKILL.md."""
    original = (
        Path(sp.__file__).resolve().parents[2]
        / "StatsPAI_full_data_analysis_skill"
        / "SKILL.md"
    )
    if not original.exists():
        pytest.skip("original skill directory not present (installed package)")
    text = original.read_text(encoding="utf-8")
    body = text[text.index("\n---\n", 4) + 5 :]
    headings = {ln[3:].strip() for ln in body.splitlines() if ln.startswith("## ")}
    packaged = "\n".join(p.read_text(encoding="utf-8") for p in REFERENCES.glob("*.md"))
    missing = [h for h in headings if f"## {h}" not in packaged]
    assert not missing, missing


def test_kwargs_calls_in_the_shipped_skill_are_all_verified(capsys):
    """Every keyword of a ``**kwargs`` call resolves to a named source."""
    ns = _gate()
    failures: list = []
    ns["check_call_keywords"](failures)
    out, _ = capsys.readouterr()
    assert not failures, out[-3000:]
    assert "keywords checked against the schema" in out


def test_quick_path_blocks_run_end_to_end(tmp_path, monkeypatch):
    """The short path is executed, not just parsed: five blocks, one session."""
    pytest.importorskip("docx")
    import warnings

    ns = _gate()
    blocks = [src for _, src in ns["_python_blocks"](REFERENCES / "quick-path.md")]
    assert len(blocks) == 5
    monkeypatch.chdir(tmp_path)
    env: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for src in blocks:
            exec(compile(src, "quick-path.md", "exec"), env)  # noqa: S102
    assert (tmp_path / "table1.docx").stat().st_size > 1000
    summary = env["card"]["assumptions"]["checks_summary"]
    # The point of step 4: an empty violations list with checks not run.
    assert summary["not_run"] >= 1 and summary["failed"] == 0
    assert env["card"]["evidence"]["outputs"]["se"] == "reference"
