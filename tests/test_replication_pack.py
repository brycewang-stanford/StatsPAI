"""Tests for ``sp.output._replication_pack``.

The pack is a wrapper around stdlib ``zipfile`` that pulls together
data, code, environment, paper, and lineage. These tests cover:

- Minimal pack (no inputs) still produces a valid zip with MANIFEST.
- Data-only pack: CSV + data manifest + content hash matches.
- Code passed inline vs. via path.
- PaperDraft-like duck type renders into ``paper/``.
- Citations → ``paper/paper.bib``.
- Results carrying ``_provenance`` aggregate into ``lineage.json``.
- ``extra_files`` end up in the archive verbatim.
- ``overwrite=False`` raises on existing path.
- ``env=False`` skips pip freeze (much faster + makes the test hermetic).
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from statspai.output._lineage import attach_provenance
from statspai.output._replication_pack import (
    ReplicationPack,
    replication_pack,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    return pd.DataFrame({
        "y": np.arange(10, dtype=float),
        "treat": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "x": np.linspace(0, 1, 10),
    })


@pytest.fixture
def script_text():
    return (
        "import statspai as sp\n"
        "import pandas as pd\n"
        "df = pd.read_csv('data/dataset.csv')\n"
        "# (placeholder analysis)\n"
    )


def _open_zip(rp):
    return zipfile.ZipFile(io.BytesIO(rp.output_path.read_bytes()), "r")


# ---------------------------------------------------------------------------
# Minimal / smoke
# ---------------------------------------------------------------------------

class TestMinimalPack:
    def test_no_inputs_still_valid_zip(self, tmp_path):
        out = tmp_path / "empty.zip"
        rp = replication_pack(None, out, env=False)
        assert isinstance(rp, ReplicationPack)
        assert out.exists() and out.stat().st_size > 0
        with _open_zip(rp) as zf:
            names = zf.namelist()
            assert "MANIFEST.json" in names
            assert "README.md" in names
            # No data, no code → those warnings recorded.
            manifest = json.loads(zf.read("MANIFEST.json"))
            assert manifest["statspai_version"]
            assert any("no data" in w.lower() for w in manifest["warnings"])
            assert any("no code" in w.lower() for w in manifest["warnings"])

    def test_summary_method(self, tmp_path):
        rp = replication_pack(None, tmp_path / "x.zip", env=False)
        s = rp.summary()
        assert "ReplicationPack" in s
        assert str(rp.output_path) in s


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TestData:
    def test_dataframe_packs_to_csv(self, tmp_path, df):
        out = tmp_path / "d.zip"
        rp = replication_pack(None, out, data=df, env=False)
        with _open_zip(rp) as zf:
            assert "data/dataset.csv" in zf.namelist()
            csv = zf.read("data/dataset.csv").decode()
            # Round-trips: parse back and compare.
            df2 = pd.read_csv(io.BytesIO(csv.encode()))
            pd.testing.assert_frame_equal(df.reset_index(drop=True),
                                           df2.reset_index(drop=True))

    def test_data_manifest_records_schema(self, tmp_path, df):
        rp = replication_pack(None, tmp_path / "d.zip", data=df, env=False)
        with _open_zip(rp) as zf:
            man = json.loads(zf.read("data/manifest.json"))
            assert man["kind"] == "DataFrame"
            assert man["shape"] == [10, 3]
            cols = {c["name"] for c in man["columns"]}
            assert cols == {"y", "treat", "x"}
            assert man["hash_sha256_prefix"]
            assert man["n_missing_total"] == 0

    def test_top_manifest_records_file_hashes(self, tmp_path, df):
        rp = replication_pack(None, tmp_path / "d.zip", data=df, env=False)
        with _open_zip(rp) as zf:
            man = json.loads(zf.read("MANIFEST.json"))
            entries = {f["path"]: f for f in man["files"]}
            assert "data/dataset.csv" in entries
            csv_bytes = zf.read("data/dataset.csv")
            import hashlib as _h
            assert entries["data/dataset.csv"]["sha256"] == \
                _h.sha256(csv_bytes).hexdigest()
            assert entries["data/dataset.csv"]["size_bytes"] == len(csv_bytes)


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------

class TestCode:
    def test_inline_code(self, tmp_path, script_text):
        rp = replication_pack(
            None, tmp_path / "c.zip", code=script_text, env=False,
        )
        with _open_zip(rp) as zf:
            assert "code/script.py" in zf.namelist()
            assert zf.read("code/script.py").decode() == script_text

    def test_code_from_path(self, tmp_path, script_text):
        sp_path = tmp_path / "analysis.py"
        sp_path.write_text(script_text, encoding="utf-8")
        rp = replication_pack(
            None, tmp_path / "c.zip", code=str(sp_path), env=False,
        )
        with _open_zip(rp) as zf:
            assert zf.read("code/script.py").decode() == script_text

    def test_no_code_warns(self, tmp_path):
        rp = replication_pack(None, tmp_path / "c.zip", env=False)
        assert any("no code" in w.lower() for w in rp.warnings)


# ---------------------------------------------------------------------------
# PaperDraft-like target
# ---------------------------------------------------------------------------

class _FakeWorkflow:
    def __init__(self, data, result):
        self.data = data
        self.result = result


class _FakePaperDraft:
    """Duck-type compatible with workflow.paper.PaperDraft for testing."""

    def __init__(self, sections, citations, fmt="markdown",
                 workflow=None, has_qmd=False):
        self.sections = sections
        self.citations = citations
        self.fmt = fmt
        self.workflow = workflow
        self._has_qmd = has_qmd

    def to_markdown(self) -> str:
        body = "\n\n".join(
            f"## {k}\n\n{v}" for k, v in self.sections.items()
        )
        return body

    def to_tex(self) -> str:
        return r"\documentclass{article}\begin{document}" + \
               self.to_markdown() + r"\end{document}"

    def to_docx(self, path):
        Path(path).write_bytes(b"PK\x03\x04 fake docx")

    def to_qmd(self) -> str:
        return "---\ntitle: 'fake'\n---\n\n" + self.to_markdown()


class TestPaperDraft:
    def test_paper_renders_to_markdown_by_default(self, tmp_path, df):
        result = SimpleNamespace(estimate=0.5)
        attach_provenance(result, function="sp.did.test", data=df)
        wf = _FakeWorkflow(data=df, result=result)
        draft = _FakePaperDraft(
            sections={"Question": "What's the effect?",
                      "Results": "Effect = 0.5"},
            citations=["Callaway B, Sant'Anna PHC. (2021). ..."],
            workflow=wf,
        )
        rp = replication_pack(draft, tmp_path / "p.zip", env=False)
        with _open_zip(rp) as zf:
            names = zf.namelist()
            assert "paper/paper.md" in names
            assert "paper/paper.bib" in names
            # Workflow.data picked up automatically.
            assert "data/dataset.csv" in names
            # Provenance picked up automatically.
            assert "lineage.json" in names

    def test_paper_format_qmd(self, tmp_path):
        draft = _FakePaperDraft(
            sections={"Q": "x"}, citations=[], fmt="markdown",
        )
        rp = replication_pack(
            draft, tmp_path / "p.zip", env=False, paper_format="qmd",
        )
        with _open_zip(rp) as zf:
            assert "paper/paper.qmd" in zf.namelist()
            qmd = zf.read("paper/paper.qmd").decode()
            assert qmd.startswith("---\n")

    def test_paper_format_tex(self, tmp_path):
        draft = _FakePaperDraft(sections={"Q": "x"}, citations=[])
        rp = replication_pack(
            draft, tmp_path / "p.zip", env=False, paper_format="tex",
        )
        with _open_zip(rp) as zf:
            assert "paper/paper.tex" in zf.namelist()


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

class TestLineage:
    def test_lineage_aggregates_from_results_list(self, tmp_path, df):
        r1 = SimpleNamespace()
        r2 = SimpleNamespace()
        attach_provenance(r1, function="sp.did.foo", data=df)
        attach_provenance(r2, function="sp.iv.bar", data=df)
        rp = replication_pack(
            [r1, r2], tmp_path / "l.zip", data=df, env=False,
        )
        with _open_zip(rp) as zf:
            assert "lineage.json" in zf.namelist()
            lin = json.loads(zf.read("lineage.json"))
            assert lin["n_runs"] == 2
            funcs = {v["function"] for v in lin["runs"].values()}
            assert funcs == {"sp.did.foo", "sp.iv.bar"}
            # Both consumed the same data hash → one input, two consumers.
            assert len(lin["data_inputs"]) == 1
            assert len(lin["data_inputs"][0]["consumers"]) == 2

    def test_no_lineage_when_no_provenance(self, tmp_path):
        rp = replication_pack(
            SimpleNamespace(), tmp_path / "l.zip", env=False,
        )
        with _open_zip(rp) as zf:
            assert "lineage.json" not in zf.namelist()


# ---------------------------------------------------------------------------
# Extras / overwrite
# ---------------------------------------------------------------------------

class TestExtras:
    def test_extra_files_included(self, tmp_path):
        rp = replication_pack(
            None, tmp_path / "e.zip",
            extra_files={"notes/HISTORY.md": "Run 1: pilot.",
                         "config.toml": b"[run]\nseed = 42\n"},
            env=False,
        )
        with _open_zip(rp) as zf:
            assert "notes/HISTORY.md" in zf.namelist()
            assert "config.toml" in zf.namelist()
            assert zf.read("notes/HISTORY.md").decode() == "Run 1: pilot."

    def test_overwrite_false_raises(self, tmp_path):
        out = tmp_path / "e.zip"
        out.write_bytes(b"existing")
        with pytest.raises(FileExistsError):
            replication_pack(None, out, env=False, overwrite=False)

    def test_overwrite_true_replaces(self, tmp_path):
        out = tmp_path / "e.zip"
        out.write_bytes(b"existing")
        rp = replication_pack(None, out, env=False, overwrite=True)
        # New file is a real zip, not the placeholder.
        with zipfile.ZipFile(out, "r") as zf:
            assert "MANIFEST.json" in zf.namelist()


# ---------------------------------------------------------------------------
# Manifest content
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_has_versions_and_timestamp(self, tmp_path, df,
                                                   script_text):
        rp = replication_pack(
            None, tmp_path / "m.zip",
            data=df, code=script_text, env=False,
        )
        m = rp.manifest
        assert m["statspai_version"]
        assert m["python_version"].count(".") == 2
        assert "T" in m["timestamp"]
        # data/dataset.csv + data/manifest.json + code/script.py + README.md
        assert isinstance(m["files"], list) and len(m["files"]) >= 4

    def test_files_listed_in_manifest_match_archive(self, tmp_path, df,
                                                      script_text):
        rp = replication_pack(
            None, tmp_path / "m.zip",
            data=df, code=script_text, env=False,
        )
        with _open_zip(rp) as zf:
            archive_names = set(zf.namelist())
        manifest_names = {f["path"] for f in rp.manifest["files"]}
        # MANIFEST.json itself isn't tracked as a "file" entry — we
        # don't checksum the manifest with itself.
        assert manifest_names <= archive_names
        for name in {"data/dataset.csv", "code/script.py", "README.md"}:
            assert name in manifest_names
            assert name in archive_names
