"""``replication_pack(strict=)`` and ``sp.verify_replication_pack`` (review §6.5).

A pack that opens is not a replication: the verifier unpacks into a fresh
directory, checks hashes, reruns ``code/script.py`` in a subprocess and
compares every recorded estimate and SE with the rerun's.
"""

import json
import textwrap
import zipfile

import pytest

import statspai as sp

SCRIPT = textwrap.dedent("""
    import pandas as pd
    import statspai as sp

    df = pd.read_csv("data/dataset.csv")
    sp.regress("log_wage ~ education + experience", data=df, robust="hc1")
    sp.logit("union ~ education", data=df)
    """)


@pytest.fixture(scope="module")
def df():
    return sp.cps_wage()


def _fits(df):
    return [
        sp.regress("log_wage ~ education + experience", data=df, robust="hc1"),
        sp.logit("union ~ education", data=df),
    ]


def _pack(tmp_path, df, script=SCRIPT, **kw):
    out = tmp_path / "pack.zip"
    sp.replication_pack(
        _fits(df),
        out,
        data=df,
        code=script,
        env=False,
        bib=False,
        include_git_sha=False,
        **kw,
    )
    return out


def test_faithful_pack_verifies(tmp_path, df):
    v = sp.verify_replication_pack(_pack(tmp_path, df))
    assert v["status"] == "verified", v.summary()
    assert v["comparison"]["n_matched"] == 2
    assert v["integrity"]["status"] == "ok"


def test_script_that_changes_the_model_is_a_mismatch(tmp_path, df):
    drifted = SCRIPT.replace('robust="hc1"', 'robust="hc3"')
    v = sp.verify_replication_pack(_pack(tmp_path, df, script=drifted))
    assert v["status"] == "mismatch"
    assert "sp.regress" in v["comparison"]["missing"]


def test_tampered_file_fails_integrity(tmp_path, df):
    path = _pack(tmp_path, df)
    with zipfile.ZipFile(path) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    members["data/dataset.csv"] = members["data/dataset.csv"].replace(b"1", b"2", 1)
    with zipfile.ZipFile(path, "w") as zf:
        for n, b in members.items():
            zf.writestr(n, b)
    assert sp.verify_replication_pack(path)["status"] == "integrity_failed"


def test_crashing_script_is_rerun_failed(tmp_path, df):
    v = sp.verify_replication_pack(_pack(tmp_path, df, script="raise SystemExit(3)\n"))
    assert v["status"] == "rerun_failed"
    assert v["rerun"]["returncode"] == 3


def test_pack_without_code_is_incomplete(tmp_path, df):
    out = tmp_path / "nocode.zip"
    sp.replication_pack(
        _fits(df), out, data=df, env=False, bib=False, include_git_sha=False
    )
    assert sp.verify_replication_pack(out)["status"] == "incomplete_pack"


def test_strict_refuses_a_pack_that_cannot_be_rerun(tmp_path, df):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="missing code"):
        sp.replication_pack(
            _fits(df),
            tmp_path / "s.zip",
            data=df,
            env=False,
            bib=False,
            include_git_sha=False,
            strict=True,
        )
    ok = tmp_path / "ok.zip"
    sp.replication_pack(
        _fits(df),
        ok,
        data=df,
        code=SCRIPT,
        env=True,
        bib=False,
        include_git_sha=False,
        strict=True,
    )
    with zipfile.ZipFile(ok) as zf:
        manifest = json.loads(zf.read("MANIFEST.json"))
        records = json.loads(zf.read("results/results.json"))
    assert manifest["strict"] is True
    assert [r["function"] for r in records] == ["sp.regress", "sp.logit"]
