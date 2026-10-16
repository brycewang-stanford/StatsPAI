"""The packaged skill over MCP (``io.modelcontextprotocol/skills``, SEP-2640).

Each check is what a host does before loading a skill: discover it, then read
every file in the manifest and verify byte size, SHA-256 digest and the
frontmatter field by field.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from statspai.agent import _protocol, _skills_ext
from statspai.agent import mcp_server as m

META = {
    _protocol.META_PROTOCOL_VERSION: "2026-07-28",
    _protocol.META_CLIENT_CAPABILITIES: {},
}


def _modern(method, **params):
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": {"_meta": dict(META), **params},
        }
    )
    return json.loads(m.handle_request(line))


def test_discover_declares_the_extension_and_resources():
    caps = _modern("server/discover")["result"]["capabilities"]
    assert "resources" in caps
    assert caps["extensions"][_skills_ext.SKILLS_EXTENSION] == {}


def test_list_entry_manifest_verifies_against_every_file():
    res = _modern("skills/list")["result"]
    assert res["resultType"] == "complete"
    assert res["ttlMs"] > 0 and res["cacheScope"] == "public"
    [entry] = res["skills"]
    assert entry["uri"] == "skill://statspai-analysis/SKILL.md"
    uris = [r["uri"] for r in entry["resources"]]
    assert entry["uri"] in uris
    assert len(uris) == len(set(uris)) and len(uris) <= 512
    assert sum(r["size"] for r in entry["resources"]) <= 16 * 1024 * 1024
    for item in entry["resources"]:
        read = _modern("resources/read", uri=item["uri"])["result"]
        raw = read["contents"][0]["text"].encode("utf-8")
        assert len(raw) == item["size"], item["uri"]
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == item["digest"]
    # Frontmatter matches the served SKILL.md; the directory name is the name.
    skill_md = _modern("resources/read", uri=entry["uri"])["result"]
    text = skill_md["contents"][0]["text"]
    assert _skills_ext.parse_frontmatter(text) == entry["frontmatter"]
    assert entry["frontmatter"]["name"] == entry["uri"].split("/")[-2]
    assert skill_md["contents"][0]["mimeType"] == "text/markdown"


def test_frontmatter_equals_what_a_yaml_parser_reads():
    yaml = pytest.importorskip("yaml")
    text = (_skills_ext.skill_dir() / "SKILL.md").read_text(encoding="utf-8")
    block = text.split("---", 2)[1]
    assert yaml.safe_load(block) == _skills_ext.parse_frontmatter(text)


def test_frontmatter_parser_refuses_what_it_cannot_read_exactly():
    with pytest.raises(ValueError, match="unsupported"):
        _skills_ext.parse_frontmatter("---\nname: x\nmetadata:\n  a: 1\n---\n")
    with pytest.raises(ValueError, match="begin"):
        _skills_ext.parse_frontmatter("# no frontmatter\n")


def test_get_by_uri_and_unknown_uri():
    got = _modern("skills/get", uri=_skills_ext.SKILL_MD_URI)["result"]
    assert got["skill"] == _modern("skills/list")["result"]["skills"][0]
    assert got["resultType"] == "complete" and got["ttlMs"] > 0
    err = _modern("skills/get", uri="skill://other/SKILL.md")["error"]
    assert err["code"] == -32602
    assert "skill://statspai-analysis/SKILL.md" in err["message"]


def test_unknown_skill_file_is_invalid_params():
    err = _modern("resources/read", uri="skill://statspai-analysis/nope.md")["error"]
    assert err["code"] == -32602
    assert "Unknown skill file" in err["message"]


def test_no_bytecode_or_hidden_files_in_the_manifest():
    entry = _skills_ext.skill_entry()
    for item in entry["resources"]:
        assert "__pycache__" not in item["uri"]
        assert not item["uri"].endswith((".pyc", ".pyo"))
        assert not item["uri"].rsplit("/", 1)[-1].startswith(".")
