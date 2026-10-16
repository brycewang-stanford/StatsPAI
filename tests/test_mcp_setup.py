"""``statspai mcp doctor`` / ``statspai mcp config``.

The failure these commands exist for: a client launches ``statspai-mcp`` by
name, the name resolves to another Python environment, and the agent talks
to an old StatsPAI without anyone noticing. Tests use a temporary ``HOME``
so no real client configuration is read or written.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from statspai import cli
from statspai.agent import _mcp_setup as ms

POSIX = pytest.mark.skipif(sys.platform.startswith("win"), reason="shebang scripts")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _fake_python(path: Path, version: str) -> Path:
    """An executable named like an interpreter that reports ``version``."""
    payload = json.dumps({"version": version, "file": "/fake/statspai/__init__.py"})
    path.write_text(f"#!/bin/sh\necho '{payload}'\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# --------------------------------------------------------------------- #
#  config
# --------------------------------------------------------------------- #


def test_launch_command_is_this_interpreter_by_absolute_path():
    command, args = ms.launch_command("core")
    assert command == os.path.abspath(sys.executable)
    assert args == ["-m", "statspai.agent.mcp_server", "--profile", "core"]


@pytest.mark.parametrize("client", ["claude-desktop", "cursor", "windsurf", "vscode"])
def test_json_clients_render_a_parseable_block(client, home):
    text = ms.render_config(client, data_roots=[str(home)])
    header, body = text.split("\n", 1)
    assert header.startswith("// ")
    block = json.loads(body)
    key = "servers" if client == "vscode" else "mcpServers"
    entry = block[key]["statspai"]
    assert entry["command"] == os.path.abspath(sys.executable)
    assert entry["args"][:2] == ["-m", "statspai.agent.mcp_server"]
    assert entry["env"]["STATSPAI_MCP_DATA_ROOTS"] == str(home.resolve())
    if client == "vscode":
        assert entry["type"] == "stdio"


def test_claude_code_renders_cli_commands():
    text = ms.render_config("claude-code")
    assert "claude mcp add --scope user statspai -- " in text
    assert "statspai.agent.mcp_server" in text


def test_codex_renders_toml():
    tomllib = pytest.importorskip("tomllib")
    text = ms.render_config("codex", data_roots=["/data"])
    parsed = tomllib.loads(text)
    entry = parsed["mcp_servers"]["statspai"]
    assert entry["args"] == ["-m", "statspai.agent.mcp_server"]
    assert "STATSPAI_MCP_DATA_ROOTS" in entry["env"]


def test_unknown_client_is_rejected():
    with pytest.raises(ValueError, match="unknown client"):
        ms.render_config("notepad")


def test_write_merges_keeps_other_servers_and_backs_up(home):
    target = home / ".cursor" / "mcp.json"
    target.parent.mkdir()
    original = {
        "mcpServers": {
            "other": {"command": "npx", "args": ["x"]},
            "statspai": {"command": "statspai-mcp"},
        },
        "unrelated": 1,
    }
    target.write_text(json.dumps(original), encoding="utf-8")
    done = ms.write_config("cursor")
    assert done["path"] == str(target)
    assert done["replaced"] == {"command": "statspai-mcp"}
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["unrelated"] == 1
    assert written["mcpServers"]["other"] == original["mcpServers"]["other"]
    assert written["mcpServers"]["statspai"]["command"] == os.path.abspath(
        sys.executable
    )
    assert json.loads(Path(done["backup"]).read_text(encoding="utf-8")) == original


def test_write_creates_a_missing_file_without_backup(home):
    done = ms.write_config("vscode")
    assert done["backup"] is None
    assert Path(done["path"]) == home / ".vscode" / "mcp.json"
    assert "statspai" in json.loads(Path(done["path"]).read_text())["servers"]


def test_write_refuses_a_file_that_is_not_an_object(home):
    target = home / ".cursor" / "mcp.json"
    target.parent.mkdir()
    target.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        ms.write_config("cursor")
    assert target.read_text(encoding="utf-8") == "[1, 2]"


def test_write_refuses_clients_without_a_json_file(home):
    with pytest.raises(ValueError, match="statspai mcp config --client codex"):
        ms.write_config("codex")


def test_cli_config_and_write(home, capsys):
    assert cli.main(["mcp", "config", "--client", "cursor"]) == 0
    assert '"statspai"' in capsys.readouterr().out
    assert cli.main(["mcp", "config", "--client", "cursor", "--write"]) == 0
    done = json.loads(capsys.readouterr().out)
    assert Path(done["path"]).exists()
    assert cli.main(["mcp", "config"]) == cli.EXIT_USAGE


# --------------------------------------------------------------------- #
#  doctor
# --------------------------------------------------------------------- #


def test_probe_reports_this_interpreters_statspai():
    import statspai

    out = ms._probe_interpreter(sys.executable)
    # The probe drops PYTHONPATH, so in a worktree it may see another
    # checkout; either way it must name a version or say why not.
    assert out.get("version") or out.get("error")
    if os.environ.get("PYTHONPATH") is None:
        assert out["version"] == statspai.__version__


def test_probe_of_a_missing_interpreter_is_an_error():
    assert "error" in ms._probe_interpreter("/nonexistent/python3")


@POSIX
def test_entry_interpreter_from_console_script_shebang(tmp_path):
    py = _fake_python(tmp_path / "python3.99", "0.0.1")
    script = tmp_path / "statspai-mcp"
    script.write_text(f"#!{py}\nimport sys\n", encoding="utf-8")
    python, how = ms.resolve_entry_interpreter({"command": str(script)})
    assert python == str(py) and "shebang" in how
    python, how = ms.resolve_entry_interpreter({"command": str(py), "args": ["-m"]})
    assert python == str(py)
    assert ms.resolve_entry_interpreter({"command": "uvx"})[0] is None


@POSIX
def test_doctor_flags_a_client_running_an_old_version(home):
    old = _fake_python(home / "python-old", "0.0.1")
    (home / ".cursor").mkdir()
    (home / ".cursor" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"statspai": {"command": str(old), "args": []}}}),
        encoding="utf-8",
    )
    report = ms.run_doctor(self_test=False, cwd=home)
    assert not report["ok"]
    [entry] = [e for e in report["client_entries"] if e["client"] == "cursor"]
    assert entry["probe"]["version"] == "0.0.1"
    assert any("0.0.1" in p and "cursor" in p for p in report["problems"])
    assert "statspai mcp config" in " ".join(report["notes"])
    assert "Problems found." in ms.format_doctor(report)


def test_doctor_finds_claude_code_user_and_project_entries(home):
    claude = {
        "mcpServers": {"statspai": {"command": "uvx", "args": ["statspai"]}},
        "projects": {
            str(home): {"mcpServers": {"sp": {"command": "statspai-mcp-missing"}}}
        },
    }
    (home / ".claude.json").write_text(json.dumps(claude), encoding="utf-8")
    entries = ms.configured_entries(home)
    scopes = sorted(e["scope"] for e in entries if e["client"] == "claude-code")
    assert scopes == ["local", "user"]


def test_doctor_self_test_round_trip(home):
    report = ms.run_doctor(self_test=True, cwd=home)
    st = report["self_test"]
    assert st["initialize_ok"] and st["discover_ok"] and st["call_ok"], st
    assert st["tools"] > 0


def test_cli_doctor_json_exit_code(home, capsys):
    code = cli.main(["mcp", "doctor", "--json", "--no-self-test"])
    report = json.loads(capsys.readouterr().out)
    assert code == (0 if report["ok"] else 1)
    assert report["statspai_version"]


def test_server_version_flag(capsys):
    from statspai.agent import mcp_server as m

    with pytest.raises(SystemExit) as exc:
        m.main(["--version"])
    assert exc.value.code == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith(f"statspai-mcp {m.SERVER_VERSION} ")
    assert "2026-07-28" in line and "\n" not in line
