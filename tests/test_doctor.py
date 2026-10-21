"""``sp.doctor()`` / ``statspai doctor`` / ``python -m statspai --where``.

The report has to tell the truth about the running process (its own
version, interpreter, prefix, backends) and about the other interpreters
on PATH, and it must never raise because a probe failed. The MCP server
has to name its own interpreter, since a client cannot otherwise see which
environment its ``statspai-mcp`` entry resolved to.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import statspai as sp
from statspai import _doctor
from statspai._rust_backend import MIN_API_VERSION

SRC = str(Path(sp.__file__).resolve().parent.parent)


def test_report_describes_this_process():
    report = sp.doctor(probe_path=False)
    assert report["statspai"]["version"] == sp.__version__
    assert report["statspai"]["location"] == str(Path(sp.__file__).resolve().parent)
    assert report["python"]["executable"] == os.path.abspath(sys.executable)
    assert report["environment"]["prefix"] == sys.prefix
    assert report["interpreters_on_path"] is None
    assert report["ok"] is (not report["problems"])
    # JSON-ready: what the CLI's --json and the MCP layer serialise.
    json.dumps(report)


def test_backends_rows_match_importlib():
    import importlib.util

    report = sp.doctor(probe_path=False)
    assert set(report["backends"]) == set(_doctor.BACKENDS)
    for name, row in report["backends"].items():
        assert row["installed"] is (importlib.util.find_spec(name) is not None)
        if name in sys.modules:
            assert row["imports"] is True


def test_rust_block_reports_the_loader_decision():
    from statspai import _rust_backend as rb

    rust = sp.doctor(probe_path=False)["rust"]
    assert rust["available"] is rb.HAS_RUST
    assert rust["min_api_version"] == MIN_API_VERSION
    if rb.HAS_RUST:
        assert rust["api_version"] >= MIN_API_VERSION


def test_doctor_does_not_import_heavy_backends():
    """Without check_imports nothing optional is imported in-process."""
    code = (
        "import sys, statspai as sp\n"
        "before = set(sys.modules)\n"
        "sp.doctor(probe_path=False)\n"
        "new = set(sys.modules) - before\n"
        "heavy = {'torch', 'jax', 'pymc', 'pyfixest', 'polars', 'numba'}\n"
        "print(sorted(heavy & new))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": SRC},
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_lazy_export_is_off_the_cold_import_path():
    code = "import sys, statspai\nprint('statspai._doctor' in sys.modules)\n"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": SRC},
        timeout=300,
    )
    assert out.stdout.strip().splitlines()[-1] == "False", out.stderr


def test_probe_reads_another_interpreter(tmp_path, monkeypatch):
    """A probe of this very interpreter recovers its own prefix and Python."""
    row = _doctor._probe(sys.executable, timeout=60)
    assert "error" not in row, row
    assert row["python_version"] == sys.version.split()[0]
    assert os.path.realpath(row["prefix"]) == os.path.realpath(sys.prefix)


def test_probe_failures_become_fields_not_exceptions(tmp_path):
    missing = str(tmp_path / "no-such-python")
    assert "error" in _doctor._probe(missing, timeout=5)
    # An executable that prints garbage and exits non-zero.
    fake = tmp_path / "python3.99"
    fake.write_text("#!/bin/sh\necho not json\nexit 3\n")
    fake.chmod(0o755)
    row = _doctor._probe(str(fake), timeout=5)
    assert row["error"] == "not json"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell stub")
def test_path_scan_finds_interpreters_once(tmp_path, monkeypatch):
    """python / python3 / python3.x are found; symlinks to one binary count once."""
    real = tmp_path / "bin" / "python3.42"
    real.parent.mkdir()
    real.write_text("#!/bin/sh\nexit 0\n")
    real.chmod(0o755)
    (tmp_path / "bin" / "python3").symlink_to(real)
    (tmp_path / "bin" / "pythonw-not-me").write_text("")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    found = _doctor._path_interpreters()
    assert len(found) == 1
    assert os.path.realpath(found[0]) == str(real.resolve())


def test_secret_environment_values_are_redacted(monkeypatch):
    monkeypatch.setenv("STATSPAI_MCP_HTTP_TOKEN", "hunter2")
    monkeypatch.setenv("STATSPAI_SKIP_RUST", "")
    env = sp.doctor(probe_path=False)["env"]
    assert env["STATSPAI_MCP_HTTP_TOKEN"] == "<set>"
    assert "hunter2" not in json.dumps(env)


def test_bad_timeout_is_refused():
    with pytest.raises(sp.MethodIncompatibility):
        sp.doctor(probe_path=False, timeout=0)


def test_install_kind_names_a_shadowed_editable_install(tmp_path):
    install = {"editable": True, "url": f"file://{tmp_path / 'checkout'}"}
    kind = _doctor._install_kind(install, str(tmp_path / "elsewhere" / "statspai"))
    assert kind.startswith("on sys.path ahead of the editable install")
    here = tmp_path / "checkout" / "src" / "statspai"
    here.mkdir(parents=True)
    assert _doctor._install_kind(install, str(here)) == "editable install"
    assert _doctor._install_kind({"editable": False}, str(here)) == "installed package"


def test_format_is_plain_text_with_every_block():
    text = _doctor.format_doctor(sp.doctor(probe_path=False))
    for label in ("StatsPAI", "interpreter", "Rust kernels", "Rscript", "Stata"):
        assert label in text


def test_cli_doctor_json(capsys):
    from statspai.cli import main

    code = main(["doctor", "--json", "--no-path"])
    report = json.loads(capsys.readouterr().out)
    assert report["statspai"]["version"] == sp.__version__
    assert code == (0 if report["ok"] else 1)


def test_python_m_statspai_where():
    from statspai.__main__ import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        assert main(["--where"]) == 0
    lines = buf.getvalue().splitlines()
    assert lines[0] == f"StatsPAI {sp.__version__}"
    assert os.path.abspath(sys.executable) in lines[1]


def test_registered_with_schema():
    assert "doctor" in sp.list_functions()
    props = sp.function_schema("doctor")["parameters"]["properties"]
    assert set(props) == {"probe_path", "check_imports", "timeout"}


# --------------------------------------------------------------------- #
#  MCP: the server names its own interpreter
# --------------------------------------------------------------------- #


def _rpc(method, params):
    from statspai.agent import mcp_server as m

    line = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    return json.loads(m.handle_request(line))


def test_mcp_instructions_name_the_interpreter():
    init = _rpc("initialize", {"protocolVersion": "2025-06-18"})["result"]
    text = init["instructions"]
    assert f"StatsPAI {sp.__version__}" in text
    assert os.path.abspath(sys.executable) in text


def test_mcp_describe_function_and_preflight_carry_server_runtime(tmp_path):
    out = _rpc(
        "tools/call",
        {"name": "describe_function", "arguments": {"name": "regress"}},
    )["result"]["structuredContent"]
    rt = out["server_runtime"]
    assert rt["statspai_version"] == sp.__version__
    assert rt["python"] == os.path.abspath(sys.executable)

    csv = tmp_path / "panel.csv"
    rows = ["id,t,y,d"] + [
        f"{i},{t},{i + t + (i % 2) * (t >= 2)},{int(i % 2 and t >= 2)}"
        for i in range(20)
        for t in range(4)
    ]
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    pre = _rpc(
        "tools/call",
        {
            "name": "preflight",
            "arguments": {
                "data_path": str(csv),
                "method": "did",
                "y": "y",
                "treat": "d",
                "time": "t",
            },
        },
    )["result"]["structuredContent"]
    assert pre["server_runtime"]["python"] == os.path.abspath(sys.executable)


def test_mcp_doctor_reports_install_kind(tmp_path, monkeypatch):
    from statspai.agent import _mcp_setup as ms

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setattr(ms.shutil, "which", lambda name: None)
    report = ms.run_doctor(self_test=False, cwd=tmp_path)
    assert report["install_kind"] == _doctor.runtime_summary()["install_kind"]
    assert "install" in ms.format_doctor(report)
