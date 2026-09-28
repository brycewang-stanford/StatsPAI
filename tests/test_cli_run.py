"""``statspai run`` / family shortcuts / ``route`` / ``mcp`` (roadmap W5).

Before these, the CLI could only list, describe and search; a shell agent
could not run an analysis and get JSON back. ``run`` and the shortcuts go
through the same dispatch layer as the MCP server, so the payload is the
agent payload and errors are structured.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from statspai.cli import SHORTCUTS, _make_parser, main


@pytest.fixture(scope="module")
def panel_csv(tmp_path_factory):
    rng = np.random.default_rng(0)
    n_units, n_periods = 50, 8
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_units), n_periods),
            "t": np.tile(np.arange(n_periods), n_units),
        }
    )
    df["g"] = np.where(df["id"] < 25, 4, 0)
    df["post"] = ((df["g"] > 0) & (df["t"] >= df["g"])).astype(int)
    df["y"] = rng.normal(size=len(df)) + 0.5 * df["post"] + 0.01 * df["id"]
    df["x"] = rng.normal(size=len(df))
    path = tmp_path_factory.mktemp("cli") / "panel.csv"
    df.to_csv(path, index=False)
    return str(path)


def _json_out(capsys) -> dict:
    out, _ = capsys.readouterr()
    return json.loads(out)


class TestRun:
    def test_run_returns_agent_payload(self, panel_csv, capsys):
        rc = main(
            [
                "run",
                "callaway_santanna",
                "--data",
                panel_csv,
                "--arg",
                "y=y",
                "--arg",
                "g=g",
                "--arg",
                "t=t",
                "--arg",
                "i=id",
            ]
        )
        assert rc == 0
        payload = _json_out(capsys)
        assert payload["method"].startswith("Callaway")
        assert payload["estimate"] == pytest.approx(0.73, abs=0.05)
        assert payload["n_obs"] == 400
        assert "violations" in payload and "next_steps" in payload
        assert payload["data_provenance"]["source_type"] == "local"

    def test_arg_values_parse_as_json(self, panel_csv, capsys):
        rc = main(
            [
                "run",
                "regress",
                "--data",
                panel_csv,
                "--arg",
                "formula=y ~ post + x",
                "--arg",
                'vce="hc1"',
                "--detail",
                "standard",
            ]
        )
        assert rc == 0
        payload = _json_out(capsys)
        assert payload["coefficients"]["post"]["estimate"] == pytest.approx(
            0.5, abs=0.3
        )

    def test_unsupported_args_are_reported_not_dropped(self, panel_csv, capsys):
        rc = main(
            [
                "run",
                "adjust_pvalues",
                "--arg",
                "pvalues=[0.01, 0.04]",
                "--arg",
                "clusterr=id",
                "--indent",
                "0",
            ]
        )
        assert rc == 0
        payload = _json_out(capsys)
        assert payload["_unsupported_args"] == ["clusterr"]

    def test_unknown_function_is_usage_error_with_suggestions(self, capsys):
        rc = main(["run", "calaway_santana", "--arg", "y=y"])
        assert rc == 2
        _, err = capsys.readouterr()
        payload = json.loads(err)
        assert "Unknown function" in payload["error"]
        assert "callaway_santanna" in payload["did_you_mean"]

    def test_estimator_error_is_structured_exit_3(self, panel_csv, capsys):
        rc = main(
            [
                "run",
                "callaway_santanna",
                "--data",
                panel_csv,
                "--arg",
                "y=wage",
                "--arg",
                "g=g",
                "--arg",
                "t=t",
                "--arg",
                "i=id",
            ]
        )
        assert rc == 3
        _, err = capsys.readouterr()
        payload = json.loads(err)
        assert "wage" in payload["error"]
        assert payload["error_kind"]
        assert "remediation" in payload

    def test_missing_data_file_is_usage_error(self, capsys):
        rc = main(
            ["run", "regress", "--data", "/nonexistent/x.csv", "--arg", "formula=y ~ x"]
        )
        assert rc == 2
        _, err = capsys.readouterr()
        assert json.loads(err)["error_kind"] == "data_load"

    def test_summary_format_and_out_file(self, panel_csv, capsys, tmp_path):
        out = tmp_path / "res.json"
        rc = main(
            [
                "run",
                "callaway_santanna",
                "--data",
                panel_csv,
                "--arg",
                "y=y",
                "--arg",
                "g=g",
                "--arg",
                "t=t",
                "--arg",
                "i=id",
                "--format",
                "summary",
                "--out",
                str(out),
            ]
        )
        assert rc == 0
        text, _ = capsys.readouterr()
        assert "ATT" in text
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["estimate"] == pytest.approx(0.73, abs=0.05)
        assert "result_id" not in saved


class TestShortcuts:
    def test_parser_has_schema_driven_flags(self):
        parser = _make_parser()
        sub = next(a for a in parser._actions if a.dest == "command")
        assert set(SHORTCUTS) <= set(sub.choices)
        did = sub.choices["did"]
        flags = {a.dest for a in did._actions}
        assert {"y", "treat", "time", "id", "method", "covariates", "cluster"} <= flags
        method = next(a for a in did._actions if a.dest == "method")
        assert "cs" in method.choices

    def test_did_shortcut(self, panel_csv, capsys):
        rc = main(
            [
                "did",
                "--data",
                panel_csv,
                "--y",
                "y",
                "--treat",
                "post",
                "--time",
                "t",
                "--id",
                "id",
                "--detail",
                "minimal",
            ]
        )
        assert rc == 0
        payload = _json_out(capsys)
        assert payload["method"].startswith("Callaway")
        assert payload["estimate"] == pytest.approx(0.73, abs=0.05)

    def test_json_typed_flag(self, panel_csv, capsys):
        rc = main(
            [
                "callaway_santanna",
                "--data",
                panel_csv,
                "--y",
                "y",
                "--g",
                "g",
                "--t",
                "t",
                "--i",
                "id",
                "--x",
                '["x"]',
                "--detail",
                "minimal",
            ]
        )
        assert rc == 0
        payload = _json_out(capsys)
        assert payload["n_obs"] == 400
        assert payload["method"].startswith("Callaway")


class TestRouteAndMcp:
    def test_route_json(self, capsys):
        rc = main(
            [
                "route",
                "did",
                "--answer",
                "design=staggered",
                "--answer",
                "timing_random=no",
                "--answer",
                "covariates=none",
                "--json",
            ]
        )
        assert rc == 0
        payload = _json_out(capsys)
        assert payload["routes"][0]["call"] == "callaway_santanna"

    def test_route_lists_families(self, capsys):
        assert main(["route", "--json"]) == 0
        assert set(_json_out(capsys)) >= {"did", "iv", "rd"}

    def test_route_bad_answer(self, capsys):
        rc = main(["route", "rd", "--answer", "running=wavy", "--json"])
        assert rc == 2
        _, err = capsys.readouterr()
        assert json.loads(err)["kind"] == "method_incompatibility"

    def test_mcp_subcommand_help(self):
        proc = subprocess.run(
            [sys.executable, "-m", "statspai.cli", "mcp", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0 and "--profile" in proc.stdout

    def test_run_over_subprocess_is_pure_json(self, panel_csv):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "statspai.cli",
                "run",
                "regress",
                "--data",
                panel_csv,
                "--arg",
                "formula=y ~ post",
                "--detail",
                "standard",
                "--indent",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-1000:]
        payload = json.loads(proc.stdout)
        assert payload["coefficients"]["post"]["estimate"] == pytest.approx(
            0.5, abs=0.3
        )
