"""Agent error contract: typed column / dependency / argument errors.

Covers the four mechanically-repairable failure families and how they
surface through every agent-facing layer:

* ``ColumnNotFound`` (kind ``column_not_found``) with ``did_you_mean``,
  still a ``MethodIncompatibility`` / ``DataInsufficient`` / ``ValueError``.
* ``MissingDependencyError`` (kind ``missing_dependency``) with the exact
  ``pip install`` command, still an ``ImportError``.
* ``remediate()`` mapping plain ``TypeError`` / ``ImportError`` into
  ``missing_arguments`` / ``unknown_argument`` / ``missing_dependency``.
* pipeline stages keeping the structured error instead of a string, and
  recording serializer fallbacks as degradations.
* the CLI's strict-JSON stdout, ``runtime_warnings`` and kind-based exit
  codes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent.remediation import remediate
from statspai.exceptions import (
    ColumnNotFound,
    DataInsufficient,
    MethodIncompatibility,
    MissingDependencyError,
    StatsPAIError,
    suggest_columns,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_units, n_periods = 40, 6
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_units), n_periods),
            "t": np.tile(np.arange(n_periods), n_units),
        }
    )
    df["d"] = ((df["id"] < 20) & (df["t"] >= 3)).astype(int)
    df["y"] = rng.normal(size=len(df)) + 0.5 * df["d"]
    df["x"] = rng.normal(size=len(df))
    return df


@pytest.fixture(scope="module")
def panel_csv(panel, tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("w9cli") / "panel.csv"
    panel.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def no_pyfixest(monkeypatch):
    """Simulate an environment without the ``fixest`` extra."""
    monkeypatch.setitem(sys.modules, "pyfixest", None)


# ---------------------------------------------------------------------------
# ColumnNotFound
# ---------------------------------------------------------------------------


class TestColumnNotFound:
    def test_did_typo_raises_column_not_found_with_suggestion(self, panel):
        with pytest.raises(ColumnNotFound) as info:
            sp.did(panel, y="yy", treat="d", id="id", time="t")
        err = info.value
        payload = err.to_dict()
        assert payload["kind"] == "column_not_found"
        assert payload["diagnostics"]["missing_columns"] == {"y": "yy"}
        assert payload["diagnostics"]["did_you_mean"] == {"y": "y"}
        assert set(payload["diagnostics"]["available_columns"]) == set(panel.columns)
        assert "did you mean 'y'" in err.recovery_hint
        assert "y='yy'" in err.recovery_hint

    def test_backward_compatible_hierarchy(self, panel):
        # Every except-clause that caught the old errors must still work.
        for exc_type in (MethodIncompatibility, DataInsufficient, ValueError):
            with pytest.raises(exc_type):
                sp.did(panel, y="yy", treat="d", id="id", time="t")

    def test_no_suggestion_when_nothing_is_close(self, panel):
        with pytest.raises(ColumnNotFound) as info:
            sp.did(panel, y="qqqqqq", treat="d", id="id", time="t")
        assert info.value.diagnostics["did_you_mean"] == {}
        assert "did you mean" not in info.value.recovery_hint

    def test_case_insensitive_exact_match_wins(self):
        assert suggest_columns(["Wage"], ["wages", "wage"]) == {"Wage": "wage"}

    def test_shared_helper_raises_column_not_found(self, panel):
        from statspai._input_validation import require_columns

        with pytest.raises(ColumnNotFound, match="Missing columns") as info:
            require_columns(panel, ["y", "xx"], function="demo")
        assert info.value.diagnostics["missing_columns"] == ["xx"]
        assert info.value.diagnostics["did_you_mean"] == {"xx": "x"}
        assert isinstance(info.value, DataInsufficient)

    def test_remediation_suggests_for_foreign_missing_column_errors(self):
        # Estimators with their own validators use the same diagnostics
        # keys; remediation must add did_you_mean for them too.
        err = MethodIncompatibility(
            "Column(s)/Covariate(s) not found in data: ['wage']",
            diagnostics={
                "missing_columns": ["wage"],
                "available_columns": ["Wage", "id"],
            },
        )
        out = remediate(err)
        assert out["category"] == "column_not_found"
        assert out["did_you_mean"] == {"wage": "Wage"}


# ---------------------------------------------------------------------------
# MissingDependencyError
# ---------------------------------------------------------------------------


class TestMissingDependency:
    def test_feols_without_pyfixest(self, panel, no_pyfixest):
        with pytest.raises(MissingDependencyError) as info:
            sp.feols("y ~ x | id", data=panel)
        err = info.value
        assert isinstance(err, ImportError) and isinstance(err, StatsPAIError)
        payload = err.to_dict()
        assert payload["kind"] == "missing_dependency"
        assert payload["diagnostics"]["package"] == "pyfixest"
        assert payload["diagnostics"]["extra"] == "fixest"
        assert payload["diagnostics"]["install"] == 'pip install "statspai[fixest]"'
        assert isinstance(err.__cause__, ImportError)

    def test_old_except_importerror_still_catches(self, panel, no_pyfixest):
        with pytest.raises(ImportError):
            sp.feols("y ~ x | id", data=panel)

    def test_require_optional_returns_module(self):
        from statspai._optional_deps import require_optional

        assert require_optional("json").__name__ == "json"

    def test_require_optional_without_extra(self, monkeypatch):
        from statspai._optional_deps import require_optional

        monkeypatch.setitem(sys.modules, "notapkg_w9", None)
        with pytest.raises(MissingDependencyError) as info:
            require_optional("notapkg_w9", pip_name="not-a-pkg")
        assert info.value.diagnostics["install"] == "pip install not-a-pkg"
        assert info.value.diagnostics["extra"] is None

    def test_plain_importerror_with_pip_hint_maps(self):
        out = remediate(ImportError("torch missing. Run `pip install torch`."))
        assert out["category"] == "missing_dependency"
        assert out["install"] == "pip install torch"

    def test_quoted_extra_is_extracted_whole(self):
        out = remediate(ImportError('Install with pip install "statspai[bayes]".'))
        assert out["install"] == 'pip install "statspai[bayes]"'

    def test_module_not_found_uses_module_name(self):
        err = ModuleNotFoundError("No module named 'xgboost.core'", name="xgboost.core")
        out = remediate(err)
        assert out["category"] == "missing_dependency"
        assert out["package"] == "xgboost"
        assert out["install"] == "pip install xgboost"


# ---------------------------------------------------------------------------
# TypeError remediation
# ---------------------------------------------------------------------------


class TestArgumentRemediation:
    def test_missing_required_arguments(self, panel):
        with pytest.raises(TypeError) as info:
            sp.did(panel)
        out = remediate(info.value)
        assert out["category"] == "missing_arguments"
        assert out["missing_arguments"] == ["y", "treat", "time"]
        assert out["function"] == "did"
        assert {"y", "treat", "time"} <= set(out["required"])

    def test_missing_argument_uses_context_when_message_has_no_name(self):
        err = TypeError("missing a required argument: 'y'")
        out = remediate(err, context={"tool": "did"})
        assert out["category"] == "missing_arguments"
        assert out["missing_arguments"] == ["y"]
        assert out["function"] == "did"

    def test_unknown_keyword_did_you_mean(self, panel):
        with pytest.raises(TypeError) as info:
            sp.regress("y ~ d", data=panel, vcee="hc1")
        out = remediate(info.value)
        assert out["category"] == "unknown_argument"
        assert out["unknown_arguments"] == ["vcee"]
        assert out["did_you_mean"] == {"vcee": "vce"}

    def test_plain_python_unexpected_keyword(self):
        err = TypeError("did() got an unexpected keyword argument 'tretment'")
        out = remediate(err)
        assert out["category"] == "unknown_argument"
        # Matches the ``treatment`` alias as well as parameter names.
        assert out["did_you_mean"]["tretment"] in {"treatment", "treat"}

    def test_unrelated_typeerror_is_not_misclassified(self):
        out = remediate(TypeError("unsupported operand type(s) for +: 'int'"))
        assert out["category"] not in {"missing_arguments", "unknown_argument"}


# ---------------------------------------------------------------------------
# Pipeline tools
# ---------------------------------------------------------------------------


class TestPipelineStructuredErrors:
    def test_primary_failure_keeps_structured_payload(self, panel):
        from statspai.agent.pipeline_tools import execute_pipeline_tool

        out = execute_pipeline_tool(
            "pipeline_did", {"y": "yy", "treat": "d", "time": "t"}, data=panel
        )
        assert out["error_kind"] == "column_not_found"
        payload = out["error_payload"]
        assert payload["diagnostics"]["did_you_mean"] == {"y": "y"}
        assert payload["recovery_hint"]
        stage = next(s for s in out["stages"] if s["name"] == "estimate")
        assert stage["status"] == "failed"
        assert stage["error"]["kind"] == "column_not_found"
        # The legacy string summary is preserved.
        assert stage["summary"].startswith("ColumnNotFound:")
        json.dumps(out)  # the envelope stays JSON-serialisable

    def test_non_taxonomy_failure_gets_remediation_payload(self):
        from statspai.agent.pipeline_tools import _safe_call

        def boom(**kw):
            raise TypeError("boom() missing 1 required positional argument: 'y'")

        result, err = _safe_call(boom)
        assert result is None
        assert str(err).startswith("TypeError: boom()")
        assert err.payload["kind"] == "missing_arguments"
        assert err.payload["remediation"]["missing_arguments"] == ["y"]

    def test_serializer_fallback_is_recorded_not_silent(self, monkeypatch):
        from statspai.agent import pipeline_tools, tools
        from statspai.workflow._degradation import WorkflowDegradedWarning

        def bad_serializer(obj, detail="standard"):
            raise RuntimeError("cannot serialise")

        monkeypatch.setattr(tools, "_default_serializer", bad_serializer)
        bag: list = []
        with pytest.warns(WorkflowDegradedWarning):
            out = pipeline_tools._light_serialize(object(), bag)
        assert out["degraded"] is True
        assert bag and bag[0]["error_type"] == "RuntimeError"
        with pytest.warns(WorkflowDegradedWarning):
            assert pipeline_tools._short_estimate(object(), bag) == ""
        assert len(bag) == 2

    def test_non_dict_serializer_output_is_recorded(self, monkeypatch):
        from statspai.agent import pipeline_tools, tools
        from statspai.workflow._degradation import WorkflowDegradedWarning

        monkeypatch.setattr(
            tools, "_default_serializer", lambda obj, detail="standard": [1, 2]
        )
        bag: list = []
        with pytest.warns(WorkflowDegradedWarning):
            out = pipeline_tools._light_serialize(object(), bag)
        assert out["degraded"] is True and bag[0]["error_type"] == "TypeError"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "statspai.cli", *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


class TestCLIContract:
    def test_column_typo_exit_4_with_did_you_mean(self, panel_csv):
        proc = _run_cli(
            "did", "--data", panel_csv, "--y", "yy", "--treat", "d", "--time", "t"
        )
        assert proc.returncode == 4, proc.stderr[-2000:]
        assert proc.stdout == ""
        err = json.loads(proc.stderr)
        assert err["error_kind"] == "column_not_found"
        assert err["remediation"]["did_you_mean"] == {"y": "y"}

    def test_success_stdout_is_single_strict_json(self, panel_csv):
        proc = _run_cli(
            "did",
            "--data",
            panel_csv,
            "--y",
            "y",
            "--treat",
            "d",
            "--time",
            "t",
            "--id",
            "id",
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        payload = json.loads(
            proc.stdout,
            parse_constant=lambda c: pytest.fail(f"non-strict JSON token {c}"),
        )
        assert isinstance(payload, dict) and "estimate" in payload

    def test_unknown_argument_exit_4(self, panel_csv):
        proc = _run_cli(
            "run",
            "regress",
            "--data",
            panel_csv,
            "--arg",
            "formula=y ~ d",
            "--arg",
            "vcee=hc1",
        )
        # The dispatcher drops unbindable kwargs and reports them; either
        # way the run must not crash and stdout must stay JSON.
        if proc.returncode == 0:
            assert json.loads(proc.stdout)["_unsupported_args"] == ["vcee"]
        else:
            assert proc.returncode == 4
            assert json.loads(proc.stderr)["error_kind"] == "unknown_argument"

    def test_generic_estimator_error_still_exit_3(self, panel_csv):
        proc = _run_cli(
            "run",
            "did",
            "--data",
            panel_csv,
            "--arg",
            "y=y",
            "--arg",
            "treat=d",
            "--arg",
            "time=t",
            "--arg",
            "method=not_a_method",
        )
        assert proc.returncode == 3, proc.stderr[-2000:]
        assert json.loads(proc.stderr)["error_kind"] == "method_incompatibility"


class TestCLIInProcess:
    def test_missing_dependency_exit_5(self, panel_csv, no_pyfixest, capsys):
        from statspai.cli import main

        rc = main(["run", "feols", "--data", panel_csv, "--arg", "fml=y ~ x | id"])
        _, err = capsys.readouterr()
        assert rc == 5, err[-2000:]
        payload = json.loads(err)
        assert payload["error_kind"] == "missing_dependency"
        assert payload["remediation"]["install"] == 'pip install "statspai[fixest]"'

    def test_nan_and_arrays_are_strict_json(self, monkeypatch, capsys):
        from statspai import cli
        from statspai.agent import tools

        def fake_execute(name, arguments, **kwargs):
            print("estimator chatter")  # must not reach stdout
            warnings.warn("few clusters", UserWarning)
            warnings.warn("few clusters", UserWarning)  # de-duplicated
            return {
                "estimate": float("nan"),
                "vec": np.array([1.0, np.nan, np.inf]),
                "frame": pd.DataFrame({"a": [1.5, np.nan]}),
                "n": np.int64(3),
            }

        monkeypatch.setattr(tools, "execute_tool", fake_execute)
        rc = cli.main(["run", "adjust_pvalues", "--arg", "pvalues=[0.1]"])
        out, err = capsys.readouterr()
        assert rc == 0
        assert "estimator chatter" in err and "estimator chatter" not in out
        payload = json.loads(
            out, parse_constant=lambda c: pytest.fail(f"non-strict token {c}")
        )
        assert payload["estimate"] is None
        assert payload["vec"] == [1.0, None, None]
        assert payload["frame"] == {"a": [1.5, None]}
        assert payload["n"] == 3
        assert payload["runtime_warnings"] == [
            {"category": "UserWarning", "message": "few clusters"}
        ]

    def test_local_json_fallback(self, monkeypatch):
        from statspai import cli

        monkeypatch.setattr(
            cli, "_json_helpers", lambda: (cli._local_clean, cli._local_json_default)
        )
        text = cli._dumps(
            {"a": np.array([np.nan, 2.0]), "b": {1, 2}, "c": float("inf")}, indent=0
        )
        assert json.loads(text) == {"a": [None, 2.0], "b": [1, 2], "c": None}

    def test_error_kind_prefers_actionable_remediation_category(self):
        from statspai.cli import KIND_EXIT_CODES, _error_kind

        payload = {
            "error": "x",
            "error_kind": "method_incompatibility",
            "remediation": {"category": "column_not_found"},
        }
        assert _error_kind(payload) == "column_not_found"
        assert KIND_EXIT_CODES["column_not_found"] == 4
        assert _error_kind({"error": "x", "remediation": {"category": "formula"}}) == (
            "formula"
        )
