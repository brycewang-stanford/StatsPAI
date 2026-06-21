"""Regression tests for the NaN/Inf JSON-leak that surfaced as

    "MCP statspai: No number after minus sign in JSON at position N"

in Claude Desktop. Root cause: ``json.dumps``'s ``default=`` callback
is **not** invoked for native Python ``float`` values, so
``float('-inf')`` was rendered as the literal token ``-Infinity``
(non-RFC-8259 — strict parsers reject it at the ``-`` with the
"No number after minus sign" message).

The fix walks every payload through :func:`_clean_floats` before
``json.dumps`` and hardens :func:`_json_default`'s numpy / pandas
branches to clean the lists/dicts they return.

These tests pin the contract: every json.dumps site in the MCP server
must produce STRICT JSON (re-parseable by ``json.loads``) even when the
underlying estimator emits ``nan`` / ``inf`` / ``-inf``.
"""

from __future__ import annotations

import json
from enum import Enum

import numpy as np
import pandas as pd
import pytest

from statspai.agent.mcp_server import (
    _clean_floats,
    _json_default,
    _jsonrpc_error,
    _jsonrpc_result,
    _make_progress_drain,
)

# ---------------------------------------------------------------------------
#  _clean_floats — the leaf scrubber
# ---------------------------------------------------------------------------


class TestCleanFloats:

    def test_native_nan_inf_become_none(self):
        assert _clean_floats(float("nan")) is None
        assert _clean_floats(float("inf")) is None
        assert _clean_floats(float("-inf")) is None

    def test_finite_floats_pass_through(self):
        assert _clean_floats(1.5) == 1.5
        assert _clean_floats(0.0) == 0.0
        assert _clean_floats(-2.5) == -2.5

    def test_walks_dict_recursively(self):
        cleaned = _clean_floats(
            {"a": 1.0, "b": float("-inf"), "c": {"d": float("nan")}}
        )
        assert cleaned == {"a": 1.0, "b": None, "c": {"d": None}}

    def test_walks_list_and_tuple(self):
        cleaned = _clean_floats([1.0, float("inf"), (float("nan"), 3.0)])
        # tuples normalise to lists (matches json.dumps behaviour)
        assert cleaned == [1.0, None, [None, 3.0]]

    def test_strings_and_bytes_left_alone(self):
        # Strings are iterable but must NOT be walked char-by-char.
        assert _clean_floats("hello") == "hello"
        assert _clean_floats(b"\xff\x00") == b"\xff\x00"

    def test_non_container_pass_through(self):
        # ints, bools, None all preserved exactly
        assert _clean_floats(42) == 42
        assert _clean_floats(True) is True
        assert _clean_floats(None) is None


# ---------------------------------------------------------------------------
#  _json_default — numpy / pandas pathways must also clean
# ---------------------------------------------------------------------------


class TestJsonDefaultNanInf:

    def test_numpy_array_with_inf_serialises_as_null(self):
        arr = np.array([1.0, np.nan, np.inf, -np.inf])
        text = json.dumps({"arr": arr}, default=_json_default)
        # Strict re-parse: would raise on '-Infinity' / 'NaN'
        parsed = json.loads(text)
        assert parsed == {"arr": [1.0, None, None, None]}

    def test_numpy_2d_array_with_inf(self):
        arr = np.array([[1.0, float("-inf")], [float("nan"), 2.0]])
        text = json.dumps({"arr": arr}, default=_json_default)
        parsed = json.loads(text)
        assert parsed == {"arr": [[1.0, None], [None, 2.0]]}

    def test_numpy_floating_scalar_inf_via_production_path(self):
        """``np.float64`` IS a Python ``float`` subclass — json.dumps'
        ``default=`` callback is therefore **bypassed** (Python's json
        module sees it as natively serialisable). This is precisely why
        :func:`_clean_floats` is needed as a pre-walk; the production
        path goes through :func:`_jsonrpc_result` which always pre-walks,
        so the leak is plugged.

        Pin: assert the **production envelope** path scrubs np.float64
        nan/inf even though _json_default alone cannot.
        """
        env = _jsonrpc_result(0, {"v": np.float64("-inf"), "w": np.float64("nan")})
        parsed = json.loads(env)  # strict re-parse
        assert parsed["result"] == {"v": None, "w": None}
        # Sanity check: the float-subclass gotcha is real — _json_default
        # alone (no pre-walk) leaks. This documents WHY _clean_floats
        # has to exist as the outer wrapper.
        leaked = json.dumps({"v": np.float64("-inf")}, default=_json_default)
        assert "-Infinity" in leaked, (
            "If this assertion ever flips, json.dumps grew a way to "
            "intercept float-subclass nan/inf and _clean_floats "
            "could be simplified."
        )

    def test_pandas_series_with_inf(self):
        s = pd.Series({"a": 1.0, "b": float("-inf"), "c": float("nan")})
        text = json.dumps({"s": s}, default=_json_default)
        parsed = json.loads(text)
        assert parsed == {"s": {"a": 1.0, "b": None, "c": None}}

    def test_pandas_dataframe_with_inf(self):
        df = pd.DataFrame({"x": [1.0, float("-inf")], "y": [float("nan"), 2.0]})
        text = json.dumps({"df": df}, default=_json_default)
        parsed = json.loads(text)
        assert parsed == {"df": {"x": [1.0, None], "y": [None, 2.0]}}

    def test_pandas_index_with_inf(self):
        idx = pd.Index([1.0, float("-inf"), 3.0])
        text = json.dumps({"idx": idx}, default=_json_default)
        assert json.loads(text) == {"idx": [1.0, None, 3.0]}

    def test_numpy_complex_with_inf_via_production_path(self):
        env = _jsonrpc_result(
            0, {"z": np.complex128(complex(float("inf"), float("nan")))}
        )
        parsed = json.loads(env)
        assert parsed["result"] == {"z": {"real": None, "imag": None}}

    def test_pandas_interval_with_inf_via_production_path(self):
        interval = pd.Interval(left=float("-inf"), right=float("inf"), closed="both")
        env = _jsonrpc_result(0, {"interval": interval})
        parsed = json.loads(env)
        assert parsed["result"] == {
            "interval": {"left": None, "right": None, "closed": "both"}
        }

    def test_enum_value_with_inf_via_production_path(self):
        class BadValue(Enum):
            value = {"cutoff": float("-inf")}

        env = _jsonrpc_result(0, {"enum": BadValue.value})
        parsed = json.loads(env)
        assert parsed["result"] == {"enum": {"cutoff": None}}


# ---------------------------------------------------------------------------
#  JSON-RPC envelopes — the actual wire-format paths Claude Desktop sees
# ---------------------------------------------------------------------------


class TestJsonRpcEnvelopes:

    def test_jsonrpc_result_strips_nan_inf_from_result(self):
        # Simulates a tool returning a degenerate-model result with
        # -inf standard error and nan t-stat (perfect collinearity).
        env = _jsonrpc_result(
            42,
            {
                "coef": 1.5,
                "se": float("-inf"),
                "tstat": float("nan"),
                "ci": [-2.0, float("inf")],
                "diag": {"r2": float("nan")},
            },
        )
        # The original bug surfaced precisely here: '-Infinity' would
        # crash strict JSON.parse. Re-parse must succeed.
        parsed = json.loads(env)
        assert parsed["result"]["se"] is None
        assert parsed["result"]["tstat"] is None
        assert parsed["result"]["ci"] == [-2.0, None]
        assert parsed["result"]["diag"] == {"r2": None}
        # Non-pathological values preserved
        assert parsed["result"]["coef"] == 1.5

    def test_jsonrpc_error_strips_nan_inf_from_data(self):
        env = _jsonrpc_error(
            request_id=7,
            code=-32000,
            message="degenerate model",
            data={"se": float("-inf"), "what": "perfect collinearity"},
        )
        parsed = json.loads(env)
        assert parsed["error"]["data"]["se"] is None
        assert parsed["error"]["data"]["what"] == "perfect collinearity"

    def test_no_invalid_json_tokens_in_output(self):
        """The original failure mode emitted '-Infinity' / 'NaN' / '
        Infinity' tokens. Pin: never appears in any envelope output."""
        env = _jsonrpc_result(
            1, {"a": float("-inf"), "b": float("nan"), "c": float("inf")}
        )
        # These substrings are the smoking-gun tokens — none should
        # appear in the wire output (would only show inside a string,
        # but we don't put them in strings either).
        assert "-Infinity" not in env
        assert "Infinity" not in env
        assert "NaN" not in env


# ---------------------------------------------------------------------------
#  Progress notifications — same json.dumps pathway, separate site
# ---------------------------------------------------------------------------


class TestProgressDrain:

    def test_progress_drain_accepts_inf_payload(self, monkeypatch):
        """Patches the module-level _PROGRESS_SINK so _make_progress_drain
        returns a real serialiser; capture its output and re-parse."""
        from statspai.agent import mcp_server as srv

        captured = []

        class _FakeSink:
            def write(self, s):
                captured.append(s)

            def flush(self):
                pass

        monkeypatch.setattr(srv, "_PROGRESS_SINK", _FakeSink(), raising=False)
        drain = srv._make_progress_drain()
        # A pathological progress payload (shouldn't happen in normal
        # tool code, but defends against estimators that pass through
        # raw float divisions).
        drain(
            {"progressToken": "tok-1", "progress": float("-inf"), "total": float("nan")}
        )
        assert captured, "progress drain produced no output"
        msg = captured[0].rstrip("\n")
        parsed = json.loads(msg)  # strict re-parse
        assert parsed["params"]["progress"] is None
        assert parsed["params"]["total"] is None


# ---------------------------------------------------------------------------
#  Re-parseability harness — the original failure was a re-parse error.
# ---------------------------------------------------------------------------


class TestStrictReparse:

    @pytest.mark.parametrize(
        "payload",
        [
            {"x": float("-inf")},
            {"x": float("inf")},
            {"x": float("nan")},
            {"nested": {"deep": {"x": float("-inf")}}},
            {"arr": [float("-inf"), float("nan"), 1.0]},
            {"np": np.array([1.0, np.nan, -np.inf])},
            {"df": pd.DataFrame({"a": [1.0, float("-inf")]})},
        ],
    )
    def test_strict_parse_after_envelope(self, payload):
        env = _jsonrpc_result(1, payload)
        # Strict parser — would raise JSONDecodeError on -Infinity / NaN
        json.loads(env)
