"""Output budget: risk detail is never lost silently, and the budget reports.

The 2026-10-02 repository review reproduced two gaps in
``statspai.agent._output_budget``:

* a tight budget cut 100 ``violations`` to one and 100 ``runtime_warnings``
  to eleven, leaving a result that reads as nearly clean unless the agent
  decodes ``truncated``;
* ``{"estimate": list(range(1000))}`` under a 100-byte budget came back at
  3,904 bytes with an empty ledger -- the budget was best effort and said
  nothing when it failed.

Risk lists are now cut after everything else and leave ``risk_summary`` +
``risk_details_complete: false``; ``output_budget`` states whether the
object was cut to fit or could not be.
"""

from __future__ import annotations

import json
import warnings

import pytest

from statspai.agent import mcp_server
from statspai.agent._output_budget import (
    RISK_KEYS,
    apply_budget,
    json_size,
    note_risk_omission,
)
from statspai.agent.mcp_server import handle_request


def _violations(n):
    return [
        {
            "kind": "assumption",
            "severity": "error" if i % 4 == 0 else "warning",
            "test": ("pretrend", "weak_iv", "overlap")[i % 3],
            "message": f"violation {i}: " + "m" * 60,
        }
        for i in range(n)
    ]


def _item(key, i):
    if key == "violations":
        return _violations(i + 1)[i]
    if key == "runtime_warnings":
        return {"category": "ConvergenceWarning", "message": f"w{i} " + "x" * 70}
    if key == "degradations":
        return {
            "section": f"step{i % 5}",
            "error_type": "ValueError",
            "message": "d" * 70,
        }
    return f"plain warning {i} " + "y" * 60


@pytest.mark.parametrize("key", RISK_KEYS)
def test_cut_risk_list_leaves_a_summary(key):
    obj = {"estimate": 1.0, "std_error": 0.1, key: [_item(key, i) for i in range(100)]}
    out, records = apply_budget(obj, 1500)
    assert json_size(out) <= 1500
    assert out["risk_details_complete"] is False
    entry = out["risk_summary"][key]
    assert entry["total"] == 100
    assert entry["shown"] == len(out[key]) < 100
    assert entry["omitted"] == 100 - len(out[key])
    assert "max_output_bytes" in out["risk_summary"]["full_details"]
    assert {r["path"] for r in records} == {f"/{key}"}
    assert out["truncated"] == records
    assert out["estimate"] == 1.0 and out["std_error"] == 0.1


def test_summary_counts_describe_the_full_list_not_the_shown_part():
    obj = {"estimate": 1.0, "violations": _violations(100)}
    out, _ = apply_budget(obj, 1500)
    entry = out["risk_summary"]["violations"]
    assert entry["by_severity"] == {"error": 25, "warning": 75}
    assert entry["categories"] == {"pretrend": 34, "weak_iv": 33, "overlap": 33}
    assert sum(entry["by_severity"].values()) == entry["total"]


def test_every_risk_category_survives_together():
    obj = {"estimate": 1.0}
    for key in RISK_KEYS:
        obj[key] = [_item(key, i) for i in range(100)]
    out, _ = apply_budget(obj, 2500)
    assert json_size(out) <= 2500
    assert out["risk_details_complete"] is False
    for key in RISK_KEYS:
        assert len(out[key]) >= 1
        assert out["risk_summary"][key]["total"] == 100


def test_tables_are_cut_before_risk_lists():
    obj = {
        "estimate": 1.0,
        "violations": _violations(5),
        "runtime_warnings": [_item("runtime_warnings", i) for i in range(5)],
        "weights": list(range(20_000)),
    }
    out, records = apply_budget(obj, 4000)
    assert json_size(out) <= 4000
    assert len(out["violations"]) == 5 and len(out["runtime_warnings"]) == 5
    assert "risk_details_complete" not in out and "risk_summary" not in out
    assert [r["path"] for r in records] == ["/weights"]
    assert out["output_budget"]["status"] == "truncated"


def test_object_that_fits_is_returned_untouched():
    obj = {"estimate": 1.0, "violations": _violations(3)}
    before = json.dumps(obj, sort_keys=True)
    out, records = apply_budget(obj, 100_000)
    assert records == []
    assert json.dumps(out, sort_keys=True) == before


def test_overflow_of_never_cut_fields_is_reported():
    obj = {"estimate": list(range(1000))}
    out, records = apply_budget(obj, 100)
    assert records == []
    assert out["estimate"] == list(range(1000))
    ledger = out["output_budget"]
    assert ledger["status"] == "unavoidable_overflow"
    assert ledger["max_bytes"] == 100
    assert ledger["actual_bytes"] == json_size(out) > 100
    assert ledger["scope"] == "structuredContent"
    assert ledger["oversized_fields"][0]["path"] == "/estimate"


@pytest.mark.parametrize("budget", [600, 1500, 5000, 20_000])
def test_actual_bytes_is_exact_and_within_budget_when_truncated(budget):
    obj = {
        "estimate": 1.0,
        "note": "说明" * 3000,  # non-ASCII text
        "violations": _violations(40),
        "table": {f"k{i}": {"v": i} for i in range(400)},
    }
    out, _ = apply_budget(obj, budget)
    ledger = out["output_budget"]
    assert ledger["actual_bytes"] == json_size(out)
    if ledger["status"] == "truncated":
        assert ledger["actual_bytes"] <= budget
    else:
        assert ledger["actual_bytes"] > budget


def test_result_card_evidence_is_never_cut():
    evidence = {
        "level": "configuration",
        "status": "estimate_only",
        "outputs": {"estimate": "reference", "se": "not_covered"},
        "artifacts": [f"tests/r_parity/{i:02d}_module.py" for i in range(60)],
    }
    obj = {
        "estimate": 1.0,
        "result_card": {
            "evidence": json.loads(json.dumps(evidence)),
            "x": list(range(900)),
        },
    }
    out, records = apply_budget(obj, 4500)
    assert out["result_card"]["evidence"] == evidence
    assert [r["path"] for r in records] == ["/result_card/x"]


def test_note_risk_omission_reports_a_producer_side_cap():
    obj = {"runtime_warnings": [_item("runtime_warnings", i) for i in range(20)]}
    note_risk_omission(obj, "runtime_warnings", 20)
    assert "risk_summary" not in obj
    note_risk_omission(obj, "runtime_warnings", 57)
    assert obj["risk_details_complete"] is False
    entry = obj["risk_summary"]["runtime_warnings"]
    assert (entry["total"], entry["shown"], entry["omitted"]) == (57, 20, 37)
    assert entry["counts_cover"] == "shown"


# ---------------------------------------------------------------------------
# Through tools/call
# ---------------------------------------------------------------------------


def _call(name, **arguments):
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    return json.loads(handle_request(raw))["result"]


def _fake(fn):
    def _execute(name, arguments, **kwargs):
        return fn()

    return _execute


def test_tools_call_keeps_risk_visible_under_a_tight_budget(monkeypatch):
    payload = {
        "estimate": 0.4,
        "std_error": 0.1,
        "violations": _violations(100),
        "degradations": [_item("degradations", i) for i in range(100)],
        "coefficients": {
            f"x{i}": {"estimate": i, "std_error": 1.0} for i in range(300)
        },
    }
    monkeypatch.setattr(
        mcp_server, "execute_tool", _fake(lambda: json.loads(json.dumps(payload)))
    )
    res = _call("regress", formula="y ~ x", max_output_bytes=3000)
    sc = res["structuredContent"]
    assert len(res["content"][0]["text"]) <= 3000
    assert json.loads(res["content"][0]["text"]) == sc
    assert sc["risk_details_complete"] is False
    assert sc["risk_summary"]["violations"]["by_severity"] == {
        "error": 25,
        "warning": 75,
    }
    assert sc["risk_summary"]["degradations"]["total"] == 100
    assert sc["output_budget"]["status"] == "truncated"


def test_tools_call_reports_more_than_twenty_distinct_warnings(monkeypatch):
    def _tool():
        for i in range(57):
            warnings.warn(f"weak instrument in spec {i}", RuntimeWarning)
        return {"estimate": 1.0}

    monkeypatch.setattr(mcp_server, "execute_tool", _fake(_tool))
    sc = _call("regress", formula="y ~ x")["structuredContent"]
    assert len(sc["runtime_warnings"]) == 20
    assert sc["risk_details_complete"] is False
    assert sc["risk_summary"]["runtime_warnings"]["total"] == 57
    assert sc["risk_summary"]["runtime_warnings"]["omitted"] == 37


def test_tools_call_reports_unavoidable_overflow(monkeypatch):
    monkeypatch.setattr(
        mcp_server, "execute_tool", _fake(lambda: {"estimate": list(range(1000))})
    )
    sc = _call("regress", formula="y ~ x", max_output_bytes=100)["structuredContent"]
    assert sc["output_budget"]["status"] == "unavoidable_overflow"
    assert len(sc["estimate"]) == 1000


def test_output_schema_documents_the_ledgers():
    props = mcp_server._RESULT_OUTPUT_SCHEMA["properties"]
    for key in ("risk_summary", "risk_details_complete", "output_budget", "truncated"):
        assert key in props
