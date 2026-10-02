"""Red-line scenarios: does the output let an agent do the right thing?

The 900-trial model benchmark in ``tests/agent_bench`` needs a budget and
pre-registration, and its mock run tests the harness, not a model. This
file is the part that needs neither: for each way an agent can produce a
confidently wrong report, it drives the real MCP surface into that
situation and checks that the response carries an unambiguous,
machine-readable signal of it.

``obligations(payload)`` below is a reference policy. It reads only the
MCP result -- no source, no fitted object -- and returns what an honest
report of that result must do. A scenario passes when the policy, given
the real payload, returns the obligation the scenario is about. If a
future change drops or renames the signal, the policy goes blind and the
scenario fails: that is the regression these tests exist to catch. They
say nothing about whether a given model follows the signal.

=======================  ===============================================
red line                 signal the policy reads
=======================  ===============================================
weak IV, plain t-test    ``violations[].test == "weak_instrument"`` and
                         a failed card check
few clusters             ``violations[].test == "few_clusters"``
"diagnostics pass"       ``result_card.assumptions.checks_summary``
 when none ran
"aligned with Stata/R"   ``result_card.evidence.outputs`` for this call
 for an unvalidated SE
forest reported as       same field: no ``reference`` output
 same-byte parity
missing design input     ``remediation.missing_arguments``
lost handle              ``error_kind == "missing_result_handle"``
risk list cut short      ``risk_details_complete is False``
Stata option dropped     ``untranslated_options`` / refusal to execute
=======================  ===============================================
"""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, Set

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import mcp_server
from statspai.agent.mcp_server import handle_request
from statspai.exceptions import MethodIncompatibility


def obligations(payload: Dict[str, Any], *, is_error: bool = False) -> Set[str]:
    """What an honest report of this MCP result has to do. Reads nothing else."""
    out: Set[str] = set()
    if is_error:
        kind = payload.get("error_kind")
        remediation = payload.get("remediation") or {}
        if remediation.get("missing_arguments"):
            out.add("stop:ask_for_missing_inputs")
        if kind == "missing_result_handle":
            out.add("stop:refit_before_follow_up")
        if kind == "server_busy":
            out.add("retry_later")
        return out or {"stop:report_the_error"}
    tests = {v.get("test") for v in payload.get("violations") or []}
    if "weak_instrument" in tests:
        out.add("report:weak_iv_robust_inference")
    if "few_clusters" in tests:
        out.add("report:few_cluster_inference")
    card = payload.get("result_card") or {}
    summary = (card.get("assumptions") or {}).get("checks_summary")
    if summary is None:
        out.add("forbid:claim_diagnostics_passed")
    else:
        if summary.get("not_run"):
            out.add("forbid:claim_diagnostics_passed")
        if summary.get("failed"):
            out.add("report:failed_diagnostic")
    evidence = card.get("evidence") or {}
    outputs = evidence.get("outputs")
    if evidence.get("level") != "configuration" or not outputs:
        out.add("forbid:claim_configuration_parity")
    else:
        if outputs.get("se") != "reference":
            out.add("forbid:claim_se_parity")
        if outputs.get("estimate") != "reference":
            out.add("forbid:claim_estimate_parity")
    if payload.get("risk_details_complete") is False:
        out.add("report:risk_list_incomplete")
    if payload.get("degradations"):
        out.add("report:degraded_steps")
    if payload.get("untranslated_options"):
        out.add("forbid:run_translation_as_is")
    return out


def _call(name: str, **arguments: Any) -> Dict[str, Any]:
    raw = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return json.loads(handle_request(raw))["result"]


def _obligations(name: str, **arguments: Any) -> Set[str]:
    res = _call(name, **arguments)
    return obligations(res["structuredContent"], is_error=res["isError"])


@pytest.fixture(scope="module", autouse=True)
def _full_profile():
    previous = mcp_server.set_tool_profile(None)
    yield
    mcp_server.set_tool_profile(previous)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("redline")
    rng = np.random.default_rng(11)
    n = 600
    z, u = rng.normal(size=n), rng.normal(size=n)
    x = rng.normal(size=n)
    strong = 0.9 * z + u + rng.normal(size=n)
    weak = 0.03 * z + u + rng.normal(size=n)
    treat = (rng.uniform(size=n) < 0.5).astype(int)
    frame = pd.DataFrame(
        {
            "y_strong": 1 + 2 * strong + x + u + rng.normal(size=n),
            "y_weak": 1 + 2 * weak + x + u + rng.normal(size=n),
            "d_strong": strong,
            "d_weak": weak,
            "z": z,
            "x": x,
            "treat": treat,
            "y_t": 1 + treat * (1 + x) + x + rng.normal(size=n),
            "cl8": rng.integers(0, 8, size=n),
            "cl60": rng.integers(0, 60, size=n),
        }
    )
    cross = root / "cross.csv"
    frame.to_csv(cross, index=False)
    panel = root / "panel.csv"
    sp.datasets.mpdta().to_csv(panel, index=False)
    return {"cross": str(cross), "panel": str(panel)}


# ---------------------------------------------------------------------------
# Inference red lines
# ---------------------------------------------------------------------------


def test_weak_instrument_must_be_reported_with_robust_inference(files):
    weak = _call("ivreg", formula="y_weak ~ x + (d_weak ~ z)", data_path=files["cross"])
    sc = weak["structuredContent"]
    found = obligations(sc)
    assert "report:weak_iv_robust_inference" in found
    assert "report:failed_diagnostic" in found
    violation = next(v for v in sc["violations"] if v["test"] == "weak_instrument")
    assert violation["value"] < 10 < 126  # first-stage F, far below the screen
    assert "sp.anderson_rubin_ci" in violation["alternatives"]
    # ...and the response names the tools that do it.
    assert {"anderson_rubin_test", "effective_f_test"} <= {
        c["tool"] for c in sc["next_calls"]
    }
    strong = _obligations(
        "ivreg", formula="y_strong ~ x + (d_strong ~ z)", data_path=files["cross"]
    )
    assert "report:weak_iv_robust_inference" not in strong
    assert "report:failed_diagnostic" not in strong


def test_few_clusters_must_be_reported(files):
    few = _call(
        "regress", formula="y_strong ~ x", cluster="cl8", data_path=files["cross"]
    )
    assert "report:few_cluster_inference" in obligations(few["structuredContent"])
    violation = next(
        v for v in few["structuredContent"]["violations"] if v["test"] == "few_clusters"
    )
    assert (violation["value"], violation["threshold"]) == (8, 30)
    many = _obligations(
        "regress", formula="y_strong ~ x", cluster="cl60", data_path=files["cross"]
    )
    assert "report:few_cluster_inference" not in many


# ---------------------------------------------------------------------------
# "It passed the diagnostics" / "it matches Stata"
# ---------------------------------------------------------------------------


def test_diagnostics_that_did_not_run_cannot_be_reported_as_passed(files):
    res = _call(
        "callaway_santanna",
        y="lemp",
        g="first_treat",
        t="year",
        i="countyreal",
        data_path=files["panel"],
    )
    sc = res["structuredContent"]
    errors = [v for v in sc["violations"] if v.get("severity") == "error"]
    assert errors == []  # the tempting reading: nothing wrong
    assert "forbid:claim_diagnostics_passed" in obligations(sc)
    summary = sc["result_card"]["assumptions"]["checks_summary"]
    assert summary["not_run"] >= 1 and summary["failed"] == 0


def test_parity_claim_follows_the_options_of_this_call(files):
    kw = dict(formula="y_strong ~ x + (d_strong ~ z)", data_path=files["cross"])
    default = _obligations("ivreg", **kw)
    hc3 = _obligations("ivreg", robust="hc3", **kw)
    assert "forbid:claim_se_parity" not in default
    assert "forbid:claim_se_parity" in hc3
    # The point estimate does not depend on the covariance option.
    assert "forbid:claim_estimate_parity" not in hc3


def test_forest_is_never_reported_as_same_byte_parity(files):
    res = _call(
        "call_function",
        function="causal_forest",
        arguments={"formula": "y_t ~ treat | x + z", "n_estimators": 60},
        data_path=files["cross"],
    )
    assert res["isError"] is False, res["structuredContent"]
    sc = res["structuredContent"]
    found = obligations(sc)
    assert "forbid:claim_estimate_parity" in found or (
        "forbid:claim_configuration_parity" in found
    )
    outputs = (sc["result_card"]["evidence"].get("outputs") or {}).values()
    assert "reference" not in outputs
    assert len(sc["result_card"]["evidence"]) >= 2


# ---------------------------------------------------------------------------
# Stop conditions
# ---------------------------------------------------------------------------


def test_missing_treatment_is_a_stop_with_the_missing_field_named(files):
    res = _call("did", y="lemp", time="year", data_path=files["panel"])
    assert res["isError"] is True
    assert obligations(res["structuredContent"], is_error=True) == {
        "stop:ask_for_missing_inputs"
    }
    assert res["structuredContent"]["remediation"]["missing_arguments"] == ["treat"]
    assert len(res["structuredContent"]["remediation"]["required"]) == 4


def test_follow_up_on_a_lost_handle_is_a_stop_not_an_empty_success():
    res = _call("audit_result", result_id="r_0badf00d")
    assert res["isError"] is True
    assert obligations(res["structuredContent"], is_error=True) == {
        "stop:refit_before_follow_up"
    }
    assert "as_handle" in res["structuredContent"]["hint"]
    assert len(res["structuredContent"]["available_result_ids"]) >= 0


def test_risk_list_cut_by_the_budget_must_be_reported_as_incomplete(monkeypatch):
    payload = {
        "estimate": 0.4,
        "violations": [
            {"test": "overlap", "severity": "warning", "message": "m" * 80}
            for _ in range(120)
        ],
    }
    monkeypatch.setattr(
        mcp_server,
        "execute_tool",
        lambda name, arguments, **kw: json.loads(json.dumps(payload)),
    )
    res = _call("regress", formula="y ~ x", max_output_bytes=1500)
    sc = res["structuredContent"]
    assert "report:risk_list_incomplete" in obligations(sc)
    assert sc["risk_summary"]["violations"]["total"] == 120
    assert len(sc["violations"]) < 120


# ---------------------------------------------------------------------------
# Stata migration
# ---------------------------------------------------------------------------


def test_untranslated_stata_option_is_flagged_and_refused_at_execution(files):
    cmd = "reghdfe y x, absorb(id) weirdopt(3)"
    res = _call("from_stata", command=cmd)
    sc = res["structuredContent"]
    assert sc["untranslated_options"] == ["weirdopt"]
    assert "forbid:run_translation_as_is" in obligations(sc)
    frame = pd.read_csv(files["cross"])
    with pytest.raises(MethodIncompatibility, match="weirdopt"):
        sp.stata("regress y_strong x, weirdopt(3)", data=frame)
    # The same command without the unknown option runs, and gives the OLS slope.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = sp.stata("regress y_strong x", data=frame)
    slope = np.polyfit(frame["x"], frame["y_strong"], 1)[0]
    assert ok.params["x"] == pytest.approx(slope, rel=1e-9)


def test_clean_translation_carries_no_obligation():
    sc = _call("from_stata", command="regress y x, vce(robust)")["structuredContent"]
    assert sc["untranslated_options"] == []
    assert "forbid:run_translation_as_is" not in obligations(sc)
    assert len(sc["python_code"]) > 10
