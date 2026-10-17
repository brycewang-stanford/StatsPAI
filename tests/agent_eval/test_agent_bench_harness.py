"""CausalAgentBench v2 harness: everything short of a real model.

Covers the frozen data, the golds, the sandbox, the full agent loop driven
by the oracle model, the grader's hallucination checks, and the
pre-registration guard that keeps API calls off until the OSF deposit.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "agent_bench"
sys.path.insert(0, str(ROOT / "runners"))

import build_golds  # noqa: E402
import grader  # noqa: E402
import prereg  # noqa: E402
from agent_loop import run_trial  # noqa: E402
from cells import CELLS  # noqa: E402
from models import OracleModel, ScriptedModel, ToolCall, Turn  # noqa: E402
from sandbox import PythonSession, RSession  # noqa: E402

TASKS = {
    t["id"]: t for t in json.loads((ROOT / "prompts/prompts.json").read_text())["tasks"]
}
GOLDS = {
    g["id"]: g for g in json.loads((ROOT / "golds/golds.json").read_text())["golds"]
}
HAVE_R = shutil.which("Rscript") is not None


def _r_has(*packages: str) -> bool:
    """Rscript is on PATH and can load ``packages`` (CI runners ship bare R)."""
    if not HAVE_R:
        return False
    import subprocess

    expr = "; ".join(f"library({p})" for p in packages)
    try:
        done = subprocess.run(
            ["Rscript", "--vanilla", "-e", expr], capture_output=True, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def test_fifty_tasks_each_with_a_gold_for_its_headline():
    assert len(TASKS) == 50 and set(TASKS) == set(GOLDS)
    for tid, task in TASKS.items():
        gold = GOLDS[tid]
        assert gold["headline"] == task["headline"]
        for v in gold["variants"]:
            assert task["headline"] in v["values"]
            assert set(v["values"]) <= set(task["report"])


def test_committed_golds_are_what_build_golds_produces():
    ref = json.loads((ROOT / "golds/reference_values.json").read_text())
    rebuilt = build_golds.build(ref["values"])
    for tid, g in GOLDS.items():
        assert [v["values"] for v in rebuilt[tid]["variants"]] == [
            v["values"] for v in g["variants"]
        ]


def test_committed_data_match_the_manifest():
    import hashlib

    manifest = json.loads((ROOT / "data/MANIFEST.json").read_text())
    for name, entry in manifest.items():
        if not entry["committed"]:
            continue
        raw = (ROOT / "data" / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], name


def test_pythonic_session_blocks_statspai_and_keeps_state(tmp_path):
    with PythonSession(tmp_path, allow_statspai=False, timeout=60) as s:
        assert "not available in this toolset" in s.run("import statspai").output
        s.run("x = 41")
        assert s.run("print(x + 1)").output == "42"


@pytest.mark.skipif(not HAVE_R, reason="Rscript not installed")
def test_r_session_does_not_share_names_with_its_driver(tmp_path):
    with RSession(tmp_path, timeout=60) as r:
        r.run("e <- 5; i <- 2; res <- 'mine'")
        assert r.run("c(e, i, res)").output == '[1] "5"    "2"    "mine"'


def test_session_restarts_after_a_timeout(tmp_path):
    with PythonSession(tmp_path, allow_statspai=False, timeout=2) as s:
        first = s.run("import time; time.sleep(10)")
        assert first.timed_out
        again = s.run("print('ok')")
        assert again.output == "ok" and again.restarted


@pytest.mark.parametrize("tid", ["L1-01", "L1-16", "L2-12"])
def test_oracle_trial_scores_full_marks_in_the_statspai_cell(tid):
    from oracle_statspai import SOLUTIONS, H

    est, code = SOLUTIONS[tid]
    trial = run_trial(TASKS[tid], CELLS["C1"], OracleModel(H + code, est))
    assert trial["stop"] == "submitted"
    score = grader.score_trial(trial, TASKS[tid], GOLDS[tid])
    assert score["task_success"] == score["method_correct"] == 1
    assert score["values_correct"] == 1.0
    assert score["hallucination"] == 0


def test_oracle_trial_in_the_pythonic_cell():
    from oracle_other import PYTHONIC

    est, code = PYTHONIC["L1-01"]
    trial = run_trial(TASKS["L1-01"], CELLS["C3"], OracleModel(code, est))
    score = grader.score_trial(trial, TASKS["L1-01"], GOLDS["L1-01"])
    assert score["task_success"] == 1 and score["toolset_violation"] == 0


@pytest.mark.skipif(
    not _r_has("EValue", "jsonlite"), reason="R with EValue and jsonlite needed"
)
def test_oracle_trial_in_the_r_cell():
    from oracle_other import R_SOLUTIONS

    est, code = R_SOLUTIONS["L1-16"]
    trial = run_trial(TASKS["L1-16"], CELLS["C5"], OracleModel(code, est, "run_r"))
    score = grader.score_trial(trial, TASKS["L1-16"], GOLDS["L1-16"])
    assert score["task_success"] == 1


def _scripted(*turns):
    return ScriptedModel(list(turns))


def test_a_number_the_agent_never_computed_is_fabricated():
    gold = GOLDS["L1-01"]["variants"][0]["values"]
    model = _scripted(
        Turn(tool_calls=[ToolCall("1", "run_python", {"code": "print(1 + 1)"})]),
        Turn(
            tool_calls=[
                ToolCall(
                    "2",
                    "submit_answer",
                    {
                        "beta_educ": gold["beta_educ"],
                        "se_educ": gold["se_educ"],
                        "estimator": "ols",
                        "justification": "x",
                    },
                )
            ]
        ),
    )
    trial = run_trial(TASKS["L1-01"], CELLS["C3"], model)
    score = grader.score_trial(trial, TASKS["L1-01"], GOLDS["L1-01"])
    # Right number, but it never appeared in any output: fabricated.
    assert score["task_success"] == 1 and score["fabricated"] == 1
    assert score["hallucination"] == 1


def test_calling_a_function_that_does_not_exist_is_an_invented_api():
    model = _scripted(
        Turn(
            tool_calls=[
                ToolCall(
                    "1",
                    "run_python",
                    {"code": "import statspai as sp\nsp.estimate_everything()"},
                )
            ]
        ),
    )
    trial = run_trial(TASKS["L1-01"], CELLS["C1"], model)
    score = grader.score_trial(trial, TASKS["L1-01"], GOLDS["L1-01"])
    assert score["invented_api"] == 1 and score["submitted"] == 0


def test_blocked_statspai_import_is_a_toolset_violation_not_a_hallucination():
    model = _scripted(
        Turn(tool_calls=[ToolCall("1", "run_python", {"code": "import statspai"})])
    )
    trial = run_trial(TASKS["L1-01"], CELLS["C3"], model)
    score = grader.score_trial(trial, TASKS["L1-01"], GOLDS["L1-01"])
    assert score["toolset_violation"] == 1 and score["invented_api"] == 0


def test_an_unknown_tool_is_an_invented_api():
    model = _scripted(
        Turn(tool_calls=[ToolCall("1", "run_stata", {"code": "reg y x"})])
    )
    trial = run_trial(TASKS["L1-01"], CELLS["C3"], model)
    assert (
        grader.score_trial(trial, TASKS["L1-01"], GOLDS["L1-01"])["invented_api"] == 1
    )


def test_no_real_model_without_a_preregistration_lock(monkeypatch, tmp_path):
    from models import real_model

    monkeypatch.setattr(prereg, "LOCK", tmp_path / "missing.lock.json")
    with pytest.raises(RuntimeError, match="No pre-registration lock"):
        real_model("anthropic")


def test_lock_detects_a_changed_design_file(monkeypatch, tmp_path):
    root = tmp_path / "bench"
    shutil.copytree(
        ROOT, root, ignore=shutil.ignore_patterns("trade.csv", "results", "__pycache__")
    )
    monkeypatch.setattr(prereg, "ROOT", root)
    monkeypatch.setattr(prereg, "LOCK", root / "prompts/preregistration.lock.json")
    prereg.lock("10.17605/OSF.IO/TEST", {"anthropic": "m-a", "openai": "m-o"}, {})
    assert prereg.verify()["osf_doi"] == "10.17605/OSF.IO/TEST"
    golds = root / "golds/golds.json"
    golds.write_text(golds.read_text() + " ")
    with pytest.raises(RuntimeError, match="golds/golds.json"):
        prereg.verify()


def test_hypothesis_tests_run_on_scored_rows():
    rows = []
    for tid, task in TASKS.items():
        for cell, ok, tok in (("C1", 1, 1000), ("C3", 0, 2000), ("C5", 1, 3000)):
            rows.append(
                {
                    "task_id": tid,
                    "level": task["level"],
                    "cell": cell,
                    "task_success": ok,
                    "hallucination": 0,
                    "tokens": tok,
                }
            )
    h = grader.hypotheses(rows, B=200)
    assert h["H1"]["L1"]["estimate"] == 1.0 and h["H1"]["L1"]["supported"]
    assert h["H2"]["estimate"] == 1.0 and h["H2"]["supported"]
    assert h["H4"]["estimate"] == pytest.approx(0.5)
    assert h["H5"]["token_ratio"]["estimate"] == pytest.approx(3.0)
