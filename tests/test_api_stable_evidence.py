"""API-stable evidence tests for hand-written non-core registry entries."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd

import statspai as sp
from statspai.target_trial import TARGET_ITEMS
from statspai.target_trial.emulate import TargetTrialResult


def test_llm_client_adapters_have_offline_contracts(monkeypatch):
    client = sp.causal_llm.echo_client(lambda role, prompt: f"{role}:{prompt}")
    assert client.chat("critic", "check") == "critic:check"
    np.testing.assert_allclose(len(client.history), 1)
    assert client.history[-1]["model"] == "echo"

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="openai-ok"))]
            )

    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_mod)
    openai = sp.causal_llm.openai_client(api_key="sk-test", model="fake-openai")
    assert openai.chat("proposer", "X -> Y") == "openai-ok"
    np.testing.assert_allclose(len(openai.history), 1)
    assert openai.history[-1]["model"] == "fake-openai"

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="thinking", thinking="private trace"),
                    SimpleNamespace(type="text", text="anthropic-ok"),
                ]
            )

    anthropic_mod = types.ModuleType("anthropic")
    anthropic_mod.Anthropic = FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_mod)
    anthropic = sp.causal_llm.anthropic_client(
        api_key="sk-ant-test",
        model="fake-anthropic",
        thinking_budget=1024,
        max_tokens=2048,
    )
    assert anthropic.chat("critic", "check graph") == "anthropic-ok"
    np.testing.assert_allclose(len(anthropic.history), 1)
    assert anthropic.history[-1]["thinking"] == "private trace"


def test_dag_example_and_recommendation_contract():
    frontdoor = sp.dag_example("frontdoor")
    assert {"X", "M", "Y"} <= set(frontdoor.nodes)

    graph = sp.dag("Z -> X; Z -> Y; X -> Y")
    recommendation = sp.dag_recommend_estimator(graph, "X", "Y")
    np.testing.assert_allclose(len(recommendation.adjustment_set), 1)
    np.testing.assert_allclose(len(recommendation.alternatives), 3)
    assert recommendation.estimator == "regress"
    assert recommendation.adjustment_set == {"Z"}
    assert "sp.regress" in recommendation.sp_call


def test_target_trial_reporting_aliases_render():
    protocol = sp.target_trial_protocol(
        eligibility="age >= 50",
        treatment_strategies=["A", "B"],
        assignment="observational emulation",
        time_zero="baseline",
        followup_end="1 year",
        outcome="Y",
        causal_contrast="ITT",
        analysis_plan="IPW",
    )
    np.testing.assert_allclose(len(protocol.treatment_strategies), 2)
    result = TargetTrialResult(
        protocol=protocol,
        estimate=1.0,
        se=0.2,
        ci=(0.6, 1.4),
        n_eligible=10,
        n_excluded_immortal=0,
        weights=np.ones(10),
        method="IPW",
    )

    checklist = sp.target_trial_checklist(result, fmt="markdown")
    np.testing.assert_allclose(
        sum(line.startswith("|") for line in checklist.splitlines()),
        len(TARGET_ITEMS) + 2,
    )
    report = sp.target_trial_report(result, fmt="text")
    assert "TARGET Statement" in checklist
    assert "Eligibility" in checklist
    assert "IPW" in report


def test_panel_compare_and_sensitivity_rr_contracts():
    ids = np.repeat(np.arange(4), 3)
    years = np.tile(np.arange(3), 4)
    x = np.linspace(0.0, 1.0, 12)
    y = 1.0 + 0.5 * x + ids * 0.1 + years * 0.05
    data = pd.DataFrame({"id": ids, "year": years, "x": x, "y": y})

    comparison = sp.panel_compare(
        data,
        "y ~ x",
        entity="id",
        time="year",
        methods=["pooled", "fe"],
    )
    assert "Pooled OLS" in comparison.columns
    assert "Panel FE (Within)" in comparison.columns

    class EventStudyResult:
        estimate = 0.5
        se = 0.1
        model_info = {
            "event_study": pd.DataFrame(
                {
                    "relative_time": [-2, -1, 1],
                    "estimate": [0.01, -0.01, 0.5],
                    "se": [0.1, 0.1, 0.1],
                }
            )
        }

    sensitivity = sp.sensitivity_rr(EventStudyResult(), Mbar=[0.0, 0.1])
    assert sensitivity.att == 0.5
    assert sensitivity.mbar_grid.tolist() == [0.0, 0.1]


def test_gformula_and_transport_top_level_aliases():
    wide = pd.DataFrame(
        {
            "id": range(6),
            "time": 0,
            "A0": [0, 1, 0, 1, 0, 1],
            "L0": [0, 0, 1, 1, 0, 1],
            "Y": [1.0, 2.0, 2.0, 3.0, 1.5, 2.5],
        }
    )
    ice = sp.gformula_ice_fn(
        wide,
        "id",
        "time",
        ["A0"],
        ["L0"],
        "Y",
        [1],
    )
    assert np.isfinite(ice.value)
    assert ice.method == "parametric-g-formula-ICE"

    source = pd.DataFrame(
        {
            "x": [0, 1, 0, 1, 0, 1],
            "a": [0, 0, 1, 1, 0, 1],
            "y": [0.0, 0.5, 1.0, 2.0, 0.1, 2.1],
        }
    )
    target = pd.DataFrame({"x": [0, 1, 1, 1]})
    transported = sp.transport_weights_fn(
        source,
        target,
        features=["x"],
        treatment="a",
        outcome="y",
    )
    w = transported.weights
    a = source["a"].to_numpy(dtype=float)
    y = source["y"].to_numpy(dtype=float)
    manual = np.sum(w * a * y) / np.sum(w * a) - np.sum(w * (1 - a) * y) / np.sum(
        w * (1 - a)
    )
    np.testing.assert_allclose(transported.effect_transported, manual)
    assert transported.ess > 0
    assert np.isfinite(transported.effect_transported)


def test_fairness_and_synth_design_frontier_helpers():
    rng = np.random.default_rng(0)
    n = 120
    protected = rng.integers(0, 2, n)
    credit = 600 + 50 * protected + rng.normal(0, 10, n)
    data = pd.DataFrame({"A": protected, "credit": credit})

    def predictor(frame):
        return frame["credit"].to_numpy() / 1000.0

    def intervention(frame, value):
        out = frame.copy()
        out["A"] = value
        out["credit"] = 600 + 50 * value + (frame["credit"] - (600 + 50 * frame["A"]))
        return out

    fairness = sp.evidence_without_injustice(
        data,
        predictor,
        protected="A",
        admissible_features=["credit"],
        scm_intervention=intervention,
        n_boot=99,
        random_state=0,
    )
    assert fairness.metric == "evidence_without_injustice"
    np.testing.assert_allclose(fairness.value, 0.0)
    np.testing.assert_allclose(fairness.ci, (0.0, 0.0))
    assert fairness.passes is True

    rows = []
    for unit in range(5):
        for time in range(4):
            rows.append(
                {
                    "unit": unit,
                    "time": time,
                    "y": unit + time * 0.1 + (0.01 if unit == 4 else 0.0),
                }
            )
    panel = pd.DataFrame(rows)
    design = sp.synth_experimental_design(
        panel,
        unit="unit",
        time="time",
        outcome="y",
        k=1,
        n_random=10,
        random_state=0,
    )
    assert len(design.selected) == 1
    assert "risk_score" in design.ranking.columns
