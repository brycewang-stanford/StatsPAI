"""The documented agent result contract holds for every result class.

AGENTS.md and ``schemas/result.schema.json`` promise ``r.to_dict(detail=...)``,
``r.violations()``, ``r.next_steps()``, ``r.result_card()`` and ``r.cite()``
on results. The core trees implement them with method-specific logic; every
other result class gets them from ``ResultProtocolMixin`` (or
``@attach_result_protocol`` for NamedTuples). These tests walk every
importable ``*Result`` class and exercise real fits.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._result_contract import AGENT_KEYS, MINIMAL_KEYS
from statspai._result_serialize import ResultProtocolMixin

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import result_protocol_audit as _audit  # noqa: E402

CONTRACT_METHODS = ("to_dict", "violations", "next_steps", "result_card", "cite")


def _gap_names() -> set:
    """``module:Class`` keys of the documented gaps, as class names."""
    return {key.rsplit(":", 1)[1] for key in _audit.AGENT_CONTRACT_GAPS}


def _gap_modules() -> Dict[str, str]:
    out = {}
    for key in _audit.AGENT_CONTRACT_GAPS:
        path, name = key.rsplit(":", 1)
        mod = path[len("src/") : -len(".py")].replace("/", ".")
        out[f"{mod}.{name}"] = name
    return out


def _all_result_classes() -> Dict[str, type]:
    classes: Dict[str, type] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for info in pkgutil.walk_packages(sp.__path__, "statspai."):
            try:
                mod = importlib.import_module(info.name)
            except Exception:  # optional dependency missing: not our concern
                continue
            for name, obj in vars(mod).items():
                if (
                    inspect.isclass(obj)
                    and obj.__module__ == mod.__name__
                    and (name.endswith("Result") or name.endswith("Results"))
                ):
                    classes[f"{obj.__module__}.{name}"] = obj
    # Public exports too (lazy attributes resolve on getattr).
    for name in dir(sp):
        if name.endswith("Result") or name.endswith("Results"):
            obj = getattr(sp, name, None)
            if inspect.isclass(obj):
                classes.setdefault(f"{obj.__module__}.{obj.__name__}", obj)
    return classes


@pytest.fixture(scope="module")
def result_classes() -> Dict[str, type]:
    return _all_result_classes()


def _accepts_detail(fn: Any) -> bool:
    return "detail" in inspect.signature(fn).parameters


def test_walk_finds_the_result_classes(result_classes):
    # Guard against the walk silently finding nothing.
    assert len(result_classes) >= 250
    assert "statspai.core.results.CausalResult" in result_classes


def test_every_result_class_has_the_agent_contract(result_classes):
    gaps = _gap_modules()
    missing: List[str] = []
    for qual, cls in sorted(result_classes.items()):
        if qual in gaps:
            continue
        absent = [m for m in CONTRACT_METHODS if not callable(getattr(cls, m, None))]
        if absent:
            missing.append(f"{qual}: missing {absent}")
            continue
        if not _accepts_detail(cls.to_dict):
            missing.append(f"{qual}: to_dict() does not accept detail=")
    assert not missing, "\n".join(missing)


def test_documented_gaps_are_still_gaps(result_classes):
    """A fixed class must leave the gap list (ratchet)."""
    for qual in _gap_modules():
        cls = result_classes.get(qual)
        if cls is None:
            continue  # optional dependency not importable here
        has_all = all(callable(getattr(cls, m, None)) for m in CONTRACT_METHODS)
        assert not (
            has_all and _accepts_detail(cls.to_dict)
        ), f"{qual} now has the contract; remove it from AGENT_CONTRACT_GAPS"


def test_static_audit_agrees_with_runtime_walk(result_classes):
    report = _audit.collect()
    static_missing = {
        row["name"]
        for row in report["per_class"]
        if "agent_contract" in row.get("missing", {})
    }
    assert static_missing == _gap_names()


# ---------------------------------------------------------------------- #
#  Behaviour on real results
# ---------------------------------------------------------------------- #


def _check_payloads(res: Any) -> Dict[str, Any]:
    minimal = res.to_dict(detail="minimal")
    assert set(MINIMAL_KEYS) <= set(minimal)
    json.dumps(minimal)
    agent = res.to_dict(detail="agent")
    assert set(MINIMAL_KEYS) <= set(agent)
    assert set(AGENT_KEYS) <= set(agent)
    text = json.dumps(agent)  # strict JSON: no NaN
    assert "NaN" not in text
    assert isinstance(agent["violations"], list)
    assert isinstance(agent["next_steps"], list)
    assert all(isinstance(s, dict) and "action" in s for s in agent["next_steps"])
    assert isinstance(res.violations(), list)
    steps = res.next_steps()
    assert isinstance(steps, list) and all(isinstance(s, dict) for s in steps)
    card = res.result_card()
    assert isinstance(card, dict)
    json.dumps(card)
    assert res.cite() is not None
    with pytest.raises(ValueError, match="detail must be"):
        res.to_dict(detail="everything")
    return agent


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_oaxaca_decomposition_result():
    res = sp.oaxaca(
        data=sp.cps_wage(),
        y="log_wage",
        group="female",
        x=["education", "experience", "tenure"],
    )
    legacy = res.to_dict()
    assert res.to_dict(detail="standard") == legacy
    agent = _check_payloads(res)
    # Standard fields stay top-level at the agent level, as for CausalResult.
    assert set(legacy) <= set(agent)
    assert agent["citation_key"] in res.bib_keys
    # The producing function is resolved from the registry (unique return
    # class), so the card carries a real evidence tier and next steps exist.
    card = res.result_card()
    assert card["function"] == "oaxaca"
    assert card["evidence"]["function_tier"] is not None
    assert card["provenance"]["result_class"].endswith("OaxacaResult")
    assert agent["next_steps"]


def test_mixin_dataclass_result_identify():
    res = sp.identify(sp.dag("Z -> X; Z -> Y; X -> Y"), treatment="X", outcome="Y")
    assert isinstance(res, ResultProtocolMixin)
    agent = _check_payloads(res)
    assert agent["identifiable"] is True
    assert agent["estimand"] == res.estimand


def test_mean_comparison_result():
    df = sp.cps_wage()
    res = sp.mean_comparison(df, ["education", "experience"], group="female")
    _check_payloads(res)


def test_namedtuple_result_keeps_tuple_semantics():
    df = sp.dgp_did(n_units=200, n_periods=8, staggered=True, seed=42)
    df["first_treat"] = df["first_treat"].fillna(0)
    cs = sp.callaway_santanna(df, y="y", g="first_treat", t="time", i="unit")
    eq = sp.pretrends_equivalence(cs)
    assert isinstance(eq, tuple)
    f_stat, f_pvalue, *_ = eq  # unpacking unchanged
    assert f_stat == eq.f_stat and f_pvalue == eq.f_pvalue
    assert eq._fields[0] == "f_stat"
    std = eq.to_dict()
    assert list(std)[: len(eq._fields)] == list(eq._fields)
    agent = _check_payloads(eq)
    assert agent["f_pvalue"] == pytest.approx(eq.f_pvalue)


def test_internal_namedtuple_flci_result():
    from statspai.did._flci import FLCIResult

    res = FLCIResult(1.0, 0.5, 0.5, 1.5, np.array([0.2, 0.8]), 0.1, 0.05)
    agent = _check_payloads(res)
    assert agent["ci"] == [0.5, 1.5]
    assert agent["estimate"] == 1.0
    # No SE recorded is not a violation; a NaN estimate is.
    assert res.violations() == []
    tests = [v["test"] for v in res._replace(estimate=float("nan")).violations()]
    assert tests == ["estimate_finite"]


def _did_fit(**kw: Any) -> Any:
    df = sp.dgp_did(n_units=100, n_periods=2, seed=0)
    df["treat"] = df["first_treat"].notna().astype(int)
    df["post"] = (df["time"] >= 1).astype(int)
    return sp.did(df, y="y", treat="treat", time="post", **kw)


def test_causal_result_next_steps_is_silent_by_default(capsys):
    res = _did_fit()
    steps = res.next_steps()
    assert steps and isinstance(steps[0], dict)
    assert capsys.readouterr().out == ""
    assert res.next_steps(print_result=True) == steps
    assert "Suggested Next Steps" in capsys.readouterr().out
    _check_payloads(res)


def test_econometric_result_next_steps_is_silent_by_default(capsys):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=100)})
    df["y"] = 1 + df.x + rng.normal(size=100)
    res = sp.regress("y ~ x", data=df)
    res.next_steps()
    assert capsys.readouterr().out == ""


def _aipw_fit(**kw: Any) -> Any:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["d"] = (rng.random(n) < 1 / (1 + np.exp(-df.x))).astype(int)
    df["y"] = df.d + df.x + rng.normal(size=n)
    return sp.aipw(df, y="y", treat="d", covariates=["x"], **kw)


def test_result_card_records_unset_seed_as_not_reproducible():
    prov = _aipw_fit(seed=None).result_card()["provenance"]
    assert prov["seed"] is None
    assert prov["reproducible"] is False
    assert prov["seed_source"] == "call"
    assert "seed_note" in prov


def test_result_card_records_explicit_and_default_seed():
    prov = _aipw_fit(seed=3).result_card()["provenance"]
    assert (prov["seed"], prov["reproducible"]) == (3, True)
    # The default seed (42) is recorded by the estimator, not invented.
    prov = _aipw_fit().result_card()["provenance"]
    assert (prov["seed"], prov["reproducible"]) == (42, True)


def test_result_card_does_not_claim_a_seed_it_did_not_see():
    # did_2x2 accepts seed= but records a curated call subset without it:
    # the card must say "unknown", not "unset".
    for kw in ({}, {"seed": 7}):
        prov = _did_fit(**kw).result_card()["provenance"]
        assert prov["seed"] is None
        assert prov["reproducible"] is None
        assert prov["seed_source"] == "not_recorded"


# ---------------------------------------------------------------------- #
#  Legacy to_dict overrides get detail= via the mixin
# ---------------------------------------------------------------------- #


@dataclass
class _LegacyResult(ResultProtocolMixin):
    estimate: float
    se: float
    diagnostics: Dict[str, Any]

    def to_dict(self, round_to: int = 3) -> Dict[str, Any]:
        return {"estimate": round(self.estimate, round_to), "custom": True}


def test_legacy_override_is_wrapped_not_changed():
    res = _LegacyResult(1.23456, float("nan"), {"n_clusters": 5})
    assert res.to_dict() == {"estimate": 1.235, "custom": True}
    assert res.to_dict(2) == {"estimate": 1.23, "custom": True}
    sig = inspect.signature(_LegacyResult.to_dict)
    assert "detail" in sig.parameters and "round_to" in sig.parameters
    agent = res.to_dict(detail="agent")
    json.dumps(agent)
    assert agent["custom"] is True
    assert agent["estimate"] == 1.235  # the class's own field wins
    tests = {v["test"] for v in agent["violations"]}
    assert {"few_clusters", "se_positive"} <= tests
    assert agent["warnings"]
    minimal = res.to_dict(detail="minimal")
    assert set(minimal) == set(MINIMAL_KEYS)


class _BrokenViolations(ResultProtocolMixin):
    def __init__(self) -> None:
        self.estimate = 1.0

    def violations(self) -> List[Dict[str, Any]]:
        raise RuntimeError("detector bug")


def test_agent_payload_records_detector_failure_instead_of_raising():
    with pytest.warns(Warning):
        agent = _BrokenViolations().to_dict(detail="agent")
    assert agent["violations"] == []
    assert any(d["section"] == "violations" for d in agent["degradations"])
