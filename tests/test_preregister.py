"""Tests for CausalQuestion preregistration (save + load)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import statspai as sp


def _fixture_question():
    return sp.causal_question(
        treatment="minimum_wage_hike",
        outcome="employment",
        design="did",
        estimand="ATT",
        time_structure="panel",
        time="year",
        id="state",
        covariates=["industry", "skill"],
        notes="Identification via 2015 state-level wage hike",
    )


# ---------------------------------------------------------------------------
# preregister / load_preregister
# ---------------------------------------------------------------------------


def test_preregister_yaml_roundtrip(tmp_path):
    q = _fixture_question()
    path = sp.preregister(q, tmp_path / "pap.yaml", note="PAP v1.0")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "treatment:" in text
    assert "minimum_wage_hike" in text

    q2 = sp.load_preregister(path)
    assert q2.treatment == q.treatment
    assert q2.outcome == q.outcome
    assert q2.estimand == q.estimand
    assert q2.design == q.design
    assert q2.covariates == q.covariates


def test_preregister_json_roundtrip(tmp_path):
    q = _fixture_question()
    path = sp.preregister(q, tmp_path / "pap.json")
    assert path.suffix == ".json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["question"]["treatment"] == q.treatment
    assert payload["metadata"]["format_version"].startswith("statspai.preregister/")

    q2 = sp.load_preregister(path)
    assert q2.treatment == q.treatment


def test_preregister_auto_suffix(tmp_path):
    q = _fixture_question()
    # No suffix -> defaults to yaml
    path = sp.preregister(q, tmp_path / "pap")
    assert path.suffix == ".yaml"


def test_preregister_rejects_bad_fmt(tmp_path):
    q = _fixture_question()
    with pytest.raises(ValueError):
        sp.preregister(q, tmp_path / "pap.txt", fmt="toml")


def test_causal_question_save_and_load_methods(tmp_path):
    q = _fixture_question()
    path = q.save(tmp_path / "plan.yaml", note="my analysis plan")
    assert path.exists()
    q2 = sp.CausalQuestion.load(path)
    assert q2.treatment == q.treatment
    assert "my analysis plan" in (q2.notes or "")


def test_to_yaml_returns_string():
    q = _fixture_question()
    text = q.to_yaml()
    assert "treatment:" in text
    assert "minimum_wage_hike" in text


def test_preregister_from_dict(tmp_path):
    d = {
        "treatment": "d",
        "outcome": "y",
        "estimand": "ATE",
        "design": "rct",
    }
    path = sp.preregister(d, tmp_path / "from_dict.yaml")
    q = sp.load_preregister(path)
    assert q.treatment == "d"
    assert q.estimand == "ATE"


def test_preregister_rejects_bad_input(tmp_path):
    with pytest.raises(TypeError):
        sp.preregister("not a question", tmp_path / "x.yaml")


def test_load_preregister_rejects_incomplete(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("metadata:\n  format_version: wrong\n", encoding="utf-8")
    with pytest.raises(ValueError):
        sp.load_preregister(path)
