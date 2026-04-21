"""
Tests for TARGET 21-item checklist (sp.target_trial).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.target_trial import TARGET_ITEMS, target_checklist, to_paper


def _minimal_result():
    """Build a minimal TargetTrialResult stub for checklist-format testing."""
    proto = sp.target_trial.protocol(
        eligibility="age >= 50",
        treatment_strategies=["statin", "no statin"],
        assignment="observational emulation",
        time_zero="diabetes diagnosis",
        followup_end="min(death, 5y)",
        outcome="incident MI",
        causal_contrast="per-protocol",
        analysis_plan="CCW + pooled logistic",
        baseline_covariates=["age", "bmi"],
    )
    # Build a TargetTrialResult directly with the stub data so we don't
    # need the full emulate() pipeline for this test.
    from statspai.target_trial.emulate import TargetTrialResult
    return TargetTrialResult(
        protocol=proto,
        estimate=0.034,
        se=0.015,
        ci=(0.005, 0.063),
        n_eligible=1_234,
        n_excluded_immortal=56,
        weights=np.ones(1_234, dtype=float),
        method="CCW-IPCW",
    )


def test_target_items_length_and_structure():
    assert len(TARGET_ITEMS) == 21
    for num, section, description in TARGET_ITEMS:
        assert num.isdigit()
        assert section
        assert description


def test_target_checklist_markdown():
    res = _minimal_result()
    md = target_checklist(res, fmt="markdown")
    # Every item number appears in the output
    for num, _, _ in TARGET_ITEMS:
        assert f"| {num} |" in md, num
    # AUTO items are filled
    assert "AUTO" in md
    assert "TODO" in md
    # Protocol values surfaced
    assert "per-protocol" in md
    assert "incident MI" in md


def test_target_checklist_text():
    res = _minimal_result()
    txt = target_checklist(res, fmt="text")
    assert "TARGET Statement" in txt
    assert "Eligibility" in txt


def test_to_paper_target_format():
    res = _minimal_result()
    out = to_paper(res, fmt="target")
    assert out == target_checklist(res, fmt="markdown")


def test_to_paper_unknown_format_errors():
    res = _minimal_result()
    with pytest.raises(ValueError, match="fmt must be"):
        to_paper(res, fmt="bogus")


def test_target_in_registry():
    fns = set(sp.list_functions())
    assert "target_trial_checklist" in fns
