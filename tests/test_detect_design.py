"""Tests for ``sp.detect_design(data, **hints)`` — heuristic study-
design identifier from a DataFrame's shape.

The function is intentionally heuristic (it reports confidence + ranks
alternatives), so these tests pin:

* Correct routing on canonical shapes (balanced panel, imbalanced panel,
  cross-section, RD with hint).
* False-positive guard: pure-noise data must NOT auto-classify as RD.
* Symmetric-pair dedup: ``(unit, time)`` and ``(time, unit)``
  candidates collapse to one entry per column-pair.
* Hint plumbing: ``unit=`` / ``time=`` / ``running_var=`` / ``cutoff=``
  pin the role and bump confidence.
* Return-shape stability: every payload is JSON-safe and bounded.
* Edge cases: empty DataFrame, single-column, non-DataFrame input.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def balanced_panel():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "firm_id": np.repeat(range(50), 10),
        "year": np.tile(range(2010, 2020), 50),
        "sales": rng.normal(size=500),
        "cost": rng.normal(size=500),
    })


@pytest.fixture
def imbalanced_panel():
    rng = np.random.default_rng(2)
    rows = []
    for i in range(30):
        for t in range(8):
            if rng.random() > 0.2:  # 80 % cell coverage
                rows.append({"unit": i, "t": t, "y": rng.normal()})
    return pd.DataFrame(rows)


@pytest.fixture
def cross_section():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(size=200),
        "y": rng.normal(size=200),
        "z": rng.normal(size=200),
    })


@pytest.fixture
def rd_dataset():
    rng = np.random.default_rng(1)
    score = np.concatenate([
        rng.uniform(40, 65, 200),
        rng.uniform(65, 90, 200),
    ])
    return pd.DataFrame({
        "score": score,
        "admit": (score >= 65).astype(int),
        "outcome": rng.normal(size=400),
    })


# ---------------------------------------------------------------------------
#  Top-level export + return shape
# ---------------------------------------------------------------------------


class TestExport:

    def test_sp_detect_design_callable(self):
        assert callable(sp.detect_design)

    def test_in_all(self):
        assert "detect_design" in sp.__all__


class TestReturnShape:

    def test_top_level_keys(self, balanced_panel):
        out = sp.detect_design(balanced_panel)
        for k in ("design", "confidence", "identified", "candidates",
                  "n_obs", "columns"):
            assert k in out, f"missing top-level key {k!r}"

    def test_design_string_in_known_set(self, balanced_panel):
        out = sp.detect_design(balanced_panel)
        assert out["design"] in ("panel", "rd", "cross_section")

    def test_confidence_in_unit_interval(self, balanced_panel):
        out = sp.detect_design(balanced_panel)
        assert 0.0 <= out["confidence"] <= 1.0

    def test_payload_strict_json_safe(self, balanced_panel):
        json.dumps(sp.detect_design(balanced_panel))

    def test_payload_bounded(self, balanced_panel):
        s = json.dumps(sp.detect_design(balanced_panel))
        assert len(s) < 4000


# ---------------------------------------------------------------------------
#  Canonical-shape routing
# ---------------------------------------------------------------------------


class TestCanonicalShapes:

    def test_balanced_panel_is_panel(self, balanced_panel):
        out = sp.detect_design(balanced_panel)
        assert out["design"] == "panel"
        assert out["confidence"] >= 0.9
        assert out["identified"]["unit"] == "firm_id"
        assert out["identified"]["time"] == "year"

    def test_imbalanced_panel_still_classified_as_panel(self,
                                                        imbalanced_panel):
        out = sp.detect_design(imbalanced_panel)
        assert out["design"] == "panel"
        # Looser confidence floor since 80 % fill rate.
        assert out["confidence"] >= 0.6

    def test_cross_section_is_cross_section(self, cross_section):
        out = sp.detect_design(cross_section)
        assert out["design"] == "cross_section"

    def test_rd_with_hint_classified_as_rd(self, rd_dataset):
        out = sp.detect_design(rd_dataset, running_var="score",
                                cutoff=65.0)
        assert out["design"] == "rd"
        assert out["identified"]["running_var"] == "score"
        assert out["identified"]["cutoff"] == 65.0

    def test_rd_without_hint_does_not_beat_cross_section(self,
                                                         cross_section):
        # Pure-noise data must NOT be auto-RD'd just because random
        # numbers split roughly evenly around their median.
        out = sp.detect_design(cross_section)
        assert out["design"] != "rd"


# ---------------------------------------------------------------------------
#  Hints
# ---------------------------------------------------------------------------


class TestHints:

    def test_unit_hint_pins_column(self, balanced_panel):
        # If we hint a NON-id column as the unit, it should still try
        # to use it.
        out = sp.detect_design(balanced_panel, unit="firm_id",
                                time="year")
        assert out["identified"]["unit"] == "firm_id"
        assert out["identified"]["time"] == "year"
        # Confidence boost from the hint vs. unhinted call.
        out_unhinted = sp.detect_design(balanced_panel)
        assert (out["confidence"] >= out_unhinted["confidence"]
                or out["confidence"] >= 0.95)

    def test_running_var_hint_promotes_rd(self, rd_dataset):
        out = sp.detect_design(rd_dataset, running_var="score",
                                cutoff=65.0)
        assert out["design"] == "rd"
        assert out["confidence"] >= 0.5


# ---------------------------------------------------------------------------
#  Symmetric-pair dedup
# ---------------------------------------------------------------------------


class TestSymmetricDedup:
    """``(firm_id, year)`` and ``(year, firm_id)`` shape-score
    identically when both columns pass the id and time filter — the
    output must list at most ONE panel candidate per column-pair."""

    def test_no_duplicate_panel_pair_in_candidates(self, balanced_panel):
        out = sp.detect_design(balanced_panel)
        panel_pairs = {
            frozenset((c["unit"], c["time"]))
            for c in out["candidates"]
            if c["design"] == "panel"
        }
        panel_count = sum(
            1 for c in out["candidates"] if c["design"] == "panel"
        )
        assert panel_count == len(panel_pairs), (
            f"{panel_count} panel candidates but only "
            f"{len(panel_pairs)} unique pairs — dedup is broken")


# ---------------------------------------------------------------------------
#  Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_empty_dataframe(self):
        out = sp.detect_design(pd.DataFrame())
        assert out["design"] == "cross_section"
        assert out["confidence"] == 0.0
        assert out["n_obs"] == 0

    def test_single_row(self):
        out = sp.detect_design(pd.DataFrame({"x": [1.0]}))
        assert out["design"] == "cross_section"

    def test_non_dataframe_raises(self):
        with pytest.raises(TypeError):
            sp.detect_design([1, 2, 3])  # list → reject

    def test_all_constant_columns_not_panel(self):
        df = pd.DataFrame({
            "x": [1] * 100,  # constant
            "y": [2] * 100,  # constant
        })
        out = sp.detect_design(df)
        assert out["design"] == "cross_section"

    def test_nan_in_unit_column_does_not_classify_as_cross_section(self):
        # A real panel with NaN-polluted unit IDs should still be
        # detected as a panel — NaN is not a 51st firm. Regression
        # guard: previously _is_id_like used dropna=False and
        # ``expected_balanced`` got inflated, dropping fill_ratio
        # below the threshold.
        rng = np.random.default_rng(7)
        rows = []
        for i in range(40):
            for t in range(8):
                rows.append({"unit_id": i, "year": t,
                             "y": rng.normal()})
        df = pd.DataFrame(rows)
        # Inject NaNs into the unit column
        nan_idx = df.sample(frac=0.10, random_state=7).index
        df.loc[nan_idx, "unit_id"] = np.nan
        out = sp.detect_design(df)
        assert out["design"] == "panel"
        assert out["identified"]["unit"] == "unit_id"


class TestWideDataFrame:
    """Bound the candidate enumeration: a 30-column DataFrame should
    not trigger 900 pair score evaluations."""

    def test_many_id_like_columns_still_completes(self):
        # 30 columns each with 25 distinct values — every pair would
        # pass _is_id_like. Without the candidate cap this triggers
        # ~900 (column-pair) × n_rows duplicated() passes.
        rng = np.random.default_rng(0)
        n = 200
        data = {f"c{i}": rng.integers(0, 25, size=n)
                for i in range(30)}
        df = pd.DataFrame(data)
        # Should complete in well under a second even on slow CI.
        import time
        t0 = time.time()
        out = sp.detect_design(df)
        elapsed = time.time() - t0
        assert elapsed < 5.0, (
            f"detect_design took {elapsed:.2f}s on a 30-col DataFrame; "
            "candidate cap may be missing")
        # Output is still well-formed.
        assert out["design"] in ("panel", "cross_section", "rd")
        assert 0.0 <= out["confidence"] <= 1.0


# ---------------------------------------------------------------------------
#  Candidates list contract
# ---------------------------------------------------------------------------


class TestCandidatesList:

    def test_candidates_sorted_by_confidence_desc(self, balanced_panel):
        out = sp.detect_design(balanced_panel)
        confidences = [c["confidence"] for c in out["candidates"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_winner_matches_top_candidate(self, balanced_panel):
        out = sp.detect_design(balanced_panel)
        assert out["design"] == out["candidates"][0]["design"]
        assert out["confidence"] == out["candidates"][0]["confidence"]

    def test_cross_section_always_present_as_fallback(self,
                                                      balanced_panel):
        out = sp.detect_design(balanced_panel)
        designs = {c["design"] for c in out["candidates"]}
        assert "cross_section" in designs
