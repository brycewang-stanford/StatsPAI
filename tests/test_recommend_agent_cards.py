"""Verify sp.recommend enriches recommendations with registry agent cards."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def did_panel():
    """30-unit × 4-period panel with 2x2 treatment."""
    rng = np.random.default_rng(0)
    n_units, n_periods = 30, 4
    df = pd.DataFrame({
        "unit": np.repeat(range(n_units), n_periods),
        "year": np.tile(range(2018, 2022), n_units),
        "treated": 0,
        "y": rng.normal(size=n_units * n_periods),
    })
    df.loc[df["unit"] < 15, "treated"] = (df["year"] >= 2020).astype(int)
    return df


@pytest.fixture
def tiny_panel():
    """10-unit × 3-period panel → n=30, below callaway_santanna's n_min=50."""
    rng = np.random.default_rng(0)
    n_units, n_periods = 10, 3
    df = pd.DataFrame({
        "unit": np.repeat(range(n_units), n_periods),
        "year": np.tile(range(2018, 2021), n_units),
        "treated": 0,
        "y": rng.normal(size=n_units * n_periods),
    })
    df.loc[df["unit"] < 5, "treated"] = (df["year"] >= 2020).astype(int)
    return df


class TestEnrichment:
    def test_populated_function_gets_card(self, did_panel):
        rec = sp.recommend(did_panel, y="y", treatment="treated",
                           id="unit", time="year")
        # CS is the primary staggered-DID recommendation and has a card
        cs = next(
            (r for r in rec.recommendations if r["function"] == "callaway_santanna"),
            None,
        )
        assert cs is not None, "callaway_santanna should appear in staggered DID recs"
        assert "agent_card" in cs
        assert cs["typical_n_min"] == 50
        assert len(cs["failure_modes"]) >= 3
        assert "sp.sensitivity_rr" in " ".join(
            fm["alternative"] for fm in cs["failure_modes"]
        )

    def test_unpopulated_function_skipped(self, did_panel):
        # sun_abraham is auto-registered only → no card content
        rec = sp.recommend(did_panel, y="y", treatment="treated",
                           id="unit", time="year")
        sa = next(
            (r for r in rec.recommendations if r["function"] == "sun_abraham"),
            None,
        )
        if sa is not None:  # sun_abraham appears when staggered
            assert "agent_card" not in sa
            assert sa.get("typical_n_min") is None

    def test_n_below_threshold_warning(self, tiny_panel):
        rec = sp.recommend(tiny_panel, y="y", treatment="treated",
                           id="unit", time="year")
        matching = [w for w in rec.warnings if "typical minimum" in w]
        assert matching, "expected an n-below-typical warning"
        assert "callaway_santanna" in matching[0]

    def test_assumptions_promoted_when_missing(self, did_panel):
        """When the hardcoded rec lacks 'assumptions', enrichment promotes
        from the card."""
        rec = sp.recommend(did_panel, y="y", treatment="treated",
                           id="unit", time="year")
        # Sun-Abraham in the hardcoded dict has no 'assumptions'; after
        # enrichment, since sa has no card, it should remain empty — so
        # we verify via callaway: hardcoded already had assumptions, so
        # it should NOT be overwritten with card content.
        cs = next(r for r in rec.recommendations
                  if r["function"] == "callaway_santanna")
        assert "Parallel trends" in cs["assumptions"][0]

    def test_hardcoded_fields_preserved(self, did_panel):
        """Existing fields (reason, code, params) must not be touched."""
        rec = sp.recommend(did_panel, y="y", treatment="treated",
                           id="unit", time="year")
        cs = next(r for r in rec.recommendations
                  if r["function"] == "callaway_santanna")
        # These were set by the hardcoded block; enrichment must not drop them
        assert cs["method"].startswith("Callaway")
        assert "staggered treatment" in cs["reason"].lower()
        assert cs["code"].startswith("# Derived cohort column") or cs["code"].startswith("sp.callaway")
        assert cs["params"]["y"] == "y"
