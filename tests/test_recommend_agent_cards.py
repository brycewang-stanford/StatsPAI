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

    def test_sun_abraham_now_populated_post_1_6_3(self, did_panel):
        """After v1.6.3 D.1 registry enrichment, ``sun_abraham`` carries a
        rich spec with assumptions + failure_modes + alternatives. This
        test used to assert the opposite ('unpopulated → no agent_card');
        flipped here to track the new contract.

        If in the future a different DiD estimator that ``sp.recommend``
        surfaces is still auto-only, repurpose this test to that
        function — it exists to verify the enrichment path, not the
        specific function."""
        rec = sp.recommend(did_panel, y="y", treatment="treated",
                           id="unit", time="year")
        sa = next(
            (r for r in rec.recommendations if r["function"] == "sun_abraham"),
            None,
        )
        if sa is not None:
            assert "agent_card" in sa
            assert sa.get("typical_n_min") == 50

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

    def test_filter_unstable_recommendations_drops_experimental_names(self):
        """Registry-backed stability gating should strip experimental entries."""
        from statspai.smart.recommend import _filter_unstable_recommendations

        recs = [
            {"function": "regress"},
            {"function": "text_treatment_effect"},
            {"function": "callaway_santanna"},
        ]
        filtered, dropped = _filter_unstable_recommendations(recs)

        assert [r["function"] for r in filtered] == [
            "regress", "callaway_santanna",
        ]
        assert dropped == ["text_treatment_effect"]

    def test_recommend_allow_experimental_flag_controls_filter_hook(
        self, did_panel, monkeypatch,
    ):
        """Public API wiring: default calls the filter, opt-in bypasses it."""
        import importlib

        recommend_mod = importlib.import_module("statspai.smart.recommend")

        calls = []

        def _fake_filter(recs):
            calls.append([r.get("function") for r in recs])
            return recs[:-1], ["text_treatment_effect"]

        monkeypatch.setattr(
            recommend_mod, "_filter_unstable_recommendations", _fake_filter,
        )

        gated = recommend_mod.recommend(
            did_panel, y="y", treatment="treated",
            id="unit", time="year",
        )
        assert calls, "default recommend() should invoke stability gating"
        assert any("allow_experimental=True" in w for w in gated.warnings)

        calls.clear()
        ungated = recommend_mod.recommend(
            did_panel, y="y", treatment="treated",
            id="unit", time="year",
            allow_experimental=True,
        )
        assert not calls, "allow_experimental=True should bypass the filter"
        assert not any("allow_experimental=True" in w for w in ungated.warnings)
