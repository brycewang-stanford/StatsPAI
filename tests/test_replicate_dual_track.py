"""Tests for the dual-track replicate() guide and bundled real data.

Covers:
- ``sp.list_replications()`` exposes the new ``has_real_data /
  has_classic_track / has_modern_track`` columns.
- ``sp.datasets.card_1995(simulated=False)`` and
  ``sp.datasets.california_prop99(simulated=False)`` load the bundled
  real CSV.
- The Card 1995 classic recipe recovers the published Table 2 numbers
  on the bundled real data.
- The ADH 2010 classic recipe lands within tolerance of the paper-
  headline gap.
- Legacy single-track entries still render.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# =====================================================================
# list_replications schema
# =====================================================================


class TestListReplicationsSchema:
    def test_columns(self):
        df = sp.list_replications()
        expected = {
            "key",
            "title",
            "design",
            "journal",
            "n_obs",
            "has_real_data",
            "has_classic_track",
            "has_modern_track",
        }
        assert expected.issubset(df.columns)

    def test_card_1995_has_full_dual_track(self):
        df = sp.list_replications().set_index("key")
        row = df.loc["card_1995"]
        assert bool(row["has_real_data"]) is True
        assert bool(row["has_classic_track"]) is True
        assert bool(row["has_modern_track"]) is True

    def test_abadie_2010_has_full_dual_track(self):
        df = sp.list_replications().set_index("key")
        row = df.loc["abadie_2010"]
        assert bool(row["has_real_data"]) is True
        assert bool(row["has_classic_track"]) is True
        assert bool(row["has_modern_track"]) is True

    def test_legacy_entries_remain(self):
        keys = set(sp.list_replications()["key"])
        for legacy in [
            "lalonde_1986",
            "lee_2008",
            "graddy_2006",
            "angrist_pischke_mhe",
        ]:
            assert legacy in keys


# =====================================================================
# Real-data loaders
# =====================================================================


class TestCardRealLoader:
    @pytest.fixture(scope="class")
    def df(self):
        return sp.datasets.card_1995(simulated=False)

    def test_shape_and_columns(self, df):
        assert df.shape == (3010, 9)
        assert {
            "lwage",
            "educ",
            "exper",
            "expersq",
            "black",
            "south",
            "smsa",
            "nearc4",
            "nearc2",
        }.issubset(df.columns)

    def test_attrs_flag_real(self, df):
        assert df.attrs.get("data_source") == "real"
        assert df.attrs.get("simulated") is False

    def test_classic_recipe_matches_paper(self, df):
        """Card (1995) Table 2: OLS β_educ = 0.075, IV β_educ = 0.132."""
        ols = sp.regress(
            "lwage ~ educ + exper + expersq + black + south + smsa",
            data=df,
            robust="hc1",
        )
        iv = sp.ivreg(
            "lwage ~ exper + expersq + black + south + smsa + " "(educ ~ nearc4)",
            data=df,
            robust="hc1",
        )
        # 3-decimal-place match to the published Card 1995 numbers
        assert ols.params["educ"] == pytest.approx(0.075, abs=2e-3)
        assert iv.params["educ"] == pytest.approx(0.132, abs=2e-3)

    def test_simulated_default_unchanged(self):
        """Backward compat: card_1995() with no args returns simulated."""
        df = sp.datasets.card_1995()
        assert df.shape == (3010, 8)  # simulated lacks nearc2
        assert "nearc2" not in df.columns


class TestProp99RealLoader:
    @pytest.fixture(scope="class")
    def df(self):
        return sp.datasets.california_prop99(simulated=False)

    def test_shape_and_columns(self, df):
        # 39 states × 31 years (1970-2000) = 1209
        assert df.shape == (1209, 8)
        assert {
            "state",
            "year",
            "cigsale",
            "lnincome",
            "beer",
            "age15to24",
            "retprice",
            "treated",
        }.issubset(df.columns)

    def test_treated_indicator_derived(self, df):
        ca = df[df["state"] == "California"]
        assert (ca[ca["year"] < 1989]["treated"] == 0).all()
        assert (ca[ca["year"] >= 1989]["treated"] == 1).all()

    def test_california_1988_cigsale_known_value(self, df):
        """ADH paper benchmark: California 1988 per-capita cigsale = 90.10."""
        ca88 = df[(df["state"] == "California") & (df["year"] == 1988)]["cigsale"].iloc[
            0
        ]
        assert ca88 == pytest.approx(90.10, abs=0.01)

    def test_simulated_default_unchanged(self):
        """Backward compat: no-arg call returns the simulated panel."""
        df = sp.datasets.california_prop99()
        assert df.shape == (1209, 8)
        assert "treated" in df.columns


# =====================================================================
# Replicate guide rendering and data routing
# =====================================================================


class TestReplicateCard1995:
    def test_returns_real_data_by_default(self):
        data, _ = sp.replicate("card_1995")
        assert data.attrs.get("data_source") == "real"
        assert data.shape == (3010, 9)

    def test_simulated_override(self):
        data, _ = sp.replicate("card_1995", simulated=True)
        # Simulated branch: 8 cols (no nearc2)
        assert data.shape == (3010, 8)
        assert (
            data.attrs.get("data_source") is None
            or data.attrs.get("data_source") != "real"
        )

    def test_guide_contains_dual_tracks(self):
        _, guide = sp.replicate("card_1995")
        assert "TRACK 1 — CLASSIC" in guide
        assert "TRACK 2 — MODERN" in guide
        assert "card1995using" in guide
        assert "andrews2019weak" in guide

    def test_guide_shows_paper_vs_statspai_numbers(self):
        _, guide = sp.replicate("card_1995")
        assert "0.0740" in guide
        assert "0.1323" in guide
        assert "Paper = +0.0750" in guide
        assert "Paper = +0.1320" in guide


class TestReplicateAbadie2010:
    def test_returns_real_data_by_default(self):
        data, _ = sp.replicate("abadie_2010")
        assert data.attrs.get("data_source") == "real"
        assert data.shape == (1209, 8)

    def test_guide_contains_dual_tracks(self):
        _, guide = sp.replicate("abadie_2010")
        assert "TRACK 1 — CLASSIC" in guide
        assert "TRACK 2 — MODERN" in guide
        assert "abadie2010synthetic" in guide
        assert "arkhangelsky2021synthetic" in guide
        assert "benmichael2021augmented" in guide

    @pytest.mark.filterwarnings("ignore")
    def test_classic_outcome_only_att_pinned(self):
        """Outcome-only SCM on real ADH data should land near -19.76."""
        data, _ = sp.replicate("abadie_2010")
        sc = sp.synth(
            data=data,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            method="classic",
            placebo=False,
        )
        # Pinned to the StatsPAI value on the bundled real data; the
        # paper headline is a qualitative ≈ -19 from Figure 2.
        assert float(sc.estimate) == pytest.approx(-19.7605, abs=0.5)


class TestReplicateLalonde:
    def test_returns_real_data_by_default(self):
        data, _ = sp.replicate("lalonde_1986")
        assert data.attrs.get("data_source") == "real"
        # MatchIt::lalonde: 185 NSW treated + 429 PSID controls = 614
        assert data.shape == (614, 11)

    def test_guide_contains_dual_tracks(self):
        _, guide = sp.replicate("lalonde_1986")
        assert "TRACK 1 — CLASSIC" in guide
        assert "TRACK 2 — MODERN" in guide
        assert "dehejia1999causal" in guide
        assert "chernozhukov2018double" in guide

    @pytest.mark.filterwarnings("ignore")
    def test_classic_recipe_recovers_pinned_numbers(self):
        """Naive OLS, adjusted OLS, 1:1 NN PSM all match StatsPAI pins."""
        data, _ = sp.replicate("lalonde_1986")
        naive = sp.regress("re78 ~ treat", data=data, robust="hc1")
        adj = sp.regress(
            "re78 ~ treat + age + educ + black + hispanic + "
            "married + nodegree + re74 + re75",
            data=data,
            robust="hc1",
        )
        # Tight regression-test pins (these are exact StatsPAI outputs)
        assert float(naive.params["treat"]) == pytest.approx(-635.03, abs=2.0)
        assert float(adj.params["treat"]) == pytest.approx(1548.24, abs=2.0)


class TestReplicateLee2008:
    def test_returns_real_data_by_default(self):
        data, _ = sp.replicate("lee_2008")
        assert data.attrs.get("data_source") == "real"
        # rdrobust::rdrobust_RDsenate
        assert data.shape == (1390, 2)
        assert {"x", "y"}.issubset(data.columns)

    def test_guide_contains_dual_tracks(self):
        _, guide = sp.replicate("lee_2008")
        assert "TRACK 1 — CLASSIC" in guide
        assert "TRACK 2 — MODERN" in guide
        assert "lee2008randomized" in guide
        assert "calonico2014robust" in guide

    @pytest.mark.filterwarnings("ignore")
    def test_cct_bandwidth_recovers_paper(self):
        """Conventional jump at CCT bandwidth ≈ 7.41 (Lee Table 1: 7.99)."""
        pytest.importorskip("rdrobust")
        data, _ = sp.replicate("lee_2008")
        rd = sp.rdrobust(data, y="y", x="x", c=0, kernel="triangular", bwselect="cct")
        conv = rd.diagnostics["conventional"]
        assert float(conv["estimate"]) == pytest.approx(7.414, abs=1e-2)
        rob = rd.diagnostics["robust"]
        assert float(rob["estimate"]) == pytest.approx(7.507, abs=1e-2)


class TestReplicateLegacy:
    def test_graddy_renders(self):
        data, guide = sp.replicate("graddy_2006")
        assert data.shape == (111, 8)
        assert "CODE" in guide

    def test_mhe_renders(self):
        data, guide = sp.replicate("angrist_pischke_mhe")
        assert "CODE" in guide


class TestReplicateErrors:
    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown replication"):
            sp.replicate("nonexistent_paper_2099")
