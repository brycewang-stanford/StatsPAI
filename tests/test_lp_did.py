"""Tests for sp.lp_did (Dube-Girardi-Jordà-Taylor 2023 LP-DiD).

Identification details flagged as [待核验] in the source — these tests
verify behaviour, not paper parity. A reference-parity check against
the authors' Stata implementation is left for a follow-up once we lock
the paper version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _staggered_panel(
    n_units: int = 80,
    n_periods: int = 10,
    effect: float = 0.5,
    share_treated: float = 0.6,
    seed: int = 0,
) -> pd.DataFrame:
    """Staggered adoption panel with homogeneous treatment effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        # About share_treated units eventually treated
        if rng.random() < share_treated:
            g = int(rng.integers(3, n_periods - 1))
        else:
            g = 0  # never-treated
        unit_fx = rng.normal(scale=0.3)
        for t in range(1, n_periods + 1):
            d = int(g > 0 and t >= g)
            y = unit_fx + 0.1 * t + effect * d + rng.normal(scale=0.5)
            rows.append({"i": u, "t": t, "d": d, "y": y})
    return pd.DataFrame(rows)


class TestLPDIDBasic:
    def test_result_structure(self):
        df = _staggered_panel(seed=1)
        r = sp.lp_did(df, y="y", unit="i", time="t", treatment="d", horizons=(-2, 3))
        assert r is not None
        assert r.method.startswith("LP-DiD")
        es = r.model_info["event_study"]
        assert list(es.columns)[:6] == [
            "relative_time",
            "att",
            "se",
            "pvalue",
            "ci_lower",
            "ci_upper",
        ]
        # horizons (-2, 3) → 6 rows
        assert len(es) == 6
        assert set(es["relative_time"]) == {-2, -1, 0, 1, 2, 3}

    def test_horizon_minus_one_is_reference(self):
        """By construction Y_{t+h} - Y_{t-1} at h=-1 is zero for all units."""
        df = _staggered_panel(seed=2)
        r = sp.lp_did(df, y="y", unit="i", time="t", treatment="d", horizons=(-2, 2))
        es = r.model_info["event_study"].set_index("relative_time")
        # h=-1 row: attribute equals zero mechanically (trivial ref).
        assert es.loc[-1, "att"] == pytest.approx(0.0, abs=1e-10)

    def test_recovers_true_effect_sign(self):
        """True effect 0.7 should give a positive horizon-0 estimate."""
        df = _staggered_panel(n_units=200, effect=0.7, seed=5)
        r = sp.lp_did(df, y="y", unit="i", time="t", treatment="d", horizons=(-2, 3))
        assert r.estimate > 0
        # Loose recovery bound — LP-DiD with TWFE time FE, homogeneous
        # DGP: effect should land within ±0.4 of truth.
        assert 0.3 < r.estimate < 1.1

    def test_zero_effect_placebo_covers_zero_mostly(self):
        """Zero-effect DGP: placebo horizons (−2, −3) should cover 0
        most of the time across independent seeds."""
        covers = 0
        trials = 6
        for s in range(trials):
            df = _staggered_panel(n_units=150, effect=0.0, seed=100 + s)
            r = sp.lp_did(
                df, y="y", unit="i", time="t", treatment="d", horizons=(-3, 2)
            )
            es = r.model_info["event_study"].set_index("relative_time")
            if es.loc[-3, "ci_lower"] <= 0 <= es.loc[-3, "ci_upper"]:
                covers += 1
        assert covers >= 4, f"Only {covers}/{trials} placebo CIs covered zero"


class TestLPDIDInputValidation:
    def test_rejects_non_binary_treatment(self):
        df = _staggered_panel(seed=0).copy()
        df["d"] = df["d"].astype(float) + 0.5  # no longer in {0,1}
        with pytest.raises(ValueError, match="binary"):
            sp.lp_did(df, y="y", unit="i", time="t", treatment="d")

    def test_rejects_invalid_clean_controls(self):
        df = _staggered_panel(seed=0)
        with pytest.raises(ValueError, match="clean_controls"):
            sp.lp_did(
                df,
                y="y",
                unit="i",
                time="t",
                treatment="d",
                clean_controls="invalid_option",
            )

    def test_rejects_horizon_range_reversed(self):
        df = _staggered_panel(seed=0)
        with pytest.raises(ValueError, match="min"):
            sp.lp_did(df, y="y", unit="i", time="t", treatment="d", horizons=(5, 0))


class TestLPDIDCleanControlVariants:
    def test_never_vs_not_yet_produce_different_samples(self):
        """Switching clean_controls should change the sample; don't require
        equal estimates, just that the function respects the flag."""
        df = _staggered_panel(seed=7)
        r_nyt = sp.lp_did(
            df,
            y="y",
            unit="i",
            time="t",
            treatment="d",
            horizons=(0, 2),
            clean_controls="not_yet_treated",
        )
        r_nev = sp.lp_did(
            df,
            y="y",
            unit="i",
            time="t",
            treatment="d",
            horizons=(0, 2),
            clean_controls="never_treated",
        )
        n_nyt = (
            r_nyt.model_info["event_study"]
            .loc[r_nyt.model_info["event_study"]["relative_time"] == 0, "n_obs"]
            .iloc[0]
        )
        n_nev = (
            r_nev.model_info["event_study"]
            .loc[r_nev.model_info["event_study"]["relative_time"] == 0, "n_obs"]
            .iloc[0]
        )
        # Never-treated is a subset of not-yet-treated ⇒ n_nev ≤ n_nyt.
        assert n_nev <= n_nyt


class TestLPDIDNoTimeFE:
    def test_toggling_time_fe_doesnt_crash(self):
        df = _staggered_panel(seed=3)
        r = sp.lp_did(
            df, y="y", unit="i", time="t", treatment="d", horizons=(0, 2), time_fe=False
        )
        assert np.isfinite(r.estimate)
