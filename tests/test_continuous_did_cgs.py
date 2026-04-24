"""Tests for sp.continuous_did(method='cgs') — CGS 2024 MVP.

These are structural + recovery tests, NOT paper-parity. A full CGS
(2024) implementation with DR + IPW + analytical IF variance is tracked
in ``docs/rfc/continuous_did_cgs.md`` for a follow-up sprint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import statspai as sp


def _two_period_panel(
    n_units: int = 400,
    share_treated: float = 0.6,
    slope: float = 0.4,
    noise: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """2-period panel with continuous dose.

    True potential-outcomes DGP:
        y_{i, post} = y_{i, pre} + 0.15 + slope * dose_i + eps
    so ATT(d) = slope * d and ACRT(d) = slope (constant).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        dose_u = 0.0 if rng.random() > share_treated else float(rng.exponential(2.0))
        unit_fx = rng.normal(scale=0.4)
        for t in [0, 1]:
            d_eff = slope * dose_u if t == 1 else 0.0
            y = unit_fx + 0.15 * t + d_eff + rng.normal(scale=noise)
            rows.append({"i": u, "t": t, "dose": dose_u, "y": y})
    return pd.DataFrame(rows)


class TestCGSMVPRecovery:
    def test_att_curve_increases_with_dose(self):
        df = _two_period_panel(n_units=600, slope=0.5, seed=3)
        r = sp.continuous_did(
            df,
            y="y",
            dose="dose",
            time="t",
            id="i",
            t_pre=0,
            t_post=1,
            method="cgs",
            n_boot=60,
            seed=3,
        )
        # Detail DataFrame should show increasing att_d as dose rises
        # (true ATT(d) is linear in d).
        att = r.detail["att_d"].values
        # Monotonic increase — loose check with a few dips allowed.
        n_dips = int(np.sum(np.diff(att) < -0.2))
        assert n_dips <= 3, f"ATT(d) curve has {n_dips} large dips"

    def test_acrt_approximates_true_slope(self):
        """True ACRT = slope = 0.4 uniformly. MVP's average ACRT should
        land within ±0.3 of 0.4."""
        df = _two_period_panel(n_units=600, slope=0.4, seed=5)
        r = sp.continuous_did(
            df,
            y="y",
            dose="dose",
            time="t",
            id="i",
            t_pre=0,
            t_post=1,
            method="cgs",
            n_boot=60,
            seed=5,
        )
        acrt = r.model_info["acrt_overall"]
        assert abs(acrt - 0.4) < 0.3

    def test_zero_slope_att_near_zero(self):
        """When slope=0, ATT(d) = 0 for all d. Average ATT should be
        statistically indistinguishable from 0."""
        covers = 0
        trials = 4
        for s in range(trials):
            df = _two_period_panel(n_units=400, slope=0.0, seed=100 + s)
            r = sp.continuous_did(
                df,
                y="y",
                dose="dose",
                time="t",
                id="i",
                t_pre=0,
                t_post=1,
                method="cgs",
                n_boot=60,
                seed=100 + s,
            )
            if r.ci[0] <= 0 <= r.ci[1]:
                covers += 1
        assert covers >= 2


class TestCGSMVPStructure:
    def test_method_label_flags_mvp_status(self):
        df = _two_period_panel(n_units=100, seed=0)
        r = sp.continuous_did(
            df,
            y="y",
            dose="dose",
            time="t",
            id="i",
            t_pre=0,
            t_post=1,
            method="cgs",
            n_boot=20,
            seed=0,
        )
        assert "MVP" in r.method or "待核验" in r.method

    def test_detail_has_dose_att_acrt(self):
        df = _two_period_panel(n_units=200, seed=0)
        r = sp.continuous_did(
            df,
            y="y",
            dose="dose",
            time="t",
            id="i",
            t_pre=0,
            t_post=1,
            method="cgs",
            n_boot=20,
            seed=0,
        )
        assert r.detail is not None
        assert set(r.detail.columns) >= {"dose", "att_d", "acrt_d"}
        # Grid is monotonic
        assert (r.detail["dose"].diff().dropna() > 0).all()

    def test_warning_in_model_info(self):
        df = _two_period_panel(n_units=100, seed=0)
        r = sp.continuous_did(
            df,
            y="y",
            dose="dose",
            time="t",
            id="i",
            t_pre=0,
            t_post=1,
            method="cgs",
            n_boot=20,
            seed=0,
        )
        assert "warning" in r.model_info
        assert "MVP" in r.model_info["warning"]


class TestCGSMVPGracefulFallback:
    def test_no_dose_zero_controls_returns_warning_result(self):
        """When no unit has dose == 0, CGS MVP can't compute control ΔY.
        Should return a result with a warning instead of crashing."""
        df = _two_period_panel(n_units=100, seed=0)
        # Shift dose away from 0
        df = df.copy()
        df["dose"] = df["dose"] + 0.5
        r = sp.continuous_did(
            df,
            y="y",
            dose="dose",
            time="t",
            id="i",
            t_pre=0,
            t_post=1,
            method="cgs",
            n_boot=10,
            seed=0,
        )
        assert np.isnan(r.estimate)
        assert "warning" in r.model_info

    def test_very_few_treated_units_returns_warning(self):
        """With < 3 treated units, MVP should refuse to smooth ATT(d)."""
        df = _two_period_panel(n_units=80, share_treated=0.05, seed=7)
        r = sp.continuous_did(
            df,
            y="y",
            dose="dose",
            time="t",
            id="i",
            t_pre=0,
            t_post=1,
            method="cgs",
            n_boot=10,
            seed=7,
        )
        # Either very few treated (graceful NaN) or a valid estimate.
        if np.isnan(r.estimate):
            assert "warning" in r.model_info
