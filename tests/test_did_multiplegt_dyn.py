"""Tests for sp.did_multiplegt_dyn (dCDH 2024) MVP.

These are structural + recovery tests, NOT paper-parity. Paper-parity
will require the R ``DIDmultiplegtDYN`` reference fixtures (see
``docs/rfc/multiplegt_dyn.md``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _staggered_on_panel(
    n_units: int = 120,
    n_periods: int = 10,
    effect: float = 0.8,
    seed: int = 0,
) -> pd.DataFrame:
    """Staggered adoption (switch-on only), homogeneous effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = int(rng.integers(3, n_periods - 2)) if rng.random() > 0.3 else 0
        unit_fx = rng.normal(scale=0.3)
        for t in range(1, n_periods + 1):
            d = int(g > 0 and t >= g)
            y = unit_fx + 0.1 * t + effect * d + rng.normal(scale=0.3)
            rows.append({"i": u, "t": t, "d": d, "y": y})
    return pd.DataFrame(rows)


class TestDidMultiplegtDynBasic:
    def test_recovers_positive_effect(self):
        df = _staggered_on_panel(n_units=200, effect=0.8, seed=1)
        r = sp.did_multiplegt_dyn(
            df,
            y="y",
            group="i",
            time="t",
            treatment="d",
            placebo=2,
            dynamic=3,
            n_boot=100,
            seed=1,
        )
        # Average dynamic effect should be close to 0.8.
        assert abs(r.estimate - 0.8) < 0.3

    def test_placebo_horizons_near_zero(self):
        df = _staggered_on_panel(n_units=200, effect=0.8, seed=2)
        r = sp.did_multiplegt_dyn(
            df,
            y="y",
            group="i",
            time="t",
            treatment="d",
            placebo=2,
            dynamic=2,
            n_boot=80,
            seed=2,
        )
        es = r.model_info["event_study"].set_index("relative_time")
        # Placebo should be near zero.
        assert abs(es.loc[-2, "att"]) < 0.3
        assert abs(es.loc[-1, "att"]) < 0.3

    def test_joint_placebo_test_does_not_reject_under_pt(self):
        """Under the DGP (which respects PT), joint placebo test should
        not reject at nominal 5%. Run a few seeds; at least 3/5 should
        give p > 0.05."""
        ok = 0
        trials = 5
        for s in range(trials):
            df = _staggered_on_panel(n_units=200, effect=0.5, seed=200 + s)
            r = sp.did_multiplegt_dyn(
                df,
                y="y",
                group="i",
                time="t",
                treatment="d",
                placebo=2,
                dynamic=2,
                n_boot=80,
                seed=200 + s,
            )
            plac = r.model_info.get("joint_placebo_test")
            if plac is not None and plac["pvalue"] > 0.05:
                ok += 1
        assert ok >= 3, f"Only {ok}/{trials} joint placebo tests had p > 0.05"

    def test_event_study_shape(self):
        df = _staggered_on_panel(seed=3)
        r = sp.did_multiplegt_dyn(
            df,
            y="y",
            group="i",
            time="t",
            treatment="d",
            placebo=2,
            dynamic=3,
            n_boot=40,
            seed=3,
        )
        es = r.model_info["event_study"]
        # Canonical event-study columns.
        for col in (
            "relative_time",
            "att",
            "se",
            "pvalue",
            "ci_lower",
            "ci_upper",
            "type",
        ):
            assert col in es.columns
        # 2 placebo + 4 dynamic (l=0..3) = 6 rows
        assert len(es) == 6
        assert set(es["relative_time"]) == {-2, -1, 0, 1, 2, 3}
        assert set(es["type"].unique()) <= {"placebo", "dynamic"}


class TestDidMultiplegtDynControlGroups:
    def test_never_vs_notyet_produce_estimates(self):
        df = _staggered_on_panel(n_units=150, seed=10)
        r_ny = sp.did_multiplegt_dyn(
            df,
            y="y",
            group="i",
            time="t",
            treatment="d",
            placebo=1,
            dynamic=2,
            control="not_yet_treated",
            n_boot=50,
            seed=10,
        )
        r_nev = sp.did_multiplegt_dyn(
            df,
            y="y",
            group="i",
            time="t",
            treatment="d",
            placebo=1,
            dynamic=2,
            control="never_treated",
            n_boot=50,
            seed=10,
        )
        assert np.isfinite(r_ny.estimate)
        assert np.isfinite(r_nev.estimate)


class TestDidMultiplegtDynInputValidation:
    def test_treatment_levels_do_not_matter_only_their_changes(self):
        """A two-level treatment at 0.5 / 1.5 is the binary one, relabelled.

        Since 1.39.0 a discrete treatment is matched on its period-one level
        instead of being refused; shifting both levels by a constant changes
        neither the switchers, nor their controls, nor the effects.
        """
        df = _staggered_on_panel(seed=0).copy()
        kwargs = dict(y="y", group="i", time="t", treatment="d", n_boot=0)
        binary = sp.did_multiplegt_dyn(df, se_method="analytic", **kwargs)
        df["d"] = df["d"].astype(float) + 0.5
        shifted = sp.did_multiplegt_dyn(df, se_method="analytic", **kwargs)
        a = binary.model_info["event_study"]
        b = shifted.model_info["event_study"]
        np.testing.assert_allclose(a["att"], b["att"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(a["se"], b["se"], rtol=0, atol=1e-12)

    def test_rejects_non_numeric_treatment(self):
        df = _staggered_on_panel(seed=0).copy()
        df["d"] = df["d"].map({0: "off", 1: "on"})
        with pytest.raises(sp.exceptions.MethodIncompatibility, match="numeric"):
            sp.did_multiplegt_dyn(
                df, y="y", group="i", time="t", treatment="d", n_boot=0
            )

    def test_rejects_invalid_control(self):
        df = _staggered_on_panel(seed=0)
        with pytest.raises(ValueError, match="control"):
            sp.did_multiplegt_dyn(
                df,
                y="y",
                group="i",
                time="t",
                treatment="d",
                control="oops",
                n_boot=5,
                seed=0,
            )

    def test_rejects_negative_horizons(self):
        df = _staggered_on_panel(seed=0)
        with pytest.raises(ValueError, match="non-negative"):
            sp.did_multiplegt_dyn(
                df,
                y="y",
                group="i",
                time="t",
                treatment="d",
                dynamic=-1,
                n_boot=5,
                seed=0,
            )


class TestDidMultiplegtDynMVPLabel:
    def test_method_label_flags_mvp_status(self):
        """MVP label must carry the [待核验] marker so users know this is
        not paper-faithful yet."""
        df = _staggered_on_panel(n_units=60, seed=0)
        r = sp.did_multiplegt_dyn(
            df,
            y="y",
            group="i",
            time="t",
            treatment="d",
            placebo=1,
            dynamic=1,
            n_boot=20,
            seed=0,
        )
        assert "待核验" in r.method or "MVP" in r.method

    def test_warning_in_model_info(self):
        df = _staggered_on_panel(n_units=60, seed=0)
        r = sp.did_multiplegt_dyn(
            df,
            y="y",
            group="i",
            time="t",
            treatment="d",
            placebo=1,
            dynamic=1,
            n_boot=20,
            seed=0,
        )
        assert "warning" in r.model_info
        # The note now says which options are available and pinned, which
        # are not, and where the roadmap lives.
        note = r.model_info["warning"]
        assert "docs/rfc/multiplegt_dyn.md" in note
        assert "controls=" in note and "trends_lin" in note
