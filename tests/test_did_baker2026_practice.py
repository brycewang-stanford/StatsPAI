"""Tests for the Baker et al. (2026) practitioner-guide surfaces.

Covers the four things the JEL practice guide asks a DiD implementation
to get right, none of which StatsPAI enforced before:

1. ω weights are part of the estimand and must never be silently dropped.
2. The parallel-trends assumption in force must be named in the result.
3. Covariate balance must be reported in *changes* as well as levels.
4. The TWFE shortcuts the paper warns about must warn.

Reference: Baker, Callaway, Cunningham, Goodman-Bacon and Sant'Anna
(2026), "Difference-in-Differences Designs: A Practitioner's Guide",
Journal of Economic Literature 64(2), 498-557.
doi:10.1257/jel.20251650
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning, MethodIncompatibility

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _staggered_panel(
    n_units: int = 320,
    seed: int = 11,
    big_effect: float = 4.0,
    small_effect: float = -1.0,
    big_weight: float = 100.0,
    small_weight: float = 5.0,
) -> pd.DataFrame:
    """Panel where ω flips the sign of the ATT.

    A quarter of units carry 20x the population weight and have a
    positive effect; the rest have a negative one. The unweighted ATT
    (average over units) and the ω-weighted ATT (average over the
    population they represent) therefore have opposite signs — exactly
    the situation §3.1 of the paper is about.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        g = int(rng.choice([3, 4, 5, 0]))
        big = rng.random() < 0.25
        w = big_weight if big else small_weight
        fe = rng.normal()
        for t in range(1, 8):
            te = 0.0 if (g == 0 or t < g) else (big_effect if big else small_effect)
            rows.append(
                {
                    "i": i,
                    "t": t,
                    "g": g,
                    "w": w,
                    "big": int(big),
                    "y": fe + 0.2 * t + te + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _staggered_panel()


KEYS = {"y": "y", "g": "g", "t": "t", "i": "i"}


# ----------------------------------------------------------------------
# 1. Weights are part of the estimand
# ----------------------------------------------------------------------


class TestWeightsChangeTheEstimand:
    def test_weighted_and_unweighted_att_differ_in_sign(self, panel):
        """The headline failure mode: ω is not a precision knob.

        Baker et al. (2026, §3.1): weighting "will also change the target
        parameter, potentially by a lot when treatment effects are
        correlated with the weights". Here they differ in sign.
        """
        unw = sp.callaway_santanna(panel, **KEYS)
        wtd = sp.callaway_santanna(panel, weights="w", **KEYS)
        assert unw.estimate < 0.5, unw.estimate
        assert wtd.estimate > 2.5, wtd.estimate
        assert np.sign(wtd.estimate) != np.sign(unw.estimate)

    def test_dispatcher_no_longer_silently_ignores_weights(self, panel):
        """`sp.did(..., weights=)` used to return the unweighted number."""
        unw = sp.did(panel, y="y", treat="g", time="t", id="i", method="cs")
        wtd = sp.did(
            panel, y="y", treat="g", time="t", id="i", method="cs", weights="w"
        )
        assert abs(wtd.estimate - unw.estimate) > 1.0

    @pytest.mark.parametrize("method", ["sun_abraham", "sdid", "bjs"])
    def test_weight_unaware_methods_refuse_rather_than_ignore(self, panel, method):
        """Silently dropping ω answers a different question (§3.7)."""
        with pytest.raises(MethodIncompatibility, match="does not implement unit"):
            sp.did(
                panel,
                y="y",
                treat="g",
                time="t",
                id="i",
                method=method,
                weights="w",
            )

    def test_constant_weights_reduce_to_unweighted(self, panel):
        """ω ≡ c must be a no-op, or the normalisation is wrong."""
        unw = sp.callaway_santanna(panel, **KEYS)
        const = sp.callaway_santanna(panel.assign(w=3.0), weights="w", **KEYS)
        assert const.estimate == pytest.approx(unw.estimate, rel=1e-10)
        assert const.se == pytest.approx(unw.se, rel=1e-10)

    def test_scale_invariance(self, panel):
        """ATT must not depend on the units ω is measured in."""
        a = sp.callaway_santanna(panel, weights="w", **KEYS)
        b = sp.callaway_santanna(panel.assign(w=panel["w"] * 1e6), weights="w", **KEYS)
        assert b.estimate == pytest.approx(a.estimate, rel=1e-10)

    @pytest.mark.parametrize("estimator", ["dr", "reg", "ipw", "ipw_abadie"])
    def test_all_four_estimators_honour_weights(self, panel, estimator):
        unw = sp.callaway_santanna(panel, estimator=estimator, **KEYS)
        wtd = sp.callaway_santanna(panel, estimator=estimator, weights="w", **KEYS)
        assert abs(wtd.estimate - unw.estimate) > 1.0

    @pytest.mark.parametrize("agg", ["simple", "dynamic", "group", "calendar"])
    def test_aggregations_use_omega_weighted_cohort_shares(self, panel, agg):
        """Cohort shares must be ω-mass, not head counts (§5.2.4)."""
        unw = sp.aggte(
            sp.callaway_santanna(panel, **KEYS), type=agg, bstrap=False, cband=False
        )
        wtd = sp.aggte(
            sp.callaway_santanna(panel, weights="w", **KEYS),
            type=agg,
            bstrap=False,
            cband=False,
        )
        assert abs(wtd.estimate - unw.estimate) > 1.0
        assert np.isfinite(wtd.se) and wtd.se > 0

    def test_cohort_sizes_become_weighted_mass(self, panel):
        wtd = sp.callaway_santanna(panel, weights="w", **KEYS)
        sizes = wtd.model_info["cohort_sizes"]
        # ω-mass totals to n (weights are renormalised to mean 1).
        assert float(sizes.sum()) == pytest.approx(wtd.model_info["n_units"], rel=1e-8)

    # -- guards ------------------------------------------------------
    def test_time_varying_weights_rejected(self, panel):
        bad = panel.copy()
        bad.loc[bad["t"] == 4, "w"] = 1.0
        with pytest.raises(MethodIncompatibility, match="varies within unit"):
            sp.callaway_santanna(bad, weights="w", **KEYS)

    def test_negative_weights_rejected(self, panel):
        bad = panel.copy()
        bad.loc[bad["i"] == 0, "w"] = -1.0
        with pytest.raises(MethodIncompatibility, match="negative"):
            sp.callaway_santanna(bad, weights="w", **KEYS)

    def test_nan_weights_rejected(self, panel):
        bad = panel.copy()
        bad.loc[bad["i"] == 0, "w"] = np.nan
        with pytest.raises(MethodIncompatibility, match="NaN or infinite"):
            sp.callaway_santanna(bad, weights="w", **KEYS)

    def test_all_zero_weights_rejected(self, panel):
        with pytest.raises(MethodIncompatibility, match="sums to"):
            sp.callaway_santanna(panel.assign(w=0.0), weights="w", **KEYS)

    def test_missing_weight_column_rejected(self, panel):
        with pytest.raises(MethodIncompatibility, match="not in the data"):
            sp.callaway_santanna(panel, weights="nope", **KEYS)


# ----------------------------------------------------------------------
# 2. The parallel-trends assumption is named
# ----------------------------------------------------------------------


class TestParallelTrendsIsNamed:
    @pytest.mark.parametrize(
        "control_group,expected",
        [("nevertreated", "PT-GT-NEV"), ("notyettreated", "PT-GT-NYT")],
    )
    def test_label_matches_comparison_group(self, panel, control_group, expected):
        r = sp.callaway_santanna(panel, control_group=control_group, **KEYS)
        assert r.model_info["parallel_trends"]["label"] == expected

    def test_covariates_make_the_assumption_conditional(self, panel):
        r = sp.callaway_santanna(panel, x=["big"], **KEYS)
        pt = r.model_info["parallel_trends"]
        assert pt["label"] == "CPT-GT-NEV"
        assert pt["conditional"] is True
        assert "overlap" in pt["also_requires"].lower()

    def test_block_carries_a_statement_and_tradeoff(self, panel):
        pt = sp.callaway_santanna(panel, **KEYS).model_info["parallel_trends"]
        for key in ("statement", "comparison", "restricts_pretrends", "tradeoff"):
            assert pt[key], key
        assert pt["reference"] == "baker2026difference"

    def test_summary_prints_the_assumption(self, panel):
        txt = sp.callaway_santanna(panel, **KEYS).summary()
        assert "Identifying assumption" in txt
        assert "PT-GT-NEV" in txt

    def test_summary_flags_the_weighted_estimand(self, panel):
        txt = sp.callaway_santanna(panel, weights="w", **KEYS).summary()
        assert "omega-weighted ATT" in txt
        assert "not a robustness check" in txt

    def test_unweighted_summary_stays_silent(self, panel):
        assert "omega-weighted" not in sp.callaway_santanna(panel, **KEYS).summary()

    def test_weighted_paragraph_is_did_only(self):
        """Several DML modules also set model_info['weighted'].

        The paragraph describes a DiD estimand — "the population the
        treated units represent" — which is wrong for a weighted DML ATE.
        It must key off the parallel-trends record, not off `weighted`
        alone.
        """
        from statspai.core.results import CausalResult

        r = CausalResult(
            method="DML PLR",
            estimand="ATE",
            estimate=2.0,
            se=0.1,
            pvalue=0.0,
            ci=(1.8, 2.2),
            alpha=0.05,
            n_obs=800,
            model_info={"weighted": True, "weights": "w"},
        )
        assert "omega-weighted ATT" not in r.summary()


# ----------------------------------------------------------------------
# 3. Balance in levels and in changes
# ----------------------------------------------------------------------


def _balance_panel(seed: int = 3) -> pd.DataFrame:
    """Level-balanced, change-imbalanced: only the ΔX panel can see it.

    ``x`` is drawn from the same distribution in both groups and stays
    flat through the base period (t <= 3), so a levels table at t = 3
    shows nothing. The groups then diverge from t = 4 onwards, which is
    visible only in ΔX. This is the paper's point that "areas that are
    poor are not the same as areas that are becoming poor" (§4.3).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(400):
        g = int(rng.choice([4, 0]))
        x0 = rng.normal(10, 2)  # same distribution in both groups
        w = rng.gamma(2, 1) * 100
        drift = -1.2 if g > 0 else -0.2
        for t in range(1, 9):
            # flat through the base period, diverging afterwards
            shift = 0.0 if t <= 3 else drift * (t - 3)
            rows.append(
                {
                    "i": i,
                    "t": t,
                    "g": g,
                    "w": w,
                    "x": x0 + shift + rng.normal(0, 0.3),
                    "y": rng.normal(),
                }
            )
    return pd.DataFrame(rows)


class TestDiDBalance:
    def test_changes_panel_catches_what_levels_misses(self):
        bal = sp.did_balance(_balance_panel(), ["x"], g="g", t="t", i="i")
        lev = bal.levels.iloc[0]
        chg = bal.changes.iloc[0]
        assert abs(lev["norm_diff"]) < 0.25, "levels should look balanced"
        assert abs(chg["norm_diff"]) > 0.25, "changes should be imbalanced"
        assert "x" in bal.flagged

    def test_both_panels_present(self):
        bal = sp.did_balance(_balance_panel(), ["x"], g="g", t="t", i="i")
        assert set(bal.table["panel"]) == {"levels", "changes"}

    def test_weighted_columns_appear_only_when_requested(self):
        df = _balance_panel()
        unw = sp.did_balance(df, ["x"], g="g", t="t", i="i")
        wtd = sp.did_balance(df, ["x"], g="g", t="t", i="i", weights="w")
        assert "w_norm_diff" not in unw.table.columns
        assert "w_norm_diff" in wtd.table.columns
        assert unw.weighted is False and wtd.weighted is True

    def test_degenerate_variance_does_not_explode(self):
        """A deterministic ΔX must not produce a 1e15 normalized difference."""
        rng = np.random.default_rng(0)
        rows = []
        for i in range(200):
            g = int(rng.choice([4, 0]))
            for t in range(1, 9):
                rows.append(
                    {"i": i, "t": t, "g": g, "x": 5.0 - 0.2 * t, "y": rng.normal()}
                )
        bal = sp.did_balance(pd.DataFrame(rows), ["x"], g="g", t="t", i="i")
        chg = bal.changes.iloc[0]
        # identical deterministic change in both groups -> exactly balanced
        assert chg["norm_diff"] == pytest.approx(0.0, abs=1e-12)
        assert not chg["flagged"]

    def test_complete_separation_reports_infinite_not_garbage(self):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(200):
            g = int(rng.choice([4, 0]))
            step = -0.5 if g > 0 else -0.2  # deterministic, but different
            for t in range(1, 9):
                rows.append(
                    {"i": i, "t": t, "g": g, "x": 5.0 + step * t, "y": rng.normal()}
                )
        bal = sp.did_balance(pd.DataFrame(rows), ["x"], g="g", t="t", i="i")
        chg = bal.changes.iloc[0]
        assert np.isinf(chg["norm_diff"])
        assert chg["flagged"]

    def test_summary_and_latex_render(self):
        bal = sp.did_balance(_balance_panel(), ["x"], g="g", t="t", i="i", weights="w")
        txt = bal.summary()
        assert "Covariate LEVELS" in txt and "Covariate CHANGES" in txt
        assert "bad control" in txt  # the covariate-vs-mechanism caveat
        assert "\\begin{tabular}" in bal.to_latex()

    def test_rejects_unknown_columns(self):
        with pytest.raises(MethodIncompatibility, match="not found"):
            sp.did_balance(_balance_panel(), ["nope"], g="g", t="t", i="i")


# ----------------------------------------------------------------------
# 4. The TWFE traps warn
# ----------------------------------------------------------------------


class TestTWFETrapsWarn:
    def test_static_twfe_median_split_warns(self, panel):
        """β_OLS = ATT_avg - average pre-trend, not ATT_avg (§5.1.3)."""
        flat = panel.assign(treated=(panel["g"] > 0).astype(int))
        with pytest.warns(AssumptionWarning, match="NOT the average"):
            sp.did(flat, y="y", treat="treated", time="t", method="twfe")

    def test_twfe_with_covariates_warns(self):
        rng = np.random.default_rng(1)
        n = 300
        df = pd.DataFrame({"i": np.repeat(range(n), 2), "t": np.tile([0, 1], n)})
        df["treated"] = np.repeat(rng.integers(0, 2, n), 2)
        df["x"] = rng.normal(size=2 * n)
        df["y"] = rng.normal(size=2 * n) + 2 * df["treated"] * df["t"]
        with pytest.warns(AssumptionWarning, match="non-convex"):
            sp.did_2x2(df, y="y", treat="treated", time="t", covariates=["x"])

    def test_two_period_twfe_does_not_warn(self):
        """Only the >2-period collapse is the trap."""
        rng = np.random.default_rng(1)
        n = 300
        df = pd.DataFrame({"i": np.repeat(range(n), 2), "t": np.tile([0, 1], n)})
        df["treated"] = np.repeat(rng.integers(0, 2, n), 2)
        df["y"] = rng.normal(size=2 * n) + 2 * df["treated"] * df["t"]
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            sp.did(df, y="y", treat="treated", time="t", method="twfe")
        assert not [c for c in caught if "NOT the average" in str(c.message)]


# ----------------------------------------------------------------------
# 5. Unknown keyword arguments are rejected, not swallowed
# ----------------------------------------------------------------------


class TestUnknownKwargsRejected:
    @pytest.mark.parametrize(
        "kwargs",
        [{"post": "post"}, {"repeated_cs": True}, {"d": "treated"}, {"typo": 1}],
    )
    def test_stale_arguments_raise(self, kwargs):
        rng = np.random.default_rng(1)
        n = 120
        df = pd.DataFrame({"i": np.repeat(range(n), 2), "t": np.tile([0, 1], n)})
        df["treated"] = np.repeat(rng.integers(0, 2, n), 2)
        df["y"] = rng.normal(size=2 * n)
        with pytest.raises(MethodIncompatibility, match="unknown keyword"):
            sp.did(df, y="y", treat="treated", time="t", **kwargs)

    def test_bjs_still_accepts_its_own_options(self, panel):
        r = sp.did(
            panel,
            y="y",
            treat="g",
            time="t",
            id="i",
            method="bjs",
            event_window=(-2, 3),
        )
        assert np.isfinite(r.estimate)


# ----------------------------------------------------------------------
# 6. cs_report implements the eight steps
# ----------------------------------------------------------------------


class TestForwardEngineeringReport:
    @pytest.fixture(scope="class")
    def report(self):
        df = _balance_panel()
        return sp.cs_report(
            df,
            y="y",
            g="g",
            t="t",
            i="i",
            x=["x"],
            weights="w",
            n_boot=100,
            random_state=0,
            verbose=False,
        )

    def test_checklist_has_all_eight_steps(self, report):
        chk = report.forward_engineering_checklist()
        assert list(chk["step"]) == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_steps_one_to_six_are_executed(self, report):
        chk = report.forward_engineering_checklist()
        done = chk[chk["step"] <= 6]["status"].tolist()
        assert all(s in {"done", "not run"} for s in done)
        assert chk.loc[chk["step"] == 2, "status"].item() == "done"

    def test_balance_evidence_attached(self, report):
        assert report.balance is not None
        assert set(report.balance.table["panel"]) == {"levels", "changes"}

    def test_estimator_triangulation_runs_all_three(self, report):
        assert set(report.estimator_comparison["estimator"]) == {
            "reg",
            "ipw",
            "dr",
        }

    def test_functional_form_test_recorded(self, report):
        assert "pvalue" in report.functional_form
        assert 0.0 <= report.functional_form["pvalue"] <= 1.0

    def test_assumption_named_in_meta_and_text(self, report):
        assert report.meta["parallel_trends"]["label"] == "CPT-GT-NEV"
        txt = report.to_text()
        assert "IDENTIFYING ASSUMPTION" in txt
        assert "EVIDENCE ON THE ASSUMPTION" in txt
        assert "COVARIATE STRATEGY" in txt
        assert "Forward-engineering checklist" in txt

    def test_weighted_flag_propagates(self, report):
        assert report.meta["weighted"] is True

    def test_no_silent_degradation(self, report):
        """Any skipped step must be recorded, never dropped."""
        assert isinstance(report.degradations, list)
        for d in report.degradations:
            assert {"section", "error_type", "message"} <= set(d)

    def test_report_without_covariates_still_works(self):
        df = _balance_panel()
        r = sp.cs_report(
            df,
            y="y",
            g="g",
            t="t",
            i="i",
            n_boot=50,
            random_state=0,
            verbose=False,
        )
        chk = r.forward_engineering_checklist()
        assert chk.loc[chk["step"] == 3, "detail"].item().endswith("(no triangulation)")
