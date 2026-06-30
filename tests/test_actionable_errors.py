"""Actionable-failure behaviour for core estimators.

The contract this suite defends: when an identifying assumption is violated,
StatsPAI must *fail loud and actionable* — the fit-time signal and the
structured ``result.violations()`` API must agree, and the payload must carry a
``recovery_hint`` plus the ``sp.<alternative>`` an agent should try next. A
silent disagreement between "what the fit warns about" and "what ``.violations()``
reports" is exactly the kind of latent inconsistency that erodes trust, so each
family gets a weak/clean pair of cases here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


def _iv_data(first_stage_coef: float, n: int = 500, seed: int = 0) -> pd.DataFrame:
    """One endogenous regressor instrumented by ``z``; ``first_stage_coef``
    tunes instrument strength (small → weak first stage)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = first_stage_coef * z + u + rng.normal(size=n)
    y = 1.0 * d + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z})


# --------------------------------------------------------------------------- #
#  IV — weak instruments
# --------------------------------------------------------------------------- #


class TestWeakInstrument:
    def test_weak_instrument_emits_typed_actionable_warning(self):
        df = _iv_data(first_stage_coef=0.02)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sp.ivreg("y ~ (d ~ z)", data=df)
        typed = [
            w.message for w in caught if isinstance(w.message, sp.AssumptionWarning)
        ]
        assert typed, "weak first stage should emit a typed AssumptionWarning"
        msg = typed[0]
        # actionable payload an agent can branch on
        assert msg.recovery_hint
        assert "sp.anderson_rubin_ci" in msg.alternative_functions
        assert msg.diagnostics["first_stage_f"] < 10
        # AssumptionWarning is still a UserWarning — old catches keep working
        assert issubclass(sp.AssumptionWarning, UserWarning)

    def test_weak_instrument_surfaced_in_violations(self):
        """The fit-time warning and the structured API must agree."""
        df = _iv_data(first_stage_coef=0.02)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.ivreg("y ~ (d ~ z)", data=df)
        # machine-readable first-stage strength is recorded
        assert "first_stage_f" in res.model_info
        assert res.model_info["first_stage_f"] < 10
        viols = res.violations()
        weak = [v for v in viols if v.get("test") == "weak_instrument"]
        assert weak, "violations() must report the weak instrument the fit warned about"
        assert "sp.anderson_rubin_ci" in weak[0]["alternatives"]

    def test_strong_instrument_is_clean(self):
        """No false positives: a strong first stage warns nothing and reports
        no weak-instrument violation."""
        df = _iv_data(first_stage_coef=2.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sp.ivreg("y ~ (d ~ z)", data=df)
        typed = [w for w in caught if isinstance(w.message, sp.AssumptionWarning)]
        assert not typed
        assert res.model_info["first_stage_f"] > 10
        assert not [v for v in res.violations() if v.get("test") == "weak_instrument"]


# --------------------------------------------------------------------------- #
#  Panel — too few clusters for cluster-robust inference
# --------------------------------------------------------------------------- #


def _balanced_panel(n_units: int, n_periods: int = 8, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        alpha = rng.normal()
        for t in range(n_periods):
            x = rng.normal()
            rows.append({"id": i, "year": t, "x": x, "y": x + alpha + rng.normal()})
    return pd.DataFrame(rows)


class TestFewClusters:
    def test_few_clusters_warns_and_surfaces(self):
        df = _balanced_panel(n_units=12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sp.panel(
                df, "y ~ x", entity="id", time="year", method="fe", cluster="entity"
            )
        typed = [
            w.message for w in caught if isinstance(w.message, sp.AssumptionWarning)
        ]
        assert typed, "few clusters should emit a typed AssumptionWarning"
        assert "sp.wild_cluster_bootstrap" in typed[0].alternative_functions
        # fit-time warning and structured API agree
        assert res.model_info["n_clusters"] == 12
        few = [v for v in res.violations() if v.get("test") == "few_clusters"]
        assert few, "violations() must surface the few-cluster risk"
        assert "sp.wild_cluster_bootstrap" in few[0]["alternatives"]

    def test_many_clusters_is_clean(self):
        df = _balanced_panel(n_units=60)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sp.panel(
                df, "y ~ x", entity="id", time="year", method="fe", cluster="entity"
            )
        assert not [w for w in caught if isinstance(w.message, sp.AssumptionWarning)]
        assert res.model_info["n_clusters"] == 60
        assert not [v for v in res.violations() if v.get("test") == "few_clusters"]

    def test_no_clustering_records_no_cluster_count(self):
        """Without cluster-robust SEs there is no n_clusters and no warning."""
        df = _balanced_panel(n_units=12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = sp.panel(
                df, "y ~ x", entity="id", time="year", method="fe", robust="robust"
            )
        assert "n_clusters" not in res.model_info
        assert not [w for w in caught if isinstance(w.message, sp.AssumptionWarning)]


# --------------------------------------------------------------------------- #
#  Synthetic control — poor pre-treatment fit
# --------------------------------------------------------------------------- #


class TestSynthPreFit:
    def test_canonical_good_synth_is_clean(self):
        """California Prop-99 is the textbook *good* synth — it must NOT fire,
        or the warning would cry wolf on the field's canonical example."""
        r = sp.synth(
            sp.california_prop99(),
            unit="state",
            time="year",
            outcome="packspercapita",
            treated_unit="California",
            treatment_time=1989,
        )
        assert not [v for v in r.violations() if v.get("test") == "synth_prefit"]

    def test_unmatchable_treated_trend_is_flagged(self):
        """A treated unit whose pre-trend no donor can match → poor pre-fit,
        surfaced with concrete alternatives."""
        rng = np.random.default_rng(3)
        rows = []
        for u in ["T"] + [f"D{i}" for i in range(8)]:
            base = 50 if u == "T" else rng.uniform(10, 30)
            slope = 5.0 if u == "T" else rng.uniform(-1, 1)
            for yr in range(1980, 1995):
                rows.append(
                    {
                        "u": u,
                        "yr": yr,
                        "y": base + slope * (yr - 1980) + rng.normal(0, 1),
                    }
                )
        r = sp.synth(
            pd.DataFrame(rows),
            unit="u",
            time="yr",
            outcome="y",
            treated_unit="T",
            treatment_time=1990,
        )
        flagged = [v for v in r.violations() if v.get("test") == "synth_prefit"]
        assert flagged, "an unmatchable treated trend should flag poor pre-fit"
        assert flagged[0]["value"] > flagged[0]["threshold"]
        assert "sp.synth_compare" in flagged[0]["alternatives"]


# --------------------------------------------------------------------------- #
#  RD — density manipulation (McCrary) at the cutoff
# --------------------------------------------------------------------------- #


class TestRDManipulation:
    def _clean_rd(self, n: int = 3000, seed: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        y = 0.5 * x + 1.0 * (x >= 0) + rng.normal(0, 0.3, n)
        return pd.DataFrame({"y": y, "x": x})

    def _manipulated_rd(self, n: int = 3000, seed: int = 8) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, n)
        mask = (x > -0.15) & (x < 0)  # sort just-below units to just-above
        x[mask & (rng.uniform(size=n) < 0.7)] *= -1
        y = 0.5 * x + 1.0 * (x >= 0) + rng.normal(0, 0.3, n)
        return pd.DataFrame({"y": y, "x": x})

    def test_clean_rd_is_silent(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = sp.rdrobust(self._clean_rd(), y="y", x="x", c=0)
        assert not [w for w in caught if isinstance(w.message, sp.AssumptionWarning)]
        assert r.model_info["mccrary"]["pvalue"] > 0.05
        assert not [v for v in r.violations() if v.get("test") == "mccrary_density"]

    def test_manipulation_warns_and_surfaces(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = sp.rdrobust(self._manipulated_rd(), y="y", x="x", c=0)
        typed = [
            w.message for w in caught if isinstance(w.message, sp.AssumptionWarning)
        ]
        assert typed, "density manipulation should warn"
        assert "sp.rddensity" in typed[0].alternative_functions
        assert r.model_info["mccrary"]["pvalue"] < 0.05
        viols = [v for v in r.violations() if v.get("test") == "mccrary_density"]
        assert viols, "violations() must surface the manipulation the fit warned about"

    def test_opt_out_skips_the_test(self):
        r = sp.rdrobust(
            self._manipulated_rd(), y="y", x="x", c=0, manipulation_test=False
        )
        assert "mccrary" not in r.model_info


# --------------------------------------------------------------------------- #
#  DID — staggered design must not silently run a biased TWFE 2x2
# --------------------------------------------------------------------------- #


def _staggered_panel(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(60):
        g = [3, 5, 7, 0][i % 4]  # three treatment cohorts + never-treated (0)
        for t in range(1, 9):
            treated = 1 if (g > 0 and t >= g) else 0
            y = 1.0 + 0.2 * t + (2.0 * (t - g + 1) if treated else 0) + rng.normal()
            rows.append({"id": i, "year": t, "first": g, "y": y})
    return pd.DataFrame(rows)


def test_staggered_twfe_fails_loud_and_actionable():
    """Forcing the naive 2x2/TWFE estimator on a staggered (multi-cohort)
    design must raise a typed, agent-catchable error pointing to the
    heterogeneity-robust estimators — never silently return a biased number."""
    df = _staggered_panel()
    with pytest.raises(sp.MethodIncompatibility) as exc:
        sp.did(df, y="y", treat="first", time="year", id="id", method="twfe")
    err = exc.value
    assert isinstance(err, sp.StatsPAIError)
    assert "sp.callaway_santanna" in err.alternative_functions
    assert "callaway" in err.recovery_hint.lower()


def test_default_did_handles_staggered_without_error():
    """The default (heterogeneity-robust) path runs the same staggered design
    fine — the loud failure is specific to forcing the biased estimator."""
    df = _staggered_panel()
    res = sp.did(df, y="y", treat="first", time="year", id="id")
    assert res is not None


# --------------------------------------------------------------------------- #
#  Taxonomy — IdentificationError is catchable through the central taxonomy
# --------------------------------------------------------------------------- #


def test_identification_error_is_taxonomy_catchable():
    """``IdentificationError`` (strict identification check) must be catchable
    as a ``StatsPAIError`` / ``IdentificationFailure`` and carry the standard
    recovery payload, so agents can handle it by kind."""
    assert issubclass(sp.IdentificationError, sp.IdentificationFailure)
    assert issubclass(sp.IdentificationError, sp.StatsPAIError)
    rep = sp.IdentificationReport(
        findings=[
            sp.DiagnosticFinding(
                severity="blocker",
                category="variation",
                message="No variation in treatment.",
            )
        ],
        design="cross_section",
        n_obs=30,
    )
    err = sp.IdentificationError(rep)
    assert err.recovery_hint
    assert err.diagnostics["design"] == "cross_section"
    assert err.report.verdict == "BLOCKERS"


# --------------------------------------------------------------------------- #
#  Inference — few cluster-robust clusters
# --------------------------------------------------------------------------- #


def test_few_clusters_surfaces_wild_bootstrap_recovery():
    """Cluster-robust inference with small G should surface an actionable
    warning even when the estimator family itself is otherwise generic."""
    res = CausalResult(
        method="regress",
        estimand="ATE",
        estimate=1.0,
        se=0.2,
        pvalue=0.01,
        ci=(0.6, 1.4),
        alpha=0.05,
        n_obs=120,
        model_info={"n_clusters": 8},
        _citation_key="regress",
    )
    few = [v for v in res.violations() if v.get("test") == "few_clusters"]
    assert few, "violations() must report too few cluster-robust clusters"
    assert few[0]["value"] == 8
    assert "sp.wild_cluster_bootstrap" in few[0]["alternatives"]
    assert few[0]["recovery_hint"]


def test_many_clusters_do_not_trigger_few_cluster_violation():
    res = CausalResult(
        method="regress",
        estimand="ATE",
        estimate=1.0,
        se=0.2,
        pvalue=0.01,
        ci=(0.6, 1.4),
        alpha=0.05,
        n_obs=600,
        model_info={"n_clusters": 30},
        _citation_key="regress",
    )
    assert not [v for v in res.violations() if v.get("test") == "few_clusters"]
