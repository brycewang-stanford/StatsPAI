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
#  Matching — residual covariate imbalance after matching
# --------------------------------------------------------------------------- #


class TestMatchingBalance:
    def _data(self, confounded: bool, n: int = 600, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        ps = 1 / (1 + np.exp(-(0.8 * x1 + 0.5 * x2))) if confounded else np.full(n, 0.5)
        d = (rng.uniform(size=n) < ps).astype(int)
        y = 1.0 + 2.0 * d + x1 + x2 + rng.normal(size=n)
        return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})

    def test_poor_balance_is_flagged(self):
        r = sp.match(
            self._data(confounded=True),
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
        )
        viols = [v for v in r.violations() if v.get("test") == "balance"]
        assert viols, "notable residual imbalance after matching should be flagged"
        assert viols[0]["value"] > viols[0]["threshold"]
        assert "sp.ebalance" in viols[0]["alternatives"]

    def test_good_balance_is_clean(self):
        """Already-balanced (randomised) data must not cry wolf."""
        r = sp.match(
            self._data(confounded=False),
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
        )
        assert not [v for v in r.violations() if v.get("test") == "balance"]


def test_rdrobust_manipulation_test_param_is_discoverable():
    """The McCrary opt-out must be visible to agents via the registry."""
    params = [
        p["name"] if isinstance(p, dict) else getattr(p, "name", None)
        for p in sp.describe_function("rdrobust")["params"]
    ]
    assert "manipulation_test" in params


# --------------------------------------------------------------------------- #
#  Regression (OLS) — too few clusters for cluster-robust inference
# --------------------------------------------------------------------------- #


def _clustered_ols(n_clusters: int, per: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_clusters):
        alpha = rng.normal()
        for _ in range(per):
            x = rng.normal()
            rows.append({"g": g, "x": x, "y": 1 + 0.5 * x + alpha + rng.normal()})
    return pd.DataFrame(rows)


class TestRegressFewClusters:
    def test_few_clusters_warns_and_surfaces(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = sp.regress("y ~ x", data=_clustered_ols(12), cluster="g")
        typed = [
            w.message for w in caught if isinstance(w.message, sp.AssumptionWarning)
        ]
        assert typed, "OLS with few clusters should warn"
        assert "sp.wild_cluster_bootstrap" in typed[0].alternative_functions
        assert r.model_info["n_clusters"] == 12
        few = [v for v in r.violations() if v.get("test") == "few_clusters"]
        assert few and "sp.wild_cluster_bootstrap" in few[0]["alternatives"]

    def test_many_clusters_is_clean(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = sp.regress("y ~ x", data=_clustered_ols(50), cluster="g")
        assert not [w for w in caught if isinstance(w.message, sp.AssumptionWarning)]
        assert r.model_info["n_clusters"] == 50
        assert not [v for v in r.violations() if v.get("test") == "few_clusters"]

    def test_no_cluster_is_silent(self):
        r = sp.regress("y ~ x", data=_clustered_ols(12))
        assert "n_clusters" not in r.model_info


# --------------------------------------------------------------------------- #
#  DML / AIPW — propensity overlap
# --------------------------------------------------------------------------- #


class TestDMLOverlap:
    def _dml(self, strength: float, n: int = 1500, seed: int = 0):
        rng = np.random.default_rng(seed)
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        ps = 1 / (1 + np.exp(-(strength * x1 + 0.6 * strength * x2)))
        d = (rng.uniform(size=n) < ps).astype(int)
        y = 1.0 + 2.0 * d + x1 + x2 + rng.normal(size=n)
        return sp.dml(
            pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2}),
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            model="irm",
        )

    def test_poor_overlap_is_flagged(self):
        viols = [
            v for v in self._dml(4.0).violations() if v.get("test") == "dml_overlap"
        ]
        assert viols and viols[0]["value"] > viols[0]["threshold"]
        assert "sp.trimming" in viols[0]["alternatives"]

    def test_good_overlap_is_clean(self):
        assert not [
            v for v in self._dml(0.6).violations() if v.get("test") == "dml_overlap"
        ]


# --------------------------------------------------------------------------- #
#  GLM — logit separation & Poisson over-dispersion
# --------------------------------------------------------------------------- #


class TestGLMDiagnostics:
    def test_logit_separation_is_flagged(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=300)
        r = sp.logit("y ~ x", data=pd.DataFrame({"y": (x > 0.3).astype(int), "x": x}))
        viols = [v for v in r.violations() if v.get("test") == "separation"]
        assert viols and viols[0]["severity"] == "error"

    def test_clean_logit_is_not_flagged(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=400)
        p = 1 / (1 + np.exp(-(0.5 + 1.0 * x)))
        y = (rng.uniform(size=400) < p).astype(int)
        r = sp.logit("y ~ x", data=pd.DataFrame({"y": y, "x": x}))
        assert not [v for v in r.violations() if v.get("test") == "separation"]

    def test_poisson_overdispersion_is_flagged(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=300)
        lam = np.exp(0.5 + 0.8 * x)
        y = rng.poisson(lam * rng.gamma(1.0, 1.0, 300))
        r = sp.poisson("y ~ x", data=pd.DataFrame({"y": y, "x": x}))
        viols = [v for v in r.violations() if v.get("test") == "overdispersion"]
        assert viols and viols[0]["value"] > viols[0]["threshold"]
        assert "sp.nbreg" in viols[0]["alternatives"]

    def test_clean_poisson_is_not_flagged(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=400)
        y = rng.poisson(np.exp(0.5 + 0.8 * x))
        r = sp.poisson("y ~ x", data=pd.DataFrame({"y": y, "x": x}))
        assert not [v for v in r.violations() if v.get("test") == "overdispersion"]


# --------------------------------------------------------------------------- #
#  Count models — excess zeros (zero inflation)
# --------------------------------------------------------------------------- #


class TestCountExcessZeros:
    def test_zero_inflated_is_flagged(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=600)
        y = rng.poisson(np.exp(0.5 + 0.8 * x))
        y[rng.uniform(size=600) < 0.4] = 0  # structural zeros
        r = sp.poisson("y ~ x", data=pd.DataFrame({"y": y, "x": x}))
        viols = [v for v in r.violations() if v.get("test") == "excess_zeros"]
        assert viols and "sp.zip_model" in viols[0]["alternatives"]

    def test_clean_poisson_has_no_excess_zeros(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=600)
        r = sp.poisson(
            "y ~ x",
            data=pd.DataFrame({"y": rng.poisson(np.exp(0.5 + 0.8 * x)), "x": x}),
        )
        assert not [v for v in r.violations() if v.get("test") == "excess_zeros"]


# --------------------------------------------------------------------------- #
#  Selection models — Heckman & Tobit
# --------------------------------------------------------------------------- #


class TestSelectionModels:
    def test_heckman_insignificant_selection_suggests_ols(self):
        rng = np.random.default_rng(2)
        n = 800
        u = rng.normal(size=n)  # corr(u, selection error) = 0 → no selection
        x, z = rng.normal(size=n), rng.normal(size=n)
        sel = (0.3 + x + z + u) > 0
        y = np.where(sel, 1 + 2 * x + rng.normal(size=n), np.nan)
        r = sp.heckman(
            pd.DataFrame({"yh": y, "sel": sel.astype(int), "x": x, "z": z}),
            y="yh",
            x=["x"],
            select="sel",
            z=["x", "z"],
        )
        viols = [v for v in r.violations() if v.get("test") == "heckman_no_selection"]
        assert viols and "sp.regress" in viols[0]["alternatives"]

    def test_tobit_extreme_censoring_is_flagged(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=600)
        y = np.maximum(0, -3 + 2 * x + rng.normal(size=600))  # ~90% censored at 0
        r = sp.tobit(pd.DataFrame({"y": y, "x": x}), y="y", x=["x"], ll=0)
        assert r.model_info["censor_pct"] > 90
        assert [v for v in r.violations() if v.get("test") == "extreme_censoring"]


# --------------------------------------------------------------------------- #
#  Survival — Cox proportional hazards
# --------------------------------------------------------------------------- #


class TestCoxPH:
    def test_ph_test_stored_and_clean_fit_is_silent(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        df = pd.DataFrame(
            {
                "t": rng.exponential(np.exp(-0.5 * x)),
                "x": x,
                "event": (rng.uniform(size=400) < 0.85).astype(int),
            }
        )
        r = sp.cox("t ~ x", data=df, event="event")
        assert "min_pvalue" in r.model_info["ph_test"]
        assert not [
            v for v in r.violations() if v.get("test") == "proportional_hazards"
        ]

    def test_ph_violation_is_surfaced(self):
        """A stored PH-test rejection must surface as a violation (the ph_test
        computation itself is exercised by the survival suite)."""
        from types import SimpleNamespace

        from statspai.core._agent_summary import econometric_violations

        stub = SimpleNamespace(
            model_info={
                "model_type": "cox proportional hazards",
                "ph_test": {"min_pvalue": 0.001, "worst_variable": "treat"},
            },
            diagnostics={},
            std_errors=None,
            params=None,
            data_info={},
        )
        viols = [
            v
            for v in econometric_violations(stub)
            if v.get("test") == "proportional_hazards"
        ]
        assert viols and "sp.aft" in viols[0]["alternatives"]


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


# --------------------------------------------------------------------------- #
#  sp.audit() is a superset of result.violations()
# --------------------------------------------------------------------------- #


def test_audit_folds_in_violations():
    """Every live violation must appear in sp.audit()'s checklist (single
    source of truth), with no name double-counted."""
    rng = np.random.default_rng(0)
    n = 1500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(4.0 * x1 + 2.4 * x2)))
    d = (rng.uniform(size=n) < ps).astype(int)
    r = sp.dml(
        pd.DataFrame(
            {"y": 1 + 2 * d + x1 + x2 + rng.normal(size=n), "d": d, "x1": x1, "x2": x2}
        ),
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        model="irm",
    )
    audit = sp.audit(r)
    names = [c["name"] for c in audit["checks"]]
    assert len(names) == len(set(names)), "audit checklist must not double-count"
    viol_names = {v["test"] for v in r.violations()}
    alias = {
        "rhat": "convergence_rhat",
        "balance": "balance_after",
        "pretrend": "parallel_trends",
        "synth_prefit": "pretreatment_fit",
    }
    for vn in viol_names:
        assert vn in names or alias.get(vn) in names, f"{vn} missing from audit"
    assert "dml_overlap" in names
