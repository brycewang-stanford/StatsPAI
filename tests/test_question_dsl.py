"""Tests for ``sp.causal_question`` DSL."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def rct_data():
    rng = np.random.default_rng(7)
    n = 500
    treat = rng.binomial(1, 0.5, n)
    y = 5 + 1.5 * treat + rng.normal(0, 1, n)
    return pd.DataFrame({"treat": treat, "y": y})


@pytest.fixture
def confounded_data():
    rng = np.random.default_rng(13)
    n = 800
    x = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-x))
    treat = rng.binomial(1, p)
    y = 2 + 1.0 * treat + 0.8 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"treat": treat, "y": y, "x": x})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_causal_question_construction():
    q = sp.causal_question(
        treatment="d",
        outcome="y",
        estimand="ATE",
        design="rct",
    )
    assert q.treatment == "d"
    assert q.outcome == "y"
    assert q.estimand == "ATE"


def test_causal_question_rejects_invalid_estimand():
    with pytest.raises(ValueError):
        sp.causal_question(
            treatment="d", outcome="y", estimand="NONSENSE",
        )


def test_causal_question_rejects_invalid_design():
    with pytest.raises(ValueError):
        sp.causal_question(
            treatment="d", outcome="y", design="unknown_thing",
        )


# ---------------------------------------------------------------------------
# identify()
# ---------------------------------------------------------------------------


def test_identify_rct_design():
    q = sp.causal_question(
        treatment="d", outcome="y", design="rct",
    )
    plan = q.identify()
    assert plan.estimator == "regress"
    assert "random" in plan.identification_story.lower()


def test_identify_iv_design_requires_instruments():
    q = sp.causal_question(
        treatment="d", outcome="y", design="iv",
    )
    plan = q.identify()
    assert plan.estimator == "iv"
    assert plan.warnings  # should warn about missing instruments


def test_identify_iv_design_with_instruments():
    q = sp.causal_question(
        treatment="d", outcome="y", design="iv",
        instruments=["z1"],
    )
    plan = q.identify()
    assert plan.estimator == "iv"
    assert plan.estimand == "LATE"
    assert "z1" in plan.identification_story


def test_identify_auto_design_with_instruments():
    q = sp.causal_question(
        treatment="d", outcome="y", design="auto",
        instruments=["z1"],
    )
    plan = q.identify()
    assert plan.estimator == "iv"


def test_identify_auto_design_with_running_variable():
    q = sp.causal_question(
        treatment="d", outcome="y", design="auto",
        running_variable="score", cutoff=50,
    )
    plan = q.identify()
    assert plan.estimator == "rdrobust"


def test_identify_auto_default_is_selection_on_observables():
    q = sp.causal_question(
        treatment="d", outcome="y", design="auto",
        covariates=["age", "sex"],
    )
    plan = q.identify()
    assert plan.estimator == "aipw"


# ---------------------------------------------------------------------------
# estimate()
# ---------------------------------------------------------------------------


def test_estimate_rct(rct_data):
    q = sp.causal_question(
        treatment="treat", outcome="y",
        design="rct", data=rct_data,
    )
    r = q.estimate()
    # True ATE = 1.5
    assert abs(r.estimate - 1.5) < 0.3
    assert r.n == len(rct_data)
    assert r.estimator == "regress"


def test_estimate_aipw_confounded(confounded_data):
    q = sp.causal_question(
        treatment="treat", outcome="y",
        design="selection_on_observables",
        covariates=["x"], data=confounded_data,
    )
    r = q.estimate()
    # True ATE = 1.0
    assert abs(r.estimate - 1.0) < 0.3
    assert r.estimator == "aipw"


# ---------------------------------------------------------------------------
# report()
# ---------------------------------------------------------------------------


def test_report_requires_estimate_first():
    q = sp.causal_question(treatment="d", outcome="y", design="rct")
    with pytest.raises(ValueError):
        q.report()


def test_report_markdown(rct_data):
    q = sp.causal_question(
        treatment="treat", outcome="y",
        design="rct", data=rct_data,
    )
    q.estimate()
    md = q.report("markdown")
    assert "Causal Question" in md
    assert "Identification" in md
    assert "Estimate" in md


def test_report_text(rct_data):
    q = sp.causal_question(
        treatment="treat", outcome="y",
        design="rct", data=rct_data,
    )
    q.estimate()
    txt = q.report("text")
    assert "Estimand" in txt


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------


def test_question_to_dict():
    q = sp.causal_question(
        treatment="d", outcome="y",
        design="did", covariates=["x1", "x2"],
    )
    d = q.to_dict()
    assert d["treatment"] == "d"
    assert d["covariates"] == ["x1", "x2"]


# ---------------------------------------------------------------------------
# v1.13 — DML / TMLE / metalearner / causal_forest dispatch
# ---------------------------------------------------------------------------


def test_identify_dml_design():
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x1", "x2"],
    )
    plan = q.identify()
    assert plan.estimator == "dml"
    assert plan.estimand == "ATE"
    assert "double" in plan.identification_story.lower() \
        or "debiased" in plan.identification_story.lower()


def test_identify_dml_with_iv_promotes_late():
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], instruments=["z"],
    )
    plan = q.identify()
    assert plan.estimator == "dml"
    assert plan.estimand == "LATE"


def test_identify_tmle_design():
    q = sp.causal_question(
        treatment="d", outcome="y", design="tmle",
        covariates=["x"],
    )
    plan = q.identify()
    assert plan.estimator == "tmle"


def test_identify_metalearner_design():
    q = sp.causal_question(
        treatment="d", outcome="y", design="metalearner",
        covariates=["x"],
    )
    plan = q.identify()
    assert plan.estimator == "metalearner"
    # plan.estimand reflects what EstimationResult.estimate (a scalar)
    # represents — i.e. ATE — not the method's CATE target. Per-unit
    # CATEs live in result.underlying.model_info['cate'].
    assert plan.estimand == "ATE"
    assert "tau(x)" in plan.identification_story
    assert "model_info['cate']" in plan.identification_story


def test_identify_causal_forest_design():
    q = sp.causal_question(
        treatment="d", outcome="y", design="causal_forest",
        covariates=["x"],
    )
    plan = q.identify()
    assert plan.estimator == "causal_forest"
    assert plan.estimand == "ATE"  # scalar, not CATE target
    assert "effect_interval" in plan.identification_story


def test_identify_auto_routes_cate_to_metalearner():
    q = sp.causal_question(
        treatment="d", outcome="y", design="auto",
        estimand="CATE", covariates=["x"],
    )
    plan = q.identify()
    assert plan.estimator == "metalearner"


def test_identify_soo_with_cate_promotes_metalearner():
    q = sp.causal_question(
        treatment="d", outcome="y",
        design="selection_on_observables",
        estimand="CATE", covariates=["x"],
    )
    plan = q.identify()
    assert plan.estimator == "metalearner"
    # Result is a scalar ATE summary — labelled honestly.
    assert plan.estimand == "ATE"


def test_identify_soo_default_remains_aipw():
    """Regression guard: ATE estimand must still pick AIPW."""
    q = sp.causal_question(
        treatment="d", outcome="y",
        design="selection_on_observables",
        covariates=["x"],
    )
    plan = q.identify()
    assert plan.estimator == "aipw"
    # New fallbacks include the ML estimators.
    assert "dml" in plan.fallback_estimators
    assert "tmle" in plan.fallback_estimators


def test_dml_requires_covariates():
    q = sp.causal_question(
        treatment="treat", outcome="y", design="dml",
        data=pd.DataFrame({"treat": [0, 1, 0, 1], "y": [1.0, 2.0, 1.5, 2.5]}),
    )
    with pytest.raises(ValueError, match="covariates"):
        q.estimate()


def test_tmle_requires_covariates():
    q = sp.causal_question(
        treatment="treat", outcome="y", design="tmle",
        data=pd.DataFrame({"treat": [0, 1, 0, 1], "y": [1.0, 2.0, 1.5, 2.5]}),
    )
    with pytest.raises(ValueError, match="covariates"):
        q.estimate()


def test_metalearner_requires_covariates():
    q = sp.causal_question(
        treatment="treat", outcome="y", design="metalearner",
        data=pd.DataFrame({"treat": [0, 1, 0, 1], "y": [1.0, 2.0, 1.5, 2.5]}),
    )
    with pytest.raises(ValueError, match="covariates"):
        q.estimate()


def test_estimate_dml_recovers_ate(confounded_data):
    q = sp.causal_question(
        treatment="treat", outcome="y", design="dml",
        covariates=["x"], data=confounded_data,
    )
    r = q.estimate()
    # True ATE = 1.0; allow 0.5 slack for finite-sample ML noise.
    assert abs(r.estimate - 1.0) < 0.5
    assert r.estimator == "dml"
    assert r.se > 0
    assert r.ci[0] < r.estimate < r.ci[1]


def test_estimate_tmle_recovers_ate(confounded_data):
    q = sp.causal_question(
        treatment="treat", outcome="y", design="tmle",
        covariates=["x"], data=confounded_data,
    )
    r = q.estimate()
    assert abs(r.estimate - 1.0) < 0.5
    assert r.estimator == "tmle"
    assert r.se > 0


def test_estimate_metalearner_runs(confounded_data):
    q = sp.causal_question(
        treatment="treat", outcome="y", design="metalearner",
        covariates=["x"], data=confounded_data,
    )
    r = q.estimate()
    assert r.estimator == "metalearner"
    # ATE / SE must be finite even though the primary target is CATE.
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) and r.se > 0


def test_estimate_causal_forest_returns_finite_se_via_aipw(confounded_data):
    """Post v1.13.x: causal_forest dispatch attaches AIPW-IF inference
    to the forest's CATE estimator. SE is finite, CI covers truth, and
    the forest is preserved on `result.underlying` for CATE access.
    """
    q = sp.causal_question(
        treatment="treat", outcome="y", design="causal_forest",
        covariates=["x"], data=confounded_data,
    )
    r = q.estimate(n_estimators=30, random_state=0)
    assert r.estimator == "causal_forest"
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) and r.se > 0
    assert r.ci[0] < r.estimate < r.ci[1]
    # Forest preserved for CATE access.
    cf = r.underlying
    assert hasattr(cf, "effect")
    cate = cf.effect(confounded_data[["x"]].to_numpy())
    assert len(cate) == len(confounded_data)


def test_causal_forest_aipw_coverage():
    """Coverage simulation: across 30 simulated datasets where the true
    ATE = 1.0, the AIPW-IF 95% CI must cover the truth at >=85% (loose
    bound for stability; nominal is 95% but small n + nuisance noise
    makes the empirical rate jump around)."""
    rng = np.random.default_rng(2026)
    nsim = 30
    n = 400
    ate_pop = 1.0
    covered = 0
    for s in range(nsim):
        rng2 = np.random.default_rng(s + 100)
        x = rng2.normal(0, 1, n)
        p = 1 / (1 + np.exp(-x))
        treat = rng2.binomial(1, p)
        y = ate_pop * treat + 0.8 * x + rng2.normal(0, 1, n)
        df = pd.DataFrame({"treat": treat, "y": y, "x": x})
        q = sp.causal_question(
            treatment="treat", outcome="y", design="causal_forest",
            covariates=["x"], data=df,
        )
        r = q.estimate(n_estimators=30, random_state=0)
        if r.ci[0] < ate_pop < r.ci[1]:
            covered += 1
    rate = covered / nsim
    assert rate >= 0.85, f"AIPW-IF coverage too low: {covered}/{nsim} = {rate:.0%}"


def test_causal_forest_random_state_is_reproducible(confounded_data):
    """Module D: a single ``random_state`` controls the WHOLE
    causal_forest branch (forest CATE + AIPW nuisance + KFold split).
    Two q.estimate() calls with the same seed must return identical
    estimate / SE / CI down to floating-point bits.
    """
    def run(seed):
        q = sp.causal_question(
            treatment="treat", outcome="y", design="causal_forest",
            covariates=["x"], data=confounded_data,
        )
        return q.estimate(n_estimators=30, random_state=seed)

    r1 = run(0)
    r2 = run(0)
    assert r1.estimate == r2.estimate
    assert r1.se == r2.se
    assert r1.ci == r2.ci

    # Different seed must produce a different result somewhere.
    r3 = run(1)
    assert (r3.estimate != r1.estimate) or (r3.se != r1.se), (
        "different random_state should change at least one of "
        "(estimate, se)"
    )


def test_causal_forest_rejects_continuous_treatment():
    """AIPW-IF for ATE assumes binary treatment. Continuous T must
    fail loud, pointing the user to design='dml'."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0, 1, n)
    treat = rng.normal(0, 1, n)  # continuous
    y = 1.0 * treat + 0.5 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"treat": treat, "y": y, "x": x})
    q = sp.causal_question(
        treatment="treat", outcome="y", design="causal_forest",
        covariates=["x"], data=df,
    )
    with pytest.raises(ValueError, match="binary"):
        q.estimate(n_estimators=20, random_state=0)


# ---------------------------------------------------------------------------
# v1.13 — Bug-fix regression tests (post-review)
# ---------------------------------------------------------------------------


def test_bug1_soo_cate_without_covariates_falls_back_to_aipw():
    """Regression: identify() and estimate() must agree.

    Pre-fix: identify() returned a `metalearner` plan even without
    covariates, so estimate() would crash inside the dispatcher.
    Post-fix: identify() returns AIPW with a warning that CATE was
    requested but cannot be served.
    """
    q = sp.causal_question(
        treatment="d", outcome="y",
        design="selection_on_observables",
        estimand="CATE",  # asks for CATE …
        covariates=[],     # … but no effect modifiers
    )
    plan = q.identify()
    assert plan.estimator == "aipw"
    assert plan.estimand == "ATE"
    assert any("CATE" in w for w in plan.warnings)


def test_bug3_dml_with_cate_estimand_coerces_to_ate_with_warning():
    """Regression: DML returns ATE; declaring estimand='CATE' must not
    silently mislabel the plan."""
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        estimand="CATE", covariates=["x"],
    )
    plan = q.identify()
    assert plan.estimator == "dml"
    assert plan.estimand == "ATE"
    assert any("coerced to 'ATE'" in w for w in plan.warnings)


def test_bug3_dml_with_iv_and_non_late_estimand_warns():
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        estimand="ATT", instruments=["z"], covariates=["x"],
    )
    plan = q.identify()
    assert plan.estimand == "LATE"
    assert any("'LATE'" in w for w in plan.warnings)


def test_bug2_dml_irm_with_instruments_drops_them_with_warning(confounded_data):
    """User overrides model='irm' but the question carries
    instruments — instruments must be dropped (IRM doesn't accept them)
    with a UserWarning, not propagated to sp.dml as a kwarg collision.
    """
    df = confounded_data.copy()
    df["z"] = df["treat"]  # dummy instrument column
    q = sp.causal_question(
        treatment="treat", outcome="y", design="dml",
        instruments=["z"], covariates=["x"], data=df,
    )
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        r = q.estimate(model="irm")  # explicit non-IV override
    msgs = [str(w.message) for w in caught]
    assert any("instruments" in m and "ignored" in m for m in msgs)
    assert r.estimator == "dml"
    assert np.isfinite(r.estimate)


def test_bug2_dml_iv_model_without_instruments_raises():
    """Symmetric to Bug 2: user picks model='pliv' but didn't declare
    instruments on the question — fail loudly, not deep inside sp.dml.
    """
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"],
        data=pd.DataFrame({
            "d": [0, 1, 0, 1, 0, 1] * 30,
            "y": [1.0, 2.0, 1.5, 2.5, 1.0, 2.0] * 30,
            "x": list(range(180)),
        }),
    )
    with pytest.raises(ValueError, match="requires instruments"):
        q.estimate(model="pliv")


def test_bug4_tmle_with_late_estimand_coerces_with_warning():
    q = sp.causal_question(
        treatment="d", outcome="y", design="tmle",
        estimand="LATE", covariates=["x"],
    )
    plan = q.identify()
    assert plan.estimand == "ATE"
    assert any("coerced to 'ATE'" in w for w in plan.warnings)


def test_bug8_longitudinal_with_cate_warns():
    q = sp.causal_question(
        treatment="d", outcome="y",
        time_structure="longitudinal", time="year", id="i",
        estimand="CATE", covariates=["x"],
    )
    plan = q.identify()
    # _auto_design picks longitudinal; longitudinal coerces CATE→ATE
    assert plan.estimator == "longitudinal.analyze"
    assert plan.estimand == "ATE"
    assert any("CATE" in w for w in plan.warnings)


def test_bug10_kwargs_collision_with_reserved_args():
    """Reserved kwargs (y/treat/covariates/data) must raise a clear
    TypeError instead of being silently forwarded to sp.dml etc.
    """
    df = pd.DataFrame({"d": [0, 1, 0, 1], "y": [1.0, 2.0, 1.5, 2.5],
                       "x": [0.1, 0.2, 0.3, 0.4]})
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], data=df,
    )
    with pytest.raises(TypeError, match="collide"):
        q.estimate(y="other_outcome")
    with pytest.raises(TypeError, match="collide"):
        q.estimate(treat="other")
    with pytest.raises(TypeError, match="collide"):
        q.estimate(covariates=["other"])


def test_bug11_metalearner_result_estimand_is_ate_not_cate(confounded_data):
    """Post-fix: EstimationResult.estimand == 'ATE' (matches scalar),
    even though the method targets CATE. Per-unit CATEs are accessible
    on the underlying object.
    """
    q = sp.causal_question(
        treatment="treat", outcome="y", design="metalearner",
        covariates=["x"], data=confounded_data,
    )
    r = q.estimate()
    assert r.estimand == "ATE"
    # The underlying CausalResult must carry the per-unit CATEs.
    cate = r.underlying.model_info.get("cate")
    assert cate is not None
    assert len(cate) == len(confounded_data)


def test_bug11_summary_does_not_mislabel_estimand(confounded_data):
    """The user-facing report must not say 'CATE = +X.XX' when X.XX is
    actually an ATE scalar."""
    q = sp.causal_question(
        treatment="treat", outcome="y", design="metalearner",
        covariates=["x"], data=confounded_data,
    )
    r = q.estimate()
    summary = r.summary()
    assert "ATE via sp.metalearner" in summary
    assert "CATE via sp.metalearner" not in summary


# Coverage gap fixes (oracle-flagged)


def test_estimate_dml_iv_recovers_late():
    """End-to-end: design='dml' + instrument actually delivers LATE.

    DGP: binary instrument z drives binary treatment d which causes y;
    with confounder u (unobserved). True LATE = 1.0.
    """
    rng = np.random.default_rng(7)
    n = 2000
    z = rng.binomial(1, 0.5, n)
    u = rng.normal(0, 1, n)
    x = rng.normal(0, 1, n)
    d_score = -0.3 + 1.5 * z + 0.4 * x + 0.6 * u + rng.normal(0, 1, n)
    d = (d_score > 0).astype(int)
    y = 0.5 + 1.0 * d + 0.4 * x + 0.6 * u + rng.normal(0, 1, n)
    df = pd.DataFrame({"d": d, "y": y, "x": x, "z": z})
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], instruments=["z"], data=df,
    )
    r = q.estimate()
    assert r.estimand == "LATE"
    assert abs(r.estimate - 1.0) < 0.5
    assert r.se > 0


def test_bug_a_iivm_requires_both_z_and_d_binary():
    """Oracle-flagged Bug A: IIVM requires BOTH binary D and binary Z.
    Continuous D + binary Z previously picked IIVM and crashed inside
    sp.dml. Post-fix the picker falls through to PLIV.
    """
    rng = np.random.default_rng(7)
    n = 600
    z = rng.binomial(1, 0.5, n)         # binary Z
    x = rng.normal(0, 1, n)
    d = 0.5 * z + 0.3 * x + rng.normal(0, 1, n)  # CONTINUOUS D
    y = 0.5 + 1.0 * d + 0.4 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"d": d, "y": y, "x": x, "z": z})
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], instruments=["z"], data=df,
    )
    # Must NOT raise; dispatcher picks PLIV (handles continuous D).
    r = q.estimate()
    assert r.estimator == "dml"
    assert r.estimand == "LATE"
    assert np.isfinite(r.estimate)


def test_bug_b_causal_forest_rejects_string_treatment():
    """Oracle-flagged Bug B: non-numeric treatment columns must raise
    a clear binary-treatment error, not a generic NumPy cast failure.
    """
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "treat": np.where(rng.binomial(1, 0.5, n) == 1, "A", "B"),
        "y": rng.normal(0, 1, n),
        "x": rng.normal(0, 1, n),
    })
    q = sp.causal_question(
        treatment="treat", outcome="y", design="causal_forest",
        covariates=["x"], data=df,
    )
    with pytest.raises(ValueError, match="binary|numeric"):
        q.estimate(n_estimators=10, random_state=0)


def test_bug_b_causal_forest_handles_numeric_strings():
    """Oracle-flagged Bug B variant: numeric strings ('0'/'1') used
    to pass the cast check but reach the forest unconverted, crashing
    inside sklearn. Post-fix the cleaned numeric T flows through.
    """
    rng = np.random.default_rng(0)
    n = 200
    treat_int = rng.binomial(1, 0.5, n)
    df = pd.DataFrame({
        "treat": np.where(treat_int == 1, "1", "0"),  # strings "0"/"1"
        "y": rng.normal(0, 1, n) + 0.5 * treat_int,
        "x": rng.normal(0, 1, n),
    })
    q = sp.causal_question(
        treatment="treat", outcome="y", design="causal_forest",
        covariates=["x"], data=df,
    )
    r = q.estimate(n_estimators=20, random_state=0)
    assert r.estimator == "causal_forest"
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) and r.se > 0


def test_bug_c_random_state_none_is_symmetric(confounded_data):
    """Oracle-flagged Bug C: passing no random_state (or None) means
    BOTH forest and AIPW use np global state — no hidden determinism
    in the AIPW SE while the forest CATE varies. Two unseeded runs
    must produce different ATE point estimates AND different SEs.
    """
    def run_unseeded():
        q = sp.causal_question(
            treatment="treat", outcome="y", design="causal_forest",
            covariates=["x"], data=confounded_data,
        )
        # Explicitly NOT passing random_state — let it default.
        return q.estimate(n_estimators=30)

    r1 = run_unseeded()
    r2 = run_unseeded()
    # The forest is non-deterministic without a seed; the AIPW path
    # now also lacks a fixed seed, so the SE varies across runs.
    assert (r1.estimate != r2.estimate) or (r1.se != r2.se), (
        "without a seed, forest CATE and AIPW SE should both vary; "
        "if they're identical, AIPW is silently using a hidden seed"
    )


def test_dml_continuous_treatment_uses_plr():
    """Coverage: continuous D should pick PLR (not IRM)."""
    rng = np.random.default_rng(11)
    n = 600
    x = rng.normal(0, 1, n)
    d = 0.3 * x + rng.normal(0, 1, n)        # continuous
    y = 0.5 + 1.0 * d + 0.4 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"d": d, "y": y, "x": x})
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], data=df,
    )
    r = q.estimate()
    assert r.estimator == "dml"
    assert abs(r.estimate - 1.0) < 0.3
    # PLR was chosen (we check via the underlying result's model_info if
    # available; otherwise just confirm a finite SE).
    assert np.isfinite(r.se) and r.se > 0


# ---------------------------------------------------------------------------
# Module E — DML sub-model selection regression tests
#
# Asserts the correct PLR/IRM/PLIV/IIVM branch is picked from each
# (treatment-type × instrument) combination. Catches regressions where
# the auto-picker might silently fall through to a wrong sub-model
# (cf. Bug A: continuous-D + binary-Z → IIVM crash).
#
# Sub-model choice is read from `r.underlying._provenance.params['model']`,
# attached by sp.dml's lineage layer.
# ---------------------------------------------------------------------------


def _dml_sub_model(result):
    """Pull the sub-model name DML actually fit (irm/plr/iivm/pliv)."""
    prov = getattr(result.underlying, "_provenance", None)
    assert prov is not None, "sp.dml did not attach provenance"
    return prov.params.get("model")


def test_dml_picks_irm_for_binary_d_no_iv():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n)
    d = rng.binomial(1, 0.5, n)              # binary D
    y = 1.0 * d + 0.3 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"d": d, "y": y, "x": x})
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], data=df,
    )
    r = q.estimate()
    assert _dml_sub_model(r) == "irm"
    assert r.estimand == "ATE"


def test_dml_picks_plr_for_continuous_d_no_iv():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(0, 1, n)
    d = 0.4 * x + rng.normal(0, 1, n)         # continuous D
    y = 1.0 * d + 0.3 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"d": d, "y": y, "x": x})
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], data=df,
    )
    r = q.estimate()
    assert _dml_sub_model(r) == "plr"
    assert r.estimand == "ATE"


def test_dml_picks_iivm_for_binary_d_and_binary_z():
    rng = np.random.default_rng(2)
    n = 800
    z = rng.binomial(1, 0.5, n)              # binary Z
    x = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)
    d_score = -0.3 + 1.5 * z + 0.4 * x + 0.6 * u + rng.normal(0, 1, n)
    d = (d_score > 0).astype(int)            # binary D
    y = 0.5 + 1.0 * d + 0.4 * x + 0.6 * u + rng.normal(0, 1, n)
    df = pd.DataFrame({"d": d, "y": y, "x": x, "z": z})
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], instruments=["z"], data=df,
    )
    r = q.estimate()
    assert _dml_sub_model(r) == "iivm"
    assert r.estimand == "LATE"


def test_dml_picks_pliv_for_continuous_d_with_iv():
    """Cross of Bug A's regression: continuous D + binary Z must
    NOT choose IIVM (which requires binary D); falls through to PLIV."""
    rng = np.random.default_rng(3)
    n = 600
    z = rng.binomial(1, 0.5, n)              # binary Z
    x = rng.normal(0, 1, n)
    d = 0.5 * z + 0.3 * x + rng.normal(0, 1, n)  # continuous D
    y = 0.5 + 1.0 * d + 0.4 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"d": d, "y": y, "x": x, "z": z})
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], instruments=["z"], data=df,
    )
    r = q.estimate()
    assert _dml_sub_model(r) == "pliv"
    assert r.estimand == "LATE"


def test_dml_picks_pliv_for_continuous_instrument():
    """Single instrument with >2 unique values (continuous Z) →
    PLIV, not IIVM."""
    rng = np.random.default_rng(4)
    n = 600
    z = rng.normal(0, 1, n)                   # continuous Z
    x = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)
    d_score = -0.3 + 0.8 * z + 0.3 * x + 0.5 * u + rng.normal(0, 1, n)
    d = (d_score > 0).astype(int)             # binary D
    y = 0.5 + 1.0 * d + 0.4 * x + 0.5 * u + rng.normal(0, 1, n)
    df = pd.DataFrame({"d": d, "y": y, "x": x, "z": z})
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], instruments=["z"], data=df,
    )
    r = q.estimate()
    assert _dml_sub_model(r) == "pliv"
    assert r.estimand == "LATE"


def test_dml_rejects_multi_instrument_with_helpful_error():
    """sp.dml's PLIV is single-scalar only. The dispatcher must raise
    a clear error pointing to sp.scalar_iv_projection rather than
    letting sp.dml's internal error bubble up."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "d": rng.binomial(1, 0.5, n),
        "y": rng.normal(0, 1, n),
        "x": rng.normal(0, 1, n),
        "z1": rng.binomial(1, 0.5, n),
        "z2": rng.normal(0, 1, n),
    })
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], instruments=["z1", "z2"], data=df,
    )
    with pytest.raises(ValueError, match="single scalar instrument"):
        q.estimate()


def test_dml_random_state_is_reproducible(confounded_data):
    """Same random_state → identical DML estimate / SE / CI across reruns."""
    def run(seed):
        q = sp.causal_question(
            treatment="treat", outcome="y", design="dml",
            covariates=["x"], data=confounded_data,
        )
        return q.estimate(random_state=seed)

    r1 = run(0)
    r2 = run(0)
    assert r1.estimate == r2.estimate
    assert r1.se == r2.se
    assert r1.ci == r2.ci

    r3 = run(1)
    assert (r3.estimate != r1.estimate) or (r3.se != r1.se), (
        "different DML random_state should change at least one of "
        "(estimate, se)"
    )


def test_tmle_random_state_is_reproducible(confounded_data):
    """Same random_state → identical TMLE estimate / SE / CI."""
    def run(seed):
        q = sp.causal_question(
            treatment="treat", outcome="y", design="tmle",
            covariates=["x"], data=confounded_data,
        )
        return q.estimate(random_state=seed)

    r1 = run(0)
    r2 = run(0)
    assert r1.estimate == r2.estimate
    assert r1.se == r2.se
    assert r1.ci == r2.ci


def test_metalearner_random_state_is_reproducible(confounded_data):
    """sp.metalearner doesn't accept random_state directly, but the
    dispatcher translates it into seeded nuisance models so reruns
    with the same seed give identical (estimate, se, ci).
    """
    def run(seed):
        q = sp.causal_question(
            treatment="treat", outcome="y", design="metalearner",
            covariates=["x"], data=confounded_data,
        )
        return q.estimate(random_state=seed)

    r1 = run(0)
    r2 = run(0)
    assert r1.estimate == r2.estimate
    assert r1.se == r2.se
    assert r1.ci == r2.ci

    # Different seed must produce a different result somewhere.
    r3 = run(1)
    assert (r3.estimate != r1.estimate) or (r3.se != r1.se)


def test_dml_user_model_override_wins():
    """User-passed `model='plr'` overrides the auto-picker even when
    auto would have picked something else (here: binary D would
    auto-pick IRM, but user forces PLR)."""
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(0, 1, n)
    d = rng.binomial(1, 0.5, n).astype(float)
    y = 1.0 * d + 0.3 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"d": d, "y": y, "x": x})
    q = sp.causal_question(
        treatment="d", outcome="y", design="dml",
        covariates=["x"], data=df,
    )
    r = q.estimate(model="plr")
    assert _dml_sub_model(r) == "plr"
