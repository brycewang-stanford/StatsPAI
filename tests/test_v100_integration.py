"""v1.0.0 integration smoke test.

Verifies that every v1.0 headline addition is reachable through
``sp.*`` and carries a registry entry discoverable by agents.

This is the contract for the v1.0 public API: if this test goes
red, the v1.0 surface is broken.
"""

from __future__ import annotations

import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------


def test_v1_version():
    # Accept any v1.x.y — this integration smoke should track the
    # public API stability contract, not the patch-level digit.
    assert sp.__version__.startswith("1.")


def test_v1_registry_size_grew():
    # v0.9.17 shipped with ~729 registered names; v1.0 should include
    # at least the scaffolded frontier modules on top of that.
    names = sp.list_functions()
    assert len(names) >= 729


# ---------------------------------------------------------------------------
# Three-school completion (v0.9.17) — must still be intact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "odds_ratio", "relative_risk", "mantel_haenszel", "bradford_hill",
    "sensitivity_specificity", "roc_curve", "cohen_kappa",
    "mr_heterogeneity", "mr_pleiotropy_egger", "mr_leave_one_out",
    "mr_steiger", "mr_presso", "mr_radial",
    "mr_mode", "mr_f_statistic",
    "longitudinal_analyze", "longitudinal_contrast",
    "regime", "always_treat", "never_treat",
    "causal_question", "CausalQuestion",
    "preregister", "load_preregister",
    "unified_sensitivity", "SensitivityDashboard",
    "dag_recommend_estimator",
    "target_trial_protocol", "target_trial_emulate",
    "target_trial_report", "target_trial_checklist",
])
def test_v097_three_school_surface_still_reachable(name):
    assert hasattr(sp, name), f"{name} disappeared from sp.*"


# ---------------------------------------------------------------------------
# v1.0 frontier modules — each must be reachable + registered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    # Bridging theorems
    "bridge", "BridgeResult",
    # Fairness
    "counterfactual_fairness", "orthogonal_to_bias",
    "demographic_parity", "equalized_odds", "fairness_audit",
    # Surrogate indices
    "surrogate_index", "long_term_from_short", "proximal_surrogate_index",
    # Multivariable MR
    "mr_multivariable", "mr_mediation", "mr_bma",
    # DiD frontiers
    "did_bcf", "cohort_anchored_event_study",
    "design_robust_event_study", "did_misclassified",
    # Conformal frontiers
    "conformal_debiased_ml", "conformal_density_ite",
    "conformal_fair_ite", "conformal_ite_multidp",
    # Proximal frontiers
    "fortified_pci", "bidirectional_pci", "pci_mtp", "select_pci_proxies",
    # QTE / RD frontiers
    "beyond_average_late", "qte_hd_panel", "rd_distribution",
    "rd_interference", "rd_multi_score",
    # Time-series causal discovery
    "pcmci", "lpcmci", "dynotears",
    # LTMLE survival / BCF longitudinal / sequential SDID
    "ltmle_survival", "sequential_sdid",
    # ML bounds
    "ml_bounds",
    # TARGET Statement 2025
    "target_trial_checklist",
    # Frontier sensitivity
    "copula_sensitivity", "survival_sensitivity",
    "calibrate_confounding_strength",
])
def test_v1_frontier_surface(name):
    assert hasattr(sp, name), f"{name} not exposed at sp.*"


@pytest.mark.parametrize("name", [
    "bridge", "did_bcf", "cohort_anchored_event_study", "pcmci",
    "ltmle_survival", "copula_sensitivity", "conformal_debiased_ml",
])
def test_v1_frontier_in_registry(name):
    sp.list_functions()  # trigger full-registry build
    from statspai.registry import _REGISTRY
    assert name in _REGISTRY, f"{name} missing from registry"


# ---------------------------------------------------------------------------
# End-to-end workflow pins
# ---------------------------------------------------------------------------


def test_v1_tte_checklist_renders():
    import pandas as pd
    proto = sp.target_trial_protocol(
        eligibility="age >= 50",
        treatment_strategies=["A", "B"],
        assignment="observational emulation",
        time_zero="t0", followup_end="5y", outcome="Y",
        causal_contrast="ITT",
        analysis_plan="IPW + pooled logistic",
    )
    df = pd.DataFrame({"age": [60, 70, 55],
                        "treat": [1, 0, 1],
                        "outcome": [1.0, 2.0, 1.5]})
    result = sp.target_trial_emulate(
        proto, df, outcome_col="outcome", treatment_col="treat",
    )
    md = result.to_paper(fmt="target")
    assert "TARGET Statement" in md
    assert "[AUTO]" in md  # items auto-filled from protocol
    assert "[TODO]" in md  # items awaiting author


def test_v1_causal_question_end_to_end():
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 300
    d = rng.binomial(1, 0.5, n)
    y = 5 + 1.4 * d + rng.normal(0, 1, n)
    df = pd.DataFrame({"treat": d, "y": y})
    q = sp.causal_question(
        treatment="treat", outcome="y",
        design="rct", data=df,
    )
    plan = q.identify()
    r = q.estimate()
    report = q.report("text")
    assert plan.estimator == "regress"
    assert abs(r.estimate - 1.4) < 0.4
    assert "Causal Question" in report
    # sensitivity() method
    dash = r.underlying.sensitivity()
    assert dash.breakdown["bias_to_flip"] > 0
