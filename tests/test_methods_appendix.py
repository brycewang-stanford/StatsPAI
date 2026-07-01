"""Tests for the auto-generated Methods and Formulas appendix.

Covers: public-API wiring, estimator resolution (exact / alias / token /
guarded-substring), zero-hallucination degradation for unregistered methods,
inference read-off from ``model_info``, citation integration, multi-result
dedup, and format handling.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult, EconometricResults
from statspai.smart.methods_appendix import MethodSpec, _resolve_spec, methods_appendix


def _causal(method, **kw):
    base = dict(
        estimand="ATT",
        estimate=1.0,
        se=0.2,
        pvalue=0.01,
        ci=(0.6, 1.4),
        alpha=0.05,
        n_obs=100,
    )
    base.update(kw)
    return CausalResult(method=method, **base)


# --------------------------------------------------------------------------
#  Public-API wiring
# --------------------------------------------------------------------------


def test_public_export():
    assert hasattr(sp, "methods_appendix")
    assert "methods_appendix" in sp.__all__
    assert callable(sp.methods_appendix)


def test_result_method_present():
    r = _causal("did")
    assert hasattr(r, "to_appendix")
    out = r.to_appendix(format="markdown")
    assert "Methods and Formulas" in out


def test_econometric_result_method_present():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(size=80), "x": rng.normal(size=80)})
    r = sp.regress("y ~ x", data=df, robust="hc1")
    out = r.to_appendix(format="text")
    assert "Methods and Formulas" in out
    assert "Produced by StatsPAI" in out


# --------------------------------------------------------------------------
#  Resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, expect_key",
    [
        ("did", "did_2x2"),
        ("Synthetic Control Method", "synth"),
        ("sdid", "sdid"),
        ("2sls", "iv"),
        ("liml", "iv"),
        ("rdrobust", "rdrobust"),
        ("rd", "rdrobust"),
        ("aipw", "aipw"),
        ("tmle", "tmle"),
        ("dml", "dml"),
        ("ipw", "ipw"),
        ("matching", "psm"),
        ("callaway_santanna", "callaway_santanna"),
        ("sun_abraham", "sun_abraham"),
        ("drdid", "drdid"),
        # Expanded causal table
        ("event_study", "event_study"),
        ("ddd", "ddd"),
        ("gsynth", "gsynth"),
        ("generalized_synthetic_control", "gsynth"),
        ("did_imputation", "did_imputation"),
        ("bjs", "did_imputation"),
        ("cic", "cic"),
        ("lp_did", "lp_did"),
        # Regression families
        ("ols", "ols"),
        ("regress", "ols"),
        ("poisson", "poisson"),
        ("fepois", "poisson"),
        ("logit", "logit"),
        ("probit", "probit"),
        ("fe", "fe"),
        ("feols", "fe"),
        ("within", "fe"),
        # Distributional / mediation / identification additions
        ("qte", "qte"),
        ("quantile_treatment_effect", "qte"),
        ("mediation", "mediation"),
        ("mediate", "mediation"),
        ("front_door", "front_door"),
        ("frontdoor", "front_door"),
        ("g_computation", "g_computation"),
        ("standardization", "g_computation"),
        ("manski_bounds", "manski_bounds"),
        ("continuous_did", "continuous_did"),
        ("bartik", "bartik"),
        ("shift_share", "bartik"),
        ("proximal", "proximal"),
        ("pci", "proximal"),
        # Design / estimator batch-2
        ("rkd", "rkd"),
        ("regression_kink", "rkd"),
        ("bunching", "bunching"),
        ("augsynth", "augsynth"),
        ("mc_panel", "mc_panel"),
        ("matrix_completion", "mc_panel"),
        ("ivqr", "ivqr"),
        ("gmm", "gmm"),
        # HTE / genetics-IV / dose-response / survival batch-3
        ("causal_forest", "causal_forest"),
        ("grf", "causal_forest"),
        ("metalearner", "metalearner"),
        ("xlearner", "metalearner"),
        ("r_learner", "r_learner"),
        ("mr", "mr"),
        ("ivw", "mr"),
        ("dose_response", "dose_response"),
        ("cox", "cox"),
        ("coxph", "cox"),
        # Survival / partial-ID / policy / sensitivity / inference batch-4
        ("kaplan_meier", "kaplan_meier"),
        ("km", "kaplan_meier"),
        ("lee_bounds", "lee_bounds"),
        ("lee_trimming", "lee_bounds"),
        ("policy_tree", "policy_tree"),
        ("policy_learning", "policy_tree"),
        ("honest_did", "honest_did"),
        ("rambachan_roth", "honest_did"),
        ("oster", "oster"),
        ("oster_bounds", "oster"),
        ("wild_cluster_bootstrap", "wild_cluster_bootstrap"),
        ("wcb", "wild_cluster_bootstrap"),
        # Neural IV / interference / time-varying confounding batch-5
        ("deepiv", "deepiv"),
        ("neural_iv", "deepiv"),
        ("interference", "interference"),
        ("spillover", "interference"),
        ("msm", "msm"),
        ("marginal_structural_model", "msm"),
        ("network_exposure", "network_exposure"),
        ("aronow_samii", "network_exposure"),
        ("stacked_did", "stacked_did"),
        ("surrogate", "surrogate"),
        ("surrogate_index", "surrogate"),
        # Decomposition family + multiway clustering batch-6
        ("oaxaca", "oaxaca"),
        ("blinder_oaxaca", "oaxaca"),
        ("rif_decomposition", "rif_decomposition"),
        ("rifreg", "rif_decomposition"),
        ("dfl_decompose", "dfl_decompose"),
        ("dfl", "dfl_decompose"),
        ("gelbach", "gelbach"),
        ("twoway_cluster", "twoway_cluster"),
        ("multiway_cluster", "twoway_cluster"),
        ("kitagawa", "kitagawa"),
        # RCT adjustment / conformal / frontier / kernel & two-stage batch-7
        ("lin", "lin"),
        ("regression_adjustment", "lin"),
        ("conformal", "conformal"),
        ("conformal_ite", "conformal"),
        ("frontier", "frontier"),
        ("sfa", "frontier"),
        ("machado_mata", "machado_mata"),
        ("kernel_iv", "kernel_iv"),
        ("kiv", "kernel_iv"),
        ("gardner_did", "gardner_did"),
        ("did2s", "gardner_did"),
        # Randomization / Bayesian / MTE / DR-learner / npIV / sensitivity b8
        ("ri", "ri"),
        ("randomization_inference", "ri"),
        ("bcf", "bcf"),
        ("mte", "mte"),
        ("marginal_treatment_effect", "mte"),
        ("dr_learner", "dr_learner"),
        ("npiv", "npiv"),
        ("nonparametric_iv", "npiv"),
        ("sensemakr", "sensemakr"),
        ("robustness_value", "sensemakr"),
        # Balancing / matching / selection / IV / ensemble batch-9
        ("ebalance", "ebalance"),
        ("entropy_balancing", "ebalance"),
        ("cbps", "cbps"),
        ("sbw", "sbw"),
        ("heckman", "heckman"),
        ("heckit", "heckman"),
        ("jive", "jive"),
        ("jackknife_iv", "jive"),
        ("super_learner", "super_learner"),
        ("superlearner", "super_learner"),
        # Bounds / time-series / spatial / structure-learning / etwfe batch-10
        ("balke_pearl", "balke_pearl"),
        ("iv_bounds", "balke_pearl"),
        ("horowitz_manski", "horowitz_manski"),
        ("causal_impact", "causal_impact"),
        ("bsts", "causal_impact"),
        ("conley", "conley"),
        ("spatial_hac", "conley"),
        ("notears", "notears"),
        ("etwfe", "etwfe"),
        ("two_way_mundlak", "etwfe"),
        # Causal discovery / weak-IV / local projections batch-11
        ("pc_algorithm", "pc_algorithm"),
        ("pc", "pc_algorithm"),
        ("fci", "fci"),
        ("lingam", "lingam"),
        ("ges", "ges"),
        ("anderson_rubin", "anderson_rubin"),
        ("ar_test", "anderson_rubin"),
        ("local_projections", "local_projections"),
        ("jorda_lp", "local_projections"),
    ],
)
def test_resolution(method, expect_key):
    spec = _resolve_spec(_causal(method))
    assert spec is not None
    assert spec.key == expect_key


def test_citation_key_overrides_method():
    r = _causal("BFGS")  # noisy optimizer name as method
    r._citation_key = "rdrobust"
    spec = _resolve_spec(r)
    assert spec is not None and spec.key == "rdrobust"


def test_short_alias_no_false_positive():
    # "sc" must not match the "sc" buried inside "obscure".
    assert _resolve_spec(_causal("some_obscure_method")) is None
    # "iv" must not fire on the "iv" inside "derivative".
    assert _resolve_spec(_causal("derivative_estimator")) is None
    # Horowitz-Manski bounds resolve to their OWN spec, never collapsing onto
    # the Manski worst-case-bounds spec.
    assert _resolve_spec(_causal("horowitz_manski")).key == "horowitz_manski"


# --------------------------------------------------------------------------
#  Zero-hallucination degradation
# --------------------------------------------------------------------------


def test_unregistered_degrades_not_hallucinates():
    out = methods_appendix(_causal("totally_unknown_xyz"), format="markdown")
    assert "not yet registered" in out
    # No invented estimand/estimator math block leaks in.
    assert "$$" not in out


# --------------------------------------------------------------------------
#  Content
# --------------------------------------------------------------------------


def test_registered_contains_formula_and_assumptions():
    out = methods_appendix(_causal("rdrobust"), format="markdown")
    assert "Estimand" in out
    assert "Estimator" in out
    assert "Identifying assumptions" in out
    assert "$$" in out  # display math present


def test_inference_read_from_model_info():
    r = _causal(
        "rdrobust",
        model_info={
            "se_method": "robust (CCT)",
            "bandwidth_h": 0.123456,
            "cluster_var": "state",
        },
    )
    out = r.to_appendix(format="text")
    assert "robust (CCT)" in out
    assert "0.1235" in out  # bandwidth formatted
    assert "state" in out


def test_citation_integration():
    out = methods_appendix(_causal("callaway_santanna"), format="markdown")
    assert "Callaway" in out and "2021" in out


def test_provenance_toggle():
    out = methods_appendix(_causal("did"), format="markdown")
    assert "Produced by StatsPAI" in out
    without = methods_appendix(
        _causal("did"), format="markdown", include_provenance=False
    )
    assert "Produced by StatsPAI" not in without


def test_assumptions_toggle_off():
    out = methods_appendix(
        _causal("rdrobust"), format="text", include_assumptions=False
    )
    assert "Identifying assumptions" not in out


# --------------------------------------------------------------------------
#  Multi-result and format handling
# --------------------------------------------------------------------------


def test_multi_result_dedup():
    r1 = _causal("did")
    r2 = _causal("2sls", estimand="LATE")
    out = methods_appendix([r1, r1, r2], format="markdown")
    assert out.count("### ") == 2


def test_latex_structure():
    out = methods_appendix(_causal("did"), format="latex")
    assert out.startswith("\\section*{Methods and Formulas}")
    assert "\\subsection*{" in out
    assert "\\[" in out


@pytest.mark.parametrize("fmt", ["latex", "markdown", "text"])
def test_all_formats_run(fmt):
    out = methods_appendix(_causal("dml"), format=fmt)
    assert isinstance(out, str) and len(out) > 0


def test_bad_format_raises():
    with pytest.raises(ValueError):
        methods_appendix(_causal("did"), format="bogus")


def test_empty_sequence_raises():
    with pytest.raises(ValueError):
        methods_appendix([], format="text")


# --------------------------------------------------------------------------
#  End-to-end on a real fitted estimator
# --------------------------------------------------------------------------


def test_end_to_end_real_did_fit():
    rng = np.random.default_rng(0)
    ids = np.repeat(np.arange(40), 2)
    time = np.tile([0, 1], 40)
    treat = (ids >= 20).astype(int)
    y = 1.0 + 2.0 * treat * time + rng.normal(size=len(ids))
    df = pd.DataFrame({"id": ids, "time": time, "treat": treat, "y": y})
    res = sp.did(df, y="y", treat="treat", time="time", id="id")
    out = res.to_appendix(format="markdown")
    assert "Methods and Formulas" in out
    assert "Estimand" in out
    assert "Inference" in out


def test_methodspec_is_frozen():
    spec = MethodSpec(
        key="x", name="X", estimand_latex="a", estimator_latex="b", prose="c"
    )
    with pytest.raises(Exception):
        spec.name = "Y"  # frozen dataclass


# --------------------------------------------------------------------------
#  Provenance (the "exact code path" leg)
# --------------------------------------------------------------------------


def test_provenance_present_by_default():
    out = methods_appendix(_causal("rdrobust"), format="text")
    assert "Produced by StatsPAI v" in out
    assert "methods spec 'rdrobust'" in out


def test_provenance_toggle_off():
    out = methods_appendix(_causal("rdrobust"), format="text", include_provenance=False)
    assert "Produced by StatsPAI" not in out


def test_provenance_unregistered_states_no_spec():
    out = methods_appendix(_causal("totally_unknown_xyz"), format="text")
    assert "no methods spec registered" in out


def test_provenance_records_version():
    out = methods_appendix(_causal("did"), format="text")
    assert sp.__version__ in out


# --------------------------------------------------------------------------
#  EconometricResults (regression family) parity
# --------------------------------------------------------------------------


def _econ(model_info):
    return EconometricResults(
        params=pd.Series({"const": 1.0, "x": 2.0}),
        std_errors=pd.Series({"const": 0.1, "x": 0.2}),
        model_info=model_info,
        data_info={"df_resid": 97},
    )


def test_econ_iv_resolves_to_iv_spec():
    r = _econ({"method": "2sls", "se_method": "robust"})
    out = r.to_appendix(format="markdown")
    assert "Instrumental Variables" in out
    assert "$$" in out  # real estimator math present
    assert "robust" in out  # inference read from model_info


def test_econ_ols_resolves_to_ols_spec():
    r = _econ({"model_type": "ols", "se_method": "HC1"})
    out = r.to_appendix(format="text")
    assert "Ordinary Least Squares" in out
    assert "(X'X)^{-1}" in out  # OLS normal-equations estimator
    assert "HC1" in out  # inference read from model_info
    assert "Produced by StatsPAI" in out  # provenance present


def test_econ_unregistered_model_degrades_gracefully():
    r = _econ({"model_type": "some_exotic_glm", "se_method": "HC1"})
    out = r.to_appendix(format="text")
    # Genuinely unregistered -> placeholder, never invented math.
    assert "not yet registered" in out
    assert "HC1" in out  # inference still reported
    assert "Produced by StatsPAI" in out  # provenance still present


def test_econ_real_regress_fit_runs():
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    df = pd.DataFrame({"y": 2.0 * x + rng.normal(size=80), "x": x})
    res = sp.regress("y ~ x", data=df)
    out = res.to_appendix(format="markdown")
    assert "Methods and Formulas" in out
    assert "Produced by StatsPAI" in out


# --------------------------------------------------------------------------
#  Citation coverage (the "verified reference" leg)
# --------------------------------------------------------------------------


def _resolves_citation(key):
    r = _causal(key)
    r._citation_key = key
    apa = r.cite(format="apa")
    return bool(apa) and not str(apa).lstrip().startswith("%")


def test_every_spec_resolves_a_citation():
    """Every registered methods spec must resolve to a real reference — the
    full traceability triple (formula + verified citation + provenance)."""
    from statspai.smart.methods_appendix import _SPECS

    missing = [s.key for s in _SPECS if not _resolves_citation(s.key)]
    assert not missing, f"specs without a resolving citation: {missing}"


def test_newly_cited_specs_carry_doi():
    # Spot-check a few of the specs whose citations were added for coverage.
    for key, token in [
        ("qte", "Firpo"),
        ("manski_bounds", "Manski"),
        ("cic", "Athey"),
        ("lp_did", "Dube"),
        ("psm", "Rosenbaum"),
        ("ddd", "Møen"),  # exercises the {\o} -> ø diacritic path
    ]:
        r = _causal(key)
        r._citation_key = key
        assert token in r.cite(format="apa"), f"{key}: missing {token!r}"


def test_working_paper_citations_upgraded_to_published():
    """RKD and augmented-SC citations point to the published journals, not the
    SSRN working-paper versions."""
    for key, venue in [
        ("rkd", "Econometrica"),
        ("augsynth", "Journal of the American Statistical Association"),
    ]:
        r = _causal(key)
        r._citation_key = key
        apa = r.cite(format="apa")
        assert venue in apa, f"{key}: expected {venue!r} in {apa!r}"
        assert "SSRN" not in apa, f"{key}: still cites SSRN"


def test_batch3_estimators_carry_verified_citation():
    for key, token in [
        ("causal_forest", "Wager"),
        ("metalearner", "Künzel"),  # exercises {\"u} -> ü diacritic
        ("r_learner", "Nie"),
        ("mr", "Burgess"),
        ("dose_response", "Hirano"),
        ("cox", "Cox"),
    ]:
        r = _causal(key)
        r._citation_key = key
        assert token in r.cite(format="apa"), f"{key}: missing {token!r}"


def test_batch4_estimators_carry_verified_citation():
    for key, token in [
        ("kaplan_meier", "Kaplan"),
        ("lee_bounds", "Lee"),
        ("policy_tree", "Athey"),
        ("honest_did", "Rambachan"),
        ("oster", "Oster"),
        ("wild_cluster_bootstrap", "Cameron"),
    ]:
        r = _causal(key)
        r._citation_key = key
        assert token in r.cite(format="apa"), f"{key}: missing {token!r}"


def test_batch5_estimators_carry_verified_citation():
    for key, token in [
        ("deepiv", "Hartford"),
        ("interference", "Hudgens"),
        ("msm", "Hernán"),  # exercises unicode author path
        ("network_exposure", "Aronow"),
        ("stacked_did", "Cengiz"),
        ("surrogate", "Athey"),
    ]:
        r = _causal(key)
        r._citation_key = key
        assert token in r.cite(format="apa"), f"{key}: missing {token!r}"


def test_batch6_estimators_carry_verified_citation():
    for key, token in [
        ("oaxaca", "Oaxaca"),
        ("rif_decomposition", "Firpo"),
        ("dfl_decompose", "DiNardo"),
        ("gelbach", "Gelbach"),
        ("twoway_cluster", "Cameron"),
        ("kitagawa", "Kitagawa"),
    ]:
        r = _causal(key)
        r._citation_key = key
        assert token in r.cite(format="apa"), f"{key}: missing {token!r}"


def test_batch7_estimators_carry_verified_citation():
    for key, token in [
        ("lin", "Lin"),
        ("conformal", "Candès"),  # exercises {\`e} -> è diacritic
        ("frontier", "Aigner"),
        ("machado_mata", "Machado"),
        ("kernel_iv", "Singh"),
        ("gardner_did", "Gardner"),
    ]:
        r = _causal(key)
        r._citation_key = key
        assert token in r.cite(format="apa"), f"{key}: missing {token!r}"


def test_batch8_estimators_carry_verified_citation():
    for key, token in [
        ("ri", "Fisher"),
        ("bcf", "Hahn"),
        ("mte", "Heckman"),
        ("dr_learner", "Kennedy"),
        ("npiv", "Newey"),
        ("sensemakr", "Cinelli"),
    ]:
        r = _causal(key)
        r._citation_key = key
        assert token in r.cite(format="apa"), f"{key}: missing {token!r}"


def test_batch9_estimators_carry_verified_citation():
    for key, token in [
        ("ebalance", "Hainmueller"),
        ("cbps", "Imai"),
        ("sbw", "Zubizarreta"),
        ("heckman", "Heckman"),
        ("jive", "Angrist"),
        ("super_learner", "Laan"),
    ]:
        r = _causal(key)
        r._citation_key = key
        assert token in r.cite(format="apa"), f"{key}: missing {token!r}"


def test_batch10_estimators_carry_verified_citation():
    for key, token in [
        ("balke_pearl", "Balke"),
        ("horowitz_manski", "Horowitz"),
        ("causal_impact", "Brodersen"),
        ("conley", "Conley"),
        ("notears", "Zheng"),
        ("etwfe", "Wooldridge"),
    ]:
        r = _causal(key)
        r._citation_key = key
        assert token in r.cite(format="apa"), f"{key}: missing {token!r}"


def test_batch11_estimators_carry_verified_citation():
    for key, token in [
        ("pc_algorithm", "Spirtes"),
        ("fci", "Spirtes"),
        ("lingam", "Shimizu"),
        ("ges", "Chickering"),
        ("anderson_rubin", "Anderson"),
        ("local_projections", "Jordà"),  # exercises {\`a} -> à diacritic
    ]:
        r = _causal(key)
        r._citation_key = key
        assert token in r.cite(format="apa"), f"{key}: missing {token!r}"
