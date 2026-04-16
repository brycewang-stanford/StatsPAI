"""
Comprehensive tests for the tier-C decomposition suite.

Covers: OB, Gelbach, DFL, FFL, Machado-Mata, Melly, CFM, Fairlie,
Bauer-Sinning, RIF, Shapley inequality, subgroup inequality,
Lerman-Yitzhaki source, Kitagawa, Das Gupta, gap-closing,
mediation, Jackson-VanderWeele disparity, and the unified
``sp.decompose`` dispatcher.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.decomposition import _common


SEED = 42


@pytest.fixture(scope="module")
def cps():
    return sp.cps_wage(n=1200, seed=SEED)


@pytest.fixture(scope="module")
def disparity():
    return sp.disparity_panel(n=1200, seed=SEED)


@pytest.fixture(scope="module")
def mincer():
    return sp.mincer_wage_panel(n=1500, seed=SEED)


# ════════════════════════════════════════════════════════════════════════
# Datasets
# ════════════════════════════════════════════════════════════════════════

def test_datasets_shapes(cps, disparity, mincer):
    assert cps.shape[0] == 1200
    assert "female" in cps.columns
    assert "log_wage" in cps.columns
    assert disparity.shape[0] == 1200
    assert "group" in disparity.columns
    assert mincer.shape[0] == 1500


def test_cps_wage_has_gap(cps):
    gap = cps[cps.female == 0].log_wage.mean() - cps[cps.female == 1].log_wage.mean()
    assert gap > 0.3, f"gender gap should be positive & large, got {gap}"


# ════════════════════════════════════════════════════════════════════════
# Existing: Oaxaca + Gelbach (regression guard)
# ════════════════════════════════════════════════════════════════════════

def test_oaxaca_basic(cps):
    r = sp.oaxaca(data=cps, y="log_wage", group="female",
                  x=["education", "experience", "tenure"])
    assert r.overall["gap"] > 0
    assert abs(r.overall["gap"] - (r.overall["explained"] + r.overall["unexplained"])) < 1e-6
    assert "education" in r.detailed["variable"].values


def test_oaxaca_pooled(cps):
    r = sp.oaxaca(data=cps, y="log_wage", group="female",
                  x=["education", "experience", "tenure"],
                  reference="pooled")
    assert r.reference == "pooled"


def test_gelbach_basic(cps):
    r = sp.gelbach(data=cps, y="log_wage",
                   base_x=["education"],
                   added_x=["experience", "tenure", "union"])
    # Deltas should sum to total change (up to floating error)
    assert abs(r.decomposition["delta"].sum() - r.total_change) < 1e-6


# ════════════════════════════════════════════════════════════════════════
# DFL
# ════════════════════════════════════════════════════════════════════════

def test_dfl_mean(cps):
    r = sp.dfl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience", "tenure", "union", "married"],
                         stat="mean")
    assert abs(r.composition + r.structure - r.gap) < 1e-6
    # In this DGP, β-differences dominate
    assert abs(r.structure) > abs(r.composition)


def test_dfl_quantile(cps):
    r = sp.dfl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience", "tenure"],
                         stat="quantile", tau=0.5)
    assert abs(r.composition + r.structure - r.gap) < 1e-6


def test_dfl_gini(cps):
    wage = np.exp(cps["log_wage"])
    df = cps.assign(wage=wage)
    r = sp.dfl_decompose(data=df, y="wage", group="female",
                         x=["education", "experience", "tenure"],
                         stat="gini")
    assert abs(r.composition + r.structure - r.gap) < 1e-4


def test_dfl_bootstrap(cps):
    r = sp.dfl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience", "tenure"],
                         stat="mean", inference="bootstrap", n_boot=30, seed=1)
    assert r.se is not None
    assert "composition" in r.se


def test_dfl_quantile_grid(cps):
    r = sp.dfl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience", "tenure"],
                         stat="quantile", tau=0.5,
                         quantile_grid=[0.25, 0.5, 0.75])
    assert r.quantile_grid is not None
    assert len(r.quantile_grid) == 3


# ════════════════════════════════════════════════════════════════════════
# FFL
# ════════════════════════════════════════════════════════════════════════

def test_ffl_mean(cps):
    r = sp.ffl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience", "tenure"],
                         stat="mean")
    total_parts = r.composition + r.structure + r.spec_error + r.reweight_error
    assert abs(total_parts - r.gap) < 1e-4
    # Detailed tables populated
    assert not r.detailed_composition.empty


def test_ffl_quantile(cps):
    r = sp.ffl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience", "tenure"],
                         stat="quantile", tau=0.5)
    total_parts = r.composition + r.structure + r.spec_error + r.reweight_error
    assert abs(total_parts - r.gap) < 1e-2


def test_ffl_variance(cps):
    r = sp.ffl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience", "tenure"],
                         stat="variance")
    total_parts = r.composition + r.structure + r.spec_error + r.reweight_error
    assert abs(total_parts - r.gap) < 1e-3


def test_ffl_gini(cps):
    # Critical: FFL Gini had been using an unweighted influence function;
    # verify that the identity E_w[RIF] = Gini_w(y) now holds.
    wage = np.exp(cps["log_wage"]).values
    df = cps.assign(wage=wage)
    r = sp.ffl_decompose(data=df, y="wage", group="female",
                         x=["education", "experience", "tenure"],
                         stat="gini")
    total_parts = r.composition + r.structure + r.spec_error + r.reweight_error
    assert abs(total_parts - r.gap) < 1e-3


def test_ffl_theil_closed_form(cps):
    # Theil_t / Theil_l / Atkinson use closed-form IFs (no O(n²) loop).
    wage = np.exp(cps["log_wage"]).values
    df = cps.assign(wage=wage)
    for stat in ["theil_t", "theil_l", "atkinson"]:
        r = sp.ffl_decompose(data=df, y="wage", group="female",
                             x=["education", "experience"],
                             stat=stat)
        total_parts = (r.composition + r.structure
                       + r.spec_error + r.reweight_error)
        assert abs(total_parts - r.gap) < 1e-2, (
            f"FFL identity broken for {stat}: sum={total_parts:.6f} "
            f"vs gap={r.gap:.6f}"
        )


def test_ffl_detailed_composition_includes_cons(cps):
    r = sp.ffl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience"], stat="mean")
    # Both detail tables should now sum to the overall component.
    assert "_cons" in r.detailed_composition["variable"].values
    assert "_cons" in r.detailed_structure["variable"].values
    assert abs(r.detailed_composition["composition"].sum()
               - r.composition) < 1e-6
    assert abs(r.detailed_structure["structure"].sum()
               - r.structure) < 1e-6


# ════════════════════════════════════════════════════════════════════════
# Machado-Mata
# ════════════════════════════════════════════════════════════════════════

def test_machado_mata(cps):
    r = sp.machado_mata(data=cps, y="log_wage", group="female",
                        x=["education", "experience", "tenure"],
                        tau_grid=[0.25, 0.5, 0.75], n_sim=200, n_tau_qr=19)
    assert len(r.quantile_grid) == 3
    for _, row in r.quantile_grid.iterrows():
        assert abs(row["composition"] + row["structure"] - row["gap"]) < 1e-6


# ════════════════════════════════════════════════════════════════════════
# Melly
# ════════════════════════════════════════════════════════════════════════

def test_melly(cps):
    r = sp.melly_decompose(data=cps, y="log_wage", group="female",
                           x=["education", "experience", "tenure"],
                           tau_grid=[0.25, 0.5, 0.75], n_tau_qr=19)
    assert len(r.quantile_grid) == 3
    for _, row in r.quantile_grid.iterrows():
        assert abs(row["composition"] + row["structure"] - row["gap"]) < 1e-6


# ════════════════════════════════════════════════════════════════════════
# CFM
# ════════════════════════════════════════════════════════════════════════

def test_cfm(cps):
    r = sp.cfm_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience", "tenure"],
                         tau_grid=[0.25, 0.5, 0.75], n_thresh=25)
    assert len(r.quantile_grid) == 3
    # CDF should be monotone in y
    cdf_a = r.cdf_grid["cdf_a"].values
    assert np.all(np.diff(cdf_a) >= -1e-10)
    assert r.ks_stat >= 0


# ════════════════════════════════════════════════════════════════════════
# Nonlinear
# ════════════════════════════════════════════════════════════════════════

def test_fairlie_logit(cps):
    r = sp.fairlie(data=cps, y="union", group="female",
                   x=["education", "experience", "tenure", "married"],
                   model="logit", n_sim=50)
    assert r.gap == pytest.approx(cps[cps.female == 0].union.mean() -
                                   cps[cps.female == 1].union.mean(), abs=1e-6)
    assert r.method == "Fairlie"


def test_bauer_sinning(cps):
    r = sp.bauer_sinning(data=cps, y="union", group="female",
                         x=["education", "experience", "tenure", "married"],
                         model="logit")
    assert abs(r.explained + r.unexplained - r.gap) < 1e-6


def test_fairlie_probit(cps):
    r = sp.fairlie(data=cps, y="union", group="female",
                   x=["education", "experience"], model="probit", n_sim=30)
    assert np.isfinite(r.explained)


# ════════════════════════════════════════════════════════════════════════
# RIF
# ════════════════════════════════════════════════════════════════════════

def test_rifreg(cps):
    r = sp.rifreg("log_wage ~ education + experience", data=cps,
                  statistic="quantile", tau=0.9)
    assert "education" in r.params.index


def test_rif_decomposition(cps):
    r = sp.rif_decomposition("log_wage ~ education + experience + tenure",
                             data=cps, group="female",
                             statistic="quantile", tau=0.5)
    assert abs(r.explained + r.unexplained - r.total_diff) < 1e-6


# ════════════════════════════════════════════════════════════════════════
# Inequality
# ════════════════════════════════════════════════════════════════════════

def test_inequality_index(cps):
    wage = np.exp(cps["log_wage"])
    for idx in ["theil_t", "theil_l", "gini", "atkinson", "cv2"]:
        v = sp.inequality_index(wage.values, index=idx)
        assert np.isfinite(v) and v >= 0


def test_subgroup_theil(cps):
    wage = np.exp(cps["log_wage"]).values
    df = cps.assign(wage=wage)
    r = sp.subgroup_decompose(data=df, y="wage", by="female", index="theil_t")
    assert abs(r.total - (r.between + r.within)) < 1e-6


def test_subgroup_gini(cps):
    wage = np.exp(cps["log_wage"]).values
    df = cps.assign(wage=wage)
    r = sp.subgroup_decompose(data=df, y="wage", by="female", index="gini")
    # Dagum decomposition: total = between + within + overlap (within rounding)
    assert r.overlap is not None


def test_shapley_inequality(cps):
    wage = np.exp(cps["log_wage"]).values
    df = cps.assign(wage=wage)
    r = sp.shapley_inequality(data=df, y="wage",
                              x=["education", "experience", "tenure"],
                              index="theil_t")
    # Shapley contributions should approximately sum to predicted index
    # (exactness depends on model; we check finite and non-trivial)
    assert len(r.shapley) == 3
    assert r.shapley["contribution"].sum() > 0


def test_source_decomposition():
    rng = np.random.default_rng(123)
    n = 500
    labor = rng.gamma(2, 30, n)
    capital = rng.gamma(1.5, 20, n)
    transfer = rng.exponential(10, n)
    df = pd.DataFrame({"labor": labor, "capital": capital, "transfer": transfer})
    r = sp.source_decompose(data=df, sources=["labor", "capital", "transfer"])
    # Shares sum to 1
    assert abs(r.sources["share"].sum() - 1.0) < 1e-6


# ════════════════════════════════════════════════════════════════════════
# Kitagawa / Das Gupta
# ════════════════════════════════════════════════════════════════════════

def test_kitagawa(cps):
    cps2 = cps.assign(educ_bin=(cps.education > 12).astype(int))
    r = sp.kitagawa_decompose(data=cps2, rate="union", group="female",
                              by="educ_bin")
    # rate + composition + interaction ≈ gap
    assert abs(r.rate_effect + r.composition_effect + r.interaction
               - r.gap) < 1e-6


def test_das_gupta():
    rng = np.random.default_rng(0)
    n = 300
    a = pd.DataFrame({"f1": rng.uniform(1, 2, n),
                      "f2": rng.uniform(0.5, 1.5, n),
                      "f3": rng.uniform(2, 3, n)})
    b = pd.DataFrame({"f1": rng.uniform(1.2, 2.2, n),
                      "f2": rng.uniform(0.3, 1.3, n),
                      "f3": rng.uniform(1.5, 2.5, n)})
    r = sp.das_gupta(a, b, factor_names=["f1", "f2", "f3"])
    assert abs(r.factor_effects["effect"].sum() - r.gap) < 1e-6


# ════════════════════════════════════════════════════════════════════════
# Causal
# ════════════════════════════════════════════════════════════════════════

def test_gap_closing_regression(cps):
    r = sp.gap_closing(data=cps, y="log_wage", group="female",
                       x=["education", "experience", "tenure"],
                       method="regression")
    assert abs(r.observed_gap - r.counterfactual_gap - r.closed_gap) < 1e-6


def test_gap_closing_aipw(cps):
    r = sp.gap_closing(data=cps, y="log_wage", group="female",
                       x=["education", "experience", "tenure"],
                       method="aipw")
    assert np.isfinite(r.counterfactual_gap)


def test_gap_closing_ipw(cps):
    r = sp.gap_closing(data=cps, y="log_wage", group="female",
                       x=["education", "experience", "tenure"],
                       method="ipw")
    assert np.isfinite(r.counterfactual_gap)


def test_mediation(disparity):
    r = sp.mediation_decompose(
        data=disparity, y="income", treatment="group", mediator="education",
        covariates=["parent_income", "age"])
    assert abs(r.nde + r.nie - r.total) < 1e-6


def test_disparity_jvw(disparity):
    r = sp.disparity_decompose(
        data=disparity, y="income", group="group", mediator="education",
        covariates=["parent_income", "age"])
    assert abs(r.initial_disparity + r.mediator_attributable
               - r.total_disparity) < 1e-6


# ════════════════════════════════════════════════════════════════════════
# Dispatcher
# ════════════════════════════════════════════════════════════════════════

def test_dispatcher_all_methods(cps, disparity):
    """Reach every registered method via the dispatcher."""
    df = cps
    x = ["education", "experience", "tenure"]

    # Mean methods
    sp.decompose("oaxaca", data=df, y="log_wage", group="female", x=x)
    sp.decompose("gelbach", data=df, y="log_wage",
                 base_x=["education"], added_x=["experience", "tenure"])
    sp.decompose("dfl", data=df, y="log_wage", group="female", x=x, stat="mean")
    sp.decompose("ffl", data=df, y="log_wage", group="female", x=x, stat="mean")

    # Quantile methods (keep grids small)
    sp.decompose("mm", data=df, y="log_wage", group="female", x=x,
                 tau_grid=[0.5], n_sim=100, n_tau_qr=11)
    sp.decompose("melly", data=df, y="log_wage", group="female", x=x,
                 tau_grid=[0.5], n_tau_qr=11)
    sp.decompose("cfm", data=df, y="log_wage", group="female", x=x,
                 tau_grid=[0.5], n_thresh=15)

    # Nonlinear
    sp.decompose("fairlie", data=df, y="union", group="female",
                 x=x, n_sim=30)
    sp.decompose("bauer_sinning", data=df, y="union", group="female", x=x)

    # Inequality
    df_wage = df.assign(wage=np.exp(df.log_wage))
    sp.decompose("subgroup", data=df_wage, y="wage", by="female",
                 index="theil_t")
    sp.decompose("shapley_inequality", data=df_wage, y="wage",
                 x=["education", "experience"], index="theil_t")

    # Kitagawa
    df2 = df.assign(educ_bin=(df.education > 12).astype(int))
    sp.decompose("kitagawa", data=df2, rate="union", group="female",
                 by="educ_bin")

    # Causal
    sp.decompose("gap_closing", data=df, y="log_wage", group="female",
                 x=x, method="regression")
    sp.decompose("mediation", data=disparity, y="income", treatment="group",
                 mediator="education", covariates=["parent_income", "age"])
    sp.decompose("causal_jvw", data=disparity, y="income", group="group",
                 mediator="education", covariates=["parent_income", "age"])


def test_dispatcher_unknown_method():
    with pytest.raises(ValueError):
        sp.decompose("nonexistent_method_xyz", data=None)


def test_available_methods_count():
    methods = sp.available_methods()
    # At least 25 method names (core + aliases)
    assert len(methods) >= 25
    for name in ["oaxaca", "dfl", "ffl", "machado_mata", "melly", "cfm",
                 "fairlie", "subgroup", "kitagawa", "gap_closing",
                 "mediation", "causal_jvw"]:
        assert name in methods


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════

def test_weighted_quantile():
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    w = np.array([1, 1, 1, 1, 1], dtype=float)
    assert _common.weighted_quantile(y, 0.5, w) == pytest.approx(3.0, abs=0.5)


def test_logit_fit():
    rng = np.random.default_rng(1)
    n = 500
    X = rng.normal(size=(n, 3))
    X = _common.add_constant(X)
    beta_true = np.array([0.2, 1.0, -0.5, 0.3])
    p = 1 / (1 + np.exp(-(X @ beta_true)))
    y = rng.binomial(1, p)
    beta, vcov = _common.logit_fit(y, X)
    # At least recover signs
    assert np.sign(beta[1]) == 1
    assert np.sign(beta[2]) == -1


def test_result_summary_methods(cps):
    """Every result class should have a working .summary()."""
    res = [
        sp.dfl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience"], stat="mean"),
        sp.ffl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience"], stat="mean"),
        sp.machado_mata(data=cps, y="log_wage", group="female",
                        x=["education"], tau_grid=[0.5], n_sim=50, n_tau_qr=9),
        sp.gap_closing(data=cps, y="log_wage", group="female",
                       x=["education", "experience"], method="regression"),
    ]
    for r in res:
        text = r.summary()
        assert isinstance(text, str) and len(text) > 0


def test_to_latex_methods(cps, disparity):
    # Every result class with a .to_latex() promise — audit them all.
    wage = np.exp(cps["log_wage"]).values
    df_wage = cps.assign(wage=wage)
    sources_df = pd.DataFrame({
        "wage": wage,
        "transfer": np.maximum(0, wage * 0.1 + np.random.default_rng(0).normal(0, 1, len(wage))),
    })
    results = [
        sp.dfl_decompose(data=cps, y="log_wage", group="female",
                         x=["education"], stat="mean"),
        sp.ffl_decompose(data=cps, y="log_wage", group="female",
                         x=["education"], stat="mean"),
        sp.machado_mata(data=cps, y="log_wage", group="female",
                        x=["education"], tau_grid=[0.5], n_sim=50, n_tau_qr=9),
        sp.melly_decompose(data=cps, y="log_wage", group="female",
                           x=["education"], tau_grid=[0.5], n_tau_qr=9),
        sp.cfm_decompose(data=cps, y="log_wage", group="female",
                         x=["education"], tau_grid=[0.5], n_thresh=10),
        sp.fairlie(data=cps, y="union", group="female",
                   x=["education"], n_sim=20),
        sp.bauer_sinning(data=cps, y="union", group="female",
                         x=["education"]),
        sp.subgroup_decompose(data=df_wage, y="wage", by="female",
                              index="theil_t"),
        sp.shapley_inequality(data=df_wage, y="wage",
                              x=["education"], index="theil_t"),
        sp.source_decompose(data=sources_df,
                            sources=["wage", "transfer"]),
        sp.kitagawa_decompose(
            data=cps.assign(e=(cps.education > 12).astype(int)),
            rate="union", group="female", by="e"),
        sp.das_gupta(
            pd.DataFrame({"a": [1.0], "b": [2.0]}),
            pd.DataFrame({"a": [1.5], "b": [1.8]}),
            factor_names=["a", "b"]),
        sp.gap_closing(data=cps, y="log_wage", group="female",
                       x=["education"], method="regression"),
        sp.mediation_decompose(data=disparity, y="income",
                               treatment="group", mediator="education",
                               covariates=["parent_income"]),
        sp.disparity_decompose(data=disparity, y="income", group="group",
                               mediator="education",
                               covariates=["parent_income"]),
    ]
    for r in results:
        latex = r.to_latex()
        assert isinstance(latex, str) and len(latex) > 0, \
            f"{type(r).__name__}.to_latex() returned empty"
        html = r._repr_html_()
        assert "<" in html and ">" in html, \
            f"{type(r).__name__}._repr_html_() not HTML"


def test_repr_html_methods(cps):
    r = sp.dfl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience"], stat="mean")
    html = r._repr_html_()
    assert "<" in html and ">" in html


# ════════════════════════════════════════════════════════════════════════
# Cross-method consistency
# ════════════════════════════════════════════════════════════════════════

def test_dfl_ffl_mean_agree(cps):
    """DFL and FFL should give almost identical mean decompositions
    because at the mean statistic, the RIF linearisation is exact."""
    x = ["education", "experience", "tenure", "union", "married"]
    r_dfl = sp.dfl_decompose(data=cps, y="log_wage", group="female",
                             x=x, stat="mean")
    r_ffl = sp.ffl_decompose(data=cps, y="log_wage", group="female",
                             x=x, stat="mean")
    # FFL's four parts should sum to gap; composition+structure should be
    # close to DFL's (spec + rw errors absorb reweighting imperfection).
    assert abs(r_dfl.gap - r_ffl.gap) < 1e-8
    # FFL structure = DFL structure at the mean (both driven by β)
    assert abs(r_dfl.structure - r_ffl.structure) < 0.05


def test_mm_melly_cfm_aligned_reference(cps):
    """Machado-Mata, Melly, CFM all use the *same* `reference` convention
    (A's β on B's X). Composition at the median should have the same
    sign across all three and be within a reasonable tolerance given the
    different estimation techniques."""
    x = ["education", "experience", "tenure"]
    r_mm = sp.machado_mata(data=cps, y="log_wage", group="female",
                           x=x, tau_grid=[0.5], n_sim=400, n_tau_qr=19)
    r_melly = sp.melly_decompose(data=cps, y="log_wage", group="female",
                                 x=x, tau_grid=[0.5], n_tau_qr=19)
    r_cfm = sp.cfm_decompose(data=cps, y="log_wage", group="female",
                             x=x, tau_grid=[0.5], n_thresh=30)
    comp_mm = r_mm.quantile_grid["composition"].iloc[0]
    comp_melly = r_melly.quantile_grid["composition"].iloc[0]
    comp_cfm = r_cfm.quantile_grid["composition"].iloc[0]
    # All three should have the same sign (not opposite).
    signs = {np.sign(comp_mm), np.sign(comp_melly), np.sign(comp_cfm)}
    signs.discard(0.0)
    assert len(signs) <= 1, (
        f"MM/Melly/CFM composition signs disagree: "
        f"MM={comp_mm:.4f}, Melly={comp_melly:.4f}, CFM={comp_cfm:.4f}"
    )


def test_dfl_mm_reference_convention_opposite(cps):
    """Document the known convention split: DFL(ref=0) composition
    should roughly equal MM(ref=1) composition, because they build
    the same economic counterfactual under opposite ref labels."""
    x = ["education", "experience", "tenure"]
    r_dfl_0 = sp.dfl_decompose(data=cps, y="log_wage", group="female",
                               x=x, stat="quantile", tau=0.5, reference=0)
    r_mm_1 = sp.machado_mata(data=cps, y="log_wage", group="female",
                             x=x, tau_grid=[0.5], reference=1,
                             n_sim=400, n_tau_qr=19)
    comp_dfl = r_dfl_0.composition
    comp_mm = r_mm_1.quantile_grid["composition"].iloc[0]
    # Rough agreement within simulation noise; both should have the same sign
    if comp_dfl != 0:
        ratio = comp_mm / comp_dfl
        assert np.sign(comp_dfl) == np.sign(comp_mm), (
            f"DFL(ref=0) and MM(ref=1) composition disagree in sign: "
            f"DFL={comp_dfl:.4f}, MM={comp_mm:.4f}"
        )
        # And roughly same magnitude (within factor 4, accounting for
        # simulation noise and different estimation approaches)
        assert 0.25 < abs(ratio) < 4.0, (
            f"DFL(ref=0)={comp_dfl:.4f} vs MM(ref=1)={comp_mm:.4f} — "
            "magnitudes too different"
        )
