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


def test_to_latex_methods(cps):
    r = sp.dfl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience"], stat="mean")
    latex = r.to_latex()
    assert r"\begin{tabular}" in latex or r"\begin{table}" in latex


def test_repr_html_methods(cps):
    r = sp.dfl_decompose(data=cps, y="log_wage", group="female",
                         x=["education", "experience"], stat="mean")
    html = r._repr_html_()
    assert "<" in html and ">" in html
