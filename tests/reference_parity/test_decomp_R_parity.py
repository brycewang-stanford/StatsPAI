"""Decomposition family vs the R packages that define the methods.

Reference: ``_fixtures/decomp_R.json`` written by ``_generate_decomp_R.R``
(DasGuptR 2.2.0, ddecompose 1.0.0, cdgd 1.0.1) from Das Gupta's own worked
examples (exported by the same script) and the simulated
``decomp_gap.csv`` / ``decomp_ye.csv`` (``_generate_decomp_data.py``).

Every comparison is on identical bytes at 1e-10 relative or tighter, except
where a logistic MLE sits in the path (1e-9: IRLS on the R side and Newton
on ours converge to the same optimum by different routes).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "decomp_R.json").read_text(encoding="utf-8"))


def _close(ours, ref, rtol=1e-10, atol=0.0):
    np.testing.assert_allclose(float(ours), float(ref), rtol=rtol, atol=atol)


def _effects(res):
    return dict(zip(res.factor_effects["factor"], res.factor_effects["effect"]))


# --------------------------------------------------------------------------
# Das Gupta / Kitagawa (DasGuptR)
# --------------------------------------------------------------------------


def test_das_gupta_two_factors_matches_dasguptr():
    d = pd.read_csv(_FIX / "decomp_dg2_1.csv")
    f = ["avg_earnings", "earner_prop"]
    r = sp.das_gupta(d[d["pop"] == "black"], d[d["pop"] == "white"], f)
    for k, v in _effects(r).items():
        _close(v, R["dg2_1"][k])


def test_das_gupta_cross_classified_sums_over_strata():
    """Four factors x six age groups: the aggregate is sum_i prod_f f_{f,i}.

    Before 1.28.0 sp.das_gupta multiplied the factor MEANS instead, which
    gave factor A 0% of the gap here (DasGuptR: 37%) and D +333% (-52%).
    """
    d = pd.read_csv(_FIX / "decomp_dg6_5.csv")
    a, b = d[d["pop"] == 1968], d[d["pop"] == 1963]
    r = sp.das_gupta(a, b, ["A", "B", "C", "D"], by="agegroup")
    _close(r.rate_a, R["dg6_5_crude"]["r1968"])
    _close(r.rate_b, R["dg6_5_crude"]["r1963"])
    for k, v in _effects(r).items():
        _close(v, R["dg6_5"][k])


def test_das_gupta_pairs_strata_by_key_not_position():
    d = pd.read_csv(_FIX / "decomp_dg6_5.csv")
    a, b = d[d["pop"] == 1968], d[d["pop"] == 1963]
    shuffled = b.sample(frac=1.0, random_state=0)
    r1 = sp.das_gupta(a, b, ["A", "B", "C", "D"], by="agegroup")
    r2 = sp.das_gupta(a, shuffled, ["A", "B", "C", "D"], by="agegroup")
    np.testing.assert_allclose(
        r1.factor_effects["effect"], r2.factor_effects["effect"], rtol=1e-13
    )


def test_das_gupta_rejects_unpaired_rows():
    d = pd.read_csv(_FIX / "decomp_dg6_5.csv")
    a, b = d[d["pop"] == 1968], d[d["pop"] == 1963]
    with pytest.raises(ValueError, match="different numbers of rows"):
        sp.das_gupta(a, b.iloc[:-1], ["A", "B", "C", "D"])
    with pytest.raises(ValueError, match="same strata"):
        sp.das_gupta(a, b.iloc[:-1], ["A", "B", "C", "D"], by="agegroup")


def test_kitagawa_matches_dasguptr_two_factor():
    d = pd.read_csv(_FIX / "decomp_dg5_1.csv")
    d["g"] = (d["pop"] != 1970).astype(int)  # group 0 = 1970 = "A"
    r = sp.kitagawa_decompose(d, rate="rate", group="g", by="age_group", weights="size")
    _close(r.composition_effect, R["kitagawa"]["size"])
    _close(r.rate_effect, R["kitagawa"]["rate"])
    _close(r.interaction, 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# gap_closing (ddecompose)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gap():
    return pd.read_csv(_FIX / "decomp_gap.csv")


@pytest.mark.parametrize("target, key", [(1, "dfl_ref0"), (0, "dfl_ref1")])
def test_gap_closing_ipw_is_dfl_reweighting(gap, target, key):
    """IPW gap closing is DiNardo-Fortin-Lemieux reweighting.

    Before 1.28.0 the weights were the reciprocal of the density ratio, so
    the reweighted sample moved away from the target distribution.
    """
    r = sp.gap_closing(
        gap,
        "y",
        "group",
        ["x1", "x2"],
        method="ipw",
        target_dist=target,
        inference="none",
        trim=0.0,
    )
    ref = R[key]
    # StatsPAI's gaps are group 0 minus group 1; ddecompose's are 1 minus 0.
    _close(r.observed_gap, -ref["observed"])
    # Either way the reweighted group's counterfactual mean mu_C splits the
    # gap as ddecompose does: counterfactual gap = -structure effect, closed
    # gap = -composition effect.
    _close(r.counterfactual_gap, -ref["structure"], rtol=1e-9)
    _close(r.closed_gap, -ref["composition"], rtol=1e-9)


def test_gap_closing_regression_is_oaxaca_counterfactual(gap):
    r = sp.gap_closing(
        gap,
        "y",
        "group",
        ["x1", "x2"],
        method="regression",
        target_dist=1,
        inference="none",
    )
    ref = R["ob_ref0"]
    _close(r.closed_gap, -ref["composition"])
    _close(r.counterfactual_gap, -ref["structure"])


def test_gap_closing_aipw_is_doubly_robust():
    """Known truth: no group effect given x, so shifting x closes the gap.

    The outcome is quadratic in x and the outcome model linear, so only the
    propensity half of AIPW is right; the estimate must still be ~0. Before
    1.28.0 the reweighting half ran backwards and this held only when the
    outcome model was also right.
    """
    rng = np.random.default_rng(1)
    n = 200_000
    g = rng.integers(0, 2, n)
    x = rng.normal(g * 1.0, 1.0)
    df = pd.DataFrame({"y": x**2 + rng.normal(0, 1, n), "group": g, "x": x})
    for target in (0, 1):
        r = sp.gap_closing(
            df, "y", "group", ["x"], method="aipw", target_dist=target, inference="none"
        )
        assert abs(r.counterfactual_gap) < 0.05, (target, r.counterfactual_gap)
        assert abs(r.observed_gap) > 0.9


# --------------------------------------------------------------------------
# Yu-Elwert (cdgd)
# --------------------------------------------------------------------------


def test_yu_elwert_efficient_matches_cdgd():
    e = pd.read_csv(_FIX / "decomp_ye.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.yu_elwert_decompose(
            e,
            "y",
            "t",
            "r",
            ["x1", "x2"],
            method="efficient",
            inference="analytic",
            trim=0.0,
        )
    ref = R["cdgd"]
    names = {
        "disparity": "total",
        "baseline": "baseline",
        "prevalence": "prevalence",
        "effect": "effect",
        "selection": "selection",
    }
    for ours, theirs in names.items():
        _close(getattr(r, ours), ref["point"][theirs], rtol=1e-9)
        _close(r.se[ours], ref["se"][theirs], rtol=1e-9)


def test_yu_elwert_efficient_components_add_up():
    e = pd.read_csv(_FIX / "decomp_ye.csv")
    r = sp.yu_elwert_decompose(
        e, "y", "t", "r", ["x1", "x2"], method="efficient", inference="none"
    )
    _close(r.baseline + r.prevalence + r.effect + r.selection, r.disparity, rtol=1e-13)


def test_yu_elwert_analytic_needs_efficient():
    e = pd.read_csv(_FIX / "decomp_ye.csv")
    with pytest.raises(ValueError, match="efficient"):
        sp.yu_elwert_decompose(e, "y", "t", "r", ["x1"], inference="analytic")


# --------------------------------------------------------------------------
# RIF regression (rifreg, dineq) and the FFL decomposition (ddecompose)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cps():
    return pd.read_csv(_FIX / "decomp_cps.csv")


_RIF_FM = "log_wage ~ education + experience"


def test_rifreg_variance_matches_rifreg(cps):
    r = sp.rifreg(_RIF_FM, data=cps, statistic="variance")
    np.testing.assert_allclose(r.params.to_numpy(), R["rifreg_variance"], rtol=1e-10)


@pytest.mark.parametrize("i, tau", [(0, 0.1), (1, 0.5), (2, 0.9)])
def test_rifreg_quantile_matches_rifreg(cps, i, tau):
    r = sp.rifreg(
        _RIF_FM, data=cps, statistic="quantile", tau=tau, quantile_convention="rifreg"
    )
    np.testing.assert_allclose(
        r.params.to_numpy(), R["rifreg_quantiles"][i], rtol=1e-10
    )


def test_rifreg_gini_matches_the_exact_gini_rif(cps):
    """dineq's RIF is rifreg's formula with the Lorenz area computed exactly.

    Before 1.28.0 the Gini RIF used the midpoint ECDF with the n/(n-1)
    Gini and averaged to neither Gini; coefficients were ~1% off.
    """
    r = sp.rifreg(_RIF_FM, data=cps, statistic="gini")
    np.testing.assert_allclose(r.params.to_numpy(), R["dineq_gini_lm"], rtol=1e-10)
    # The stock rifreg Gini integrates the Lorenz curve numerically; the gap
    # to the exact value is its quadrature error, not a convention.
    np.testing.assert_allclose(r.params.to_numpy(), R["rifreg_gini_stock"], rtol=1e-4)


def test_gini_rif_averages_to_the_gini(cps):
    from statspai.decomposition._common import gini_population, influence_function

    y = cps["log_wage"].to_numpy()
    _close(influence_function(y, "gini").mean(), gini_population(y), rtol=1e-12)


_FFL_TERMS = {
    "gap": "observed",
    "composition": "composition",
    "structure": "structure",
    "spec_error": "specification",
    "reweight_error": "reweighting",
}


def _ffl(cps, stat, reference, tau=0.5):
    return sp.ffl_decompose(
        cps,
        "log_wage",
        "female",
        ["education", "experience", "tenure"],
        stat=stat,
        tau=tau,
        reference=reference,
        trim=0.0,
        inference="none",
        quantile_convention="rifreg",
    )


@pytest.mark.parametrize("reference, tag", [(1, "ref0"), (0, "ref1")])
@pytest.mark.parametrize(
    "stat, key, tau",
    [
        ("variance", "variance", 0.5),
        ("gini", "gini", 0.5),
        ("quantile", "q10", 0.1),
        ("quantile", "q50", 0.5),
        ("quantile", "q90", 0.9),
    ],
)
def test_ffl_matches_ob_decompose_reweighted(cps, stat, key, tau, reference, tag):
    """Every term of the reweighted RIF decomposition, both directions.

    StatsPAI's reference=1 reweights group 0 onto group 1's covariates, as
    ddecompose's reference_0 = TRUE; StatsPAI's gaps are group 0 minus 1,
    so every term is the negative of ddecompose's. Before 1.28.0 the two
    error terms were swapped and, with reference=1, did not add up.
    """
    r = _ffl(cps, stat, reference, tau)
    ref = R[f"ffl_{key}_{tag}"]
    # 1e-9 relative, with a 1e-12 absolute floor: the error terms are
    # differences of nearly equal products, so the logit MLE's last digits
    # (IRLS in R, Newton here) reach them at ~1e-12 absolute.
    for ours, theirs in _FFL_TERMS.items():
        _close(getattr(r, ours), -ref[theirs], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("reference", [0, 1])
@pytest.mark.parametrize("stat", ["mean", "variance", "gini", "quantile"])
def test_ffl_terms_add_up_to_the_gap(cps, stat, reference):
    r = _ffl(cps, stat, reference)
    total = r.composition + r.structure + r.spec_error + r.reweight_error
    _close(total, r.gap, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("reference, tag", [(1, "ref0"), (0, "ref1")])
@pytest.mark.parametrize(
    "stat, key, tau",
    [
        ("variance", "variance", 0.5),
        ("gini", "gini", 0.5),
        ("quantile", "q10", 0.1),
        ("quantile", "q50", 0.5),
        ("quantile", "q90", 0.9),
    ],
)
def test_dfl_non_mean_statistics_match_ddecompose(cps, stat, key, tau, reference, tag):
    """DFL beyond the mean (Track A module 31_dfl covers the mean).

    stat_convention="hmisc" is ddecompose's weighted variance / quantile;
    the Gini reference is ddecompose run with the exact plug-in Gini, since
    its built-in Gini integrates the Lorenz curve numerically.
    """
    r = sp.dfl_decompose(
        cps,
        "log_wage",
        "female",
        ["education", "experience", "tenure"],
        stat=stat,
        tau=tau,
        reference=reference,
        trim=0.0,
        inference="none",
        stat_convention="hmisc",
    )
    ref = R[f"dfl_{key}_{tag}"]
    # The Gini gap is a difference of two ~0.1 values equal to 1e-3, so
    # identical formulas summed in a different order meet at ~1e-12 absolute.
    _close(r.gap, -ref["observed"], rtol=1e-10, atol=1e-12)
    _close(r.composition, -ref["composition"], rtol=1e-9, atol=1e-12)
    _close(r.structure, -ref["structure"], rtol=1e-9, atol=1e-12)


def test_dfl_stat_conventions_agree_on_unit_weights(cps):
    """The two conventions only differ once weights are non-uniform."""
    kw = dict(stat="variance", trim=0.0, inference="none")
    a = sp.dfl_decompose(cps, "log_wage", "female", ["education"], **kw)
    b = sp.dfl_decompose(
        cps, "log_wage", "female", ["education"], stat_convention="hmisc", **kw
    )
    _close(a.gap, b.gap, rtol=1e-13)
    assert a.composition != b.composition


def test_dfl_rejects_unknown_convention(cps):
    with pytest.raises(Exception, match="convention"):
        sp.dfl_decompose(
            cps,
            "log_wage",
            "female",
            ["education"],
            stat="variance",
            inference="none",
            stat_convention="stata",
        )


# --------------------------------------------------------------------------
# Stata references: b1x2 (Gelbach), ineqdeco, descogini
# --------------------------------------------------------------------------

ST = json.loads((_FIX / "decomp_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stata_data():
    return pd.read_csv(_FIX / "decomp_stata.csv")


@pytest.mark.parametrize(
    "vce, key", [("robust", "gelbach"), ("nonrobust", "gelbach_homoskedastic")]
)
def test_gelbach_matches_b1x2(stata_data, vce, key):
    """Contributions and their joint covariance, as Gelbach's own command.

    Before 1.28.0 the SEs were a two-term delta method that dropped the
    covariance between the auxiliary and long-regression estimates.
    """
    added = ["experience", "tenure", "union"]
    r = sp.gelbach(stata_data, "log_wage", ["education"], added, vce=vce)
    ref = ST[key]
    if vce == "robust":
        np.testing.assert_allclose(
            r.decomposition["delta"], ST["gelbach"]["delta"], rtol=1e-10
        )
        _close(r.total_change, ST["gelbach"]["total"], rtol=1e-10)
    np.testing.assert_allclose(r.vcov.to_numpy(), np.array(ref["V"]), rtol=1e-9)
    np.testing.assert_allclose(
        r.decomposition["se"], np.sqrt(np.diag(ref["V"])), rtol=1e-9
    )
    _close(r.total_se**2, ref["V_total"], rtol=1e-9)


@pytest.mark.parametrize(
    "index, ge", [("theil_l", "ge0"), ("theil_t", "ge1"), ("ge2", "ge2")]
)
def test_subgroup_decompose_matches_ineqdeco(stata_data, index, ge):
    r = sp.subgroup_decompose(stata_data, y="wage", by="female", index=index)
    ref = ST["ineqdeco"]
    _close(r.total, ref[ge], rtol=1e-12)
    _close(r.within, ref[f"within_{ge}"], rtol=1e-12)
    _close(r.between, ref[f"between_{ge}"], rtol=1e-11)


def test_source_decompose_matches_descogini(stata_data):
    src = ["wage", "capital", "transfer"]
    r = sp.source_decompose(stata_data, src, gini="population")
    ref = ST["descogini"]
    _close(r.total_gini, ref["gtotal"], rtol=1e-12)
    for row in r.sources.itertuples():
        s = ref[row.source]
        _close(row.share, s["S"], rtol=1e-12)
        _close(row.gini_k, s["G"], rtol=1e-12)
        _close(row.gini_corr, s["R"], rtol=1e-12)
        _close(row.pct_of_gini / 100.0, s["share"], rtol=1e-12)


def test_source_decompose_conventions_share_every_ratio(stata_data):
    """The n/(n-1) factor cancels from S_k, R_k and each source's share."""
    src = ["wage", "capital", "transfer"]
    a = sp.source_decompose(stata_data, src)
    b = sp.source_decompose(stata_data, src, gini="population")
    n = len(stata_data)
    _close(a.total_gini, b.total_gini * n / (n - 1), rtol=1e-12)
    for col in ("share", "gini_corr", "pct_of_gini"):
        np.testing.assert_allclose(a.sources[col], b.sources[col], rtol=1e-12)


# ---- Yun / Bauer-Sinning nonlinear decomposition vs mvdcmp -----------------

_MV_NAMES = ["education", "experience", "_cons"]


def _mv_check(stata_data, model, rtol, atol):
    r = sp.bauer_sinning(
        stata_data,
        y="union",
        group="female",
        x=["education", "experience"],
        model=model,
    )
    ref = ST[f"mvdcmp_{model}"]
    b = np.array(ref["b"])
    V = np.array(ref["V"])
    _close(r.explained, b[6], rtol=rtol, atol=atol)
    _close(r.unexplained, b[7], rtol=rtol, atol=atol)
    _close(r.gap, b[8], rtol=rtol, atol=atol)
    det = r.detailed.set_index("variable")
    un = r.detailed_unexplained.set_index("variable")
    for i, name in enumerate(_MV_NAMES):
        if name != "_cons":
            _close(det.loc[name, "contribution"], b[i], rtol=rtol, atol=atol)
        _close(un.loc[name, "contribution"], b[3 + i], rtol=rtol, atol=atol)
    labels = [f"explained:{n}" for n in _MV_NAMES] + [
        f"unexplained:{n}" for n in _MV_NAMES
    ]
    np.testing.assert_allclose(
        r.vcov.loc[labels, labels].to_numpy(), V[:6, :6], rtol=rtol, atol=atol
    )
    _close(r.se["explained"] ** 2, V[6, 6], rtol=rtol)
    _close(r.se["unexplained"] ** 2, V[7, 7], rtol=rtol)
    _close(r.se["gap"] ** 2, V[8, 8], rtol=rtol)


def test_bauer_sinning_logit_matches_mvdcmp(stata_data):
    """Point estimates, Yun-weighted detail, delta-method covariance.

    Logit's canonical link lets Stata's default tolerance reach the MLE, so
    this is exact up to the two fits' last digits.
    """
    _mv_check(stata_data, "logit", rtol=1e-9, atol=1e-13)


def test_bauer_sinning_probit_is_aligned_with_mvdcmp(stata_data):
    """Aligned, not bit-exact: mvdcmp's probit stops at Stata's default
    tolerance (see the next test), which moves every term by <= 1e-6."""
    _mv_check(stata_data, "probit", rtol=2e-6, atol=1e-12)


def test_statspai_probit_is_statas_tightly_converged_probit(stata_data):
    from statspai.decomposition.nonlinear import _probit_fit

    fits = ST["probit_fits"]
    for g in (0, 1):
        d = stata_data[stata_data["female"] == g]
        X = np.column_stack([np.ones(len(d)), d["education"], d["experience"]])
        beta, _ = _probit_fit(d["union"].to_numpy(dtype=float), X)
        ours = np.array([beta[1], beta[2], beta[0]])
        np.testing.assert_allclose(ours, fits[f"tight_{g}"], rtol=1e-12)
    # ... and Stata's default fit, which mvdcmp uses, is the one that is short.
    assert not np.allclose(fits["default_1"], fits["tight_1"], rtol=1e-10, atol=0.0)
