"""R parity for the methods of Rosenbaum, *An Introduction to the Theory of
Observational Studies* (Springer, 2025).

The book's analyses run on three R packages by its author: ``weightedRank``
(sensitivity analysis in block designs), ``iTOS`` (two-criteria matching,
balance evaluation, Noether's test, amplification) and ``tightenBlock``.
This file pins the StatsPAI counterparts against them.

Data. The committed fixtures are simulated to the shape of the book's
examples (``_fixtures/_generate_rosenbaum_itos_data.R``); the book's own
NHANES extracts are not redistributed. The reference numbers are in
``_fixtures/rosenbaum_itos_R.json``, written by
``_fixtures/_generate_rosenbaum_itos.R``; both sides read the same CSV
bytes. To run every comparison on the book's data instead, write them to a
directory with ``_generate_rosenbaum_itos_data.R --book DIR`` and set
``STATSPAI_ITOS_FIXTURES=DIR`` for the reference script and for pytest. The
numbers that run produced are in
``docs/dev/2026-10-06-rosenbaum-itos-review.md``.

Tolerances. Everything except the permutation benchmark is deterministic,
and the budget is the strict one of ``CLAUDE.md`` section 5.1 (relative
1e-6); deviates and moments agree to about 1e-12. The places where the
comparison is something other than "same number" are these, each with its
mechanism:

* p-values. R forms ``1 - pnorm(z)``, which keeps about ``1e-16 / p``
  relative digits; the deviates are compared tightly and the p-values with
  an absolute floor.
* ``wgtRankCI`` solves for the interval ends by bisection to ``1e-5`` on a
  deviate rounded to five decimals; the ends here are solved to ``1e-11``
  and are compared to ``2e-5``.
* ``estPower`` with ``ssratio != 1`` divides the bounding standard
  deviation by the ratio instead of its square root. The value here follows
  the variance; R's number is rebuilt from our quantities to show that this
  is the only difference, and ``test_power_estimate_tracks_simulated_power``
  shows which of the two the truth agrees with.
* ``makematch`` hands its costs to ``rcbalance::callrelax``, which
  truncates them to integers. On truncated costs the optimal objective here
  equals R's exactly; on the costs as given it is lower than R's, and equals
  that of an independent LP solver.
* ``senstrat`` with its default ``method="BU"`` takes hypergeometric moments
  from ``BiasedUrn`` at precision ``1e-7``. Its exact ``method="RK"`` is
  matched to 1e-9; the ``"BU"`` rows are held to 1e-6.
* ``wgtRankC`` and ``gwgtRankC`` are two formulations of the conditional
  test that coincide when no outcomes are tied and differ slightly when
  some are. ``conditional=True`` is the second; the first is matched on
  untied data.
* ``evalBal`` permutes with R's generator. The p-values of the matched
  sample are deterministic and matched; the shares of better-balanced
  simulated experiments are compared within Monte Carlo error only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

FIX = Path(
    os.environ.get("STATSPAI_ITOS_FIXTURES", Path(__file__).parent / "_fixtures")
)
TIGHT = 1e-9
PS_X = [
    "age",
    "female",
    "education",
    "smokenow",
    "smokeQuit",
    "bpRX",
    "bmi",
    "vigor",
    "waisthip",
]


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(FIX / name, float_precision="round_trip")


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "rosenbaum_itos_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def hdl():
    return _csv("rosenbaum_itos_hdl.csv").to_numpy(dtype=float)


@pytest.fixture(scope="module")
def bp():
    return _csv("rosenbaum_itos_bp.csv")[["b", "n", "p"]].to_numpy(dtype=float)


@pytest.fixture(scope="module")
def peri():
    return _csv("rosenbaum_itos_peri.csv")


@pytest.fixture(scope="module")
def peri_blocks(peri):
    pairs = peri[peri["pair"] == 1]
    return (
        pairs["pd"].to_numpy().reshape(-1, 4),
        pairs["z"].to_numpy(dtype=float).reshape(-1, 4),
    )


@pytest.fixture(scope="module")
def binge():
    return _csv("rosenbaum_itos_binge.csv")


def _p_close(ours: float, theirs: float) -> bool:
    """p-values agree to 1e-6 relative, above the floor that ``1 - pnorm``
    leaves in the reference."""
    return abs(ours - theirs) <= 1e-6 * theirs + 2e-15


# ------------------------------------------------------------------ wgtRank


@pytest.mark.parametrize(
    "phi", ["wilc", "quade", "u868", "u878", "u888", "u858", "mixed"]
)
@pytest.mark.parametrize("gamma", [1, 2, 3.5, 5])
def test_weighted_rank_one_treated_per_block(ref, hdl, phi, gamma):
    r = ref["wgtRank"][f"{phi} {gamma}"]
    res = sp.weighted_rank(hdl, gamma=gamma, phi=phi)
    n = hdl.shape[0]
    assert res.deviate == pytest.approx(r["Deviate"], rel=TIGHT)
    # wgtRank reports the statistic as a mean over blocks
    assert res.statistic / n == pytest.approx(r["Statistic"], rel=TIGHT)
    assert res.expectation / n == pytest.approx(r["Expectation"], rel=TIGHT)
    assert res.variance / n**2 == pytest.approx(r["Variance"], rel=TIGHT)
    assert _p_close(res.pvalue, r["pval"])


def test_weighted_rank_pairs_and_named_triple_agree(ref, bp):
    r = ref["wgtRank_pairs"]
    a = sp.weighted_rank(bp[:, :2], gamma=2.3, phi="u878")
    b = sp.weighted_rank(bp[:, :2], gamma=2.3, phi=(8, 7, 8))
    assert a.deviate == pytest.approx(r["Deviate"], rel=TIGHT)
    assert a.deviate == b.deviate


@pytest.mark.parametrize("phi", ["wilc", "quade", "u868", "u878"])
@pytest.mark.parametrize("gamma", [1, 2])
@pytest.mark.parametrize("alt", ["greater", "twosided", "less"])
def test_weighted_rank_estimates(ref, hdl, phi, gamma, alt):
    r = ref["wgtRankCI"][f"{phi} {gamma} {alt}"]
    alternative = "two-sided" if alt == "twosided" else alt
    res = sp.weighted_rank(
        hdl, gamma=gamma, phi=phi, alternative=alternative, estimates=True
    )
    assert res.estimate == pytest.approx(r["estimate"], abs=2e-5)
    for ours, theirs in zip(res.conf_int, r["confidence"]):
        if isinstance(theirs, str):
            assert np.isinf(ours) and (ours > 0) == (theirs == "Inf")
        else:
            assert ours == pytest.approx(theirs, abs=2e-5)


@pytest.mark.parametrize(
    "key, phis, gamma",
    [
        ("u868 u878 3", ["u868", "u878"], 3),
        ("u868 u878 5", ["u868", "u878"], 5),
        ("wilc mixed 4", ["wilc", "mixed"], 4),
    ],
)
def test_weighted_rank_adaptive(ref, hdl, key, phis, gamma):
    r = ref["wgtRanktt"][key]
    res = sp.weighted_rank(hdl, gamma=gamma, phi=phis)
    assert res.pvalue == pytest.approx(r["jointP"], rel=1e-8)
    assert res.diagnostics["correlation"][0][1] == pytest.approx(r["cor12"], rel=TIGHT)
    for (_, row), theirs in zip(res.detail.iterrows(), r["detail"]):
        assert row["deviate"] == pytest.approx(theirs[1], rel=TIGHT)
    # the joint p-value pays for the choice
    assert res.pvalue >= res.detail["pvalue"].min()


def test_weighted_rank_scores_gap_and_lower_tail(ref, bp):
    r = ref["dwgtRank"]
    n = bp.shape[0]
    rev = bp[:, ::-1]
    cases = {
        "second_factor": sp.weighted_rank(
            rev,
            gamma=1.45,
            phi=(8, 8, 8),
            alternative="less",
            scores=(1, 2, 5),
            block_scale="gap",
        ),
        "stratified_wilcoxon": sp.weighted_rank(
            rev, gamma=1.45, phi=(1, 1, 1), alternative="less", scores=(1, 2, 3)
        ),
        "pairs": sp.weighted_rank(bp[:, :2], gamma=2.3, phi=(8, 7, 8)),
        "gap": sp.weighted_rank(bp, gamma=1.5, phi=(8, 6, 8), block_scale="gap"),
        "less": sp.weighted_rank(bp, gamma=1.5, phi=(8, 6, 8), alternative="less"),
    }
    for key, res in cases.items():
        assert res.deviate == pytest.approx(r[key]["Deviate"], rel=TIGHT), key
        assert res.statistic / n == pytest.approx(r[key]["Statistic"], rel=TIGHT), key
        assert _p_close(res.pvalue, r[key]["pval"]), key


@pytest.mark.parametrize("phi", ["wilc", "quade", "u868", "u878"])
@pytest.mark.parametrize("gamma", [1, 2, 4])
def test_weighted_rank_two_treated_per_block(ref, peri_blocks, phi, gamma):
    """``gwgtRank`` goes through ``senstrat(method="BU")``, hence 1e-6."""
    y, z = peri_blocks
    r = ref["gwgtRank"][f"{phi} {gamma}"]
    res = sp.weighted_rank(y, treated=z, gamma=gamma, phi=phi)
    assert res.deviate == pytest.approx(r["detail"]["Deviate"], rel=1e-6)
    assert res.diagnostics["n_treated_per_block"] == [2]


def test_weighted_rank_long_format_and_taylor_bound(ref, peri):
    r = ref["gwgtRank"]["taylor"]
    pairs = peri[peri["pair"] == 1]
    sep = sp.weighted_rank("pd", data=pairs, treat="z", block="block", gamma=3)
    tay = sp.weighted_rank(
        "pd", data=pairs, treat="z", block="block", gamma=3, bound="taylor"
    )
    assert sep.deviate == pytest.approx(r["sep"]["Deviate"], rel=1e-6)
    assert tay.deviate == pytest.approx(r["lin"]["Deviate"], rel=1e-6)
    assert tay.pvalue >= sep.pvalue - 1e-15


@pytest.mark.parametrize(
    "key, gamma, phi, alt",
    [
        ("m20_g11", 11, (20, 19, 20), "greater"),
        ("u878_g7", 7, "u878", "greater"),
        ("u222_g45", 4.5, (2, 2, 2), "greater"),
        ("mixed_g3", 3, "mixed", "greater"),
        ("less_g2", 2, "u878", "less"),
    ],
)
def test_weighted_rank_conditional(ref, peri_blocks, key, gamma, phi, alt):
    y, z = peri_blocks
    r = ref["gwgtRankC"][key]
    res = sp.weighted_rank(
        y, treated=z, gamma=gamma, phi=phi, conditional=True, alternative=alt
    )
    assert res.statistic == pytest.approx(r["detail"]["T"], rel=TIGHT)
    assert res.expectation == pytest.approx(r["detail"]["E(T)"], rel=TIGHT)
    assert res.variance == pytest.approx(r["detail"]["var(T)"], rel=TIGHT)
    assert res.deviate == pytest.approx(r["detail"]["Deviate"], rel=TIGHT)
    assert res.diagnostics["n_decisive"] == r["counts"]["Decisive"]
    assert res.diagnostics["n_blocks_tied_extremes"] == r["counts"]["Relevant Ties"]


def test_weighted_rank_conditional_sets_and_ties(ref, peri, hdl, bp):
    sets = peri[peri["pair"] == 0]
    r = ref["gwgtRankC"]["sets_g89"]
    res = sp.weighted_rank(
        "pd", data=sets, treat="z", block="block", gamma=8.9, phi="u878",
        conditional=True,
    )  # fmt: skip
    assert res.deviate == pytest.approx(r["detail"]["Deviate"], rel=TIGHT)

    tied = sp.weighted_rank(hdl, gamma=6, phi="u878", conditional=True)
    assert tied.deviate == pytest.approx(
        ref["gwgtRankC"]["hdl_g6"]["detail"]["Deviate"], rel=TIGHT
    )
    # wgtRankC ranks within the extremes instead; with ties it is another test
    assert tied.diagnostics["n_blocks_tied_extremes"] > 0
    assert tied.pvalue != pytest.approx(ref["wgtRankC_tied"]["pval"], rel=1e-4)
    untied = sp.weighted_rank(bp, gamma=1.8, phi="u878", conditional=True)
    assert untied.deviate == pytest.approx(ref["wgtRankC_untied"]["deviate"], rel=TIGHT)
    assert untied.diagnostics["n_blocks_tied_extremes"] == 0


# -------------------------------------------------- evidence factors, tools


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("g23_u145", dict(gamma=2.3, upsilon=1.45)),
        ("g26_u17", dict(gamma=2.6, upsilon=1.7)),
        ("fisher", dict(gamma=2.6, upsilon=1.7, trunc=1.0)),
        (
            "range_123",
            dict(
                gamma=2,
                upsilon=2,
                phi=((8, 6, 8), (8, 6, 8)),
                scores=(1, 2, 3),
                block_scale="range",
            ),
        ),
        ("less", dict(gamma=1.5, upsilon=1.2, alternative="less")),
    ],
)
def test_evidence_factors(ref, bp, key, kwargs):
    r = ref["ef2C"][key]["pvals"]
    res = sp.evidence_factors(bp, **kwargs)
    assert _p_close(res.pvalue_treated_vs_control1, r["TreatedVSControl1"])
    assert _p_close(res.pvalue_control2_vs_others, r["Control2vsOthers"])
    assert res.pvalue == pytest.approx(r["Combined"], rel=1e-6, abs=2e-15)


def test_truncated_product_and_amplify(ref):
    r = ref["truncatedP"]
    assert sp.truncated_product([0.01, 0.3]) == pytest.approx(r["a"], rel=TIGHT)
    assert sp.truncated_product([0.01, 0.3, 0.15, 0.04], trunc=0.1) == pytest.approx(
        r["b"], rel=TIGHT
    )
    assert sp.truncated_product([0.5, 0.3]) == r["c"] == 1
    assert sp.truncated_product([0.01, 0.3, 0.6], trunc=1) == pytest.approx(
        r["d"], rel=TIGHT
    )
    assert sp.truncated_product([0.19, 0.02, 0.2, 0.7, 0.0004]) == pytest.approx(
        r["e"], rel=TIGHT
    )
    # trunc = 1 is Fisher's method
    assert sp.truncated_product([0.01, 0.3, 0.6], trunc=1) == pytest.approx(
        stats.combine_pvalues([0.01, 0.3, 0.6], method="fisher").pvalue, rel=1e-12
    )
    a = ref["amplify"]
    assert sp.amplify(4, 7) == a["a"] == 9
    np.testing.assert_allclose(sp.amplify(4, np.arange(5, 20)), a["b"], rtol=1e-14)
    assert sp.amplify(1.45, 2.5) == pytest.approx(a["c"], rel=1e-14)


def test_noether(ref):
    d = _csv("rosenbaum_itos_pairs.csv")["d"].to_numpy()
    r = ref["noether"]
    cases = {
        "sign": sp.noether_test(d, f=0, gamma=3),
        "top_third": sp.noether_test(d, gamma=3),
        "two_sided": sp.noether_test(d, f=1 / 3, alternative="two-sided"),
        "less": sp.noether_test(-d, gamma=2, alternative="less"),
    }
    for key, res in cases.items():
        assert res.diagnostics["n_pairs_used"] == r[key]["number.pairs"], key
        assert res.statistic == r[key]["positive.pairs"], key
        assert res.pvalue == pytest.approx(r[key]["pval"], rel=1e-10), key
    # the bound crosses alpha exactly at gamma_critical
    top = cases["top_third"]
    at = sp.noether_test(d, gamma=top.gamma_critical).pvalue
    assert at == pytest.approx(0.05, rel=1e-5)


# ------------------------------------------------------------------- power


@pytest.mark.parametrize("phi", ["wilc", "quade", "u868", "u878", "u888", "mixed"])
def test_weighted_rank_power(ref, hdl, phi):
    r = ref["estPower"][phi]
    res = sp.weighted_rank_power(hdl, [1, 2, 3, 4, 5, 7, 9], phi=phi)
    np.testing.assert_allclose(res.power["power"], r["power"], rtol=1e-8, atol=1e-13)
    assert res.mean == pytest.approx(r["jackm"], rel=TIGHT)
    assert res.variance == pytest.approx(r["jackv"], rel=TIGHT)


def test_weighted_rank_power_sample_ratio_differs_from_estpower_by_one_term(ref, hdl):
    """R divides the bounding sd by the ratio, not by its square root."""
    gammas = [1, 2, 3, 4, 5, 7, 9]
    s = 1000 / 406
    ours = sp.weighted_rank_power(hdl, gammas, phi="u868", sample_ratio=s)
    theirs = np.array(ref["estPower"]["u868_ratio"]["power"])
    n = hdl.shape[0]
    crit = stats.norm.isf(0.05)
    rebuilt = []
    for g in gammas:
        w = sp.weighted_rank(hdl, gamma=g, phi="u868")
        e_bar, sd_bar = w.expectation / n, np.sqrt(w.variance) / n
        z = (e_bar - ours.mean + crit * sd_bar / s) / np.sqrt(ours.variance / s)
        rebuilt.append(stats.norm.sf(z))
    np.testing.assert_allclose(rebuilt, theirs, rtol=1e-8, atol=1e-13)
    # more blocks, so the wrong scaling flatters the power
    assert np.all(ours.power["power"].to_numpy() <= theirs + 1e-12)
    assert np.max(theirs - ours.power["power"].to_numpy()) > 0.02


# ------------------------------------------------------- stratified bounds


@pytest.mark.parametrize("gamma", [1, 1.5, 3])
@pytest.mark.parametrize("alt", ["greater", "less"])
def test_rosenbaum_stratified_small_blocks(ref, peri, gamma, alt):
    """819 blocks of two shapes, exact moments on both sides."""
    r = ref["senstrat_blocks"][f"{gamma} {alt}"]
    kw = dict(gamma=gamma, alternative=alt)
    tay = sp.rosenbaum_stratified(peri, "pd", "z", "block", **kw).detail.iloc[0]
    sep = sp.rosenbaum_stratified(
        peri, "pd", "z", "block", bound="separable", **kw
    ).detail.iloc[0]
    assert tay["deviate"] == pytest.approx(r["lin"]["Deviate"], rel=TIGHT)
    assert tay["variance"] == pytest.approx(r["lin"]["Variance"], rel=TIGHT)
    assert sep["deviate"] == pytest.approx(r["sep"]["Deviate"], rel=TIGHT)
    assert sep["statistic"] == pytest.approx(r["sep"]["Statistic"], rel=1e-12)


def _binge_never(binge):
    d = binge[binge["AlcGroup"] != "P"].copy()
    d["z"] = (d["AlcGroup"] == "B").astype(int)
    d["stratum"] = d["ageC"].astype(str) + "_" + d["female"].astype(str)
    return d


def test_rosenbaum_stratified_large_strata(ref, binge):
    """Against the exact ``method="RK"``, on the rows it was run on."""
    d = _binge_never(binge).iloc[:1200]
    exact = ref["senstrat_strata"]["rank 1.3 RK"]
    d = d.assign(score=stats.rankdata(d["bpCombined"]))
    res = sp.rosenbaum_stratified(d, "score", "z", "stratum", gamma=1.3, score="raw")
    row = res.detail.iloc[0]
    assert res.diagnostics["n_strata"] == 8
    assert row["deviate"] == pytest.approx(exact["lin"]["Deviate"], rel=TIGHT)
    assert row["expectation"] == pytest.approx(exact["lin"]["Expected"], rel=TIGHT)
    assert row["variance"] == pytest.approx(exact["lin"]["Variance"], rel=TIGHT)
    # With eight large strata the separable approximation is optimistic,
    # which is what the Taylor bound is for.
    assert row["pvalue_taylor"] >= row["pvalue_separable"]
    sep = sp.rosenbaum_stratified(
        d, "score", "z", "stratum", gamma=1.3, score="raw", bound="separable"
    ).detail.iloc[0]
    assert sep["deviate"] == pytest.approx(exact["sep"]["Deviate"], rel=TIGHT)
    assert sep["deviate"] >= row["deviate"]


@pytest.mark.parametrize("gamma", [1.1, 1.3])
@pytest.mark.parametrize(
    "score, key", [("rank", "rank"), ("aligned_rank", "aligned"), ("raw", "raw")]
)
def test_rosenbaum_stratified_scores(ref, binge, gamma, score, key):
    """Against ``senstrat(method="BU")``: BiasedUrn's 1e-7 precision."""
    d = _binge_never(binge)
    r = ref["senstrat_strata"][f"{key} {gamma}"]
    row = sp.rosenbaum_stratified(
        d, "bpCombined", "z", "stratum", gamma=gamma, score=score
    ).detail.iloc[0]
    assert row["statistic"] == pytest.approx(r["lin"]["Statistic"], rel=1e-12)
    assert row["deviate"] == pytest.approx(r["lin"]["Deviate"], rel=1e-6)


@pytest.mark.parametrize("gamma", [1, 1.2, 2])
@pytest.mark.parametrize("alt", ["greater", "less"])
def test_rosenbaum_stratified_single_stratum(ref, binge, gamma, alt):
    """One stratum is ``sen2sample``: the worst case is searched directly."""
    d = _binge_never(binge).iloc[:300]
    r = ref["sen2sample"][f"{gamma} {alt}"]
    res = sp.rosenbaum_stratified(d, "bpCombined", "z", gamma=gamma, alternative=alt)
    row = res.detail.iloc[0]
    assert row["deviate"] == pytest.approx(r["detail"]["Deviate"], rel=TIGHT)
    assert row["variance"] == pytest.approx(r["detail"]["Variance"], rel=TIGHT)
    assert _p_close(res.pvalue, r["pval"])


# ------------------------------------------------------------ matched pairs


@pytest.mark.parametrize("gamma", [1, 1.5, 2, 2.5])
def test_pairs_agree_across_functions(ref, bp, gamma):
    """For pairs with equal block weights the weighted rank test is the
    signed-rank test, and ``sp.rosenbaum_bounds`` already matched DOS2."""
    r = ref["pairs"][str(gamma)]
    old = sp.rosenbaum_bounds(bp[:, 0], bp[:, 1], gamma_grid=[gamma])
    assert old.pvalue_upper[0] == pytest.approx(r["pval_greater"], rel=1e-8)
    new = sp.weighted_rank(bp[:, :2], gamma=gamma, phi="wilcoxon")
    assert _p_close(new.pvalue, r["wgt_wilc"])
    u = sp.weighted_rank(bp[:, :2], gamma=gamma, phi=(8, 7, 8))
    assert _p_close(u.pvalue, r["senU"])


# ---------------------------------------------------------------- matching


def _versus(binge, group):
    d = binge[binge["AlcGroup"].isin(["B", group])].copy()
    d["z"] = (d["AlcGroup"] == "B").astype(int)
    d = d.sort_values(["z", "SEQN"], ascending=[False, True])
    return d.set_index("SEQN", drop=False)


def test_cost_terms(ref, binge):
    from statspai.matching.two_criteria import _term_cost

    r = ref["matching"]["terms"]
    six = binge[binge["SEQN"].isin(r["seqn"])].sort_values("SEQN")
    treated = (six["AlcGroup"] == "B").to_numpy()
    cases = {
        "mahalanobis": dict(type="mahalanobis", on=["age", "female"]),
        "near_exact": dict(type="near_exact", on="female"),
        "caliper_one_step": dict(type="caliper", on="age", width=10, two_step=False),
        "caliper": dict(type="caliper", on="age", width=10),
        "caliper_asymmetric": dict(type="caliper", on="age", width=(-2, 10)),
        "caliper_default": dict(type="caliper", on="age"),
        "integer": dict(type="integer", on="education", penalty=3),
    }
    for key, term in cases.items():
        np.testing.assert_allclose(
            _term_cost(term, six, treated), np.array(r[key]), rtol=1e-12, atol=1e-12
        )
    past = binge[binge["AlcGroup"] != "N"]
    q = _term_cost(
        dict(type="quantile", on="age", probs=[0.25, 0.5, 0.75], penalty=5),
        past,
        (past["AlcGroup"] == "B").to_numpy(),
    )
    assert q.sum() == r["quantile_sum"]


def test_propensity_caliper_with_fine_balance(ref, binge):
    """The design of Section 4.3 of the book: all costs are integers, so the
    two optimal objectives are the same number."""
    r = ref["matching"]
    d = _versus(binge, "N")
    from statspai.matching.two_criteria import _fit_pscore

    d["pscore"] = _fit_pscore(d, (d["z"] == 1).to_numpy(), PS_X)
    np.testing.assert_allclose(
        d["pscore"].iloc[:5], r["propensity"]["p_head"], rtol=1e-9
    )
    d["pcat"] = sum((d["pscore"] > c).astype(int) for c in (0.05, 0.1, 0.15, 0.2))
    terms = dict(
        ps="pscore",
        pair=[dict(type="caliper", on="pscore", penalty=10)],
        balance=[dict(type="integer", on="pcat")],
    )
    one = sp.two_criteria_match(d, "z", **terms)
    assert (
        one.total_cost == r["propensity"]["left_true"] + r["propensity"]["right_true"]
    )
    assert one.n_sets == int(d["z"].sum()) and one.n_unmatched == 0
    if one.balance_cost == 0:
        # fine balance: the categories of the score have the same counts
        m = one.matched
        assert (
            m[m.z == 1]["pcat"].value_counts().sort_index().tolist()
            == m[m.z == 0]["pcat"].value_counts().sort_index().tolist()
        )
    two = sp.two_criteria_match(d, "z", ratio=2, **terms)
    assert two.total_cost == pytest.approx(
        r["propensity_1to2"]["left_true"] + r["propensity_1to2"]["right_true"]
    )
    assert two.matched.groupby("mset").size().eq(3).all()


def _book_terms():
    pair = [
        dict(type="integer", on="ageC", penalty=100),
        dict(type="near_exact", on="female", penalty=10000),
        dict(type="near_exact", on="bpRX", penalty=10000),
        dict(type="near_exact", on="vigor", penalty=10),
        dict(type="integer", on="smokenow", penalty=10),
        dict(
            type="mahalanobis",
            on=["age", "bpRX", "female", "education", "smokenow", "smokeQuit",
                "bmi", "vigor", "waisthip"],
        ),
    ]  # fmt: skip
    balance = [
        dict(type="mahalanobis", on=["female", "age", "pscore"]),
        dict(type="integer", on="education", penalty=10),
        dict(type="integer", on="smokenow", penalty=1000),
        dict(type="integer", on="smokeQuit", penalty=10),
        dict(type="caliper", on="pscore", width=(-1, 0.03), penalty=10),
    ]
    return pair, balance


def _lp_objective(pair, balance, use, ratio=1):
    """The same network as a linear program, solved by HiGHS."""
    from scipy import sparse
    from scipy.optimize import linprog

    n_t, n_c = pair.shape
    nx = n_t * n_c
    cost = np.concatenate([pair.ravel(), use, balance.ravel()])
    t_of = np.repeat(np.arange(n_t), n_c)
    c_of = np.tile(np.arange(n_c), n_t)
    ix, ic = np.arange(nx), np.arange(n_c)
    rows = [t_of, n_t + c_of, n_t + ic, n_t + n_c + ic, n_t + n_c + c_of,
            n_t + 2 * n_c + t_of]  # fmt: skip
    cols = [ix, ix, nx + ic, nx + ic, nx + n_c + ix, nx + n_c + ix]
    vals = [np.ones(nx), np.ones(nx), -np.ones(n_c), np.ones(n_c), -np.ones(nx),
            np.ones(nx)]  # fmt: skip
    a_eq = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(2 * n_t + 2 * n_c, 2 * nx + n_c),
    )
    b_eq = np.concatenate([np.full(n_t, ratio), np.zeros(2 * n_c), np.full(n_t, ratio)])
    return linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=(0, 1), method="highs").fun


@pytest.mark.parametrize("group, key", [("N", "never"), ("P", "past")])
def test_book_match(ref, binge, group, key):
    """The cost structure of the two matches behind the book's ``bingeM``:
    eleven terms, and for the smaller reservoir a cost on unlike controls."""
    from statspai.matching.two_criteria import _fit_pscore, _sum_terms

    r = ref["matching"][key]
    d = _versus(binge, group)
    treated = (d["z"] == 1).to_numpy()
    d["pscore"] = _fit_pscore(d, treated, PS_X)
    pair_terms, balance_terms = _book_terms()
    pair = _sum_terms(pair_terms, d, treated, "pair")
    balance = _sum_terms(balance_terms, d, treated, "balance")
    assert pair.sum() == pytest.approx(r["left_sum"], rel=1e-10)
    assert balance.sum() == pytest.approx(r["right_sum"], rel=1e-10)
    use = None
    if group == "P":
        use = ((d["pscore"][~treated] < 0.4) & (d["age"][~treated] > 42)) * 1000.0
        use = use.to_numpy()

    # (a) R truncates costs to integers; on those costs the optimum is R's.
    cut = sp.two_criteria_match(
        d, "z", pair=np.floor(pair), balance=np.floor(balance), control_cost=use
    )
    assert cut.total_cost == r["objective_truncated"]

    # (b) On the costs as given the optimum is no higher than R's solution,
    # priced at those costs ...
    fit = sp.two_criteria_match(
        d, "z", ps="pscore", pair=pair_terms, balance=balance_terms, control_cost=use
    )
    theirs = r["left_true"] + r["right_true"] + r["cc"]
    assert fit.total_cost <= theirs + 1e-9
    # ... and is the optimum: an LP solver reaches the same value.
    if group == "P":
        assert fit.total_cost == pytest.approx(
            _lp_objective(pair, balance, use), rel=1e-10
        )
    # (c) every treated unit is matched, and the score is better balanced
    assert fit.n_sets == int(treated.sum()) and fit.n_unmatched == 0
    assert abs(fit.balance.loc["pscore", "smd_after"]) < abs(
        fit.balance.loc["pscore", "smd_before"]
    )


@pytest.mark.parametrize("ratio", [1, 2, 3])
def test_solver_is_optimal_for_several_controls(binge, ratio):
    from statspai.matching.two_criteria import _fit_pscore, _sum_terms

    d = _versus(binge, "N")
    n_treated = int(d["z"].sum())
    d = pd.concat([d.iloc[:60], d.iloc[n_treated : n_treated + 500]])
    treated = (d["z"] == 1).to_numpy()
    d["pscore"] = _fit_pscore(d, treated, PS_X)
    pair_terms, balance_terms = _book_terms()
    pair = _sum_terms(pair_terms, d, treated, "pair")
    balance = _sum_terms(balance_terms, d, treated, "balance")
    fit = sp.two_criteria_match(d, "z", pair=pair, balance=balance, ratio=ratio)
    lp = _lp_objective(pair, balance, np.zeros(pair.shape[1]), ratio)
    assert fit.total_cost == pytest.approx(lp, rel=1e-10)
    assert fit.matched.groupby("mset").size().eq(ratio + 1).all()
    assert fit.pairs["control"].is_unique


# -------------------------------------------------------------- tightening


def _floor_costs(monkeypatch):
    """Make the solver see truncated costs, as R's does."""
    from statspai.matching import two_criteria as tc

    real = tc.solve_two_criteria

    def cut(pair, balance, use, skip, ratio):
        return real(np.floor(pair), np.floor(balance), np.floor(use), skip, ratio)

    monkeypatch.setattr(tc, "solve_two_criteria", cut)


def test_tighten_blocks_bmi(ref, monkeypatch):
    """1-to-3 blocks tightened to 1-to-2 with BMI balanced."""
    r = ref["tighten"]["bmi_1to2"]
    d = _csv("rosenbaum_itos_blocks.csv").set_index("SEQN", drop=False)
    d["bmicat"] = sum((d["bmi"] > c).astype(int) for c in (22.5, 27.5, 32.5))
    kw = dict(covariates=["age", "education"], fine_balance=["ibmi", "bmicat"], ratio=2)
    fit = sp.tighten_blocks(d, "z", "block", **kw)
    assert fit.diagnostics["block_penalty"] == r["block_penalty"]
    n_blocks = ref["tighten"]["n_blocks"]
    assert fit.n_sets == n_blocks and len(fit.matched) == 3 * n_blocks
    assert fit.matched.groupby("mset")["block"].nunique().eq(1).all()
    theirs = r["left_true"] + r["right_true"]
    assert fit.total_cost <= theirs + 1e-9
    _floor_costs(monkeypatch)
    cut = sp.tighten_blocks(d, "z", "block", **kw)
    pair_trunc = np.floor(cut.pairs["pair_cost"]).sum()
    assert pair_trunc + cut.balance_cost == r["objective_truncated"]


@pytest.mark.parametrize(
    "key, subset", [("dentist_all", None), ("dentist_150", 150), ("dentist_50", 50)]
)
def test_tighten_blocks_subset(ref, monkeypatch, key, subset):
    """Dropping blocks that cannot be balanced, at three prices."""
    r = ref["tighten"][key]
    d = _csv("rosenbaum_itos_dentist.csv").set_index("SEQN", drop=False)
    assert [len(d), int(d["z"].sum())] == ref["tighten"]["dentist_n"]
    _floor_costs(monkeypatch)
    fit = sp.tighten_blocks(
        d, "z", "block",
        covariates=["age", "education"], fine_balance=["education"],
        subset_cost=subset,
    )  # fmt: skip
    assert fit.n_unmatched == r["n_skipped"]
    skipped = float(subset or 0) * fit.n_unmatched
    assert (
        np.floor(fit.pairs["pair_cost"]).sum() + fit.balance_cost + skipped
        == r["objective_truncated"] + skipped
    )


# ------------------------------------------------------------ balance check


def _matched_sample(binge):
    bm = _csv("rosenbaum_itos_matched.csv")
    return bm.merge(binge.drop(columns=["AlcGroup"]), on="SEQN", how="left")


BAL_X = ["age", "female", "education", "bmi", "waisthip", "vigor", "smokenow",
         "bpRX", "smokeQuit"]  # fmt: skip


@pytest.mark.parametrize(
    "key, kwargs, past_only",
    [
        ("auto", dict(), True),
        ("fisher_all", dict(trunc=1.0), False),
        ("t", dict(test="t"), True),
        ("w", dict(test="wilcoxon"), True),
        ("five_levels", dict(max_levels=5), True),
    ],
)
def test_balance_vs_randomization_actual(ref, binge, key, kwargs, past_only):
    r = ref["evalBal"][key]["actual"]
    m = _matched_sample(binge)
    if past_only:
        m = m[m["AlcGroup"] != "N"]
    res = sp.balance_vs_randomization(m, "z", BAL_X, n_sim=50, random_state=0, **kwargs)
    for c in BAL_X:
        assert res.table.loc[c, "actual"] == pytest.approx(r[c], rel=1e-8), c
    assert res.table.loc["min_p", "actual"] == pytest.approx(r["minP"], rel=1e-8)
    assert res.table.loc["truncated_product", "actual"] == pytest.approx(
        r["tProduct"], rel=1e-8
    )
    assert res.table.loc["n_below_alpha", "actual"] == r["nLEalpha"]


def test_balance_vs_randomization_shares_within_monte_carlo_error(ref, binge):
    """Stochastic screen: two generators, 2000 permutations each."""
    r = ref["evalBal"]["auto"]
    m = _matched_sample(binge)
    m = m[m["AlcGroup"] != "N"]
    res = sp.balance_vs_randomization(m, "z", BAL_X, n_sim=2000, random_state=11)
    for c in BAL_X + ["min_p"]:
        a, b = (
            res.table.loc[c, "share_better"],
            r["share_better"][c if c != "min_p" else "minP"],
        )
        se = np.sqrt((a * (1 - a) + b * (1 - b)) / 2000 + 1e-6)
        assert abs(a - b) < 5 * se, c
    # under randomization the p-value of a continuous covariate is uniform
    assert res.sim[["age", "bmi", "waisthip"]].median().between(0.45, 0.55).all()
