"""R parity for the propensity-score workflow of Barrett, D'Agostino McGowan
and Gerke, *Causal Inference in R* (https://www.r-causal.org).

The book works through one loop: fit a propensity score, turn it into
weights for a chosen estimand, check what the weights did to the sample
(effective sample size, standardized differences, variance ratios, KS,
energy distance, a weighted AUC), fit the weighted outcome model with a
variance that knows the weights were estimated, compare with
g-computation, and ask how strong an unmeasured confounder would have to
be. This file pins each step against the R package the book uses for it.

The reference numbers are in ``_fixtures/barrett_causal_inference_in_r_R.json``,
written by ``_fixtures/_generate_barrett_causal_inference_in_r.R`` on data
simulated there; both sides read the same CSV bytes.

Tolerances. Everything is deterministic, so the budget is the strict parity
one of ``CLAUDE.md`` section 5.1 (relative 1e-6); most gaps are below 1e-10
and the bounds say so. Three items are conventions, each reproduced
exactly once the convention is applied:

* ``propensity::ipw`` multiplies the M-estimation variance by ``n/(n-1)``;
  ``sp.ipw(se_method="sandwich")`` uses divisor ``n`` (Stata ``teffects``).
* ``marginaleffects`` differentiates numerically, so its delta-method
  standard errors agree with the analytic ones here to about 1e-7.
* ``halfmoon``'s AUC is not the Mann-Whitney probability: on this
  fixture, which has no tied scores, it is 3e-4 below the rank statistic
  that ``wilcox.test`` and ``pROC::auc`` both return. The unweighted AUC
  is pinned to those two, the weighted one to its definition, and
  ``halfmoon`` only to three decimals.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
X = ["x1", "x2", "b"]
TIGHT = 1e-9


@pytest.fixture(scope="module")
def ref():
    path = FIX / "barrett_causal_inference_in_r_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(FIX / "barrett_ps_workflow.csv")


@pytest.fixture(scope="module")
def ps(df):
    return sp.propensity_score(df, "t", X).to_numpy()


def test_propensity_score(ps, ref):
    assert ps == pytest.approx(np.array(ref["ps"]), rel=1e-9)


# --- weights ---------------------------------------------------------------


@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC", "ATM", "ATO"])
def test_weights_and_ess_match_propensity_and_halfmoon(df, ps, ref, estimand):
    w = sp.ps_weights(ps, df["t"].to_numpy(), estimand)
    assert w == pytest.approx(np.array(ref["weights"][estimand]), rel=TIGHT)
    assert sp.ess(w) == pytest.approx(ref["ess"][estimand], rel=TIGHT)


def test_stabilized_truncated_and_grouped(df, ps, ref):
    t = df["t"].to_numpy()
    w = sp.ps_weights(ps, t, stabilize=True)
    assert w == pytest.approx(np.array(ref["weights_stabilized"]), rel=TIGHT)
    w = sp.ps_weights(ps, t, truncate=(0.05, 0.95), truncate_scale="quantile")
    assert w == pytest.approx(np.array(ref["trunc_weights"]), rel=TIGHT)
    by = sp.ess(sp.ps_weights(ps, t), by=t)
    assert by.loc[0] == pytest.approx(ref["ess_by_group"]["0"], rel=TIGHT)
    assert by.loc[1] == pytest.approx(ref["ess_by_group"]["1"], rel=TIGHT)


def test_atu_is_atc_and_series_keep_their_index(df, ps):
    t = df["t"]
    a = sp.ps_weights(ps, t, "ATU")
    assert isinstance(a, pd.Series) and a.index.equals(t.index)
    assert a.to_numpy() == pytest.approx(sp.ps_weights(ps, t.to_numpy(), "ATC"))


def test_weight_edge_cases():
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        sp.ps_weights([0.0, 0.5], [1, 0])
    with pytest.raises(ValueError, match="0/1"):
        sp.ps_weights([0.2, 0.5], [2, 0])
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.ps_weights([0.2, 0.5], [1, 0], "ATT", stabilize=True)
    with pytest.raises(ValueError, match="sigma"):
        sp.ps_weights([0.2, 0.5], [1.0, 0.3], exposure="continuous")
    with pytest.raises(ValueError, match="missing"):
        sp.ess([1.0, np.nan])
    assert sp.ess(np.full(7, 3.0)) == pytest.approx(7.0)


def test_continuous_exposure_weights_are_the_density_ratio():
    rng = np.random.default_rng(0)
    z = rng.normal(size=400)
    x = 1 + 0.8 * z + rng.normal(size=400)
    mu, sigma = 1 + 0.8 * z, 1.0
    from scipy.stats import norm

    w = sp.ps_weights(mu, x, exposure="continuous", sigma=sigma, stabilize=True)
    expected = norm.pdf(x, x.mean(), x.std(ddof=1)) / norm.pdf(x, mu, sigma)
    assert w == pytest.approx(expected, rel=1e-12)
    # stabilised weights have mean close to one under a correct model
    assert abs(w.mean() - 1) < 0.1


# --- trimming --------------------------------------------------------------


def test_crump_cutoff_matches_propensity(df, ps, ref):
    """⚠️ Through 1.38.0 the cutoff solved ``alpha = 1/(2 E[g])`` instead of
    ``alpha (1 - alpha) = 1/(2 E[g])`` on a 500-point grid and trimmed too
    little."""
    from statspai.matching.ps_diagnostics import _crump_alpha

    assert _crump_alpha(ps) == pytest.approx(ref["crump"]["cutoff"], rel=1e-10)
    kept = sp.trimming(df, treatment="t", covariates=X)
    dropped = sorted(set(df.index + 1) - set(kept.index + 1))
    assert dropped == ref["crump"]["trimmed"]


def test_crump_cutoff_satisfies_its_defining_equation():
    rng = np.random.default_rng(1)
    from statspai.matching.ps_diagnostics import _crump_alpha

    ps = rng.beta(0.6, 0.6, size=5000)
    a = _crump_alpha(ps)
    g = 1 / (ps * (1 - ps))
    gamma = 1 / (a * (1 - a))
    assert gamma == pytest.approx(2 * g[g <= gamma].mean(), rel=1e-10)
    # good overlap: nothing to trim
    assert _crump_alpha(rng.uniform(0.3, 0.7, size=500)) == 0.0


# --- IPW -------------------------------------------------------------------


@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATM", "ATO"])
def test_ipw_sandwich_matches_propensity_ipw(df, ref, estimand):
    r = sp.ipw(df, "y", "t", X, estimand=estimand, se_method="sandwich")
    est, se = ref["ipw"][estimand]
    assert r.estimate == pytest.approx(est, rel=TIGHT)
    n = len(df)
    assert r.se * np.sqrt(n / (n - 1)) == pytest.approx(se, rel=1e-9)


def test_ipw_atc_and_its_alias(df, ref):
    a = sp.ipw(df, "y", "t", X, estimand="ATC", se_method="sandwich")
    b = sp.ipw(df, "y", "t", X, estimand="ATU", se_method="sandwich")
    assert a.estimate == pytest.approx(ref["ipw"]["ATC"], rel=TIGHT)
    assert (b.estimate, b.se) == (a.estimate, a.se)


def test_ipw_overlap_estimand_agrees_with_overlap_weights(df):
    a = sp.ipw(df, "y", "t", X, estimand="ATO", se_method="sandwich")
    b = sp.overlap_weights(df, "y", "t", X, n_bootstrap=50, seed=0)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-9)
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.ipw(df, "y", "t", X, estimand="ATM", normalize=False)


# --- balance ---------------------------------------------------------------


def test_balance_matches_cobalt_and_halfmoon(df, ps, ref):
    w = sp.ps_weights(ps, df["t"].to_numpy())
    cb = {row["_row"]: row for row in ref["cobalt"]["unweighted_denominator"]}
    hm = ref["halfmoon"]
    unw = sp.balance_diagnostics(df, "t", X, weights=w, ps=ps, sd_denom="unweighted")
    for v in ("x1", "x2"):
        row = unw.table.loc[v]
        assert row["smd_raw"] == pytest.approx(cb[v]["Diff.Un"], rel=TIGHT)
        assert row["smd_weighted"] == pytest.approx(cb[v]["Diff.Adj"], rel=TIGHT)
        assert row["variance_ratio_weighted"] == pytest.approx(
            cb[v]["V.Ratio.Adj"], rel=TIGHT
        )
        assert row["ks_stat_weighted"] == pytest.approx(cb[v]["KS.Adj"], rel=TIGHT)
    assert unw.table.loc["x1", "variance_ratio_weighted"] == pytest.approx(
        hm["vr_x1"], rel=TIGHT
    )
    s = unw.summary_stats
    assert s["energy_distance_raw"] == pytest.approx(hm["energy_raw"], rel=TIGHT)
    assert s["energy_distance_weighted"] == pytest.approx(
        hm["energy_weighted"], rel=TIGHT
    )
    assert sp.energy_distance(df, "t", X, weights=w) == pytest.approx(
        hm["energy_weighted"], rel=TIGHT
    )
    assert s["effective_sample_size_treated"] == pytest.approx(
        ref["ess_by_group"]["1"], rel=TIGHT
    )


def test_unit_weights_reproduce_the_unweighted_columns(df):
    """⚠️ Through 1.38.0 the weighted variance had divisor ``sum(w)`` and the
    unweighted one ``n - 1``, so ``weights=1`` did not return the raw
    statistics."""
    one = sp.balance_diagnostics(df, "t", X, weights=np.ones(len(df)))
    assert one.table["smd_weighted"].to_numpy() == pytest.approx(
        one.table["smd_raw"].to_numpy(), rel=1e-12
    )
    # and the weighted variance is free of the scale of the weights
    w = np.random.default_rng(0).uniform(0.5, 3, len(df))
    a = sp.balance_diagnostics(df, "t", X, weights=w).table
    b = sp.balance_diagnostics(df, "t", X, weights=40 * w).table
    assert a["variance_ratio_weighted"].to_numpy() == pytest.approx(
        b["variance_ratio_weighted"].to_numpy(), rel=1e-12
    )
    with pytest.raises(ValueError, match="sd_denom"):
        sp.balance_diagnostics(df, "t", X, sd_denom="pooled")


def test_auc_unweighted_matches_and_weighted_is_mann_whitney(df, ps, ref):
    t = df["t"].to_numpy()
    assert sp.auc(t, ps) == pytest.approx(ref["auc"]["wilcoxon"], rel=TIGHT)
    assert sp.auc(t, ps) == pytest.approx(ref["auc"]["proc"], rel=TIGHT)
    assert sp.auc(t, ps) == pytest.approx(ref["auc"]["observed"], abs=2e-3)
    w = sp.ps_weights(ps, t)
    pos, neg = t == 1, t == 0
    wins = (ps[pos][:, None] > ps[neg][None, :]) + 0.5 * (
        ps[pos][:, None] == ps[neg][None, :]
    )
    brute = (w[pos][:, None] * w[neg][None, :] * wins).sum() / (
        w[pos].sum() * w[neg].sum()
    )
    roc = sp.roc_curve(t, ps, weights=w)
    assert roc.auc == pytest.approx(brute, rel=1e-12)
    assert np.isnan(roc.auc_se)
    # halfmoon's weighted value: same to three decimals, not to parity
    assert roc.auc == pytest.approx(ref["auc"]["w"], abs=2e-3)
    assert sp.auc(t, ps, weights=np.ones(len(t))) == pytest.approx(sp.auc(t, ps))


# --- implied regression weights -------------------------------------------


def test_implied_weights_match_lmw(df, ref):
    uri = sp.implied_weights(df, "t", X)
    assert uri.to_numpy() == pytest.approx(np.array(ref["lmw"]["uri"]), rel=1e-8)
    t, c = df["t"] == 1, df["t"] == 0
    diff = np.average(df["y"][t], weights=uri[t]) - np.average(
        df["y"][c], weights=uri[c]
    )
    assert diff == pytest.approx(ref["lmw"]["ols"], rel=1e-9)
    for est, key in (("ATE", "mri_ate"), ("ATT", "mri_att")):
        w = sp.implied_weights(df, "t", X, interactions=True, estimand=est)
        assert w.to_numpy() == pytest.approx(np.array(ref["lmw"][key]), rel=1e-8)
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.implied_weights(df, "t", X, estimand="ATT")


# --- g-computation ----------------------------------------------------------


def test_marginal_contrasts_match_marginaleffects(df, ref):
    g = ref["gcomp_logit"]
    fit = sp.logit("yb ~ C(t) + x1 + x2 + b", data=df)
    for subset, key in ((None, "rd"), ("t == 1", "att"), ("t == 0", "atc")):
        c = sp.contrast(fit, data=df, variable="t", subset=subset).iloc[0]
        assert c["contrast"] == pytest.approx(g[key][0], rel=1e-9)
        assert c["se"] == pytest.approx(g[key][1], rel=1e-6)
    for effect, key, log_key in (
        ("ratio", "rr", "lnrr"),
        ("odds_ratio", "or", "lnor"),
    ):
        c = sp.contrast(fit, data=df, variable="t", effect=effect).iloc[0]
        assert c["contrast"] == pytest.approx(g[key][0], rel=1e-9)
        assert c["se"] == pytest.approx(g[log_key][1], rel=1e-6)
        assert c["ci_lower"] == pytest.approx(g[key][1], rel=1e-6)
        assert c["ci_upper"] == pytest.approx(g[key][2], rel=1e-6)


def test_subpopulation_effects_with_an_interaction(df, ref):
    g = ref["gcomp_ols"]
    fit = sp.regress("y ~ C(t)*x1 + x2 + b", data=df)
    for subset, key in ((None, "ate"), ("t == 1", "att"), ("t == 0", "atc")):
        c = sp.contrast(fit, data=df, variable="t", subset=subset).iloc[0]
        assert c["contrast"] == pytest.approx(g[key][0], rel=1e-9)
        assert c["se"] == pytest.approx(g[key][1], rel=1e-6)
    m = sp.margins(fit, data=df, variables=["t"], subset=(df["t"] == 1).to_numpy())
    assert m["dy/dx"].iloc[0] == pytest.approx(g["att"][0], rel=1e-9)
    # a frame in which one level is absent cannot be contrasted: say so
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="subset="):
        sp.contrast(fit, data=df[df["t"] == 1], variable="t")
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.contrast(fit, data=df, variable="t", subset="t == 5")


def test_g_computation_effect_on_the_untreated(df):
    kw = dict(n_boot=20, seed=0)
    ate = sp.g_computation(df, "y", "t", X, **kw).estimate
    atc = sp.g_computation(df, "y", "t", X, estimand="ATC", **kw).estimate
    atu = sp.g_computation(df, "y", "t", X, estimand="ATU", **kw).estimate
    # one linear model without interaction: the three populations agree
    assert atc == pytest.approx(ate, rel=1e-10)
    assert atu == atc


# --- unmeasured confounder ---------------------------------------------------


def test_confounder_adjust_matches_tipr(ref):
    t = ref["tipr"]
    A = sp.confounder_adjust

    def adj(**kw):
        return float(A(**kw)["effect_adjusted"].iloc[0])

    assert adj(
        effect=6.58, confounder_outcome_effect=-2.3, exposure_confounder_effect=-0.17
    ) == pytest.approx(t["adjust_coef"]["effect_adjusted"], rel=1e-12)
    assert adj(
        effect=-12.5,
        confounder_outcome_effect=-10,
        exposed_prev=0.26,
        unexposed_prev=0.05,
    ) == pytest.approx(t["adjust_coef_binary"]["effect_adjusted"], rel=1e-12)
    rr = dict(effect=1.5, confounder_outcome_effect=1.8)
    assert adj(**rr, exposure_confounder_effect=0.5, measure="rr") == pytest.approx(
        t["adjust_rr"]["rr_adjusted"], rel=1e-12
    )
    assert adj(
        **rr, exposed_prev=0.4, unexposed_prev=0.1, measure="rr"
    ) == pytest.approx(t["adjust_rr_binary"]["rr_adjusted"], rel=1e-12)
    common = dict(exposure_confounder_effect=0.5, rare_outcome=False)
    assert adj(**rr, **common, measure="or") == pytest.approx(
        t["adjust_or_common"]["rr_adjusted"], rel=1e-12
    )
    assert adj(**rr, **common, measure="hr") == pytest.approx(
        t["adjust_hr_common"]["rr_adjusted"], rel=1e-12
    )
    assert adj(
        **rr, exposed_prev=0.4, unexposed_prev=0.1, measure="hr", rare_outcome=False
    ) == pytest.approx(t["adjust_hr_binary_common"]["rr_adjusted"], rel=1e-12)
    # a rare-outcome odds ratio is treated as a risk ratio
    assert adj(**rr, exposure_confounder_effect=0.5, measure="or") == pytest.approx(
        t["adjust_rr"]["rr_adjusted"], rel=1e-12
    )
    # point estimate and both limits move together
    out = A(
        [-12.5, -13.4, -11.6],
        confounder_outcome_effect=-10,
        exposed_prev=0.26,
        unexposed_prev=0.05,
    )
    assert out["effect_adjusted"].tolist() == pytest.approx([-10.4, -11.3, -9.5])


def test_confounder_tip_matches_tipr(ref):
    t = ref["tipr"]
    T = sp.confounder_tip

    def one(col, **kw):
        return float(T(**kw)[col].iloc[0])

    d, g, n = (
        "exposure_confounder_effect",
        "confounder_outcome_effect",
        "n_unmeasured_confounders",
    )
    assert one(d, effect=6.58, confounder_outcome_effect=-2.3) == pytest.approx(
        t["tip_coef_d"][d], rel=1e-12
    )
    assert one(g, effect=-10.2, exposure_confounder_effect=4) == pytest.approx(
        t["tip_coef_g"][g], rel=1e-12
    )
    assert one(
        n, effect=6.58, exposure_confounder_effect=-0.5, confounder_outcome_effect=-2.3
    ) == pytest.approx(t["tip_coef_n"][n], rel=1e-12)
    assert one(
        d, effect=1.2, confounder_outcome_effect=2.5, measure="rr"
    ) == pytest.approx(t["tip_rr_d"][d], rel=1e-12)
    assert one(
        n,
        effect=1.5,
        exposure_confounder_effect=0.3,
        confounder_outcome_effect=1.3,
        measure="rr",
    ) == pytest.approx(t["tip_rr_n"][n], rel=1e-12)
    b = dict(effect=1.2, measure="rr")
    assert one(g, **b, exposed_prev=0.5, unexposed_prev=0.1) == pytest.approx(
        t["tip_bin_g"][g], rel=1e-12
    )
    assert one(
        "exposed_prev", **b, unexposed_prev=0.1, confounder_outcome_effect=2.5
    ) == pytest.approx(t["tip_bin_p1"]["exposed_confounder_prev"], rel=1e-12)
    assert one(
        "unexposed_prev", **b, exposed_prev=0.5, confounder_outcome_effect=2.5
    ) == pytest.approx(t["tip_bin_p0"]["unexposed_confounder_prev"], rel=1e-12)
    assert one(
        n,
        effect=1.5,
        exposed_prev=0.4,
        unexposed_prev=0.1,
        confounder_outcome_effect=1.3,
        measure="rr",
    ) == pytest.approx(t["tip_bin_n"][n], rel=1e-12)
    assert one(
        "exposed_prev",
        effect=0.8,
        unexposed_prev=0.1,
        confounder_outcome_effect=0.5,
        measure="rr",
    ) == pytest.approx(t["tip_bin_protective"]["exposed_confounder_prev"], rel=1e-12)
    hr = T(1.5, confounder_outcome_effect=2, measure="hr", rare_outcome=False)
    assert float(hr[d].iloc[0]) == pytest.approx(t["tip_hr_common"][d], rel=1e-12)
    assert float(hr["effect_observed"].iloc[0]) == pytest.approx(
        t["tip_hr_common"]["effect_observed"], rel=1e-12
    )
    assert one(
        g,
        effect=1.5,
        exposed_prev=0.5,
        unexposed_prev=0.1,
        measure="or",
        rare_outcome=False,
    ) == pytest.approx(t["tip_or_binary_common"][g], rel=1e-12)


def test_tipping_is_the_inverse_of_adjusting():
    tip = sp.confounder_tip(
        1.7, exposed_prev=0.6, unexposed_prev=0.2, measure="rr"
    ).iloc[0]
    back = sp.confounder_adjust(
        1.7,
        confounder_outcome_effect=tip["confounder_outcome_effect"],
        exposed_prev=0.6,
        unexposed_prev=0.2,
        measure="rr",
    )
    assert float(back["effect_adjusted"].iloc[0]) == pytest.approx(1.0, abs=1e-12)


def test_confounder_functions_fail_loudly():
    with pytest.warns(UserWarning, match="no binary confounder"):
        out = sp.confounder_tip(3.0, exposed_prev=0.5, unexposed_prev=0.4, measure="rr")
    assert np.isnan(out["confounder_outcome_effect"].iloc[0])
    with pytest.warns(UserWarning, match="away from the null"):
        out = sp.confounder_tip(
            1.5,
            exposure_confounder_effect=-0.3,
            confounder_outcome_effect=1.3,
            measure="rr",
        )
    assert out["n_unmeasured_confounders"].iloc[0] == 0
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.confounder_adjust(
            1.0,
            confounder_outcome_effect=2,
            exposure_confounder_effect=0.1,
            exposed_prev=0.2,
            unexposed_prev=0.1,
        )
    with pytest.raises(ValueError, match="must be positive"):
        sp.confounder_adjust(
            -1.0,
            confounder_outcome_effect=2,
            exposure_confounder_effect=1,
            measure="rr",
        )
    with pytest.raises(ValueError, match="prevalence"):
        sp.confounder_adjust(
            1.0, confounder_outcome_effect=2, exposed_prev=1.2, unexposed_prev=0.1
        )


# --- DAGs --------------------------------------------------------------------

DAGS = {
    "book": (
        "emm -> wait; close -> wait; season -> wait; temp -> wait; temp -> emm",
        "emm",
        "wait",
    ),
    "two_sizes": ("W -> X; P -> W; Q -> W; P -> Y; Q -> Y; X -> Y", "X", "Y"),
    "chain": ("A -> B; B -> C; C -> D; A -> D; E -> D; B -> E", "A", "D"),
}


def _sets(raw):
    # jsonlite unboxes one-element sets to a string
    return sorted(sorted([s] if isinstance(s, str) else s) for s in raw)


@pytest.mark.parametrize("name", list(DAGS))
def test_adjustment_sets_and_equivalence_class_match_dagitty(ref, name):
    spec, x, y = DAGS[name]
    g = sp.dag(spec)
    r = ref["dag"][name]

    def got(minimal):
        return sorted(sorted(s) for s in g.adjustment_sets(x, y, minimal=minimal))

    assert got(True) == _sets(r["minimal"])
    assert got(False) == _sets(r["all"])
    cls = g.equivalence_class()
    assert cls["n_dags"] == r["n_equivalent"] == len(g.equivalent_dags())
    assert [list(e) for e in cls["undirected"]] == sorted(r["undirected"])


def test_minimal_sets_of_different_sizes_are_all_returned():
    """⚠️ Through 1.38.0 ``minimal=True`` stopped at the smallest size that
    had a valid set and returned only ``{W}`` here."""
    g = sp.dag("W -> X; P -> W; Q -> W; P -> Y; Q -> Y; X -> Y")
    sets = [sorted(s) for s in g.adjustment_sets("X", "Y")]
    assert sorted(sets) == [["P", "Q"], ["W"]]


def test_equivalent_dags_share_independencies_and_refuse_large_graphs():
    g = sp.dag("A -> B; B -> C; C -> D; A -> D; E -> D; B -> E")
    base = {(a, b, frozenset(z)) for a, b, z in g.implied_independencies()}
    for member in g.equivalent_dags():
        other = {(a, b, frozenset(z)) for a, b, z in member.implied_independencies()}
        assert other == base
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        g.equivalent_dags(max_edges=3)
    # a collider's edges are fixed in every member
    cls = sp.dag("A -> C; B -> C").equivalence_class()
    assert cls == {"directed": [("A", "C"), ("B", "C")], "undirected": [], "n_dags": 1}


def test_no_warnings_on_the_happy_path(df, ps):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.ps_weights(ps, df["t"].to_numpy(), "ATO")
        sp.confounder_tip(6.58, confounder_outcome_effect=-7)
        sp.energy_distance(df, "t", X)


# --- a continuous exposure -------------------------------------------------


def test_contrast_of_two_exposure_levels_through_a_spline(df, ref):
    """Chapter 13: move everyone from one dose to another and difference."""
    fit = sp.regress("y ~ bs(x2, df=3) + t + x1 + b", data=df)
    c = sp.margins_at(fit, data=df, at={"x2": [-1, 1]}, contrast="first").iloc[0]
    est, se = ref["gcomp_spline"]
    assert c["contrast"] == pytest.approx(est, rel=1e-9)
    assert c["se"] == pytest.approx(se, rel=1e-6)
    assert c["versus"] == "x2=-1"
    # the two margins alone do not give this standard error
    m = sp.margins_at(fit, data=df, at={"x2": [-1, 1]})
    assert c["contrast"] == pytest.approx(m["margin"].iloc[1] - m["margin"].iloc[0])
    assert c["se"] != pytest.approx(np.hypot(*m["se"]), rel=1e-3)

    logit = sp.logit("yb ~ bs(x2, df=3) + t + x1", data=df)
    g = ref["gcomp_spline_logit"]
    m = sp.margins_at(logit, data=df, at={"x2": [-1, 1]})
    assert m["margin"].tolist() == pytest.approx(g[1:], rel=1e-8)
    c = sp.margins_at(logit, data=df, at={"x2": [-1, 1]}, contrast="first")
    assert c["contrast"].iloc[0] == pytest.approx(g[0], rel=1e-8)


def test_margins_at_contrast_options(df):
    fit = sp.regress("y ~ x2 + t + x1", data=df)
    grid = {"x2": [-1, 0, 2]}
    first = sp.margins_at(fit, data=df, at=grid, contrast="first")
    adjacent = sp.margins_at(fit, data=df, at=grid, contrast="adjacent")
    slope = fit.params["x2"]
    assert first["contrast"].tolist() == pytest.approx([slope, 3 * slope])
    assert adjacent["contrast"].tolist() == pytest.approx([slope, 2 * slope])
    assert adjacent["versus"].tolist() == ["x2=-1", "x2=0"]
    # a linear term: the standard error is |distance| times the coefficient's
    assert first["se"].iloc[0] == pytest.approx(fit.std_errors["x2"], rel=1e-9)
    sub = sp.margins_at(fit, data=df, at={"x2": [0]}, subset="t == 1")
    assert sub.attrs["n"] == int((df["t"] == 1).sum())
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.margins_at(fit, data=df, at={"x2": [0]}, contrast="first")
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.margins_at(fit, data=df, at=grid, contrast="pairwise")
