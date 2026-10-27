"""Round-6 option scan: the open items left by round 5.

Each test states the correct behaviour against a known truth, an identity
or a reference implementation (Stata 18 / R numbers are pinned inline with
the command that produced them).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import statspai as sp
from statspai.exceptions import MethodIncompatibility

# --------------------------------------------------------------------- #
#  1. sp.aipw(design=) keeps each PSU in one cross-fitting fold
# --------------------------------------------------------------------- #


def _psu_frame(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(40):
        u = rng.normal()
        d = int(rng.integers(2))  # cluster-randomised treatment
        for _ in range(8):
            x = rng.normal()
            rows.append(
                {
                    "psu": f"c{g:02d}",
                    "x": x,
                    "d": d,
                    "y": 1.0 + 0.5 * d + 0.8 * x + u + rng.normal(),
                    "w": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_aipw_design_folds_follow_psus():
    df = _psu_frame()
    design = sp.svydesign(df, weights="w", cluster="psu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_design = sp.aipw(df, y="y", treat="d", covariates=["x"], design=design)
        by_cluster = sp.aipw(df, y="y", treat="d", covariates=["x"], cluster="psu")
    # Equal weights and the PSUs as clusters: the folds, so the cross-fitted
    # nuisances and the point estimate, must be those of cluster=.
    assert by_design.model_info["fold_unit"] == "psu"
    assert by_design.estimate == pytest.approx(by_cluster.estimate, rel=1e-10)


def test_aipw_design_without_psus_keeps_row_folds():
    df = _psu_frame()
    design = sp.svydesign(df, weights="w")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_design = sp.aipw(df, y="y", treat="d", covariates=["x"], design=design)
        plain = sp.aipw(df, y="y", treat="d", covariates=["x"], weights="w")
    assert by_design.model_info["fold_unit"] == "row"
    assert by_design.estimate == pytest.approx(plain.estimate, rel=1e-10)


# --------------------------------------------------------------------- #
#  2. sp.pate flags separation of the participation model
# --------------------------------------------------------------------- #


def _separated_samples():
    rng = np.random.default_rng(0)
    n = 80
    x_exp = rng.uniform(0, 1, n)
    d = rng.integers(0, 2, n)
    exp = pd.DataFrame({"x": x_exp, "d": d, "y": d * (1 + x_exp) + rng.normal(0, 1, n)})
    tgt = pd.DataFrame({"x": rng.uniform(2, 3, 120)})  # no overlap at all
    return exp, tgt


def test_pate_warns_and_records_complete_separation():
    exp, tgt = _separated_samples()
    with pytest.warns(sp.exceptions.AssumptionWarning, match="separat"):
        res = sp.pate(
            exp, tgt, y="y", treatment="d", covariates=["x"], n_boot=20, seed=1
        )
    assert res.diagnostics["separation"] == "complete"
    assert res.diagnostics["n_participation_clipped"] == len(exp)


def test_pate_no_separation_flag_with_overlap():
    rng = np.random.default_rng(1)
    exp = pd.DataFrame({"x": rng.normal(0.3, 1, 100), "d": rng.integers(0, 2, 100)})
    exp["y"] = exp["d"] + rng.normal(size=100)
    tgt = pd.DataFrame({"x": rng.normal(0, 1, 150)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.exceptions.AssumptionWarning)
        res = sp.pate(
            exp, tgt, y="y", treatment="d", covariates=["x"], n_boot=20, seed=1
        )
    assert res.diagnostics["separation"] is None


# --------------------------------------------------------------------- #
#  3. Two-sided normal-approximation power counts both tails
# --------------------------------------------------------------------- #

_ZERO_EFFECT_CALLS = [
    lambda: sp.power_rct(n=400, effect_size=0.0),
    lambda: sp.power_did(n=400, effect_size=0.0, n_periods=6, n_treated_periods=3),
    lambda: sp.power_rd(n=400, effect_size=0.0),
    lambda: sp.power_iv(n=400, effect_size=0.0, first_stage_f=20),
    lambda: sp.power_cluster_rct(
        n_clusters=40, cluster_size=10, effect_size=0.0, icc=0.05
    ),
    lambda: sp.power_ols(n=400, effect_size=0.0),
    lambda: sp.power_two_proportions(n=400, p1=0.4, p2=0.4),
    lambda: sp.power_two_proportions(n=400, p1=0.4, p2=0.4, test="chi2"),
]


@pytest.mark.parametrize("call", _ZERO_EFFECT_CALLS)
def test_two_sided_power_at_zero_effect_is_alpha(call):
    assert float(call().power) == pytest.approx(0.05, abs=1e-12)


def test_power_rct_counts_the_lower_tail():
    # Stata 18: power onemean 0 0.01, n(100) sd(1) knownsd -> 0.051146302810811
    # (both tails). power_rct with n=400, ratio 1 has se = 1/sqrt(100) = 0.1.
    got = float(sp.power_rct(n=400, effect_size=0.01).power)
    assert got == pytest.approx(0.051146302810811, abs=1e-12)


def test_power_logrank_counts_both_tails():
    res = sp.power_logrank(n=200, hazard_ratio=0.95, prob_event=0.5)
    shift = np.sqrt(100 * 0.25) * abs(np.log(0.95))
    z = norm.ppf(0.975)
    assert float(res.power) == pytest.approx(
        norm.cdf(shift - z) + norm.cdf(-shift - z), abs=1e-14
    )


def test_one_sided_power_unchanged_at_zero_effect():
    res = sp.power_two_proportions(n=400, p1=0.4, p2=0.4, alternative="one-sided")
    assert float(res.power) == pytest.approx(0.05, abs=1e-12)


# --------------------------------------------------------------------- #
#  4. power_two_proportions(test="chi2") is Stata / R power for the
#     Pearson test
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs, stata",
    [
        # Stata 18: power twoproportions 0.5 0.65, n(400) [test(chi2)]
        (dict(n=400, p1=0.5, p2=0.65), 0.861470522956074),
        # ... nratio(2) nfractional (group sizes 133.33 / 266.67, as here;
        # without nfractional Stata floors them to 133 / 266: 0.820797469169701)
        (dict(n=400, p1=0.5, p2=0.65, ratio=2.0), 0.821735832105281),
        # power twoproportions 0.5 0.52, n(400)
        (dict(n=400, p1=0.5, p2=0.52), 0.068481523865120),
        # ... onesided (R power.prop.test(alternative="one.sided") agrees)
        (dict(n=400, p1=0.5, p2=0.65, alternative="one-sided"), 0.920100766897105),
    ],
)
def test_power_two_proportions_chi2_matches_stata(kwargs, stata):
    res = sp.power_two_proportions(test="chi2", **kwargs)
    assert float(res.power) == pytest.approx(stata, abs=1e-12)
    assert res.params["test"] == "chi2"


@pytest.mark.parametrize("ratio, stata_n", [(1.0, 339), (2.0, 378)])
def test_power_two_proportions_chi2_sample_size_matches_stata(ratio, stata_n):
    # Stata 18: power twoproportions 0.5 0.65, power(0.8) [nratio(2)]
    # nfractional -> N = 338.62 (R power.prop.test: 2 x 169.3114) / 377.922;
    # the smallest integer total with fractional group sizes is its ceiling.
    # (Stata's integer default rounds each group up instead: 340 / 378.)
    res = sp.power_two_proportions(
        p1=0.5, p2=0.65, ratio=ratio, power_target=0.8, test="chi2"
    )
    assert res.n == stata_n


def test_power_two_proportions_default_is_still_wald():
    res = sp.power_two_proportions(n=400, p1=0.5, p2=0.65)
    se = np.sqrt(0.25 / 200 + 0.65 * 0.35 / 200)
    z = norm.ppf(0.975)
    want = norm.cdf(0.15 / se - z) + norm.cdf(-0.15 / se - z)
    assert float(res.power) == pytest.approx(want, abs=1e-14)
    assert res.params["test"] == "wald"


def test_power_two_proportions_rejects_unknown_test():
    with pytest.raises(MethodIncompatibility):
        sp.power_two_proportions(n=400, p1=0.5, p2=0.65, test="fisher")


# --------------------------------------------------------------------- #
#  5. cluster_matched_pair: interval label follows alpha; coverage
# --------------------------------------------------------------------- #


def _matched_pairs(rng, n_pairs=40, het=0.0):
    rows = []
    for p in range(n_pairs):
        x = rng.normal()
        for arm in (0, 1):
            u = rng.normal(0, 0.5)
            for _ in range(10):
                y = 2 * x + u + (1.0 + het * x) * arm + rng.normal()
                rows.append({"y": y, "c": 2 * p + arm, "d": arm, "p": p})
    return pd.DataFrame(rows)


def test_cluster_matched_pair_summary_labels_its_level():
    df = _matched_pairs(np.random.default_rng(0))
    res = sp.cluster_matched_pair(df, "y", "c", "d", "p", alpha=0.10)
    assert "90% CI" in res.summary()
    assert "95% CI" not in res.summary()


def test_cluster_matched_pair_coverage_homogeneous_effect():
    # Known truth 1.0, homogeneous effect: the pair-difference variance is
    # consistent and the interval covers at about the nominal rate.
    rng = np.random.default_rng(11)
    hits = 0
    reps = 400
    for _ in range(reps):
        res = sp.cluster_matched_pair(_matched_pairs(rng), "y", "c", "d", "p")
        hits += res.ci[0] <= 1.0 <= res.ci[1]
    assert 0.92 <= hits / reps <= 0.98


# --------------------------------------------------------------------- #
#  6. dnc_gnn_did: embedding rows are matched to units by id
# --------------------------------------------------------------------- #


def _dnc_panel(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_units = 60
    ids = [f"u{k}" for k in range(n_units)]  # 'u10' sorts before 'u2'
    f = rng.normal(size=n_units)  # network position, drives trend and uptake
    treated = (f + rng.normal(0, 0.5, n_units)) > 0
    rows = []
    for k, uid in enumerate(ids):
        g = 3 if treated[k] else 0
        u = rng.normal()
        for t in range(1, 6):
            post = int(g > 0 and t >= g)
            rows.append(
                {
                    "id": uid,
                    "time": t,
                    "g": g,
                    "y": u + (0.5 + 1.0 * f[k]) * t + 2.0 * post + rng.normal(0, 0.2),
                    "nc_y": u + rng.normal(0, 0.2),
                    "nc_d": u + rng.normal(0, 0.2),
                }
            )
    df = pd.DataFrame(rows)
    emb = pd.DataFrame({"f": f}, index=pd.Index(ids, name="id"))
    return df, emb


def test_dnc_gnn_did_dataframe_embedding_aligned_by_id():
    df, emb = _dnc_panel()
    shuffled = emb.sample(frac=1.0, random_state=0)  # row order is irrelevant
    res = sp.dnc_gnn_did(
        df,
        y="y",
        treat="g",
        time="time",
        id="id",
        nc_outcome=["nc_y"],
        nc_exposure=["nc_d"],
        embedding=shuffled,
        n_boot=30,
    )
    # The feature f enters the trend; adjusting for it recovers the effect 2.
    assert res.estimate == pytest.approx(2.0, abs=0.15)
    assert res.diagnostics["embedding_used"] is True


def test_dnc_gnn_did_ndarray_in_ambiguous_order_is_refused():
    df, emb = _dnc_panel()
    # Units appear as u0, u1, ..., u59 but sort as u0, u1, u10, ...: an array
    # has no unit labels, so its row order cannot be told apart.
    with pytest.raises(MethodIncompatibility, match="embedding"):
        sp.dnc_gnn_did(
            df,
            y="y",
            treat="g",
            time="time",
            id="id",
            nc_outcome=["nc_y"],
            nc_exposure=["nc_d"],
            embedding=emb.to_numpy(),
            n_boot=10,
        )


def test_dnc_gnn_did_without_embedding_reports_no_network_feature():
    df, _ = _dnc_panel()
    res = sp.dnc_gnn_did(
        df,
        y="y",
        treat="g",
        time="time",
        id="id",
        nc_outcome=["nc_y"],
        nc_exposure=["nc_d"],
        n_boot=10,
    )
    assert res.diagnostics["network_features"] is None


# --------------------------------------------------------------------- #
#  7. The 'parmigiani' interference alias is deprecated
# --------------------------------------------------------------------- #


def test_parmigiani_alias_warns():
    with pytest.warns(DeprecationWarning, match="network_hte"):
        with pytest.raises(TypeError):
            sp.interference("parmigiani")  # no data: the warning comes first


def test_network_hte_name_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(TypeError):
            sp.interference("network_hte")


# --------------------------------------------------------------------- #
#  8. sp.cross_validate: DML folds / seed, reference=, alpha
# --------------------------------------------------------------------- #


def _dml_frame():
    rng = np.random.default_rng(5)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = 0.5 * x1 + rng.normal(size=n)
    y = 1.0 * d + x1 - x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def test_cross_validate_dml_honours_n_folds_and_seed():
    df = _dml_frame()
    cv = sp.cross_validate(
        df,
        "dml",
        y="y",
        treatment="d",
        covariates=["x1", "x2"],
        engines=["statspai"],
        n_folds=3,
        seed=7,
    )
    est = [e for e in cv.estimates if e.engine == "statspai"][0]
    ref = sp.dml(
        df, y="y", treat="d", covariates=["x1", "x2"], n_folds=3, random_state=7
    )
    assert est.coef == pytest.approx(ref.estimate, rel=1e-12)
    assert est.se == pytest.approx(ref.se, rel=1e-12)


def test_cross_validate_misspelt_reference_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=50)})
    df["y"] = df["x"] + rng.normal(size=50)
    with pytest.raises(MethodIncompatibility) as info:
        sp.cross_validate(
            df,
            "ols",
            formula="y ~ x",
            treatment="x",
            engines=["statspai"],
            reference="statspia",
        )
    assert info.value.recovery_hint


def test_cross_validate_intervals_follow_alpha():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=80)})
    df["y"] = 0.5 * df["x"] + rng.normal(size=80)
    cv = sp.cross_validate(
        df,
        "ols",
        formula="y ~ x",
        treatment="x",
        engines=["statspai", "pyfixest"],
        alpha=0.10,
    )
    fit = sp.regress("y ~ x", data=df)
    tidy = fit.tidy(conf_level=0.90).set_index("term")
    for e in cv.estimates:
        if e.status != "ok":
            continue
        assert e.ci_lower == pytest.approx(
            tidy.loc["x", "conf_low"], rel=1e-8
        ), e.engine
        assert e.ci_upper == pytest.approx(
            tidy.loc["x", "conf_high"], rel=1e-8
        ), e.engine


# --------------------------------------------------------------------- #
#  9. relative_risk with a zero cell: Stata cs reports the plain ratio
# --------------------------------------------------------------------- #


def test_relative_risk_zero_cell_is_the_plain_ratio():
    # Stata 18: csi 0 20 80 80 -> r(rr) = 0, CI missing.
    with pytest.warns(UserWarning, match="interval"):
        res = sp.epi.relative_risk(0, 80, 20, 80)
    assert res.estimate == 0.0
    assert np.isnan(res.ci[0]) and np.isnan(res.ci[1])
    # csi 20 0 80 80 -> rr missing (infinite risk ratio)
    with pytest.warns(UserWarning):
        res2 = sp.epi.relative_risk(20, 80, 0, 80)
    assert res2.estimate == np.inf


def test_relative_risk_zero_noncase_cell_uses_raw_counts():
    # Stata 18: csi 10 5 0 5 (exposed: 10 cases, 0 non-cases) -> rr 2,
    # [1.076109420254179, 3.717094121390733]. The Haldane default gave 1.909.
    res = sp.epi.relative_risk(10, 0, 5, 5)
    assert res.estimate == pytest.approx(2.0, rel=1e-14)
    assert res.ci[0] == pytest.approx(1.076109420254179, rel=1e-12)
    assert res.ci[1] == pytest.approx(3.717094121390733, rel=1e-12)


def test_relative_risk_haldane_is_opt_in():
    res = sp.epi.relative_risk(0, 80, 20, 80, correction="haldane")
    want = (0.5 / 81.0) / (20.5 / 101.0)
    assert res.estimate == pytest.approx(want, rel=1e-14)
    assert np.isfinite(res.ci[0]) and np.isfinite(res.ci[1])


def test_relative_risk_without_zero_cells_matches_stata():
    # csi 40 20 60 80: rr 2, [1.26300636440305, 3.167046590371353]
    res = sp.epi.relative_risk(40, 60, 20, 80)
    assert res.estimate == pytest.approx(2.0, rel=1e-14)
    assert res.ci[0] == pytest.approx(1.26300636440305, rel=1e-10)
    assert res.ci[1] == pytest.approx(3.167046590371353, rel=1e-10)


# --------------------------------------------------------------------- #
#  10. PAF interval with an estimated exposure prevalence
# --------------------------------------------------------------------- #


def _numeric_multinomial_se(a, b, c, d):
    """Delta method on log(1 - PAF) with a numerical gradient (multinomial)."""
    n = np.array([a, b, c, d], dtype=float)
    N = n.sum()
    p = n / N

    def lnq(pp):
        pe = pp[0] + pp[1]
        r = pp[0] + pp[2]
        r0 = pp[2] / (pp[2] + pp[3])
        return np.log(r0 / r) + 0 * pe

    grad = np.zeros(4)
    h = 1e-7
    for i in range(4):
        e = np.zeros(4)
        e[i] = h
        grad[i] = (lnq(p + e) - lnq(p - e)) / (2 * h)
    cov = (np.diag(p) - np.outer(p, p)) / N
    return float(np.sqrt(grad @ cov @ grad))


def test_paf_interval_includes_prevalence_uncertainty():
    a, b, c, d = 40, 60, 20, 80
    res = sp.epi.attributable_risk(a, b, c, d)
    se = _numeric_multinomial_se(a, b, c, d)
    z = norm.ppf(0.975)
    lnq = np.log(1 - res.paf)
    assert res.paf_ci[0] == pytest.approx(1 - np.exp(lnq + z * se), rel=1e-6)
    assert res.paf_ci[1] == pytest.approx(1 - np.exp(lnq - z * se), rel=1e-6)
    # Stata 18: glm y i.x, family(binomial) link(log) vce(robust); punaf,
    # atspec(x=0) eform vce(unconditional) -> [0.090436, 0.5113653], whose
    # log(1 - PAF) SE 0.158510628919060 carries the robust N/(N-1).
    stata_se = 0.158510628919060
    assert se * np.sqrt(200 / 199) == pytest.approx(stata_se, rel=1e-6)


def test_paf_fixed_prevalence_keeps_the_conditional_interval():
    res = sp.epi.attributable_risk(40, 60, 20, 80, prevalence="fixed")
    assert res.paf_ci == pytest.approx(
        (0.0942846836814939, 0.5092890266546513), rel=1e-14
    )


def test_paf_interval_coverage_under_random_sampling():
    # Known truth: multinomial sampling, P_e = 0.3, R0 = 0.45, RR = 2.
    rng = np.random.default_rng(2)
    pe, r0, rr, N = 0.3, 0.45, 2.0, 400
    p = np.array([pe * rr * r0, pe * (1 - rr * r0), (1 - pe) * r0, (1 - pe) * (1 - r0)])
    truth = pe * (rr - 1) / (1 + pe * (rr - 1))
    hits = 0
    reps = 3000
    for _ in range(reps):
        a, b, c, d = rng.multinomial(N, p)
        lo, hi = sp.epi.attributable_risk(int(a), int(b), int(c), int(d)).paf_ci
        hits += lo <= truth <= hi
    # 'fixed' covers about 0.914 here; the estimated-prevalence interval 0.947.
    assert 0.935 <= hits / reps <= 0.962


# --------------------------------------------------------------------- #
#  11. ipcw: the event column must be 0/1
# --------------------------------------------------------------------- #


def _ipcw_frame():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    return pd.DataFrame(
        {"time": rng.exponential(size=n), "event": rng.binomial(1, 0.7, n), "x": x}
    )


@pytest.mark.parametrize("bad", [2.0, 0.5, -1])
def test_ipcw_rejects_non_binary_event(bad):
    df = _ipcw_frame()
    df["event"] = df["event"].astype(float)
    df.loc[0, "event"] = bad
    with pytest.raises(MethodIncompatibility) as info:
        sp.ipcw(df, time="time", event="event", censor_covariates=["x"])
    assert info.value.recovery_hint


def test_ipcw_accepts_bool_and_float_binary_events():
    df = _ipcw_frame()
    base = sp.ipcw(df, time="time", event="event", censor_covariates=["x"])
    as_bool = sp.ipcw(
        df.assign(event=df["event"].astype(bool)),
        time="time",
        event="event",
        censor_covariates=["x"],
    )
    as_float = sp.ipcw(
        df.assign(event=df["event"].astype(float)),
        time="time",
        event="event",
        censor_covariates=["x"],
    )
    np.testing.assert_array_equal(base.weights, as_bool.weights)
    np.testing.assert_array_equal(base.weights, as_float.weights)
