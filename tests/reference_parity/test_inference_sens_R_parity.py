"""Inference / sensitivity family vs the R packages that define the methods.

Reference: ``_fixtures/inference_sens_R.json`` written by
``_generate_inference_sens_R.R`` (package versions in its ``_meta`` block:
sandwich 3.1.1, clubSandwich 0.6.2, summclust 0.7.0, fwildclusterboot
0.14.3, ri2 0.5.0, DOS2 0.5.2, rbounds 2.2, EValue 4.1.4, robomit 1.0.7 on
R 4.5.2) from the CSVs that ``_fixtures/_generate_inference_sens_data.py``
writes, so both sides read identical bytes.

Conventions each comparison depends on
--------------------------------------
* ``cluster_robust_se``: CR1 = ``vcovCL(type="HC1", cadjust=TRUE)``, i.e.
  G/(G-1) * (n-1)/(n-k) per component; two-way is Cameron-Gelbach-Miller
  with each component's own G (``multi0=FALSE``). The two-way matrix on this
  design is positive definite, so the PSD projection StatsPAI applies by
  default (sandwich's ``fix=TRUE``) is inactive -- the fixture carries both
  ``fix`` settings to prove it.
* ``cr3_jackknife_vcov`` = (G-1)/G * sum_g (b_(g) - b)(b_(g) - b)', the
  delete-one-cluster jackknife centred at the FULL-SAMPLE estimate
  (``vcovJK(center="estimate")``; MacKinnon-Nielsen-Webb's CV3, which
  ``summclust`` computes). ``clubSandwich``'s CR3 is the same matrix without
  the (G-1)/G factor -- asserted as an identity, not a tolerance.
* ``jackknife_se`` centres at the MEAN of the replicates
  (``vcovJK(center="mean")``, Stata's ``vce(jackknife)`` default).
* Wild cluster bootstrap: restricted (WCR), Rademacher, G = 12 and
  B >= 2^12, so fwildclusterboot and StatsPAI both enumerate all 4096 sign
  vectors and the bootstrap distribution is exact. p = #{|t*| > |t|} / B
  (strict: the identity draw and its negation tie exactly). p-values are
  multiples of 1/4096 and are compared exactly; t at 1e-10. The
  test-inversion CI is the location of a jump of a step function;
  fwildclusterboot's uniroot is run at tol = 1e-13 and StatsPAI bisects to
  machine precision, so the endpoints are compared at 1e-9 relative.
* Randomization inference: every design has at most 4900 assignments, so
  ri2 (via randomizr) and StatsPAI enumerate the same set; p = #{|T*| >=
  |T|} / N including the observed assignment. p-values compared exactly
  (up to 1e-12 for the rational), statistics at 1e-12.
* Rosenbaum bounds: DOS2::senWilcox is Rosenbaum's own code -- zeros ranked
  with the other pairs and given weight 0, no continuity correction.
  rbounds::psens drops the zeros before ranking (``zero_method="wilcox"``)
  and ROUNDS its bounds to 4 decimals, so it is compared after rounding.
  The sign-test bound is the exact binomial tail, stats::binom.test.
* E-values: EValue's own functions; evalues.RD searches a grid built with
  ``seq`` while StatsPAI uses ``np.arange`` -- the grid points differ in the
  last bits (observed 6e-14), hence 1e-12.
* Oster: robomit rounds delta* and beta* to 6 decimals before returning
  them, so the R side is compared at 5e-7 absolute; the full-precision
  reference is Stata ``psacalc`` (test_inference_sens_stata_parity.py).
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
R = json.loads((_FIX / "inference_sens_R.json").read_text(encoding="utf-8"))
REG = pd.read_csv(_FIX / "inference_sens_reg.csv")
TIGHT = 1e-10


def _close(ours, ref, rtol=TIGHT, atol=0.0):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float),
        np.asarray(ref, dtype=float),
        rtol=rtol,
        atol=atol,
    )


def _design():
    X = np.column_stack([np.ones(len(REG)), REG[["d", "x1", "x2"]].to_numpy(float)])
    y = REG["y"].to_numpy(float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return X, y, y - X @ b


def _se(V):
    return np.sqrt(np.diag(np.asarray(V, dtype=float)))


# --------------------------------------------------------------------------
# Cluster-robust variance
# --------------------------------------------------------------------------


def test_cluster_robust_se_one_way_cr1_matches_vcovCL():
    X, _, e = _design()
    _close(
        sp.cluster_robust_se(X, e, REG["s12"].to_numpy()),
        _se(R["cluster"]["V_cr1_s12"]),
    )


def test_cluster_robust_se_without_df_adjustment_matches_cr0():
    X, _, e = _design()
    _close(
        sp.cluster_robust_se(X, e, REG["s12"].to_numpy(), df_adjust=False),
        _se(R["cluster"]["V_cr0_s12"]),
    )


def test_cluster_robust_se_two_way_matches_vcovCL():
    X, _, e = _design()
    clusters = [REG["s12"].to_numpy(), REG["yr"].to_numpy()]
    ref = R["cluster"]["V_cr1_s12_yr"]
    # The CGM matrix is positive definite here, so sandwich's fix=TRUE (the
    # PSD projection StatsPAI applies by default) changes nothing ...
    _close(ref, R["cluster"]["V_cr1_s12_yr_fix"], rtol=0.0)
    assert np.all(np.linalg.eigvalsh(np.asarray(ref)) > 0)
    # ... and both StatsPAI settings reproduce it.
    _close(sp.cluster_robust_se(X, e, clusters), _se(ref))
    _close(sp.cluster_robust_se(X, e, clusters, psd_correct=False), _se(ref))


def test_cr3_jackknife_vcov_matches_vcovJK_centred_at_estimate():
    X, y, _ = _design()
    V = sp.cr3_jackknife_vcov(X, y, REG["s12"].to_numpy())
    _close(V, R["cluster"]["V_jk_estimate"])
    # summclust's CV3 (its vcov carries every coefficient, intercept first)
    assert R["cluster"]["summclust_coef_names"] == ["(Intercept)", "d", "x1", "x2"]
    _close(V, R["cluster"]["summclust_vcov_cv3"])


def test_cr3_jackknife_vcov_is_clubsandwich_cr3_times_g_minus_1_over_g():
    """Identity: the jackknife centred at b equals the analytic CR3 of
    clubSandwich scaled by (G-1)/G (b_(g) - b = -(X'X)^-1 X_g'(I-H_gg)^-1 e_g)."""
    X, y, _ = _design()
    G = REG["s12"].nunique()
    V = sp.cr3_jackknife_vcov(X, y, REG["s12"].to_numpy())
    _close(V, np.asarray(R["cluster"]["V_clubsandwich_cr3"]) * (G - 1) / G)


def test_jackknife_se_matches_vcovJK_centred_at_mean():
    fit = sp.regress("y ~ d + x1 + x2", data=REG)
    jk = sp.jackknife_se(fit, REG, cluster="s12")
    _close(jk.std_errors.to_numpy(), _se(R["cluster"]["V_jk_mean"]))
    assert jk.model_info["jackknife_dof"] == REG["s12"].nunique() - 1


# --------------------------------------------------------------------------
# Wild cluster bootstrap (fwildclusterboot, full enumeration)
# --------------------------------------------------------------------------

WILD_CASES = [
    ("d_h0", "d", 0.0),
    ("x1_h0", "x1", 0.0),
    ("d_h02", "d", 0.2),
    ("x2_h0", "x2", 0.0),
]


@pytest.mark.parametrize("key,var,h0", WILD_CASES, ids=[c[0] for c in WILD_CASES])
def test_wild_cluster_bootstrap_matches_fwildclusterboot(key, var, h0):
    ref = R["wild"][key]
    out = sp.wild_cluster_bootstrap(
        REG, "y", ["d", "x1", "x2"], "s12", test_var=var, h0=h0, n_boot=9999, seed=1
    )
    assert out["enumerated"] and out["n_boot"] == ref["B"] == 4096
    assert out["p_boot"] * 4096 == pytest.approx(ref["p"] * 4096, abs=1e-9)
    _close(out["t_stat"], ref["t"])
    # Identity: draws w and -w give t* and -t*, so exceedances come in pairs.
    assert round(out["p_boot"] * 4096) % 2 == 0


@pytest.mark.parametrize("key,var", [("d_h0", "d"), ("x1_h0", "x1")])
def test_wild_cluster_boot_matches_fwildclusterboot(key, var):
    ref = R["wild"][key]
    fit = sp.regress("y ~ d + x1 + x2", data=REG)
    out = sp.wild_cluster_boot(fit, REG, cluster="s12", variable=var, n_boot=9999)
    assert out["enumerated"] and out["n_boot"] == 4096
    assert out["p_boot"] * 4096 == pytest.approx(ref["p"] * 4096, abs=1e-9)
    _close(out["t_stat"], ref["t"])


@pytest.mark.parametrize("key,var", [("d_h0", "d"), ("x1_h0", "x1"), ("x2_h0", "x2")])
def test_wild_cluster_ci_inv_matches_fwildclusterboot(key, var):
    ref = R["wild"][key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.wild_cluster_ci_inv(
            REG,
            "y",
            ["d", "x1", "x2"],
            "s12",
            test_var=var,
            n_boot=9999,
            weight_type="rademacher",
        )
    _close(out["ci"], ref["ci"], rtol=1e-9)


# --------------------------------------------------------------------------
# Randomization inference (ri2, full enumeration)
# --------------------------------------------------------------------------

RI_DATA = {
    "simple": pd.read_csv(_FIX / "inference_sens_ri_simple.csv"),
    "cluster": pd.read_csv(_FIX / "inference_sens_ri_cluster.csv"),
    "strat": pd.read_csv(_FIX / "inference_sens_ri_strat.csv"),
}


def _ri_close(p_ours, p_ref, n):
    assert p_ours * n == pytest.approx(p_ref * n, abs=1e-9)


@pytest.mark.parametrize(
    "key,stat",
    [("simple_diff", "diff_means"), ("simple_t", "t"), ("simple_ks", "ks")],
)
def test_ri_test_matches_ri2(key, stat):
    ref = R["ri"][key]
    out = sp.ri_test(RI_DATA["simple"], "y", "d", stat=stat, n_perms=10000)
    assert out["exact"] and out["n_perms"] == ref["n_assign"] == 924
    _close(out["observed"], ref["observed"], rtol=1e-12)
    _ri_close(out["p_value"], ref["p_two"], 924)
    _ri_close(out["p_one_sided"], ref["p_upper"], 924)


def test_ri_test_cluster_randomization_matches_ri2():
    ref = R["ri"]["cluster_diff"]
    out = sp.ri_test(RI_DATA["cluster"], "y", "d", n_perms=10000, cluster="cl")
    assert out["exact"] and out["n_perms"] == ref["n_assign"] == 70
    _close(out["observed"], ref["observed"], rtol=1e-12)
    _ri_close(out["p_value"], ref["p_two"], 70)
    _ri_close(out["p_one_sided"], ref["p_upper"], 70)


@pytest.mark.parametrize(
    "key,statistic,kw",
    [
        ("simple_diff", "ate", {}),
        ("simple_ks", "ks", {}),
        ("simple_ranksum", "rank_sum", {}),
        ("cluster_diff", "ate", {"cluster": "cl"}),
        ("strat_diff", "ate", {"stratify": "st"}),
    ],
)
def test_fisher_exact_matches_ri2(key, statistic, kw):
    ref = R["ri"][key]
    data = RI_DATA[key.split("_")[0]]
    out = sp.fisher_exact(data, "y", "d", statistic=statistic, n_perm=10000, **kw)
    n = ref["n_assign"]
    assert out.n_perm == n
    _close(out.statistic, ref["observed"], rtol=1e-12)
    _ri_close(out.p_value, ref["p_two"], n)
    _ri_close(out.p_one_sided, ref["p_upper"], n)


def test_ri_symmetric_design_identity():
    """With n1 = n0 the assignment set is closed under swapping arms, so the
    difference-in-means distribution is symmetric: p_two = 2 * p_upper."""
    out = sp.ri_test(RI_DATA["simple"], "y", "d", n_perms=10000)
    assert out["p_value"] == pytest.approx(2 * out["p_one_sided"], abs=1e-15)


# --------------------------------------------------------------------------
# Rosenbaum bounds
# --------------------------------------------------------------------------

PAIRS = pd.read_csv(_FIX / "inference_sens_pairs.csv")
DIFF = PAIRS["diff"].to_numpy(float)
ZERO = np.zeros_like(DIFF)
GAMMAS = R["rosenbaum"]["gamma"]


@pytest.mark.parametrize(
    "alternative,key",
    [
        ("greater", "senwilcox_greater"),
        ("less", "senwilcox_less"),
        ("two-sided", "senwilcox_twosided"),
    ],
)
def test_rosenbaum_wilcoxon_matches_dos2_senwilcox(alternative, key):
    out = sp.rosenbaum_bounds(DIFF, ZERO, alternative=alternative, gamma_grid=GAMMAS)
    _close(out.pvalue_upper, R["rosenbaum"][key], rtol=1e-12)


def test_rosenbaum_gamma_alias_and_long_format_agree():
    wide = sp.rosenbaum_gamma(PAIRS["y_t"], PAIRS["y_c"], gamma_grid=GAMMAS)
    long = pd.concat(
        [
            pd.DataFrame({"y": PAIRS["y_t"], "w": 1, "pair": PAIRS["pair"]}),
            pd.DataFrame({"y": PAIRS["y_c"], "w": 0, "pair": PAIRS["pair"]}),
        ]
    )
    via_df = sp.rosenbaum_bounds(
        data=long, y="y", treat="w", pair_id="pair", gamma_grid=GAMMAS
    )
    _close(wide.pvalue_upper, R["rosenbaum"]["senwilcox_greater"], rtol=1e-12)
    _close(via_df.pvalue_upper, R["rosenbaum"]["senwilcox_greater"], rtol=1e-12)


def test_rosenbaum_wilcox_zero_method_matches_rbounds_psens():
    out = sp.rosenbaum_bounds(DIFF, ZERO, gamma_grid=GAMMAS, zero_method="wilcox")
    assert R["rosenbaum"]["psens_gamma"] == GAMMAS
    # psens rounds its bounds to 4 decimals itself.
    np.testing.assert_array_equal(
        np.round(out.pvalue_upper, 4), R["rosenbaum"]["psens_upper"]
    )
    np.testing.assert_array_equal(
        np.round(out.pvalue_lower, 4), R["rosenbaum"]["psens_lower"]
    )


def test_rosenbaum_sign_test_matches_binom_test():
    g = sp.rosenbaum_bounds(DIFF, ZERO, gamma_grid=GAMMAS, method="sign")
    _close(g.pvalue_upper, R["rosenbaum"]["sign_upper_greater"], rtol=1e-12)
    _close(g.pvalue_lower, R["rosenbaum"]["sign_lower_greater"], rtol=1e-12)
    le = sp.rosenbaum_bounds(
        DIFF, ZERO, gamma_grid=GAMMAS, method="sign", alternative="less"
    )
    _close(le.pvalue_upper, R["rosenbaum"]["sign_upper_less"], rtol=1e-12)


def test_rosenbaum_bounds_coincide_at_gamma_one():
    """Identity: with no hidden bias the upper and lower bounds are one p-value."""
    for method in ("wilcoxon", "sign"):
        out = sp.rosenbaum_bounds(DIFF, ZERO, gamma_grid=[1.0], method=method)
        assert out.pvalue_upper[0] == pytest.approx(out.pvalue_lower[0], rel=1e-15)


def test_rosenbaum_two_sided_is_direction_symmetric():
    """Negating every difference must not change a two-sided bound (the
    pre-fix implementation reported p = 0 at Gamma = 3 for the negated data)."""
    a = sp.rosenbaum_bounds(DIFF, ZERO, alternative="two-sided", gamma_grid=GAMMAS)
    b = sp.rosenbaum_bounds(-DIFF, ZERO, alternative="two-sided", gamma_grid=GAMMAS)
    _close(a.pvalue_upper, b.pvalue_upper, rtol=1e-14)


# --------------------------------------------------------------------------
# E-values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    R["evalue"]["rd"],
    ids=lambda c: f"{c['cells']}_t{c['true']}_a{c['alpha']}_g{c['grid']}",
)
def test_evalue_rd_matches_EValue(case):
    out = sp.evalue_rd(
        *case["cells"], true=case["true"], alpha=case["alpha"], grid=case["grid"]
    )
    _close(out["evalue_estimate"], case["est"], rtol=1e-12)
    _close(out["evalue_ci"], case["lower"], rtol=1e-12)


@pytest.mark.parametrize(
    "case", R["evalue"]["bias_factor"], ids=lambda c: f"{c['rr_eu']}x{c['rr_ud']}"
)
def test_bias_factor_matches_EValue_multi_bound(case):
    _close(sp.bias_factor(case["rr_eu"], case["rr_ud"]), case["bias"], rtol=1e-14)


def test_bias_factor_at_the_evalue_recovers_the_risk_ratio():
    """Identity: the E-value e of RR solves B(e, e) = RR."""
    for rr in (1.3, 2.0, 4.5):
        e = sp.evalue(estimate=rr, measure="RR")["evalue_estimate"]
        assert sp.bias_factor(e, e) == pytest.approx(rr, rel=1e-13)


@pytest.mark.parametrize(
    "case", R["evalue"]["from_result_rr"], ids=lambda c: str(c["cells"])
)
def test_evalue_from_result_matches_twoXtwoRR_then_evalues_RR(case):
    res = sp.relative_risk(*case["cells"])
    _close(res.estimate, case["rr"], rtol=1e-13)
    _close(res.ci, [case["lo"], case["hi"]], rtol=1e-13)
    out = sp.evalue_from_result(res, measure="RR")
    _close(out["evalue_estimate"], case["e_point"], rtol=1e-13)
    _close(out["evalue_ci"], case["e_ci"], rtol=1e-13)


# --------------------------------------------------------------------------
# Oster (2019) -- robomit, rounded to 6 decimals by the package
# --------------------------------------------------------------------------

ROUND6 = 5e-7 + 1e-12


def test_oster_bounds_matches_robomit():
    ref = R["oster"]
    out = sp.oster_bounds(REG, y="y", treat="t", controls=["x1", "x2"])
    assert out["method"] == "exact"
    _close(out["r_max"], ref["rmax13"], rtol=1e-13)
    _close(out["delta_for_zero"], ref["delta_rm13"], rtol=0, atol=ROUND6)
    _close(out["beta_adjusted"], ref["beta_rm13_d1"], rtol=0, atol=ROUND6)
    for delta, key in ((0.5, "beta_rm13_d05"), (2.0, "beta_rm13_d2")):
        o = sp.oster_bounds(REG, y="y", treat="t", controls=["x1", "x2"], delta=delta)
        _close(o["beta_adjusted"], ref[key], rtol=0, atol=ROUND6)
    o1 = sp.oster_bounds(REG, y="y", treat="t", controls=["x1", "x2"], r_max=1.0)
    _close(o1["delta_for_zero"], ref["delta_rm1"], rtol=0, atol=ROUND6)


def test_oster_delta_matches_robomit():
    ref = R["oster"]
    out = sp.oster_delta(REG, "y", ["t"], ["x1", "x2"], n_boot=20)
    _close(out.model_info["r_max"], ref["rmax13"], rtol=1e-13)
    _close(out.model_info["delta_star"], ref["delta_rm13"], rtol=0, atol=ROUND6)
    _close(out.model_info["beta_star_delta1"], ref["beta_rm13_d1"], rtol=0, atol=ROUND6)
