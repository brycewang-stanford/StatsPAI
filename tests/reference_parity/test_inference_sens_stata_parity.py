"""Inference / sensitivity family vs Stata.

Reference: ``_fixtures/inference_sens_stata.json`` written by
``_fixtures/_generate_inference_sens_stata.do`` (Stata 18 MP; SSC boottest
4.5.3, ritest 1.1.7, rbounds 1.1.6, psacalc 2.1 -- the ``*!`` version lines
are recorded in ``_meta``) on the same CSVs as the R fixture.

What each comparison depends on
-------------------------------
* ``regress, vce(cluster s12)``: CR1, G/(G-1) * (N-1)/(N-K) -- the default
  of ``sp.cluster_robust_se``. 1e-10.
* ``regress, vce(jackknife, cluster(s12) double)`` centres the
  delete-one-cluster replicates at their mean (``sp.jackknife_se``); ``mse``
  centres them at the full-sample estimate (``sp.cr3_jackknife_vcov``). Both
  scale by (G-1)/G and use t(G-1) for p-values and intervals. 1e-10
  (p-values 1e-9). Without ``double`` the jackknife prefix stores its
  replicates in float and the SEs move by ~7e-8 relative; that default is
  recorded too and asserted to sit within float precision of ours.
* ``boottest`` (Roodman): WCR, Rademacher, G = 12 <= log2(reps), so boottest
  enumerates all 4096 sign vectors -- p is exact on both sides and compared
  exactly (it is a multiple of 1/4096); t at 1e-10. The subcluster case
  clusters the CRVE at g6 (6 clusters) and flips signs at s12
  (``bootcluster(s12)``).
* boottest's confidence interval is NOT a parity target (T4). In the
  enumerated regime the p-value is a step function of the null value, and
  boottest's Chandrupatla search (``boottest.mata``, ``search()``) returns
  early -- when |p - alpha| < 1/B, or when a probe lands on a plateau whose
  p equals a bracket end's -- so the endpoint it reports is an arbitrary
  point of the final bracket. On this design both of its endpoints are
  values its own test rejects (p = 204/4096 and 202/4096 < 0.05). StatsPAI
  and R fwildclusterboot (pinned in the R test to 1e-9) return the jump; the
  test asserts that evidence directly.
* ``ritest`` is run over the complete assignment set of each design
  (``samplingsourcefile()``, one column per assignment); c = #{|T| >= |T(obs)|
  - 1e-7}. p compared exactly. ritest keeps T(obs) in single precision (the
  observed 0.713245153427124 vs 0.7132451475949962 in double), so the
  observed statistic is compared at float epsilon, 2^-23 relative.
* ``rbounds`` (Gangl) ranks zero differences with the others and weights them
  0 (``zero_method="pratt"``), no continuity correction; sig+ / sig- are the
  upper / lower bounds. sig- is computed as ``1 - normprob(z)``, which has
  absolute error ~1e-16, so it is compared at 1e-15 absolute (its smallest
  value is 1.6e-10).
* ``psacalc`` is Oster's own implementation; exact delta (for beta = 0 and
  beta = 0.3) and exact beta* (quadratic at delta = 1, cubic at 0.5 and 2),
  with and without ``mcontrol()``. 1e-12 (observed <= 5e-14).
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
S = json.loads((_FIX / "inference_sens_stata.json").read_text(encoding="utf-8"))
REG = pd.read_csv(_FIX / "inference_sens_reg.csv")
VARS = ["_cons", "d", "x1", "x2"]


def _close(ours, ref, rtol=1e-10, atol=0.0):
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
    return X, y, b, y - X @ b


# --------------------------------------------------------------------------
# regress, vce(cluster) / vce(jackknife)
# --------------------------------------------------------------------------


def test_cluster_robust_se_matches_regress_vce_cluster():
    ref = S["regress_cluster_s12"]
    X, _, b, e = _design()
    _close(b, [ref[f"b_{v}"] for v in VARS])
    _close(
        sp.cluster_robust_se(X, e, REG["s12"].to_numpy()),
        [ref[f"se_{v}"] for v in VARS],
    )


def test_jackknife_se_matches_vce_jackknife_cluster():
    ref = S["jackknife_s12"]
    fit = sp.regress("y ~ d + x1 + x2", data=REG)
    jk = sp.jackknife_se(fit, REG, cluster="s12")
    names = ["Intercept", "d", "x1", "x2"]
    assert list(jk.params.index) == names
    _close(jk.params.to_numpy(), [ref[f"b_{v}"] for v in VARS])
    _close(jk.std_errors.to_numpy(), [ref[f"se_{v}"] for v in VARS])
    assert jk.data_info["df_resid"] == ref["df_r"] == 11
    _close(np.asarray(jk.pvalues), [ref[f"p_{v}"] for v in VARS], rtol=1e-9)
    ci = jk.conf_int()
    _close(ci.iloc[:, 0].to_numpy(), [ref[f"ll_{v}"] for v in VARS])
    _close(ci.iloc[:, 1].to_numpy(), [ref[f"ul_{v}"] for v in VARS])


def test_stata_float_jackknife_is_within_float_precision():
    """Stata's default (no ``double``) keeps replicates in single precision."""
    fit = sp.regress("y ~ d + x1 + x2", data=REG)
    jk = sp.jackknife_se(fit, REG, cluster="s12")
    ours, flt = float(jk.std_errors["d"]), S["jackknife_s12_float"]["se_d"]
    assert abs(ours / flt - 1) < 2.0**-23
    assert abs(ours / flt - 1) > 1e-10  # it is a real, float-sized difference


def test_cr3_jackknife_vcov_matches_vce_jackknife_mse():
    ref = S["jackknife_mse_s12"]
    X, y, _, _ = _design()
    V = sp.cr3_jackknife_vcov(X, y, REG["s12"].to_numpy())
    _close(np.sqrt(np.diag(V)), [ref[f"se_{v}"] for v in VARS])


# --------------------------------------------------------------------------
# boottest
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,var,h0",
    [
        ("boottest_d_h0", "d", 0.0),
        ("boottest_x1_h0", "x1", 0.0),
        ("boottest_d_h02", "d", 0.2),
    ],
)
def test_wild_cluster_bootstrap_matches_boottest(key, var, h0):
    ref = S[key]
    out = sp.wild_cluster_bootstrap(
        REG, "y", ["d", "x1", "x2"], "s12", test_var=var, h0=h0, n_boot=9999
    )
    assert out["n_boot"] == ref["reps"] == 4096
    assert out["p_boot"] * 4096 == pytest.approx(ref["p"] * 4096, abs=1e-9)
    _close(out["t_stat"], ref["t"])


def test_wild_cluster_boot_matches_boottest():
    ref = S["boottest_d_h0"]
    fit = sp.regress("y ~ d + x1 + x2", data=REG)
    out = sp.wild_cluster_boot(fit, REG, cluster="s12", variable="d", n_boot=9999)
    assert out["p_boot"] * 4096 == pytest.approx(ref["p"] * 4096, abs=1e-9)
    _close(out["t_stat"], ref["t"])


def test_subcluster_wild_bootstrap_matches_boottest_bootcluster():
    ref = S["boottest_sub_d"]
    out = sp.subcluster_wild_bootstrap(
        REG,
        "y",
        ["d", "x1", "x2"],
        "g6",
        subcluster="s12",
        test_var="d",
        n_boot=9999,
        weight_type="rademacher",
    )
    assert out["enumerated"] and out["n_boot"] == ref["reps"] == 4096
    assert out["n_clusters"] == 6 and out["n_subclusters"] == 12
    assert out["p_boot"] * 4096 == pytest.approx(ref["p"] * 4096, abs=1e-9)
    _close(out["t_stat"], ref["t"])


@pytest.mark.parametrize("key,var", [("boottest_d_h0", "d"), ("boottest_x1_h0", "x1")])
def test_boottest_ci_endpoints_are_rejected_values_outside_the_jumps(key, var):
    """T4: boottest stops its CI search early on a step function.

    ``search()`` returns as soon as |p - alpha| < 1/B, or immediately when a
    probe's p equals a bracket end's p ("violation of monotonicity") -- which
    on a step-function p happens whenever the probe lands on a plateau. The
    reported endpoint is therefore some point of the bracket, not the jump.
    Evidence, using StatsPAI's p-value function (identical to boottest's at
    every null value compared above):
      * boottest's endpoints are REJECTED values (p < alpha), so its interval
        contains points its own test rejects;
      * StatsPAI's endpoints are the jumps: p >= alpha just inside, p < alpha
        just outside -- and they equal R fwildclusterboot's (R test, 1e-9);
      * StatsPAI's interval therefore sits strictly inside boottest's."""
    ref = S[key]

    def p_at(h0):
        return sp.wild_cluster_bootstrap(
            REG, "y", ["d", "x1", "x2"], "s12", test_var=var, h0=h0, n_boot=9999
        )["p_boot"]

    for end in (ref["ci_lo"], ref["ci_hi"]):
        assert p_at(end) < 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = sp.wild_cluster_ci_inv(
            REG,
            "y",
            ["d", "x1", "x2"],
            "s12",
            test_var=var,
            n_boot=9999,
            weight_type="rademacher",
        )["ci"]
    assert ref["ci_lo"] < ours[0] and ours[1] < ref["ci_hi"]
    eps = 1e-9 * (ours[1] - ours[0])
    assert p_at(ours[0] + eps) >= 0.05 > p_at(ours[0] - eps)
    assert p_at(ours[1] - eps) >= 0.05 > p_at(ours[1] + eps)


# --------------------------------------------------------------------------
# ritest over the full assignment set
# --------------------------------------------------------------------------

RI_DATA = {
    k: pd.read_csv(_FIX / f"inference_sens_ri_{k}.csv")
    for k in ("simple", "cluster", "strat")
}
FLOAT_EPS = 2.0**-23


@pytest.mark.parametrize(
    "key,stat",
    [("ri_simple_diff", "diff_means"), ("ri_simple_t", "t"), ("ri_simple_ks", "ks")],
)
def test_ri_test_matches_ritest(key, stat):
    ref = S[key]
    out = sp.ri_test(RI_DATA["simple"], "y", "d", stat=stat, n_perms=10000)
    n = int(ref["reps"])
    assert out["n_perms"] == n
    _close(out["observed"], ref["observed"], rtol=FLOAT_EPS)
    assert out["p_value"] * n == pytest.approx(ref["p_two"] * n, abs=1e-9)
    assert out["p_one_sided"] * n == pytest.approx(ref["p_upper"] * n, abs=1e-9)


@pytest.mark.parametrize(
    "key,statistic,kw",
    [
        ("ri_simple_diff", "ate", {}),
        ("ri_simple_ks", "ks", {}),
        ("ri_simple_ranksum", "rank_sum", {}),
        ("ri_cluster_diff", "ate", {"cluster": "cl"}),
        ("ri_strat_diff", "ate", {"stratify": "st"}),
    ],
)
def test_fisher_exact_matches_ritest(key, statistic, kw):
    ref = S[key]
    data = RI_DATA[key.split("_")[1]]
    out = sp.fisher_exact(data, "y", "d", statistic=statistic, n_perm=10000, **kw)
    n = int(ref["reps"])
    assert out.n_perm == n
    _close(out.statistic, ref["observed"], rtol=FLOAT_EPS)
    assert out.p_value * n == pytest.approx(ref["p_two"] * n, abs=1e-9)
    assert out.p_one_sided * n == pytest.approx(ref["p_upper"] * n, abs=1e-9)


# --------------------------------------------------------------------------
# rbounds
# --------------------------------------------------------------------------


def test_rosenbaum_bounds_match_rbounds():
    ref = S["rbounds"]
    gam = [ref[f"gamma_{i}"] for i in range(1, 6)]
    diff = pd.read_csv(_FIX / "inference_sens_pairs.csv")["diff"].to_numpy(float)
    out = sp.rosenbaum_bounds(diff, np.zeros_like(diff), gamma_grid=gam)
    _close(out.pvalue_upper, [ref[f"sig_plus_{i}"] for i in range(1, 6)], rtol=1e-12)
    _close(
        out.pvalue_lower,
        [ref[f"sig_minus_{i}"] for i in range(1, 6)],
        rtol=0,
        atol=1e-15,
    )


# --------------------------------------------------------------------------
# psacalc
# --------------------------------------------------------------------------

OSTER_RTOL = 1e-12


def test_oster_bounds_matches_psacalc():
    ref = S["psacalc"]
    base = dict(y="y", treat="t", controls=["x1", "x2"])
    out = sp.oster_bounds(REG, **base)
    _close(out["r2_long"], ref["r_tilde"], rtol=1e-13)
    _close(out["r_max"], ref["rmax13"], rtol=1e-13)
    _close(out["delta_for_zero"], ref["delta_rm13"], rtol=OSTER_RTOL)
    _close(out["beta_adjusted"], ref["beta_rm13_d1"], rtol=OSTER_RTOL)
    _close(
        out["beta_adjusted_alternatives"][0], ref["beta_rm13_d1_alt1"], rtol=OSTER_RTOL
    )
    for delta, key in ((0.5, "beta_rm13_d05"), (2.0, "beta_rm13_d2")):
        _close(
            sp.oster_bounds(REG, delta=delta, **base)["beta_adjusted"],
            ref[key],
            rtol=OSTER_RTOL,
        )
    one = sp.oster_bounds(REG, r_max=1.0, **base)
    _close(one["delta_for_zero"], ref["delta_rm1"], rtol=OSTER_RTOL)
    _close(one["beta_adjusted"], ref["beta_rm1_d1"], rtol=OSTER_RTOL)


def test_oster_delta_matches_psacalc():
    ref = S["psacalc"]
    out = sp.oster_delta(REG, "y", ["t"], ["x1", "x2"], n_boot=20)
    _close(out.model_info["r_max"], ref["rmax13"], rtol=1e-13)
    _close(out.model_info["delta_star"], ref["delta_rm13"], rtol=OSTER_RTOL)
    _close(out.model_info["beta_star_delta1"], ref["beta_rm13_d1"], rtol=OSTER_RTOL)
    # identified set at delta = 1 is [beta*, beta_tilde]
    assert out.lower == pytest.approx(ref["beta_rm13_d1"], rel=OSTER_RTOL)
    # further x_base entries are psacalc's mcontrol(): in both regressions
    mc = sp.oster_delta(REG, "y", ["t", "x2"], ["x1"], n_boot=20)
    _close(mc.model_info["delta_star"], ref["delta_rm13_mc_x2"], rtol=OSTER_RTOL)
    _close(
        mc.model_info["beta_star_delta1"], ref["beta_rm13_d1_mc_x2"], rtol=OSTER_RTOL
    )


def test_oster_exact_delta_for_a_nonzero_target():
    from statspai.diagnostics._oster import oster_delta_exact, oster_inputs

    ref = S["psacalc"]
    inp = oster_inputs(REG, "y", "t", ["x1", "x2"])
    _close(
        oster_delta_exact(inp, ref["rmax13"], beta=0.3),
        ref["delta_rm13_beta03"],
        rtol=OSTER_RTOL,
    )


def test_oster_exact_roots_are_consistent():
    """Identity: beta*(delta*) = 0 -- the delta that drives the bias-adjusted
    coefficient to zero, fed back into the cubic, returns a root at zero."""
    from statspai.diagnostics._oster import (
        oster_beta_exact,
        oster_delta_exact,
        oster_inputs,
    )

    inp = oster_inputs(REG, "y", "t", ["x1", "x2"])
    rm = min(1.0, 1.3 * inp["r_t"])
    d_star = oster_delta_exact(inp, rm, beta=0.0)
    roots = oster_beta_exact(inp, rm, delta=d_star)["roots"]
    assert min(abs(r) for r in roots) < 1e-10
