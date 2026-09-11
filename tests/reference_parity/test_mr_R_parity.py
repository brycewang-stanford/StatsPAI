"""Mendelian randomisation family vs the R packages that define the methods.

Reference: ``_fixtures/mr_R.json`` written by ``_generate_mr_R.R`` from
MendelianRandomization 0.10.0, TwoSampleMR 0.7.9, RadialMR 1.2.4,
MRPRESSO 1.0 and mr.raps 0.4.3 on the 28-variant LDL-C -> CHD summary
statistics that ship with MendelianRandomization (``_fixtures/mr_ldl.csv``,
exported by the same script so both sides read identical bytes).

Every deterministic quantity is compared at 1e-10 relative or tighter; the
exceptions are listed where they occur:

* bootstrap standard errors (median, mode, PRESSO) are Monte Carlo on both
  sides and are not compared -- only the point estimates, which are
  deterministic;
* ``mr_raps`` with a robust loss solves a tau^2 root and integrates the
  loss's Gaussian moments; R's ``uniroot`` stops at its default tolerance
  and ``integrate`` at 1.2e-4 relative. The sandwich formula is therefore
  pinned exactly at R's own estimates and moments, and the fitted values
  are graded aligned (beta 5e-5, SE 5e-4, tau^2 1.5e-3) with a test showing
  StatsPAI's root is the more accurate one. The Gaussian (L2) fits, which
  need neither, match at 1e-8.
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
R = json.loads((_FIX / "mr_R.json").read_text(encoding="utf-8"))
D = pd.read_csv(_FIX / "mr_ldl.csv")
BX, BY, SX, SY = (D[c].to_numpy() for c in ("bx", "by", "bxse", "byse"))
N_EXP, N_OUT = 17723, 60801

TIGHT = 1e-10


def _close(ours, ref, rtol=TIGHT, atol=0.0):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float),
        np.asarray(ref, dtype=float),
        rtol=rtol,
        atol=atol,
    )


# --------------------------------------------------------------------------
# IVW / Egger
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["default", "fixed", "random"])
def test_ivw_matches_mendelianrandomization(model):
    r = sp.mr_ivw(BX, BY, SX, SY, model=model)
    ref = R[f"ivw_{model}"]
    _close(r["estimate"], ref["est"])
    _close(r["se"], ref["se"])


def test_ivw_default_is_random_effects_with_rse():
    r = sp.mr_ivw(BX, BY, SX, SY)
    assert r["model"] == R["ivw_default"]["model"] == "random"
    _close(r["rse"], R["ivw_default"]["rse"])
    _close(r["Q"], R["ivw_default"]["Q"])


def test_ivw_heterogeneity_matches_twosamplemr():
    r = sp.mr_heterogeneity(BX, BY, SY)
    ref = R["tsmr_ivw"]
    _close(r.Q, ref["Q"])
    assert r.Q_df == ref["Q_df"]
    # Q_p = 3e-10. Before 1.28.0 this was `1 - chi2.cdf`, accurate only to
    # ~1e-16 absolute, i.e. 1e-7 relative here; the survival function is exact.
    _close(r.Q_p, ref["Q_pval"])


def test_egger_matches_mendelianrandomization():
    r = sp.mr_egger(BX, BY, SX, SY)
    ref = R["egger"]
    _close(r["estimate"], ref["est"])
    _close(r["se"], ref["se"])
    _close(r["intercept"], ref["int"])
    _close(r["intercept_se"], ref["int_se"])


def test_egger_intercept_test_matches_twosamplemr():
    r = sp.mr_pleiotropy_egger(BX, BY, SY)
    ref = R["tsmr_egger"]
    _close(r.intercept, ref["b_i"])
    _close(r.se, ref["se_i"])
    _close(r.p_value, ref["pval_i"])


def test_egger_heterogeneity_matches_twosamplemr():
    r = sp.mr_heterogeneity(BX, BY, SY, method="egger")
    ref = R["tsmr_egger"]
    _close(r.Q, ref["Q"])
    assert r.Q_df == ref["Q_df"]
    _close(r.Q_p, ref["Q_pval"])


def test_mean_f_statistic_matches_mendelianrandomization():
    r = sp.mr_f_statistic(BX, SX)
    _close(r.f_mean, R["ivw_default"]["Fstat"])


def test_leave_one_out_is_default_ivw_on_each_subset():
    t = sp.mr_leave_one_out(BX, BY, SY).table
    _close(t["estimate"], R["loo"]["est"])
    _close(t["se"], R["loo"]["se"])


def test_multivariable_ivw_matches_mr_mvivw():
    df = pd.DataFrame(
        {"ldl": BX, "hdl": D["hdl"], "tg": D["tg"], "beta_y": BY, "se_y": SY}
    )
    r = sp.mr_multivariable(df, exposures=["ldl", "hdl", "tg"])
    _close(r.direct_effect["estimate"], R["mvivw"]["est"])
    _close(r.direct_effect["se"], R["mvivw"]["se"])


# --------------------------------------------------------------------------
# Median / mode point estimates
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, key",
    [
        ({}, "median_weighted"),
        ({"weighting": "simple"}, "median_simple"),
        ({"penalized": True}, "median_penalized"),
    ],
)
def test_median_estimates_match_mr_median(kwargs, key):
    r = sp.mr_median(BX, BY, SX, SY, n_boot=50, seed=0, **kwargs)
    _close(r["estimate"], R[key])


@pytest.mark.parametrize(
    "method, key", [("weighted", "mode_weighted"), ("simple", "mode_unweighted")]
)
def test_mode_estimates_match_mr_mbe(method, key):
    r = sp.mr_mode(BX, BY, SX, SY, method=method, n_boot=50, seed=0)
    # mr_mbe takes the arg-max of density() over its 512-point grid on
    # [min - 3h, max + 3h]; StatsPAI maximises over the same grid, so the
    # estimate is the same grid point, not merely a nearby value.
    _close(r.estimate, R[key])


# --------------------------------------------------------------------------
# Constrained maximum likelihood
# --------------------------------------------------------------------------


def test_cml_path_matches_mr_cml_for_every_k():
    r = sp.mr_cml(BX, BY, SX, SY, n=N_EXP, K_max=6)
    for row, ref in zip(r.path.itertuples(), R["cml_per_k"]):
        assert row.K == ref["K"]
        # cML iterates to |d theta| <= 1e-7 on both sides from the same
        # start in the same update order, so the paths coincide to 1e-9.
        _close(row.estimate, ref["est"], rtol=1e-9)
        _close(row.se, ref["se"], rtol=1e-9)


def test_cml_bic_selection_matches_mr_cml():
    r = sp.mr_cml(BX, BY, SX, SY, n=N_EXP)
    ref = R["cml_bic"]
    _close(r.estimate, ref["est"], rtol=1e-9)
    _close(r.se, ref["se"], rtol=1e-9)
    assert list(np.flatnonzero(r.invalid_snps) + 1) == ref["invalid"]


def test_cml_model_average_matches_mr_cml():
    r = sp.mr_cml(BX, BY, SX, SY, n=N_EXP, model_average=True)
    _close(r.estimate, R["cml_ma"]["est"], rtol=1e-9)
    _close(r.se, R["cml_ma"]["se"], rtol=1e-9)


def test_cml_without_sample_size_warns():
    with pytest.warns(UserWarning, match="n"):
        sp.mr_cml(BX, BY, SX, SY)


# --------------------------------------------------------------------------
# Robust adjusted profile score
# --------------------------------------------------------------------------


def test_raps_simple_matches_mr_raps():
    r = sp.mr_raps(BX, BY, SX, SY, loss="l2", over_dispersion=False)
    _close(r.estimate, R["raps_simple"]["est"], rtol=1e-8)
    _close(r.se, R["raps_simple"]["se"], rtol=1e-8)


def test_raps_l2_overdispersed_matches_mr_raps():
    r = sp.mr_raps(BX, BY, SX, SY, loss="l2", over_dispersion=True)
    ref = R["raps_l2"]
    _close(r.estimate, ref["est"], rtol=1e-8)
    _close(r.se, ref["se"], rtol=1e-8)
    _close(r.tau2, ref["tau2"], rtol=1e-8)
    _close(r.tau2_se, ref["tau2_se"], rtol=1e-7)


_K = {"huber": 1.345, "tukey": 4.685}


@pytest.mark.parametrize("loss", ["huber", "tukey"])
def test_raps_robust_sandwich_formula_at_r_estimates(loss):
    """The variance formula, pinned where it cannot be blamed on a solver.

    Evaluated at R's own (beta, tau^2) with R's own integrate() moments,
    StatsPAI's sandwich reproduces R's standard errors exactly; everything
    that separates the fitted values below is how tightly each side solves.
    """
    from statspai.mendelian.frontier.raps import _robust_vcov

    ref = R[f"raps_{loss}"]
    c = ref["consts"]
    V = _robust_vcov(
        BX, BY, SX, SY, ref["est"], ref["tau2"], (c["delta"], c["c1"], c["c2"], c["c3"])
    )
    _close(np.sqrt(V[0, 0]), ref["se"])
    _close(np.sqrt(V[1, 1]), ref["tau2_se"])


@pytest.mark.parametrize("loss", ["huber", "tukey"])
def test_raps_robust_moments_are_the_package_integrals(loss):
    from statspai.mendelian.frontier.raps import _gauss_moment, _rho

    rho = _rho(loss, _K[loss])
    c = R[f"raps_{loss}"]["consts"]
    delta = _gauss_moment(lambda x: x * rho(x, 1))
    # R's integrate() stops at rel.tol = 1.2e-4 and the Huber rho'' has a
    # jump, so its moments are good to ~1e-4; scipy quad reaches ~1e-12.
    _close(delta, c["delta"], rtol=1e-4)
    _close(_gauss_moment(lambda x: rho(x, 1) ** 2), c["c1"], rtol=1e-4)
    _close(_gauss_moment(lambda x: x**2 * rho(x, 2)), c["c3"], rtol=2e-4)


@pytest.mark.parametrize("loss", ["huber", "tukey"])
def test_raps_robust_fit_solves_its_equation_tighter_than_r(loss):
    """R's tau^2 comes from uniroot at its default tol (~1e-4 * scale).

    Plugging each side's (beta, tau^2) into the tau^2 estimating equation
    shows StatsPAI's root is the more accurate one -- so the remaining gap
    to R is R's stopping rule, not a different estimator.
    """
    from statspai.mendelian.frontier.raps import _gauss_moment, _rho

    rho = _rho(loss, _K[loss])
    delta = _gauss_moment(lambda x: x * rho(x, 1))

    def tau_eq(b, t2):
        v = t2 + SY**2 + SX**2 * b**2
        t = (BY - b * BX) / np.sqrt(v)
        return float(np.sum(SX**2 * (t * rho(t, 1) - delta) / v))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.mr_raps(BX, BY, SX, SY, loss=loss, over_dispersion=True)
    ref = R[f"raps_{loss}"]
    assert abs(tau_eq(r.estimate, r.tau2)) < abs(tau_eq(ref["est"], ref["tau2"]))
    # Aligned, not bit-exact: beta to 3e-5, its SE to 4e-4 and tau^2 to 1e-3
    # (Tukey is the looser of the two; the SE inherits the tau^2 gap).
    _close(r.estimate, ref["est"], rtol=5e-5)
    _close(r.se, ref["se"], rtol=5e-4)
    _close(r.tau2, ref["tau2"], rtol=1.5e-3)


# --------------------------------------------------------------------------
# Radial, Steiger, PRESSO
# --------------------------------------------------------------------------


def test_radial_contributions_match_ivw_radial():
    r = sp.mr_radial(BX, BY, SY, bonferroni=False)
    ref = R["radial"]
    # RadialMR reports the square-root weight sqrt(w_j) = |bx_j| / se_y,j.
    _close(np.sqrt(r.table["W"]), ref["Wj"])
    _close(r.table["q_contribution"], ref["Qj"])
    _close(r.total_Q, ref["Q"])
    assert [i + 1 for i in r.outliers] == ref["outliers"]


def test_radial_bonferroni_default_flags_a_subset():
    strict = sp.mr_radial(BX, BY, SY)
    assert set(strict.outliers) <= {i - 1 for i in R["radial"]["outliers"]}


def test_steiger_matches_twosamplemr():
    r = sp.mr_steiger(BX, SX, N_EXP, BY, SY, N_OUT)
    ref = R["steiger_fwd"]
    _close(r.r2_exposure, ref["r2_exp"])
    _close(r.r2_outcome, ref["r2_out"])
    assert r.correct_direction is ref["dir"]
    # p = 1.8e-73: only a survival-function evaluation gets this right.
    _close(r.steiger_pvalue, ref["p"], rtol=1e-12)


def test_presso_matches_mrpresso():
    r = sp.mr_presso(BX, BY, SX, SY, n_boot=2000, seed=1)
    ref = R["presso"]
    _close(r.raw_estimate, ref["raw_est"])
    _close(r.raw_se, ref["raw_se"])
    _close(r.global_test_rss_obs, ref["RSSobs"])
    assert [i + 1 for i in r.outliers] == ref["outliers"]
    _close(r.outlier_corrected_estimate, ref["cor_est"])
    _close(r.outlier_corrected_se, ref["cor_se"])


def test_dispatcher_routes_to_the_same_ivw():
    r = sp.mr("ivw", beta_exposure=BX, beta_outcome=BY, se_exposure=SX, se_outcome=SY)
    _close(r["estimate"], R["ivw_default"]["est"])
    _close(r["se"], R["ivw_default"]["se"])
