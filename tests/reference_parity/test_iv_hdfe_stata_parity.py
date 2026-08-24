"""Reference parity: ``sp.iv(absorb=...)`` against Stata ``ivreghdfe`` / ``acreg``.

The design under test is the one a county x year-month policy-intensity
study needs — the shape of Zhang et al. (2026, *Science*): high-dimensional
county and year-month fixed effects, an interaction instrument, inference
clustered at the county (the same dimension that is absorbed), and spatial
robustness. Every number below was produced locally with **Stata 18 MP**,
``ivreghdfe 1.1.4`` / ``ivreg2 4.1.12`` / ``reghdfe 6.13.1`` / ``acreg 1.1.0``
on ``_fixtures/iv_hdfe_panel.csv`` read at full double precision::

    import delimited "iv_hdfe_panel.csv", clear stringcols(_all)
    foreach v of varlist _all {
        gen double d_`v' = real(`v')
        drop `v'
        rename d_`v' `v'
    }
    ivreghdfe shannon temp wind (policy = z), absorb(county ym) cluster(county)
    ivreghdfe shannon temp wind (policy = z), absorb(county ym) cluster(county ym)
    ivreghdfe shannon temp wind (policy = z), absorb(county ym) robust
    ivreghdfe shannon temp wind (policy = z), absorb(county ym)
    ivreghdfe shannon temp wind (policy = z z2), absorb(county ym) cluster(county)
    ivreghdfe shannon temp wind (policy = z z2), absorb(county ym) cluster(county) liml
    ivreghdfe shannon temp wind (policy = z z2), absorb(county ym) cluster(county) gmm2s
    ivreghdfe shannon temp wind (policy = z z2), absorb(county ym) robust liml
    ivreghdfe shannon temp wind (policy = z z2), absorb(county ym) robust fuller(1)
    acreg shannon temp wind (policy = z), spatial latitude(lat) longitude(lon) ///
        dist(500) pfe1(county) pfe2(ym)
    acreg shannon temp wind (policy = z), spatial latitude(lat) longitude(lon) ///
        dist(500) pfe1(county) pfe2(ym) hac bartlett lag(3) id(county) time(ym)

Two documented convention gaps, both asserted here rather than papered over:

* **k-class robust/cluster meat.** StatsPAI builds the sandwich meat from
  ``A X`` with ``A = I - kappa M_W`` (the k-class influence function);
  ``ivreg2`` uses the 2SLS projection ``X_hat = P_W X`` even for LIML and
  Fuller. The two agree to ``O(kappa - 1)``, here a few parts per million.
* **GMM variance.** ``ivreg2 gmm2s`` reports the efficient-GMM variance
  ``q (X'W S^-1 W'X)^-1``; StatsPAI defaults to the full sandwich, which
  stays valid when the weight matrix is not the efficient one. Pass
  ``gmm_vcov="efficient"`` for the ``ivreg2`` number — asserted exactly.

References
----------
- Baum, C. F., Schaffer, M. E. and Stillman, S. (2007). "Enhanced routines
  for instrumental variables/generalized method of moments estimation and
  testing." *The Stata Journal*, 7(4), 465-506.
  doi:10.1177/1536867X0800700402 [@baum2007enhanced]
- Colella, F., Lalive, R., Sakalli, S. O. and Thoenig, M. (2023).
  "acreg: Arbitrary correlation regression." *The Stata Journal*.
  doi:10.1177/1536867X231162031 [@colella2023acreg]
- Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2011). "Robust Inference
  With Multiway Clustering." *Journal of Business & Economic Statistics*,
  29(2), 238-249. doi:10.1198/jbes.2010.07136 [@cameron2011robust]
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "iv_hdfe_panel.csv"

JUST_IDENTIFIED = "shannon ~ (policy ~ z) + temp + wind"
OVER_IDENTIFIED = "shannon ~ (policy ~ z + z2) + temp + wind"
ABSORB = ["county", "ym"]

# --- ivreghdfe, just-identified -----------------------------------------
STATA_B_JUST = -2.40046966112762e-02
STATA_2SLS = {
    "cluster_county": 1.66539421841015e-02,
    "cluster_county_ym": 1.32017925482374e-02,
    "robust": 1.51196074647892e-02,
    "iid": 1.54843554455976e-02,
}
STATA_KP_WALD_F = 2.09620268591704e03

# --- ivreghdfe, over-identified -----------------------------------------
STATA_OVER = {
    # (method kwargs, coefficient, std error)
    "2sls_cluster": (-2.02181534724847e-02, 1.48590973542333e-02),
    "liml_cluster": (-2.02844273799157e-02, 1.48625967398737e-02),
    "gmm_cluster": (-1.92236423546054e-02, 1.47730887837631e-02),
    "2sls_robust": (-2.02181534724847e-02, 1.38634772627139e-02),
    "liml_robust": (-2.02844273799157e-02, 1.38673165488081e-02),
    "fuller_robust": (-2.01456601802608e-02, 1.38592795584793e-02),
}
STATA_LIML_KAPPA = 1.00016601909601e00
STATA_FULLER_KAPPA = 9.99818313949975e-01
STATA_HANSEN_J = 3.94762457041120e-01

# --- ivreg2 rank tests: (n_instruments, vcov) -> (rk LM, rk Wald F) -----
STATA_RANK = {
    (1, "cluster_county"): (3.890494388810e01, 2.096202685917e03),
    (1, "robust"): (3.016257439897e02, 1.067328216941e03),
    (1, "iid"): (1.050054334105e03, 1.568816278508e03),
    (1, "cluster_county_ym"): (7.584019002102e00, 6.536918316461e01),
    (2, "cluster_county"): (3.921732349896e01, 1.454750653746e03),
    (2, "robust"): (3.199160933481e02, 1.067987183952e03),
    (2, "iid"): (1.210402994671e03, 9.906676203536e02),
    (2, "cluster_county_ym"): (8.869274237587e00, 2.337287398327e03),
}
# Cragg-Donald F, i.e. e(cdf), just-identified spec. It is not vcov-free:
# the clustered run charges fewer absorbed DOF (the county FE is nested),
# which raises it.
STATA_CRAGG_DONALD_CLUSTERED = 1.637674344865e03
STATA_CRAGG_DONALD_IID = 1.568816278508e03

# --- ivreg2 over-identification tests, over-identified spec -------------
# Sargan under i.i.d. errors; Hansen J once the vcov is robust/clustered.
STATA_OVERID = {
    "iid": ("Sargan", 4.780779854272e-01),
    "robust": ("Hansen J", 4.927960914351e-01),
    "cluster_county": ("Hansen J", 3.947624570411e-01),
    "cluster_county_ym": ("Hansen J", 5.086494899161e-01),
}

# --- acreg spatial ------------------------------------------------------
ACREG_SPATIAL_SE = 1.86485730389168e-02
ACREG_SPATIAL_HAC_SE = 1.52339004530449e-02

# --- reghdfe OLS (the confounded benchmark) -----------------------------
STATA_OLS = (1.47516412028955e-01, 1.28949611710288e-02)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("cluster_county", {"cluster": "county"}),
        ("cluster_county_ym", {"cluster": ["county", "ym"]}),
        ("robust", {"robust": "hc1"}),
        ("iid", {}),
    ],
)
def test_absorbed_2sls_matches_ivreghdfe(panel, label, kwargs):
    """2SLS + HDFE + (multiway) clustering, to machine precision."""
    res = sp.iv(JUST_IDENTIFIED, data=panel, absorb=ABSORB, **kwargs)
    np.testing.assert_allclose(res.params["policy"], STATA_B_JUST, rtol=1e-12)
    np.testing.assert_allclose(res.std_errors["policy"], STATA_2SLS[label], rtol=1e-10)


def test_nested_fe_is_dropped_from_the_dof_charge(panel):
    """``cluster="county"`` makes the county FE redundant (``e(df_a)=23``)."""
    res = sp.iv(JUST_IDENTIFIED, data=panel, absorb=ABSORB, cluster="county")
    assert res.model_info["fe_nested_in_cluster"] == ["county"]
    # ivreghdfe: Absorbed FE table reports county 120/120 redundant, ym 24/1
    # redundant -> df_a = 23, and e(sdofminus) = 23.
    assert res.model_info["fe_dof_charged"] == 23

    both = sp.iv(JUST_IDENTIFIED, data=panel, absorb=ABSORB, cluster=["county", "ym"])
    assert both.model_info["fe_nested_in_cluster"] == ["county", "ym"]
    # e(df_a) = 0 but e(sdofminus) = 1: the constant is still charged.
    assert both.model_info["fe_dof_charged"] == 1

    plain = sp.iv(JUST_IDENTIFIED, data=panel, absorb=ABSORB, robust="hc1")
    assert plain.model_info["fe_nested_in_cluster"] == []
    assert plain.model_info["fe_dof_charged"] == 143  # 120 + 23


def test_kleibergen_paap_wald_f_matches_ivreghdfe(panel):
    res = sp.iv(JUST_IDENTIFIED, data=panel, absorb=ABSORB, cluster="county")
    np.testing.assert_allclose(
        res.diagnostics["KP rk Wald F"], STATA_KP_WALD_F, rtol=1e-8
    )


@pytest.mark.parametrize(
    "label,kwargs,rtol_se",
    [
        ("2sls_cluster", {"cluster": "county"}, 1e-10),
        ("2sls_robust", {"robust": "hc1"}, 1e-10),
        # LIML/Fuller: coefficients exact, SE within the documented
        # k-class-meat convention gap (ivreg2 uses X_hat, StatsPAI uses AX).
        ("liml_cluster", {"cluster": "county", "method": "liml"}, 5e-5),
        ("liml_robust", {"robust": "hc1", "method": "liml"}, 5e-5),
        (
            "fuller_robust",
            {"robust": "hc1", "method": "fuller", "fuller_alpha": 1.0},
            5e-5,
        ),
    ],
)
def test_absorbed_kclass_matches_ivreghdfe(panel, label, kwargs, rtol_se):
    ref_b, ref_se = STATA_OVER[label]
    res = sp.iv(OVER_IDENTIFIED, data=panel, absorb=ABSORB, **kwargs)
    np.testing.assert_allclose(res.params["policy"], ref_b, rtol=1e-10)
    np.testing.assert_allclose(res.std_errors["policy"], ref_se, rtol=rtol_se)


def test_absorbed_liml_and_fuller_kappa_match_ivreghdfe(panel):
    liml = sp.iv(
        OVER_IDENTIFIED, data=panel, absorb=ABSORB, robust="hc1", method="liml"
    )
    np.testing.assert_allclose(liml.model_info["kappa"], STATA_LIML_KAPPA, rtol=1e-10)
    fuller = sp.iv(
        OVER_IDENTIFIED,
        data=panel,
        absorb=ABSORB,
        robust="hc1",
        method="fuller",
        fuller_alpha=1.0,
    )
    np.testing.assert_allclose(
        fuller.model_info["kappa"], STATA_FULLER_KAPPA, rtol=1e-8
    )


def test_absorbed_gmm_efficient_vcov_matches_ivreg2(panel):
    ref_b, ref_se = STATA_OVER["gmm_cluster"]
    res = sp.iv(
        OVER_IDENTIFIED,
        data=panel,
        absorb=ABSORB,
        cluster="county",
        method="gmm",
        gmm_vcov="efficient",
    )
    np.testing.assert_allclose(res.params["policy"], ref_b, rtol=1e-10)
    np.testing.assert_allclose(res.std_errors["policy"], ref_se, rtol=1e-10)

    # The default sandwich is a different (more agnostic) estimator, but on
    # an efficiently weighted GMM fit it must land in the same neighbourhood.
    sandwich = sp.iv(
        OVER_IDENTIFIED,
        data=panel,
        absorb=ABSORB,
        cluster="county",
        method="gmm",
    )
    np.testing.assert_allclose(sandwich.std_errors["policy"], ref_se, rtol=5e-3)


def test_absorbed_iv_conley_matches_acreg(panel):
    """Spatial HAC on an absorbed 2SLS fit == ``acreg ... pfe1() pfe2()``."""
    res = sp.iv(JUST_IDENTIFIED, data=panel, absorb=ABSORB)
    spatial = sp.conley(
        res, panel, lat="lat", lon="lon", dist_cutoff=500, distance="planar"
    )
    np.testing.assert_allclose(
        spatial.std_errors["policy"], ACREG_SPATIAL_SE, rtol=1e-10
    )

    # Spatial + serial: acreg carries a (numerically zero) constant column
    # through the HAC block, which moves the last few digits only.
    spacetime = sp.conley(
        res,
        panel,
        lat="lat",
        lon="lon",
        dist_cutoff=500,
        distance="planar",
        time="ym",
        lag_cutoff=3,
        unit="county",
        time_kernel="bartlett",
    )
    np.testing.assert_allclose(
        spacetime.std_errors["policy"], ACREG_SPATIAL_HAC_SE, rtol=1e-3
    )


def test_ols_is_confounded_but_iv_is_not(panel):
    """The fixture has a real endogeneity problem, so the test has teeth."""
    pytest.importorskip(
        "pyfixest", reason="sp.feols is backed by the optional [fixest] extra"
    )
    ols = sp.feols(
        "shannon ~ policy + temp + wind | county + ym",
        data=panel,
        vcov={"CRV1": "county"},
    )
    np.testing.assert_allclose(ols.params["policy"], STATA_OLS[0], rtol=1e-6)
    iv = sp.iv(JUST_IDENTIFIED, data=panel, absorb=ABSORB, cluster="county")
    # OLS says +0.148, IV says -0.024: opposite signs, as designed.
    assert ols.params["policy"] > 0 > iv.params["policy"]


# =======================================================================
#  Weak-identification and over-identification diagnostics
# =======================================================================

VCOV_KWARGS = {
    "cluster_county": {"cluster": "county"},
    "cluster_county_ym": {"cluster": ["county", "ym"]},
    "robust": {"robust": "hc1"},
    "iid": {},
}


@pytest.mark.parametrize("n_z", [1, 2])
@pytest.mark.parametrize(
    "label", ["cluster_county", "cluster_county_ym", "robust", "iid"]
)
def test_kleibergen_paap_matches_ranktest(panel, n_z, label):
    """rk LM and rk Wald F, for every vcov ivreghdfe supports.

    These are the numbers a referee reads off the ivreghdfe output: the
    underidentification LM test and the weak-identification Wald F. Both
    have to track the estimator's own vcov -- a heteroskedasticity-only rk
    F next to cluster-robust coefficients overstates instrument strength.
    """
    formula = JUST_IDENTIFIED if n_z == 1 else OVER_IDENTIFIED
    res = sp.iv(formula, data=panel, absorb=ABSORB, **VCOV_KWARGS[label])
    ref_lm, ref_f = STATA_RANK[(n_z, label)]
    np.testing.assert_allclose(res.diagnostics["KP rk LM"], ref_lm, rtol=1e-8)
    np.testing.assert_allclose(res.diagnostics["KP rk Wald F"], ref_f, rtol=1e-8)


@pytest.mark.parametrize(
    "label", ["cluster_county", "cluster_county_ym", "robust", "iid"]
)
def test_overidentification_test_follows_the_vcov(panel, label):
    """Sargan under i.i.d. errors, Hansen J under robust/clustered."""
    expected_name, expected_stat = STATA_OVERID[label]
    res = sp.iv(OVER_IDENTIFIED, data=panel, absorb=ABSORB, **VCOV_KWARGS[label])
    key = f"{expected_name} statistic"
    assert key in res.diagnostics, sorted(res.diagnostics)
    np.testing.assert_allclose(res.diagnostics[key], expected_stat, rtol=1e-8)


def test_effective_f_equals_rank_f_with_one_instrument(panel):
    """With k_z = 1 the Olea-Pflueger F and the KP rk Wald F coincide.

    Both reduce to the same robust first-stage Wald statistic, so this is
    a genuine cross-check of two independently coded paths against the
    same Stata number.
    """
    for label, kwargs in VCOV_KWARGS.items():
        res = sp.iv(JUST_IDENTIFIED, data=panel, absorb=ABSORB, **kwargs)
        ref = STATA_RANK[(1, label)][1]
        np.testing.assert_allclose(
            res.diagnostics["Olea-Pflueger effective F"], ref, rtol=1e-8
        )
        np.testing.assert_allclose(res.diagnostics["KP rk Wald F"], ref, rtol=1e-8)


def test_classical_first_stage_f_matches_cragg_donald(panel):
    """The i.i.d. effective F is ivreg2's Cragg-Donald statistic."""
    out = sp.effective_f_test(
        panel,
        endog="policy",
        instruments=["z"],
        exog=["temp", "wind"],
        absorb=ABSORB,
        vcov="classic",
    )
    np.testing.assert_allclose(out["F_eff"], STATA_RANK[(1, "iid")][1], rtol=1e-8)
    np.testing.assert_allclose(out["first_stage_F"], STATA_CRAGG_DONALD_IID, rtol=1e-8)


def test_iv_diag_bundle_reproduces_the_stata_panel(panel):
    """One call must reproduce the whole ivreghdfe output block."""
    diag = sp.iv_diag(
        panel,
        y="shannon",
        endog="policy",
        instruments=["z"],
        exog=["temp", "wind"],
        absorb=ABSORB,
        cluster="county",
        n_boot=50,
        random_state=0,
    )
    np.testing.assert_allclose(diag.beta_2sls, STATA_B_JUST, rtol=1e-10)
    np.testing.assert_allclose(diag.se_2sls, STATA_2SLS["cluster_county"], rtol=1e-8)
    np.testing.assert_allclose(
        diag.kp_rk_f, STATA_RANK[(1, "cluster_county")][1], rtol=1e-8
    )
    np.testing.assert_allclose(
        diag.kp_rk_lm, STATA_RANK[(1, "cluster_county")][0], rtol=1e-8
    )
    np.testing.assert_allclose(
        diag.first_stage_F, STATA_CRAGG_DONALD_CLUSTERED, rtol=1e-8
    )
    # The AR set is size-correct whatever the first stage looks like; with
    # this strong an instrument it should sit close to the Wald interval.
    lo, hi = diag.ar_ci
    assert lo < STATA_B_JUST < hi
    assert abs(lo) < 0.2 and abs(hi) < 0.2
