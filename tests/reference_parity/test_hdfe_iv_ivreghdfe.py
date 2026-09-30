"""``sp.hdfe_ols("y ~ exog | FE | endog ~ inst")`` against Stata ``ivreghdfe``.

Reference: ivreghdfe 1.1.4 / ivreg2 4.1.11 / reghdfe 6.13.1 on
``_fixtures/hdfe_iv.csv`` (``_fixtures/_generate_hdfe_iv_Stata.do``): firm FE
nested in the cluster, a city x quarter FE, 15 singletons. Coefficients, SEs,
the t reference df and every identification statistic ivreghdfe prints
(Kleibergen-Paap rk LM and Wald F, Cragg-Donald F, Anderson-Rubin, Hansen J /
Sargan, first-stage F). Printed at 17 significant digits; tolerance rel 1e-7
covers the absorber's convergence tolerance (observed <= 5e-9).

Origin: Zheng, Huang & Zhu (2026) Table 6 (KP F = 509.244, KP LM = 30.714);
before 1.33 HDFE IV went through pyfixest (about 4 minutes on 1.5M rows) and
reported no weak-instrument diagnostics.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

DATA = pd.read_csv(pathlib.Path(__file__).parent / "_fixtures" / "hdfe_iv.csv")
FE = "firm + city^q"
REL = 1e-7


def _fit(iv="d ~ z1 + z2", exog="w", **kw):
    return sp.hdfe_ols(f"y ~ {exog} | {FE} | {iv}", DATA, tol=1e-12, **kw)


def test_cluster_two_instruments():
    r = _fit(cluster="ind")
    np.testing.assert_allclose(
        r.coef[["d", "w"]], [1.0180579528230513, -0.41248814512422199], rtol=REL
    )
    V = np.array(
        [
            [0.0032904777819914989, -0.00024720485875300631],
            [-0.00024720485875300631, 0.0018799251888273884],
        ]
    )
    np.testing.assert_allclose(r.vcov, V, rtol=REL)
    assert r.n_obs == 785 and r.n_singletons_dropped == 15
    assert r.df_inference == 24
    iv = r.iv_diagnostics
    assert iv["kp_rk_wald_F"] == pytest.approx(124.8692874304197, rel=REL)
    assert iv["kp_rk_lm"] == pytest.approx(19.513974550640789, rel=REL)
    assert iv["kp_rk_lm_df"] == 2
    assert iv["cragg_donald_F"] == pytest.approx(178.35783084021691, rel=REL)
    assert iv["anderson_rubin_F"] == pytest.approx(48.619690878755819, rel=REL)
    assert iv["anderson_rubin_chi2"] == pytest.approx(107.1689092694123, rel=REL)
    assert iv["hansen_j"] == pytest.approx(1.4986729187880203, rel=REL)
    fs = iv["first_stage"]["d"]
    assert fs["F"] == pytest.approx(124.86928743041979, rel=REL)
    assert fs["chi2"] == pytest.approx(275.24044462845569, rel=REL)
    assert fs["partial_r2"] == pytest.approx(0.32496180398333491, rel=REL)
    assert r.r2_within == pytest.approx(0.7478902293369859, rel=REL)
    assert r.r2_a_within == pytest.approx(0.73328009437942576, rel=REL)
    assert r.rmse == pytest.approx(0.89075900109674411, rel=REL)
    assert "Kleibergen-Paap" in r.summary()


def test_cluster_just_identified_no_exog():
    r = _fit(iv="d ~ z1", exog="1", cluster="ind")
    assert list(r.coef.index) == ["d"]
    assert float(r.coef["d"]) == pytest.approx(1.0143164570187264, rel=REL)
    assert float(r.se["d"]) == pytest.approx(0.078239080068530448, rel=REL)
    iv = r.iv_diagnostics
    assert iv["kp_rk_wald_F"] == pytest.approx(167.83274338194036, rel=REL)
    assert iv["cragg_donald_F"] == pytest.approx(258.57385959829963, rel=REL)
    assert iv["kp_rk_lm"] == pytest.approx(19.116638296227546, rel=REL)
    assert np.isnan(iv["hansen_j"])


def test_robust():
    r = _fit(vce="robust")
    np.testing.assert_allclose(
        r.se[["d", "w"]], [0.058183285821852344, 0.046223213398657211], rtol=REL
    )
    assert r.df_inference == 560
    iv = r.iv_diagnostics
    assert iv["kp_rk_wald_F"] == pytest.approx(123.62435676816656, rel=REL)
    assert iv["cragg_donald_F"] == pytest.approx(134.55064431805846, rel=REL)
    assert iv["kp_rk_lm"] == pytest.approx(154.1179812884819, rel=REL)
    assert iv["hansen_j"] == pytest.approx(2.5096820172450967, rel=REL)


def test_iid():
    r = _fit()
    np.testing.assert_allclose(
        r.se[["d", "w"]], [0.054928257846192735, 0.044831504559966824], rtol=REL
    )
    iv = r.iv_diagnostics
    assert iv["underid_test"].startswith("Anderson")
    assert iv["kp_rk_lm"] == pytest.approx(255.09501612691787, rel=REL)
    assert iv["cragg_donald_F"] == pytest.approx(134.55064431805835, rel=REL)
    assert iv["hansen_j"] == pytest.approx(2.4653823915086575, rel=REL)
    assert np.isnan(iv["kp_rk_wald_F"])


def test_iv_boundaries():
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="Underidentified"):
        _fit(iv="d + w ~ z1", exog="1", cluster="ind")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="weights"):
        _fit(cluster="ind", weights="z2")
    with pytest.raises(ValueError, match="fixed effect"):
        sp.hdfe_ols("y ~ w | | d ~ z1", DATA)


def test_sp_iv_absorb_drops_rows_with_a_missing_interaction_component():
    """``sp.iv(absorb='city^q')`` used to turn a missing ``q`` into a "nan"
    level and keep the row (reghdfe drops it): on the 1.5M-row replication
    of Zheng, Huang and Zhu (2026, Table 6) that gave -0.136797 instead of
    ivreghdfe's -0.1381572."""
    d = DATA.copy()
    d.loc[::17, "q"] = np.nan
    a = sp.iv("y ~ w + (d ~ z1 + z2)", data=d, absorb="firm + city^q", cluster="ind")
    b = sp.hdfe_ols("y ~ w | firm + city^q | d ~ z1 + z2", d, cluster="ind", tol=1e-12)
    assert a.data_info["nobs"] == b.n_obs == 732
    assert float(a.params["d"]) == pytest.approx(float(b.coef["d"]), rel=1e-9)
    assert float(a.std_errors["d"]) == pytest.approx(float(b.se["d"]), rel=1e-9)
