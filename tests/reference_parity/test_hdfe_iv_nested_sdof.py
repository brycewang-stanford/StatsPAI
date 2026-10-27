"""``sp.hdfe_ols`` IV when every absorbed fixed effect is nested in the cluster.

Reference: Stata 18 MP, ivreghdfe 1.1.4 / ivreg2 4.1.11 / reghdfe 6.13.1 on
the committed ``_fixtures/hdfe_iv.csv`` (``firm`` nested in ``ind``),
``_fixtures/_generate_hdfe_iv_nested_Stata.do``, printed with ``%24.17g``.

With every absorbed FE nested in the cluster reghdfe reports ``e(df_a) = 0``
and ivreghdfe still charges one degree of freedom for the swept-out constant:
``e(sdofminus) = 1``. Until 1.41 ``sp.hdfe_ols`` charged 0 there, so its
clustered SEs were too small by ~1/(2N) relative (6.4e-4 here at N = 785;
1e-7 at the 5M-row census benchmark, where fixest and ivreghdfe agree with
each other and not with us -- ``benchmarks/hdfe/iv_census_scale_bench.py``).
The partially nested design (``firm + city^q``, ``e(sdofminus) = 41 =
e(df_a)``) was already right and is kept as a control.

Tolerance 1e-7 relative, as ``test_hdfe_iv_ivreghdfe.py`` (absorber
convergence; observed <= 1e-10 with ``tol=1e-12``).
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

DATA = pd.read_csv(pathlib.Path(__file__).parent / "_fixtures" / "hdfe_iv.csv")
REL = 1e-7

# absorb -> (N, sdofminus, b_d, se_d, se_w, rkf, cdf, idstat, j)
STATA = {
    "firm": (
        785,
        1,
        1.0039502909272089,
        0.048084531110987372,
        0.043500948880801714,
        178.52847607449067,
        194.99734345671342,
        20.392209537879278,
        2.0163517811607039,
    ),
    "ind": (
        800,
        1,
        1.0186611298071833,
        0.043396696824074266,
        0.039604275823932243,
        238.85754716655958,
        199.13389925717351,
        21.452240812202561,
        2.2075751449796011,
    ),
    "firm + ind": (
        785,
        1,
        1.0039502909272089,
        0.048084531110987372,
        0.043500948880801714,
        178.5284760744907,
        194.99734345671342,
        20.392209537879278,
        2.0163517811607039,
    ),
    "firm + city^q": (
        785,
        41,
        1.0180579528230513,
        0.057362686321262006,
        0.043358104073257034,
        124.8692874304197,
        178.35783084021691,
        19.513974550640789,
        1.4986729187880203,
    ),
}


@pytest.mark.parametrize("absorb", sorted(STATA))
def test_matches_ivreghdfe(absorb):
    n, sdof, b_d, se_d, se_w, rkf, cdf, idstat, j = STATA[absorb]
    r = sp.hdfe_ols(f"y ~ w | {absorb} | d ~ z1 + z2", DATA, cluster="ind", tol=1e-12)
    iv = r.iv_diagnostics
    assert r.n_obs == n
    assert iv["sdofminus"] == sdof
    assert float(r.coef["d"]) == pytest.approx(b_d, rel=REL)
    np.testing.assert_allclose(r.se[["d", "w"]], [se_d, se_w], rtol=REL)
    assert iv["kp_rk_wald_F"] == pytest.approx(rkf, rel=REL)
    assert iv["cragg_donald_F"] == pytest.approx(cdf, rel=REL)
    assert iv["kp_rk_lm"] == pytest.approx(idstat, rel=REL)
    assert iv["hansen_j"] == pytest.approx(j, rel=REL)


def test_all_nested_residual_df_charges_the_constant():
    """df_r = N - K - 1, and charging 0 instead (the 1.41 behaviour) moves
    the SE by sqrt((N-K-1)/(N-K)) - 1 = -6.4e-4, far outside REL: the
    comparison above discriminates between the two conventions."""
    r = sp.hdfe_ols("y ~ w | firm | d ~ z1 + z2", DATA, cluster="ind", tol=1e-12)
    N, K = r.n_obs, 2
    assert r.df_resid == N - K - 1
    assert abs(np.sqrt((N - K - 1) / (N - K)) - 1) > 1000 * REL
