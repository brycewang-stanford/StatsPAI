"""The two absorbed-IV entry points must stay numerically identical.

``sp.iv(..., absorb=)`` (the IV dispatcher: LIML / GMM / multiway clusters)
and ``sp.hdfe_ols("y ~ x | fe | d ~ z")`` (the reghdfe absorber, 2SLS) are
separate code paths that both claim ``ivreghdfe``. Each is pinned to Stata
elsewhere (``test_hdfe_iv_ivreghdfe.py``, ``test_iv_hdfe_inference.py``);
this gate fails the moment they drift apart, which is how the missing-``a^b``
sample bug of ``sp.iv`` showed up (Table 6 of Zheng, Huang and Zhu 2026:
-0.136797 against -0.1381572). Tolerance rel 1e-8: two sweeps to 1e-12.
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

DATA = pd.read_csv(pathlib.Path(__file__).parent / "_fixtures" / "hdfe_iv.csv")
DATA_MISSING = DATA.assign(q=DATA.q.where(np.arange(len(DATA)) % 17 != 0))
REL = 1e-8

VCE = {
    "cluster": ({"cluster": "ind"}, {"cluster": "ind"}),
    "robust": ({"robust": "hc1"}, {"vce": "robust"}),
    "iid": ({}, {}),
}


@pytest.mark.parametrize("vce", sorted(VCE))
@pytest.mark.parametrize("data", [DATA, DATA_MISSING], ids=["complete", "missing_q"])
@pytest.mark.parametrize("insts", ["z1", "z1 + z2"])
def test_sp_iv_absorb_equals_hdfe_ols_iv(vce, data, insts):
    iv_kw, h_kw = VCE[vce]
    a = sp.iv(f"y ~ w + (d ~ {insts})", data=data, absorb="firm + city^q", **iv_kw)
    b = sp.hdfe_ols(f"y ~ w | firm + city^q | d ~ {insts}", data, tol=1e-12, **h_kw)
    assert a.data_info["nobs"] == b.n_obs
    for nm in ("d", "w"):
        assert float(a.params[nm]) == pytest.approx(float(b.coef[nm]), rel=REL)
        assert float(a.std_errors[nm]) == pytest.approx(float(b.se[nm]), rel=REL)
    D, iv = a.diagnostics, b.iv_diagnostics
    assert D["KP rk LM"] == pytest.approx(iv["kp_rk_lm"], rel=REL)
    if vce != "iid":
        assert D["KP rk Wald F"] == pytest.approx(iv["kp_rk_wald_F"], rel=REL)
    if "+" in insts:
        j = D.get("Hansen J statistic", D.get("Sargan statistic"))
        assert j == pytest.approx(iv["hansen_j"], rel=1e-6)


def test_effective_f_is_computed_with_interacted_fe():
    """It used to be replaced by "OP effective F error": KeyError('city^q')."""
    a = sp.iv("y ~ (d ~ z1)", data=DATA, absorb="firm + city^q", cluster="ind")
    assert "OP effective F error" not in a.diagnostics
    # one instrument, clustered: the effective F is the KP rk Wald F
    assert a.diagnostics["Olea-Pflueger effective F"] == pytest.approx(
        a.diagnostics["KP rk Wald F"], rel=1e-7
    )
