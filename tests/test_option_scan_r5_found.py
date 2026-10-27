"""Option scan, round 5: defects found outside the scanned packages.

* ``sp.regress("y ~ x | fe")`` dropped the fixed-effect block.
* ``sp.iv(absorb=)`` rescaled its SEs for the absorbed degrees of freedom
  but left t statistics and p-values at the unscaled values.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def fe_df():
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame(
        {"fe": rng.integers(0, 20, n), "z": rng.normal(size=n), "x": rng.normal(size=n)}
    )
    a = rng.normal(size=20)[df.fe] * 2
    df["d"] = df.z + a + rng.normal(size=n)
    df["y"] = 1 + 0.2 * df.d + 3 * a + df.x + 3 * rng.normal(size=n)
    return df


def test_regress_refuses_fixed_effect_block(fe_df):
    with pytest.raises(MethodIncompatibility):
        sp.regress("y ~ d + x | fe", data=fe_df)


@pytest.mark.parametrize("robust", ["nonrobust", "hc1"])
def test_iv_absorb_t_and_p_follow_the_reported_se(fe_df, robust):
    pf = pytest.importorskip("pyfixest")
    r = sp.iv("y ~ x + (d ~ z)", data=fe_df, absorb=["fe"], robust=robust)
    np.testing.assert_allclose(r.tvalues, r.params / r.std_errors, rtol=1e-12)
    ref = (
        pf.feols(
            "y ~ x | fe | d ~ z",
            data=fe_df,
            vcov="iid" if robust == "nonrobust" else "HC1",
        )
        .tidy()
        .loc["d"]
    )
    assert r.std_errors["d"] == pytest.approx(ref["Std. Error"], rel=1e-10)
    assert r.tvalues["d"] == pytest.approx(ref["t value"], rel=1e-10)
    assert r.pvalues["d"] == pytest.approx(ref["Pr(>|t|)"], rel=1e-8)
    V = np.asarray(r.data_info["var_cov"])
    np.testing.assert_allclose(np.sqrt(np.diag(V)), r.std_errors, rtol=1e-12)


def test_regress_refuses_iv_block(fe_df):
    with pytest.raises(MethodIncompatibility):
        sp.regress("y ~ x + (d ~ z)", data=fe_df)
