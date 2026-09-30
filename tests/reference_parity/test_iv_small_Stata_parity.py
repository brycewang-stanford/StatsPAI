"""``sp.iv(small=)`` reproduces ``ivregress`` with and without ``small``.

``sp.iv`` has always reported ``ivregress ..., small`` statistics. Stata's
default is large-sample: ``N`` divisor, HC0 under ``vce(robust)``, no
finite-sample factor under ``vce(cluster)``, and z p-values -- which a
replication of a paper that ran plain ``ivregress`` could not reproduce
(ADH / Kinship, the top-5 replication list). ``small=False`` gives those.

Reference: Stata 18 on ``_fixtures/ssc_presets.csv``
(``_generate_iv_small_Stata.do``). SEs agree to 1e-12; p-values to 1e-9
(Stata's local macro keeps 12 significant digits).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
STATA = json.loads((_FIX / "iv_small_Stata.json").read_text(encoding="utf-8"))

VCE = {"unadjusted": {}, "robust": {"robust": "robust"}, "cluster": {"cluster": "ind"}}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "ssc_presets.csv")


@pytest.mark.parametrize("small", [True, False])
@pytest.mark.parametrize("vce", list(VCE))
@pytest.mark.parametrize("method", ["2sls", "liml"])
def test_matches_ivregress(data, method, vce, small):
    ref = STATA[f"{method}_{vce}" + ("_small" if small else "")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.iv("y ~ w + (x ~ z)", data=data, method=method, small=small, **VCE[vce])
    assert r.params["x"] == pytest.approx(ref["b"], rel=1e-12)
    assert r.std_errors["x"] == pytest.approx(ref["se"], rel=1e-12)
    assert r.pvalues["x"] == pytest.approx(ref["p"], rel=1e-9)
    if not small:
        assert r.data_info["inference"] == "z"
        assert r.model_info["small"] is False
        # conf_int follows the same z convention
        ci = r.conf_int()
        half = (ci.loc["x"].iloc[1] - ci.loc["x"].iloc[0]) / 2
        assert half == pytest.approx(1.959963984540054 * ref["se"], rel=1e-10)


def test_small_false_rejects_what_ivregress_does_not_define(data):
    with pytest.raises(sp.MethodIncompatibility, match="HC3"):
        sp.iv("y ~ w + (x ~ z)", data=data, robust="hc3", small=False)
    with pytest.raises(sp.MethodIncompatibility, match="2SLS and LIML"):
        sp.iv("y ~ w + (x ~ z)", data=data, method="gmm", small=False)
    with pytest.raises(sp.MethodIncompatibility, match="absorb"):
        sp.iv("y ~ w + (x ~ z)", data=data, absorb="firm", small=False)
    with pytest.raises(sp.MethodIncompatibility, match="small"):
        sp.iv("y ~ w + (x ~ z)", data=data, small="no")
