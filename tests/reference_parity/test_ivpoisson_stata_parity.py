"""Cross-language parity: ``sp.ivpoisson`` against Stata 18 ``ivpoisson gmm``.

Fixture: ``_fixtures/_generate_ivpoisson_stata.do``. The data are generated
in Stata and exported at ``%21.16e``, so both sides read the same bytes.

Every block is held to 1e-6 on coefficients, standard errors and Hansen's
J. The observed gaps are between 1e-15 and 2e-7; what is left is Stata's
stopping rule for the GMM criterion (ours iterates Gauss-Newton to 1e-10).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
BLOCKS = {
    "additive_twostep": {},
    "additive_onestep": dict(method="onestep"),
    "additive_igmm": dict(method="igmm"),
    "multiplicative_twostep": dict(errors="multiplicative"),
    "multiplicative_onestep": dict(errors="multiplicative", method="onestep"),
    "multiplicative_igmm": dict(errors="multiplicative", method="igmm"),
    "multiplicative_justid": dict(errors="multiplicative", instruments=["z1"]),
    "multiplicative_cluster": dict(errors="multiplicative", cluster="clust"),
    "additive_cluster": dict(cluster="clust"),
    "additive_vcecluster_only": dict(cluster="clust"),
    "additive_wrobust_vcecluster": dict(cluster="clust", wmatrix="robust"),
    "additive_unadjusted": dict(vce="unadjusted"),
}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "ivpoisson_data.csv")


@pytest.fixture(scope="module")
def stata() -> dict:
    return json.loads((_FIX / "ivpoisson_stata.json").read_text(encoding="utf-8"))


def _fit(block: str, data: pd.DataFrame):
    kw = {"instruments": ["z1", "z2"], **BLOCKS[block]}
    return sp.ivpoisson(data, y="y", x=["x1", "x2"], endog="w", **kw)


@pytest.mark.parametrize("block", sorted(BLOCKS))
def test_matches_stata(block, data, stata):
    res, ref = _fit(block, data), stata[block]
    assert [f"y:{n}" for n in res.params.index] == ref["names"]
    np.testing.assert_allclose(res.params.values, ref["b"], rtol=1e-6)
    np.testing.assert_allclose(res.std_errors.values, ref["se"], rtol=1e-6)
    info = res.model_info
    assert info["j_df"] == int(ref["J_df"])
    if info["j_df"] > 0:
        assert info["j_stat"] == pytest.approx(ref["J"], rel=1e-6)
    else:
        assert info["j_stat"] is None and info["j_pvalue"] is None


def test_the_weight_matrix_follows_vce_unless_given(data, stata):
    """Stata's wmatrix() defaults to the vce() type; the fixture shows it."""
    assert stata["additive_vcecluster_only"]["b"] == stata["additive_cluster"]["b"]
    assert stata["additive_wrobust_vcecluster"]["b"] != stata["additive_cluster"]["b"]
    a = _fit("additive_wrobust_vcecluster", data)
    b = _fit("additive_twostep", data)
    np.testing.assert_allclose(a.params.values, b.params.values, rtol=1e-12)
    assert not np.allclose(a.std_errors.values, b.std_errors.values)


def test_multiplicative_recovers_the_truth_where_poisson_does_not(data):
    """The do-file's DGP has a multiplicative error correlated with ``w``.

    Slopes 0.4 (w), 0.3 (x1), -0.5 (x2).
    """
    truth = np.array([0.4, 0.3, -0.5])
    mult = _fit("multiplicative_twostep", data)
    z = (mult.params.values[:3] - truth) / mult.std_errors.values[:3]
    assert np.all(np.abs(z) < 2.5)
    # Poisson with no instruments is far off on the endogenous slope.
    naive = sp.poisson(data=data, y="y", x=["w", "x1", "x2"], robust="robust")
    assert (naive.params["w"] - 0.4) / naive.std_errors["w"] > 5


def test_onestep_j_carries_its_caveat(data):
    info = _fit("additive_onestep", data).model_info
    assert "homoskedastic" in info["j_note"]
    assert "j_note" not in _fit("additive_twostep", data).model_info


def test_refusals(data):
    bad = sp.exceptions.MethodIncompatibility
    with pytest.raises(bad, match="endog"):
        sp.ivpoisson(data, y="y", x=["x1"], instruments=["z1"])
    with pytest.raises(bad, match="excluded instrument"):
        sp.ivpoisson(data, y="y", endog=["w", "x1"], instruments=["z1"])
    with pytest.raises(bad, match="errors="):
        sp.ivpoisson(data, y="y", endog="w", instruments=["z1"], errors="x")
    with pytest.raises(bad, match="wmatrix='cluster'"):
        sp.ivpoisson(data, y="y", endog="w", instruments=["z1"], wmatrix="cluster")
    neg = data.assign(y=data["y"] - 1.0)
    with pytest.raises(bad, match="negative"):
        sp.ivpoisson(neg, y="y", endog="w", instruments=["z1"])


def test_from_stata_round_trip(data, stata):
    cases = {
        "additive_twostep": "ivpoisson gmm y x1 x2 (w = z1 z2)",
        "multiplicative_cluster": (
            "ivpoisson gmm y x1 x2 (w = z1 z2), multiplicative "
            "vce(cluster clust) wmatrix(cluster clust)"
        ),
        "additive_wrobust_vcecluster": (
            "ivpoisson gmm y x1 x2 (w = z1 z2), vce(cluster clust) wmatrix(robust)"
        ),
        "additive_igmm": "ivpoisson gmm y x1 x2 (w = z1 z2), igmm",
    }
    for block, command in cases.items():
        out = sp.from_stata(command)
        assert out["ok"] and out["untranslated_options"] == [], out
        res = sp.ivpoisson(data, **out["arguments"])
        np.testing.assert_allclose(res.params.values, stata[block]["b"], rtol=1e-6)
        np.testing.assert_allclose(res.std_errors.values, stata[block]["se"], rtol=1e-6)
    refused = sp.from_stata("ivpoisson cfunction y x1 (w = z1)")
    assert not refused["ok"] and "control function" in refused["error"]


def test_cite_returns_the_registered_reference(data):
    assert "mullahy1997instrumental" in _fit("multiplicative_twostep", data).cite()
