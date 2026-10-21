"""Cluster-robust ``vcov`` spellings on sp.feols / fepois / feglm / hdfe_ols.

``vcov={"CL1": "id"}`` used to die inside pyfixest with a bare
``AssertionError``. Every spelling that clearly means one-way (or
multiway) CRV1 / CRV3 clustering now maps to the same internal request,
so the result must be *identical* (``==``, not approximately equal) to
``cluster="id"``; unknown spellings raise ``MethodIncompatibility`` with a
recovery hint.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core._vcov_spellings import fixest_vcov, house_style_vce
from statspai.exceptions import MethodIncompatibility

CRV1_SPELLINGS = [
    {"CL1": "unit"},
    {"cl1": "unit"},
    {"CRV1": "unit"},
    {"crv1": "unit"},
    {"CR1": "unit"},
    {"cluster": "unit"},
    {"Cluster": "unit"},
    {"CRV1": ["unit"]},
    "cluster:unit",
    "cluster: unit",
    "cluster(unit)",
    "cluster = unit",
    "CL1:unit",
    "CRV1(unit)",
    "~unit",
    "~ unit",
    "cluster unit",
    "vce(cluster unit)",
    "cl unit",
]


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    df = sp.dgp_did(n_units=60, n_periods=6, staggered=False, seed=0)
    rng = np.random.default_rng(0)
    df["x"] = rng.normal(size=len(df))
    df["cnt"] = rng.poisson(2.0, size=len(df))
    df["b"] = (rng.random(len(df)) < 0.4).astype(int)
    df["region"] = df["unit"] % 7
    return df


def _fit(fn, data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fn == "feols":
            return sp.feols("y ~ x | unit + time", data, **kw)
        if fn == "fepois":
            return sp.fepois("cnt ~ x | unit + time", data, **kw)
        return sp.feglm("b ~ x | unit + time", data, family="logit", **kw)


@pytest.mark.parametrize("fn", ["feols", "fepois", "feglm"])
@pytest.mark.parametrize("spec", CRV1_SPELLINGS, ids=repr)
def test_crv1_spellings_equal_cluster_argument(panel, fn, spec):
    base = _fit(fn, panel, cluster="unit")
    res = _fit(fn, panel, vcov=spec)
    pd.testing.assert_series_equal(res.params, base.params)
    pd.testing.assert_series_equal(res.std_errors, base.std_errors)


@pytest.mark.parametrize("spec", [{"CL1": "unit"}, "cluster:unit"], ids=repr)
def test_crv1_spellings_on_the_pyfixest_backend(panel, spec):
    base = _fit("feols", panel, vcov={"CRV1": "unit"}, backend="pyfixest")
    res = _fit("feols", panel, vcov=spec, backend="pyfixest")
    pd.testing.assert_series_equal(res.std_errors, base.std_errors)


@pytest.mark.parametrize(
    "spec",
    [{"CL3": "unit"}, {"cr3": "unit"}, "CRV3:unit", "cr3(unit)"],
    ids=repr,
)
def test_crv3_spellings(panel, spec):
    base = _fit("feols", panel, vcov={"CRV3": "unit"})
    res = _fit("feols", panel, vcov=spec)
    pd.testing.assert_series_equal(res.std_errors, base.std_errors)


@pytest.mark.parametrize(
    "spec",
    [
        {"CL1": "unit+region"},
        {"cluster": ["unit", "region"]},
        "cluster:unit+region",
        "~unit + region",
        "cluster(unit region)",
    ],
    ids=repr,
)
def test_multiway_spellings(panel, spec):
    base = _fit("feols", panel, vcov={"CRV1": "unit+region"})
    res = _fit("feols", panel, vcov=spec)
    pd.testing.assert_series_equal(res.std_errors, base.std_errors)


def test_vce_alias_and_bare_cluster(panel):
    base = _fit("feols", panel, cluster="unit")
    for kw in (
        {"vce": {"CL1": "unit"}},
        {"vce": "cluster:unit"},
        {"vcov": "cluster", "cluster": "unit"},
    ):
        pd.testing.assert_series_equal(
            _fit("feols", panel, **kw).std_errors, base.std_errors
        )


def test_missing_cluster_rows_are_marked_out_the_same_way(panel):
    data = panel.copy()
    data["cl"] = data["unit"].astype(float)
    data.loc[data.index[:6], "cl"] = np.nan
    base = _fit("feols", data, cluster="cl")
    res = _fit("feols", data, vcov="cluster:cl")
    assert res.model_info["n_missing_cluster_dropped"] == 6
    pd.testing.assert_series_equal(res.std_errors, base.std_errors)


@pytest.mark.parametrize("fn", ["feols", "fepois", "feglm"])
@pytest.mark.parametrize(
    "spec",
    [
        {"foo": "unit"},
        {"CR2": "unit"},
        {"CRV1": "unit", "CRV3": "unit"},
        {"CRV1": 3},
        "bogus",
        "cluster",
    ],
    ids=repr,
)
def test_unrecognised_spellings_raise_a_statspai_error(panel, fn, spec):
    with pytest.raises(MethodIncompatibility) as info:
        _fit(fn, panel, vcov=spec)
    assert "cluster=" in info.value.recovery_hint
    assert not isinstance(info.value.__cause__, AssertionError) or "backend" in str(
        info.value
    )


def test_backend_assertion_is_reraised(panel):
    """A string pyfixest rejects with an AssertionError is re-raised here."""
    with pytest.raises(MethodIncompatibility, match="rejected") as info:
        _fit("feols", panel, vcov="not-a-vcov", backend="pyfixest")
    assert isinstance(info.value.__cause__, AssertionError)


@pytest.mark.parametrize(
    "vce",
    ["cluster unit", "vce(cluster unit)", "cluster:unit", "cluster(unit)", "~unit"],
)
def test_hdfe_ols_cluster_spellings(panel, vce):
    base = sp.hdfe_ols("y ~ x | unit + time", panel, cluster="unit")
    res = sp.hdfe_ols("y ~ x | unit + time", panel, vce=vce)
    pd.testing.assert_series_equal(res.std_errors, base.std_errors)


def test_hdfe_ols_unknown_vce_is_a_statspai_error(panel):
    with pytest.raises(MethodIncompatibility) as info:
        sp.hdfe_ols("y ~ x | unit + time", panel, vce="bogus")
    assert "cluster=" in info.value.recovery_hint


def test_normaliser_leaves_other_requests_alone():
    for value in (
        "hetero",
        "HC1",
        "iid",
        "NW",
        "DK",
        "wild",
        "cr2",
        "vce(robust)",
        None,
    ):
        assert house_style_vce(value) == value
    assert fixest_vcov("hetero", None, function="f") == ("hetero", None)
    assert fixest_vcov({"cl": " a + b "}, None, function="f") == ({"CRV1": "a+b"}, None)
