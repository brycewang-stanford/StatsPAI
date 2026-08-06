"""Tests for the canonical vcov vocabulary and unknown-kwarg rejection.

Regression cover for the two silent-correctness traps found by the
RCT/quasi-experiment teaching notebook:

1. ``sp.regress(..., vcov={"CRV1": "firm"})`` — the pyfixest spelling a
   user carries over from ``sp.feols`` — was accepted and *silently
   dropped*, returning unclustered standard errors.
2. Any misspelled keyword (``robsut="hc1"``) fell through ``**kwargs``
   into ``fit()`` and vanished, silently yielding default SEs.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def clustered():
    """Panel with a strong cluster effect: clustered SEs must differ."""
    rng = np.random.default_rng(0)
    n = 600
    g = rng.integers(0, 15, n)
    g_effect = rng.normal(0, 2, size=15)[g]
    df = pd.DataFrame({"x": rng.normal(size=n), "g": g})
    df["y"] = df["x"] + g_effect + rng.normal(size=n)
    return df


@pytest.fixture(scope="module")
def iv_clustered():
    rng = np.random.default_rng(0)
    n = 900
    g = rng.integers(0, 15, n)
    g_effect = rng.normal(0, 2, size=15)[g]
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    x = z + u + rng.normal(size=n)
    y = 2 * x + u + g_effect + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "g": g})


class TestRegressVcovAlias:
    def test_crv1_equals_cluster(self, clustered):
        native = sp.regress("y ~ x", clustered, cluster="g").std_errors["x"]
        alias = sp.regress("y ~ x", clustered, vcov={"CRV1": "g"}).std_errors["x"]
        assert alias == pytest.approx(native, rel=1e-12)

    def test_crv1_is_not_silently_dropped(self, clustered):
        """The original bug: vcov= returned plain (unclustered) SEs."""
        plain = sp.regress("y ~ x", clustered).std_errors["x"]
        alias = sp.regress("y ~ x", clustered, vcov={"CRV1": "g"}).std_errors["x"]
        assert alias != pytest.approx(plain, rel=1e-6)

    def test_crv2_and_crv3_map_to_small_sample_kinds(self, clustered):
        cr2_native = sp.regress("y ~ x", clustered, cluster="g", vce="CR2").std_errors[
            "x"
        ]
        cr2_alias = sp.regress("y ~ x", clustered, vcov={"CRV2": "g"}).std_errors["x"]
        assert cr2_alias == pytest.approx(cr2_native, rel=1e-12)

        cr3_native = sp.regress("y ~ x", clustered, cluster="g", vce="CR3").std_errors[
            "x"
        ]
        cr3_alias = sp.regress("y ~ x", clustered, vcov={"CRV3": "g"}).std_errors["x"]
        assert cr3_alias == pytest.approx(cr3_native, rel=1e-12)
        # The three corrections must actually differ from one another.
        assert len({round(cr2_alias, 9), round(cr3_alias, 9)}) == 2

    def test_scalar_spellings(self, clustered):
        hc1 = sp.regress("y ~ x", clustered, robust="hc1").std_errors["x"]
        for spelling in ("hetero", "HC1", "robust"):
            got = sp.regress("y ~ x", clustered, vcov=spelling).std_errors["x"]
            assert got == pytest.approx(hc1, rel=1e-12), spelling
        iid = sp.regress("y ~ x", clustered).std_errors["x"]
        assert sp.regress("y ~ x", clustered, vcov="iid").std_errors["x"] == (
            pytest.approx(iid, rel=1e-12)
        )

    def test_conflicting_spellings_raise(self, clustered):
        with pytest.raises(MethodIncompatibility, match="two vocabularies"):
            sp.regress("y ~ x", clustered, vcov={"CRV1": "g"}, cluster="g")
        with pytest.raises(MethodIncompatibility, match="two vocabularies"):
            sp.regress("y ~ x", clustered, vcov="hetero", robust="hc1")

    def test_unknown_vcov_raises(self, clustered):
        with pytest.raises(MethodIncompatibility, match="unknown vcov"):
            sp.regress("y ~ x", clustered, vcov="nonsense")
        with pytest.raises(MethodIncompatibility, match="unknown cluster-robust"):
            sp.regress("y ~ x", clustered, vcov={"CRVX": "g"})
        with pytest.raises(MethodIncompatibility, match="exactly one entry"):
            sp.regress("y ~ x", clustered, vcov={"CRV1": "g", "CRV2": "g"})
        with pytest.raises(MethodIncompatibility, match="column name"):
            sp.regress("y ~ x", clustered, vcov={"CRV1": 3})
        with pytest.raises(MethodIncompatibility, match="string or a one-entry"):
            sp.regress("y ~ x", clustered, vcov=["CRV1", "g"])


class TestIvregVcovAlias:
    def test_crv1_equals_cluster(self, iv_clustered):
        native = sp.ivreg("y ~ (x ~ z)", iv_clustered, cluster="g").std_errors["x"]
        alias = sp.ivreg("y ~ (x ~ z)", iv_clustered, vcov={"CRV1": "g"}).std_errors[
            "x"
        ]
        assert alias == pytest.approx(native, rel=1e-12)

    def test_hetero_equals_hc1(self, iv_clustered):
        native = sp.ivreg("y ~ (x ~ z)", iv_clustered, robust="hc1").std_errors["x"]
        alias = sp.ivreg("y ~ (x ~ z)", iv_clustered, vcov="hetero").std_errors["x"]
        assert alias == pytest.approx(native, rel=1e-12)

    def test_conflict_raises(self, iv_clustered):
        with pytest.raises(MethodIncompatibility, match="two vocabularies"):
            sp.ivreg("y ~ (x ~ z)", iv_clustered, vcov={"CRV1": "g"}, cluster="g")


class TestUnknownKwargsRejected:
    def test_regress_typo_raises(self, clustered):
        with pytest.raises(TypeError, match="robsut"):
            sp.regress("y ~ x", clustered, robsut="hc1")

    def test_ivreg_typo_raises(self, iv_clustered):
        with pytest.raises(TypeError, match="robsut"):
            sp.ivreg("y ~ (x ~ z)", iv_clustered, robsut="hc1")

    def test_error_message_explains_why(self, clustered):
        with pytest.raises(TypeError, match="rejected rather than ignored"):
            sp.regress("y ~ x", clustered, clsuter="g")

    def test_supported_kwargs_still_pass_through(self, clustered):
        # weights= is a real option and must survive the new gate.
        w = np.full(len(clustered), 2.0)
        res = sp.regress("y ~ x", clustered, weights=w)
        assert np.isfinite(res.params["x"])
