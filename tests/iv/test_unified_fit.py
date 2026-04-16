"""Tests for the unified sp.iv.fit dispatcher."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
import statspai.iv as iv


@pytest.fixture
def dgp():
    rng = np.random.default_rng(99)
    n = 2000
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    x = rng.normal(size=n)
    eps = rng.normal(size=n)
    d = 0.5 * z1 + 0.4 * z2 + 0.4 * eps + rng.normal(size=n, scale=0.5)
    y = 1 + 2.0 * d + 0.5 * x + eps
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "x": x})


class TestUnifiedFit:
    def test_2sls_default(self, dgp):
        res = iv.fit("y ~ (d ~ z1 + z2) + x", data=dgp)
        assert abs(res.params["d"] - 2.0) < 0.15
        # Augmented diagnostics attached by default
        assert "KP rk Wald F" in res.diagnostics
        assert "SW conditional F (d)" not in res.diagnostics  # single endog → no SW line
        assert "Olea-Pflueger effective F" in res.diagnostics

    def test_alias_tsls(self, dgp):
        a = iv.fit("y ~ (d ~ z1 + z2) + x", data=dgp, method="tsls")
        b = iv.fit("y ~ (d ~ z1 + z2) + x", data=dgp, method="2sls")
        assert abs(a.params["d"] - b.params["d"]) < 1e-10

    def test_liml(self, dgp):
        res = iv.fit("y ~ (d ~ z1 + z2) + x", data=dgp, method="liml")
        assert abs(res.params["d"] - 2.0) < 0.1

    def test_liml_reduces_many_weak_iv_bias(self):
        """With many weak instruments, LIML should be markedly less biased than 2SLS."""
        rng = np.random.default_rng(1)
        n = 500
        zs = [rng.normal(size=n) for _ in range(10)]
        eps = rng.normal(size=n)
        d = 0.04 * sum(zs) + 0.7 * eps + rng.normal(size=n, scale=0.3)
        y = 1 + 2.0 * d + eps
        df = pd.DataFrame({"y": y, "d": d, **{f"z{i+1}": z for i, z in enumerate(zs)}})
        formula = "y ~ (d ~ " + "+".join(f"z{i+1}" for i in range(10)) + ")"
        b2sls = iv.fit(formula, data=df, method="2sls").params["d"]
        bliml = iv.fit(formula, data=df, method="liml").params["d"]
        # True β = 2.0 ; 2SLS biased toward OLS (≈ 2.5+), LIML closer to 2.0
        assert abs(bliml - 2.0) < abs(b2sls - 2.0)

    def test_ujive_method(self, dgp):
        res = iv.fit("y ~ (d ~ z1 + z2) + x", data=dgp, method="ujive")
        assert abs(res.params["d"] - 2.0) < 0.25
        assert res.first_stage_f > 10

    def test_two_endog_adds_sw(self, dgp):
        # Build a 2-endog variant
        rng = np.random.default_rng(3)
        n = len(dgp)
        z3 = rng.normal(size=n)
        eps = rng.normal(size=n)
        d2 = 0.6 * dgp["z2"].values + 0.7 * z3 + 0.3 * eps + rng.normal(size=n, scale=0.5)
        df = dgp.assign(z3=z3, d2=d2, y=dgp["y"] - 1.5 * d2)
        res = iv.fit("y ~ (d + d2 ~ z1 + z2 + z3) + x", data=df)
        # SW per-endog should be attached
        assert any("SW conditional F" in key for key in res.diagnostics)

    def test_unknown_method_raises(self, dgp):
        with pytest.raises(ValueError, match="Unknown method"):
            iv.fit("y ~ (d ~ z1 + z2) + x", data=dgp, method="quantum_iv")

    def test_re_export_2sls_matches_ivreg(self, dgp):
        res_fit = iv.fit("y ~ (d ~ z1 + z2) + x", data=dgp)
        res_old = sp.ivreg("y ~ (d ~ z1 + z2) + x", data=dgp)
        assert abs(res_fit.params["d"] - res_old.params["d"]) < 1e-10
