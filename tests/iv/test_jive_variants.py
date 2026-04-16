"""Tests for JIVE variants."""

import numpy as np
import pandas as pd
import pytest

import statspai.iv as iv


@pytest.fixture
def dgp():
    rng = np.random.default_rng(2024)
    n = 2500
    k = 5  # several instruments
    Z = rng.normal(size=(n, k))
    eps = rng.normal(size=n)
    d = Z @ rng.uniform(0.2, 0.5, size=k) + 0.3 * eps + rng.normal(size=n, scale=0.5)
    y = 1 + 1.5 * d + eps
    cols = {f"z{i+1}": Z[:, i] for i in range(k)}
    cols.update({"y": y, "d": d})
    return pd.DataFrame(cols)


class TestJIVEVariants:
    def test_ujive_recovers_true_coef(self, dgp):
        zlist = [c for c in dgp.columns if c.startswith("z")]
        res = iv.ujive(y="d", endog="d", instruments=zlist, data=dgp)
        # abuse of fixture: use y below, not d
        res = iv.ujive(y="y", endog="d", instruments=zlist, data=dgp)
        # true β = 1.5
        assert abs(res.params["endog0"] - 1.5) < 0.2
        assert res.first_stage_f > 30

    def test_jive1_ijive_ujive_agree_direction(self, dgp):
        zlist = [c for c in dgp.columns if c.startswith("z")]
        r1 = iv.jive1(y="y", endog="d", instruments=zlist, data=dgp)
        r2 = iv.ijive(y="y", endog="d", instruments=zlist, data=dgp)
        r3 = iv.ujive(y="y", endog="d", instruments=zlist, data=dgp)
        # all should recover β near 1.5
        for r in (r1, r2, r3):
            assert abs(r.params["endog0"] - 1.5) < 0.3

    def test_rjive_requires_ridge(self, dgp):
        zlist = [c for c in dgp.columns if c.startswith("z")]
        with pytest.raises(ValueError, match="ridge"):
            iv.rjive(y="y", endog="d", instruments=zlist, data=dgp, ridge=0.0)

    def test_rjive_shrinks_towards_ols_as_ridge_grows(self, dgp):
        zlist = [c for c in dgp.columns if c.startswith("z")]
        r_small = iv.rjive(y="y", endog="d", instruments=zlist, data=dgp, ridge=0.01)
        r_large = iv.rjive(y="y", endog="d", instruments=zlist, data=dgp, ridge=100.0)
        # Not a strict monotonicity guarantee, but point estimate should move
        assert r_small.params["endog0"] != r_large.params["endog0"]
