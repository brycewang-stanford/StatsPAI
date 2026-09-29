"""``sp.ivreg(vce='wild')`` (WRE) against Stata ``boottest`` after ``ivregress``.

Reference: Stata 18, ``ivregress 2sls y w (x = z), vce(cluster cl)`` then
``boottest x, reps(2000) weight(rademacher)`` on ``_fixtures/iv_wild_data.csv``
(10 clusters, so all 2^10 sign vectors are enumerated and the p-value is
exact), from ``_fixtures/_generate_iv_wild_Stata.do``.

The confidence interval used to be the null-imposed bootstrap-t quantiles
placed around the estimate -- valid only at the null value itself (ADH,
AER 2013: [-0.496, -0.078] against ``boottest``'s [-0.580, -0.112]). It is now
the test-inversion confidence set ``boottest`` reports, over the same draws.
One endogenous regressor takes a vectorised path (Frisch-Waugh-Lovell per
draw); several fall back to the per-draw loop.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "iv_wild_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "iv_wild_data.csv")


def test_wre_matches_boottest(ref, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.ivreg(
            "y ~ (x ~ z) + w",
            data=data,
            cluster="cl",
            vce="wild",
            wild_reps=2000,
            wild_weight_type="rademacher",
        )
    assert float(r.pvalues["x"]) == pytest.approx(ref["p"], abs=1e-12)
    width = ref["hi"] - ref["lo"]
    # boottest locates the bounds to its ptol (1e-3)
    assert float(r.conf_int_lower["x"]) == pytest.approx(ref["lo"], abs=1e-3 * width)
    assert float(r.conf_int_upper["x"]) == pytest.approx(ref["hi"], abs=1e-3 * width)


def test_ci_is_test_inversion(data):
    """p at each bound is at the level: the set is {b0 : p(b0) > alpha}."""
    import importlib

    iw = importlib.import_module("statspai.inference.iv_wild")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.ivreg("y ~ (x ~ z) + w", data=data, cluster="cl")
        out = iw.iv_wild_bootstrap(fit, data, "cl", "x", n_boot=2000)
        lo, hi = out["ci_boot"]
        inside = iw.iv_wild_bootstrap(
            fit, data, "cl", "x", n_boot=2000, beta0=lo + 1e-3
        )
        outside = iw.iv_wild_bootstrap(
            fit, data, "cl", "x", n_boot=2000, beta0=lo - 1e-3
        )
    assert inside["p_boot"] > 0.05 >= outside["p_boot"]
    assert lo < float(fit.params["x"]) < hi


def test_two_endogenous_regressors_use_the_loop(data):
    rng = np.random.default_rng(1)
    d = data.assign(z2=rng.normal(size=len(data)))
    d["x2"] = 0.5 * d.z2 + rng.normal(size=len(d))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.ivreg(
            "y ~ (x + x2 ~ z + z2) + w",
            data=d,
            cluster="cl",
            vce="wild",
            wild_reps=199,
            seed=1,
        )
    lo, hi = float(r.conf_int_lower["x"]), float(r.conf_int_upper["x"])
    assert np.isfinite(lo) and np.isfinite(hi) and lo < float(r.params["x"]) < hi
