"""``sp.two_step_bootstrap``.

* With a step 1 that estimates nothing, it is the cluster bootstrap of
  step 2 (``sp.bootstrap`` with the same seed): identical replicates.
* Two-stage least squares by hand (step 1: first stage, step 2: OLS on the
  fitted value) has a known answer: the bootstrap SE of the second-stage
  slope estimates the 2SLS sampling SE, which ``sp.ivreg`` gives in closed
  form (cluster-robust); the second stage's own SE does not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ColumnNotFound, MethodIncompatibility


@pytest.fixture(scope="module")
def iv_data():
    rng = np.random.default_rng(0)
    G = 200
    g = np.repeat(np.arange(G), 5)
    n = len(g)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = z + rng.normal(size=n) + 0.5 * u
    return pd.DataFrame({"g": g, "z": z, "d": d, "y": 0.5 * d + u})


def _first_stage(frame):
    b = np.polyfit(frame.z, frame.d, 1)
    return frame.assign(dhat=np.polyval(b, frame.z))


def _second(frame):
    return sp.regress("y ~ dhat", data=frame, cluster="g")


def test_nonrandom_step1_is_the_plain_cluster_bootstrap(iv_data):
    two = sp.two_step_bootstrap(
        lambda f: f,
        lambda f: sp.regress("y ~ d", data=f),
        iv_data,
        cluster="g",
        n_boot=99,
        seed=3,
    )
    one = sp.bootstrap(
        iv_data,
        lambda f: sp.regress("y ~ d", data=f),
        cluster="g",
        n_boot=99,
        seed=3,
    )
    np.testing.assert_array_equal(two.std_errors.to_numpy(), one.std_errors.to_numpy())
    np.testing.assert_array_equal(two.params.to_numpy(), one.params.to_numpy())


def test_manual_2sls_bootstrap_matches_the_iv_standard_error(iv_data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.two_step_bootstrap(
            _first_stage, _second, iv_data, cluster="g", n_boot=999, seed=1
        )
        iv = sp.ivreg("y ~ (d ~ z)", data=iv_data, cluster="g")
    # same point estimate: 2SLS = OLS of y on the first-stage fit
    assert res.params["dhat"] == pytest.approx(iv.params["d"], abs=1e-10)
    # B = 999: the bootstrap SE has Monte Carlo sd ~ SE / sqrt(2B) ~ 2.2%;
    # with 200 clusters CR1 and the cluster bootstrap agree to O(1/G).
    # 10% is about 4.5 Monte Carlo sds.
    assert res.std_errors["dhat"] == pytest.approx(iv.std_errors["d"], rel=0.10)
    # the reported naive SE is step 2's own, on the original data
    own = _second(_first_stage(iv_data)).std_errors
    pd.testing.assert_series_equal(
        res.model_info["naive_se"], own.astype(float), check_names=False
    )
    np.testing.assert_allclose(
        res.model_info["se_ratio"], res.std_errors / own, rtol=1e-14
    )


def test_refusals(iv_data):
    with pytest.raises(MethodIncompatibility, match="callables"):
        sp.two_step_bootstrap(None, _second, iv_data)
    with pytest.raises(MethodIncompatibility, match="n_boot"):
        sp.two_step_bootstrap(_first_stage, _second, iv_data, n_boot=5)
    with pytest.raises(ColumnNotFound):
        sp.two_step_bootstrap(_first_stage, _second, iv_data, cluster="nope")
    with pytest.raises(MethodIncompatibility, match="DataFrame"):
        sp.two_step_bootstrap(lambda f: f.d, _second, iv_data, n_boot=20)
    with pytest.raises(MethodIncompatibility, match="idcluster"):
        sp.two_step_bootstrap(_first_stage, _second, iv_data, idcluster="cid")
