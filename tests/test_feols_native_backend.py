"""``sp.feols(backend='native')``: pyfixest's answer from the HDFE kernel.

Busting the Princelings (QJE 2019) ran ``feols`` on 5.7 million rows: 12 GB
and minutes, where ``sp.hdfe_ols`` took seconds. ``backend='native'`` fits
the same regression on the native kernel and reports it as ``feols``; on
2M rows with two effects it took 1.9 s against 110 s.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.fixest.wrapper import _native_eligible


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n = 20000
    df = pd.DataFrame(
        dict(
            f1=rng.integers(0, 1500, n),
            f2=rng.integers(0, 100, n),
            x1=rng.normal(size=n),
            x2=rng.normal(size=n),
            w=rng.uniform(0.5, 2, n),
        )
    )
    df["y"] = df.x1 - 0.5 * df.x2 + rng.normal(size=n)
    df["c"] = df.f1 // 10
    return df


@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "c"}])
@pytest.mark.parametrize("weights", [None, "w"])
def test_native_equals_pyfixest(data, vcov, weights):
    if weights and vcov == "hetero":
        # the native robust menu is unweighted: refuse, never ignore
        with pytest.raises(ValueError, match="weights"):
            sp.feols(
                "y ~ x1 + x2 | f1 + f2",
                data=data,
                vcov=vcov,
                weights=weights,
                backend="native",
            )
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.feols("y ~ x1 + x2 | f1 + f2", data=data, vcov=vcov, weights=weights)
        b = sp.feols(
            "y ~ x1 + x2 | f1 + f2",
            data=data,
            vcov=vcov,
            weights=weights,
            backend="native",
        )
    np.testing.assert_allclose(b.params[a.params.index], a.params, rtol=1e-10)
    # SEs agree to the demeaning tolerance.
    np.testing.assert_allclose(b.std_errors[a.params.index], a.std_errors, rtol=1e-8)
    assert b.model_info["backend"] == "statspai-native"


def test_eligibility_and_validation(data):
    assert _native_eligible("y ~ x1 | f1", "iid", None, {})
    assert _native_eligible("y ~ x1 | f1", {"CRV1": "c"}, None, {})
    assert not _native_eligible("y ~ x1 | f1 | d ~ z", "iid", None, {})
    assert not _native_eligible("y ~ csw(x1, x2) | f1", "iid", None, {})
    assert not _native_eligible("y ~ i(x1) | f1", "iid", None, {})
    assert not _native_eligible("y ~ x1 | f1", "hetero", None, {}, "w")
    with pytest.raises(sp.MethodIncompatibility, match="backend"):
        sp.feols("y ~ x1 | f1", data=data, backend="fast")
