"""Zero-inflated and hurdle fits under quasi-complete separation.

Found on the ``trips`` data of Croissant (2025), *Microeconometrics with
R*: one regressor predicts "not a structural zero" perfectly, its
coefficient in the binary part runs off, and ``1 / (1 + exp(-x))``
overflowed inside the complex-step Hessian. One NaN there made **every**
standard error NaN, including those of the count part, which R's
``pscl::zeroinfl`` reports without trouble on the same data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special

import statspai as sp
from statspai.exceptions import ConvergenceWarning
from statspai.regression.zeroinflated import _log_expit, _logaddexp


@pytest.fixture(scope="module")
def separated() -> pd.DataFrame:
    rng = np.random.default_rng(20261004)
    n = 1500
    x = rng.normal(size=n)
    s = (rng.uniform(size=n) < 0.15).astype(float)
    mu = np.exp(0.6 + 0.4 * x + 0.5 * s)
    structural_zero = rng.uniform(size=n) < 1 / (1 + np.exp(-(-0.5 + 0.7 * x)))
    y = np.where(structural_zero, 0, rng.poisson(mu))
    # s == 1 never has a zero of either kind: separation in the binary part.
    y = np.where((s == 1) & (y == 0), 1, y)
    return pd.DataFrame({"y": y, "x": x, "s": s})


@pytest.mark.parametrize("fn", ["zip_model", "zinb", "hurdle"])
def test_other_standard_errors_survive(fn, separated):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = getattr(sp, fn)(data=separated, y="y", x=["x", "s"])
    se = res.std_errors
    assert np.all(np.isfinite(se.values)), se
    flat = [name for name in se.index if se[name] > 1e4]
    assert len(flat) == 1 and flat[0].endswith("s"), flat
    ok = se.drop(flat)
    assert np.all((ok > 0) & (ok < 1.0)), ok
    messages = [
        str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
    ]
    assert any(flat[0] in m and "separation" in m for m in messages), messages


def test_no_warning_without_separation():
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.normal(size=n)
    zero = rng.uniform(size=n) < 0.3
    df = pd.DataFrame(
        {"y": np.where(zero, 0, rng.poisson(np.exp(0.5 + 0.3 * x))), "x": x}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        sp.zip_model(data=df, y="y", x=["x"])
        sp.hurdle(data=df, y="y", x=["x"])


def test_stable_logistic_helpers():
    z = np.array([-800.0, -30.0, -1.0, 0.0, 1.0, 30.0, 800.0])
    with np.errstate(over="raise", invalid="raise"):
        out = _log_expit(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, -np.logaddexp(0.0, -z), rtol=1e-13, atol=1e-320)
    assert out[0] == pytest.approx(-800.0) and out[-1] == pytest.approx(0.0, abs=1e-300)
    # Complex-step derivative of log expit is 1 - expit, also at the extremes.
    d = np.imag(_log_expit(z + 1e-30j)) / 1e-30
    np.testing.assert_allclose(d, special.expit(-z), rtol=1e-12, atol=1e-320)
    a, b = np.array([-900.0, 2.0, 5.0]), np.array([3.0, 2.0, -900.0])
    np.testing.assert_allclose(_logaddexp(a, b), np.logaddexp(a, b), rtol=1e-14)
