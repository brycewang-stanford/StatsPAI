"""End-to-end numerical-coherence test for the IV / 2SLS workflow.

On a DGP with a strong instrument and an endogenous regressor (true
coefficient 1.0, OLS biased upward by an omitted confounder), the whole IV
story must hold together: OLS is biased, 2SLS recovers the truth and is less
biased than OLS, the first-stage F flags a strong instrument, and the Hausman
test rejects exogeneity (confirming IV was needed).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_BETA = 1.0


@pytest.fixture
def iv_data():
    rng = np.random.RandomState(7)
    n = 2000
    z = rng.randn(n)  # instrument
    u = rng.randn(n)  # omitted confounder
    x = 0.8 * z + 1.0 * u + rng.randn(n) * 0.5  # strong first stage
    y = TRUE_BETA * x + 2.0 * u + rng.randn(n)  # OLS biased by u
    return pd.DataFrame({"y": y, "x": x, "z": z})


def test_ols_is_biased_iv_recovers_truth(iv_data):
    ols = sp.regress("y ~ x", iv_data)
    iv = sp.ivreg("y ~ (x ~ z)", iv_data)

    ols_b = float(ols.params["x"])
    iv_b = float(iv.params["x"])
    # OLS is materially biased away from the truth ...
    assert abs(ols_b - TRUE_BETA) > 0.5
    # ... while 2SLS recovers it ...
    assert iv_b == pytest.approx(TRUE_BETA, abs=0.25)
    # ... and is strictly less biased than OLS.
    assert abs(iv_b - TRUE_BETA) < abs(ols_b - TRUE_BETA)


def test_first_stage_is_strong(iv_data):
    iv = sp.ivreg("y ~ (x ~ z)", iv_data)
    f = iv.diagnostics["First-stage F (x)"]
    # Stock-Yogo rule of thumb: F > 10 => not weak.
    assert f > 10


def test_hausman_rejects_exogeneity(iv_data):
    iv = sp.ivreg("y ~ (x ~ z)", iv_data)
    # The regressor is endogenous by construction, so the Hausman test should
    # reject the null of exogeneity.
    assert iv.diagnostics["Hausman p-value"] < 0.05


def test_iv_ci_brackets_truth(iv_data):
    iv = sp.ivreg("y ~ (x ~ z)", iv_data)
    lo = float(iv.conf_int_lower["x"])
    hi = float(iv.conf_int_upper["x"])
    assert lo < hi
    assert lo <= TRUE_BETA <= hi
