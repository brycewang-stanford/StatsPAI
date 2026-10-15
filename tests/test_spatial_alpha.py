"""``alpha=`` reaches the intervals of the spatial ML fits.

sp.sar / sem / sdm / slx / sac accepted alpha and never used it: the
interval was 95% whatever was asked (found 2026-10-16 while exposing
signature parameters in the agent schemas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def ring():
    rng = np.random.default_rng(0)
    n = 60
    W = np.zeros((n, n))
    for i in range(n):
        W[i, (i - 1) % n] = 1.0
        W[i, (i + 1) % n] = 1.0
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": 1.0 + 0.8 * x + rng.normal(scale=0.5, size=n), "x": x})
    return W, df


@pytest.mark.parametrize("fn", ["sar", "sem", "sdm", "slx", "sac"])
def test_alpha_sets_the_interval_level(fn, ring):
    W, df = ring
    fit = getattr(sp, fn)
    r95 = fit(W, data=df, formula="y ~ x")
    r90 = fit(W, data=df, formula="y ~ x", alpha=0.10)
    assert r90.alpha == 0.10 and r95.alpha == 0.05
    np.testing.assert_allclose(r90.params, r95.params, rtol=0, atol=0)
    w95 = (r95.conf_int_upper - r95.conf_int_lower).to_numpy()
    w90 = (r90.conf_int_upper - r90.conf_int_lower).to_numpy()
    # Same SEs, so the width ratio is the ratio of the critical values.
    ratio = w90 / w95
    assert np.all(ratio < 1)
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-12)
