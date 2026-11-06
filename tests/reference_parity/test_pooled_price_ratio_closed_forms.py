"""``sp.pooled_price_ratio``: closed forms of the pooled price ratio.

There is no Stata or R command for the pooled price ratio, so the evidence
is known truth (T1): a discrete distribution worked by hand, the conditional
means of the uniform and beta distributions (midpoint-quantile
samples, so the only error is the discretisation), and the equivalence of
frequency weights with duplicated rows. The end-to-end use inside
``sp.risk_class_selection`` is checked in
``test_risk_class_selection_known_truth.py``; input errors, the markup
no-trade flag and the bootstrap contract stay in
``tests/test_adverse_selection.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import statspai as sp


def test_ppr_discrete_exact():
    # p = 0.1 (weight 2), 0.2, 0.4. At r = 0.1: E[P|P>=r] = 0.8 / 4 = 0.2 and
    # T = (0.2/0.8) / (0.1/0.9) = 2.25. At 0.2: 0.3, T = (0.3/0.7)/(0.2/0.8).
    res = sp.pooled_price_ratio([0.1, 0.2, 0.4], weights=[2, 1, 1])
    c = res.curve
    assert c["quantile"].tolist() == pytest.approx([0.0, 0.5, 0.75])
    assert c["mean_above"].tolist() == pytest.approx([0.2, 0.3, 0.4])
    t_02 = (0.3 / 0.7) / 0.25
    assert c["T"].tolist() == pytest.approx([2.25, t_02, 1.0], rel=1e-12)
    assert res.estimate == pytest.approx(0.0, abs=1e-12)  # top point included
    r6 = sp.pooled_price_ratio([0.1, 0.2, 0.4], weights=[2, 1, 1], upto_quantile=0.6)
    assert r6.estimate == pytest.approx(t_02 - 1, rel=1e-12)
    assert r6.argmin_r == 0.2


def test_ppr_uniform_closed_form():
    # P ~ U(a, b): E[P | P >= r] = (r + b) / 2. Midpoint quantile sample of
    # size n reproduces it to O((b - a) / n).
    a, b, n = 0.02, 0.3, 20000
    pv = a + (b - a) * (np.arange(n) + 0.5) / n
    res = sp.pooled_price_ratio(pv)
    c = res.curve.iloc[::1000]
    m = (c["r"] + b) / 2
    expected = (m / (1 - m)) / (c["r"] / (1 - c["r"]))
    np.testing.assert_allclose(c["T"], expected, rtol=1e-4)


def test_ppr_beta_t80_closed_form():
    # P ~ Beta(2, 30): E[P 1{P >= r}] = mean * (1 - I_r(3, 30)).
    a_, b_, n = 2.0, 30.0, 40000
    pv = stats.beta.ppf((np.arange(n) + 0.5) / n, a_, b_)
    res = sp.pooled_price_ratio(pv)
    r = np.linspace(stats.beta.ppf(0.001, a_, b_), stats.beta.ppf(0.8, a_, b_), 4000)
    mean = a_ / (a_ + b_)
    m = mean * (1 - special.betainc(a_ + 1, b_, r)) / (1 - special.betainc(a_, b_, r))
    t = (m / (1 - m)) / (r / (1 - r)) - 1
    # Grid and sample discretisation: agreement to 2e-3 relative.
    assert res.estimate == pytest.approx(t.min(), rel=2e-3)


def test_ppr_weights_equal_duplicated_rows():
    rng = np.random.default_rng(2)
    pv = np.round(rng.beta(2, 20, size=300), 3)
    w = rng.integers(1, 4, size=300)
    a = sp.pooled_price_ratio(pv, weights=w)
    b = sp.pooled_price_ratio(np.repeat(pv, w))
    pd.testing.assert_frame_equal(a.curve, b.curve, rtol=1e-12)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12)
