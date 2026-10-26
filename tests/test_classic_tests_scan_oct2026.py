"""Option scan of the classic tests (2026-10-27): suspects drafted in phase 1.

The round-3 scan (tests/test_inference_scan3_classic_tests.py) already
covers these functions; this file adds the precision of the Sidak
adjustment in ``sp.oneway`` for very small p-values.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp


def test_oneway_sidak_keeps_precision_for_tiny_p():
    rng = np.random.default_rng(0)
    g = np.repeat([1, 2, 3], 30)
    y = np.array([0.0, 5.0, 10.0])[g - 1] + 0.5 * rng.normal(size=g.size)
    df = pd.DataFrame({"y": y, "g": g})
    res = sp.oneway(df, "y", by="g", compare="sidak")
    est = res.estimates
    n, k = g.size, 3
    pairs = k * (k - 1) / 2
    d = est["differences"].iloc[1, 0]  # mean(2) - mean(1)
    t = d / np.sqrt(est["ms_within"] * (1 / 30 + 1 / 30))
    p_raw = 2 * stats.t.sf(abs(t), n - k)
    assert 0 < p_raw < 1e-20  # the regime where the naive form cancels
    ref = -np.expm1(pairs * np.log1p(-p_raw))
    assert est["pvalues"].iloc[1, 0] == pytest.approx(ref, rel=1e-10, abs=0)
