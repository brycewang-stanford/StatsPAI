"""Tests for RD aliases: multi_cutoff_rd / geographic_rd / boundary_rd."""

import warnings
import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import statspai as sp


def test_multi_cutoff_rd_recovers_jumps():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.uniform(-2, 2, n)
    # Different jumps at each cutoff
    y = x + 0.5 * (x > -0.5) + 1.0 * (x > 0.5) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x": x})
    r = sp.multi_cutoff_rd(df, y="y", x="x", cutoffs=[-0.5, 0.5])
    assert r.pooled_estimate > 0
    assert r.n_cutoffs == 2
    assert len(r.cutoff_results) == 2


def test_multi_cutoff_rd_same_as_rdmc():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.uniform(-2, 2, n)
    y = x + 0.5 * (x > 0) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x": x})
    r1 = sp.multi_cutoff_rd(df, y="y", x="x", cutoffs=[0.0])
    r2 = sp.rdmc(df, y="y", x="x", cutoffs=[0.0])
    assert r1.pooled_estimate == pytest.approx(r2.pooled_estimate)


def test_aliases_registered():
    fns = sp.list_functions()
    for name in ["multi_cutoff_rd", "geographic_rd", "boundary_rd", "multi_score_rd"]:
        assert name in fns
