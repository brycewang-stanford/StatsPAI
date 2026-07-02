"""Analytical parity: sp.dist_iv / sp.kan_dlate distributional-LATE recovery.

Distributional IV (Wald-style quantile LATE) with a binary instrument. Under a
constant treatment effect the LATE is flat across quantiles at beta, and the
point estimate must be finite for *every* ordinary data draw. This guards the
correctness fix for the degenerate median split that previously returned a
silent all-NaN ``late_q`` whenever the binary instrument had more 1s than 0s.
Analytical evidence tier (known-truth recovery on deterministic DGPs).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

BETA = 1.5
QUANTILES = np.array([0.25, 0.5, 0.75])


def _simulate(seed, n=1500):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, n)
    d = ((0.3 + 0.5 * z + rng.normal(0, 0.3, n)) > 0.5).astype(int)
    y = BETA * d + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y, "d": d, "z": z})


def test_never_silently_nan_across_seeds():
    # Includes seeds where the binary instrument has more 1s than 0s
    # (median == 1), the case that previously produced all-NaN.
    for seed in range(12):
        r = sp.dist_iv(
            _simulate(seed),
            y="y",
            treat="d",
            instrument="z",
            quantiles=QUANTILES,
            n_boot=10,
        )
        assert np.all(np.isfinite(np.asarray(r.late_q, dtype=float)))


def test_recovers_flat_late_at_constant_effect():
    # Average across seeds recovers the constant effect at every quantile.
    stacks = []
    for seed in range(8):
        r = sp.dist_iv(
            _simulate(seed),
            y="y",
            treat="d",
            instrument="z",
            quantiles=QUANTILES,
            n_boot=10,
        )
        stacks.append(np.asarray(r.late_q, dtype=float))
    mean_by_q = np.mean(stacks, axis=0)
    assert np.max(np.abs(mean_by_q - BETA)) < 0.2


def test_kan_dlate_matches_dist_iv():
    df = _simulate(0)
    a = sp.dist_iv(df, y="y", treat="d", instrument="z", quantiles=QUANTILES, n_boot=10)
    b = sp.kan_dlate(
        df, y="y", treat="d", instrument="z", quantiles=QUANTILES, n_boot=10
    )
    assert np.allclose(
        np.asarray(a.late_q, dtype=float),
        np.asarray(b.late_q, dtype=float),
        equal_nan=True,
    )
