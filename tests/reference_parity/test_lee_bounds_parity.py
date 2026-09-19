"""Analytical parity: sp.lee_bounds exact trimming-bound identity (Lee 2009).

Lee bounds under monotone sample selection (paper.bib lee2009training) are an
exact closed form: with trimming fraction p = (s1 - s0) / s1, the bounds are
the top/bottom-trimmed treated means minus the observed control mean:

    lower == mean(y1[y1 <= Q_{1-p}(y1)]) - mean(y0)
    upper == mean(y1[y1 >= Q_p(y1)])     - mean(y0)

with Q the sample quantile of Stata's ``_pctile`` (``x_(floor(q n) + 1)``),
i.e. Lee's (2009) estimator as Stata ``leebounds`` computes it for a
continuous outcome (cross-checked in ``test_teffects_R_parity.py``).

All reported quantities (bounds, trimming fraction, retention rates) match the
hand-computed definition to machine precision (observed <= 1e-16), and the
bounds bracket the true constant effect planted in the DGP.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

BETA = 1.0
N = 6000


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    d = rng.integers(0, 2, N)
    s = (rng.random(N) < (0.6 + 0.2 * d)).astype(int)  # monotone selection
    y = BETA * d + rng.normal(0, 1, N)
    df = pd.DataFrame({"y": np.where(s == 1, y, np.nan), "d": d, "s": s})
    return df, sp.lee_bounds(df, y="y", treat="d", selection="s")


def test_bounds_match_hand_trimmed_means(fitted):
    df, r = fitted
    mi = r.model_info
    y1 = df.loc[(df.d == 1) & (df.s == 1), "y"].to_numpy()
    y0 = df.loc[(df.d == 0) & (df.s == 1), "y"].to_numpy()
    s1 = df.loc[df.d == 1, "s"].mean()
    s0 = df.loc[df.d == 0, "s"].mean()
    p = (s1 - s0) / s1
    # Lee's sample-quantile rule with Stata _pctile's quantile definition:
    # the q-quantile of n sorted values is x_(floor(q n) + 1) (1-based) when
    # q n is not an integer.
    ys = np.sort(y1)
    q_lo = ys[int(np.floor((1 - p) * len(ys)))]
    q_hi = ys[int(np.floor(p * len(ys)))]
    lower = y1[y1 <= q_lo].mean() - y0.mean()
    upper = y1[y1 >= q_hi].mean() - y0.mean()
    assert mi["lower_bound"] == pytest.approx(lower, abs=1e-12)
    assert mi["upper_bound"] == pytest.approx(upper, abs=1e-12)
    assert mi["trimming_fraction"] == pytest.approx(p, abs=1e-15)
    assert mi["retention_treated"] == pytest.approx(s1, abs=1e-15)
    assert mi["retention_control"] == pytest.approx(s0, abs=1e-15)


def test_bounds_bracket_true_effect(fitted):
    _, r = fitted
    mi = r.model_info
    assert mi["lower_bound"] <= BETA <= mi["upper_bound"]
    assert mi["bound_width"] == pytest.approx(
        mi["upper_bound"] - mi["lower_bound"], abs=1e-15
    )
