"""Analytical parity: sp.front_door identification under unobserved confounding.

The front-door criterion (Pearl 1995) identifies the causal effect of T on Y
through a mediator M even when an *unobserved* U confounds T and Y, provided
M is unconfounded with U and fully mediates T. On a known DGP

    U ~ N(0,1);  T = 1{0.8 U + e_T > 0};  M = 1{0.6 T + e_M > 0}
    Y = beta_MY * M + 1.2 U + e_Y

the population front-door ATE is (P(M=1|T=1) - P(M=1|T=0)) * beta_MY =
(Phi(0.6) - 0.5) * 1.5 ~ 0.3386. The estimator recovers it, while a naive
Y ~ T regression is biased upward by the confounder. Analytical evidence tier
(known-truth recovery on a deterministic DGP; no cross-package reference).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

BETA_MY = 1.5
POP_ATE = (norm.cdf(0.6) - 0.5) * BETA_MY  # ~0.3386


def _simulate(seed, n=5000):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, n)
    T = (0.8 * U + rng.normal(0, 1, n) > 0).astype(int)
    M = (0.6 * T + rng.normal(0, 1, n) > 0).astype(int)
    Y = BETA_MY * M + 1.2 * U + rng.normal(0, 0.5, n)
    return pd.DataFrame({"Y": Y, "T": T, "M": M})


def test_recovers_population_front_door_ate():
    df = _simulate(0)
    r = sp.front_door(df, y="Y", treat="T", mediator="M")
    assert float(r.estimate) == pytest.approx(POP_ATE, abs=0.05)


def test_less_biased_than_naive_confounded_regression():
    df = _simulate(0)
    fd = float(sp.front_door(df, y="Y", treat="T", mediator="M").estimate)
    # Naive Y ~ T is confounded by U (which drives both T and Y) => upward bias.
    t = df["T"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    naive = np.cov(y, t, bias=True)[0, 1] / np.var(t)
    assert naive > fd + 0.2  # front-door strips a sizeable confounding bias
    assert abs(fd - POP_ATE) < abs(naive - POP_ATE)
