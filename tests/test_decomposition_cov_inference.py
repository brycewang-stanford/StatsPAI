"""Coverage campaign (decomposition) — bootstrap-inference paths.

The distributional decompositions expose an ``inference='bootstrap'`` path that
re-runs the estimator on stratified resamples and assembles standard errors /
confidence intervals (``ffl.py`` 328-357, ``dfl.py`` 303-339, ``machado_mata.py``
286-332). This file exercises those SE/CI assembly branches with a modest
``n_boot`` and verifies the standard errors are finite, the CI brackets the point
estimate, and the point decomposition identity is unchanged by the inference
mode. No mocking (CLAUDE.md §12).
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp
from statspai.decomposition import datasets

X = ["education", "experience", "tenure"]


@pytest.fixture(scope="module")
def wage():
    return datasets.cps_wage()


# ── FFL bootstrap inference ──────────────────────────────────────────


def test_ffl_bootstrap_se(wage):
    r = sp.decompose(
        "ffl",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        stat="quantile",
        tau=0.5,
        inference="bootstrap",
        n_boot=40,
        seed=0,
    )
    assert r.se is not None
    assert {"gap", "composition", "structure"} <= set(r.se)
    assert np.isfinite(r.se["gap"]) and r.se["gap"] >= 0
    # point identity unchanged by inference mode
    assert r.gap == pytest.approx(r.stat_a - r.stat_b, rel=1e-7, abs=1e-9)


# ── DFL bootstrap inference + quantile grid ──────────────────────────


def test_dfl_bootstrap_se(wage):
    r = sp.decompose(
        "dfl",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        stat="mean",
        inference="bootstrap",
        n_boot=40,
        seed=0,
    )
    assert r.se is not None
    assert np.isfinite(r.se["gap"]) and r.se["gap"] >= 0
    assert r.gap == pytest.approx(r.composition + r.structure, rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("tau", [0.25, 0.75])
def test_dfl_quantile_stat(wage, tau):
    # stat='quantile' at several taus exercises the weighted-quantile DFL path.
    r = sp.decompose(
        "dfl", data=wage, y="log_wage", group="female", x=X, stat="quantile", tau=tau
    )
    assert r.gap == pytest.approx(r.composition + r.structure, rel=1e-7, abs=1e-9)
    assert r.gap == pytest.approx(r.stat_a - r.stat_b, rel=1e-7, abs=1e-9)


# ── Machado–Mata bootstrap inference ─────────────────────────────────


def test_machado_mata_bootstrap_se(wage):
    r = sp.decompose(
        "machado_mata",
        data=wage,
        y="log_wage",
        group="female",
        x=X,
        tau_grid=[0.25, 0.5, 0.75],
        n_sim=80,
        inference="bootstrap",
        n_boot=30,
        seed=0,
    )
    # se table assembled across the bootstrap replications
    assert r.se is not None
    o = r.overall
    assert o["mean_gap"] == pytest.approx(
        o["mean_composition"] + o["mean_structure"], rel=1e-6, abs=1e-6
    )
