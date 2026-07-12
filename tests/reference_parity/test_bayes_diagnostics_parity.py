"""Analytical parity: Bayesian causal estimators — posterior recovery + NUTS
convergence diagnostics.

On a clean, strongly-identified DGP the posterior should concentrate on the
known-truth parameter and the sampler should converge: ``rhat < 1.01`` and a
healthy effective sample size (StatsPAI warns below 400, per CLAUDE.md §11).
These guards run ``sp.bayes_iv`` (LATE recovery) and ``sp.bayes_rd`` (sharp-RD
jump recovery) with a small, deterministic NUTS budget. Analytical evidence
tier (known-truth posterior recovery on deterministic DGPs).

Requires the ``[bayes]`` extra (PyMC + ArviZ); skipped otherwise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

pytest.importorskip("pymc")
pytest.importorskip("arviz")


@pytest.fixture(scope="module")
def bayes_iv_fit():
    rng = np.random.default_rng(1)
    n = 1200
    z = rng.normal(0, 1, n)
    u = rng.normal(0, 1, n)                      # confounder
    x = 1.0 * z + 0.6 * u + rng.normal(0, 0.3, n)  # endogenous treatment
    y = 1.5 * x + u                               # true LATE = 1.5
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.bayes_iv(
            df, y="y", treat="x", instrument="z",
            draws=800, tune=800, chains=2, random_state=0,
        )


def test_bayes_iv_posterior_recovers_late(bayes_iv_fit):
    assert float(bayes_iv_fit.posterior_mean) == pytest.approx(1.5, abs=0.15)


def test_bayes_iv_hdi_is_valid_interval(bayes_iv_fit):
    lo = float(bayes_iv_fit.hdi_lower)
    hi = float(bayes_iv_fit.hdi_upper)
    assert lo < hi
    assert lo <= float(bayes_iv_fit.posterior_mean) <= hi


def test_bayes_iv_converged(bayes_iv_fit):
    assert float(bayes_iv_fit.rhat) < 1.01
    assert float(bayes_iv_fit.ess) > 400


@pytest.fixture(scope="module")
def bayes_rd_fit():
    rng = np.random.default_rng(0)
    n = 1500
    tau = 3.0
    run = rng.uniform(-1, 1, n)
    y = 1.0 + 2.0 * run + tau * (run >= 0.0) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "run": run})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.bayes_rd(
            df, y="y", running="run", cutoff=0.0,
            draws=800, tune=800, chains=2, random_state=0,
        )


def test_bayes_rd_posterior_recovers_jump(bayes_rd_fit):
    assert float(bayes_rd_fit.posterior_mean) == pytest.approx(3.0, abs=0.3)


def test_bayes_rd_converged(bayes_rd_fit):
    assert float(bayes_rd_fit.rhat) < 1.01
    assert float(bayes_rd_fit.ess) > 400
