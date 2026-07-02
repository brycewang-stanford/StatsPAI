"""Analytical parity: sp.bootstrap exact contract + consistency.

The nonparametric bootstrap has two machine-exact contract identities and one
statistical-consistency property:

    estimate == statistic(full sample)          (exact)
    se       == std(boot_distribution, ddof=1)  (exact)
    se       ~= analytic SE of the mean = sd/sqrt(n)   (consistency)

The first two are asserted to machine precision (observed diff 0); the third
to Monte-Carlo tolerance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(5, 2, 3000)})
    r = sp.bootstrap(df, lambda d: float(d["x"].mean()), n_boot=3000, seed=0)
    return df, r


def test_estimate_equals_full_sample_statistic(fitted):
    df, r = fitted
    assert float(r.estimate) == pytest.approx(float(df["x"].mean()), abs=1e-12)


def test_se_equals_bootstrap_dispersion(fitted):
    _, r = fitted
    assert float(r.se) == pytest.approx(
        float(np.std(r.boot_distribution, ddof=1)), abs=1e-12
    )


def test_se_consistent_with_analytic(fitted):
    df, r = fitted
    analytic = float(df["x"].std(ddof=1) / np.sqrt(len(df)))
    assert float(r.se) == pytest.approx(analytic, rel=0.1)
    assert float(r.ci_lower) < float(r.estimate) < float(r.ci_upper)
