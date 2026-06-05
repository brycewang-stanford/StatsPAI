"""Coverage tests for statspai.rd frontier modules.

Covers rd_interference, rd_multi_score, rd_distribution, rd_bayes_hte,
rd_distributional_design across kernels / auto-bandwidth / error paths,
plus the dispatcher routing for the running/cutoff-style methods.
Real synthetic RD data; structural properties asserted.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _rd_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.uniform(-1, 1, n)
    Rn = R + 0.3 * rng.standard_normal(n)
    treat = (R >= 0).astype(int)
    Y = 0.5 * R + 1.5 * treat + rng.standard_normal(n) * 0.3
    return pd.DataFrame({
        "y": Y, "r": R, "rn": Rn,
        "x1": rng.standard_normal(n), "x2": rng.standard_normal(n),
    })


@pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
def test_rd_interference_kernels(kernel):
    df = _rd_data()
    res = sp.rd_interference(df, y="y", running="r", neighbour_running="rn",
                             cutoff=0.0, bandwidth=0.6, kernel=kernel)
    assert isinstance(res, sp.RDInterferenceResult)
    assert np.isfinite(res.direct_effect)


def test_rd_interference_auto_bandwidth():
    df = _rd_data()
    res = sp.rd_interference(df, y="y", running="r", neighbour_running="rn",
                             cutoff=0.0)
    assert np.isfinite(res.direct_effect)


def test_rd_multi_score_properties_and_error():
    rng = np.random.default_rng(2)
    n = 600
    r1 = rng.uniform(-1, 1, n)
    r2 = rng.uniform(-1, 1, n)
    treat = ((r1 >= 0) & (r2 >= 0)).astype(int)
    Y = 0.3 * r1 + 0.3 * r2 + 2.0 * treat + rng.standard_normal(n) * 0.4
    df = pd.DataFrame({"y": Y, "r1": r1, "r2": r2})
    res = sp.rd_multi_score(df, y="y", running_vars=["r1", "r2"],
                            cutoffs=[0.0, 0.0], bandwidth=0.6)
    assert 0 <= res.boundary_share <= 1
    with pytest.raises(ValueError, match="len"):
        sp.rd_multi_score(df, y="y", running_vars=["r1"], cutoffs=[0.0, 0.5])


def test_rd_distribution():
    df = _rd_data()
    res = sp.rd_distribution(df, y="y", running="r", cutoff=0.0,
                             quantiles=np.array([0.25, 0.5, 0.75]),
                             bandwidth=0.6)
    assert len(res.qte) == 3


def test_rd_distributional_design():
    df = _rd_data()
    res = sp.rd_distributional_design(df, y="y", running="r", cutoff=0.0,
                                      quantiles=np.array([0.25, 0.5, 0.75]),
                                      bandwidth=0.6)
    assert len(res.rdd_effect) == 3
    assert len(res.rkd_effect) == 3


def test_rd_bayes_hte():
    df = _rd_data()
    res = sp.rd_bayes_hte(df, y="y", running="r", covariates=["x1", "x2"],
                          cutoff=0.0, bandwidth=0.6, n_draws=200)
    assert len(res.cate) == len(df)
    assert np.isfinite(res.posterior_mean)


def test_dispatcher_running_cutoff_methods():
    df = _rd_data()
    # interference via dispatcher (uses running/cutoff renaming)
    res = sp.rd.fit(df, y="y", x="r", c=0.0, method="interference",
                    neighbour_running="rn", bandwidth=0.6)
    assert isinstance(res, sp.RDInterferenceResult)
    # bayes_hte via dispatcher
    res2 = sp.rd.fit(df, y="y", x="r", c=0.0, method="bayes_hte",
                     covariates=["x1", "x2"], bandwidth=0.6, n_draws=150)
    assert np.isfinite(res2.posterior_mean)
    # distribution via dispatcher
    res3 = sp.rd.fit(df, y="y", x="r", c=0.0, method="distribution",
                     quantiles=np.array([0.5]), bandwidth=0.6)
    assert len(res3.qte) == 1
