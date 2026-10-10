"""Fit statistics and stopping rules that used to cost more than the fit.

* the ROC area of ``sp.logit`` / ``sp.probit`` was a Python loop over the
  sorted scores (60% of the run time at n = 200,000);
* ``sp.nbreg`` on data that are not overdispersed spent its whole outer
  iteration budget and reported non-convergence, because the dispersion
  wanders where the likelihood is flat.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.regression.logit_probit import _roc_auc


@pytest.mark.parametrize("decimals", [None, 1, 0])
def test_roc_auc_is_the_share_of_correctly_ordered_pairs(decimals):
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    p = rng.uniform(size=300)
    if decimals is not None:
        p = np.round(p, decimals)  # ties count one half
    pos, neg = p[y == 1], p[y == 0]
    diff = pos[:, None] - neg[None, :]
    want = ((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size
    assert _roc_auc(y, p) == pytest.approx(want, rel=1e-13)


def test_roc_auc_is_undefined_without_both_classes_or_with_nan():
    assert np.isnan(_roc_auc(np.ones(5), np.linspace(0.1, 0.9, 5)))
    assert np.isnan(_roc_auc(np.array([0, 1, 1]), np.array([0.2, np.nan, 0.7])))


@pytest.mark.parametrize("dispersion", ["mean", "constant"])
def test_nbreg_stops_when_the_likelihood_is_flat_in_the_dispersion(dispersion):
    # Poisson counts: the negative binomial has nothing to add, and the
    # coefficients are the Poisson coefficients.
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    df = pd.DataFrame({"y": rng.poisson(np.exp(0.2 + 0.3 * x)), "x": x})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sp.nbreg("y ~ x", df, dispersion=dispersion)
    assert not any("did not converge" in str(w.message) for w in caught)
    assert fit.model_info["converged"]
    assert fit.model_info["iterations"] < 20
    pois = sp.poisson("y ~ x", df)
    # the fitted dispersion is ~1e-8 or a sampling-noise positive value;
    # either way it moves the slope by far less than its standard error
    assert abs(fit.params["x"] - pois.params["x"]) < 0.05 * pois.std_errors["x"]


def test_nbreg_with_real_overdispersion_is_unaffected_by_the_stall_rule():
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=n)
    mu = np.exp(0.3 + 0.5 * x)
    df = pd.DataFrame({"y": rng.negative_binomial(2.0, 2.0 / (2.0 + mu)), "x": x})
    fit = sp.nbreg("y ~ x", df)
    # joint MLE: the score of (beta, ln alpha) vanishes
    assert fit.model_info["gradient_norm"] < 1e-6
    assert fit.model_info["dispersion"] == pytest.approx(0.5, abs=0.08)


def _peak_mb(fn):
    import tracemalloc

    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


def test_fracreg_and_ivqreg_do_not_allocate_n_by_n():
    # Both built a dense n x n matrix (the diagonal weights; the projection
    # on the exogenous regressors), which is 288 MB at this n and 320 GB at
    # n = 200,000. What they need is linear in n.
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    endo = 0.8 * z + 0.5 * u + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": z, "endo": endo, "y": 1 + endo + x + u})
    df["share"] = 1 / (1 + np.exp(-(0.3 * x + rng.normal(size=n))))
    n_by_n = n * n * 8 / 1e6
    assert _peak_mb(lambda: sp.fracreg(df, "share", ["x"])) < n_by_n / 10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a 5-point grid is too coarse for SEs
        peak = _peak_mb(lambda: sp.ivqreg(df, "y", "endo", "z", exog=["x"], n_grid=5))
    assert peak < n_by_n / 10


def _rd_frame(n):
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, n)
    d = (x >= 0).astype(float)
    t = np.arange(n) - n // 2
    return pd.DataFrame(
        {
            "y": 0.5 * d + x + 0.3 * rng.normal(size=n),
            "x": x,
            "t": t.astype(float),
            "yt": 0.4 * (t >= 0) + rng.normal(size=n),
            "d": rng.integers(0, 2, n),
            "z": rng.normal(size=n),
        }
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda df: sp.rdit(df, y="yt", time="t", cutoff=0),
        lambda df: sp.rdmc(df, "y", "x", [-0.3, 0.3], bandwidth=0.6),
        lambda df: sp.rd_distributional_design(df, "y", "x", bandwidth=0.9),
        lambda df: sp.qte(
            df, "y", "d", method="conditional_qr", controls=["z"], n_boot=2
        ),
        lambda df: sp.robreg("y ~ x + z", df, method="m", init="lad"),
    ],
    ids=[
        "rdit",
        "rdmc",
        "rd_distributional_design",
        "qte_conditional_qr",
        "robreg_lad",
    ],
)
def test_weighted_least_squares_steps_stay_linear_in_memory(call):
    # Each of these wrote the weights of a weighted fit as np.diag(w), or
    # the identity blocks of an LP as dense matrices.
    n = 6000
    df = _rd_frame(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _peak_mb(lambda: call(df)) < n * n * 8 / 1e6 / 10
