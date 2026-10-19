"""``sp.bart_causal``: known truth, identities and input checks.

Comparison with R bcf / bartCause by seed replication is in
``tests/reference_parity/test_bart_causal_seed_screen.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _dgp(n=300, seed=0, het=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    p = 1 / (1 + np.exp(-0.8 * x[:, 0]))
    z = rng.binomial(1, p)
    tau = 1.0 + (0.8 * (x[:, 1] > 0) if het else 0.0)
    y = x[:, 0] + 0.5 * x[:, 1] ** 2 + tau * z + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "z": z, "x1": x[:, 0], "x2": x[:, 1]})
    return df, tau * np.ones(n)


@pytest.mark.parametrize("method", ["bcf", "bart"])
def test_interval_covers_the_true_ate(method):
    df, tau = _dgp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.bart_causal(
            df,
            "y",
            "z",
            ["x1", "x2"],
            method=method,
            draws=400,
            burnin=400,
            chains=2,
            seed=7,
        )
    assert r.ci[0] < tau.mean() < r.ci[1]
    assert r.diagnostics["rhat"] < 1.1


def test_bcf_recovers_effect_heterogeneity_and_att():
    df, tau = _dgp(n=400, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.bart_causal(
            df,
            "y",
            "z",
            ["x1", "x2"],
            method="bcf",
            estimand="ATT",
            draws=400,
            burnin=400,
            chains=2,
            seed=3,
        )
    hi = r.cate["mean"][df.x2 > 0].mean()
    lo = r.cate["mean"][df.x2 <= 0].mean()
    assert 0.4 < hi - lo < 1.2  # true difference 0.8
    att = tau[df.z == 1].mean()
    assert r.ci[0] < att < r.ci[1]
    # the estimand draws are the sample means of the individual draws
    np.testing.assert_allclose(
        r.draws.to_numpy(),
        r.tau_draws[:, df.z.to_numpy() == 1].mean(axis=1),
        rtol=1e-12,
    )


def test_precision_weighted_sweep_equals_rescaled_noise():
    # Identity used by the effect forest: constant precision weights c
    # with noise sigma2 give the leaf marginal of noise sigma2 / c.
    from statspai.mcmc.trees import _cutpoints, _Forest

    rng = np.random.default_rng(0)
    n = 150
    X = rng.normal(size=(n, 2))
    y = np.sin(X[:, 0]) + rng.normal(scale=0.3, size=n)
    cuts = [_cutpoints(X[:, v], 50) for v in range(2)]
    xcut = np.column_stack([np.searchsorted(c, X[:, v]) for v, c in enumerate(cuts)])
    ncut = np.array([c.size for c in cuts])
    fits = []
    for weighted in (False, True):
        f = _Forest(xcut, ncut, 10, 0.01, 0.95, 2.0, 6, 1)
        f.k["seed"](11)
        for _ in range(30):
            if weighted:
                f.sweep(y, 0.2, np.full(n, 4.0))
            else:
                f.sweep(y, 0.05)
        fits.append(f.total.copy())
    np.testing.assert_allclose(fits[0], fits[1], rtol=1e-9, atol=1e-12)


def test_input_checks():
    df, _ = _dgp(n=60)
    with pytest.raises(MethodIncompatibility):
        sp.bart_causal(df, "y", "z", ["x1"], method="xbart")
    with pytest.raises(MethodIncompatibility):
        sp.bart_causal(df, "y", "z", ["x1"], estimand="LATE")
    with pytest.raises(MethodIncompatibility):
        sp.bart_causal(df.assign(z=df.z + 1), "y", "z", ["x1"])
    with pytest.raises(MethodIncompatibility):
        sp.bart_causal(df.assign(p=1.0), "y", "z", ["x1"], propensity="p")
