"""``sp.msprt``: the mixture SPRT has no deterministic reference
implementation, so the evidence is an identity (the closed-form mixture
likelihood ratio equals its defining integral) and Monte Carlo checks of the
guarantee itself (S-level: fixed seeds, tolerances stated as Monte Carlo
standard errors).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


@pytest.mark.parametrize(
    "est,var,tau,theta0", [(0.3, 0.01, 0.2, 0.0), (-0.1, 0.5, 1.0, 0.2)]
)
def test_lambda_equals_the_mixture_integral(est, var, tau, theta0):
    # Lambda = int f(est | theta) / f(est | theta0) dN(theta; theta0, tau^2).
    sd = np.sqrt(var)

    def integrand(th):
        return (
            stats.norm.pdf(est, th, sd)
            / stats.norm.pdf(est, theta0, sd)
            * stats.norm.pdf(th, theta0, tau)
        )

    val, _ = integrate.quad(
        integrand, theta0 - 12 * tau, theta0 + 12 * tau, epsabs=1e-13
    )
    assert sp.msprt_lambda(est, var, tau, theta0) == pytest.approx(val, rel=1e-9)


def test_path_matches_closed_form_and_running_minimum():
    rng = np.random.default_rng(3)
    d = rng.integers(0, 2, size=300)
    df = pd.DataFrame({"d": d, "y": 0.3 * d + rng.normal(size=300)})
    r = sp.msprt(df, "y", "d", tau=0.3, sigma=1.0)
    p = r.path
    v = 1.0 / p.n_treated + 1.0 / p.n_control
    lam = sp.msprt_lambda(p.estimate.to_numpy(), v.to_numpy(), 0.3)
    np.testing.assert_allclose(p.likelihood_ratio, lam, rtol=1e-12)
    np.testing.assert_allclose(
        p.p_value, np.minimum.accumulate(np.minimum(1, 1 / lam)), rtol=1e-12
    )
    assert np.all(np.diff(p.ci_upper - p.ci_lower) <= 1e-12)  # intersection shrinks
    # The confidence sequence is the set of theta0 not rejected at alpha.
    k = len(p) - 1
    lo_k = p.estimate[k] - np.sqrt(
        v[k] * (v[k] + 0.09) / 0.09 * (2 * np.log(20) + np.log((v[k] + 0.09) / v[k]))
    )
    assert sp.msprt_lambda(p.estimate[k], v[k], 0.3, theta0=lo_k) == pytest.approx(20.0)


def _simulate(effect, n, reps, seed, sigma=1.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        d = rng.integers(0, 2, size=n)
        y = effect * d + sigma * rng.normal(size=n)
        out.append(pd.DataFrame({"d": d, "y": y}))
    return out


def test_type_one_error_under_continuous_monitoring():
    # Known sigma: P(sup Lambda >= 1/alpha) <= alpha by Ville's inequality.
    # 400 reps, n = 400 looks each; MC SE of a 0.05 rate is 0.011, tolerance
    # alpha + 3 MC SE.
    reps = _simulate(0.0, 400, 400, seed=20261019)
    rej = np.mean([sp.msprt(df, "y", "d", tau=0.2, sigma=1.0).rejected for df in reps])
    assert rej <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 400)


def test_confidence_sequence_covers_uniformly():
    # The CS contains the true effect at every look with probability >= 0.95.
    reps = _simulate(0.4, 300, 300, seed=7)
    miss = [
        bool(np.any((r.path.ci_lower > 0.4) | (r.path.ci_upper < 0.4)))
        for r in (sp.msprt(df, "y", "d", tau=0.3, sigma=1.0) for df in reps)
    ]
    assert np.mean(miss) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 300)


def test_detects_a_real_effect():
    reps = _simulate(0.5, 600, 100, seed=8)
    rej = np.mean([sp.msprt(df, "y", "d", tau=0.5, sigma=1.0).rejected for df in reps])
    assert rej >= 0.9


def test_plugin_variance_and_batched_looks():
    rng = np.random.default_rng(5)
    n = 240
    df = pd.DataFrame(
        {
            "d": rng.integers(0, 2, size=n),
            "y": rng.normal(size=n),
            "day": np.repeat(np.arange(12), n // 12),
        }
    )
    r = sp.msprt(df, "y", "d", tau=0.2, time="day", min_per_arm=2)
    assert len(r.path) == 12
    last = r.path.iloc[-1]
    g = df.groupby("d")["y"]
    v = (g.var() / g.count()).sum()
    assert last.se == pytest.approx(np.sqrt(v), rel=1e-12)
    assert last.estimate == pytest.approx(g.mean()[1] - g.mean()[0], rel=1e-12)


def test_invalid_inputs():
    df = pd.DataFrame({"d": [0, 1, 0, 1], "y": [1.0, 2.0, 1.5, 2.5]})
    with pytest.raises(MethodIncompatibility):
        sp.msprt(df, "y", "d", tau=0)
    with pytest.raises(MethodIncompatibility):
        sp.msprt(df.assign(d=[0, 1, 2, 1]), "y", "d", tau=1)
    with pytest.raises(DataInsufficient):
        sp.msprt(df, "y", "d", tau=1, min_per_arm=5)


def test_default_plug_in_keeps_the_level():
    # 2026-10-20 probe: with 2 observations per arm at the first looks the
    # plug-in variance made the always-valid test reject 8-10% of the time
    # under the null. The default now waits for 10 per arm. 300 seeded null
    # replications (n = 600, tau = 0.3): rejection rate observed 3.0%;
    # the bound 0.05 + 2 * MC SE (0.0126) would still fail at 2 per arm
    # (0.095 in 600 replications).
    rej = 0
    R = 300
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        df = pd.DataFrame({"d": rng.integers(0, 2, 600), "y": rng.normal(size=600)})
        rej += sp.msprt(df, "y", "d", tau=0.3).p_value < 0.05
    assert rej / R <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / R)
