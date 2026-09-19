"""Analytical parity: sp.pate transportability recovery.

The Population Average Treatment Effect reweights an experimental sample to a
target population's covariate distribution, so it estimates
``E_target[tau(X)]`` rather than the study SATE. On a known DGP with a
covariate-modified effect tau(x) = 1 + 0.5 x, an experiment drawn from
X ~ N(0, 1) has SATE = 1.0, while a target population with X ~ N(1, 1) has
PATE = E[1 + 0.5 X] = 1.5. The estimator tracks the *target* value, not the
sample SATE. Analytical evidence tier (known-truth recovery on a deterministic
DGP; no cross-package reference).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

TARGET_PATE = 1.5
SAMPLE_SATE = 1.0


def _simulate(seed, n=4000):
    rng = np.random.default_rng(seed)
    xe = rng.normal(0, 1, n)
    de = rng.integers(0, 2, n)
    ye = (1 + 0.5 * xe) * de + 0.7 * xe + rng.normal(0, 0.5, n)
    exp = pd.DataFrame({"y": ye, "d": de, "x": xe})
    tgt = pd.DataFrame({"x": rng.normal(1, 1, n)})  # shifted target
    return exp, tgt


def _fit(exp, tgt, seed):
    return sp.pate(
        exp, tgt, y="y", treatment="d", covariates=["x"], method="ipw", seed=seed
    )


def test_recovers_target_pate_across_seeds():
    ests = []
    for seed in range(6):
        exp, tgt = _simulate(seed)
        ests.append(float(_fit(exp, tgt, seed).estimate))
    assert float(np.mean(ests)) == pytest.approx(TARGET_PATE, abs=0.1)


def test_reweighting_moves_estimate_toward_target():
    # PATE should sit clearly above the sample SATE (1.0) since the target
    # population is shifted to higher X where tau(x) is larger.
    exp, tgt = _simulate(0)
    est = float(_fit(exp, tgt, 0).estimate)
    assert est > (SAMPLE_SATE + TARGET_PATE) / 2  # past the midpoint toward target


def _dr_design(seed, n_exp=20_000, n_tgt=40_000):
    """Participation is logistic in x (so the participation model is correct)
    while tau(x) = 1 + x^2 is quadratic (so the linear outcome model with a
    D*x interaction is WRONG). Target estimand: E[1 + x^2 | S = 0]."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, 400_000)
    s = rng.uniform(size=x.size) < 1 / (1 + np.exp(-(-0.5 + 0.5 * x)))
    xe = rng.choice(x[s], n_exp, replace=False)
    xt = rng.choice(x[~s], n_tgt, replace=False)
    d = rng.integers(0, 2, n_exp)
    y = xe**2 + d * (1 + xe**2) + rng.normal(0, 1, n_exp)
    truth = 1 + float(np.mean(x[~s] ** 2))
    return pd.DataFrame({"y": y, "d": d, "x": xe}), pd.DataFrame({"x": xt}), truth


def test_aipw_is_doubly_robust_when_only_the_participation_model_is_right():
    """Known-truth check of the AIPW augmentation term.

    With the participation model correct and the outcome model misspecified,
    the doubly robust estimator must stay consistent. Until 1.28 the
    augmentation was scaled by n_exp / n_tgt and normalised by the sum of ALL
    odds weights instead of each arm's; on this design (16 replications) it
    was biased by -0.168 (MC SE 0.013) against -0.001 (MC SE 0.016) now.
    The bound 0.06 is ~4 MC SEs.
    """
    errs = []
    for seed in range(16):
        exp, tgt, truth = _dr_design(seed)
        est = sp.pate(
            exp,
            tgt,
            y="y",
            treatment="d",
            covariates=["x"],
            method="aipw",
            n_boot=2,
            seed=0,
            trim=1e-6,
        ).estimate
        errs.append(float(est) - truth)
    assert abs(float(np.mean(errs))) < 0.06
