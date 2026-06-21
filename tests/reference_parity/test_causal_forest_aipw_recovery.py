"""Self-contained recovery test for the causal-forest AIPW ATE/ATT.

No R, no fixtures: builds a clean-overlap deterministic DGP with an
analytically known average treatment effect and asserts that
``sp.causal_forest.average_treatment_effect`` (the doubly-robust AIPW
estimand grf reports) recovers the truth within 4 standard errors.  It
also asserts that the **plug-in** mean of the CATE predictions
(``cf.ate()``) is materially more biased than the AIPW estimate, which
is the whole reason the headline aggregation was switched to AIPW.

This is the Tier-1 (no external language required) evidence behind the
module-13 causal-forest parity row; the cross-language agreement with
``grf`` itself is checked in ``test_grf_parity.py`` against a committed
grf fixture, and re-verified live in ``tests/r_parity/13_causal_forest``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

SEED = 42
N = 4000
K = 5


def _clean_overlap_dgp(seed: int = SEED, n: int = N):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, K))
    e = 0.5 + 0.2 * np.tanh(X[:, 0])  # propensity in [0.30, 0.70]
    T = (rng.uniform(size=n) < e).astype(int)
    tau = 1.0 + 0.5 * X[:, 1]  # heterogeneous CATE, E[tau] = 1
    Y = X[:, 0] + 0.5 * X[:, 2] + tau * T + rng.normal(scale=1.0, size=n)
    return X, T, Y, float(tau.mean()), float(tau[T == 1].mean())


@pytest.fixture(scope="module")
def fitted():
    X, T, Y, true_ate, true_att = _clean_overlap_dgp()
    cf = sp.causal_forest(
        Y=Y,
        T=T,
        X=X,
        n_estimators=2000,
        random_state=SEED,
        discrete_treatment=True,
    )
    return cf, true_ate, true_att


def test_aipw_ate_recovers_truth(fitted):
    cf, true_ate, _ = fitted
    r = cf.average_treatment_effect(target_sample="all")
    assert r["method"] == "aipw"
    z = abs(r["estimate"] - true_ate) / r["se"]
    assert z < 4.0, (
        f"AIPW ATE={r['estimate']:.4f} (SE {r['se']:.4f}) is {z:.2f} SE "
        f"from the known truth {true_ate:.4f}"
    )


def test_aipw_att_recovers_truth(fitted):
    cf, _, true_att = fitted
    r = cf.average_treatment_effect(target_sample="treated")
    z = abs(r["estimate"] - true_att) / r["se"]
    assert z < 4.0, (
        f"AIPW ATT={r['estimate']:.4f} (SE {r['se']:.4f}) is {z:.2f} SE "
        f"from the known truth {true_att:.4f}"
    )


def test_clean_overlap_clip_stabilizes_att(fitted):
    """Explicit overlap clipping stabilizes ATT on the clean-overlap DGP.

    The clean-overlap DGP has true propensities in [0.30, 0.70].  A
    flexible treatment nuisance forest can still report near-certain
    treatment probabilities in finite samples.  The public
    ``average_treatment_effect(..., clip=...)`` knob should let users
    encode that overlap knowledge without changing CATE training.
    """
    cf, _, true_att = fitted
    r = cf.average_treatment_effect(target_sample="treated", clip=0.25)
    assert abs(r["estimate"] - true_att) < 0.01


@pytest.mark.parametrize("seed", [7, 42, 2026])
def test_clean_overlap_aipw_is_stable_across_seeds(seed):
    """AIPW aggregation should not be a one-seed parity accident.

    This guard is intentionally Python-only and smaller than the live
    Track-A forest fixture, so it runs in the normal test suite.  The DGP
    has known clean overlap; with overlap-consistent clipping, both ATE
    and ATT should recover their known targets comfortably within the
    reported AIPW standard errors across independent forest seeds.
    """
    X, T, Y, true_ate, true_att = _clean_overlap_dgp(seed=seed, n=2500)
    cf = sp.causal_forest(
        Y=Y,
        T=T,
        X=X,
        n_estimators=800,
        random_state=seed,
        discrete_treatment=True,
    )
    for target_sample, truth in (("all", true_ate), ("treated", true_att)):
        r = cf.average_treatment_effect(target_sample=target_sample, clip=0.25)
        z = abs(r["estimate"] - truth) / r["se"]
        assert z < 2.5, (
            f"seed={seed} target={target_sample}: AIPW estimate "
            f"{r['estimate']:.4f} (SE {r['se']:.4f}) is {z:.2f} SE from "
            f"known truth {truth:.4f}"
        )


def test_aipw_ci_covers_truth(fitted):
    cf, true_ate, _ = fitted
    r = cf.average_treatment_effect(target_sample="all")
    assert r["ci_low"] <= true_ate <= r["ci_high"], (
        f"95% AIPW CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}] does not "
        f"cover the truth {true_ate:.4f}"
    )


def test_plugin_is_more_biased_than_aipw(fitted):
    """The plug-in CATE average should be further from truth than AIPW.

    This is the quantitative justification for using AIPW as the headline
    aggregation: on a clean-overlap DGP the plug-in mean overshoots.
    """
    cf, true_ate, _ = fitted
    plug_in = cf.ate()
    aipw = cf.average_treatment_effect(target_sample="all")["estimate"]
    assert abs(plug_in - true_ate) > abs(aipw - true_ate), (
        f"plug-in |bias|={abs(plug_in - true_ate):.4f} should exceed AIPW "
        f"|bias|={abs(aipw - true_ate):.4f} (plug-in={plug_in:.4f}, "
        f"aipw={aipw:.4f}, truth={true_ate:.4f})"
    )
