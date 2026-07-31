"""Uniform (simultaneous) inference over the quantile grid (WP-7).

Why this exists
---------------
A QTE analysis makes statements about a *curve*: "the effect is zero at
every quantile", "the effect is constant", "treatment shifts the whole
distribution". A row of pointwise 95% intervals does not support any of
them -- with K quantiles the chance that all K intervals cover at once is
far below 95%, and roughly ``0.05 * K`` spurious pointwise rejections are
expected under the null.

This module adds:

* :func:`statspai.qte._core.uniform_band` -- a multiplier-bootstrap band
  with **simultaneous** coverage;
* ``QTEResult.test_no_effect()`` -- H0: ``QTE(tau) = 0`` at every tau;
* ``QTEResult.test_constant_effect()`` -- H0: ``QTE(tau)`` does not vary,
  i.e. "an ATE is an adequate summary". This is the hypothesis that
  distinguishes a genuine distributional finding from a mean effect.

Anchors
-------
A. **Simultaneous coverage.** The uniform band attains ~95% joint coverage
   over 9 quantiles where the pointwise band attains ~76%. Both numbers are
   asserted: a band that is merely *wider* is not evidence it is *correct*.
B. **The uniform critical value exceeds z.** ~2.64 vs 1.96 here.
C. **Size and power of both curve tests**, on three DGPs whose middle case
   is the discriminating one:

   ===============  ==================  ====================
   DGP              reject "no effect"  reject "constant"
   ===============  ==================  ====================
   no effect        ~0.05               ~0.05
   constant shift   ~1.00               ~0.05   <- key
   scale change     ~1.00               ~1.00
   ===============  ==================  ====================

   A test that always rejects passes the power rows and fails the null row;
   one that never rejects does the reverse. Only a correct test passes all
   six cells, and only the middle row separates "there is an effect" from
   "the effect varies".
D. **Rearrangement** is a projection that restores monotonicity.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.qte._core import (  # noqa: E402
    functional_test,
    multiplier_bootstrap,
    rearrange,
    uniform_band,
)

TAUS_WIDE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TAUS_TEST = [0.15, 0.3, 0.5, 0.7, 0.85]


def _sim(kind: str, seed: int, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n)
    if kind == "null":
        y = rng.normal(0, 1, n)
    elif kind == "shift":
        y = 0.4 * d + rng.normal(0, 1, n)
    elif kind == "scale":
        y = np.where(d == 1, rng.normal(0, 1.8, n), rng.normal(0, 1, n))
    else:  # pragma: no cover
        raise ValueError(kind)
    return pd.DataFrame({"y": y, "d": d})


# ── A / B. simultaneous coverage ───────────────────────────────────── #


def test_uniform_band_has_simultaneous_coverage():
    """Nominal 95% JOINT coverage over 9 quantiles.

    Asserts the pointwise band undercovers too -- otherwise a band that is
    simply very wide would pass, and we would have no evidence the
    multiplier bootstrap is doing anything principled.
    """
    true_qte, reps = 1.5, 300
    pw_all = uni_all = 0
    crits = []
    for s in range(reps):
        rng = np.random.default_rng(s)
        n = 2000
        d = rng.integers(0, 2, n)
        y = true_qte * d + rng.normal(0, 1, n)
        r = sp.qte(
            pd.DataFrame({"y": y, "d": d}),
            y="y",
            treatment="d",
            method="firpo_qte",
            quantiles=TAUS_WIDE,
            se="analytic",
        )
        pw_all += bool(np.all((r.ci_lower <= true_qte) & (true_qte <= r.ci_upper)))
        uni_all += bool(
            np.all((r.ci_lower_uniform <= true_qte) & (true_qte <= r.ci_upper_uniform))
        )
        crits.append(r.uniform_crit)

    pointwise = pw_all / reps
    uniform = uni_all / reps
    assert 0.90 < uniform < 0.99, f"uniform joint coverage {uniform}"
    # The pointwise band MUST undercover jointly; if it did not, the uniform
    # band would be solving a non-problem.
    assert pointwise < 0.90, f"pointwise joint coverage {pointwise}"
    assert uniform > pointwise + 0.08


def test_uniform_critical_value_exceeds_pointwise_z():
    r = sp.qte(
        _sim("shift", 0),
        y="y",
        treatment="d",
        method="firpo_qte",
        quantiles=TAUS_WIDE,
        se="analytic",
    )
    assert r.uniform_crit > 1.96
    assert r.uniform_crit < 5.0  # sanity: not blowing up
    # And the band is correspondingly wider at every tau.
    assert np.all(r.ci_lower_uniform <= r.ci_lower + 1e-12)
    assert np.all(r.ci_upper_uniform >= r.ci_upper - 1e-12)


def test_uniform_band_absent_on_bootstrap_path():
    """The band needs influence functions; bootstrap SEs do not provide them."""
    r = sp.qte(
        _sim("shift", 1, n=600),
        y="y",
        treatment="d",
        method="firpo_qte",
        quantiles=[0.25, 0.5, 0.75],
        se="bootstrap",
        n_boot=20,
    )
    assert r.ci_lower_uniform is None
    with pytest.raises(ValueError, match="analytic"):
        r.test_no_effect()


# ── C. curve-level hypothesis tests ────────────────────────────────── #


def _reject_rates(kind: str, reps: int = 120):
    pn, pc = [], []
    for s in range(reps):
        r = sp.qte(
            _sim(kind, s),
            y="y",
            treatment="d",
            method="firpo_qte",
            quantiles=TAUS_TEST,
            se="analytic",
        )
        pn.append(r.test_no_effect(n_boot=400, seed=s)[1])
        pc.append(r.test_constant_effect(n_boot=400, seed=s)[1])
    return (
        float((np.asarray(pn) < 0.05).mean()),
        float((np.asarray(pc) < 0.05).mean()),
    )


def test_curve_tests_have_correct_size_under_the_null():
    no_eff, const = _reject_rates("null")
    assert 0.005 < no_eff < 0.15, f"'no effect' size {no_eff}"
    assert 0.005 < const < 0.15, f"'constant' size {const}"


def test_constant_shift_is_detected_but_not_called_heterogeneous():
    """The discriminating case.

    A real but CONSTANT effect must reject "no effect anywhere" and must NOT
    reject "the effect is constant". A test that conflates the two -- or one
    that never rejects -- fails here.
    """
    no_eff, const = _reject_rates("shift")
    assert no_eff > 0.9, f"failed to detect a real effect: {no_eff}"
    assert const < 0.20, f"falsely called a constant effect heterogeneous: {const}"


def test_scale_change_is_detected_as_heterogeneous():
    """Treatment changes the spread, not the mean: only a curve-level test
    of constancy can see it."""
    no_eff, const = _reject_rates("scale")
    assert no_eff > 0.9, no_eff
    assert const > 0.8, f"failed to detect tau-varying effect: {const}"


@pytest.mark.parametrize("kind", ["ks", "cvm"])
def test_both_statistics_work(kind):
    r = sp.qte(
        _sim("scale", 3),
        y="y",
        treatment="d",
        method="firpo_qte",
        quantiles=TAUS_TEST,
        se="analytic",
    )
    stat, p = r.test_constant_effect(kind=kind, n_boot=300, seed=0)
    assert np.isfinite(stat) and 0.0 <= p <= 1.0
    assert p < 0.05


# ── D. primitives ──────────────────────────────────────────────────── #


def test_rearrange_is_a_monotone_projection():
    x = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    r = rearrange(x)
    assert np.all(np.diff(r) >= 0)
    # A projection: sorting preserves the multiset.
    np.testing.assert_allclose(np.sort(x), r)
    # Idempotent.
    np.testing.assert_allclose(rearrange(r), r)


def test_rearrange_reduces_distance_to_a_monotone_truth():
    """Chernozhukov-Fernandez-Val-Galichon (2010): rearranging an estimate of
    a monotone curve never increases the Lp distance to the truth."""
    rng = np.random.default_rng(0)
    truth = np.linspace(0.0, 1.0, 25)  # monotone by construction
    worse = 0
    for _ in range(200):
        noisy = truth + rng.normal(0, 0.2, len(truth))
        if np.linalg.norm(rearrange(noisy) - truth) > np.linalg.norm(noisy - truth):
            worse += 1
    assert worse == 0, f"rearrangement hurt in {worse}/200 draws"


@pytest.mark.parametrize("weights", ["rademacher", "gaussian", "mammen"])
def test_multiplier_bootstrap_shapes_and_weights(weights):
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(500, 6))
    draws = multiplier_bootstrap(psi, n_boot=200, seed=0, weights=weights)
    assert draws.shape == (200,)
    assert np.all(draws >= 0)
    assert np.isfinite(draws).all()


def test_multiplier_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError, match="2-D"):
        multiplier_bootstrap(np.zeros(10))
    with pytest.raises(ValueError, match="rademacher"):
        multiplier_bootstrap(np.zeros((10, 2)), weights="nope")


def test_functional_test_rejects_bad_kind():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="'ks' or 'cvm'"):
        functional_test(np.zeros(3), rng.normal(size=(50, 3)), kind="nope")


def test_uniform_band_returns_consistent_pieces():
    rng = np.random.default_rng(1)
    n, k = 400, 5
    psi = rng.normal(size=(n, k))
    est = np.arange(k, dtype=float)
    lo, hi, se, crit = uniform_band(est, psi, alpha=0.05, n_boot=300, seed=0)
    np.testing.assert_allclose(hi - lo, 2 * crit * se)
    np.testing.assert_allclose((lo + hi) / 2, est)
    assert crit > 1.96
