"""Reference parity / analytical anchors: ``sp.dist_iv``, ``sp.kan_dlate``.

Estimand
--------
The complier quantile treatment effect

    QTE_c(tau) = F^{-1}_{Y(1)|c}(tau) - F^{-1}_{Y(0)|c}(tau)

identified by Abadie kappa weighting (Abadie 2002, 2003; Frolich & Melly
2013 with covariates).

Why this file was rewritten (WP-1)
----------------------------------
The previous version of this file only probed a **pure location shift with
full compliance**, where the true QTE is a flat line at ``beta``.  That is
the one design in which a quantile estimand is indistinguishable from a mean
estimand, so it passed even though ``sp.dist_iv`` (<= 1.20.0) implemented a
"Wald ratio of quantiles",

    LATE_q = [Q(tau|Z=1) - Q(tau|Z=0)] / [E(D|Z=1) - E(D|Z=0)],

which is inconsistent for any quantile estimand because the quantile
operator is not linear.  Two things had to change:

1. **Partial compliance** — with always-takers and never-takers the Wald
   denominator is no longer 1, and the old estimator's ``1/Delta_p``
   inflation becomes visible.  ``test_wald_ratio_of_quantiles_regression``
   pins the exact number the old code produced so a revert cannot pass.
2. **Non-degenerate truth** — ``test_recovers_quantile_fan`` uses a DGP whose
   true QTE *varies with tau*.  An estimator that silently reports a mean
   effect at every quantile fails it.

Anchors
-------
A. **Partial-compliance level** (30% always-taker / 50% complier / 20%
   never-taker; complier ``Y1 = Y0 + 2``).  True ``QTE(tau) == 2.0`` at every
   tau.  Hand-set from the DGP, not read off the estimator.
B. **Quantile fan** (same compliance structure; complier ``Y1 = 2 * Y0`` with
   ``Y0 ~ N(0,1)``).  True ``QTE(tau) = 2*Phi^-1(tau) - Phi^-1(tau) =
   Phi^-1(tau)`` — computed from ``scipy.stats.norm``, not the estimator.
C. **Cross-implementation agreement** with ``sp.beyond_average_late``, which
   reaches the same estimand by an independently written code path.
D. **Analytic-SE calibration** — Monte-Carlo coverage of the influence-
   function CI against the nominal 95%.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

TAUS = np.array([0.25, 0.5, 0.75])

# Compliance structure shared by anchors A-C.
P_ALWAYS, P_COMPLIER = 0.30, 0.50  # never-takers get the remaining 0.20
ALWAYS_Y, NEVER_Y = 10.0, -10.0  # far-tail masses, so mixing is punishing
DELTA = 2.0  # anchor A: constant complier shift


def _late_sample(seed: int, n: int, effect: str) -> pd.DataFrame:
    """One-sided-noncompliance LATE design with a known complier QTE.

    ``effect='shift'`` -> Y1 = Y0 + DELTA      (true QTE flat at DELTA)
    ``effect='scale'`` -> Y1 = 2 * Y0          (true QTE = Phi^-1(tau))
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    kind = np.where(u < P_ALWAYS, "a", np.where(u < P_ALWAYS + P_COMPLIER, "c", "n"))
    z = rng.integers(0, 2, n)
    d = np.where(kind == "a", 1, np.where(kind == "n", 0, z))
    y0 = rng.normal(0.0, 1.0, n)
    y1 = y0 + DELTA if effect == "shift" else 2.0 * y0
    y_complier = np.where(z == 1, y1, y0)
    y = np.where(kind == "a", ALWAYS_Y, np.where(kind == "n", NEVER_Y, y_complier))
    return pd.DataFrame({"y": y, "d": d.astype(int), "z": z})


# ── A. constant complier effect under partial compliance ───────────── #


def test_recovers_constant_complier_qte_under_partial_compliance():
    """True QTE == DELTA at every tau; Delta_p = 0.5, so a mean-Wald
    rescaling would land at 2*DELTA."""
    res = sp.dist_iv(
        _late_sample(7, 200_000, "shift"),
        y="y",
        treat="d",
        instrument="z",
        quantiles=TAUS,
    )
    assert np.all(np.abs(res.late_q - DELTA) < 0.05), res.late_q
    # complier share is Delta_p = P(complier) = 0.50 by construction
    assert abs(res.complier_share - P_COMPLIER) < 0.01


def test_wald_ratio_of_quantiles_regression():
    """Guard the <=1.20.0 defect: the old estimator inflated by 1/Delta_p.

    With Delta_p = 0.5 the old code returned ~2*DELTA = 4.0.  Assert we are
    nowhere near it, so reintroducing the quantile-Wald ratio fails loudly.
    """
    res = sp.dist_iv(
        _late_sample(7, 200_000, "shift"),
        y="y",
        treat="d",
        instrument="z",
        quantiles=TAUS,
    )
    old_biased_value = DELTA / P_COMPLIER  # == 4.0
    assert np.all(np.abs(res.late_q - old_biased_value) > 1.5), (
        "dist_iv reproduced the pre-1.21 quantile-Wald-ratio bias: " f"{res.late_q}"
    )


# ── B. non-degenerate truth: the QTE fan ───────────────────────────── #


def test_recovers_quantile_fan():
    """Complier Y1 = 2*Y0 with Y0 ~ N(0,1) => QTE(tau) = Phi^-1(tau).

    The truth VARIES with tau (negative low, zero at the median, positive
    high).  Any estimator that reports one number per sample -- or that
    reports a mean effect at every quantile -- cannot pass this.
    """
    taus = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    truth = stats.norm.ppf(taus)  # from scipy, not from the estimator
    res = sp.dist_iv(
        _late_sample(11, 200_000, "scale"),
        y="y",
        treat="d",
        instrument="z",
        quantiles=taus,
    )
    assert np.all(
        np.abs(res.late_q - truth) < 0.06
    ), f"estimated {np.round(res.late_q, 3)} vs truth {np.round(truth, 3)}"
    # The fan must actually be a fan: strictly increasing, straddling zero.
    assert np.all(np.diff(res.late_q) > 0)
    assert res.late_q[0] < -0.5 and res.late_q[-1] > 0.5
    assert abs(res.late_q[2]) < 0.06  # median effect is 0


# ── C. cross-implementation agreement ──────────────────────────────── #


def test_agrees_with_beyond_average_late():
    """Two independently written code paths, same estimand, same answer."""
    df = _late_sample(3, 50_000, "shift")
    a = sp.dist_iv(df, y="y", treat="d", instrument="z", quantiles=TAUS, se="none")
    b = sp.beyond_average_late(
        df, y="y", treat="d", instrument="z", quantiles=TAUS, n_boot=2
    )
    assert np.allclose(a.late_q, b.late_q, atol=1e-10), (a.late_q, b.late_q)


# ── D. covariates must bind (was silently ignored pre-1.21) ────────── #


def test_covariates_change_the_estimate_and_remove_bias():
    """Z is randomised only CONDITIONAL on x, so ignoring x is biased.

    True complier effect is 1.5.  Pre-1.21 ``covariates=`` was accepted and
    then dropped on the floor, so both calls returned the same biased number.
    """
    rng = np.random.default_rng(1)
    n = 40_000
    x = rng.normal(size=n)
    pi = 1.0 / (1.0 + np.exp(-0.8 * x))  # P(Z=1|X) depends on x
    z = (rng.random(n) < pi).astype(int)
    d = ((0.2 + 0.6 * z + 0.3 * x + rng.normal(0, 0.3, n)) > 0.5).astype(int)
    y = 1.5 * d + 0.7 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "d": d, "z": z, "x": x})

    naive = sp.dist_iv(df, y="y", treat="d", instrument="z", quantiles=TAUS, se="none")
    adj = sp.dist_iv(
        df,
        y="y",
        treat="d",
        instrument="z",
        covariates=["x"],
        quantiles=TAUS,
        se="none",
    )
    assert not np.allclose(naive.late_q, adj.late_q), "covariates were ignored"
    # Adjusting for x must move the estimate toward the truth.
    assert np.mean(np.abs(adj.late_q - 1.5)) < np.mean(np.abs(naive.late_q - 1.5))


# ── E. analytic influence-function SE calibration ──────────────────── #


def test_analytic_se_coverage():
    """Monte-Carlo coverage of the IF-based CI. Nominal 95%.

    Band [0.90, 0.99] is wide enough for 200 reps (MC s.e. ~1.5pp) but
    still fails a badly-scaled or fabricated SE.
    """
    true_beta, reps = 1.5, 200
    hits = np.zeros(len(TAUS))
    used = 0
    for s in range(reps):
        rng = np.random.default_rng(s)
        n = 3000
        z = rng.integers(0, 2, n)
        d = ((0.2 + 0.6 * z + rng.normal(0, 0.3, n)) > 0.5).astype(int)
        y = true_beta * d + rng.normal(0, 1, n)
        r = sp.dist_iv(
            pd.DataFrame({"y": y, "d": d, "z": z}),
            y="y",
            treat="d",
            instrument="z",
            quantiles=TAUS,
        )
        if not np.isfinite(r.se_q).all():
            continue
        used += 1
        hits += ((r.ci_low <= true_beta) & (true_beta <= r.ci_high)).astype(float)
    assert used > reps * 0.9
    coverage = hits / used
    assert np.all((coverage > 0.90) & (coverage < 0.99)), coverage


# ── F. loud failure, not silent NaN ────────────────────────────────── #


def test_non_binary_instrument_raises():
    df = _late_sample(0, 2000, "shift")
    df["z"] = np.arange(len(df)) % 3
    with pytest.raises(ValueError, match="binary"):
        sp.dist_iv(df, y="y", treat="d", instrument="z", quantiles=TAUS)


def test_zero_first_stage_warns_loudly():
    """An instrument that does not move treatment must be flagged.

    The complier share is only positive here by sampling noise, and every
    downstream quantity divides by it.  Depending on the draw this is either
    a hard error (share <= 0) or a weak-first-stage warning -- never a
    silent, confident-looking estimate.
    """
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "d": rng.integers(0, 2, n),  # independent of z
            "z": rng.integers(0, 2, n),
        }
    )
    try:
        with pytest.warns(UserWarning, match="weak first stage"):
            sp.dist_iv(df, y="y", treat="d", instrument="z", quantiles=TAUS)
    except ValueError as exc:  # share came out <= 0 on this draw
        assert "complier share" in str(exc)


def test_strong_first_stage_does_not_warn():
    """The weak-instrument guard must not fire on a healthy design."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sp.dist_iv(
            _late_sample(2, 20_000, "shift"),
            y="y",
            treat="d",
            instrument="z",
            quantiles=TAUS,
        )


def test_point_estimates_never_silently_nan():
    """Regression guard for the degenerate-median-split bug: seeds where the
    binary instrument has more 1s than 0s previously produced all-NaN."""
    for seed in range(12):
        r = sp.dist_iv(
            _late_sample(seed, 4000, "shift"),
            y="y",
            treat="d",
            instrument="z",
            quantiles=TAUS,
            se="none",
        )
        assert np.isfinite(r.late_q).all(), f"seed {seed}: {r.late_q}"


# ── G. kan_dlate deprecation ───────────────────────────────────────── #


def test_kan_dlate_is_deprecated_and_matches_dist_iv():
    df = _late_sample(5, 20_000, "shift")
    with pytest.warns(DeprecationWarning, match="kan_dlate"):
        k = sp.kan_dlate(df, y="y", treat="d", instrument="z", quantiles=TAUS)
    d = sp.dist_iv(df, y="y", treat="d", instrument="z", quantiles=TAUS)
    assert np.allclose(k.late_q, d.late_q)
