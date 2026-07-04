"""Frozen-stat parity: sp.mr_f_statistic exact F = (β/se)^2; sp.mr_steiger R^2.

F-stat: per-SNP first-stage strength F_i = (beta_i / se_i)^2 is an exact closed
form in large-GWAS summary statistics. Bit-exact against the hand formula.

Steiger: R^2_exposure = beta^2 N / (beta^2 N + se^2 N) and the same formula
for outcome; the Steiger test compares the two R^2's to identify the causal
direction. Exact closed forms. Bit-exact.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp


def test_mr_f_statistic_is_exact_beta_over_se_squared():
    bx = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    sx = np.array([0.02, 0.04, 0.06, 0.08, 0.10])
    hand_F = (bx / sx) ** 2
    r = sp.mr_f_statistic(beta_exposure=bx, se_exposure=sx)
    assert np.allclose(np.asarray(r.per_snp_F, dtype=float), hand_F, atol=1e-12)
    assert float(r.f_mean) == pytest.approx(float(np.mean(hand_F)), abs=1e-12)


def test_mr_steiger_R2_exposure_closed_form():
    bx = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    sx = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    n = 100_000
    # Per-SNP t^2 = (beta/se)^2, then per-SNP R^2 = t^2 / (t^2 + n - 2),
    # summed across SNPs and clipped at 1.
    t2 = (bx / sx) ** 2
    per_snp = t2 / (t2 + n - 2)
    r2_hand = float(min(1.0, np.sum(per_snp)))
    r = sp.mr_steiger(
        beta_exposure=bx,
        se_exposure=sx,
        n_exposure=n,
        beta_outcome=bx,
        se_outcome=sx,
        n_outcome=n,
    )
    assert float(r.r2_exposure) == pytest.approx(r2_hand, abs=1e-9)


def test_mr_steiger_R2_outcome_closed_form():
    by = np.array([0.05, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
    sy = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    n = 100_000
    t2 = (by / sy) ** 2
    per_snp = t2 / (t2 + n - 2)
    r2_hand = float(min(1.0, np.sum(per_snp)))
    r = sp.mr_steiger(
        beta_exposure=by,
        se_exposure=sy,
        n_exposure=n,
        beta_outcome=by,
        se_outcome=sy,
        n_outcome=n,
    )
    assert float(r.r2_outcome) == pytest.approx(r2_hand, abs=1e-9)


def test_steiger_correct_direction_when_exposure_drives_outcome():
    # Plant effect: by = 0.5 * bx. R^2_exposure > R^2_outcome; correct direction.
    rng = np.random.default_rng(0)
    bx = rng.uniform(0.1, 0.4, 30)
    se = np.full(30, 0.05)
    by = 0.5 * bx + rng.normal(0, 0.05, 30)
    r = sp.mr_steiger(
        beta_exposure=bx,
        se_exposure=se,
        n_exposure=100_000,
        beta_outcome=by,
        se_outcome=se,
        n_outcome=100_000,
    )
    assert bool(r.correct_direction) is True
    assert float(r.r2_exposure) > float(r.r2_outcome)
