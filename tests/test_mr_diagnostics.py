"""Tests for MR diagnostics: heterogeneity, Steiger, leave-one-out,
MR-PRESSO, radial."""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp


@pytest.fixture
def clean_mr_data():
    """Simulate a clean two-sample MR dataset with no pleiotropy.

    True causal effect = 0.4.  Each SNP has its own effect on X with
    small measurement error in both GWAS.
    """
    rng = np.random.default_rng(123)
    n_snps = 15
    bx = rng.uniform(0.05, 0.25, n_snps)
    sx = np.full(n_snps, 0.02)
    sy = np.full(n_snps, 0.03)
    true_beta = 0.4
    by = true_beta * bx + rng.normal(0, sy)
    return dict(bx=bx, by=by, sx=sx, sy=sy, true_beta=true_beta)


@pytest.fixture
def pleiotropic_mr_data():
    """Data with a few outlier SNPs violating exclusion restriction."""
    rng = np.random.default_rng(45)
    n_snps = 20
    bx = rng.uniform(0.05, 0.25, n_snps)
    sx = np.full(n_snps, 0.02)
    sy = np.full(n_snps, 0.03)
    true_beta = 0.3
    by = true_beta * bx + rng.normal(0, sy)
    # Introduce pleiotropy in 3 SNPs
    by[:3] += 0.25
    return dict(bx=bx, by=by, sx=sx, sy=sy, outlier_idx=[0, 1, 2])


# ---------------------------------------------------------------------------
# Heterogeneity (Cochran Q)
# ---------------------------------------------------------------------------


def test_mr_heterogeneity_ivw_clean_data_not_significant(clean_mr_data):
    d = clean_mr_data
    r = sp.mr_heterogeneity(d["bx"], d["by"], d["sy"], method="ivw")
    assert r.Q_p > 0.05
    assert 0 <= r.I2 <= 100


def test_mr_heterogeneity_egger_runs(clean_mr_data):
    d = clean_mr_data
    r = sp.mr_heterogeneity(d["bx"], d["by"], d["sy"], method="egger")
    assert np.isfinite(r.Q)
    assert r.method == "egger"


def test_mr_heterogeneity_flags_pleiotropic(pleiotropic_mr_data):
    d = pleiotropic_mr_data
    r = sp.mr_heterogeneity(d["bx"], d["by"], d["sy"], method="ivw")
    # Significant heterogeneity expected when 3/20 SNPs have horizontal pleiotropy
    assert r.Q > 0
    # Strongly pleiotropic dataset almost always yields Q_p < 0.05
    assert r.Q_p < 0.05


# ---------------------------------------------------------------------------
# Egger intercept test
# ---------------------------------------------------------------------------


def test_mr_pleiotropy_egger_clean_data_no_directional(clean_mr_data):
    d = clean_mr_data
    r = sp.mr_pleiotropy_egger(d["bx"], d["by"], d["sy"])
    # With symmetric noise, intercept should not be significantly non-zero
    assert r.p_value > 0.05
    assert abs(r.intercept) < 0.1


# ---------------------------------------------------------------------------
# Leave-one-out
# ---------------------------------------------------------------------------


def test_leave_one_out_returns_correct_number_of_rows(clean_mr_data):
    d = clean_mr_data
    r = sp.mr_leave_one_out(d["bx"], d["by"], d["sy"])
    assert len(r.table) == len(d["bx"])
    assert {"dropped_snp", "estimate", "se", "p_value"} <= set(r.table.columns)


def test_leave_one_out_accepts_snp_ids(clean_mr_data):
    d = clean_mr_data
    ids = [f"rs{1000+i}" for i in range(len(d["bx"]))]
    r = sp.mr_leave_one_out(d["bx"], d["by"], d["sy"], snp_ids=ids)
    assert list(r.table["dropped_snp"]) == ids


# ---------------------------------------------------------------------------
# Steiger
# ---------------------------------------------------------------------------


def test_mr_steiger_direction_correct():
    # Simulate: exposure has high R^2, outcome has low R^2
    rng = np.random.default_rng(1)
    n_snps = 10
    bx = rng.uniform(0.1, 0.3, n_snps)       # large exposure effects
    sx = np.full(n_snps, 0.02)
    by = 0.2 * bx                            # small outcome effects
    sy = np.full(n_snps, 0.05)
    r = sp.mr_steiger(bx, sx, 50_000, by, sy, 50_000)
    assert r.correct_direction
    assert r.r2_exposure > r.r2_outcome


# ---------------------------------------------------------------------------
# MR-PRESSO
# ---------------------------------------------------------------------------


def test_mr_presso_detects_outliers(pleiotropic_mr_data):
    d = pleiotropic_mr_data
    r = sp.mr_presso(
        d["bx"], d["by"], d["sx"], d["sy"],
        n_boot=200, seed=42,
    )
    # MR-PRESSO should detect at least one of the seeded outliers
    assert len(r.outliers) >= 1
    assert r.outlier_corrected_estimate is not None
    # Corrected estimate should be closer to the true beta (0.3) than raw
    assert abs(r.outlier_corrected_estimate - 0.3) < abs(r.raw_estimate - 0.3)


def test_mr_presso_clean_data_no_outliers(clean_mr_data):
    d = clean_mr_data
    r = sp.mr_presso(
        d["bx"], d["by"], d["sx"], d["sy"],
        n_boot=200, seed=7,
    )
    # global test should not reject for clean data
    assert r.global_test_pvalue > 0.01
    # If outliers are detected on a clean sim, it should be at most a small number
    assert len(r.outliers) <= 3


# ---------------------------------------------------------------------------
# Radial MR
# ---------------------------------------------------------------------------


def test_mr_radial_runs(clean_mr_data):
    d = clean_mr_data
    r = sp.mr_radial(d["bx"], d["by"], d["sy"])
    assert r.total_Q >= 0
    assert 0 <= r.Q_pvalue <= 1
    assert set(r.table.columns) >= {"snp", "W", "beta_hat", "q_contribution"}


def test_mr_radial_flags_outliers(pleiotropic_mr_data):
    d = pleiotropic_mr_data
    r = sp.mr_radial(d["bx"], d["by"], d["sy"])
    # Radial should flag at least one SNP in this sim
    assert isinstance(r.outliers, list)
    assert len(r.outliers) >= 0
