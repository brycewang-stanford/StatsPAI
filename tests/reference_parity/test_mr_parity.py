"""Mendelian Randomization reference-parity tests.

Because the canonical R packages (``TwoSampleMR``, ``MendelianRandomization``)
are not available in CI, this module builds a known-truth simulation and
checks the StatsPAI MR suite against both (a) the analytic truth and
(b) cross-estimator agreement properties that are guaranteed by theory
(Bowden et al. 2015, 2016; Verbanck et al. 2018; Hartwig et al. 2017).

The MR functions here return dictionaries of the form
``{'estimate': ..., 'se': ..., 'ci_lower': ..., 'ci_upper': ...}`` for
simple estimators, and dataclass-style objects for more specialised
methods (:class:`MRPressoResult`, :class:`RadialResult`,
:class:`LeaveOneOutResult`).  The tests handle both shapes.

Targets validated:

1. **IVW recovers the truth** under balanced pleiotropy.
2. **Egger intercept ≈ 0** under balanced pleiotropy.
3. **Egger intercept flags directional pleiotropy**.
4. **Weighted median is robust** to a large minority of invalid IVs.
5. **MR-PRESSO detects a contaminated SNP**.
6. **Leave-one-out** never swings > 5 SE from full-sample IVW on clean data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------


def _simulate_mr_twosample(
    n_snps: int = 30,
    true_beta: float = 0.30,
    pleiotropy_mean: float = 0.0,
    pleiotropy_sd: float = 0.005,
    n_exposure: int = 20_000,
    n_outcome: int = 20_000,
    seed: int = 0,
) -> tuple:
    """Return (bx, by, se_x, se_y) ndarrays for two-sample MR."""
    rng = np.random.default_rng(seed)
    alpha = rng.normal(0.1, 0.03, n_snps)
    se_x = np.full(n_snps, 1.0 / np.sqrt(n_exposure))
    se_y = np.full(n_snps, np.sqrt(2.0 / n_outcome))
    bx = rng.normal(alpha, se_x)
    pleio = rng.normal(pleiotropy_mean, pleiotropy_sd, n_snps)
    by = true_beta * alpha + pleio + rng.normal(0.0, se_y, n_snps)
    return bx, by, se_x, se_y


def _estimate(r: Any) -> float:
    """Extract a point estimate from an MR result (dict or dataclass)."""
    if isinstance(r, dict):
        return float(r["estimate"])
    return float(getattr(r, "estimate"))


def _se(r: Any) -> float:
    if isinstance(r, dict):
        return float(r["se"])
    return float(getattr(r, "se"))


# ---------------------------------------------------------------------------
# IVW + Egger
# ---------------------------------------------------------------------------


class TestIVWBalancedPleiotropy:

    def test_ivw_recovers_truth(self):
        bx, by, se_x, se_y = _simulate_mr_twosample(seed=0)
        r = sp.mr.mr_ivw(bx, by, se_x, se_y)
        truth = 0.30
        est = _estimate(r); se = _se(r)
        assert abs(est - truth) <= 4 * se, (
            f"IVW {est:.3f} vs truth {truth} (SE {se:.3f})"
        )

    def test_egger_intercept_near_zero_when_balanced(self):
        bx, by, se_x, se_y = _simulate_mr_twosample(seed=0)
        r = sp.mr.mr_egger(bx, by, se_x, se_y)
        intercept = r["intercept"]
        int_se = r["intercept_se"]
        assert abs(intercept) <= 3 * int_se, (
            f"Egger intercept {intercept:.4f} exceeds 3*SE ({int_se:.4f})"
        )


class TestMREggerDirectionalPleiotropy:

    def test_egger_detects_shift(self):
        bx, by, se_x, se_y = _simulate_mr_twosample(
            seed=1, pleiotropy_mean=0.04, pleiotropy_sd=0.003,
        )
        r = sp.mr.mr_egger(bx, by, se_x, se_y)
        intercept = r["intercept"]
        assert intercept > 0.01, (
            f"Egger intercept failed to detect +0.04 pleiotropic shift: {intercept}"
        )
        assert r["intercept_p"] < 0.05


# ---------------------------------------------------------------------------
# Robust estimators
# ---------------------------------------------------------------------------


class TestWeightedMedianRobust:

    def test_median_robust_to_outliers(self):
        bx, by, se_x, se_y = _simulate_mr_twosample(seed=2)
        n_bad = int(0.3 * len(bx))
        by = by.copy()
        by[:n_bad] += 0.20
        r = sp.mr.mr_median(bx, by, se_x, se_y, n_boot=200, seed=0)
        assert abs(r["estimate"] - 0.30) < 0.15, (
            f"Median broke on 30% outliers: {r['estimate']:.3f}"
        )


class TestMRPressoOutlierFlag:

    def test_presso_flags_outlier(self):
        bx, by, se_x, se_y = _simulate_mr_twosample(seed=4)
        by = by.copy()
        by[0] += 0.5
        r = sp.mr.mr_presso(bx, by, se_x, se_y, n_boot=100, seed=0)
        outliers = list(getattr(r, "outliers", []) or [])
        # Either SNP 0 is flagged, or the corrected estimate moves closer to truth.
        corrected = float(getattr(r, "outlier_corrected_estimate", r.raw_estimate))
        assert (0 in outliers) or abs(corrected - 0.30) < abs(r.raw_estimate - 0.30), (
            f"PRESSO outliers={outliers}, raw={r.raw_estimate:.3f}, "
            f"corrected={corrected:.3f}"
        )


# ---------------------------------------------------------------------------
# Leave-one-out
# ---------------------------------------------------------------------------


class TestLeaveOneOut:

    def test_loo_stable(self):
        bx, by, se_x, se_y = _simulate_mr_twosample(seed=5, pleiotropy_sd=0.002)
        ivw_full = sp.mr.mr_ivw(bx, by, se_x, se_y)
        loo = sp.mr.mr_leave_one_out(bx, by, se_y)
        table = getattr(loo, "table", None)
        if table is None:
            table = getattr(loo, "results", None)
        if table is None and isinstance(loo, dict):
            table = loo.get("table", None)
        if table is None and hasattr(loo, "__dict__"):
            # Pull first DataFrame attribute
            for v in loo.__dict__.values():
                if isinstance(v, pd.DataFrame):
                    table = v
                    break
        assert isinstance(table, pd.DataFrame)
        est_cols = [c for c in table.columns
                    if c.lower() in ("estimate", "b", "beta", "ivw", "ivw_estimate")]
        assert est_cols, f"LOO table has no estimate column: {table.columns.tolist()}"
        vals = table[est_cols[0]].to_numpy(dtype=float)
        max_dev = float(np.max(np.abs(vals - ivw_full["estimate"])))
        assert max_dev < 5 * ivw_full["se"], (
            f"LOO swung by {max_dev:.4f} > 5*IVW SE ({ivw_full['se']:.4f})"
        )


# ---------------------------------------------------------------------------
# MR-Radial cross-check: beta_hat per SNP ≈ ratio estimator
# ---------------------------------------------------------------------------


class TestMRRadialBetaSNP:

    def test_radial_beta_matches_ratio(self):
        bx, by, se_x, se_y = _simulate_mr_twosample(seed=9)
        r = sp.mr.mr_radial(bx, by, se_y)
        table = getattr(r, "table", None)
        if table is None:
            table = getattr(r, "results", None)
        if table is None and hasattr(r, "__dict__"):
            for v in r.__dict__.values():
                if isinstance(v, pd.DataFrame):
                    table = v
                    break
        assert isinstance(table, pd.DataFrame)
        assert "beta_hat" in table.columns
        # Radial's per-SNP beta_hat should equal by/bx up to rescaling
        # (Wald ratios), except where bx ≈ 0.
        wald = by / bx
        rho = np.corrcoef(wald, table["beta_hat"].to_numpy(dtype=float))[0, 1]
        assert rho > 0.9, (
            f"Radial beta_hat correlation with Wald ratios = {rho:.3f}"
        )
