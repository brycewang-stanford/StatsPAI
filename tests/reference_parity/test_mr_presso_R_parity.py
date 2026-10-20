"""sp.mr_presso against R MRPRESSO 1.0, run as a black box.

Reference: ``_fixtures/mr_presso_R.json`` written by
``_fixtures/_generate_mr_presso_R.R`` (MRPRESSO 1.0, GPL-3; only its exported
``mr_presso()`` is called) on three scenarios: the committed 28-variant LDL-C
-> CHD data (``mr_ldl.csv``) and 15 simulated variants with and without two
gross pleiotropic outliers (``mr_presso_sim.csv``). Each scenario was run
under seeds 1-6 with ``NbDistribution = 1000``.

Evidence levels (CLAUDE.md section 5.1):

* T2, rtol 1e-10: everything MR-PRESSO computes deterministically -- the raw
  and outlier-corrected IVW fits (estimate, SE, t-based p), the observed RSS
  (global and per variant), the distortion coefficient, and the outlier set,
  which is the same in all six reference runs because every flagged variant
  has a simulated p of 0.
* S (stochastic screen, reported but not a parity claim): the global and
  outlier p-values are Monte Carlo on both sides. Their means over six runs
  of 1,000 draws are compared with a tolerance of four Monte Carlo standard
  errors of the difference, ``4 sqrt(2 q (1 - q) / 6000)`` for a share ``q``.
* Not compared: the distortion-test p-value. Verbanck et al. (2018) describe
  its null only in words, and the StatsPAI reading (resample the outliers
  from the non-outliers; see ``docs/dev/cleanroom/mr_presso.md``) gives
  about 0.60 (LDL) and 0.80 (simulated) where MRPRESSO gives 0.52 and 0.93.
  The gap is recorded there as unexplained; the previous StatsPAI
  implementation did not reproduce MRPRESSO here either.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.mendelian.diagnostics import _presso_rss

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "mr_presso_R.json").read_text(encoding="utf-8"))
_LDL = pd.read_csv(_FIX / "mr_ldl.csv")
_SIM = pd.read_csv(_FIX / "mr_presso_sim.csv")
DATA = {
    "ldl": tuple(_LDL[c].to_numpy() for c in ("bx", "by", "bxse", "byse")),
    "sim_out": tuple(_SIM[c].to_numpy() for c in ("bx", "by_out", "bxse", "byse")),
    "sim_clean": tuple(_SIM[c].to_numpy() for c in ("bx", "by_clean", "bxse", "byse")),
}
SEEDS = R["provenance"]["seeds"]
B = R["provenance"]["NbDistribution"]
TIGHT = 1e-10


def _close(ours, ref, rtol=TIGHT):
    np.testing.assert_allclose(float(ours), float(ref), rtol=rtol, atol=0.0)


@pytest.fixture(scope="module")
def fits():
    return {
        nm: [sp.mr_presso(*DATA[nm], n_boot=B, seed=s) for s in SEEDS] for nm in DATA
    }


@pytest.mark.parametrize("scenario", ["ldl", "sim_out", "sim_clean"])
def test_deterministic_outputs_match(fits, scenario):
    ref = R[scenario][0]
    r = fits[scenario][0]
    _close(r.raw_estimate, ref["raw"]["est"])
    _close(r.raw_se, ref["raw"]["sd"])
    _close(r.raw_p, ref["raw"]["p"])
    _close(r.global_test_rss_obs, ref["rss_obs"])
    if ref.get("outliers") is None:
        # Global test does not reject: MRPRESSO runs no outlier test and
        # reports no corrected estimate; neither does StatsPAI.
        assert all(f.outlier_pvalues is None for f in fits[scenario])
        assert all(f.outlier_corrected_estimate is None for f in fits[scenario])
        return
    for run in R[scenario]:
        assert run["outliers"] == ref["outliers"]
    for f in fits[scenario]:
        assert [i + 1 for i in f.outliers] == ref["outliers"]
    _close(r.outlier_corrected_estimate, ref["cor"]["est"])
    _close(r.outlier_corrected_se, ref["cor"]["sd"])
    _close(r.outlier_corrected_p, ref["cor"]["p"])
    _close(r.distortion_coefficient, ref["distortion_coef"])


@pytest.mark.parametrize("scenario", ["ldl", "sim_out"])
def test_per_variant_observed_rss_matches(scenario):
    """MRPRESSO's outlier table lists (by_j - b_{-j} bx_j)^2 per variant."""
    bx, by, _, sy = DATA[scenario]
    total, resid2 = _presso_rss(bx, by, sy)
    np.testing.assert_allclose(resid2, R[scenario][0]["rss_obs_j"], rtol=TIGHT)
    _close(total, R[scenario][0]["rss_obs"])


def _mc_tol(q: float, n: int) -> float:
    q = min(max(q, 1.0 / n), 1 - 1.0 / n)
    return 4.0 * np.sqrt(2.0 * q * (1.0 - q) / n)


@pytest.mark.parametrize("scenario", ["ldl", "sim_out", "sim_clean"])
def test_global_pvalue_stochastic_screen(fits, scenario):
    """S level: mean of six k/B p-values on each side, MC tolerance."""
    n = len(SEEDS) * B
    ref = np.mean([run["global_p"] for run in R[scenario]])
    ours = np.mean([f.global_test_pvalue for f in fits[scenario]])
    assert abs(ours - ref) <= _mc_tol(ref, n)


@pytest.mark.parametrize("scenario", ["ldl", "sim_out"])
def test_outlier_pvalues_stochastic_screen(fits, scenario):
    """S level on the per-variant exceedance shares (adjusted p / J).

    Variants whose Bonferroni-adjusted p was capped at 1 in any run on
    either side carry no share information and are skipped.
    """
    n = len(SEEDS) * B
    n_var = len(DATA[scenario][0])
    ref_p = np.array([run["outlier_p"] for run in R[scenario]])
    our_p = np.array([f.outlier_pvalues for f in fits[scenario]])
    usable = (ref_p < 1).all(axis=0) & (our_p < 1).all(axis=0)
    assert usable.sum() >= 2
    for j in np.flatnonzero(usable):
        q_ref = ref_p[:, j].mean() / n_var
        q_our = our_p[:, j].mean() / n_var
        assert abs(q_our - q_ref) <= _mc_tol(q_ref, n), j


def test_pvalues_are_k_over_b(fits):
    for f in fits["ldl"]:
        for p in [f.global_test_pvalue, f.distortion_p]:
            k = p * B
            assert abs(k - round(k)) < 1e-9
