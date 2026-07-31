"""Inference validity for ``sp.distributional_te`` (WP-4).

What this file guards
---------------------
Up to and including 1.20.0 the "KS p-value" was

    ks_pvalue = mean(boot_ks >= ks_stat)

where ``boot_ks[b] = sup_y |DTE_b(y)|`` -- the bootstrap statistic
**uncentred**.  That distribution is centred on the estimate, not on the
null, so it is stochastically larger than the observed statistic whatever
the truth.  Measured on a pure no-effect DGP (40 seeds, n = 400,
n_boot = 200) the reported p-value **never fell below 0.565** and rejected
at the 5% level **0.0%** of the time.  It was not a p-value; it could not
reject at any conventional level.

The fix recentres the bootstrap: under H0 the sampling distribution of
``sup|DTE_hat|`` is approximated by that of ``sup|DTE_b - DTE_hat|``.  A
Cramer-von Mises companion (integrated squared deviation) was added because
a sup-statistic is insensitive to broad, shallow shifts.

A one-sided fix is easy to fake -- a test that never rejects has perfect
size and zero power, and a test that always rejects has perfect power and
no size control.  **Every test here therefore asserts BOTH directions.**

Anchors
-------
A. **Size**: on a no-effect DGP the rejection rate sits near the nominal
   level and the p-values are uniform (a KS test against U(0,1)).
B. **Power**: on a location shift the tests reject essentially always.
C. **Regression guard**: p-values must reach below 0.5 on null data, which
   the old statistic provably never did.
D. **Quantile inversion** no longer snaps to grid nodes.
E. **Distribution regression** produces monotone conditional CDFs in [0,1].
F. Bootstrap failures are recorded via ``record_degradation``, not
   swallowed.
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

# Keep these modest: each rep runs a full bootstrap.
REPS, N, N_BOOT = 120, 400, 200


def _pvalues(effect: float, reps: int = REPS):
    ks, cvm = [], []
    for s in range(reps):
        rng = np.random.default_rng(s)
        d = rng.integers(0, 2, N)
        y = effect * d + rng.normal(0, 1, N)
        r = sp.distributional_te(
            pd.DataFrame({"y": y, "d": d}),
            y="y",
            treatment="d",
            method="ipw",
            quantiles=[0.5],
            n_boot=N_BOOT,
            seed=s,
        )
        ks.append(r.ks_pvalue)
        cvm.append(r.cvm_pvalue)
    return np.asarray(ks), np.asarray(cvm)


# ── A. size under the null ─────────────────────────────────────────── #


def test_size_under_null():
    """No treatment effect at all: rejection rate must sit near 5%."""
    ks, cvm = _pvalues(0.0)
    ks_rate = float((ks < 0.05).mean())
    cvm_rate = float((cvm < 0.05).mean())
    # Band allows Monte-Carlo error at REPS reps (s.e. ~2pp) while still
    # failing both a zero-power test (the old behaviour) and a test that
    # over-rejects.
    assert 0.005 < ks_rate < 0.15, f"KS size {ks_rate}"
    assert 0.005 < cvm_rate < 0.15, f"CvM size {cvm_rate}"


def test_null_pvalues_are_uniform():
    """A valid p-value is U(0,1) under H0. The old statistic was not."""
    ks, _ = _pvalues(0.0)
    assert stats.kstest(ks, "uniform").pvalue > 0.01, (
        f"null p-values not uniform: {np.sort(ks)[:10]}"
    )


# ── B. power ───────────────────────────────────────────────────────── #


def test_power_against_location_shift():
    """delta = 0.5 on unit-variance outcomes: must reject nearly always.

    Pairs with the size test: together they rule out both degenerate fixes
    (never reject / always reject).
    """
    ks, cvm = _pvalues(0.5)
    assert float((ks < 0.05).mean()) > 0.8, f"KS power {(ks < 0.05).mean()}"
    assert float((cvm < 0.05).mean()) > 0.8, f"CvM power {(cvm < 0.05).mean()}"


# ── C. explicit regression guard on the old defect ─────────────────── #


def test_null_pvalues_reach_below_half():
    """The pre-1.21 statistic never produced a p-value below 0.565.

    On null data a valid p-value must frequently land in the lower half of
    [0, 1]. This is the cheapest possible detector for a reversion to the
    uncentred bootstrap.
    """
    ks, _ = _pvalues(0.0, reps=40)
    assert ks.min() < 0.5, f"minimum null p-value was {ks.min():.4f}"
    assert 0.3 < float((ks < 0.5).mean()) < 0.7, (
        f"fraction below 0.5 was {(ks < 0.5).mean():.3f}, expected ~0.5"
    )


# ── D. quantile inversion is no longer grid-snapped ────────────────── #


def test_quantiles_not_snapped_to_grid():
    """With interpolation the estimate must not coincide with a grid node.

    Pre-1.21 ``_quantile_from_cdf`` returned ``grid[idx]``, so every quantile
    was exactly one of ``n_grid`` values and carried a discretisation error
    that shrank only with ``n_grid``, never with n.
    """
    from statspai.qte.distributional import _quantile_from_cdf

    grid = np.linspace(0.0, 10.0, 11)  # coarse on purpose
    cdf = grid / 10.0  # exact uniform CDF
    taus = np.array([0.25, 0.55, 0.77])
    q = _quantile_from_cdf(grid, cdf, taus)
    # True quantiles of U(0,10) are 2.5, 5.5, 7.7 -- none is a grid node.
    np.testing.assert_allclose(q, [2.5, 5.5, 7.7], atol=1e-9)
    assert not np.any(np.isin(q, grid))


def test_quantile_accuracy_improves_with_n_not_grid():
    """A coarse grid must not cap accuracy once interpolation is in place."""
    rng = np.random.default_rng(3)
    n = 40_000
    d = rng.integers(0, 2, n)
    y = 2.0 * d + rng.normal(0, 1, n)
    res = sp.distributional_te(
        pd.DataFrame({"y": y, "d": d}),
        y="y",
        treatment="d",
        method="ipw",
        quantiles=[0.25, 0.5, 0.75],
        n_grid=40,  # deliberately coarse: cells are ~0.25 wide
        n_boot=20,
        seed=0,
    )
    # Grid spacing alone would allow ~0.25 of error; interpolation beats it.
    assert np.all(np.abs(res.qte_effects - 2.0) < 0.12), res.qte_effects


# ── E. distribution regression replaces the linear probability model ── #


def test_conditional_cdf_is_monotone_and_bounded():
    """``_fit_cond_cdf_ctrl`` must return a genuine CDF in y for each unit.

    The old LinearRegression version was a linear probability model: not
    bounded (it was clipped afterwards) and not monotone in y.
    """
    from statspai.qte.distributional import _fit_cond_cdf_ctrl

    rng = np.random.default_rng(7)
    n = 600
    X = rng.normal(size=(n, 2))
    Y = X[:, 0] * 2.0 + rng.normal(size=n)
    grid = np.linspace(Y.min(), Y.max(), 25)
    mu = _fit_cond_cdf_ctrl(X, Y, X, grid)

    assert mu.shape == (n, len(grid))
    assert np.all(mu >= 0.0) and np.all(mu <= 1.0)
    # Monotone non-decreasing in y for every observation.
    assert np.all(np.diff(mu, axis=1) >= -1e-12)


def test_dr_method_recovers_known_shift():
    rng = np.random.default_rng(11)
    n = 8000
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-0.8 * x))
    d = (rng.random(n) < p).astype(int)
    y = 1.5 * d + 0.7 * x + rng.normal(0, 1, n)
    res = sp.distributional_te(
        pd.DataFrame({"y": y, "d": d, "x": x}),
        y="y",
        treatment="d",
        x=["x"],
        method="dr",
        quantiles=[0.25, 0.5, 0.75],
        n_boot=20,
        seed=0,
    )
    assert np.all(np.abs(res.qte_effects - 1.5) < 0.3), res.qte_effects


# ── F. degradation is recorded, not swallowed ──────────────────────── #


def test_result_exposes_degradation_fields():
    rng = np.random.default_rng(2)
    n = 400
    d = rng.integers(0, 2, n)
    y = rng.normal(size=n)
    res = sp.distributional_te(
        pd.DataFrame({"y": y, "d": d}),
        y="y",
        treatment="d",
        method="ipw",
        quantiles=[0.5],
        n_boot=30,
        seed=0,
    )
    # Contract: these exist whether or not anything degraded.
    assert hasattr(res, "degradations")
    assert res.n_boot_failed == 0
    assert np.isfinite(res.cvm_stat)


def test_cvm_reported_in_summary():
    rng = np.random.default_rng(4)
    n = 300
    d = rng.integers(0, 2, n)
    y = 1.0 * d + rng.normal(size=n)
    res = sp.distributional_te(
        pd.DataFrame({"y": y, "d": d}),
        y="y",
        treatment="d",
        method="ipw",
        quantiles=[0.5],
        n_boot=30,
        seed=0,
    )
    out = res.summary()
    assert "CvM" in out and "KS" in out


@pytest.mark.parametrize("method", ["ipw", "dr", "cic"])
def test_all_methods_produce_valid_pvalues(method):
    rng = np.random.default_rng(5)
    n = 800
    if method == "cic":
        # CiC wants the 4-cell encoding 0..3.
        g = rng.integers(0, 4, n)
        y = rng.normal(size=n) + (g == 3) * 1.0
        df = pd.DataFrame({"y": y, "d": g})
        kw = {}
    else:
        x = rng.normal(size=n)
        d = (rng.random(n) < 1 / (1 + np.exp(-0.5 * x))).astype(int)
        y = 1.0 * d + 0.5 * x + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "x": x})
        kw = {"x": ["x"]}
    res = sp.distributional_te(
        df, y="y", treatment="d", method=method, quantiles=[0.5],
        n_boot=40, seed=0, **kw
    )
    assert 0.0 <= res.ks_pvalue <= 1.0
    assert 0.0 <= res.cvm_pvalue <= 1.0
