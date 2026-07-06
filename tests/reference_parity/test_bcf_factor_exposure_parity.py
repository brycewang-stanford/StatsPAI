"""Analytical parity: sp.bcf_factor_exposure identities + effect recovery.

``sp.bcf_factor_exposure`` projects a high-dimensional exposure vector onto
factors, runs a Bayesian Causal Forest (BCF) per binarised factor, and sums
the per-factor ATEs into a total mixture ATE. It has no cross-package
reference, so the honest grade is ``analytical-only``.

Two anchors:

1. **Exact adding-up identities.** By construction the total mixture ATE is
   the sum of the per-factor ATEs and the total variance is the sum of the
   per-factor variances (local-independence aggregation). Both hold to
   machine precision.

2. **Known-effect recovery.** With supplied one-hot loadings that select two
   exposures carrying *known* binary treatment effects (2.0 and 1.0), the
   per-factor ATEs and their total recover the planted effects within a band
   (BCF is a regularised tree ensemble, so the band is a few tenths, not
   machine precision).

The design is deterministic (fixed RNG + ``random_state``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import statspai as sp


def _make_frame():
    rng = np.random.default_rng(0)
    n, p = 800, 6
    Z = rng.normal(size=(n, p))
    x0 = rng.normal(size=n)  # pre-exposure confounder
    # Median-split treatments on the first two exposures.
    F0 = Z[:, 0] - Z[:, 0].mean()
    F1 = Z[:, 1] - Z[:, 1].mean()
    T0 = (F0 > np.median(F0)).astype(int)
    T1 = (F1 > np.median(F1)).astype(int)
    d0, d1 = 2.0, 1.0
    Y = 1.0 + d0 * T0 + d1 * T1 + 0.5 * x0 + rng.normal(0, 0.5, n)
    cols = {f"z{j}": Z[:, j] for j in range(p)}
    cols.update({"Y": Y, "x0": x0})
    df = pd.DataFrame(cols)
    # One-hot loadings: factor F1 = z0, factor F2 = z1.
    L = np.zeros((p, 2))
    L[0, 0] = 1.0
    L[1, 1] = 1.0
    loadings = pd.DataFrame(
        L, index=[f"z{j}" for j in range(p)], columns=["F1", "F2"]
    )
    return df, loadings, (d0, d1)


def _fit():
    df, loadings, truth = _make_frame()
    res = sp.bcf_factor_exposure(
        df,
        y="Y",
        exposures=[f"z{j}" for j in range(6)],
        covariates=["x0"],
        loadings=loadings,
        n_bootstrap=40,
        random_state=0,
    )
    return res, truth


def test_bcf_factor_exposure_adding_up_identities():
    res, _ = _fit()
    # Total mixture ATE == sum of per-factor ATEs, exactly.
    assert abs(
        res.total_mixture_ate - float(res.per_factor_ate["ate"].sum())
    ) <= 1e-9
    # Total SE == sqrt(sum of per-factor variances), exactly.
    agg_se = float(np.sqrt((res.per_factor_ate["se"] ** 2).sum()))
    assert abs(res.total_mixture_se - agg_se) <= 1e-9


def test_bcf_factor_exposure_recovers_planted_effects():
    res, (d0, d1) = _fit()
    ate = res.per_factor_ate["ate"]
    # Each factor recovers its planted binary effect within a BCF band.
    assert abs(float(ate.loc["F1"]) - d0) <= 0.3
    assert abs(float(ate.loc["F2"]) - d1) <= 0.3
    # Total mixture ATE recovers the sum of planted effects.
    assert abs(res.total_mixture_ate - (d0 + d1)) <= 0.4
