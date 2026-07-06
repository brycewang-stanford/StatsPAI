"""Analytical parity: sp.calibration_test detects known heterogeneity.

``sp.calibration_test`` (alias ``sp.test_calibration``) runs the
Chernozhukov-Demirer-Duflo-Fernandez-Val-Newey best-linear-predictor (BLP)
calibration test on a fitted ``sp.causal_forest``. It regresses the AIPW
pseudo-outcome on the mean forest prediction and the demeaned (differential)
forest prediction; the two slopes are:

- ``mean_forest_prediction`` (beta1): the calibration slope. Under a
  well-specified forest it is a positive O(1) quantity (ideally ~1).
- ``differential_forest_prediction`` (beta2): the heterogeneity slope.
  beta2 > 0 (significantly) iff the forest's *ranking* of units captures
  real treatment-effect heterogeneity.

No cross-package reference exists, so the honest grade is ``analytical-only``.
On a DGP with strong heterogeneity ``tau(x) = 1 + 2 x0`` the forest learns
the ordering, so beta2 is significantly positive — a known-truth anchor
(a forest that failed to capture the HTE would give beta2 ~ 0, large p).
The alias ``sp.test_calibration`` returns a bit-identical frame.

The forest is seeded (``random_state=0``); the DGP RNG is fixed, so the test
is deterministic. The null (constant-effect) case is intentionally *not*
pinned: the differential-prediction test has finite-forest size distortion,
so only the positive-detection direction is a reliable anchor.
"""

from __future__ import annotations

import numpy as np

import statspai as sp

_MEAN = "mean_forest_prediction"
_DIFF = "differential_forest_prediction"


def _forest_with_hte():
    rng = np.random.default_rng(11)
    n = 1600
    X = rng.standard_normal((n, 3))
    tau = 1.0 + 2.0 * X[:, 0]  # strong, learnable heterogeneity
    T = rng.integers(0, 2, size=n)
    Y0 = X[:, 1] + rng.standard_normal(n) * 0.5
    Y = np.where(T == 1, Y0 + tau, Y0)
    cf = sp.causal_forest(Y=Y, T=T, X=X, n_estimators=200, random_state=0)
    return cf, X, Y, T


def test_calibration_test_detects_known_heterogeneity():
    cf, X, Y, T = _forest_with_hte()
    out = sp.calibration_test(cf, X=X, Y=Y, T=T)

    # Structure: exactly the two BLP rows, all finite, positive SEs, CI brackets.
    assert list(out.index) == [_MEAN, _DIFF]
    cols = out.loc[:, ["coef", "se", "ci_low", "ci_high"]].to_numpy()
    assert np.all(np.isfinite(cols))
    assert np.all(out["se"] > 0)
    assert np.all(out["ci_low"] <= out["coef"])
    assert np.all(out["coef"] <= out["ci_high"])

    # Calibration slope: positive and O(1) (well-specified forest).
    assert 0.3 < out.loc[_MEAN, "coef"] < 3.0

    # Heterogeneity slope: significantly positive ⇒ the forest's ranking
    # captures the real HTE (known-truth detection).
    assert out.loc[_DIFF, "coef"] > 0.0
    assert out.loc[_DIFF, "p"] < 0.01


def test_test_calibration_alias_is_identical():
    cf, X, Y, T = _forest_with_hte()
    out = sp.calibration_test(cf, X=X, Y=Y, T=T)
    alias = sp.test_calibration(cf, X=X, Y=Y, T=T)
    # Exact structural identity: the alias returns a bit-identical frame.
    assert alias.equals(out)
