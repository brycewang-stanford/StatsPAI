"""
Partially Linear IV (PLIV) model for DML.

Model: ``Y = theta * D + g(X) + eps``, ``E[eps | Z, X] = 0``.

Neyman-orthogonal score:
    psi = (Y - g(X) - theta*(D - m(X))) * (Z - r(X))

DML1 ratio estimator:
    theta = sum(y_tilde * z_tilde) / sum(d_tilde * z_tilde)

SE via influence-function variance of the ratio.
"""

import numpy as np

from ._base import _DoubleMLBase


class DoubleMLPLIV(_DoubleMLBase):
    """Partially linear IV DML — endogenous D with continuous/binary Z."""

    _MODEL_TAG = 'PLIV'
    _ESTIMAND = 'LATE'
    _REQUIRES_INSTRUMENT = True
    _BINARY_TREATMENT = False
    _BINARY_INSTRUMENT = False

    def _fit_one_rep(self, Y, D, X, Z, n, rng_seed):
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rng_seed)
        y_resid = np.zeros(n)
        d_resid = np.zeros(n)
        z_resid = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            ml_g = clone(self.ml_g)
            ml_g.fit(X[train_idx], Y[train_idx])
            y_resid[test_idx] = Y[test_idx] - ml_g.predict(X[test_idx])

            ml_m = clone(self.ml_m)
            ml_m.fit(X[train_idx], D[train_idx])
            d_resid[test_idx] = D[test_idx] - ml_m.predict(X[test_idx])

            ml_r = clone(self.ml_r)
            ml_r.fit(X[train_idx], Z[train_idx])
            z_resid[test_idx] = Z[test_idx] - ml_r.predict(X[test_idx])

        denom = float(np.sum(z_resid * d_resid))
        scale = float(np.sqrt(np.sum(z_resid**2) * np.sum(d_resid**2)))
        if abs(denom) < 1e-6 * max(scale, 1.0):
            raise RuntimeError(
                f"Degenerate PLIV first stage: residualized instrument is "
                f"(near-)orthogonal to residualized treatment. "
                f"|E[z̃·d̃]| = {abs(denom):.2e}, scale = {scale:.2e}. "
                f"Instrument likely weak or irrelevant conditional on X."
            )
        theta = float(np.sum(z_resid * y_resid) / denom)

        psi = (y_resid - theta * d_resid) * z_resid
        J = -np.mean(z_resid * d_resid)
        sigma2 = np.mean(psi**2)
        se = float(np.sqrt(sigma2 / (J**2 * n))) if abs(J) > 1e-10 else 0.0
        return theta, se
