"""
Partially Linear Regression (PLR) model for DML.

Model: ``Y = theta * D + g(X) + eps``, ``D = m(X) + v``.

Neyman-orthogonal score:
    psi(W; theta, g, m) = (Y - g(X) - theta*(D - m(X))) * (D - m(X))

Closed-form DML1 estimator:
    theta = sum( y_tilde * d_tilde ) / sum( d_tilde * d_tilde )
    y_tilde = Y - g_hat(X),  d_tilde = D - m_hat(X).
"""

import numpy as np

from ._base import _DoubleMLBase


class DoubleMLPLR(_DoubleMLBase):
    """Partially linear regression DML (continuous or binary D, no IV)."""

    _MODEL_TAG = 'PLR'
    _ESTIMAND = 'ATE'
    _REQUIRES_INSTRUMENT = False
    _BINARY_TREATMENT = False  # PLR is agnostic to D type

    def _fit_one_rep(self, Y, D, X, Z, n, rng_seed):
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rng_seed)
        y_resid = np.zeros(n)
        d_resid = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            ml_g = clone(self.ml_g)
            ml_g.fit(X[train_idx], Y[train_idx])
            y_resid[test_idx] = Y[test_idx] - ml_g.predict(X[test_idx])

            ml_m = clone(self.ml_m)
            ml_m.fit(X[train_idx], D[train_idx])
            d_resid[test_idx] = D[test_idx] - ml_m.predict(X[test_idx])

        denom = float(np.sum(d_resid * d_resid))
        if denom < 1e-12:
            raise RuntimeError("PLR denominator ≈ 0; check covariate informativeness.")
        theta = float(np.sum(d_resid * y_resid) / denom)

        psi = y_resid - theta * d_resid
        J = -np.mean(d_resid**2)
        sigma2 = np.mean((d_resid * psi)**2)
        se = float(np.sqrt(sigma2 / (J**2 * n))) if abs(J) > 1e-10 else 0.0
        return theta, se
