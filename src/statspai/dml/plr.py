"""
Partially Linear Regression (PLR) model for DML.

Model: ``Y = theta * D + g(X) + eps``, ``D = m(X) + v``.

Neyman-orthogonal score:
    psi(W; theta, g, m) = (Y - g(X) - theta*(D - m(X))) * (D - m(X))

Closed-form DML1 estimator (unweighted):
    theta = sum( y_tilde * d_tilde ) / sum( d_tilde * d_tilde )
    y_tilde = Y - g_hat(X),  d_tilde = D - m_hat(X).

Weighted variant (with sample_weight w_i):
    theta = sum( w * y_tilde * d_tilde ) / sum( w * d_tilde² )
    Var(theta) = sum( w² * psi_score² ) / ( sum(w * d_tilde²) )²
where psi_score_i = (y_tilde_i - theta * d_tilde_i) * d_tilde_i.
"""

import numpy as np

from ._base import _DoubleMLBase


class DoubleMLPLR(_DoubleMLBase):
    """Partially linear regression DML (continuous or binary D, no IV)."""

    _MODEL_TAG = 'PLR'
    _ESTIMAND = 'ATE'
    _REQUIRES_INSTRUMENT = False
    _ML_M_TARGET_BINARY = False  # PLR is agnostic to D type
    _SUPPORTS_SAMPLE_WEIGHT = True

    def _fit_one_rep(self, Y, D, X, Z, n, rng_seed, sample_weight=None):
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rng_seed)
        y_resid = np.zeros(n)
        d_resid = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            w_train = (
                sample_weight[train_idx] if sample_weight is not None else None
            )
            ml_g = self._fit_weighted(self.ml_g, X[train_idx], Y[train_idx], w_train)
            y_resid[test_idx] = Y[test_idx] - ml_g.predict(X[test_idx])

            ml_m = self._fit_weighted(self.ml_m, X[train_idx], D[train_idx], w_train)
            d_resid[test_idx] = D[test_idx] - ml_m.predict(X[test_idx])

        if sample_weight is None:
            denom = float(np.sum(d_resid * d_resid))
            if denom < 1e-12:
                raise RuntimeError(
                    "PLR denominator ≈ 0; check covariate informativeness."
                )
            theta = float(np.sum(d_resid * y_resid) / denom)
            psi_inner = y_resid - theta * d_resid
            psi_score = psi_inner * d_resid
            J = -np.mean(d_resid ** 2)
            sigma2 = float(np.mean(psi_score ** 2))
            se = (
                float(np.sqrt(sigma2 / (J ** 2 * n)))
                if abs(J) > 1e-10 else 0.0
            )
        else:
            w = sample_weight
            denom = float(np.sum(w * d_resid * d_resid))
            if denom < 1e-12:
                raise RuntimeError(
                    "PLR weighted denominator ≈ 0; check covariate "
                    "informativeness or weight distribution."
                )
            theta = float(np.sum(w * d_resid * y_resid) / denom)
            psi_inner = y_resid - theta * d_resid
            psi_score = psi_inner * d_resid
            # Z-estimator sandwich variance for a weighted moment:
            #     M(θ) = (1/W) Σ w_i ψ_score_i,   W = Σ w_i
            # Var(θ̂) = ( Σ w_i² ψ_score_i² ) / ( Σ w_i d_resid_i² )²
            num = float(np.sum((w ** 2) * (psi_score ** 2)))
            se = float(np.sqrt(num)) / abs(denom) if denom != 0 else 0.0

        # Diagnostics: residual scales, partial correlation, and a crude
        # within-R² for each nuisance — analogous to the panel_dml
        # diagnostics block. These help users sanity-check that the ML
        # nuisances are doing useful residualisation. We use *unweighted*
        # second moments so the numbers are comparable across calls
        # regardless of weight scale.
        var_y = float(np.var(Y))
        var_d = float(np.var(D))
        self._last_rep_diagnostics = {
            "y_resid_std": float(np.std(y_resid)),
            "d_resid_std": float(np.std(d_resid)),
            "partial_corr_yd": float(
                np.corrcoef(y_resid, d_resid)[0, 1]
            ) if (np.std(y_resid) > 0 and np.std(d_resid) > 0) else 0.0,
            "ml_g_within_r2": (
                1.0 - float(np.var(y_resid) / var_y) if var_y > 0 else 0.0
            ),
            "ml_m_within_r2": (
                1.0 - float(np.var(d_resid) / var_d) if var_d > 0 else 0.0
            ),
            "weighted": sample_weight is not None,
        }
        return theta, se
