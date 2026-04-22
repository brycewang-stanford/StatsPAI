"""
Interactive Regression Model (IRM) for DML.

Binary D. Efficient influence function for ATE (AIPW / cross-fitted
doubly-robust score):

    psi = g(1, X) - g(0, X)
          + D*(Y - g(1, X)) / m(X)
          - (1-D)*(Y - g(0, X)) / (1 - m(X))

theta_ATE = mean(psi);  SE = sd(psi) / sqrt(n).
"""

import numpy as np

from ._base import _DoubleMLBase


class DoubleMLIRM(_DoubleMLBase):
    """Interactive regression DML — binary D, ATE via AIPW."""

    _MODEL_TAG = 'IRM'
    _ESTIMAND = 'ATE'
    _REQUIRES_INSTRUMENT = False
    _BINARY_TREATMENT = True
    # Subgroups (rows with D=1 / D=0 in the training fold) below this
    # size fall back to the subgroup mean rather than fitting a flexible
    # GBM. Mirrors the same protection in DoubleMLIIVM — fitting 100
    # boosted trees on a handful of rows overfits wildly and poisons
    # the influence function for the entire test fold.
    _MIN_SUBGROUP_FIT = 10

    def _fit_one_rep(self, Y, D, X, Z, n, rng_seed):
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        if not set(np.unique(D)).issubset({0, 1}):
            from statspai.exceptions import MethodIncompatibility
            raise MethodIncompatibility(
                "model='irm' requires binary treatment (0/1).",
                recovery_hint=(
                    "Use model='plr' for continuous treatment, or "
                    "sp.multi_treatment for multi-valued treatments."
                ),
                diagnostics={"treat_values": sorted(map(float, np.unique(D)))[:10]},
                alternative_functions=["sp.dml", "sp.multi_treatment"],
            )
        if len(np.unique(D)) < 2:
            from statspai.exceptions import IdentificationFailure
            raise IdentificationFailure(
                "model='irm' requires variation in D (both 0 and 1); "
                "ATE is not identified with a constant treatment.",
                recovery_hint=(
                    "Treatment is constant in the sample — check the filter / "
                    "data pipeline."
                ),
                diagnostics={"n_unique_D": int(len(np.unique(D)))},
                alternative_functions=[],
            )

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rng_seed)
        psi_scores = np.zeros(n)
        min_fit = self._MIN_SUBGROUP_FIT

        for train_idx, test_idx in kf.split(X):
            D_tr, D_te = D[train_idx], D[test_idx]
            Y_tr, Y_te = Y[train_idx], Y[test_idx]
            X_tr, X_te = X[train_idx], X[test_idx]

            mask1 = D_tr == 1
            if mask1.sum() >= min_fit:
                ml_g1 = clone(self.ml_g)
                ml_g1.fit(X_tr[mask1], Y_tr[mask1])
                g1_hat = ml_g1.predict(X_te)
            elif mask1.sum() > 0:
                # Too few treated rows to trust a flexible learner; use
                # the subgroup mean as a stable biased fallback.
                g1_hat = np.full(len(test_idx), float(np.mean(Y_tr[mask1])))
            else:
                g1_hat = np.zeros(len(test_idx))

            mask0 = D_tr == 0
            if mask0.sum() >= min_fit:
                ml_g0 = clone(self.ml_g)
                ml_g0.fit(X_tr[mask0], Y_tr[mask0])
                g0_hat = ml_g0.predict(X_te)
            elif mask0.sum() > 0:
                g0_hat = np.full(len(test_idx), float(np.mean(Y_tr[mask0])))
            else:
                g0_hat = np.zeros(len(test_idx))

            # Propensity: need both arms present in training fold
            if len(np.unique(D_tr)) < 2:
                # Constant D in training fold — propensity collapses to
                # the empirical mean; use it.
                m_hat = np.full(len(test_idx), float(np.mean(D_tr)))
            else:
                ml_m = clone(self.ml_m)
                ml_m.fit(X_tr, D_tr)
                if hasattr(ml_m, 'predict_proba'):
                    m_hat = ml_m.predict_proba(X_te)[:, 1]
                else:
                    m_hat = ml_m.predict(X_te)
            m_hat = np.clip(m_hat, 0.01, 0.99)

            psi_scores[test_idx] = (
                g1_hat - g0_hat
                + D_te * (Y_te - g1_hat) / m_hat
                - (1 - D_te) * (Y_te - g0_hat) / (1 - m_hat)
            )

        theta = float(np.mean(psi_scores))
        se = float(np.std(psi_scores, ddof=1) / np.sqrt(n))
        return theta, se
