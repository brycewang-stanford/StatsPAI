"""
Interactive IV Model (IIVM) for DML.

Binary D, binary Z. Estimates LATE (compliers) via the Neyman-orthogonal
ratio of two doubly-robust scores:

    psi_a = g(1, X) - g(0, X)
            + Z*(Y - g(1, X))/m(X)
            - (1-Z)*(Y - g(0, X))/(1 - m(X))

    psi_b = r(1, X) - r(0, X)
            + Z*(D - r(1, X))/m(X)
            - (1-Z)*(D - r(0, X))/(1 - m(X))

    theta_LATE = E[psi_a] / E[psi_b]

where g(z, X) = E[Y|Z=z, X], r(z, X) = E[D|Z=z, X], m(X) = P(Z=1|X).
SE via delta-method on the ratio.
"""

import numpy as np

from ._base import _DoubleMLBase


class DoubleMLIIVM(_DoubleMLBase):
    """Interactive IV DML — binary D, binary Z, LATE via Wald."""

    _MODEL_TAG = 'IIVM'
    _ESTIMAND = 'LATE'
    _REQUIRES_INSTRUMENT = True
    _BINARY_TREATMENT = True
    _BINARY_INSTRUMENT = True

    def _fit_one_rep(self, Y, D, X, Z, n, rng_seed):
        from sklearn.base import clone
        from sklearn.model_selection import KFold

        if not set(np.unique(Z)).issubset({0, 1}):
            raise ValueError(
                "model='iivm' requires a binary (0/1) instrument Z. "
                "For continuous instruments use model='pliv'."
            )
        if not set(np.unique(D)).issubset({0, 1}):
            raise ValueError(
                "model='iivm' requires a binary (0/1) treatment D. "
                "For continuous treatments use model='pliv'."
            )
        # Identification requires variation in both Z AND D. Without
        # variation in D the LATE is trivially non-identified and the
        # nuisance regressions blow up; we'd rather fail loud here
        # than return a giant-SE garbage estimate from near-zero
        # first-stage compliance.
        if len(np.unique(Z)) < 2:
            raise ValueError(
                "model='iivm' requires variation in Z (saw a single value). "
                "The instrument must take both 0 and 1 in the data."
            )
        if len(np.unique(D)) < 2:
            raise ValueError(
                "model='iivm' requires variation in D (saw a single value). "
                "The treatment must take both 0 and 1 — with no compliance "
                "variation, LATE is not identified."
            )

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rng_seed)
        g1 = np.zeros(n)
        g0 = np.zeros(n)
        r1 = np.zeros(n)
        r0 = np.zeros(n)
        m_hat = np.zeros(n)

        for train_idx, test_idx in kf.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            Y_tr, Z_tr, D_tr = Y[train_idx], Z[train_idx], D[train_idx]

            # g(1, X), g(0, X) — outcome under each Z arm
            mask_z1 = Z_tr == 1
            mask_z0 = Z_tr == 0
            g1[test_idx] = self._fit_predict_subgroup(
                self.ml_g, X_tr[mask_z1], Y_tr[mask_z1], X_te, Y_tr[mask_z1]
            )
            g0[test_idx] = self._fit_predict_subgroup(
                self.ml_g, X_tr[mask_z0], Y_tr[mask_z0], X_te, Y_tr[mask_z0]
            )

            # r(z, X) = P(D=1 | Z=z, X) — first-stage compliance
            r1[test_idx] = self._fit_predict_classifier(
                self.ml_r, X_tr[mask_z1], D_tr[mask_z1], X_te
            )
            r0[test_idx] = self._fit_predict_classifier(
                self.ml_r, X_tr[mask_z0], D_tr[mask_z0], X_te
            )

            # m(X) = P(Z=1 | X) — instrument propensity
            ml_m = clone(self.ml_m)
            ml_m.fit(X_tr, Z_tr)
            if hasattr(ml_m, 'predict_proba'):
                m_hat[test_idx] = ml_m.predict_proba(X_te)[:, 1]
            else:
                m_hat[test_idx] = ml_m.predict(X_te)

        m_hat = np.clip(m_hat, 0.01, 0.99)
        r1 = np.clip(r1, 1e-4, 1 - 1e-4)
        r0 = np.clip(r0, 1e-4, 1 - 1e-4)

        psi_a = (
            g1 - g0
            + Z * (Y - g1) / m_hat
            - (1 - Z) * (Y - g0) / (1 - m_hat)
        )
        psi_b = (
            r1 - r0
            + Z * (D - r1) / m_hat
            - (1 - Z) * (D - r0) / (1 - m_hat)
        )

        num = float(np.mean(psi_a))
        den = float(np.mean(psi_b))
        if abs(den) < 1e-6:
            raise RuntimeError(
                f"Degenerate IIVM first stage: E[psi_b] ≈ {den:.2e}. "
                "Compliance (first-stage effect of Z on D) is near zero; "
                "LATE is not identified."
            )
        theta = num / den
        influence = (psi_a - theta * psi_b) / den
        sigma2 = float(np.var(influence, ddof=1))
        se = float(np.sqrt(sigma2 / n))
        return theta, se

    # ----- small helpers kept here (local to IIVM) -----

    # Subgroups below this size fall back to a constant (subgroup mean)
    # rather than fitting a flexible learner on pathologically small
    # data. Fitting a gradient-boosted forest on <10 rows almost always
    # overfits and poisons the influence function for the whole test
    # fold; falling back to the mean is biased but stable.
    _MIN_SUBGROUP_FIT = 10

    @staticmethod
    def _fit_predict_subgroup(learner, X_sub, y_sub, X_te, fallback_y):
        """Fit `learner` on a subgroup; fall back to subgroup mean if too small."""
        from sklearn.base import clone
        if len(X_sub) >= DoubleMLIIVM._MIN_SUBGROUP_FIT:
            clf = clone(learner)
            clf.fit(X_sub, y_sub)
            return clf.predict(X_te)
        if len(fallback_y) > 0:
            return np.full(len(X_te), float(np.mean(fallback_y)))
        return np.zeros(len(X_te))

    @staticmethod
    def _fit_predict_classifier(learner, X_sub, d_sub, X_te):
        """Fit a classifier on (X_sub, d_sub); fall back to mean(d_sub)."""
        from sklearn.base import clone
        if (
            len(X_sub) >= DoubleMLIIVM._MIN_SUBGROUP_FIT
            and len(np.unique(d_sub)) > 1
        ):
            clf = clone(learner)
            clf.fit(X_sub, d_sub)
            if hasattr(clf, 'predict_proba'):
                return clf.predict_proba(X_te)[:, 1]
            return clf.predict(X_te)
        if len(d_sub) > 0:
            return np.full(len(X_te), float(np.mean(d_sub)))
        return np.zeros(len(X_te))
