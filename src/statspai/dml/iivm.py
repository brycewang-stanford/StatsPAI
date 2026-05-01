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
Weighted variant (with sample_weight w_i):

    theta_LATE,w = E_w[psi_a] / E_w[psi_b]
    Var(theta_w) = sum(w_i^2 * (psi_a_i - theta_w * psi_b_i)^2)
                   / (sum(w_i * psi_b_i)^2)

where E_w denotes the empirical weighted mean under the supplied sample
weights.

Folds are stratified by Z so each training fold contains both arms of
the instrument; fitting g(z, X) on an empty subgroup would otherwise
silently fall back to zero and bias every test-fold influence-function
contribution.
"""

import numpy as np

from ._base import _DoubleMLBase


class DoubleMLIIVM(_DoubleMLBase):
    """Interactive IV DML — binary D, binary Z, LATE via Wald."""

    _MODEL_TAG = 'IIVM'
    _ESTIMAND = 'LATE'
    _REQUIRES_INSTRUMENT = True
    # IIVM ml_m models the *instrument* propensity P(Z=1|X), not D.
    # ml_r models compliance r(z, X) = P(D=1|Z=z, X). Both targets are
    # binary so both default to classifiers.
    _ML_M_TARGET_BINARY = True
    _ML_R_TARGET_BINARY = True
    _SUPPORTS_SAMPLE_WEIGHT = True
    # Instrument-propensity clip — same role as the IRM propensity clip.
    _PSCORE_CLIP_LO = 0.01
    _PSCORE_CLIP_HI = 0.99
    # First-stage compliance clip — r1, r0 do NOT appear in denominators
    # of the AIPW score, so a tighter clip mainly bounds extrapolated
    # predictions for stability.
    _COMPLIANCE_CLIP_LO = 1e-4
    _COMPLIANCE_CLIP_HI = 1 - 1e-4

    def _fit_one_rep(self, Y, D, X, Z, n, rng_seed, sample_weight=None):
        from sklearn.model_selection import StratifiedKFold

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
        # StratifiedKFold needs n_folds ≤ min class count in the
        # stratification variable (Z here).
        nZ0, nZ1 = int(np.sum(Z == 0)), int(np.sum(Z == 1))
        if min(nZ0, nZ1) < self.n_folds:
            raise ValueError(
                f"model='iivm' with n_folds={self.n_folds} requires at least "
                f"n_folds rows under each instrument arm; got "
                f"n(Z=0)={nZ0}, n(Z=1)={nZ1}. Reduce n_folds or check "
                f"instrument balance."
            )

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=rng_seed,
        )
        if sample_weight is None:
            w_full = None
        else:
            w_arr = np.asarray(sample_weight, dtype=float)
            # Classifiers with regularisation can react to a pure scale
            # change in sample_weight. Normalise to mean 1 so w and c*w
            # define the same weighted empirical measure in practice.
            w_full = w_arr * (len(w_arr) / float(np.sum(w_arr)))
        g1 = np.zeros(n)
        g0 = np.zeros(n)
        r1 = np.zeros(n)
        r0 = np.zeros(n)
        m_hat = np.zeros(n)
        n_fallback_g1 = 0
        n_fallback_g0 = 0
        n_fallback_r1 = 0
        n_fallback_r0 = 0

        for train_idx, test_idx in skf.split(X, Z):
            X_tr, X_te = X[train_idx], X[test_idx]
            Y_tr, Z_tr, D_tr = Y[train_idx], Z[train_idx], D[train_idx]
            w_tr = w_full[train_idx] if w_full is not None else None

            # g(1, X), g(0, X) — outcome under each Z arm
            mask_z1 = Z_tr == 1
            mask_z0 = Z_tr == 0
            g1[test_idx], used_fb_g1 = self._fit_predict_subgroup(
                self.ml_g, X_tr[mask_z1], Y_tr[mask_z1], X_te,
                fallback_y=Y_tr[mask_z1],
                weights_sub=w_tr[mask_z1] if w_tr is not None else None,
                arm_label="Z=1 outcome g(1, X)",
            )
            g0[test_idx], used_fb_g0 = self._fit_predict_subgroup(
                self.ml_g, X_tr[mask_z0], Y_tr[mask_z0], X_te,
                fallback_y=Y_tr[mask_z0],
                weights_sub=w_tr[mask_z0] if w_tr is not None else None,
                arm_label="Z=0 outcome g(0, X)",
            )

            # r(z, X) = P(D=1 | Z=z, X) — first-stage compliance
            r1[test_idx], used_fb_r1 = self._fit_predict_classifier(
                self.ml_r, X_tr[mask_z1], D_tr[mask_z1], X_te,
                weights_sub=w_tr[mask_z1] if w_tr is not None else None,
                arm_label="Z=1 first-stage r(1, X)",
            )
            r0[test_idx], used_fb_r0 = self._fit_predict_classifier(
                self.ml_r, X_tr[mask_z0], D_tr[mask_z0], X_te,
                weights_sub=w_tr[mask_z0] if w_tr is not None else None,
                arm_label="Z=0 first-stage r(0, X)",
            )
            n_fallback_g1 += int(used_fb_g1)
            n_fallback_g0 += int(used_fb_g0)
            n_fallback_r1 += int(used_fb_r1)
            n_fallback_r0 += int(used_fb_r0)

            # m(X) = P(Z=1 | X) — instrument propensity
            ml_m = self._fit_weighted(self.ml_m, X_tr, Z_tr, w_tr)
            if hasattr(ml_m, 'predict_proba'):
                m_hat[test_idx] = ml_m.predict_proba(X_te)[:, 1]
            else:
                m_hat[test_idx] = ml_m.predict(X_te)

        m_hat_raw = m_hat.copy()
        m_hat = np.clip(m_hat, self._PSCORE_CLIP_LO, self._PSCORE_CLIP_HI)
        r1 = np.clip(r1, self._COMPLIANCE_CLIP_LO, self._COMPLIANCE_CLIP_HI)
        r0 = np.clip(r0, self._COMPLIANCE_CLIP_LO, self._COMPLIANCE_CLIP_HI)

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

        if w_full is None:
            w = np.ones(n, dtype=float)
            W = float(n)
        else:
            w = w_full
            W = float(np.sum(w))
        num = float(np.sum(w * psi_a) / W)
        den = float(np.sum(w * psi_b) / W)
        if abs(den) < 1e-6:
            raise RuntimeError(
                f"Degenerate IIVM first stage: E[psi_b] ≈ {den:.2e}. "
                "Compliance (first-stage effect of Z on D) is near zero; "
                "LATE is not identified."
            )
        theta = num / den
        phi = psi_a - theta * psi_b
        if w_full is None:
            influence = phi / den
            sigma2 = float(np.var(influence, ddof=1))
            se = float(np.sqrt(sigma2 / n))
        else:
            score = w * phi
            num_var = float(np.sum(score ** 2))
            den_var = abs(float(np.sum(w * psi_b)))
            se = float(np.sqrt(num_var)) / den_var if den_var > 0 else 0.0

        n_clipped_lo = int(np.sum(m_hat_raw < self._PSCORE_CLIP_LO))
        n_clipped_hi = int(np.sum(m_hat_raw > self._PSCORE_CLIP_HI))
        self._last_rep_diagnostics = {
            "pscore_z_min": float(np.min(m_hat_raw)),
            "pscore_z_max": float(np.max(m_hat_raw)),
            "pscore_z_p01": float(np.quantile(m_hat_raw, 0.01)),
            "pscore_z_p99": float(np.quantile(m_hat_raw, 0.99)),
            "n_clipped_below": n_clipped_lo,
            "n_clipped_above": n_clipped_hi,
            "n_subgroup_fallback_g1": n_fallback_g1,
            "n_subgroup_fallback_g0": n_fallback_g0,
            "n_subgroup_fallback_r1": n_fallback_r1,
            "n_subgroup_fallback_r0": n_fallback_r0,
            "first_stage_E_psi_b": den,
            "weighted": w_full is not None,
        }
        return theta, se

    # ----- small helpers kept here (local to IIVM) -----

    # Subgroups below this size fall back to a constant (subgroup mean)
    # rather than fitting a flexible learner on pathologically small
    # data. Fitting a gradient-boosted forest on <10 rows almost always
    # overfits and poisons the influence function for the whole test
    # fold; falling back to the mean is biased but stable.
    _MIN_SUBGROUP_FIT = 10

    @staticmethod
    def _fit_predict_subgroup(
        learner, X_sub, y_sub, X_te, fallback_y, weights_sub=None,
        arm_label="(unspecified)",
    ):
        """Fit ``learner`` on a subgroup; fall back to subgroup mean if too small.

        Returns ``(predictions, fallback_used)`` where ``fallback_used``
        is True iff the subgroup mean replaced a flexible fit.
        """
        from statspai.exceptions import IdentificationFailure

        if len(X_sub) >= DoubleMLIIVM._MIN_SUBGROUP_FIT:
            clf = DoubleMLIIVM._fit_weighted(learner, X_sub, y_sub, weights_sub)
            return clf.predict(X_te), False
        if len(fallback_y) > 0:
            if weights_sub is not None and float(np.sum(weights_sub)) > 0:
                fallback_mean = float(np.average(fallback_y, weights=weights_sub))
            else:
                fallback_mean = float(np.mean(fallback_y))
            return (
                np.full(len(X_te), fallback_mean),
                True,
            )
        raise IdentificationFailure(
            f"IIVM cross-fit produced an empty subgroup for {arm_label}; "
            "aborting rather than biasing the AIPW score with zeros.",
            recovery_hint=(
                "Reduce n_folds and rerun, or inspect the joint balance "
                "of (D, Z) in the data."
            ),
            diagnostics={"arm": arm_label, "subgroup_size": int(len(X_sub))},
            alternative_functions=[],
        )

    @staticmethod
    def _fit_predict_classifier(
        learner, X_sub, d_sub, X_te, weights_sub=None,
        arm_label="(unspecified)",
    ):
        """Fit a classifier on (X_sub, d_sub); fall back to mean(d_sub).

        Returns ``(predictions, fallback_used)``.
        """
        from statspai.exceptions import IdentificationFailure

        if (
            len(X_sub) >= DoubleMLIIVM._MIN_SUBGROUP_FIT
            and len(np.unique(d_sub)) > 1
        ):
            clf = DoubleMLIIVM._fit_weighted(learner, X_sub, d_sub, weights_sub)
            if hasattr(clf, 'predict_proba'):
                return clf.predict_proba(X_te)[:, 1], False
            return clf.predict(X_te), False
        if len(d_sub) > 0:
            if weights_sub is not None and float(np.sum(weights_sub)) > 0:
                fallback_mean = float(np.average(d_sub, weights=weights_sub))
            else:
                fallback_mean = float(np.mean(d_sub))
            return np.full(len(X_te), fallback_mean), True
        raise IdentificationFailure(
            f"IIVM cross-fit produced an empty subgroup for {arm_label}; "
            "aborting rather than biasing the first-stage score with zeros.",
            recovery_hint=(
                "Reduce n_folds and rerun, or inspect the joint balance "
                "of (D, Z) in the data."
            ),
            diagnostics={"arm": arm_label, "subgroup_size": int(len(X_sub))},
            alternative_functions=[],
        )
