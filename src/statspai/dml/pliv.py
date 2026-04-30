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
    _ML_M_TARGET_BINARY = False
    _ML_R_TARGET_BINARY = False

    # First-stage degeneracy threshold on |corr(z̃, d̃)|. Below this
    # the instrument is functionally orthogonal to the residualised
    # treatment after the ML control function — the ratio estimator
    # explodes. The previous threshold of 1e-6 was scale-invariant but
    # too lenient: a real weak instrument can have |corr| ~ 1e-3 and
    # still pass. ``1e-3`` is conservative enough to catch numerical
    # collapse; *separately* a partial-correlation diagnostic is
    # exposed so the user can apply weak-IV inference (effective F,
    # AR test) at their preferred threshold.
    _FIRST_STAGE_CORR_FLOOR = 1e-3

    def _fit_one_rep(self, Y, D, X, Z, n, rng_seed, sample_weight=None):
        # The base class refuses ``sample_weight`` upstream because
        # ``_SUPPORTS_SAMPLE_WEIGHT`` is False on this subclass. We
        # accept the kwarg only to keep the override signature
        # compatible with the base.
        del sample_weight
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
        sum_z2 = float(np.sum(z_resid ** 2))
        sum_d2 = float(np.sum(d_resid ** 2))
        scale = float(np.sqrt(max(sum_z2, 0.0) * max(sum_d2, 0.0)))
        partial_corr = denom / scale if scale > 0 else 0.0
        # Two distinct degeneracy modes need separate guards:
        #   (i) ML residualisation drove z_resid to (near-)zero variance —
        #       e.g., Z is a deterministic function of X, fully absorbed
        #       by ml_r. Then ``partial_corr`` is a ratio of floating-point
        #       noise and is *random*, not small; checking |corr| alone
        #       misses this case. Detect via the residual-variance ratio.
        #   (ii) z_resid has variance but is (near-)orthogonal to d_resid
        #       — the standard weak-instrument case. Detect via
        #       |partial_corr|.
        var_z_total = float(np.var(Z)) if Z is not None else 0.0
        var_z_resid = sum_z2 / max(n, 1)
        if var_z_total > 0 and (var_z_resid / var_z_total) < 1e-10:
            raise RuntimeError(
                f"Degenerate PLIV first stage: ML residualisation absorbed "
                f"essentially all of Z's variance "
                f"(Var(z̃)/Var(Z) = {var_z_resid / var_z_total:.2e}). "
                f"The instrument is collinear with X — drop it and find "
                f"an instrument with conditional-on-X variation."
            )
        if abs(partial_corr) < self._FIRST_STAGE_CORR_FLOOR:
            raise RuntimeError(
                f"Weak / degenerate PLIV first stage: |partial corr(z̃, d̃)| "
                f"= {abs(partial_corr):.2e} below floor "
                f"{self._FIRST_STAGE_CORR_FLOOR:.0e}. The ML-residualised "
                f"instrument is (near-)orthogonal to the ML-residualised "
                f"treatment; the ratio estimator is not numerically "
                f"identified. Consider a different instrument, or run "
                f"sp.weakrobust / sp.anderson_rubin_test to check that "
                f"weak-IV-robust inference still has power."
            )
        theta = float(np.sum(z_resid * y_resid) / denom)

        psi = (y_resid - theta * d_resid) * z_resid
        J = -np.mean(z_resid * d_resid)
        sigma2 = np.mean(psi**2)
        se = float(np.sqrt(sigma2 / (J**2 * n))) if abs(J) > 1e-10 else 0.0

        # Approximate first-stage F (informative weak-IV diagnostic)
        # using the partial correlation: F_partial ≈ (n-K) ρ² / (1-ρ²).
        # K is unknown (ML nuisance has no fixed dof), so we use n as an
        # upper bound on (n - K) — the resulting F is mildly optimistic.
        rho2 = partial_corr ** 2
        first_stage_F = (
            float((n) * rho2 / (1.0 - rho2))
            if rho2 < 1.0 - 1e-12
            else float("inf")
        )
        self._last_rep_diagnostics = {
            "first_stage_partial_corr": float(partial_corr),
            "first_stage_F_approx": first_stage_F,
            "z_resid_std": float(np.std(z_resid)),
            "d_resid_std": float(np.std(d_resid)),
        }
        return theta, se
