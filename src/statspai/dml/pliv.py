"""
Partially Linear IV (PLIV) model for DML.

Model: ``Y = theta * D + g(X) + eps``, ``E[eps | Z, X] = 0``.

Neyman-orthogonal score:
    psi = (Y - g(X) - theta*(D - m(X))) * (Z - r(X))

DML2 (pooled-moment) ratio estimator:
    theta = sum(y_tilde * z_tilde) / sum(d_tilde * z_tilde)

Weighted variant (with sample_weight w_i):
    theta = sum(w * y_tilde * z_tilde) / sum(w * d_tilde * z_tilde)
    Var(theta) = sum(w^2 * psi_i^2) / (sum(w * d_tilde * z_tilde))^2
where psi_i = (y_tilde_i - theta * d_tilde_i) * z_tilde_i.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ._base import _DoubleMLBase


def _anderson_rubin_set(
    y_resid: np.ndarray, d_resid: np.ndarray, z_resid: np.ndarray, alpha: float
) -> Dict[str, Any]:
    """Weak-instrument-robust confidence set for the PLIV coefficient.

    At a hypothesised ``theta`` the orthogonal score is ``m_i(theta) =
    (y_i - theta*d_i) * z_i`` in residualised variables, and ``C(theta) =
    n * mean(m)^2 / var(m)`` is chi-squared(1) whether or not the
    instrument is strong (Anderson and Rubin 1949, applied to the
    cross-fitted score). The set ``{theta: C(theta) <= q}`` is the solution
    of a quadratic inequality, so it is found exactly: an interval when
    the first stage is significant at level ``alpha``, and otherwise the
    whole line or the complement of an interval.
    """
    from scipy import stats

    n = len(y_resid)
    q = float(stats.chi2.ppf(1.0 - alpha, 1))
    u, w = y_resid * z_resid, d_resid * z_resid
    a, b = float(np.mean(u)), float(np.mean(w))
    # (n + q) * (a - theta*b)^2 - q * mean((u - theta*w)^2) <= 0
    A = (n + q) * b * b - q * float(np.mean(w * w))
    B = -2.0 * (n + q) * a * b + 2.0 * q * float(np.mean(u * w))
    C = (n + q) * a * a - q * float(np.mean(u * u))
    disc = B * B - 4.0 * A * C
    out: Dict[str, Any] = {
        "level": 1.0 - alpha,
        "statistic_at_zero": float(n * a * a / np.var(u)),
        "first_stage_statistic": float(n * b * b / np.var(w)),
        "critical_value": q,
    }
    out["p_value_at_zero"] = float(stats.chi2.sf(out["statistic_at_zero"], 1))
    if abs(A) < 1e-300:  # pragma: no cover - measure-zero boundary
        out.update(kind="unbounded", intervals=[(-np.inf, np.inf)])
        return out
    if disc < 0:
        if A > 0:
            out.update(kind="empty", intervals=[])
        else:
            out.update(kind="unbounded", intervals=[(-np.inf, np.inf)])
        return out
    r1 = (-B - np.sqrt(disc)) / (2.0 * A)
    r2 = (-B + np.sqrt(disc)) / (2.0 * A)
    lo, hi = float(min(r1, r2)), float(max(r1, r2))
    if A > 0:
        out.update(kind="interval", intervals=[(lo, hi)])
    else:
        out.update(kind="disjoint", intervals=[(-np.inf, lo), (hi, np.inf)])
    return out


class DoubleMLPLIV(_DoubleMLBase):
    """Partially linear IV DML — endogenous D with continuous/binary Z.

    Direct entry point for the partially linear IV model
    ``Y = theta * D + g(X) + eps`` with ``E[eps | Z, X] = 0``. Usually
    reached through the dispatcher ``sp.dml(..., model='pliv')``; an
    instrument column must be supplied.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> z = rng.normal(size=n)
    >>> d = 0.8 * z + 0.5 * x1 + rng.normal(size=n)
    >>> y = 1.5 * d + x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "z": z, "x1": x1, "x2": x2})
    >>> est = sp.DoubleMLPLIV(
    ...     df, y="y", treat="d", covariates=["x1", "x2"],
    ...     instrument="z", n_folds=3,
    ... )
    >>> res = est.fit()
    >>> bool(np.isfinite(res.estimate))
    True
    """

    _MODEL_TAG = "PLIV"
    _ESTIMAND = "LATE"
    _REQUIRES_INSTRUMENT = True
    _ML_M_TARGET_BINARY = False
    _ML_R_TARGET_BINARY = False
    _SUPPORTS_SAMPLE_WEIGHT = True
    # PLIV has a single (partialling-out) score; declared so the option
    # is accepted and surfaced in model_info, matching DoubleML's PLIV.
    _VALID_SCORES = {"partialling out"}
    _DEFAULT_SCORE = "partialling out"

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

    def _fit_one_rep(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray,
        Z: np.ndarray,
        n: int,
        rng_seed: int,
        sample_weight: Optional[np.ndarray] = None,
        fold_indices: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        if sample_weight is None:
            w_full: Optional[np.ndarray] = None
        else:
            w_arr = np.asarray(sample_weight, dtype=float)
            # Nuisance learners can be numerically sensitive to a pure
            # rescaling of sample_weight even though the target weighted
            # estimand is not. Normalise to mean 1 so w and c*w define
            # the same fitting problem in practice.
            w_full = w_arr * (len(w_arr) / float(np.sum(w_arr)))

        splits = self._make_splits(X, rng_seed=rng_seed, fold_indices=fold_indices)
        y_resid: np.ndarray = np.zeros(n, dtype=float)
        d_resid: np.ndarray = np.zeros(n, dtype=float)
        z_resid: np.ndarray = np.zeros(n, dtype=float)

        for train_idx, test_idx in splits:
            w_train = w_full[train_idx] if w_full is not None else None
            ml_g = self._fit_weighted(
                self.ml_g,
                X[train_idx],
                Y[train_idx],
                w_train,
            )
            y_resid[test_idx] = Y[test_idx] - self._predict_nuisance(
                ml_g, X[test_idx], Y, "ml_g"
            )

            ml_m = self._fit_weighted(
                self.ml_m,
                X[train_idx],
                D[train_idx],
                w_train,
            )
            d_resid[test_idx] = D[test_idx] - self._predict_nuisance(
                ml_m, X[test_idx], D, "ml_m"
            )

            ml_r = self._fit_weighted(
                self.ml_r,
                X[train_idx],
                Z[train_idx],
                w_train,
            )
            z_resid[test_idx] = Z[test_idx] - self._predict_nuisance(
                ml_r, X[test_idx], Z, "ml_r"
            )

        if w_full is None:
            w: np.ndarray = np.ones(n, dtype=float)
            label = "PLIV"
        else:
            w = w_full
            label = "PLIV weighted"

        W = float(np.sum(w))
        denom = float(np.sum(w * z_resid * d_resid))
        sum_z2 = float(np.sum(w * (z_resid**2)))
        sum_d2 = float(np.sum(w * (d_resid**2)))
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
        if w_full is None:
            var_z_total = float(np.var(Z)) if Z is not None else 0.0
        else:
            z_bar = float(np.sum(w * Z) / W)
            var_z_total = float(np.sum(w * ((Z - z_bar) ** 2)) / W)
        var_z_resid = sum_z2 / max(W, 1.0)
        if var_z_total > 0 and (var_z_resid / var_z_total) < 1e-10:
            raise RuntimeError(
                f"Degenerate PLIV first stage: ML residualisation absorbed "
                f"essentially all of Z's variance "
                f"(Var(z̃)/Var(Z) = {var_z_resid / var_z_total:.2e}). "
                f"The instrument is collinear with X — drop it and find "
                f"an instrument with conditional-on-X variation."
            )
        if abs(partial_corr) < self._FIRST_STAGE_CORR_FLOOR:
            raise RuntimeError(  # pragma: no cover
                f"Weak / degenerate PLIV first stage: |partial corr(z̃, d̃)| "
                f"= {abs(partial_corr):.2e} below floor "
                f"{self._FIRST_STAGE_CORR_FLOOR:.0e}. The ML-residualised "
                f"instrument is (near-)orthogonal to the ML-residualised "
                f"treatment; the ratio estimator is not numerically "
                f"identified. Consider a different instrument, or run "
                f"sp.weakrobust / sp.anderson_rubin_test to check that "
                f"weak-IV-robust inference still has power."
            )
        if abs(denom) < 1e-12:
            raise RuntimeError(  # pragma: no cover
                f"{label} denominator ≈ 0; the ML-residualised instrument "
                f"is effectively orthogonal to the ML-residualised treatment "
                f"under the supplied weight measure."
            )
        theta = float(np.sum(w * z_resid * y_resid) / denom)

        psi = (y_resid - theta * d_resid) * z_resid
        if w_full is None:
            J = -np.mean(z_resid * d_resid)
            sigma2 = np.mean(psi**2)
            se = float(np.sqrt(sigma2 / (J**2 * n))) if abs(J) > 1e-10 else 0.0
        else:
            num = float(np.sum((w**2) * (psi**2)))
            se = float(np.sqrt(num)) / abs(denom) if denom != 0 else 0.0

        # Linear score elements (DoubleML convention) for the base class's
        # cluster-robust recomputation.
        self._last_rep_score = {
            "psi_a": -z_resid * d_resid,
            "psi_b": z_resid * y_resid,
            "splits": splits,
            "weights": w_full,
        }

        # Approximate first-stage F (informative weak-IV diagnostic)
        # using the partial correlation: F_partial ≈ (n-K) ρ² / (1-ρ²).
        # K is unknown (ML nuisance has no fixed dof), so we use n as an
        # upper bound on (n - K) — the resulting F is mildly optimistic.
        rho2 = partial_corr**2
        first_stage_F = (
            float((n) * rho2 / (1.0 - rho2)) if rho2 < 1.0 - 1e-12 else float("inf")
        )
        if w_full is None and self.cluster is None:
            self._last_rep_ar = _anderson_rubin_set(
                y_resid, d_resid, z_resid, self.alpha
            )
        self._last_rep_diagnostics = {
            "first_stage_partial_corr": float(partial_corr),
            "first_stage_F_approx": first_stage_F,
            "z_resid_std": float(np.std(z_resid)),
            "d_resid_std": float(np.std(d_resid)),
            "weighted": w_full is not None,
        }
        return theta, se
