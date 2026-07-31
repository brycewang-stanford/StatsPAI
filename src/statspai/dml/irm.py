"""
Interactive Regression Model (IRM) for DML.

Binary D. Efficient influence function for ATE (AIPW / cross-fitted
doubly-robust score):

    psi = g(1, X) - g(0, X)
          + D*(Y - g(1, X)) / m(X)
          - (1-D)*(Y - g(0, X)) / (1 - m(X))

theta_ATE = mean(psi);  SE = sd(psi) / sqrt(n).

Folds are stratified by D so that each training fold contains both
arms — without stratification, a small data set or small ``n_folds``
can produce a degenerate fold whose subgroup-fitted nuisance is junk.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._base import _DoubleMLBase


class DoubleMLIRM(_DoubleMLBase):
    """Interactive regression DML — binary D, ATE via AIPW.

    Direct entry point for the interactive regression model with a
    binary treatment; estimates the ATE via the cross-fitted
    doubly-robust (AIPW) score. Usually reached through the dispatcher
    ``sp.dml(..., model='irm')``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> ps = 1 / (1 + np.exp(-(0.5 * x1)))
    >>> d = (rng.uniform(size=n) < ps).astype(int)
    >>> y = 1.0 * d + x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    >>> est = sp.DoubleMLIRM(
    ...     df, y="y", treat="d", covariates=["x1", "x2"], n_folds=3,
    ... )
    >>> res = est.fit()
    >>> bool(np.isfinite(res.estimate))
    True
    """

    _MODEL_TAG = "IRM"
    _ESTIMAND = "ATE"
    _REQUIRES_INSTRUMENT = False
    _ML_M_TARGET_BINARY = True  # ml_m models D ∈ {0, 1}
    # DoubleML-compatible score / IPW options. score='ATE' (default)
    # reproduces the historical StatsPAI AIPW score bit-for-bit;
    # score='ATTE' targets the effect on the treated. normalize_ipw and
    # trimming_threshold mirror DoubleML's IRM knobs (defaults =
    # historical behaviour: no normalization, symmetric 0.01 trim).
    _VALID_SCORES = {"ATE", "ATTE"}
    _DEFAULT_SCORE = "ATE"
    _USES_IPW = True
    # Subgroups (rows with D=1 / D=0 in the training fold) below this
    # size fall back to the subgroup mean rather than fitting a flexible
    # GBM. Mirrors the same protection in DoubleMLIIVM — fitting 100
    # boosted trees on a handful of rows overfits wildly and poisons
    # the influence function for the entire test fold.
    _MIN_SUBGROUP_FIT = 10
    # Symmetric clip on the propensity score m̂(X) = P(D=1 | X). With a
    # mass of observations near 0 or 1 the AIPW score blows up; the clip
    # trades a small amount of bias for stability and is a standard
    # convention (cf. Crump et al. 2009).
    _PSCORE_CLIP_LO = 0.01
    _PSCORE_CLIP_HI = 0.99
    _SUPPORTS_SAMPLE_WEIGHT = True

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
        # Need at least one row per arm per fold for stratified splitting.
        n0, n1 = int(np.sum(D == 0)), int(np.sum(D == 1))
        if min(n0, n1) < self.n_folds:
            from statspai.exceptions import IdentificationFailure  # pragma: no cover

            raise IdentificationFailure(  # pragma: no cover
                f"model='irm' with n_folds={self.n_folds} requires at "
                f"least n_folds rows in each treatment arm; got "
                f"n(D=0)={n0}, n(D=1)={n1}.",
                recovery_hint=(
                    "Reduce n_folds (try n_folds=2 or 3), or check whether "
                    "the smaller arm should be excluded from the analysis."
                ),
                diagnostics={"n_D0": n0, "n_D1": n1, "n_folds": self.n_folds},
                alternative_functions=[],
            )

        splits = self._make_splits(
            X, rng_seed=rng_seed, fold_indices=fold_indices, stratify=D
        )
        # Accumulate the cross-fitted nuisance predictions over folds;
        # trimming / IPW-normalization / the (ATE or ATTE) score are then
        # applied once on the full out-of-fold vectors so that
        # full-sample quantities (p̂ = mean(D), the IPW normalizing
        # constants) match DoubleML exactly.
        g1_full: np.ndarray = np.zeros(n, dtype=float)
        g0_full: np.ndarray = np.zeros(n, dtype=float)
        m_hat_full: np.ndarray = np.zeros(n, dtype=float)
        min_fit = self._MIN_SUBGROUP_FIT
        n_fallback_g1 = 0
        n_fallback_g0 = 0

        for train_idx, test_idx in splits:
            # Test-fold Y/D are no longer needed inside the loop: the
            # score is assembled once on the full out-of-fold vectors
            # after cross-fitting (see below). Only training-fold rows
            # are used here to fit the nuisances.
            D_tr = D[train_idx]
            Y_tr = Y[train_idx]
            X_tr, X_te = X[train_idx], X[test_idx]
            w_tr = sample_weight[train_idx] if sample_weight is not None else None

            mask1 = D_tr == 1
            if mask1.sum() >= min_fit:
                w_sub = w_tr[mask1] if w_tr is not None else None
                ml_g1 = self._fit_weighted(self.ml_g, X_tr[mask1], Y_tr[mask1], w_sub)
                g1_hat = ml_g1.predict(X_te)
            elif mask1.sum() > 0:
                # Too few treated rows to trust a flexible learner; use
                # the subgroup mean as a stable biased fallback. Use the
                # weighted subgroup mean when sample weights are given so
                # the fallback respects the survey design.
                if w_tr is not None:
                    w_sub = w_tr[mask1]
                    g1_hat = np.full(
                        len(test_idx),
                        float(np.average(Y_tr[mask1], weights=w_sub)),
                    )
                else:
                    g1_hat = np.full(len(test_idx), float(np.mean(Y_tr[mask1])))
                n_fallback_g1 += 1
            else:
                # StratifiedKFold should preclude this — guard anyway.
                # pragma: no cover
                from statspai.exceptions import IdentificationFailure

                raise IdentificationFailure(  # pragma: no cover
                    "IRM cross-fit produced a training fold with no D=1 "
                    "rows despite stratification; aborting rather than "
                    "biasing g(1, X) with zeros.",
                    recovery_hint=(
                        "Reduce n_folds and rerun, or inspect treatment "
                        "balance in the data."
                    ),
                    diagnostics={"fold_n_D1": int(mask1.sum())},
                    alternative_functions=[],
                )

            mask0 = D_tr == 0
            if mask0.sum() >= min_fit:
                w_sub = w_tr[mask0] if w_tr is not None else None
                ml_g0 = self._fit_weighted(
                    self.ml_g,
                    X_tr[mask0],
                    Y_tr[mask0],
                    w_sub,
                )
                g0_hat = ml_g0.predict(X_te)
            elif mask0.sum() > 0:
                if w_tr is not None:
                    w_sub = w_tr[mask0]
                    g0_hat = np.full(
                        len(test_idx),
                        float(np.average(Y_tr[mask0], weights=w_sub)),
                    )
                else:
                    g0_hat = np.full(len(test_idx), float(np.mean(Y_tr[mask0])))
                n_fallback_g0 += 1
            else:
                # pragma: no cover
                from statspai.exceptions import IdentificationFailure

                raise IdentificationFailure(  # pragma: no cover
                    "IRM cross-fit produced a training fold with no D=0 "
                    "rows despite stratification; aborting rather than "
                    "biasing g(0, X) with zeros.",
                    recovery_hint=(
                        "Reduce n_folds and rerun, or inspect treatment "
                        "balance in the data."
                    ),
                    diagnostics={"fold_n_D0": int(mask0.sum())},
                    alternative_functions=[],
                )

            ml_m = self._fit_weighted(self.ml_m, X_tr, D_tr, w_tr)
            if hasattr(ml_m, "predict_proba"):
                m_hat = ml_m.predict_proba(X_te)[:, 1]
            else:
                m_hat = ml_m.predict(X_te)
            g1_full[test_idx] = g1_hat
            g0_full[test_idx] = g0_hat
            m_hat_full[test_idx] = m_hat

        # ---- trimming + IPW normalization + score (full-vector) ------
        lo, hi = self.trimming_threshold, 1.0 - self.trimming_threshold
        m_clip = np.clip(m_hat_full, lo, hi)
        u1 = Y - g1_full
        u0 = Y - g0_full

        if self.score == "ATE" and not self.normalize_ipw:
            # Historical StatsPAI AIPW-ATE score — preserved bit-for-bit
            # so default ``sp.dml(model='irm')`` output never moves.
            psi_scores = (
                g1_full - g0_full + D * u1 / m_clip - (1 - D) * u0 / (1 - m_clip)
            )
            if sample_weight is None:
                theta = float(np.mean(psi_scores))
                # Influence-function variance normalised by n, matching
                # DoubleML, the ``normalize_ipw`` / ATTE branch below
                # (which uses mean(psi**2)), the weighted branch, and
                # PLR / PLIV. This path used ddof=1 until v1.21, which
                # inflated the SE by sqrt(n/(n-1)) relative to all four.
                se = float(np.std(psi_scores, ddof=0) / np.sqrt(n))
            else:
                # Weighted Z-estimator: θ̂ = Σ w ψ / Σ w; sandwich SE
                # Var(θ̂) = Σ w_i² (ψ_i − θ̂)² / (Σ w_i)².
                w = sample_weight
                W = float(np.sum(w))
                theta = float(np.sum(w * psi_scores) / W)
                num = float(np.sum((w**2) * (psi_scores - theta) ** 2))
                se = float(np.sqrt(num)) / W
        else:
            # ATTE and/or normalize_ipw — DoubleML-matching orthogonal
            # score. Sample weights are not combined with these (the
            # DoubleML 'weights' object is a GATE construct, not survey
            # weights) — fail loudly rather than mix conventions.
            if sample_weight is not None:
                from statspai.exceptions import MethodIncompatibility

                raise MethodIncompatibility(
                    "dml.irm: sample_weight is currently supported only for "
                    "score='ATE' with normalize_ipw=False. For weighted ATTE "
                    "or normalized-IPW estimands, drop sample_weight."
                )
            # DoubleML _propensity_score_adjustment: optionally renormalize
            # the IPW weights so each arm's inverse-propensity mass is 1.
            if self.normalize_ipw:
                mean_t1 = float(np.mean(D / m_clip))
                mean_t0 = float(np.mean((1.0 - D) / (1.0 - m_clip)))
                m_use = D * (m_clip * mean_t1) + (1.0 - D) * (
                    1.0 - (1.0 - m_clip) * mean_t0
                )
            else:
                m_use = m_clip
            ipw_term = D * u1 / m_use - (1.0 - D) * u0 / (1.0 - m_use)
            if self.score == "ATE":
                psi_b = (g1_full - g0_full) + ipw_term
                psi_a = -np.ones(n, dtype=float)
            else:  # ATTE
                p_hat = float(np.mean(D))
                weights = D / p_hat
                weights_bar = m_use / p_hat
                psi_b = weights * (g1_full - g0_full) + weights_bar * ipw_term
                psi_a = -weights / float(np.mean(weights))
            mean_a = float(np.mean(psi_a))
            theta = float(-np.mean(psi_b) / mean_a)
            psi = psi_a * theta + psi_b
            # DoubleML asymptotic variance: σ̂² = E[ψ²]/E[ψ_a]²; SE = √(σ̂²/n).
            sigma2 = float(np.mean(psi**2) / (mean_a**2))
            se = float(np.sqrt(sigma2 / n))
            # Stash so that ``psi_scores - theta`` equals the influence
            # function ψ (= ψ_a·θ + ψ_b), keeping the residual block below
            # score-agnostic.
            psi_scores = psi + theta

        m_hat = m_clip  # legacy alias kept for the diagnostics block below

        # Overlap diagnostics: how many propensities were clipped, and
        # the empirical distribution. Surface to the user via model_info.
        n_clipped_lo = int(np.sum(m_hat_full < lo))
        n_clipped_hi = int(np.sum(m_hat_full > hi))
        self._last_rep_diagnostics = {
            "pscore_min": float(np.min(m_hat_full)),
            "pscore_max": float(np.max(m_hat_full)),
            "pscore_p01": float(np.quantile(m_hat_full, 0.01)),
            "pscore_p99": float(np.quantile(m_hat_full, 0.99)),
            "n_clipped_below": n_clipped_lo,
            "n_clipped_above": n_clipped_hi,
            "n_subgroup_fallback_g1": n_fallback_g1,
            "n_subgroup_fallback_g0": n_fallback_g0,
            "weighted": sample_weight is not None,
        }
        # Stash IRM-relevant residuals for sensitivity / diagnostics.
        # We don't save per-fold g1/g0 — surface ψ_scores and m_hat_full
        # so dml_diagnostics can build overlap plots, and provide
        # y_resid/d_resid as defined for sensitivity. The natural "D
        # residual" is D - m̂(X); for "Y residual" we use ψ - θ̂ (see
        # below).
        d_resid = D - m_hat_full
        # For IRM, y_resid uses ψ - θ̂ (the score residual) which is the
        # influence function input — appropriate for OVB sensitivity per
        # the Long-Story-Short paper §5.2 (IRM example).
        y_resid = psi_scores - theta
        self._last_rep_residuals = {
            "y_resid": y_resid,
            "d_resid": d_resid,
            "pscore": m_hat_full,
        }
        return theta, se
