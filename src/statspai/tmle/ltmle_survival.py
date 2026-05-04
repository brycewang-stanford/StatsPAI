"""
Longitudinal TMLE for survival outcomes (discrete-time pooled hazards).

Estimates the counterfactual survival function ``S^{a}(t) = P(T > t | do(A))``
under static or dynamic treatment regimes, with optional right censoring,
via a discrete-time pooled-logistic hazard model targeted at each time
interval.

This is the survival analogue of :func:`ltmle`. It treats survival as a
sequence of K binary hazards ``h_k = P(T = k | T >= k, history)`` and
runs the LTMLE recursion on the pooled hazard scale, which is the
standard approach for causal survival analysis (van der Laan & Gruber
2012; Stitelman-van der Laan 2010; Cai & van der Laan 2019).

Data layout
-----------
Wide-format DataFrame with one row per subject and columns:

* ``T_k`` for k=1..K — the *event indicator* at interval k
  (1 if event at k, 0 otherwise)
* ``C_k`` for k=1..K — the *observed-at-k indicator* (optional)
* ``A_k`` for k=1..K — treatment at interval k
* ``L_k`` — time-varying covariates at interval k

Subjects contribute to the hazard regression at interval k only while
they are at risk (no event and uncensored through k-1).

References
----------
van der Laan, M.J. & Gruber, S. (2012). "Targeted minimum loss based
estimation of causal effects of multiple time point interventions."
*IJB*, 8(1). [@vanderlaan2012targeted]

Stitelman, O.M. & van der Laan, M.J. (2010). "Collaborative targeted
maximum likelihood for time to event data." *IJB*, 6(1). [@stitelman2010collaborative]

Cai, W. & van der Laan, M. J. (2020). "One‐step targeted maximum
likelihood estimation for time‐to‐event outcomes." *Biometrics*.
doi 10.1111/biom.13172. [@cai2020step]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

# sklearn is imported lazily inside ``_fit_logit`` so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches ltmle_survival.


Regime = Union[Sequence[int], Callable[[int, Dict[str, np.ndarray]], np.ndarray]]


@dataclass
class LTMLESurvivalResult:
    """Counterfactual survival curves and contrasts."""

    times: np.ndarray
    survival_treated: np.ndarray
    survival_control: np.ndarray
    rmst_treated: float
    rmst_control: float
    rmst_difference: float
    rmst_se: float
    rmst_ci: tuple
    rmst_pvalue: float
    risk_difference_final: float
    risk_difference_final_se: float
    K: int
    n_obs: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lo, hi = self.rmst_ci
        return (
            "LTMLE for survival\n"
            "-------------------\n"
            f"  K (time intervals)          : {self.K}\n"
            f"  N                           : {self.n_obs}\n"
            f"  Treated survival S(K)       : {self.survival_treated[-1]:.4f}\n"
            f"  Control survival S(K)       : {self.survival_control[-1]:.4f}\n"
            f"  Risk difference at K        : {self.risk_difference_final:+.4f}"
            f"   (SE={self.risk_difference_final_se:.4f})\n"
            f"  RMST treated                : {self.rmst_treated:.4f}\n"
            f"  RMST control                : {self.rmst_control:.4f}\n"
            f"  RMST difference             : {self.rmst_difference:+.4f}"
            f"   (SE={self.rmst_se:.4f})\n"
            f"  RMST 95% CI                 : [{lo:.4f}, {hi:.4f}]\n"
            f"  RMST p-value                : {self.rmst_pvalue:.4f}"
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "time": self.times,
            "S_treated": self.survival_treated,
            "S_control": self.survival_control,
            "risk_diff": self.survival_control - self.survival_treated,
        })

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers (mirror ltmle.py's style)
# ═══════════════════════════════════════════════════════════════════════

def _fit_logit(X: np.ndarray, y: np.ndarray):
    if np.all(y == y[0]):
        class _Const:
            def __init__(self, p):
                self.p = p

            def predict_proba(self, X):
                return np.column_stack([
                    1 - self.p * np.ones(X.shape[0]),
                    self.p * np.ones(X.shape[0]),
                ])
        return _Const(float(y[0]))
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    lr.fit(X, y)
    return lr


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(X)
    return prob[:, 1] if prob.ndim == 2 else prob


def _safe_logit(p, eps=1e-6):
    return logit(np.clip(p, eps, 1 - eps))


# ═══════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════

def ltmle_survival(
    data: pd.DataFrame,
    event_indicators: Sequence[str],
    treatments: Sequence[str],
    covariates_time: Sequence[Sequence[str]],
    baseline: Optional[Sequence[str]] = None,
    censoring: Optional[Sequence[str]] = None,
    regime_treated: Optional[Regime] = None,
    regime_control: Optional[Regime] = None,
    propensity_bounds: Tuple[float, float] = (0.01, 0.99),
    alpha: float = 0.05,
) -> LTMLESurvivalResult:
    """
    LTMLE for a discrete-time survival outcome under dynamic regimes.

    Parameters
    ----------
    data : DataFrame
        Wide-format, one row per subject.
    event_indicators : sequence of str, length K
        Column names for the per-interval event indicator ``T_k``
        (``1`` if the event occurs *in* interval k, ``0`` otherwise).
    treatments : sequence of str, length K
        Treatment column per interval.
    covariates_time : sequence of sequences of str, length K
        Time-varying covariates at each interval.
    baseline : list of str, optional
        Time-invariant baseline covariates.
    censoring : sequence of str, optional length K
        ``1`` if the subject is observed *through* interval k, ``0`` if
        right-censored at or before k. If omitted, no censoring.
    regime_treated, regime_control : static sequence or callable
        Treatment regimes, same semantics as :func:`ltmle`.
    propensity_bounds : tuple, default (0.01, 0.99)
    alpha : float, default 0.05

    Returns
    -------
    LTMLESurvivalResult
        Contains counterfactual survival curves, restricted-mean
        survival time (RMST) contrasts, and a terminal risk
        difference.

    Notes
    -----
    Estimation strategy:

    1. For each interval k, fit a pooled-logistic hazard
       :math:`h_k = P(T=k | T \\ge k, A_k, L_k, W)` on the at-risk set.
    2. Obtain counterfactual hazards under the regime by predicting
       at ``A_k = regime[k]``.
    3. Apply an LTMLE-style targeting correction at each interval
       using the clever covariate
       :math:`H_k = I(\\bar A_{1:k} = \\bar{regime})
                   / [\\prod_j g_j(regime_j | H_j)
                     \\prod_j P(C_j=1 | H_j, A_j)]`.
    4. Accumulate the survival curve
       :math:`S^{regime}(k) = \\prod_{j \\le k} (1 - h^{regime,*}_j)`.
    5. Report RMST =  :math:`\\sum_{k=1}^K S(k)` and the terminal risk
       difference :math:`S^{control}(K) - S^{treated}(K)`.

    This implementation uses logistic regression for nuisance models
    (self-contained). Advanced users can swap in the
    :class:`SuperLearner` by monkey-patching ``_fit_logit``.

    Notes
    -----
    **Influence function — Cai & van der Laan (2020) form.** The EIF for
    :math:`E[S^a(t)]` is

    .. math::

       D^*(O) = -\\sum_{j \\le t} H_j \\cdot \\frac{S^a(t)}{S^a(j)}
                                   \\cdot (T_j - h_j^*)
                + (S^a(t) - \\psi_S(t))

    The survival ratio :math:`S^a(t)/S^a(j)` is essential — without it
    earlier intervals' contributions are overweighted, inflating the
    SE. The RMST EIF is the per-:math:`k` sum of these. The terminal
    risk difference's EIF is the EIF of :math:`S^a(K)` *alone*, NOT the
    cross-:math:`k` RMST sum. v1.11.4 and earlier collapsed all
    intervals into a single IC and used it for both RMST and terminal
    RD, leading to over-conservative RMST SE *and* incorrect terminal
    SE. This module now computes the two ICs separately.

    **One-step targeting residual.** Like :func:`ltmle`, the targeting
    step uses a one-step linear approximation rather than fully
    iterated MLE. The targeting equation residual at each interval is
    therefore near-zero rather than identically zero, and the EIF
    expression above ignores that residual. Coverage may be slightly
    anti-conservative when nuisance models are flexible; use a CV-LTMLE
    workflow (not yet exposed) for critical inference.
    """
    event_indicators = list(event_indicators)
    treatments = list(treatments)
    covariates_time = [list(c) for c in covariates_time]
    baseline = list(baseline or [])
    K = len(event_indicators)
    if len(treatments) != K or len(covariates_time) != K:
        raise ValueError(
            "event_indicators, treatments, and covariates_time must "
            "all have the same length K"
        )
    if censoring is not None and len(censoring) != K:
        raise ValueError("censoring must have length K if provided")

    if regime_treated is None:
        regime_treated = [1] * K
    if regime_control is None:
        regime_control = [0] * K
    if not callable(regime_treated) and len(regime_treated) != K:
        raise ValueError("regime_treated must have length K or be callable")
    if not callable(regime_control) and len(regime_control) != K:
        raise ValueError("regime_control must have length K or be callable")

    df = data.copy().reset_index(drop=True)
    n = len(df)

    # ── Materialise regimes (same pattern as ltmle.py) ────────────────
    def _materialise(regime: Regime) -> np.ndarray:
        if not callable(regime):
            return np.tile(np.asarray(list(regime), dtype=int), (n, 1))
        mat = np.zeros((n, K), dtype=int)
        history: Dict[str, np.ndarray] = {
            c: df[c].to_numpy(dtype=float) for c in baseline
        }
        for k in range(K):
            for c in covariates_time[k]:
                history[c] = df[c].to_numpy(dtype=float)
            a = np.asarray(regime(k, history), dtype=int).reshape(-1)
            if a.size != n or not set(np.unique(a)).issubset({0, 1}):
                raise ValueError(
                    f"Dynamic regime at k={k} must return a length-{n} "
                    "binary vector."
                )
            mat[:, k] = a
            history[f"__regime_A_{k}"] = a.astype(float)
        return mat

    regime_treated_mat = _materialise(regime_treated)
    regime_control_mat = _materialise(regime_control)

    # ── Fit propensity and censoring models per interval ─────────────
    propensities: List[np.ndarray] = []
    cens_probs: List[np.ndarray] = []
    for k in range(K):
        hist_cols = list(baseline)
        for j in range(k):
            hist_cols += [treatments[j]] + covariates_time[j]
        hist_cols += covariates_time[k]
        X_k = (df[hist_cols].to_numpy(dtype=float)
               if hist_cols else np.ones((n, 0)))
        X_k = np.column_stack([np.ones(n), X_k])

        A_k = df[treatments[k]].to_numpy(dtype=int)
        model_g = _fit_logit(X_k, A_k)
        g_k = np.clip(_predict_proba(model_g, X_k), *propensity_bounds)
        propensities.append(g_k)

        if censoring is not None:
            X_c = np.column_stack([X_k, A_k.astype(float).reshape(-1, 1)])
            C_k = df[censoring[k]].to_numpy(dtype=int)
            if np.all(C_k == 1):
                cens_probs.append(np.ones(n))
            else:
                m_c = _fit_logit(X_c, C_k)
                p_c = np.clip(_predict_proba(m_c, X_c), *propensity_bounds)
                cens_probs.append(p_c)
        else:
            cens_probs.append(np.ones(n))

    # ── Fit & target discrete-time hazards under each regime ─────────
    def _run_regime(regime_mat: np.ndarray):
        """Returns per-subject sequences plus the population survival curve.

        Returned arrays (all per-subject):

        * ``survival_curve`` (K,) — population S^a(k) = E_n[S^a(k) | i]
          for k = 1..K
        * ``S_seq`` (n × K) — per-subject S^a(k) for k = 1..K
        * ``h_star_seq`` (n × K) — per-subject targeted hazard h_k^*(regime)
        * ``H_seq`` (n × K) — per-subject clever-covariate value (zero
          outside the regime-following + observed + at-risk indicator)
        * ``T_seq`` (n × K) — observed event indicator
        * ``mask_seq`` (n × K) — at-risk + observed mask (only these
          subjects contribute to the EIF score at each k)

        The downstream EIF computation uses these directly, so any
        target functional (S(K), RMST, RD at K) gets its own correct
        influence function — see :func:`_eif_survival_K` and
        :func:`_eif_rmst` below.
        """
        # cum_follow[i] = True while subject i has followed the regime
        # so far AND been uncensored AND event-free; once False it stays.
        cum_follow = np.ones(n, dtype=bool)
        cum_weight = np.ones(n)
        at_risk = np.ones(n, dtype=bool)   # hasn't had the event yet

        S_prev = np.ones(n)
        survival_curve = np.ones(K + 1)    # S(0) = 1
        S_seq = np.zeros((n, K))
        h_star_seq = np.zeros((n, K))
        H_seq = np.zeros((n, K))
        T_seq = np.zeros((n, K))
        mask_seq = np.zeros((n, K), dtype=bool)

        for k in range(K):
            hist_cols = list(baseline)
            for j in range(k):
                hist_cols += [treatments[j]] + covariates_time[j]
            hist_cols += covariates_time[k]
            X_hist = (df[hist_cols].to_numpy(dtype=float)
                      if hist_cols else np.ones((n, 0)))
            X_hist = np.column_stack([np.ones(n), X_hist])

            A_k = df[treatments[k]].to_numpy(dtype=int)
            a_tgt = regime_mat[:, k]

            # Subjects still at risk (no event, uncensored so far)
            if censoring is not None:
                C_k_obs = df[censoring[k]].to_numpy(dtype=int)
            else:
                C_k_obs = np.ones(n, dtype=int)
            T_k = df[event_indicators[k]].to_numpy(dtype=int)

            # Hazard regression on at-risk + uncensored subjects
            mask = at_risk & (C_k_obs == 1)
            if mask.sum() < 5:
                # Too few at risk — assume zero hazard and coast.
                h_hat = np.zeros(n)
                h_hat_regime = np.zeros(n)
            else:
                X_h = np.column_stack([X_hist[mask], A_k[mask].astype(float)])
                y_h = T_k[mask]
                m = _fit_logit(X_h, y_h)
                X_h_all = np.column_stack([X_hist, A_k.astype(float)])
                h_hat_raw = _predict_proba(m, X_h_all)
                X_h_regime = np.column_stack([X_hist, a_tgt.astype(float)])
                h_hat_regime = _predict_proba(m, X_h_regime)
                h_hat = h_hat_raw

            # Clever covariate — product of 1/g along regime & 1/p_c
            g_k = propensities[k]
            g_regime = np.where(a_tgt == 1, g_k, 1 - g_k)
            p_c = cens_probs[k]
            inc = 1.0 / np.maximum(g_regime, 1e-6) / np.maximum(p_c, 1e-6)
            new_cum_weight = cum_weight * inc

            # Targeting step on logit scale (pooled-logistic)
            indicator = cum_follow & (A_k == a_tgt) & (C_k_obs == 1) & at_risk
            H = np.where(indicator, new_cum_weight, 0.0)

            # Targeting is a one-step logistic update applied to the
            # *regime-counterfactual* hazard (the quantity whose
            # survival curve we report), not the observed-arm hazard.
            # The TMLE offset is therefore logit(h_hat_regime); the
            # residual uses the observed-arm h_hat only to score
            # epsilon against observed outcomes under the clever
            # covariate.
            resid = np.where(mask, T_k.astype(float) - h_hat, 0.0)
            denom = float(np.sum(H[mask] ** 2))
            if denom > 1e-10:
                eps = float(np.sum(H[mask] * resid[mask]) / denom)
            else:
                eps = 0.0
            offset_regime = _safe_logit(
                np.clip(h_hat_regime, 1e-6, 1 - 1e-6)
            )
            h_star_regime = expit(offset_regime + eps * new_cum_weight)

            # Survival update: S(k) = S(k-1) * (1 - h*)
            S_new = S_prev * (1.0 - h_star_regime)

            # Stash the per-subject sequences for the EIF post-pass.
            # The proper EIF for S^a(K) requires the survival ratio
            # S^a(K)/S^a(j); for RMST it requires the per-k partial sum
            # of those ratios. Both need every k's S, h*, H, T, mask —
            # which we now record rather than collapse to a single
            # cross-time sum.
            S_seq[:, k] = S_new
            h_star_seq[:, k] = h_star_regime
            H_seq[:, k] = H
            T_seq[:, k] = T_k
            mask_seq[:, k] = mask

            survival_curve[k + 1] = float(np.mean(S_new))

            # Update running state
            at_risk = at_risk & (T_k == 0) & (C_k_obs == 1)
            cum_follow = indicator
            cum_weight = new_cum_weight
            S_prev = S_new

        return (survival_curve[1:], S_seq, h_star_seq, H_seq,
                T_seq, mask_seq)

    def _eif_survival_at_k(target_k: int, S_seq: np.ndarray,
                            h_star_seq: np.ndarray, H_seq: np.ndarray,
                            T_seq: np.ndarray) -> np.ndarray:
        """EIF of E[S^a(target_k)] under the stacked nuisance fits.

        D*(O) = -∑_{j≤target_k} H_j · (T_j − h_j^*) · S^a(target_k)/S^a(j)
                + (S^a(target_k) − ψ_S(target_k))

        S_seq[:, j] is per-subject S^a(j+1) (1-indexed), so for the
        ratio at column j we use ``S_seq[:, j]`` ≈ S^a(j+1) — call
        sites compensate by using `target_k` 0-indexed (j ≤ target_k).
        """
        n = S_seq.shape[0]
        S_K = S_seq[:, target_k]
        ic = np.zeros(n)
        for j in range(target_k + 1):
            S_j = S_seq[:, j]
            # Ratio S^a(target_k)/S^a(j); guard against S_j ≈ 0.
            ratio = np.where(S_j > 1e-12, S_K / np.maximum(S_j, 1e-12), 0.0)
            ic -= H_seq[:, j] * (T_seq[:, j] - h_star_seq[:, j]) * ratio
        ic_centred = ic + (S_K - float(np.mean(S_K)))
        return ic_centred

    def _eif_rmst(S_seq: np.ndarray, h_star_seq: np.ndarray,
                   H_seq: np.ndarray, T_seq: np.ndarray) -> np.ndarray:
        """EIF of E[∑_{k=1}^K S^a(k)] = E[RMST^a].

        D*(O) = ∑_{k=1}^K D*_{S^a(k)}(O)  (linearity in target functional).
        """
        n = S_seq.shape[0]
        K_local = S_seq.shape[1]
        ic = np.zeros(n)
        for tk in range(K_local):
            ic += _eif_survival_at_k(tk, S_seq, h_star_seq, H_seq, T_seq)
        return ic

    surv_t, S_t_seq, h_t_seq, H_t_seq, T_t_seq, mask_t_seq = _run_regime(
        regime_treated_mat,
    )
    surv_c, S_c_seq, h_c_seq, H_c_seq, T_c_seq, mask_c_seq = _run_regime(
        regime_control_mat,
    )
    S_t = surv_t                  # population survival curve (treated)
    S_c = surv_c                  # population survival curve (control)

    rmst_t = float(np.sum(S_t))
    rmst_c = float(np.sum(S_c))
    rmst_diff = rmst_t - rmst_c

    # Proper RMST EIF: linear sum across k of S^a(k) EIFs, with the
    # full survival-product factor S^a(k)/S^a(j) (Cai & van der Laan
    # 2020). Earlier versions of this module dropped the survival
    # ratio and used the same cross-time IC for both RMST and the
    # terminal risk difference — see the v1.11.5 changelog audit note.
    ic_rmst_t = _eif_rmst(S_t_seq, h_t_seq, H_t_seq, T_t_seq)
    ic_rmst_c = _eif_rmst(S_c_seq, h_c_seq, H_c_seq, T_c_seq)
    diff_ic_rmst = ic_rmst_t - ic_rmst_c
    rmst_se = float(np.std(diff_ic_rmst, ddof=1) / np.sqrt(n))
    z = rmst_diff / rmst_se if rmst_se > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    rmst_ci = (rmst_diff - crit * rmst_se, rmst_diff + crit * rmst_se)

    # Terminal risk difference: its EIF is the EIF of S^a(K) only,
    # NOT the cumulative-across-K RMST EIF. v1.11.4 and earlier reused
    # the RMST IC here — overstating the SE since the RMST IC contains
    # contributions from all earlier intervals.
    rd_final = S_c[-1] - S_t[-1]
    ic_SK_t = _eif_survival_at_k(K - 1, S_t_seq, h_t_seq, H_t_seq, T_t_seq)
    ic_SK_c = _eif_survival_at_k(K - 1, S_c_seq, h_c_seq, H_c_seq, T_c_seq)
    rd_final_se = float(np.std(ic_SK_c - ic_SK_t, ddof=1) / np.sqrt(n))

    return LTMLESurvivalResult(
        times=np.arange(1, K + 1),
        survival_treated=np.asarray(S_t, dtype=float),
        survival_control=np.asarray(S_c, dtype=float),
        rmst_treated=rmst_t,
        rmst_control=rmst_c,
        rmst_difference=rmst_diff,
        rmst_se=rmst_se,
        rmst_ci=(float(rmst_ci[0]), float(rmst_ci[1])),
        rmst_pvalue=pval,
        risk_difference_final=float(rd_final),
        risk_difference_final_se=float(rd_final_se),
        K=K,
        n_obs=n,
        detail={
            "propensity_summary": [
                (float(p.min()), float(p.max())) for p in propensities
            ],
            "regime_treated_callable": callable(regime_treated),
            "regime_control_callable": callable(regime_control),
        },
    )


__all__ = ["ltmle_survival", "LTMLESurvivalResult"]
