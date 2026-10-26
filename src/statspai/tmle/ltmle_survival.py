"""
Longitudinal TMLE for survival outcomes (discrete-time pooled hazards).

Estimates the counterfactual survival function ``S^{a}(t) = P(T > t | do(A))``
under static or dynamic treatment regimes, with optional right censoring,
via a discrete-time pooled-logistic hazard model targeted at each time
interval.

This is the survival analogue of :func:`ltmle`. For each horizon ``t``
the risk ``P(T^a <= t)`` is a longitudinal mean of the cumulative event
indicator, estimated by the sequential-regression (iterated conditional
expectation) TMLE of van der Laan & Gruber (2012), with the survival
structure built in: a unit whose event has occurred keeps outcome 1, so
it is not refitted at later intervals (R ``ltmle`` with
``survivalOutcome = TRUE``).

Data layout
-----------
Wide-format DataFrame with one row per subject and columns:

* ``T_k`` for k=1..K — the *event indicator* at interval k
  (1 if event at k, 0 otherwise)
* ``C_k`` for k=1..K — the *observed-at-k indicator* (optional)
* ``A_k`` for k=1..K — treatment at interval k
* ``L_k`` — time-varying covariates at interval k

Subjects enter the regressions at interval k only while they are at risk
(no event before k) and uncensored. An indicator may also be cumulative
(1 from the event onwards); values after the event are ignored.

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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from .._result_serialize import ResultProtocolMixin
from .ltmle import _fluctuate, _logit_fit

Regime = Union[Sequence[int], Callable[[int, Dict[str, np.ndarray]], np.ndarray]]


@dataclass
class LTMLESurvivalResult(ResultProtocolMixin):
    """Counterfactual survival curves and contrasts.

    Produced by :func:`ltmle_survival`. Holds the treated/control
    counterfactual survival curves, the restricted-mean survival time
    (RMST) contrast with its standard error and confidence interval,
    and the terminal risk difference. Use :meth:`to_frame` for a tidy
    per-interval survival table.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n, K = 200, 3
    >>> A = rng.integers(0, 2, size=(n, K))
    >>> L = rng.normal(0, 1, size=(n, K))
    >>> T = np.zeros((n, K), dtype=int)
    >>> for k in range(K):
    ...     haz = 1 / (1 + np.exp(-(-1.5 - 0.5 * A[:, k] + 0.3 * L[:, k])))
    ...     T[:, k] = (rng.uniform(size=n) < haz).astype(int)
    >>> df = pd.DataFrame({
    ...     **{f"T{k+1}": T[:, k] for k in range(K)},
    ...     **{f"A{k+1}": A[:, k] for k in range(K)},
    ...     **{f"L{k+1}": L[:, k] for k in range(K)},
    ... })
    >>> res = sp.ltmle_survival(
    ...     df,
    ...     event_indicators=["T1", "T2", "T3"],
    ...     treatments=["A1", "A2", "A3"],
    ...     covariates_time=[["L1"], ["L2"], ["L3"]],
    ... )
    >>> isinstance(res, sp.LTMLESurvivalResult)
    True
    >>> len(res.times)
    3
    """

    _citation_keys = (
        "vanderlaan2012targeted",
        "stitelman2010collaborative",
        "cai2020step",
    )

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
        return pd.DataFrame(
            {
                "time": self.times,
                "S_treated": self.survival_treated,
                "S_control": self.survival_control,
                "risk_diff": self.survival_control - self.survival_treated,
            }
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n, K = 200, 3
    >>> A = rng.integers(0, 2, size=(n, K))
    >>> L = rng.normal(0, 1, size=(n, K))
    >>> T = np.zeros((n, K), dtype=int)
    >>> for k in range(K):
    ...     haz = 1 / (1 + np.exp(-(-1.5 - 0.5 * A[:, k] + 0.3 * L[:, k])))
    ...     T[:, k] = (rng.uniform(size=n) < haz).astype(int)
    >>> df = pd.DataFrame({
    ...     **{f"T{k+1}": T[:, k] for k in range(K)},
    ...     **{f"A{k+1}": A[:, k] for k in range(K)},
    ...     **{f"L{k+1}": L[:, k] for k in range(K)},
    ... })
    >>> res = sp.ltmle_survival(
    ...     df,
    ...     event_indicators=["T1", "T2", "T3"],
    ...     treatments=["A1", "A2", "A3"],
    ...     covariates_time=[["L1"], ["L2"], ["L3"]],
    ... )
    >>> res.K
    3
    >>> res.n_obs
    200
    >>> bool(0.0 <= res.survival_treated[-1] <= 1.0)
    True
    >>> list(res.to_frame().columns)
    ['time', 'S_treated', 'S_control', 'risk_diff']

    References
    ----------
    [@vanderlaan2012targeted], [@stitelman2010collaborative], [@cai2020step]

    Notes
    -----
    **Algorithm.** For each horizon ``t`` the risk ``P(T^a <= t)`` is the
    mean of the cumulative event indicator ``Ybar_t`` under the regime,
    estimated by sequential regression as in :func:`ltmle` (van der Laan &
    Gruber 2012). Backwards over ``k = t, ..., 1``:

    1. Fit a main-terms logistic (quasi-binomial) regression of the next
       pseudo-outcome (``Ybar_t`` at ``k = t``) on the history through
       ``A_k`` among units at risk at ``k`` (no earlier event) and
       uncensored through ``k``; predict with every ``A_j``, ``j <= k``,
       set to the regime (a dynamic regime is evaluated on the observed
       history), bounded to ``[1e-4, 0.9999]``.
    2. Target it with an intercept-only logistic fluctuation, offset the
       logit of the prediction, weights ``1 / prod_{j<=k} g_j`` over the
       units that follow the regime and are uncensored through ``k``.
    3. A unit whose event happened before ``k`` keeps pseudo-outcome 1
       (the survival structure; R ``ltmle``'s ``survivalOutcome = TRUE``).

    ``S(t) = 1 - mean(Q*_1)``; its influence curve is minus
    ``sum_k H_k (Q*_{k+1} - Q*_k) + Q*_1 - risk``. RMST is ``sum_t S(t)``
    with the summed influence curves; the terminal risk difference uses
    the influence curves of ``S(K)`` alone. Treatment and censoring models
    are main-terms logistic regressions on the at-risk units, the
    treatment probabilities bounded by ``propensity_bounds``.

    Before the Oct 2026 scan this function fitted one hazard per interval
    and predicted it with only the *current* treatment set to the regime,
    earlier treatments at their observed values, which is biased whenever
    past treatment affects later hazards (on a two-interval randomised
    design with a lagged effect the treated survival at ``t = 2`` was
    about 0.06 too low).
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
    lo_b, hi_b = propensity_bounds

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

    # ── Observed status ───────────────────────────────────────────────
    # ``event_before[:, k]``: the event happened in an interval before k.
    # Once it has, later columns (treatment, covariates, censoring, event)
    # carry no information and may be missing.
    T_raw = np.column_stack([df[c].to_numpy(dtype=float) for c in event_indicators])
    T_obs = np.where(np.isfinite(T_raw), T_raw, 0.0)
    cum_event = np.maximum.accumulate(T_obs, axis=1)
    event_before = np.column_stack([np.zeros(n), cum_event[:, :-1]]) > 0
    at_risk = ~event_before
    if censoring is not None:
        C_obs = np.column_stack([df[c].to_numpy(dtype=float) == 1 for c in censoring])
        uncens = np.cumprod(C_obs | event_before, axis=1).astype(bool)
    else:
        uncens = np.ones((n, K), dtype=bool)
    uncens_before = np.column_stack([np.ones(n, dtype=bool), uncens[:, :-1]])
    observed = at_risk & uncens  # event indicator of interval k is observed
    if np.any(observed & ~np.isfinite(T_raw)):
        raise ValueError(
            "ltmle_survival: an event indicator is missing for a unit that "
            "is at risk and uncensored in that interval."
        )
    A_obs = np.column_stack([df[a].to_numpy(dtype=float) for a in treatments])

    def _hist_cols(k: int) -> List[str]:
        cols = list(baseline)
        for j in range(k):
            cols += [treatments[j]] + list(covariates_time[j])
        return cols + list(covariates_time[k])

    def _design(k: int) -> Tuple[np.ndarray, List[str]]:
        cols = _hist_cols(k) + [treatments[k]]
        X = df[cols].to_numpy(dtype=float)
        bad = ~np.isfinite(X).all(axis=1) & at_risk[:, k] & uncens_before[:, k]
        if bad.any():
            raise ValueError(
                f"ltmle_survival: missing history or treatment values at "
                f"interval {k + 1} for {int(bad.sum())} unit(s) still at risk "
                "and uncensored."
            )
        # Rows after an event or censoring are never fitted or used.
        return np.column_stack([np.ones(n), np.nan_to_num(X)]), cols

    def _fit_predict(
        X: np.ndarray, y: np.ndarray, rows: np.ndarray, X_pred: np.ndarray
    ) -> np.ndarray:
        if not rows.any():
            return np.full(len(X_pred), 0.5)
        yr = y[rows]
        if np.all(yr == yr[0]):
            return np.full(len(X_pred), float(yr[0]))
        beta = _logit_fit(X[rows], yr)
        return np.asarray(expit(X_pred @ beta), dtype=float)

    # ── Treatment and censoring models on the at-risk, uncensored set ──
    prob_a1: List[np.ndarray] = []
    prob_c1: List[np.ndarray] = []
    for k in range(K):
        X_k, _ = _design(k)
        X_g = X_k[:, :-1]  # history through L_k, without A_k
        rows = at_risk[:, k] & uncens_before[:, k]
        p_a = _fit_predict(X_g, A_obs[:, k], rows, X_g)
        prob_a1.append(np.clip(p_a, lo_b, hi_b))
        if censoring is not None:
            c_k = (df[censoring[k]].to_numpy(dtype=float) == 1).astype(float)
            p_c = _fit_predict(X_k, c_k, rows, X_k)
            prob_c1.append(np.clip(p_c, lo_b, 1.0))
        else:
            prob_c1.append(np.ones(n))

    # ── Sequential-regression TMLE, one horizon at a time ─────────────
    def _risk_at(
        t: int, regime_mat: np.ndarray
    ) -> Tuple[float, np.ndarray, List[float]]:
        """TMLE of P(event by interval t+1 under the regime) (0-indexed t),
        with its influence curve and the targeting epsilons."""
        g_step = np.column_stack(
            [
                np.where(regime_mat[:, k] == 1, prob_a1[k], 1 - prob_a1[k]) * prob_c1[k]
                for k in range(t + 1)
            ]
        )
        cum_g = np.cumprod(g_step, axis=1)
        follow = np.cumprod(A_obs[:, : t + 1] == regime_mat[:, : t + 1], axis=1)
        follow = follow.astype(bool)

        Q_next = cum_event[:, t].astype(float)  # observed outcome at t
        ic = np.zeros(n)
        eps_list: List[float] = []
        for k in reversed(range(t + 1)):
            X_obs, cols = _design(k)
            X_reg = X_obs.copy()
            for j in range(k + 1):
                X_reg[:, 1 + cols.index(treatments[j])] = regime_mat[:, j]
            # Units still at risk at k whose next outcome is known: the
            # observed event (k = t) or the targeted prediction from k + 1.
            rows = at_risk[:, k] & uncens[:, k]
            q_pred = np.clip(_fit_predict(X_obs, Q_next, rows, X_reg), 1e-4, 0.9999)
            off = logit(q_pred)
            subs = rows & follow[:, k]
            w = np.where(subs, 1.0 / cum_g[:, k], 0.0)
            eps = _fluctuate(Q_next[subs], off[subs], w[subs])
            q_star = expit(off + eps)
            ic = ic + np.where(subs, (Q_next - q_star) / cum_g[:, k], 0.0)
            eps_list.append(float(eps))
            # An event before k fixes the outcome at 1.
            Q_next = np.where(at_risk[:, k], q_star, 1.0)
        psi = float(np.mean(Q_next))
        return psi, ic + Q_next - psi, eps_list[::-1]

    def _curve(regime_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
        """Survival S(t) = 1 - risk(t), t = 1..K, and its influence curves
        (n x K)."""
        surv = np.empty(K)
        ics = np.empty((n, K))
        eps_all = []
        for t in range(K):
            risk, ic_risk, eps = _risk_at(t, regime_mat)
            surv[t] = 1.0 - risk
            ics[:, t] = -ic_risk
            eps_all.append(eps)
        return surv, ics, eps_all

    S_t, IC_t, eps_t = _curve(regime_treated_mat)
    S_c, IC_c, eps_c = _curve(regime_control_mat)

    rmst_t = float(np.sum(S_t))
    rmst_c = float(np.sum(S_c))
    rmst_diff = rmst_t - rmst_c
    # RMST is a sum of survival probabilities, so its influence curve is
    # the sum of theirs.
    diff_ic_rmst = IC_t.sum(axis=1) - IC_c.sum(axis=1)
    rmst_se = float(np.std(diff_ic_rmst, ddof=1) / np.sqrt(n))
    z = rmst_diff / rmst_se if rmst_se > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    rmst_ci = (rmst_diff - crit * rmst_se, rmst_diff + crit * rmst_se)

    # Terminal risk difference (treated minus control risk) and its SE
    # from the influence curves of S(K) alone.
    rd_final = S_c[-1] - S_t[-1]
    rd_final_se = float(np.std(IC_c[:, -1] - IC_t[:, -1], ddof=1) / np.sqrt(n))

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
            "propensity_summary": [(float(p.min()), float(p.max())) for p in prob_a1],
            "regime_treated_callable": callable(regime_treated),
            "regime_control_callable": callable(regime_control),
            "algorithm": "sequential-regression TMLE per horizon (survival outcome)",
            "survival_se_treated": IC_t.std(axis=0, ddof=1) / np.sqrt(n),
            "survival_se_control": IC_c.std(axis=0, ddof=1) / np.sqrt(n),
            "epsilons_treated": eps_t,
            "epsilons_control": eps_c,
        },
    )


__all__ = ["ltmle_survival", "LTMLESurvivalResult"]
