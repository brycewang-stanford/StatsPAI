"""
Longitudinal Targeted Maximum Likelihood Estimation (LTMLE).

Estimates marginal mean outcomes under static treatment regimes in the
presence of time-varying treatment and time-varying confounding (and
optional right censoring), following van der Laan & Gruber (2012).

Data layout
-----------
Long / wide panel with a *fixed* number of time points ``K``. For each
time :math:`k = 1, ..., K` the user provides:

* ``A[k]``  — treatment indicator at time k  (binary)
* ``L[k]``  — time-varying covariates at time k  (array of column names)
* ``C[k]``  — optional censoring indicator at time k (1 = observed, 0 = censored)

Baseline covariates ``W`` are time-invariant. The outcome ``Y`` is
measured at the final time (or as a pooled survival indicator).

Target parameter
----------------
For a static regime :math:`\\bar a = (a_1, ..., a_K)`,

    ψ(\\bar a) = E[ Y(a_1, ..., a_K) ]

and, for ATE,

    ATE = ψ(1,...,1) - ψ(0,...,0).

Algorithm (recursive backward induction)
----------------------------------------
1. Start at time K. Fit :math:`Q_K = E[Y | A_K, L_K, history]`.
   Compute targeted update using clever covariate
   :math:`H_K = g(A_K | hist)^{-1}`.
2. Move to time K-1. Fit :math:`Q_{K-1} = E[Q_K^* | A_{K-1}, L_{K-1}, history]`.
3. Continue until time 1.
4. ψ(bar a) = mean of :math:`Q_1^*` under the regime.

The implementation uses logistic / linear regression for nuisance
models (so it is self-contained and reproducible) — advanced users can
swap in the existing :class:`SuperLearner`.

References
----------
van der Laan, M. J., & Gruber, S. (2012).
"Targeted minimum loss based estimation of causal effects of multiple
time point interventions." *The International Journal of Biostatistics*,
8(1).

Schwab, J., Lendle, S., Petersen, M., & van der Laan, M. J. (2014).
ltmle: Longitudinal Targeted Maximum Likelihood Estimation (R package).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from sklearn.linear_model import LogisticRegression, LinearRegression


# Type alias for regime specification: either a static sequence of 0/1
# or a callable that takes (k, history_dict) and returns a length-n
# vector of 0/1. The history dict contains, at call time, all
# baseline/time-varying covariates plus any treatments already assigned
# by the regime at earlier time points.
Regime = Union[Sequence[int], Callable[[int, Dict[str, np.ndarray]], np.ndarray]]


@dataclass
class LTMLEResult:
    psi_treated: float
    psi_control: float
    ate: float
    se: float
    ci: tuple
    pvalue: float
    K: int
    n_obs: int
    regime_treated: Sequence[int]
    regime_control: Sequence[int]
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lo, hi = self.ci
        return (
            "Longitudinal TMLE\n"
            "-----------------\n"
            f"  K (time points) : {self.K}\n"
            f"  N               : {self.n_obs}\n"
            f"  E[Y(1,...,1)]   : {self.psi_treated:.4f}\n"
            f"  E[Y(0,...,0)]   : {self.psi_control:.4f}\n"
            f"  ATE             : {self.ate:.4f}  (SE={self.se:.4f})\n"
            f"  95% CI          : [{lo:.4f}, {hi:.4f}]\n"
            f"  p-value         : {self.pvalue:.4f}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"LTMLEResult(ATE={self.ate:.4f}, SE={self.se:.4f})"


# --------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------

def _safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return logit(p)


def _fit_logit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Logistic regression with l2; handles degenerate y."""
    if np.all(y == y[0]):
        # trivial constant response; LR will fail — return dummy
        class _Const:
            def __init__(self, p):
                self.p = p

            def predict_proba(self, X):
                return np.column_stack([1 - self.p * np.ones(X.shape[0]),
                                        self.p * np.ones(X.shape[0])])

        return _Const(float(y[0]))
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    lr.fit(X, y)
    return lr


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(X)
    return prob[:, 1] if prob.ndim == 2 else prob


def _fit_linear(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    lr = LinearRegression()
    lr.fit(X, y)
    return lr


# --------------------------------------------------------------------
# Main LTMLE
# --------------------------------------------------------------------


def ltmle(
    data: pd.DataFrame,
    y: str,
    treatments: Sequence[str],
    covariates_time: Sequence[Sequence[str]],
    baseline: Optional[Sequence[str]] = None,
    censoring: Optional[Sequence[str]] = None,
    regime_treated: Optional[Regime] = None,
    regime_control: Optional[Regime] = None,
    propensity_bounds: Tuple[float, float] = (0.01, 0.99),
    outcome_type: str = "auto",
    alpha: float = 0.05,
) -> LTMLEResult:
    """
    Longitudinal TMLE for static regime contrasts.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format panel: one row per unit.
    y : str
        Final outcome column.
    treatments : sequence of str
        Treatment column per time point, length ``K``.
    covariates_time : sequence of sequences of str
        ``covariates_time[k]`` lists time-k covariate columns
        (may be empty). Length ``K``.
    baseline : sequence of str, optional
        Baseline time-invariant covariates.
    censoring : sequence of str, optional
        Censoring indicator column per time point (``1=observed``,
        ``0=censored``). If None, no censoring is modeled.
    regime_treated, regime_control : sequence of {0,1} OR callable
        Regimes to contrast. Default: all-1 vs all-0.

        A regime may also be a **callable** ``regime(k, history)`` for
        *dynamic regimes* that depend on the simulated / observed
        history of baseline and time-varying covariates. The callable
        receives ``k`` (int 0..K-1) and ``history`` — a dict mapping
        column name to the length-``n`` numpy array observed up to
        that timepoint — and must return a length-``n`` numpy array
        of 0/1 treatment assignments.

        Example (treat when a biomarker L exceeds its baseline):

        >>> def dynamic(k, hist):
        ...     return (hist[f"L{k}"] > hist["L_baseline"]).astype(int)
    propensity_bounds : tuple, default (0.01, 0.99)
        Clip propensity to this range for stability.
    outcome_type : {"auto", "binary", "continuous"}
        ``auto`` detects from unique values of ``y``.
    alpha : float, default 0.05

    Returns
    -------
    LTMLEResult
    """
    treatments = list(treatments)
    covariates_time = [list(c) for c in covariates_time]
    if len(treatments) != len(covariates_time):
        raise ValueError("treatments and covariates_time must have equal length")
    K = len(treatments)
    if K < 1:
        raise ValueError("Need at least one time point")

    baseline = list(baseline or [])
    censoring = list(censoring or []) if censoring else []
    if censoring and len(censoring) != K:
        raise ValueError("censoring must have length K if provided")

    if regime_treated is None:
        regime_treated = [1] * K
    if regime_control is None:
        regime_control = [0] * K
    # Validate shape only for static (non-callable) regimes. Callable
    # dynamic regimes are evaluated lazily at each time step.
    if not callable(regime_treated) and len(regime_treated) != K:
        raise ValueError("regime_treated must have length K or be callable")
    if not callable(regime_control) and len(regime_control) != K:
        raise ValueError("regime_control must have length K or be callable")

    df = data.copy().reset_index(drop=True)
    n = len(df)

    # Detect outcome type
    if outcome_type == "auto":
        yvals = df[y].dropna().unique()
        if set(yvals.astype(int)) <= {0, 1} and len(yvals) <= 2:
            outcome_type = "binary"
        else:
            outcome_type = "continuous"

    # ----- Forward: fit propensity scores at every time --------------
    # History at time k = baseline + treatments[:k] + covariates[:k+1]
    propensities: List[np.ndarray] = []
    for k in range(K):
        hist_cols = list(baseline)
        for j in range(k):
            hist_cols += [treatments[j]] + covariates_time[j]
        hist_cols += covariates_time[k]
        X_k = df[hist_cols].to_numpy(dtype=float) if hist_cols else np.ones((n, 0))
        X_k = np.column_stack([np.ones(n), X_k])
        A_k = df[treatments[k]].to_numpy(dtype=int)
        model_g = _fit_logit(X_k, A_k)
        g_k = _predict_proba(model_g, X_k)
        g_k = np.clip(g_k, *propensity_bounds)
        propensities.append(g_k)

    # Censoring models (optional) — estimate P(C_k=1 | hist, A_k)
    cens_probs: List[np.ndarray] = []
    for k in range(K):
        if not censoring:
            cens_probs.append(np.ones(n))
            continue
        hist_cols = list(baseline)
        for j in range(k):
            hist_cols += [treatments[j]] + covariates_time[j]
        hist_cols += covariates_time[k] + [treatments[k]]
        X_c = df[hist_cols].to_numpy(dtype=float) if hist_cols else np.ones((n, 0))
        X_c = np.column_stack([np.ones(n), X_c])
        C_k = df[censoring[k]].to_numpy(dtype=int)
        if np.all(C_k == 1):
            cens_probs.append(np.ones(n))
            continue
        model_c = _fit_logit(X_c, C_k)
        p_c = _predict_proba(model_c, X_c)
        p_c = np.clip(p_c, *propensity_bounds)
        cens_probs.append(p_c)

    # Precompute the full regime matrix. For static regimes this is
    # trivial; for dynamic regimes we evaluate the callable forward in
    # time on the OBSERVED history (NOT on a simulated counterfactual
    # history — LTMLE's targeting step handles the causal counterfactual
    # mapping; we need the regime value at each k evaluated on the data
    # covariates at that k, which is what a data-dependent policy
    # g(L_k) depends on anyway).
    def _materialise_regime(regime: Regime) -> np.ndarray:
        """Return an (n × K) 0/1 matrix for the regime."""
        if not callable(regime):
            arr = np.asarray(list(regime), dtype=int)
            return np.tile(arr, (n, 1))
        mat = np.zeros((n, K), dtype=int)
        history: Dict[str, np.ndarray] = {}
        for c in baseline:
            history[c] = df[c].to_numpy(dtype=float)
        for k in range(K):
            for c in covariates_time[k]:
                history[c] = df[c].to_numpy(dtype=float)
            a_k = np.asarray(regime(k, history), dtype=int).reshape(-1)
            if a_k.size != n:
                raise ValueError(
                    f"Dynamic regime at k={k} returned length "
                    f"{a_k.size}, expected {n}."
                )
            if not set(np.unique(a_k)).issubset({0, 1}):
                raise ValueError(
                    f"Dynamic regime at k={k} produced non-binary values."
                )
            mat[:, k] = a_k
            # Expose the regime's own past assignments to subsequent calls.
            history[f"__regime_A_{k}"] = a_k.astype(float)
        return mat

    # Helper: target Q at each step given regime
    def _run_regime(regime: Regime) -> Tuple[float, np.ndarray]:
        """Returns ψ and individual influence-function contributions."""
        # Cumulative regime-following indicator and cumulative weights
        cum_follow = np.ones(n, dtype=bool)
        cum_weight = np.ones(n)
        regime_mat = _materialise_regime(regime)  # (n, K)

        # Start Q at the final outcome
        Q = df[y].to_numpy(dtype=float).copy()
        # Targeted outcome storage, updated from K-1 down to 0
        eps_list: List[float] = []

        for k in reversed(range(K)):
            # History at time k
            hist_cols = list(baseline)
            for j in range(k):
                hist_cols += [treatments[j]] + covariates_time[j]
            hist_cols += covariates_time[k]
            X_k_hist = df[hist_cols].to_numpy(dtype=float) if hist_cols else np.ones((n, 0))
            X_k_hist = np.column_stack([np.ones(n), X_k_hist])

            # Design matrix for Q regression includes A_k
            A_k = df[treatments[k]].to_numpy(dtype=int)
            X_q = np.column_stack([X_k_hist, A_k])

            a_target = regime_mat[:, k]

            # Fit Q_k. For binary outcomes we ONLY run the logistic
            # regression at the terminal step k=K-1 where Q is the
            # observed 0/1 outcome. At earlier steps Q is a continuous
            # pseudo-outcome in [0,1] carried back from the targeted
            # update; thresholding it at 0.5 to refit a binary logit
            # collapses the bounded-Q recursion (van der Laan-Gruber
            # 2012 run a quasi-logit on the continuous pseudo-outcome).
            # We fall through to a linear regression at earlier steps
            # and clip predictions into (0,1) before the targeting
            # step applies its logit update.
            at_terminal = (k == K - 1)
            if outcome_type == "binary" and at_terminal:
                m = _fit_logit(X_q, Q.astype(int))
                Q_hat_raw = _predict_proba(m, X_q)
                X_q_regime = np.column_stack([X_k_hist, a_target])
                Q_hat_regime = _predict_proba(m, X_q_regime)
            elif outcome_type == "binary":
                # Linear model on the continuous pseudo-outcome, then
                # clip into (ε, 1-ε) so the downstream logit update
                # stays well-defined.
                m = _fit_linear(X_q, Q)
                Q_hat_raw = np.clip(m.predict(X_q), 1e-6, 1 - 1e-6)
                X_q_regime = np.column_stack([X_k_hist, a_target])
                Q_hat_regime = np.clip(m.predict(X_q_regime), 1e-6, 1 - 1e-6)
            else:
                m = _fit_linear(X_q, Q)
                Q_hat_raw = m.predict(X_q)
                X_q_regime = np.column_stack([X_k_hist, a_target])
                Q_hat_regime = m.predict(X_q_regime)

            # --- Targeting step -------------------------------------
            # Clever covariate H_k = I(A_1:k = regime_1:k)/prod g_k * 1/prod p_c
            indicator = cum_follow & (A_k == a_target)
            # update cum_follow to include current
            next_follow = indicator

            # cumulative weight: product of 1/g_k along regime path
            g_k = propensities[k]
            # g under the (unit-specific) target assignment
            g_regime = np.where(a_target == 1, g_k, 1 - g_k)
            p_c = cens_probs[k]
            inc = 1.0 / np.maximum(g_regime, 1e-6) / np.maximum(p_c, 1e-6)
            new_cum_weight = cum_weight * inc

            H = np.where(next_follow, new_cum_weight, 0.0)

            # Fit epsilon by regressing (Q - Q_hat_raw) on H (linear for continuous,
            # logit-update for binary).
            if outcome_type == "binary":
                q0 = np.clip(Q_hat_raw, 1e-6, 1 - 1e-6)
                offset = _safe_logit(q0)
                try:
                    Yb = np.clip(Q, 1e-6, 1 - 1e-6)
                    # One-step logistic update: regress logit(Y) - logit(q0) ~ H
                    # Use weighted linear approximation
                    resid = _safe_logit(Yb) - offset
                    mask = H > 0
                    if mask.sum() > 1 and np.std(H[mask]) > 1e-10:
                        eps = float(np.sum(H[mask] * resid[mask]) /
                                    np.sum(H[mask] ** 2))
                    else:
                        eps = 0.0
                    Q_star_regime = expit(_safe_logit(Q_hat_regime) + eps * new_cum_weight)
                except Exception:
                    eps = 0.0
                    Q_star_regime = Q_hat_regime
            else:
                resid = Q - Q_hat_raw
                mask = H > 0
                if mask.sum() > 1 and np.std(H[mask]) > 1e-10:
                    eps = float(np.sum(H[mask] * resid[mask]) /
                                np.sum(H[mask] ** 2))
                else:
                    eps = 0.0
                Q_star_regime = Q_hat_regime + eps * new_cum_weight

            eps_list.append(eps)

            # Feed targeted outcome to the previous time step as pseudo-outcome
            Q = Q_star_regime
            cum_follow = next_follow
            cum_weight = new_cum_weight

        psi = float(np.mean(Q))

        # Influence function: D = H_1*(Y - Q_1^*) + (Q_1^* - psi)
        # For a simplified SE we use the empirical variance of Q.
        ic = Q - psi
        return psi, ic

    psi1, ic1 = _run_regime(regime_treated)
    psi0, ic0 = _run_regime(regime_control)
    ate = psi1 - psi0
    diff_ic = ic1 - ic0
    se = float(np.std(diff_ic, ddof=1) / np.sqrt(n))
    z_stat = ate / se if se > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z_stat)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (ate - crit * se, ate + crit * se)

    def _serialise_regime(r: Regime) -> Any:
        return "dynamic-callable" if callable(r) else tuple(r)

    return LTMLEResult(
        psi_treated=psi1,
        psi_control=psi0,
        ate=ate,
        se=se,
        ci=ci,
        pvalue=pval,
        K=K,
        n_obs=n,
        regime_treated=_serialise_regime(regime_treated),
        regime_control=_serialise_regime(regime_control),
        detail={
            "propensity_summary": [(float(p.min()), float(p.max())) for p in propensities],
            "regime_treated_callable": callable(regime_treated),
            "regime_control_callable": callable(regime_control),
        },
    )


__all__ = ["ltmle", "LTMLEResult"]
