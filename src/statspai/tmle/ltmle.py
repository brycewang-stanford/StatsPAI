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
8(1). [@vanderlaan2012targeted]

Lendle, S. D., Schwab, J., Petersen, M. L., & van der Laan, M. J. (2017).
"ltmle: An R Package Implementing Targeted Minimum Loss-Based Estimation
for Longitudinal Data." *Journal of Statistical Software*, 81(1). [@lendle2017ltmle]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from .._result_serialize import ResultProtocolMixin

_LTMLE_ALTERNATIVES = ["sp.ltmle", "sp.tmle.ltmle", "sp.ltmle_survival"]


def _ltmle_error(
    message: str,
    *,
    diagnostics: Optional[Dict[str, Any]] = None,
    recovery_hint: str = "Check LTMLE column names and regime options.",
) -> MethodIncompatibility:
    return MethodIncompatibility(
        message,
        recovery_hint=recovery_hint,
        diagnostics=diagnostics,
        alternative_functions=_LTMLE_ALTERNATIVES,
    )


# Type alias for regime specification: either a static sequence of 0/1
# or a callable that takes (k, history_dict) and returns a length-n
# vector of 0/1. The history dict contains, at call time, all
# baseline/time-varying covariates plus any treatments already assigned
# by the regime at earlier time points.
Regime = Union[
    Sequence[int],
    Callable[[int, Dict[str, np.ndarray]], np.ndarray],
]


@dataclass
class LTMLEResult(ResultProtocolMixin):
    """Structured output of :func:`ltmle`.

    Holds the treated/control marginal means, their ATE contrast, and the
    associated inference (``se``, ``ci``, ``pvalue``).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> l0 = rng.normal(size=n)
    >>> a0 = rng.binomial(1, 1 / (1 + np.exp(-0.4 * l0)))
    >>> l1 = 0.3 * l0 + 0.2 * a0 + rng.normal(size=n)
    >>> a1 = rng.binomial(1, 1 / (1 + np.exp(-0.4 * l1)))
    >>> y = 1.0 + 0.4 * a0 + 0.3 * a1 + 0.2 * l1 + rng.normal(scale=0.2, size=n)
    >>> df = pd.DataFrame({"L0": l0, "A0": a0, "L1": l1, "A1": a1, "Y": y})
    >>> res = sp.ltmle(
    ...     df, y="Y", treatments=["A0", "A1"],
    ...     covariates_time=[["L0"], ["L1"]],
    ... )
    >>> isinstance(res, sp.LTMLEResult)
    True
    >>> float(res.ate)  # doctest: +SKIP
    0.71
    """

    _citation_keys = ("lendle2017ltmle", "vanderlaan2012targeted")

    psi_treated: float
    psi_control: float
    ate: float
    se: float
    ci: tuple[float, float]
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


def _safe_logit(p: Any, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.asarray(logit(p), dtype=float)


def _fit_logit(X: np.ndarray, y: np.ndarray) -> Any:
    """Logistic regression with l2; handles degenerate y."""
    if np.all(y == y[0]):
        # trivial constant response; LR will fail — return dummy
        class _Const:
            def __init__(self, p: float) -> None:
                self.p = p

            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                return np.column_stack(
                    [
                        1 - self.p * np.ones(X.shape[0]),
                        self.p * np.ones(X.shape[0]),
                    ]
                )

        return _Const(float(y[0]))
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
    lr.fit(X, y)
    return lr


def _predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(X)
    out = prob[:, 1] if prob.ndim == 2 else prob
    return np.asarray(out, dtype=float)


def _fit_linear(X: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.linear_model import LinearRegression

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

    Notes
    -----
    **SE caveat — simplified influence function.** The reported SE uses
    the in-sample empirical variance of the regime-marginalised
    pseudo-outcome :math:`Q_1^*` minus the plug-in estimate
    (``ic = Q - psi``). The full LTMLE EIF (van der Laan & Gruber 2012)
    is

    .. math::

       D^*(O) = \\sum_{k=1}^{K} H_k \\cdot \\mathbb{1}\\{\\bar A_{1:k}=\\bar a,
                                                         \\bar C_{1:k}=1\\}
                                  (Q_{k+1}^* - Q_k^*) + (Q_1^* - \\psi)

    The first sum is in-sample zero ONLY when the targeting equation
    has been iterated to convergence at every time point. This module
    uses a one-step linear (or quasi-logistic) approximation, so the
    sum is *near* zero but not identically zero. The reported SE
    therefore drops a finite-sample residual and is mildly
    **anti-conservative** when nuisance models are flexible — CI
    coverage may be below the nominal :math:`1-\\alpha`. For inference
    that requires honest coverage with rich ML nuisances, use the
    full CV-LTMLE / iterated-targeting path (not yet exposed; tracked
    as a follow-up).

    **Binary-outcome targeting** uses a one-step linear approximation
    to the Bernoulli MLE
    :math:`\\hat\\epsilon \\approx \\sum H \\cdot (\\mathrm{logit}(Y) -
    \\mathrm{logit}(\\hat Q)) / \\sum H^2`, which is accurate near
    :math:`\\epsilon=0` but biased for moderate :math:`\\epsilon`.

    **Targeting-step failures.** If the binary-outcome fluctuation step
    fails at a time point, :math:`\\epsilon` is set to 0 for that step
    (no targeting update — the estimate degrades toward untargeted
    g-computation at that time point); a ``ConvergenceWarning`` is
    emitted and the failed step indices are recorded per regime in
    ``detail['targeting_failures']``. Per-step epsilons (time order
    k=0..K-1) are exposed in ``detail['epsilons']``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> l0 = rng.normal(size=n)
    >>> a0 = rng.binomial(1, 1 / (1 + np.exp(-0.4 * l0)))
    >>> l1 = 0.3 * l0 + 0.2 * a0 + rng.normal(size=n)
    >>> a1 = rng.binomial(1, 1 / (1 + np.exp(-0.4 * l1)))
    >>> y = 1.0 + 0.4 * a0 + 0.3 * a1 + 0.2 * l1 + rng.normal(scale=0.2, size=n)
    >>> df = pd.DataFrame({"L0": l0, "A0": a0, "L1": l1, "A1": a1, "Y": y})
    >>> res = sp.ltmle(
    ...     df, y="Y", treatments=["A0", "A1"],
    ...     covariates_time=[["L0"], ["L1"]],
    ... )
    >>> bool(np.isfinite(res.ate))
    True
    """
    if not isinstance(data, pd.DataFrame):
        raise _ltmle_error(
            "ltmle data must be a pandas DataFrame.",
            diagnostics={"type": type(data).__name__},
            recovery_hint="Pass a wide-format pandas DataFrame.",
        )
    treatments = list(treatments)
    covariates_time = [list(c) for c in covariates_time]
    if len(treatments) != len(covariates_time):
        raise _ltmle_error(
            "treatments and covariates_time must have equal length.",
            diagnostics={
                "n_treatments": len(treatments),
                "n_covariate_blocks": len(covariates_time),
            },
            recovery_hint="Pass one covariate block for each treatment time.",
        )
    K = len(treatments)
    if K < 1:
        raise DataInsufficient(
            "ltmle needs at least one time point.",
            recovery_hint="Pass at least one treatment column.",
            diagnostics={"K": K},
            alternative_functions=_LTMLE_ALTERNATIVES,
        )

    baseline = list(baseline or [])
    censoring = list(censoring or []) if censoring else []
    if censoring and len(censoring) != K:
        raise _ltmle_error(
            "censoring must have length K if provided.",
            diagnostics={"K": K, "n_censoring": len(censoring)},
            recovery_hint="Pass one censoring indicator per treatment time.",
        )
    if outcome_type not in {"auto", "binary", "continuous"}:
        raise _ltmle_error(
            "outcome_type must be 'auto', 'binary', or 'continuous'.",
            diagnostics={"outcome_type": outcome_type},
            recovery_hint="Use outcome_type='auto', 'binary', or 'continuous'.",
        )
    if not (0 < alpha < 1):
        raise _ltmle_error(
            f"alpha must be in (0, 1), got {alpha}.",
            diagnostics={"alpha": alpha},
            recovery_hint="Use a confidence level such as alpha=0.05.",
        )
    if len(propensity_bounds) != 2:
        raise _ltmle_error(
            "propensity_bounds must contain exactly two values.",
            diagnostics={"propensity_bounds": list(propensity_bounds)},
            recovery_hint="Use propensity_bounds=(0.01, 0.99).",
        )
    p_lo, p_hi = float(propensity_bounds[0]), float(propensity_bounds[1])
    if not (0 < p_lo < p_hi < 1):
        raise _ltmle_error(
            "propensity_bounds must satisfy 0 < lower < upper < 1.",
            diagnostics={"propensity_bounds": [p_lo, p_hi]},
            recovery_hint="Use bounds such as (0.01, 0.99).",
        )
    propensity_bounds = (p_lo, p_hi)

    required = [y] + treatments + baseline + censoring
    for block in covariates_time:
        required.extend(block)
    missing = set(required) - set(data.columns)
    if missing:
        raise _ltmle_error(
            f"Missing columns: {missing}",
            diagnostics={"missing_columns": sorted(str(col) for col in missing)},
            recovery_hint="Pass LTMLE column names present in the DataFrame.",
        )

    if regime_treated is None:
        regime_treated = [1] * K
    if regime_control is None:
        regime_control = [0] * K
    # Validate shape only for static (non-callable) regimes. Callable
    # dynamic regimes are evaluated lazily at each time step.
    if not callable(regime_treated) and len(regime_treated) != K:
        raise _ltmle_error(
            "regime_treated must have length K or be callable.",
            diagnostics={"K": K, "regime_length": len(regime_treated)},
            recovery_hint="Pass a static regime with one value per time point.",
        )
    if not callable(regime_control) and len(regime_control) != K:
        raise _ltmle_error(
            "regime_control must have length K or be callable.",
            diagnostics={"K": K, "regime_length": len(regime_control)},
            recovery_hint="Pass a static regime with one value per time point.",
        )

    df = data.copy().reset_index(drop=True)
    n = len(df)
    if n < 2:
        raise DataInsufficient(
            "ltmle requires at least two observations.",
            recovery_hint="Provide more rows for longitudinal TMLE.",
            diagnostics={"n": n},
            alternative_functions=_LTMLE_ALTERNATIVES,
        )
    # Guard an all-/mostly-NaN outcome up front: otherwise the nuisance fit
    # leaks a cryptic sklearn ``ValueError: Input y contains NaN`` instead of a
    # StatsPAI message naming the outcome (censoring is handled via the
    # ``censoring`` argument, not NaN outcomes).
    if int(df[y].notna().sum()) < 2:
        raise DataInsufficient(
            f"ltmle: outcome '{y}' has fewer than two non-missing values; "
            "the nuisance models cannot be fit.",
            recovery_hint="Provide a non-missing outcome; encode dropout via "
            "the `censoring` argument rather than NaN outcomes.",
            diagnostics={"n_nonmissing_outcome": int(df[y].notna().sum())},
            alternative_functions=_LTMLE_ALTERNATIVES,
        )

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
            hist_cols += [treatments[j]] + list(covariates_time[j])
        hist_cols += list(covariates_time[k])
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
            hist_cols += [treatments[j]] + list(covariates_time[j])
        hist_cols += list(covariates_time[k]) + [treatments[k]]
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
                raise _ltmle_error(
                    f"Dynamic regime at k={k} returned length "
                    f"{a_k.size}, expected {n}.",
                    diagnostics={"k": k, "length": a_k.size, "expected": n},
                    recovery_hint="Return one treatment assignment per row.",
                )
            if not set(np.unique(a_k)).issubset({0, 1}):
                raise _ltmle_error(
                    f"Dynamic regime at k={k} produced non-binary values.",
                    diagnostics={"k": k, "values": np.unique(a_k).tolist()},
                    recovery_hint="Return only 0/1 treatment assignments.",
                )
            mat[:, k] = a_k
            # Expose the regime's own past assignments to subsequent calls.
            history[f"__regime_A_{k}"] = a_k.astype(float)
        return mat

    # Helper: target Q at each step given regime
    def _run_regime(regime: Regime) -> Tuple[float, np.ndarray, List[float], List[int]]:
        """Returns ψ, influence-function contributions, per-step
        epsilons (time order k=0..K-1) and failed targeting steps."""
        # Cumulative regime-following indicator and cumulative weights
        cum_follow = np.ones(n, dtype=bool)
        cum_weight = np.ones(n)
        regime_mat = _materialise_regime(regime)  # (n, K)

        # Start Q at the final outcome
        Q = df[y].to_numpy(dtype=float).copy()
        # Targeted outcome storage, updated from K-1 down to 0
        eps_list: List[float] = []
        targeting_failures: List[int] = []

        for k in reversed(range(K)):
            # History at time k
            hist_cols = list(baseline)
            for j in range(k):
                hist_cols += [treatments[j]] + list(covariates_time[j])
            hist_cols += list(covariates_time[k])
            X_k_hist = (
                df[hist_cols].to_numpy(dtype=float) if hist_cols else np.ones((n, 0))
            )
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
            at_terminal = k == K - 1
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
            # Clever covariate H_k = I(A_1:k = regime_1:k, C_1:k = 1) /
            #                       prod_j g_j(regime) * prod_j P(C_j=1).
            # The censoring indicator MUST gate the regime-following
            # mask: previously this code used 1/p_c to inflate weights
            # but did not exclude censored units from H, so censored
            # rows continued contributing to the targeting equation
            # past their censoring time with arbitrarily large weights.
            # ltmle_survival.py already does it correctly; ltmle.py is
            # being brought into line.
            if censoring:
                C_k_obs = df[censoring[k]].to_numpy(dtype=int)
            else:
                C_k_obs = np.ones(n, dtype=int)
            indicator = cum_follow & (A_k == a_target) & (C_k_obs == 1)
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
                        eps = float(
                            np.sum(H[mask] * resid[mask]) / np.sum(H[mask] ** 2)
                        )
                    else:
                        eps = 0.0
                    Q_star_regime = expit(
                        _safe_logit(Q_hat_regime) + eps * new_cum_weight
                    )
                except Exception as exc:
                    eps = 0.0
                    Q_star_regime = Q_hat_regime
                    targeting_failures.append(k)
                    warnings.warn(
                        f"ltmle: targeting step k={k} failed "
                        f"({type(exc).__name__}: {exc}); epsilon set to 0 "
                        "(no targeting update at this time point — the "
                        "estimate degrades toward untargeted "
                        "g-computation).",
                        ConvergenceWarning,
                        stacklevel=3,
                    )
            else:
                resid = Q - Q_hat_raw
                mask = H > 0
                if mask.sum() > 1 and np.std(H[mask]) > 1e-10:
                    eps = float(np.sum(H[mask] * resid[mask]) / np.sum(H[mask] ** 2))
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
        # eps_list was appended k=K-1..0; reverse into time order.
        return psi, ic, eps_list[::-1], sorted(targeting_failures)

    psi1, ic1, eps1, fail1 = _run_regime(regime_treated)
    psi0, ic0, eps0, fail0 = _run_regime(regime_control)
    ate = psi1 - psi0
    diff_ic = ic1 - ic0
    se = float(np.std(diff_ic, ddof=1) / np.sqrt(n))
    z_stat = ate / se if se > 0 else 0.0
    pval = float(2 * stats.norm.sf(abs(z_stat)))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (ate - crit * se, ate + crit * se)

    def _serialise_regime(r: Regime) -> Any:
        return "dynamic-callable" if callable(r) else tuple(r)

    _result = LTMLEResult(
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
            "propensity_summary": [
                (float(p.min()), float(p.max())) for p in propensities
            ],
            "regime_treated_callable": callable(regime_treated),
            "regime_control_callable": callable(regime_control),
            # Per-step targeting epsilons (time order k=0..K-1) and the
            # steps where the binary fluctuation failed (eps forced to 0).
            "epsilons": {"treated": eps1, "control": eps0},
            "targeting_failures": {"treated": fail1, "control": fail0},
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.tmle.ltmle",
            params={
                "y": y,
                "treatments": list(treatments),
                "covariates_time": [list(c) for c in covariates_time],
                "baseline": list(baseline) if baseline else None,
                "censoring": list(censoring) if censoring else None,
                "regime_treated_callable": callable(regime_treated),
                "regime_control_callable": callable(regime_control),
                "propensity_bounds": list(propensity_bounds),
                "outcome_type": outcome_type,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


__all__ = ["ltmle", "LTMLEResult"]
