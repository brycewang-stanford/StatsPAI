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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility

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


def _logit_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unpenalised (quasi-)binomial logistic MLE by Newton-Raphson.

    ``y`` may be fractional in [0, 1]: the quasi-binomial score equations
    are the binomial ones, so this is R's
    ``glm(family = quasibinomial())`` fit. Rank-deficient designs are
    handled with a pseudo-inverse step (R drops aliased columns; the
    fitted values agree).
    """
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        mu = expit(X @ beta)
        W = mu * (1 - mu)
        grad = X.T @ (y - mu)
        info = (X * W[:, None]).T @ X
        step = np.linalg.lstsq(info, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    else:
        warnings.warn(
            "ltmle: a logistic regression did not converge in 100 Newton "
            "steps (separation?).",
            ConvergenceWarning,
            stacklevel=3,
        )
    return beta


def _logit_fit_predict(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Fit on ``rows`` and return fitted probabilities for every unit."""
    if np.all(y[rows] == y[rows][0]):
        return np.full(len(y), float(y[rows][0]))
    return expit(X @ _logit_fit(X[rows], y[rows]))


def _fluctuate(q_next: np.ndarray, off: np.ndarray, w: np.ndarray) -> float:
    """Intercept-only weighted logistic fluctuation with offset.

    Solves ``sum w (q_next - expit(off + eps)) = 0`` for ``eps`` (R:
    ``glm(Q.kplus1 ~ -1 + S1 + offset(off), weights = w,
    family = quasibinomial)``).
    """
    if q_next.size == 0 or not np.any(w > 0):
        return 0.0
    eps = 0.0
    for _ in range(100):
        mu = expit(off + eps)
        score = float(np.sum(w * (q_next - mu)))
        info = float(np.sum(w * mu * (1 - mu)))
        step = score / info
        eps += step
        if abs(step) < 1e-13:
            break
    else:
        raise ValueError("fluctuation did not converge")
    return eps


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
    propensity_bounds: Tuple[float, float] = (0.01, 1.0),
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

    propensity_bounds : tuple, default (0.01, 1.0)
        Bounds on the *cumulative* probability of following the regime
        (and remaining uncensored) up to each time point, ``prod_j g_j``
        -- the ``gbounds`` of R ``ltmle``, whose default this is.
    outcome_type : {"auto", "binary", "continuous"}
        ``auto`` detects from unique values of ``y``.
    alpha : float, default 0.05

    Returns
    -------
    LTMLEResult

    Notes
    -----
    **Algorithm.** Sequential regression TMLE of van der Laan & Gruber
    (2012) as implemented in R ``ltmle`` (non-stratified, ``glm``
    learners). A continuous outcome is mapped to ``[0, 1]`` by
    ``(Y - min Y) / (max Y - min Y)``. Backwards over ``k = K-1, ..., 0``:

    1. Fit a logistic (quasi-binomial) regression of the current
       pseudo-outcome ``Q*_{k+1}`` on the full history up to ``A_k``
       among units uncensored through ``k``; predict for everyone with
       ``A_0..A_k`` set to the regime, bounded to ``[1e-4, 0.9999]``.
    2. Target it: an intercept-only logistic fluctuation with offset
       ``logit Q_k`` and weights ``1 / prod_{j<=k} g_j`` (bounded by
       ``propensity_bounds``) over units that follow the regime and are
       uncensored through ``k``; the fitted intercept solves the
       efficient-score equation exactly.
    3. Add ``H_k (Q*_{k+1} - Q*_k)`` to the influence curve, with
       ``H_k = I(follow, uncensored) / prod_{j<=k} g_j``.

    ``psi = mean(Q*_0)``; the SE is ``sd(IC_treated - IC_control) /
    sqrt(n)`` (R ``ltmle``'s ``variance.method = "ic"``). Treatment and
    censoring models are main-terms logistic regressions on the full
    observed history.

    Before 1.29 this function fitted linear models to the pseudo-
    outcomes, set only the current treatment to the regime, and targeted
    binary outcomes with a linearised update on ``logit(Y)`` (``Y`` in
    ``{0, 1}`` clipped to ``1e-6``, i.e. ``+/-13.8``). On a two-period
    binary fixture it returned an ATE of 0.757 where R ``ltmle`` returns
    0.373; the continuous path was a different (consistent) estimator.

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
    if not (0 < p_lo < p_hi <= 1):
        raise _ltmle_error(
            "propensity_bounds must satisfy 0 < lower < upper <= 1.",
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

    # Outcome scale: continuous Y is mapped to [0, 1] (R ltmle, Yrange=NULL).
    Y_raw = df[y].to_numpy(dtype=float)
    if outcome_type == "continuous":
        y_min = float(np.nanmin(Y_raw))
        y_range = float(np.nanmax(Y_raw) - y_min)
        if y_range <= 0:
            raise _ltmle_error(
                f"ltmle: outcome '{y}' is constant.",
                diagnostics={"y_min": y_min},
                recovery_hint="Provide a non-degenerate outcome.",
            )
    else:
        y_min, y_range = 0.0, 1.0
    Y_scaled = (Y_raw - y_min) / y_range

    def _hist_cols(k: int) -> List[str]:
        cols = list(baseline)
        for j in range(k):
            cols += list(covariates_time[j]) + [treatments[j]]
        cols += list(covariates_time[k])
        return cols

    # Uncensored-through-k indicators (C_k observed after A_k).
    if censoring:
        C_obs = np.column_stack([df[c].to_numpy(dtype=float) == 1 for c in censoring])
        uncens = np.cumprod(C_obs, axis=1).astype(bool)
    else:
        uncens = np.ones((n, K), dtype=bool)
    uncens_before = np.column_stack([np.ones(n, dtype=bool), uncens[:, :-1]])

    # ----- Forward: treatment and censoring models (observed history) ---
    A_obs = np.column_stack([df[a].to_numpy(dtype=float) for a in treatments])
    prob_a1: List[np.ndarray] = []
    prob_c1: List[np.ndarray] = []
    for k in range(K):
        H = df[_hist_cols(k)].to_numpy(dtype=float)
        X_k = np.column_stack([np.ones(n), H])
        fit_rows = uncens_before[:, k]
        prob_a1.append(_logit_fit_predict(X_k, A_obs[:, k], fit_rows))
        if censoring:
            X_c = np.column_stack([X_k, A_obs[:, k]])
            c_k = (df[censoring[k]].to_numpy(dtype=float) == 1).astype(float)
            prob_c1.append(_logit_fit_predict(X_c, c_k, fit_rows))
        else:
            prob_c1.append(np.ones(n))
    propensities = prob_a1

    # Precompute the full regime matrix. For static regimes this is
    # trivial; for dynamic regimes we evaluate the callable forward in
    # time on the OBSERVED history (as R ltmle's abar matrix does).
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

    def _run_regime(regime: Regime) -> Tuple[float, np.ndarray, List[float], List[int]]:
        """Returns psi (original scale), influence curve (original scale),
        per-step epsilons (time order k=0..K-1) and failed targeting steps."""
        regime_mat = _materialise_regime(regime).astype(float)  # (n, K)
        # Cumulative g of following the regime and staying uncensored.
        g_step = np.column_stack(
            [
                np.where(regime_mat[:, k] == 1, prob_a1[k], 1 - prob_a1[k]) * prob_c1[k]
                for k in range(K)
            ]
        )
        cum_g = np.clip(np.cumprod(g_step, axis=1), p_lo, p_hi)
        follow = np.cumprod(A_obs == regime_mat, axis=1).astype(bool)

        Q_next = Y_scaled.copy()  # Q*_{k+1}; starts at the (scaled) outcome
        ic = np.zeros(n, dtype=float)
        eps_list: List[float] = []
        failures: List[int] = []
        for k in reversed(range(K)):
            # Design: [1, history through L_k, A_k]; the prediction design
            # sets every A_j (j <= k) to the regime (R ltmle's SetA).
            cols = _hist_cols(k)
            X_obs = np.column_stack(
                [np.ones(n), df[cols + [treatments[k]]].to_numpy(dtype=float)]
            )
            X_reg = X_obs.copy()
            for j in range(k + 1):
                pos = 1 + (cols + [treatments[k]]).index(treatments[j])
                X_reg[:, pos] = regime_mat[:, j]
            fit_rows = uncens[:, k] & np.isfinite(Q_next)
            beta = _logit_fit(X_obs[fit_rows], Q_next[fit_rows])
            q_pred = np.clip(expit(X_reg @ beta), 1e-4, 0.9999)
            off = logit(q_pred)

            subs = uncens[:, k] & follow[:, k]
            w = np.where(subs, 1.0 / cum_g[:, k], 0.0)
            try:
                eps = _fluctuate(Q_next[subs], off[subs], w[subs])
            except (np.linalg.LinAlgError, FloatingPointError, ValueError) as exc:
                eps = 0.0
                failures.append(k)
                warnings.warn(
                    f"ltmle: targeting step k={k} failed "
                    f"({type(exc).__name__}: {exc}); epsilon set to 0 "
                    "(no targeting update at this time point).",
                    ConvergenceWarning,
                    stacklevel=3,
                )
            q_star = expit(off + eps)
            ic = ic + np.where(
                subs, (np.nan_to_num(Q_next) - q_star) / cum_g[:, k], 0.0
            )
            eps_list.append(float(eps))
            Q_next = q_star

        psi_scaled = float(np.mean(Q_next))
        ic = ic + (Q_next - psi_scaled)
        return (
            y_min + y_range * psi_scaled,
            y_range * ic,
            eps_list[::-1],
            sorted(failures),
        )

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
