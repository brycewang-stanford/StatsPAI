"""Modified treatment policies by sequential TMLE: ``sp.lmtp``.

The parameter is ``E[Y(d)]``, the mean outcome had each treatment been
replaced by ``d(A_t)`` -- a function of its *natural* value (an additive
shift, a multiplicative change, a cap). The estimator is the sequentially
doubly robust TMLE for longitudinal modified treatment policies, written
from the published method and checked against R ``lmtp`` 1.5.4 (AGPL; a
black box, outputs only) on fixtures:

* density ratios ``g^d(a | h) / g(a | h)`` come from a classifier that
  separates the natural rows (label 0) from the policy rows (label 1) of
  the stacked data, ``r = p / (1 - p)`` evaluated at the natural rows; the
  history in both halves is the natural one;
* the ratios are trimmed at the ``trim`` quantile of *all* of them (the
  whole ``n x K`` matrix, R's type-7 quantile);
* outcomes are scaled to ``[0, 1]`` (continuous: by the observed range or
  ``bounds``); the last outcome regression is a linear (continuous) or
  logistic (binary) regression, earlier ones regress the (continuous)
  pseudo-outcome linearly; predictions are bounded to ``[bound, 1 -
  bound]``;
* each step is targeted by an intercept-only logistic fluctuation with
  offset ``logit Q`` and weights the cumulative ratios; the policy value is
  the mean of the targeted ``Q_1`` at the policy, and the standard error
  ``sd(EIF) / sqrt(n)`` with ``EIF = Q*_1(d) + sum_t prod_{s<=t} r_s
  (Q*_{t+1}(d) - Q*_t)``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

Shift = Union[float, Callable[[np.ndarray], np.ndarray]]


def _logit_fit(
    X: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray] = None,
    off: Optional[np.ndarray] = None,
) -> np.ndarray:
    w = np.ones(len(y)) if w is None else w
    off = np.zeros(len(y)) if off is None else off
    b = np.zeros(X.shape[1])
    for _ in range(200):
        mu = expit(X @ b + off)
        g = X.T @ (w * (y - mu))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        b = b + step
        if np.max(np.abs(step)) < 1e-12:
            break
    return b


class _Linear:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Linear":
        self.b = np.linalg.lstsq(X, y, rcond=None)[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.b


class _Logistic:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Logistic":
        self.b = _logit_fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return expit(X @ self.b)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1 - p, p])


def _clone(model: Any) -> Any:
    from sklearn.base import clone

    return clone(model)


def _apply_shift(shift: Shift, a: np.ndarray) -> np.ndarray:
    if callable(shift):
        out = np.asarray(shift(a.copy()), dtype=float).reshape(-1)
        if out.shape != a.shape:
            raise MethodIncompatibility(
                "lmtp: the shift function must return one value per row.",
                recovery_hint="Write shift as a vectorised function of the "
                "treatment array, e.g. lambda a: a + 1.",
            )
        return out
    return a + float(shift)


@accepts_aliases(seed="random_state")
def lmtp(
    data: pd.DataFrame,
    y: str,
    treatments: Sequence[str],
    shift: Shift,
    *,
    covariates_time: Optional[Sequence[Sequence[str]]] = None,
    baseline: Optional[Sequence[str]] = None,
    outcome_type: str = "auto",
    bounds: Optional[Tuple[float, float]] = None,
    folds: int = 1,
    trim: float = 0.999,
    bound: float = 1e-5,
    outcome_learner: Any = None,
    ratio_learner: Any = None,
    random_state: int = 42,
    alpha: float = 0.05,
) -> CausalResult:
    """Mean outcome under a modified treatment policy (sequential TMLE).

    Parameters
    ----------
    data : pd.DataFrame
        Wide data, one row per unit.
    y : str
        Final outcome (continuous, or 0/1).
    treatments : sequence of str
        Treatment column at each time point, in time order.
    shift : float or callable
        The policy: a number is added to each treatment (``A + delta``); a
        callable maps the array of natural values of one treatment to the
        values under the policy (e.g. ``lambda a: np.where(a > 80, a - 15,
        a)``), applied at every time point.
    covariates_time : sequence of sequences of str, optional
        Time-varying covariates measured before each treatment.
    baseline : sequence of str, optional
        Baseline covariates (in every history).
    outcome_type : {'auto', 'continuous', 'binary'}, default 'auto'
    bounds : (float, float), optional
        Range used to scale a continuous outcome to ``[0, 1]``; default the
        observed range.
    folds : int, default 1
        Cross-fitting folds for the nuisance models (``1`` = fit and
        predict on the full sample, the setting the R comparison uses).
    trim : float, default 0.999
        Density ratios above this quantile of all ratios are set to it.
    bound : float, default 1e-5
        Outcome predictions are bounded to ``[bound, 1 - bound]``.
    outcome_learner : sklearn regressor, optional
        Learner for every outcome regression (default: linear regression,
        and logistic regression for a binary final outcome).
    ratio_learner : sklearn classifier, optional
        Classifier for the density ratios (default: logistic regression).
    random_state : int, default 42
        Seed of the cross-fitting split (``seed=`` is accepted as the
        same argument).
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``estimate`` is ``E[Y(d)]``. ``model_info`` holds the natural-course
        mean ``E[Y]``, the effect ``E[Y(d)] - E[Y]`` with its standard error
        (difference of influence functions, R ``lmtp_contrast``), the
        density ratios and the influence function.

    Notes
    -----
    Equals R ``lmtp::lmtp_tmle`` 1.5.4 with ``SL.glm`` learners and
    ``folds = 1`` to 1e-7 (one and two time points, continuous and binary
    outcomes; the remaining gap is R glm's convergence tolerance), see
    ``tests/reference_parity/test_lmtp_parity.py``. A deterministic
    modified treatment policy needs a treatment whose shifted value has
    support under the natural distribution; static policies on a binary
    treatment (``set A = 1``) are what ``sp.ltmle`` estimates. Censoring
    and the SDR estimator are not implemented.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> w = rng.normal(size=n)
    >>> a = w + rng.normal(size=n)
    >>> yv = a + w + rng.normal(size=n)
    >>> df = pd.DataFrame({'W': w, 'A': a, 'Y': yv})
    >>> r = sp.lmtp(df, 'Y', ['A'], shift=1.0, baseline=['W'])
    >>> bool(abs(r.model_info['effect'] - 1.0) < 0.3)
    True
    """
    treatments = list(treatments)
    K = len(treatments)
    if K < 1:
        raise MethodIncompatibility("lmtp: at least one treatment column is needed.")
    covariates_time = [list(c) for c in (covariates_time or [[] for _ in range(K)])]
    if len(covariates_time) != K:
        raise MethodIncompatibility(
            "lmtp: covariates_time needs one list per time point.",
            recovery_hint="Pass [] for a time point without new covariates.",
        )
    baseline = list(baseline or [])
    cols = [y] + treatments + baseline + [c for b in covariates_time for c in b]
    missing = sorted({c for c in cols if c not in data.columns})
    if missing:
        raise MethodIncompatibility(
            f"lmtp: columns {missing} not found.", diagnostics={"missing": missing}
        )
    if not (0.5 < trim <= 1.0) or not (0.0 < bound < 0.5):
        raise MethodIncompatibility("lmtp: need 0.5 < trim <= 1 and 0 < bound < 0.5.")
    df = data[list(dict.fromkeys(cols))].dropna()
    n = len(df)
    if n < 20:
        raise DataInsufficient(f"lmtp: {n} complete rows are too few.")
    Y = df[y].to_numpy(dtype=float)
    if outcome_type == "auto":
        outcome_type = "binary" if set(np.unique(Y)) <= {0.0, 1.0} else "continuous"
    if outcome_type not in ("binary", "continuous"):
        raise MethodIncompatibility(
            "lmtp: outcome_type must be 'continuous' or 'binary'."
        )
    if outcome_type == "binary":
        if not set(np.unique(Y)) <= {0.0, 1.0}:
            raise MethodIncompatibility("lmtp: a binary outcome must be 0/1.")
        lo, hi = 0.0, 1.0
    else:
        lo, hi = (
            (float(Y.min()), float(Y.max())) if bounds is None else map(float, bounds)
        )
        if not hi > lo:
            raise MethodIncompatibility("lmtp: the outcome bounds are degenerate.")
    ys = np.clip((Y - lo) / (hi - lo), 0.0, 1.0)

    nat = {c: df[c].to_numpy(dtype=float) for c in df.columns}
    pol = {a: _apply_shift(shift, nat[a]) for a in treatments}

    def history(t: int) -> List[str]:
        h = list(baseline)
        for s in range(t):
            h += covariates_time[s] + [treatments[s]]
        return h + covariates_time[t]

    if folds <= 1:
        splits = [(np.arange(n), np.arange(n))]
    else:
        from sklearn.model_selection import KFold

        splits = list(
            KFold(n_splits=int(folds), shuffle=True, random_state=random_state).split(
                ys
            )
        )

    def design(t: int, policy_t: bool) -> np.ndarray:
        cols_t = [nat[c] for c in history(t)]
        a = pol[treatments[t]] if policy_t else nat[treatments[t]]
        return np.column_stack([np.ones(n)] + cols_t + [a])

    # density ratios
    raw = np.zeros((n, K))
    for t in range(K):
        Xn, Xp = design(t, False), design(t, True)
        for tr, te in splits:
            Xs = np.vstack([Xn[tr], Xp[tr]])
            lab = np.r_[np.zeros(len(tr)), np.ones(len(tr))]
            clf = _Logistic() if ratio_learner is None else _clone(ratio_learner)
            clf.fit(Xs if ratio_learner is None else Xs[:, 1:], lab)
            p = clf.predict_proba(Xn[te] if ratio_learner is None else Xn[te][:, 1:])[
                :, 1
            ]
            p = np.clip(p, 1e-12, 1 - 1e-12)
            raw[te, t] = p / (1 - p)
    cap = float(np.quantile(raw, trim)) if trim < 1 else np.inf
    ratios = np.minimum(raw, cap)
    cum = np.cumprod(ratios, axis=1)

    Qn = np.zeros((n, K + 1))
    Qp = np.zeros((n, K + 1))
    Qn[:, K] = Qp[:, K] = ys
    pseudo = ys.copy()
    eps_all: List[float] = []
    for t in reversed(range(K)):
        Xn, Xp = design(t, False), design(t, True)
        qn = np.empty(n)
        qp = np.empty(n)
        last_binary = t == K - 1 and outcome_type == "binary"
        for tr, te in splits:
            if outcome_learner is None:
                m = _Logistic() if last_binary else _Linear()
                m.fit(Xn[tr], pseudo[tr])
                qn[te], qp[te] = m.predict(Xn[te]), m.predict(Xp[te])
            else:
                m = _clone(outcome_learner).fit(Xn[tr][:, 1:], pseudo[tr])
                qn[te], qp[te] = m.predict(Xn[te][:, 1:]), m.predict(Xp[te][:, 1:])
        qn = np.clip(qn, bound, 1 - bound)
        qp = np.clip(qp, bound, 1 - bound)
        eps = float(_logit_fit(np.ones((n, 1)), pseudo, w=cum[:, t], off=logit(qn))[0])
        eps_all.append(eps)
        Qn[:, t] = expit(logit(qn) + eps)
        Qp[:, t] = expit(logit(qp) + eps)
        pseudo = Qp[:, t]

    psi_s = float(Qp[:, 0].mean())
    eif_s = Qp[:, 0].copy()
    for t in range(K):
        eif_s += cum[:, t] * (Qp[:, t + 1] - Qn[:, t])
    eif = (hi - lo) * (eif_s - psi_s)
    psi = lo + (hi - lo) * psi_s
    se = float(np.std(eif, ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1 - alpha / 2)
    # contrast with the natural course, E[Y], whose EIF is Y - mean(Y)
    eif_nat = Y - Y.mean()
    diff_if = eif - eif_nat
    effect = psi - float(Y.mean())
    se_eff = float(np.std(diff_if, ddof=1) / np.sqrt(n))
    info: Dict[str, Any] = {
        "natural_course": float(Y.mean()),
        "effect": effect,
        "effect_se": se_eff,
        "effect_ci": (effect - z * se_eff, effect + z * se_eff),
        "density_ratios": ratios,
        "ratio_trim_value": cap,
        "n_ratios_trimmed": int((raw > cap).sum()),
        "max_cumulative_ratio": float(cum.max()),
        "influence_function": eif,
        "epsilons": eps_all[::-1],
        "folds": int(max(folds, 1)),
        "outcome_type": outcome_type,
        "K": K,
    }
    return CausalResult(
        method="LMTP (sequential TMLE for a modified treatment policy)",
        estimand="E[Y(d)]",
        estimate=psi,
        se=se,
        pvalue=float("nan"),
        ci=(psi - z * se, psi + z * se),
        alpha=alpha,
        n_obs=n,
        detail=None,
        model_info=info,
    )
