"""Longitudinal TMLE for a working marginal structural model: ``sp.ltmle_msm``.

The parameter is the projection of the counterfactual outcome means
``E[Y(d)]`` of a set of regimes ``d`` onto a logistic working model
``logit m(d, V) = x(d, V)' beta`` in summary measures of the regime (and
baseline covariates ``V``), weighted by ``h(d, V)``: R ``ltmle``'s
``ltmleMSM`` (Petersen, Schwab, Gruber, Blaser, Schomaker and van der
Laan, 2014). The estimator was written from that method and checked
against ``ltmle`` 1.3-0 (GPL-2; used as a black box, outputs only):

* the sequential outcome regressions are fitted separately for each regime
  (all units uncensored at the node, history with the observed
  treatments; predictions with the treatments set to the regime) and
  bounded to ``[1e-4, 0.9999]``, as in ``sp.ltmle``;
* at each node one fluctuation is shared by all regimes: a logistic
  regression of ``Q*_{k+1}(d)`` with offset ``logit Q_k(d)``, covariates
  ``x(d, V)`` and weights ``h(d, V) w / g_cum(d)`` over the units that
  follow ``d`` and are uncensored through the node, stacked over regimes;
* ``beta`` is the weighted logistic (quasi-binomial) fit of the targeted
  ``Q*_0(d)`` on ``x(d, V)`` stacked over units and regimes; the influence
  curve is ``M^{-1} sum_d h x (sum_k H_k (Q*_{k+1} - Q*_k) + Q*_0 - m)``
  with ``M = mean(sum_d h x x' m (1 - m))``;
* ``msm_weights='empirical'`` is ``h(d)``, the unweighted share of units
  whose observed treatments match ``d`` (``d(i)`` for a dynamic regime)
  and who remain uncensored; a regime identical to an earlier one gets 0.

A continuous outcome is mapped to ``[0, 1]`` by its observed range and the
working model describes the scaled mean, as in ``ltmle``.

References
----------
[@lendle2017ltmle], [@vanderlaan2012targeted]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability
from .ltmle import Regime, _logit_fit, _logit_fit_predict

_ALTS = ["sp.ltmle", "sp.ltmle_msm", "sp.lmtp"]


def _err(message: str, hint: str, **diag: Any) -> MethodIncompatibility:
    return MethodIncompatibility(
        message,
        recovery_hint=hint,
        diagnostics=diag or None,
        alternative_functions=_ALTS,
    )


@dataclass
class LTMLEMSMResult(ResultProtocolMixin):
    """Working marginal structural model fitted by :func:`ltmle_msm`.

    Attributes
    ----------
    coefficients : pd.DataFrame
        One row per working-model term: TMLE ``estimate``, influence-curve
        ``se``, ``z``, ``pvalue``, ``ci_lower`` / ``ci_upper`` and the IPTW
        estimate ``iptw``.
    influence_curve : np.ndarray
        ``n x p`` influence curve of the TMLE coefficients.
    regime_means : pd.DataFrame
        For each regime, the mean of the targeted ``Q*_0(d)`` (on the
        outcome's scale) and its weight.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 300
    >>> L0 = rng.normal(size=n)
    >>> A0 = rng.binomial(1, 0.5, size=n)
    >>> L1 = 0.5 * L0 + rng.normal(size=n)
    >>> A1 = rng.binomial(1, 0.5, size=n)
    >>> Y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 * (A0 + A1) + 0.2 * L1))))
    >>> df = pd.DataFrame(dict(L0=L0, A0=A0, L1=L1, A1=A1, Y=Y))
    >>> r = sp.ltmle_msm(df, 'Y', ['A0', 'A1'], [['L0'], ['L1']],
    ...                  [(0, 0), (1, 0), (0, 1), (1, 1)],
    ...                  {'cum_a': [0, 1, 1, 2]})
    >>> isinstance(r, sp.LTMLEMSMResult)
    True
    """

    _citation_keys = ("lendle2017ltmle", "vanderlaan2012targeted")

    coefficients: pd.DataFrame
    influence_curve: np.ndarray
    regime_means: pd.DataFrame
    n_obs: int
    K: int
    alpha: float
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def params(self) -> pd.Series:
        return self.coefficients["estimate"]

    @property
    def vcov(self) -> pd.DataFrame:
        ic = self.influence_curve
        v = np.cov(ic, rowvar=False, ddof=1) / ic.shape[0]
        idx = self.coefficients.index
        return pd.DataFrame(np.atleast_2d(v), index=idx, columns=idx)

    def summary(self) -> str:
        lines = [
            "Longitudinal TMLE, working marginal structural model",
            "-" * 52,
            f"  K (time points) : {self.K}",
            f"  N               : {self.n_obs}",
            f"  Regimes         : {len(self.regime_means)}",
            "",
            self.coefficients.round(4).to_string(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _regime_matrix(
    regime: Union[Regime, np.ndarray],
    n: int,
    K: int,
    df: pd.DataFrame,
    baseline: List[str],
    covariates_time: Sequence[Sequence[str]],
) -> np.ndarray:
    """``n x K`` 0/1 matrix of a static, callable or per-unit regime."""
    if callable(regime):
        mat = np.zeros((n, K), dtype=float)
        history: Dict[str, np.ndarray] = {
            c: df[c].to_numpy(dtype=float) for c in baseline
        }
        for k in range(K):
            for c in covariates_time[k]:
                history[c] = df[c].to_numpy(dtype=float)
            a_k = np.asarray(regime(k, history), dtype=float).reshape(-1)
            if a_k.size != n or not np.isin(a_k, (0.0, 1.0)).all():
                raise _err(
                    f"ltmle_msm: a dynamic regime returned an invalid "
                    f"assignment at time point {k + 1}.",
                    "Return one 0/1 value per row.",
                    k=k,
                )
            mat[:, k] = a_k
            history[f"__regime_A_{k}"] = a_k
        return mat
    arr = np.asarray(regime, dtype=float)
    if arr.ndim == 1:
        if arr.size != K:
            raise _err(
                f"ltmle_msm: a static regime needs {K} values; got {arr.size}.",
                "Give one 0/1 value per time point.",
            )
        arr = np.tile(arr, (n, 1))
    if arr.shape != (n, K) or not np.isin(arr, (0.0, 1.0)).all():
        raise _err(
            "ltmle_msm: a per-unit regime must be an n x K array of 0/1.",
            "Pass static sequences, callables, or n x K arrays.",
        )
    return arr


def _weighted_logit(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, off: Optional[np.ndarray] = None
) -> np.ndarray:
    """Quasi-binomial weighted logistic fit (Newton), optional offset."""
    off = np.zeros(len(y)) if off is None else off
    b = np.zeros(X.shape[1])
    for _ in range(100):
        mu = expit(off + X @ b)
        g = X.T @ (w * (y - mu))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.lstsq(H, g, rcond=None)[0]
        b = b + step
        if np.max(np.abs(step)) < 1e-13:
            b_out: np.ndarray = b
            return b_out
    raise NumericalInstability(
        "ltmle_msm: a logistic fit did not converge in 100 Newton steps.",
        recovery_hint="Check the working model for separation or collinear "
        "summary measures.",
    )


def ltmle_msm(
    data: pd.DataFrame,
    y: str,
    treatments: Sequence[str],
    covariates_time: Sequence[Sequence[str]],
    regimes: Sequence[Union[Regime, np.ndarray]],
    summary_measures: Union[pd.DataFrame, Dict[str, Sequence[float]]],
    *,
    msm_terms: Optional[Sequence[str]] = None,
    baseline: Optional[Sequence[str]] = None,
    censoring: Optional[Sequence[str]] = None,
    msm_weights: Any = "empirical",
    propensity_bounds: Tuple[float, float] = (0.01, 1.0),
    outcome_type: str = "auto",
    weights: Optional[str] = None,
    alpha: float = 0.05,
) -> LTMLEMSMResult:
    """Longitudinal TMLE of a working marginal structural model (R ``ltmleMSM``).

    Parameters
    ----------
    data : pd.DataFrame
        Wide data, one row per unit.
    y : str
        Final outcome (0/1, or continuous: scaled to ``[0, 1]`` by its
        observed range).
    treatments : sequence of str
        Binary treatment column per time point (``K`` of them).
    covariates_time : sequence of sequences of str
        Covariates measured before each treatment (``K`` lists, may be
        empty).
    regimes : sequence
        The regimes ``d_1, ..., d_J``: each a static sequence of ``K``
        0/1 values, a callable ``regime(k, history)`` as in ``sp.ltmle``,
        or an ``n x K`` 0/1 array (one assignment per unit).
    summary_measures : pd.DataFrame or dict
        One row per regime (``J`` rows), columns naming the measures of the
        regime the working model uses (e.g. the number of treated periods).
    msm_terms : sequence of str, optional
        Right-hand side of the working model: columns of
        ``summary_measures`` and / or baseline covariates; an intercept is
        always included. Default: every summary measure.
    baseline : sequence of str, optional
        Baseline covariates (in every history; usable in ``msm_terms``).
    censoring : sequence of str, optional
        Censoring indicator after each treatment (1 = observed).
    msm_weights : {'empirical'}, None or array, default 'empirical'
        Projection weights ``h(d, V)``: the share of units whose observed
        treatments match each regime (R's default), ``None`` for equal
        weights, or a length-``J`` (or ``n x J``) array.
    propensity_bounds : tuple, default (0.01, 1.0)
        Bounds on the cumulative probability of following a regime and
        remaining uncensored (R ``gbounds``).
    outcome_type : {'auto', 'binary', 'continuous'}, default 'auto'
    weights : str, optional
        Observation (sampling) weights column (R ``observation.weights``).
    alpha : float, default 0.05

    Returns
    -------
    LTMLEMSMResult
        ``coefficients`` (TMLE estimates with influence-curve standard
        errors and the IPTW estimates), the influence curve and the
        targeted regime means.

    Notes
    -----
    Equals R ``ltmle::ltmleMSM`` 1.3-0 (``SL.library = NULL``,
    ``variance.method = "ic"``, one final outcome node) to 1e-8 on two-
    period designs with static, dynamic and censored regimes, observation
    weights and user projection weights
    (``tests/reference_parity/test_ltmle_msm_parity.py``). Pooling over
    several outcome nodes (``final.Ynodes``) and stratified fits are not
    implemented. The influence-curve standard error can be optimistic
    under near-positivity violations (R's ``variance.method='tmle'`` is
    not implemented).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> L0 = rng.normal(size=n)
    >>> A0 = rng.binomial(1, 1 / (1 + np.exp(-0.5 * L0)))
    >>> L1 = 0.5 * L0 + 0.3 * A0 + rng.normal(size=n)
    >>> A1 = rng.binomial(1, 1 / (1 + np.exp(-0.5 * L1)))
    >>> Y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 0.4 * (A0 + A1) + 0.3 * L1))))
    >>> df = pd.DataFrame(dict(L0=L0, A0=A0, L1=L1, A1=A1, Y=Y))
    >>> regs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    >>> r = sp.ltmle_msm(df, 'Y', ['A0', 'A1'], [['L0'], ['L1']], regs,
    ...                  {'cum_a': [0, 1, 1, 2]})
    >>> list(r.coefficients.index)
    ['(Intercept)', 'cum_a']
    """
    if not isinstance(data, pd.DataFrame):
        raise _err("ltmle_msm: data must be a pandas DataFrame.", "Pass a DataFrame.")
    treatments = list(treatments)
    covariates_time = [list(c) for c in covariates_time]
    K = len(treatments)
    if K < 1 or len(covariates_time) != K:
        raise _err(
            "ltmle_msm: treatments and covariates_time need one entry per "
            "time point.",
            "Pass [] for a time point without covariates.",
        )
    baseline = list(baseline or [])
    censoring = list(censoring or [])
    if censoring and len(censoring) != K:
        raise _err(
            "ltmle_msm: censoring needs one indicator per time point.",
            "Pass the indicator after each treatment, 1 = observed.",
        )
    sm = pd.DataFrame(summary_measures).reset_index(drop=True)
    J = len(regimes)
    if J < 2 or len(sm) != J:
        raise _err(
            f"ltmle_msm: {J} regimes and {len(sm)} rows of summary measures; "
            "need at least two regimes and one row per regime.",
            "Give summary_measures one row per regime.",
        )
    terms = list(sm.columns) if msm_terms is None else list(msm_terms)
    bad = [t for t in terms if t not in sm.columns and t not in baseline]
    if bad:
        raise _err(
            f"ltmle_msm: msm_terms {bad} are neither summary measures nor "
            "baseline covariates.",
            "Use summary_measures columns or names listed in baseline=.",
        )
    cols = [y] + treatments + baseline + censoring
    cols += [c for b in covariates_time for c in b]
    if weights is not None:
        cols.append(weights)
    missing = sorted({c for c in cols if c not in data.columns})
    if missing:
        raise _err(
            f"ltmle_msm: columns {missing} not found.",
            "Check the column names.",
            missing=missing,
        )
    if not 0 < alpha < 1:
        raise _err(f"ltmle_msm: alpha must be in (0, 1); got {alpha}.", "Use 0.05.")
    p_lo, p_hi = (float(v) for v in propensity_bounds)
    if not 0 < p_lo < p_hi <= 1:
        raise _err(
            "ltmle_msm: propensity_bounds must satisfy 0 < lower < upper <= 1.",
            "Use (0.01, 1.0), R's default.",
        )

    df = data.reset_index(drop=True)
    n = len(df)
    if n < 10:
        raise DataInsufficient(
            f"ltmle_msm: {n} rows are too few.",
            recovery_hint="Use more data.",
            alternative_functions=_ALTS,
        )
    Y_raw = df[y].to_numpy(dtype=float)
    Yo = Y_raw[np.isfinite(Y_raw)]
    if outcome_type == "auto":
        outcome_type = "binary" if set(np.unique(Yo)) <= {0.0, 1.0} else "continuous"
    if outcome_type not in ("binary", "continuous"):
        raise _err(
            "ltmle_msm: outcome_type must be 'auto', 'binary' or 'continuous'.",
            "Use outcome_type='auto'.",
        )
    if outcome_type == "binary":
        y_min, y_range = 0.0, 1.0
    else:
        y_min, y_range = float(Yo.min()), float(np.ptp(Yo))
        if y_range <= 0:
            raise _err("ltmle_msm: the outcome is constant.", "Check the outcome.")
    Ys = (Y_raw - y_min) / y_range
    if weights is not None:
        obs_w = df[weights].to_numpy(dtype=float)
        if not np.all(np.isfinite(obs_w)) or np.any(obs_w <= 0):
            raise _err(
                "ltmle_msm: weights must be finite and strictly positive.",
                "Drop rows with zero or missing weight.",
            )
    else:
        obs_w = np.ones(n)

    def hist_cols(k: int) -> List[str]:
        h = list(baseline)
        for j in range(k):
            h += list(covariates_time[j]) + [treatments[j]]
        return h + list(covariates_time[k])

    if censoring:
        C_obs = np.column_stack([df[c].to_numpy(dtype=float) == 1 for c in censoring])
        uncens = np.cumprod(C_obs, axis=1).astype(bool)
    else:
        uncens = np.ones((n, K), dtype=bool)
    uncens_before = np.column_stack([np.ones(n, dtype=bool), uncens[:, :-1]])
    A_obs = np.column_stack([df[a].to_numpy(dtype=float) for a in treatments])

    # treatment and censoring models on the observed history
    prob_a1: List[np.ndarray] = []
    prob_c1: List[np.ndarray] = []
    for k in range(K):
        X_k = np.column_stack([np.ones(n), df[hist_cols(k)].to_numpy(dtype=float)])
        rows = uncens_before[:, k]
        prob_a1.append(_logit_fit_predict(X_k, A_obs[:, k], rows, obs_w))
        if censoring:
            X_c = np.column_stack([X_k, A_obs[:, k]])
            c_k = C_obs[:, k].astype(float)
            prob_c1.append(_logit_fit_predict(X_c, c_k, rows, obs_w))
        else:
            prob_c1.append(np.ones(n))

    mats = [_regime_matrix(r, n, K, df, baseline, covariates_time) for r in regimes]
    follow, cum_g = [], []
    for mat in mats:
        g_step = np.column_stack(
            [
                np.where(mat[:, k] == 1, prob_a1[k], 1 - prob_a1[k]) * prob_c1[k]
                for k in range(K)
            ]
        )
        cum_g.append(np.clip(np.cumprod(g_step, axis=1), p_lo, p_hi))
        follow.append(np.cumprod(A_obs == mat, axis=1).astype(bool) & uncens)

    # projection weights h(d, V), n x J
    if isinstance(msm_weights, str):
        if msm_weights != "empirical":
            raise _err(
                f"ltmle_msm: msm_weights must be 'empirical', None or an "
                f"array; got {msm_weights!r}.",
                "Use 'empirical' (R's default).",
            )
        # share of units (unweighted) whose observed treatments match the
        # regime and who stay uncensored; a regime identical to an earlier
        # one for every unit gets weight 0
        match = np.column_stack(
            [np.all(A_obs == m, axis=1) & uncens[:, -1] for m in mats]
        )
        share = match.mean(axis=0)
        for j in range(J):
            if any(np.array_equal(mats[j], mats[i]) for i in range(j)):
                share[j] = 0.0
        h = np.tile(share, (n, 1))
    elif msm_weights is None:
        h = np.ones((n, J))
    else:
        hw = np.asarray(msm_weights, dtype=float)
        if hw.shape == (J,):
            h = np.tile(hw, (n, 1))
        elif hw.shape == (n, J):
            h = hw.copy()
        else:
            raise _err(
                f"ltmle_msm: msm_weights must have length {J} or shape "
                f"({n}, {J}); got {hw.shape}.",
                "Give one weight per regime.",
            )
        if not np.all(np.isfinite(h)) or np.any(h < 0):
            raise _err(
                "ltmle_msm: msm_weights must be finite and non-negative.",
                "Check the weights.",
            )

    def design(j: int) -> np.ndarray:
        xs = [np.ones(n)]
        for t in terms:
            if t in sm.columns:
                xs.append(np.full(n, float(sm.loc[j, t])))
            else:
                xs.append(df[t].to_numpy(dtype=float))
        return np.column_stack(xs)

    Xd = [design(j) for j in range(J)]
    p = Xd[0].shape[1]

    Q_next = [Ys.copy() for _ in range(J)]
    ic_sum = [np.zeros(n) for _ in range(J)]
    eps_all: List[List[float]] = []
    for k in reversed(range(K)):
        hc = hist_cols(k)
        names = hc + [treatments[k]]
        X_obs = np.column_stack([np.ones(n), df[names].to_numpy(dtype=float)])
        Qk = []
        for j in range(J):
            X_reg = X_obs.copy()
            for m in range(k + 1):
                X_reg[:, 1 + names.index(treatments[m])] = mats[j][:, m]
            rows = uncens[:, k] & np.isfinite(Q_next[j])
            beta_q = _logit_fit(X_obs[rows], Q_next[j][rows], obs_w[rows])
            Qk.append(np.clip(expit(X_reg @ beta_q), 1e-4, 0.9999))
        sub = [follow[j][:, k] for j in range(J)]
        Xs = np.vstack([Xd[j][sub[j]] for j in range(J)])
        ys = np.concatenate([Q_next[j][sub[j]] for j in range(J)])
        off = np.concatenate([logit(Qk[j][sub[j]]) for j in range(J)])
        ws = np.concatenate(
            [(h[:, j] * obs_w / cum_g[j][:, k])[sub[j]] for j in range(J)]
        )
        keep = ws > 0
        eps = (
            _weighted_logit(Xs[keep], ys[keep], ws[keep], off[keep])
            if keep.sum() > 0
            else np.zeros(p)
        )
        eps_all.append(eps.tolist())
        for j in range(J):
            q_star = expit(logit(Qk[j]) + Xd[j] @ eps)
            ic_sum[j] = ic_sum[j] + np.where(
                sub[j],
                obs_w * (np.nan_to_num(Q_next[j]) - q_star) / cum_g[j][:, k],
                0.0,
            )
            Q_next[j] = q_star

    # working model on the targeted means, and IPTW
    X_all = np.vstack(Xd)
    h_all = np.concatenate([h[:, j] * obs_w for j in range(J)])
    beta = _weighted_logit(X_all, np.concatenate(Q_next), h_all)
    fin = [follow[j][:, K - 1] for j in range(J)]
    w_iptw = np.concatenate(
        [np.where(fin[j], h[:, j] * obs_w / cum_g[j][:, K - 1], 0.0) for j in range(J)]
    )
    y_iptw = np.concatenate(
        [np.where(fin[j], np.nan_to_num(Ys), 0.0) for j in range(J)]
    )
    beta_iptw = _weighted_logit(
        X_all[w_iptw > 0], y_iptw[w_iptw > 0], w_iptw[w_iptw > 0]
    )

    M = np.zeros((p, p))
    D = np.zeros((n, p))
    for j in range(J):
        m = expit(Xd[j] @ beta)
        M += (Xd[j] * (h[:, j] * obs_w * m * (1 - m))[:, None]).T @ Xd[j]
        D += (h[:, j] * (ic_sum[j] + obs_w * (Q_next[j] - m)))[:, None] * Xd[j]
    M /= n
    IC = D @ np.linalg.inv(M).T
    se = np.sqrt(np.var(IC, axis=0, ddof=1) / n)
    z = beta / se
    crit = stats.norm.ppf(1 - alpha / 2)
    index = ["(Intercept)"] + terms
    coef = pd.DataFrame(
        {
            "estimate": beta,
            "se": se,
            "z": z,
            "pvalue": 2 * stats.norm.sf(np.abs(z)),
            "ci_lower": beta - crit * se,
            "ci_upper": beta + crit * se,
            "iptw": beta_iptw,
        },
        index=index,
    )
    labels = []
    for r in regimes:
        if callable(r):
            labels.append("dynamic")
        elif np.ndim(r) == 1:
            labels.append(",".join(str(int(v)) for v in np.ravel(r)))
        else:
            labels.append("per-unit")
    means = pd.DataFrame(
        {
            "regime": labels,
            "mean_Q0": [
                y_min + y_range * float(np.sum(obs_w * Q_next[j]) / obs_w.sum())
                for j in range(J)
            ],
            "weight": h.mean(axis=0),
        }
    )
    return LTMLEMSMResult(
        coefficients=coef,
        influence_curve=IC,
        regime_means=means,
        n_obs=n,
        K=K,
        alpha=alpha,
        detail={
            "epsilons": eps_all[::-1],
            "outcome_type": outcome_type,
            "outcome_scale": (y_min, y_range),
            "propensity_bounds": (p_lo, p_hi),
            "msm_weights": (
                msm_weights
                if isinstance(msm_weights, str) or msm_weights is None
                else "user"
            ),
            "n_regimes": J,
        },
    )


__all__ = ["ltmle_msm", "LTMLEMSMResult"]
