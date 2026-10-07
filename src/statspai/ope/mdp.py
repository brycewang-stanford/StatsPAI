"""
Policy evaluation in dynamic systems from one long trajectory.

Two estimators for data in which today's action changes tomorrow's
state.

:func:`mdp_policy_value`
    The long-run average outcome under a target policy in a Markov
    decision process whose state is observed. The estimator combines an
    excess-reward function ``Q`` (how much better than average it is to
    start from a state) with the ratio ``omega`` of the stationary state
    distributions under the target policy and under the data::

        V_hat = sum_t omega(X_t) rho_t (Y_t + Q(X_{t+1}) - Q(X_t))
                / sum_t omega(X_t) rho_t ,

    where ``rho_t`` is one over the probability of the action taken when
    it agrees with the policy and zero otherwise. It is consistent if
    either ``Q`` or ``omega`` is right, and its error does not grow with
    the length of the trajectory, unlike sequential inverse-probability
    weighting.

:func:`marginal_policy_effect`
    When part of the state is not observed, the value of a different
    policy is hard to learn, but the effect of a *small* change to the
    current one is not. Removing a fraction ``eps`` of the treatments
    that occur changes the average outcome by ``-eps * theta``, where
    ``theta`` averages, over periods, the treatment probability times
    the effect of treating now on the sum of current and future
    outcomes.

References
----------
[@liao2022batch], [@kallus2022efficiently]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility

PolicyLike = Union[Callable[[pd.DataFrame], Any], str, int, float]


@dataclass
class MDPPolicyValueResult(ResultProtocolMixin):
    """Result of :func:`mdp_policy_value`.

    Attributes
    ----------
    value : float
        Estimated long-run average outcome under the target policy, or
        its difference from the baseline policy when one was given.
    se : float
    ci : tuple of float
    value_policy, value_baseline : float
        The two long-run values (``value_baseline`` is ``nan`` without a
        baseline).
    n_obs : int
        Number of transitions used.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> s, rows = 0, []
    >>> for t in range(600):
    ...     w = int(rng.random() < 0.5)
    ...     rows.append((s, w, s + w + rng.normal()))
    ...     s = int(rng.random() < (0.7 if w else 0.3))
    >>> df = pd.DataFrame(rows, columns=["s", "w", "y"])
    >>> r = sp.mdp_policy_value(df, "y", "w", ["s"], policy=1, propensity=0.5)
    >>> isinstance(r, sp.MDPPolicyValueResult)
    True
    """

    _citation_keys = ("liao2022batch", "kallus2022efficiently")

    value: float
    se: float
    ci: Tuple[float, float]
    pvalue: float
    value_policy: float
    value_baseline: float
    n_obs: int
    alpha: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lines = [
            "Long-run average value in a Markov decision process",
            f"  transitions = {self.n_obs}",
            f"  value under the policy   : {self.value_policy:.6g}",
        ]
        if np.isfinite(self.value_baseline):
            lines.append(f"  value under the baseline : {self.value_baseline:.6g}")
            lines.append(f"  difference               : {self.value:.6g}")
        lines.append(
            f"  se = {self.se:.6g}, {100 * (1 - self.alpha):g}% CI = "
            f"[{self.ci[0]:.6g}, {self.ci[1]:.6g}]"
        )
        return "\n".join(lines)


def _transitions(
    data: pd.DataFrame, trajectory: Optional[str], time: Optional[str]
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Sorted data and, for each row, the position of the next row (-1 at
    the end of a trajectory)."""
    keys = [k for k in (trajectory, time) if k is not None]
    df = data.sort_values(keys, kind="stable") if keys else data
    df = df.reset_index(drop=True)
    nxt = np.arange(1, len(df) + 1)
    if trajectory is not None:
        ids = df[trajectory].to_numpy()
        last = np.r_[ids[1:] != ids[:-1], True]
    else:
        last = np.zeros(len(df), dtype=bool)
        last[-1] = True
    nxt[last] = -1
    return df, nxt


def _target_actions(
    policy: PolicyLike, states: pd.DataFrame, df: pd.DataFrame
) -> np.ndarray:
    if callable(policy):
        out = np.asarray(policy(states))
        if out.shape != (len(states),):
            raise MethodIncompatibility(
                "a callable policy must return one action per row of the "
                "state columns"
            )
        return np.asarray(out)
    if isinstance(policy, str):
        if policy not in df.columns:
            raise MethodIncompatibility(f"policy column {policy!r} not found")
        return np.asarray(df[policy].to_numpy())
    return np.full(len(states), policy)


def _feature_matrix(
    states: pd.DataFrame, features: Any
) -> Tuple[np.ndarray, np.ndarray, str]:
    """``(phi, psi, kind)``: the full basis (spans the constant) and the
    basis with the constant direction removed."""
    if features is None or (isinstance(features, str) and features == "tabular"):
        codes = states.astype(str).agg("|".join, axis=1)
        dummies = pd.get_dummies(codes, dtype=float).to_numpy()
        if dummies.shape[1] > 200:
            raise MethodIncompatibility(
                f"the state takes {dummies.shape[1]} distinct values; pass "
                "features= (numeric columns or a callable) instead of a "
                "table"
            )
        return dummies, dummies[:, 1:], "tabular"
    if callable(features):
        psi = np.asarray(features(states), dtype=float)
    else:
        missing = [c for c in features if c not in states.columns]
        if missing:
            raise MethodIncompatibility(
                f"features must be among the state columns; missing {missing}"
            )
        psi = states[list(features)].to_numpy(dtype=float)
    if psi.ndim == 1:
        psi = psi[:, None]
    if psi.shape[0] != len(states) or not np.all(np.isfinite(psi)):
        raise MethodIncompatibility("features must be finite, one row per period")
    return np.column_stack([np.ones(len(states)), psi]), psi, "linear"


def _one_policy(
    y: np.ndarray,
    rho: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    cur: np.ndarray,
    nxt: np.ndarray,
) -> Dict[str, Any]:
    """Nuisances and scores for one target policy."""
    f = phi[cur]
    g, gn = psi[cur], psi[nxt]
    yy, r = y[cur], rho[cur]
    m = cur.size
    p = phi.shape[1]
    # Excess-reward function: sum_t rho_t phi_t (Y_t - eta + (psi_{t+1} - psi_t)'b) = 0.
    lhs = np.column_stack(
        [(f * r[:, None]).sum(axis=0), -(f * r[:, None]).T @ (gn - g)]
    )
    rhs = (f * (r * yy)[:, None]).sum(axis=0)
    sol, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    eta, beta = float(sol[0]), sol[1:]
    q, qn = g @ beta, gn @ beta
    # Stationary ratio omega = phi'a:
    #   sum_t omega_t (rho_t psi_{t+1} - psi_t) = 0 and mean(omega) = 1.
    lhs_w = np.vstack([(r[:, None] * gn - g).T @ f / m, f.mean(axis=0)[None, :]])
    rhs_w = np.zeros(p)
    rhs_w[-1] = 1.0
    a, *_ = np.linalg.lstsq(lhs_w, rhs_w, rcond=None)
    omega = f @ a
    weight = omega * r
    denom = float(weight.mean())
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        raise DataInsufficient(
            "the data contain too few periods in which the action taken "
            "agrees with the policy"
        )
    value = float(np.mean(weight * (yy + qn - q)) / denom)
    score = weight * (yy + qn - q - value) / denom
    return {
        "value": value,
        "score": score,
        "eta": eta,
        "omega": omega,
        "q": q,
        "share_followed": float(np.mean(r > 0)),
    }


def mdp_policy_value(
    data: pd.DataFrame,
    y: str,
    treat: str,
    state: Sequence[str],
    policy: PolicyLike,
    *,
    baseline: Optional[PolicyLike] = None,
    propensity: Union[None, str, float] = None,
    features: Any = None,
    trajectory: Optional[str] = None,
    time: Optional[str] = None,
    alpha: float = 0.05,
) -> MDPPolicyValueResult:
    """
    Long-run average outcome under a policy in a Markov decision process.

    Parameters
    ----------
    data : pd.DataFrame
        One row per period, in time order (or sortable by ``time``),
        from one or several trajectories.
    y : str
        Outcome (reward) of the period.
    treat : str
        Action taken in the period.
    state : sequence of str
        Columns of the state observed before the action. The state must
        carry everything through which past actions affect the future.
    policy : callable, str or scalar
        The target policy: a function of the state columns returning one
        action per row, the name of a column holding the action the
        policy would take, or a single action taken always.
    baseline : callable, str or scalar, optional
        A second policy. If given, the result is the difference in
        long-run value between ``policy`` and ``baseline``.
    propensity : str or float, optional
        Probability with which the action *actually taken* was chosen,
        given the state: a column, or one number if every action had the
        same probability. If omitted it is estimated by the frequency of
        the action in each distinct state, which requires a state with
        few distinct values.
    features : "tabular", sequence of str or callable, optional
        Basis for the excess-reward function and the stationary ratio.
        ``"tabular"`` (default) uses one indicator per distinct state
        and is exact for a finite state space. A list of numeric state
        columns, or a callable returning a matrix, gives a linear basis
        (a constant is added).
    trajectory : str, optional
        Column identifying trajectories; transitions are not formed
        across them.
    time : str, optional
        Column to sort by within a trajectory.
    alpha : float, default 0.05

    Returns
    -------
    MDPPolicyValueResult

    Notes
    -----
    The assumptions are those of a time-homogeneous Markov decision
    process observed in its stationary regime: the state is fully
    observed, the action depends on the past only through the state,
    and both the data and the target policy visit the same states
    (``detail["min_omega"]`` and ``detail["max_omega"]`` report the
    range of the estimated stationary ratio; a warning is issued when
    it is negative, which a linear basis can produce when overlap is
    poor).

    The two nuisance functions are fitted on the same trajectory, by
    moment equations that are linear in the basis. With the tabular
    basis the estimate coincides with the plug-in value of the estimated
    transition model. The standard error uses the fact that the
    summands are martingale differences at the truth, so no correction
    for serial correlation is applied; it does not account for the
    estimation of ``propensity`` when that is omitted.

    References
    ----------
    [@liao2022batch], [@kallus2022efficiently]

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> s, rows = 0, []
    >>> for t in range(2000):
    ...     w = int(rng.random() < 0.5)
    ...     rows.append((s, w, s + w + rng.normal()))
    ...     s = int(rng.random() < (0.7 if w else 0.3))
    >>> df = pd.DataFrame(rows, columns=["s", "w", "y"])
    >>> r = sp.mdp_policy_value(df, "y", "w", ["s"], policy=1, baseline=0,
    ...                         propensity=0.5)
    >>> bool(r.ci[0] < 1.4 < r.ci[1])   # true difference: 1 + 0.7 - 0.3
    True
    """
    if not 0 < alpha < 1:
        raise MethodIncompatibility("alpha must be in (0, 1)")
    state = list(state)
    cols = [y, treat] + state + [c for c in (trajectory, time) if c is not None]
    if isinstance(propensity, str):
        cols.append(propensity)
    for pol in (policy, baseline):
        if isinstance(pol, str):
            cols.append(pol)
    missing = [c for c in dict.fromkeys(cols) if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"columns not found in data: {missing}")
    if not state:
        raise MethodIncompatibility("at least one state column is required")
    df, nxt = _transitions(data[list(dict.fromkeys(cols))].dropna(), trajectory, time)
    cur = np.flatnonzero(nxt >= 0)
    if cur.size < 20:
        raise DataInsufficient("at least twenty transitions are needed")
    states = df[state]
    yv = df[y].to_numpy(dtype=float)
    wv = df[treat].to_numpy()
    phi, psi, kind = _feature_matrix(states, features)

    estimated_propensity = propensity is None
    if propensity is None:
        codes = states.astype(str).agg("|".join, axis=1)
        if codes.nunique() > 200:
            raise MethodIncompatibility(
                "propensity= is required when the state takes many values"
            )
        cell = pd.DataFrame({"s": codes, "w": wv})
        e_taken = (
            cell.groupby(["s", "w"])["w"].transform("size")
            / cell.groupby("s")["w"].transform("size")
        ).to_numpy(dtype=float)
    elif isinstance(propensity, str):
        e_taken = df[propensity].to_numpy(dtype=float)
    else:
        e_taken = np.full(len(df), float(propensity))
    if not np.all(np.isfinite(e_taken)) or (e_taken <= 0).any() or (e_taken > 1).any():
        raise MethodIncompatibility(
            "propensity must be the probability of the action taken, in (0, 1]"
        )

    fits: List[Dict[str, Any]] = []
    for pol in (policy, baseline):
        if pol is None:
            continue
        target = _target_actions(pol, states, df)
        rho = np.where(wv == target, 1.0 / e_taken, 0.0)
        if rho[cur].sum() == 0:
            raise DataInsufficient(
                "the action taken never agrees with the policy, so its value "
                "cannot be learned from these data"
            )
        fit = _one_policy(yv, rho, phi, psi, cur, nxt[cur])
        if float(fit["omega"].min()) < 0:
            warnings.warn(
                "the estimated stationary ratio is negative for some periods "
                f"(minimum {float(fit['omega'].min()):.3g}); the policy visits "
                "states the data rarely reach, or the basis is too coarse.",
                AssumptionWarning,
                stacklevel=2,
            )
        fits.append(fit)

    m = cur.size
    value_policy = fits[0]["value"]
    value_baseline = fits[1]["value"] if len(fits) > 1 else float("nan")
    score = fits[0]["score"] - (fits[1]["score"] if len(fits) > 1 else 0.0)
    value = value_policy - (value_baseline if len(fits) > 1 else 0.0)
    se = float(np.sqrt(np.sum(score**2)) / m)
    crit = float(stats.norm.ppf(1 - alpha / 2))
    z = value / se if se > 0 else np.nan
    return MDPPolicyValueResult(
        value=float(value),
        se=se,
        ci=(float(value - crit * se), float(value + crit * se)),
        pvalue=float(2 * stats.norm.sf(abs(z))),
        value_policy=float(value_policy),
        value_baseline=float(value_baseline),
        n_obs=int(m),
        alpha=float(alpha),
        detail={
            "basis": kind,
            "n_basis": int(phi.shape[1]),
            "propensity_estimated": bool(estimated_propensity),
            "share_followed": fits[0]["share_followed"],
            "min_omega": float(fits[0]["omega"].min()),
            "max_omega": float(fits[0]["omega"].max()),
            "bellman_value": fits[0]["eta"],
        },
    )


# --------------------------------------------------------------------
# Marginal policy effect
# --------------------------------------------------------------------


@dataclass
class MarginalPolicyEffectResult(ResultProtocolMixin):
    """Result of :func:`marginal_policy_effect`.

    Attributes
    ----------
    estimate : float
        ``theta``: removing a fraction ``eps`` of the treatments that
        occur lowers the average outcome per period by ``eps * theta``.
    se : float
    ci : tuple of float
    per_treatment : float
        ``theta`` divided by the share of treated periods: the average
        effect of one treatment on the sum of the outcomes of the
        current and the next ``horizon`` periods.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> w = rng.binomial(1, 0.3, 500)
    >>> yv = 1.0 * w + rng.normal(size=500)
    >>> df = pd.DataFrame({"y": yv, "w": w, "x": rng.normal(size=500)})
    >>> r = sp.marginal_policy_effect(df, "y", "w", ["x"], horizon=2)
    >>> isinstance(r, sp.MarginalPolicyEffectResult)
    True
    """

    _citation_keys = ()

    estimate: float
    se: float
    ci: Tuple[float, float]
    pvalue: float
    per_treatment: float
    horizon: int
    n_obs: int
    alpha: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        return (
            "Marginal policy effect\n"
            f"  periods = {self.n_obs}, look-ahead = {self.horizon}\n"
            f"  theta = {self.estimate:.6g} (se {self.se:.6g}), "
            f"{100 * (1 - self.alpha):g}% CI = [{self.ci[0]:.6g}, "
            f"{self.ci[1]:.6g}]\n"
            f"  effect per treatment on current and future outcomes: "
            f"{self.per_treatment:.6g}"
        )


def _logit_fit(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        grad = X.T @ (w - p)
        hess = (X * (p * (1 - p))[:, None]).T @ X + 1e-10 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return np.asarray(1.0 / (1.0 + np.exp(-(X @ beta))))


def marginal_policy_effect(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    *,
    horizon: int,
    propensity: Union[None, str, float] = None,
    trajectory: Optional[str] = None,
    time: Optional[str] = None,
    hac_lags: Optional[int] = None,
    alpha: float = 0.05,
) -> MarginalPolicyEffectResult:
    """
    Effect of removing a small share of treatments in a dynamic system.

    Parameters
    ----------
    data : pd.DataFrame
        One row per period, in time order (or sortable by ``time``).
    y : str
        Outcome of the period.
    treat : str
        Binary (0/1) treatment of the period.
    covariates : sequence of str
        Observed part of the state, given which treatment is as good as
        random. May be empty for a randomized treatment.
    horizon : int
        Number of future periods over which a treatment can still affect
        outcomes. The effect is on the sum of the outcomes of the
        current period and the next ``horizon`` periods.
    propensity : str or float, optional
        Probability of treatment given the covariates: a column or one
        number. If omitted, a logistic regression on the covariates.
    trajectory, time : str, optional
        Trajectory identifier and time column.
    hac_lags : int, optional
        Lags of the Bartlett kernel for the standard error. Defaults to
        ``horizon`` plus ``floor(4 (T / 100) ** (2 / 9))``.
    alpha : float, default 0.05

    Returns
    -------
    MarginalPolicyEffectResult

    Notes
    -----
    The estimand is ``theta = mean_t E[e(X_t) * Delta_t(X_t)]``, where
    ``e`` is the treatment probability and ``Delta_t`` the effect of
    treating in period ``t`` on the outcomes of periods ``t`` to
    ``t + horizon``, given the observed covariates. It is identified
    without observing the whole state, provided treatment depends on the
    past only through the covariates. It answers "what are the
    treatments that occur worth, at the margin", not "what if no one
    were treated".

    The estimator is doubly robust. Writing ``G_t`` for the forward sum
    of outcomes and ``m0`` for its linear regression on the covariates
    among untreated periods, it averages
    ``W_t (G_t - m0) - (1 - W_t) e / (1 - e) (G_t - m0)``. The forward
    sums of neighbouring periods overlap, so the standard error uses a
    heteroskedasticity and autocorrelation consistent estimate. The last
    ``horizon`` periods of each trajectory have no complete forward sum
    and are dropped.

    A ``horizon`` that is too short leaves out part of the effect; one
    that is too long adds noise. Report the estimate for several values.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 3000
    >>> w = rng.binomial(1, 0.3, n)
    >>> u = np.zeros(n)
    >>> for t in range(1, n):
    ...     u[t] = 0.5 * u[t - 1] + w[t - 1] + rng.normal()
    >>> yv = w + u + rng.normal(size=n)   # one treatment is worth 1 + 1 + 0.5
    >>> df = pd.DataFrame({"y": yv, "w": w})
    >>> r = sp.marginal_policy_effect(df, "y", "w", [], horizon=2, propensity=0.3)
    >>> bool(r.ci[0] < 0.3 * 2.5 < r.ci[1])
    True
    """
    if not 0 < alpha < 1:
        raise MethodIncompatibility("alpha must be in (0, 1)")
    if int(horizon) != horizon or horizon < 0:
        raise MethodIncompatibility("horizon must be a non-negative integer")
    K = int(horizon)
    covariates = list(covariates)
    cols = [y, treat] + covariates + [c for c in (trajectory, time) if c is not None]
    if isinstance(propensity, str):
        cols.append(propensity)
    missing = [c for c in dict.fromkeys(cols) if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"columns not found in data: {missing}")
    keys = [k for k in (trajectory, time) if k is not None]
    df = data[list(dict.fromkeys(cols))].dropna()
    df = (df.sort_values(keys, kind="stable") if keys else df).reset_index(drop=True)
    wv = df[treat].to_numpy(dtype=float)
    if not np.isin(wv, (0.0, 1.0)).all():
        raise MethodIncompatibility(f"{treat!r} must be coded 0/1")
    yv = df[y].to_numpy(dtype=float)
    group = (
        df[trajectory].to_numpy() if trajectory is not None else np.zeros(len(df), int)
    )

    # Forward sums within a trajectory; incomplete ones are dropped.
    forward = np.full(len(df), np.nan)
    bounds = np.flatnonzero(np.r_[True, group[1:] != group[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = yv[a:b]
        if seg.size > K:
            cs = np.concatenate([[0.0], np.cumsum(seg)])
            forward[a : b - K] = cs[K + 1 :] - cs[: seg.size - K]
    keep = np.isfinite(forward)
    T = int(keep.sum())
    if T < 30:
        raise DataInsufficient(
            "fewer than thirty periods have a complete forward sum; shorten "
            "the horizon"
        )
    G, W = forward[keep], wv[keep]
    X = np.column_stack(
        [np.ones(T)] + [df[c].to_numpy(dtype=float)[keep] for c in covariates]
    )
    if W.min() == W.max():
        raise DataInsufficient("both treated and untreated periods are needed")
    if propensity is None:
        e = _logit_fit(X, W)
    elif isinstance(propensity, str):
        e = df[propensity].to_numpy(dtype=float)[keep]
    else:
        e = np.full(T, float(propensity))
    if not np.all(np.isfinite(e)) or (e <= 0).any() or (e >= 1).any():
        raise MethodIncompatibility("the treatment probability must lie in (0, 1)")
    if float(e.max()) > 0.99:
        warnings.warn(
            "some treatment probabilities exceed 0.99; untreated periods "
            "with such covariates carry very large weights.",
            AssumptionWarning,
            stacklevel=2,
        )
    control = W == 0
    beta, *_ = np.linalg.lstsq(X[control], G[control], rcond=None)
    resid = G - X @ beta
    psi = W * resid - (1 - W) * e / (1 - e) * resid
    theta = float(psi.mean())

    lags = int(hac_lags) if hac_lags is not None else K + int(4 * (T / 100) ** (2 / 9))
    if lags < 0:
        raise MethodIncompatibility("hac_lags must be non-negative")
    u = psi - theta
    g_keep = group[keep]
    pos = np.flatnonzero(keep)
    lrv = float(u @ u)
    for lag in range(1, min(lags, T - 1) + 1):
        # Products only within a trajectory and at the stated distance.
        same = (g_keep[lag:] == g_keep[:-lag]) & (pos[lag:] - pos[:-lag] == lag)
        lrv += (
            2.0 * (1 - lag / (lags + 1)) * float(np.sum(u[lag:][same] * u[:-lag][same]))
        )
    se = float(np.sqrt(max(lrv, 0.0)) / T)
    crit = float(stats.norm.ppf(1 - alpha / 2))
    z = theta / se if se > 0 else np.nan
    share = float(W.mean())
    return MarginalPolicyEffectResult(
        estimate=theta,
        se=se,
        ci=(theta - crit * se, theta + crit * se),
        pvalue=float(2 * stats.norm.sf(abs(z))),
        per_treatment=theta / share,
        horizon=K,
        n_obs=T,
        alpha=float(alpha),
        detail={
            "hac_lags": lags,
            "treated_share": share,
            "propensity_estimated": propensity is None,
            "n_trajectories": int(len(bounds) - 1),
        },
    )


__all__ = [
    "mdp_policy_value",
    "MDPPolicyValueResult",
    "marginal_policy_effect",
    "MarginalPolicyEffectResult",
]
