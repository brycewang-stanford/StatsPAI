"""
Adaptive experiments: bandit allocation and inference afterwards.

Three pieces.

* :func:`bandit_allocate` turns the data collected so far into
  assignment probabilities for the next subject (Thompson sampling,
  upper confidence bounds, epsilon-greedy).
* :func:`bandit_experiment` runs such a rule sequentially on a reward
  sampler or on a table of potential outcomes, and records the
  assignment probabilities that later inference needs.
* :func:`adaptive_inference` gives confidence intervals for arm means
  and contrasts from adaptively collected data.

Sample means and inverse-probability-weighted means are not
asymptotically normal when assignment probabilities depend on earlier
outcomes. Weighting each observation by the inverse *square root* of
its assignment probability makes the conditional variance of every
term the same, which restores a martingale central limit theorem.

References
----------
[@thompson1933likelihood], [@lai1985asymptotically], [@hadad2021confidence]
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

_ALGORITHMS = ("thompson", "ucb", "epsilon_greedy", "uniform")
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(80)
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(400)


# --------------------------------------------------------------------
# Probability that each arm is best
# --------------------------------------------------------------------


def _prob_best_gaussian(mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """P(arm k has the largest draw) for independent normal posteriors."""
    K = mean.shape[0]
    if K == 2:
        scale = float(np.hypot(sd[0], sd[1]))
        p0 = (
            float(stats.norm.cdf((mean[0] - mean[1]) / scale))
            if scale > 0
            else float(mean[0] > mean[1]) + 0.5 * float(mean[0] == mean[1])
        )
        return np.array([p0, 1.0 - p0])
    out = np.empty(K)
    for k in range(K):
        x = mean[k] + sd[k] * _GH_NODES
        others = np.ones_like(x)
        for j in range(K):
            if j != k:
                others *= stats.norm.cdf((x - mean[j]) / max(sd[j], 1e-300))
        out[k] = float(_GH_WEIGHTS @ others) / np.sqrt(2 * np.pi)
    out = np.clip(out, 0.0, None)
    return np.asarray(out / out.sum())


def _prob_best_beta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """P(arm k has the largest draw) for independent Beta posteriors."""
    x = 0.5 * (_GL_NODES + 1.0)
    w = 0.5 * _GL_WEIGHTS
    K = a.shape[0]
    pdf = np.array([stats.beta.pdf(x, a[k], b[k]) for k in range(K)])
    cdf = np.array([stats.beta.cdf(x, a[k], b[k]) for k in range(K)])
    out = np.empty(K)
    for k in range(K):
        others = np.prod(np.delete(cdf, k, axis=0), axis=0)
        out[k] = float(w @ (pdf[k] * others))
    out = np.clip(out, 0.0, None)
    return np.asarray(out / out.sum())


def _apply_floor(prob: np.ndarray, floor: float) -> np.ndarray:
    """Raise every probability to at least ``floor``, keeping the sum at 1."""
    K = prob.shape[0]
    if floor <= 0:
        return prob
    if floor * K > 1 + 1e-12:
        raise MethodIncompatibility(
            f"prob_floor={floor} is infeasible with {K} arms (needs floor <= 1/K)"
        )
    p = prob.copy()
    for _ in range(K):
        low = p < floor
        if not low.any():
            break
        free = ~low
        mass = 1.0 - floor * low.sum()
        p[low] = floor
        if free.any():
            p[free] = p[free] / p[free].sum() * mass
    return np.asarray(p / p.sum())


def _allocation(
    counts: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray,
    *,
    algorithm: str,
    model: str,
    sigma: Optional[float],
    horizon: Optional[int],
    ucb_scale: float,
    epsilon: float,
    prob_floor: float,
    min_pulls: int,
) -> np.ndarray:
    K = counts.shape[0]
    if algorithm == "uniform":
        return np.full(K, 1.0 / K)
    short = counts < min_pulls
    if short.any():
        # Initialisation: spread the next draws over under-sampled arms.
        p = short / short.sum()
        return np.asarray(p, dtype=float)
    mean = sums / counts
    if model == "bernoulli":
        scale = np.sqrt(np.clip(mean * (1 - mean), 1e-12, None))
    elif sigma is not None:
        scale = np.full(K, float(sigma))
    else:
        total = counts.sum()
        resid = float((sumsq - counts * mean**2).sum())
        dof = max(total - K, 1)
        pooled = np.sqrt(max(resid / dof, 0.0))
        scale = np.full(K, pooled if pooled > 0 else 1.0)

    if algorithm == "thompson":
        if model == "bernoulli":
            p = _prob_best_beta(1.0 + sums, 1.0 + counts - sums)
        else:
            p = _prob_best_gaussian(mean, scale / np.sqrt(counts))
    elif algorithm == "ucb":
        n_total = horizon if horizon is not None else max(int(counts.sum()) + 1, 2)
        index = mean + ucb_scale * scale * np.sqrt(np.log(n_total) / counts)
        p = np.zeros(K)
        p[int(np.argmax(index))] = 1.0
    else:  # epsilon_greedy
        p = np.full(K, epsilon / K)
        p[int(np.argmax(mean))] += 1.0 - epsilon
    return _apply_floor(p, prob_floor)


def _check_algorithm(algorithm: str, model: str, epsilon: float) -> None:
    if algorithm not in _ALGORITHMS:
        raise MethodIncompatibility(f"algorithm must be one of {list(_ALGORITHMS)}")
    if model not in ("gaussian", "bernoulli"):
        raise MethodIncompatibility("model must be 'gaussian' or 'bernoulli'")
    if not 0 <= epsilon <= 1:
        raise MethodIncompatibility("epsilon must be in [0, 1]")


def bandit_allocate(
    data: pd.DataFrame,
    y: str,
    arm: str,
    *,
    arms: Optional[Sequence[Any]] = None,
    algorithm: str = "thompson",
    model: str = "gaussian",
    sigma: Optional[float] = None,
    horizon: Optional[int] = None,
    ucb_scale: float = 2.0,
    epsilon: float = 0.1,
    prob_floor: float = 0.0,
    min_pulls: int = 2,
) -> pd.Series:
    """
    Assignment probabilities for the next subject of an adaptive experiment.

    Parameters
    ----------
    data : pd.DataFrame
        Outcomes observed so far, one row per subject. May be empty.
    y : str
        Outcome (reward) column; larger is better.
    arm : str
        Column with the arm each subject received.
    arms : sequence, optional
        All arm labels, including arms not yet tried. Defaults to the
        labels present in ``data``.
    algorithm : {"thompson", "ucb", "epsilon_greedy", "uniform"}
        ``"thompson"`` assigns each arm with the posterior probability
        that it is the best one. ``"ucb"`` puts probability one on the
        arm with the largest upper confidence bound
        ``mean + ucb_scale * sd * sqrt(log(horizon) / n)``.
        ``"epsilon_greedy"`` plays the best arm so far with probability
        ``1 - epsilon`` and a uniformly drawn arm otherwise.
    model : {"gaussian", "bernoulli"}
        Reward model behind the posterior. ``"gaussian"`` uses a flat
        prior on each mean; ``"bernoulli"`` uses uniform Beta priors and
        needs 0/1 rewards.
    sigma : float, optional
        Known reward standard deviation for the Gaussian model. By
        default the pooled within-arm standard deviation is used.
    horizon : int, optional
        Planned number of subjects, used by ``"ucb"``. Defaults to the
        number observed so far plus one.
    ucb_scale : float, default 2.0
        Width multiplier of the upper confidence bound.
    epsilon : float, default 0.1
        Exploration share of ``"epsilon_greedy"``.
    prob_floor : float, default 0.0
        Lower bound on every assignment probability. A positive floor
        keeps later inference well behaved.
    min_pulls : int, default 2
        Arms with fewer observations are drawn first, with equal
        probability among them.

    Returns
    -------
    pd.Series
        Probabilities indexed by arm label, summing to one.

    References
    ----------
    [@thompson1933likelihood], [@lai1985asymptotically]

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"arm": ["a", "b", "a", "b"], "y": [1.0, 0.2, 0.8, 0.1]})
    >>> p = sp.bandit_allocate(df, "y", "arm", sigma=1.0)
    >>> bool(p["a"] > p["b"])
    True
    """
    _check_algorithm(algorithm, model, epsilon)
    for col in (y, arm):
        if col not in data.columns:
            raise MethodIncompatibility(f"column {col!r} not found in data")
    labels = list(arms) if arms is not None else sorted(pd.unique(data[arm]))
    if len(labels) < 2:
        raise DataInsufficient("at least two arms are needed; pass arms=")
    unknown = set(pd.unique(data[arm])) - set(labels)
    if unknown:
        raise MethodIncompatibility(f"arms in data but not in arms=: {sorted(unknown)}")
    yv = data[y].to_numpy(dtype=float)
    if not np.all(np.isfinite(yv)):
        raise DataInsufficient(f"{y!r} contains missing or non-finite values")
    if model == "bernoulli" and not np.isin(yv, (0.0, 1.0)).all():
        raise MethodIncompatibility("model='bernoulli' needs 0/1 rewards")
    av = data[arm].to_numpy()
    counts = np.array([(av == lab).sum() for lab in labels], dtype=float)
    sums = np.array([yv[av == lab].sum() for lab in labels])
    sumsq = np.array([(yv[av == lab] ** 2).sum() for lab in labels])
    p = _allocation(
        counts,
        sums,
        sumsq,
        algorithm=algorithm,
        model=model,
        sigma=sigma,
        horizon=horizon,
        ucb_scale=ucb_scale,
        epsilon=epsilon,
        prob_floor=prob_floor,
        min_pulls=int(min_pulls),
    )
    return pd.Series(p, index=pd.Index(labels, name=arm), name="probability")


# --------------------------------------------------------------------
# Sequential experiment
# --------------------------------------------------------------------


@dataclass
class BanditExperimentResult(ResultProtocolMixin):
    """Result of :func:`bandit_experiment`.

    Attributes
    ----------
    data : pd.DataFrame
        One row per period: ``t``, ``arm``, ``reward``, ``prob`` (the
        probability with which the chosen arm was assigned) and one
        ``prob_<arm>`` column per arm.
    arms : pd.DataFrame
        Number of pulls and mean reward by arm.
    regret : float or None
        Cumulative shortfall of the expected reward relative to always
        playing the best arm; available when ``true_means`` was given or
        potential outcomes were supplied.

    Examples
    --------
    >>> import statspai as sp
    >>> exp = sp.bandit_experiment(
    ...     lambda k, rng: [0.0, 0.5][k] + rng.normal(), 100, n_arms=2,
    ...     sigma=1.0, seed=0)
    >>> isinstance(exp, sp.BanditExperimentResult)
    True
    >>> list(exp.data.columns)
    ['t', 'arm', 'reward', 'prob', 'prob_0', 'prob_1']
    """

    _citation_keys = ("thompson1933likelihood", "lai1985asymptotically")

    data: pd.DataFrame
    arms: pd.DataFrame
    algorithm: str
    model: str
    n_periods: int
    regret: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lines = [
            f"Adaptive experiment ({self.algorithm}, {self.model} rewards), "
            f"T = {self.n_periods}",
            self.arms.to_string(index=False),
        ]
        if self.regret is not None:
            lines.append(f"Cumulative regret: {self.regret:.4g}")
        return "\n".join(lines)


def bandit_experiment(
    reward: Union[
        Callable[[int, np.random.Generator], float], np.ndarray, pd.DataFrame
    ],
    n_periods: Optional[int] = None,
    *,
    n_arms: Optional[int] = None,
    algorithm: str = "thompson",
    model: str = "gaussian",
    sigma: Optional[float] = None,
    ucb_scale: float = 2.0,
    epsilon: float = 0.1,
    prob_floor: float = 0.0,
    min_pulls: int = 2,
    batch_size: int = 1,
    true_means: Optional[Sequence[float]] = None,
    seed: Optional[int] = None,
) -> BanditExperimentResult:
    """
    Run a sequential multi-armed bandit experiment.

    Parameters
    ----------
    reward : callable, ndarray or DataFrame
        Either a function ``reward(k, rng)`` returning the outcome of
        pulling arm ``k`` (an integer from 0), or a ``T x K`` table of
        potential outcomes whose row ``t`` holds the outcome each arm
        would produce in period ``t``.
    n_periods : int, optional
        Number of subjects. Required with a callable; defaults to the
        number of rows of a potential-outcome table.
    n_arms : int, optional
        Number of arms. Required with a callable.
    algorithm, model, sigma, ucb_scale, epsilon, prob_floor, min_pulls
        As in :func:`bandit_allocate`. ``"ucb"`` uses ``n_periods`` as
        its horizon.
    batch_size : int, default 1
        Assignment probabilities are recomputed every ``batch_size``
        periods.
    true_means : sequence of float, optional
        Mean reward of each arm, used to report regret. With a
        potential-outcome table the column means are used by default.
    seed : int, optional

    Returns
    -------
    BanditExperimentResult
        ``.data`` can be passed to :func:`adaptive_inference`.

    Notes
    -----
    Rules that drive assignment probabilities to zero quickly collect
    high rewards during the experiment but leave little information
    about the arms they abandon. If confidence intervals are wanted
    afterwards, set ``prob_floor`` so that every arm keeps being
    sampled.

    References
    ----------
    [@thompson1933likelihood], [@lai1985asymptotically]

    Examples
    --------
    >>> import statspai as sp
    >>> means = [0.0, 0.5]
    >>> exp = sp.bandit_experiment(
    ...     lambda k, rng: means[k] + rng.normal(), 300, n_arms=2,
    ...     sigma=1.0, prob_floor=0.05, true_means=means, seed=0)
    >>> exp.data.shape[0]
    300
    """
    _check_algorithm(algorithm, model, epsilon)
    if batch_size < 1:
        raise MethodIncompatibility("batch_size must be at least 1")
    rng = np.random.default_rng(seed)
    table: Optional[np.ndarray] = None
    sampler: Optional[Callable[[int, np.random.Generator], float]] = None
    if callable(reward):
        sampler = reward
        if n_periods is None or n_arms is None:
            raise MethodIncompatibility(
                "a callable reward needs n_periods= and n_arms="
            )
        K, T = int(n_arms), int(n_periods)
        labels: List[Any] = list(range(K))
    else:
        if isinstance(reward, pd.DataFrame):
            labels = list(reward.columns)
            table = reward.to_numpy(dtype=float)
        else:
            table = np.asarray(reward, dtype=float)
            if table.ndim != 2:
                raise MethodIncompatibility(
                    "a potential-outcome table must be two-dimensional (T x K)"
                )
            labels = list(range(table.shape[1]))
        K = table.shape[1]
        T = int(n_periods) if n_periods is not None else table.shape[0]
        if T > table.shape[0]:
            raise DataInsufficient(
                "n_periods exceeds the rows of the potential-outcome table"
            )
        if n_arms is not None and int(n_arms) != K:
            raise MethodIncompatibility("n_arms does not match the table")
        if true_means is None:
            true_means = table.mean(axis=0)
    if K < 2:
        raise DataInsufficient("at least two arms are needed")
    if T < 1:
        raise DataInsufficient("n_periods must be positive")

    counts = np.zeros(K)
    sums = np.zeros(K)
    sumsq = np.zeros(K)
    chosen = np.empty(T, dtype=int)
    rewards = np.empty(T)
    probs = np.empty((T, K))
    p = np.full(K, 1.0 / K)
    for t in range(T):
        if t % batch_size == 0:
            p = _allocation(
                counts,
                sums,
                sumsq,
                algorithm=algorithm,
                model=model,
                sigma=sigma,
                horizon=T,
                ucb_scale=ucb_scale,
                epsilon=epsilon,
                prob_floor=prob_floor,
                min_pulls=int(min_pulls),
            )
        k = int(rng.choice(K, p=p))
        if table is not None:
            r = float(table[t, k])
        else:
            assert sampler is not None
            r = float(sampler(k, rng))
        if not np.isfinite(r):
            raise DataInsufficient(f"non-finite reward in period {t}")
        if model == "bernoulli" and r not in (0.0, 1.0):
            raise MethodIncompatibility("model='bernoulli' needs 0/1 rewards")
        chosen[t], rewards[t], probs[t] = k, r, p
        counts[k] += 1
        sums[k] += r
        sumsq[k] += r * r

    frame = pd.DataFrame(
        {
            "t": np.arange(1, T + 1),
            "arm": [labels[k] for k in chosen],
            "reward": rewards,
            "prob": probs[np.arange(T), chosen],
        }
    )
    for k, lab in enumerate(labels):
        frame[f"prob_{lab}"] = probs[:, k]
    with np.errstate(invalid="ignore", divide="ignore"):
        arm_means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    arm_table = pd.DataFrame(
        {"arm": labels, "pulls": counts.astype(int), "mean": arm_means}
    )
    regret = None
    if true_means is not None:
        mu = np.asarray(true_means, dtype=float)
        if mu.shape != (K,):
            raise MethodIncompatibility("true_means must have one entry per arm")
        arm_table["true_mean"] = mu
        regret = float((mu.max() - mu[chosen]).sum())
    return BanditExperimentResult(
        data=frame,
        arms=arm_table,
        algorithm=algorithm,
        model=model,
        n_periods=T,
        regret=regret,
        detail={
            "prob_floor": prob_floor,
            "batch_size": batch_size,
            "min_assigned_prob": float(frame["prob"].min()),
            "arm_labels": labels,
        },
    )


# --------------------------------------------------------------------
# Inference after adaptive data collection
# --------------------------------------------------------------------


@dataclass
class AdaptiveInferenceResult(ResultProtocolMixin):
    """Result of :func:`adaptive_inference`.

    Attributes
    ----------
    estimates : pd.DataFrame
        Arm means with standard errors and confidence intervals.
    contrasts : pd.DataFrame
        Differences between arm means.
    method : str

    Examples
    --------
    >>> import statspai as sp
    >>> exp = sp.bandit_experiment(
    ...     lambda k, rng: [0.0, 0.5][k] + rng.normal(), 200, n_arms=2,
    ...     sigma=1.0, prob_floor=0.1, seed=0)
    >>> r = sp.adaptive_inference(exp.data, "reward", "arm", "prob")
    >>> isinstance(r, sp.AdaptiveInferenceResult)
    True
    >>> r.contrasts["contrast"].tolist()
    ['1 - 0']
    """

    _citation_keys = ("hadad2021confidence",)

    estimates: pd.DataFrame
    contrasts: pd.DataFrame
    method: str
    n_obs: int
    alpha: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        return (
            f"Inference after adaptive data collection ({self.method}), "
            f"T = {self.n_obs}\n{self.estimates.to_string(index=False)}\n\n"
            f"Contrasts:\n{self.contrasts.to_string(index=False)}"
        )


def adaptive_inference(
    data: pd.DataFrame,
    y: str,
    arm: str,
    prob: Optional[str] = None,
    *,
    probs: Optional[Union[Sequence[str], Dict[Any, str]]] = None,
    method: str = "aw",
    contrasts: Optional[Sequence[Tuple[Any, Any]]] = None,
    time: Optional[str] = None,
    alpha: float = 0.05,
) -> AdaptiveInferenceResult:
    """
    Confidence intervals for arm means after an adaptive experiment.

    Parameters
    ----------
    data : pd.DataFrame
        One row per subject, in the order of assignment (or sortable by
        ``time``).
    y : str
        Outcome column.
    arm : str
        Column with the assigned arm.
    prob : str, optional
        Column with the probability with which the *assigned* arm was
        chosen, given the data available at that time. Sufficient for
        ``method="aw"``, ``"ipw"`` and ``"mean"``.
    probs : sequence of str or dict, optional
        Columns with the assignment probability of *every* arm in each
        period: a dict ``{arm label: column}`` or a list aligned with
        the sorted arm labels. Required for ``method="aipw"``.
    method : {"aw", "aipw", "ipw", "mean"}, default "aw"
        ``"aw"`` weights each observation of an arm by the inverse
        square root of its assignment probability and normalises by the
        sum of those weights. ``"aipw"`` applies square-root-probability
        weights to augmented inverse-probability scores whose outcome
        model is the arm's running mean over earlier periods. Both have
        asymptotically normal studentised statistics under adaptive
        assignment. ``"ipw"`` and ``"mean"`` are the inverse-probability
        weighted mean and the sample mean with their usual standard
        errors; they are reported for comparison and their intervals are
        **not** valid when assignment depended on earlier outcomes.
    contrasts : sequence of (arm, arm), optional
        Pairs ``(a, b)`` for which ``mean(a) - mean(b)`` is reported.
        Defaults to every arm against the first.
    time : str, optional
        Column to sort by before computing ``"aipw"`` running means.
    alpha : float, default 0.05

    Returns
    -------
    AdaptiveInferenceResult

    Notes
    -----
    The potential outcomes are assumed independent and identically
    distributed over time, and assignment may depend on earlier
    outcomes only. Assignment probabilities must be the ones actually
    used. Validity needs probabilities that do not fall to zero too
    fast; a warning is issued when an assigned arm had probability
    below ``1 / T`` and an error when a probability is zero or the
    assignment was deterministic.

    The variance of a contrast is the sum of the two arm variances: the
    weighted terms of different arms are uncorrelated because no period
    contributes an outcome to both.

    References
    ----------
    [@hadad2021confidence]

    Examples
    --------
    >>> import statspai as sp
    >>> means = [0.0, 0.5]
    >>> exp = sp.bandit_experiment(
    ...     lambda k, rng: means[k] + rng.normal(), 400, n_arms=2,
    ...     sigma=1.0, prob_floor=0.05, seed=1)
    >>> r = sp.adaptive_inference(exp.data, "reward", "arm", "prob")
    >>> r.estimates.shape[0]
    2
    """
    if method not in ("aw", "aipw", "ipw", "mean"):
        raise MethodIncompatibility("method must be 'aw', 'aipw', 'ipw' or 'mean'")
    if not 0 < alpha < 1:
        raise MethodIncompatibility("alpha must be in (0, 1)")
    needed = [y, arm] + ([prob] if prob else []) + ([time] if time else [])
    for col in needed:
        if col not in data.columns:
            raise MethodIncompatibility(f"column {col!r} not found in data")
    df = data.sort_values(time, kind="stable") if time else data
    yv = df[y].to_numpy(dtype=float)
    av = df[arm].to_numpy()
    T = yv.shape[0]
    if not np.all(np.isfinite(yv)):
        raise DataInsufficient(f"{y!r} contains missing or non-finite values")
    labels = sorted(pd.unique(av))
    K = len(labels)
    if K < 2:
        raise DataInsufficient("at least two arms must have been assigned")

    E: Optional[np.ndarray] = None
    if probs is not None:
        if isinstance(probs, dict):
            missing = [lab for lab in labels if lab not in probs]
            if missing:
                raise MethodIncompatibility(f"probs has no column for arms {missing}")
            cols = [probs[lab] for lab in labels]
        else:
            cols = list(probs)
            if len(cols) != K:
                raise MethodIncompatibility(
                    "probs must list one column per assigned arm, in sorted "
                    "arm order, or be a dict {arm: column}"
                )
        for col in cols:
            if col not in df.columns:
                raise MethodIncompatibility(f"column {col!r} not found in data")
        E = df[cols].to_numpy(dtype=float)
    if method == "aipw" and E is None:
        raise MethodIncompatibility(
            "method='aipw' needs probs=, the assignment probability of every "
            "arm in every period"
        )
    idx = np.array([labels.index(a) for a in av])
    if prob is not None:
        e_assigned = df[prob].to_numpy(dtype=float)
    elif E is not None:
        e_assigned = E[np.arange(T), idx]
    elif method == "mean":
        e_assigned = np.ones(T)
    else:
        raise MethodIncompatibility(
            "pass prob= (or probs=) with the assignment " "probabilities"
        )
    if not np.all(np.isfinite(e_assigned)) or (e_assigned <= 0).any():
        raise MethodIncompatibility(
            "assignment probabilities of the assigned arms must be positive"
        )
    if (e_assigned > 1 + 1e-9).any():
        raise MethodIncompatibility("assignment probabilities exceed one")
    if method in ("aw", "aipw") and np.mean(e_assigned >= 1 - 1e-12) > 0.5:
        raise MethodIncompatibility(
            "most assignments were deterministic (probability one), as under "
            "an upper-confidence-bound rule; the weighted estimators need "
            "randomized assignment"
        )
    if method in ("aw", "aipw") and float(e_assigned.min()) < 1.0 / T:
        warnings.warn(
            "an assigned arm had probability below 1/T (smallest: "
            f"{float(e_assigned.min()):.2e}); the normal approximation may "
            "be poor when probabilities decay this fast.",
            AssumptionWarning,
            stacklevel=2,
        )

    crit = float(stats.norm.ppf(1 - alpha / 2))
    est = np.empty(K)
    var = np.empty(K)
    for k in range(K):
        ind = idx == k
        n_k = int(ind.sum())
        if n_k < 2:
            # One observation fixes the estimate and leaves no variance.
            est[k] = float(yv[ind].mean())
            var[k] = np.nan
            warnings.warn(
                f"arm {labels[k]!r} was assigned once; its standard error "
                "is not available.",
                AssumptionWarning,
                stacklevel=2,
            )
            continue
        if method == "mean":
            est[k] = yv[ind].mean()
            var[k] = yv[ind].var(ddof=1) / n_k
        elif method == "ipw":
            term = ind * yv / e_assigned
            est[k] = term.mean()
            var[k] = term.var(ddof=1) / T
        elif method == "aw":
            w = ind / np.sqrt(e_assigned)
            est[k] = float(w @ yv / w.sum())
            var[k] = float(((w * (yv - est[k])) ** 2).sum() / w.sum() ** 2)
        else:  # aipw
            assert E is not None
            e_k = E[:, k]
            if (e_k[ind] <= 0).any() or not np.all(np.isfinite(e_k)):
                raise MethodIncompatibility(
                    "probs contains a non-positive probability for an " "assigned arm"
                )
            run_sum = np.concatenate([[0.0], np.cumsum(ind * yv)[:-1]])
            run_n = np.concatenate([[0.0], np.cumsum(ind)[:-1]])
            m_hat = np.where(run_n > 0, run_sum / np.maximum(run_n, 1), 0.0)
            score = m_hat + np.where(ind, (yv - m_hat) / np.where(ind, e_k, 1.0), 0.0)
            h = np.sqrt(np.clip(e_k, 0.0, None))
            est[k] = float(h @ score / h.sum())
            var[k] = float(((h * (score - est[k])) ** 2).sum() / h.sum() ** 2)

    se = np.sqrt(var)
    estimates = pd.DataFrame(
        {
            "arm": labels,
            "estimate": est,
            "se": se,
            "ci_lo": est - crit * se,
            "ci_hi": est + crit * se,
            "n": [int((idx == k).sum()) for k in range(K)],
        }
    )
    pairs = [(lab, labels[0]) for lab in labels[1:]] if contrasts is None else contrasts
    rows = []
    for a, b in pairs:
        if a not in labels or b not in labels:
            raise MethodIncompatibility(f"contrast ({a!r}, {b!r}) names an unknown arm")
        ia, ib = labels.index(a), labels.index(b)
        d = float(est[ia] - est[ib])
        s = float(np.sqrt(var[ia] + var[ib]))
        rows.append(
            {
                "contrast": f"{a} - {b}",
                "estimate": d,
                "se": s,
                "pvalue": float(2 * stats.norm.sf(abs(d / s))) if s > 0 else np.nan,
                "ci_lo": d - crit * s,
                "ci_hi": d + crit * s,
            }
        )
    return AdaptiveInferenceResult(
        estimates=estimates,
        contrasts=pd.DataFrame(
            rows, columns=["contrast", "estimate", "se", "pvalue", "ci_lo", "ci_hi"]
        ),
        method=method,
        n_obs=T,
        alpha=alpha,
        detail={
            "valid_under_adaptivity": method in ("aw", "aipw"),
            "min_assigned_prob": float(e_assigned.min()),
        },
    )


__all__ = [
    "bandit_allocate",
    "bandit_experiment",
    "adaptive_inference",
    "BanditExperimentResult",
    "AdaptiveInferenceResult",
]
