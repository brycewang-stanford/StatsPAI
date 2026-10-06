"""
Out-of-sample predictive accuracy from posterior draws: Pareto smoothed
importance sampling, leave-one-out cross-validation, WAIC, K-fold
cross-validation and model comparison on the expected log predictive
density.

Everything here works on a matrix of pointwise log-likelihoods, one row
per posterior draw and one column per observation, so it serves any
sampler: the fits of :func:`statspai.bayes_regress`, a PyMC trace, or
draws produced elsewhere.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import special

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility, StatsPAIWarning

# ---------------------------------------------------------------------
# generalized Pareto fit and Pareto smoothing
# ---------------------------------------------------------------------

_PRIOR_BS = 3.0
_PRIOR_K = 10.0
_MIN_TAIL = 5


def _gpd_fit(x: np.ndarray) -> Tuple[float, float]:
    """Shape and scale of a generalized Pareto fitted to sorted exceedances.

    The empirical Bayes estimator of Zhang and Stephens (2009): a
    posterior mean of ``theta = -k / sigma`` over a data-dependent grid
    under a weak prior, then ``k`` by profile. The shape is finally
    shrunk toward 0.5 with the weight of ten observations, as Vehtari
    et al. (2024, appendix) do to stabilise small tails.

    ``x`` must be sorted ascending and positive.
    """
    n = x.size
    m = 30 + int(np.sqrt(n))
    j = np.arange(1, m + 1, dtype=float)
    quartile = x[int(n / 4.0 + 0.5) - 1]
    theta = 1.0 / x[-1] + (1.0 - np.sqrt(m / (j - 0.5))) / (_PRIOR_BS * quartile)
    k_of_theta = np.log1p(-theta[:, None] * x[None, :]).mean(axis=1)
    profile = n * (np.log(-theta / k_of_theta) - k_of_theta - 1.0)
    weights = np.exp(-special.logsumexp(profile[None, :] - profile[:, None], axis=1))
    keep = weights >= 10.0 * np.finfo(float).eps
    weights = weights[keep] / weights[keep].sum()
    theta_hat = float(np.sum(theta[keep] * weights))
    k = float(np.log1p(-theta_hat * x).mean())
    sigma = -k / theta_hat
    k = (n * k + _PRIOR_K * 0.5) / (n + _PRIOR_K)
    return k, float(sigma)


def _gpd_quantile(p: np.ndarray, k: float, sigma: float) -> np.ndarray:
    if abs(k) < 1e-30:
        return np.asarray(-sigma * np.log1p(-p))
    return np.asarray(sigma * np.expm1(-k * np.log1p(-p)) / k)


def _tail_length(n_draws: int, r_eff: float) -> int:
    return int(np.ceil(min(0.2 * n_draws, 3.0 * np.sqrt(n_draws / r_eff))))


def _psis_one(lw: np.ndarray, r_eff: float) -> Tuple[np.ndarray, float]:
    """Smoothed, normalised log weights and the Pareto shape for one column."""
    lw = lw - lw.max()
    n_draws = lw.size
    tail_len = _tail_length(n_draws, r_eff)
    khat = np.inf
    if tail_len >= _MIN_TAIL and tail_len < n_draws:
        order = np.argsort(lw, kind="stable")
        tail_ids = order[n_draws - tail_len :]
        cutoff = lw[order[n_draws - tail_len - 1]]
        tail = lw[tail_ids]
        if np.unique(tail).size >= _MIN_TAIL and tail[-1] > cutoff:
            exp_cutoff = np.exp(cutoff)
            exceed = np.exp(tail) - exp_cutoff
            positive = exceed > 0
            if positive.sum() >= _MIN_TAIL:
                khat, sigma = _gpd_fit(exceed[positive])
                if np.isfinite(khat):
                    probs = (np.arange(1, tail_len + 1) - 0.5) / tail_len
                    smoothed = np.log(_gpd_quantile(probs, khat, sigma) + exp_cutoff)
                    lw = lw.copy()
                    lw[tail_ids] = np.minimum(smoothed, 0.0)
    lw = lw - special.logsumexp(lw)
    return lw, float(khat)


def k_threshold(n_draws: int) -> float:
    """Largest Pareto shape at which the smoothed estimate is reliable.

    ``min(1 - 1 / log10(S), 0.7)`` for ``S`` draws (Vehtari et al. 2024):
    0.7 from 2,200 draws on, lower for shorter chains.
    """
    return float(min(1.0 - 1.0 / np.log10(n_draws), 0.7))


@dataclass
class PSISResult(ResultProtocolMixin):
    """Result of :func:`psis`.

    Attributes
    ----------
    log_weights : ndarray, shape (draws, observations)
        Smoothed log importance weights, each column normalised to sum
        to one on the natural scale.
    pareto_k : ndarray
        Estimated shape of the generalized Pareto tail of the raw ratios.
    n_eff : ndarray
        Effective sample size of the smoothed weights,
        ``r_eff / sum(w ** 2)``.
    k_threshold : float
        See :func:`k_threshold`.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> out = sp.psis(np.random.default_rng(0).normal(size=(500, 2)))
    >>> out.pareto_k.shape, out.weights().shape
    ((2,), (500, 2))
    """

    log_weights: np.ndarray
    pareto_k: np.ndarray
    n_eff: np.ndarray
    r_eff: np.ndarray
    k_threshold: float

    _citation_keys = ("vehtari2024pareto", "zhang2009new")

    def weights(self) -> np.ndarray:
        """The smoothed weights on the natural scale."""
        return np.asarray(np.exp(self.log_weights))

    def summary(self) -> str:
        table = _k_table(self.pareto_k, self.n_eff, self.k_threshold)
        return "Pareto smoothed importance sampling\n" + str(
            table.to_string(index=False)
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _as_matrix(x: Any, what: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise MethodIncompatibility(
            f"{what} must be a matrix with one row per posterior draw and "
            f"one column per observation; got {arr.ndim} dimensions."
        )
    if not np.all(np.isfinite(arr)):
        raise MethodIncompatibility(
            f"{what} contains missing or infinite values. An observation "
            "with zero likelihood under some draw has no finite log "
            "predictive density; check the model for that observation."
        )
    return arr


def _r_eff_vector(r_eff: Any, n_obs: int) -> np.ndarray:
    if r_eff is None:
        return np.ones(n_obs)
    out: np.ndarray = np.asarray(r_eff, dtype=float).reshape(-1)
    if out.size == 1:
        out = np.asarray(np.repeat(out, n_obs))
    if out.size != n_obs or np.any(~np.isfinite(out)) or np.any(out <= 0):
        raise MethodIncompatibility(
            "r_eff must be a positive number or one positive number per " "observation."
        )
    return out


def psis(log_ratios: Any, r_eff: Any = None) -> PSISResult:
    """Pareto smoothed importance sampling.

    Stabilises importance weights by replacing the largest ones with the
    expected order statistics of a generalized Pareto distribution fitted
    to the upper tail. The fitted shape ``k`` is also the diagnostic:
    it measures how heavy the tail of the ratios is, and so how many
    draws a reliable estimate needs.

    Parameters
    ----------
    log_ratios : array, shape (draws,) or (draws, observations)
        Log importance ratios. For leave-one-out cross-validation these
        are minus the pointwise log-likelihoods.
    r_eff : float or array, optional
        Relative efficiency of the draws (effective sample size divided
        by the number of draws) for each column. Default 1, which is
        right for independent draws and optimistic for a slowly mixing
        chain. It sets the tail length and scales ``n_eff``.

    Returns
    -------
    PSISResult

    Notes
    -----
    The tail has ``ceil(min(0.2 S, 3 sqrt(S / r_eff)))`` draws. Smoothed
    weights are truncated at the largest raw weight. A column whose tail
    has fewer than five distinct values is left unsmoothed and gets
    ``k = inf``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> out = sp.psis(rng.normal(size=(1000, 3)))
    >>> out.log_weights.shape
    (1000, 3)
    >>> bool(np.allclose(np.exp(out.log_weights).sum(axis=0), 1.0))
    True

    References
    ----------
    vehtari2024pareto, zhang2009new
    """
    lr = _as_matrix(log_ratios, "log_ratios")
    n_draws, n_obs = lr.shape
    if n_draws < 25:
        raise DataInsufficient(
            f"psis needs at least 25 draws to fit a tail; got {n_draws}."
        )
    reff = _r_eff_vector(r_eff, n_obs)
    lw = np.empty_like(lr)
    khat = np.empty(n_obs)
    for i in range(n_obs):
        lw[:, i], khat[i] = _psis_one(lr[:, i], reff[i])
    n_eff = reff / np.exp(special.logsumexp(2.0 * lw, axis=0))
    return PSISResult(
        log_weights=lw,
        pareto_k=khat,
        n_eff=n_eff,
        r_eff=reff,
        k_threshold=k_threshold(n_draws),
    )


def _k_table(khat: np.ndarray, n_eff: np.ndarray, threshold: float) -> pd.DataFrame:
    edges = [(-np.inf, threshold, "good"), (threshold, 1.0, "bad")]
    edges.append((1.0, np.inf, "very bad"))
    rows = []
    for lo, hi, label in edges:
        inside = (khat > lo) & (khat <= hi)
        rows.append(
            {
                "range": f"({lo:.2g}, {hi:.2g}]",
                "label": label,
                "count": int(inside.sum()),
                "pct": 100.0 * float(inside.mean()) if khat.size else np.nan,
                "min_n_eff": float(n_eff[inside].min()) if inside.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# result object shared by loo / waic / kfold
# ---------------------------------------------------------------------


@dataclass
class LOOResult(ResultProtocolMixin):
    """Expected log predictive density of a model, with its pieces.

    Returned by :func:`loo`, :func:`waic` and :func:`kfold`.

    Attributes
    ----------
    kind : {'loo', 'waic', 'kfold'}
    estimates : pd.DataFrame
        Rows ``elpd``, ``p`` (effective number of parameters) and ``ic``
        (``-2 elpd``); columns ``estimate`` and ``se``. The standard
        errors measure how much the sum over observations would vary in
        another sample of the same size, not Monte Carlo error.
    pointwise : pd.DataFrame
        One row per observation: ``elpd``, ``p``, ``ic`` and, for
        ``kind='loo'``, ``mcse_elpd``, ``pareto_k`` and ``n_eff``.
    elpd, se_elpd, p, se_p, ic, se_ic : float
        The entries of ``estimates``.
    mcse_elpd : float
        Monte Carlo standard error of ``elpd`` (``kind='loo'``); ``nan``
        when some Pareto shape exceeds the threshold, since the error of
        those terms is not estimable.
    k_threshold : float
        Pareto shapes above it mark observations whose leave-one-out
        density is not reliably estimated.
    n_obs, n_draws : int
    log_weights : ndarray or None
        The smoothed log weights (``kind='loo'``), kept for
        :func:`loo_predict`.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y, mu = rng.normal(size=20), rng.normal(0, 0.2, size=(500, 1))
    >>> ll = -0.5 * (y - mu) ** 2
    >>> out = sp.loo(ll)
    >>> out.kind, out.n_obs
    ('loo', 20)
    """

    kind: str
    estimates: pd.DataFrame
    pointwise: pd.DataFrame
    elpd: float
    se_elpd: float
    p: float
    se_p: float
    ic: float
    se_ic: float
    n_obs: int
    n_draws: int
    mcse_elpd: float = float("nan")
    k_threshold: float = float("nan")
    log_weights: Optional[np.ndarray] = field(default=None, repr=False)
    notes: List[str] = field(default_factory=list)
    model_info: Dict[str, Any] = field(default_factory=dict)

    _citation_keys = ("vehtari2017practical", "vehtari2024pareto")

    @property
    def pareto_k(self) -> np.ndarray:
        """Pareto shape by observation (``kind='loo'``)."""
        if "pareto_k" not in self.pointwise:
            raise MethodIncompatibility(
                f"Pareto shapes belong to PSIS leave-one-out, not to {self.kind}."
            )
        return np.asarray(self.pointwise["pareto_k"].to_numpy())

    def pareto_k_table(self) -> pd.DataFrame:
        """Counts of observations by reliability of their estimate."""
        return _k_table(
            self.pareto_k, self.pointwise["n_eff"].to_numpy(), self.k_threshold
        )

    def bad_observations(self) -> np.ndarray:
        """Positions of the observations whose Pareto shape is too high."""
        return np.flatnonzero(self.pareto_k > self.k_threshold)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "kind": self.kind,
            "elpd": self.elpd,
            "se_elpd": self.se_elpd,
            "p": self.p,
            "se_p": self.se_p,
            "ic": self.ic,
            "se_ic": self.se_ic,
            "mcse_elpd": None if np.isnan(self.mcse_elpd) else self.mcse_elpd,
            "n_obs": self.n_obs,
            "n_draws": self.n_draws,
            "notes": list(self.notes),
        }
        if self.kind == "loo":
            out["k_threshold"] = self.k_threshold
            out["n_bad_k"] = int(self.bad_observations().size)
            out["max_pareto_k"] = float(np.max(self.pareto_k))
        return out

    def summary(self) -> str:
        title = {
            "loo": "Leave-one-out cross-validation (PSIS)",
            "waic": "Widely applicable information criterion",
            "kfold": "K-fold cross-validation",
        }[self.kind]
        lines = [
            title,
            f"Computed from {self.n_draws} draws and {self.n_obs} observations.",
            "",
            self.estimates.to_string(float_format=lambda v: f"{v:.1f}"),
        ]
        if self.kind == "loo":
            lines.append("")
            if np.isnan(self.mcse_elpd):
                lines.append("Monte Carlo SE of elpd is not available.")
            else:
                lines.append(f"Monte Carlo SE of elpd: {self.mcse_elpd:.2g}")
            lines.append("")
            lines.append("Pareto k diagnostic:")
            lines.append(
                self.pareto_k_table().to_string(
                    index=False, float_format=lambda v: f"{v:.1f}"
                )
            )
        for note in self.notes:
            lines.extend(["", f"Note: {note}"])
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def plot(self) -> Any:
        """Pareto shapes by observation, with the reliability threshold."""
        import matplotlib.pyplot as plt

        k = self.pareto_k
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.scatter(np.arange(k.size), k, s=12)
        ax.axhline(self.k_threshold, linestyle="--", linewidth=1)
        ax.set_xlabel("observation")
        ax.set_ylabel("Pareto k")
        fig.tight_layout()
        return fig


def _estimates(pointwise: pd.DataFrame) -> pd.DataFrame:
    n = len(pointwise)
    rows = {}
    for name in ("elpd", "p", "ic"):
        col = pointwise[name].to_numpy()
        se = float(np.sqrt(n * col.var(ddof=1))) if n > 1 else float("nan")
        rows[name] = {"estimate": float(col.sum()), "se": se}
    return pd.DataFrame(rows).T


def _build(
    kind: str,
    pointwise: pd.DataFrame,
    n_draws: int,
    **extra: Any,
) -> LOOResult:
    est = _estimates(pointwise)
    return LOOResult(
        kind=kind,
        estimates=est,
        pointwise=pointwise,
        elpd=float(est.loc["elpd", "estimate"]),
        se_elpd=float(est.loc["elpd", "se"]),
        p=float(est.loc["p", "estimate"]),
        se_p=float(est.loc["p", "se"]),
        ic=float(est.loc["ic", "estimate"]),
        se_ic=float(est.loc["ic", "se"]),
        n_obs=len(pointwise),
        n_draws=int(n_draws),
        **extra,
    )


def pointwise_log_lik(x: Any) -> np.ndarray:
    """The draws-by-observations log-likelihood matrix behind ``x``.

    ``x`` is a fitted Bayesian model with a ``log_lik()`` method or the
    matrix itself.
    """
    getter = getattr(x, "log_lik", None)
    if callable(getter):
        return _as_matrix(getter(), "log_lik()")
    if isinstance(x, pd.DataFrame):
        return _as_matrix(x.to_numpy(), "log_lik")
    if isinstance(x, (LOOResult, str)) or np.isscalar(x):
        raise MethodIncompatibility(
            "Expected a fitted Bayesian model with a log_lik() method or a "
            "matrix of pointwise log-likelihoods (draws by observations)."
        )
    return _as_matrix(x, "log_lik")


def _relative_efficiency(x: Any, ll: np.ndarray) -> Optional[np.ndarray]:
    """Relative efficiency of ``exp(log_lik)`` from a model's own chains."""
    chain = getattr(x, "chain", None)
    if chain is None or not callable(getattr(x, "log_lik", None)):
        return None
    chain_ids = np.asarray(chain)
    if chain_ids.shape != (ll.shape[0],):
        return None
    from .diagnostics import _ess_1d

    lik = np.exp(ll - ll.max(axis=0))
    ids = np.unique(chain_ids)
    out = np.empty(ll.shape[1])
    for i in range(ll.shape[1]):
        ess = 0.0
        for c in ids:
            series = lik[chain_ids == c, i]
            ess += _ess_1d(series) if np.ptp(series) > 0 else series.size
        out[i] = min(ess / ll.shape[0], 1.0)
    return out


def loo(x: Any, r_eff: Any = None, save_weights: bool = True) -> LOOResult:
    """Leave-one-out cross-validation by Pareto smoothed importance sampling.

    Estimates the expected log predictive density for new data,
    ``elpd = sum_i log p(y_i | y_{-i})``, from one fit: the posterior
    draws are reweighted to stand in for the posterior without
    observation ``i``. The Pareto shape of each observation's weights
    says whether that reweighting can be trusted.

    Parameters
    ----------
    x : fitted model or array, shape (draws, observations)
        A result with a ``log_lik()`` method (``sp.bayes_regress``), or
        the pointwise log-likelihood matrix.
    r_eff : float or array, optional
        Relative efficiency of the draws by observation. For a fitted
        model it is computed from the model's chains; for a bare matrix
        it defaults to 1 (independent draws).
    save_weights : bool, default True
        Keep the smoothed log weights on the result, for
        :func:`loo_predict` and :func:`loo_r2`.

    Returns
    -------
    LOOResult

    Notes
    -----
    ``p`` is the effective number of parameters, the difference between
    the within-sample log predictive density and ``elpd``. A value far
    above the number of parameters in the model, or many high Pareto
    shapes, points to a misspecified model or a few very influential
    observations; :func:`kfold` is the remedy that always works.

    The reliability threshold on the Pareto shape depends on the number
    of draws (see :func:`k_threshold`). The fixed 0.7 of older software
    and of Vehtari, Gelman and Gabry (2017) is its limit for long chains.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=80)})
    >>> df["y"] = 1 + 2 * df["x"] + rng.normal(size=80)
    >>> fit = sp.bayes_regress("y ~ x", df, draws=1000, burnin=300, seed=1)
    >>> out = sp.loo(fit)
    >>> bool(1.5 < out.p < 5)
    True

    References
    ----------
    vehtari2017practical, vehtari2024pareto
    """
    ll = pointwise_log_lik(x)
    n_draws, n_obs = ll.shape
    if r_eff is None:
        r_eff = _relative_efficiency(x, ll)
    smooth = psis(-ll, r_eff=r_eff)
    lw = smooth.log_weights
    elpd_i = special.logsumexp(ll + lw, axis=0)
    lpd_i = special.logsumexp(ll, axis=0) - np.log(n_draws)
    # Monte Carlo error of each term: variance of the self-normalised
    # importance sampling estimate of the predictive density, carried to
    # the log scale by the delta method.
    centred = np.exp(ll - elpd_i[None, :]) - 1.0
    var_ratio = np.exp(special.logsumexp(2.0 * lw, axis=0, b=centred**2))
    mcse_i = np.sqrt(var_ratio / smooth.r_eff)
    pointwise = pd.DataFrame(
        {
            "elpd": elpd_i,
            "mcse_elpd": mcse_i,
            "p": lpd_i - elpd_i,
            "ic": -2.0 * elpd_i,
            "pareto_k": smooth.pareto_k,
            "n_eff": smooth.n_eff,
        }
    )
    bad = smooth.pareto_k > smooth.k_threshold
    notes: List[str] = []
    if bad.any():
        notes.append(
            f"{int(bad.sum())} of {n_obs} Pareto k values exceed "
            f"{smooth.k_threshold:.2f}; the estimate for those observations "
            "is unreliable. Use sp.kfold, or look at them: they are the "
            "observations the model finds most surprising."
        )
        warnings.warn(notes[-1], StatsPAIWarning, stacklevel=2)
    mcse = float("nan") if bad.any() else float(np.sqrt(np.sum(mcse_i**2)))
    return _build(
        "loo",
        pointwise,
        n_draws,
        mcse_elpd=mcse,
        k_threshold=smooth.k_threshold,
        log_weights=lw if save_weights else None,
        notes=notes,
    )


def waic(x: Any) -> LOOResult:
    """Widely applicable information criterion.

    ``elpd_waic = sum_i [log mean_s p(y_i | theta_s) - var_s log p(y_i |
    theta_s)]``: the within-sample log predictive density minus the
    posterior variance of each pointwise log-likelihood (Watanabe 2010).
    Asymptotically equal to leave-one-out cross-validation; in finite
    samples :func:`loo` is more robust and carries its own diagnostic,
    so prefer it.

    Parameters
    ----------
    x : fitted model or array, shape (draws, observations)

    Returns
    -------
    LOOResult
        With ``kind='waic'``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y, mu = rng.normal(size=20), rng.normal(0, 0.2, size=(500, 1))
    >>> ll = -0.5 * (y - mu) ** 2
    >>> round(sp.waic(ll).ic, 1) == round(-2 * sp.waic(ll).elpd, 1)
    True

    References
    ----------
    watanabe2010asymptotic, vehtari2017practical
    """
    ll = pointwise_log_lik(x)
    n_draws = ll.shape[0]
    if n_draws < 2:
        raise DataInsufficient("waic needs at least two draws.")
    lpd_i = special.logsumexp(ll, axis=0) - np.log(n_draws)
    p_i = ll.var(axis=0, ddof=1)
    pointwise = pd.DataFrame(
        {"elpd": lpd_i - p_i, "p": p_i, "ic": -2.0 * (lpd_i - p_i)}
    )
    notes: List[str] = []
    high = int(np.sum(p_i > 0.4))
    if high:
        notes.append(
            f"{high} of {len(p_i)} pointwise variances exceed 0.4, where the "
            "WAIC approximation is poor. Use sp.loo."
        )
        warnings.warn(notes[-1], StatsPAIWarning, stacklevel=2)
    return _build("waic", pointwise, n_draws, notes=notes)


# ---------------------------------------------------------------------
# K-fold
# ---------------------------------------------------------------------


def kfold_split(
    n: int,
    k: int = 10,
    seed: Optional[int] = None,
    groups: Optional[Any] = None,
    stratify: Optional[Any] = None,
) -> np.ndarray:
    """Assign ``n`` observations to ``k`` folds of near-equal size.

    Parameters
    ----------
    n : int
    k : int, default 10
    seed : int, optional
    groups : array-like, optional
        Keep the observations of a group in one fold (clustered data).
    stratify : array-like, optional
        Spread the levels of this variable evenly over folds.

    Returns
    -------
    ndarray of int
        Fold of each observation, ``0 .. k - 1``.

    Examples
    --------
    >>> import statspai as sp
    >>> sorted(set(sp.kfold_split(10, k=5, seed=1).tolist()))
    [0, 1, 2, 3, 4]
    """
    if groups is not None and stratify is not None:
        raise MethodIncompatibility("Pass groups or stratify, not both.")
    rng = np.random.default_rng(seed)
    if k < 2 or k > n:
        raise MethodIncompatibility(
            f"k must be between 2 and the number of observations; got {k}."
        )
    if groups is not None:
        codes, uniques = pd.factorize(np.asarray(groups))
        if len(uniques) < k:
            raise MethodIncompatibility(f"{len(uniques)} groups cannot fill {k} folds.")
        fold_of_group = np.arange(len(uniques)) % k
        rng.shuffle(fold_of_group)
        return np.asarray(fold_of_group[codes])
    if stratify is not None:
        out = np.empty(n, dtype=int)
        codes, _ = pd.factorize(np.asarray(stratify))
        start = 0
        for c in np.unique(codes):
            idx = np.flatnonzero(codes == c)
            rng.shuffle(idx)
            out[idx] = (start + np.arange(idx.size)) % k
            start += idx.size
        return out
    out = np.arange(n) % k
    rng.shuffle(out)
    return out


def kfold(
    fit: Any,
    k: int = 10,
    folds: Optional[Any] = None,
    seed: Optional[int] = None,
    refit: Optional[Callable[[pd.DataFrame], Any]] = None,
    data: Optional[pd.DataFrame] = None,
) -> LOOResult:
    """K-fold cross-validation of a Bayesian model by refitting.

    Holds out each fold in turn, refits on the rest, and scores every
    held-out observation by its log predictive density averaged over the
    posterior draws of the fit that did not see it. Slower than
    :func:`loo` and free of its importance-sampling approximation: the
    choice when Pareto shapes are high, and the only valid one when
    whole groups must be left out together.

    Parameters
    ----------
    fit : fitted model
        A result of ``sp.bayes_regress``. It is refitted with the same
        settings on each training set.
    k : int, default 10
    folds : array-like of int, optional
        Fold of each observation, e.g. from :func:`kfold_split` with
        ``groups=``. Overrides ``k`` and ``seed``.
    seed : int, optional
        Seed of the random fold assignment.
    refit : callable, optional
        ``refit(train) -> fitted model`` for a model that does not carry
        its own refitting recipe; the returned object must have
        ``log_lik(data)``. Requires ``data``.
    data : DataFrame, optional
        The estimation data, when ``refit`` is given.

    Returns
    -------
    LOOResult
        With ``kind='kfold'``. ``p`` is the within-sample log predictive
        density of the full fit minus the cross-validated one.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=60)})
    >>> df["y"] = 1 + 2 * df["x"] + rng.normal(size=60)
    >>> fit = sp.bayes_regress("y ~ x", df, draws=500, burnin=200, seed=1)
    >>> out = sp.kfold(fit, k=5, seed=1)
    >>> out.kind
    'kfold'

    References
    ----------
    vehtari2017practical
    """
    if refit is None:
        refit_own = getattr(fit, "_refit", None)
        frame = getattr(fit, "_frame", None)
        if not callable(refit_own) or frame is None:
            raise MethodIncompatibility(
                "This model does not carry a refitting recipe. Pass "
                "refit=lambda train: <fit on train> and data=."
            )
        refit, data = refit_own, frame
    elif data is None:
        raise MethodIncompatibility("refit= needs data=.")
    assert data is not None
    n = len(data)
    if folds is None:
        fold = kfold_split(n, k=k, seed=seed)
    else:
        fold = np.asarray(folds).reshape(-1)
        if fold.size != n:
            raise MethodIncompatibility(
                f"folds has {fold.size} entries for {n} observations."
            )
        fold = pd.factorize(fold, sort=True)[0]
    labels = np.unique(fold)
    if labels.size < 2:
        raise MethodIncompatibility("Cross-validation needs at least two folds.")
    elpd_i = np.empty(n)
    n_draws = 0
    for label in labels:
        held = fold == label
        sub = refit(data.loc[~held])
        ll = _as_matrix(sub.log_lik(data.loc[held]), "log_lik(held-out data)")
        n_draws = ll.shape[0]
        elpd_i[held] = special.logsumexp(ll, axis=0) - np.log(n_draws)
    full = pointwise_log_lik(fit)
    if full.shape[1] != n:
        raise MethodIncompatibility(
            "The fitted model and data have different numbers of observations."
        )
    lpd_i = special.logsumexp(full, axis=0) - np.log(full.shape[0])
    pointwise = pd.DataFrame(
        {"elpd": elpd_i, "p": lpd_i - elpd_i, "ic": -2.0 * elpd_i, "fold": fold}
    )
    return _build("kfold", pointwise, n_draws, model_info={"k": int(labels.size)})


# ---------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------


def loo_compare(*models: Any, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Compare models on expected log predictive density.

    Parameters
    ----------
    *models : LOOResult, or a single dict of them
        Results of :func:`loo`, :func:`waic` or :func:`kfold` computed on
        the same observations. Fitted models are accepted and passed
        through :func:`loo` first.
    names : list of str, optional
        Labels; ``model1``, ``model2``, ... by default, or the keys of
        the dict.

    Returns
    -------
    pd.DataFrame
        One row per model, best first. ``elpd_diff`` is the difference
        from the best model and ``se_diff`` its standard error, computed
        from the paired pointwise differences (much smaller than the
        standard errors of the two totals when the models' predictions
        are correlated, which they nearly always are). The remaining
        columns repeat each model's ``elpd``, ``p`` and ``ic``.

    Notes
    -----
    A difference of less than about 4 is small whatever its standard
    error; for larger ones compare ``elpd_diff`` with ``se_diff``. The
    normal approximation behind ``se_diff`` is itself unreliable with
    fewer than about 100 observations or when the models are nearly
    identical.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y, mu = rng.normal(size=30), rng.normal(0, 0.2, size=(500, 1))
    >>> a = -0.5 * (y - mu) ** 2
    >>> out = sp.loo_compare(sp.loo(a), sp.loo(a - 0.1), names=["a", "b"])
    >>> out.index[0]
    'a'

    References
    ----------
    vehtari2017practical
    """
    if len(models) == 1 and isinstance(models[0], dict):
        names = list(models[0].keys()) if names is None else names
        models = tuple(models[0].values())
    if len(models) < 2:
        raise MethodIncompatibility("loo_compare needs at least two models.")
    results = [m if isinstance(m, LOOResult) else loo(m) for m in models]
    labels = (
        [f"model{i + 1}" for i in range(len(results))]
        if names is None
        else [str(v) for v in names]
    )
    if len(labels) != len(results):
        raise MethodIncompatibility("names must have one label per model.")
    n_obs = {r.n_obs for r in results}
    if len(n_obs) != 1:
        raise MethodIncompatibility(
            "The models were evaluated on different numbers of observations "
            f"({sorted(n_obs)}); predictive densities are comparable only on "
            "the same data. A transformed outcome also needs the Jacobian "
            "added to its pointwise log-likelihoods."
        )
    kinds = {r.kind for r in results}
    if len(kinds) != 1:
        warnings.warn(
            f"Comparing estimates of different kinds ({sorted(kinds)}).",
            StatsPAIWarning,
            stacklevel=2,
        )
    n = n_obs.pop()
    order = np.argsort([-r.elpd for r in results], kind="stable")
    best = results[order[0]].pointwise["elpd"].to_numpy()
    rows = []
    for j in order:
        r = results[j]
        diff = r.pointwise["elpd"].to_numpy() - best
        se = float(np.sqrt(n * diff.var(ddof=1))) if n > 1 else float("nan")
        rows.append(
            {
                "elpd_diff": float(diff.sum()),
                "se_diff": 0.0 if j == order[0] else se,
                "elpd": r.elpd,
                "se_elpd": r.se_elpd,
                "p": r.p,
                "se_p": r.se_p,
                "ic": r.ic,
                "se_ic": r.se_ic,
            }
        )
    return pd.DataFrame(rows, index=[labels[j] for j in order])


# ---------------------------------------------------------------------
# leave-one-out predictions
# ---------------------------------------------------------------------


def _loo_with_weights(fit: Any, loo_result: Optional[LOOResult]) -> LOOResult:
    if loo_result is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StatsPAIWarning)
            loo_result = loo(fit)
    if loo_result.kind != "loo" or loo_result.log_weights is None:
        raise MethodIncompatibility(
            "A PSIS leave-one-out result with its weights is required "
            "(sp.loo(fit, save_weights=True))."
        )
    return loo_result


def loo_predict(
    fit: Any,
    loo_result: Optional[LOOResult] = None,
    what: str = "mean",
    draws: Optional[Any] = None,
) -> np.ndarray:
    """Leave-one-out prediction of each observation from the other ones.

    The posterior expectation of ``E[y_i | x_i]`` reweighted to the
    posterior that did not see observation ``i``. Unlike fitted values,
    these do not benefit from having been fitted to the point they
    predict, so residuals from them are honest.

    Parameters
    ----------
    fit : fitted model
        A result with ``predict(what='draws')`` (``sp.bayes_regress``).
    loo_result : LOOResult, optional
        The output of ``sp.loo(fit)``; computed when omitted.
    what : {'mean', 'linear'}
        Scale of the prediction: the expected outcome or the linear
        index.
    draws : array, shape (draws, observations), optional
        Predict this quantity instead, e.g. a matrix of your own
        posterior functionals.

    Returns
    -------
    ndarray
        One prediction per observation.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=60)})
    >>> df["y"] = 1 + 2 * df["x"] + rng.normal(size=60)
    >>> fit = sp.bayes_regress("y ~ x", df, draws=500, burnin=200, seed=1)
    >>> sp.loo_predict(fit).shape
    (60,)

    References
    ----------
    vehtari2017practical
    """
    res = _loo_with_weights(fit, loo_result)
    if draws is None:
        if what == "mean":
            values = np.asarray(fit.predict(what="draws"), dtype=float)
        elif what == "linear":
            values = np.asarray(fit.posterior_linpred(), dtype=float)
        else:
            raise MethodIncompatibility("what must be 'mean' or 'linear'.")
    else:
        values = _as_matrix(draws, "draws")
    assert res.log_weights is not None
    if values.shape != res.log_weights.shape:
        raise MethodIncompatibility(
            f"The predictions have shape {values.shape}; the weights "
            f"{res.log_weights.shape}."
        )
    return np.asarray(np.sum(np.exp(res.log_weights) * values, axis=0))


__all__ = [
    "PSISResult",
    "LOOResult",
    "psis",
    "loo",
    "waic",
    "kfold",
    "kfold_split",
    "loo_compare",
    "loo_predict",
    "k_threshold",
    "pointwise_log_lik",
]
