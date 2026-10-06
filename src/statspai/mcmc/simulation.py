"""
Simulation-based Bayesian inference: ``sp.abc``.

For models that can be simulated but whose likelihood cannot be written
down. Two methods:

* ``'rejection'``: approximate Bayesian computation. Draw parameters from
  the prior, simulate summary statistics, keep the draws whose summaries
  fall close to the observed ones, optionally correct them by a local
  linear regression on the summaries.
* ``'synthetic'``: Bayesian synthetic likelihood. Assume the summaries
  are normal given the parameters, estimate their mean and covariance by
  simulation at each parameter value, and run Metropolis-Hastings on the
  resulting likelihood.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import linalg

from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._results import posterior_table
from .regress import BayesRegressResult


class _Prior:
    """A prior given as independent ``scipy.stats`` distributions or as a
    sampler ``prior(rng, size)``."""

    def __init__(self, prior: Any) -> None:
        self.dists: Optional[List[Any]] = None
        self.sampler: Optional[Callable[..., Any]] = None
        if callable(prior):
            self.sampler = prior
        else:
            dists = list(prior) if isinstance(prior, (list, tuple)) else [prior]
            if not dists or not all(
                hasattr(d, "rvs") and hasattr(d, "logpdf") for d in dists
            ):
                raise MethodIncompatibility(
                    "prior must be a list of frozen scipy.stats distributions, "
                    "one per parameter, or a function prior(rng, size) that "
                    "returns a (size, d) array of draws."
                )
            self.dists = dists

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.dists is not None:
            return np.column_stack(
                [d.rvs(size=size, random_state=rng) for d in self.dists]
            ).astype(float)
        assert self.sampler is not None
        out = np.asarray(self.sampler(rng, size), dtype=float)
        if out.ndim == 1:
            out = out[:, None]
        if out.shape[0] != size:
            raise MethodIncompatibility(
                f"prior(rng, size) must return size rows; got shape {out.shape}."
            )
        return out

    def logpdf(self, theta: np.ndarray) -> float:
        if self.dists is None:
            raise MethodIncompatibility(
                "method='synthetic' evaluates the prior density, so prior must "
                "be a list of scipy.stats distributions, not a sampler."
            )
        return float(sum(d.logpdf(t) for d, t in zip(self.dists, theta)))


def _simulate_many(
    simulate: Callable[..., Any],
    theta: np.ndarray,
    rng: np.random.Generator,
    vectorized: bool,
    n_stats: Optional[int],
) -> np.ndarray:
    if vectorized:
        out = np.asarray(simulate(theta, rng), dtype=float)
        if out.ndim == 1:
            out = out[:, None]
        if out.shape[0] != theta.shape[0]:
            raise MethodIncompatibility(
                "With vectorized=True, simulate(theta, rng) must return one "
                f"row of summaries per row of theta; got shape {out.shape}."
            )
    else:
        first = np.atleast_1d(np.asarray(simulate(theta[0], rng), dtype=float))
        out = np.empty((theta.shape[0], first.size))
        out[0] = first
        for i in range(1, theta.shape[0]):
            out[i] = np.atleast_1d(np.asarray(simulate(theta[i], rng), dtype=float))
    if n_stats is not None and out.shape[1] != n_stats:
        raise MethodIncompatibility(
            f"simulate returns {out.shape[1]} summaries; observed has {n_stats}."
        )
    return out


def abc(
    simulate: Callable[..., Any],
    observed: Any,
    prior: Any,
    method: str = "rejection",
    n_sim: int = 100000,
    tol: Optional[float] = None,
    quantile: float = 0.01,
    adjust: str = "none",
    scale: Any = "mad",
    n_synthetic: int = 100,
    draws: int = 5000,
    burnin: int = 1000,
    names: Optional[Sequence[str]] = None,
    vectorized: bool = False,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> BayesRegressResult:
    """Approximate Bayesian computation and Bayesian synthetic likelihood.

    Posterior inference for a model that can be simulated but has no
    tractable likelihood. The data enter only through summary statistics.

    Parameters
    ----------
    simulate : callable
        ``simulate(theta, rng)`` returns the summary statistics of one
        data set simulated at the parameter vector ``theta``. With
        ``vectorized=True`` it receives an ``(N, d)`` array and returns an
        ``(N, s)`` array.
    observed : array
        The summary statistics of the data.
    prior : list of scipy.stats distributions, or callable
        One frozen distribution per parameter (independent priors), or
        ``prior(rng, size)`` returning a ``(size, d)`` array.
        ``method='synthetic'`` needs the first form.
    method : {'rejection', 'synthetic'}, default 'rejection'
    n_sim : int, default 100000
        Rejection: number of prior draws.
    tol : float, optional
        Rejection: accept a draw when the scaled distance between its
        summaries and the observed ones is at most ``tol``. Default: keep
        the closest ``quantile`` of the draws.
    quantile : float, default 0.01
    adjust : {'none', 'linear'}, default 'none'
        Rejection: ``'linear'`` regresses the accepted parameters on
        their summaries with Epanechnikov weights and moves every draw to
        the observed summaries.
    scale : {'mad', 'sd'}, None or array, default 'mad'
        Rejection: each summary is divided by its median absolute
        deviation (or standard deviation, or the given numbers) over the
        simulations before distances are taken. ``None`` leaves the
        summaries as they are.
    n_synthetic : int, default 100
        Synthetic likelihood: simulations per parameter value.
    draws, burnin : int
        Synthetic likelihood: length of the Metropolis-Hastings chain.
    names : list of str, optional
        Parameter names. Default ``theta1``, ``theta2``, ...
    vectorized : bool, default False
    seed : int, optional
    level : float, default 0.95

    Returns
    -------
    BayesRegressResult
        ``model='abc'``. ``draws`` are the accepted (and adjusted)
        parameters, or the chain. ``acceptance_rate`` is the share of
        prior draws kept, or the Metropolis acceptance rate.
        ``model_info`` reports the tolerance and the scaling used.

    Notes
    -----
    Rejection ABC targets the posterior given that the summaries fall
    within ``tol`` of the observed ones, not the posterior given the
    data. It equals the posterior given the summaries as ``tol`` goes to
    zero, and the posterior given the data only if the summaries are
    sufficient. Check sensitivity to ``quantile``.

    The regression adjustment is exact when the parameters are linear in
    the summaries with a spread that does not depend on them, and an
    extrapolation otherwise.

    The synthetic likelihood assumes normal summaries. Its posterior
    depends on ``n_synthetic`` through the noise of the estimated mean
    and covariance; a few times the number of summaries is the minimum.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import statspai as sp
    >>> y = np.random.default_rng(0).normal(1.0, 1.0, size=50)
    >>> def simulate(theta, rng):
    ...     return rng.normal(theta[:, 0], 1 / np.sqrt(50))[:, None]
    >>> fit = sp.abc(simulate, [y.mean()], [stats.norm(0, 3)], n_sim=20000,
    ...              vectorized=True, seed=1)
    >>> bool(abs(fit.params["theta1"] - y.mean()) < 0.1)
    True

    References
    ----------
    beaumont2002approximate, wood2010statistical
    """
    meth = str(method).lower()
    if meth not in ("rejection", "synthetic"):
        raise MethodIncompatibility(
            f"method must be 'rejection' or 'synthetic'; got {method!r}."
        )
    adj = str(adjust).lower()
    if adj not in ("none", "linear"):
        raise MethodIncompatibility(
            f"adjust must be 'none' or 'linear'; got {adjust!r}."
        )
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    if not callable(simulate):
        raise MethodIncompatibility("simulate must be a function simulate(theta, rng).")
    s_obs = np.atleast_1d(np.asarray(observed, dtype=float))
    if s_obs.ndim != 1 or not np.isfinite(s_obs).all():
        raise MethodIncompatibility("observed must be a finite vector of summaries.")
    pr = _Prior(prior)
    rng = np.random.default_rng(seed)
    probe = pr.sample(rng, 2)
    d = probe.shape[1]
    pnames = (
        [f"theta{j + 1}" for j in range(d)]
        if names is None
        else [str(v) for v in names]
    )
    if len(pnames) != d:
        raise MethodIncompatibility(
            f"names has {len(pnames)} entries for {d} parameters."
        )
    info: Dict[str, Any] = {"method": meth, "n_summaries": int(s_obs.size)}
    notes: List[str] = []

    if meth == "rejection":
        if n_sim < 100:
            raise DataInsufficient("n_sim must be at least 100.")
        if tol is None and not 0.0 < quantile <= 1.0:
            raise MethodIncompatibility(f"quantile must be in (0, 1]; got {quantile}.")
        theta = pr.sample(rng, int(n_sim))
        S = _simulate_many(simulate, theta, rng, vectorized, s_obs.size)
        ok = np.isfinite(S).all(axis=1)
        if not ok.all():
            notes.append(
                f"{int((~ok).sum())} simulations returned non-finite "
                "summaries and were dropped"
            )
            theta, S = theta[ok], S[ok]
        sc: np.ndarray
        if scale is None:
            sc = np.ones(s_obs.size)
        elif isinstance(scale, str):
            key = scale.lower()
            if key == "mad":
                sc = np.median(np.abs(S - np.median(S, axis=0)), axis=0)
            elif key == "sd":
                sc = S.std(axis=0)
            else:
                raise MethodIncompatibility(
                    f"scale must be 'mad', 'sd', None or an array; got {scale!r}."
                )
        else:
            sc = np.broadcast_to(np.asarray(scale, dtype=float), s_obs.shape).copy()
        if np.any(sc <= 0):
            raise MethodIncompatibility(
                "A summary statistic does not vary across simulations (zero "
                "scale); drop it or pass scale=."
            )
        dist = np.sqrt((((S - s_obs) / sc) ** 2).sum(axis=1))
        eps = float(np.quantile(dist, quantile)) if tol is None else float(tol)
        if eps <= 0:
            raise MethodIncompatibility("tol must be positive.")
        keep = dist <= eps
        n_acc = int(keep.sum())
        if n_acc < max(20, 5 * (s_obs.size + 1)):
            raise DataInsufficient(
                f"Only {n_acc} of {len(dist)} simulations fall within the "
                "tolerance. Raise n_sim, tol or quantile, or use fewer summaries."
            )
        th, Sk, dk = theta[keep], S[keep], dist[keep]
        if adj == "linear":
            w = 1.0 - (dk / eps) ** 2  # Epanechnikov
            Z = np.column_stack([np.ones(n_acc), (Sk - s_obs) / sc])
            sw = np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(Z * sw, th * sw, rcond=None)
            th = th - Z[:, 1:] @ coef[1:]
            # an unweighted sample from the weighted adjusted draws
            th = th[rng.choice(n_acc, size=n_acc, replace=True, p=w / w.sum())]
            info["effective_draws"] = float(w.sum() ** 2 / (w**2).sum())
        d_df = pd.DataFrame(th, columns=pnames)
        chain_idx = np.zeros(n_acc, dtype=int)
        rate = n_acc / len(dist)
        info.update(
            tol=eps,
            n_sim=int(len(dist)),
            n_accepted=n_acc,
            scale=pd.Series(sc, index=[f"s{j + 1}" for j in range(s_obs.size)]),
            adjust=adj,
        )
        sampler = "rejection" + (", local linear adjustment" if adj == "linear" else "")
        n_draws, burn = n_acc, 0
    else:
        if n_synthetic < s_obs.size + 3:
            raise DataInsufficient(
                f"n_synthetic={n_synthetic} is too small to estimate the "
                f"covariance of {s_obs.size} summaries."
            )
        if draws < 100 or burnin < 0:
            raise MethodIncompatibility(
                "draws must be at least 100 and burnin non-negative."
            )
        m = int(n_synthetic)

        def synth_loglik(t: np.ndarray) -> float:
            S = _simulate_many(
                simulate, np.tile(t, (m, 1)), rng, vectorized, s_obs.size
            )
            if not np.isfinite(S).all():
                return -np.inf
            mu = S.mean(axis=0)
            cov = np.atleast_2d(np.cov(S, rowvar=False))
            try:
                chol = linalg.cholesky(cov, lower=True)
            except linalg.LinAlgError:
                return -np.inf
            zz = linalg.solve_triangular(chol, s_obs - mu, lower=True)
            return float(-np.log(np.diag(chol)).sum() - 0.5 * zz @ zz)

        # pilot: where the prior predictive comes closest to the data, and
        # how the parameters covary there
        n_pilot = max(2000, 200 * d)
        tp = pr.sample(rng, n_pilot)
        Sp = _simulate_many(simulate, tp, rng, vectorized, s_obs.size)
        okp = np.isfinite(Sp).all(axis=1)
        tp, Sp = tp[okp], Sp[okp]
        scp = np.median(np.abs(Sp - np.median(Sp, axis=0)), axis=0)
        scp = np.where(scp > 0, scp, 1.0)
        dp = np.sqrt((((Sp - s_obs) / scp) ** 2).sum(axis=1))
        near = np.argsort(dp)[: max(10 * d, n_pilot // 50)]
        cur = tp[near].mean(axis=0)
        prop = np.atleast_2d(np.cov(tp[near], rowvar=False)) * (2.38**2 / d)
        prop += 1e-12 * np.eye(d) * max(np.trace(prop), 1e-12)
        chol_p = linalg.cholesky(prop, lower=True)
        lp = pr.logpdf(cur) + synth_loglik(cur)
        tries = 0
        while not np.isfinite(lp) and tries < 50:
            cur = tp[near[tries % len(near)]]
            lp = pr.logpdf(cur) + synth_loglik(cur)
            tries += 1
        if not np.isfinite(lp):
            raise MethodIncompatibility(
                "The synthetic likelihood is not finite near the data. The "
                "summaries may be degenerate (zero variance) at these "
                "parameter values."
            )
        n_iter = int(burnin) + int(draws)
        chain = np.empty((n_iter, d))
        accepted = 0
        adapt_until = int(burnin) // 2
        for it in range(n_iter):
            cand = cur + chol_p @ rng.standard_normal(d)
            lpri = pr.logpdf(cand)
            lp_c = lpri + synth_loglik(cand) if np.isfinite(lpri) else -np.inf
            if np.log(rng.random()) < lp_c - lp:
                cur, lp = cand, lp_c
                accepted += 1
            chain[it] = cur
            if it + 1 == adapt_until and adapt_until >= 200:
                # one re-tune of the proposal from the first half of burn-in
                emp = np.atleast_2d(np.cov(chain[: it + 1], rowvar=False))
                if np.all(np.linalg.eigvalsh(emp) > 0):
                    chol_p = linalg.cholesky(emp * (2.38**2 / d), lower=True)
        d_df = pd.DataFrame(chain[int(burnin) :], columns=pnames)
        chain_idx = np.zeros(int(draws), dtype=int)
        rate = accepted / n_iter
        info.update(n_synthetic=m, n_pilot=int(n_pilot))
        sampler = "Metropolis-Hastings on a Gaussian synthetic likelihood"
        n_draws, burn = int(draws), int(burnin)

    table, diag = posterior_table(d_df, chain_idx, 1, level)
    if meth == "rejection":
        # accepted draws are independent
        table["ess"] = info.get("effective_draws", float(len(d_df)))
        table["mcse"] = table["sd"] / np.sqrt(table["ess"])
        diag["min_ess"] = float(table["ess"].min())
    diag["warnings"].extend(notes)
    res = BayesRegressResult(
        model="abc",
        formula=f"{meth} ABC, {s_obs.size} summaries",
        params=table["mean"].copy(),
        std_errors=table["sd"].copy(),
        table=table,
        draws=d_df,
        chain=chain_idx,
        n_obs=int(s_obs.size),
        n_draws=n_draws,
        burnin=burn,
        thin=1,
        chains=1,
        sampler=sampler,
        acceptance_rate=float(rate),
        prior={
            "prior": (
                "independent scipy.stats distributions" if pr.dists else "user sampler"
            )
        },
        level=level,
        model_info=info,
        diagnostics_info=diag,
        _model=None,
    )
    if meth == "synthetic" and (rate < 0.05 or diag["min_ess"] < 100):
        text = (
            f"The chain mixes slowly (acceptance {rate:.2f}, smallest "
            f"effective sample size {diag['min_ess']:.0f}). A noisy synthetic "
            "likelihood makes the chain stick: raise n_synthetic or draws."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    return res
