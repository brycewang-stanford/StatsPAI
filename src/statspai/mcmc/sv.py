"""
Stochastic volatility by MCMC: ``sp.stochvol``.

    y_t = exp(h_t / 2) eps_t,            eps_t ~ N(0, 1)
    h_t = mu + phi (h_{t-1} - mu) + sigma eta_t

The log of ``y_t^2`` is ``h_t`` plus a log chi-square(1) error. That error
is replaced by a mixture of seven normals, which makes the model linear
and Gaussian given the mixture indicators, so the whole volatility path is
drawn in one block by forward filtering and backward sampling. The priors
are those of the R package ``stochvol``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._core import check_mcmc_args, rtruncnorm, spawn_rngs
from ._results import posterior_table
from .regress import BayesRegressResult

# Seven-normal approximation of the log chi-square(1) density: weights,
# means (before the common shift of -1.2704) and variances. The test suite
# checks the approximation against the exact density.
_MIX_Q = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
_MIX_M = (
    np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819])
    - 1.2704
)
_MIX_V = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])

_KERNEL: Dict[str, Any] = {}


def _ffbs_py(
    ystar: np.ndarray,
    off: np.ndarray,
    var: np.ndarray,
    mu: float,
    phi: float,
    sig2: float,
    z: np.ndarray,
) -> np.ndarray:
    """Draw the AR(1) state path given ``ystar_t = h_t + off_t + N(0, var_t)``."""
    n = ystar.shape[0]
    m = np.zeros(n)
    c = np.zeros(n)
    a_prev = mu
    r_prev = sig2 / (1.0 - phi * phi)
    for t in range(n):
        if t > 0:
            a_prev = mu + phi * (m[t - 1] - mu)
            r_prev = phi * phi * c[t - 1] + sig2
        q = r_prev + var[t]
        k = r_prev / q
        m[t] = a_prev + k * (ystar[t] - off[t] - a_prev)
        c[t] = r_prev * (1.0 - k)
    h = np.zeros(n)
    h[n - 1] = m[n - 1] + np.sqrt(c[n - 1]) * z[n - 1]
    for t in range(n - 2, -1, -1):
        r_next = phi * phi * c[t] + sig2
        j = phi * c[t] / r_next
        mean = m[t] + j * (h[t + 1] - mu - phi * (m[t] - mu))
        v = c[t] - j * j * r_next
        if v < 0.0:
            v = 0.0
        h[t] = mean + np.sqrt(v) * z[t]
    return h


def _ffbs() -> Any:
    if "f" not in _KERNEL:
        from numba import njit  # type: ignore[import-untyped]

        _KERNEL["f"] = njit(cache=True)(_ffbs_py)
    return _KERNEL["f"]


def _sv_sweep(
    rng: np.random.Generator,
    ystar: np.ndarray,
    h: np.ndarray,
    theta: np.ndarray,
    prior: Tuple[float, float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One sweep: indicators, volatility path, then (mu, phi, sigma2)."""
    b_mu, B_mu, a0, b0, B_sig = prior
    mu, phi, sig2 = theta
    n = ystar.shape[0]
    # 1. mixture indicators
    dev = ystar[:, None] - h[:, None] - _MIX_M[None, :]
    logp = (
        np.log(_MIX_Q)[None, :]
        - 0.5 * np.log(_MIX_V)[None, :]
        - 0.5 * dev * dev / _MIX_V
    )
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    cum = np.cumsum(p, axis=1)
    s = (rng.random(n)[:, None] * cum[:, -1:] > cum).sum(axis=1)
    s = np.minimum(s, 6)
    # 2. volatility path
    h = _ffbs()(ystar, _MIX_M[s], _MIX_V[s], mu, phi, sig2, rng.standard_normal(n))
    # 3a. mu
    prec = 1.0 / B_mu + ((1.0 - phi * phi) + (n - 1) * (1.0 - phi) ** 2) / sig2
    num = (
        b_mu / B_mu
        + ((1.0 - phi * phi) * h[0] + (1.0 - phi) * (h[1:] - phi * h[:-1]).sum()) / sig2
    )
    mu = num / prec + rng.standard_normal() / np.sqrt(prec)
    # 3b. phi: proposal from the AR regression, corrected for the prior and
    #     the stationary density of the first state
    x = h[:-1] - mu
    sxx = float(x @ x)
    cand = float(
        rtruncnorm(
            rng,
            np.array([float(x @ (h[1:] - mu)) / sxx]),
            np.sqrt(sig2 / sxx),
            -1.0,
            1.0,
        )[0]
    )

    def g(ph: float) -> float:
        return float(
            (a0 - 1.0) * np.log1p(ph)
            + (b0 - 1.0) * np.log1p(-ph)
            + 0.5 * np.log1p(-ph * ph)
            - (1.0 - ph * ph) * (h[0] - mu) ** 2 / (2.0 * sig2)
        )

    if np.log(rng.random()) < g(cand) - g(phi):
        phi = cand
    # 3c. sigma2: inverse-gamma proposal from the likelihood, corrected for
    #     the Gamma(1/2, 1 / (2 B)) prior
    e = h[1:] - mu - phi * (h[:-1] - mu)
    ss = (1.0 - phi * phi) * (h[0] - mu) ** 2 + float(e @ e)
    cand = (ss / 2.0) / rng.gamma((n - 1) / 2.0)
    if np.log(rng.random()) < -(cand - sig2) / (2.0 * B_sig):
        sig2 = cand
    # 4. interweaving: redraw (mu, sigma) with the standardised path
    #    (h - mu) / sigma held fixed, where both are regression
    #    coefficients of the linearised observation equation
    sig = np.sqrt(sig2)
    ht = (h - mu) / sig
    w = 1.0 / _MIX_V[s]
    yy = ystar - _MIX_M[s]
    p00 = w.sum() + 1.0 / B_mu
    p01 = float(w @ ht)
    p11 = float(w @ (ht * ht)) + 1.0 / B_sig
    r0 = float(w @ yy) + b_mu / B_mu
    r1 = float(w @ (ht * yy))
    det = p00 * p11 - p01 * p01
    m0 = (p11 * r0 - p01 * r1) / det
    m1 = (p00 * r1 - p01 * r0) / det
    l00 = np.sqrt(p11 / det)
    l10 = -p01 / det / l00
    l11 = np.sqrt(max(p00 / det - l10 * l10, 0.0))
    z0, z1 = rng.standard_normal(2)
    mu = m0 + l00 * z0
    sig_new = m1 + l10 * z0 + l11 * z1
    h = mu + sig_new * ht
    sig2 = sig_new * sig_new
    return h, np.array([mu, phi, sig2]), s


def stochvol(
    y: Union[str, np.ndarray, pd.Series],
    data: Optional[pd.DataFrame] = None,
    demean: bool = True,
    mu_prior: Tuple[float, float] = (0.0, 100.0),
    phi_prior: Tuple[float, float] = (5.0, 1.5),
    sigma_prior: float = 1.0,
    draws: int = 10000,
    burnin: int = 2000,
    thin: int = 1,
    chains: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> BayesRegressResult:
    """Stochastic volatility model for a return series.

    The log variance ``h_t`` of the returns follows a stationary AR(1)
    with level ``mu``, persistence ``phi`` and volatility of volatility
    ``sigma``. Unlike GARCH the variance has its own shock, and the whole
    path of volatilities is estimated with its uncertainty.

    Parameters
    ----------
    y : array, Series or str
        Returns (not prices), or a column name of ``data``.
    data : DataFrame, optional
    demean : bool, default True
        Subtract the sample mean first.
    mu_prior : (mean, variance), default (0, 100)
        Normal prior of the level of the log variance.
    phi_prior : (a, b), default (5, 1.5)
        ``(phi + 1) / 2 ~ Beta(a, b)``: prior mean of ``phi`` 0.54, most
        mass on positive persistence.
    sigma_prior : float, default 1
        ``sigma^2 ~ Gamma(1/2, 1 / (2 * sigma_prior))``, equivalently
        ``+-sigma ~ N(0, sigma_prior)``.
    draws, burnin, thin, chains, seed, level
        As in :func:`statspai.bayes_regress`.

    Returns
    -------
    BayesRegressResult
        ``model='stochvol'``, parameters ``mu``, ``phi``, ``sigma``.
        ``model_info['volatility']`` is a DataFrame with the posterior
        mean and interval of the standard deviation ``exp(h_t / 2)`` at
        every date.

    Notes
    -----
    The log chi-square error of ``log y_t^2`` is approximated by a
    mixture of seven normals; the approximation error is small next to
    posterior uncertainty in samples of the usual size. Exact zeros in
    ``y`` are offset by a small constant before taking logs, and the
    result says so.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> h = np.zeros(300)
    >>> for t in range(1, 300):
    ...     h[t] = -1 + 0.95 * (h[t - 1] + 1) + 0.2 * rng.normal()
    >>> y = np.exp(h / 2) * rng.normal(size=300)
    >>> fit = sp.stochvol(y, draws=500, burnin=300, seed=1)
    >>> list(fit.params.index)
    ['mu', 'phi', 'sigma']
    >>> fit.model_info["volatility"].shape
    (300, 3)

    References
    ----------
    kastner2014ancillarity, ramirezhassan2026introduction
    """
    check_mcmc_args(draws, burnin, thin, chains)
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    if isinstance(y, str):
        if data is None or y not in data.columns:
            raise MethodIncompatibility(f"{y!r} is not a column of data.")
        series = data[y]
    else:
        series = pd.Series(np.asarray(y, dtype=float).reshape(-1))
    index = series.index
    r = series.to_numpy(dtype=float)
    if not np.isfinite(r).all():
        raise MethodIncompatibility(
            "The series has missing or infinite values; remove them first."
        )
    n = r.size
    if n < 30:
        raise DataInsufficient(f"{n} observations are too few for a volatility model.")
    if demean:
        r = r - r.mean()
    if np.allclose(r, 0.0):
        raise MethodIncompatibility("The series is constant.")
    b_mu, B_mu = (float(v) for v in mu_prior)
    a0, b0 = (float(v) for v in phi_prior)
    B_sig = float(sigma_prior)
    if B_mu <= 0 or a0 <= 0 or b0 <= 0 or B_sig <= 0:
        raise MethodIncompatibility("The prior scales must be positive.")
    notes = []
    offset = 0.0
    if np.any(r == 0.0):
        offset = 1e-6 * float(np.mean(r * r))
        notes.append(
            f"{int((r == 0).sum())} zero returns: {offset:.3g} was added to "
            "the squared returns before taking logs"
        )
    ystar = np.log(r * r + offset)
    prior = (b_mu, B_mu, a0, b0, B_sig)
    n_iter = burnin + draws * thin
    rngs = spawn_rngs(seed, chains)
    pieces = []
    store = []
    for ch in range(chains):
        rng = rngs[ch]
        theta = np.array([float(ystar.mean() + 1.27), 0.9, 0.05])
        if chains > 1:
            theta[0] += rng.normal()
            theta[1] = rng.uniform(0.5, 0.97)
        h = np.full(n, theta[0])
        out = np.empty((draws, 3))
        hs = np.empty((draws, n), dtype=np.float32)
        kept = 0
        for it in range(n_iter):
            h, theta, _ = _sv_sweep(rng, ystar, h, theta, prior)
            if it >= burnin and (it - burnin) % thin == 0:
                out[kept] = (theta[0], theta[1], np.sqrt(theta[2]))
                hs[kept] = h
                kept += 1
        pieces.append(out)
        store.append(hs)
    hs_all = np.vstack(store)
    sd_path = np.exp(0.5 * hs_all)
    lo = (1.0 - level) / 2.0
    vol = pd.DataFrame(
        {
            "mean": sd_path.mean(axis=0),
            "lower": np.quantile(sd_path, lo, axis=0),
            "upper": np.quantile(sd_path, 1.0 - lo, axis=0),
        },
        index=index,
    )
    d_df = pd.DataFrame(np.vstack(pieces), columns=["mu", "phi", "sigma"])
    chain_idx = np.repeat(np.arange(chains), draws)
    table, diag = posterior_table(d_df, chain_idx, chains, level)
    diag["warnings"].extend(notes)
    res = BayesRegressResult(
        model="stochvol",
        formula="y_t = exp(h_t / 2) eps_t",
        params=table["mean"].copy(),
        std_errors=table["sd"].copy(),
        table=table,
        draws=d_df,
        chain=chain_idx,
        n_obs=n,
        n_draws=draws * chains,
        burnin=burnin,
        thin=thin,
        chains=chains,
        sampler="Gibbs with interweaving (normal-mixture approximation, FFBS)",
        acceptance_rate=None,
        prior={
            "mu": f"N({b_mu:g}, {B_mu:g})",
            "phi": f"(phi + 1) / 2 ~ Beta({a0:g}, {b0:g})",
            "sigma2": f"Gamma(0.5, rate {1 / (2 * B_sig):g})",
        },
        level=level,
        model_info={"volatility": vol, "offset": offset, "demean": bool(demean)},
        diagnostics_info=diag,
        _model=None,
    )
    if diag["min_ess"] < 100:
        text = (
            "The chain mixes slowly: effective sample size "
            f"{diag['min_ess']:.0f} for '{table['ess'].idxmin()}'. Increase "
            "draws; the volatility of volatility is the slowest parameter."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    return res
