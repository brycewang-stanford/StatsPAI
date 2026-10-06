"""
Bayesian ARIMA: ``sp.bayes_arima``.

The exact Gaussian likelihood of a stationary and invertible ARMA(p, q)
process is evaluated by the Kalman filter and the posterior is explored by
random-walk Metropolis. The AR and MA coefficients are parameterised by
their partial autocorrelations, which turns the stationarity and
invertibility restrictions into the open cube (-1, 1)^(p+q). The prior is
uniform over the admissible region of the coefficients themselves, and it
is properly normalised, so marginal likelihoods can be compared across
orders.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import special

from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._core import (
    check_mcmc_args,
    find_mode,
    log_invgamma,
    random_walk_metropolis,
    spawn_rngs,
)
from ._results import posterior_table
from .regress import BayesRegressResult

_KERNEL: Dict[str, Any] = {}


def _arma_filter_py(
    w: np.ndarray, phi: np.ndarray, theta: np.ndarray
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Kalman filter of a zero-mean ARMA process with unit innovation
    variance. Returns ``sum(log F_t)``, ``sum(v_t^2 / F_t)`` and the
    predicted state and its covariance one step past the sample."""
    p = phi.shape[0]
    q = theta.shape[0]
    r = max(p, q + 1)
    tmat = np.zeros((r, r))
    for i in range(p):
        tmat[i, 0] = phi[i]
    for i in range(r - 1):
        tmat[i, i + 1] = 1.0
    rvec = np.zeros(r)
    rvec[0] = 1.0
    for i in range(q):
        rvec[i + 1] = theta[i]
    rr = np.outer(rvec, rvec)
    # stationary covariance: vec(P) = (I - T kron T)^{-1} vec(R R')
    kron = np.kron(tmat, tmat)
    pmat: np.ndarray = np.linalg.solve(np.eye(r * r) - kron, rr.reshape(r * r)).reshape(
        r, r
    )
    pmat = 0.5 * (pmat + pmat.T)
    a = np.zeros(r)
    sum_log_f = 0.0
    sum_sq = 0.0
    for t in range(w.shape[0]):
        f = pmat[0, 0]
        v = w[t] - a[0]
        sum_log_f += np.log(f)
        sum_sq += v * v / f
        k = pmat[:, 0] / f
        a = a + k * v
        pmat = pmat - np.outer(k, pmat[0, :])
        a = tmat @ a
        pmat = tmat @ pmat @ tmat.T + rr
        pmat = 0.5 * (pmat + pmat.T)
    return sum_log_f, sum_sq, a, pmat


def _arma_filter() -> Any:
    if "f" not in _KERNEL:
        from numba import njit  # type: ignore[import-untyped]

        _KERNEL["f"] = njit(cache=True)(_arma_filter_py)
    return _KERNEL["f"]


def pacf_to_coefs(r: np.ndarray) -> np.ndarray:
    """Coefficients of the stationary AR polynomial with partial
    autocorrelations ``r`` (Durbin-Levinson recursion)."""
    a = np.zeros(0)
    for rk in np.asarray(r, dtype=float):
        a = np.concatenate([a - rk * a[::-1], [rk]])
    return a


def coefs_to_pacf(a: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pacf_to_coefs`. Values outside (-1, 1) mean the
    polynomial is not stationary."""
    a = np.array(a, dtype=float)
    out = np.zeros(a.size)
    for k in range(a.size - 1, -1, -1):
        rk = a[k]
        out[k] = rk
        if abs(rk) >= 1.0:
            out[:k] = np.nan
            break
        a = (a[:k] + rk * a[:k][::-1]) / (1.0 - rk * rk)
    return out


def _log_pacf_prior(r: np.ndarray) -> float:
    """Log density of partial autocorrelations under a uniform prior on
    the admissible region of the coefficients."""
    val = 0.0
    for k, rk in enumerate(r, start=1):
        a = (k - 1) // 2 + 1.0
        b = k // 2 + 1.0
        val += (
            (a - 1.0) * np.log1p(rk)
            + (b - 1.0) * np.log1p(-rk)
            - (a + b - 1.0) * np.log(2.0)
            - special.betaln(a, b)
        )
    return float(val)


class _ArmaModel:
    """Posterior kernel of an ARMA(p, q) model on the unconstrained scale
    ``(const, atanh(pacf_ar), atanh(pacf_ma), log sigma2)``."""

    def __init__(
        self,
        w: np.ndarray,
        p: int,
        q: int,
        constant: bool,
        mean_prior: Tuple[float, float],
        sigma2_prior: Tuple[float, float],
    ) -> None:
        self.w = np.ascontiguousarray(w, dtype=float)
        self.p, self.q, self.constant = p, q, constant
        self.m0, self.v0 = mean_prior
        self.a0, self.d0 = sigma2_prior
        self.names: List[str] = (
            (["const"] if constant else [])
            + [f"ar.L{i + 1}" for i in range(p)]
            + [f"ma.L{i + 1}" for i in range(q)]
            + ["sigma2"]
        )
        self.k = len(self.names)

    # -- transforms --------------------------------------------------------
    def split(self, u: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
        c = int(self.constant)
        mu = float(u[0]) if c else 0.0
        return mu, u[c : c + self.p], u[c + self.p : c + self.p + self.q], float(u[-1])

    def from_u(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_2d(u)
        out = np.empty_like(u)
        for i, row in enumerate(u):
            mu, ua, um, ls = self.split(row)
            parts: List[Any] = []
            if self.constant:
                parts.append([mu])
            parts.append(pacf_to_coefs(np.tanh(ua)))
            # an invertible MA polynomial 1 + theta(z) is a stationary AR
            # polynomial in -theta
            parts.append(-pacf_to_coefs(np.tanh(um)))
            parts.append([np.exp(ls)])
            out[i] = np.concatenate(parts)
        return out

    def to_u(self, draws: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(draws)
        out = np.empty_like(d)
        c = int(self.constant)
        for i, row in enumerate(d):
            parts: List[Any] = []
            if c:
                parts.append([row[0]])
            parts.append(np.arctanh(coefs_to_pacf(row[c : c + self.p])))
            parts.append(
                np.arctanh(coefs_to_pacf(-row[c + self.p : c + self.p + self.q]))
            )
            parts.append([np.log(row[-1])])
            out[i] = np.concatenate(parts)
        return out

    # -- densities ---------------------------------------------------------
    def log_likelihood(
        self, mu: float, phi: np.ndarray, theta: np.ndarray, sigma2: float
    ) -> float:
        slf, ssq, _, _ = _arma_filter()(self.w - mu, phi, theta)
        n = self.w.size
        return float(
            -0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * slf - 0.5 * ssq / sigma2
        )

    def log_kernel(self, u: np.ndarray) -> float:
        mu, ua, um, ls = self.split(np.asarray(u, dtype=float))
        if abs(ls) > 500 or np.any(np.abs(ua) > 18) or np.any(np.abs(um) > 18):
            return -np.inf
        ra, rm = np.tanh(ua), np.tanh(um)
        s2 = float(np.exp(ls))
        val = self.log_likelihood(mu, pacf_to_coefs(ra), -pacf_to_coefs(rm), s2)
        # priors, with the Jacobians of tanh and exp
        val += _log_pacf_prior(ra) + float(np.log1p(-ra * ra).sum())
        val += _log_pacf_prior(rm) + float(np.log1p(-rm * rm).sum())
        val += log_invgamma(s2, self.a0 / 2.0, self.d0 / 2.0) + ls
        if self.constant:
            val += (
                -0.5 * np.log(2.0 * np.pi * self.v0)
                - 0.5 * (mu - self.m0) ** 2 / self.v0
            )
        return float(val) if np.isfinite(val) else -np.inf

    def mode(self) -> Tuple[np.ndarray, np.ndarray]:
        start = np.zeros(self.k)
        if self.constant:
            start[0] = self.w.mean()
        start[-1] = np.log(self.w.var())

        def neg(u: np.ndarray) -> float:
            v = self.log_kernel(u)
            return -v if np.isfinite(v) else 1e300

        return find_mode(neg, start, what="ARMA posterior")


def _integrate(w_path: np.ndarray, tails: List[float]) -> np.ndarray:
    """Undo the differencing: ``tails[j]`` is the last value of the series
    differenced ``j`` times, for ``j = d - 1, ..., 0``."""
    out = w_path
    for last in tails:
        out = last + np.cumsum(out, axis=-1)
    return out


def bayes_arima(
    y: Union[str, np.ndarray, pd.Series],
    data: Optional[pd.DataFrame] = None,
    order: Tuple[int, int, int] = (1, 0, 0),
    constant: Optional[bool] = None,
    mean_prior: Tuple[float, float] = (0.0, 1e6),
    sigma2_prior: Tuple[float, float] = (0.001, 0.001),
    horizon: int = 0,
    draws: int = 10000,
    burnin: int = 2000,
    thin: int = 1,
    chains: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
    tune: Optional[float] = None,
) -> BayesRegressResult:
    """Bayesian ARIMA(p, d, q) with the exact Gaussian likelihood.

    The series is differenced ``d`` times and the result is modelled as a
    stationary and invertible ARMA(p, q) process around a constant. The
    posterior is sampled by random-walk Metropolis.

    Parameters
    ----------
    y : array, Series or str
        The series, or a column name of ``data``.
    data : DataFrame, optional
    order : (p, d, q), default (1, 0, 0)
    constant : bool, optional
        Include a constant: the mean of the series when ``d = 0``, the
        drift when ``d > 0``. Default ``True`` when ``d = 0`` and ``False``
        otherwise.
    mean_prior : (mean, variance), default (0, 1e6)
        Normal prior of the constant.
    sigma2_prior : (alpha0, delta0), default (0.001, 0.001)
        ``sigma2 ~ IG(alpha0 / 2, delta0 / 2)``.
    horizon : int, default 0
        Number of periods to forecast. The posterior predictive mean and
        interval go to ``model_info['forecast']``.
    draws, burnin, thin, chains, seed, level
        As in :func:`statspai.bayes_regress`.
    tune : float, optional
        Scale of the proposal relative to the curvature at the mode.
        Default ``2.38 / sqrt(k)``.

    Returns
    -------
    BayesRegressResult
        ``model='arima'`` with parameters ``const``, ``ar.L1`` ...,
        ``ma.L1`` ... and ``sigma2``. The MA polynomial is
        ``1 + ma.L1 L + ...``, the convention of :func:`statspai.arima`.
        ``log_marginal_likelihood()`` is available for comparing orders
        with the same ``d``.

    Notes
    -----
    The prior of the AR coefficients is uniform over the stationary
    region, and that of the MA coefficients is uniform over the
    invertible region. Both regions are bounded, so the prior is proper
    without any tuning constant.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(300)
    >>> for t in range(1, 300):
    ...     y[t] = 0.6 * y[t - 1] + rng.normal()
    >>> fit = sp.bayes_arima(y, order=(1, 0, 0), draws=1000, burnin=300, seed=1)
    >>> list(fit.params.index)
    ['const', 'ar.L1', 'sigma2']
    >>> bool(0.4 < fit.params["ar.L1"] < 0.8)
    True

    References
    ----------
    ramirezhassan2026introduction
    """
    check_mcmc_args(draws, burnin, thin, chains)
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    try:
        p, d, q = (int(v) for v in order)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"order must be three integers (p, d, q); got {order!r}."
        ) from exc
    if min(p, d, q) < 0 or tuple(order) != (p, d, q):
        raise MethodIncompatibility(
            f"order must be three non-negative integers; got {order!r}."
        )
    if horizon < 0 or int(horizon) != horizon:
        raise MethodIncompatibility(
            f"horizon must be a non-negative integer; got {horizon}."
        )
    if isinstance(y, str):
        if data is None or y not in data.columns:
            raise MethodIncompatibility(f"{y!r} is not a column of data.")
        raw = data[y].to_numpy(dtype=float)
        name = y
    else:
        raw = np.asarray(y, dtype=float).reshape(-1)
        name = getattr(y, "name", None) or "y"
    if not np.isfinite(raw).all():
        raise MethodIncompatibility(
            "The series has missing or infinite values; remove them first."
        )
    if constant is None:
        constant = d == 0
    m0, v0 = (float(v) for v in mean_prior)
    a0, d0 = (float(v) for v in sigma2_prior)
    if v0 <= 0 or a0 <= 0 or d0 <= 0:
        raise MethodIncompatibility("The prior scales must be positive.")
    w = raw.copy()
    tails: List[float] = []
    for _ in range(d):
        tails.append(float(w[-1]))
        w = np.diff(w)
    tails = tails[::-1]
    n = w.size
    k = int(constant) + p + q + 1
    if n < max(10, 3 * k):
        raise DataInsufficient(
            f"{n} observations after differencing are too few for {k} parameters."
        )
    if np.allclose(w, w[0]):
        raise MethodIncompatibility("The series is constant after differencing.")

    mdl = _ArmaModel(w, p, q, bool(constant), (m0, v0), (a0, d0))
    mode, cov = mdl.mode()
    scale = 2.38 / np.sqrt(k) if tune is None else float(tune)
    if scale <= 0:
        raise MethodIncompatibility("tune must be positive.")
    n_iter = burnin + draws * thin
    rngs = spawn_rngs(seed, chains)
    pieces, accepts = [], []
    for c in range(chains):
        start = mode.copy()
        if chains > 1:
            start = start + np.linalg.cholesky(cov) @ rngs[c].standard_normal(k)
        out, acc = random_walk_metropolis(
            rngs[c], mdl.log_kernel, start, scale * scale * cov, n_iter
        )
        pieces.append(mdl.from_u(out[burnin:n_iter:thin]))
        accepts.append(acc)
    d_df = pd.DataFrame(np.vstack(pieces), columns=mdl.names)
    chain_idx = np.repeat(np.arange(chains), draws)
    table, diag = posterior_table(d_df, chain_idx, chains, level)
    info: Dict[str, Any] = {
        "order": (p, d, q),
        "constant": bool(constant),
        "series": str(name),
        "mode": pd.Series(mdl.from_u(mode)[0], index=mdl.names),
    }

    if horizon:
        rng = rngs[0]
        arr = d_df.to_numpy()
        take = np.linspace(0, len(arr) - 1, min(len(arr), 2000)).astype(int)
        c0 = int(constant)
        r = max(p, q + 1)
        sims = np.empty((take.size, int(horizon)))
        for j, row in enumerate(arr[take]):
            mu = row[0] if c0 else 0.0
            phi, theta, s2 = row[c0 : c0 + p], row[c0 + p : c0 + p + q], row[-1]
            _, _, a, pm = _arma_filter()(w - mu, phi, theta)
            # the predicted state has covariance sigma2 * pm
            vals, vecs = np.linalg.eigh(pm)
            root = vecs * np.sqrt(np.clip(vals, 0.0, None))
            state = a + np.sqrt(s2) * (root @ rng.standard_normal(r))
            rvec = np.concatenate([[1.0], theta, np.zeros(r - 1 - q)])
            for hstep in range(int(horizon)):
                sims[j, hstep] = mu + state[0]
                nxt = np.zeros(r)
                nxt[: r - 1] = state[1:]
                nxt[:p] += phi * state[0]
                state = nxt + rvec * np.sqrt(s2) * rng.standard_normal()
        paths = _integrate(sims, tails)
        lo = (1.0 - level) / 2.0
        info["forecast"] = pd.DataFrame(
            {
                "mean": paths.mean(axis=0),
                "sd": paths.std(axis=0, ddof=1),
                "lower": np.quantile(paths, lo, axis=0),
                "upper": np.quantile(paths, 1.0 - lo, axis=0),
            },
            index=pd.RangeIndex(1, int(horizon) + 1, name="step"),
        )

    res = BayesRegressResult(
        model="arima",
        formula=f"{name} ~ ARIMA({p}, {d}, {q})",
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
        sampler="random-walk Metropolis on the partial autocorrelations",
        acceptance_rate=float(np.mean(accepts)),
        prior={
            "const": f"N({m0:g}, {v0:g})" if constant else "none",
            "ar": "uniform over the stationary region",
            "ma": "uniform over the invertible region",
            "sigma2": f"IG({a0 / 2:g}, {d0 / 2:g})",
        },
        level=level,
        model_info=info,
        diagnostics_info=diag,
        _model=mdl,
    )
    acc = res.acceptance_rate or 0.0
    if acc < 0.05 or diag["min_ess"] < 100:
        text = (
            f"The chain mixes slowly (acceptance {acc:.2f}, smallest "
            f"effective sample size {diag['min_ess']:.0f}). Increase draws "
            "or lower tune. A posterior piled up against the unit circle "
            "suggests differencing the series."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    return res
