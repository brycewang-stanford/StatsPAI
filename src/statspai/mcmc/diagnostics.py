"""
Convergence diagnostics and summaries for MCMC output.

Every function here is a deterministic function of the draws, so each can
be checked digit for digit against R ``coda`` (Plummer, Best, Cowles and
Vines 2006) on the same chain. The conventions are coda's: the variance of
a chain mean comes from the spectral density at frequency zero of an
autoregression fitted by Yule-Walker with the order chosen by AIC.

* :func:`mcmc_summary`  -- mean, sd, naive and time-series standard error,
  quantiles, effective sample size.
* :func:`mcmc_ess`      -- effective sample size.
* :func:`geweke_diag`   -- Geweke (1992) equality of means of an early and
  a late window.
* :func:`raftery_diag`  -- Raftery and Lewis (1992) run length for a
  quantile.
* :func:`heidel_diag`   -- Heidelberger and Welch (1983) stationarity and
  half-width tests.
* :func:`gelman_rubin`  -- Gelman and Rubin (1992) potential scale
  reduction factor over several chains.
* :func:`hpd_interval`  -- shortest interval holding a given posterior mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import special, stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

ChainLike = Union[np.ndarray, pd.DataFrame, pd.Series, Sequence[float], Any]


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def _as_draws(chain: ChainLike) -> pd.DataFrame:
    """Coerce draws to a (draws x parameters) DataFrame.

    Accepts a 1-D or 2-D array, a Series, a DataFrame, or any object with
    a ``draws`` attribute holding one of those (the results of
    :func:`statspai.bayes_regress` and friends).
    """
    obj = getattr(chain, "draws", chain)
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, pd.Series):
        df = obj.to_frame(name=obj.name if obj.name is not None else "x")
    else:
        arr = np.asarray(obj, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise MethodIncompatibility(
                "MCMC draws must be 1-D or 2-D (draws x parameters); got an "
                f"array of shape {arr.shape}. For several chains pass a list "
                "of arrays to sp.gelman_rubin."
            )
        names = [f"var{j + 1}" for j in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=names)
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    if df.shape[0] < 2:
        raise DataInsufficient(
            "MCMC diagnostics need at least two draws; got " f"{df.shape[0]}."
        )
    if not np.isfinite(df.to_numpy()).all():
        bad = [str(c) for c in df.columns[~np.isfinite(df.to_numpy()).all(axis=0)]]
        raise MethodIncompatibility(
            "MCMC draws contain missing or infinite values in: " + ", ".join(bad) + "."
        )
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Spectral density at zero from an AR fit (coda's spectrum0.ar)
# --------------------------------------------------------------------------


def _yule_walker_aic(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """AR coefficients and innovation variance, order chosen by AIC.

    Yule-Walker fit with the conventions documented for R ``stats::ar``:
    autocovariances with divisor ``n``, maximum order
    ``min(n - 1, floor(10 log10 n))``, AIC ``n log v_k + 2k`` and the
    innovation variance rescaled by ``n / (n - (k + 1))``.
    """
    n = x.size
    xc = x - x.mean()
    order_max = int(min(n - 1, np.floor(10.0 * np.log10(n))))
    order_max = max(order_max, 0)
    # autocovariances r_0 .. r_order_max, divisor n
    r = np.empty(order_max + 1)
    for k in range(order_max + 1):
        r[k] = np.dot(xc[: n - k], xc[k:]) / n
    if r[0] <= 0.0:
        return np.zeros(0), 0.0
    phis: List[np.ndarray] = [np.zeros(0)]
    v = np.empty(order_max + 1)
    v[0] = r[0]
    prev = np.zeros(0)
    for k in range(1, order_max + 1):
        if v[k - 1] <= 0.0:
            # perfectly predictable series: stop the recursion
            v[k:] = v[k - 1]
            phis.extend([prev] * (order_max - k + 1))
            break
        acc = r[k] - (np.dot(prev, r[k - 1 : 0 : -1]) if k > 1 else 0.0)
        kk = acc / v[k - 1]
        cur = np.empty(k)
        if k > 1:
            cur[: k - 1] = prev - kk * prev[::-1]
        cur[k - 1] = kk
        v[k] = v[k - 1] * (1.0 - kk * kk)
        phis.append(cur)
        prev = cur
    with np.errstate(divide="ignore", invalid="ignore"):
        aic = n * np.log(v) + 2.0 * np.arange(order_max + 1)
    aic = np.where(np.isfinite(aic), aic, -np.inf)
    order = int(np.argmin(aic))
    var_pred = v[order] * n / (n - (order + 1))
    return phis[order], float(var_pred)


def _spectrum0_ar(x: np.ndarray) -> float:
    """Spectral density at frequency zero of ``x`` from an AR(AIC) fit.

    The long-run variance of the series: the variance of the mean of ``n``
    draws is this number over ``n``. A constant series has value 0.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float("nan")
    # A series that is an exact linear trend (in particular a constant)
    # carries no sampling noise.
    t = np.arange(1, n + 1, dtype=float)
    tc = t - t.mean()
    slope = np.dot(tc, x - x.mean()) / np.dot(tc, tc)
    resid = x - x.mean() - slope * tc
    scale = max(1.0, float(np.abs(x).max()))
    if resid.std(ddof=1) <= 1.5e-8 * scale:
        return 0.0
    ar, var_pred = _yule_walker_aic(x)
    denom = 1.0 - ar.sum()
    return float(var_pred / (denom * denom))


def _ess_1d(x: np.ndarray) -> float:
    s0 = _spectrum0_ar(x)
    if not np.isfinite(s0) or s0 == 0.0:
        return 0.0
    return float(x.size * x.var(ddof=1) / s0)


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------


@dataclass
class MCMCDiagnostic(ResultProtocolMixin):
    """A convergence diagnostic for each parameter of a chain.

    Attributes
    ----------
    name : str
        Which diagnostic this is.
    table : pd.DataFrame
        One row per parameter. The columns depend on the diagnostic.
    settings : dict
        The arguments the diagnostic was run with.
    passed : bool or None
        Whether every parameter passes at the diagnostic's own threshold,
        when the diagnostic defines one.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> draws = np.random.default_rng(0).normal(size=(2000, 2))
    >>> out = sp.geweke_diag(draws)
    >>> list(out.table.columns)
    ['z', 'p_value']
    >>> isinstance(out, sp.MCMCDiagnostic)
    True
    """

    name: str
    table: pd.DataFrame
    settings: Dict[str, Any] = field(default_factory=dict)
    passed: Optional[bool] = None

    def summary(self) -> str:
        lines = [f"{self.name}"]
        if self.settings:
            lines.append(
                "  " + ", ".join(f"{k} = {v}" for k, v in self.settings.items())
            )
        lines.append("")
        lines.append(self.table.to_string(float_format=lambda v: f"{v:.4g}"))
        if self.passed is not None:
            lines.append("")
            lines.append(
                "All parameters pass."
                if self.passed
                else "At least one parameter fails; see the table."
            )
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        return self.table.copy()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


# --------------------------------------------------------------------------
# Public functions
# --------------------------------------------------------------------------


def mcmc_ess(chain: ChainLike) -> pd.Series:
    """Effective sample size of each parameter of an MCMC chain.

    The number of independent draws that would estimate the posterior mean
    as precisely as the autocorrelated chain does: ``n var(x) / S(0)``,
    with ``S(0)`` the spectral density at zero of an AR model fitted to
    the chain (order by AIC). Same definition as R ``coda::effectiveSize``.

    Parameters
    ----------
    chain : array-like, DataFrame or fitted Bayesian result
        Draws, one row per iteration and one column per parameter.

    Returns
    -------
    pd.Series
        Effective sample size by parameter.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> x = np.random.default_rng(1).normal(size=5000)
    >>> bool(4000 < sp.mcmc_ess(x).iloc[0] < 6000)
    True

    References
    ----------
    plummer2006coda
    """
    df = _as_draws(chain)
    return pd.Series(
        [_ess_1d(df[c].to_numpy()) for c in df.columns], index=df.columns, name="ess"
    )


def hpd_interval(chain: ChainLike, prob: float = 0.95) -> pd.DataFrame:
    """Highest posterior density interval of each parameter.

    The shortest interval that holds ``prob`` of the draws (Chen and Shao
    1999). For a unimodal posterior it is the HPD region; for a symmetric
    one it coincides with the equal-tailed interval. Same rule as R
    ``coda::HPDinterval``.

    Parameters
    ----------
    chain : array-like, DataFrame or fitted Bayesian result
    prob : float, default 0.95
        Posterior mass inside the interval.

    Returns
    -------
    pd.DataFrame
        Columns ``lower`` and ``upper``, one row per parameter.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> x = np.random.default_rng(2).exponential(size=20000)
    >>> iv = sp.hpd_interval(x, prob=0.9)
    >>> bool(iv["lower"].iloc[0] < 0.01)
    True

    References
    ----------
    chen1999monte
    """
    if not 0.0 < prob < 1.0:
        raise MethodIncompatibility(f"prob must be in (0, 1); got {prob}.")
    df = _as_draws(chain)
    n = df.shape[0]
    gap = int(max(1, min(n - 1, round(n * prob))))
    rows = []
    for c in df.columns:
        v = np.sort(df[c].to_numpy())
        width = v[gap:] - v[: n - gap]
        i = int(np.argmin(width))
        rows.append((v[i], v[i + gap]))
    return pd.DataFrame(rows, index=df.columns, columns=["lower", "upper"])


def mcmc_summary(
    chain: ChainLike,
    quantiles: Sequence[float] = (0.025, 0.25, 0.5, 0.75, 0.975),
    hpd: Optional[float] = None,
) -> pd.DataFrame:
    """Posterior summary of an MCMC chain.

    Mean, standard deviation, the naive standard error of the mean
    (``sd / sqrt(n)``, valid only for independent draws), the time-series
    standard error (which accounts for autocorrelation), the effective
    sample size and quantiles. The first four columns and the quantiles are
    those of R ``summary(coda::mcmc(x))``.

    Parameters
    ----------
    chain : array-like, DataFrame or fitted Bayesian result
        Draws, one row per iteration and one column per parameter.
    quantiles : sequence of float
        Posterior quantiles to report.
    hpd : float, optional
        If given, also report the highest posterior density interval with
        this mass (columns ``hpd_lower``, ``hpd_upper``).

    Returns
    -------
    pd.DataFrame
        One row per parameter.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> draws = np.random.default_rng(3).normal(size=(4000, 2))
    >>> list(sp.mcmc_summary(draws).columns[:5])
    ['mean', 'sd', 'naive_se', 'ts_se', 'ess']

    References
    ----------
    plummer2006coda
    """
    df = _as_draws(chain)
    n = df.shape[0]
    out = pd.DataFrame(index=df.columns)
    out["mean"] = df.mean(axis=0)
    out["sd"] = df.std(axis=0, ddof=1)
    out["naive_se"] = out["sd"] / np.sqrt(n)
    s0 = np.array([_spectrum0_ar(df[c].to_numpy()) for c in df.columns])
    out["ts_se"] = np.sqrt(s0 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ess"] = np.where(s0 > 0, n * out["sd"].to_numpy() ** 2 / s0, 0.0)
    for q in quantiles:
        out[f"q{100 * q:g}"] = df.quantile(q, axis=0)
    if hpd is not None:
        iv = hpd_interval(df, prob=hpd)
        out["hpd_lower"] = iv["lower"]
        out["hpd_upper"] = iv["upper"]
    return out


def geweke_diag(
    chain: ChainLike, frac1: float = 0.1, frac2: float = 0.5
) -> MCMCDiagnostic:
    """Geweke (1992) convergence diagnostic.

    Compares the mean of the first ``frac1`` of the chain with the mean of
    the last ``frac2``. If the chain is stationary the two means are equal
    and the standardised difference is asymptotically standard normal.
    Standard errors use the spectral density at zero of each window, so
    autocorrelation is accounted for. Same statistic as R
    ``coda::geweke.diag``.

    Parameters
    ----------
    chain : array-like, DataFrame or fitted Bayesian result
    frac1 : float, default 0.1
        Fraction of the chain in the early window.
    frac2 : float, default 0.5
        Fraction of the chain in the late window.

    Returns
    -------
    MCMCDiagnostic
        ``table`` has the z-score and its two-sided p-value by parameter;
        ``passed`` is True when every ``|z| < 1.96``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> x = np.random.default_rng(4).normal(size=3000)
    >>> out = sp.geweke_diag(x)
    >>> bool(abs(out.table["z"].iloc[0]) < 4)
    True

    References
    ----------
    geweke1992evaluating
    """
    if not (0.0 < frac1 < 1.0 and 0.0 < frac2 < 1.0) or frac1 + frac2 > 1.0:
        raise MethodIncompatibility(
            "frac1 and frac2 must be in (0, 1) with frac1 + frac2 <= 1; got "
            f"frac1={frac1}, frac2={frac2}."
        )
    df = _as_draws(chain)
    n = df.shape[0]
    # iterations are 1..n; windows as documented for coda
    end_a = int(np.ceil(1 + frac1 * (n - 1)))
    start_b = int(np.floor(n - frac2 * (n - 1)))
    zs = []
    for c in df.columns:
        x = df[c].to_numpy()
        a = x[:end_a]
        b = x[start_b - 1 :]
        va = _spectrum0_ar(a) / a.size
        vb = _spectrum0_ar(b) / b.size
        with np.errstate(divide="ignore", invalid="ignore"):
            zs.append((a.mean() - b.mean()) / np.sqrt(va + vb))
    z = np.asarray(zs, dtype=float)
    table = pd.DataFrame(
        {"z": z, "p_value": 2.0 * stats.norm.sf(np.abs(z))}, index=df.columns
    )
    finite = np.isfinite(z)
    passed = bool(finite.all() and (np.abs(z) < 1.959963984540054).all())
    return MCMCDiagnostic(
        name="Geweke diagnostic (equality of early and late means)",
        table=table,
        settings={"frac1": frac1, "frac2": frac2},
        passed=passed,
    )


def raftery_diag(
    chain: ChainLike,
    q: float = 0.025,
    r: float = 0.005,
    s: float = 0.95,
    converge_eps: float = 0.001,
) -> MCMCDiagnostic:
    """Raftery and Lewis (1992) run-length diagnostic.

    How long a chain must be to estimate the posterior quantile ``q`` to
    within ``+/- r`` with probability ``s``. The chain is reduced to the
    indicator of being below its ``q`` quantile, thinned until that
    indicator behaves as a first-order Markov chain, and the two-state
    chain's transition probabilities give the burn-in ``M`` and the total
    run ``N`` (burn-in included). The dependence factor ``I = N / Nmin``
    compares the total with the length an independent sample would need;
    values above 5 indicate strong autocorrelation. Same output as R
    ``coda::raftery.diag``, which prints ``I`` to three significant
    digits.

    Parameters
    ----------
    chain : array-like, DataFrame or fitted Bayesian result
    q : float, default 0.025
        Quantile to be estimated.
    r : float, default 0.005
        Desired margin of error of the estimated cumulative probability.
    s : float, default 0.95
        Probability of attaining that margin.
    converge_eps : float, default 0.001
        Precision required for the estimated time to convergence.

    Returns
    -------
    MCMCDiagnostic
        ``table`` columns: ``thin`` (k), ``burnin`` (M), ``total`` (N),
        ``n_min`` (Nmin) and ``dependence_factor`` (I). ``passed`` is True
        when every dependence factor is at most 5.

    Raises
    ------
    DataInsufficient
        If the chain is shorter than ``Nmin``, the length needed even with
        independent draws.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> x = np.random.default_rng(5).normal(size=5000)
    >>> out = sp.raftery_diag(x)
    >>> int(out.table["n_min"].iloc[0])
    3746

    References
    ----------
    raftery1992how
    """
    for nm, val in (("q", q), ("r", r), ("s", s)):
        if not 0.0 < val < 1.0:
            raise MethodIncompatibility(f"{nm} must be in (0, 1); got {val}.")
    df = _as_draws(chain)
    n = df.shape[0]
    phi = stats.norm.ppf(0.5 * (1.0 + s))
    n_min = int(np.ceil(q * (1.0 - q) * phi**2 / r**2))
    rows: List[Tuple[Any, ...]] = []
    if n_min > n:
        raise DataInsufficient(
            f"The chain has {n} draws but at least {n_min} are needed to "
            f"estimate the {q} quantile to within {r} with probability {s} "
            "even if the draws were independent. Run a longer chain or "
            "relax q / r / s."
        )
    for c in df.columns:
        x = df[c].to_numpy()
        below = x <= np.quantile(x, q)
        k = 0
        while True:
            k += 1
            z = below[::k].astype(np.int64)
            m = z.size
            if m < 3:
                break
            # 2 x 2 x 2 table of consecutive triples
            idx = 4 * z[:-2] + 2 * z[1:-1] + z[2:]
            tab = np.bincount(idx, minlength=8).astype(float).reshape(2, 2, 2)
            g2 = 0.0
            for i1 in range(2):
                for i2 in range(2):
                    for i3 in range(2):
                        cnt = tab[i1, i2, i3]
                        if cnt != 0.0:
                            fitted = (
                                tab[i1, i2, :].sum()
                                * tab[:, i2, i3].sum()
                                / tab[:, i2, :].sum()
                            )
                            g2 += 2.0 * cnt * np.log(cnt / fitted)
            bic = g2 - 2.0 * np.log(m - 2)
            if bic < 0:
                break
        z = below[::k].astype(np.int64)
        tr = np.bincount(2 * z[:-1] + z[1:], minlength=4).astype(float).reshape(2, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = tr[0, 1] / tr[0, :].sum()
            beta = tr[1, 0] / tr[1, :].sum()
            burn = np.log(converge_eps * (alpha + beta) / max(alpha, beta)) / np.log(
                abs(1.0 - alpha - beta)
            )
            keep = (
                (2.0 - alpha - beta)
                * alpha
                * beta
                * phi**2
                / ((alpha + beta) ** 3 * r**2)
            )
        if np.isfinite(burn) and np.isfinite(keep):
            n_burn = int(np.ceil(burn) * k)
            n_keep = int(np.ceil(keep) * k)
            rows.append((k, n_burn, n_burn + n_keep, n_min, (n_burn + n_keep) / n_min))
        else:
            rows.append((k, np.nan, np.nan, n_min, np.nan))
    table = pd.DataFrame(
        rows,
        index=df.columns,
        columns=["thin", "burnin", "total", "n_min", "dependence_factor"],
    )
    dep = table["dependence_factor"].to_numpy(dtype=float)
    passed = bool(np.isfinite(dep).all() and (dep <= 5.0).all())
    return MCMCDiagnostic(
        name="Raftery-Lewis run-length diagnostic",
        table=table,
        settings={"q": q, "r": r, "s": s, "converge_eps": converge_eps},
        passed=passed,
    )


def _cramer_von_mises_cdf(q: float, n_terms: int = 4) -> float:
    """Limiting distribution function of the Cramer-von Mises statistic.

    Anderson and Darling (1952) series,
    ``P(W <= q) = sum_k G(k+1/2) sqrt(4k+1) / (G(k+1) pi^{3/2} sqrt(q))
    exp(-u_k) K_{1/4}(u_k)`` with ``u_k = (4k+1)^2 / (16 q)``.
    """
    if not np.isfinite(q):
        return float("nan")
    if q <= 0.0:
        return 0.0
    total = 0.0
    for k in range(n_terms):
        u = (4 * k + 1) ** 2 / (16.0 * q)
        if u > 11.512925464970229:  # -log(1e-5): the term is negligible
            continue
        zc = (
            special.gamma(k + 0.5)
            * np.sqrt(4 * k + 1)
            / (special.gamma(k + 1) * np.pi**1.5 * np.sqrt(q))
        )
        total += zc * np.exp(-u) * special.kv(0.25, u)
    return float(total)


def heidel_diag(
    chain: ChainLike, eps: float = 0.1, pvalue: float = 0.05
) -> MCMCDiagnostic:
    """Heidelberger and Welch (1983) stationarity and half-width tests.

    Stationarity: a Cramer-von Mises test that the chain is a draw from a
    stationary process, applied to the whole chain and then, if it
    rejects, after discarding the first 10%, 20%, ... up to 50%. The
    half-width test then asks whether the part of the chain retained
    estimates the mean precisely enough: it passes when the half-width of
    the 95% interval for the mean, relative to the mean, is below ``eps``.
    Same procedure as R ``coda::heidel.diag``.

    Parameters
    ----------
    chain : array-like, DataFrame or fitted Bayesian result
    eps : float, default 0.1
        Target ratio of half-width to mean.
    pvalue : float, default 0.05
        Significance level of the stationarity test.

    Returns
    -------
    MCMCDiagnostic
        ``table`` columns: ``stationary`` (bool), ``start`` (first
        iteration retained, 1-based), ``p_value``, ``halfwidth_passed``,
        ``mean`` and ``halfwidth``. For a chain that fails stationarity the
        last four are missing.

    Notes
    -----
    The half-width test is relative to the mean, so it is uninformative for
    a parameter whose posterior mean is near zero.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> x = 5 + np.random.default_rng(6).normal(size=4000)
    >>> out = sp.heidel_diag(x)
    >>> bool(out.table["stationary"].iloc[0])
    True

    References
    ----------
    heidelberger1983simulation
    """
    df = _as_draws(chain)
    n_iter = df.shape[0]
    # candidate first iterations (1-based): 1, 1 + n/10, ... up to n/2
    step = n_iter / 10.0
    starts = []
    cur = 1.0
    while cur <= n_iter / 2.0 + 1e-9:
        starts.append(cur)
        cur += step
    rows: List[Tuple[Any, ...]] = []
    for c in df.columns:
        x = df[c].to_numpy()
        half_from = int(np.ceil(n_iter / 2.0))  # first iteration >= n/2
        s0 = _spectrum0_ar(x[half_from - 1 :])
        converged = False
        y = x
        first = 1
        pv = float("nan")
        for st in starts:
            first = int(np.ceil(st - 1e-9))
            y = x[first - 1 :]
            m = y.size
            ybar = y.mean()
            b = np.cumsum(y) - ybar * np.arange(1, m + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                stat = float(np.sum(b * b / (m * s0)) / m)
            if np.isfinite(stat):
                cdf = _cramer_von_mises_cdf(stat)
                pv = 1.0 - cdf
                if cdf < 1.0 - pvalue:
                    converged = True
                    break
        if converged:
            m = y.size
            ybar = float(y.mean())
            hw = 1.96 * np.sqrt(_spectrum0_ar(y) / m)
            with np.errstate(divide="ignore", invalid="ignore"):
                ok = bool(np.isfinite(hw) and abs(hw / ybar) <= eps)
            rows.append((True, first, pv, ok, ybar, hw))
        else:
            rows.append((False, np.nan, pv, np.nan, np.nan, np.nan))
    table = pd.DataFrame(
        rows,
        index=df.columns,
        columns=[
            "stationary",
            "start",
            "p_value",
            "halfwidth_passed",
            "mean",
            "halfwidth",
        ],
    )
    passed = bool(table["stationary"].astype(bool).all())
    return MCMCDiagnostic(
        name="Heidelberger-Welch stationarity and half-width tests",
        table=table,
        settings={"eps": eps, "pvalue": pvalue},
        passed=passed,
    )


def _as_chain_list(chains: Any) -> Tuple[List[np.ndarray], List[str]]:
    if isinstance(chains, np.ndarray) and chains.ndim == 3:
        arrs = [chains[i] for i in range(chains.shape[0])]
        names = [f"var{j + 1}" for j in range(chains.shape[2])]
        return [np.asarray(a, dtype=float) for a in arrs], names
    if isinstance(chains, (list, tuple)):
        dfs = [_as_draws(c) for c in chains]
        names = [str(c) for c in dfs[0].columns]
        return [d.to_numpy() for d in dfs], names
    raise MethodIncompatibility(
        "gelman_rubin needs several chains: a list of (draws x parameters) "
        "arrays / DataFrames / fitted results, or a 3-D array "
        "(chains x draws x parameters)."
    )


def gelman_rubin(
    chains: Any,
    confidence: float = 0.95,
    split: bool = False,
    autoburnin: bool = False,
) -> MCMCDiagnostic:
    """Gelman and Rubin (1992) potential scale reduction factor.

    Compares the variance between several chains started from dispersed
    values with the variance within them. A factor near 1 says the chains
    have mixed; the conventional alarm is above 1.1 (1.01 for the split
    version). Point estimate and upper confidence limit as in R
    ``coda::gelman.diag`` (with the Brooks and Gelman 1998 degrees of
    freedom correction), plus the multivariate factor.

    Parameters
    ----------
    chains : list of chains, or array (chains x draws x parameters)
        At least two chains of equal length.
    confidence : float, default 0.95
        Coverage of the upper limit.
    split : bool, default False
        Split every chain in two before computing the factor, which also
        detects a single chain that drifts. Use this when only one chain is
        available.
    autoburnin : bool, default False
        Discard the first half of every chain first. ``coda::gelman.diag``
        does this by default; here the draws are assumed to be post
        burn-in already.

    Returns
    -------
    MCMCDiagnostic
        ``table`` columns ``psrf`` and ``upper``; ``settings['mpsrf']`` is
        the multivariate factor (None for one parameter).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(7)
    >>> out = sp.gelman_rubin([rng.normal(size=(1000, 2)) for _ in range(3)])
    >>> bool((out.table["psrf"] < 1.05).all())
    True

    References
    ----------
    gelman1992inference, brooks1998general
    """
    if isinstance(chains, (list, tuple)) or (
        isinstance(chains, np.ndarray) and chains.ndim == 3
    ):
        arrs, names = _as_chain_list(chains)
    else:
        d = _as_draws(chains)
        arrs, names = [d.to_numpy()], [str(c) for c in d.columns]
        if not split:
            raise MethodIncompatibility(
                "gelman_rubin needs at least two chains. With a single chain "
                "pass split=True to compare its two halves."
            )
    if autoburnin:
        arrs = [a[a.shape[0] // 2 :] for a in arrs]
    if split:
        halves = []
        for a in arrs:
            h = a.shape[0] // 2
            halves.extend([a[:h], a[a.shape[0] - h :]])
        arrs = halves
    lens = {a.shape[0] for a in arrs}
    if len(lens) != 1:
        raise MethodIncompatibility(
            "All chains must have the same length; got lengths " f"{sorted(lens)}."
        )
    m = len(arrs)
    n = arrs[0].shape[0]
    if m < 2:
        raise MethodIncompatibility("gelman_rubin needs at least two chains.")
    if n < 2:
        raise DataInsufficient("Each chain needs at least two draws.")
    x = np.stack(arrs)  # m x n x p
    p = x.shape[2]
    chain_means = x.mean(axis=1)  # m x p
    grand = chain_means.mean(axis=0)
    # within and between covariance matrices
    w_mat = np.zeros((p, p))
    for i in range(m):
        w_mat += np.cov(x[i], rowvar=False, ddof=1).reshape(p, p)
    w_mat /= m
    b_mat = n * np.cov(chain_means, rowvar=False, ddof=1).reshape(p, p)
    w = np.diag(w_mat)
    b = np.diag(b_mat)
    s2 = np.stack([x[i].var(axis=0, ddof=1) for i in range(m)])  # m x p
    var_w = s2.var(axis=0, ddof=1) / m
    var_b = 2.0 * b**2 / (m - 1)
    cov_wb = (n / m) * np.array(
        [
            np.cov(s2[:, j], chain_means[:, j] ** 2, ddof=1)[0, 1]
            - 2.0 * grand[j] * np.cov(s2[:, j], chain_means[:, j], ddof=1)[0, 1]
            for j in range(p)
        ]
    )
    v = (n - 1) * w / n + (1.0 + 1.0 / m) * b / n
    var_v = (
        (n - 1) ** 2 * var_w
        + (1.0 + 1.0 / m) ** 2 * var_b
        + 2.0 * (n - 1) * (1.0 + 1.0 / m) * cov_wb
    ) / n**2
    with np.errstate(divide="ignore", invalid="ignore"):
        df_v = 2.0 * v**2 / var_v
        df_adj = (df_v + 3.0) / (df_v + 1.0)
        b_df = m - 1
        w_df = 2.0 * w**2 / var_w
        r2_fixed = (n - 1) / n
        r2_random = (1.0 + 1.0 / m) * (1.0 / n) * (b / w)
        r2_est = r2_fixed + r2_random
        r2_up = r2_fixed + stats.f.ppf((1.0 + confidence) / 2.0, b_df, w_df) * r2_random
        psrf = np.sqrt(df_adj * r2_est)
        upper = np.sqrt(df_adj * r2_up)
    mpsrf: Optional[float] = None
    if p > 1:
        try:
            eig = np.linalg.eigvals(np.linalg.solve(w_mat, b_mat / n))
            lam = float(np.max(eig.real))
            mpsrf = float(np.sqrt((1.0 - 1.0 / n) + (1.0 + 1.0 / m) * lam))
        except np.linalg.LinAlgError:
            mpsrf = None
    table = pd.DataFrame({"psrf": psrf, "upper": upper}, index=names)
    threshold = 1.01 if split else 1.1
    passed = bool(np.isfinite(psrf).all() and (psrf < threshold).all())
    return MCMCDiagnostic(
        name="Gelman-Rubin potential scale reduction factor",
        table=table,
        settings={
            "chains": m,
            "draws_per_chain": n,
            "split": split,
            "confidence": confidence,
            "mpsrf": mpsrf,
            "threshold": threshold,
        },
        passed=passed,
    )
