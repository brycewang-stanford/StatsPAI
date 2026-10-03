"""
General-purpose bootstrap inference framework.

Works with any StatsPAI estimator — pass a fitting function and get
bootstrap confidence intervals, standard errors, and p-values.

Supports:
- **Nonparametric bootstrap** (resample rows)
- **Pairs cluster bootstrap** (resample clusters)
- **Block bootstrap** (for time-series / panel data)
- **Percentile, BCa, and normal** confidence intervals

References
----------
Efron, B. and Tibshirani, R.J. (1993).
*An Introduction to the Bootstrap*. Chapman & Hall.

Cameron, A.C., Gelbach, J.B. and Miller, D.L. (2008).
"Bootstrap-Based Improvements for Inference with Clustered Errors."
*Review of Economics and Statistics*, 90(3), 414-427. [@cameron2008bootstrap]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

if TYPE_CHECKING:  # pragma: no cover
    from ..core.results import EconometricResults


@dataclass
class BootstrapResult(ResultProtocolMixin):
    """Container for bootstrap inference results.

    Returned by :func:`bootstrap`. Holds the point estimate, the
    bootstrap standard error, the confidence interval, the two-sided
    p-value, and the full bootstrap distribution.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"y": rng.normal(1.0, 1.0, 200),
    ...                    "x": rng.normal(0.0, 1.0, 200)})
    >>> res = sp.bootstrap(df, lambda d: d["y"].mean() - d["x"].mean(),
    ...                    n_boot=200, seed=0)
    >>> type(res).__name__
    'BootstrapResult'
    >>> res.n_boot
    200
    >>> res.ci_method
    'percentile'
    >>> bool(res.ci_lower < res.estimate < res.ci_upper)
    True
    """

    _citation_keys = ("cameron2008bootstrap",)

    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    alpha: float
    n_boot: int
    boot_distribution: np.ndarray
    ci_method: str

    def summary(self) -> str:
        lines = [
            "Bootstrap Inference",
            f"  Estimate:   {self.estimate:.6f}",
            f"  Std. Error: {self.se:.6f}",
            f"  CI ({1 - self.alpha:.0%}):    [{self.ci_lower:.6f}, "
            f"{self.ci_upper:.6f}]  ({self.ci_method})",
            f"  p-value:    {self.pvalue:.4f}",
            f"  Replications: {self.n_boot}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def bootstrap(
    data: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    n_boot: int = 1000,
    cluster: Optional[str] = None,
    block: Optional[str] = None,
    ci_method: str = "percentile",
    alpha: float = 0.05,
    seed: Optional[int] = None,
    null_value: float = 0.0,
    idcluster: Optional[str] = None,
) -> Union[BootstrapResult, "EconometricResults"]:
    """
    General-purpose bootstrap inference.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    statistic : callable
        A function of a (possibly resampled) DataFrame returning the
        statistic: a float, a vector (array / Series), or a fitted result
        with ``.params`` -- ``lambda d: sp.regress("y ~ x", data=d)`` is
        Stata's ``bootstrap _b: regress y x``. A vector statistic returns
        an ``EconometricResults`` with bootstrap standard errors, the
        bootstrap covariance (``vcov()``), z statistics and normal-based
        intervals as Stata reports them; percentile intervals, the bias and
        the replicates are in ``model_info``.
    n_boot : int, default 1000
        Number of bootstrap replications.
    cluster : str, optional
        Column name for cluster bootstrap (resample clusters, not rows).
    block : str, optional
        Column name for block bootstrap (resample blocks, preserve
        within-block ordering). Useful for panel/time-series data.
    ci_method : str, default 'percentile'
        CI method: ``'percentile'``, ``'bca'``, or ``'normal'``.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed.
    null_value : float, default 0.0
        Value under H0 for p-value computation.
    idcluster : str, optional
        With ``cluster=``, the name of a new column holding a distinct id
        for every resampled cluster (Stata ``idcluster()``): a cluster drawn
        twice becomes two groups, which a fixed-effects statistic must see
        to stay the cluster bootstrap. Absorb or cluster on this column
        inside ``statistic``.

    Returns
    -------
    BootstrapResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"y": rng.normal(1.0, 1.0, 200),
    ...                    "x": rng.normal(0.0, 1.0, 200)})
    >>> boot = sp.bootstrap(df, lambda d: d["y"].mean() - d["x"].mean(),
    ...                     n_boot=500, seed=0)
    >>> type(boot).__name__
    'BootstrapResult'
    >>> boot.n_boot
    500
    >>> bool(boot.se > 0)
    True
    """
    rng = np.random.RandomState(seed)
    if idcluster is not None and cluster is None:
        raise MethodIncompatibility("idcluster= requires cluster=.")

    # Original estimate
    base = data
    if idcluster is not None:
        base = data.assign(**{idcluster: pd.factorize(data[cluster])[0]})
    theta_raw = statistic(base)
    names, theta_vec = _as_statistic_vector(theta_raw)
    if names is not None:
        return _bootstrap_vector(
            data,
            statistic,
            names,
            theta_vec,
            n_boot=n_boot,
            cluster=cluster,
            block=block,
            idcluster=idcluster,
            alpha=alpha,
            rng=rng,
            seed=seed,
        )
    theta_hat = float(theta_vec)

    # Bootstrap distribution
    boot_stats = np.empty(n_boot)
    rows = _cluster_rows(data, cluster) if cluster is not None else None
    for b in range(n_boot):
        boot_data = _resample(
            data, rng, cluster=cluster, block=block, idcluster=idcluster, rows=rows
        )
        try:
            boot_stats[b] = statistic(boot_data)
        except Exception:
            boot_stats[b] = np.nan

    # Remove failed replications -- counted and reported, not dropped
    # silently.
    n_failed = int(np.isnan(boot_stats).sum())
    boot_stats = boot_stats[~np.isnan(boot_stats)]
    n_valid = len(boot_stats)
    if n_failed:
        warnings.warn(
            f"bootstrap: {n_failed}/{n_boot} replications failed or returned "
            "NaN and were excluded (Stata reports these as 'x').",
            RuntimeWarning,
            stacklevel=2,
        )

    if n_valid < 10:
        raise RuntimeError(
            f"Only {n_valid}/{n_boot} bootstrap replications succeeded. "
            "Check that your statistic function handles resampled data."
        )

    # Standard error
    se = float(np.std(boot_stats, ddof=1))

    # Confidence interval
    if ci_method == "percentile":
        ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    elif ci_method == "normal":
        z = sp_stats.norm.ppf(1 - alpha / 2)
        ci_lower = theta_hat - z * se
        ci_upper = theta_hat + z * se
    elif ci_method == "bca":
        ci_lower, ci_upper = _bca_ci(
            theta_hat, boot_stats, data, statistic, alpha, rng, cluster, block
        )
    else:
        raise ValueError(
            f"Unknown ci_method: {ci_method}. Use 'percentile', 'normal', or 'bca'."
        )

    # Two-sided p-value (fraction of boot dist more extreme than null)
    centered = boot_stats - theta_hat  # center at estimate
    pvalue = float(np.mean(np.abs(centered) >= abs(theta_hat - null_value)))
    pvalue = max(pvalue, 1 / (n_valid + 1))  # minimum p-value

    _result = BootstrapResult(
        estimate=theta_hat,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        pvalue=pvalue,
        alpha=alpha,
        n_boot=n_valid,
        boot_distribution=boot_stats,
        ci_method=ci_method,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.bootstrap",
            params={
                "statistic": getattr(statistic, "__name__", "anonymous"),
                "n_boot": n_boot,
                "cluster": cluster,
                "block": block,
                "ci_method": ci_method,
                "alpha": alpha,
                "seed": seed,
                "null_value": null_value,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ====================================================================== #
#  Resampling strategies
# ====================================================================== #


def _resample(
    data: pd.DataFrame,
    rng: np.random.RandomState,
    cluster: Optional[str] = None,
    block: Optional[str] = None,
    idcluster: Optional[str] = None,
    rows: Optional[dict] = None,
) -> pd.DataFrame:
    """Resample data according to the specified strategy.

    ``rows`` (``_cluster_rows``) can be passed to avoid regrouping on every
    replication.
    """
    n = len(data)

    if cluster is not None:
        # Cluster bootstrap: resample clusters, keep all rows within
        clusters = data[cluster].unique()
        boot_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        if rows is None:
            rows = _cluster_rows(data, cluster)
        idx = np.concatenate([rows[c] for c in boot_clusters])
        out = data.iloc[idx].reset_index(drop=True)
        if idcluster is not None:
            # Stata idcluster(): each draw is its own cluster.
            out[idcluster] = np.repeat(
                np.arange(len(boot_clusters)), [len(rows[c]) for c in boot_clusters]
            )
        return out

    elif block is not None:
        # Block bootstrap: resample blocks (e.g., time periods)
        blocks = data[block].unique()
        boot_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        frames = [data[data[block] == b] for b in boot_blocks]
        return pd.concat(frames, ignore_index=True)

    else:
        # Nonparametric: resample rows
        idx = rng.choice(n, size=n, replace=True)
        return data.iloc[idx].reset_index(drop=True)


def _cluster_rows(data: pd.DataFrame, cluster: str) -> dict:
    """Row positions of each cluster, in data order."""
    return {
        c: np.asarray(ix) for c, ix in data.groupby(cluster, sort=False).indices.items()
    }


def _as_statistic_vector(value: Any) -> Tuple[Optional[List[str]], Any]:
    """``(names, values)`` for a vector statistic, ``(None, float)`` for a scalar."""
    if hasattr(value, "params") and isinstance(getattr(value, "params"), pd.Series):
        p = value.params
        return [str(k) for k in p.index], p.to_numpy(dtype=float)
    if isinstance(value, pd.Series):
        return [str(k) for k in value.index], value.to_numpy(dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0 or arr.size == 1:
        return None, float(arr.reshape(-1)[0])
    return [f"_b{j}" for j in range(arr.size)], arr.reshape(-1)


def _bootstrap_vector(
    data: pd.DataFrame,
    statistic: Callable,
    names: List[str],
    theta: np.ndarray,
    *,
    n_boot: int,
    cluster: Optional[str],
    block: Optional[str],
    idcluster: Optional[str],
    alpha: float,
    rng: np.random.RandomState,
    seed: Optional[int],
) -> "EconometricResults":
    """Stata ``bootstrap _b``: SEs and covariance from the replicates."""
    from ..core.results import EconometricResults

    p = len(names)
    draws = np.full((n_boot, p), np.nan)
    rows = _cluster_rows(data, cluster) if cluster is not None else None
    for b in range(n_boot):
        boot_data = _resample(
            data, rng, cluster=cluster, block=block, idcluster=idcluster, rows=rows
        )
        try:
            got_names, got = _as_statistic_vector(statistic(boot_data))
        except Exception:
            continue
        if got_names is None:
            continue
        if got_names != names:
            # A replicate that drops or adds a term (collinearity in the
            # resample) is aligned by name; missing terms stay NaN.
            got = pd.Series(got, index=got_names).reindex(names).to_numpy()
        draws[b] = got
    ok = np.all(np.isfinite(draws), axis=1)
    n_failed = int((~ok).sum())
    if n_failed:
        warnings.warn(
            f"bootstrap: {n_failed}/{n_boot} replications failed or had a "
            "missing coefficient and were excluded (Stata reports these as "
            "'x').",
            RuntimeWarning,
            stacklevel=3,
        )
    draws = draws[ok]
    if len(draws) < 10:
        raise DataInsufficient(
            f"Only {len(draws)}/{n_boot} bootstrap replications succeeded. "
            "Check that your statistic function handles resampled data."
        )
    V = np.cov(draws, rowvar=False, ddof=1).reshape(p, p)
    se = np.sqrt(np.diag(V))
    pct = pd.DataFrame(
        {
            "ci_lower": np.percentile(draws, 100 * alpha / 2, axis=0),
            "ci_upper": np.percentile(draws, 100 * (1 - alpha / 2), axis=0),
        },
        index=names,
    )
    res = EconometricResults(
        params=pd.Series(theta, index=names),
        std_errors=pd.Series(se, index=names),
        model_info={
            "model_type": "Bootstrap",
            "method": "bootstrap"
            + (f", cluster({cluster})" if cluster else "")
            + (f", block({block})" if block else ""),
            "n_boot": int(len(draws)),
            "n_failed": n_failed,
            "cluster": cluster,
            "idcluster": idcluster,
            "seed": seed,
            "bias": pd.Series(draws.mean(axis=0) - theta, index=names),
            "percentile_ci": pct,
            "boot_distribution": pd.DataFrame(draws, columns=names),
            "alpha": alpha,
        },
        data_info={
            "nobs": int(len(data)),
            "var_cov": V,
            "var_names": list(names),
            "inference": "z",
        },
        diagnostics={},
    )
    res.alpha = alpha
    res._compute_statistics()
    return res


# ====================================================================== #
#  BCa confidence interval
# ====================================================================== #


def _bca_ci(
    theta_hat: float,
    boot_stats: np.ndarray,
    data: pd.DataFrame,
    statistic: Callable,
    alpha: float,
    rng: np.random.RandomState,
    cluster: Optional[str],
    block: Optional[str],
) -> tuple:
    """Bias-corrected and accelerated (BCa) confidence interval."""
    # Bias correction factor z0
    prop_less = np.mean(boot_stats < theta_hat)
    z0 = sp_stats.norm.ppf(max(min(prop_less, 0.999), 0.001))

    # Acceleration factor a (jackknife).
    #
    # The jackknife leave-one-out unit must match the *resampling* unit of
    # the bootstrap that produced ``boot_stats``: with ``cluster=`` the
    # bootstrap resamples clusters, so the jackknife deletes whole clusters
    # (leave-one-cluster-out), not individual rows — otherwise ``a`` is
    # estimated on a different design than the interval it corrects.
    _JACK_CAP = 200
    if cluster is not None and cluster in data.columns:
        units = list(pd.unique(data[cluster]))

        def _delete(u: object) -> pd.DataFrame:
            return data[data[cluster] != u].reset_index(drop=True)

    else:
        units = list(range(len(data)))

        def _delete(u: object) -> pd.DataFrame:
            return data.drop(data.index[u]).reset_index(drop=True)

    # Cap the jackknife for cost. Sample the units *at random* rather than
    # taking the first K, whose statistic is systematically biased when the
    # data are ordered (by group, time, …); warn that ``a`` is approximate.
    n_units = len(units)
    if n_units > _JACK_CAP:
        pick = rng.choice(n_units, size=_JACK_CAP, replace=False)
        units = [units[int(k)] for k in pick]
        warnings.warn(
            f"bootstrap: BCa acceleration is estimated from a random "
            f"{_JACK_CAP}-unit jackknife subsample of {n_units} "
            f"{'clusters' if cluster is not None else 'observations'} "
            "(exact jackknife capped for cost); the interval endpoints are "
            "approximate.",
            RuntimeWarning,
            stacklevel=2,
        )

    jack_vals: List[float] = []
    n_failed = 0
    for u in units:
        try:
            jack_vals.append(float(statistic(_delete(u))))
        except Exception:
            # Do NOT impute theta_hat: a zeroed deviation biases the
            # acceleration toward 0 (BCa silently degrades to BC). Drop the
            # point and surface the loss instead.
            n_failed += 1
    if n_failed:
        warnings.warn(
            f"bootstrap: {n_failed}/{len(units)} BCa jackknife replicate(s) "
            "failed and were dropped from the acceleration estimate.",
            RuntimeWarning,
            stacklevel=2,
        )

    jack_stats = np.asarray(jack_vals, dtype=float)
    if jack_stats.size < 2:
        # Too few usable jackknife points to estimate acceleration.
        a = 0.0
    else:
        jack_mean = np.mean(jack_stats)
        diff = jack_mean - jack_stats
        denom = np.sum(diff**2)
        a = float(np.sum(diff**3) / (6 * denom**1.5)) if denom > 0 else 0.0

    # Adjusted quantiles
    z_alpha = sp_stats.norm.ppf(alpha / 2)
    z_1alpha = sp_stats.norm.ppf(1 - alpha / 2)

    a1 = sp_stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    a2 = sp_stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    ci_lower = float(np.percentile(boot_stats, 100 * np.clip(a1, 0.001, 0.999)))
    ci_upper = float(np.percentile(boot_stats, 100 * np.clip(a2, 0.001, 0.999)))

    return ci_lower, ci_upper
