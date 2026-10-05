"""BDS test of independence for a time series.

Applied to the residuals of a fitted model it asks whether any dependence
is left, linear or not.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["bds"]

#: The pairwise comparison holds an n x n table of booleans.
_MAX_OBS = 20000


def _close_pairs(table: np.ndarray, dim: int) -> np.ndarray:
    """Pairs of ``dim``-histories whose every coordinate is within epsilon.

    ``table[i, j]`` says whether observations ``i`` and ``j`` are close.
    Histories are indexed by their last observation, so the result has
    ``n - dim + 1`` rows.
    """
    n = table.shape[0]
    joint = table[dim - 1 :, dim - 1 :].copy()
    for back in range(1, dim):
        joint &= table[dim - 1 - back : n - back, dim - 1 - back : n - back]
    return joint


def _share_close(joint: np.ndarray) -> float:
    """Share of distinct pairs that are close: the correlation integral."""
    n = joint.shape[0]
    # the diagonal is always True; count the pairs above it
    return float((joint.sum() - n) / (n * (n - 1.0)))


def bds(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, Any],
    y: Optional[str] = None,
    *,
    max_dim: int = 2,
    epsilon: Optional[float] = None,
    distance: float = 1.5,
) -> pd.DataFrame:
    """BDS test that a series is independent and identically distributed.

    For each embedding dimension ``m`` the test compares the share of pairs
    of ``m``-histories that stay within ``epsilon`` of each other,
    ``C_m``, with what independence implies, ``C_1 ** m``. The scaled
    difference is asymptotically standard normal under the null. It has
    power against linear dependence, nonlinear dependence and chaos alike,
    so it is usually run on the residuals of a fitted model: a rejection
    says the model has left structure behind, not what kind.

    Parameters
    ----------
    data : DataFrame, Series, array or fitted result
        The series in time order. A fitted regression is accepted and its
        residuals are tested.
    y : str, optional
        Column, when ``data`` is a DataFrame.
    max_dim : int, default 2
        Largest embedding dimension. Dimensions ``2..max_dim`` are tested.
    epsilon : float, optional
        Radius within which two observations count as close. Default:
        ``distance`` standard deviations of the series.
    distance : float, default 1.5
        Multiple of the sample standard deviation (divisor ``n - 1``) used
        when ``epsilon`` is not given.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``dim``, with the columns ``statistic``, ``pvalue``
        (two-sided, standard normal), ``c_m`` and ``c_1`` (the two
        correlation integrals, both computed on the last ``n - m + 1``
        observations) and ``nobs`` (that number). ``attrs`` carries
        ``epsilon`` and ``k``.

    Notes
    -----
    The normal approximation is poor in short series: with fewer than
    about 200 observations, or ``m`` large relative to ``n``, the test
    rejects a true null too often and the p-values should be bootstrapped.

    On residuals the limit law is unchanged for a linear model fitted by
    least squares, but not for the standardised residuals of a GARCH model.

    The statistic is built as statsmodels' ``bds`` builds it: the
    one-dimensional integral is recomputed on the observations each
    dimension actually uses, and the variance uses the whole series.

    References
    ----------
    broock1996test

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> out = sp.bds(rng.normal(size=500), max_dim=3)
    >>> list(out.index), list(out.columns)
    ([2, 3], ['statistic', 'pvalue', 'c_m', 'c_1', 'nobs'])
    >>> bool((out["pvalue"] > 0.05).all())
    True
    """
    if isinstance(data, pd.DataFrame):
        if y is None:
            if data.shape[1] != 1:
                raise MethodIncompatibility(
                    "bds: y= names the column to test.",
                    recovery_hint='sp.bds(df, "resid")',
                )
            values = data.iloc[:, 0]
        elif y not in data.columns:
            raise MethodIncompatibility(
                f"bds: column {y!r} not found.", recovery_hint="Check the name."
            )
        else:
            values = data[y]
    else:
        info = getattr(data, "data_info", None)
        values = data
        if isinstance(info, dict) and info.get("residuals") is not None:
            values = info["residuals"]
    x = np.asarray(values, dtype=float).ravel()
    if np.isnan(x).any():
        raise MethodIncompatibility(
            "bds: the series has missing values.",
            recovery_hint="Drop them; the test needs consecutive observations.",
        )
    n = x.size
    max_dim = int(max_dim)
    if max_dim < 2:
        raise MethodIncompatibility("bds: max_dim must be at least 2.")
    if n <= max_dim + 2:
        raise DataInsufficient(
            f"bds: {n} observations are too few for dimension {max_dim}."
        )
    if n > _MAX_OBS:
        raise MethodIncompatibility(
            f"bds: {n} observations need a {n} x {n} comparison table.",
            recovery_hint=f"Test a window of at most {_MAX_OBS} observations.",
        )
    if epsilon is None:
        if not distance > 0:
            raise MethodIncompatibility("bds: distance must be positive.")
        epsilon = float(distance * np.std(x, ddof=1))
    epsilon = float(epsilon)
    if not epsilon > 0:
        raise MethodIncompatibility(
            "bds: epsilon must be positive (is the series constant?)."
        )

    table = np.abs(x[:, None] - x[None, :]) < epsilon

    # Variance terms from the whole series: c is the probability that two
    # observations are close, k that three are mutually chained.
    c = _share_close(table)
    row = table.sum(axis=1).astype(float)
    k = float(
        (np.sum(row**2) - 3.0 * row.sum() + 2.0 * n) / (n * (n - 1.0) * (n - 2.0))
    )

    rows = []
    for m in range(2, max_dim + 1):
        used = n - (m - 1)
        c_m = _share_close(_close_pairs(table, m))
        c_1 = _share_close(table[m - 1 :, m - 1 :])
        cross = sum(k ** (m - j) * c ** (2 * j) for j in range(1, m))
        variance = 4.0 * (
            k**m
            + 2.0 * cross
            + (m - 1) ** 2 * c ** (2 * m)
            - m**2 * k * c ** (2 * m - 2)
        )
        if not variance > 0:
            stat = float("nan")
        else:
            stat = float(np.sqrt(used) * (c_m - c_1**m) / np.sqrt(variance))
        rows.append(
            {
                "dim": m,
                "statistic": stat,
                "pvalue": float(2.0 * stats.norm.sf(abs(stat))),
                "c_m": c_m,
                "c_1": c_1,
                "nobs": used,
            }
        )
    out = pd.DataFrame(rows).set_index("dim")
    out.attrs["epsilon"] = epsilon
    out.attrs["k"] = k
    return out
