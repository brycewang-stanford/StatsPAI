"""Hendren's pooled price ratio and the implicit tax of adverse selection.

A contract that pays one if a loss happens is actuarially fair for a
person with loss probability ``r`` at the premium ``r / (1 - r)``. If the
contract is bought by everyone with risk ``r`` or more, it breaks even
only at ``m / (1 - m)`` with ``m = E[P | P >= r]``. The ratio of the two
premiums is the pooled price ratio::

    T(r) = [E[P | P >= r] / (1 - E[P | P >= r])] / [r / (1 - r)]

and the implicit tax is ``t(r) = T(r) - 1``: how much more than her own
fair price the person at ``r`` must pay to buy into the pool of everyone
riskier. Hendren's no-trade condition says the market cannot offer the
contract when ``T(r)`` exceeds the marginal rate of substitution between
the loss and no-loss states, ``u'(w - L) / u'(w)``, at every ``r`` below
one. Writing that ratio as ``1 + markup``, no trade occurs when
``t(r) >= markup`` everywhere.

The distribution of ``P`` is the distribution of private risk within a
market segment (the people the insurer cannot tell apart), for example
predicted probabilities from a model that adds genetic scores to the
variables insurers price on.

References
----------
[@hendren2013private]
[@azevedo2026genetic]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._results import PooledPriceRatioResult

__all__ = ["pooled_price_ratio"]


def _curve(p: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """Exact pooled price ratio at every support point of a weighted sample."""
    order = np.argsort(p, kind="stable")
    p, w = p[order], w[order]
    uniq, start = np.unique(p, return_index=True)
    # Weighted totals from each support point upward (ties included).
    tail_w = np.cumsum(w[::-1])[::-1][start]
    tail_wp = np.cumsum((w * p)[::-1])[::-1][start]
    total = float(w.sum())
    mean_above = tail_wp / tail_w
    below = 1.0 - tail_w / total
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (mean_above / (1.0 - mean_above)) / (uniq / (1.0 - uniq))
    return pd.DataFrame(
        {
            "quantile": np.clip(below, 0.0, 1.0),
            "r": uniq,
            "mean_above": mean_above,
            "T": ratio,
            "t": ratio - 1.0,
        }
    )


def _summaries(curve: pd.DataFrame, upto: float) -> Dict[str, Any]:
    # Hendren's condition ranges over the support below one; at r = 0 the
    # ratio is infinite, which never binds.
    inner = curve[(curve["r"] > 0) & (curve["r"] < 1)]
    if inner.empty:
        raise DataInsufficient(
            "no risk values strictly between 0 and 1",
            recovery_hint="Pass probabilities in (0, 1).",
        )
    # r <= Q(upto) exactly when the mass strictly below r is less than upto.
    head = inner[inner["quantile"] < upto]
    if head.empty:
        head = inner.iloc[:1]
    i = int(np.argmin(head["t"].to_numpy()))
    return {
        "t_min": float(head["t"].iloc[i]),
        "argmin_r": float(head["r"].iloc[i]),
        "t_min_all": float(inner["t"].min()),
        "head": head,
        "inner": inner,
    }


def pooled_price_ratio(
    p: Union[str, Sequence[float], np.ndarray, pd.Series],
    data: Optional[pd.DataFrame] = None,
    weights: Optional[Union[str, Sequence[float], np.ndarray]] = None,
    upto_quantile: float = 0.8,
    markup: Optional[float] = None,
    n_boot: int = 0,
    seed: Optional[int] = 0,
    alpha: float = 0.05,
) -> PooledPriceRatioResult:
    """Pooled price ratio, implicit tax and Hendren's no-trade condition.

    Parameters
    ----------
    p : str or array-like
        Loss probabilities of the people in one market segment (a column of
        ``data``, or the values themselves), each in ``[0, 1]``.
    data : pandas.DataFrame, optional
        Needed when ``p`` or ``weights`` is a column name.
    weights : str or array-like, optional
        Population weights.
    upto_quantile : float, default 0.8
        The summary ``estimate`` is the minimum implicit tax over risk
        values up to this quantile of the distribution (``t80`` with the
        default, the statistic Hendren reports and Azevedo et al. adopt).
    markup : float, optional
        Willingness to pay above the actuarially fair price, as a share
        (``0.2`` means 20 percent above), i.e. ``u'(w - L) / u'(w) - 1``.
        When given, ``no_trade`` reports whether ``t(r) >= markup`` at
        every risk value below one.
    n_boot : int, default 0
        Bootstrap draws (resampling people) for an interval around the
        minimum implicit tax. ``0`` skips the bootstrap.
    seed : int, optional
    alpha : float, default 0.05

    Returns
    -------
    PooledPriceRatioResult
        ``curve`` has one row per distinct risk value with its quantile,
        ``E[P | P >= r]``, ``T`` and ``t``; ``estimate`` is the minimum
        implicit tax up to ``upto_quantile``.

    Notes
    -----
    The curve is computed exactly on the empirical distribution: at each
    distinct value ``r`` the mean is taken over everyone with risk at least
    ``r``, ties included. ``quantile`` is the share of the population with
    risk strictly below ``r``, so ``r`` is at or below the
    ``upto_quantile`` quantile exactly when ``quantile < upto_quantile``.
    At the top of the support the pool is the person alone and ``t`` is
    zero, which is why the summary stops at a quantile below one.

    The bootstrap resamples the given probabilities only. When they are
    predictions from a fitted model, the estimation error of that model is
    not included; bootstrap the whole pipeline for that.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> risk = rng.beta(2, 30, size=5000)
    >>> res = sp.pooled_price_ratio(risk)
    >>> bool(res.estimate > 0)
    True
    >>> res.curve.columns.tolist()
    ['quantile', 'r', 'mean_above', 'T', 't']

    References
    ----------
    [@hendren2013private]
    [@azevedo2026genetic]
    """
    if not 0 < upto_quantile <= 1:
        raise MethodIncompatibility(
            "upto_quantile must be in (0, 1]",
            recovery_hint="Pass e.g. upto_quantile=0.8.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            "alpha must be in (0, 1)", recovery_hint="Pass e.g. alpha=0.05."
        )
    pv = _resolve(p, data, "p")
    if weights is None:
        wv = np.ones_like(pv)
    else:
        wv = _resolve(weights, data, "weights")
        if wv.shape != pv.shape:
            raise MethodIncompatibility(
                "weights must have one entry per probability",
                recovery_hint="Pass weights of the same length as p.",
            )
    ok = np.isfinite(pv) & np.isfinite(wv)
    pv, wv = pv[ok], wv[ok]
    if pv.size < 2:
        raise DataInsufficient(
            "need at least two probabilities",
            recovery_hint="Pass the risk distribution of a market segment.",
        )
    if (pv < 0).any() or (pv > 1).any():
        raise MethodIncompatibility(
            "probabilities must lie in [0, 1]",
            recovery_hint="Pass predicted loss probabilities, not an index.",
        )
    if (wv < 0).any() or wv.sum() <= 0:
        raise MethodIncompatibility(
            "weights must be non-negative with a positive total",
            recovery_hint="Drop negative weights.",
        )

    curve = _curve(pv, wv)
    s = _summaries(curve, upto_quantile)

    no_trade = no_trade_upto = None
    if markup is not None:
        if markup < 0:
            raise MethodIncompatibility(
                "markup must be non-negative",
                recovery_hint="Pass the willingness to pay above the fair "
                "price as a share, e.g. markup=0.2.",
            )
        no_trade = bool((s["inner"]["t"] >= markup).all())
        no_trade_upto = bool((s["head"]["t"] >= markup).all())

    se = ci = None
    n_failed = 0
    if n_boot:
        if n_boot < 2:
            raise MethodIncompatibility(
                "n_boot must be 0 or at least 2",
                recovery_hint="Pass e.g. n_boot=200.",
            )
        rng = np.random.default_rng(seed)
        draws = np.full(n_boot, np.nan)
        for b in range(n_boot):
            idx = rng.integers(0, pv.size, size=pv.size)
            try:
                draws[b] = _summaries(_curve(pv[idx], wv[idx]), upto_quantile)["t_min"]
            except DataInsufficient:
                n_failed += 1
        se = float(np.nanstd(draws, ddof=1))
        ci = (
            float(np.nanquantile(draws, alpha / 2)),
            float(np.nanquantile(draws, 1 - alpha / 2)),
        )

    return PooledPriceRatioResult(
        curve=curve,
        estimate=s["t_min"],
        argmin_r=s["argmin_r"],
        t_min_all=s["t_min_all"],
        upto_quantile=upto_quantile,
        markup=markup,
        no_trade=no_trade,
        no_trade_upto=no_trade_upto,
        se=se,
        ci=ci,
        alpha=alpha,
        n_obs=int(pv.size),
        diagnostics={
            "n_dropped_nonfinite": int((~ok).sum()),
            "n_support_points": int(len(curve)),
            "n_boot_failed": n_failed,
        },
    )


def _resolve(
    value: Union[str, Sequence[float], np.ndarray, pd.Series],
    data: Optional[pd.DataFrame],
    name: str,
) -> np.ndarray:
    if isinstance(value, str):
        if data is None:
            raise MethodIncompatibility(
                f"{name}={value!r} is a column name but no data was given",
                recovery_hint=f"Pass data=, or the {name} values themselves.",
            )
        from .._input_validation import check_columns

        check_columns(data, {name: value}, function="pooled_price_ratio")
        col: np.ndarray = data[value].to_numpy(dtype=float)
        return col
    return np.asarray(value, dtype=float).ravel()
