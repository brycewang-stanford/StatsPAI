"""Simulation-based power for any estimator (design diagnosis by Monte Carlo).

``sp.power("simulation", dgp=..., estimator=..., n=...)`` draws ``n_sims``
datasets from a user data-generating process, applies an estimator to each
and reports the diagnosands of the design: power (rejection rate of the null
of no effect), bias, RMSE, coverage of the confidence interval, the mean
standard error and the standard deviation of the estimates. Every quantity
carries its Monte Carlo standard error so that the simulation size can be
judged.
"""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from ..exceptions import MethodIncompatibility, NumericalInstability

if TYPE_CHECKING:
    from .power import PowerResult

__all__ = ["power_simulation"]


def _extract(
    res: Any, term: Optional[str], alpha: float
) -> Tuple[float, float, float, float, float]:
    """``(estimate, se, pvalue, ci_low, ci_high)`` from an estimator's output."""
    if isinstance(res, dict):
        res = tuple(res.get(k) for k in ("estimate", "se", "pvalue", "ci"))
        est, se, p, ci = res
        return _finish(est, se, p, ci, alpha)
    if isinstance(res, (tuple, list)):
        vals = list(res) + [None] * (4 - len(res))
        return _finish(vals[0], vals[1], vals[2], vals[3], alpha)
    if term is not None and hasattr(res, "params"):
        params = res.params
        if term not in getattr(params, "index", []):
            raise MethodIncompatibility(
                f"term {term!r} is not among the estimator's coefficients.",
                recovery_hint=f"Pick one of {list(getattr(params, 'index', []))}.",
            )
        est = float(params[term])
        se = float(res.std_errors[term]) if hasattr(res, "std_errors") else None
        p = float(res.pvalues[term]) if hasattr(res, "pvalues") else None
        ci = None
        if callable(getattr(res, "conf_int", None)):
            try:
                tab = res.conf_int(alpha=alpha)
            except TypeError:
                tab = res.conf_int() if abs(alpha - 0.05) < 1e-12 else None
            if tab is not None:
                ci = (float(tab.loc[term].iloc[0]), float(tab.loc[term].iloc[1]))
        return _finish(est, se, p, ci, alpha)
    if hasattr(res, "estimate") and hasattr(res, "se"):
        ci = getattr(res, "ci", None)
        if ci is not None and abs(float(getattr(res, "alpha", alpha)) - alpha) > 1e-12:
            ci = None
        return _finish(res.estimate, res.se, getattr(res, "pvalue", None), ci, alpha)
    raise MethodIncompatibility(
        f"Cannot read an estimate from {type(res).__name__}.",
        recovery_hint=(
            "Return a StatsPAI result with .estimate / .se (e.g. CausalResult), a "
            "regression result together with term=..., a tuple (estimate, se[, "
            "pvalue[, (ci_low, ci_high)]]) or a dict with those keys."
        ),
    )


def _finish(
    est: Any, se: Any, p: Any, ci: Any, alpha: float
) -> Tuple[float, float, float, float, float]:
    est_f = float(est)
    se_f = float("nan") if se is None else float(se)
    z = ndtri(1.0 - alpha / 2.0)
    if p is None:
        p = 2.0 * ndtr(-abs(est_f / se_f)) if se_f > 0 else float("nan")
    if ci is None:
        ci = (est_f - z * se_f, est_f + z * se_f)
    return est_f, se_f, float(p), float(ci[0]), float(ci[1])


def _call_dgp(dgp: Callable, n: Any, rng: np.random.Generator, effect: Any) -> Any:
    params = inspect.signature(dgp).parameters
    kwargs: Dict[str, Any] = {}
    if "effect_size" in params and effect is not None:
        kwargs["effect_size"] = effect
    return dgp(n, rng, **kwargs)


def power_simulation(
    n: Any,
    effect_size: Optional[float] = None,
    *,
    dgp: Callable[..., pd.DataFrame],
    estimator: Callable[[pd.DataFrame], Any],
    n_sims: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    term: Optional[str] = None,
    true_value: Optional[float] = None,
) -> "PowerResult":
    """Monte Carlo power, bias, RMSE and coverage of an estimator.

    Parameters
    ----------
    n : int or sequence of int
        Sample size(s) passed to ``dgp``; a sequence gives a power curve.
    effect_size : float, optional
        Passed to ``dgp`` as ``effect_size=`` when the callable accepts it.
        Also the true value for bias and coverage unless ``true_value`` is
        given.
    dgp : callable
        ``dgp(n, rng)`` (or ``dgp(n, rng, effect_size=...)``) returning one
        simulated dataset; ``rng`` is a ``numpy.random.Generator`` to draw
        every random number from.
    estimator : callable
        ``estimator(data)`` returning a StatsPAI result with ``.estimate``
        and ``.se`` (``CausalResult``), a regression result (then give
        ``term``), a tuple ``(estimate, se[, pvalue[, (lo, hi)]])`` or a
        dict with those keys. The result's own p-value and confidence
        interval are used when it has them.
    n_sims : int, default 500
        Simulated datasets per sample size.
    alpha : float, default 0.05
        Level of the test of a zero effect.
    seed : int, optional
        Seed; the same seed reproduces every dataset.
    term : str, optional
        Coefficient to read from a regression result.
    true_value : float, optional
        Estimand value used for bias, RMSE and coverage (default
        ``effect_size``). Without either, those are not reported.

    Returns
    -------
    PowerResult
        ``power`` (rejection rate); ``params`` holds, per sample size,
        ``bias``, ``rmse``, ``coverage``, ``mean_se``, ``sd_estimate``,
        ``n_failed`` and the Monte Carlo standard errors ``mc_se_power``,
        ``mc_se_bias`` and ``mc_se_coverage``; ``params['table']`` is the
        same as a DataFrame.

    Notes
    -----
    Draws that raise an exception are counted in ``n_failed`` and left out;
    if more than half fail the call raises. The Monte Carlo standard error of
    the power is ``sqrt(p (1 - p) / n_sims)``.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> def dgp(n, rng, effect_size=0.0):
    ...     d = rng.permutation(np.repeat([0, 1], n // 2))
    ...     return pd.DataFrame({"d": d, "y": effect_size * d + rng.normal(size=n)})
    >>> est = lambda df: sp.difference_in_means(df, "y", "d")
    >>> r = sp.power("simulation", n=200, effect_size=0.4, dgp=dgp,
    ...              estimator=est, n_sims=200, seed=1)
    >>> 0.7 < r.power < 0.9
    True
    """
    from .power import PowerResult

    if not callable(dgp) or not callable(estimator):
        raise MethodIncompatibility(
            "power('simulation') needs callables dgp= and estimator=.",
            recovery_hint=(
                "dgp(n, rng) -> DataFrame; estimator(df) -> result with "
                ".estimate and .se."
            ),
        )
    n_sims = int(n_sims)
    if n_sims < 2:
        raise MethodIncompatibility(
            f"n_sims must be at least 2; got {n_sims}.",
            recovery_hint="E.g. n_sims=500.",
        )
    if not (0 < alpha < 1):
        raise MethodIncompatibility(
            f"alpha must be in (0, 1); got {alpha!r}.", recovery_hint="E.g. alpha=0.05."
        )
    truth = true_value if true_value is not None else effect_size
    sizes = np.atleast_1d(np.asarray(list(n) if isinstance(n, range) else n))
    scalar_n = np.ndim(n) == 0
    root = np.random.SeedSequence(seed)
    rows = []
    for size, child in zip(sizes, root.spawn(len(sizes))):
        size_v = size.item() if hasattr(size, "item") else size
        rng = np.random.default_rng(child)
        draws = []
        failed = 0
        last_error: Optional[BaseException] = None
        for _ in range(n_sims):
            try:
                data = _call_dgp(dgp, size_v, rng, effect_size)
                draws.append(_extract(estimator(data), term, alpha))
            except MethodIncompatibility:
                raise
            except Exception as exc:  # a draw where the estimator fails
                failed += 1
                last_error = exc
        if failed:
            warnings.warn(
                f"power('simulation'): {failed} of {n_sims} draws failed at "
                f"n={size_v} and were left out (last error: {last_error!r}).",
                RuntimeWarning,
                stacklevel=2,
            )
        if len(draws) < n_sims / 2:
            raise NumericalInstability(
                f"power('simulation'): {failed} of {n_sims} draws failed at "
                f"n={size_v}.",
                recovery_hint="Check that dgp and estimator run on one draw by hand.",
                diagnostics={
                    "n": size_v,
                    "n_failed": failed,
                    "last_error": repr(last_error),
                },
            )
        arr = np.asarray(draws, dtype=float)
        est, se, p, lo, hi = arr.T
        m = len(arr)
        rej = float(np.mean(p <= alpha))
        row: Dict[str, Any] = {
            "n": size_v,
            "power": rej,
            "mc_se_power": float(np.sqrt(rej * (1 - rej) / m)),
            "mean_estimate": float(np.mean(est)),
            "sd_estimate": float(np.std(est, ddof=1)),
            "mean_se": float(np.nanmean(se)),
            "n_sims": m,
            "n_failed": failed,
        }
        if truth is not None:
            err = est - float(truth)
            cov = float(np.mean((lo <= truth) & (truth <= hi)))
            row.update(
                {
                    "bias": float(np.mean(err)),
                    "mc_se_bias": float(np.std(est, ddof=1) / np.sqrt(m)),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "coverage": cov,
                    "mc_se_coverage": float(np.sqrt(cov * (1 - cov) / m)),
                }
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    pw = table["power"].to_numpy()
    params: Dict[str, Any] = {
        "alpha": alpha,
        "n_sims": n_sims,
        "seed": seed,
        "true_value": truth,
        "table": table,
    }
    for col in table.columns:
        if col != "n":
            vals = table[col].to_numpy()
            params[col] = vals.item() if scalar_n else vals
    return PowerResult(
        power_val=float(pw[0]) if scalar_n else pw,
        n=n,
        effect_size=effect_size,
        design="simulation",
        params=params,
    )
