"""Refit bootstrap for estimators whose fitted model is itself random.

A neural estimator's plug-in standard error holds the trained network
fixed. That leaves out the sampling variability of the network and the
run-to-run variability of training, which together are most of the error.
The only standard error that carries both is the spread of the estimate
across refits on resampled data, each from its own initialisation.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

__all__ = ["apply_refit_bootstrap"]

# Below this many successful refits a standard deviation is not worth
# reporting as a standard error.
_MIN_REFITS = 5


def apply_refit_bootstrap(
    result: Any,
    data: pd.DataFrame,
    refit: Callable[[pd.DataFrame, int], Any],
    *,
    n_refits: int,
    random_state: int,
    alpha: float,
    label: str,
    cluster: Optional[str] = None,
) -> Any:
    """Replace ``result``'s SE, CI and p-value by refit-bootstrap ones.

    Parameters
    ----------
    result : CausalResult
        The fit on the full sample. Modified in place and returned.
    data : pd.DataFrame
        The rows the estimator was fitted on.
    refit : callable
        ``refit(resampled_data, seed)`` returns a fitted result with an
        ``estimate`` attribute. It is given a different seed each time so
        the refits do not share an initialisation.
    n_refits : int
        Number of bootstrap refits; at least 5.
    random_state : int
        Seed for the resampling.
    alpha : float
        Significance level of the interval.
    label : str
        Name used in messages.
    cluster : str, optional
        Column to resample by instead of rows.
    """
    if n_refits < _MIN_REFITS:
        from ..exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            f"{label}: refit_bootstrap must be 0 or at least {_MIN_REFITS}, "
            f"got {n_refits}."
        )
    rng = np.random.default_rng(random_state)
    n = len(data)
    if cluster is not None:
        codes, uniques = pd.factorize(data[cluster])
        groups = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    draws = []
    errors = []
    for b in range(n_refits):
        if cluster is None:
            idx = rng.integers(0, n, size=n)
        else:
            picked = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[g] for g in picked])
        sample = data.iloc[idx].reset_index(drop=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = refit(sample, int(random_state) + 1 + b)
            value = float(fit.estimate)
        except Exception as exc:  # a degenerate resample; reported below
            errors.append(exc)
            continue
        if np.isfinite(value):
            draws.append(value)
    if len(draws) < n_refits:
        from ._fallback import warn_fallback

        warn_fallback(
            f"{n_refits - len(draws)} of {n_refits} {label} bootstrap refits",
            errors[0] if errors else None,
            "the standard error uses the remaining refits",
        )
    if len(draws) < _MIN_REFITS:
        from ..exceptions import NumericalInstability

        raise NumericalInstability(
            f"{label}: only {len(draws)} of {n_refits} bootstrap refits "
            "succeeded; no standard error can be formed."
        )
    se = float(np.std(draws, ddof=1))
    estimate = float(result.estimate)
    z = float(sp_stats.norm.ppf(1 - alpha / 2))
    info = result.model_info
    info["se_plugin"] = float(result.se)
    result.se = se
    result.ci = (estimate - z * se, estimate + z * se)
    result.pvalue = (
        float(2 * sp_stats.norm.sf(abs(estimate) / se)) if se > 0 else float("nan")
    )
    info["se_method"] = "refit_bootstrap"
    info["se_valid_for_ate"] = True
    info["se_note"] = (
        f"Standard deviation of the estimate over {len(draws)} refits on "
        "bootstrap resamples, each with its own initialisation."
    )
    info["refit_bootstrap"] = {
        "n_refits": int(n_refits),
        "n_ok": int(len(draws)),
        "draws": np.asarray(draws, dtype=float),
        "mean": float(np.mean(draws)),
    }
    return result
