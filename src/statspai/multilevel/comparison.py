"""
Model-comparison helpers for mixed-effects models.

``lrtest`` performs a likelihood-ratio test between two nested fits of
``mixed()`` / ``meglm()``.  When the difference lies purely in the
number of variance components being tested, the asymptotic reference
distribution is a mixture of chi-squareds (χ̄²) rather than a plain
χ² — we apply the Self-Liang (1987) 50/50 mixture correction
automatically for single-component boundary tests; for multi-component
boundary tests we fall back to a conservative average of the
bracketing χ² df's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class LRTestResult:
    chi2: float
    df: float
    p_value: float
    boundary_corrected: bool
    restricted_logL: float
    full_logL: float

    def summary(self) -> str:
        tag = " (χ̄² boundary-corrected)" if self.boundary_corrected else ""
        return (
            f"LR test{tag}: chi² = {self.chi2:.4f}, df = {self.df:.1f}, "
            f"p-value = {self.p_value:.4g}"
        )

    def __float__(self) -> float:
        return float(self.chi2)


def _n_free_params(result: Any) -> int:
    """Guess the total number of free parameters of a mixed-model fit."""
    if hasattr(result, "n_params"):
        return int(result.n_params)
    # Fallback for MEGLMResult / generic objects.
    npars = getattr(result, "_n_total_params", None)
    if npars is not None:
        return int(npars)
    return int(len(result.params)) + int(getattr(result, "_n_cov_params", 1))


def _n_variance_params(result: Any) -> int:
    if hasattr(result, "_n_cov_params"):
        return int(result._n_cov_params)
    return 1


def lrtest(
    restricted: Any,
    full: Any,
    boundary: "bool | None" = None,
) -> LRTestResult:
    """
    Likelihood-ratio test comparing a *restricted* and a *full* model.

    Parameters
    ----------
    restricted, full
        Two fitted mixed models.  ``full`` should strictly nest
        ``restricted`` — i.e. the parameter space of the restricted
        model is a subset of the full model's.
    boundary
        Whether to apply the Self-Liang χ̄² boundary correction.  When
        ``None`` (default) we infer it from whether the restriction
        touches a variance component — the only parameters that live on
        the boundary of their support.

    Returns
    -------
    LRTestResult
    """
    ll_r = float(restricted.log_likelihood)
    ll_f = float(full.log_likelihood)
    chi2 = max(2.0 * (ll_f - ll_r), 0.0)

    k_r = _n_free_params(restricted)
    k_f = _n_free_params(full)
    df = max(k_f - k_r, 0)
    var_df = _n_variance_params(full) - _n_variance_params(restricted)

    if boundary is None:
        boundary = var_df > 0

    if boundary and df == 1:
        # Classic 50/50 mixture of χ²_0 and χ²_1 (Self-Liang 1987).
        p = 0.5 * (1.0 - stats.chi2.cdf(chi2, 1))
    elif boundary and df >= 2:
        # Conservative: average of χ²_(df-1) and χ²_df tail probabilities.
        p = 0.5 * (
            (1.0 - stats.chi2.cdf(chi2, df - 1))
            + (1.0 - stats.chi2.cdf(chi2, df))
        )
    else:
        p = 1.0 - stats.chi2.cdf(chi2, df) if df > 0 else 1.0

    return LRTestResult(
        chi2=chi2,
        df=float(df),
        p_value=float(p),
        boundary_corrected=bool(boundary),
        restricted_logL=ll_r,
        full_logL=ll_f,
    )


__all__ = ["lrtest", "LRTestResult"]
