"""Ordered outcomes in the positive-correlation test.

With coverage (or claims) in ``J >= 3`` ordered levels, the latent-error
model of the residual test becomes an ordered probit: level ``j`` is
observed when ``kappa_{j-1} < x'b + e <= kappa_j``. The score of the
bivariate model with error correlation ``rho`` at ``rho = 0`` is still the
product of the two generalized residuals: the derivative of the
bivariate normal density with respect to ``rho`` at zero is
``u v phi(u) phi(v)``, so the derivative of a rectangle's probability is
``(phi(l1) - phi(u1)) (phi(l2) - phi(u2))``, and dividing by the
probability at ``rho = 0`` gives ``e1 e2`` with
``e = (phi(l) - phi(u)) / (Phi(u) - Phi(l))``. A binary outcome is the
case ``J = 2`` and gives the probit generalized residual.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

#: Upper limit on the number of levels treated as an ordered outcome.
MAX_LEVELS = 20


def outcome_kind(values: np.ndarray, name: str) -> str:
    """``'binary'`` for 0/1 (or constant) codes, ``'ordered'`` for 3-20 levels."""
    if np.isin(values, (0.0, 1.0)).all():
        return "binary"
    levels = np.unique(values)
    if 3 <= levels.size <= MAX_LEVELS:
        return "ordered"
    raise MethodIncompatibility(
        f"{name!r} must be coded 0/1 or take 3 to {MAX_LEVELS} ordered values "
        f"(found {levels.size} distinct values)",
        recovery_hint=f"Recode {name!r} as a 0/1 indicator or as ordered "
        "levels (e.g. 0, 1, 2 for coverage tiers or claim counts top-coded).",
    )


def ordered_generalized_residual(df: pd.DataFrame, y: str, x: List[str]) -> np.ndarray:
    """``E[e | y, x]`` of an ordered probit of ``y`` on ``x``."""
    yv = df[y].to_numpy(dtype=float)
    levels = np.unique(yv)
    idx = np.searchsorted(levels, yv)
    if x:
        from ..regression.multinomial import oprobit

        fit = oprobit(data=df, y=y, x=x)
        beta = fit.params.reindex(x).to_numpy(dtype=float)
        cuts = fit.model_info["cutpoints"]
        kappa = np.array([cuts[f"cut{j + 1}"] for j in range(len(cuts))], dtype=float)
        xb = df[x].to_numpy(dtype=float) @ beta
    else:
        shares = np.cumsum(np.bincount(idx, minlength=levels.size))[:-1] / yv.size
        kappa = stats.norm.ppf(shares)
        xb = np.zeros(yv.size)
    if kappa.size != levels.size - 1 or not np.isfinite(kappa).all():
        raise DataInsufficient(
            f"the ordered probit of {y!r} did not return {levels.size - 1} "
            "finite cutpoints",
            recovery_hint="Merge sparse levels of the outcome.",
        )
    bounds = np.r_[-np.inf, kappa, np.inf]
    lo = bounds[idx] - xb
    hi = bounds[idx + 1] - xb
    # Probability of the interval, from the tail that keeps precision.
    prob = np.where(
        lo > 0,
        stats.norm.sf(lo) - stats.norm.sf(hi),
        stats.norm.cdf(hi) - stats.norm.cdf(lo),
    )
    if (prob <= 0).any():
        raise DataInsufficient(
            f"the ordered probit of {y!r} gives some observed levels zero "
            "probability",
            recovery_hint="Drop or merge the covariate cells that predict "
            "the outcome perfectly.",
        )
    return np.asarray((stats.norm.pdf(lo) - stats.norm.pdf(hi)) / prob)
