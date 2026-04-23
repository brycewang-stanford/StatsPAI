"""
Rosenbaum (2002, 2010) sensitivity analysis for matched observational studies.

Given matched pairs (treated, control), the Rosenbaum framework asks:
how large an unmeasured confounder (odds ratio Gamma) would be needed
to render the p-value non-significant?

Under the null of no effect, if treatment assignment within matched
pair i has odds ratio bounded by Gamma, the signed-rank (Wilcoxon) or
sign-test p-value is bounded between a lower and an upper extremum.
The smallest Gamma >= 1 at which the upper-bound p-value exceeds alpha
is the *critical* Gamma — higher Gamma means the study is more robust
to hidden bias.

Two flavours are implemented:

* ``method="wilcoxon"`` — Rosenbaum (2002, §4) Wilcoxon signed-rank
  bounds. Works for continuous outcomes.
* ``method="sign"`` — Binomial / sign-test bounds. Works for binary
  or any paired data; simpler and distribution-free.

References
----------
Rosenbaum, P. R. (2002). *Observational Studies*, 2nd ed. Springer.
Rosenbaum, P. R. (2010). *Design of Observational Studies*. Springer. [@rosenbaum2002observational]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RosenbaumResult:
    """Result container for :func:`rosenbaum_bounds`."""

    gamma_grid: np.ndarray
    pvalue_lower: np.ndarray
    pvalue_upper: np.ndarray
    gamma_critical: float
    method: str
    alpha: float
    n_pairs: int
    statistic: float
    alternative: str
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:  # pragma: no cover - thin formatter
        lines = [
            "Rosenbaum Sensitivity Bounds",
            "============================",
            f"Method           : {self.method}",
            f"Alternative      : {self.alternative}",
            f"Pairs (n)        : {self.n_pairs}",
            f"Statistic        : {self.statistic:.4f}",
            f"Critical Gamma*  : {self.gamma_critical:.4f}  (alpha={self.alpha})",
            "",
            "Gamma    p_lower   p_upper",
        ]
        for g, lo, hi in zip(self.gamma_grid, self.pvalue_lower, self.pvalue_upper):
            lines.append(f"{g:7.3f}  {lo:8.4f}  {hi:8.4f}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RosenbaumResult(gamma*={self.gamma_critical:.3f}, "
            f"method={self.method}, n_pairs={self.n_pairs})"
        )


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _normal_sf(z: float) -> float:
    """One-sided upper-tail probability for a standard normal."""
    return float(stats.norm.sf(z))


def _sign_test_bound(T_plus: int, n: int, p: float) -> float:
    """Upper-tail Binomial(n, p) p-value P(X >= T_plus)."""
    if n == 0:
        return 1.0
    return float(stats.binom.sf(T_plus - 1, n, p))


def _wilcoxon_bound(
    ranks_abs: np.ndarray,
    signs: np.ndarray,
    p: float,
    alternative: str,
) -> float:
    """
    Compute Rosenbaum's bounding p-value for Wilcoxon signed-rank under
    odds-ratio Gamma implied by ``p``.

    Each nonzero rank ``r_i`` contributes a Bernoulli(p)·r_i summand.
    The test statistic is T = sum_i sign_i * r_i. Under the worst case
    ``p`` we compare the *observed* T to the extreme moments.
    """
    ranks_abs = np.asarray(ranks_abs, dtype=float)
    if ranks_abs.size == 0:
        return 1.0

    T_obs = float(np.sum(ranks_abs * (signs > 0)))
    mean = p * float(ranks_abs.sum())
    var = (p * (1.0 - p)) * float(np.sum(ranks_abs ** 2))

    if var <= 0:
        # Degenerate — one-sided p at the corner.
        return 1.0 if T_obs < mean else 0.0

    z = (T_obs - 0.5 - mean) / np.sqrt(var)
    if alternative == "greater":
        return _normal_sf(z)
    if alternative == "less":
        return _normal_sf(-(T_obs + 0.5 - mean) / np.sqrt(var))
    # two-sided
    return float(2.0 * min(_normal_sf(abs(z)), 0.5))


def _wilcoxon_ranks(diffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed-rank transformation with mid-ranks for ties; drops zeros."""
    mask = diffs != 0
    d = diffs[mask]
    ranks = stats.rankdata(np.abs(d), method="average")
    signs = np.where(d > 0, 1, -1)
    return ranks, signs


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------

def rosenbaum_bounds(
    treated: Optional[Sequence[float]] = None,
    control: Optional[Sequence[float]] = None,
    *,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    treat: Optional[str] = None,
    pair_id: Optional[str] = None,
    method: str = "wilcoxon",
    alternative: str = "greater",
    gamma_grid: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
) -> RosenbaumResult:
    """
    Compute Rosenbaum bounds on a paired observational study.

    You can pass either two parallel arrays ``(treated, control)`` of
    matched outcomes, or a long-format ``DataFrame`` with columns
    ``y``, ``treat``, ``pair_id``.

    Parameters
    ----------
    treated, control : array-like, optional
        Outcome in the treated / control unit of each matched pair
        (same length). Ignored if ``data`` is provided.
    data : pd.DataFrame, optional
        Long-format data. Must contain exactly two rows per ``pair_id``,
        one with ``treat=1`` and one with ``treat=0``.
    method : {"wilcoxon", "sign"}, default "wilcoxon"
        Wilcoxon signed-rank bound (continuous) or binomial sign test
        (robust / binary).
    alternative : {"greater", "less", "two-sided"}, default "greater"
        Direction of the alternative hypothesis for the treatment effect.
    gamma_grid : sequence of float, optional
        Gamma values (>= 1) over which to compute bounding p-values.
        Default: ``np.arange(1.0, 3.01, 0.1)``.
    alpha : float, default 0.05
        Significance level used to report ``gamma_critical``.

    Returns
    -------
    RosenbaumResult
        ``gamma_critical`` is the smallest Gamma in the grid at which the
        upper-bound p-value exceeds ``alpha``. It is ``inf`` if the study
        is insensitive across the grid, and ``1.0`` if already sensitive.
    """
    if method not in {"wilcoxon", "sign"}:
        raise ValueError("method must be 'wilcoxon' or 'sign'")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    # ---- Build paired differences -----------------------------------
    if data is not None:
        if not all([y, treat, pair_id]):
            raise ValueError("When `data` is given, y/treat/pair_id are required")
        d = data[[y, treat, pair_id]].dropna().copy()
        wide = d.pivot(index=pair_id, columns=treat, values=y)
        if 0 not in wide.columns or 1 not in wide.columns:
            raise ValueError("treat column must contain both 0 and 1 values")
        wide = wide.dropna()
        diffs = (wide[1].values - wide[0].values).astype(float)
    else:
        if treated is None or control is None:
            raise ValueError("Provide either (treated, control) or `data`")
        t = np.asarray(treated, dtype=float)
        c = np.asarray(control, dtype=float)
        if t.shape != c.shape:
            raise ValueError("treated/control must have the same length")
        diffs = t - c

    n_pairs = int(diffs.size)
    if n_pairs == 0:
        raise ValueError("No matched pairs after cleaning")

    if gamma_grid is None:
        gamma_grid = np.round(np.arange(1.0, 3.01, 0.1), 3)
    gamma_grid = np.asarray(gamma_grid, dtype=float)
    if np.any(gamma_grid < 1.0):
        raise ValueError("gamma_grid must be >= 1")

    # ---- Compute bounds per Gamma ----------------------------------
    lowers = np.empty_like(gamma_grid)
    uppers = np.empty_like(gamma_grid)

    if method == "wilcoxon":
        ranks_abs, signs = _wilcoxon_ranks(diffs)
        statistic = float(np.sum(ranks_abs * (signs > 0)))
        for i, gamma in enumerate(gamma_grid):
            p_low = 1.0 / (1.0 + gamma)
            p_high = gamma / (1.0 + gamma)
            # worst case toward alternative = p_high
            if alternative == "less":
                p_upper_case, p_lower_case = p_low, p_high
            else:
                p_upper_case, p_lower_case = p_high, p_low
            uppers[i] = _wilcoxon_bound(ranks_abs, signs, p_upper_case, alternative)
            lowers[i] = _wilcoxon_bound(ranks_abs, signs, p_lower_case, alternative)
    else:  # sign test
        nonzero = diffs[diffs != 0]
        n_nz = int(nonzero.size)
        if alternative == "less":
            statistic = float(np.sum(nonzero < 0))
            target = statistic
        else:
            statistic = float(np.sum(nonzero > 0))
            target = statistic
        for i, gamma in enumerate(gamma_grid):
            p_low = 1.0 / (1.0 + gamma)
            p_high = gamma / (1.0 + gamma)
            if alternative == "less":
                p_upper_case, p_lower_case = p_low, p_high
            else:
                p_upper_case, p_lower_case = p_high, p_low
            p_hi = _sign_test_bound(int(target), n_nz, p_upper_case)
            p_lo = _sign_test_bound(int(target), n_nz, p_lower_case)
            if alternative == "two-sided":
                p_hi = min(1.0, 2 * p_hi)
                p_lo = min(1.0, 2 * p_lo)
            uppers[i] = p_hi
            lowers[i] = p_lo

    # ---- Critical Gamma --------------------------------------------
    above = np.where(uppers > alpha)[0]
    if above.size == 0:
        gamma_crit = float("inf")
    else:
        gamma_crit = float(gamma_grid[above[0]])

    detail = pd.DataFrame(
        {
            "Gamma": gamma_grid,
            "p_lower": lowers,
            "p_upper": uppers,
            "reject_upper": uppers <= alpha,
        }
    )

    return RosenbaumResult(
        gamma_grid=gamma_grid,
        pvalue_lower=lowers,
        pvalue_upper=uppers,
        gamma_critical=gamma_crit,
        method=method,
        alpha=alpha,
        n_pairs=n_pairs,
        statistic=float(statistic),
        alternative=alternative,
        detail=detail,
    )


# Convenience alias matching the Γ naming in Rosenbaum's textbook.
rosenbaum_gamma = rosenbaum_bounds


__all__ = ["rosenbaum_bounds", "rosenbaum_gamma", "RosenbaumResult"]
