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
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin


@dataclass
class RosenbaumResult(ResultProtocolMixin):
    """Result container for :func:`rosenbaum_bounds`.

    Examples
    --------
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> treated = rng.normal(1.0, 1.0, 80)
    >>> control = rng.normal(0.0, 1.0, 80)
    >>> res = sp.rosenbaum_bounds(treated, control)
    >>> type(res).__name__
    'RosenbaumResult'
    >>> res.n_pairs
    80
    """

    _citation_keys = ("rosenbaum2002observational",)

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

    def to_dict(self) -> dict:
        """JSON-safe dict of every field (agent-native serialization)."""
        from .._result_serialize import result_to_dict

        return result_to_dict(self)

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


def _wilcoxon_z(ranks: np.ndarray, positive: np.ndarray, p: float) -> float:
    """Rosenbaum's standardized signed-rank deviate at assignment probability ``p``.

    ``T = sum(ranks[positive])`` against the moments of a sum of independent
    ``Bernoulli(p) * rank`` terms: ``E = p * sum(ranks)``,
    ``Var = p (1 - p) * sum(ranks**2)``. No continuity correction -- none of
    ``DOS2::senWilcox`` (Rosenbaum's own implementation), ``rbounds::psens``
    or Stata ``rbounds`` applies one.
    """
    ranks = np.asarray(ranks, dtype=float)
    T_obs = float(np.sum(ranks[positive]))
    mean = p * float(ranks.sum())
    var = p * (1.0 - p) * float(np.sum(ranks**2))
    if var <= 0:
        return 0.0 if T_obs == mean else float(np.sign(T_obs - mean) * np.inf)
    return (T_obs - mean) / float(np.sqrt(var))


def _wilcoxon_ranks(diffs: np.ndarray, zero_method: str) -> np.ndarray:
    """Mid-ranks of ``|diffs|`` with zero differences carrying rank 0.

    ``zero_method="pratt"`` ranks the zeros together with the non-zero
    differences and then drops them (``rank(|d|) * (|d| > 0)``), as
    ``DOS2::senWilcox`` and Stata ``rbounds`` do; ``"wilcox"`` discards the
    zeros before ranking, as ``rbounds::psens`` does. The two agree when no
    difference is exactly zero.
    """
    ad = np.abs(diffs)
    if zero_method == "pratt":
        return stats.rankdata(ad, method="average") * (ad > 0)
    ranks = np.zeros_like(ad)
    nz = ad > 0
    ranks[nz] = stats.rankdata(ad[nz], method="average")
    return ranks


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
    zero_method: str = "pratt",
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
    zero_method : {"pratt", "wilcox"}, default "pratt"
        Treatment of pairs with a zero difference in the Wilcoxon bound.
        ``"pratt"`` ranks them with the other pairs and gives them no weight
        (Rosenbaum's ``DOS2::senWilcox``, Stata ``rbounds``); ``"wilcox"``
        drops them before ranking (``rbounds::psens``). Identical when no
        difference is zero. Ignored by the sign test, which always drops
        zeros.

    Returns
    -------
    RosenbaumResult
        ``gamma_critical`` is the smallest Gamma in the grid at which the
        upper-bound p-value exceeds ``alpha``. It is ``inf`` if the study
        is insensitive across the grid, and ``1.0`` if already sensitive.

    Examples
    --------
    Two parallel arrays of matched-pair outcomes (Wilcoxon bound):

    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> treated = rng.normal(1.0, 1.0, 80)
    >>> control = rng.normal(0.0, 1.0, 80)
    >>> res = sp.rosenbaum_bounds(treated, control)
    >>> res.gamma_critical >= 1.0
    True

    A binary / robust sign-test bound on an insensitive example:

    >>> import numpy as np
    >>> t = np.array([2.0, 3.0, 4.0, 5.0])
    >>> c = np.array([1.0, 1.0, 1.0, 1.0])
    >>> bounds = sp.rosenbaum_bounds(t, c, method="sign", gamma_grid=[1.0, 2.0])
    >>> bool(np.all(bounds.pvalue_upper >= bounds.pvalue_lower))
    True
    """
    if method not in {"wilcoxon", "sign"}:
        raise ValueError("method must be 'wilcoxon' or 'sign'")
    if zero_method not in {"pratt", "wilcox"}:
        raise ValueError("zero_method must be 'pratt' or 'wilcox'")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")

    # ---- Build paired differences -----------------------------------
    if data is not None:
        if not all([y, treat, pair_id]):
            raise ValueError("When `data` is given, y/treat/pair_id are required")
        _miss = [c for c in (y, treat, pair_id) if c not in data.columns]
        if _miss:
            raise ValueError(
                f"rosenbaum_bounds: columns {_miss} not found in data "
                f"(available: {list(data.columns)})."
            )
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
        gamma_grid_arr = np.round(np.arange(1.0, 3.01, 0.1), 3)
    else:
        gamma_grid_arr = np.asarray(gamma_grid, dtype=float)
    if np.any(gamma_grid_arr < 1.0):
        raise ValueError("gamma_grid must be >= 1")

    # ---- Compute bounds per Gamma ----------------------------------
    lowers = np.empty_like(gamma_grid_arr)
    uppers = np.empty_like(gamma_grid_arr)

    # Bounding p-values (Rosenbaum 2002, ch. 4). Under hidden bias of at most
    # Gamma the chance that the treated unit of a pair is the one with the
    # larger response lies in [1/(1+Gamma), Gamma/(1+Gamma)]; the upper bound
    # on the one-sided p-value evaluates the statistic's null distribution at
    # the end of that interval least favourable to the alternative, the lower
    # bound at the other end. "less" is the "greater" computation applied to
    # the negated differences, and the two-sided bound is twice the smaller
    # one-sided bound, capped at 1 (DOS2::senWilcox).
    def _one_sided(d: np.ndarray, p: float) -> tuple:
        if method == "wilcoxon":
            ranks = _wilcoxon_ranks(d, zero_method)
            z = _wilcoxon_z(ranks, d > 0, p)
            return float(stats.norm.sf(z)), float(np.sum(ranks[d > 0]))
        nonzero = d[d != 0]
        t_plus = int(np.sum(nonzero > 0))
        return _sign_test_bound(t_plus, int(nonzero.size), p), float(t_plus)

    for i, gamma in enumerate(gamma_grid_arr):
        p_high = gamma / (1.0 + gamma)
        p_low = 1.0 / (1.0 + gamma)
        if alternative == "greater":
            uppers[i], statistic = _one_sided(diffs, p_high)
            lowers[i], _ = _one_sided(diffs, p_low)
        elif alternative == "less":
            uppers[i], statistic = _one_sided(-diffs, p_high)
            lowers[i], _ = _one_sided(-diffs, p_low)
        else:
            up_g, statistic = _one_sided(diffs, p_high)
            up_l, _ = _one_sided(-diffs, p_high)
            lo_g, _ = _one_sided(diffs, p_low)
            lo_l, _ = _one_sided(-diffs, p_low)
            uppers[i] = min(1.0, 2.0 * min(up_g, up_l))
            lowers[i] = min(1.0, 2.0 * min(lo_g, lo_l))

    # ---- Critical Gamma --------------------------------------------
    above = np.where(uppers > alpha)[0]
    if above.size == 0:
        gamma_crit = float("inf")
    else:
        gamma_crit = float(gamma_grid_arr[above[0]])

    detail = pd.DataFrame(
        {
            "Gamma": gamma_grid_arr,
            "p_lower": lowers,
            "p_upper": uppers,
            "reject_upper": uppers <= alpha,
        }
    )

    return RosenbaumResult(
        gamma_grid=gamma_grid_arr,
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
