"""Tests for comparing distributions: ranks, medians, variances and means.

* :func:`ranksum` -- Wilcoxon rank-sum (Mann-Whitney) test of two groups
  (Stata ``ranksum``, R ``wilcox.test``).
* :func:`signrank` -- Wilcoxon matched-pairs signed-rank test (Stata
  ``signrank``).
* :func:`kwallis` -- Kruskal-Wallis test of several groups (Stata
  ``kwallis``, R ``kruskal.test``).
* :func:`spearman` -- Spearman's rank correlation (Stata ``spearman``).
* :func:`ktau` -- Kendall's tau-a and tau-b with the score test (Stata
  ``ktau``).
* :func:`ksmirnov` -- two-sample Kolmogorov-Smirnov test (Stata
  ``ksmirnov``).
* :func:`median_test` -- the median test (Stata ``median``).
* :func:`robvar` -- Levene's test of equal variances and the two
  Brown-Forsythe variants (Stata ``robvar``).
* :func:`oneway` -- one-way analysis of variance with Bartlett's test and
  pairwise comparisons (Stata ``oneway``).

Every p-value is the large-sample one the Stata command prints; the
statistics are corrected for ties as stated in each function. The rows
with a missing value in a variable the test uses are left out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = [
    "ClassicTestResult",
    "ranksum",
    "signrank",
    "kwallis",
    "spearman",
    "ktau",
    "ksmirnov",
    "median_test",
    "robvar",
    "oneway",
]


@dataclass
class ClassicTestResult(ResultProtocolMixin):
    """Result of one of the tests of this module.

    Attributes
    ----------
    method : str
        The test.
    statistic : float
        The test statistic named in ``statistic_name``.
    pvalue : float
        Its p-value (two-sided where the test has sides).
    df : float or tuple, optional
        Degrees of freedom of the reference distribution.
    n_obs : int
        Rows used.
    estimates : dict
        The other numbers the test reports (rank sums, variances, the
        correlation, ...), by name.
    table : pandas.DataFrame, optional
        The table by group, where the test has one.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6, 7, 8],
    ...                    "g": [0, 0, 0, 0, 1, 1, 1, 1]})
    >>> res = sp.ranksum(df, "y", by="g")
    >>> type(res).__name__, res.statistic_name, res.n_obs
    ('ClassicTestResult', 'z', 8)
    >>> sorted(res.estimates)[:2]
    ['adjusted_variance', 'porder']
    """

    method: str
    statistic: float
    pvalue: float
    statistic_name: str = "z"
    df: Any = None
    n_obs: int = 0
    estimates: Dict[str, Any] = field(default_factory=dict)
    table: Optional[pd.DataFrame] = None

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def summary(self) -> str:
        lines = [self.method, "-" * len(self.method)]
        if self.table is not None:
            lines += [self.table.to_string(), ""]
        for key, value in self.estimates.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                lines += [f"{key}:", value.to_string()]
            else:
                lines.append(f"{key:>24s} = {value:.6g}" if isinstance(value, float)
                             else f"{key:>24s} = {value}")  # fmt: skip
        shown = self.statistic_name
        if self.df is not None:
            shown += f"({self.df})" if not isinstance(self.df, tuple) else str(self.df)
        lines.append(f"{shown:>24s} = {self.statistic:.4f}")
        lines.append(f"{'p-value':>24s} = {self.pvalue:.4f}")
        lines.append(f"{'N':>24s} = {self.n_obs}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ------------------------------------------------------------------ helpers
def _frame(data: Any, who: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            f"{who}: data must be a DataFrame.",
            recovery_hint=f"sp.{who}(df, ...)",
        )
    return data


def _column(data: pd.DataFrame, name: Any, who: str, numeric: bool = True) -> pd.Series:
    if name not in data.columns:
        raise MethodIncompatibility(
            f"{who}: column {name!r} is not in data.",
            recovery_hint="Check the column name.",
            diagnostics={"columns": [str(c) for c in data.columns][:20]},
        )
    col = data[name]
    if numeric and not pd.api.types.is_numeric_dtype(col):
        raise MethodIncompatibility(
            f"{who}: column {name!r} is not numeric.",
            recovery_hint="Encode it as a number first.",
        )
    return col


def _groups(
    data: pd.DataFrame, y: str, by: str, who: str, exactly: Optional[int] = None
) -> Tuple[List[Any], List[np.ndarray]]:
    """The values of ``y`` by level of ``by``, levels in sorted order."""
    data = _frame(data, who)
    col = _column(data, y, who).astype(float)
    grp = _column(data, by, who, numeric=False)
    keep = col.notna() & grp.notna()
    col, grp = col[keep], grp[keep]
    levels = sorted(pd.unique(grp))
    if exactly is not None and len(levels) != exactly:
        raise MethodIncompatibility(
            f"{who}: {by!r} must have exactly {exactly} groups, it has "
            f"{len(levels)}.",
            recovery_hint="Filter the data to the two groups to compare.",
        )
    if len(levels) < 2:
        raise DataInsufficient(
            f"{who}: {by!r} has fewer than two groups with data.",
            recovery_hint="Check the grouping column.",
        )
    return levels, [col[grp == lv].to_numpy(dtype=float) for lv in levels]


def _percentile(x: np.ndarray, p: float) -> float:
    """Stata's percentile: with ``h = n p / 100``, the mean of the h-th and
    (h + 1)-th order statistics when h is an integer, else the next one."""
    x = np.sort(x)
    n = x.size
    h = n * p / 100.0
    k = int(np.floor(h + 1e-12))
    if abs(h - k) < 1e-12:
        return float((x[max(k, 1) - 1] + x[min(k + 1, n) - 1]) / 2.0)
    return float(x[min(k + 1, n) - 1])


def _with_totals(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["total"] = out.sum(axis=1)
    out.loc["total"] = out.sum(axis=0)
    return out


def _tie_sizes(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.unique(values, return_counts=True)[1], dtype=float)


# ------------------------------------------------------------------ ranksum
def ranksum(data: pd.DataFrame, y: str, by: str) -> ClassicTestResult:
    """Wilcoxon rank-sum (Mann-Whitney) test that two groups have the same
    distribution.

    The statistic is the sum of the ranks of the first group (the lower
    value of ``by``) in the pooled sample, ties ranked by their mean rank.
    Its variance under the null is ``n1 n2 (N + 1) / 12`` less the
    correction for ties ``n1 n2 sum(t^3 - t) / (12 N (N - 1))``, and
    ``z = (T - n1 (N + 1) / 2) / sd`` is referred to the standard normal
    without a continuity correction (Stata ``ranksum``; R's
    ``wilcox.test(correct = FALSE)`` gives the same p-value).

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        The variable to compare.
    by : str
        Grouping column with exactly two values.

    Returns
    -------
    ClassicTestResult
        ``statistic`` is z. ``estimates`` holds the rank sums, the
        variances and ``porder``, the estimate of P(y1 > y2) + P(y1 = y2)/2.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"y": [1.0, 3, 5, 7, 2, 4, 6, 9, 11],
    ...                    "g": [0, 0, 0, 0, 1, 1, 1, 1, 1]})
    >>> res = sp.ranksum(df, "y", by="g")
    >>> round(res.statistic, 3)
    -0.98
    """
    levels, (a, b) = _groups(data, y, by, "ranksum", exactly=2)
    n1, n2 = a.size, b.size
    n = n1 + n2
    pooled = np.concatenate([a, b])
    ranks = stats.rankdata(pooled)
    t1 = float(ranks[:n1].sum())
    expected = n1 * (n + 1) / 2.0
    unadjusted = n1 * n2 * (n + 1) / 12.0
    ties = _tie_sizes(pooled)
    adjustment = -n1 * n2 * float(np.sum(ties**3 - ties)) / (12.0 * n * (n - 1))
    variance = unadjusted + adjustment
    if variance <= 0:
        raise DataInsufficient(
            "ranksum: every value is the same, so the ranks carry no " "information.",
            recovery_hint="Check the variable.",
        )
    z = (t1 - expected) / np.sqrt(variance)
    table = pd.DataFrame(
        {
            "obs": [n1, n2, n],
            "rank_sum": [t1, float(ranks[n1:].sum()), n * (n + 1) / 2.0],
            "expected": [expected, n2 * (n + 1) / 2.0, n * (n + 1) / 2.0],
        },
        index=[levels[0], levels[1], "combined"],
    )
    return ClassicTestResult(
        method="Two-sample Wilcoxon rank-sum (Mann-Whitney) test",
        statistic=float(z),
        pvalue=float(2 * stats.norm.sf(abs(z))),
        statistic_name="z",
        n_obs=n,
        estimates={
            "unadjusted_variance": unadjusted,
            "tie_adjustment": adjustment,
            "adjusted_variance": variance,
            "porder": float((t1 - n1 * (n1 + 1) / 2.0) / (n1 * n2)),
        },
        table=table,
    )


# ----------------------------------------------------------------- signrank
def signrank(
    data: pd.DataFrame, y: str, other: Optional[str] = None, value: float = 0.0
) -> ClassicTestResult:
    """Wilcoxon matched-pairs signed-rank test.

    The differences ``y - other`` (or ``y - value``) are ranked by absolute
    size, ties and zeros included, and the ranks of the positive ones are
    summed. Under the null of a distribution symmetric about zero the sum
    has mean ``n (n + 1) / 4`` less the share of the zeros and variance
    ``n (n + 1) (2n + 1) / 24``, less ``sum(t^3 - t) / 48`` for ties among
    the nonzero differences and ``n0 (n0 + 1) (2 n0 + 1) / 24`` for the
    ``n0`` zeros (Stata ``signrank``).

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
    other : str, optional
        The paired variable. Without it ``y`` is compared with ``value``.
    value : float, default 0

    Returns
    -------
    ClassicTestResult
        ``statistic`` is z.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"before": [10.0, 12, 9, 14, 11, 13],
    ...                    "after": [12.0, 15, 9, 13, 15, 18]})
    >>> res = sp.signrank(df, "after", other="before")
    >>> res.n_obs
    6
    """
    data = _frame(data, "signrank")
    a = _column(data, y, "signrank").astype(float)
    b = (
        _column(data, other, "signrank").astype(float)
        if other is not None
        else pd.Series(float(value), index=a.index)
    )
    d = (a - b).dropna().to_numpy(dtype=float)
    n = d.size
    if n < 2:
        raise DataInsufficient(
            "signrank: fewer than two pairs.", recovery_hint="Check the columns."
        )
    ranks = stats.rankdata(np.abs(d))
    positive = float(ranks[d > 0].sum())
    negative = float(ranks[d < 0].sum())
    zero = float(ranks[d == 0].sum())
    n0 = int(np.sum(d == 0))
    expected = (n * (n + 1) / 2.0 - zero) / 2.0
    unadjusted = n * (n + 1) * (2 * n + 1) / 24.0
    ties = _tie_sizes(np.abs(d[d != 0]))
    tie_adjustment = -float(np.sum(ties**3 - ties)) / 48.0
    zero_adjustment = -n0 * (n0 + 1) * (2 * n0 + 1) / 24.0
    variance = unadjusted + tie_adjustment + zero_adjustment
    if variance <= 0:
        raise DataInsufficient(
            "signrank: every difference is zero.", recovery_hint="Check the columns."
        )
    z = (positive - expected) / np.sqrt(variance)
    table = pd.DataFrame(
        {
            "obs": [int(np.sum(d > 0)), int(np.sum(d < 0)), n0, n],
            "rank_sum": [positive, negative, zero, n * (n + 1) / 2.0],
            "expected": [expected, expected, zero, n * (n + 1) / 2.0],
        },
        index=["positive", "negative", "zero", "all"],
    )
    return ClassicTestResult(
        method="Wilcoxon signed-rank test",
        statistic=float(z),
        pvalue=float(2 * stats.norm.sf(abs(z))),
        statistic_name="z",
        n_obs=n,
        estimates={
            "unadjusted_variance": unadjusted,
            "tie_adjustment": tie_adjustment,
            "zero_adjustment": zero_adjustment,
            "adjusted_variance": variance,
        },
        table=table,
    )


# ------------------------------------------------------------------ kwallis
def kwallis(data: pd.DataFrame, y: str, by: str) -> ClassicTestResult:
    """Kruskal-Wallis test that several groups have the same distribution.

    ``H = 12 / (N (N + 1)) sum(R_j^2 / n_j) - 3 (N + 1)`` with ``R_j`` the
    rank sum of group j, referred to chi-squared with ``k - 1`` degrees of
    freedom. The statistic corrected for ties divides H by
    ``1 - sum(t^3 - t) / (N^3 - N)``; it is the one R's ``kruskal.test``
    reports and the second one Stata's ``kwallis`` prints.

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
    by : str
        Grouping column.

    Returns
    -------
    ClassicTestResult
        ``statistic`` / ``pvalue`` are the tie-corrected ones;
        ``estimates['chi2_unadjusted']`` and ``['p_unadjusted']`` the
        others.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6, 7, 8, 9],
    ...                    "g": [1, 1, 1, 2, 2, 2, 3, 3, 3]})
    >>> round(sp.kwallis(df, "y", by="g").statistic, 3)
    7.2
    """
    levels, parts = _groups(data, y, by, "kwallis")
    pooled = np.concatenate(parts)
    n = pooled.size
    ranks = stats.rankdata(pooled)
    sums, sizes, start = [], [], 0
    for part in parts:
        sums.append(float(ranks[start : start + part.size].sum()))
        sizes.append(part.size)
        start += part.size
    h = 12.0 / (n * (n + 1)) * sum(r**2 / m for r, m in zip(sums, sizes)) - 3 * (n + 1)
    ties = _tie_sizes(pooled)
    correction = 1.0 - float(np.sum(ties**3 - ties)) / (n**3 - n)
    if correction <= 0:
        raise DataInsufficient(
            "kwallis: every value is the same.", recovery_hint="Check the variable."
        )
    h_ties = h / correction
    df = len(levels) - 1
    return ClassicTestResult(
        method="Kruskal-Wallis equality-of-populations rank test",
        statistic=float(h_ties),
        pvalue=float(stats.chi2.sf(h_ties, df)),
        statistic_name="chi2",
        df=df,
        n_obs=n,
        estimates={
            "chi2_unadjusted": float(h),
            "p_unadjusted": float(stats.chi2.sf(h, df)),
        },
        table=pd.DataFrame({"obs": sizes, "rank_sum": sums}, index=levels),
    )


# ----------------------------------------------------------------- spearman
def _pair(
    data: pd.DataFrame, x: str, y: str, who: str
) -> Tuple[np.ndarray, np.ndarray]:
    data = _frame(data, who)
    a = _column(data, x, who).astype(float)
    b = _column(data, y, who).astype(float)
    keep = a.notna() & b.notna()
    if int(keep.sum()) < 3:
        raise DataInsufficient(
            f"{who}: fewer than three complete pairs.",
            recovery_hint="Check the columns for missing values.",
        )
    return a[keep].to_numpy(), b[keep].to_numpy()


def spearman(data: pd.DataFrame, x: str, y: str) -> ClassicTestResult:
    """Spearman's rank correlation and the test that it is zero.

    The correlation of the ranks (ties get their mean rank). The p-value
    refers ``t = rho sqrt((n - 2) / (1 - rho^2))`` to Student's t with
    ``n - 2`` degrees of freedom, as Stata's ``spearman`` and R's
    ``cor.test(method = "spearman", exact = FALSE)`` do.

    Parameters
    ----------
    data : pandas.DataFrame
    x, y : str

    Returns
    -------
    ClassicTestResult
        ``statistic`` is rho itself; ``estimates['t']`` the t statistic.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5], "y": [2.0, 1, 4, 3, 5]})
    >>> round(sp.spearman(df, "x", "y").statistic, 2)
    0.8
    """
    a, b = _pair(data, x, y, "spearman")
    n = a.size
    rho = float(np.corrcoef(stats.rankdata(a), stats.rankdata(b))[0, 1])
    if abs(rho) >= 1:
        t, p = float("inf"), 0.0
    else:
        t = rho * np.sqrt((n - 2) / (1 - rho**2))
        p = float(2 * stats.t.sf(abs(t), n - 2))
    return ClassicTestResult(
        method="Spearman's rank correlation",
        statistic=rho,
        pvalue=p,
        statistic_name="rho",
        df=n - 2,
        n_obs=n,
        estimates={"t": float(t)},
    )


# --------------------------------------------------------------------- ktau
def ktau(data: pd.DataFrame, x: str, y: str) -> ClassicTestResult:
    """Kendall's rank correlations tau-a and tau-b and the score test.

    The score ``S`` is the number of concordant pairs less the number of
    discordant ones; ``tau-a = S / (n (n - 1) / 2)`` and tau-b divides S by
    the geometric mean of the numbers of pairs not tied on x and not tied
    on y. The variance of S under independence is corrected for ties
    (Kendall 1975), and ``z = (|S| - 1) / sd`` carries a continuity
    correction, as in Stata's ``ktau``.

    Parameters
    ----------
    data : pandas.DataFrame
    x, y : str

    Returns
    -------
    ClassicTestResult
        ``statistic`` is tau-b; ``estimates`` holds ``tau_a``, ``score``,
        ``se_score`` and ``z``.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5], "y": [2.0, 1, 4, 3, 5]})
    >>> round(sp.ktau(df, "x", "y").statistic, 2)
    0.6
    """
    a, b = _pair(data, x, y, "ktau")
    n = a.size
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    # concordant minus discordant pairs from the cumulated table
    below = np.cumsum(np.cumsum(table, axis=0), axis=1)
    conc = np.zeros_like(table)
    conc[1:, 1:] = below[:-1, :-1]
    left = np.cumsum(np.cumsum(table[:, ::-1], axis=0), axis=1)[:, ::-1]
    disc = np.zeros_like(table)
    disc[1:, :-1] = left[:-1, 1:]
    score = float((table * conc).sum() - (table * disc).sum())
    pairs = n * (n - 1) / 2.0
    t = table.sum(axis=1)
    u = table.sum(axis=0)
    tied_x = float(np.sum(t * (t - 1)) / 2.0)
    tied_y = float(np.sum(u * (u - 1)) / 2.0)
    denominator = np.sqrt((pairs - tied_x) * (pairs - tied_y))
    if denominator <= 0:
        raise DataInsufficient(
            "ktau: one of the variables does not vary.",
            recovery_hint="Check the columns.",
        )
    variance = (
        n * (n - 1) * (2 * n + 5)
        - np.sum(t * (t - 1) * (2 * t + 5))
        - np.sum(u * (u - 1) * (2 * u + 5))
    ) / 18.0
    if n > 2:
        variance += (np.sum(t * (t - 1) * (t - 2)) * np.sum(u * (u - 1) * (u - 2))) / (
            9.0 * n * (n - 1) * (n - 2)
        )
    variance += np.sum(t * (t - 1)) * np.sum(u * (u - 1)) / (2.0 * n * (n - 1))
    se = float(np.sqrt(variance))
    z = float(np.sign(score) * (abs(score) - 1) / se) if score != 0 else 0.0
    return ClassicTestResult(
        method="Kendall's rank correlation",
        statistic=float(score / denominator),
        pvalue=float(min(1.0, 2 * stats.norm.sf(abs(z)))),
        statistic_name="tau_b",
        n_obs=n,
        estimates={
            "tau_a": float(score / pairs),
            "score": score,
            "se_score": se,
            "z": z,
        },
    )


# ----------------------------------------------------------------- ksmirnov
def ksmirnov(data: pd.DataFrame, y: str, by: str) -> ClassicTestResult:
    """Two-sample Kolmogorov-Smirnov test of equal distributions.

    ``D`` is the largest difference between the two empirical distribution
    functions. The one-sided p-values are ``exp(-2 m n / (m + n) D^2)`` and
    the combined one is the limiting Kolmogorov distribution evaluated at
    ``D sqrt(m n / (m + n))``; both are large-sample approximations, as
    printed by Stata's ``ksmirnov`` (which also offers an exact p-value
    for small samples, not computed here).

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
    by : str
        Grouping column with exactly two values.

    Returns
    -------
    ClassicTestResult
        ``statistic`` is the combined D. ``table`` has one row per
        direction: the first group's distribution function above the
        second's (its values tend to be smaller) and below it.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6, 7, 8],
    ...                    "g": [0, 0, 0, 0, 1, 1, 1, 1]})
    >>> sp.ksmirnov(df, "y", by="g").statistic
    1.0
    """
    levels, (a, b) = _groups(data, y, by, "ksmirnov", exactly=2)
    m, n = a.size, b.size
    grid = np.unique(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / m
    fb = np.searchsorted(np.sort(b), grid, side="right") / n
    d_plus = float(max(np.max(fa - fb), 0.0))
    d_minus = float(min(np.min(fa - fb), 0.0))
    d = max(d_plus, -d_minus)
    scale = m * n / (m + n)
    table = pd.DataFrame(
        {
            "D": [d_plus, d_minus, d],
            "pvalue": [
                float(np.exp(-2 * scale * d_plus**2)),
                float(np.exp(-2 * scale * d_minus**2)),
                float(stats.kstwobign.sf(d * np.sqrt(scale))),
            ],
        },
        index=[levels[0], levels[1], "combined"],
    )
    return ClassicTestResult(
        method="Two-sample Kolmogorov-Smirnov test",
        statistic=d,
        pvalue=float(table.loc["combined", "pvalue"]),
        statistic_name="D",
        n_obs=m + n,
        estimates={"unique_values": int(grid.size)},
        table=table,
    )


# -------------------------------------------------------------- median test
def median_test(data: pd.DataFrame, y: str, by: str) -> ClassicTestResult:
    """Median test: do the groups have the same share of values above the
    overall median?

    Each observation is classed as greater than the median of the pooled
    sample or not, and Pearson's chi-squared test of independence is
    applied to the resulting table. With two groups the continuity
    corrected statistic is reported as well (Stata ``median``).

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
    by : str

    Returns
    -------
    ClassicTestResult
        ``table`` is the 2 x k table of counts.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6, 7, 8],
    ...                    "g": [0, 0, 0, 0, 1, 1, 1, 1]})
    >>> sp.median_test(df, "y", by="g").statistic
    8.0
    """
    levels, parts = _groups(data, y, by, "median_test")
    pooled = np.concatenate(parts)
    median = float(np.median(pooled))
    counts = np.array(
        [
            [int(np.sum(p <= median)) for p in parts],
            [int(np.sum(p > median)) for p in parts],
        ]
    )
    if (counts.sum(axis=1) == 0).any():
        raise DataInsufficient(
            "median_test: no value lies above the median.",
            recovery_hint="The variable is (almost) constant.",
        )
    chi2, p, df, _ = stats.chi2_contingency(counts, correction=False)
    estimates: Dict[str, Any] = {"median": median}
    if len(levels) == 2:
        corrected = stats.chi2_contingency(counts, correction=True)
        estimates["chi2_corrected"] = float(corrected[0])
        estimates["p_corrected"] = float(corrected[1])
    return ClassicTestResult(
        method="Median test",
        statistic=float(chi2),
        pvalue=float(p),
        statistic_name="chi2",
        df=int(df),
        n_obs=int(pooled.size),
        estimates=estimates,
        table=_with_totals(
            pd.DataFrame(counts, index=["not above", "above"], columns=levels)
        ),
    )


# ------------------------------------------------------------------- robvar
def _levene(parts: List[np.ndarray], centre: Any) -> float:
    z = [np.abs(p - centre(p)) for p in parts]
    n = sum(p.size for p in parts)
    k = len(parts)
    grand = np.concatenate(z).mean()
    between = sum(p.size * (zz.mean() - grand) ** 2 for p, zz in zip(parts, z))
    within = sum(float(np.sum((zz - zz.mean()) ** 2)) for zz in z)
    return float((n - k) / (k - 1) * between / within)


def robvar(data: pd.DataFrame, y: str, by: str) -> ClassicTestResult:
    """Robust tests of equal variances across groups.

    ``W0`` is Levene's F statistic on the absolute deviations from the
    group means; ``W50`` replaces the mean by the median and ``W10`` by
    the 10 percent trimmed mean, here the mean of the values between the
    group's 10th and 90th percentiles (Brown and Forsythe). Each is referred to
    F with ``k - 1`` and ``N - k`` degrees of freedom (Stata ``robvar``).

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
    by : str

    Returns
    -------
    ClassicTestResult
        ``statistic`` / ``pvalue`` are those of W0; ``estimates`` holds
        ``W50``, ``W10`` and their p-values.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"y": [1.0, 2, 3, 4, 10, 20, 30, 40],
    ...                    "g": [0, 0, 0, 0, 1, 1, 1, 1]})
    >>> round(sp.robvar(df, "y", by="g").statistic, 3)
    9.624
    """
    levels, parts = _groups(data, y, by, "robvar")
    if min(p.size for p in parts) < 2:
        raise DataInsufficient(
            "robvar: a group has fewer than two observations.",
            recovery_hint="Drop or merge the small group.",
        )
    n = sum(p.size for p in parts)
    k = len(parts)

    def trimmed(p: np.ndarray) -> float:
        # the mean of the values from the 10th to the 90th percentile of
        # the group, the bounds included (the definition in robvar.ado)
        lo, hi = _percentile(p, 10), _percentile(p, 90)
        return float(p[(p >= lo) & (p <= hi)].mean())

    w0 = _levene(parts, np.mean)
    w50 = _levene(parts, np.median)
    w10 = _levene(parts, trimmed)
    df = (k - 1, n - k)
    table = pd.DataFrame(
        {
            "mean": [p.mean() for p in parts],
            "sd": [p.std(ddof=1) for p in parts],
            "freq": [p.size for p in parts],
        },
        index=levels,
    )
    pooled = np.concatenate(parts)
    table.loc["total"] = [pooled.mean(), pooled.std(ddof=1), n]
    return ClassicTestResult(
        method="Robust tests of equal variances (Levene, Brown-Forsythe)",
        statistic=w0,
        pvalue=float(stats.f.sf(w0, *df)),
        statistic_name="W0",
        df=df,
        n_obs=n,
        estimates={
            "W50": w50,
            "p_W50": float(stats.f.sf(w50, *df)),
            "W10": w10,
            "p_W10": float(stats.f.sf(w10, *df)),
        },
        table=table,
    )


# ------------------------------------------------------------------- oneway
def oneway(
    data: pd.DataFrame, y: str, by: str, compare: Optional[str] = None
) -> ClassicTestResult:
    """One-way analysis of variance.

    The F test that the group means are equal, Bartlett's test that the
    group variances are equal and, on request, every pairwise difference
    of means with a p-value adjusted for the number of comparisons (Stata
    ``oneway``, R ``aov`` / ``bartlett.test``).

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
    by : str
    compare : {None, 'bonferroni', 'sidak', 'scheffe'}
        Adjustment of the pairwise comparisons. Each difference is the
        mean of the row group less the mean of the column group, tested
        with the pooled within-group variance.

    Returns
    -------
    ClassicTestResult
        ``statistic`` is F with ``df = (k - 1, N - k)``. ``estimates``
        holds the sums of squares, mean squares, ``bartlett_chi2`` /
        ``bartlett_p`` and, with ``compare``, the tables
        ``differences`` and ``pvalues``.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6, 7, 8, 9],
    ...                    "g": [1, 1, 1, 2, 2, 2, 3, 3, 3]})
    >>> sp.oneway(df, "y", by="g").statistic
    27.0
    """
    if compare not in (None, "bonferroni", "sidak", "scheffe"):
        raise MethodIncompatibility(
            f"oneway: unknown compare {compare!r}.",
            recovery_hint="Use 'bonferroni', 'sidak' or 'scheffe'.",
        )
    levels, parts = _groups(data, y, by, "oneway")
    k = len(parts)
    n = sum(p.size for p in parts)
    if n <= k:
        raise DataInsufficient(
            "oneway: no within-group degrees of freedom.",
            recovery_hint="Each group needs more than one observation overall.",
        )
    pooled = np.concatenate(parts)
    means = np.array([p.mean() for p in parts])
    sizes = np.array([p.size for p in parts], dtype=float)
    ss_between = float(np.sum(sizes * (means - pooled.mean()) ** 2))
    ss_within = float(sum(np.sum((p - p.mean()) ** 2) for p in parts))
    ms_between, ms_within = ss_between / (k - 1), ss_within / (n - k)
    f = ms_between / ms_within if ms_within > 0 else float("inf")
    estimates: Dict[str, Any] = {
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_between + ss_within,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "ms_total": (ss_between + ss_within) / (n - 1),
        "df_between": k - 1,
        "df_within": n - k,
        "r2": ss_between / (ss_between + ss_within),
        "rmse": float(np.sqrt(ms_within)),
    }
    variances = np.array([p.var(ddof=1) if p.size > 1 else np.nan for p in parts])
    if np.all(sizes > 1) and np.all(variances > 0):
        dof = sizes - 1
        pooled_var = float(np.sum(dof * variances) / np.sum(dof))
        m = np.sum(dof) * np.log(pooled_var) - np.sum(dof * np.log(variances))
        c = 1 + (np.sum(1 / dof) - 1 / np.sum(dof)) / (3 * (k - 1))
        estimates["bartlett_chi2"] = float(m / c)
        estimates["bartlett_p"] = float(stats.chi2.sf(m / c, k - 1))
    if compare is not None:
        diff = pd.DataFrame(np.nan, index=levels, columns=levels)
        pval = pd.DataFrame(np.nan, index=levels, columns=levels)
        pairs = k * (k - 1) / 2
        for i in range(k):
            for j in range(i):
                d = means[i] - means[j]
                t = d / np.sqrt(ms_within * (1 / sizes[i] + 1 / sizes[j]))
                p = 2 * stats.t.sf(abs(t), n - k)
                if compare == "bonferroni":
                    p = min(1.0, p * pairs)
                elif compare == "sidak":
                    p = 1 - (1 - p) ** pairs
                else:
                    p = float(stats.f.sf(t**2 / (k - 1), k - 1, n - k))
                diff.iloc[i, j], pval.iloc[i, j] = d, p
        estimates["differences"] = diff
        estimates["pvalues"] = pval
    table = pd.DataFrame(
        {
            "mean": means,
            "sd": np.sqrt(variances),
            "freq": sizes.astype(int),
        },
        index=levels,
    )
    table.loc["total"] = [pooled.mean(), pooled.std(ddof=1), n]
    return ClassicTestResult(
        method="One-way analysis of variance",
        statistic=float(f),
        pvalue=float(stats.f.sf(f, k - 1, n - k)),
        statistic_name="F",
        df=(k - 1, n - k),
        n_obs=n,
        estimates=estimates,
        table=table,
    )
