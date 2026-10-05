"""Standardised effect sizes of a two-group comparison, and the correlation test.

``esize``
    Cohen's d, Hedges's g, Glass's Delta and the point-biserial correlation
    of a difference between two group means, with confidence intervals from
    the noncentral t distribution (Stata ``esize twosample``; R
    ``effsize::cohen.d(noncentral = TRUE)``).
``cor_test``
    Pearson's correlation, or the partial correlation given covariates, with
    its t test and Fisher-z confidence interval (R ``cor.test``,
    ``ggm::pcor.test``; the significance level of Stata ``pwcorr, sig``).

A p-value says whether a difference is distinguishable from zero; the effect
size says how large it is on a scale that does not depend on the sample
size, which is the number a power calculation (:func:`statspai.power_ttest`)
takes as input.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from ..exceptions import DataInsufficient, MethodIncompatibility
from .rank_tests import ClassicTestResult, _column, _frame, _groups

__all__ = ["esize", "cor_test"]


def _nct_limits(t: float, df: float, alpha: float) -> Tuple[float, float]:
    """Noncentrality parameters that put ``t`` at the two tail quantiles.

    The interval for the noncentrality is the set of values under which the
    observed statistic is not in either ``alpha / 2`` tail; an effect size is the noncentrality times a known
    constant, so its interval follows.
    """

    def tail(ncp: float, target: float) -> float:
        value = float(stats.nct.cdf(t, df, ncp))
        if math.isnan(value):
            # far in a tail the series does not converge; the limit is known
            value = 0.0 if ncp > t else 1.0
        return value - target

    # The interval is a few standard errors of t wide; widen until it brackets.
    step = (float(stats.norm.isf(alpha / 2.0)) + 2.0) * math.sqrt(
        1.0 + t * t / (2.0 * df)
    )
    lo_a, hi_b = t - step, t + step
    while tail(lo_a, 1.0 - alpha / 2.0) < 0:
        lo_a -= step
    while tail(hi_b, alpha / 2.0) > 0:
        hi_b += step
    lower = optimize.brentq(tail, lo_a, t, args=(1.0 - alpha / 2.0,), xtol=1e-12)
    upper = optimize.brentq(tail, t, hi_b, args=(alpha / 2.0,), xtol=1e-12)
    return float(lower), float(upper)


def _hedges_factor(m: float) -> float:
    """``Gamma(m/2) / (sqrt(m/2) Gamma((m-1)/2))``: the bias of d, exactly."""
    return float(
        math.exp(special.gammaln(m / 2.0) - special.gammaln((m - 1.0) / 2.0))
        / math.sqrt(m / 2.0)
    )


def esize(
    data: pd.DataFrame,
    y: str,
    by: str,
    *,
    unequal: bool = False,
    alpha: float = 0.05,
) -> ClassicTestResult:
    """Effect sizes for the difference between two group means.

    Reports, with confidence intervals,

    * **Cohen's d**: the difference in means over the pooled standard
      deviation;
    * **Hedges's g**: d times the exact small-sample correction
      ``Gamma(m/2) / (sqrt(m/2) Gamma((m-1)/2))``, ``m = n1 + n2 - 2``,
      which removes the upward bias of d;
    * **Glass's Delta**: the difference over the standard deviation of one
      group alone (``delta1`` the first group's, ``delta2`` the second's),
      for when a treatment changes the spread as well as the mean and one
      group is the natural yardstick;
    * the **point-biserial correlation** between the outcome and group
      membership.

    The difference is the mean of the group with the smaller value of
    ``by`` minus the mean of the other, as in :func:`statspai.ttest`.

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        Outcome.
    by : str
        Column with exactly two groups.
    unequal : bool, default False
        Use Satterthwaite's degrees of freedom for the intervals of d, g
        and the point-biserial correlation (Stata's ``unequal``). The
        estimates of d, g and Delta do not change, nor do Delta's intervals.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    ClassicTestResult
        ``statistic`` is Cohen's d. ``table`` has one row per effect size
        with ``estimate``, ``ci_lower`` and ``ci_upper``; ``estimates``
        carries the group sizes, means and standard deviations, the t
        statistic and its degrees of freedom.

    Notes
    -----
    The intervals invert the noncentral t distribution of the t statistic,
    so they are exact under normality and are not symmetric about the
    estimate. They agree with Stata's ``esize twosample`` to rounding. R's
    ``effsize::cohen.d`` uses a normal approximation unless
    ``noncentral = TRUE``, and corrects g by ``1 - 3 / (4m - 1)``, which
    differs from the exact factor in the seventh digit at ``m = 400``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"g": np.repeat(["a", "b"], 100)})
    >>> df["y"] = rng.normal(size=200) + 0.5 * (df.g == "b")
    >>> res = sp.esize(df, "y", by="g")
    >>> list(res.table.index)
    ['cohens_d', 'hedges_g', 'glass_delta1', 'glass_delta2', 'point_biserial_r']
    >>> bool(res.table.loc["cohens_d", "ci_upper"] < 0)   # a minus b
    True
    """
    if not (0.0 < alpha < 1.0):
        raise MethodIncompatibility("esize: alpha must be between 0 and 1.")
    levels, groups = _groups(data, y, by, "esize", exactly=2)
    a, b = groups
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        raise DataInsufficient(
            "esize: each group needs at least two observations.",
            recovery_hint="Check the grouping column for near-empty groups.",
        )
    m1, m2 = float(a.mean()), float(b.mean())
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    df_pooled = n1 + n2 - 2.0
    s_pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df_pooled)
    if not (s_pooled > 0 and v1 > 0 and v2 > 0):
        raise DataInsufficient(
            "esize: the outcome does not vary within a group, so a "
            "standardised difference is undefined.",
        )
    diff = m1 - m2
    scale = math.sqrt(1.0 / n1 + 1.0 / n2)
    d = diff / s_pooled
    g = d * _hedges_factor(df_pooled)

    if unequal:
        se = math.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )
        t = diff / se
    else:
        df = df_pooled
        t = d / scale
    lo, hi = _nct_limits(t, df, alpha)

    rows: List[Tuple[str, float, float, float]] = [
        ("cohens_d", d, lo * scale, hi * scale),
        (
            ("hedges_g", g, lo * scale * g / d, hi * scale * g / d)
            if d != 0
            else ("hedges_g", g, lo * scale, hi * scale)
        ),
    ]
    for label, var_j, n_j in (("glass_delta1", v1, n1), ("glass_delta2", v2, n2)):
        delta = diff / math.sqrt(var_j)
        # one group's variance, so that group's degrees of freedom
        l_j, h_j = _nct_limits(delta / scale, n_j - 1.0, alpha)
        rows.append((label, delta, l_j * scale, h_j * scale))
    r_pb = t / math.sqrt(t * t + df)
    rows.append(
        (
            "point_biserial_r",
            r_pb,
            lo / math.sqrt(lo * lo + df),
            hi / math.sqrt(hi * hi + df),
        )
    )
    table = pd.DataFrame(
        [r[1:] for r in rows],
        index=[r[0] for r in rows],
        columns=["estimate", "ci_lower", "ci_upper"],
    )
    return ClassicTestResult(
        method="Effect size of a difference in means"
        + (" (unequal variances)" if unequal else ""),
        statistic=float(d),
        pvalue=float(2 * stats.t.sf(abs(t), df)),
        statistic_name="d",
        df=float(df),
        n_obs=int(n1 + n2),
        estimates={
            "groups": f"{levels[0]} - {levels[1]}",
            "n1": int(n1),
            "n2": int(n2),
            "mean1": m1,
            "mean2": m2,
            "sd1": math.sqrt(v1),
            "sd2": math.sqrt(v2),
            "sd_pooled": s_pooled,
            "t": float(t),
            "level": 1.0 - alpha,
        },
        table=table,
    )


def cor_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    covariates: Optional[Sequence[str]] = None,
    *,
    alpha: float = 0.05,
) -> ClassicTestResult:
    """Pearson correlation, or partial correlation, with test and interval.

    Without ``covariates`` this is the correlation of ``x`` and ``y``; with
    them, the correlation of what is left of each after a linear regression
    on the covariates. ``t = r sqrt(df / (1 - r^2))`` with
    ``df = n - 2 - k`` (``k`` covariates) is Student's t under no
    correlation and joint normality, and the interval is Fisher's:
    ``tanh(atanh(r) -/+ z / sqrt(n - 3 - k))``.

    A partial correlation of zero is what a causal graph implies for two
    nodes that a set of others d-separates, under linearity: conditioning on
    a mediator should remove a correlation, conditioning on a collider
    should create one. :meth:`statspai.dag.DAG.test_implications` runs this
    test for every implication of a graph.

    Parameters
    ----------
    data : pandas.DataFrame
    x, y : str
        Numeric columns.
    covariates : sequence of str, optional
        Numeric columns to partial out. Rows with a missing value in any
        column used are dropped.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    ClassicTestResult
        ``statistic`` is the correlation; ``pvalue`` two-sided;
        ``estimates`` has ``t``, ``ci_lower``, ``ci_upper`` and
        ``n_covariates``.

    Notes
    -----
    Matches R's ``cor.test`` (estimate, t, df, p-value and interval) and,
    with covariates, the t test of ``ggm::pcor.test``. For rank
    correlations see :func:`statspai.spearman` and :func:`statspai.ktau`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=500)
    >>> m = x + rng.normal(size=500)           # x -> m -> y
    >>> y = m + rng.normal(size=500)
    >>> df = pd.DataFrame({"x": x, "m": m, "y": y})
    >>> bool(sp.cor_test(df, "x", "y").pvalue < 0.001)
    True
    >>> bool(sp.cor_test(df, "x", "y", covariates=["m"]).pvalue > 0.05)
    True
    """
    who = "cor_test"
    data = _frame(data, who)
    if not (0.0 < alpha < 1.0):
        raise MethodIncompatibility("cor_test: alpha must be between 0 and 1.")
    if isinstance(covariates, str):
        covariates = [covariates]
    covs = list(covariates or [])
    if x == y or x in covs or y in covs:
        raise MethodIncompatibility(
            "cor_test: x, y and the covariates must be different columns."
        )
    cols: List[Any] = [x, y, *covs]
    frame = pd.concat(
        [_column(data, c, who).astype(float) for c in cols], axis=1
    ).dropna()
    n, k = len(frame), len(covs)
    if n < k + 4:
        raise DataInsufficient(
            f"cor_test: {n} complete rows are too few for {k} covariates "
            "(the interval needs n - k - 3 > 0)."
        )
    a = frame[x].to_numpy()
    b = frame[y].to_numpy()
    if k:
        Z = np.column_stack([np.ones(n), frame[covs].to_numpy()])
        a = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
        b = b - Z @ np.linalg.lstsq(Z, b, rcond=None)[0]
    else:
        a = a - a.mean()
        b = b - b.mean()
    saa, sbb = float(a @ a), float(b @ b)
    if not (saa > 0 and sbb > 0):
        raise DataInsufficient(
            "cor_test: a variable is constant"
            + (" given the covariates" if k else "")
            + ", so the correlation is undefined."
        )
    r = float(a @ b) / math.sqrt(saa * sbb)
    r = max(-1.0, min(1.0, r))
    df = n - 2 - k
    if abs(r) >= 1.0:
        t, p = math.copysign(math.inf, r), 0.0
        lo = hi = r
    else:
        t = r * math.sqrt(df / (1.0 - r * r))
        p = float(2 * stats.t.sf(abs(t), df))
        half = float(stats.norm.isf(alpha / 2.0)) / math.sqrt(n - 3 - k)
        lo = math.tanh(math.atanh(r) - half)
        hi = math.tanh(math.atanh(r) + half)
    return ClassicTestResult(
        method=("Partial correlation" if k else "Pearson's correlation")
        + f" of {x} and {y}"
        + (f" given {', '.join(map(str, covs))}" if k else ""),
        statistic=r,
        pvalue=p,
        statistic_name="r",
        df=int(df),
        n_obs=int(n),
        estimates={
            "t": float(t),
            "ci_lower": float(lo),
            "ci_upper": float(hi),
            "level": 1.0 - alpha,
            "n_covariates": int(k),
        },
    )
