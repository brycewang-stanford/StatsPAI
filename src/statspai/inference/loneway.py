"""One-way analysis of variance with the intraclass correlation.

``sp.loneway`` is the companion of :func:`sp.oneway` for many groups: the
same between / within decomposition, read as a random-effects model. It
reports how much of the variance of the outcome is shared within a group
(the intraclass correlation), the standard deviations of the group effect
and of the within-group error, and the reliability of a group mean. The
intraclass correlation is what a design effect or a Moulton factor is
computed from, so it is the number to look at before deciding how to
cluster standard errors.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility
from .rank_tests import ClassicTestResult, _groups

__all__ = ["loneway"]


def loneway(
    data: pd.DataFrame,
    y: str,
    by: str,
    *,
    alpha: float = 0.05,
    exact: bool = False,
) -> ClassicTestResult:
    """One-way analysis of variance and the intraclass correlation.

    The analysis-of-variance estimator of the intraclass correlation in
    the one-way random-effects model ``y_ij = mu + a_i + e_ij``, with its
    asymptotic standard error (Stata ``loneway``).

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        The response.
    by : str
        The grouping column (the cluster).
    alpha : float, default 0.05
        One minus the level of the confidence interval.
    exact : bool, default False
        Report the exact confidence interval from the F distribution in
        place of the normal one. It needs groups of equal size.

    Returns
    -------
    ClassicTestResult
        ``statistic`` is the F statistic of equal group means with ``df =
        (k - 1, N - k)``. ``estimates`` holds

        * ``icc`` -- the intraclass correlation ``(F - 1) / (F - 1 + g)``,
          where ``g = (N - sum n_i^2 / N) / (k - 1)`` is the average group
          size that the unbalanced design calls for; a negative estimate is
          reported as 0 with ``icc_truncated = True``;
        * ``icc_se``, ``icc_ci`` -- the asymptotic standard error and the
          interval ``icc -/+ z * se`` (or the exact interval);
        * ``sd_between``, ``sd_within`` -- standard deviations of the group
          effect and of the error within groups;
        * ``reliability`` -- of a group mean of ``g`` observations,
          ``g * icc / (1 + (g - 1) * icc)``, and ``avg_group_size`` = ``g``;
        * the sums of squares and mean squares of :func:`sp.oneway`, and
          ``r2``.

    Notes
    -----
    The normal interval stops at zero below and is left open above: with
    few groups and a correlation near one its upper end exceeds 1, as
    Stata's does. When the between-group mean square is smaller than the
    within-group one the correlation is reported as 0 and ``sd_between``
    is missing.

    With groups of ``g`` observations and intraclass correlation ``rho``
    of both the outcome and a regressor fixed within groups, a variance
    that ignores the grouping is too small by the factor ``1 + (g - 1)
    rho`` (the design effect).

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"y": [1.0, 2, 3, 11, 12, 13, 21, 22, 23],
    ...                    "g": [1, 1, 1, 2, 2, 2, 3, 3, 3]})
    >>> res = sp.loneway(df, "y", by="g")
    >>> round(res.estimates["icc"], 4), round(res.estimates["sd_between"], 4)
    (0.9901, 9.9833)

    References
    ----------
    [@donner1986review]
    """
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"loneway: alpha must lie strictly between 0 and 1, got {alpha!r}.",
            recovery_hint="alpha=0.05 gives a 95% interval.",
        )
    levels, parts = _groups(data, y, by, "loneway")
    k = len(parts)
    sizes = np.array([p.size for p in parts], dtype=float)
    n = float(sizes.sum())
    if k < 2 or n <= k:
        raise DataInsufficient(
            "loneway: at least two groups and one within-group degree of "
            "freedom are needed.",
            recovery_hint="Each group needs more than one observation overall.",
            diagnostics={"groups": k, "n_obs": int(n)},
        )
    pooled = np.concatenate(parts)
    means = np.array([p.mean() for p in parts])
    ss_between = float(np.sum(sizes * (means - pooled.mean()) ** 2))
    ss_within = float(sum(np.sum((p - p.mean()) ** 2) for p in parts))
    ms_between, ms_within = ss_between / (k - 1), ss_within / (n - k)
    f = ms_between / ms_within if ms_within > 0 else float("inf")

    sum2, sum3 = float(np.sum(sizes**2)), float(np.sum(sizes**3))
    g = (n - sum2 / n) / (k - 1)
    if np.isfinite(f):
        rho = (f - 1.0) / (f - 1.0 + g)
    else:
        rho = 1.0
    truncated = rho < 0
    if truncated:
        rho = 0.0
    # Donner (1986), the large-sample variance for groups of unequal size
    spread = sum2 - 2.0 * sum3 / n + sum2**2 / n**2
    variance = (
        2.0
        * (1.0 - rho) ** 2
        / g**2
        * (
            (1.0 + rho * (g - 1.0)) ** 2 / (n - k)
            + (
                (k - 1.0) * (1.0 - rho) * (1.0 + rho * (2.0 * g - 1.0))
                + rho**2 * spread
            )
            / (k - 1.0) ** 2
        )
    )
    se = float(np.sqrt(variance))
    if exact:
        if np.ptp(sizes) != 0:
            raise MethodIncompatibility(
                "loneway: the exact interval is defined for groups of equal size.",
                recovery_hint="Drop exact=True for the asymptotic interval.",
                diagnostics={
                    "min_size": int(sizes.min()),
                    "max_size": int(sizes.max()),
                },
            )
        f_low = f / stats.f.ppf(1 - alpha / 2, k - 1, n - k)
        f_high = f * stats.f.ppf(1 - alpha / 2, n - k, k - 1)
        ci = ((f_low - 1) / (f_low + g - 1), (f_high - 1) / (f_high + g - 1))
    else:
        z = float(stats.norm.ppf(1 - alpha / 2))
        # the lower end stops at zero, the upper end is left as it falls
        ci = (max(rho - z * se, 0.0), rho + z * se)
    between = (ms_between - ms_within) / g
    sd_between = float(np.sqrt(between)) if between >= 0 else float("nan")
    estimates: Dict[str, Any] = {
        "icc": float(rho),
        "icc_se": se,
        "icc_ci": (float(ci[0]), float(ci[1])),
        "icc_truncated": bool(truncated),
        "sd_between": sd_between,
        "sd_within": float(np.sqrt(ms_within)),
        "reliability": float(g * rho / (1.0 + (g - 1.0) * rho)),
        "avg_group_size": float(g),
        "n_groups": k,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_between + ss_within,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "df_between": k - 1,
        "df_within": int(n - k),
        "r2": ss_between / (ss_between + ss_within),
    }
    return ClassicTestResult(
        method="One-way analysis of variance (intraclass correlation)",
        statistic=float(f),
        pvalue=float(stats.f.sf(f, k - 1, n - k)),
        statistic_name="F",
        df=(k - 1, int(n - k)),
        n_obs=int(n),
        estimates=estimates,
    )
