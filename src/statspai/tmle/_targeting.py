"""Helpers of :mod:`statspai.tmle.tmle` that do not depend on the estimand.

* :func:`parameter_table` -- the treatment-specific means and the
  contrasts built from them (difference, ratio, odds ratio), each with the
  standard error its influence function implies.
* :func:`cluster_se` -- the standard error of a mean from its influence
  function, with or without clusters.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def parameter_table(
    ic1: np.ndarray,
    ic0: np.ndarray,
    ey1: float,
    ey0: float,
    se_of: Callable[[np.ndarray], float],
    alpha: float,
    nonnegative_outcome: bool,
    unit_interval_outcome: bool,
) -> pd.DataFrame:
    """Treatment-specific means and the contrasts built from them.

    ``ic1`` / ``ic0`` are the influence functions of the two targeted means.
    The difference is linear in them. The ratio and the odds ratio are
    handled on the log scale, where the delta method gives the influence
    functions ``ic1 / EY1 - ic0 / EY0`` and
    ``ic1 / (EY1 (1 - EY1)) - ic0 / (EY0 (1 - EY0))``; their intervals are
    built there and exponentiated, so they are not symmetric around the
    estimate, and the reported ``se`` is the natural-scale delta-method one
    (``estimate * se_log``). Rows whose parameter is undefined for the data
    are omitted: the ratio needs a non-negative outcome with positive means
    (a ratio of means of a variable that changes sign has no stable
    interpretation), the odds ratio an outcome in the unit interval.
    """
    z = float(sp_stats.norm.ppf(1 - alpha / 2))
    rows = []

    def linear(name: str, est: float, ic: np.ndarray) -> None:
        se = se_of(ic)
        pv = float(2 * sp_stats.norm.sf(abs(est / se))) if se > 0 else np.nan
        rows.append((name, est, se, est - z * se, est + z * se, pv, np.nan, np.nan))

    def on_log_scale(name: str, est: float, ic_log: np.ndarray) -> None:
        se_log = se_of(ic_log)
        log_est = float(np.log(est))
        pv = (
            float(2 * sp_stats.norm.sf(abs(log_est / se_log))) if se_log > 0 else np.nan
        )
        rows.append(
            (
                name,
                est,
                est * se_log,
                float(np.exp(log_est - z * se_log)),
                float(np.exp(log_est + z * se_log)),
                pv,
                log_est,
                se_log,
            )
        )

    linear("EY1", ey1, ic1)
    linear("EY0", ey0, ic0)
    linear("ATE", ey1 - ey0, ic1 - ic0)
    if nonnegative_outcome and ey1 > 0 and ey0 > 0:
        on_log_scale("RR", ey1 / ey0, ic1 / ey1 - ic0 / ey0)
        if unit_interval_outcome and ey1 < 1 and ey0 < 1:
            on_log_scale(
                "OR",
                (ey1 / (1 - ey1)) / (ey0 / (1 - ey0)),
                ic1 / (ey1 * (1 - ey1)) - ic0 / (ey0 * (1 - ey0)),
            )
    return pd.DataFrame(
        rows,
        columns=[
            "parameter",
            "estimate",
            "se",
            "ci_lower",
            "ci_upper",
            "pvalue",
            "log_estimate",
            "se_log",
        ],
    )


def cluster_se(cl_codes: Optional[np.ndarray], n: int) -> Callable[[np.ndarray], float]:
    """Standard error of a mean from its influence function.

    ``sd(ic) / sqrt(n)`` for independent rows; with clusters the centred
    cluster totals with the ``G / (G - 1)`` factor.
    """

    def se_of(ic: np.ndarray) -> float:
        if cl_codes is None:
            return float(np.std(ic, ddof=1) / np.sqrt(n))
        S = np.bincount(cl_codes, weights=ic)
        G = S.shape[0]
        return float(np.sqrt(G / (G - 1) * np.sum((S - S.mean()) ** 2)) / n)

    return se_of
