"""Critical-value conventions for the regression-based event studies.

``sp.event_study``, ``sp.sun_abraham`` and ``sp.did_2x2`` report the same
standard errors as Stata and R, but each program turns a standard error
into a p-value and a confidence interval with its own reference
distribution. Measured on the castle-doctrine and divorce panels
(2026-10-21, Stata 18 ``reghdfe`` / ``eventstudyinteract`` / ``regress``
and R ``fixest`` 0.14.0, run as black boxes):

* cluster-robust SEs: Student t with ``G - 1`` degrees of freedom
  (``G`` = number of clusters), in Stata and in fixest alike;
* heteroskedasticity-robust or classical SEs: t with ``N - K``, where
  ``K`` counts the slopes plus the absorbed fixed-effect levels.

The StatsPAI defaults are unchanged: the two event studies use the
normal distribution and ``did_2x2`` uses t with ``N - K`` also under
clustering. ``parity='stata'`` / ``parity='r'`` switch to the reference
program's distribution. The convention used is recorded in
``model_info['conventions']`` (keys ``parity``, ``crit_dist``,
``crit_df``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from scipy import stats

from ..exceptions import MethodIncompatibility

PARITY_VALUES = ("r", "stata")


def check_parity(parity: Any, function: str) -> Optional[str]:
    """Validate a ``parity=`` argument; returns ``None``, ``'r'`` or ``'stata'``."""
    if parity is None:
        return None
    if isinstance(parity, str) and parity.strip().lower() in PARITY_VALUES:
        return parity.strip().lower()
    raise MethodIncompatibility(
        f"{function}: parity={parity!r} is not recognised.",
        recovery_hint=(
            "Pass parity='stata' or parity='r' to use that program's "
            "critical values, or leave parity=None for the StatsPAI default."
        ),
        diagnostics={"parity": repr(parity), "allowed": list(PARITY_VALUES)},
    )


def crit_spec(
    parity: Optional[str],
    *,
    clustered: bool,
    n_clusters: Optional[int],
    df_resid: Optional[float],
    default: Tuple[str, Optional[float]] = ("normal", None),
) -> Tuple[str, Optional[float]]:
    """``(crit_dist, crit_df)`` for a regression event study.

    ``default`` is what the estimator reports with ``parity=None``.
    Under ``parity='stata'`` or ``'r'`` the reference distribution is
    t(G - 1) for cluster-robust SEs and t(N - K) otherwise; Stata and
    fixest agree on both.
    """
    if parity is None:
        return default
    if clustered:
        if n_clusters is None or n_clusters < 2:
            raise MethodIncompatibility(
                f"parity={parity!r} needs at least two clusters for t(G-1) "
                f"critical values; got G={n_clusters}.",
                recovery_hint="Cluster on a variable with two or more levels.",
            )
        return "t", float(n_clusters - 1)
    if df_resid is None or df_resid < 1:
        raise MethodIncompatibility(
            f"parity={parity!r}: the residual degrees of freedom are "
            f"{df_resid}; t(N-K) critical values are undefined.",
            recovery_hint="Use parity=None (normal critical values).",
        )
    return "t", float(df_resid)


def crit_value(alpha: float, dist: str, df: Optional[float]) -> float:
    """Two-sided critical value at level ``alpha``."""
    if dist == "normal":
        return float(stats.norm.ppf(1 - alpha / 2))
    return float(stats.t.ppf(1 - alpha / 2, df))


def two_sided_p(z: float, dist: str, df: Optional[float]) -> float:
    """Two-sided p-value of the statistic ``z``."""
    if dist == "normal":
        return float(2 * stats.norm.sf(abs(z)))
    return float(2 * stats.t.sf(abs(z), df))


def conventions(
    parity: Optional[str], dist: str, df: Optional[float], **extra: Any
) -> Dict[str, Any]:
    """The ``model_info['conventions']`` record (shared key names)."""
    out: Dict[str, Any] = {
        "parity": parity,
        "crit_dist": dist,
        "crit_df": None if df is None else int(round(df)),
    }
    out.update(extra)
    return out
