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

In this release the StatsPAI defaults are unchanged: the two event
studies use the normal distribution and ``did_2x2`` uses t with ``N - K``
also under clustering. ``parity='stata'`` / ``parity='r'`` switch to the
reference program's distribution, and so does ``crit_dist='t'``, which
names the convention without naming a program. The default becomes
``crit_dist='t'`` in 1.41 (owner decision 2026-10-21); until then a call
whose intervals or p-values would change warns (``FutureWarning``), and
``crit_dist='normal'`` (event studies) / ``crit_dist='residual'``
(``did_2x2``) keeps today's numbers without the warning. The
convention used is recorded in ``model_info['conventions']`` (keys
``parity``, ``crit_dist``, ``crit_df``).

Callaway-Sant'Anna is not affected: R ``did`` and Stata ``csdid`` both
use the normal distribution.
"""

from __future__ import annotations

import os
import sys
import warnings
from types import FrameType
from typing import Any, Dict, Optional, Tuple

from scipy import stats

from ..exceptions import MethodIncompatibility

PARITY_VALUES = ("r", "stata")

_STATSPAI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_crit_dist(
    crit_dist: Any,
    parity: Optional[str],
    *,
    function: str,
    legacy: str,
) -> Optional[str]:
    """Validate ``crit_dist=`` against the function's values and ``parity``.

    ``legacy`` is the value that reproduces the function's pre-1.41
    default (``'normal'`` for the event studies, ``'residual'`` for
    ``did_2x2``). Returns ``None``, ``'t'`` or ``legacy``.
    """
    if crit_dist is None:
        return None
    allowed = ("t", legacy)
    value = crit_dist.strip().lower() if isinstance(crit_dist, str) else crit_dist
    if value not in allowed:
        raise MethodIncompatibility(
            f"{function}: crit_dist={crit_dist!r} is not recognised.",
            recovery_hint=(
                f"Pass crit_dist='t' (t(G-1) with clustered SEs, t(N-K) "
                f"otherwise, as Stata and fixest) or crit_dist={legacy!r} "
                "(the pre-1.41 default)."
            ),
            diagnostics={"crit_dist": repr(crit_dist), "allowed": list(allowed)},
        )
    if parity is not None and value != "t":
        raise MethodIncompatibility(
            f"{function}: crit_dist={value!r} contradicts parity={parity!r}; "
            f"parity={parity!r} uses that program's t critical values "
            "(crit_dist='t').",
            recovery_hint=(
                "Drop crit_dist= (parity sets it), or drop parity= and keep "
                f"crit_dist={value!r}."
            ),
            diagnostics={"parity": parity, "crit_dist": value},
        )
    return value


def _external_stacklevel() -> int:
    """``stacklevel`` that attributes a warning to the first caller frame
    outside ``statspai`` (the user's call, not a decorator wrapper)."""
    level = 2  # the caller of this helper's caller issues warnings.warn
    frame: Optional[FrameType] = sys._getframe(2)
    prefix = _STATSPAI_DIR + os.sep
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        prefix
    ):
        level += 1
        frame = frame.f_back
    return level


def warn_default_change(
    function: str,
    *,
    old: Tuple[str, Optional[float]],
    new: Optional[Tuple[str, Optional[float]]],
    legacy: str,
) -> None:
    """``FutureWarning`` for an implicit default that changes in 1.41.

    ``old`` is the ``(crit_dist, crit_df)`` this call reports, ``new`` the
    one ``crit_dist='t'`` would give (``None`` when it would raise, e.g.
    a single cluster). Silent when the two coincide.
    """
    if new is not None and new == old:
        return

    def _name(spec: Tuple[str, Optional[float]]) -> str:
        dist, df = spec
        return "normal" if dist == "normal" else f"t({int(round(df or 0))})"

    after = (
        f"the {_name(new)} distribution here"
        if new is not None
        else "an error here (t critical values are undefined for this fit)"
    )
    warnings.warn(
        f"{function}: p-values and confidence intervals use the "
        f"{_name(old)} distribution; from 1.41 the default is "
        f"crit_dist='t', which gives {after}: t(G-1) with cluster-robust "
        "SEs and t(N-K) otherwise, as Stata reghdfe / regress / "
        "eventstudyinteract and R fixest report. Estimates and standard "
        "errors do not change. Pass crit_dist='t' (or parity='stata' / "
        f"'r') to adopt it now, or crit_dist={legacy!r} to keep these "
        "numbers. See MIGRATION.md.",
        FutureWarning,
        stacklevel=_external_stacklevel(),
    )


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
    crit_dist: Optional[str] = None,
) -> Tuple[str, Optional[float]]:
    """``(crit_dist, crit_df)`` for a regression event study.

    ``default`` is what the estimator reports with ``parity=None`` and
    ``crit_dist`` unset or set to the legacy value. Under
    ``parity='stata'`` / ``'r'`` or ``crit_dist='t'`` the reference
    distribution is t(G - 1) for cluster-robust SEs and t(N - K)
    otherwise; Stata and fixest agree on both.
    """
    if parity is None and crit_dist != "t":
        return default
    what = f"parity={parity!r}" if parity is not None else "crit_dist='t'"
    if clustered:
        if n_clusters is None or n_clusters < 2:
            raise MethodIncompatibility(
                f"{what} needs at least two clusters for t(G-1) "
                f"critical values; got G={n_clusters}.",
                recovery_hint="Cluster on a variable with two or more levels.",
            )
        return "t", float(n_clusters - 1)
    if df_resid is None or df_resid < 1:
        raise MethodIncompatibility(
            f"{what}: the residual degrees of freedom are "
            f"{df_resid}; t(N-K) critical values are undefined.",
            recovery_hint="Use parity=None (normal critical values).",
        )
    return "t", float(df_resid)


def t_spec_or_none(
    *,
    clustered: bool,
    n_clusters: Optional[int],
    df_resid: Optional[float],
) -> Optional[Tuple[str, Optional[float]]]:
    """What ``crit_dist='t'`` would give, or ``None`` where it raises."""
    try:
        return crit_spec(
            None,
            clustered=clustered,
            n_clusters=n_clusters,
            df_resid=df_resid,
            crit_dist="t",
        )
    except MethodIncompatibility:
        return None


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
