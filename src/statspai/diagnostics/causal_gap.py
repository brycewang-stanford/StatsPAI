"""Critical causal gap: an assumption-free sensitivity summary.

An estimate targets a statistical estimand ``psi``. The causal quantity of
interest ``psi*`` equals it only under identifying assumptions; their
difference ``psi - psi*`` is the *causal gap*. Without modelling where a
gap would come from (an unmeasured confounder, a positivity violation,
selection), one can still report how large it would have to be to change
the conclusion: the *critical causal gap* is the smallest gap at which the
confidence interval for ``psi*`` reaches a stated value, the null or a
threshold of practical importance.

It complements the tools that parametrise the violation, such as
:func:`sp.evalue`, :func:`sp.sensemakr` and :func:`sp.confounder_tip`. It
says nothing about which violation is plausible; that judgement is left to
the reader.

References
----------
[@schuler2022introduction]
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility

_SCALES = ("difference", "ratio")


def _from_result(result: Any) -> Tuple[float, Optional[float], Tuple[float, float]]:
    try:
        est = float(result.estimate)
        lo, hi = (float(v) for v in result.ci)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "causal_gap: the first argument must be a number or a result "
            "with scalar .estimate and .ci.",
            recovery_hint="Pass the estimate and se= (or ci=) directly.",
        ) from exc
    se = getattr(result, "se", None)
    return est, (None if se is None else float(se)), (lo, hi)


def causal_gap(
    estimate: Any,
    se: Optional[float] = None,
    *,
    ci: Optional[Tuple[float, float]] = None,
    null: Any = None,
    alpha: float = 0.05,
    scale: str = "difference",
    n_grid: int = 101,
) -> pd.DataFrame:
    """Smallest causal gap that would change the conclusion.

    Parameters
    ----------
    estimate : float or result object
        Point estimate of the statistical estimand, or a fitted result
        with scalar ``.estimate`` and ``.ci`` (its interval is then used
        as reported, at the level it was computed at).
    se : float, optional
        Standard error. On ``scale='ratio'`` it is the standard error of
        the *log* of the estimate. Not needed when ``ci`` is given.
    ci : (float, float), optional
        Confidence interval for the statistical estimand. Takes
        precedence over ``se``; an interval built on another scale (the
        log scale of a ratio, say) is used as it stands.
    null : float or array-like, optional
        Value(s) the interval has to stay clear of: the null hypothesis,
        or the smallest effect of practical interest. Defaults to 0 on the
        difference scale and 1 on the ratio scale. Several values give
        several rows.
    alpha : float, default 0.05
        Level of the interval built from ``se``. Ignored when the interval
        comes from ``ci`` or from a result.
    scale : {'difference', 'ratio'}, default 'difference'
        Whether the gap shifts the estimate (``psi* = psi - gap``) or
        divides it (``psi* = psi / gap``; risk, odds and hazard ratios).
    n_grid : int, default 101
        Number of hypothetical gaps in the curve stored in
        ``.attrs['curve']``.

    Returns
    -------
    pandas.DataFrame
        One row per ``null`` with

        - ``estimate``, ``ci_lower``, ``ci_upper``: the inputs;
        - ``conclusion_holds``: whether the interval excludes ``null`` to
          begin with;
        - ``critical_gap``: the gap ``psi - psi*`` at which the interval
          for ``psi*`` first touches ``null``; signed on the difference
          scale, a factor on the ratio scale, and 0 (or 1) when the
          interval already covers ``null``;
        - ``gap_to_point``: the gap that would move the point estimate
          itself to ``null``;
        - ``share_of_estimate``: ``critical_gap`` relative to the distance
          between the estimate and ``null`` (on the log scale for ratios):
          the fraction of the apparent effect that would have to be
          spurious.

        ``.attrs['curve']`` holds, for the first ``null``, a grid of
        hypothetical gaps with the implied estimate and interval of
        ``psi*`` (columns ``gap``, ``estimate``, ``ci_lower``,
        ``ci_upper``, ``conclusion_holds``), ready to plot.

    Notes
    -----
    The interval is taken as given: a gap moves its centre and leaves its
    width alone, because the width reflects sampling error in the
    statistical estimand and has nothing to do with identification.

    Examples
    --------
    An estimate of 1 with a 95% interval of half-width 0.5 stops being
    statistically significant once the causal gap reaches 0.5:

    >>> import statspai as sp
    >>> tbl = sp.causal_gap(1.0, ci=(0.5, 1.5))
    >>> float(tbl['critical_gap'].iloc[0])
    0.5

    If only effects above 0.3 matter, a gap of 0.2 is enough:

    >>> round(float(sp.causal_gap(1.0, ci=(0.5, 1.5), null=0.3)
    ...             ['critical_gap'].iloc[0]), 10)
    0.2

    A risk ratio of 1.4 (1.26 to 1.56) survives any bias factor below
    1.26:

    >>> tbl = sp.causal_gap(1.4, ci=(1.26, 1.56), scale='ratio')
    >>> float(tbl['critical_gap'].iloc[0])
    1.26

    References
    ----------
    [@schuler2022introduction]
    """
    scale = str(scale).lower()
    if scale not in _SCALES:
        raise MethodIncompatibility(
            f"causal_gap: scale must be one of {_SCALES}, got {scale!r}"
        )
    if not (0 < alpha < 1):
        raise MethodIncompatibility("causal_gap: alpha must lie in (0, 1)")
    if int(n_grid) < 2:
        raise MethodIncompatibility("causal_gap: n_grid must be at least 2")
    ratio = scale == "ratio"

    if isinstance(estimate, (int, float, np.integer, np.floating)):
        est = float(estimate)
    else:
        est, se_res, ci_res = _from_result(estimate)
        if ci is None:
            ci = ci_res
        if se is None and not ratio:
            se = se_res
    if not np.isfinite(est):
        raise MethodIncompatibility("causal_gap: the estimate must be finite")
    if ratio and est <= 0:
        raise MethodIncompatibility(
            "causal_gap: scale='ratio' needs a positive estimate"
        )

    if ci is not None:
        lo, hi = (float(v) for v in ci)
    elif se is not None:
        if not (np.isfinite(se) and se > 0):
            raise MethodIncompatibility("causal_gap: se must be positive and finite")
        z = float(stats.norm.ppf(1 - alpha / 2))
        if ratio:
            lo, hi = est * np.exp(-z * se), est * np.exp(z * se)
        else:
            lo, hi = est - z * se, est + z * se
    else:
        raise MethodIncompatibility(
            "causal_gap: give se= or ci=.",
            recovery_hint="On scale='ratio', se is that of the log estimate.",
        )
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= est <= hi):
        raise MethodIncompatibility(
            "causal_gap: the interval must be finite and contain the estimate",
            diagnostics={"estimate": est, "ci": (lo, hi)},
        )
    if ratio and lo <= 0:
        raise MethodIncompatibility(
            "causal_gap: scale='ratio' needs a positive lower limit"
        )

    if null is None:
        null = 1.0 if ratio else 0.0
    nulls = np.atleast_1d(np.asarray(null, dtype=float))
    if not np.isfinite(nulls).all() or (ratio and np.any(nulls <= 0)):
        raise MethodIncompatibility(
            "causal_gap: null must be finite"
            + (" and positive on the ratio scale" if ratio else "")
        )

    # Work on the additive scale; ratios are handled through their logs.
    f = np.log if ratio else (lambda v: v)
    back = np.exp if ratio else (lambda v: v)
    e, l, h = f(est), f(lo), f(hi)

    rows = []
    for m_raw in nulls:
        m = f(m_raw)
        holds = bool(l > m or h < m)
        if l > m:
            crit = l - m
        elif h < m:
            crit = h - m
        else:
            crit = 0.0
        to_point = e - m
        share = crit / to_point if to_point != 0 else np.nan
        rows.append(
            {
                "null": float(m_raw),
                "estimate": est,
                "ci_lower": lo,
                "ci_upper": hi,
                "conclusion_holds": holds,
                "critical_gap": float(back(crit)),
                "gap_to_point": float(back(to_point)),
                "share_of_estimate": float(share),
            }
        )
    out = pd.DataFrame(rows)

    # Curve for the first null: gaps from none to twice the gap that moves
    # the point estimate onto the null (or one interval width if it is
    # already there).
    m0 = f(nulls[0])
    span = 2.0 * (e - m0) if e != m0 else (h - l)
    gaps = np.linspace(0.0, span, int(n_grid))
    adj_l, adj_h = l - gaps, h - gaps
    out.attrs["curve"] = pd.DataFrame(
        {
            "gap": back(gaps),
            "estimate": back(e - gaps),
            "ci_lower": back(adj_l),
            "ci_upper": back(adj_h),
            "conclusion_holds": (adj_l > m0) | (adj_h < m0),
        }
    )
    out.attrs["scale"] = scale
    return out
