"""Least "wiggly" confound path through an event study's Wald region.

Freyaldenhoven, Hansen, Perez Perez and Shapiro (Suggestion 6 of their
event-study chapter; Stata ``xteventplot, smpath()``) plot, next to the
estimated event-time path ``delta_hat``, the smoothest polynomial confound
that is statistically consistent with it: the polynomial
``delta*_k(v) = sum_j v_j k^(j-1)`` with the fewest coefficients ``p*`` whose
path lies in the Wald region
``CR = {d : (d - delta_hat)' V^-1 (d - delta_hat) <= c}``, and among those the
one with the smallest ``v_p*^2``. The order reported is the degree
``p* - 1``, as Stata prints it ("Order 0" is a constant).

Here both steps are solved exactly. On the subspace of polynomials that are
zero at the normalised event times, the Wald statistic is a quadratic in the
coefficients with minimum ``q_min(p)``; ``p*`` is the first ``p`` with
``q_min(p) <= c``, and the leading coefficient ranges over an interval whose
end nearest zero is attained at a unique point of the ellipsoid (closed
form). Stata minimises the same objective numerically (its help file warns
that the optimiser may stop with "flat region" or "missing derivative"
messages and that the result is approximate), so agreement with it is a
stochastic-screen comparison, not parity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["xtevent_smpath"]


def _event_inputs(result: CausalResult):
    info = getattr(result, "model_info", None) or {}
    es = info.get("event_study")
    v_ev = info.get("vcov_event")
    if not isinstance(es, pd.DataFrame) or not isinstance(v_ev, pd.DataFrame):
        raise MethodIncompatibility(
            "xtevent_smpath needs an sp.xtevent event study (model_info "
            "'event_study' and 'vcov_event').",
            recovery_hint="Fit sp.xtevent(..., window=...) without static=True "
            "and pass its result.",
            diagnostics={"result_type": type(result).__name__},
        )
    if info.get("proxy") or info.get("trend"):
        raise MethodIncompatibility(
            "xtevent_smpath is defined on the plain event study; this fit "
            "uses proxy= or trend=.",
            recovery_hint="Refit sp.xtevent without proxy= / trend= for the "
            "smoothest-path plot (Stata draws it on the same estimates).",
            diagnostics={"proxy": bool(info.get("proxy")), "trend": info.get("trend")},
        )
    return es, v_ev, info


def _path_frame(es, est_times, b, W, w, scale, powers, crit, df, level, q_by):
    """The output table for scaled coefficients ``w`` (``len(w) - 1`` = degree)."""
    coef = w / scale**powers
    k_all = es["relative_time"].to_numpy(dtype=float)
    path = (k_all[:, None] ** powers) @ coef
    fitted = (np.asarray(est_times, dtype=float)[:, None] ** powers) @ coef
    r = fitted - b
    out = pd.DataFrame(
        {
            "relative_time": es["relative_time"].to_numpy(),
            "att": es["att"].to_numpy(dtype=float),
            "in_region": es["relative_time"].isin(est_times).to_numpy(),
            "smpath": path,
        }
    )
    out.attrs.update(
        order=len(w) - 1,
        coefficients=coef.tolist(),
        wald_by_order=list(q_by),
        wald=float(r @ W @ r),
        critical=crit,
        df=df,
        level=float(level),
    )
    return out


def xtevent_smpath(
    result: CausalResult,
    postwindow: Optional[int] = None,
    maxorder: int = 10,
    level: float = 0.95,
) -> pd.DataFrame:
    """Smoothest polynomial confound path through an event study's Wald region.

    The "least wiggly" path of Freyaldenhoven, Hansen, Perez Perez and
    Shapiro that Stata ``xteventplot, smpath(line)`` draws after
    ``xtevent``: the lowest-order polynomial in event time that passes
    through the normalised coefficient(s) and lies in the Wald confidence
    region of the estimated coefficients, and, among those, the one whose
    highest-order coefficient is smallest in magnitude.

    Parameters
    ----------
    result : CausalResult
        An :func:`sp.xtevent` event study (not ``static=True``, ``trend=`` or
        ``proxy=``).
    postwindow : int, optional
        Number of post-event coefficients (event times ``0 .. postwindow - 1``
        after the normalised one, counted from event time 0) that enter the
        Wald region (Stata ``postwindow()``); the path is still evaluated over
        the whole window. Default: all of them, endpoints included.
    maxorder : int, default 10
        Highest polynomial degree tried (Stata ``maxorder()``, between 1 and
        10); degree 0 (a constant, which the normalisation makes the zero
        path) is always tried first.
    level : float, default 0.95
        Coverage of the Wald region: the chi-square ``level`` quantile with as
        many degrees of freedom as estimated coefficients in it.

    Returns
    -------
    pd.DataFrame
        One row per event time of the window: ``relative_time``, the
        estimate ``att`` (zero at the normalised time), ``in_region`` (whether
        the coefficient enters the Wald region) and ``smpath``, the path.
        ``attrs`` holds ``order`` (the degree, ``p* - 1``), ``coefficients``
        (``v*``, constant first, in powers of event time), ``wald`` (the
        path's Wald statistic, at most ``critical``), ``wald_by_order`` (the
        least Wald statistic at each degree tried, Stata's "Wald value"
        lines), ``critical`` and ``df``.

    Raises
    ------
    MethodIncompatibility
        If the result is not a plain :func:`sp.xtevent` event study.
    DataInsufficient
        If no polynomial up to ``maxorder`` reaches the Wald region.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> d = pd.DataFrame({"i": np.repeat(np.arange(40), 12),
    ...                   "t": np.tile(np.arange(12), 40)})
    >>> d["z"] = np.repeat(rng.integers(0, 2, 40), 12) * (d["t"] >= 6)
    >>> d["y"] = d["z"] + rng.normal(size=len(d))
    >>> r = sp.xtevent(d, y="y", policy="z", panel="i", time="t", window=2)
    >>> path = sp.xtevent_smpath(r)
    >>> bool(path.attrs["wald"] <= path.attrs["critical"] + 1e-9)
    True

    Notes
    -----
    Stata reaches the same objective by numerical optimisation and documents
    its answer as approximate, so the two are compared as a screen
    (``tests/reference_parity/test_xtevent_smpath_Stata_screen.py``), not
    held to a parity tolerance. Here ``p*`` is the first ``p`` whose
    least-squares Wald statistic on polynomials vanishing at the normalised
    times is at most the critical value, and the leading coefficient is the
    end, nearest zero, of the interval it spans over the region's
    intersection with those polynomials; both are closed form.

    References
    ----------
    [@freyaldenhoven2019event]
    """
    es, v_ev, info = _event_inputs(result)
    maxorder = int(maxorder)
    if not 1 <= maxorder <= 10:
        raise MethodIncompatibility(
            "maxorder must be between 1 and 10.",
            recovery_hint="Stata's default is maxorder(10).",
            diagnostics={"maxorder": maxorder},
        )
    if not 0.0 < float(level) < 1.0:
        raise MethodIncompatibility(
            "level must lie strictly between 0 and 1.",
            recovery_hint="Use level=0.95 for a 95% Wald region.",
            diagnostics={"level": level},
        )
    es = es.sort_values("relative_time").reset_index(drop=True)
    k_all = es["relative_time"].to_numpy(dtype=float)
    zeroed = set(int(t) for t in info.get("normalized", []))
    est_times = [int(t) for t in v_ev.index]
    if postwindow is not None:
        postwindow = int(postwindow)
        if postwindow < 1:
            raise MethodIncompatibility(
                "postwindow must be a positive integer.",
                recovery_hint="Omit it to use every post-event coefficient.",
                diagnostics={"postwindow": postwindow},
            )
        est_times = [t for t in est_times if t < 0 or t < postwindow]
    b = es.set_index("relative_time").loc[est_times, "att"].to_numpy(dtype=float)
    V = v_ev.loc[est_times, est_times].to_numpy(dtype=float)
    W = np.linalg.pinv(V)
    df = len(est_times)
    crit = float(stats.chi2.ppf(level, df))
    scale = float(max(np.max(np.abs(k_all)), 1.0))
    k_est = np.asarray(est_times, dtype=float) / scale
    k_norm = np.asarray(sorted(zeroed), dtype=float) / scale

    q_by: list = []  # least Wald statistic by degree, as Stata prints them
    for p in range(1, maxorder + 2):  # p coefficients = degree p - 1
        powers = np.arange(p)
        X = k_est[:, None] ** powers
        if k_norm.size:
            # Polynomials that vanish at the normalised times: v = N z.
            C = k_norm[:, None] ** powers
            _, sv, vt = np.linalg.svd(C)
            rank = int(np.sum(sv > 1e-12 * max(sv.max(), 1.0)))
            N = vt[rank:].T
        else:
            N = np.eye(p)
        if N.shape[1] == 0:
            # Only the zero polynomial passes through the normalised times:
            # feasible when the no-effect, no-confound path is in the region.
            q_by.append(float(b @ W @ b))
            if q_by[-1] > crit * (1.0 + 1e-12):
                continue
            w = np.zeros(p)
            return _path_frame(
                es, est_times, b, W, w, scale, powers, crit, df, level, q_by
            )
        A = X @ N
        H = A.T @ W @ A
        g = A.T @ W @ b
        z0 = np.linalg.lstsq(H, g, rcond=None)[0]
        resid = A @ z0 - b
        q_min = float(resid @ W @ resid)
        q_by.append(q_min)
        if q_min > crit * (1.0 + 1e-12):
            continue
        # Leading (scaled) coefficient is c'z over the ellipsoid
        # (z - z0)' H (z - z0) <= crit - q_min.
        c = N[p - 1]
        h_inv_c = np.linalg.lstsq(H, c, rcond=None)[0]
        spread = float(c @ h_inv_c)
        slack = max(crit - q_min, 0.0)
        centre = float(c @ z0)
        if spread <= 0 or slack == 0:
            z = z0
        else:
            radius = np.sqrt(slack * spread)
            if abs(centre) <= radius:
                # Zero is reachable, so a lower degree would have been
                # feasible; only without a normalised time (never after
                # sp.xtevent) can this happen, at degree 0.
                z = z0 - h_inv_c * (centre / spread)
            else:
                step = np.sign(centre) * np.sqrt(slack / spread)
                z = z0 - h_inv_c * step
        w = N @ z
        return _path_frame(es, est_times, b, W, w, scale, powers, crit, df, level, q_by)
    raise DataInsufficient(
        f"No polynomial of order up to {maxorder} lies in the {level:.0%} Wald "
        "region of the event-time coefficients.",
        recovery_hint="Raise maxorder (at most 10) or lower level.",
        diagnostics={
            "maxorder": maxorder,
            "df": df,
            "critical": crit,
            "wald_by_order": q_by,
        },
    )
