"""Input checks, robust variance and risk-set concordance for ``sp.cox``.

Kept apart from ``models.py`` so that the weighted / counting-process
pieces of the Cox fit (case weights, ``(start, stop]`` rows, the
small-sample convention of the sandwich) live in one place.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

#: Small-sample conventions of the robust / clustered Cox variance.
#: ``'r'``: none (``coxph``); ``'stata'``: ``N/(N-1)`` robust and
#: ``G/(G-1)`` clustered (``stcox, vce()``).
COX_SSC = ("r", "stata")


def resolve_cox_ssc(ssc: Any) -> Optional[str]:
    """Validate ``ssc=``; ``None`` keeps the historical mixed convention."""
    if ssc is None:
        return None
    key = str(ssc).lower()
    if key == "none":
        key = "r"
    if key not in COX_SSC:
        raise MethodIncompatibility(
            f"ssc must be 'r' or 'stata'; got {ssc!r}.",
            recovery_hint=(
                "ssc='r' reproduces survival::coxph (no factor); ssc='stata' "
                "reproduces stcox, vce(robust) / vce(cluster)."
            ),
            diagnostics={"ssc": ssc},
        )
    return key


def check_cox_inputs(
    T: np.ndarray,
    E: np.ndarray,
    w: Optional[np.ndarray],
    start: Optional[np.ndarray],
    *,
    robust: Any,
    cluster: Optional[str],
    weights_as: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Validate weights and interval starts.

    Returns the weights and, when some weights are zero, the mask of rows
    to keep: a zero-weight row adds nothing to any sum of the fit, so it is
    dropped (and not counted in ``N``).
    """
    if start is not None:
        bad = ~(start < T)
        if bad.any():
            raise DataInsufficient(
                f"start must be strictly less than the duration on every row; "
                f"{int(bad.sum())} row(s) have start >= stop.",
                recovery_hint=(
                    "Drop or fix the zero-length intervals (R coxph refuses "
                    "them too; Stata stset excludes them)."
                ),
                diagnostics={"n_bad_intervals": int(bad.sum())},
            )
    if w is None:
        return None, None
    if not np.all(np.isfinite(w)) or (w < 0).any():
        raise MethodIncompatibility(
            "weights must be finite and non-negative.",
            recovery_hint="Check the weight column for negative or infinite values.",
            diagnostics={"n_negative": int((w < 0).sum())},
        )
    keep = w > 0
    if not keep.any() or not (E[keep] == 1).any():
        raise DataInsufficient(
            "No event carries a positive weight.",
            recovery_hint="Check the weight column; every event has weight 0.",
        )
    non_integer = bool(np.any(w != np.floor(w)))
    if (
        non_integer
        and weights_as is None
        and cluster is None
        and robust in ("nonrobust", None)
    ):
        warnings.warn(
            "sp.cox: non-integer weights with the model-based variance, which "
            "treats them as frequency counts. For inverse-probability or survey "
            "weights the variance should be robust: R coxph switches to it by "
            "default for such weights and Stata stcox [pw=] always uses it. Pass "
            "weights_as='sampling' (or robust='hc0' / cluster=).",
            UserWarning,
            stacklevel=3,
        )
    if keep.all():
        return w, None
    return w, keep


def cox_robust_vcov(
    info_inv: np.ndarray,
    scores: np.ndarray,
    clusters: Optional[np.ndarray],
    ssc: Optional[str],
) -> np.ndarray:
    """Sandwich ``I^-1 (sum s s') I^-1`` of the (weighted) score residuals.

    With ``clusters`` the residuals are summed within cluster first. The
    factor is ``G/(G-1)`` clustered unless ``ssc='r'``, and ``N/(N-1)``
    unclustered only when ``ssc='stata'``.
    """
    n = scores.shape[0]
    if clusters is not None:
        codes, uniq = pd.factorize(pd.Series(clusters), sort=False)
        G = len(uniq)
        S = np.zeros((G, scores.shape[1]))
        np.add.at(S, codes, scores)
        factor = 1.0 if ssc == "r" else (G / (G - 1.0) if G > 1 else 1.0)
    else:
        S = scores
        factor = n / (n - 1.0) if (ssc == "stata" and n > 1) else 1.0
    meat = S.T @ S
    return np.asarray(factor * info_inv @ meat @ info_inv)


class _Fenwick:
    """Weighted Fenwick tree over integer positions ``0..n-1``."""

    def __init__(self, n: int):
        self.n = n
        self.t = [0.0] * (n + 1)

    def add(self, i: int, v: float) -> None:
        i += 1
        t = self.t
        n = self.n
        while i <= n:
            t[i] += v
            i += i & (-i)

    def prefix(self, i: int) -> float:
        """Sum over positions ``< i``."""
        s = 0.0
        t = self.t
        while i > 0:
            s += t[i]
            i -= i & (-i)
        return s


def risk_set_concordance(
    score: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    start: Optional[np.ndarray] = None,
    w: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
) -> float:
    """Concordance over the risk sets of a (weighted, counting-process) fit.

    A pair is comparable when row ``i`` fails at ``t`` while row ``k`` is at
    risk at ``t`` (``start_k < t <= stop_k``, same stratum) and does not
    itself fail at ``t``; it is concordant when ``i`` has the higher risk
    score, and a tie in the score counts one half. Pairs are weighted by
    ``w_i w_k``. This is R ``concordance()`` of a ``coxph`` fit (for which
    the higher linear predictor predicts the earlier failure); without
    weights, entry times or strata it is Harrell's C.

    The sweep runs over event times from the latest, adding a row when the
    time reaches its stop and removing it when it reaches its start, with
    a Fenwick tree over the score ranks: ``O(n log n)``.
    """
    T = np.asarray(T, dtype=float)
    E = np.asarray(E) == 1
    n = len(T)
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    st = np.full(n, -np.inf) if start is None else np.asarray(start, dtype=float)
    if strata is None:
        s_code = np.zeros(n, dtype=np.int64)
    else:
        s_code = pd.factorize(np.asarray(strata), sort=True)[0].astype(np.int64)
    _, rank = np.unique(np.asarray(score, dtype=float), return_inverse=True)
    R = int(rank.max()) + 1 if n else 1
    pos = s_code * R + rank  # stratum block, then score rank
    tree = _Fenwick(int(s_code.max() + 1) * R if n else 1)

    stop_order = np.argsort(-T, kind="stable")
    start_order = np.argsort(-st, kind="stable")
    ev_times = np.unique(T[E])[::-1]
    ia = 0
    ir = 0
    num = 0.0
    den = 0.0
    dead_idx = np.flatnonzero(E)
    dead_by_time = pd.Series(dead_idx).groupby(T[dead_idx]).apply(np.asarray)
    for t in ev_times:
        while ia < n and T[stop_order[ia]] >= t:
            k = stop_order[ia]
            tree.add(int(pos[k]), float(w[k]))
            ia += 1
        while ir < n and st[start_order[ir]] >= t:
            k = start_order[ir]
            if T[k] >= t:  # only rows already added can leave
                tree.add(int(pos[k]), -float(w[k]))
            ir += 1
        D = dead_by_time[t]
        # deaths at t are not compared with one another
        for k in D:
            tree.add(int(pos[k]), -float(w[k]))
        for i in D:
            base = int(s_code[i]) * R
            lo = tree.prefix(base)
            less = tree.prefix(int(pos[i])) - lo
            leq = tree.prefix(int(pos[i]) + 1) - lo
            tot = tree.prefix(base + R) - lo
            num += w[i] * (less + 0.5 * (leq - less))
            den += w[i] * tot
        for k in D:
            tree.add(int(pos[k]), float(w[k]))
    if den <= 0:
        return 0.5
    return float(num / den)
