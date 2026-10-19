"""Weighted Lee (2009) bounds: ``sp.lee_bounds(weights=...)``.

Stata ``leebounds y d [pw=w]`` (v1.5) rescales the weights to mean one on
the estimation sample and then uses them everywhere: the retention rates,
the trimming quantile (``_pctile`` with probability weights), the mass kept
at a tied threshold, the trimmed and untrimmed means and the terms of the
analytic variance. This module does the same, with one deliberate
difference: in the branch without ties at the threshold, ``leebounds``
computes the untrimmed arm's variance term of the *upper* bound without
the weights (``sum y if s == 1 & t == 0``), while the lower bound and the
tied branch weight it. Here it is weighted in every branch; the
reference test rebuilds Stata's number from the unweighted variance.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility


def weighted_pctile(y: np.ndarray, w: np.ndarray, pct: float) -> float:
    """Stata ``_pctile`` with probability weights.

    With the observations sorted and ``W_i`` the running sum of weights,
    ``P = W_n pct / 100``: the average of ``y_i`` and ``y_{i+1}`` when some
    ``W_i == P``, otherwise the first ``y_i`` with ``W_i > P``.
    """
    order = np.argsort(y, kind="mergesort")
    ys, ws = y[order], w[order]
    cum = np.cumsum(ws)
    P = cum[-1] * pct / 100.0
    eq = np.flatnonzero(cum == P)
    if eq.size and eq[0] + 1 < len(ys):
        i = int(eq[0])
        return float(0.5 * (ys[i] + ys[i + 1]))
    above = np.flatnonzero(cum > P)
    return float(ys[above[0]] if above.size else ys[-1])


def _iw_var(y: np.ndarray, w: np.ndarray) -> float:
    """``summarize [iweight]`` variance: divisor ``sum(w) - 1``."""
    mu = np.sum(w * y) / np.sum(w)
    return float(np.sum(w * (y - mu) ** 2) / (np.sum(w) - 1.0))


def weighted_lee(
    y_sel: np.ndarray,
    t_sel: np.ndarray,
    w_sel: np.ndarray,
    t_all: np.ndarray,
    s_all: np.ndarray,
    w_all: np.ndarray,
    q: float,
    rounding: bool,
) -> Tuple[float, float, float, float, Dict[bool, bool]]:
    """Trimmed-arm bounds and variances with weights.

    ``t`` marks the arm that is trimmed (the more retained one). Returns
    ``(contrast_low, contrast_high, var_low, var_high)`` for the trimmed
    arm's bottom-kept and top-kept means minus the untrimmed arm's mean,
    and whether each side took the tied-threshold branch.
    """
    nall = float(np.sum(w_all))
    est = np.sum(w_all * ((s_all == 1) & (t_all == 1))) / nall
    esnt = np.sum(w_all * ((s_all == 1) & (t_all == 0))) / nall
    et = np.sum(w_all * (t_all == 1)) / nall
    oddsc = np.sum(w_all * ((s_all == 0) & (t_all == 0))) / nall / esnt
    oddst = np.sum(w_all * ((s_all == 0) & (t_all == 1))) / nall / est
    nall_r = float(np.round(nall))
    vp = (1 - q) ** 2 * (oddst / et + oddsc / (1 - et))
    yt, wt = y_sel[t_sel == 1], w_sel[t_sel == 1]
    yc, wc = y_sel[t_sel == 0], w_sel[t_sel == 0]
    mean_c = float(np.sum(wc * yc) / np.sum(wc))
    vc = _iw_var(yc, wc) / esnt
    from .lee_manski import _stata_local_macro

    trim = 100.0 * q
    if rounding:
        trim = _stata_local_macro(trim)

    ties = {}

    def side(top: bool) -> Tuple[float, float]:
        pct = trim if top else 100.0 - trim
        if rounding:
            pct = _stata_local_macro(pct)
        thr = weighted_pctile(yt, wt, pct)
        if rounding:
            thr = _stata_local_macro(thr)
        at = yt == thr
        ties[top] = bool(at.any())
        if not at.any():
            keep = yt >= thr if top else yt <= thr
            yk, wk = yt[keep], wt[keep]
            mu = float(np.sum(wk * yk) / np.sum(wk))
            vb1 = _iw_var(yk, wk) / float(np.sum(wk))
        else:
            beyond = yt > thr if top else yt < thr
            neth = float(np.sum(wt[at]))
            nbth = float(np.sum(wt[beyond]))
            sbth = float(np.sum(wt[beyond] * yt[beyond]))
            ntall = float(np.sum(wt))
            stie = (ntall * (1 - q) - nbth) / neth
            mass = nbth + stie * neth
            mu = (sbth + stie * neth * thr) / mass
            ysq = yt[beyond] ** 2
            if rounding:  # leebounds generates y^2 as a float variable
                ysq = ysq.astype(np.float32).astype(float)
            sq = float(np.sum(wt[beyond] * ysq))
            vb1 = ((sq + stie * neth * thr**2) / mass - mu**2) / mass
        vb2 = (thr - mu) ** 2 * q / (est * (1 - q))
        vb3 = ((thr - mu) / (1 - q)) ** 2 * vp
        return mu - mean_c, vb1 + (vb2 + vb3 + vc) / nall_r

    lo, v_lo = side(False)
    hi, v_hi = side(True)
    return lo, hi, v_lo, v_hi, ties


def _f32(v: np.ndarray) -> np.ndarray:
    """A value stored by Stata ``generate`` (float) and read back."""
    return np.asarray(v, dtype=np.float32).astype(float)


def weighted_point(
    Y: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    W: np.ndarray,
    trimming: str,
) -> Dict[str, float]:
    """Bounds, analytic variances and retention rates on one sample.

    ``trimming='leebounds'`` reproduces ``leebounds``' storage of the
    weights as ``float`` (``generate`` without a type) and its 16-digit
    macros; ``'exact'`` uses the weights as given and no rounding.
    """
    stata = trimming == "leebounds"
    w_raw = _f32(W) if stata else np.asarray(W, dtype=float)
    w = w_raw / np.mean(w_raw)
    if stata:
        w = _f32(w)
    p1 = float(np.sum(w_raw * S * (D == 1)) / np.sum(w_raw * (D == 1)))
    p0 = float(np.sum(w_raw * S * (D == 0)) / np.sum(w_raw * (D == 0)))
    if p1 == 0 or p0 == 0:
        raise DataInsufficient(
            "lee_bounds: one treatment group has zero weighted retention.",
            recovery_hint="Check the selection indicator in each arm.",
        )
    t = D if p1 >= p0 else 1.0 - D
    q = (max(p1, p0) - min(p1, p0)) / max(p1, p0)
    sel = S == 1
    lo, hi, v_lo, v_hi, ties = weighted_lee(Y[sel], t[sel], w[sel], t, S, w, q, stata)
    if p1 >= p0:
        lb, ub, var_lb, var_ub = lo, hi, v_lo, v_hi
        tie_lb, tie_ub = ties[False], ties[True]
    else:  # control trimmed: treated mean minus the control's kept means
        lb, ub, var_lb, var_ub = -hi, -lo, v_hi, v_lo
        tie_lb, tie_ub = ties[True], ties[False]
    return {
        "lower": float(lb),
        "upper": float(ub),
        "var_lower": float(var_lb),
        "var_upper": float(var_ub),
        "p1": p1,
        "p0": p0,
        "q": float(q),
        "tie_branch": {"lower": tie_lb, "upper": tie_ub},
    }


def weighted_bounds(
    Y: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    W: np.ndarray,
    trimming: str,
    se_method: str,
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, Any]:
    """Point estimates plus analytic or (weight-carrying) bootstrap SEs."""
    if trimming not in ("leebounds", "exact"):
        raise MethodIncompatibility(
            "lee_bounds(weights=...) trims a share of the weighted mass, which "
            "is defined for trimming='exact' (fractional weight at a tied "
            "threshold) and trimming='leebounds' (Stata leebounds [pw=])."
            f" Got trimming={trimming!r}.",
            recovery_hint="Pass trimming='leebounds' or trimming='exact'.",
        )
    if not np.all(np.isfinite(W)) or np.any(W < 0) or not np.any(W > 0):
        raise MethodIncompatibility(
            "lee_bounds: weights must be finite and non-negative, not all zero.",
            recovery_hint="Drop or fix rows with negative or missing weights.",
        )
    out = weighted_point(Y, D, S, W, trimming)
    if se_method == "analytic":
        out["se_lower"] = float(np.sqrt(out["var_lower"]))
        out["se_upper"] = float(np.sqrt(out["var_upper"]))
        return out
    rng = np.random.RandomState(random_state)
    n = len(D)
    draws = []
    for _ in range(int(n_bootstrap)):
        idx = rng.choice(n, size=n, replace=True)
        try:
            b = weighted_point(Y[idx], D[idx], S[idx], W[idx], trimming)
        except (DataInsufficient, ZeroDivisionError, FloatingPointError, IndexError):
            continue
        draws.append((b["lower"], b["upper"]))
    if len(draws) < max(20, n_bootstrap // 2):
        raise MethodIncompatibility(
            "lee_bounds: too few usable bootstrap draws with these weights.",
            recovery_hint="Use se_method='analytic'.",
        )
    arr = np.asarray(draws)
    out["se_lower"] = float(np.std(arr[:, 0], ddof=1))
    out["se_upper"] = float(np.std(arr[:, 1], ddof=1))
    return out
