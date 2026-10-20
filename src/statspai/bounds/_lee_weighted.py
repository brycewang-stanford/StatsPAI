"""Lee (2009) bounds with sampling weights (``sp.lee_bounds(weights=...)``).

Clean-room rewrite (2026-10-20); Stata ``leebounds [pw=]`` was used only as
a black box. Specification, sources and the experiments behind every rule:
``docs/dev/cleanroom/leebounds.md``.

The weights are rescaled to mean one over the estimation sample. Retention
rates, trimming quantiles (``_pctile`` with weights), kept mass, means and
the analytic variance (Lee 2009, Proposition 3) are all weighted; at a
threshold shared by several observations the tied ones carry the mass still
needed, in proportion to their weights. ``trimming='leebounds'`` also
reproduces Stata's single-precision storage of the weights and of the
squared outcomes it averages, and its 16-digit macros.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility


def weighted_pctile(
    y: np.ndarray, w: np.ndarray, pct: float, tol: float = 1e-15
) -> float:
    """Stata ``_pctile`` with weights ([D] pctile, Methods and formulas).

    With ``W(i)`` the running weight total over the sorted values and
    ``P = sum(w) pct / 100``: the first ``x(i)`` with ``W(i) > P``, or the
    average of ``x(i-1)`` and ``x(i)`` when ``W(i-1) = P`` (equality up to
    a relative ``tol``; ``1e-15`` as for the unweighted command).
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(y, kind="stable")
    y, w = y[order], w[order]
    running = np.cumsum(w)
    P = running[-1] * pct / 100.0
    tol = tol * max(abs(P), 1.0)
    beyond = np.nonzero(running > P + tol)[0]
    i = int(beyond[0]) if len(beyond) else len(y) - 1
    if i > 0 and abs(running[i - 1] - P) <= tol:
        return float((y[i - 1] + y[i]) / 2.0)
    return float(y[i])


def _iw_var(y: np.ndarray, w: np.ndarray) -> float:
    """``summarize [iweight]`` variance: divisor ``sum(w) - 1``."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    total = float(np.sum(w))
    mean = float(np.sum(w * y)) / total
    return float(np.sum(w * (y - mean) ** 2) / (total - 1.0))


def _f32(v: np.ndarray) -> np.ndarray:
    """A value stored in a Stata ``float`` variable and read back."""
    return np.asarray(v, dtype=np.float32).astype(float)


def _check_weights(W: np.ndarray) -> None:
    if not np.all(np.isfinite(W)) or np.any(W < 0) or not np.any(W > 0):
        raise MethodIncompatibility(
            "lee_bounds: weights must be finite and non-negative, not all zero.",
            recovery_hint="Drop or fix rows with negative or missing weights.",
        )


def _side(
    y: np.ndarray, w: np.ndarray, q: float, top: bool, held: bool
) -> Tuple[float, float, float, float, bool]:
    """One trimmed tail of the trimmed arm.

    Returns ``(mean, second-moment variance, kept weight, threshold, tie)``.
    The variance divides by the kept weight at a tie (fractional share) and
    by the kept weight minus one otherwise.
    """
    from .lee_manski import _LEE_RULE_TOL, _stata_local_macro

    pct = 100.0 * q if top else 100.0 * (1.0 - q)
    if held:
        threshold = _stata_local_macro(weighted_pctile(y, w, _stata_local_macro(pct)))
    else:
        threshold = weighted_pctile(y, w, pct, tol=_LEE_RULE_TOL)
    tied = y == threshold
    if tied.any():
        inside = y > threshold if top else y < threshold
        keep = (1.0 - q) * float(np.sum(w))
        rest = keep - float(np.sum(w[inside]))
        squares = y[inside] ** 2
        if held:
            squares = _f32(squares)
        mean = (float(np.sum(w[inside] * y[inside])) + rest * threshold) / keep
        second = (float(np.sum(w[inside] * squares)) + rest * threshold**2) / keep
        return mean, second - mean**2, keep, threshold, True
    kept = y >= threshold if top else y <= threshold
    wk, yk = w[kept], y[kept]
    total = float(np.sum(wk))
    mean = float(np.sum(wk * yk)) / total
    s2 = float(np.sum(wk * (yk - mean) ** 2)) / (total - 1.0)
    return mean, s2, total, threshold, False


def weighted_point(
    Y: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    W: np.ndarray,
    trimming: str,
) -> Dict[str, Any]:
    """Bounds, analytic variances and retention rates on one sample."""
    held = trimming == "leebounds"
    if held:
        # weights held in single precision, then rescaled and held again;
        # the retention rates use the weights as first held
        w_rates = _f32(W)
        w = _f32(w_rates / np.mean(w_rates))
    else:
        w = W / np.mean(W)
        w_rates = w

    def rates(v: np.ndarray) -> Tuple[float, float]:
        t1, t0 = np.sum(v * D), np.sum(v * (1.0 - D))
        if t1 <= 0 or t0 <= 0:
            raise DataInsufficient(
                "lee_bounds: one treatment arm has no positive weight.",
                recovery_hint="Check the treatment indicator and the weights.",
            )
        return float(np.sum(v * S * D) / t1), float(np.sum(v * S * (1.0 - D)) / t0)

    p1, p0 = rates(w_rates)
    if p1 == 0 or p0 == 0:
        raise DataInsufficient(
            "lee_bounds: one treatment group has zero weighted retention.",
            recovery_hint="Check the selection indicator in each arm.",
        )
    treated_trimmed = p1 >= p0
    T = D if treated_trimmed else 1.0 - D
    hi, lo = (p1, p0) if treated_trimmed else (p0, p1)
    q = (hi - lo) / hi
    r1, r0 = rates(w)
    hi_r, lo_r = (r1, r0) if treated_trimmed else (r0, r1)
    q_rescaled = (hi_r - lo_r) / hi_r

    sel_t = (S == 1) & (T == 1)
    sel_o = (S == 1) & (T == 0)
    y_t, w_t = Y[sel_t], w[sel_t]
    y_o, w_o = Y[sel_o], w[sel_o]
    mean_other = float(np.sum(w_o * y_o) / np.sum(w_o))

    n = len(w)
    total = float(np.sum(w))
    e_st = float(np.sum(w * S * T)) / total
    e_t = float(np.sum(w * T)) / total
    e_so = float(np.sum(w * S * (1.0 - T))) / total
    s_other = float(np.sum(w * S * (1.0 - T)) / np.sum(w * (1.0 - T)))
    v_other = _iw_var(y_o, w_o) / e_so

    sides = {}
    for top in (False, True):
        mean, s2, kept, threshold, tie = _side(y_t, w_t, q, top, held)
        gap2 = (threshold - mean) ** 2
        v_trim = (
            s2 * n / kept
            + gap2 * q / (e_st * (1.0 - q))
            + gap2
            * ((1.0 - s_other) - q_rescaled * (1.0 - e_t))
            / (e_t * s_other * (1.0 - e_t))
        )
        sides[top] = (mean, (v_trim + v_other) / n, tie)

    # treated trimmed: lower bound from the bottom tail; control trimmed:
    # lower bound subtracts the control's top tail
    low, up = (
        (sides[False], sides[True]) if treated_trimmed else (sides[True], sides[False])
    )
    sign = 1.0 if treated_trimmed else -1.0
    return {
        "lower": sign * (low[0] - mean_other),
        "upper": sign * (up[0] - mean_other),
        "var_lower": float(low[1]),
        "var_upper": float(up[1]),
        "p1": p1,
        "p0": p0,
        "q": q,
        "tie_branch": {"lower": bool(low[2]), "upper": bool(up[2])},
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
    if trimming not in ("exact", "leebounds"):
        raise MethodIncompatibility(
            "lee_bounds(weights=...) trims a share of the weighted mass, which "
            "is defined for trimming='exact' (fractional weight at a tied "
            "threshold) and trimming='leebounds' (Stata leebounds [pw=]). Got "
            f"trimming={trimming!r}.",
            recovery_hint="Pass trimming='leebounds' or trimming='exact'.",
        )
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(D, dtype=float)
    S = np.asarray(S, dtype=float)
    W = np.asarray(W, dtype=float)
    _check_weights(W)
    out = weighted_point(Y, D, S, W, trimming)
    if se_method == "analytic":
        out["se_lower"] = float(np.sqrt(out["var_lower"]))
        out["se_upper"] = float(np.sqrt(out["var_upper"]))
        return out
    rng = np.random.RandomState(random_state)
    n = len(Y)
    draws = np.full((n_bootstrap, 2), np.nan)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            r = weighted_point(Y[idx], D[idx], S[idx], W[idx], trimming)
        except (DataInsufficient, ZeroDivisionError, FloatingPointError):
            continue
        draws[b] = r["lower"], r["upper"]
    out["se_lower"] = float(np.nanstd(draws[:, 0], ddof=1))
    out["se_upper"] = float(np.nanstd(draws[:, 1], ddof=1))
    return out
