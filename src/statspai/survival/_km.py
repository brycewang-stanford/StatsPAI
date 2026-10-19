"""Weighted / delayed-entry Kaplan-Meier with the infinitesimal-jackknife
variance (R ``survival::survfit`` with ``weights=``, ``Surv(start, stop,
event)`` and ``robust=``).

The estimator is ``S(t) = prod_{t_j <= t} (1 - d_j / n_j)`` with ``d_j``
the weighted deaths at ``t_j`` and ``n_j`` the weighted number at risk
(``start < t_j <= stop``). Its derivative with respect to the weight of
row ``i`` is

    U_i(t) = S(t) sum_{t_j <= t} (d_j Y_i(t_j) - n_j dN_i(t_j))
                                  / (n_j (n_j - d_j)),

and the infinitesimal-jackknife variance sums ``(w_i U_i(t))^2``, first
within cluster when one is given. This is the robust variance R reports
by default for non-integer weights; without weights the Greenwood
variance is the default there and here.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple

import numpy as np
import pandas as pd


def km_weighted_curve(
    T: np.ndarray,
    E: np.ndarray,
    w: np.ndarray,
    start: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, ...]:
    """Event times, weighted at-risk / deaths / censored, survival, and the
    per-time pieces the variances need."""
    T = np.asarray(T, dtype=float)
    dead = np.asarray(E) == 1
    times = np.unique(T[dead])
    # weighted number at risk at each event time: stop >= t and start < t
    order = np.argsort(T)
    cw = np.concatenate([[0.0], np.cumsum(w[order])])
    n_stop_ge = cw[-1] - cw[np.searchsorted(T[order], times, side="left")]
    if start is not None:
        so = np.argsort(start)
        cs = np.concatenate([[0.0], np.cumsum(w[so])])
        n_start_ge = cs[-1] - cs[np.searchsorted(start[so], times, side="left")]
        n_risk = n_stop_ge - n_start_ge
    else:
        n_risk = n_stop_ge
    idx = np.searchsorted(times, T)
    in_t = (idx < len(times)) & (times[np.minimum(idx, len(times) - 1)] == T)
    d = np.bincount(idx[dead], weights=w[dead], minlength=len(times))
    cens_rows = in_t & ~dead
    c = np.bincount(idx[cens_rows], weights=w[cens_rows], minlength=len(times))
    surv = np.cumprod(1.0 - d / n_risk)
    return times, n_risk, d, c, surv


def _ij_sweep(
    T: np.ndarray,
    E: np.ndarray,
    w: np.ndarray,
    times: np.ndarray,
    n_risk: np.ndarray,
    d: np.ndarray,
    start: Optional[np.ndarray],
    codes: np.ndarray,
    G: int,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(j, acc)`` over the event times, ``acc`` holding the cluster
    totals of ``w_i U_i(t_j) / S(t_j)``.

    ``U_i(t_m) / S(t_m) = sum_{j <= m} Y_i(t_j) a_j - dN_i(t_j) b_j`` with
    ``a_j = d_j / (n_j (n_j - d_j))`` and ``b_j = 1 / (n_j - d_j)``: a row
    adds ``w_i a_j`` while at risk (event times inside ``(start_i,
    stop_i]``) and ``-w_i b_j`` at its own death. One pass with running
    totals, ``O(n + G m)``. ``acc`` is updated in place; use it before the
    next step.
    """
    dead = np.asarray(E) == 1
    m = len(times)
    with np.errstate(divide="ignore", invalid="ignore"):
        full = n_risk > d
        a = np.where(full, d / (n_risk * (n_risk - d)), 0.0)
        b = np.where(full, 1.0 / (n_risk - d), 0.0)
    hi = np.searchsorted(times, T, side="right")
    lo = (
        np.zeros(len(T), dtype=np.int64)
        if start is None
        else np.searchsorted(times, start, side="right")
    )
    dead_rows = np.flatnonzero(dead)
    dead_at = np.searchsorted(times, T[dead_rows])
    order = np.argsort(dead_at, kind="stable")
    dead_rows, dead_at = dead_rows[order], dead_at[order]
    dead_bounds = np.searchsorted(dead_at, np.arange(m + 1))
    enter_order = np.argsort(lo, kind="stable")
    enter_bounds = np.searchsorted(lo[enter_order], np.arange(m + 1))
    leave_order = np.argsort(hi, kind="stable")
    leave_bounds = np.searchsorted(hi[leave_order], np.arange(m + 1), side="right")
    acc = np.zeros(G)
    w_active = np.zeros(G)  # weight of the rows at risk, by cluster
    active = np.zeros(len(T), dtype=bool)
    lp = 0
    for j in range(m):
        ent = enter_order[enter_bounds[j] : enter_bounds[j + 1]]
        ent = ent[lo[ent] < hi[ent]]
        active[ent] = True
        np.add.at(w_active, codes[ent], w[ent])
        lv = leave_order[lp : leave_bounds[j]]
        lp = leave_bounds[j]
        lv = lv[active[lv]]
        active[lv] = False
        np.add.at(w_active, codes[lv], -w[lv])
        acc += w_active * a[j]
        rows = dead_rows[dead_bounds[j] : dead_bounds[j + 1]]
        if len(rows):
            np.add.at(acc, codes[rows], -w[rows] * b[j])
        yield j, acc


def _codes(cluster: Optional[np.ndarray], n: int) -> Tuple[np.ndarray, int]:
    if cluster is None:
        return np.arange(n), n
    codes, uniq = pd.factorize(pd.Series(cluster), sort=False)
    return np.asarray(codes), len(uniq)


def km_ij_variance(
    T: np.ndarray,
    E: np.ndarray,
    w: np.ndarray,
    times: np.ndarray,
    n_risk: np.ndarray,
    d: np.ndarray,
    surv: np.ndarray,
    start: Optional[np.ndarray] = None,
    cluster: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Infinitesimal-jackknife variance of ``S`` at each event time (the
    sum over clusters of the squared totals of ``w_i U_i``)."""
    T = np.asarray(T, dtype=float)
    codes, G = _codes(cluster, len(T))
    var = np.zeros(len(times))
    for j, acc in _ij_sweep(T, E, w, times, n_risk, d, start, codes, G):
        var[j] = float(acc @ acc) * surv[j] ** 2
    return var


def km_ij_rmst_influence(
    T: np.ndarray,
    E: np.ndarray,
    w: np.ndarray,
    times: np.ndarray,
    n_risk: np.ndarray,
    d: np.ndarray,
    surv: np.ndarray,
    tau: float,
    codes: np.ndarray,
    G: int,
    start: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Cluster totals of ``w_i`` times the influence of row ``i`` on
    ``RMST(tau) = int_0^tau S(t) dt``: ``int_0^tau U_i(t) dt``, the step
    function ``U_i`` integrated between consecutive event times."""
    T = np.asarray(T, dtype=float)
    out = np.zeros(G)
    nxt = np.append(times[1:], np.inf)
    for j, acc in _ij_sweep(T, E, w, times, n_risk, d, start, codes, G):
        if times[j] >= tau:
            break
        width = min(nxt[j], tau) - times[j]
        out += acc * (surv[j] * width)
    return out


def km_effective_n(
    T: np.ndarray,
    w: np.ndarray,
    times: np.ndarray,
    start: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``(sum_R w)^2 / sum_R w^2`` over the risk set of each event time,
    the effective number at risk of Xie and Liu's variance."""
    order = np.argsort(T)
    Ts = T[order]

    def at_risk(v: np.ndarray) -> np.ndarray:
        c = np.concatenate([[0.0], np.cumsum(v[order])])
        out = c[-1] - c[np.searchsorted(Ts, times, side="left")]
        if start is not None:
            so = np.argsort(start)
            cs = np.concatenate([[0.0], np.cumsum(v[so])])
            out = out - (cs[-1] - cs[np.searchsorted(start[so], times, side="left")])
        return np.asarray(out)

    return np.asarray(at_risk(w) ** 2 / at_risk(w**2))


def km_table_weighted(
    T: np.ndarray,
    E: np.ndarray,
    w: Optional[np.ndarray],
    start: Optional[np.ndarray],
    cluster: Optional[np.ndarray],
    se_method: str,
    alpha: float,
    conf_type: str,
) -> pd.DataFrame:
    """Life table of the weighted / delayed-entry Kaplan-Meier."""
    from scipy import stats

    T = np.asarray(T, dtype=float)
    w = np.ones(len(T)) if w is None else np.asarray(w, dtype=float)
    times, n_risk, d, c, surv = km_weighted_curve(T, E, w, start)
    if se_method == "ij":
        var = km_ij_variance(T, E, w, times, n_risk, d, surv, start, cluster)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_log = np.where(surv > 0, var / surv**2, 0.0)
    elif se_method == "xie_liu":
        # Greenwood with the effective number at risk in the denominator
        M = km_effective_n(T, w, times, start)
        with np.errstate(divide="ignore", invalid="ignore"):
            inc = np.where(n_risk > d, d / (M * (n_risk - d)), 0.0)
        var_log = np.cumsum(inc)
        var = surv**2 * var_log
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            inc = np.where(n_risk > d, d / (n_risk * (n_risk - d)), 0.0)
        var_log = np.cumsum(inc)
        var = surv**2 * var_log
    se = np.sqrt(var)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lo, ci_hi = _km_ci(surv, se, var_log, z, conf_type)
    body = pd.DataFrame(
        {
            "time": times,
            "n_risk": n_risk,
            "n_event": d,
            "n_censor": c,
            "survival": surv,
            "std_err": se,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        }
    )
    t0 = pd.DataFrame(
        [
            {
                "time": 0.0,
                "n_risk": (
                    float(w.sum())
                    if start is None
                    else float(n_risk[0]) if len(n_risk) else 0.0
                ),
                "n_event": 0.0,
                "n_censor": 0.0,
                "survival": 1.0,
                "std_err": 0.0,
                "ci_lower": 1.0,
                "ci_upper": 1.0,
            }
        ]
    )
    return pd.concat([t0, body], ignore_index=True)


def _km_ci(
    surv: np.ndarray, se: np.ndarray, var_log: np.ndarray, z: float, conf_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Pointwise interval on the plain, log or log-log scale (as
    ``_km_table``), from the variance of ``log S``."""
    plain = (conf_type == "plain") | (var_log <= 0) | ~((surv > 0.0) & (surv < 1.0))
    ci_lo = np.maximum(0.0, surv - z * se)
    ci_hi = np.minimum(1.0, surv + z * se)
    if conf_type != "plain":
        undefined = plain & (surv <= 0.0)
        ci_lo[undefined] = np.nan
        ci_hi[undefined] = np.nan
        ok = ~plain
        with np.errstate(divide="ignore", invalid="ignore"):
            if conf_type == "log":
                half = z * np.sqrt(var_log)
                lo = surv * np.exp(-half)
                hi = np.minimum(1.0, surv * np.exp(half))
            else:
                half = z * np.sqrt(var_log) / np.abs(np.log(surv))
                lo = surv ** np.exp(half)
                hi = surv ** np.exp(-half)
        ci_lo[ok] = lo[ok]
        ci_hi[ok] = hi[ok]
    return ci_lo, ci_hi


def resolve_km_se(se_method: Any, w: Optional[np.ndarray], cluster: Any) -> str:
    """R's rule for the default: robust (IJ) with a cluster or non-integer
    weights, Greenwood otherwise."""
    from ..exceptions import MethodIncompatibility

    if se_method is None:
        if cluster is not None:
            return "ij"
        if w is not None and np.any(w != np.floor(w)):
            return "ij"
        return "greenwood"
    key = str(se_method).lower()
    if key in ("robust", "jackknife", "infinitesimal_jackknife"):
        key = "ij"
    if key in ("xieliu", "xie-liu"):
        key = "xie_liu"
    if key not in ("greenwood", "ij", "xie_liu"):
        raise MethodIncompatibility(
            f"se_method must be 'greenwood', 'ij' or 'xie_liu'; got {se_method!r}.",
            recovery_hint="Use 'ij' (robust; R survfit robust=TRUE) or 'greenwood'.",
        )
    if key != "ij" and cluster is not None:
        raise MethodIncompatibility(
            "cluster= needs the infinitesimal-jackknife variance.",
            recovery_hint="Use se_method='ij' (or leave it unset) with cluster=.",
        )
    return key
