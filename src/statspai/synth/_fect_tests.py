"""Diagnostic tests for ``sp.fect``: placebo, carryover, leave-one-period-out.

Ports of the tests in Liu, Wang and Xu's R package ``fect`` (2.4.x):

* **placebo test** (``fect(placeboTest = TRUE, placebo.period = ...)``):
  the untreated cells of the treated units in a window of pre-treatment
  periods are removed from the fit and their out-of-sample effect, which
  should be zero, is averaged and tested;
* **carryover test** (``carryoverTest = TRUE, carryover.period = ...``):
  with treatment reversals, the untreated cells in a window of periods
  after the treatment switches off are removed the same way;
* **leave-one-period-out** (``loo = TRUE``): every pre-treatment period in
  turn is removed from the fit and its effect estimated out of sample;
* the pre-trend **F test** and **TOST equivalence test** of
  ``fect:::diagtest`` on the in-sample (or leave-one-out) pre-period
  effects.

Relative periods use fect's coding: ``0`` is the last untreated period
before onset, ``1`` the first treated period; after a switch-off, ``1`` is
the first untreated period.

One deliberate departure from ``fect``: for ``vartype = "jackknife"``
``diagtest`` builds the F statistic's covariance matrix as ``cov()`` of
the raw leave-one-unit-out replicates. That is the jackknife variance
divided by ``(n - 1)^2 / n`` (about ``n``), so the R F statistic is
inflated by that factor and its p-value rejects parallel trends in data
that satisfy them. StatsPAI uses the jackknife covariance
``(n - 1)/n * sum (r_i - rbar)(r_i - rbar)'``; ``F_fect = F * (n - 1)^2 / n``
exactly (pinned in the parity test). The bootstrap path is the same in
both.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility


def normalize_window(
    period: Any, *, name: str, upper: Optional[int], lower: Optional[int]
) -> Tuple[int, int]:
    """``period`` (int or (lo, hi)) as an inclusive integer window."""
    vals = [period] if np.isscalar(period) else list(period)
    if len(vals) not in (1, 2):
        raise MethodIncompatibility(
            f"fect: {name} must be one period or a (first, last) pair",
            recovery_hint=f"Pass e.g. {name}=(-2, 0).",
        )
    lo, hi = int(vals[0]), int(vals[-1])
    if lo > hi:
        lo, hi = hi, lo
    if upper is not None and hi > upper:
        raise MethodIncompatibility(
            f"fect: {name} must be <= {upper} in fect's coding "
            "(0 = last untreated period)",
            recovery_hint=f"Pass {name}=(-2, 0) for the last three pre-periods.",
        )
    if lower is not None and lo < lower:
        raise MethodIncompatibility(
            f"fect: {name} must be >= {lower} (1 = first period after the "
            "treatment switches off)",
            recovery_hint=f"Pass {name}=(1, 2).",
        )
    return lo, hi


def off_term(D: np.ndarray, Iobs: np.ndarray, get_term: Callable) -> np.ndarray:
    """fect ``get_term(type = "off")`` for every unit (rows = time).

    ``get_term`` fills unobserved cells with the previous status before
    flipping it, and filling commutes with flipping, so the off-coding is
    the on-coding of ``1 - D``.
    """
    T, N = D.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        if Iobs[:, j].sum() > 0:
            out[:, j] = get_term(1.0 - D[:, j], Iobs[:, j])
    return out


def has_reversals(D: np.ndarray, Iobs: np.ndarray) -> bool:
    """Some unit's observed treatment goes 1 -> 0 (fect ``hasRevs``)."""
    for j in range(D.shape[1]):
        d = D[Iobs[:, j] == 1, j]
        if d.size and np.any(np.maximum.accumulate(d) != d):
            return True
    return False


def window_mean(
    eff: np.ndarray, term: np.ndarray, lo: int, hi: int
) -> Tuple[float, int]:
    """Mean effect over observed cells whose relative period is in [lo, hi]."""
    sel = ~np.isnan(eff) & ~np.isnan(term) & (term >= lo) & (term <= hi)
    n = int(sel.sum())
    return (float(eff[sel].mean()) if n else np.nan), n


def replicate_se(reps: Sequence[float], vce: str) -> float:
    """fect's replicate standard error (sd for bootstrap, jackknife SE)."""
    r = np.asarray([v for v in reps if np.isfinite(v)], dtype=float)
    if r.size < 2:
        return np.nan
    if vce == "bootstrap":
        return float(np.std(r, ddof=1))
    n = r.size
    return float(np.sqrt((n - 1) / n * np.sum((r - r.mean()) ** 2)))


def tost_p(coef: float, se: float, threshold: float) -> float:
    """fect's two one-sided tests p-value for ``|coef| < threshold``."""
    p1 = stats.norm.sf((coef + threshold) / se)
    p2 = stats.norm.sf((threshold - coef) / se)
    return float(max(p1, p2))


def effect_test(
    est: float, reps: Sequence[float], vce: str, alpha: float, threshold: float
) -> Dict[str, Any]:
    """fect's ``est.placebo`` / ``est.carryover`` row plus its TOST p-value."""
    se = replicate_se(reps, vce)
    z = stats.norm.ppf(1 - alpha / 2)
    if np.isfinite(se) and se > 0:
        pvalue = float(2 * stats.norm.sf(abs(est) / se))
        equiv = tost_p(est, se, threshold)
    else:
        pvalue = equiv = np.nan
    return {
        "estimate": float(est),
        "se": se,
        "ci": (float(est - z * se), float(est + z * se)),
        "pvalue": pvalue,
        "equiv_pvalue": equiv,
        "tost_threshold": float(threshold),
        "n_reps": int(np.isfinite(np.asarray(reps, dtype=float)).sum()),
    }


def _replicate_cov(R: np.ndarray, vce: str) -> np.ndarray:
    """Covariance of the estimates from replicates (rows = estimates)."""
    n = R.shape[1]
    S = np.cov(R, ddof=1)
    S = np.atleast_2d(S)
    if vce == "jackknife":
        S = S * (n - 1) ** 2 / n
    return S


def pretrend_tests(
    time: np.ndarray,
    count: np.ndarray,
    att: np.ndarray,
    se: np.ndarray,
    reps: np.ndarray,
    *,
    vce: str,
    tost_threshold: float,
    f_threshold: float,
    proportion: float,
    pre_periods: Optional[Sequence[int]],
    loo_att: Optional[np.ndarray] = None,
    loo_se: Optional[np.ndarray] = None,
    loo_reps: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """``fect:::diagtest`` without the placebo / carryover branches.

    ``time`` / ``count`` / ``att`` / ``se`` describe the in-sample event
    study; ``reps`` is (periods x replicates). With the ``loo_*`` arrays
    (one row per pre-period, ``time <= 0``) the leave-one-out version is
    computed instead, exactly as ``diagtest`` does when ``x$loo`` is set.
    """
    time = np.asarray(time, dtype=float)
    count = np.asarray(count, dtype=float)
    max_pre = time[(count >= count.max() * (proportion or 0.0)) & (time <= 0)]
    if pre_periods is None:
        pre = max_pre
    elif len(pre_periods) > 0:
        rng_ = np.arange(int(pre_periods[0]), int(pre_periods[-1]) + 1)
        pre = np.intersect1d(rng_, max_pre)
    else:
        pre = np.array([])
    n_pre_all = int(np.sum(time <= 0))
    out: Dict[str, Any] = {"tost_threshold": float(tost_threshold)}
    if pre.size == 0:
        out.update(f_stat=np.nan, f_p=np.nan, f_equiv_p=np.nan, tost_equiv_p=np.nan)
        return out
    N_bar = float(count[np.isin(time, pre)].max())
    pos = np.flatnonzero(np.isin(time, pre))
    if loo_att is not None:
        pos = pos[pos < loo_att.shape[0]]
    if pos.size == n_pre_all:
        pos = pos[1:]
    if loo_att is not None:
        D = loo_att[pos]
        R = loo_reps[pos]
        est_rows, se_rows = loo_att, loo_se
    else:
        D = att[pos]
        R = reps[pos]
        est_rows, se_rows = att[pos], se[pos]
    R = R[:, np.all(np.isfinite(R), axis=0)]
    k = pos.size
    f_stat = f_p = f_equiv = np.nan
    if k > 0 and R.shape[1] >= 2:
        S = _replicate_cov(R, vce)
        try:
            psi = float(D @ np.linalg.solve(S, D))
        except np.linalg.LinAlgError:
            psi = np.nan
        # fect: (N_bar - k) / ((N_bar - 1) k), no F test unless positive
        scale = (N_bar - k) / ((N_bar - 1) * k) if N_bar > k else -1.0
        if np.isfinite(psi) and scale > 0:
            f_stat = psi * scale
            f_p = float(stats.f.sf(f_stat, k, N_bar - k))
            f_equiv = float(stats.ncf.cdf(f_stat, k, N_bar - k, N_bar * f_threshold))
    tost = [
        tost_p(float(e), float(s), tost_threshold)
        for e, s in zip(est_rows, se_rows)
        if np.isfinite(e) and np.isfinite(s) and s > 0
    ]
    out.update(
        f_stat=float(f_stat),
        f_p=f_p,
        f_threshold=float(f_threshold),
        f_equiv_p=f_equiv,
        df1=int(k),
        df2=float(N_bar - k),
        N_bar=float(N_bar),
        tost_equiv_p=float(max(tost)) if tost else np.nan,
        pre_periods=[int(v) for v in time[pos]],
    )
    return out


def drop_short_units(
    II: np.ndarray, D: np.ndarray, Iobs: np.ndarray, min_t0: int, what: str
) -> np.ndarray:
    """Units kept after hiding cells (fect's second ``min.T0`` pass)."""
    T0 = II.sum(axis=0)
    treated = (D * Iobs).sum(axis=0) > 0
    keep = T0 >= min_t0
    if not keep[treated].any():
        raise DataInsufficient(
            f"fect: all treated units have been removed in the {what} test",
            recovery_hint=f"Use a shorter {what} window or a smaller min_t0.",
            diagnostics={"min_t0": int(min_t0)},
        )
    return keep


def period_table(fits: List[Dict[str, Any]], periods: np.ndarray) -> np.ndarray:
    """(periods x replicates) matrix of by-period effects; NaN when absent."""
    M = np.full((len(periods), len(fits)), np.nan)
    for b, f in enumerate(fits):
        lookup = dict(zip(np.asarray(f["time"]).tolist(), np.asarray(f["att"])))
        for i, k in enumerate(periods.tolist()):
            if k in lookup:
                M[i, b] = lookup[k]
    return M


def resample(
    fit: Callable[..., Dict[str, Any]],
    Yz: np.ndarray,
    D: np.ndarray,
    Iobs: np.ndarray,
    X: Optional[np.ndarray],
    hide: Optional[np.ndarray],
    tr: np.ndarray,
    co: np.ndarray,
    vce: str,
    n_boot: int,
    rng: np.random.Generator,
    skip: Tuple[type, ...],
) -> List[Tuple[int, np.ndarray, Dict[str, Any]]]:
    """fect's unit resampling: ``(key, columns, fit)`` per successful draw.

    ``vce='bootstrap'`` resamples treated and control units separately with
    replacement (key = draw number); ``'jackknife'`` leaves out each unit
    once (key = the unit left out). ``fit(Yb, Db, Ib, Xb, hide_b)`` refits.
    """
    N = Yz.shape[1]
    draws = n_boot if vce == "bootstrap" else N
    out: List[Tuple[int, np.ndarray, Dict[str, Any]]] = []
    for b in range(draws):
        if vce == "bootstrap":
            idx = np.concatenate(
                [
                    rng.choice(tr, size=len(tr), replace=True),
                    (
                        rng.choice(co, size=len(co), replace=True)
                        if len(co)
                        else np.zeros(0, int)
                    ),
                ]
            ).astype(int)
        else:
            idx = np.array([j for j in range(N) if j != b])
        Db, Ib = D[:, idx], Iobs[:, idx]
        if ((Db * Ib).sum(axis=0) > 0).sum() == 0 or (Db == 0).sum() == 0:
            continue
        try:
            fb = fit(
                Yz[:, idx],
                Db,
                Ib,
                X[:, idx, :] if X is not None else None,
                None if hide is None else hide[:, idx],
            )
        except skip:
            continue
        if np.isfinite(fb["att_avg"]):
            out.append((b, idx, fb))
    return out


def run_loo(
    refit: Callable[..., Dict[str, Any]],
    Yz: np.ndarray,
    D: np.ndarray,
    Iobs: np.ndarray,
    X: Optional[np.ndarray],
    T_on: np.ndarray,
    periods: Sequence[int],
    *,
    vce: str,
    n_boot: int,
    rng: np.random.Generator,
    min_t0: int,
    alpha: float,
    skip: Tuple[type, ...],
) -> Dict[str, Any]:
    """fect's leave-one-period-out loop over the pre-treatment periods.

    For each period ``k`` the cells with relative period ``k`` leave the
    fit, units left with fewer than ``min_t0`` fitted periods are dropped,
    and the effect at ``k`` and its resampling SE are recorded. Replicate
    columns are keyed by the unit left out (jackknife, original unit index)
    or the draw number (bootstrap), as fect stores ``pre.att.boot``.
    """
    N = Yz.shape[1]
    n_cols = N if vce == "jackknife" else n_boot
    periods = sorted(int(k) for k in periods)
    att = np.full(len(periods), np.nan)
    se = np.full(len(periods), np.nan)
    count = np.zeros(len(periods), dtype=int)
    reps = np.full((len(periods), n_cols), np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    # fect walks the periods from the latest to the earliest
    for i in sorted(range(len(periods)), key=lambda i_: -periods[i_]):
        k = periods[i]
        hide = (Iobs == 1) & (T_on == k)
        II = ((Iobs == 1) & (D == 0) & ~hide).astype(int)
        T0 = II.sum(axis=0)
        treated = (D * Iobs).sum(axis=0) > 0
        keep = T0 >= min_t0
        if not keep[treated].any():
            continue
        cols = np.flatnonzero(keep)
        Yk, Dk, Ik, hk = Yz[:, cols], D[:, cols], Iobs[:, cols], hide[:, cols]
        Xk = X[:, cols, :] if X is not None else None
        fit = refit(Yk, Dk, Ik, Xk, hk)
        lookup = dict(zip(np.asarray(fit["time"]).tolist(), range(len(fit["time"]))))
        if k not in lookup:
            continue
        att[i] = float(fit["att"][lookup[k]])
        count[i] = int(fit["count"][lookup[k]])
        trk = fit["treated_units"]
        cok = np.array([j for j in range(len(cols)) if j not in set(trk.tolist())])
        for key, _, fb in resample(
            refit, Yk, Dk, Ik, Xk, hk, trk, cok, vce, n_boot, rng, skip
        ):
            lk = dict(zip(np.asarray(fb["time"]).tolist(), np.asarray(fb["att"])))
            if k in lk:
                col = int(cols[key]) if vce == "jackknife" else int(key)
                reps[i, col] = lk[k]
        se[i] = replicate_se(reps[i], vce)
    pvalue = 2 * stats.norm.sf(np.abs(att) / se)
    table = {
        "period": np.asarray(periods),
        "att": att,
        "se": se,
        "ci_lower": att - z * se,
        "ci_upper": att + z * se,
        "pvalue": pvalue,
        "count": count,
    }
    return {"table": table, "reps": reps}
