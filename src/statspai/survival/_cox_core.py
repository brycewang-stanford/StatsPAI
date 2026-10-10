"""Risk-set arithmetic for the Cox partial likelihood, in O(n log n).

Every quantity of the partial likelihood is a sum over risk sets
``R(t) = {k : T_k >= t}`` of the same stratum. Sorting each stratum by
descending time turns those sums into running totals, so the likelihood,
its derivatives, the score residuals, the baseline hazard and the pieces
of the proportional-hazards test all cost one sort and a few passes over
the data, whatever the number of distinct event times. (Recomputing each
risk set from scratch is quadratic when times are continuous: 20,000
observations took two minutes.)

Two reparametrisations keep the running totals accurate; the partial
likelihood is invariant to both, so neither changes any reported number
beyond rounding:

* the columns of ``X`` are centred, which removes the cancellation in
  ``S2 / S0 - xbar xbar'`` for regressors with a large mean (a year);
* within each stratum the linear index is shifted so that its maximum is
  zero, which keeps ``exp`` from overflowing and keeps the totals of
  different strata on one scale (strata are separated by subtracting the
  running total at the start of the stratum).

The information matrix never needs the ``p x p`` array of each risk set:
``sum_g a_g S2_g`` with ``S2_g = sum_{k in R_g} r_k x_k x_k'`` equals
``X' diag(r_k A_k) X`` with ``A_k`` the sum of ``a_g`` over the risk sets
that contain ``k``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


class RiskSets:
    """Sorted layout of the risk sets of one sample.

    Rows are ordered by stratum, then by descending time; a *group* is a
    run of rows with the same stratum and time. Row ``k`` is at risk at
    group ``g`` of its stratum iff ``g >= group[k]``.
    """

    def __init__(
        self, T: Any, E: Any, strata: Optional[Any] = None, order: str = "sorted"
    ):
        T = np.asarray(T, dtype=float)
        dead = np.asarray(E) == 1
        n = len(T)
        if strata is None:
            codes = np.zeros(n, dtype=np.int64)
            n_strata = 1
        else:
            codes, uniques = pd.factorize(np.asarray(strata), sort=order == "sorted")
            codes = codes.astype(np.int64)
            n_strata = len(uniques)
        # lexsort is stable: rows tied on (stratum, time) keep data order
        self.order = np.lexsort((-T, codes))
        self.n = n
        self.n_strata = n_strata
        self.T = T[self.order]
        self.dead = dead[self.order]
        self.codes = codes[self.order]

        new = np.ones(n, dtype=bool)
        if n > 1:
            new[1:] = (self.codes[1:] != self.codes[:-1]) | (self.T[1:] != self.T[:-1])
        self.start = np.flatnonzero(new)  # first row of each group
        self.group = np.cumsum(new) - 1  # group of each row
        self.n_groups = len(self.start)
        self.end = np.append(self.start[1:], n) - 1  # last row of each group
        self.g_time = self.T[self.start]
        self.g_code = self.codes[self.start]
        self.d = (
            np.add.reduceat(self.dead.astype(np.int64), self.start)
            if n
            else np.zeros(0, np.int64)
        )

        # first row / first and last group of the stratum of each group
        s_new = np.ones(self.n_groups, dtype=bool)
        if self.n_groups > 1:
            s_new[1:] = self.g_code[1:] != self.g_code[:-1]
        s_first_group = np.flatnonzero(s_new)
        s_of_group = np.cumsum(s_new) - 1
        self.s_first_row = self.start[s_first_group][s_of_group]
        s_last_group = np.append(s_first_group[1:], self.n_groups) - 1
        self.s_last_group = s_last_group[s_of_group]
        self._s_first_group = s_first_group
        self._s_of_group = s_of_group

        # the Efron terms: one entry per death, (event group, l = 0..d-1)
        self.ev = np.flatnonzero(self.d > 0)  # event groups
        d_ev = self.d[self.ev]
        self.term_group = np.repeat(self.ev, d_ev)
        first = np.cumsum(d_ev) - d_ev
        self.term_ell = np.arange(int(d_ev.sum())) - np.repeat(first, d_ev)
        self.term_d = np.repeat(d_ev, d_ev)
        self.term_start = first  # first term of each event group

    # -- running totals -------------------------------------------------
    def at_risk(self, values: np.ndarray) -> np.ndarray:
        """Sum of ``values`` (sorted rows) over the risk set of each group."""
        c = np.cumsum(values, axis=0)
        out = c[self.end]
        if self.n_strata > 1:
            first = self.s_first_row
            has = first > 0
            out = out.copy()
            out[has] -= c[first[has] - 1]
        return np.asarray(out)

    def over_groups(self, values: np.ndarray) -> np.ndarray:
        """Sum of ``values`` (sorted rows) within each group."""
        return np.add.reduceat(values, self.start, axis=0)

    def containing(self, per_group: np.ndarray) -> np.ndarray:
        """For each row, the sum of ``per_group`` over the groups at whose
        time the row is at risk (its own group and the later-sorted ones of
        its stratum, i.e. the earlier times)."""
        # accumulate from the far end of each stratum, smallest terms first:
        # with one stratum there is no subtraction at all
        rc = np.cumsum(per_group[::-1], axis=0)[::-1]
        if self.n_strata > 1:
            nxt = self.s_last_group + 1
            has = nxt < self.n_groups
            rc = rc.copy()
            rc[has] -= rc[nxt[has]]
        return np.asarray(rc[self.group])

    def stratum_scale(self, xb_sorted: np.ndarray) -> np.ndarray:
        """Per-row maximum of the linear index within its stratum."""
        if self.n_strata == 1:
            return np.full(self.n, xb_sorted.max() if self.n else 0.0)
        row_first = self.start[self._s_first_group]
        m = np.maximum.reduceat(xb_sorted, row_first)
        return np.asarray(m[self._s_of_group][self.group])


class CoxKernel:
    """Partial likelihood of one design on fixed risk sets."""

    def __init__(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        strata: Optional[np.ndarray] = None,
        breslow: bool = False,
        order: str = "sorted",
    ):
        self.rs = RiskSets(T, E, strata, order=order)
        X = np.asarray(X, dtype=float)
        self.X_raw = X[self.rs.order]
        self.center = X.mean(axis=0) if len(X) else np.zeros(X.shape[1])
        self.X = self.X_raw - self.center
        self.breslow = breslow
        self.p = X.shape[1]
        rs = self.rs
        self.c = np.zeros(len(rs.term_ell)) if breslow else rs.term_ell / rs.term_d
        self.dead_sum_x = self.X[rs.dead].sum(axis=0)

    # -- building blocks -------------------------------------------------
    def _risk(self, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scaled risk scores, the scaled index, and the per-row scale."""
        xb = self.X @ beta
        m = self.rs.stratum_scale(xb)
        xbn = xb - m
        return np.exp(xbn), xbn, m

    def _terms(self, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``S0`` and ``D0`` per group and the denominator of each term."""
        rs = self.rs
        S0 = rs.at_risk(r)
        D0 = rs.over_groups(r * rs.dead)
        denom = S0[rs.term_group] - self.c * D0[rs.term_group]
        return S0, D0, denom

    def _xbar(self, r: np.ndarray, denom: np.ndarray) -> np.ndarray:
        """Risk-set mean of ``x`` for each term, ``(n_deaths, p)``."""
        rs = self.rs
        rx = self.X * r[:, None]
        S1 = rs.at_risk(rx)
        num = S1[rs.term_group]
        if not self.breslow:
            D1 = rs.over_groups(rx * rs.dead[:, None])
            num = num - self.c[:, None] * D1[rs.term_group]
        return np.asarray(num / denom[:, None])

    def _per_event_group(self, per_term: np.ndarray) -> np.ndarray:
        """Sum the terms of each event group; zero for the other groups."""
        rs = self.rs
        out = np.zeros((rs.n_groups,) + per_term.shape[1:])
        if len(per_term):
            out[rs.ev] = np.add.reduceat(per_term, rs.term_start, axis=0)
        return out

    # -- likelihood -------------------------------------------------------
    def neg_loglik(self, beta: np.ndarray) -> float:
        r, xbn, _ = self._risk(np.asarray(beta, dtype=float))
        _, _, denom = self._terms(r)
        return float(np.log(denom).sum() - xbn[self.rs.dead].sum())

    def score_hessian(
        self, beta: np.ndarray, group_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Score and Hessian of the log partial likelihood.

        ``group_weights`` (one per group) weights each event time's
        contribution to the Hessian; the score is returned unweighted.
        """
        rs = self.rs
        r, _, _ = self._risk(np.asarray(beta, dtype=float))
        _, _, denom = self._terms(r)
        xbar = self._xbar(r, denom)
        score = self.dead_sum_x - xbar.sum(axis=0)

        inv = 1.0 / denom
        w_term = (
            np.ones(len(denom))
            if group_weights is None
            else group_weights[rs.term_group]
        )
        a = self._per_event_group(inv * w_term)
        w_row = r * rs.containing(a)
        if not self.breslow:
            b = self._per_event_group(self.c * inv * w_term)
            w_row = w_row - r * rs.dead * b[rs.group]
        info = (self.X * w_row[:, None]).T @ self.X - (xbar * w_term[:, None]).T @ xbar
        return score, -info

    def information(self, beta: np.ndarray, group_weights: np.ndarray) -> np.ndarray:
        """``sum_g w_g V_g``, the information with event times weighted."""
        return -self.score_hessian(beta, group_weights)[1]

    def score_residuals(self, beta: np.ndarray) -> np.ndarray:
        """Per-observation score residuals, in the original row order."""
        rs = self.rs
        r, _, _ = self._risk(np.asarray(beta, dtype=float))
        _, _, denom = self._terms(r)
        xbar = self._xbar(r, denom)
        inv = 1.0 / denom
        a = self._per_event_group(inv)
        q = self._per_event_group(xbar * inv[:, None])
        H = rs.containing(a)
        XH = rs.containing(q)
        res = -r[:, None] * (self.X * H[:, None] - XH)

        mean_xbar = self._per_event_group(xbar) / np.maximum(rs.d, 1)[:, None]
        dead = rs.dead
        res[dead] += self.X[dead] - mean_xbar[rs.group[dead]]
        if not self.breslow:
            b = self._per_event_group(self.c * inv)
            qc = self._per_event_group(xbar * (self.c * inv)[:, None])
            g = rs.group[dead]
            res[dead] += r[dead, None] * (self.X[dead] * b[g][:, None] - qc[g])
        out = np.empty_like(res)
        out[rs.order] = res
        return np.asarray(out)

    # -- by event time ----------------------------------------------------
    def event_time_scores(
        self, beta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Event groups (ascending time within stratum), their times and
        score contributions, and one Schoenfeld residual per death with its
        group, under the tie rule of the fit."""
        rs = self.rs
        r, _, _ = self._risk(np.asarray(beta, dtype=float))
        _, _, denom = self._terms(r)
        xbar = self._xbar(r, denom)
        xbar_sum = self._per_event_group(xbar)
        dead_x = rs.over_groups(self.X * rs.dead[:, None])
        # ascending time within stratum = reversed group order per stratum
        ev = rs.ev[np.lexsort((rs.g_time[rs.ev], rs.g_code[rs.ev]))]
        scores = dead_x[ev] - xbar_sum[ev]
        mean_xbar = xbar_sum / np.maximum(rs.d, 1)[:, None]
        # deaths: stratum, ascending time, data order within a time
        rows = np.flatnonzero(rs.dead)
        rows = rows[np.lexsort((rs.order[rows], rs.T[rows], rs.codes[rows]))]
        resid = self.X[rows] - mean_xbar[rs.group[rows]]
        return ev, rs.g_time[ev], scores, rs.group[rows], resid

    def breslow_expected_x(self, beta: np.ndarray) -> np.ndarray:
        """``S1 / S0`` of each group (no tie correction), uncentred."""
        r, _, _ = self._risk(np.asarray(beta, dtype=float))
        S0 = self.rs.at_risk(r)
        S1 = self.rs.at_risk(self.X * r[:, None])
        return np.asarray(S1 / S0[:, None] + self.center)

    def breslow_increments(
        self, beta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stratum code, time and ``d / sum_R exp(x'b)`` of each event
        group, ascending in time within stratum."""
        rs = self.rs
        beta = np.asarray(beta, dtype=float)
        r, _, m = self._risk(beta)
        S0 = rs.at_risk(r)
        # undo the scaling and the centring: the hazard is not invariant
        with np.errstate(over="ignore", under="ignore"):
            scale = np.exp(m[rs.start] + self.center @ beta)
        ev = rs.ev[np.lexsort((rs.g_time[rs.ev], rs.g_code[rs.ev]))]
        return rs.g_code[ev], rs.g_time[ev], rs.d[ev] / (S0[ev] * scale[ev])


def dominance_counts(
    ins_pos: np.ndarray,
    ins_val: np.ndarray,
    qry_pos: np.ndarray,
    qry_val: np.ndarray,
) -> Tuple[int, int]:
    """Count pairs (insert a, query b) with ``pos_a < pos_b`` and
    ``val_a < val_b``, and those with ``val_a == val_b``.

    Positions are non-negative integers. Each pair is counted at the one
    bit where the two positions first differ, so the cost is one sort per
    bit of the largest position.
    """
    if len(ins_pos) == 0 or len(qry_pos) == 0:
        return 0, 0
    _, inv = np.unique(np.concatenate([ins_val, qry_val]), return_inverse=True)
    iv = inv[: len(ins_val)].astype(np.int64)
    qv = inv[len(ins_val) :].astype(np.int64)
    R = int(inv.max()) + 1
    ins_pos = ins_pos.astype(np.int64)
    qry_pos = qry_pos.astype(np.int64)
    less = 0
    equal = 0
    top = int(max(ins_pos.max(), qry_pos.max()))
    for k in range(top.bit_length()):
        a = ((ins_pos >> k) & 1) == 0
        b = ((qry_pos >> k) & 1) == 1
        if not a.any() or not b.any():
            continue
        keys = np.sort((ins_pos[a] >> (k + 1)) * R + iv[a])
        base = (qry_pos[b] >> (k + 1)) * R
        lo = np.searchsorted(keys, base, side="left")
        lt = np.searchsorted(keys, base + qv[b], side="left")
        le = np.searchsorted(keys, base + qv[b], side="right")
        less += int((lt - lo).sum())
        equal += int((le - lt).sum())
    return less, equal


def harrell_c(score: np.ndarray, T: np.ndarray, E: np.ndarray) -> float:
    """Harrell's C: among the pairs in which one subject fails while the
    other is known to outlive it (a later time, or censored at the same
    time), the share in which the one that fails has the higher score,
    ties in the score counting one half."""
    T = np.asarray(T, dtype=float)
    dead = np.asarray(E) == 1
    if not dead.any():
        return 0.5
    # process times from the latest: censored at a time are compared with
    # the failures at that time, failures at one time not with each other
    _, g = np.unique(-T, return_inverse=True)
    g = g.astype(np.int64)
    pos = np.where(dead, 3 * g + 2, 3 * g)
    less, equal = dominance_counts(pos, score, 3 * g[dead] + 1, score[dead])
    # comparable pairs: for each failure, everyone inserted before it
    order_pos = np.sort(pos)
    total = int(np.searchsorted(order_pos, 3 * g[dead] + 1, side="left").sum())
    if total == 0:
        return 0.5
    return (less + 0.5 * equal) / total


def km_left_survival(T: np.ndarray, E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Distinct times (ascending) and the Kaplan-Meier estimate just
    before each."""
    T = np.asarray(T, dtype=float)
    dead = np.asarray(E) == 1
    times, inv, counts = np.unique(T, return_inverse=True, return_counts=True)
    n_dead = np.bincount(inv, weights=dead, minlength=len(times))
    n_risk = len(T) - (np.cumsum(counts) - counts)
    surv = np.cumprod(1.0 - n_dead / n_risk)
    left = np.ones(len(times))
    left[1:] = surv[:-1]
    return times, left
