"""Exact partial likelihood for tied failure times.

When ``d`` subjects fail at the same recorded time and the ties come from
grouping (time is continuous, the clock is coarse), the probability that
exactly these ``d`` are the ones that fail, given the risk set ``R`` and
that ``d`` of its members fail, is

    exp(sum_{i in D} x_i' b) / e_d(r_k : k in R),    r_k = exp(x_k' b),

with ``e_d`` the elementary symmetric polynomial of degree ``d`` (the sum,
over all subsets of ``R`` of size ``d``, of the product of their risk
scores). This is the conditional-logistic likelihood of the event time; R
``coxph(ties = "exact")`` and Stata ``stcox, exactp`` maximise it. Efron
and Breslow are approximations to it that agree with it when no times are
tied.

``e_d`` and its first two derivatives are built by the recurrence over the
members of the risk set

    e_j(1..i) = e_j(1..i-1) + r_i e_{j-1}(1..i-1),

run level by level in ``j`` with cumulative sums over ``i``, so an event
time costs ``O(|R| d p^2)`` instead of the ``C(|R|, d)`` of the
definition. Risk scores are rescaled within each risk set (``e_d`` is
homogeneous of degree ``d``) so that the polynomial stays inside double
range for up to several hundred tied failures.

Event times with a single failure are handled by the ordinary kernel
(:class:`statspai.survival._cox_core.CoxKernel`), where the three rules
coincide.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility, NumericalInstability
from ._cox_core import CoxKernel

#: Refuse a fit whose tied event times need more than this many
#: ``|R| * d * p^2`` operations per likelihood evaluation.
MAX_EXACT_WORK = 4e9


def check_exact_options(
    *, weights: Any, robust: Any, cluster: Any, design: Any
) -> None:
    """Refuse the options the exact partial likelihood does not define."""
    asked = []
    if weights is not None:
        asked.append("weights=")
    if cluster is not None:
        asked.append("cluster=")
    elif robust is not None and str(robust).lower() not in ("nonrobust",):
        asked.append(f"robust={robust!r}")
    if design is not None:
        asked.append("design=")
    if asked:
        raise MethodIncompatibility(
            "sp.cox: ties='exact' is the conditional likelihood of each event "
            f"time and does not take {', '.join(asked)}. Case weights and "
            "per-subject score residuals (which the robust and clustered "
            "variances are built from) are not defined for it; R coxph and "
            "Stata stcox, exactp refuse them too.",
            recovery_hint=(
                "Use ties='efron' with these options, or ties='exact' with the "
                "model-based variance and no weights."
            ),
            diagnostics={"options": asked},
        )


class ExactCoxKernel:
    """Exact partial likelihood on fixed risk sets.

    Parameters mirror :class:`CoxKernel`; case weights are not defined for
    the exact likelihood.
    """

    def __init__(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        strata: Optional[np.ndarray] = None,
        start: Optional[np.ndarray] = None,
    ):
        X = np.asarray(X, dtype=float)
        T = np.asarray(T, dtype=float)
        dead = np.asarray(E) == 1
        n, p = X.shape
        self.p = p
        self.center = X.mean(axis=0) if n else np.zeros(p)
        self.Xc = X - self.center
        if strata is None:
            codes = np.zeros(n, dtype=np.int64)
        else:
            codes = pd.factorize(np.asarray(strata), sort=True)[0].astype(np.int64)
        start_arr = None if start is None else np.asarray(start, dtype=float)

        # event times with two or more failures, per stratum
        tied: List[Tuple[np.ndarray, np.ndarray]] = []
        single = dead.copy()
        work = 0.0
        frame = pd.DataFrame(
            {"s": codes[dead], "t": T[dead], "i": np.flatnonzero(dead)}
        )
        for (s, t), grp in frame.groupby(["s", "t"], sort=True):
            if len(grp) < 2:
                continue
            at_risk = (codes == s) & (T >= t)
            if start_arr is not None:
                at_risk &= start_arr < t
            risk_idx = np.flatnonzero(at_risk)
            dead_idx = grp["i"].to_numpy()
            tied.append((risk_idx, dead_idx))
            single[dead_idx] = False
            work += float(len(risk_idx)) * len(dead_idx) * max(p, 1) ** 2
        if work > MAX_EXACT_WORK:
            raise MethodIncompatibility(
                "sp.cox: ties='exact' is too expensive on these data (the tied "
                f"event times need about {work:.1e} operations per likelihood "
                "evaluation).",
                recovery_hint=(
                    "Use ties='efron', which approximates the exact likelihood "
                    "closely unless a large share of each risk set fails at "
                    "once; for genuinely discrete time fit a discrete-time "
                    "model (sp.logit on person-period data)."
                ),
                diagnostics={"work": work, "n_tied_times": len(tied)},
            )
        self.tied = tied
        self.n_tied_times = len(tied)
        self.max_tied = max((len(d) for _, d in tied), default=1)
        # single-failure times: the ordinary kernel on the same risk sets
        # (rows keep their times, so the risk sets are unchanged)
        self._single = CoxKernel(
            X, T, single.astype(float), strata, breslow=True, start=start_arr
        )
        self._any_single = bool(single.any())

    # ------------------------------------------------------------------
    def _tied_group(
        self, beta: np.ndarray, risk_idx: np.ndarray, dead_idx: np.ndarray, deriv: bool
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Log-likelihood, score and Hessian contribution of one tied time."""
        Xr = self.Xc[risk_idx]
        xb = Xr @ beta
        d = len(dead_idx)
        shift = float(xb.max())
        r = np.exp(xb - shift)
        scale = float(r.sum()) / d
        r = r / scale
        m = len(r)
        e_prev = np.ones(m + 1)
        if deriv:
            p = self.p
            d_prev = np.zeros((m + 1, p))
            h_prev = np.zeros((m + 1, p, p))
            xx = Xr[:, :, None] * Xr[:, None, :]
        for _ in range(d):
            a = r * e_prev[:-1]
            e_new = np.concatenate([[0.0], np.cumsum(a)])
            if deriv:
                d_lag = d_prev[:-1]
                x_d = Xr[:, :, None] * d_lag[:, None, :]
                b_h = a[:, None, None] * xx + r[:, None, None] * (
                    x_d + x_d.transpose(0, 2, 1) + h_prev[:-1]
                )
                h_prev = np.concatenate([np.zeros((1, p, p)), np.cumsum(b_h, axis=0)])
                b_d = a[:, None] * Xr + r[:, None] * d_lag
                d_prev = np.concatenate([np.zeros((1, p)), np.cumsum(b_d, axis=0)])
            e_prev = e_new
        e_d = float(e_prev[-1])
        if not (np.isfinite(e_d) and e_d > 0):
            raise NumericalInstability(
                "sp.cox: the exact partial likelihood left double range at an "
                f"event time with {d} tied failures among {m} at risk.",
                recovery_hint="Use ties='efron' for data with this many ties.",
            )
        xb_dead = self.Xc[dead_idx] @ beta
        ll = float(xb_dead.sum()) - d * (shift + np.log(scale)) - np.log(e_d)
        if not deriv:
            return ll, None, None
        g = d_prev[-1] / e_d
        score = self.Xc[dead_idx].sum(axis=0) - g
        hess = -(h_prev[-1] / e_d - np.outer(g, g))
        return ll, score, hess

    def neg_loglik(self, beta: np.ndarray) -> float:
        """Minus the exact log partial likelihood."""
        beta = np.asarray(beta, dtype=float)
        total = -self._single.neg_loglik(beta) if self._any_single else 0.0
        for risk_idx, dead_idx in self.tied:
            total += self._tied_group(beta, risk_idx, dead_idx, deriv=False)[0]
        return -float(total)

    def score_hessian(self, beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Score and Hessian of the exact log partial likelihood."""
        beta = np.asarray(beta, dtype=float)
        if self._any_single:
            score, hess = self._single.score_hessian(beta)
            score, hess = score.copy(), hess.copy()
        else:
            score, hess = np.zeros(self.p), np.zeros((self.p, self.p))
        for risk_idx, dead_idx in self.tied:
            _, s_g, h_g = self._tied_group(beta, risk_idx, dead_idx, deriv=True)
            score += s_g
            hess += h_g
        return score, hess
