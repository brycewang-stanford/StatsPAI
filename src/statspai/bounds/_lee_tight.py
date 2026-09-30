"""Covariate-tightened Lee (2009) bounds.

With discrete covariates ``X`` the trimming is done cell by cell: in cell
``x`` the arm with higher retention is trimmed by its own share
``p_x = (q_hi,x - q_lo,x) / q_hi,x`` and the trimmed means are averaged with
weights ``Pr(X = x | selected, lower-retention arm)`` -- under
monotonicity the retained members of the lower-retention arm are exactly
the always-observed, whose covariate distribution the bounds refer to
[@lee2009training]. The other arm contributes its
overall retained mean. The bounds can only narrow: averaging cell-specific
trimmed means over the always-observed never widens the interval.

This is what Stata's ``leebounds, tight()`` computes (Tauchmann, SSC
v1.5); with ``trimming='leebounds'`` the point estimates agree to the last
digit. Its analytic variance is an approximation: cell variances combined
with squared weights, a between-cell term divided by the retained count of
the trimmed arm (``ntall``), and the variance of the overall control mean,
ignoring the covariance between the estimated weights and that mean.
``ntall`` is only reassigned inside ``leebounds``' tie branch, so it holds
the last tied context's count -- and when no tie occurred the expression
reads ``between / + vc``. ``trimming='leebounds'`` reproduces that number
for number (and records which case applied); the other trimmings use the
bootstrap, which covers the whole procedure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def _trimmed(
    y: np.ndarray, p: float, top: bool, trimming: str
) -> Tuple[float, float, np.ndarray, np.ndarray, bool]:
    """(mean, threshold, kept values, weights, tie branch fired)."""
    from .lee_manski import _lee_trimmed, _stata_local_macro, _stata_pctile

    y = np.sort(np.asarray(y, dtype=float))
    if trimming != "leebounds":
        if p <= 0:
            return (
                float(y.mean()),
                float(y[0] if top else y[-1]),
                y,
                np.ones(len(y)),
                False,
            )
        mean, thr, vals, w = _lee_trimmed(y, p, top=top, trimming=trimming)
        return mean, thr, vals, w, bool(np.any(w != 1.0))
    # leebounds (leetbound): trim through local macros; out-of-range shares
    # use the sample extreme as the threshold.
    n = len(y)
    trim = _stata_local_macro(100.0 * p)
    pct = trim if top else 100.0 - trim
    if 0 < pct < 100:
        raw_thr = _stata_pctile(y, pct)
    else:
        raw_thr = float(y.min() if pct <= 0 else y.max())
    thr = _stata_local_macro(raw_thr)
    at = y == thr
    if not at.any():
        keep = y >= thr if top else y <= thr
        vals = y[keep]
        w = np.ones(len(vals))
        return float(vals.mean()), raw_thr, vals, w, False
    beyond = y > thr if top else y < thr
    n_beyond = int(beyond.sum())
    frac = (n * (1.0 - trim / 100.0) - n_beyond) / int(at.sum())
    vals = np.concatenate([y[beyond], y[at]])
    w = np.concatenate([np.ones(n_beyond), np.full(int(at.sum()), frac)])
    return float(np.sum(vals * w) / np.sum(w)), thr, vals, w, True


def tightened_lee_bounds(
    y: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    cells: np.ndarray,
    trimming: str,
    analytic: bool = False,
) -> Dict[str, Any]:
    """Tightened bounds (and, for ``trimming='leebounds'``, Stata's variance).

    ``y`` is the outcome for every unit (NaN where not selected), ``D`` the
    0/1 treatment, ``S`` the 0/1 selection indicator and ``cells`` integer
    cell codes.
    """
    D = np.asarray(D, dtype=float)
    S = np.asarray(S, dtype=float)
    cells = np.asarray(cells)
    q1, q0 = S[D == 1].mean(), S[D == 0].mean()
    if q1 == q0:
        raise MethodIncompatibility(
            "lee_bounds: retention is identical in both arms; the effect is "
            "point identified and there is nothing to tighten."
        )
    trim_treated = q1 > q0
    hi = (D == 1) if trim_treated else (D == 0)  # arm that is trimmed
    lo = ~hi
    codes = np.unique(cells)  # leebounds loops cells 1..ncat in order
    sizes = pd.crosstab(cells, D)
    if sizes.shape[1] < 2 or (sizes.to_numpy() <= 0).any():
        raise MethodIncompatibility(
            "lee_bounds: some covariate cells lack treated or control units; "
            "use coarser covariates (Stata: 'cells without variation in "
            "treatment')."
        )
    sel_rates = {}
    for c in codes:
        m = cells == c
        r_hi, r_lo = S[m & hi].mean(), S[m & lo].mean()
        if r_hi == 0 or r_lo == 0:
            raise MethodIncompatibility(
                "lee_bounds: some covariate cells have no selected outcome in "
                "one arm; use coarser covariates."
            )
        sel_rates[c] = (r_hi, r_lo)
    diffs = np.array([a - b for a, b in sel_rates.values()])
    hetero = bool(diffs.min() < 0 < diffs.max())

    y_lo = y[lo & (S == 1)]
    rows: List[Dict[str, Any]] = []
    # ``ntall`` as leebounds leaves it: the treated-retained count of the
    # last context whose tie branch fired (overall first, then cell by cell,
    # upper before lower).
    ntall: Optional[float] = None
    p_all = (S[hi].mean() - S[lo].mean()) / S[hi].mean()
    y_hi_all = y[hi & (S == 1)]
    for top in (True, False):
        if _trimmed(y_hi_all, p_all, top, trimming)[4]:
            ntall = float(len(y_hi_all))
    for c in codes:
        m = cells == c
        r_hi, r_lo = sel_rates[c]
        p = (r_hi - r_lo) / r_hi
        y_hi = y[m & hi & (S == 1)]
        if trimming != "leebounds":
            p = max(p, 0.0)
        up = _trimmed(y_hi, p, True, trimming)
        dn = _trimmed(y_hi, p, False, trimming)
        for part in (up, dn):
            if part[4]:
                ntall = float(len(y_hi))
        row: Dict[str, Any] = dict(
            cell=c,
            n=int(m.sum()),
            weight=float(np.sum(m & lo & (S == 1))),
            trim=float(p),
            mean_top=up[0],
            mean_bottom=dn[0],
            retention_trimmed_arm=float(r_hi),
            retention_other_arm=float(r_lo),
        )
        if analytic:
            nall = float(m.sum())
            hi_c = m & hi
            est = np.sum(hi_c & (S == 1)) / nall
            esnt = np.sum(m & lo & (S == 1)) / nall
            et = np.sum(hi_c) / nall
            oddsc = np.sum(m & lo & (S == 0)) / nall / esnt
            oddst = np.sum(hi_c & (S == 0)) / nall / est
            vp = (1 - p) ** 2 * (oddst / (et * nall) + oddsc / ((1 - et) * nall))
            for key, part in (("var_top", up), ("var_bottom", dn)):
                mu, thr, vals, w, tie = part
                if tie:
                    mass = np.sum(w)
                    vb1 = (np.sum(w * vals**2) / mass - mu**2) / mass
                else:
                    vb1 = np.var(vals, ddof=1) / len(vals)
                vb2 = (thr - mu) ** 2 * p / ((1 - p) * est * nall)
                vb3 = ((thr - mu) / (1 - p)) ** 2 * vp
                row[key] = float(vb1 + vb2 + vb3)
        rows.append(row)

    cells_df = pd.DataFrame(rows)
    w = cells_df["weight"].to_numpy() / cells_df["weight"].sum()
    m_lo = float(np.mean(y_lo))
    t_top = float(w @ cells_df["mean_top"].to_numpy())
    t_bot = float(w @ cells_df["mean_bottom"].to_numpy())
    out: Dict[str, Any] = dict(
        cells=cells_df.assign(weight=w),
        hetero=hetero,
        trimmed_arm="treatment" if trim_treated else "control",
    )
    if trim_treated:
        out["lower"], out["upper"] = t_bot - m_lo, t_top - m_lo
    else:
        out["lower"], out["upper"] = m_lo - t_top, m_lo - t_bot
    if analytic:
        vc = float(np.var(y_lo, ddof=1) / len(y_lo))

        def _var(mean_col: str, var_col: str) -> float:
            tm = cells_df[mean_col].to_numpy()
            within = float(np.sum(w**2 * cells_df[var_col].to_numpy()))
            between = float(w @ tm**2 - (w @ tm) ** 2)
            if ntall is None:
                return within + between / vc  # leebounds: "/`ntall'+`vc'", ntall empty
            return within + between / ntall + vc

        v_top, v_bot = _var("mean_top", "var_top"), _var("mean_bottom", "var_bottom")
        out["var_lower"], out["var_upper"] = (
            (v_bot, v_top) if trim_treated else (v_top, v_bot)
        )
        out["ntall"] = ntall
    return out


def cell_codes(data: pd.DataFrame, covariates: List[str]) -> np.ndarray:
    """Integer cell codes for the covariate combinations (``egen group()``)."""
    return data.groupby(list(covariates), sort=True).ngroup().to_numpy()
