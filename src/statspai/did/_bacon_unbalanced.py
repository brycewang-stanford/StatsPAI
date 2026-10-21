"""Goodman-Bacon decomposition on an unbalanced panel (Stata ``ddtiming``).

``sp.bacon_decomposition(balance='observed_rows')``. Theorem 1 of
Goodman-Bacon (2021) is an identity for a balanced panel; on an unbalanced
one it is applied to the observed rows, the rule of Stata ``ddtiming``
(Goldring, version 0.2), established black-box (its help file and output on
the panels of ``tests/reference_parity/_fixtures/bacon_ddtiming_data.csv``;
its source was not read):

* a unit's timing group is the first period in which it is observed treated;
  a unit treated in its first observed period is already treated (it serves
  only as a control), whatever that period is;
* each 2x2 estimate is the two-way fixed-effects coefficient on the observed
  rows of the two groups inside Theorem 1's window (every period against
  never- or already-treated units; periods before the later group's date for
  earlier-vs-later; from the earlier group's date for later-vs-earlier);
* each weight is Theorem 1's expression with the group share ``n_k`` the
  group's share of all observed rows and ``D_k`` the treated share of the
  group's observed rows, normalised over all pairs.

The weights then need not be positive (``D_k - D_l`` can change sign when
rows are missing), a 2x2 can have no treatment variation left after its
fixed effects (``ddtiming`` reports 0 there and keeps the weight; here the
estimate is NaN, ``identified`` is False, and the pair adds nothing to
``weighted_sum``, which is ``ddtiming``'s ``e(dd)``), and ``weighted_sum``
is not the TWFE coefficient.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

#: Timing-group codes of the never- and the already-treated units.
NEVER = np.inf
ALWAYS = -np.inf


def twfe_coefficient(
    y: np.ndarray, d: np.ndarray, unit: np.ndarray, period: np.ndarray
) -> float:
    """Coefficient on ``d`` with unit and period effects, any balance.

    Exact: the unit effects are removed by within-unit demeaning, the period
    effects by least squares on the within-unit-demeaned period dummies
    (the panels this runs on have few periods). NaN when ``d`` has no
    variation left after the two sets of effects.
    """
    _, u = np.unique(unit, return_inverse=True)
    _, t = np.unique(period, return_inverse=True)
    n_u = int(u.max()) + 1

    def within(M: np.ndarray) -> np.ndarray:
        cnt = np.bincount(u, minlength=n_u).astype(float)
        sums = np.vstack(
            [np.bincount(u, weights=M[:, j], minlength=n_u) for j in range(M.shape[1])]
        ).T
        return M - (sums / cnt[:, None])[u]

    T = np.zeros((len(t), int(t.max()) + 1))
    T[np.arange(len(t)), t] = 1.0
    Tw = within(T[:, 1:]) if T.shape[1] > 1 else np.zeros((len(t), 0))
    V = within(np.column_stack([y, d]).astype(float))
    if Tw.shape[1]:
        V = V - Tw @ np.linalg.lstsq(Tw, V, rcond=None)[0]
    ry, rd = V[:, 0], V[:, 1]
    ss = float(rd @ rd)
    if ss <= 1e-10 * max(float(d @ d), 1.0):
        return float("nan")
    return float(rd @ ry) / ss


def timing_groups(df: pd.DataFrame, treat: str, time: str, id_col: str) -> pd.Series:
    """Per-unit timing group: first observed treated period, ALWAYS or NEVER."""
    first_obs = df.groupby(id_col)[time].min()
    treated = df.loc[df[treat] == 1]
    first_treated = treated.groupby(id_col)[time].min()
    groups = first_treated.reindex(first_obs.index).astype(float)
    groups[groups.isna()] = NEVER
    groups[first_treated.reindex(first_obs.index) == first_obs] = ALWAYS
    return groups


def _pair_weight(n: Dict, D: Dict, treated: float, control: float) -> float:
    """Theorem 1's (unnormalised) weight of one 2x2 from shares and means."""
    if control in (NEVER, ALWAYS):
        nk, nc = n[treated], n[control]
        s = nk / (nk + nc)
        return (nk + nc) ** 2 * s * (1 - s) * D[treated] * (1 - D[treated])
    early, late = (treated, control) if treated < control else (control, treated)
    nk, nl, Dk, Dl = n[early], n[late], D[early], D[late]
    s = nk / (nk + nl)
    if treated < control:  # earlier treated vs later (not yet treated) control
        return (
            ((nk + nl) * (1 - Dl)) ** 2
            * s
            * (1 - s)
            * (Dk - Dl)
            / (1 - Dl)
            * (1 - Dk)
            / (1 - Dl)
        )
    return ((nk + nl) * Dk) ** 2 * s * (1 - s) * (Dl / Dk) * (Dk - Dl) / Dk


def decompose_observed_rows(
    df: pd.DataFrame, y: str, treat: str, time: str, id_col: str
) -> Tuple[pd.DataFrame, float]:
    """The 2x2 table (weights unnormalised) and the TWFE coefficient."""
    groups = timing_groups(df, treat, time, id_col)
    g = df[id_col].map(groups).to_numpy(dtype=float)
    yv = df[y].to_numpy(dtype=float)
    dv = df[treat].to_numpy(dtype=float)
    unit = df[id_col].to_numpy()
    period = df[time].to_numpy()
    n_rows = float(len(df))
    labels = sorted(set(g.tolist()))
    n = {k: float((g == k).sum()) / n_rows for k in labels}
    D = {k: float(dv[g == k].mean()) for k in labels}
    beta_twfe = twfe_coefficient(yv, dv, unit, period)

    rows: List[Dict] = []
    for control in labels:
        for treated in labels:
            if treated == control or treated in (NEVER, ALWAYS):
                continue
            mask = (g == treated) | (g == control)
            if control == NEVER:
                kind, ctrl = "Treated vs Untreated", "Never"
            elif control == ALWAYS:
                kind, ctrl = "Later vs Always Treated", "Always"
            elif treated < control:
                kind, ctrl = "Earlier vs Later Treated", control
                mask &= period < control
            else:
                kind, ctrl = "Later vs Earlier Treated", control
                mask &= period >= control
            est = twfe_coefficient(yv[mask], dv[mask], unit[mask], period[mask])
            rows.append(
                {
                    "type": kind,
                    "treated": treated,
                    "control": ctrl,
                    "estimate": est,
                    "weight": _pair_weight(n, D, treated, control),
                    "identified": bool(np.isfinite(est)),
                    "time_lower": period[mask].min(),
                    "time_upper": period[mask].max(),
                }
            )
    return pd.DataFrame(rows), beta_twfe
