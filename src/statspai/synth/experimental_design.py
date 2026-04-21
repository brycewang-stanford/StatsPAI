"""
Synthetic Controls for Experimental Design — Abadie & Zhao (2025/2026).

Flips the classic synthetic-control workflow: instead of using SC to build
a *post*-treatment counterfactual, use it *before* treatment begins to
decide **which units to treat**.

Given a panel of pre-treatment outcomes for N candidate units and a budget
``k`` (number of units to treat), rank candidates by the quality of the
synthetic control that can be constructed for each one from the remaining
donors.  The theory (Abadie & Zhao 2025/2026, MIT working paper) shows that

    Var[ATT_hat | assignment D] ≈ sum_{i in D} sigma^2_i

where ``sigma^2_i`` is the (feasible) pre-period MSPE of the SC fit for
unit ``i``.  Minimizing this over a budget-``k`` assignment reduces to

    D* = argmin_{|D|=k} sum_{i in D} sigma^2_i

which, under the exchangeability-of-donors assumption, is solved by picking
the ``k`` candidates with the smallest leave-one-out pre-period MSPE.

References
----------
Abadie, A. & Zhao, J. (2025/2026). Synthetic Controls for Experimental
Design. MIT working paper.  *Advances in Economics and Econometrics*
(Cambridge UP, 2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._core import solve_simplex_weights


__all__ = [
    "synth_experimental_design",
    "SynthExperimentalDesignResult",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SynthExperimentalDesignResult:
    """Structured output of :func:`synth_experimental_design`.

    Attributes
    ----------
    selected : list of unit ids
        The ``k`` units recommended for treatment.
    ranking : pandas.DataFrame
        All candidates with columns
        ``[unit, pre_mspe, pre_rmse, effective_donors, risk_score, selected]``
        sorted by ``risk_score`` ascending (best first).
    weights : dict[unit_id, ndarray]
        Leave-one-out SC weight vectors (aligned to ``donor_units``) —
        useful for the post-experiment analysis and for diagnostics.
    donor_units : list
        The donor pool that each candidate was matched against
        (candidates excluded from each other's donor pool by default).
    expected_variance : float
        Sum of pre-period MSPEs over ``selected`` — proxy for the
        post-experiment ATT-variance under Abadie-Zhao 2025/2026 Eq. (3).
    baseline_variance : float
        Same quantity for a random-``k`` assignment (average over
        ``n_random`` draws); the gain is
        ``baseline_variance - expected_variance``.
    method : str
        Always ``'abadie_zhao_2025'``.
    diagnostics : dict
        Extra metadata (n_units, pre_periods, solver, etc.).
    """

    selected: List[Any]
    ranking: pd.DataFrame
    weights: Dict[Any, np.ndarray]
    donor_units: List[Any]
    expected_variance: float
    baseline_variance: float
    method: str = "abadie_zhao_2025"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    def summary(self) -> str:
        n = len(self.ranking)
        k = len(self.selected)
        gain = self.baseline_variance - self.expected_variance
        gain_pct = (
            100 * gain / self.baseline_variance if self.baseline_variance > 0 else 0.0
        )
        lines = [
            "Synthetic Controls for Experimental Design (Abadie-Zhao 2025/2026)",
            "-" * 66,
            f"  Candidates evaluated   : {n}",
            f"  Donor pool size        : {len(self.donor_units)}",
            f"  Treatment budget k     : {k}",
            f"  Selected units         : {list(self.selected)[:10]}"
            + ("..." if k > 10 else ""),
            f"  Expected sum-MSPE      : {self.expected_variance:.6f}",
            f"  Baseline (random k)    : {self.baseline_variance:.6f}",
            f"  Variance reduction     : {gain:.6f}  ({gain_pct:.1f}% below random)",
            "",
            "  Top of ranking:",
        ]
        head = self.ranking.head(min(5, n)).to_string(index=False)
        lines.append(head)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": list(self.selected),
            "expected_variance": float(self.expected_variance),
            "baseline_variance": float(self.baseline_variance),
            "n_candidates": int(len(self.ranking)),
            "n_donors": int(len(self.donor_units)),
            "method": self.method,
            "diagnostics": dict(self.diagnostics),
        }


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _build_wide_panel(
    data: pd.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
) -> pd.DataFrame:
    """Pivot long-format panel into (unit, time) wide matrix."""
    wide = data.pivot_table(index=unit, columns=time, values=outcome, aggfunc="mean")
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    return wide


def _leave_one_out_sc(
    y_i: np.ndarray,
    donor_matrix: np.ndarray,
    penalization: float = 0.0,
) -> Tuple[np.ndarray, float, float]:
    """Fit simplex SC for one candidate against donors.

    Returns
    -------
    w : ndarray (n_donors,)
        Nonneg simplex weights.
    mspe : float
        Pre-period mean squared prediction error.
    eff_donors : float
        Effective donor count ``1 / sum(w^2)`` (inverse Herfindahl).
    """
    # solve_simplex_weights(y, X) fits y ~ X w
    # where X is (T_pre, n_donors) and y is (T_pre,)
    w = solve_simplex_weights(y_i, donor_matrix, penalization=penalization)
    resid = y_i - donor_matrix @ w
    mspe = float(np.mean(resid ** 2))
    herf = float(np.sum(w ** 2))
    eff = float(1.0 / herf) if herf > 0 else float("nan")
    return w, mspe, eff


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def synth_experimental_design(
    data: pd.DataFrame,
    *,
    unit: str,
    time: str,
    outcome: str,
    k: int,
    candidates: Optional[Sequence[Any]] = None,
    donors: Optional[Sequence[Any]] = None,
    pre_period: Optional[Tuple[Any, Any]] = None,
    risk: str = "mspe",
    concentration_weight: float = 0.0,
    penalization: float = 0.0,
    n_random: int = 500,
    random_state: Optional[int] = None,
) -> SynthExperimentalDesignResult:
    """Pick ``k`` treated units to minimize the expected SC post-ATT variance.

    Parameters
    ----------
    data : DataFrame (long format)
        Must contain columns ``[unit, time, outcome]``.
    unit, time, outcome : str
        Column names for the panel.
    k : int
        Number of units to select for treatment.  Must satisfy
        ``1 <= k <= len(candidates) - 1``.
    candidates : sequence, optional
        Units eligible for treatment.  Defaults to all units.
    donors : sequence, optional
        Units available as donors.  Defaults to "all units NOT in
        ``candidates``"; if ``candidates`` covers all units we fall back
        to a leave-one-out protocol where each candidate's donor pool is
        every *other* unit.
    pre_period : (start, end), optional
        Closed interval of pre-treatment periods.  Defaults to all
        timestamps in ``data``.
    risk : {'mspe', 'rmse'}, default 'mspe'
        Loss functional for ranking candidates.
    concentration_weight : float, default 0.0
        Penalty on donor-weight concentration (Herfindahl):
        ``risk_score = loss + lambda * H(w)`` where
        ``H(w) = sum(w_j^2)``.  Abadie-Zhao show that for a fixed
        pre-MSPE, less-concentrated donors give tighter post-period
        confidence intervals.
    penalization : float, default 0.0
        Ridge penalty passed to the simplex solver (Doudchenko &
        Imbens 2016 style).
    n_random : int, default 500
        Monte-Carlo draws used to estimate ``baseline_variance``
        (the expected sum-MSPE under random-``k`` selection).
    random_state : int, optional

    Returns
    -------
    SynthExperimentalDesignResult

    Notes
    -----
    The practical recipe (Abadie-Zhao 2025/2026, Section 4) is:

    1. For each candidate unit ``i``, solve the simplex SC problem against
       the donor pool restricted to **non-candidates** (to avoid coupling
       risk scores across candidates).
    2. Record the pre-period MSPE as the plug-in estimate of
       ``sigma^2_i``.
    3. Pick the ``k`` candidates with the smallest ``risk_score``:
       ``loss_i + lambda * H(w_i)``.

    The implementation degrades gracefully when ``candidates`` covers all
    units: we then use per-candidate *leave-one-out* donor pools.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.utils.dgp_synth(n_units=40, n_periods=20, seed=0)
    >>> res = sp.synth_experimental_design(
    ...     df, unit='unit', time='time', outcome='y',
    ...     k=5, pre_period=(0, 19), random_state=0,
    ... )
    >>> res.selected  # doctest: +SKIP
    [12, 7, 23, 4, 30]
    >>> print(res.summary())  # doctest: +SKIP
    """
    # --- Validation --------------------------------------------------------
    for col in (unit, time, outcome):
        if col not in data.columns:
            raise ValueError(f"column '{col}' not in data")
    if risk not in ("mspe", "rmse"):
        raise ValueError(f"risk must be 'mspe' or 'rmse', got {risk!r}")
    if concentration_weight < 0:
        raise ValueError("concentration_weight must be >= 0")

    # --- Build wide panel --------------------------------------------------
    wide = _build_wide_panel(data, unit=unit, time=time, outcome=outcome)
    all_units = list(wide.index)
    if pre_period is not None:
        lo, hi = pre_period
        time_cols = [t for t in wide.columns if lo <= t <= hi]
        if not time_cols:
            raise ValueError(f"pre_period {pre_period} selected 0 periods")
        wide = wide[time_cols]
    if wide.isna().any().any():
        raise ValueError(
            "Panel is unbalanced or has NaN in pre_period. "
            "Balance before calling synth_experimental_design()."
        )

    # --- Candidates & donors ----------------------------------------------
    cand = list(candidates) if candidates is not None else list(all_units)
    unknown = set(cand) - set(all_units)
    if unknown:
        raise ValueError(f"candidates not in panel: {sorted(unknown)}")
    if not (1 <= k <= len(cand) - 1):
        raise ValueError(f"k must be in [1, {len(cand) - 1}], got {k}")

    if donors is not None:
        donor_list = list(donors)
        unknown_d = set(donor_list) - set(all_units)
        if unknown_d:
            raise ValueError(f"donors not in panel: {sorted(unknown_d)}")
        fixed_donor_pool: Optional[List[Any]] = donor_list
    elif set(cand) == set(all_units):
        fixed_donor_pool = None  # leave-one-out mode
    else:
        fixed_donor_pool = [u for u in all_units if u not in set(cand)]

    # --- Fit SC per candidate ---------------------------------------------
    rows: List[Dict[str, Any]] = []
    weights_map: Dict[Any, np.ndarray] = {}
    donor_union: List[Any] = (
        list(fixed_donor_pool) if fixed_donor_pool is not None else list(all_units)
    )

    for i_unit in cand:
        y_i = wide.loc[i_unit].to_numpy(dtype=float)
        if fixed_donor_pool is None:
            donor_ids = [u for u in all_units if u != i_unit]
        else:
            donor_ids = [u for u in fixed_donor_pool if u != i_unit]
        if len(donor_ids) < 2:
            raise ValueError(f"unit {i_unit!r} has fewer than 2 donors")
        X = wide.loc[donor_ids].to_numpy(dtype=float).T  # (T_pre, n_donors)
        w, mspe, eff = _leave_one_out_sc(y_i, X, penalization=penalization)
        rmse = float(np.sqrt(mspe))
        loss = mspe if risk == "mspe" else rmse
        herf = float(np.sum(w ** 2))
        risk_score = loss + concentration_weight * herf
        rows.append({
            "unit": i_unit,
            "pre_mspe": mspe,
            "pre_rmse": rmse,
            "effective_donors": eff,
            "herfindahl": herf,
            "risk_score": risk_score,
        })
        weights_map[i_unit] = np.array([w[donor_ids.index(u)] if u in donor_ids else 0.0
                                        for u in donor_union], dtype=float)

    ranking = pd.DataFrame(rows).sort_values("risk_score").reset_index(drop=True)
    selected = ranking["unit"].iloc[:k].tolist()
    ranking["selected"] = ranking["unit"].isin(selected)

    expected_var = float(
        ranking.loc[ranking["selected"], "pre_mspe"].sum()
    )

    # --- Baseline: random k-subset expected sum-MSPE ----------------------
    rng = np.random.default_rng(random_state)
    vals = ranking["pre_mspe"].to_numpy()
    if n_random <= 0:
        baseline = float(np.mean(vals) * k)
    else:
        draws = np.empty(n_random)
        n = len(vals)
        for b in range(n_random):
            idx = rng.choice(n, size=k, replace=False)
            draws[b] = vals[idx].sum()
        baseline = float(draws.mean())

    return SynthExperimentalDesignResult(
        selected=selected,
        ranking=ranking,
        weights=weights_map,
        donor_units=donor_union,
        expected_variance=expected_var,
        baseline_variance=baseline,
        method="abadie_zhao_2025",
        diagnostics={
            "n_units": int(len(all_units)),
            "n_candidates": int(len(cand)),
            "n_donors": int(len(donor_union)),
            "T_pre": int(wide.shape[1]),
            "risk": risk,
            "concentration_weight": float(concentration_weight),
            "penalization": float(penalization),
            "n_random": int(max(n_random, 0)),
            "leave_one_out_mode": fixed_donor_pool is None,
        },
    )
