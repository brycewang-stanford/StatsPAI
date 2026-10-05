"""
Pre-treatment unit selection by leave-one-out synthetic-control fit.

Given a panel of pre-treatment outcomes for N candidate units and a budget
``k`` (number of units to treat), rank candidates by the pre-period mean
squared prediction error (MSPE) of the synthetic control that can be built
for each one from the remaining donors, optionally penalising
concentrated donor weights, and pick the ``k`` best-fitted candidates.

.. warning::
   This is a heuristic, **not** the synthetic control design of Abadie &
   Zhao ("Synthetic Controls for Experimental Design", arXiv:2108.02196).
   Their design jointly chooses treated weights ``w`` and control weights
   ``v`` (disjoint supports, a cardinality constraint on the treated set)
   so that both weighted averages reproduce the population average of the
   pre-treatment predictors; it is a mixed-integer quadratic programme
   (the authors' replication code, github.com/jinglongzhao2/SCDesign,
   solves it with Gurobi). Earlier versions of this module attributed a
   variance formula ``Var[ATT | D] ~ sum_{i in D} sigma_i^2`` to that paper;
   no such formula appears there, and the ``expected_variance`` below is a
   heuristic sum of pre-period MSPEs, not a variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._input_validation import require_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._core import solve_simplex_weights

__all__ = [
    "synth_experimental_design",
    "SynthExperimentalDesignResult",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class SynthExperimentalDesignResult(ResultProtocolMixin):
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
        Sum of pre-period MSPEs over ``selected`` (a heuristic fit score,
        not a variance).
    baseline_variance : float
        Same quantity for a random-``k`` assignment (average over
        ``n_random`` draws); the gain is
        ``baseline_variance - expected_variance``.
    method : str
        ``'loo_sc_fit_ranking'`` (was ``'abadie_zhao_2025'``, a
        misattribution; see the module docstring), or
        ``'population_matching_search'`` under ``criterion='population'``.
    diagnostics : dict
        Extra metadata (n_units, pre_periods, solver, etc.).

    Notes
    -----
    Under ``criterion='population'`` the fields read differently:
    ``ranking`` has one row per unit with its ``role`` and ``weight``,
    ``weights`` holds the two vectors ``'treated'`` and ``'control'``
    (aligned to ``donor_units``, here all units), ``expected_variance`` is
    the sum of the two pre-period MSPEs of the chosen set and
    ``baseline_variance`` its mean over the sets searched.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> df = sp.utils.dgp_synth(n_units=40, n_periods=20, seed=0)
    >>> res = sp.synth_experimental_design(
    ...     df, unit="unit", time="time", outcome="y",
    ...     k=5, pre_period=(0, 19), random_state=0,
    ... )
    >>> len(res.selected)
    5
    """

    selected: List[Any]
    ranking: pd.DataFrame
    weights: Dict[Any, np.ndarray]
    donor_units: List[Any]
    expected_variance: float
    baseline_variance: float
    method: str = "loo_sc_fit_ranking"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        # The dataclass repr prints every weight vector and the full ranking.
        return (
            f"SynthExperimentalDesignResult(selected={list(self.selected)!r}, "
            f"n_candidates={len(self.ranking)}, "
            f"expected_variance={self.expected_variance:.6g}, "
            f"baseline_variance={self.baseline_variance:.6g})"
        )

    def summary(self) -> str:
        n = len(self.ranking)
        k = len(self.selected)
        gain = self.baseline_variance - self.expected_variance
        gain_pct = (
            100 * gain / self.baseline_variance if self.baseline_variance > 0 else 0.0
        )
        if self.method == "population_matching_search":
            d = self.diagnostics
            return "\n".join(
                [
                    "Treated set chosen to track the population average",
                    "-" * 66,
                    f"  Units                  : {n}",
                    f"  Treatment budget k     : {d.get('k')}",
                    f"  Treated (weight > 0)   : {list(self.selected)}",
                    f"  Treated-side pre RMSE  : {d.get('rmse_treated'):.6g}",
                    f"  Control-side pre RMSE  : {d.get('rmse_control'):.6g}",
                    f"  Loss (sum of MSPEs)    : {self.expected_variance:.6g}",
                    f"  Mean loss, random sets : {self.baseline_variance:.6g}",
                    f"  Sets searched          : {d.get('n_search')}"
                    "  (random search, not a global optimum)",
                ]
            )
        lines = [
            "Unit selection by leave-one-out SC fit (not the Abadie-Zhao design)",
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
    mspe = float(np.mean(resid**2))
    herf = float(np.sum(w**2))
    eff = float(1.0 / herf) if herf > 0 else float("nan")
    return w, mspe, eff


def _population_design(
    data: pd.DataFrame,
    wide: pd.DataFrame,
    *,
    unit: str,
    cand: List[Any],
    k: int,
    donors: Optional[Sequence[Any]],
    population_weights: Optional[str],
    penalization: float,
    n_search: int,
    random_state: Optional[int],
) -> "SynthExperimentalDesignResult":
    """Random search for a treated set that tracks the population average."""
    if donors is not None:
        raise MethodIncompatibility(
            "criterion='population' uses every unit outside the treated set "
            "as a control; donors= does not apply."
        )
    if n_search < 1:
        raise MethodIncompatibility("n_search must be at least 1.")
    units = list(wide.index)
    Y = wide.to_numpy(dtype=float).T  # (T_pre, n_units)
    if population_weights is None:
        share = np.full(len(units), 1.0 / len(units))
    else:
        if population_weights not in data.columns:
            raise MethodIncompatibility(
                f"population_weights column {population_weights!r} not found."
            )
        per_unit = data.groupby(unit)[population_weights]
        if (per_unit.nunique(dropna=False) > 1).any():
            raise MethodIncompatibility(
                f"{population_weights!r} must be constant within unit."
            )
        size = per_unit.first().reindex(units).to_numpy(dtype=float)
        if not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise MethodIncompatibility(
                f"{population_weights!r} must be positive for every unit."
            )
        share = size / size.sum()
    target = Y @ share
    pos = {u: j for j, u in enumerate(units)}
    cand_pos = np.array([pos[u] for u in cand])
    all_pos = np.arange(len(units))

    def evaluate(treated: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
        control = np.setdiff1d(all_pos, treated)
        w = solve_simplex_weights(target, Y[:, treated], penalization=penalization)
        v = solve_simplex_weights(target, Y[:, control], penalization=penalization)
        mse_t = float(np.mean((target - Y[:, treated] @ w) ** 2))
        mse_c = float(np.mean((target - Y[:, control] @ v) ** 2))
        return mse_t + mse_c, w, v, mse_t

    rng = np.random.default_rng(random_state)
    best: Optional[Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = None
    losses = np.empty(n_search)
    for b in range(n_search):
        treated = np.sort(rng.choice(cand_pos, size=k, replace=False))
        loss, w, v, mse_t = evaluate(treated)
        losses[b] = loss
        if best is None or loss < best[0]:
            best = (loss, treated, w, v, mse_t)
    assert best is not None
    loss, treated, w, v, mse_t = best
    control = np.setdiff1d(all_pos, treated)
    treated_w = dict(zip([units[j] for j in treated], w))
    control_w = dict(zip([units[j] for j in control], v))
    selected = [u for u, wt in treated_w.items() if wt > 1e-6]
    rows = [
        {
            "unit": u,
            "role": "treated" if u in treated_w else "control",
            "weight": float(treated_w[u] if u in treated_w else control_w.get(u, 0.0)),
            "population_share": float(share[pos[u]]),
            "selected": u in selected,
        }
        for u in units
    ]
    ranking = (
        pd.DataFrame(rows)
        .sort_values(["role", "weight"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return SynthExperimentalDesignResult(
        selected=selected,
        ranking=ranking,
        weights={
            "treated": np.array([treated_w.get(u, 0.0) for u in units]),
            "control": np.array([control_w.get(u, 0.0) for u in units]),
        },
        donor_units=units,
        expected_variance=float(loss),
        baseline_variance=float(losses.mean()),
        method="population_matching_search",
        diagnostics={
            "n_units": len(units),
            "n_candidates": len(cand),
            "k": k,
            "T_pre": int(Y.shape[0]),
            "n_search": n_search,
            "rmse_treated": float(np.sqrt(mse_t)),
            "rmse_control": float(np.sqrt(loss - mse_t)),
            "population_weights": population_weights,
            "penalization": float(penalization),
        },
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@accepts_aliases(id="unit", y="outcome")
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
    criterion: str = "loo_fit",
    population_weights: Optional[str] = None,
    n_search: int = 500,
) -> SynthExperimentalDesignResult:
    """Pick the ``k`` candidates with the best leave-one-out SC pre-fit.

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
        ``H(w) = sum(w_j^2)`` (a heuristic).
    penalization : float, default 0.0
        Ridge penalty passed to the simplex solver (Doudchenko &
        Imbens 2016 style).
    n_random : int, default 500
        Monte-Carlo draws used to estimate ``baseline_variance``
        (the expected sum-MSPE under random-``k`` selection).
    random_state : int, optional
    criterion : {'loo_fit', 'population'}, default 'loo_fit'
        ``'loo_fit'`` ranks each candidate by how well the other units
        reproduce it (the recipe below). ``'population'`` looks for a
        treated set that stands in for the whole market: among ``n_search``
        random sets of ``k`` candidates it keeps the one for which a
        simplex-weighted average of the set and a simplex-weighted average
        of all remaining units both track the population average of the
        outcome over the pre-period (the smallest sum of the two
        pre-period MSPEs). The experiment then estimates the effect on the
        market, not on the units that happen to be easy to predict. Units
        of the set that get zero weight are left out of ``selected``. The
        result holds both weight vectors in ``weights['treated']`` and
        ``weights['control']``. It is a random search, so it returns a good
        set, not the optimum; ``risk`` and ``concentration_weight`` are not
        used.
    population_weights : str, optional
        For ``criterion='population'``: a column, constant within unit,
        whose shares define the population average (city population, say).
        Default: every unit counts equally.
    n_search : int, default 500
        For ``criterion='population'``: number of random treated sets tried.

    Returns
    -------
    SynthExperimentalDesignResult

    Notes
    -----
    The recipe (a heuristic; see the module docstring) is:

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
    require_columns(data, [unit, time, outcome], function="synth_experimental_design")
    if risk not in ("mspe", "rmse"):
        raise ValueError(f"risk must be 'mspe' or 'rmse', got {risk!r}")
    if concentration_weight < 0:
        raise ValueError("concentration_weight must be >= 0")

    # --- Build wide panel --------------------------------------------------
    wide = _build_wide_panel(data, unit=unit, time=time, outcome=outcome)
    all_units = list(wide.index)
    # Catch a collapsed panel (empty / fully-missing / single-unit input)
    # *here*, with a message naming the real problem — otherwise it slips
    # through to the ``k`` range check below as a misleading
    # "k must be in [1, -1]" error.
    if len(all_units) < 2 or wide.shape[1] < 1:
        raise DataInsufficient(
            "synth_experimental_design: the panel collapsed to "
            f"{len(all_units)} unit(s) x {int(wide.shape[1])} period(s) after "
            "pivoting — an empty, fully-missing, or single-unit panel cannot "
            "design a synthetic-control experiment.",
            recovery_hint="Provide a balanced panel with multiple units and "
            "at least one pre-period observation.",
            diagnostics={
                "n_units": len(all_units),
                "n_periods": int(wide.shape[1]),
            },
        )
    if pre_period is not None:
        lo, hi = pre_period
        time_cols = [t for t in wide.columns if lo <= t <= hi]
        if not time_cols:
            raise ValueError(f"pre_period {pre_period} selected 0 periods")
        wide = wide[time_cols]
    if wide.isna().any().any():
        raise DataInsufficient(
            "synth_experimental_design: the panel is unbalanced or has NaN in "
            "the pre-period; synthetic control needs a balanced donor matrix.",
            recovery_hint="Balance the panel (e.g. sp.balance_panel) or restrict "
            "pre_period to fully-observed dates before calling.",
            diagnostics={"n_missing_cells": int(wide.isna().to_numpy().sum())},
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

    if criterion not in ("loo_fit", "population"):
        raise MethodIncompatibility(
            f"criterion must be 'loo_fit' or 'population', got {criterion!r}"
        )
    if criterion == "population":
        return _population_design(
            data,
            wide,
            unit=unit,
            cand=cand,
            k=int(k),
            donors=donors,
            population_weights=population_weights,
            penalization=penalization,
            n_search=int(n_search),
            random_state=random_state,
        )
    if population_weights is not None:
        raise MethodIncompatibility(
            "population_weights= is used by criterion='population' only."
        )

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
            raise ValueError(
                f"unit {i_unit!r} has fewer than 2 donors"
            )  # pragma: no cover
        X = wide.loc[donor_ids].to_numpy(dtype=float).T  # (T_pre, n_donors)
        w, mspe, eff = _leave_one_out_sc(y_i, X, penalization=penalization)
        rmse = float(np.sqrt(mspe))
        loss = mspe if risk == "mspe" else rmse
        herf = float(np.sum(w**2))
        risk_score = loss + concentration_weight * herf
        rows.append(
            {
                "unit": i_unit,
                "pre_mspe": mspe,
                "pre_rmse": rmse,
                "effective_donors": eff,
                "herfindahl": herf,
                "risk_score": risk_score,
            }
        )
        weights_map[i_unit] = np.array(
            [w[donor_ids.index(u)] if u in donor_ids else 0.0 for u in donor_union],
            dtype=float,
        )

    ranking = pd.DataFrame(rows).sort_values("risk_score").reset_index(drop=True)
    selected = ranking["unit"].iloc[:k].tolist()
    ranking["selected"] = ranking["unit"].isin(selected)

    expected_var = float(ranking.loc[ranking["selected"], "pre_mspe"].sum())

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
        method="loo_sc_fit_ranking",
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
