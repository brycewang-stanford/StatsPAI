"""Survey calibration (raking / post-stratification / linear calibration).

Adjusts design weights so that the weighted sample matches known
population totals (margins) on a set of auxiliary variables. This is
the ``survey::calibrate()`` and ``survey::rake()`` functionality from
R's survey package — previously unavailable in a Python econometrics
toolkit.

Three calibration methods:

- **Raking** (Deming & Stephan 1940) — iterative proportional fitting
  that adjusts weights to match marginal distributions one variable
  at a time. Converges to the minimum-entropy distance solution.
- **Linear calibration** (Deville & Särndal 1992) — find weights
  closest to the design weights (chi-squared distance) such that
  weighted totals of X exactly equal population totals.
- **Post-stratification** — special case where cells are defined
  by the crossing of categorical variables.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


@dataclass
class CalibrationResult(ResultProtocolMixin):
    calibrated_weights: np.ndarray
    method: str
    converged: bool
    iterations: int
    weight_summary: Dict[str, float]

    def summary(self) -> str:
        lines = [
            f"Survey Calibration ({self.method})",
            "-" * 40,
            f"Converged : {self.converged} ({self.iterations} iterations)",
            f"Min weight: {self.weight_summary['min']:.4f}",
            f"Max weight: {self.weight_summary['max']:.4f}",
            f"Mean weight: {self.weight_summary['mean']:.4f}",
            f"CV(weight) : {self.weight_summary['cv']:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


@accepts_aliases(weights="weight")
def rake(
    data: pd.DataFrame,
    margins: Dict[str, Dict],
    weight: Optional[str] = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> CalibrationResult:
    """Raking (iterative proportional fitting).

    Starting from the design weights, cycles through the margins and
    rescales the weights within each category so its weighted share equals
    the target, until every margin holds.  The fixed point is the
    raking-ratio (multiplicative-distance) calibration: R
    ``survey::rake`` run to convergence and
    ``survey::calibrate(calfun = "raking")`` give the same weights.

    Parameters
    ----------
    data : pd.DataFrame
    margins : dict
        ``{column_name: {category: target}}``.  Targets may be proportions or
        population counts; each margin is normalised to shares.  Every
        category observed in ``data[column_name]`` must have a target and
        every target category must occur in the data (R ``rake`` also
        refuses both).
    weight : str, optional
        Existing design weight column. If ``None``, starts with equal weights.
    max_iter : int
        Maximum number of full sweeps over the margins.
    tol : float
        Convergence: after a sweep, every weighted category share is within
        ``tol`` (relative) of its target.  Scale-free, unlike the 1.28.0
        criterion (see Notes).

    Returns
    -------
    CalibrationResult
        ``calibrated_weights`` sum to 1 (multiply by the population size to
        get R ``weights(rake(...))`` when the margins are counts).

    Notes
    -----
    Up to 1.28.0 convergence was declared when the largest absolute
    change in the sum-to-one weights fell below ``tol = 1e-6``.  Those
    weights are O(1/n), so for large samples the loop stopped after one or
    two sweeps with margins still off (4e-4 relative at n = 100 000).

    The calibrated weights are only weights: passing them to
    ``sp.svydesign`` treats them as fixed, which is not R's
    calibration-adjusted linearisation variance (residuals of y on the
    calibration variables).  For standard errors that account for the
    calibration use ``design.calibrate(margins=...)`` on a
    :class:`SurveyDesign` (``test_survey_calibrated_design_parity.py``).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({
    ...     "sex": rng.choice(["M", "F"], size=n),
    ...     "age_group": rng.choice(["18-34", "35-54", "55+"], size=n),
    ... })
    >>> margins = {
    ...     "sex": {"M": 0.49, "F": 0.51},
    ...     "age_group": {"18-34": 0.30, "35-54": 0.40, "55+": 0.30},
    ... }
    >>> res = sp.rake(df, margins=margins)
    >>> bool(res.converged)
    True
    """
    n = len(data)
    if weight is not None:
        w = data[weight].to_numpy(dtype=float).copy()
    else:
        w = np.ones(n, dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise MethodIncompatibility(
            "design weights must be finite and strictly positive"
        )
    w = w / w.sum()

    # (masks, target shares) per margin, validated up front
    plan = []
    sums = []
    for col, targets in margins.items():
        vals = data[col].to_numpy()
        present = set(pd.unique(vals))
        missing_target = present - set(targets)
        if missing_target:
            raise MethodIncompatibility(
                f"margin '{col}': categories {sorted(map(str, missing_target))} "
                "occur in the data but have no target",
                recovery_hint="Supply a target for every category of the margin.",
            )
        absent = [c for c in targets if c not in present]
        if absent:
            raise MethodIncompatibility(
                f"margin '{col}': target categories {sorted(map(str, absent))} "
                "do not occur in the data"
            )
        tvals = np.array([float(targets[c]) for c in targets])
        if np.any(~np.isfinite(tvals)) or np.any(tvals <= 0):
            raise MethodIncompatibility(f"margin '{col}': targets must be positive")
        sums.append(tvals.sum())
        plan.append([(vals == c, t / tvals.sum()) for c, t in zip(targets, tvals)])
    if len(sums) > 1 and np.ptp(sums) > 1e-8 * max(sums):
        warnings.warn(
            f"margins have different totals {sums}; each is normalised to "
            "shares, so only the proportions are matched.",
            UserWarning,
            stacklevel=2,
        )

    def _max_rel_gap() -> float:
        return float(max(abs(w[m].sum() - t) / t for cells in plan for m, t in cells))

    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        for cells in plan:
            for mask, target in cells:
                w[mask] *= target / w[mask].sum()
        if _max_rel_gap() < tol:
            converged = True
            break
    if not converged:
        warnings.warn(
            f"raking did not converge in {max_iter} sweeps "
            f"(max relative margin gap {_max_rel_gap():.3g} >= tol={tol:g})",
            RuntimeWarning,
            stacklevel=2,
        )
    w = w / w.sum()

    return CalibrationResult(
        calibrated_weights=w,
        method="raking",
        converged=converged,
        iterations=iteration,
        weight_summary={
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "cv": float(w.std() / w.mean()) if w.mean() > 0 else 0.0,
        },
    )


@accepts_aliases(weights="weight")
def linear_calibration(
    data: pd.DataFrame,
    totals: Dict[str, float],
    weight: Optional[str] = None,
) -> CalibrationResult:
    """Deville-Särndal (1992) linear calibration.

    Find calibrated weights ``w_i = g_i * d_i`` minimising the chi-squared
    distance ``Σ (w_i - d_i)² / d_i = Σ d_i (g_i - 1)²`` subject to
    ``Σ w_i x_{ik} = T_k`` for each auxiliary variable k with known total
    T_k.  Closed form ``g = 1 + X (X' D X)^{-1} (T - X' d)``; identical to R
    ``survey::calibrate(design, ~ 0 + x1 + ..., population, calfun =
    "linear")`` (unbounded).  No intercept is added: include a column of
    ones with total N for one (a dummy column for a category count).

    Returns weights only; for calibration-aware standard errors use
    ``design.calibrate(totals=...)`` on a :class:`SurveyDesign`.

    Parameters
    ----------
    totals : dict
        ``{column_name: population_total}`` for continuous auxiliary variables.
    weight : str, optional
        Design weight column. If ``None``, uses equal weights.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({
    ...     "income": rng.normal(50.0, 10.0, size=n),
    ...     "age": rng.normal(40.0, 12.0, size=n),
    ... })
    >>> totals = {"income": df["income"].sum() * 1.05,
    ...           "age": df["age"].sum() * 0.98}
    >>> res = sp.linear_calibration(df, totals=totals)
    >>> cal_total = float((res.calibrated_weights * df["income"]).sum())
    >>> bool(abs(cal_total - totals["income"]) < 1e-4)
    True
    """
    n = len(data)
    if weight is not None:
        d = data[weight].to_numpy(dtype=float)
    else:
        d = np.ones(n)
    var_names = list(totals.keys())
    T_pop = np.array([totals[v] for v in var_names])
    X = data[var_names].to_numpy(dtype=float)

    # Current weighted totals
    T_current = (d[:, None] * X).sum(axis=0)

    # g-weights: g = 1 + X (X' D X)^{-1} (T - T_current) where D = diag(d)
    DX = d[:, None] * X
    XtDX = X.T @ DX
    if np.linalg.matrix_rank(XtDX) < XtDX.shape[0]:
        warnings.warn(
            "calibration variables are collinear; using the pseudo-inverse "
            "(the totals of the redundant columns are matched only if they "
            "are consistent)",
            UserWarning,
            stacklevel=2,
        )
        lam = np.linalg.pinv(XtDX) @ (T_pop - T_current)
    else:
        lam = np.linalg.solve(XtDX, T_pop - T_current)
    g = 1.0 + X @ lam
    w = d * g

    return CalibrationResult(
        calibrated_weights=w,
        method="linear (Deville-Särndal)",
        converged=True,
        iterations=1,
        weight_summary={
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "cv": float(w.std() / w.mean()) if w.mean() > 0 else 0.0,
        },
    )
