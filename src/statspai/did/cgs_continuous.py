"""Callaway-Goodman-Bacon-Sant'Anna DiD with a continuous treatment.

A continuous dose has no single ATT to report. Units dosed at 0.2 and at
0.8 are different comparisons, and the two-way fixed-effects coefficient
averages them with weights nobody chose -- weights that can be negative.
CGS replace that coefficient with two dose-indexed curves:

``ATT(d)``
    the effect of dose ``d`` versus no dose, among units that got ``d``.

``ACRT(d)``
    the causal response at ``d``, the derivative of ``ATT`` -- what a
    question about a *marginal* change in the dose is actually asking.

See :mod:`statspai.did._contdid` for the estimation details. Pinned against
``contdid`` 0.1.1, the authors' own package: the fitted curves, the overall
ATT and the overall ACRT all agree to machine precision.

References
----------
callaway2024difference
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._contdid import cont_did_cell, default_dose_grid, knots_by_quantile

__all__ = ["ContinuousDoseResult", "cgs_continuous_did"]


@dataclass
class ContinuousDoseResult(ResultProtocolMixin):
    """Dose-response curves from :func:`cgs_continuous_did`."""

    dose: np.ndarray
    att_d: np.ndarray
    acrt_d: np.ndarray
    overall_att: float
    overall_acrt: float
    overall_acrt_se: float
    n_units: int
    n_cells: int
    degree: int
    knots: np.ndarray
    curve_basis: str
    control_group: str
    alpha: float = 0.05
    detail: Optional[pd.DataFrame] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    method: str = "Continuous-treatment DiD (Callaway, Goodman-Bacon & Sant'Anna)"

    @property
    def ci(self) -> tuple:
        z = float(stats.norm.ppf(1 - self.alpha / 2))
        return (
            self.overall_acrt - z * self.overall_acrt_se,
            self.overall_acrt + z * self.overall_acrt_se,
        )

    def to_frame(self) -> pd.DataFrame:
        """The curves, one row per dose on the grid."""
        return pd.DataFrame(
            {"dose": self.dose, "att_d": self.att_d, "acrt_d": self.acrt_d}
        )

    def summary(self) -> str:
        lo, hi = self.ci
        lines = [
            self.method,
            "=" * len(self.method),
            f"  units            : {self.n_units}",
            f"  (g, t) cells     : {self.n_cells}",
            f"  control group    : {self.control_group}",
            f"  spline           : degree {self.degree}, "
            f"{len(self.knots)} interior knot(s)",
            f"  dose grid        : {len(self.dose)} points, "
            f"[{self.dose.min():.4g}, {self.dose.max():.4g}]",
            "",
            f"  overall ATT      : {self.overall_att:.6f}",
            f"  overall ACRT     : {self.overall_acrt:.6f} "
            f"(se {self.overall_acrt_se:.6f}, "
            f"{100 * (1 - self.alpha):.0f}% CI [{lo:.6f}, {hi:.6f}])",
            "",
            "ACRT is the derivative of ATT in the dose. A flat ACRT means "
            "the effect scales linearly with the dose; a falling one means "
            "later units of treatment buy less than earlier ones.",
        ]
        if self.curve_basis == "reference":
            lines.append("")
            lines.append(
                "curve_basis='reference' reproduces contdid 0.1.1's reported "
                "curves, which are evaluated on a basis rescaled to the dose "
                "grid rather than the fitted range. The curves are then a "
                "rescaled version of the fitted dose response and do not line "
                "up with the overall ACRT above."
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "overall_att": self.overall_att,
            "overall_acrt": self.overall_acrt,
            "overall_acrt_se": self.overall_acrt_se,
            "n_units": self.n_units,
            "n_cells": self.n_cells,
            "degree": self.degree,
            "knots": np.asarray(self.knots).tolist(),
            "curve_basis": self.curve_basis,
            "control_group": self.control_group,
            "alpha": self.alpha,
            "curves": self.to_frame().to_dict(orient="list"),
            "diagnostics": dict(self.diagnostics),
        }


def cgs_continuous_did(
    data: pd.DataFrame,
    y: str,
    *,
    dose: str,
    time: str,
    unit: str,
    cohort: str,
    degree: int = 3,
    num_knots: int = 0,
    knots: Optional[Sequence[float]] = None,
    dose_grid: Optional[Sequence[float]] = None,
    control_group: str = "nevertreated",
    curve_basis: str = "fitted",
    alpha: float = 0.05,
) -> ContinuousDoseResult:
    """ATT(d) and ACRT(d) for a continuous treatment.

    Parameters
    ----------
    data : DataFrame
        Long-format panel.
    y : str
        Outcome column.
    dose : str
        Continuous treatment intensity. Zero for untreated units.
    time : str
        Period column.
    unit : str
        Unit identifier.
    cohort : str
        First-treatment period; ``0`` marks never-treated units.
    degree : int, default 3
        B-spline degree for the dose. ``degree=1`` with no interior knots
        gives a constant ACRT -- the "effect per unit of dose" reading --
        and is the right starting point when the sample is small.
    num_knots : int, default 0
        Interior knots, placed at equally spaced quantiles of the positive
        doses. Ignored when ``knots`` is given. There is no automatic
        selector: more knots buy flexibility at the cost of variance, and
        the choice is recorded on the result.
    knots : sequence of float, optional
        Explicit interior knots.
    dose_grid : sequence of float, optional
        Doses at which to report the curves. Defaults to the 10th-99th
        percentiles of the positive doses, which is where there are data to
        support them.
    control_group : {"nevertreated", "notyettreated"}, default "nevertreated"
    curve_basis : {"fitted", "reference"}, default "fitted"
        Which basis the reported curves are evaluated on.

        ``"fitted"`` uses the same basis the regression was fitted on, so
        ``acrt_d`` is the derivative of ``att_d`` and both are the fitted
        function.

        ``"reference"`` rebuilds the basis with its boundary at the ends of
        the *dose grid* instead, which is what ``contdid`` 0.1.1 reports.
        Because the fitted coefficients are then applied to a differently
        scaled basis, the reported curves are a rescaled version of the
        fitted dose response -- on the parity fixture the reported ACRT sits
        10% above the ``overall_acrt`` the same call returns, and at degree 1
        the gap is exactly the ratio of the two ranges. Use it only to
        reproduce output from that package.
    alpha : float, default 0.05

    Returns
    -------
    ContinuousDoseResult

    Notes
    -----
    The overall ACRT standard error comes from the influence function of the
    per-cell regression. ``contdid`` routes its standard errors through the
    ``pte`` package's aggregation layer, which is not replicated here, so
    the two differ by a few percent even though the point estimates agree
    exactly.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(400):
    ...     g = 2 if i < 200 else 0
    ...     d = float(rng.uniform(0.1, 1.0)) if g else 0.0
    ...     fe = rng.normal()
    ...     for t in (1, 2):
    ...         eff = 2.0 * d if (g and t >= g) else 0.0
    ...         rows.append((i, t, g, d, fe + 0.5 * t + eff + rng.normal(0, 0.3)))
    >>> df = pd.DataFrame(rows, columns=["id", "t", "g", "dose", "y"])
    >>> res = sp.cgs_continuous_did(
    ...     df, y="y", dose="dose", time="t", unit="id", cohort="g", degree=1
    ... )
    >>> bool(abs(res.overall_acrt - 2.0) < 0.4)
    True

    References
    ----------
    callaway2024difference
    """
    context = "cgs_continuous_did"
    if control_group not in {"nevertreated", "notyettreated"}:
        raise MethodIncompatibility(
            f"{context}: control_group must be 'nevertreated' or " "'notyettreated'.",
            diagnostics={"context": context, "control_group": control_group},
        )
    if curve_basis not in {"fitted", "reference"}:
        raise MethodIncompatibility(
            f"{context}: curve_basis must be 'fitted' or 'reference'.",
            diagnostics={"context": context, "curve_basis": curve_basis},
        )
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(
            f"{context}: alpha must be in (0, 1).",
            diagnostics={"context": context, "alpha": alpha},
        )
    for col in (y, dose, time, unit, cohort):
        if col not in data.columns:
            raise MethodIncompatibility(
                f"{context}: column {col!r} not in data.",
                diagnostics={"context": context, "columns": list(data.columns)},
            )

    df = data.copy()
    periods = sorted(df[time].unique())
    cohorts = [g for g in sorted(df[cohort].unique()) if g != 0]
    if not cohorts:
        raise DataInsufficient(
            f"{context}: no treated cohorts (every unit has cohort 0).",
            diagnostics={"context": context},
        )

    all_units = pd.Index(sorted(df[unit].unique()))
    n_units = len(all_units)
    pos = pd.Series(np.arange(n_units), index=all_units)

    # One row per unit: the dose is a unit-level attribute, and quantiles
    # taken over the long format would weight units by how many periods they
    # happen to be observed for.
    unit_doses = (
        df.drop_duplicates(subset=[unit]).loc[lambda f: f[dose] > 0, dose]
    ).to_numpy(dtype=float)
    grid = (
        default_dose_grid(unit_doses)
        if dose_grid is None
        else np.asarray(dose_grid, dtype=float)
    )
    knots_arr = np.asarray(
        knots_by_quantile(unit_doses, num_knots) if knots is None else knots,
        dtype=float,
    )

    cells: List[Dict[str, Any]] = []
    for g in cohorts:
        base = g - 1
        if base not in periods:
            warnings.warn(
                f"{context}: cohort {g!r} has no period {base!r} to use as the "
                "base period, so it contributes no cells.",
                UserWarning,
                stacklevel=2,
            )
            continue
        for t in [p for p in periods if p >= g]:
            frame = df[df[time].isin([base, t])]
            if control_group == "notyettreated":
                is_control = (frame[cohort] == 0) | (frame[cohort] > t)
            else:
                is_control = frame[cohort] == 0
            frame = frame[(frame[cohort] == g) | is_control]
            pre = frame[frame[time] == base].set_index(unit)
            post = frame[frame[time] == t].set_index(unit)
            ids = pre.index.intersection(post.index)
            if len(ids) == 0:
                continue
            pre, post = pre.loc[ids], post.loc[ids]
            dy = post[y].to_numpy(dtype=float) - pre[y].to_numpy(dtype=float)
            # The dose only counts for the cohort being estimated; controls
            # enter at dose zero however they are eventually treated.
            d_cell = np.where(
                pre[cohort].to_numpy() == g, pre[dose].to_numpy(dtype=float), 0.0
            )
            try:
                fit = cont_did_cell(
                    dose=d_cell,
                    dy=dy,
                    dose_grid=grid,
                    degree=degree,
                    knots=knots_arr,
                )
            except DataInsufficient:
                continue
            if curve_basis == "reference":
                fit = _reference_curves(fit, d_cell, dy, grid, degree, knots_arr)
            psi = np.zeros(n_units, dtype=float)
            psi[pos.reindex(np.asarray(ids)).to_numpy()] = (
                n_units / len(ids)
            ) * fit.influence
            cells.append(
                {
                    "cohort": g,
                    "time": t,
                    "att_overall": fit.att_overall,
                    "acrt_overall": fit.acrt_overall,
                    "n_treated": fit.n_treated,
                    "n_control": fit.n_control,
                    "_att_d": fit.att_d,
                    "_acrt_d": fit.acrt_d,
                    "_influence": psi,
                }
            )

    if not cells:
        raise DataInsufficient(
            f"{context}: no estimable (cohort, period) cells. Check that each "
            "treated cohort has a period before it and some zero-dose units.",
            diagnostics={"context": context, "cohorts": cohorts},
        )

    w = np.array([float(c["n_treated"]) for c in cells])
    w = w / w.sum() if w.sum() > 0 else np.full(len(cells), 1.0 / len(cells))
    att_d = np.sum([wi * c["_att_d"] for wi, c in zip(w, cells)], axis=0)
    acrt_d = np.sum([wi * c["_acrt_d"] for wi, c in zip(w, cells)], axis=0)
    overall_att = float(np.sum(w * np.array([c["att_overall"] for c in cells])))
    overall_acrt = float(np.sum(w * np.array([c["acrt_overall"] for c in cells])))
    psi = np.sum([wi * c["_influence"] for wi, c in zip(w, cells)], axis=0)
    se = float(np.sqrt(np.mean(psi**2) / n_units))

    detail = pd.DataFrame(
        [{k: v for k, v in c.items() if not k.startswith("_")} for c in cells]
    )
    return ContinuousDoseResult(
        dose=grid,
        att_d=att_d,
        acrt_d=acrt_d,
        overall_att=overall_att,
        overall_acrt=overall_acrt,
        overall_acrt_se=se,
        n_units=n_units,
        n_cells=len(cells),
        degree=degree,
        knots=knots_arr,
        curve_basis=curve_basis,
        control_group=control_group,
        alpha=float(alpha),
        detail=detail,
        diagnostics={
            "cohorts": cohorts,
            "influence_function": psi,
            "cell_weights": w.tolist(),
        },
    )


def _reference_curves(fit, d_cell, dy, grid, degree, knots_arr):
    """Re-evaluate the curves on a basis anchored at the grid's endpoints.

    This is what ``contdid`` 0.1.1 reports. The fit is unchanged; only the
    basis the coefficients are applied to differs, so ``acrt_d`` stops being
    the derivative of ``att_d``. Kept behind ``curve_basis='reference'`` for
    reproducing that package's output.
    """
    from ._contdid import bspline_basis

    lo, hi = float(np.min(grid)), float(np.max(grid))
    slope = fit.coefficients[1:]
    B = bspline_basis(grid, degree=degree, knots=knots_arr, lower=lo, upper=hi)
    dB = bspline_basis(
        grid, degree=degree, knots=knots_arr, lower=lo, upper=hi, deriv=1
    )
    control_mean = float(np.mean(dy[d_cell == 0]))
    fit.att_d = fit.coefficients[0] + B @ slope - control_mean
    fit.acrt_d = dB @ slope
    return fit
