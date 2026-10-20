"""
Design-based (Taylor linearisation) variance of any estimator from its
influence function -- the one place where ``design=`` is turned into a
standard error.

An estimator that accepts ``design=SurveyDesign`` does three things:

1. fits with the design weights (``design.weights``) as sampling weights;
2. builds its per-row *linearised contributions* ``u_i`` -- the terms with
   ``theta_hat - theta ~= sum_i u_i`` (for a weighted M-estimator,
   ``u_i = w_i IF_i / sum(w)``, i.e. the weight is already inside);
3. calls :func:`design_vcov` (and :func:`design_df` for the reference
   distribution).

:func:`design_vcov` is the first-stage (ultimate-cluster) variance of the
PSU totals of ``u`` within strata -- R ``survey::svyrecvar`` for a
single-stage design (strata, PSUs, FPC, lonely-PSU rule, calibration), the
same routine ``sp.svymean`` / ``sp.svyglm`` use. Nothing estimator-specific
lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..exceptions import MethodIncompatibility
from .estimators import _design_dof, _design_vcov

if TYPE_CHECKING:
    from .design import SurveyDesign


@dataclass(frozen=True)
class DesignRows:
    """The estimator's rows located inside a survey design.

    Attributes
    ----------
    design : SurveyDesign
    rows : ndarray of int
        Position, in ``design.data``, of every row the estimator uses (in
        the estimator's own order).
    weights : ndarray
        The design weights of those rows.
    """

    design: "SurveyDesign"
    rows: np.ndarray
    weights: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        """Boolean indicator over the design's rows of the rows in use."""
        m = np.zeros(self.design.n, dtype=bool)
        m[self.rows] = True
        return m


def check_design(
    design: Any,
    data: pd.DataFrame,
    *,
    function: str,
    weights: Optional[str] = None,
    cluster: Any = None,
    allow_calibrated: bool = True,
) -> "SurveyDesign":
    """Validate a ``design=`` argument against the estimator's ``data``.

    The design must be a :class:`SurveyDesign` built on the same rows as
    ``data`` (same length and index): the estimator reads its variables
    from ``data`` and the sampling structure from the design. The design
    carries the weights and the PSUs, so ``weights=`` and ``cluster=``
    must not be given as well.
    """
    from ._replicate_refit import refuse_replicate
    from .design import SurveyDesign

    refuse_replicate(design, function)
    if not isinstance(design, SurveyDesign):
        raise MethodIncompatibility(
            f"{function}: design= must be a SurveyDesign "
            f"(got {type(design).__name__}).",
            recovery_hint="Build it with sp.svydesign(data, weights=, "
            "strata=, cluster=, fpc=) and pass the same data frame as data=.",
        )
    if weights is not None or cluster is not None:
        given = [n for n, v in (("weights", weights), ("cluster", cluster)) if v]
        raise MethodIncompatibility(
            f"{function}: design= already carries the sampling weights and "
            f"the PSUs; {' and '.join(given)}= cannot be given as well.",
            recovery_hint="Declare weights and clusters in sp.svydesign(...) "
            "and drop them from this call.",
            diagnostics={"given": given},
        )
    if len(data) != design.n or not data.index.equals(design.data.index):
        raise MethodIncompatibility(
            f"{function}: data has {len(data)} rows but the design was "
            f"declared on {design.n} rows (or on a different index). The "
            "estimator must see the rows the design describes; filtering "
            "the data first would drop strata and PSUs from the variance.",
            recovery_hint="Pass the data frame the design was built from "
            "(design.data). For a subpopulation keep all rows and use the "
            "estimator on the full design.",
            diagnostics={"n_data": len(data), "n_design": design.n},
        )
    if not allow_calibrated and design._calibration is not None:
        raise MethodIncompatibility(
            f"{function}: a calibrated design is not supported here; the "
            "calibration residualises row-level scores, and this estimator's "
            "influence function is defined per unit, not per row.",
            recovery_hint="Use the calibrated weights as an ordinary "
            "design: sp.svydesign(data, weights=cal.weights, ...).",
        )
    return design


def design_rows(
    design: "SurveyDesign", data: pd.DataFrame, columns: Sequence[str]
) -> DesignRows:
    """Rows of ``data`` with no missing value in ``columns``, in order.

    These are exactly the rows ``data[columns].dropna()`` keeps, as
    positions in the design, so an estimator that drops incomplete rows
    can place its contributions back without index look-ups (which break
    on repeated index labels).
    """
    cols = list(dict.fromkeys(c for c in columns if c is not None))
    complete = ~data[cols].isna().any(axis=1).to_numpy()
    rows = np.flatnonzero(complete)
    return DesignRows(design=design, rows=rows, weights=design.weights[rows])


def design_vcov(
    contributions: np.ndarray,
    design: "SurveyDesign",
    rows: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Linearisation covariance of an estimator from its contributions.

    Parameters
    ----------
    contributions : (m,) or (m, p) array
        ``u_i`` with ``theta_hat - theta ~= sum_i u_i``; the sampling
        weight is already inside (``u_i = w_i IF_i / sum(w)`` for a
        weighted M-estimator with influence function ``IF``).
    design : SurveyDesign
    rows : array of int, optional
        Positions of the ``m`` rows in the design. Rows of the design
        that are not listed contribute zero but keep their stratum and
        PSU in the variance (domain estimation, as R ``svyglm`` treats
        rows with a missing variable). Default: all rows, in order.

    Returns
    -------
    (p, p) ndarray
        ``sum_h (1 - f_h) n_h / (n_h - 1) sum_j (t_hj - tbar_h)(...)'`` over
        PSU totals ``t_hj`` -- R ``survey::svyrecvar`` for a one-stage
        design, with the design's lonely-PSU rule and, for a calibrated
        design, the calibration residuals.
    """
    u = np.asarray(contributions, dtype=np.float64)
    if u.ndim == 1:
        u = u[:, None]
    if rows is None:
        if u.shape[0] != design.n:
            raise MethodIncompatibility(
                f"design_vcov: {u.shape[0]} contributions for a design of "
                f"{design.n} rows; pass rows= to place them."
            )
        full = u
    else:
        rows = np.asarray(rows)
        if rows.shape[0] != u.shape[0]:
            raise MethodIncompatibility(
                f"design_vcov: {u.shape[0]} contributions but {rows.shape[0]} rows."
            )
        full = np.zeros((design.n, u.shape[1]))
        np.add.at(full, rows, u)
    if not np.all(np.isfinite(full)):
        raise MethodIncompatibility(
            "design_vcov: the influence function has non-finite entries.",
            recovery_hint="Check the estimator's fit (separation, empty arms).",
        )
    out: np.ndarray = _design_vcov(full, design)
    return out


def design_se_of(
    design: "SurveyDesign", rows: np.ndarray, n: int
) -> Callable[[np.ndarray], float]:
    """SE of a mean-type estimator from influence rows scaled to mean one.

    For estimators that write ``theta_hat - theta ~= mean(u)`` over their
    ``n`` rows with the (mean-one) sampling weight inside ``u``: the
    contributions are ``u / n``.
    """

    def se_of(u: np.ndarray) -> float:
        return float(np.sqrt(design_vcov(np.asarray(u) / n, design, rows)[0, 0]))

    return se_of


def design_df(design: "SurveyDesign", rows: Optional[np.ndarray] = None) -> float:
    """Design degrees of freedom, #PSU - #strata (Stata ``e(df_r)``).

    With ``rows`` (the rows an estimator used) strata without a used row
    are dropped and every PSU of the remaining strata counts, as Stata
    ``svy, subpop()``; with all rows in use it is R ``degf``.
    """
    if rows is None or len(rows) == design.n:
        return _design_dof(design)
    mask = np.zeros(design.n, dtype=bool)
    mask[np.asarray(rows)] = True
    return _design_dof(design, mask, "stata")


def design_inference(
    estimate: float, se: float, df: float, alpha: float
) -> Tuple[Tuple[float, float], float]:
    """t-based interval and two-sided p-value with the design df."""
    crit = float(sp_stats.t.ppf(1 - alpha / 2, df=df))
    ci = (estimate - crit * se, estimate + crit * se)
    if not (se > 0):
        return ci, float("nan")
    return ci, float(2 * sp_stats.t.sf(abs(estimate / se), df=df))


def design_info(design: "SurveyDesign", df: float) -> Dict[str, Any]:
    """Summary of the design recorded in a result's ``model_info``."""
    return {
        "n": int(design.n),
        "n_strata": int(design._strata_codes.max()) + 1,
        "n_psu": int(design._psu_codes.max()) + 1,
        "df": float(df),
        "weights": design._weight_col,
        "strata": design.strata_col,
        "cluster": design.cluster_col,
        "fpc": design.fpc_col,
        "lonely_psu": design.lonely_psu,
        "calibrated": (
            None if design._calibration is None else design._calibration["method"]
        ),
    }


def unit_design_rows(
    design: "SurveyDesign",
    data: pd.DataFrame,
    unit: str,
    units: Sequence[Any],
    *,
    function: str,
) -> DesignRows:
    """First design row of each panel unit, checking the design is per unit.

    A panel estimator's influence function has one entry per unit; the
    unit's weight, stratum and PSU must be the same on all its rows (the
    unit is the sampled element). Returns the first row of each unit in
    ``units`` order.
    """
    ids = data[unit].to_numpy()
    codes, uniq = pd.factorize(pd.Series(ids), sort=False)
    for name, values in (
        ("weights", design.weights),
        ("strata", design._strata_codes),
        ("PSU", design._psu_codes),
    ):
        per_unit = pd.Series(values).groupby(codes).nunique()
        if (per_unit > 1).any():
            raise MethodIncompatibility(
                f"{function}: the survey {name} vary within a panel unit "
                f"({int((per_unit > 1).sum())} units). A panel unit is one "
                "sampled element: its weight, stratum and PSU must be the "
                "same in every period.",
                recovery_hint="Use the base-period weight on all of a "
                "unit's rows (or the longitudinal weight) and unit-level "
                "strata / PSUs.",
            )
    first = np.unique(codes, return_index=True)[1]  # first row per code
    lookup = pd.Series(first, index=pd.Index(uniq))
    try:
        rows = lookup.loc[list(units)].to_numpy()
    except KeyError as exc:  # pragma: no cover - units come from data
        raise MethodIncompatibility(
            f"{function}: a unit of the estimator is not in the design."
        ) from exc
    return DesignRows(design=design, rows=rows, weights=design.weights[rows])


class DesignVariance:
    """Design variance for an influence-function matrix on given rows.

    For estimators that write ``Var(theta) = psi' psi / n^2`` (the R ``did``
    convention: ``psi`` has one row per unit, ``n`` units): the
    contributions are ``psi / n`` and their design covariance replaces
    ``psi' psi / n^2``. ``rows`` places the ``psi`` rows in the design.
    Passed where an estimator accepts cluster ids for its influence-function
    standard errors.
    """

    is_survey_design = True

    def __init__(self, design: "SurveyDesign", rows: np.ndarray):
        self.design = design
        self.rows = np.asarray(rows)
        self.df = design_df(design, self.rows)

    def vcov(self, psi: np.ndarray, n: int) -> np.ndarray:
        return design_vcov(np.asarray(psi, dtype=float) / n, self.design, self.rows)

    def se(self, psi: np.ndarray, n: int) -> np.ndarray:
        psi = np.asarray(psi, dtype=float)
        squeeze = psi.ndim == 1
        out = np.sqrt(np.maximum(np.diag(self.vcov(psi, n)), 0.0))
        return np.asarray(out[0] if squeeze else out)

    def info(self) -> Dict[str, Any]:
        return design_info(self.design, self.df)


def refuse_design_result(result: Any, function: str) -> None:
    """Refuse a post-estimation step that would ignore a fit's design."""
    mi = getattr(result, "model_info", None) or {}
    if mi.get("survey_design") is not None:
        raise MethodIncompatibility(
            f"{function}: this result was fitted with design= (a complex "
            "survey design); this step would rebuild its variance from the "
            "influence functions without the strata, PSUs and fpc.",
            recovery_hint="Use the aggregations and standard errors the fit "
            "already reports (result.detail, model_info['event_study']), or "
            "refit without design=.",
        )


def t_inference_frame(
    frame: pd.DataFrame, alpha: float, df: float, est: str = "att"
) -> pd.DataFrame:
    """Recompute ``ci_lower`` / ``ci_upper`` / ``pvalue`` with t(df)."""
    if frame is None or len(frame) == 0 or "se" not in frame:
        return frame
    out = frame.copy()
    crit = float(sp_stats.t.ppf(1 - alpha / 2, df=df))
    b = out[est].to_numpy(dtype=float)
    s = out["se"].to_numpy(dtype=float)
    ok = np.isfinite(s) & (s > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["pvalue"] = np.where(
            ok,
            2 * sp_stats.t.sf(np.abs(b / np.where(ok, s, 1.0)), df=df),
            out["pvalue"],
        )
    out["ci_lower"] = np.where(ok, b - crit * s, out["ci_lower"])
    out["ci_upper"] = np.where(ok, b + crit * s, out["ci_upper"])
    return out


def design_var_diag(
    contributions: np.ndarray,
    design: "SurveyDesign",
    rows: Optional[np.ndarray] = None,
    block: int = 64,
) -> np.ndarray:
    """Diagonal of :func:`design_vcov` for many columns, in column blocks."""
    u = np.asarray(contributions, dtype=np.float64)
    if u.ndim == 1:
        u = u[:, None]
    out = np.empty(u.shape[1])
    for lo in range(0, u.shape[1], block):
        hi = min(lo + block, u.shape[1])
        out[lo:hi] = np.diag(design_vcov(u[:, lo:hi], design, rows))
    return out
