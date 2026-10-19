"""``design=`` on the regression entry points (``sp.regress``, ``sp.logit``,
``sp.probit``): the fit uses the design weights as sampling weights and the
covariance is ``B M B`` with ``B`` the inverse weighted information and ``M``
the linearisation covariance of the score rows -- R ``svyglm`` / Stata
``svy:``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility
from .linearization import check_design, design_df, design_info, design_vcov

if TYPE_CHECKING:
    from ..core.results import EconometricResults
    from .design import SurveyDesign


def refuse_design_options(function: str, **given: Any) -> None:
    """``design=`` fixes the weights and the variance; refuse other requests."""
    clash = [k for k, v in given.items() if v not in (None, False, "nonrobust")]
    if clash:
        raise MethodIncompatibility(
            f"{function}: design= fixes the weights and the variance "
            "(survey linearisation); it cannot be combined with "
            + ", ".join(f"{k}=" for k in clash)
            + ".",
            recovery_hint="Declare weights / clusters in sp.svydesign(...) and "
            "drop the other variance options.",
            diagnostics={"given": clash},
        )


def positions(
    data: pd.DataFrame, labels: Optional[pd.Index], function: str
) -> np.ndarray:
    """Positions in ``data`` of the fitted rows (labels ``None`` = all rows)."""
    if labels is None or len(labels) == len(data):
        return np.arange(len(data))
    if not data.index.is_unique:
        raise MethodIncompatibility(
            f"{function}: rows were dropped for missing values and the data "
            "index has repeated labels, so the fitted rows cannot be placed "
            "in the design.",
            recovery_hint="Give the data a unique index (reset_index) before "
            "building the design.",
        )
    pos = data.index.get_indexer(labels)
    if (pos < 0).any():  # pragma: no cover - labels come from data
        raise MethodIncompatibility(f"{function}: fitted rows not found in data.")
    return np.asarray(pos)


def apply_design_vcov(
    result: Any, V: np.ndarray, df: float, design: "SurveyDesign", label: str
) -> Any:
    """Install a design covariance and t(df) inference on a fitted result."""
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    result.std_errors = pd.Series(se, index=result.params.index)
    info: Dict[str, Any] = dict(result.data_info)
    info.pop("inference", None)
    info["df_inference"] = float(df)
    info["var_cov"] = V
    result.data_info = info
    result._compute_statistics()
    mi = dict(result.model_info)
    mi["vcov_type"] = label
    mi["se_type"] = label
    mi["survey_design"] = design_info(design, df)
    result.model_info = mi
    return result


def design_sandwich(
    scores: np.ndarray,
    bread: np.ndarray,
    design: "SurveyDesign",
    rows: np.ndarray,
) -> np.ndarray:
    """``B M B`` with ``M`` the design covariance of the score rows."""
    meat = design_vcov(scores, design, rows)
    V: np.ndarray = bread @ meat @ bread
    return V


def regress_with_design(
    formula: str,
    data: pd.DataFrame,
    design: Any,
    collinear: str,
) -> "EconometricResults":
    """``sp.regress(formula, data, design=)``: WLS + survey linearisation."""
    from ..regression.ols import regress

    svy = check_design(design, data, function="regress")
    work = data.reset_index(drop=True)
    work["__svy_weight__"] = svy.weights
    base = regress(
        formula,
        work,
        weights="__svy_weight__",
        weights_as="sampling",
        collinear=collinear,
    )
    di = base.data_info
    rows = positions(work, di.get("sample_index"), "regress")
    X = np.asarray(di["X"], dtype=float)
    y = np.asarray(di["y"], dtype=float).ravel()
    w = svy.weights[rows]
    beta = base.params.to_numpy(dtype=float)
    resid = y - X @ beta
    bread = np.linalg.inv((X * w[:, None]).T @ X)
    scores = (w * resid)[:, None] * X
    V = design_sandwich(scores, bread, svy, rows)
    df = design_df(svy, rows)
    info = dict(base.data_info)
    info["sample_index"] = data.index[rows]
    info.pop("analytic_weights", None)
    base.data_info = info
    mi = dict(base.model_info)
    for key in ("weights_as", "n_effective_weights", "robust"):
        mi.pop(key, None)
    mi["weighted"] = True
    base.model_info = mi
    out: "EconometricResults" = apply_design_vcov(
        base, V, df, svy, "Survey linearisation (strata / PSU / fpc; R svyglm)"
    )
    return out
