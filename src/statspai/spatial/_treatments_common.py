"""Shared row assembly for the alternative spatial treatment estimators."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility


def _ring_row(k: int, edges: np.ndarray, wk: np.ndarray) -> Dict[str, Any]:
    lo, hi = float(edges[k]), float(edges[k + 1])
    return {
        "ring": k,
        "lower": lo,
        "upper": hi,
        "midpoint": 0.5 * (lo + hi),
        "n_units": int((wk > 0).any(axis=1).sum()),
    }


def _missing_cells() -> Dict[str, float]:
    keys = (
        "estimate se ci_lower ci_upper pvalue ri_pvalue mu_treated "
        "mu_control V_t V_c V_x"
    ).split()
    return {key: np.nan for key in keys}


def _inference_cells(
    k: int,
    mu_t: float,
    mu_c: float,
    var: float,
    draws_est: Optional[np.ndarray],
    z: float,
) -> Dict[str, float]:
    """Estimate, interval and p-values of one ring from its variance."""
    est = float(mu_t - mu_c)
    if var < 0:
        warnings.warn(
            AssumptionWarning(
                f"Ring {k}: the variance estimate is negative ({var:.3g}); "
                "the standard error is set to 0.",
                recovery_hint="Use more candidate locations or a wider ring.",
            ),
            stacklevel=3,
        )
    se = float(np.sqrt(max(var, 0.0)))
    pval = float(2 * stats.norm.sf(abs(est) / se)) if se > 0 else np.nan
    ri_p = np.nan
    if draws_est is not None:
        ok = np.isfinite(draws_est)
        if ok.any():
            tol = 1e-12 * max(1.0, abs(est))
            hits = np.sum(np.abs(draws_est[ok]) >= abs(est) - tol)
            ri_p = float((1 + hits) / (1 + ok.sum()))
    return {
        "estimate": est,
        "se": se,
        "ci_lower": est - z * se,
        "ci_upper": est + z * se,
        "pvalue": pval,
        "ri_pvalue": ri_p,
        "mu_treated": float(mu_t),
        "mu_control": float(mu_c),
    }


def _covariate_matrix(
    frame: pd.DataFrame, cols: Optional[Sequence[str]], name: str
) -> Optional[np.ndarray]:
    """Numeric design columns (categorical ones expanded to indicators)."""
    if not cols:
        return None
    from ..core._covariates import expand_categorical_covariates

    frame, cols, _ = expand_categorical_covariates(
        frame, list(cols), function="spatial_treatment"
    )
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise DataInsufficient(
            f"{name} has no column(s) {missing}.",
            recovery_hint=f"Name covariate columns present in {name}.",
            diagnostics={"missing": missing},
        )
    try:
        X = frame[list(cols)].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"Covariates of {name} must be numeric or categorical.",
            recovery_hint="Convert the covariate columns to numbers.",
        ) from exc
    if not np.all(np.isfinite(X)):
        raise DataInsufficient(
            f"Covariates of {name} contain missing values.",
            recovery_hint=f"Drop or impute the rows of {name} with missing covariates.",
            diagnostics={"n_bad": int((~np.isfinite(X)).any(axis=1).sum())},
        )
    return np.asarray(X, dtype=float)


def alternative_rows(
    method: str,
    D: np.ndarray,
    W: np.ndarray,
    p: np.ndarray,
    Y: np.ndarray,
    edges: np.ndarray,
    kernel: str,
    location_weights: bool,
    d0: float,
    draws: Optional[np.ndarray],
    z: float,
    data: pd.DataFrame,
    locations: pd.DataFrame,
    covariates: Optional[Sequence[str]],
    location_covariates: Optional[Sequence[str]],
    propensity: Optional[str],
    n_folds: int,
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], List[int], Optional[Dict[str, Any]]]:
    """Per-ring table rows for ``method='nearest'`` and ``method='dml'``."""
    from .treatments import _ring_weights

    rows: List[Dict[str, Any]] = []
    empty: List[int] = []
    dml_info: Optional[Dict[str, Any]] = None
    ring_w = []
    for k in range(edges.size - 1):
        wk = _ring_weights(D, edges[k], edges[k + 1], kernel, first=(k == 0))
        if location_weights:
            tot = wk.sum(axis=0)
            wk = np.divide(wk, tot, out=np.zeros_like(wk), where=tot > 0)
        ring_w.append(wk)
    if method == "nearest":
        from ._treatments_nearest import nearest_rings

        fits = nearest_rings(D, W, p, Y, ring_w, d0, draws)
        for k, (wk, fit) in enumerate(zip(ring_w, fits)):
            row = _ring_row(k, edges, wk)
            row.update(n_pairs_treated=fit["n_pairs_treated"])
            row.update(n_pairs_control=fit["n_pairs_control"])
            if "comp" not in fit:
                empty.append(k)
                row.update(_missing_cells(), V_cov=np.nan, V_bound=np.nan)
            else:
                comp = fit["comp"]
                row.update(
                    _inference_cells(
                        k, fit["mu_t"], fit["mu_c"], comp["var"], fit["ri_draws"], z
                    ),
                    V_t=np.nan,
                    V_c=np.nan,
                    V_x=np.nan,
                    V_cov=comp["V_cov"],
                    V_bound=comp["V_bound"],
                )
            rows.append(row)
    else:
        from ._treatments_dml import dml_rings

        Xunit = _covariate_matrix(data, covariates, "data")
        Zloc = _covariate_matrix(locations, location_covariates, "locations")
        dml = dml_rings(
            D,
            W,
            Y,
            ring_w,
            d0,
            None if propensity is None and Zloc is not None else p,
            Zloc,
            Xunit,
            n_folds,
            rng,
        )
        dml_info = dml
        for k, (wk, fit) in enumerate(zip(ring_w, dml["rings"])):
            row = _ring_row(k, edges, wk)
            if fit is None:
                empty.append(k)
                row.update(
                    n_pairs_treated=int(((wk > 0) & (W[None, :] == 1)).sum()),
                    n_pairs_control=int(((wk > 0) & (W[None, :] == 0)).sum()),
                )
                row.update(_missing_cells())
            else:
                comp = fit["comp"]
                row.update(
                    n_pairs_treated=fit["n_t"],
                    n_pairs_control=fit["n_c"],
                )
                row.update(
                    _inference_cells(k, fit["mu_t"], fit["mu_c"], comp["var"], None, z),
                    V_t=comp["Vt"],
                    V_c=comp["Vc"],
                    V_x=comp["Vx"],
                )
            rows.append(row)
    return rows, empty, dml_info
