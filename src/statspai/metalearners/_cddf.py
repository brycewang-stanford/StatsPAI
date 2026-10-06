"""Shared core of the BLP / GATES regressions (``sp.blp_test``, ``sp.gate_test``).

Both are weighted least-squares regressions with weights
``1 / (p (1 - p))`` of the outcome on a control block (an intercept and,
optionally, a baseline proxy ``B(Z)``) plus treatment-residual regressors
``D - p`` interacted with the CATE proxy ``S(Z)``. This module holds the
weighted fit, its covariance options, the quantile grouping and the
out-of-fold proxy construction, so the two public functions cannot drift
apart. The reference implementation is R ``GenericML`` (``BLP``, ``GATES``,
``quantile_group``); with identical inputs the numbers agree to the
floating-point floor (``tests/reference_parity/test_ml_causal_R_parity.py``).
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

VCOV_TYPES = ("const", "HC0", "HC1", "HC2", "HC3")


def resolve_vector(value: Any, data: pd.DataFrame, n: int, name: str) -> np.ndarray:
    """A per-row numeric vector given as an array or a column name."""
    if isinstance(value, str):
        if value not in data.columns:
            raise MethodIncompatibility(
                f"{name}: column '{value}' not found in data.",
                recovery_hint=f"Pass an existing column or an array for {name}.",
            )
        arr = data[value].to_numpy(dtype=float)[:n]
    else:
        arr = np.asarray(value, dtype=float).ravel()
    if arr.shape[0] != n:
        raise MethodIncompatibility(
            f"{name} must have one value per row ({n}); got {arr.shape[0]}.",
            recovery_hint=f"Align {name} with the estimation sample.",
        )
    if not np.isfinite(arr).all():
        raise MethodIncompatibility(
            f"{name} contains non-finite values.",
            recovery_hint=f"Drop or impute non-finite {name} rows.",
        )
    return np.asarray(arr)


def check_propensity(p: np.ndarray) -> None:
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise NumericalInstability(
            "Propensity scores must lie strictly inside (0, 1); the "
            "regression weights are 1 / (p (1 - p)).",
            recovery_hint="Trim or clip the propensity scores away from 0 and 1.",
        )


def wls_fit(
    y: np.ndarray, X: np.ndarray, w: np.ndarray, vcov_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted least squares plus the ``sandwich::vcovHC`` covariance.

    All types follow ``sandwich::vcovHC`` applied to a weighted ``lm``,
    whose estimating functions are ``w_i e_i x_i``: ``const`` uses the
    constant ``sum((w e)^2) / (n - k)`` (GenericML's default), HC0-HC3 the
    per-observation squares with the leverages of the weighted design.
    """
    if vcov_type not in VCOV_TYPES:
        raise MethodIncompatibility(
            f"vcov_type must be one of {VCOV_TYPES}; got {vcov_type!r}.",
            recovery_hint="Use 'const' (GenericML's default) or 'HC1'.",
        )
    n, k = X.shape
    if n <= k:
        raise DataInsufficient(
            f"The regression has {k} coefficients but only {n} observations.",
            recovery_hint="Use fewer groups or a larger sample.",
        )
    Xw = X * w[:, None]
    XtWX = X.T @ Xw
    if np.linalg.matrix_rank(XtWX) < k:
        raise NumericalInstability(
            "The BLP / GATES design matrix is rank deficient.",
            recovery_hint=(
                "Check for a constant proxy, an empty group, or a baseline "
                "proxy collinear with the intercept."
            ),
        )
    bread = np.linalg.inv(XtWX)
    beta = bread @ (Xw.T @ y)
    e = y - X @ beta
    if vcov_type == "const":
        # sandwich::vcovHC(type = "const") on a weighted lm works on the
        # estimating functions w_i e_i x_i: omega is the constant
        # sum((w e)^2) / (n - k) and the meat is omega * X'X. This is not
        # vcov(lm) (sigma^2 (X'WX)^{-1}) unless the weights are constant.
        omega_c = float(np.sum((w * e) ** 2) / (n - k))
        return beta, omega_c * (bread @ (X.T @ X) @ bread)
    if vcov_type in ("HC2", "HC3"):
        h = w * np.einsum("ij,jk,ik->i", X, bread, X)
        adj = (1.0 - h) if vcov_type == "HC2" else (1.0 - h) ** 2
        omega = (w * e) ** 2 / adj
    else:
        omega = (w * e) ** 2
    meat = (X * omega[:, None]).T @ X
    V = bread @ meat @ bread
    if vcov_type == "HC1":
        V = V * n / (n - k)
    return beta, V


def quantile_groups(x: np.ndarray, n_groups: int) -> np.ndarray:
    """Group labels ``0..K-1`` by type-7 quantile cut points, left-closed.

    ``GenericML::quantile_group``: group 1 is ``x < q_1``, group ``k`` is
    ``q_{k-1} <= x < q_k``, the last is ``x >= q_{K-1}``, with ``q`` from R's
    default ``quantile`` (numpy's ``linear`` method).
    """
    cuts = np.quantile(x, np.arange(1, n_groups) / n_groups)
    labels = np.searchsorted(cuts, x, side="right")
    sizes = np.bincount(labels, minlength=n_groups)
    if np.any(sizes < 2):
        raise DataInsufficient(
            "The quantile cut points leave a group with fewer than two units "
            f"(sizes {sizes.tolist()}).",
            recovery_hint="Use fewer groups; the proxy has too many ties.",
        )
    return labels


def cross_fit_proxy(
    result: Any,
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    n_folds: int,
    seed: int,
) -> Optional[np.ndarray]:
    """Out-of-fold CATE predictions from a fitted metalearner's estimator.

    Returns None when ``result`` carries no refittable estimator.
    """
    info = getattr(result, "model_info", None)
    est = info.get("_estimator") if isinstance(info, dict) else None
    if est is None or not callable(getattr(est, "fit", None)):
        return None
    from sklearn.model_selection import KFold

    out = np.empty(len(Y))
    for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
        model = copy.deepcopy(est)
        model.fit(X[tr], Y[tr], D[tr])
        out[te] = np.asarray(model.effect(X[te]), dtype=float).ravel()
    return out


def resolve_proxy(
    result: Any,
    proxy: Any,
    data: pd.DataFrame,
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    n_folds: int,
    seed: int,
    in_sample: np.ndarray,
    context: str,
) -> Tuple[np.ndarray, str]:
    """The CATE proxy ``S(Z)`` and a label for where it came from.

    ``proxy='cross_fit'`` refits the result's learner out of fold when it
    can; a raw array passed as ``result`` is used as supplied (it is the
    caller's responsibility that it was not fit on these outcomes). A
    fitted model without a refittable learner (e.g. a causal forest) falls
    back to its in-sample predictions with a warning.
    """
    n = len(Y)
    if isinstance(proxy, str):
        if proxy == "in_sample":
            return in_sample, "in_sample"
        if proxy != "cross_fit":
            raise MethodIncompatibility(
                f"{context}: proxy must be 'cross_fit', 'in_sample' or an array.",
                recovery_hint="Use proxy='cross_fit' (the default).",
            )
        if isinstance(result, (np.ndarray, list, tuple, pd.Series)):
            return in_sample, "supplied"
        cf = cross_fit_proxy(result, X, Y, D, n_folds, seed)
        if cf is not None:
            return cf, "cross_fit"
        warnings.warn(
            f"{context}: the CATE proxy is the model's in-sample prediction "
            "(it has no refittable learner). Predictions fit on the same "
            "outcomes overstate heterogeneity; pass out-of-sample predictions "
            "via proxy=.",
            UserWarning,
            stacklevel=3,
        )
        return in_sample, "in_sample"
    return resolve_vector(proxy, data, n, "proxy"), "supplied"
