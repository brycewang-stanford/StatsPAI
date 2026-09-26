"""Coefficient covariance and reference distribution for post-estimation.

``sp.test``, ``sp.lincom`` and ``sp.margins`` need two things from a fit that
the result objects store in different places:

* the **full** covariance matrix of ``result.params``. Rebuilding it from the
  standard errors as a diagonal matrix treats every pair of estimates as
  uncorrelated, which silently mis-states any restriction or delta-method
  gradient that involves more than one coefficient (``test x1 = x2`` after
  ``regress, vce(robust)``: F = 154.4 against Stata's 167.0);
* the **reference distribution** of the fit's own inference: t / F with
  ``df`` degrees of freedom, or z / chi-squared.

Both are resolved here once.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

__all__ = ["coefficient_covariance", "inference_df", "require_covariance"]

_COVARIANCE_KEYS = ("var_cov", "vcov", "cov_params", "cov")


def _standard_errors(result: Any, k: int) -> np.ndarray:
    se = np.asarray(getattr(result, "std_errors"), dtype=float).reshape(-1)
    if se.size != k:
        raise MethodIncompatibility(
            f"result has {k} coefficients but {se.size} standard errors.",
        )
    return se


def _as_matrix(candidate: Any, params: pd.Series) -> Optional[np.ndarray]:
    """``candidate`` as a (k, k) array aligned to ``params``, or None."""
    k = len(params)
    if isinstance(candidate, pd.DataFrame):
        try:
            candidate = candidate.loc[params.index, params.index]
        except KeyError:
            if candidate.shape != (k, k):
                return None
        candidate = candidate.to_numpy(dtype=float)
    try:
        V = np.asarray(candidate, dtype=float)
    except (TypeError, ValueError):
        return None
    if V.shape != (k, k) or not np.all(np.isfinite(V)):
        return None
    return V


def coefficient_covariance(result: Any) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Return ``(V, se)`` for ``result.params``.

    ``V`` is the full covariance matrix when the result carries one **and**
    its diagonal reproduces the reported standard errors (relative 1e-6).
    Several estimators replace ``std_errors`` after the fit -- CR2 / wild
    bootstrap / Conley paths -- without touching the stored matrix; that
    stale matrix describes different standard errors than the ones printed,
    so it is not used. ``V`` is None when no consistent matrix exists.
    """
    params = result.params
    if not isinstance(params, pd.Series):
        params = pd.Series(np.atleast_1d(np.asarray(params, dtype=float)))
    k = len(params)
    se = _standard_errors(result, k)

    candidates = []
    for store_name in ("data_info", "model_info"):
        store = getattr(result, store_name, None)
        if isinstance(store, dict):
            candidates.extend(store.get(key) for key in _COVARIANCE_KEYS)
    inner = getattr(result, "_results", None)
    if inner is not None:
        candidates.append(getattr(inner, "var_cov", None))
    # sp.panel results wrap a linearmodels fit whose ``cov`` is the matrix
    # behind the reported (possibly clustered) standard errors.
    linearmodels_fit = getattr(result, "_lm_result", None)
    if linearmodels_fit is not None:
        candidates.append(getattr(linearmodels_fit, "cov", None))
    candidates.append(getattr(result, "vcov", None))

    for cand in candidates:
        if cand is None or callable(cand):
            continue
        V = _as_matrix(cand, params)
        if V is None:
            continue
        diag_se = np.sqrt(np.clip(np.diag(V), 0.0, None))
        if np.allclose(diag_se, se, rtol=1e-6, atol=1e-12, equal_nan=True):
            return V, se
    return None, se


def require_covariance(result: Any, weights: np.ndarray, what: str) -> np.ndarray:
    """Covariance to use for a restriction / gradient matrix ``weights``.

    ``weights`` is ``(q, k)``. With a consistent full matrix it is returned.
    Without one, the diagonal is exact only when a single coefficient enters
    a single restriction; anything else raises instead of assuming the
    estimates are uncorrelated.
    """
    V, se = coefficient_covariance(result)
    if V is not None:
        return V
    W = np.atleast_2d(np.asarray(weights, dtype=float))
    involved = np.count_nonzero(np.any(W != 0, axis=0))
    if W.shape[0] == 1 and involved <= 1:
        return np.diag(se**2)
    raise MethodIncompatibility(
        f"{what} involves several coefficients, which needs their full "
        "covariance matrix; this result carries only standard errors (or a "
        "stored matrix that no longer matches them). Using the standard "
        "errors alone would assume the estimates are uncorrelated.",
        recovery_hint=(
            "Refit with an estimator that stores data_info['var_cov'], or "
            "test one coefficient at a time."
        ),
    )


def inference_df(
    result: Any, stores: Tuple[str, ...] = ("data_info", "model_info")
) -> float:
    """Degrees of freedom of the fit's reference distribution; ``inf`` = z.

    ``data_info['inference'] == 'z'`` marks likelihood-based fits (Stata
    reports z / chi2 after ``logit``, ``poisson``, ``glm`` ...).
    ``data_info['df_inference']`` overrides the residual degrees of freedom
    (``G - 1`` under clustering, as ``regress, vce(cluster)`` uses);
    otherwise ``df_resid`` is used, and ``inf`` when neither is recorded.
    ``stores`` lists the result dictionaries to consult, in order.
    """
    for store_name in stores:
        store = getattr(result, store_name, None)
        if not isinstance(store, dict):
            continue
        if store.get("inference") == "z":
            return float("inf")
        for key in ("df_inference", "df_resid"):
            if key in store and store[key] is not None:
                try:
                    df = float(store[key])
                except (TypeError, ValueError):
                    continue
                return df if np.isfinite(df) and df > 0 else float("inf")
    return float("inf")
