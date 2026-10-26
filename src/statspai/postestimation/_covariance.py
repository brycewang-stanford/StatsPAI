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

__all__ = [
    "AttrArray",
    "coefficient_covariance",
    "coefficient_scale",
    "inference_df",
    "require_covariance",
]

_COVARIANCE_KEYS = ("var_cov", "vcov", "cov_params", "cov")


class AttrArray:
    """A read-only matrix that can live in ``DataFrame.attrs``.

    pandas compares the ``attrs`` of the inputs of ``pd.concat`` with ``==``
    and needs a single bool -- and ``print`` of a wide table concatenates
    the halves it shows. A bare ndarray there makes both raise ("the truth
    value of an array ... is ambiguous"). This wrapper compares by value and
    returns a bool, and behaves as the array for reading (``np.asarray``,
    indexing, ``@``, ``.T``, ``.shape``).
    """

    __slots__ = ("_a",)

    def __init__(self, a: Any) -> None:
        arr = np.array(a, dtype=float)
        arr.setflags(write=False)
        object.__setattr__(self, "_a", arr)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return self._a if dtype is None else self._a.astype(dtype)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AttrArray):
            other = other._a
        try:
            return bool(np.array_equal(self._a, np.asarray(other), equal_nan=True))
        except (TypeError, ValueError):
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    __hash__ = None  # type: ignore[assignment]

    def __getitem__(self, key: Any) -> Any:
        return self._a[key]

    def __len__(self) -> int:
        return len(self._a)

    def __iter__(self) -> Any:
        return iter(self._a)

    def __matmul__(self, other: Any) -> Any:
        return self._a @ np.asarray(other)

    def __rmatmul__(self, other: Any) -> Any:
        return np.asarray(other) @ self._a

    def __deepcopy__(self, memo: Any) -> "AttrArray":
        return self  # immutable

    def __copy__(self) -> "AttrArray":
        return self

    def __reduce__(self) -> Any:
        return (AttrArray, (self._a,))

    @property
    def T(self) -> np.ndarray:
        return self._a.T

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._a.shape

    @property
    def ndim(self) -> int:
        return self._a.ndim

    def __repr__(self) -> str:
        return f"AttrArray(shape={self._a.shape})"


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


class _CoefficientScaleView:
    """A fit reported as ``exp(b)`` (``irr=True``) seen on the ``b`` scale."""

    def __init__(
        self,
        params: pd.Series,
        std_errors: pd.Series,
        data_info: dict,
        model_info: dict,
    ) -> None:
        self.params = params
        self.std_errors = std_errors
        self.data_info = data_info
        self.model_info = model_info


def coefficient_scale(result: Any) -> Any:
    """``result`` on the scale of its estimated coefficients.

    ``sp.poisson`` / ``sp.nbreg`` / ``sp.xtnbreg`` with ``irr=True`` report
    ``exp(b)`` with standard errors ``exp(b) se(b)``. Stata's ``test`` /
    ``lincom`` / ``nlcom`` after ``poisson, irr`` work on ``e(b) = b``, so
    these functions read such a fit through this view: ``b``, ``se(b)`` and
    the covariance of ``b``. Ancillary parameters (names starting with
    ``/``) are not exponentiated by the fits and are left as they are.
    Any other result is returned unchanged.
    """
    model_info = getattr(result, "model_info", None) or {}
    params = getattr(result, "params", None)
    if not model_info.get("irr") or not isinstance(params, pd.Series):
        return result
    names = [str(n) for n in params.index]
    expo = np.array([not n.startswith("/") for n in names])
    reported = params.to_numpy(dtype=float)
    se = _standard_errors(result, len(params))
    b = reported.copy()
    b[expo] = np.log(reported[expo])
    se_b = se.copy()
    se_b[expo] = se[expo] / reported[expo]
    data_info = dict(getattr(result, "data_info", None) or {})
    # sp.xtnbreg(irr=True) maps its covariance to the IRR scale (J V J'),
    # sp.poisson / sp.nbreg keep the b-scale matrix: undo the former.
    V_reported, _ = coefficient_covariance(result)
    if V_reported is not None:
        scale = np.where(expo, 1.0 / reported, 1.0)
        data_info["var_cov"] = V_reported * np.outer(scale, scale)
    view_info = dict(model_info)
    view_info["irr"] = False
    return _CoefficientScaleView(
        pd.Series(b, index=params.index),
        pd.Series(se_b, index=params.index),
        data_info,
        view_info,
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
