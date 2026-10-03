"""
OLS regression implementation with comprehensive features
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import solve_triangular

from .._aliases import accepts_aliases
from ..core._vcov_spec import markout_clusters
from ..core.base import BaseEstimator, BaseModel
from ..core.results import EconometricResults
from ..core.utils import _coerce_string_extension_dtypes, create_design_matrices
from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)

_NORMAL_EQUATION_COND_MAX = 1e8
_LOW_ORDER_DEP_MAX_WORK = 50_000

_OlsKernel = Callable[
    [np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]
_SandwichKernel = Callable[[np.ndarray, np.ndarray, np.ndarray, str], np.ndarray]
_ClusterMeatKernel = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
_HacMeatKernel = Callable[[np.ndarray, np.ndarray, Optional[int]], np.ndarray]


def _validate_analytic_weights(
    weights: Any,
    n: int,
    *,
    context: str,
) -> np.ndarray:
    """Validate Stata-style analytic weights for OLS/WLS paths."""
    try:
        w = np.asarray(weights, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(f"{context}: weights must be numeric") from exc
    if w.shape[0] != n:
        raise MethodIncompatibility(
            f"{context}: weights length ({w.shape[0]}) does not match "
            f"the number of observations ({n})."
        )
    if not np.isfinite(w).all():
        raise DataInsufficient(f"{context}: weights contain NaN or infinite values.")
    if (w <= 0).any():
        raise MethodIncompatibility(
            f"{context}: weights must be strictly positive "
            "(analytic/`aweight` semantics)."
        )
    return w


def _validate_ols_arrays(
    y: Any,
    X: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated float64 OLS arrays with aligned rows."""
    try:
        y_arr = np.asarray(y, dtype=float).ravel()
        X_arr = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: y and X must be numeric arrays"
        ) from exc
    if X_arr.ndim != 2:
        raise MethodIncompatibility(f"{context}: X must be 2-D, got ndim={X_arr.ndim}")
    if y_arr.shape[0] != X_arr.shape[0]:
        raise MethodIncompatibility(
            f"{context}: y has {y_arr.shape[0]} rows but X has "
            f"{X_arr.shape[0]} rows"
        )
    if y_arr.shape[0] < 1:
        raise DataInsufficient(f"{context}: data must contain at least one row")
    if not np.isfinite(y_arr).all():
        raise DataInsufficient(f"{context}: y contains non-finite values")
    if not np.isfinite(X_arr).all():
        raise DataInsufficient(f"{context}: X contains non-finite values")
    return y_arr, X_arr


def _crossprod_fit_if_well_conditioned(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OLS via cross-products for well-conditioned designs.

    QR remains the fallback for ill-conditioned certification cases. The
    cross-product path is only used when the small k x k system is comfortably
    conditioned, so the expected normal-equation precision loss stays far below
    the estimator tolerances while avoiding QR's fixed cost on common designs.
    """
    XtX = X.T @ X
    cond = np.linalg.cond(XtX)
    if not np.isfinite(cond) or cond > _NORMAL_EQUATION_COND_MAX:
        raise np.linalg.LinAlgError("ill-conditioned cross-product system")
    XtX_inv = np.linalg.solve(XtX, np.eye(XtX.shape[0]))
    params = XtX_inv @ (X.T @ y)
    fitted = X @ params
    residuals = y - fitted
    return params, fitted, residuals, XtX_inv


def _qr_fit_with_bread(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OLS via QR, returning coefficients and ``(X'X)^-1`` from the same R."""
    Q, R = np.linalg.qr(X)
    params = np.linalg.solve(R, Q.T @ y)
    fitted = X @ params
    residuals = y - fitted
    R_inv = np.linalg.solve(R, np.eye(R.shape[0]))
    XtX_inv = R_inv @ R_inv.T
    return params, fitted, residuals, XtX_inv


def _centered_intercept_bread(
    *,
    n: int,
    k: int,
    const_col: int,
    other: List[int],
    const_value: float,
    x_mean: np.ndarray,
    slope_xtx_inv: np.ndarray,
) -> np.ndarray:
    """Assemble ``(X'X)^-1`` for ``X=[c, Z]`` from centered slope bread."""
    XtX_inv = np.empty((k, k), dtype=float)
    mean_bread = x_mean @ slope_xtx_inv
    XtX_inv[const_col, const_col] = 1.0 / (n * const_value * const_value) + float(
        mean_bread @ x_mean
    ) / (const_value * const_value)
    cross = -mean_bread / const_value
    for pos, j in enumerate(other):
        XtX_inv[const_col, j] = cross[pos]
        XtX_inv[j, const_col] = cross[pos]
    for pos_i, i in enumerate(other):
        for pos_j, j in enumerate(other):
            XtX_inv[i, j] = slope_xtx_inv[pos_i, pos_j]
    return XtX_inv


def _detect_constant_column(X: np.ndarray) -> Optional[int]:
    """Index of the intercept column (exactly constant and non-zero), or None.

    Used to enable the mean-centered (Frisch-Waugh-Lovell) fit. Detection is
    by exact equality (``ptp == 0``): a patsy / design-matrix intercept is
    exactly ``1.0`` in every row, so this never misfires on a merely
    near-constant real regressor.
    """
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.ptp(col) == 0 and col[0] != 0:
            return j
    return None


def _detect_perfect_collinearity(X: np.ndarray, var_names: List[str]) -> None:
    """Raise :class:`NumericalInstability` on an exactly rank-deficient design.

    Perfect collinearity leaves coefficients unidentified; without a guard the
    least-squares solve returns enormous garbage (e.g. ``1e14``) with no signal
    — a silent-failure violation of the "fail loudly" rule.

    Detection is deliberately **structural** (duplicate / proportional columns
    and zero-variance regressors) rather than conditioning-based. A singular-
    value / rank tolerance loose enough to catch real collinearity also flags
    legitimately ill-conditioned *full-rank* designs: the NIST StRD Filippelli
    benchmark has ``s_min/s_max ~ 6e-16`` — numerically *more* singular than an
    exactly duplicated column — yet it is full rank and must fit. Structural
    detection separates the two cleanly: the worst off-diagonal |correlation|
    across every NIST ill-conditioned design is ~0.999, far under the
    ``1 - 1e-8`` duplicate threshold here. The trade-off is that a general
    exact dependence among 3+ columns (not reducible to a pairwise duplicate or
    a constant column) is intentionally *not* auto-detected, because a detector
    that caught it could not also pass Filippelli.
    """
    n, k = X.shape
    names = list(var_names) if var_names is not None else [f"x{i}" for i in range(k)]

    # 1) Zero-variance non-intercept regressor: no identifying variation, and
    #    collinear with the intercept when one is present.
    for j in range(k):
        if names[j] == "Intercept":
            continue
        col = X[:, j]
        if np.ptp(col) <= 1e-12 * max(1.0, float(np.max(np.abs(col)))):
            raise NumericalInstability(
                f"Regressor '{names[j]}' is constant (no variation); its "
                f"coefficient is not identified — perfectly collinear with "
                f"the intercept.",
                recovery_hint=(
                    f"Drop '{names[j]}', or remove the intercept if it is the "
                    f"only regressor."
                ),
                diagnostics={"zero_variance_regressor": names[j]},
            )

    # 2) Duplicate / proportional columns (|corr| == 1), including
    #    complementary 0/1 dummies (the dummy-variable trap). Needs >=3 rows for
    #    a meaningful correlation; smaller-n degeneracy is caught elsewhere.
    #    Keep this path lean: ``np.corrcoef`` is convenient but expensive on
    #    the hot ``sp.regress`` path, so compute only the small k x k Gram
    #    matrix needed for this structural check.
    if k >= 2 and n >= 3:
        centered = X - X.mean(axis=0)
        norms = np.sqrt(np.sum(centered * centered, axis=0))
        for i in range(k):
            if norms[i] == 0:
                continue
            for j in range(i + 1, k):
                if norms[j] == 0:
                    continue
                c = float(centered[:, i] @ centered[:, j]) / (norms[i] * norms[j])
                if np.isfinite(c) and abs(c) >= 1.0 - 1e-8:
                    raise NumericalInstability(
                        f"Regressors '{names[i]}' and '{names[j]}' are "
                        f"perfectly collinear (|correlation| = {abs(c):.10f}); "
                        f"the design matrix is rank-deficient and their "
                        f"coefficients are not separately identified.",
                        recovery_hint=(f"Drop one of '{names[i]}' or '{names[j]}'."),
                        diagnostics={
                            "collinear_pair": [names[i], names[j]],
                            "abs_correlation": float(abs(c)),
                        },
                    )


def _written_column_order(
    formula: Optional[str], design_info: Any, k: int
) -> List[int]:
    """Design columns in the order the formula writes its terms.

    patsy moves categorical terms ahead of numeric ones, so ``d1 + C(g)``
    arrives as ``C(g), d1``. Stata scans ``d1 i.g`` as written; the
    collinearity scan follows the written order so the same member is
    omitted. Falls back to the design order when the terms cannot be
    matched.
    """
    identity = list(range(k))
    if not formula or design_info is None:
        return identity
    try:
        import patsy

        written = list(patsy.ModelDesc.from_formula(formula).rhs_termlist)
        slices = design_info.term_slices
        keyed = []
        for pos, (term, sl) in enumerate(slices.items()):
            rank = written.index(term) if term in written else len(written) + pos
            keyed.extend((rank, j) for j in range(sl.start, sl.stop))
    except Exception:  # patsy absent or an unparsable formula: design order
        return identity
    order = [j for _, j in sorted(keyed, key=lambda t: t[0])]
    return order if sorted(order) == identity else identity


def _collinear_omissions(X: np.ndarray, var_names: List[str]) -> List[Tuple[int, str]]:
    """Columns Stata's ``regress`` would omit, scanned left to right.

    Returns ``(column index, reason)`` pairs. A column is omitted when it adds
    nothing to the columns kept before it, so the *later* member of a
    collinear set goes, as with Stata's ``_rmcoll``. Three structural cases
    are covered:

    * a zero column, or a constant column after the intercept;
    * a column proportional to a kept one (``|corr| = 1``; through the origin
      when the design has no constant);
    * an exact dependence among discrete columns -- intercept and integer-
      valued regressors such as dummies (the dummy-variable trap,
      ``C(g)`` plus a hand-made level dummy). Integer data carry exact
      dependencies, so a pivot-free Cholesky sweep of their cross-product in
      column order separates them from full-rank columns by many orders of
      magnitude.

    Continuous columns are not swept against each other: a relative
    tolerance that caught a three-way continuous dependence would also drop a
    column of NIST's full-rank Filippelli design (``s_min / s_max ~ 6e-16``).
    Those dependencies are caught after the cross-product fit fails (see
    :func:`_detect_low_order_linear_dependence`).
    """
    n, k = X.shape
    names = list(var_names)
    omitted: List[Tuple[int, str]] = []
    if k == 0 or n == 0:
        return omitted

    has_const = False
    for j in range(k):
        col = X[:, j]
        if np.ptp(col) == 0 and col[0] != 0:
            has_const = True
            break

    kept: List[int] = []
    const_kept: Optional[int] = None
    # Discrete sweep state: kept discrete columns and the Cholesky factor of
    # their cross-product (grown one column at a time).
    disc_kept: List[int] = []
    L = np.zeros((0, 0))
    centered = X - X.mean(axis=0) if has_const else X
    norms = np.sqrt(np.sum(centered * centered, axis=0))

    for j in range(k):
        col = X[:, j]
        scale = max(1.0, float(np.max(np.abs(col))))
        if not np.any(col):
            omitted.append((j, "zero column"))
            continue
        is_const = np.ptp(col) <= 1e-12 * scale
        if is_const:
            if const_kept is not None:
                omitted.append((j, f"constant, collinear with '{names[const_kept]}'"))
                continue
            const_kept = j
        else:
            dup = None
            for i in kept:
                if i == const_kept or norms[i] == 0 or norms[j] == 0:
                    continue
                c = float(centered[:, i] @ centered[:, j]) / (norms[i] * norms[j])
                if np.isfinite(c) and abs(c) >= 1.0 - 1e-8:
                    dup = i
                    break
            if dup is not None:
                omitted.append((j, f"collinear with '{names[dup]}'"))
                continue

        is_discrete = bool(
            np.all(np.isfinite(col))
            and np.all(col == np.round(col))
            and float(np.max(np.abs(col))) <= 1e6
        )
        if is_discrete:
            g = X[:, disc_kept].T @ col if disc_kept else np.zeros(0)
            gjj = float(col @ col)
            v = solve_triangular(L, g, lower=True) if disc_kept else np.zeros(0)
            d = gjj - float(v @ v)
            if d <= 1e-9 * gjj:
                omitted.append((j, "collinear with earlier discrete regressors"))
                continue
            m = len(disc_kept)
            L_new = np.zeros((m + 1, m + 1))
            L_new[:m, :m] = L
            L_new[m, :m] = v
            L_new[m, m] = np.sqrt(d)
            L = L_new
            disc_kept.append(j)
        kept.append(j)
    return omitted


def _detect_low_order_linear_dependence(
    X: np.ndarray,
    var_names: Optional[List[str]],
) -> None:
    """Raise when a column is exactly spanned by two other columns.

    This guard is intentionally run only after a design has already failed the
    well-conditioned cross-product path. It catches structural mistakes such as
    ``x_sum = x1 + x2`` without adding work to ordinary regressions or using
    a rank tolerance that would reject NIST's ill-conditioned full-rank cases.
    """
    n, k = X.shape
    if k < 3:
        return
    # Keep the QR fallback cheap for wide designs. Pairwise duplicate and
    # constant-column failures are already caught by the hot-path structural
    # check above; this targeted search handles the common low-order mistakes.
    if n * k * (k - 1) * (k - 2) // 2 > _LOW_ORDER_DEP_MAX_WORK:
        return

    names = list(var_names) if var_names is not None else [f"x{i}" for i in range(k)]
    eps = np.finfo(float).eps
    # Scan from the last column: the later member of a dependent set is the
    # one reported (and omitted by ``collinear='omit'``), as in Stata.
    for target in reversed(range(k)):
        y_col = X[:, target]
        y_norm = float(np.linalg.norm(y_col))
        others = [idx for idx in range(k) if idx != target]
        for first_pos, first in enumerate(others[:-1]):
            for second in others[first_pos + 1 :]:
                basis = X[:, [first, second]]
                coeffs, *_ = np.linalg.lstsq(basis, y_col, rcond=None)
                fitted = basis @ coeffs
                residual_norm = float(np.linalg.norm(y_col - fitted))
                scale = y_norm + float(np.linalg.norm(fitted)) + 1.0
                if residual_norm <= 256 * eps * scale:
                    raise NumericalInstability(
                        f"Regressor '{names[target]}' is an exact linear "
                        f"combination of '{names[first]}' and "
                        f"'{names[second]}'; the design matrix is "
                        f"rank-deficient and coefficients are not separately "
                        f"identified.",
                        recovery_hint=(
                            f"Drop '{names[target]}' or one of "
                            f"'{names[first]}'/'{names[second]}'."
                        ),
                        diagnostics={
                            "linear_dependence": {
                                "target": names[target],
                                "basis": [names[first], names[second]],
                                "coefficients": [
                                    float(coeffs[0]),
                                    float(coeffs[1]),
                                ],
                            }
                        },
                    )


def _numba_kernels() -> tuple[
    _OlsKernel,
    _SandwichKernel,
    _ClusterMeatKernel,
    _HacMeatKernel,
]:
    """Load accelerated kernels only when OLS is actually estimated."""
    from ..core._numba_kernels import cluster_meat, hac_meat, ols_fit, sandwich_hc

    return (
        cast(_OlsKernel, ols_fit),
        cast(_SandwichKernel, sandwich_hc),
        cast(_ClusterMeatKernel, cluster_meat),
        cast(_HacMeatKernel, hac_meat),
    )


class OLSEstimator(BaseEstimator):
    """
    Ordinary Least Squares estimator with robust standard errors
    """

    def estimate(
        self,
        y: np.ndarray,
        X: np.ndarray,
        robust: str = "nonrobust",
        cluster: Optional[pd.Series] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Estimate OLS parameters

        Parameters
        ----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Independent variables (including constant if desired)
        robust : str, default 'nonrobust'
            Type of standard errors ('nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac')
        cluster : pd.Series, optional
            Cluster variable for clustered standard errors
        **kwargs
            Additional options

        Returns
        -------
        Dict[str, Any]
            Estimation results
        """
        y, X = _validate_ols_arrays(y, X, context="OLSEstimator")
        n, k = X.shape
        # A constant outcome has no variation to explain: R-squared is
        # undefined and the fit is degenerate. Warn explicitly rather than
        # relying on a NumPy divide-by-zero RuntimeWarning, which newer NumPy
        # (>= 2.x) no longer reliably emits.
        if n > 1 and float(np.ptp(np.asarray(y, dtype=float))) == 0.0:
            warnings.warn(
                "OLS outcome has zero variance (constant y): the model cannot "
                "explain any variation and R-squared is undefined.",
                stacklevel=2,
            )
        if n <= k:
            raise DataInsufficient(
                "OLS requires more observations than parameters to estimate "
                "residual variance and standard errors: "
                f"nobs={n}, parameters={k}, residual df={n - k}."
            )
        # Stata ``regress, vce(robust)`` is HC1; ``True`` means the same.
        robust_key = "hc1" if robust is True else str(robust).lower()
        ewc_df: Optional[int] = None
        if robust_key == "robust":
            robust_key = "hc1"
        elif robust_key in ("false", "none", "ols", "oim", "iid", "classical"):
            robust_key = "nonrobust"
        var_names = kwargs.pop("var_names", None)
        if var_names is not None:
            var_names = list(var_names)
            if len(var_names) != k:
                var_names = None

        # ---- Analytic weights (Stata ``aweight``) ---------------------------
        # When weights are supplied we fit WLS by running the unweighted kernel
        # on the sqrt(w)-transformed design: with X̃ = √w ⊙ X and ỹ = √w ⊙ y the
        # kernel returns β̂ = (X'WX)⁻¹X'Wy and residuals r̃ = √w(y − Xβ̂), so
        # (X̃'X̃)⁻¹ is the correct WLS "bread" and Σr̃² = Σ w r² feeds every VCE
        # branch unchanged (classical, HCk, cluster). Point estimates and all
        # standard errors are invariant to weight scaling, so normalising to
        # Σw = n only pins the reported σ̂/RMSE and R² to Stata's aweight
        # convention. The unweighted path is byte-identical (sw stays None).
        weights = kwargs.get("weights", None)
        X_orig, y_orig, sw = X, y, None
        if weights is not None:
            w = _validate_analytic_weights(
                weights,
                n,
                context="OLS analytic weights",
            )
            w = w * (n / w.sum())  # Stata aweight normalisation: Σw = n
            sw = np.sqrt(w)
            X = X * sw[:, None]
            y = y * sw

        # Fast OLS via Numba-accelerated kernel (graceful fallback)
        (
            _fast_ols,
            _fast_sandwich_hc,
            _fast_cluster_meat,
            _fast_hac_meat,
        ) = _numba_kernels()

        # Mean-centered (Frisch-Waugh-Lovell) fit when an intercept is present.
        # Fitting the raw design when y (or a regressor) carries a large
        # constant offset destroys the slope coefficients through catastrophic
        # cancellation: the kernel projects y ~ 1e12 onto contrast directions
        # and only ~3 significant digits of the O(1) signal survive (NIST StRD
        # SmLs07-09). Centering first makes the slope regression operate on
        # O(1) deviations; FWL guarantees identical coefficients to the raw fit
        # in exact arithmetic, so well-conditioned designs are unchanged to
        # machine precision while offset designs recover to the float64 floor.
        const_col = _detect_constant_column(X)
        if const_col is not None and k > 1:
            other = [j for j in range(k) if j != const_col]
            X_other = X[:, other]
            x_mean = X_other.mean(axis=0)
            y_mean = y.mean()
            X_centered = X_other - x_mean
            y_centered = y - y_mean
            try:
                (
                    slopes,
                    _,
                    residuals,
                    slope_xtx_inv,
                ) = _crossprod_fit_if_well_conditioned(X_centered, y_centered)
            except np.linalg.LinAlgError:
                other_names = None
                if var_names is not None:
                    other_names = [var_names[j] for j in other]
                _detect_low_order_linear_dependence(X_centered, other_names)
                try:
                    slopes, _, residuals, slope_xtx_inv = _qr_fit_with_bread(
                        X_centered, y_centered
                    )
                except np.linalg.LinAlgError:
                    params, fitted_values, residuals = _fast_ols(X, y)
                    XtX_inv = np.linalg.pinv(X.T @ X)
                    warnings.warn("X'X matrix is singular, using pseudo-inverse")
                else:
                    params = np.empty(k, dtype=float)
                    for pos, j in enumerate(other):
                        params[j] = slopes[pos]
                    const_value = float(X[0, const_col])
                    params[const_col] = (y_mean - x_mean @ slopes) / const_value
                    fitted_values = y - residuals
                    XtX_inv = _centered_intercept_bread(
                        n=n,
                        k=k,
                        const_col=const_col,
                        other=other,
                        const_value=const_value,
                        x_mean=x_mean,
                        slope_xtx_inv=slope_xtx_inv,
                    )
            else:
                params = np.empty(k, dtype=float)
                for pos, j in enumerate(other):
                    params[j] = slopes[pos]
                const_value = float(X[0, const_col])
                params[const_col] = (y_mean - x_mean @ slopes) / const_value
                fitted_values = y - residuals

                # Reuse the centered cross-product inverse for the OLS bread.
                # For X=[c, Z], with centered Zc and A=(Zc'Zc)^-1:
                # inv(X'X) = [[1/(n c^2)+mAm'/c^2, -mA/c], [-Am/c, A]].
                XtX_inv = _centered_intercept_bread(
                    n=n,
                    k=k,
                    const_col=const_col,
                    other=other,
                    const_value=const_value,
                    x_mean=x_mean,
                    slope_xtx_inv=slope_xtx_inv,
                )
        else:
            # Use cross-products only for comfortably conditioned designs; QR
            # remains the certified fallback for hard numerical cases.
            try:
                (
                    params,
                    fitted_values,
                    residuals,
                    XtX_inv,
                ) = _crossprod_fit_if_well_conditioned(X, y)
            except np.linalg.LinAlgError:
                _detect_low_order_linear_dependence(X, var_names)
                try:
                    params, fitted_values, residuals, XtX_inv = _qr_fit_with_bread(X, y)
                except np.linalg.LinAlgError:
                    params, fitted_values, residuals = _fast_ols(X, y)
                    XtX_inv = np.linalg.pinv(X.T @ X)
                    warnings.warn("X'X matrix is singular, using pseudo-inverse")

        # Variance-covariance via accelerated sandwich kernels
        if cluster is not None:
            cluster_arr = np.asarray(cluster)
            if cluster_arr.shape[0] != n:
                raise MethodIncompatibility(
                    "cluster length does not match the estimation sample after "
                    f"missing-data filtering: got {cluster_arr.shape[0]}, "
                    f"expected {n}."
                )
            if pd.isna(cluster_arr).any():
                raise DataInsufficient(
                    "Cluster-robust OLS inference requires non-missing cluster "
                    "labels for every observation in the estimation sample."
                )
            n_clusters = len(pd.unique(cluster_arr))
            if n_clusters < 2:
                raise DataInsufficient(
                    "Cluster-robust OLS inference requires at least two clusters."
                )
            meat = _fast_cluster_meat(X, residuals, cluster_arr)
            correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
            var_cov = correction * XtX_inv @ meat @ XtX_inv
        elif robust_key == "nonrobust":
            sigma2 = np.sum(residuals**2) / (n - k)
            var_cov = sigma2 * XtX_inv
        elif robust_key in ["hc0", "hc1", "hc2", "hc3"]:
            var_cov = _fast_sandwich_hc(X, residuals, XtX_inv, robust_key)
        elif robust_key == "hac":
            lags = kwargs.get("lags", None)
            if lags is not None and lags >= n:
                raise DataInsufficient(
                    f"HAC: {lags} autocovariances need more than {n} " "observations.",
                    recovery_hint="Lower hac_lags.",
                )
            meat = _fast_hac_meat(X, residuals, lags)
            var_cov = XtX_inv @ meat @ XtX_inv
            if kwargs.get("hac_small"):
                # Stata `newey` / sandwich::NeweyWest(adjust = TRUE)
                var_cov = var_cov * (n / (n - k))
        elif robust_key == "ewc":
            # Equal-weighted cosine estimator of the long-run variance
            # (Lazarus, Lewis, Stock and Watson 2018, eq. 10): the average of
            # the outer products of the first nu Type II cosine transforms
            # of z_t = x_t * u_t.
            nu = kwargs.get("ewc_df", None)
            if nu is None:
                nu = int(np.floor(0.4 * n ** (2.0 / 3.0)))  # their eq. (4)
            if not 1 <= nu < n:
                raise DataInsufficient(
                    f"EWC: {nu} cosine terms with {n} observations.",
                    recovery_hint="ewc_df must be between 1 and N - 1.",
                )
            ewc_df = int(nu)
            moments = X * residuals[:, None]
            t_mid = np.arange(1, n + 1) - 0.5
            cosines = np.cos(np.pi * np.outer(t_mid, np.arange(1, ewc_df + 1)) / n)
            lam = np.sqrt(2.0 / n) * (cosines.T @ moments)  # (nu, k)
            omega = lam.T @ lam / ewc_df
            var_cov = XtX_inv @ (n * omega) @ XtX_inv
        else:
            raise MethodIncompatibility(f"Unknown robust option: {robust}")

        std_errors = np.sqrt(np.diag(var_cov))

        # Model diagnostics. For WLS, report fitted/residuals on the ORIGINAL
        # scale and use weighted TSS/RSS so R²/RMSE match Stata's aweight output.
        # Without a constant the total sum of squares is taken about zero:
        # R-squared, its adjustment and the F statistic are the uncentred
        # ones (Stata `regress, noconstant`, R `lm(y ~ x - 1)`). Measured
        # about the mean, a line through the origin can have a negative
        # R-squared.
        has_const = _detect_constant_column(X_orig) is not None
        if sw is not None:
            fitted_values = X_orig @ params
            residuals = y_orig - fitted_values
            wn = sw**2
            ybar_w = np.sum(wn * y_orig) / np.sum(wn) if has_const else 0.0
            tss = np.sum(wn * (y_orig - ybar_w) ** 2)
            rss = np.sum(wn * residuals**2)
        else:
            tss = np.sum((y - np.mean(y)) ** 2) if has_const else np.sum(y**2)
            rss = np.sum(residuals**2)
        df_model = k - 1 if has_const else k
        if tss <= 0:
            r_squared = np.nan
            adj_r_squared = np.nan
        else:
            r_squared = 1 - rss / tss
            adj_r_squared = 1 - (rss / (n - k)) / (tss / (n - 1 if has_const else n))

        # F statistic: every slope is zero (with a constant), every
        # coefficient is zero (without one).
        if df_model > 0 and n > k and np.isfinite(r_squared):
            denom = (1 - r_squared) / (n - k)
            if denom <= 0:
                # Exact fit (R² == 1): F diverges. NIST StRD certifies this as
                # "Infinity" (e.g. Wampler1/2); report it without tripping a
                # divide-by-zero warning.
                f_stat = np.inf
                f_pvalue = 0.0
            else:
                f_stat = (r_squared / df_model) / denom
                f_pvalue = stats.f.sf(f_stat, df_model, n - k)
        else:
            f_stat = f_pvalue = np.nan

        # With a robust, clustered or HAC covariance the classical F (a ratio
        # of sums of squares) is not a test of anything: the model F is the
        # Wald statistic b'V^{-1}b / q on that covariance, as Stata and
        # R's lmtest::waldtest(vcov=) report it. Its denominator degrees of
        # freedom are N - K, or G - 1 with G clusters.
        if (
            (robust_key != "nonrobust" or cluster is not None)
            and df_model > 0
            and n > k
        ):
            const_col = _detect_constant_column(X_orig)
            tested = [j for j in range(k) if j != const_col]
            df2 = n - k
            if cluster is not None:
                df2 = int(len(pd.unique(np.asarray(cluster).ravel()))) - 1
            f_stat = f_pvalue = np.nan
            if df2 >= len(tested):
                b_t = params[tested]
                v_t = var_cov[np.ix_(tested, tested)]
                try:
                    wald = float(b_t @ np.linalg.solve(v_t, b_t))
                except np.linalg.LinAlgError:
                    wald = np.nan
                q = len(tested)
                if np.isfinite(wald) and wald >= 0 and robust_key == "ewc":
                    # B cosine terms: the Wald statistic is Hotelling's
                    # T^2(q, B), and (B - q + 1) / (B q) W is F(q, B - q + 1)
                    df_ewc = ewc_df - q + 1
                    if df_ewc >= 1:
                        f_stat = wald * df_ewc / (ewc_df * q)
                        f_pvalue = float(stats.f.sf(f_stat, q, df_ewc))
                elif np.isfinite(wald) and wald >= 0:
                    f_stat = wald / q
                    f_pvalue = float(stats.f.sf(f_stat, q, df2))

        return {
            "params": params,
            "std_errors": std_errors,
            "var_cov": var_cov,
            "fitted_values": fitted_values,
            "residuals": residuals,
            "r_squared": r_squared,
            "adj_r_squared": adj_r_squared,
            "f_statistic": f_stat,
            "f_pvalue": f_pvalue,
            "nobs": n,
            "df_model": df_model,
            "df_resid": n - k,
            "rss": rss,
            "tss": tss,
            "ewc_df": ewc_df,
        }

    def _robust_cov_matrix(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        XtX_inv: np.ndarray,
        robust_type: str,
    ) -> np.ndarray:
        """Calculate heteroskedasticity-robust covariance matrix"""
        n, k = X.shape

        if robust_type == "hc0":
            # White (1980)
            weights = residuals**2
        elif robust_type == "hc1":
            # Degree of freedom correction
            weights = (n / (n - k)) * residuals**2
        elif robust_type == "hc2":
            # MacKinnon and White (1985)
            h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
            weights = residuals**2 / (1 - h)
        elif robust_type == "hc3":
            # Davidson and MacKinnon (1993)
            h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
            weights = residuals**2 / (1 - h) ** 2

        # Sandwich estimator
        meat = (X * (weights)[:, None]).T @ X
        return np.asarray(XtX_inv @ meat @ XtX_inv, dtype=float)

    def _hac_cov_matrix(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        XtX_inv: np.ndarray,
        lags: Optional[int] = None,
    ) -> np.ndarray:
        """Calculate HAC (Newey-West) covariance matrix"""
        n, k = X.shape

        if lags is None:
            # Automatic lag selection (Newey-West rule)
            lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

        # Calculate centered moments.  The HAC meat is intentionally
        # unnormalised so ``XtX_inv @ meat @ XtX_inv`` has the same scale as
        # HC and clustered covariance estimators.
        moments = X * residuals[:, np.newaxis]

        # Gamma_0 (contemporaneous covariance)
        gamma_0 = moments.T @ moments

        # Gamma_j for j = 1, ..., lags
        gamma_sum = gamma_0.copy()
        for j in range(1, lags + 1):
            gamma_j = moments[j:].T @ moments[:-j]
            weight = 1 - j / (lags + 1)  # Bartlett kernel
            gamma_sum += weight * (gamma_j + gamma_j.T)

        return np.asarray(XtX_inv @ gamma_sum @ XtX_inv, dtype=float)

    def _cluster_cov_matrix(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        XtX_inv: np.ndarray,
        cluster: pd.Series,
    ) -> np.ndarray:
        """Calculate clustered standard errors"""
        n, k = X.shape

        # Get unique clusters
        clusters = cluster.unique()
        n_clusters = len(clusters)

        # Calculate cluster sum of moments
        meat = np.zeros((k, k))
        for cluster_id in clusters:
            cluster_idx = cluster == cluster_id
            X_c = X[cluster_idx]
            resid_c = residuals[cluster_idx]
            moments_c = (X_c * resid_c[:, np.newaxis]).sum(axis=0)
            meat += np.outer(moments_c, moments_c)

        # Finite sample correction
        correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))

        return np.asarray(correction * XtX_inv @ meat @ XtX_inv, dtype=float)


class OLSRegression(BaseModel):
    """
    OLS regression model with comprehensive functionality
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        var_names: Optional[List[str]] = None,
        collinear: str = "omit",
    ) -> None:
        """
        Initialize OLS regression

        Parameters
        ----------
        formula : str, optional
            Regression formula (e.g., "y ~ x1 + x2")
        data : pd.DataFrame, optional
            Data containing variables
        y : np.ndarray, optional
            Dependent variable (alternative to formula)
        X : np.ndarray, optional
            Independent variables (alternative to formula)
        var_names : List[str], optional
            Variable names when using y, X directly
        collinear : {'omit', 'raise'}, default 'omit'
            Rank-deficient design: omit the later member of each collinear
            set with a warning (Stata ``regress``), or raise
            :class:`NumericalInstability`.
        """
        super().__init__()
        if collinear not in ("omit", "raise"):
            raise MethodIncompatibility(
                f"collinear must be 'omit' or 'raise', got {collinear!r}"
            )
        self.collinear = collinear

        self.formula = formula
        self.data = data
        self.y = y
        self.X = X
        self.var_names = var_names
        self._design_info = None
        self.estimator = OLSEstimator()

    def _resolve_weights(
        self,
        weights: Any,
        design_index: Optional[pd.Index],
    ) -> np.ndarray:
        """Resolve and validate analytic regression weights (Stata ``aweight``).

        Accepts a column name (resolved against ``self.data`` and aligned to the
        design's row index, so rows dropped for missing data stay aligned) or an
        array-like of length ``nobs``. Fails loudly on length mismatch, NaN/inf,
        or non-positive weights rather than silently producing wrong estimates.
        """
        if isinstance(weights, str):
            if self.data is None or weights not in self.data.columns:
                raise ValueError(f"weights='{weights}' is not a column in the data.")
            col = self.data[weights]
            if design_index is not None:
                col = col.reindex(design_index)
            wv = np.asarray(col, dtype=float).ravel()
        else:
            wv = np.asarray(weights, dtype=float).ravel()
        if self.y is None:
            raise MethodIncompatibility(
                "OLS analytic weights cannot be resolved before y is prepared."
            )
        return _validate_analytic_weights(
            wv,
            self.y.shape[0],
            context="OLS analytic weights",
        )

    def fit(
        self,
        robust: str = "nonrobust",
        cluster: Optional[str] = None,
        **kwargs: Any,
    ) -> EconometricResults:
        """
        Fit the OLS model

        Parameters
        ----------
        robust : str, default 'nonrobust'
            Type of standard errors
        cluster : str, optional
            Variable name for clustering
        **kwargs
            Additional options

        Returns
        -------
        EconometricResults
            Fitted model results
        """
        # Prepare data
        design_index: Optional[pd.Index] = None
        if self.formula is not None and self.data is not None:
            y_df, X_df = create_design_matrices(self.formula, self.data)
            self._design_info = getattr(X_df, "design_info", None)
            design_index = y_df.index
            self.y = y_df.values.ravel()
            self.X = X_df.values
            self.var_names = list(X_df.columns)
            self.dependent_var = y_df.columns[0]
        elif self.y is not None and self.X is not None:
            self.dependent_var = "y"
        else:
            raise ValueError("Must provide either (formula, data) or (y, X)")

        self.y, self.X = _validate_ols_arrays(
            self.y, self.X, context="OLSRegression.fit"
        )
        if self.var_names is None:
            self.var_names = [f"x{i}" for i in range(self.X.shape[1])]
        elif len(self.var_names) != self.X.shape[1]:
            raise MethodIncompatibility(
                f"OLSRegression.fit: var_names has {len(self.var_names)} "
                f"entries but X has {self.X.shape[1]} columns"
            )

        # A rank-deficient design: omit the later member of each collinear set
        # as Stata's ``regress`` does (``collinear='omit'``, the default), or
        # fail loudly (``collinear='raise'``). Never return unidentified
        # coefficients.
        collinear = getattr(self, "collinear", "omit")
        omitted: List[Dict[str, str]] = []
        if collinear == "omit":
            order = _written_column_order(
                self.formula, self._design_info, self.X.shape[1]
            )
            drops = [
                (order[j], r)
                for j, r in _collinear_omissions(
                    self.X[:, order], [self.var_names[i] for i in order]
                )
            ]
            if drops:
                omitted = [
                    {"variable": self.var_names[j], "reason": r} for j, r in drops
                ]
                keep = [
                    j for j in range(self.X.shape[1]) if j not in {d[0] for d in drops}
                ]
                self.X = self.X[:, keep]
                self.var_names = [self.var_names[j] for j in keep]
        _detect_perfect_collinearity(self.X, self.var_names)

        # Resolve analytic regression weights (Stata ``aweight`` semantics).
        # These were previously accepted via **kwargs and *silently ignored*,
        # returning unweighted OLS — a fail-silently correctness bug. Now they
        # are resolved, validated, and threaded into the WLS kernel.
        if kwargs.get("weights", None) is not None:
            kwargs["weights"] = self._resolve_weights(kwargs["weights"], design_index)

        # Handle clustering
        cluster_var: Optional[pd.Series] = None
        if cluster and self.data is not None:
            if cluster not in self.data.columns:
                raise MethodIncompatibility(
                    f"cluster='{cluster}' is not a column in the data."
                )
            cluster_var = self.data[cluster]
            if design_index is not None:
                cluster_var = cluster_var.reindex(design_index)

        # Estimate model
        while True:
            try:
                results = self.estimator.estimate(
                    self.y,
                    self.X,
                    robust=robust,
                    cluster=cluster_var,
                    var_names=self.var_names,
                    **kwargs,
                )
                break
            except NumericalInstability as exc:
                # A continuous exact dependence (``x3 = x1 + x2``) only shows
                # once the cross-product fit fails; omit its later member.
                dep = (getattr(exc, "diagnostics", None) or {}).get("linear_dependence")
                if (
                    collinear != "omit"
                    or not dep
                    or dep["target"] not in self.var_names
                ):
                    raise
                j = self.var_names.index(dep["target"])
                omitted.append(
                    {
                        "variable": dep["target"],
                        "reason": "linear combination of '"
                        + "', '".join(dep["basis"])
                        + "'",
                    }
                )
                self.X = np.delete(self.X, j, axis=1)
                self.var_names = [v for i, v in enumerate(self.var_names) if i != j]

        if omitted:
            warnings.warn(
                "regress: "
                + "; ".join(
                    f"note: {o['variable']} omitted because of collinearity "
                    f"({o['reason']})"
                    for o in omitted
                )
                + ". Pass collinear='raise' to fail instead.",
                UserWarning,
                stacklevel=3,
            )

        # Create results object
        params = pd.Series(results["params"], index=self.var_names)
        std_errors = pd.Series(results["std_errors"], index=self.var_names)

        # Surface the number of clusters so few-cluster inference risk is
        # machine-readable (result.violations()) and warned loudly at fit time,
        # mirroring sp.panel — cluster-robust SEs are unreliable with few
        # clusters (Cameron-Gelbach-Miller 2008).
        if cluster_var is not None:
            from ..core._agent_summary import (
                _FEW_CLUSTERS_MIN,
                FEW_CLUSTERS_HINT,
                effective_n_clusters,
            )

            n_clusters_obs = int(pd.Series(cluster_var).nunique())
            n_clusters_eff = effective_n_clusters(cluster_var)
            if (
                n_clusters_obs >= _FEW_CLUSTERS_MIN
                and n_clusters_eff < _FEW_CLUSTERS_MIN
            ):
                # Enough clusters by count, few in effect: a handful of
                # them hold most of the sample.
                top_share = float(
                    pd.Series(cluster_var).value_counts(normalize=True).iloc[0]
                )
                warnings.warn(
                    AssumptionWarning(
                        f"{n_clusters_obs} clusters for cluster='{cluster}', "
                        f"but unequal in size: the effective number is "
                        f"{n_clusters_eff:.1f} (< {_FEW_CLUSTERS_MIN}; the "
                        f"largest holds {100 * top_share:.0f}% of the rows). "
                        "Cluster-robust t-tests over-reject in that case as "
                        "they do with few clusters.",
                        recovery_hint=FEW_CLUSTERS_HINT,
                        diagnostics={
                            "n_clusters": n_clusters_obs,
                            "n_clusters_effective": n_clusters_eff,
                            "largest_cluster_share": top_share,
                            "threshold": _FEW_CLUSTERS_MIN,
                        },
                        alternative_functions=[
                            "sp.wild_cluster_bootstrap",
                            "sp.wild_cluster_ci_inv",
                        ],
                    ),
                    stacklevel=2,
                )
            if n_clusters_obs < _FEW_CLUSTERS_MIN:
                warnings.warn(
                    AssumptionWarning(
                        f"Only {n_clusters_obs} clusters (< {_FEW_CLUSTERS_MIN}) "
                        f"for cluster='{cluster}' — cluster-robust SEs are "
                        "downward-biased and t-tests over-reject with few "
                        "clusters.",
                        recovery_hint=FEW_CLUSTERS_HINT,
                        diagnostics={
                            "n_clusters": n_clusters_obs,
                            "threshold": _FEW_CLUSTERS_MIN,
                        },
                        alternative_functions=[
                            "sp.wild_cluster_bootstrap",
                            "sp.wild_cluster_ci_inv",
                        ],
                    ),
                    stacklevel=2,
                )

        model_info = {
            "model_type": "OLS",
            "method": "Least Squares",
            "robust": robust,
            "cluster": cluster,
        }
        if kwargs.get("weights", None) is not None:
            model_info["weighted"] = True
        if omitted:
            model_info["omitted"] = omitted
        if cluster_var is not None:
            model_info["n_clusters"] = n_clusters_obs
            model_info["n_clusters_effective"] = n_clusters_eff

        data_info = {
            "nobs": results["nobs"],
            "df_model": results["df_model"],
            "df_resid": results["df_resid"],
            "dependent_var": self.dependent_var,
            "fitted_values": results["fitted_values"],
            "residuals": results["residuals"],
            "X": self.X,
            "y": self.y,
            "var_cov": results.get("var_cov"),
            "var_names": self.var_names,
            # sums of squares on the scale the R-squared uses (weighted, with
            # analytic weights normalised to sum to N)
            "rss": float(results["rss"]),
            "tss": float(results["tss"]),
        }
        # The rows of ``data`` the fit used, and the analytic weights on
        # them: a variance computed afterwards on the stored design has to
        # align its cluster keys to these rows and weight the same way.
        if design_index is not None:
            data_info["sample_index"] = design_index
        if kwargs.get("weights", None) is not None:
            data_info["analytic_weights"] = np.asarray(kwargs["weights"], dtype=float)
        if cluster_var is not None:
            # Stata ``regress, vce(cluster)``: t / F with G - 1 degrees of
            # freedom. p-values and intervals used t(N - K), which on 40
            # clusters and 1,200 observations understated a p-value by 17
            # orders of magnitude.
            data_info["df_inference"] = int(n_clusters_obs) - 1
        if results.get("ewc_df"):
            # the EWC t ratio is t(nu) and the rescaled Wald is F(m, nu-m+1)
            data_info["df_inference"] = int(results["ewc_df"])
            data_info["hotelling_df"] = int(results["ewc_df"])

        rss_per_obs = results["rss"] / results["nobs"]
        if rss_per_obs <= 0:
            log_likelihood = np.inf
            aic = -np.inf
            bic = -np.inf
        else:
            log_likelihood = (
                -0.5 * results["nobs"] * (np.log(2 * np.pi * rss_per_obs) + 1)
            )
            # -2 log L + 2k and -2 log L + k log N with k the number of
            # coefficients: Stata's `estat ic`. (R's AIC() counts the error
            # variance as one more parameter.)
            n_coef = len(params)
            aic = -2.0 * log_likelihood + 2 * n_coef
            bic = -2.0 * log_likelihood + np.log(results["nobs"]) * n_coef

        diagnostics = {
            "R-squared": results["r_squared"],
            "Adj. R-squared": results["adj_r_squared"],
            "F-statistic": results["f_statistic"],
            "Prob (F-statistic)": results["f_pvalue"],
            "Log-Likelihood": log_likelihood,
            "AIC": aic,
            "BIC": bic,
        }

        fitted_result = EconometricResults(
            params=params,
            std_errors=std_errors,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )

        self._results = fitted_result
        self.is_fitted = True
        return fitted_result

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        what: str = "mean",
        alpha: float = 0.05,
        return_df: bool = False,
    ) -> "np.ndarray | pd.DataFrame":
        """Generate predictions from the fitted OLS model.

        Parameters
        ----------
        data : pd.DataFrame, optional
            New data at which to predict. If ``None``, returns the
            in-sample fitted values.
        what : {"mean", "confidence", "prediction"}, default "mean"
            - ``"mean"`` — point predictions only (default).
            - ``"confidence"`` — point + ``(1-alpha)`` confidence interval
              for the conditional mean ``E[y | x]``.
            - ``"prediction"`` — point + ``(1-alpha)`` prediction interval
              for a new observation (wider than the CI by ``sqrt(sigma^2)``).
        alpha : float, default 0.05
            Significance level for the interval.
        return_df : bool, default False
            Return a DataFrame with columns ``["yhat", "lower", "upper"]``.
            Ignored (forces True) when ``what != "mean"``.

        Returns
        -------
        np.ndarray or pd.DataFrame
            Point predictions, optionally with interval columns.
        """
        if not self.is_fitted:
            raise MethodIncompatibility(
                "Model must be fitted before prediction.",
                recovery_hint="Call fit() before predict().",
                diagnostics={"is_fitted": False},
            )
        assert self._results is not None  # guaranteed by is_fitted
        valid_what = {"mean", "confidence", "prediction"}
        if not isinstance(what, str) or what not in valid_what:
            raise MethodIncompatibility(
                "`what` must be 'mean', 'confidence', or 'prediction'; "
                f"got {what!r}.",
                recovery_hint="Choose one of: mean, confidence, prediction.",
                diagnostics={"what": repr(what), "valid": sorted(valid_what)},
            )
        if what != "mean":
            try:
                alpha = float(alpha)
            except (TypeError, ValueError) as exc:
                raise MethodIncompatibility(
                    "`alpha` must be a finite number in the open interval (0, 1).",
                    recovery_hint=(
                        "Use alpha=0.05 for 95% intervals, or another value "
                        "strictly between 0 and 1."
                    ),
                    diagnostics={"alpha": repr(alpha)},
                ) from exc
            if not np.isfinite(alpha) or not (0.0 < alpha < 1.0):
                raise MethodIncompatibility(
                    "`alpha` must be a finite number in the open interval (0, 1).",
                    recovery_hint=(
                        "Use alpha=0.05 for 95% intervals, or another value "
                        "strictly between 0 and 1."
                    ),
                    diagnostics={"alpha": repr(alpha)},
                )

        # In-sample path
        if data is None:
            yhat = np.asarray(self._results.fitted_values()).ravel()
            if what == "mean" and not return_df:
                return yhat
            # Fall through to interval machinery using the training design X.
            if self.X is None:
                raise MethodIncompatibility(
                    "Model design matrix is unavailable for prediction.",
                    recovery_hint="Refit the model before calling predict().",
                    diagnostics={"missing_state": "X"},
                )
            X_new = np.asarray(self.X, dtype=float)
        else:
            if self.formula is None:
                raise MethodIncompatibility(
                    "Out-of-sample prediction requires the model to have been fit "
                    "with a formula (not raw y, X arrays).",
                    recovery_hint=(
                        "Fit OLSRegression with formula=... and data=..., or "
                        "call predict() without new data for in-sample fitted values."
                    ),
                    diagnostics={"formula": None},
                )
            # Build X from the RHS of the formula. patsy's dmatrices() wants
            # the LHS variable present in `data`; at prediction time we only
            # have the regressors, so use dmatrix on the RHS only.
            from patsy import PatsyError, build_design_matrices, dmatrix

            if self.var_names is None:
                raise MethodIncompatibility(
                    "Model variable names are unavailable; refit the model "
                    "before out-of-sample prediction.",
                    recovery_hint=(
                        "Refit OLSRegression with a formula-backed design before "
                        "calling predict(data=...)."
                    ),
                    diagnostics={"missing_state": "var_names"},
                )
            var_names = list(self.var_names)
            # pandas >= 3.0 string columns are StringDtype, which patsy cannot
            # sniff; coerce to object so prediction rebuilds the same design.
            data = _coerce_string_extension_dtypes(data)
            try:
                if self._design_info is not None:
                    X_df = build_design_matrices(
                        [self._design_info],
                        data,
                        return_type="dataframe",
                    )[0]
                else:
                    rhs = self.formula.split("~", 1)[1].strip()
                    X_df = dmatrix(rhs, data, return_type="dataframe")
            except (PatsyError, KeyError, ValueError) as exc:
                raise MethodIncompatibility(
                    "Could not build prediction design matrix from new data.",
                    recovery_hint=(
                        "Check that new data contains the formula regressors "
                        "and only categorical levels seen during model fitting."
                    ),
                    diagnostics={"formula": self.formula, "error": str(exc)},
                ) from exc
            missing = [nm for nm in var_names if nm not in X_df.columns]
            if missing:
                raise MethodIncompatibility(
                    f"New data is missing columns produced by the formula: {missing}",
                    recovery_hint=(
                        "Use data compatible with the fitted formula design, "
                        "or refit the model with the desired design."
                    ),
                    diagnostics={"missing_columns": missing},
                )
            X_new = np.asarray(X_df[var_names].values, dtype=float)
            params = np.asarray(self._results.params)
            yhat = X_new @ params

        if what == "mean" and not return_df:
            return yhat
        if what == "mean":
            return pd.DataFrame({"yhat": yhat})

        params = np.asarray(self._results.params)
        # Covariance of the estimated coefficients
        cov_source = (
            self._results.data_info.get("cov_params", None)
            if hasattr(self._results, "data_info")
            else None
        )
        if cov_source is None and hasattr(self._results, "data_info"):
            cov_source = self._results.data_info.get("var_cov", None)
        if cov_source is not None:
            cov = np.asarray(cov_source, dtype=float)
        else:
            # Reconstruct from std_errors (diagonal approximation if full cov missing)
            se = np.asarray(self._results.std_errors)
            cov = np.diag(se**2)
        if cov.shape != (params.shape[0], params.shape[0]):
            raise MethodIncompatibility(
                "Coefficient covariance matrix shape does not match model "
                "parameters.",
                recovery_hint=(
                    "Refit the model or provide a result object with a square "
                    "covariance matrix aligned to params."
                ),
                diagnostics={
                    "covariance_shape": list(cov.shape),
                    "n_parameters": int(params.shape[0]),
                },
            )

        # var(x' beta) = x' Σ x
        var_mean = np.einsum("ij,jk,ik->i", X_new, cov, X_new)
        var_mean = np.maximum(var_mean, 0.0)
        se_mean = np.sqrt(var_mean)

        df_resid = self._results.data_info.get("df_resid", np.inf)
        t_crit = stats.t.ppf(1 - alpha / 2, df_resid)

        if what == "confidence":
            lower = yhat - t_crit * se_mean
            upper = yhat + t_crit * se_mean
        elif what == "prediction":
            sigma2 = self._results.diagnostics.get("sigma2", None)
            if sigma2 is None:
                # fall back to residual variance
                e = np.asarray(self._results.data_info.get("residuals", []))
                sigma2 = float(e @ e) / df_resid if len(e) else 0.0
            se_pred = np.sqrt(var_mean + float(sigma2))
            lower = yhat - t_crit * se_pred
            upper = yhat + t_crit * se_pred
        else:
            raise MethodIncompatibility(
                "`what` must be 'mean', 'confidence', or 'prediction'; "
                f"got {what!r}.",
                recovery_hint="Choose one of: mean, confidence, prediction.",
                diagnostics={"what": repr(what)},
            )

        out = pd.DataFrame({"yhat": yhat, "lower": lower, "upper": upper})
        return out


def _sample_codes(base: Any, data: pd.DataFrame, column: str) -> np.ndarray:
    """Integer codes of ``data[column]`` on the rows ``base`` was fitted on.

    The fit drops rows with a missing value in the formula; a cluster key
    taken from ``data`` positionally would then belong to other rows.
    """
    if column not in data.columns:
        raise MethodIncompatibility(f"cluster='{column}' is not a column in the data.")
    keys = data[column]
    idx = (getattr(base, "data_info", None) or {}).get("sample_index")
    if idx is not None and len(idx) != len(data):
        keys = keys.loc[idx]
    if keys.isna().any():
        raise MethodIncompatibility(
            f"cluster='{column}' has missing values on the estimation sample.",
            recovery_hint=f"Drop the rows where {column!r} is missing first.",
        )
    codes = np.asarray(pd.Categorical(keys).codes)
    if len(codes) != len(base.data_info["y"]):
        raise MethodIncompatibility(
            f"cluster='{column}' cannot be aligned to the estimation sample "
            f"({len(codes)} keys for {len(base.data_info['y'])} fitted rows).",
            recovery_hint="Give the data a unique index, or drop missing rows "
            "before calling sp.regress.",
        )
    return codes


def _refuse_weights(weights: Any, what: str) -> None:
    """``weights=`` with a variance that would ignore them is an error."""
    if weights is not None:
        raise MethodIncompatibility(
            f"regress: weights= is not implemented with {what}.",
            recovery_hint="Use vce='hc1', cluster=, vce='cr2' or two-way "
            "clustering with weights, or drop weights=.",
            diagnostics={"vce": what},
        )


@accepts_aliases(vce="robust")
@markout_clusters
def regress(
    formula: str,
    data: pd.DataFrame,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    weights: Optional[Any] = None,
    *,
    vce: Optional[str] = None,
    vcov: Optional[Any] = None,
    conley_lat: Optional[str] = None,
    conley_lon: Optional[str] = None,
    conley_cutoff: Optional[float] = None,
    collinear: str = "omit",
    hac_lags: Optional[int] = None,
    hac_small: bool = False,
    ewc_df: Optional[int] = None,
    **kwargs: Any,
) -> EconometricResults:
    """
    Convenient function for OLS regression

    Parameters
    ----------
    formula : str
        Regression formula
    data : pd.DataFrame
        Data containing variables
    robust : str, default 'nonrobust'
        Type of standard errors ('nonrobust', 'hc0'–'hc3', 'hac', 'ewc';
        case-insensitive). ``'hac'`` is Newey-West with normal critical
        values. ``'ewc'`` is the equal-weighted cosine estimator of
        Lazarus, Lewis, Stock and Watson (2018), with t and F critical
        values whose degrees of freedom are the number of cosine terms: in
        time-series regressions it rejects a true null much less often than
        Newey-West with a short lag length.
    cluster : str or list of two str, optional
        Variable name for clustering. A list of two names gives two-way
        clustering by inclusion-exclusion, with the whole meat scaled by
        ``G_min/(G_min - 1) * (N - 1)/(N - K)``. Stata 18's
        ``regress, vce(cluster a b)`` and R ``sandwich::vcovCL`` scale each
        of the three components by its own ``G/(G - 1)`` instead;
        ``sp.twoway_cluster(result, data, a, b)`` returns that version.
    weights : str or array-like, optional
        Analytic regression weights (Stata ``aweight`` semantics). Pass a
        column name or an array of length ``nobs``. Fits WLS — point
        estimates, classical / HC1-HC3 / clustered / CR2 SEs and R² match
        ``regress y x [aw=w]``. Also honoured by ``vce='cr3'`` and two-way
        clustering; refused with ``vce='wild'`` and ``vce='conley'``, which
        do not implement them. Weights must be strictly positive and finite;
        invalid weights raise ``ValueError`` rather than being silently
        ignored.
    vcov : str or dict, optional
        pyfixest/``fixest``-style variance specification, accepted as a
        canonical alias so the same spelling works across ``sp.regress``
        and ``sp.feols``: ``"iid"`` / ``"hetero"`` / ``"HC0"``–``"HC3"``,
        or a cluster dict such as ``{"CRV1": "firm"}`` (``CRV2`` /
        ``CRV3`` select the matching small-sample correction). Mutually
        exclusive with ``robust`` / ``cluster`` / ``vce``.
    hac_lags : int, optional
        With ``robust='hac'``: the number of autocovariances in the
        Newey-West (Bartlett kernel) estimator, Stata's ``newey, lag(#)``
        and R's ``sandwich::NeweyWest(lag = #)``. Default: the Newey-West
        (1994) rule ``floor(4 * (T / 100) ** (2 / 9))``. The rows must be in
        time order. Stock and Watson's truncation parameter
        ``m = 0.75 * T ** (1 / 3)`` corresponds to ``hac_lags = m - 1``.
    hac_small : bool, default False
        With ``robust='hac'``: multiply the covariance by ``N / (N - K)``,
        as Stata's ``newey`` and ``sandwich::NeweyWest(adjust = TRUE)`` do.
        The default (no factor) is ``NeweyWest(adjust = FALSE)`` and
        statsmodels' ``cov_type='HAC'``.
    ewc_df : int, optional
        With ``robust='ewc'``: the number of cosine terms ``nu``, which is
        also the degrees of freedom of the t distribution used for p-values
        and intervals. Default ``floor(0.4 * T ** (2 / 3))``. The rows must
        be in time order. Joint tests through :func:`statspai.test` use the
        rescaled statistic ``(nu - m + 1) / (nu * m) * W``, which is
        ``F(m, nu - m + 1)``.
    **kwargs
        Additional options. Unrecognised keywords raise ``TypeError``
        rather than being ignored — a misspelled option must not
        silently fall back to default standard errors.

    Returns
    -------
    EconometricResults
        Fitted model results. With ``robust='hac'``,
        ``model_info['hac_lags']`` is the lag length used; with
        ``robust='ewc'``, ``model_info['ewc_df']`` the number of cosines.

    References
    ----------
    [@newey1987simple],
    [@lazarus2018har]

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> results = sp.regress("log_wage ~ education + experience", data=df)
    >>> bool(results.params["education"] > 0)
    True

    >>> results = sp.regress("log_wage ~ education + experience", data=df,
    ...                      robust='hc1', cluster='union')
    >>> "education" in results.params.index
    True
    """
    # --- vcov= (pyfixest spelling) -> native robust=/cluster=/vce= ---
    # Accepted as a canonical cross-estimator alias; previously it fell
    # through **kwargs and was dropped, silently returning default SEs.
    from ..core._vcov_spec import normalize_vcov

    if vcov is not None:
        _robust, _cluster, _vce = normalize_vcov(
            vcov=vcov,
            robust=robust,
            cluster=cluster,
            vce=vce,
            function="regress",
        )
        # The @accepts_aliases(vce="robust") decorator folds vce= into
        # robust=, and the CR2/CR3 dispatch below reads `robust`; write
        # the small-sample kind there rather than into the dead `vce`.
        cluster = _cluster
        if _vce is not None:
            robust = _vce
        else:
            robust = _robust if _robust is not None else "nonrobust"

    # --- Stata grammar: vce='robust' / True / 'cluster firm' / 'vce(hc3)' ---
    # ``vce='robust'`` (Stata's most common spelling) used to raise, and
    # ``vce='cluster'`` worked only by accident when cluster= was also set.
    if not isinstance(robust, dict):
        from ..core._vcov_spec import parse_se_request

        _se = parse_se_request(
            robust,
            cluster,
            function="regress",
            supported=(
                "nonrobust",
                "robust",
                "hc0",
                "hc1",
                "hc2",
                "hc3",
                "hac",
                "ewc",
                "cluster",
                "cr2",
                "cr3",
                "jackknife",
                "wild",
                "conley",
            ),
            multiway=True,
        )
        cluster = _se.cluster
        # regress, vce(robust) is HC1; a cluster request is carried by cluster=.
        robust = {"robust": "hc1", "cluster": "nonrobust"}.get(_se.kind, _se.kind)

    # --- Input validation (Stata-quality error messages) ---
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"'data' must be a pandas DataFrame, got {type(data).__name__}. "
            f"Example: sp.regress('y ~ x', data=df)"
        )
    if data.empty:
        raise ValueError("DataFrame is empty — no observations to regress.")
    # Check formula variables exist in data
    if "~" in formula:
        import re

        lhs, rhs = formula.split("~", 1)
        # Strip function calls: C(...), I(...), np.log(...), bs(...), etc.
        rhs_stripped = re.sub(r"[A-Za-z_][\w.]*\s*\([^)]*\)", "", rhs)
        # Split on operators
        rhs_stripped = re.sub(r"[+*:\-]", " ", rhs_stripped)
        tokens = rhs_stripped.split()
        # Keep only bare column identifiers (no digits, no '1'/'0')
        bare_vars = [
            v for v in tokens if re.match(r"^[A-Za-z_]\w*$", v) and v not in ("1", "0")
        ]
        # Include LHS dep var
        dep_check = lhs.strip()
        all_vars = (
            [dep_check] if re.match(r"^[A-Za-z_]\w*$", dep_check) else []
        ) + bare_vars
        missing = [v for v in all_vars if v not in data.columns]
        if missing:
            available = ", ".join(sorted(data.columns)[:10])
            raise ValueError(
                f"Variable(s) not found in data: {missing}. "
                f"Available columns: {available}"
                + (" ..." if len(data.columns) > 10 else "")
            )
    # Check for all-NaN outcome
    dep_var = formula.split("~")[0].strip()
    if dep_var in data.columns and data[dep_var].isna().all():
        raise ValueError(
            f"Outcome variable '{dep_var}' is entirely NaN — "
            f"cannot estimate regression."
        )

    # ``weights`` is an explicit parameter (Stata ``aweight`` semantics) but
    # the downstream ``OLSRegression.fit`` consumes it via ``**kwargs``; only
    # re-inject when provided so the no-weights path stays byte-identical.
    if weights is not None:
        kwargs["weights"] = weights

    # --- vce= dispatch (canonical alias of robust=; vce='CR2'/'CR3'/'conley'
    # take the native efficient path on the stored design; vce=cluster=[a,b]
    # runs CGM-2011 two-way; vce='jackknife' is an alias of CR3).
    vce_kw = kwargs.pop("vce", None) or robust
    if isinstance(vce_kw, str) and vce_kw.lower() in ("cr2", "cr3", "jackknife"):
        if cluster is None:
            raise MethodIncompatibility(
                f"regress(vce={vce_kw!r}) requires cluster=... (a cluster-robust "
                "small-sample adjustment)."
            )
        kind = "CR3" if vce_kw.lower() in ("cr3", "jackknife") else "CR2"
        base = regress(
            formula=formula,
            data=data,
            robust="nonrobust",
            cluster=cluster,
            collinear=collinear,
            weights=weights,
        )
        from ..inference.jackknife import cr_vcov_ols

        cl_codes = _sample_codes(base, data, cluster)
        se = cr_vcov_ols(
            base, cl_codes, power=0.5 if kind == "CR2" else 1.0, small_sample=False
        )
        base.std_errors = se
        # Reference distribution: t(G - 1), as Stata's vce(hc2 clustvar)
        # and vce(jackknife, cluster()) use. Through 1.36.0 this was the
        # normal, which at 6 clusters rejected a true null 12% (CR2) and
        # 7% (CR3) of the time (tests/reliability/few_clusters.py).
        from ..inference.jackknife import set_cluster_t_inference

        set_cluster_t_inference(base, se, len(np.unique(cl_codes)))
        base.model_info = dict(base.model_info)
        base.model_info["vcov_type"] = (
            f"{kind} cluster-robust (Pustejovsky-Tipton 2018; matches R "
            "sandwich::vcovCL)"
        )
        base.model_info["cluster"] = cluster
        return base

    if isinstance(vce_kw, str) and vce_kw.lower() == "conley":
        if conley_lat is None or conley_lon is None or conley_cutoff is None:
            raise MethodIncompatibility(
                "regress(vce='conley') requires conley_lat=, conley_lon=, and "
                "conley_cutoff= (planar distance cutoff in km; matches Stata acreg)."
            )
        _refuse_weights(weights, "vce='conley'")
        base = regress(
            formula=formula,
            data=data,
            robust="nonrobust",
            cluster=None,
            collinear=collinear,
        )
        from scipy import stats as _stats

        from ..inference.conley import ols_conley_vcov

        se = ols_conley_vcov(base, data, conley_lat, conley_lon, conley_cutoff)
        base.std_errors = se
        z = base.params / se
        base.pvalues = pd.Series(2 * _stats.norm.sf(np.abs(z)), index=base.params.index)
        crit = _stats.norm.ppf(0.975)
        base.conf_int_lower = base.params - crit * se
        base.conf_int_upper = base.params + crit * se
        base.model_info = dict(base.model_info)
        # These SEs replace the fitted ones and their p-values are normal;
        # mark the fit so conf_int() / tidy() / sp.test use z as well.
        base.data_info = dict(base.data_info, inference="z")
        base.model_info["vcov_type"] = (
            f"Conley spatial HAC (acreg planar, {conley_cutoff} km; uniform)"
        )
        return base

    if isinstance(vce_kw, str) and vce_kw.lower() == "jackknife":
        # leave-one-cluster-out 2SLS-style jackknife. Re-uses the same path as
        # the dedicated ``sp.jackknife_se`` standalone.
        if cluster is None:
            raise MethodIncompatibility(
                "regress(vce='jackknife') requires cluster=... ."
            )
        base = regress(
            formula=formula,
            data=data,
            robust="nonrobust",
            cluster=cluster,
            collinear=collinear,
        )
        from ..inference.jackknife import jackknife_se

        jk = jackknife_se(base, data, cluster=cluster)
        return jk

    if isinstance(vce_kw, str) and vce_kw.lower() in (
        "wild",
        "wildbootstrap",
        "wild_cluster",
        "wcr",
        "boottest",
    ):
        if cluster is None:
            raise MethodIncompatibility(
                "regress(vce='wild') requires cluster=... (the wild cluster "
                "bootstrap resamples residuals within clusters)."
            )
        from ..inference.jackknife import wild_cluster_boot as _wcb

        _refuse_weights(weights, "vce='wild'")
        base = regress(
            formula=formula,
            data=data,
            robust={"CRV1": cluster},
            cluster=cluster,
            collinear=collinear,
        )
        # The bootstrap reuses the stored design, which it can only align to
        # the rows that were fitted.
        _idx = base.data_info.get("sample_index")
        if _idx is not None and len(_idx) != len(data):
            data = data.loc[_idx]
        wild_reps = kwargs.pop("wild_reps", 999)
        wild_weight_type = kwargs.pop("wild_weight_type", "rademacher")
        seed = kwargs.pop("seed", None)
        se = pd.Series(np.nan, index=base.params.index, dtype=float)
        pvals = pd.Series(np.nan, index=base.params.index, dtype=float)
        ci_lo = pd.Series(np.nan, index=base.params.index, dtype=float)
        ci_hi = pd.Series(np.nan, index=base.params.index, dtype=float)
        for var in base.params.index:
            out = _wcb(
                base,
                data,
                cluster=cluster,
                variable=str(var),
                n_boot=wild_reps,
                weight_type=wild_weight_type,
                seed=seed,
            )
            se[var] = out["se_cluster"]
            pvals[var] = out["p_boot"]
            ci_lo[var], ci_hi[var] = out["ci_boot"]
        base.std_errors = se
        base.pvalues = pvals
        base.conf_int_lower = ci_lo
        base.conf_int_upper = ci_hi
        base.model_info = dict(base.model_info)
        # These SEs replace the fitted ones and their p-values are normal;
        # mark the fit so conf_int() / tidy() / sp.test use z as well.
        base.data_info = dict(base.data_info, inference="z")
        base.model_info["vcov_type"] = (
            f"WCR wild cluster bootstrap (Cameron-Gelbach-Miller 2008, "
            f"{wild_reps} reps, {wild_weight_type})"
        )
        base.model_info["cluster"] = cluster
        base.model_info["n_boot"] = wild_reps
        return base

    # Two-way: vce=cluster=[a,b] (list) takes the CGM-2011 inclusion-exclusion
    # sandwich on the projected-score meat.
    if isinstance(cluster, (list, tuple)) and len(cluster) == 2:
        c1, c2 = cluster
        base = regress(
            formula=formula,
            data=data,
            robust="nonrobust",
            cluster=c1,
            collinear=collinear,
            weights=weights,
        )
        from ..inference.jackknife import two_way_correction_ols

        c1_codes = _sample_codes(base, data, c1)
        c2_codes = _sample_codes(base, data, c2)
        c12_codes = pd.factorize(
            pd.Series(list(zip(c1_codes.tolist(), c2_codes.tolist())))
        )[0]
        se = two_way_correction_ols(base, c1_codes, c2_codes, c12_codes)
        base.std_errors = se
        # t(G_min - 1), the degrees of freedom Stata 18's
        # regress, vce(cluster a b) and fixest report; the normal was
        # used through 1.36.0.
        from ..inference.jackknife import set_cluster_t_inference

        set_cluster_t_inference(
            base, se, min(len(np.unique(c1_codes)), len(np.unique(c2_codes)))
        )
        base.model_info = dict(base.model_info)
        base.model_info["vcov_type"] = "two-way cluster (CGM 2011)"
        base.model_info["cluster"] = list(cluster)
        return base

    # Anything still in kwargs is unrecognised: reject rather than let it
    # fall through to fit(**kwargs) and vanish (a misspelled option must
    # not silently return default standard errors).
    from ..core._vcov_spec import reject_unknown_kwargs

    reject_unknown_kwargs(kwargs, function="regress", known=("weights",))

    model = OLSRegression(formula=formula, data=data, collinear=collinear)
    robust_kw = vce_kw if vce_kw is not None else robust
    is_hac = str(robust_kw).lower() == "hac" and cluster is None
    if (hac_lags is not None or hac_small) and not is_hac:
        raise MethodIncompatibility(
            "regress: hac_lags= and hac_small= only apply to robust='hac'; "
            f"this call asks for {robust_kw!r}"
            + (f" clustered on {cluster!r}." if cluster is not None else "."),
            recovery_hint="Add robust='hac', or drop the two options.",
        )
    if hac_lags is not None:
        if isinstance(hac_lags, bool) or int(hac_lags) != hac_lags or hac_lags < 0:
            raise MethodIncompatibility(
                f"regress: hac_lags must be a non-negative integer, got "
                f"{hac_lags!r}.",
                recovery_hint="hac_lags=0 is the heteroskedasticity-robust "
                "HC0 estimator.",
            )
        kwargs["lags"] = int(hac_lags)
    if hac_small:
        kwargs["hac_small"] = True
    is_ewc = str(robust_kw).lower() == "ewc"
    if is_ewc and cluster is not None:
        raise MethodIncompatibility(
            "regress: robust='ewc' is a time-series covariance estimator and "
            "cannot be combined with cluster=.",
            recovery_hint="Drop cluster= for a single time series, or use "
            "cluster-robust standard errors for a panel.",
        )
    if ewc_df is not None:
        if not is_ewc:
            raise MethodIncompatibility(
                f"regress: ewc_df= only applies to robust='ewc'; this call "
                f"asks for {robust_kw!r}.",
                recovery_hint="Add robust='ewc', or drop ewc_df.",
            )
        if isinstance(ewc_df, bool) or int(ewc_df) != ewc_df or ewc_df < 1:
            raise MethodIncompatibility(
                f"regress: ewc_df must be a positive integer, got {ewc_df!r}.",
                recovery_hint="The default is floor(0.4 * T ** (2 / 3)).",
            )
        kwargs["ewc_df"] = int(ewc_df)
    _result = model.fit(robust=robust_kw, cluster=cluster, **kwargs)
    if is_ewc:
        _result.model_info["ewc_df"] = int(_result.data_info["df_inference"])
    if is_hac:
        n_used = int(_result.data_info["nobs"])
        _result.model_info["hac_lags"] = (
            int(hac_lags)
            if hac_lags is not None
            else int(np.floor(4 * (n_used / 100) ** (2 / 9)))
        )
        _result.model_info["hac_small"] = bool(hac_small)
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.regress",
            params={
                "formula": formula,
                "robust": robust,
                "cluster": cluster,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ("weights", "vcov", "se_type")
                },
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover — provenance must never break fit
        pass
    return _result
