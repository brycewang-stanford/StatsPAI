"""
OLS regression implementation with comprehensive features
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
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
    for target in range(k):
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


def _numba_kernels() -> (
    tuple[
        _OlsKernel,
        _SandwichKernel,
        _ClusterMeatKernel,
        _HacMeatKernel,
    ]
):
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
        robust_key = str(robust).lower()
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
            meat = _fast_hac_meat(X, residuals, lags)
            var_cov = XtX_inv @ meat @ XtX_inv
        else:
            raise MethodIncompatibility(f"Unknown robust option: {robust}")

        std_errors = np.sqrt(np.diag(var_cov))

        # Model diagnostics. For WLS, report fitted/residuals on the ORIGINAL
        # scale and use weighted TSS/RSS so R²/RMSE match Stata's aweight output.
        if sw is not None:
            fitted_values = X_orig @ params
            residuals = y_orig - fitted_values
            wn = sw**2
            ybar_w = np.sum(wn * y_orig) / np.sum(wn)
            tss = np.sum(wn * (y_orig - ybar_w) ** 2)
            rss = np.sum(wn * residuals**2)
        else:
            tss = np.sum((y - np.mean(y)) ** 2)
            rss = np.sum(residuals**2)
        if tss <= 0:
            r_squared = np.nan
            adj_r_squared = np.nan
        else:
            r_squared = 1 - rss / tss
            adj_r_squared = 1 - (rss / (n - k)) / (tss / (n - 1))

        # F-statistic (assuming constant in first column)
        if k > 1 and np.isfinite(r_squared):
            r_squared_restricted = 0  # R² from constant-only model
            denom = (1 - r_squared) / (n - k)
            if denom <= 0:
                # Exact fit (R² == 1): F diverges. NIST StRD certifies this as
                # "Infinity" (e.g. Wampler1/2); report it without tripping a
                # divide-by-zero warning.
                f_stat = np.inf
                f_pvalue = 0.0
            else:
                f_stat = ((r_squared - r_squared_restricted) / (k - 1)) / denom
                f_pvalue = 1 - stats.f.cdf(f_stat, k - 1, n - k)
        else:
            f_stat = f_pvalue = np.nan

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
            "df_model": k - 1,
            "df_resid": n - k,
            "rss": rss,
            "tss": tss,
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
            h = np.diag(X @ XtX_inv @ X.T)
            weights = residuals**2 / (1 - h)
        elif robust_type == "hc3":
            # Davidson and MacKinnon (1993)
            h = np.diag(X @ XtX_inv @ X.T)
            weights = residuals**2 / (1 - h) ** 2

        # Sandwich estimator
        meat = X.T @ np.diag(weights) @ X
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
        """
        super().__init__()

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

        # Fail loudly on an exactly rank-deficient design rather than returning
        # unidentified garbage coefficients.
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
        results = self.estimator.estimate(
            self.y,
            self.X,
            robust=robust,
            cluster=cluster_var,
            var_names=self.var_names,
            **kwargs,
        )

        # Create results object
        params = pd.Series(results["params"], index=self.var_names)
        std_errors = pd.Series(results["std_errors"], index=self.var_names)

        # Surface the number of clusters so few-cluster inference risk is
        # machine-readable (result.violations()) and warned loudly at fit time,
        # mirroring sp.panel — cluster-robust SEs are unreliable with few
        # clusters (Cameron-Gelbach-Miller 2008).
        if cluster_var is not None:
            from ..core._agent_summary import _FEW_CLUSTERS_MIN

            n_clusters_obs = int(pd.Series(cluster_var).nunique())
            if n_clusters_obs < _FEW_CLUSTERS_MIN:
                warnings.warn(
                    AssumptionWarning(
                        f"Only {n_clusters_obs} clusters (< {_FEW_CLUSTERS_MIN}) "
                        f"for cluster='{cluster}' — cluster-robust SEs are "
                        "downward-biased and t-tests over-reject with few "
                        "clusters.",
                        recovery_hint=(
                            "Report sp.wild_cluster_bootstrap (or "
                            "sp.wild_cluster_ci_inv for CIs), correct with few "
                            "clusters."
                        ),
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
        if cluster_var is not None:
            model_info["n_clusters"] = n_clusters_obs

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
        }

        rss_per_obs = results["rss"] / results["nobs"]
        if rss_per_obs <= 0:
            log_likelihood = np.inf
            aic = -np.inf
            bic = -np.inf
        else:
            log_likelihood = (
                -0.5 * results["nobs"] * (np.log(2 * np.pi * rss_per_obs) + 1)
            )
            aic = results["nobs"] * np.log(rss_per_obs) + 2 * (results["df_model"] + 1)
            bic = results["nobs"] * np.log(rss_per_obs) + np.log(results["nobs"]) * (
                results["df_model"] + 1
            )

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


@accepts_aliases(vce="robust")
def regress(
    formula: str,
    data: pd.DataFrame,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    weights: Optional[Any] = None,
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
        Type of standard errors ('nonrobust', 'hc0'–'hc3', 'hac';
        case-insensitive)
    cluster : str, optional
        Variable name for clustering
    weights : str or array-like, optional
        Analytic regression weights (Stata ``aweight`` semantics). Pass a
        column name or an array of length ``nobs``. Fits WLS — point
        estimates, classical / robust / clustered SEs and R² match
        ``regress y x [aw=w]``. Weights must be strictly positive and finite;
        invalid weights raise ``ValueError`` rather than being silently
        ignored.
    **kwargs
        Additional options

    Returns
    -------
    EconometricResults
        Fitted model results

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
    model = OLSRegression(formula=formula, data=data)
    _result = model.fit(robust=robust, cluster=cluster, **kwargs)
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
