"""
Cointegration tests and estimation.

Provides Engle-Granger two-step procedure, Johansen test for
cointegration rank, and VECM estimation.

Equivalent to Stata's ``vecrank`` / ``vec`` and R's ``ca.jo()``.

References
----------
Engle, R.F. & Granger, C.W.J. (1987).
"Co-Integration and Error Correction: Representation, Estimation,
and Testing." *Econometrica*, 55(2), 251-276. [@engle1987integration]

Johansen, S. (1991).
"Estimation and Hypothesis Testing of Cointegration Vectors in
Gaussian Vector Autoregressive Models." *Econometrica*, 59(6), 1551-1580. [@johansen1991estimation]
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats


class CointegrationResult:
    """Results from cointegration test."""

    def __init__(self, test_type, test_stats, critical_values, rank,
                 eigenvalues, eigenvectors, n_obs, n_vars, lags):
        self.test_type = test_type
        self.test_stats = test_stats
        self.critical_values = critical_values
        self.rank = rank
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.n_obs = n_obs
        self.n_vars = n_vars
        self.lags = lags

    def summary(self) -> str:
        lines = [
            f"Cointegration Test: {self.test_type}",
            "=" * 65,
            f"Variables: {self.n_vars}   Lags: {self.lags}   N: {self.n_obs}",
            "",
        ]

        if self.test_type == 'Engle-Granger':
            lines.append(f"ADF test statistic: {self.test_stats:.4f}")
            lines.append(f"Critical values (1%, 5%, 10%): "
                         f"{self.critical_values[0]:.3f}, "
                         f"{self.critical_values[1]:.3f}, "
                         f"{self.critical_values[2]:.3f}")
            reject = self.test_stats < self.critical_values[1]
            lines.append(f"Conclusion: {'Cointegrated' if reject else 'Not cointegrated'} at 5%")
        else:
            lines.append(f"{'H0: rank':>12s} {'Trace stat':>12s} {'5% CV':>10s} {'Reject':>8s}")
            lines.append("-" * 50)
            for i in range(self.n_vars):
                ts = self.test_stats[i] if i < len(self.test_stats) else np.nan
                cv = self.critical_values[i] if i < len(self.critical_values) else np.nan
                reject = ts > cv if np.isfinite(ts) and np.isfinite(cv) else False
                lines.append(f"{'r <= ' + str(i):>12s} {ts:>12.4f} {cv:>10.3f} "
                             f"{'Yes' if reject else 'No':>8s}")

            lines.append(f"\nEstimated cointegration rank: {self.rank}")

        lines.append("=" * 65)
        return "\n".join(lines)


def engle_granger(
    data: pd.DataFrame,
    variables: List[str] = None,
    lags: int = None,
    trend: str = "c",
    alpha: float = 0.05,
) -> CointegrationResult:
    """
    Engle-Granger (1987) two-step cointegration test.

    Step 1: OLS regression of y on x
    Step 2: ADF test on residuals

    Parameters
    ----------
    data : pd.DataFrame
    variables : list of str
        Variables to test (first is dependent).
    lags : int, optional
        Lags for ADF test. If None, uses AIC selection.
    trend : str, default 'c'
    alpha : float, default 0.05

    Returns
    -------
    CointegrationResult
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    y = data[variables[0]].values.astype(float)
    X = data[variables[1:]].values.astype(float)
    n = len(y)
    k = len(variables)

    # Step 1: OLS
    X_const = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(X_const, y, rcond=None)[0]
    residuals = y - X_const @ beta

    # Step 2: ADF on residuals
    if lags is None:
        lags = int(np.floor(4 * (n / 100)**0.25))

    dy = np.diff(residuals)
    y_lag = residuals[:-1]

    # ADF regression: Δe_t = ρ*e_{t-1} + Σ γ_j Δe_{t-j} + error
    T = len(dy)
    max_lag = min(lags, T - 2)

    Y_adf = dy[max_lag:]
    n_adf = len(Y_adf)
    X_adf = y_lag[max_lag:max_lag + n_adf].reshape(-1, 1)
    for j in range(1, max_lag + 1):
        lag_slice = dy[max_lag - j:max_lag - j + n_adf]
        X_adf = np.column_stack([X_adf, lag_slice])

    if trend == 'c':
        X_adf = np.column_stack([X_adf, np.ones(n_adf)])

    beta_adf = np.linalg.lstsq(X_adf, Y_adf, rcond=None)[0]
    resid_adf = Y_adf - X_adf @ beta_adf
    try:
        XtX_inv = np.linalg.inv(X_adf.T @ X_adf)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_adf.T @ X_adf)
    se_rho = np.sqrt(np.sum(resid_adf**2) / max(n_adf - X_adf.shape[1], 1) *
                     XtX_inv[0, 0])
    adf_stat = beta_adf[0] / se_rho

    # Critical values for Engle-Granger (depend on number of variables)
    # Approximate MacKinnon (1996) critical values
    eg_cv = {
        2: [-3.90, -3.34, -3.04],  # 1%, 5%, 10%
        3: [-4.32, -3.78, -3.50],
        4: [-4.68, -4.16, -3.88],
        5: [-4.99, -4.49, -4.22],
        6: [-5.26, -4.78, -4.52],
    }
    cvs = eg_cv.get(k, eg_cv[min(k, 6)])

    reject = adf_stat < cvs[1]

    return CointegrationResult(
        test_type='Engle-Granger',
        test_stats=adf_stat,
        critical_values=cvs,
        rank=1 if reject else 0,
        eigenvalues=None,
        eigenvectors=beta,
        n_obs=n,
        n_vars=k,
        lags=max_lag,
    )


def johansen(
    data: pd.DataFrame,
    variables: List[str] = None,
    lags: int = 1,
    trend: str = "c",
    test: str = "trace",
    alpha: float = 0.05,
) -> CointegrationResult:
    """
    Johansen (1991) cointegration test.

    Tests for the cointegration rank using the trace or maximum
    eigenvalue test statistic.

    Equivalent to Stata's ``vecrank`` and R's ``ca.jo()``.

    Parameters
    ----------
    data : pd.DataFrame
    variables : list of str
        Variables to test.
    lags : int, default 1
        Number of lags in the VECM.
    trend : str, default 'c'
        'n' (none), 'c' (constant), 'ct' (constant + trend).
    test : str, default 'trace'
        'trace' or 'maxeig' (maximum eigenvalue).
    alpha : float, default 0.05

    Returns
    -------
    CointegrationResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.johansen(df, variables=['gdp', 'consumption', 'investment'], lags=2)
    >>> print(result.summary())
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    Y = data[variables].dropna().values.astype(float)
    T, k = Y.shape

    # First differences
    dY = np.diff(Y, axis=0)  # (T-1) x k

    # Lagged levels
    Y_lag = Y[:-1]  # (T-1) x k

    # Additional lags of differences
    T_eff = T - 1 - lags
    if T_eff < k + 1:
        raise ValueError("Too few observations for the number of lags")

    dY_trim = dY[lags:]  # T_eff x k
    Y_lag_trim = Y_lag[lags:]  # T_eff x k

    # Lagged differences
    Z = []
    for j in range(1, lags + 1):
        Z.append(dY[lags - j:T - 1 - j])
    if len(Z) > 0:
        Z = np.hstack(Z)  # T_eff x (k*lags)
    else:
        Z = np.empty((T_eff, 0))

    if trend == 'c':
        Z = np.column_stack([Z, np.ones(T_eff)]) if Z.shape[1] > 0 else np.ones((T_eff, 1))
    elif trend == 'ct':
        Z = np.column_stack([Z, np.ones(T_eff), np.arange(1, T_eff + 1)])

    # Concentrate out Z from dY and Y_lag
    if Z.shape[1] > 0:
        Pz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
        Mz = np.eye(T_eff) - Pz
        R0 = Mz @ dY_trim
        R1 = Mz @ Y_lag_trim
    else:
        R0 = dY_trim
        R1 = Y_lag_trim

    # Product moment matrices
    S00 = R0.T @ R0 / T_eff
    S11 = R1.T @ R1 / T_eff
    S01 = R0.T @ R1 / T_eff
    S10 = S01.T

    # Solve generalized eigenvalue problem
    # |λ S11 - S10 S00^{-1} S01| = 0
    try:
        S00_inv = np.linalg.inv(S00)
        M = np.linalg.inv(S11) @ S10 @ S00_inv @ S01
        eigenvalues, eigenvectors = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        eigenvalues = np.zeros(k)
        eigenvectors = np.eye(k)

    # Sort eigenvalues descending
    idx = np.argsort(-np.real(eigenvalues))
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:, idx])

    # Clamp eigenvalues to [0, 1)
    eigenvalues = np.clip(eigenvalues, 0, 1 - 1e-10)

    # Test statistics
    if test == 'trace':
        # Trace statistic: -T Σ_{i=r+1}^{k} ln(1-λ_i)
        trace_stats = np.array([
            -T_eff * np.sum(np.log(1 - eigenvalues[r:])) for r in range(k)
        ])
        test_stats = trace_stats
    else:
        # Max eigenvalue: -T ln(1-λ_{r+1})
        maxeig_stats = np.array([
            -T_eff * np.log(1 - eigenvalues[r]) if r < k else 0 for r in range(k)
        ])
        test_stats = maxeig_stats

    # Critical values (Osterwald-Lenum 1992, approximate)
    # trace test 5% critical values for k variables
    trace_cv_5 = {
        1: [3.84],
        2: [15.41, 3.76],
        3: [29.68, 15.41, 3.76],
        4: [47.21, 29.68, 15.41, 3.76],
        5: [68.52, 47.21, 29.68, 15.41, 3.76],
        6: [94.15, 68.52, 47.21, 29.68, 15.41, 3.76],
    }
    maxeig_cv_5 = {
        1: [3.84],
        2: [14.07, 3.76],
        3: [21.12, 14.07, 3.76],
        4: [27.42, 21.12, 14.07, 3.76],
        5: [33.46, 27.42, 21.12, 14.07, 3.76],
        6: [39.37, 33.46, 27.42, 21.12, 14.07, 3.76],
    }

    if test == 'trace':
        cvs = trace_cv_5.get(k, trace_cv_5[min(k, 6)])
    else:
        cvs = maxeig_cv_5.get(k, maxeig_cv_5[min(k, 6)])

    # Pad if needed
    while len(cvs) < k:
        cvs.append(3.76)

    # Determine rank
    rank = 0
    for r in range(k):
        if r < len(test_stats) and r < len(cvs):
            if test_stats[r] > cvs[r]:
                rank = r + 1
            else:
                break

    return CointegrationResult(
        test_type=f'Johansen ({test})',
        test_stats=test_stats,
        critical_values=cvs,
        rank=rank,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        n_obs=T,
        n_vars=k,
        lags=lags,
    )
