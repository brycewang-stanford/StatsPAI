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
Gaussian Vector Autoregressive Models." *Econometrica*, 59(6),
1551-1580. [@johansen1991estimation]
"""

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin


class CointegrationResult(ResultProtocolMixin):
    """Results from cointegration test.

    Produced by :func:`engle_granger` (and the Johansen routine). Exposes
    the test statistic(s), critical values, estimated cointegration rank
    and a formatted ``.summary()``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> T = 200
    >>> x = np.cumsum(rng.normal(size=T))      # random walk
    >>> y = 2.0 * x + rng.normal(size=T)       # cointegrated with x
    >>> df = pd.DataFrame({"y": y, "x": x})
    >>> res = sp.engle_granger(df, variables=["y", "x"])
    >>> type(res).__name__
    'CointegrationResult'
    >>> res.test_type
    'Engle-Granger'
    >>> res.n_vars
    2
    >>> isinstance(res.summary(), str)
    True

    References
    ----------
    [@engle1987integration]
    """

    def __init__(
        self,
        test_type: str,
        test_stats: Any,
        critical_values: list[float],
        rank: int,
        eigenvalues: Optional[np.ndarray],
        eigenvectors: Optional[np.ndarray],
        n_obs: int,
        n_vars: int,
        lags: int,
    ) -> None:
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

        if self.test_type == "Engle-Granger":
            lines.append(f"ADF test statistic: {self.test_stats:.4f}")
            lines.append(
                f"Critical values (1%, 5%, 10%): "
                f"{self.critical_values[0]:.3f}, "
                f"{self.critical_values[1]:.3f}, "
                f"{self.critical_values[2]:.3f}"
            )
            reject = self.test_stats < self.critical_values[1]
            conclusion = "Cointegrated" if reject else "Not cointegrated"
            lines.append(f"Conclusion: {conclusion} at 5%")
        else:
            lines.append(
                f"{'H0: rank':>12s} {'Trace stat':>12s}"
                f" {'5% CV':>10s} {'Reject':>8s}"
            )
            lines.append("-" * 50)
            for i in range(self.n_vars):
                ts = self.test_stats[i] if i < len(self.test_stats) else np.nan
                cv = (
                    self.critical_values[i] if i < len(self.critical_values) else np.nan
                )
                reject = ts > cv if np.isfinite(ts) and np.isfinite(cv) else False
                lines.append(
                    f"{'r <= ' + str(i):>12s} {ts:>12.4f}"
                    f" {cv:>10.3f} {'Yes' if reject else 'No':>8s}"
                )

            lines.append(f"\nEstimated cointegration rank: {self.rank}")

        lines.append("=" * 65)
        return "\n".join(lines)


def engle_granger(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    lags: Optional[int] = None,
    trend: str = "c",
    alpha: float = 0.05,
) -> CointegrationResult:
    """
    Engle-Granger (1987) two-step cointegration test.

    Step 1: OLS regression of ``y`` (first variable) on the remaining
    variables and the deterministic terms selected by ``trend``.
    Step 2: augmented Dickey-Fuller regression on the step-1 residuals
    **without** deterministic terms,

    .. math:: \\Delta e_t = \\rho e_{t-1} + \\sum_{j=1}^{L} \\gamma_j
              \\Delta e_{t-j} + u_t ,

    whose ``t``-ratio on ``rho`` is the test statistic. This is the
    specification of Stata's ``egranger`` (Schaffer, SSC) and the one the
    MacKinnon critical values are tabulated for; the deterministic terms
    live in step 1 only.

    Parameters
    ----------
    data : pd.DataFrame
    variables : list of str
        Variables to test (first is dependent).
    lags : int, optional
        Number of lagged differences ``L`` in the step-2 regression
        (Stata ``egranger, lags()``). If None, uses the fixed rule
        ``int(4 * (n / 100) ** 0.25)``.
    trend : {'c', 'ct', 'ctt'}, default 'c'
        Deterministic terms in step 1: constant; constant + linear trend
        (``egranger, trend``); constant + linear + quadratic trend
        (``egranger, qtrend``). The trend is ``t = 0, 1, ...``.
    alpha : {0.01, 0.05, 0.10}, default 0.05
        Level at which ``rank`` (1 = cointegrated) is decided.

    Returns
    -------
    CointegrationResult
        ``test_stats`` is the ADF ``t``-ratio; ``critical_values`` the
        (1%, 5%, 10%) MacKinnon (2010) response-surface critical values for
        ``N = len(variables)`` series and ``T = n - 1`` (the ``egranger``
        convention); ``eigenvectors`` holds the step-1 coefficients
        (constant, regressors, trend terms).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> T = 200
    >>> x = np.cumsum(rng.normal(size=T))      # random walk
    >>> y = 2.0 * x + rng.normal(size=T)       # cointegrated with x
    >>> df = pd.DataFrame({"y": y, "x": x})
    >>> res = sp.engle_granger(df, variables=["y", "x"])
    >>> res.test_type
    'Engle-Granger'
    >>> res.n_vars
    2
    >>> bool(res.rank in (0, 1))
    True
    >>> isinstance(res.summary(), str)
    True

    References
    ----------
    [@engle1987integration]
    """
    from ._critvals import mackinnon_cv

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
    if trend not in ("c", "ct", "ctt"):
        raise ValueError("trend must be 'c', 'ct' or 'ctt' for engle_granger")
    level_pct = {0.01: 1, 0.05: 5, 0.10: 10}.get(round(float(alpha), 2))
    if level_pct is None:
        raise ValueError(
            "alpha must be 0.01, 0.05 or 0.10 (MacKinnon tabulates only these)"
        )
    k = len(variables)
    if not 2 <= k <= 12:
        raise ValueError("engle_granger needs 2..12 variables (MacKinnon 2010 tables)")

    Z = data[list(variables)].dropna().to_numpy(dtype=float)
    y = Z[:, 0]
    X = Z[:, 1:]
    n = len(y)

    # Step 1: OLS with the deterministic terms
    tt = np.arange(n, dtype=float)
    det = [np.ones(n)]
    if trend in ("ct", "ctt"):
        det.append(tt)
    if trend == "ctt":
        det.append(tt**2)
    X_const = np.column_stack([det[0], X] + det[1:])
    beta = np.linalg.lstsq(X_const, y, rcond=None)[0]
    residuals = y - X_const @ beta

    # Step 2: ADF on residuals, no deterministic terms
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** 0.25))
    lags = int(lags)
    if lags < 0:
        raise ValueError("lags must be >= 0")

    dy = np.diff(residuals)
    y_lag = residuals[:-1]
    T = len(dy)
    max_lag = min(lags, T - 2)

    Y_adf = dy[max_lag:]
    n_adf = len(Y_adf)
    X_adf = y_lag[max_lag : max_lag + n_adf].reshape(-1, 1)
    for j in range(1, max_lag + 1):
        lag_slice = dy[max_lag - j : max_lag - j + n_adf]
        X_adf = np.column_stack([X_adf, lag_slice])

    beta_adf = np.linalg.lstsq(X_adf, Y_adf, rcond=None)[0]
    resid_adf = Y_adf - X_adf @ beta_adf
    XtX_inv = np.linalg.inv(X_adf.T @ X_adf)
    se_rho = np.sqrt(np.sum(resid_adf**2) / (n_adf - X_adf.shape[1]) * XtX_inv[0, 0])
    adf_stat = float(beta_adf[0] / se_rho)

    # MacKinnon (2010) response surface, T = (step-1 observations) - 1
    cvs = [mackinnon_cv(trend, k, lvl, n - 1) for lvl in (1, 5, 10)]
    reject = adf_stat < cvs[(1, 5, 10).index(level_pct)]

    _result = CointegrationResult(
        test_type="Engle-Granger",
        test_stats=adf_stat,
        critical_values=cvs,
        rank=1 if reject else 0,
        eigenvalues=None,
        eigenvectors=beta,
        n_obs=n,
        n_vars=k,
        lags=max_lag,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.timeseries.engle_granger",
            params={
                "variables": list(variables) if variables else None,
                "lags": lags,
                "trend": trend,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


_JOHANSEN_TREND_ALIASES = {
    "n": "n",
    "none": "n",
    "rc": "rc",
    "rconstant": "rc",
    "c": "c",
    "constant": "c",
    "rt": "rt",
    "rtrend": "rt",
    "ct": "ct",
    "trend": "ct",
}


def johansen(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    lags: int = 1,
    trend: str = "c",
    test: str = "trace",
    alpha: float = 0.05,
) -> CointegrationResult:
    """
    Johansen (1991) cointegration test.

    Tests for the cointegration rank using the trace or maximum
    eigenvalue test statistic.

    Equivalent to Stata's ``vecrank`` and R's ``urca::ca.jo()``.

    Parameters
    ----------
    data : pd.DataFrame
    variables : list of str
        Variables to test.
    lags : int, default 1
        Number of lagged **differences** in the VECM. This is Stata's
        ``vecrank, lags(p)`` minus one and ``ca.jo(K = lags + 1)``.
    trend : str, default 'c'
        Deterministic specification (Stata ``vecrank, trend()`` names in
        parentheses): ``'n'`` (``none``); ``'rc'`` (``rconstant``, constant
        restricted to the cointegrating space = ``ca.jo(ecdet="const")``);
        ``'c'`` (``constant``, unrestricted constant =
        ``ca.jo(ecdet="none")``); ``'rt'`` (``rtrend``, trend restricted to
        the cointegrating space, unrestricted constant =
        ``ca.jo(ecdet="trend")``); ``'ct'`` (``trend``, unrestricted
        constant and trend). Stata's long names are accepted as aliases.
    test : {'trace', 'maxeig'}, default 'trace'
    alpha : {0.05, 0.01}, default 0.05
        Level of the Osterwald-Lenum critical values (the table Stata's
        ``vecrank`` uses) that decide ``rank``.

    Returns
    -------
    CointegrationResult
        ``test_stats[r]`` tests H0: rank <= r (r = 0..k-1);
        ``eigenvalues`` are the k largest squared canonical correlations;
        ``eigenvectors`` are the cointegrating vectors (columns), normalised
        so the first element is 1 (``ca.jo``'s ``@V`` convention; restricted
        cases carry the deterministic coefficient as the last row).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> trend = np.cumsum(rng.normal(size=200))  # common stochastic trend
    >>> df = pd.DataFrame({
    ...     'gdp': trend + rng.normal(scale=0.5, size=200),
    ...     'consumption': 0.8 * trend + rng.normal(scale=0.5, size=200),
    ...     'investment': 0.3 * trend + rng.normal(scale=0.5, size=200),
    ... })
    >>> result = sp.johansen(
    ...     df, variables=['gdp', 'consumption', 'investment'], lags=2)
    >>> type(result).__name__
    'CointegrationResult'
    >>> isinstance(result.summary(), str)
    True
    """
    from ._critvals import JOHANSEN_CV

    case = _JOHANSEN_TREND_ALIASES.get(str(trend).lower())
    if case is None:
        raise ValueError(
            "trend must be one of 'n', 'rc', 'c', 'rt', 'ct' (or Stata's "
            "none / rconstant / constant / rtrend / trend)"
        )
    if test not in ("trace", "maxeig"):
        raise ValueError("test must be 'trace' or 'maxeig'")
    alpha_key = round(float(alpha), 2)
    if alpha_key not in (0.05, 0.01):
        raise ValueError("alpha must be 0.05 or 0.01 (Osterwald-Lenum table)")
    lags = int(lags)
    if lags < 0:
        raise ValueError("lags must be >= 0")

    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    Y = data[variables].dropna().values.astype(float)
    T, k = Y.shape

    dY = np.diff(Y, axis=0)  # (T-1) x k
    Y_lag = Y[:-1]  # (T-1) x k

    T_eff = T - 1 - lags
    if T_eff < k + 1:
        raise ValueError("Too few observations for the number of lags")

    dY_trim = dY[lags:]  # T_eff x k
    Y_lag_trim = Y_lag[lags:]  # T_eff x k
    # time index of the effective sample; any origin gives the same
    # statistics (a shift is absorbed by the constant)
    tt = np.arange(lags + 2, T + 1, dtype=float)

    lag_blocks: list[np.ndarray] = [
        dY[lags - j : T - 1 - j] for j in range(1, lags + 1)
    ]
    Z = np.hstack(lag_blocks) if lag_blocks else np.empty((T_eff, 0))

    # Deterministic terms: unrestricted ones enter Z (concentrated out),
    # restricted ones are appended to the lagged levels.
    if case in ("c", "rt", "ct"):
        Z = np.column_stack([Z, np.ones(T_eff)])
    if case == "ct":
        Z = np.column_stack([Z, tt])
    if case == "rc":
        Y_lag_trim = np.column_stack([Y_lag_trim, np.ones(T_eff)])
    elif case == "rt":
        Y_lag_trim = np.column_stack([Y_lag_trim, tt])

    if Z.shape[1] > 0:
        coefs0 = np.linalg.lstsq(Z, dY_trim, rcond=None)[0]
        coefs1 = np.linalg.lstsq(Z, Y_lag_trim, rcond=None)[0]
        R0 = dY_trim - Z @ coefs0
        R1 = Y_lag_trim - Z @ coefs1
    else:
        R0 = dY_trim
        R1 = Y_lag_trim

    S00 = R0.T @ R0 / T_eff
    S11 = R1.T @ R1 / T_eff
    S01 = R0.T @ R1 / T_eff
    S10 = S01.T

    # |lambda S11 - S10 S00^{-1} S01| = 0, solved in symmetric form
    C = np.linalg.cholesky(S11)
    Cinv = np.linalg.inv(C)
    Msym = Cinv @ S10 @ np.linalg.solve(S00, S01) @ Cinv.T
    Msym = (Msym + Msym.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(Msym)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    V = Cinv.T @ eigvecs[:, order]
    with np.errstate(divide="ignore", invalid="ignore"):
        V = V / V[0, :]

    eigenvalues = np.clip(eigvals[:k], 0.0, 1.0 - 1e-15)
    eigenvectors = V

    if test == "trace":
        test_stats = np.array(
            [-T_eff * np.sum(np.log(1 - eigenvalues[r:])) for r in range(k)]
        )
    else:
        test_stats = np.array([-T_eff * np.log(1 - eigenvalues[r]) for r in range(k)])

    table = JOHANSEN_CV[test][alpha_key][case]
    # critical value for H0 rank <= r is indexed by k - r
    cvs = [table[k - r - 1] if k - r <= len(table) else np.nan for r in range(k)]
    if k > len(table):
        import warnings

        warnings.warn(
            f"johansen: critical values are tabulated for k - r <= {len(table)}; "
            "the rank decision ignores hypotheses beyond the table.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Rank: first r whose statistic does not exceed its critical value
    rank = 0
    for r in range(k):
        if np.isfinite(cvs[r]) and test_stats[r] > cvs[r]:
            rank = r + 1
        else:
            break

    _result = CointegrationResult(
        test_type=f"Johansen ({test})",
        test_stats=test_stats,
        critical_values=cvs,
        rank=rank,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        n_obs=T,
        n_vars=k,
        lags=lags,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.timeseries.johansen",
            params={
                "variables": list(variables) if variables else None,
                "lags": lags,
                "trend": trend,
                "test": test,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
