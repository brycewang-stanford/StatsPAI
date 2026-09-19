"""
Vector Autoregression (VAR) and Granger causality.

Provides VAR estimation, impulse response functions (IRF),
forecast error variance decomposition (FEVD), and Granger causality tests.

Equivalent to Stata's ``var`` / ``vargranger`` and R's ``vars::VAR()``.

References
----------
Lutkepohl, H. (2005).
"New Introduction to Multiple Time Series Analysis."
*Springer*.

Granger, C.W.J. (1969).
"Investigating Causal Relations by Econometric Models and Cross-spectral
Methods."
*Econometrica*, 37(3), 424-438. [@granger1969investigating]
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


class VARResult(ResultProtocolMixin):
    """Results from VAR estimation.

    Returned by :func:`var`; carries coefficient tables, the residual
    covariance, information criteria, and helper methods ``.irf()``,
    ``.granger_test()`` and ``.plot_irf()``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(3)
    >>> T = 200
    >>> x = np.zeros(T); y = np.zeros(T)
    >>> for t in range(1, T):
    ...     x[t] = 0.5 * x[t - 1] + rng.normal()
    ...     y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 1] + rng.normal()
    >>> df = pd.DataFrame({"y": y, "x": x})
    >>> vr = sp.var(df, variables=["y", "x"], lags=2)
    >>> type(vr).__name__
    'VARResult'
    >>> vr.var_names
    ['y', 'x']
    >>> vr.lags
    2
    >>> print(vr.summary().splitlines()[0])
    Vector Autoregression (VAR)

    References
    ----------
    lutkepohl2005new
    """

    def __init__(
        self,
        coefs: Dict[str, pd.DataFrame],
        se: Dict[str, np.ndarray],
        residuals: pd.DataFrame,
        sigma_u: pd.DataFrame,
        var_names: List[str],
        lags: int,
        n_obs: int,
        aic: float,
        bic: float,
        hqic: float,
        det_sigma: float,
        log_likelihood: float,
        se_df: str = "stata",
    ) -> None:
        self.coefs = coefs  # dict: var_name -> DataFrame of coefficients
        self.se = se
        self.residuals = residuals
        self.sigma_u = sigma_u  # residual covariance matrix
        self.var_names = var_names
        self.lags = lags
        self.n_obs = n_obs
        self.aic = aic
        self.bic = bic
        self.hqic = hqic
        self.det_sigma = det_sigma
        self.log_likelihood = log_likelihood
        self.se_df = se_df
        self._companion: Optional[np.ndarray] = None
        self._B: Optional[np.ndarray] = None
        self._k = len(var_names)
        self._lags = lags
        self._trend: Optional[str] = None
        self._XtX_inv: Optional[np.ndarray] = None
        self._coef_sigma_u: Optional[np.ndarray] = None

    def summary(self) -> str:
        k = len(self.var_names)
        lines = [
            "Vector Autoregression (VAR)",
            "=" * 60,
            f"Lags: {self.lags:<10d} Variables: {k}",
            f"N obs: {self.n_obs:<10d} Log-lik: {self.log_likelihood:.2f}",
            (f"AIC: {self.aic:.4f}   BIC: {self.bic:.4f}   " f"HQIC: {self.hqic:.4f}"),
            "=" * 60,
        ]
        for var_name in self.var_names:
            lines.append(f"\nEquation: {var_name}")
            lines.append("-" * 60)
            coef_df = self.coefs[var_name]
            lines.append(
                f"{'Variable':<20s} {'Coef':>10s} {'SE':>10s}"
                f" {'t':>8s} {'P>|t|':>8s}"
            )
            lines.append("-" * 60)
            for idx, row in coef_df.iterrows():
                t_val = row["coef"] / row["se"] if row["se"] > 0 else np.nan
                p_val = 2 * stats.t.sf(abs(t_val), self.n_obs - coef_df.shape[0])
                lines.append(
                    f"{idx:<20s} {row['coef']:>10.4f} "
                    f"{row['se']:>10.4f} {t_val:>8.3f} {p_val:>8.4f}"
                )
        return "\n".join(lines)

    def irf(
        self,
        periods: int = 20,
        impulse: Optional[str] = None,
        response: Optional[str] = None,
        orthogonal: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compute impulse response functions (see :func:`irf`)."""
        return irf(
            self,
            periods=periods,
            impulse=impulse,
            response=response,
            orthogonal=orthogonal,
            **kwargs,
        )

    def granger_test(self, caused: str, causing: str) -> Dict[str, Any]:
        """Test Granger causality."""
        return granger_causality(self, caused=caused, causing=causing)

    def plot_irf(
        self,
        periods: int = 20,
        orthogonal: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Plot impulse response functions."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        k = len(self.var_names)
        irf_result = self.irf(periods=periods, orthogonal=orthogonal)
        irfs = irf_result["irf"]

        fig, axes = plt.subplots(k, k, figsize=(4 * k, 3 * k), squeeze=False)
        for i, resp in enumerate(self.var_names):
            for j, imp in enumerate(self.var_names):
                ax = axes[i][j]
                key = f"{imp} -> {resp}"
                if key in irfs:
                    ax.plot(range(periods + 1), irfs[key], "b-", lw=1.5)
                    ax.axhline(0, color="gray", ls="--", lw=0.5)
                ax.set_title(key, fontsize=9)
                if i == k - 1:
                    ax.set_xlabel("Period")
        plt.tight_layout()
        return fig


def _lag_matrix(data: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Create lagged matrix for VAR estimation."""
    n, k = data.shape
    Y = data[lags:]  # dependent (T-p) x k
    X_parts = []
    for lag in range(1, lags + 1):
        X_parts.append(data[lags - lag : n - lag])
    X = np.hstack(X_parts)
    # Add constant
    X = np.column_stack([X, np.ones(X.shape[0])])
    return Y, X


def var(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    lags: int = 1,
    trend: str = "c",
    alpha: float = 0.05,
    se_df: str = "stata",
) -> VARResult:
    """
    Estimate a Vector Autoregression (VAR) model.

    Equivalent to Stata's ``var y1 y2, lags(1/p)`` and R's ``vars::VAR()``.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data.
    variables : list of str, optional
        Variable names. If None, uses all numeric columns.
    lags : int, default 1
        Number of lags.
    trend : str, default 'c'
        Trend: 'c' (constant), 'ct' (constant + trend), 'n' (none).
    alpha : float, default 0.05
        Significance level.
    se_df : {'stata', 'ml', 'r', 'unbiased'}, default 'stata'
        Residual-variance denominator for coefficient standard errors.
        ``'stata'``/``'ml'`` uses ``T`` and matches Stata ``var`` default
        conditional-MLE standard errors. ``'r'``/``'unbiased'`` uses
        ``T - k_params`` and matches the equation-by-equation ``lm()``
        standard errors returned inside R ``vars::VAR()``.

    Returns
    -------
    VARResult
        VAR estimation results with .irf(), .granger_test(), .plot_irf().

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(3)
    >>> T = 200
    >>> x = np.zeros(T); y = np.zeros(T)
    >>> for t in range(1, T):
    ...     x[t] = 0.5 * x[t - 1] + rng.normal()
    ...     y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 1] + rng.normal()
    >>> df = pd.DataFrame({"y": y, "x": x})
    >>> result = sp.var(df, variables=["y", "x"], lags=2)
    >>> result.var_names
    ['y', 'x']
    >>> result.lags
    2
    >>> fig = result.plot_irf()  # doctest: +SKIP
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
    se_df_key = str(se_df).lower()
    if se_df_key not in {"stata", "ml", "r", "unbiased"}:
        raise ValueError("se_df must be one of {'stata', 'ml', 'r', 'unbiased'}")

    var_data = data[variables].dropna().values.astype(float)
    var_names = list(variables)
    n, k = var_data.shape

    Y, X = _lag_matrix(var_data, lags)
    T = Y.shape[0]

    if trend == "ct":
        X = np.column_stack([X, np.arange(1, T + 1)])
    elif trend == "n":
        X = X[:, :-1]  # remove constant

    # OLS for each equation
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ X.T @ Y  # (kp+1) x k

    residuals = Y - X @ B
    rss = residuals.T @ residuals
    sigma_u = rss / T

    # Build coefficient tables
    coefs = {}
    se_dict = {}
    n_params = X.shape[1]
    if se_df_key in {"r", "unbiased"}:
        if T <= n_params:
            raise ValueError(
                "Cannot use se_df='r'/'unbiased' when T <= number of " "VAR regressors"
            )
        coef_sigma_u = rss / (T - n_params)
    else:
        coef_sigma_u = sigma_u

    for eq_idx, var_name in enumerate(var_names):
        eq_resid_var = coef_sigma_u[eq_idx, eq_idx]
        eq_se = np.sqrt(eq_resid_var * np.diag(XtX_inv))

        # Build index names
        idx_names = []
        for lag in range(1, lags + 1):
            for v in var_names:
                idx_names.append(f"L{lag}.{v}")
        if trend in ["c", "ct"]:
            idx_names.append("_cons")
        if trend == "ct":
            idx_names.append("_trend")

        coef_df = pd.DataFrame(
            {
                "coef": B[:, eq_idx],
                "se": eq_se,
            },
            index=idx_names,
        )
        coefs[var_name] = coef_df
        se_dict[var_name] = eq_se

    # Information criteria
    det_sigma = np.linalg.det(sigma_u)
    log_lik = -(T * k / 2) * (1 + np.log(2 * np.pi)) - (T / 2) * np.log(det_sigma)
    n_total_params = k * n_params
    aic = -2 * log_lik / T + 2 * n_total_params / T
    bic = -2 * log_lik / T + np.log(T) * n_total_params / T
    hqic = -2 * log_lik / T + 2 * np.log(np.log(T)) * n_total_params / T

    result = VARResult(
        coefs=coefs,
        se=se_dict,
        residuals=pd.DataFrame(residuals, columns=var_names),
        sigma_u=pd.DataFrame(sigma_u, index=var_names, columns=var_names),
        var_names=var_names,
        lags=lags,
        n_obs=T,
        aic=aic,
        bic=bic,
        hqic=hqic,
        det_sigma=det_sigma,
        log_likelihood=log_lik,
        se_df=se_df_key,
    )

    # Store for IRF computation
    result._B = B
    result._k = k
    result._lags = lags
    result._trend = trend
    # Store (X'X)^{-1} so granger_causality can form the proper coefficient
    # covariance σ²·(X'X)^{-1} for its Wald test (not just the SE diagonal).
    result._XtX_inv = XtX_inv
    result._coef_sigma_u = coef_sigma_u

    return result


def granger_causality(
    var_result: Optional[VARResult] = None,
    data: Optional[pd.DataFrame] = None,
    caused: Optional[str] = None,
    causing: Optional[str] = None,
    lags: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Granger causality test.

    Tests whether ``causing`` variable Granger-causes ``caused`` variable.

    Parameters
    ----------
    var_result : VARResult, optional
        Pre-estimated VAR model.
    data : pd.DataFrame, optional
        Data (if var_result not provided).
    caused : str
        Variable being tested for causation.
    causing : str or list of str
        Variable(s) hypothesised to cause; a list tests all their lags
        jointly (Stata ``vargranger``'s ``ALL`` row when it lists every
        other variable).
    lags : int, optional
        Number of lags (if fitting new VAR).

    Returns
    -------
    dict
        Keys: 'F_stat', 'p_value', 'df1', 'df2', 'chi2', 'chi2_p_value',
        'caused', 'causing', 'reject'. ``chi2`` is the Wald statistic ``W``
        with its ``chi2(df1)`` p-value (Stata ``vargranger`` after ``var``);
        ``F_stat = W / df1`` is referred to ``F(df1, T - m)`` (Stata
        ``vargranger`` after ``var, small``). The residual variance in ``W``
        follows the fitted model's ``se_df``: ``'stata'`` (divisor ``T``) or
        ``'r'`` (divisor ``T - m``, Stata ``var, small dfk``). ``reject``
        uses the F p-value at 5%.

    Examples
    --------
    ``x`` Granger-causes ``y`` by construction, but not vice versa:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(5)
    >>> T = 200
    >>> x = np.zeros(T)
    >>> y = np.zeros(T)
    >>> for t in range(1, T):
    ...     x[t] = 0.5 * x[t - 1] + rng.normal()
    ...     y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 1] + rng.normal()
    >>> df = pd.DataFrame({"y": y, "x": x})
    >>> vr = sp.var(df, variables=["y", "x"], lags=2)
    >>> gc = sp.granger_causality(vr, caused="y", causing="x")
    >>> print(gc["p_value"] < 0.05)  # x helps predict y
    True

    Or fit the VAR implicitly from data:

    >>> gc2 = sp.granger_causality(data=df, caused="x", causing="y", lags=2)
    >>> print(gc2["reject"])  # y does not Granger-cause x
    False
    """
    if caused is None or causing is None:
        raise MethodIncompatibility(
            "granger_causality requires both caused=... and causing=....",
            diagnostics={"caused": caused, "causing": causing},
        )

    if var_result is None:
        if data is None:
            raise MethodIncompatibility("Provide var_result or data")
        if lags is None:
            lags = 1
        causing_vars = [causing] if isinstance(causing, str) else list(causing)
        var_result = var(data, variables=[caused] + causing_vars, lags=lags)

    k = var_result._k
    p = var_result._lags
    T = var_result.n_obs
    var_names = var_result.var_names

    causing_list = [causing] if isinstance(causing, str) else list(causing)
    if caused in causing_list:
        raise MethodIncompatibility(
            "granger_causality: 'causing' must not contain the caused variable."
        )
    causing_idx = [var_names.index(c) for c in causing_list]

    # Lags of every causing variable in the caused equation
    restrict_indices = []
    for lag in range(p):
        for ci in causing_idx:
            restrict_indices.append(lag * k + ci)

    coef_df = var_result.coefs[caused]
    coefs_all = coef_df["coef"].values
    R = np.zeros((len(restrict_indices), len(coefs_all)))
    for i, ri in enumerate(restrict_indices):
        R[i, ri] = 1

    r = R @ coefs_all
    coef_sigma_u = getattr(var_result, "_coef_sigma_u", var_result.sigma_u)
    if isinstance(coef_sigma_u, pd.DataFrame):
        sigma2 = coef_sigma_u.loc[caused, caused]
    else:
        sigma2 = coef_sigma_u[var_names.index(caused), var_names.index(caused)]

    # Coefficient covariance for the `caused` equation is σ²_caused·(X'X)^{-1},
    # where X is the stacked lagged-regressor design matrix. The full matrix
    # (not just its diagonal) is required because the restricted coefficients
    # — the p lags of `causing` — are mutually correlated.
    XtX_inv = getattr(var_result, "_XtX_inv", None)
    if XtX_inv is None:
        raise MethodIncompatibility(
            "This VARResult predates the Granger covariance fix and does not "
            "carry (X'X)^{-1}. Re-fit the model with sp.var(...) and retry."
        )
    V = sigma2 * np.asarray(XtX_inv)

    # Wald statistic W = (Rβ)'(R V R')^{-1}(Rβ).
    #   chi2 = W ~ chi2(q)            (Stata ``vargranger`` after ``var``)
    #   F    = W / q ~ F(q, T - m)    (Stata ``vargranger`` after ``var, small``)
    mid = R @ V @ R.T
    try:
        W = float(r @ np.linalg.solve(mid, r))
    except np.linalg.LinAlgError:
        W = np.nan
    df1 = len(restrict_indices)
    F_stat = W / df1
    df2 = T - len(coefs_all)
    p_value = stats.f.sf(F_stat, df1, df2) if np.isfinite(F_stat) else np.nan
    chi2_p = stats.chi2.sf(W, df1) if np.isfinite(W) else np.nan

    return {
        "F_stat": F_stat,
        "p_value": p_value,
        "df1": df1,
        "df2": df2,
        "chi2": W,
        "chi2_p_value": chi2_p,
        "caused": caused,
        "causing": causing,
        "reject": p_value < 0.05 if np.isfinite(p_value) else None,
    }


def irf(
    var_result: VARResult,
    periods: int = 20,
    impulse: Optional[str] = None,
    response: Optional[str] = None,
    orthogonal: bool = True,
    cumulative: bool = False,
    sigma_df: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute impulse response functions from VAR.

    Simple responses are the MA coefficients ``Phi_s`` of the VAR;
    orthogonalised responses are ``Phi_s P`` with ``P`` the lower Cholesky
    factor of the residual covariance (variable order = Cholesky order).

    Parameters
    ----------
    var_result : VARResult
        Estimated VAR model.
    periods : int, default 20
        Number of periods for IRF.
    impulse : str, optional
        Impulse variable (if None, all).
    response : str, optional
        Response variable (if None, all).
    orthogonal : bool, default True
        Orthogonalized (Cholesky) IRF.
    cumulative : bool, default False
        Return cumulative responses ``sum_{j<=s}`` (Stata ``cirf`` /
        ``coirf``; ``vars::irf(cumulative = TRUE)``).
    sigma_df : {'ml', 'unbiased'}, optional
        Divisor of the residual covariance that the orthogonalisation uses:
        ``'ml'`` = ``T`` (Stata ``irf create`` after ``var``), ``'unbiased'``
        = ``T - m`` with ``m`` regressors per equation (``vars::Psi``, and
        Stata after ``var, dfk``). Default follows the fitted model's
        ``se_df`` (``'stata'``/``'ml'`` -> ``'ml'``, ``'r'``/``'unbiased'`` ->
        ``'unbiased'``). Irrelevant when ``orthogonal=False``.

    Returns
    -------
    dict
        Keys: 'irf' (dict of arrays), 'periods'.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(3)
    >>> T = 200
    >>> x = np.zeros(T); y = np.zeros(T)
    >>> for t in range(1, T):
    ...     x[t] = 0.5 * x[t - 1] + rng.normal()
    ...     y[t] = 0.3 * y[t - 1] + 0.4 * x[t - 1] + rng.normal()
    >>> df = pd.DataFrame({"y": y, "x": x})
    >>> vr = sp.var(df, variables=["y", "x"], lags=2)
    >>> out = sp.irf(vr, periods=10, impulse="x", response="y")
    >>> sorted(out.keys())
    ['irf', 'periods']
    >>> len(out["irf"]["x -> y"])  # s = 0, 1, ..., 10
    11
    >>> out["periods"][0]
    0
    >>> bool(np.isfinite(out["irf"]["x -> y"]).all())
    True

    References
    ----------
    lutkepohl2005new
    """
    k = var_result._k
    p = var_result._lags
    B = var_result._B
    if B is None:
        raise MethodIncompatibility(
            "VARResult does not contain coefficient matrices for IRF. "
            "Re-fit the model with sp.var(...) and retry."
        )
    var_names = var_result.var_names

    # Build companion matrix
    # Extract autoregressive coefficients (exclude constant/trend)
    A_list = []
    for lag in range(p):
        A_lag = B[lag * k : (lag + 1) * k, :].T  # k x k
        A_list.append(A_lag)

    # Compute MA representation: Φ_s = Σ Φ_{s-j} A_j
    Phi = [np.eye(k)]  # Φ_0 = I
    for s in range(1, periods + 1):
        Phi_s = np.zeros((k, k))
        for j in range(min(s, p)):
            Phi_s += Phi[s - j - 1] @ A_list[j]
        Phi.append(Phi_s)

    # Orthogonalize via Cholesky
    sigma_u = np.asarray(
        (
            var_result.sigma_u.values
            if isinstance(var_result.sigma_u, pd.DataFrame)
            else var_result.sigma_u
        ),
        dtype=float,
    )
    if sigma_df is None:
        sigma_df = (
            "unbiased"
            if str(getattr(var_result, "se_df", "stata")).lower() in {"r", "unbiased"}
            else "ml"
        )
    if sigma_df not in ("ml", "unbiased"):
        raise ValueError("sigma_df must be 'ml' or 'unbiased'")
    if sigma_df == "unbiased":
        m = B.shape[0]  # regressors per equation
        sigma_u = sigma_u * var_result.n_obs / (var_result.n_obs - m)
    if orthogonal:
        P = np.linalg.cholesky(sigma_u)
    else:
        P = np.eye(k)

    # Build IRF dict
    irfs = {}
    imp_vars = [impulse] if impulse else var_names
    resp_vars = [response] if response else var_names

    for imp in imp_vars:
        imp_idx = var_names.index(imp)
        for resp in resp_vars:
            resp_idx = var_names.index(resp)
            key = f"{imp} -> {resp}"
            irf_values = np.array([Phi[s] @ P[:, imp_idx] for s in range(periods + 1)])
            path = irf_values[:, resp_idx]
            irfs[key] = np.cumsum(path) if cumulative else path

    return {"irf": irfs, "periods": list(range(periods + 1))}
