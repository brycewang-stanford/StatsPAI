"""
Advanced instrumental variables estimators.

Implements LIML, Fuller's k-class, JIVE (Jackknife IV), and LASSO-IV
for settings with many/weak instruments.

Equivalent to Stata's ``ivregress liml`` and R's ``AER::ivreg()``.

References
----------
Anderson, T.W. & Rubin, H. (1949).
"Estimation of the Parameters of a Single Equation in a Complete System
of Stochastic Equations." *Annals of Math. Stats.*, 20(1), 46-63. [@anderson1949estimation]

Fuller, W.A. (1977).
"Some Properties of a Modification of the Limited Information Estimator."
*Econometrica*, 45(4), 939-953. [@fuller1977some]

Angrist, J.D., Imbens, G.W. & Krueger, A.B. (1999).
"Jackknife Instrumental Variables Estimation."
*Journal of Applied Econometrics*, 14(1), 57-67. [@angrist1999jackknife]

Belloni, A., Chen, D., Chernozhukov, V. & Hansen, C. (2012).
"Sparse Models and Methods for Optimal Instruments with an Application
to Eminent Domain." *Econometrica*, 80(6), 2369-2429. [@belloni2012sparse]

Chernozhukov, V., Hansen, C. & Spindler, M. (2016). "hdm:
High-Dimensional Metrics." *The R Journal*, 8(2), 185-199.
[@chernozhukov2016hdm] — reference R implementation (``rlassoIV``).
"""

import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..core._vcov_spec import markout_clusters
from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility


@accepts_aliases(vce="robust")
@markout_clusters
def liml(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x_endog: Optional[List[str]] = None,
    x_exog: Optional[List[str]] = None,
    z: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    fuller: Optional[float] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Limited Information Maximum Likelihood (LIML) estimator.

    More robust to weak instruments than 2SLS. Fuller's modification
    provides improved finite-sample properties.

    Equivalent to Stata's ``ivregress liml y (x_endog = z) x_exog``.

    Parameters
    ----------
    formula : str, optional
        Formula: "y ~ x_exog | x_endog | z" or "y ~ x_exog + (x_endog ~ z)".
    data : pd.DataFrame
    y : str
        Outcome variable.
    x_endog : list of str
        Endogenous regressors.
    x_exog : list of str
        Exogenous regressors (included instruments).
    z : list of str
        Excluded instruments.
    robust : str, default 'nonrobust'
    cluster : str, optional
    fuller : float, optional
        Fuller's constant (typically 1 or 4). If None, pure LIML.
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 300
    >>> z1, z2 = rng.normal(size=n), rng.normal(size=n)   # excluded instruments
    >>> exper = rng.normal(size=n)                        # exogenous control
    >>> u = rng.normal(size=n)                            # endogeneity
    >>> educ = 0.6 * z1 + 0.5 * z2 + 0.3 * exper + u + rng.normal(size=n)
    >>> lwage = 1.0 + 0.8 * educ + 0.2 * exper + 1.5 * u + rng.normal(size=n)
    >>> df = pd.DataFrame({'lwage': lwage, 'educ': educ, 'exper': exper,
    ...                    'z1': z1, 'z2': z2})
    >>> result = sp.liml(data=df, y='lwage', x_endog=['educ'],
    ...                  x_exog=['exper'], z=['z1', 'z2'])
    >>> bool(abs(result.params['educ'] - 0.8) < 0.2)
    True
    """
    if formula is not None:
        # Parse IV formula
        parts = formula.replace("(", "|").replace(")", "").replace("~", "|").split("|")
        if len(parts) >= 3:
            y = parts[0].strip()
            x_exog = [v.strip() for v in parts[1].split("+") if v.strip()]
            x_endog = [v.strip() for v in parts[2].split("+") if v.strip()]
            if len(parts) >= 4:
                z = [v.strip() for v in parts[3].split("+") if v.strip()]

    if x_exog is None:
        x_exog = []
    if data is None or y is None or x_endog is None or z is None:
        raise MethodIncompatibility(
            "liml requires data, y, x_endog, and z unless all are supplied "
            "by formula",
            recovery_hint=(
                "Pass data= plus y=, x_endog=, z=, or provide a complete " "IV formula."
            ),
        )

    df = data.dropna(subset=[y] + x_endog + x_exog + z)
    n = len(df)

    Y = df[y].values.astype(float)
    X_endog = df[x_endog].values.astype(float).reshape(n, -1)
    X_exog = (
        np.column_stack([np.ones(n)] + [df[v].values for v in x_exog])
        if x_exog
        else np.ones((n, 1))
    )
    Z_excl = df[z].values.astype(float).reshape(n, -1)

    # All instruments: exogenous regressors + excluded instruments
    Z_all = np.column_stack([X_exog, Z_excl])

    # All regressors: exogenous + endogenous
    X_all = np.column_stack([X_exog, X_endog])
    k = X_all.shape[1]

    # Projections are applied, never formed: the n x n matrices used here
    # before made sp.liml O(n^2) in memory (10 GB at n = 20,000).
    from .iv import _projector

    proj_x = _projector(X_exog, np.linalg.inv(X_exog.T @ X_exog))
    proj_z = _projector(Z_all, np.linalg.inv(Z_all.T @ Z_all))

    # Compute LIML κ via the Anderson (1951) generalized symmetric
    # eigenvalue problem:  S_exog v = κ S_full v , with
    #   S_full = W0' M_full W0   (residuals from full model)
    #   S_exog = W0' M_exog W0   (residuals from exog-only model)
    # and W0 = [Y, X_endog]. Both are symmetric PSD and
    # S_exog ≽ S_full in the Loewner order (extra residualisation
    # shrinks SSR), so all eigenvalues κ ≥ 1 and κ_LIML is the SMALLEST.
    #
    # The previous implementation used ``np.linalg.eigvals(inv(A) @ B)``
    # on the non-symmetric product, which silently returned complex
    # eigenvalues and produced a biased κ — the same bug already fixed
    # in ``iv.py::_liml_kappa``. Fixed here by aligning to that
    # canonical implementation: ``scipy.linalg.eigh(S_exog, S_full)``.
    W = np.column_stack([Y.reshape(-1, 1), X_endog])
    S_full = W.T @ (W - proj_z(W))  # W0' M_full W0
    S_exog = W.T @ (W - proj_x(W))  # W0' M_exog W0

    try:
        from scipy.linalg import eigh as _sp_eigh

        eigvals = _sp_eigh(S_exog, S_full, eigvals_only=True)
        kappa = float(np.min(eigvals))
        if not np.isfinite(kappa) or kappa < 1 - 1e-8:
            warnings.warn(
                f"LIML κ = {kappa} outside expected [1, ∞); falling back "
                "to 2SLS (κ = 1).",
                RuntimeWarning,
                stacklevel=2,
            )
            kappa = 1.0
    except Exception:
        warnings.warn(
            "LIML generalized eigenvalue solve failed; falling back to 2SLS.",
            RuntimeWarning,
            stacklevel=2,
        )
        kappa = 1.0

    if fuller is not None:
        kappa = kappa - fuller / (n - Z_all.shape[1])

    # k-class estimator: β = (X'(I - κMz)X)^{-1} X'(I - κMz)Y, with
    # (I - κMz) X = (1 - κ) X + κ Pz X.
    PzX = proj_z(X_all)
    AX = (1.0 - kappa) * X_all + kappa * PzX
    XtWX = X_all.T @ AX
    try:
        XtWY = AX.T @ Y
        beta = np.linalg.solve(XtWX, XtWY)
    except np.linalg.LinAlgError:
        beta = np.full(k, np.nan)

    resid = Y - X_all @ beta

    # Standard errors
    try:
        XtX_inv = np.linalg.inv(XtWX) if not np.any(np.isnan(beta)) else np.eye(k)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtWX)

    # Stata grammar: vce='robust' / True / 'cluster firm'.
    from ..core._vcov_spec import parse_se_request

    se_req = parse_se_request(
        robust,
        cluster,
        function="liml",
        supported=("nonrobust", "robust", "hc0", "hc1", "cluster"),
    )
    robust, cluster = se_req.kind, se_req.cluster

    # Sandwich meat: the instrument-projected regressors P_Z X, for every
    # k-class estimator. This is what Stata's ``ivregress liml`` and
    # ``linearmodels.IVLIML`` compute (bit-for-bit against Stata 18 in
    # tests/reference_parity/test_vce_grammar_stata_parity.py). The k-class
    # first-order-condition form (I - kappa M_Z) X used here before differs
    # by (kappa - 1) M_Z X, which is asymptotically negligible and zero at
    # kappa = 1 (2SLS), but moved robust / cluster SEs ~0.05% away from both
    # references. Bread and meat use the small-sample convention of
    # ``ivregress ..., small``, like the classical SE below.
    scores = PzX * resid[:, None]
    if robust == "cluster":
        clusters = df[cluster].values
        unique_cl, codes = np.unique(clusters, return_inverse=True)
        n_cl = len(unique_cl)
        summed = np.zeros((n_cl, k))
        np.add.at(summed, codes, scores)
        correction = n_cl / (n_cl - 1) * (n - 1) / (n - k)
        var_cov = correction * XtX_inv @ (summed.T @ summed) @ XtX_inv
    elif robust in ("robust", "hc0", "hc1"):
        var_cov = XtX_inv @ (scores.T @ scores) @ XtX_inv
        if robust != "hc0":
            # ``vce(robust) small`` = N/(N-K); sp.liml(robust=...) used to
            # return the unscaled HC0 while sp.iv(method='liml') returned HC1.
            var_cov = var_cov * (n / (n - k))
    else:
        sigma2 = np.sum(resid**2) / (n - k)
        var_cov = sigma2 * XtX_inv

    se = np.sqrt(np.diag(var_cov))

    # Variable names
    var_names = ["_cons"] + x_exog + x_endog if x_exog else ["_cons"] + x_endog

    params = pd.Series(beta, index=var_names)
    std_errors = pd.Series(se, index=var_names)

    # Diagnostics
    tss = np.sum((Y - Y.mean()) ** 2)
    rss = np.sum(resid**2)
    r2 = 1 - rss / tss

    model_name = "LIML" if fuller is None else f"Fuller (a={fuller})"

    # First-stage strength: the binding (weakest) excluded-instrument F across
    # endogenous regressors, so result.violations() flags weak instruments even
    # when the workflow picks LIML (the weak-IV-robust estimator). Additive —
    # does not touch the LIML point estimate or SE.
    try:
        from .iv import _first_stage_diagnostics

        _fs = _first_stage_diagnostics(X_exog, X_endog, Z_all, n, Z_excl.shape[1])
        _fvals = [
            fs["f_statistic"]
            for fs in _fs
            if fs.get("f_statistic") is not None and np.isfinite(fs["f_statistic"])
        ]
    except (np.linalg.LinAlgError, ValueError, KeyError):
        _fvals = []
    first_stage_f = float(min(_fvals)) if _fvals else None

    _result = EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "alpha": alpha,
            "model_type": model_name,
            "method": "Limited Information Maximum Likelihood",
            "kappa": kappa,
            "fuller_constant": fuller,
            "endog_vars": x_endog,
            "instruments": z,
            "first_stage_f": first_stage_f,
        },
        data_info={
            "n_obs": n,
            "df_resid": n - k,
            "dep_var": y,
        },
        diagnostics={
            "r_squared": r2,
            "kappa": kappa,
            "n_instruments": len(z),
            "n_endogenous": len(x_endog),
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.iv.liml",
            params={
                "formula": formula,
                "y": y,
                "x_endog": list(x_endog) if x_endog else None,
                "x_exog": list(x_exog) if x_exog else None,
                "z": list(z) if z else None,
                "robust": robust,
                "cluster": cluster,
                "fuller": fuller,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


@accepts_aliases(vce="robust")
def jive(
    data: pd.DataFrame,
    y: str,
    x_endog: List[str],
    x_exog: Optional[List[str]] = None,
    z: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    variant: str = "jive1",
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Jackknife Instrumental Variables Estimation (JIVE).

    Reduces finite-sample bias from many instruments by using
    leave-one-out fitted values as instruments.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    x_endog : list of str
    x_exog : list of str, optional
    z : list of str
    robust : str, default 'nonrobust'
    cluster : str, optional
    variant : str, default 'jive1'
        Angrist, Imbens & Krueger (1999) jackknife instrument, used as the
        instrument in an IV second stage (Stata ``jive``'s ``ujive1`` /
        ``ujive2``):

        - ``'jive1'``: ``(Z_i pi - h_i x_i) / (1 - h_i)`` -- the
          leave-one-out first-stage fitted value.
        - ``'jive2'``: ``(Z_i pi - h_i x_i) / (1 - 1/n)``.

        ``h_i`` is the leverage of the first-stage regression on the
        included exogenous regressors and the excluded instruments.
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Notes
    -----
    Point estimates and the ``robust`` (HC0) standard errors reproduce
    Stata ``jive`` (Stata Journal package st0108, SJ 6-3) ``ujive1`` /
    ``ujive2``, and so does the default standard error, the homoskedastic
    IV sandwich ``s^2 (X_J'X)^-1 X_J'X_J (X_J'X)^-1`` with
    ``s^2 = e'e / (n - k)``. ``cluster`` uses the ``G/(G-1)`` factor only;
    no reference implementation offers clustered JIVE.

    Through 1.28.0 ``variant='jive2'`` computed ``fitted / (1 - h_i)``,
    which is not a jackknife instrument (it keeps observation ``i``'s own
    contribution), and the default standard error was
    ``s^2 (X_J'X)^-1``, which omits the sandwich and understated the SE
    by half on the reference fixture.

    References
    ----------
    angrist1999jackknife

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 300
    >>> Z = rng.normal(size=(n, 5))                  # many instruments
    >>> u = rng.normal(size=n)
    >>> x = Z @ np.full(5, 0.4) + u + rng.normal(size=n)
    >>> y = 1.0 * x + 1.5 * u + rng.normal(size=n)
    >>> df = pd.DataFrame(Z, columns=[f'z{i}' for i in range(5)])
    >>> df['y'], df['x'] = y, x
    >>> result = sp.jive(df, y='y', x_endog=['x'],
    ...                  z=[f'z{i}' for i in range(5)])
    >>> bool(abs(result.params['x'] - 1.0) < 0.2)  # bias-reduced IV estimate
    True
    """
    if x_exog is None:
        x_exog = []
    if z is None:
        raise MethodIncompatibility(
            "jive requires z excluded instruments",
            recovery_hint="Pass z=[...] with at least one excluded instrument.",
        )

    df = data.dropna(subset=[y] + x_endog + x_exog + z)
    n = len(df)

    Y = df[y].values.astype(float)
    X_endog = df[x_endog].values.astype(float).reshape(n, -1)
    X_exog = (
        np.column_stack([np.ones(n)] + [df[v].values for v in x_exog])
        if x_exog
        else np.ones((n, 1))
    )
    Z_excl = df[z].values.astype(float).reshape(n, -1)
    Z_all = np.column_stack([X_exog, Z_excl])

    if variant not in ("jive1", "jive2"):
        raise MethodIncompatibility(
            f"variant must be 'jive1' or 'jive2'; got {variant!r}"
        )

    # First-stage leverages and fitted values through a thin QR, so the
    # n x n projection matrix is never formed.
    Q, _ = np.linalg.qr(Z_all)
    h_ii = np.einsum("ij,ij->i", Q, Q)

    X_endog_hat = np.zeros_like(X_endog, dtype=np.float64)
    for j in range(X_endog.shape[1]):
        x_j = X_endog[:, j]
        loo_num = Q @ (Q.T @ x_j) - h_ii * x_j
        if variant == "jive1":
            X_endog_hat[:, j] = loo_num / np.maximum(1 - h_ii, 1e-10)
        else:
            X_endog_hat[:, j] = loo_num / (1.0 - 1.0 / n)

    # IV second stage with the jackknife instruments
    X_all = np.column_stack([X_exog, X_endog])
    X_hat = np.column_stack([X_exog, X_endog_hat])

    k = X_all.shape[1]

    try:
        XhX_inv = np.linalg.inv(X_hat.T @ X_all)
        beta = XhX_inv @ (X_hat.T @ Y)
    except np.linalg.LinAlgError:
        XhX_inv = np.full((k, k), np.nan)
        beta = np.full(k, np.nan)

    resid = Y - X_all @ beta

    if cluster is not None:
        clusters = df[cluster].values
        _, codes = np.unique(clusters, return_inverse=True)
        n_cl = int(codes.max()) + 1
        sums = np.zeros((n_cl, k))
        np.add.at(sums, codes, X_hat * resid[:, None])
        meat = sums.T @ sums * (n_cl / (n_cl - 1))
    elif robust != "nonrobust":
        meat = (X_hat * (resid**2)[:, None]).T @ X_hat
    else:
        sigma2 = float(resid @ resid) / (n - k)
        meat = sigma2 * (X_hat.T @ X_hat)
    var_cov = XhX_inv @ meat @ XhX_inv.T

    se = np.sqrt(np.abs(np.diag(var_cov)))

    var_names = ["_cons"] + x_exog + x_endog if x_exog else ["_cons"] + x_endog
    params = pd.Series(beta, index=var_names)
    std_errors = pd.Series(se, index=var_names)

    # First-stage strength so result.violations() flags weak instruments — JIVE
    # is chosen precisely under weak/many instruments, so the diagnostic belongs
    # on the result. Additive: does not touch the JIVE point estimate or SE.
    try:
        from .iv import _first_stage_diagnostics

        _fs = _first_stage_diagnostics(X_exog, X_endog, Z_all, n, Z_excl.shape[1])
        _fvals = [
            fs["f_statistic"]
            for fs in _fs
            if fs.get("f_statistic") is not None and np.isfinite(fs["f_statistic"])
        ]
    except (np.linalg.LinAlgError, ValueError, KeyError):
        _fvals = []
    first_stage_f = float(min(_fvals)) if _fvals else None

    _result = EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "alpha": alpha,
            "model_type": f"JIVE ({variant.upper()})",
            "method": "Jackknife Instrumental Variables",
            "endog_vars": x_endog,
            "instruments": z,
            "variant": variant,
            "first_stage_f": first_stage_f,
        },
        data_info={"n_obs": n, "df_resid": n - k, "dep_var": y},
        diagnostics={"n_instruments": len(z), "n_endogenous": len(x_endog)},
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.iv.jive",
            params={
                "y": y,
                "x_endog": list(x_endog),
                "x_exog": list(x_exog) if x_exog else None,
                "z": list(z) if z else None,
                "robust": robust,
                "cluster": cluster,
                "variant": variant,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


@accepts_aliases(vce="robust")
def lasso_iv(
    data: pd.DataFrame,
    y: str,
    x_endog: Optional[List[str]] = None,
    x_exog: Optional[List[str]] = None,
    z: Optional[List[str]] = None,
    robust: str = "robust",
    cluster: Optional[str] = None,
    penalty: str = "bic",
    alpha: float = 0.05,
    d: Optional[Any] = None,
) -> EconometricResults:
    """
    LASSO-selected instrumental variables (information-criterion penalty).

    Uses LASSO to select relevant instruments from a large set, then
    estimates IV/2SLS with the selected instruments. The Lasso penalty is
    chosen by BIC / AIC over a fixed grid or by cross-validation, on
    instruments partialled of the exogenous regressors.

    This is *not* the estimator of Belloni, Chen, Chernozhukov & Hansen
    (2012), which selects instruments with the rigorous plug-in penalty and
    heteroskedasticity-adapted loadings: that estimator is
    :func:`sp.rlasso_iv` (``select_X=False``), which reproduces R
    ``hdm::rlassoIV``. Through 1.28.0 this function's docstring and
    ``model_info['method']`` attributed it to BCCH (2012).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    x_endog : list of str
    x_exog : list of str, optional
    z : list of str
        Full set of candidate instruments.
    robust : str, default 'robust'
    cluster : str, optional
    penalty : str, default 'bic'
        Instrument selection criterion: 'bic', 'aic', 'cv'.
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(7)
    >>> n = 400
    >>> # 20 candidate instruments, only the first 5 are relevant
    >>> Z = rng.normal(size=(n, 20))
    >>> u = rng.normal(size=n)                            # endogeneity
    >>> educ = Z[:, :5] @ np.full(5, 0.5) + u + rng.normal(size=n)
    >>> lwage = 1.0 + 0.7 * educ + 1.2 * u + rng.normal(size=n)
    >>> df = pd.DataFrame(Z, columns=[f'z{i}' for i in range(20)])
    >>> df['lwage'], df['educ'] = lwage, educ
    >>> result = sp.lasso_iv(df, y='lwage', x_endog=['educ'],
    ...                      z=[f'z{i}' for i in range(20)])
    >>> bool(abs(result.params['educ'] - 0.7) < 0.2)
    True
    """
    from ..core._param_aliases import resolve_alias

    if isinstance(d, str):
        d = [d]
    x_endog = resolve_alias("x_endog", x_endog, "d", d)
    if x_endog is None:
        raise TypeError("lasso_iv() missing required argument: 'x_endog' (alias 'd')")
    if isinstance(x_endog, str):
        x_endog = [x_endog]
    if x_exog is None:
        x_exog = []
    if z is None:
        raise MethodIncompatibility(
            "lasso_iv requires z candidate instruments",
            recovery_hint="Pass z=[...] with the candidate instrument set.",
        )

    df = data.dropna(subset=[y] + x_endog + x_exog + z)
    n = len(df)

    Z_candidates = df[z].values.astype(float)
    X_exog_mat = (
        np.column_stack([np.ones(n)] + [df[v].values for v in x_exog])
        if x_exog
        else np.ones((n, 1))
    )

    # Partial out exogenous regressors from instruments and endogenous vars
    # (least-squares residuals; the n x n annihilator is never formed).
    def _resid(M: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(X_exog_mat, M, rcond=None)
        return M - X_exog_mat @ coef

    Z_tilde = _resid(Z_candidates)  # residualized instruments
    X_endog_tilde = _resid(df[x_endog].values.astype(float).reshape(n, -1))

    # LASSO selection for each endogenous variable
    selected_z_indices = set()

    for j in range(X_endog_tilde.shape[1]):
        x_j = X_endog_tilde[:, j]

        # Cross-validated LASSO
        from sklearn.linear_model import Lasso, LassoCV

        if penalty == "cv":
            lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
            lasso.fit(Z_tilde, x_j)
        else:
            # Use BIC to select lambda
            alphas = np.logspace(-4, 1, 50)
            best_bic = np.inf
            best_alpha = alphas[0]

            for a in alphas:
                lasso_temp = Lasso(alpha=a, max_iter=10000)
                lasso_temp.fit(Z_tilde, x_j)
                pred = lasso_temp.predict(Z_tilde)
                rss = np.sum((x_j - pred) ** 2)
                k_sel = np.sum(np.abs(lasso_temp.coef_) > 1e-10)
                if penalty == "bic":
                    criterion = n * np.log(rss / n) + k_sel * np.log(n)
                else:
                    criterion = n * np.log(rss / n) + 2 * k_sel

                if criterion < best_bic:
                    best_bic = criterion
                    best_alpha = a

            lasso = Lasso(alpha=best_alpha, max_iter=10000)
            lasso.fit(Z_tilde, x_j)

        # Selected instruments
        sel_idx = np.where(np.abs(lasso.coef_) > 1e-10)[0]
        selected_z_indices.update(sel_idx.tolist())

    selected_z = [z[i] for i in sorted(selected_z_indices)]

    if len(selected_z) == 0:
        warnings.warn("LASSO selected no instruments. Using all instruments.")
        selected_z = z

    # 2SLS with selected instruments. Build a formula string to drive the
    # current ``sp.iv`` formula-only API (``y ~ (endog ~ z) + exog``).
    # Map legacy ``robust='robust'`` to the modern HC1 enum.
    from ..regression.iv import iv

    endog_str = " + ".join(x_endog)
    z_str = " + ".join(selected_z)
    exog_str = (" + " + " + ".join(x_exog)) if x_exog else ""
    formula = f"{y} ~ ({endog_str} ~ {z_str}){exog_str}"
    iv_robust = "hc1" if robust == "robust" else robust
    result = iv(formula=formula, data=df, robust=iv_robust, cluster=cluster)

    # Add LASSO-specific info
    result.model_info["model_type"] = "LASSO-IV (2SLS with selected instruments)"
    result.model_info["method"] = (
        f"Lasso instrument selection ({penalty}-chosen penalty) + 2SLS; for the "
        "BCCH (2012) plug-in-penalty estimator use sp.rlasso_iv"
    )
    result.model_info["n_candidate_instruments"] = len(z)
    result.model_info["n_selected_instruments"] = len(selected_z)
    result.model_info["selected_instruments"] = selected_z
    result.model_info["selection_criterion"] = penalty

    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.iv.lasso_iv",
            params={
                "y": y,
                "x_endog": list(x_endog),
                "x_exog": list(x_exog) if x_exog else None,
                "z_candidates": list(z),
                "selected_instruments": list(selected_z),
                "penalty": penalty,
                "robust": robust,
                "cluster": cluster,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return result
