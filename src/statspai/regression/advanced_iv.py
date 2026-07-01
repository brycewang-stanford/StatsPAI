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
to Eminent Domain." *Econometrica*, 80(6), 2369-2429. [@belloni2011sparse]

Chernozhukov, V., Hansen, C. & Spindler, M. (2016). "hdm:
High-Dimensional Metrics." *The R Journal*, 8(2), 185-199.
[@chernozhukov2016hdm] — reference R implementation (``rlassoIV``).
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility


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

    # Residual maker for X_exog
    Px_exog = X_exog @ np.linalg.solve(X_exog.T @ X_exog, X_exog.T)
    Mx = np.eye(n) - Px_exog

    # Projection onto all instruments
    Pz = Z_all @ np.linalg.solve(Z_all.T @ Z_all, Z_all.T)
    Mz = np.eye(n) - Pz

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
    S_full = W.T @ Mz @ W  # W0' M_full W0
    S_exog = W.T @ Mx @ W  # W0' M_exog W0

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

    # k-class estimator: β = (X'(I - κMz)X)^{-1} X'(I - κMz)Y
    I_kMz = np.eye(n) - kappa * Mz
    try:
        XtWX = X_all.T @ I_kMz @ X_all
        XtWY = X_all.T @ I_kMz @ Y
        beta = np.linalg.solve(XtWX, XtWY)
    except np.linalg.LinAlgError:
        beta = np.full(k, np.nan)

    resid = Y - X_all @ beta

    # Standard errors
    try:
        XtX_inv = (
            np.linalg.inv(X_all.T @ I_kMz @ X_all)
            if not np.any(np.isnan(beta))
            else np.eye(k)
        )
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_all.T @ I_kMz @ X_all)

    # k-class FOC X' (I − κ M_Z) (y − X β) = 0 implies the score per
    # observation is (AX)_i u_i with AX = (I − κ M_Z) X. Meat must use
    # AX, not raw X — the same projection-in-meat invariant as the 2SLS
    # path in regression/iv.py::_k_class_fit (fixed in v1.6.4). Matches
    # Cameron–Miller (2015), Stata ivregress, and linearmodels.IVLIML.
    AX = I_kMz @ X_all
    if cluster is not None:
        clusters = df[cluster].values
        unique_cl = np.unique(clusters)
        n_cl = len(unique_cl)
        meat = np.zeros((k, k))
        for cl in unique_cl:
            cl_mask = clusters == cl
            score = AX[cl_mask].T @ resid[cl_mask]
            meat += np.outer(score, score)
        correction = n_cl / (n_cl - 1) * (n - 1) / (n - k)
        var_cov = correction * XtX_inv @ meat @ XtX_inv
    elif robust != "nonrobust":
        Omega = np.diag(resid**2)
        var_cov = XtX_inv @ (AX.T @ Omega @ AX) @ XtX_inv
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
        'jive1' (Angrist et al. 1999) or 'jive2' (alternative).
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

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

    # Leave-one-out projection for each endogenous variable
    Pz = Z_all @ np.linalg.solve(Z_all.T @ Z_all, Z_all.T)
    h_ii = np.diag(Pz)

    X_endog_hat = np.zeros_like(X_endog)

    for j in range(X_endog.shape[1]):
        x_j = X_endog[:, j]
        fitted = Pz @ x_j  # regular fitted values

        if variant == "jive1":
            # JIVE1: x̂_i = (fitted_i - h_ii * x_i) / (1 - h_ii)
            X_endog_hat[:, j] = (fitted - h_ii * x_j) / np.maximum(1 - h_ii, 1e-10)
        else:
            # JIVE2: x̂_i = fitted_i / (1 - h_ii)
            X_endog_hat[:, j] = fitted / np.maximum(1 - h_ii, 1e-10)

    # 2SLS with jackknife instruments
    X_all = np.column_stack([X_exog, X_endog])
    X_hat = np.column_stack([X_exog, X_endog_hat])

    k = X_all.shape[1]

    try:
        XhX = X_hat.T @ X_all
        XhY = X_hat.T @ Y
        beta = np.linalg.solve(XhX, XhY)
    except np.linalg.LinAlgError:
        beta = np.full(k, np.nan)

    resid = Y - X_all @ beta

    # SE
    try:
        XhX_inv = np.linalg.inv(X_hat.T @ X_all)
    except np.linalg.LinAlgError:
        XhX_inv = np.eye(k)

    if cluster is not None:
        clusters = df[cluster].values
        unique_cl = np.unique(clusters)
        n_cl = len(unique_cl)
        meat = np.zeros((k, k))
        for cl in unique_cl:
            cl_mask = clusters == cl
            score = X_hat[cl_mask].T @ resid[cl_mask]
            meat += np.outer(score, score)
        correction = n_cl / (n_cl - 1)
        var_cov = correction * XhX_inv @ meat @ XhX_inv.T
    else:
        sigma2 = np.sum(resid**2) / (n - k)
        mid = (
            X_hat.T
            @ (np.diag(resid**2) if robust != "nonrobust" else sigma2 * np.eye(n))
            @ X_hat
        )
        if robust != "nonrobust":
            var_cov = XhX_inv @ mid @ XhX_inv.T
        else:
            var_cov = sigma2 * np.linalg.inv(X_hat.T @ X_all)

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


def lasso_iv(
    data: pd.DataFrame,
    y: str,
    x_endog: List[str],
    x_exog: Optional[List[str]] = None,
    z: Optional[List[str]] = None,
    robust: str = "robust",
    cluster: Optional[str] = None,
    penalty: str = "bic",
    alpha: float = 0.05,
) -> EconometricResults:
    """
    LASSO-selected instrumental variables.

    Uses LASSO to select relevant instruments from a large set,
    then estimates IV/2SLS with selected instruments.
    Belloni, Chen, Chernozhukov & Hansen (2012).

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
    Px = X_exog_mat @ np.linalg.solve(X_exog_mat.T @ X_exog_mat, X_exog_mat.T)
    Mx = np.eye(n) - Px

    Z_tilde = Mx @ Z_candidates  # residualized instruments
    X_endog_tilde = Mx @ df[x_endog].values.astype(float).reshape(n, -1)

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
    result.model_info["method"] = "Belloni-Chen-Chernozhukov-Hansen (2012)"
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
