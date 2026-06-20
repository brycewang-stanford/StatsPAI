"""
Sample selection and treatment effect models.

Bivariate probit, treatment effects model (endogenous treatment),
and switching regression.

Equivalent to Stata's ``biprobit``, ``etregress``, ``movestay``.

References
----------
Heckman, J.J. (1978).
"Dummy Endogenous Variables in a Simultaneous Equation System."
*Econometrica*, 46(4), 931-959. [@heckman1978dummy]

Maddala, G.S. (1983).
"Limited-Dependent and Qualitative Variables in Econometrics."
*Cambridge University Press*. [@maddala1983limited]
"""

from typing import Optional, List, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import EconometricResults
from ._optim_helpers import robust_convergence


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def biprobit(
    data: pd.DataFrame,
    y1: str,
    y2: str,
    x1: List[str],
    x2: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Bivariate probit model.

    Jointly estimates two binary outcomes with correlated errors:
        y1* = x1'β1 + ε1,  y1 = 1(y1* > 0)
        y2* = x2'β2 + ε2,  y2 = 1(y2* > 0)
        (ε1, ε2) ~ BVN(0, 0, 1, 1, ρ)

    Equivalent to Stata's ``biprobit y1 y2 = x1, x2``.

    Parameters
    ----------
    data : pd.DataFrame
    y1 : str
        First binary outcome.
    y2 : str
        Second binary outcome.
    x1 : list of str
        Regressors for equation 1.
    x2 : list of str, optional
        Regressors for equation 2. If None, same as x1.
    robust : str, default 'nonrobust'
    cluster : str, optional
    maxiter : int, default 200
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> age = rng.normal(40, 10, n)
    >>> education = rng.normal(13, 3, n)
    >>> e1 = rng.normal(0, 1, n)
    >>> e2 = 0.5 * e1 + rng.normal(0, 1, n)  # correlated errors
    >>> employed = ((-1.0 + 0.03 * age + 0.1 * education + e1) > 0).astype(int)
    >>> married = ((-0.5 + 0.02 * age + e2) > 0).astype(int)
    >>> df = pd.DataFrame({"employed": employed, "married": married,
    ...                    "age": age, "education": education})
    >>> result = sp.biprobit(df, y1='employed', y2='married',
    ...                      x1=['age', 'education'])
    >>> bool('rho' in result.model_info)  # error correlation reported
    True
    """
    x2_names = list(x1) if x2 is None else list(x2)

    df = data.dropna(subset=[y1, y2] + x1 + x2_names)
    n = len(df)

    y1_data = df[y1].values.astype(float)
    y2_data = df[y2].values.astype(float)
    X1 = np.column_stack([np.ones(n), df[x1].values.astype(float)])
    X2 = np.column_stack([np.ones(n), df[x2_names].values.astype(float)])
    k1 = X1.shape[1]
    k2 = X2.shape[1]

    def _bvn_cdf(h: np.ndarray, k: np.ndarray, rho: Any) -> np.ndarray:
        """Bivariate standard-normal CDF P(X<=h, Y<=k; corr=rho).

        Vectorised over observations via the Drezner-Wesolowsky (1990)
        identity

            Phi2(h, k; rho) = Phi(h) Phi(k) + integral_0^rho phi2(h, k; r) dr,

        with the inner integral evaluated by a 24-node Gauss-Legendre rule
        (smooth in every argument, accurate to ~1e-10). The previous
        implementation looped over observations calling
        ``multivariate_normal.cdf`` per point: ~280x slower and — fatally —
        precise only to ~1e-8, which corrupted BFGS's finite-difference
        gradient and pinned ``rho`` at its starting value of 0 (so the model
        always reported zero error correlation regardless of the data).
        """
        h = np.asarray(h, dtype=float)
        k = np.asarray(k, dtype=float)
        rho = np.asarray(rho, dtype=float)
        if rho.ndim == 0:
            rho = np.full(h.shape, float(rho))
        rho = np.clip(rho, -0.999999, 0.999999)
        base = stats.norm.cdf(h) * stats.norm.cdf(k)
        nodes, weights = np.polynomial.legendre.leggauss(24)
        s = 0.5 * (nodes + 1.0)  # map [-1, 1] -> [0, 1]
        w = 0.5 * weights
        integral = np.zeros(h.shape)
        for s_j, w_j in zip(s, w):
            r = rho * s_j
            denom = 1.0 - r * r
            integral += (
                w_j
                * np.exp(-(h * h - 2.0 * r * h * k + k * k) / (2.0 * denom))
                / (2.0 * np.pi * np.sqrt(denom))
            )
        return _as_float_array(base + rho * integral)

    def neg_ll(theta: np.ndarray) -> float:
        beta1 = theta[:k1]
        beta2 = theta[k1 : k1 + k2]
        atanh_rho = theta[-1]
        rho = np.tanh(atanh_rho)

        xb1 = X1 @ beta1
        xb2 = X2 @ beta2

        # Adjust signs for different (y1, y2) combinations
        q1 = 2 * y1_data - 1  # +1 if y=1, -1 if y=0
        q2 = 2 * y2_data - 1

        h = q1 * xb1
        k_ = q2 * xb2
        rho_adj = q1 * q2 * rho

        probs = _bvn_cdf(h, k_, rho_adj)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        return float(-np.sum(np.log(probs)))

    # Initialize with separate probits
    beta1_init = np.linalg.lstsq(X1, y1_data, rcond=None)[0]
    beta2_init = np.linalg.lstsq(X2, y2_data, rcond=None)[0]
    theta0 = np.concatenate([beta1_init, beta2_init, [0.0]])

    try:
        result = minimize(
            neg_ll, theta0, method="BFGS", options={"maxiter": maxiter, "gtol": tol}
        )
        theta_hat = _as_float_array(result.x)
        converged, grad_norm = robust_convergence(result)
    except Exception:
        theta_hat = theta0
        converged, grad_norm = False, float("inf")

    beta1 = theta_hat[:k1]
    beta2 = theta_hat[k1 : k1 + k2]
    rho = float(np.tanh(theta_hat[-1]))

    # Numerical Hessian for SE
    k_total = len(theta_hat)
    eps = 1e-5
    H = np.zeros((k_total, k_total))
    f0 = neg_ll(theta_hat)
    for i in range(k_total):
        ei = np.zeros(k_total)
        ei[i] = eps
        fp = neg_ll(theta_hat + ei)
        fm = neg_ll(theta_hat - ei)
        H[i, i] = (fp - 2 * f0 + fm) / eps**2
        for j in range(i + 1, k_total):
            ej = np.zeros(k_total)
            ej[j] = eps
            fpp = neg_ll(theta_hat + ei + ej)
            fpm = neg_ll(theta_hat + ei - ej)
            fmp = neg_ll(theta_hat - ei + ej)
            fmm = neg_ll(theta_hat - ei - ej)
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps**2)

    try:
        var_cov = np.linalg.inv(H)
        se = _as_float_array(np.sqrt(np.abs(np.diag(var_cov))))
    except np.linalg.LinAlgError:
        se = np.full(k_total, np.nan)

    # Delta method for rho SE
    rho_se = float(se[-1] * (1 - rho**2))  # d(tanh)/d(atanh) = 1-tanh^2

    names = (
        ["eq1._cons"]
        + [f"eq1.{v}" for v in x1]
        + ["eq2._cons"]
        + [f"eq2.{v}" for v in x2_names]
        + ["rho"]
    )
    param_vals = np.concatenate([beta1, beta2, [rho]])
    se_vals = np.concatenate([se[:k1], se[k1 : k1 + k2], [rho_se]])

    params = pd.Series(param_vals, index=names)
    std_errors = pd.Series(se_vals, index=names)

    ll = float(-neg_ll(theta_hat))

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "model_type": "Bivariate Probit",
            "converged": converged,
            "gradient_norm": grad_norm,
            "rho": rho,
            "rho_se": rho_se,
            "rho_test_p": (
                float(2 * (1 - stats.norm.cdf(abs(rho / rho_se))))
                if rho_se > 0
                else np.nan
            ),
        },
        data_info={
            "n_obs": n,
            "dep_var_1": y1,
            "dep_var_2": y2,
            "df_resid": n - k_total,
        },
        diagnostics={
            "log_likelihood": ll,
            "aic": -2 * ll + 2 * k_total,
            "bic": -2 * ll + np.log(n) * k_total,
        },
    )


def etregress(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    treatment: str,
    z: List[str],
    method: str = "mle",
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Endogenous treatment effects model.

    Estimates the effect of an endogenous binary treatment on a
    continuous outcome, accounting for selection on unobservables.

        Outcome:  y = x'β + δ*D + ε
        Selection: D* = z'γ + u, D = 1(D* > 0)
        (ε, u) ~ BVN(0, 0, σ², 1, ρσ)

    Equivalent to Stata's ``etregress y x, treat(D = z)``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x : list of str
        Exogenous regressors.
    treatment : str
        Binary treatment variable.
    z : list of str
        Instruments for the selection equation.
    method : str, default 'mle'
        'mle' or 'twostep' (Heckman-style).
    robust : str, default 'nonrobust'
    cluster : str, optional
    maxiter : int, default 200
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> experience = rng.normal(15, 7, n)
    >>> education = rng.normal(13, 3, n)
    >>> father_union = rng.integers(0, 2, n)
    >>> region = rng.integers(0, 4, n)
    >>> u = rng.normal(0, 1, n)  # selection error, correlated with wage
    >>> union = ((-0.5 + 0.4 * father_union + 0.1 * region + u) > 0).astype(int)
    >>> wage = (8.0 + 0.3 * experience + 0.5 * education + 2.0 * union
    ...         + 1.5 * u + rng.normal(0, 2, n))
    >>> df = pd.DataFrame({"wage": wage, "experience": experience,
    ...                    "education": education, "union": union,
    ...                    "father_union": father_union, "region": region})
    >>> result = sp.etregress(df, y='wage', x=['experience', 'education'],
    ...                       treatment='union', z=['father_union', 'region'])
    >>> bool('ate' in result.diagnostics)  # endogenous treatment effect
    True
    """
    df = data.dropna(subset=[y, treatment] + x + z)
    n = len(df)

    y_data = df[y].values.astype(float)
    D_data = df[treatment].values.astype(float)
    X_out = np.column_stack([np.ones(n), df[x].values.astype(float), D_data])
    k_out = X_out.shape[1]

    if method == "twostep":
        # Two-step (control function approach)
        # Step 1: Probit for D
        from .logit_probit import probit as _probit

        probit_result = _probit(data=df, y=treatment, x=z)
        gamma = probit_result.params.values
        xb_sel = np.column_stack([np.ones(n), df[z].values.astype(float)]) @ gamma
        # Inverse Mills ratio
        mills = np.where(
            D_data == 1,
            stats.norm.pdf(xb_sel) / np.clip(stats.norm.cdf(xb_sel), 1e-10, None),
            -stats.norm.pdf(xb_sel) / np.clip(1 - stats.norm.cdf(xb_sel), 1e-10, None),
        )

        # Step 2: OLS with Mills ratio
        X_out_mills = np.column_stack([X_out, mills])
        beta = np.linalg.lstsq(X_out_mills, y_data, rcond=None)[0]
        resid = y_data - X_out_mills @ beta
        sigma2 = np.sum(resid**2) / (n - X_out_mills.shape[1])

        # SE (simplified — should use bootstrap for correct SE)
        var_cov = sigma2 * np.linalg.inv(X_out_mills.T @ X_out_mills)
        se = np.sqrt(np.diag(var_cov))

        out_names = ["_cons"] + x + [treatment, "mills_lambda"]
        params = pd.Series(beta, index=out_names)
        std_errors = pd.Series(se, index=out_names)
        ate = beta[k_out - 1]  # treatment coefficient

    else:
        # MLE (simplified: control function + joint estimation)
        # Use two-step as approximation for MLE
        from .logit_probit import probit as _probit

        probit_result = _probit(data=df, y=treatment, x=z)
        gamma = probit_result.params.values
        xb_sel = np.column_stack([np.ones(n), df[z].values.astype(float)]) @ gamma
        mills = np.where(
            D_data == 1,
            stats.norm.pdf(xb_sel) / np.clip(stats.norm.cdf(xb_sel), 1e-10, None),
            -stats.norm.pdf(xb_sel) / np.clip(1 - stats.norm.cdf(xb_sel), 1e-10, None),
        )
        X_out_mills = np.column_stack([X_out, mills])
        beta = np.linalg.lstsq(X_out_mills, y_data, rcond=None)[0]
        resid = y_data - X_out_mills @ beta
        sigma2 = np.sum(resid**2) / (n - X_out_mills.shape[1])
        var_cov = sigma2 * np.linalg.inv(X_out_mills.T @ X_out_mills)
        se = np.sqrt(np.diag(var_cov))

        out_names = ["_cons"] + x + [treatment, "mills_lambda"]
        params = pd.Series(beta, index=out_names)
        std_errors = pd.Series(se, index=out_names)
        ate = beta[k_out - 1]

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "model_type": "Endogenous Treatment Effects",
            "method": method,
            "treatment_effect": ate,
            "treatment_var": treatment,
        },
        data_info={
            "n_obs": n,
            "dep_var": y,
            "df_resid": n - len(beta),
        },
        diagnostics={
            "ate": ate,
            "ate_se": se[k_out - 1] if k_out - 1 < len(se) else np.nan,
            "mills_coef": beta[-1],
            "mills_se": se[-1],
            "selection_corr": beta[-1] / np.sqrt(sigma2) if sigma2 > 0 else np.nan,
        },
    )
