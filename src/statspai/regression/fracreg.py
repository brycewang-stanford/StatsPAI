"""
Fractional response models and beta regression.

For outcomes bounded in [0,1] (proportions, rates, shares).

Papke & Wooldridge (1996) quasi-MLE (fractional logit/probit),
and beta regression (Ferrari & Cribari-Neto 2004).

Equivalent to Stata's ``fracreg`` and R's ``betareg::betareg()``.

References
----------
Papke, L.E. & Wooldridge, J.M. (1996).
"Econometric Methods for Fractional Response Variables with an
Application to 401(k) Plan Participation Rates."
*Journal of Applied Econometrics*, 11(6), 619-632. [@papke1996econometric]

Ferrari, S.L.P. & Cribari-Neto, F. (2004).
"Beta Regression for Modelling Rates and Proportions."
*Journal of Applied Statistics*, 31(7), 799-815. [@ferrari2004beta]
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from .._aliases import accepts_aliases
from ..core._vcov_spec import markout_clusters
from ..core.results import EconometricResults
from ._optim_helpers import robust_convergence


@accepts_aliases(vce="robust")
@markout_clusters
def fracreg(
    data: pd.DataFrame = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    link: str = "logit",
    robust: str = "robust",
    cluster: Optional[str] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Fractional response model (Papke & Wooldridge 1996).

    Quasi-MLE for outcomes in [0, 1]. Uses Bernoulli log-likelihood
    with robust (sandwich) standard errors.

    Equivalent to Stata's ``fracreg logit y x``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable in [0, 1].
    x : list of str
        Regressors.
    link : str, default 'logit'
        Link function: 'logit' or 'probit'.
    robust : str, default 'robust'
        Always use robust SE (quasi-MLE).
    cluster : str, optional
        Cluster variable for clustered SE.
    maxiter : int, default 100
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> income = rng.normal(0, 1, n)
    >>> age = rng.normal(0, 1, n)
    >>> mu = 1 / (1 + np.exp(-(0.5 + 0.8 * income - 0.3 * age)))
    >>> df = pd.DataFrame({
    ...     'participation_rate': np.clip(
    ...         mu + rng.normal(0, 0.05, n), 0.01, 0.99
    ...     ),
    ...     'income': income,
    ...     'age': age,
    ... })
    >>> result = sp.fracreg(df, y='participation_rate', x=['income', 'age'])
    >>> bool(isinstance(result.summary(), str))
    True
    """
    if data is None or y is None or x is None:
        raise ValueError("fracreg requires data, y, and x")

    x_names = list(x)
    df = data.dropna(subset=[y] + x_names)
    n = len(df)

    y_data = df[y].values.astype(float)
    if np.any((y_data < 0) | (y_data > 1)):
        raise ValueError(
            "fracreg: the outcome must lie in [0, 1]; "
            f"{int(((y_data < 0) | (y_data > 1)).sum())} observation(s) do not."
        )
    X_data = np.column_stack([np.ones(n), df[x_names].values.astype(float)])
    k = X_data.shape[1]
    var_names = ["_cons"] + x_names

    if link == "logit":

        def g(xb: np.ndarray) -> np.ndarray:
            xb = np.clip(xb, -500, 500)
            return np.asarray(1 / (1 + np.exp(-xb)), dtype=float)

        def g_prime(xb: np.ndarray) -> np.ndarray:
            p = g(xb)
            return np.asarray(p * (1 - p), dtype=float)

    elif link == "probit":

        def g(xb: np.ndarray) -> np.ndarray:
            return np.asarray(stats.norm.cdf(xb), dtype=float)

        def g_prime(xb: np.ndarray) -> np.ndarray:
            return np.asarray(stats.norm.pdf(xb), dtype=float)

    else:
        raise ValueError(f"Unknown link: {link}")

    # Quasi-MLE via IRLS (Bernoulli working likelihood)
    beta = np.zeros(k)

    for iteration in range(maxiter):
        xb = X_data @ beta
        mu = g(xb)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)
        dmu = g_prime(xb)

        # Working weights and residuals
        w = dmu**2 / (mu * (1 - mu))
        w = np.clip(w, 1e-10, 1e10)
        z = xb + (y_data - mu) / dmu

        # WLS
        W = np.diag(w)
        try:
            XtWX = X_data.T @ W @ X_data
            XtWz = X_data.T @ W @ z
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    from scipy import special

    from ..core._vcov import ml_vcov
    from ..core._vcov_spec import parse_se_request
    from ._optim_helpers import inverse_information, ml_newton_polish, se_from_vcov

    # Stata grammar. fracreg is a quasi-MLE: like Stata's fracreg it offers
    # only sandwich variances (vce(robust), the default, or vce(cluster)).
    se_req = parse_se_request(
        robust,
        cluster,
        function="fracreg",
        supported=("robust", "hc0", "hc1", "cluster"),
    )
    robust, cluster = se_req.kind, se_req.cluster

    def obs_qll(theta: np.ndarray) -> np.ndarray:
        """Per-observation Bernoulli quasi-log-likelihood; complex-safe."""
        eta = X_data @ theta
        if link == "logit":
            return y_data * eta - np.log1p(np.exp(eta))
        return y_data * special.log_ndtr(eta) + (1 - y_data) * special.log_ndtr(-eta)

    # IRLS stops on |delta beta| < tol; exact Newton steps on the quasi-
    # likelihood take it to the optimum Stata's fracreg reports. The bread
    # is the observed information, which for the probit link differs from
    # the IRLS weight matrix.
    beta, score, H, _ = ml_newton_polish(obs_qll, beta)

    # Final predictions
    xb = X_data @ beta
    mu = g(xb)
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    dmu = g_prime(xb)

    # Quasi-log-likelihood
    qll = float(np.sum(obs_qll(beta)))

    # Stata conventions (core._vcov.ml_vcov): vce(robust) carries N/(N-1),
    # which the previous hand-rolled sandwich omitted.
    clusters = df[cluster].values if cluster is not None else None
    var_cov = ml_vcov(inverse_information(H), score, kind=robust, clusters=clusters)

    se = se_from_vcov(var_cov)
    params = pd.Series(beta, index=var_names)
    std_errors = pd.Series(se, index=var_names)

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "alpha": alpha,
            "model_type": f"Fractional {link.title()} (Papke-Wooldridge)",
            "link": link,
            "quasi_ll": qll,
        },
        data_info={
            "n_obs": n,
            "dep_var": y,
            "df_resid": n - k,
            # z / chi2 inference, as Stata's fracreg.
            "inference": "z",
            "var_cov": var_cov,
        },
        diagnostics={
            "quasi_log_likelihood": qll,
            "aic": -2 * qll + 2 * k,
            "bic": -2 * qll + np.log(n) * k,
        },
    )


@accepts_aliases(vce="robust")
@markout_clusters
def betareg(
    data: pd.DataFrame = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    z: Optional[List[str]] = None,
    link: str = "logit",
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Beta regression (Ferrari & Cribari-Neto 2004).

    Full parametric model for outcomes in (0, 1) using the Beta distribution.

    Equivalent to R's ``betareg::betareg()`` and Stata's ``betareg``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome in (0, 1).
    x : list of str
        Regressors for the mean equation.
    z : list of str, optional
        Regressors for the precision equation. If None, constant precision.
    link : str, default 'logit'
        Link for mean: 'logit', 'probit', 'cloglog'.
    robust : str, default 'nonrobust'
    cluster : str, optional
    maxiter : int, default 200
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> price = rng.normal(0, 1, n)
    >>> quality = rng.normal(0, 1, n)
    >>> mu = 1 / (1 + np.exp(-(0.3 + 0.6 * price - 0.4 * quality)))
    >>> df = pd.DataFrame({
    ...     'share': np.clip(mu + rng.normal(0, 0.05, n), 0.01, 0.99),
    ...     'price': price,
    ...     'quality': quality,
    ... })
    >>> result = sp.betareg(df, y='share', x=['price', 'quality'])
    >>> bool(isinstance(result.summary(), str))
    True
    """
    if data is None or y is None or x is None:
        raise ValueError("betareg requires data, y, and x")

    from scipy import special

    from ..core._vcov import ml_vcov
    from ..core._vcov_spec import parse_se_request
    from ._optim_helpers import inverse_information, ml_newton_polish, se_from_vcov

    # Stata grammar; robust= and cluster= used to be accepted and ignored.
    se_req = parse_se_request(
        robust,
        cluster,
        function="betareg",
        supported=("nonrobust", "robust", "hc0", "hc1", "cluster"),
    )
    robust, cluster = se_req.kind, se_req.cluster

    x_names = list(x)
    z_names = list(z) if z is not None else []
    df = data.dropna(
        subset=[y] + x_names + z_names + ([cluster] if cluster is not None else [])
    )
    n = len(df)

    y_data = df[y].values.astype(float)
    # The beta density is zero at 0 and 1. Values on or outside the boundary
    # used to be clipped to [1e-6, 1 - 1e-6] without notice, which changes
    # the likelihood arbitrarily; refuse instead, as Stata's betareg does.
    outside = (y_data <= 0) | (y_data >= 1)
    if outside.any():
        raise ValueError(
            f"betareg: the outcome must lie strictly inside (0, 1); "
            f"{int(outside.sum())} observation(s) do not. Use sp.fracreg for "
            "outcomes that take the values 0 or 1."
        )

    X_mean = np.column_stack([np.ones(n), df[x_names].values.astype(float)])
    k_mean = X_mean.shape[1]
    mean_names = ["_cons"] + x_names

    if z_names:
        X_prec = np.column_stack([np.ones(n), df[z_names].values.astype(float)])
        prec_names = ["_cons_phi"] + [f"phi_{v}" for v in z_names]
    else:
        X_prec = np.ones((n, 1))
        prec_names = ["_cons_phi"]
    k_prec = X_prec.shape[1]

    # Inverse mean links (Stata betareg: logit, probit, cloglog, loglog).
    # ``link='cloglog'`` used to fall through to the logit link silently.
    inverse_links: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "logit": lambda eta: 1.0 / (1.0 + np.exp(-eta)),
        "probit": lambda eta: special.ndtr(eta),
        "cloglog": lambda eta: 1.0 - np.exp(-np.exp(eta)),
        "loglog": lambda eta: np.exp(-np.exp(-eta)),
    }
    if link not in inverse_links:
        raise ValueError(
            f"betareg: unknown link {link!r}; use one of {sorted(inverse_links)}."
        )
    g = inverse_links[link]
    log_y = np.log(y_data)
    log_1my = np.log1p(-y_data)

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        """Per-observation beta log-likelihood; complex-step safe."""
        mu = g(X_mean @ theta[:k_mean])
        phi = np.exp(X_prec @ theta[k_mean:])
        a = mu * phi
        b = (1 - mu) * phi
        return (
            special.loggamma(phi)
            - special.loggamma(a)
            - special.loggamma(b)
            + (a - 1) * log_y
            + (b - 1) * log_1my
        )

    def neg_log_lik(theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            value = float(-np.sum(np.real(obs_loglik(theta))))
        return value if np.isfinite(value) else 1e300

    # Initialize
    theta0 = np.zeros(k_mean + k_prec)
    theta0[k_mean] = np.log(5)  # initial precision

    try:
        result = minimize(
            neg_log_lik,
            theta0,
            method="BFGS",
            options={"maxiter": maxiter, "gtol": tol},
        )
        theta_start = np.asarray(result.x, dtype=float)
        bfgs_converged, _ = robust_convergence(result)
    except Exception:
        theta_start = theta0
        bfgs_converged = False

    # Exact Newton steps to the optimum, then the observed information and
    # per-observation scores for the oim / robust / cluster variances.
    theta_hat, scores, H, newton_steps = ml_newton_polish(obs_loglik, theta_start)
    grad_norm = float(np.linalg.norm(scores.sum(axis=0)))
    converged = bool(bfgs_converged or grad_norm < 1e-6)
    k_total = len(theta_hat)

    cluster_vals = df[cluster].values if cluster is not None else None
    var_cov = ml_vcov(
        inverse_information(H), scores, kind=robust, clusters=cluster_vals
    )
    se = se_from_vcov(var_cov)

    all_names = mean_names + prec_names
    params = pd.Series(theta_hat, index=all_names)
    std_errors = pd.Series(se, index=all_names)

    ll = -neg_log_lik(theta_hat)

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "alpha": alpha,
            "model_type": "Beta Regression (Ferrari-Cribari-Neto)",
            "link": link,
            "n_mean_params": k_mean,
            "n_precision_params": k_prec,
            "converged": converged,
            "gradient_norm": grad_norm,
            "newton_steps": newton_steps,
            "robust": robust,
            "cluster": cluster,
        },
        data_info={
            "n_obs": n,
            "dep_var": y,
            "df_resid": n - k_total,
            # z / chi2 inference, as Stata's betareg.
            "inference": "z",
            "var_cov": var_cov,
        },
        diagnostics={
            "log_likelihood": ll,
            "aic": -2 * ll + 2 * k_total,
            "bic": -2 * ll + np.log(n) * k_total,
        },
    )
