"""
Truncated regression model.

MLE estimation for outcomes that are only observed within a range
(left-truncated, right-truncated, or both).

Equivalent to Stata's ``truncreg`` and R's ``truncreg::truncreg()``.

References
----------
Hausman, J.A. & Wise, D.A. (1977).
"Social Experimentation, Truncated Distributions, and Efficient
Estimation." *Econometrica*, 45(4), 919-938. [@hausman1977social]
"""

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .._aliases import accepts_aliases
from ..core._vcov_spec import markout_clusters
from ..core.results import EconometricResults
from ._optim_helpers import robust_convergence


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


@accepts_aliases(vce="robust")
@markout_clusters
def truncreg(
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    ll: Optional[float] = None,
    ul: Optional[float] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Truncated regression (MLE).

    For samples where the outcome is only observed if it falls
    within [ll, ul]. Different from censoring (Tobit): here the
    *observation itself* is missing outside the range.

    Equivalent to Stata's ``truncreg y x, ll(0)`` and
    R's ``truncreg::truncreg()``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x : list of str
        Regressors.
    ll : float, optional
        Lower truncation point. None = no lower truncation.
    ul : float, optional
        Upper truncation point. None = no upper truncation.
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
    >>> n = 400
    >>> education = rng.integers(8, 18, n)
    >>> experience = rng.integers(0, 30, n)
    >>> wage = (2.0 + 0.4 * education + 0.1 * experience
    ...         + rng.normal(0, 2, n))
    >>> df = pd.DataFrame({'wage': wage, 'education': education,
    ...                    'experience': experience})
    >>> df = df[df['wage'] > 0]  # observed only if wage > 0 (truncated)
    >>> result = sp.truncreg(df, y='wage',
    ...                      x=['education', 'experience'], ll=0)
    >>> print(result.summary())  # doctest: +SKIP
    """
    if data is None:
        raise ValueError("'data' must be provided.")
    if y is None or x is None:
        raise ValueError("Both 'y' and 'x' must be provided.")
    if ll is not None and ul is not None and ll >= ul:
        raise ValueError(f"Lower limit ({ll}) must be < upper limit ({ul})")

    import warnings

    from scipy import special

    from ..core._vcov import ml_vcov
    from ..core._vcov_spec import parse_se_request
    from ._optim_helpers import inverse_information, ml_newton_polish, se_from_vcov

    # Stata grammar; robust= and cluster= used to be accepted and ignored.
    se_req = parse_se_request(
        robust,
        cluster,
        function="truncreg",
        supported=("nonrobust", "robust", "hc0", "hc1", "cluster"),
    )
    robust, cluster = se_req.kind, se_req.cluster

    x_names = list(x)
    df = data.dropna(subset=[y] + x_names + ([cluster] if cluster is not None else []))

    # Observations outside the truncation limits cannot have come from the
    # truncated distribution; Stata's truncreg drops them, and so do we
    # (previously they were kept and the density evaluated where it is 0).
    inside = np.ones(len(df), dtype=bool)
    if ll is not None:
        inside &= df[y].values > ll
    if ul is not None:
        inside &= df[y].values < ul
    n_truncated = int((~inside).sum())
    if n_truncated:
        warnings.warn(
            f"truncreg: dropped {n_truncated} observation(s) outside the "
            "truncation limits, as Stata's truncreg does.",
            stacklevel=2,
        )
        df = df[inside]
    n = len(df)

    y_data = df[y].values.astype(float)
    X_data = np.column_stack([np.ones(n), df[x_names].values.astype(float)])
    k = X_data.shape[1]
    var_names = ["_cons"] + x_names

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        """Per-observation log-likelihood at (beta, ln sigma); complex-safe."""
        ln_sigma = theta[k]
        sigma = np.exp(ln_sigma)
        xb = X_data @ theta[:k]
        z = (y_data - xb) / sigma
        out = -0.5 * z * z - 0.5 * np.log(2.0 * np.pi) - ln_sigma
        if ll is not None and ul is not None:
            out = out - np.log(
                special.ndtr((ul - xb) / sigma) - special.ndtr((ll - xb) / sigma)
            )
        elif ll is not None:
            out = out - special.log_ndtr((xb - ll) / sigma)
        elif ul is not None:
            out = out - special.log_ndtr((ul - xb) / sigma)
        return out

    def neg_log_lik(theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            value = float(-np.sum(obs_loglik(theta)))
        return value if np.isfinite(value) else 1e300

    # Initialize with OLS
    beta_init = np.linalg.lstsq(X_data, y_data, rcond=None)[0]
    resid = y_data - X_data @ beta_init
    sigma_init = np.std(resid)
    theta0 = np.concatenate([beta_init, [np.log(max(sigma_init, 0.1))]])

    result = minimize(
        neg_log_lik, theta0, method="BFGS", options={"maxiter": maxiter, "gtol": tol}
    )
    # BFGS stops on its gradient tolerance; finish with exact Newton steps.
    theta_hat, scores, H, newton_steps = ml_newton_polish(
        obs_loglik, _as_float_array(result.x)
    )
    grad_norm = float(np.linalg.norm(scores.sum(axis=0)))
    converged = bool(robust_convergence(result)[0] or grad_norm < 1e-6)

    beta_hat = theta_hat[:k]
    sigma_hat = float(np.exp(theta_hat[k]))

    cluster_vals = df[cluster].values if cluster is not None else None
    var_cov = ml_vcov(
        inverse_information(H), scores, kind=robust, clusters=cluster_vals
    )
    se = se_from_vcov(var_cov)
    k_total = len(theta_hat)

    all_names = var_names + ["ln_sigma"]
    all_params = np.concatenate([beta_hat, [theta_hat[k]]])

    params = pd.Series(all_params, index=all_names)
    std_errors = pd.Series(se, index=all_names)

    ll_val = float(-neg_log_lik(theta_hat))

    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info={
            "alpha": alpha,
            "model_type": "Truncated Regression",
            "lower_limit": ll,
            "upper_limit": ul,
            "sigma": sigma_hat,
            "converged": converged,
            "gradient_norm": grad_norm,
            "newton_steps": newton_steps,
            "robust": robust,
            "cluster": cluster,
            "n_truncated": n_truncated,
        },
        data_info={
            "n_obs": n,
            "dep_var": y,
            "df_resid": n - k - 1,
            # z / chi2 inference, as Stata's truncreg; covariance of
            # (beta, ln_sigma) in params order for sp.test / sp.lincom.
            "inference": "z",
            "var_cov": var_cov,
        },
        diagnostics={
            "log_likelihood": ll_val,
            "sigma": sigma_hat,
            "aic": -2 * ll_val + 2 * k_total,
            "bic": -2 * ll_val + np.log(n) * k_total,
        },
    )
