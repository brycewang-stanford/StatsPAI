"""
Zero-Inflated and Hurdle Count Models.

Implements Zero-Inflated Poisson (ZIP), Zero-Inflated Negative Binomial
(ZINB), and Hurdle models for count data with excess zeros.

Zero-Inflated models assume two data-generating processes:
  1. A binary process (logit) that generates structural zeros with
     probability π_i = Λ(z_i'γ).
  2. A count process (Poisson or NB2) that generates counts (including
     sampling zeros) with mean μ_i = exp(x_i'β).

Hurdle models differ: zeros come from ONE process only (the logit gate),
and positive counts come from a truncated-at-zero count distribution.

References
----------
Lambert, D. (1992).
"Zero-Inflated Poisson Regression, with an Application to Defects in
Manufacturing." *Technometrics*, 34(1), 1-14. [@lambert1992zero]

Vuong, Q.H. (1989).
"Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses."
*Econometrica*, 57(2), 307-333. [@vuong1989likelihood]

Cameron, A.C. and P.K. Trivedi (2013).
*Regression Analysis of Count Data*, 2nd ed. Cambridge University Press.
[@cameron2013regression]

Mullahy, J. (1986).
"Specification and Testing of Some Modified Count Data Models."
*Journal of Econometrics*, 33(3), 341-365. [@mullahy1986specification]
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .._aliases import accepts_aliases
from ..core._vcov import ml_vcov
from ..core._vcov_spec import markout_clusters
from ..core.results import EconometricResults
from ..core.utils import parse_formula
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._optim_helpers import (
    inverse_information,
    ml_newton_polish,
    require_two_clusters,
    robust_convergence,
    se_from_vcov,
)


class _FittedArraysDict(Dict[str, Any]):
    """``diagnostics`` of the zero-inflated and hurdle fits.

    Through 1.38 it also held three vectors of fitted values, one entry
    per observation, which every summary, JSON export and agent payload
    then carried along. They live in ``data_info`` now. The old keys still
    answer here, with a ``DeprecationWarning``, but are no longer listed
    when the dictionary is iterated or printed.
    """

    def __init__(self, scalars: Dict[str, Any], arrays: Dict[str, Any]):
        super().__init__(scalars)
        self._arrays = dict(arrays)

    def _legacy(self, key: Any) -> Any:
        import warnings

        warnings.warn(
            f"diagnostics[{key!r}] is deprecated and will be removed in "
            f"StatsPAI 1.41; read data_info[{key!r}] instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return self._arrays[key]

    def __missing__(self, key: Any) -> Any:
        if key in self._arrays:
            return self._legacy(key)
        raise KeyError(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        if key in self._arrays:
            return self._legacy(key)
        return default

    def __contains__(self, key: object) -> bool:
        return dict.__contains__(self, key) or key in self._arrays

    def __reduce__(self) -> Any:
        return (_FittedArraysDict, (dict(self), self._arrays))


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _logit(z: np.ndarray) -> np.ndarray:
    """Logistic sigmoid, numerically stable."""
    return _as_float_array(special.expit(z))


def _warn_flat_terms(se: pd.Series, function: str) -> None:
    """Warn when the likelihood is flat along some coefficient.

    Under quasi-complete separation in the binary part a coefficient runs
    off towards infinity and its standard error explodes (R ``pscl``
    prints ~2e4 at its looser stopping rule, the exact optimum gives far
    more). The other coefficients are unaffected, so the fit is returned,
    but the term is named rather than left for the reader to spot.
    """
    flat = [str(name) for name, v in se.items() if np.isfinite(v) and v > 1e4]
    if flat:
        import warnings

        from ..exceptions import ConvergenceWarning

        warnings.warn(
            f"{function}: the likelihood is flat along {flat} (standard error "
            "above 1e4), which is what quasi-complete separation in the "
            "zero / hurdle part looks like. Those coefficients are not "
            "identified; the remaining estimates and standard errors are "
            "unaffected. Drop or recode the separating regressor to remove "
            "the term.",
            ConvergenceWarning,
            stacklevel=3,
        )


def _log_expit(z: np.ndarray) -> np.ndarray:
    """``log(expit(z))`` without overflow; complex-step safe.

    ``1 / (1 + exp(-z))`` overflows once ``-z`` passes ~709, and one
    overflowing row turns the whole complex-step Hessian into NaN. That
    happens whenever a binary-part coefficient runs off under
    quasi-complete separation.
    """
    neg = np.real(z) < 0
    return np.asarray(np.where(neg, z, 0.0) - np.log1p(np.exp(np.where(neg, z, -z))))


def _logaddexp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``log(exp(a) + exp(b))``; complex-step safe (``np.logaddexp`` is not)."""
    a_big = np.real(a) >= np.real(b)
    hi = np.where(a_big, a, b)
    lo = np.where(a_big, b, a)
    return np.asarray(hi + np.log1p(np.exp(lo - hi)))


def _log_poisson_pmf(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Log P(Y=y | mu) for Poisson, vectorized."""
    return _as_float_array(
        y * np.log(np.maximum(mu, 1e-20)) - mu - special.gammaln(y + 1)
    )


def _log_nb2_pmf(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """
    Log P(Y=y | mu, alpha) for NB2 parameterization.

    Var(Y) = mu + alpha * mu^2.  Let r = 1/alpha.
    P(Y=y) = Gamma(y+r)/(Gamma(r)*y!) * (r/(r+mu))^r * (mu/(r+mu))^y
    """
    r = 1.0 / max(alpha, 1e-10)
    log_p = (
        special.gammaln(y + r)
        - special.gammaln(r)
        - special.gammaln(y + 1)
        + r * np.log(r / (r + mu))
        + y * np.log(np.maximum(mu, 1e-20) / (r + mu))
    )
    return _as_float_array(log_p)


def _require_plain_columns(data: pd.DataFrame, names: List[str], function: str) -> None:
    """Every regressor of this family is a column of ``data``.

    The zero-inflated / hurdle estimators read names as columns; a patsy
    term (``C(g)``, ``x1:x2``) or a misspelled ``inflate=`` name used to end
    in a bare pandas ``KeyError``.
    """
    missing = [str(v) for v in names if v not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"{function}: {missing} not found among the columns of data. "
            "This estimator takes column names only; formula terms such as "
            "C(g), I(x**2) or x1:x2 are not expanded.",
            recovery_hint=(
                "Check the spelling, or build the term as a column first "
                "(pd.get_dummies for a factor, a product for an interaction)."
            ),
            diagnostics={"missing": missing},
        )


def _omit_collinear(
    X: np.ndarray, names: List[str], function: str
) -> Tuple[np.ndarray, List[str], List[Dict[str, str]]]:
    """Drop dependent columns with a note, as the other ML estimators do."""
    from ..core._collinear import drop_collinear

    X, names, omitted, _ = drop_collinear(X, names, function, stacklevel=5)
    return X, names, omitted


def _build_matrices(
    data: Optional[pd.DataFrame],
    formula: Optional[str],
    y: Optional[str],
    x: Optional[List[str]],
    inflate: Optional[List[str]],
    function: str = "zip_model",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], str, pd.DataFrame]:
    """
    Parse inputs and return
    (Y, X_count, X_inflate, count_names, inflate_names, dep_var, df).

    X matrices include a constant column. Collinear regressors are omitted
    from each equation; the records are left in ``df.attrs['omitted']``.
    """
    if data is None:
        raise ValueError("`data` must be provided.")
    if formula is not None:
        parsed = parse_formula(formula)
        dep_var = str(parsed["dependent"])
        x_vars = [str(name) for name in parsed["exogenous"]]
    else:
        if y is None or x is None:
            raise ValueError("Provide either `formula` or both `y` and `x`.")
        dep_var = y
        x_vars = list(x)

    if inflate is None:
        inflate_vars = list(x_vars)
    else:
        inflate_vars = list(inflate)

    all_vars = list(set([dep_var] + x_vars + inflate_vars))
    _require_plain_columns(data, [dep_var] + x_vars + inflate_vars, function)
    df = data[all_vars].dropna()

    Y = df[dep_var].values.astype(float)
    if np.any(Y < 0) or not np.all(Y == Y.astype(int)):
        raise ValueError("Dependent variable must contain non-negative integers.")
    Y = Y.astype(int)

    # Count equation design matrix (with constant)
    X_count = np.column_stack(
        [np.ones(len(df))] + [df[v].values.astype(float) for v in x_vars]
    )
    count_names = ["const"] + x_vars

    # Inflate equation design matrix (with constant)
    X_inflate = np.column_stack(
        [np.ones(len(df))] + [df[v].values.astype(float) for v in inflate_vars]
    )
    inflate_names = ["inflate_const"] + [f"inflate_{v}" for v in inflate_vars]

    # A dependent regressor leaves the likelihood flat in one direction
    # (a constant column gave offsetting coefficients of order 1e4).
    X_count, count_names, om_c = _omit_collinear(X_count, count_names, function)
    X_inflate, inflate_names, om_i = _omit_collinear(X_inflate, inflate_names, function)
    df.attrs["omitted"] = om_c + om_i

    return Y, X_count, X_inflate, count_names, inflate_names, dep_var, df


#: SE kinds the zero-inflated / hurdle family implements.
_SE_KINDS = ("nonrobust", "robust", "hc0", "hc1", "cluster")


def _parse_se(robust: Any, cluster: Any, function: str) -> Tuple[str, Any]:
    """Resolve ``robust=`` / ``cluster=`` through the shared Stata grammar."""
    from ..core._vcov_spec import parse_se_request

    req = parse_se_request(robust, cluster, function=function, supported=_SE_KINDS)
    return req.kind, req.cluster


def _numerical_score(
    neg_loglik: Callable[[np.ndarray], float],
    theta: np.ndarray,
    n_obs: int,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute per-observation numerical score (gradient of log-lik contribution)."""
    # This is an approximation — compute gradient of total neg_loglik
    k = len(theta)
    grad = np.zeros(k)
    for j in range(k):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[j] += eps
        theta_m[j] -= eps
        grad[j] = (neg_loglik(theta_p) - neg_loglik(theta_m)) / (2 * eps)
    return _as_float_array(grad)


def _vuong_test(
    loglik_model1: np.ndarray, loglik_model2: np.ndarray
) -> Dict[str, float]:
    """
    Vuong (1989) non-nested likelihood ratio test.

    H0: models are equivalent.
    V > 1.96 favours model 1; V < -1.96 favours model 2.

    Parameters
    ----------
    loglik_model1, loglik_model2 : array of per-obs log-likelihoods.

    Returns
    -------
    dict with vuong_stat, vuong_p.
    """
    m = loglik_model1 - loglik_model2
    n = len(m)
    m_bar = m.mean()
    s_m = m.std(ddof=1)
    if s_m < 1e-15:
        return {"vuong_stat": 0.0, "vuong_p": 1.0}
    V = np.sqrt(n) * m_bar / s_m
    p = 2 * stats.norm.sf(np.abs(V))
    return {"vuong_stat": float(V), "vuong_p": float(p)}


# ===================================================================
# ZIP — Zero-Inflated Poisson
# ===================================================================


@accepts_aliases(vce="robust")
@markout_clusters
def zip_model(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    inflate: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Zero-Inflated Poisson (ZIP) regression via MLE.

    Two-part model:
      - Inflate equation: logit model for P(structural zero) = Λ(z'γ)
      - Count equation:  Poisson model with mean μ = exp(x'β)

    Equivalent to Stata's ``zip y x, inflate(z)``.

    Parameters
    ----------
    formula : str, optional
        Patsy-style formula for the count equation, e.g. "y ~ x1 + x2".
    data : pd.DataFrame
        Dataset.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Count-equation regressors (alternative to formula).
    inflate : list of str, optional
        Inflation-equation regressors. Default: same as count regressors.
    robust : str, default "nonrobust"
        "nonrobust", "HC0", "HC1", etc.
    cluster : str, optional
        Cluster variable name for clustered standard errors.
    maxiter : int, default 200
        Maximum iterations for optimizer.
    tol : float, default 1e-8
        Convergence tolerance.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
        Coefficients for both equations, Vuong test, diagnostics.
        ``diagnostics['vuong_stat']`` compares the fit with a Poisson
        regression fitted on its own. Read it as a description of fit,
        not as a test for zero inflation: the two models are nested on a
        boundary and the statistic is not standard normal under the null
        [@wilson2015misuse].

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> age = rng.normal(0, 1, n)
    >>> chronic = rng.integers(0, 2, n)
    >>> visits = rng.poisson(np.exp(0.5 + 0.3 * age))
    >>> visits[rng.random(n) < 0.3] = 0  # excess structural zeros
    >>> df = pd.DataFrame({'visits': visits, 'age': age, 'chronic': chronic})
    >>> result = sp.zip_model(data=df, y='visits', x=['age'],
    ...                       inflate=['chronic'])
    >>> print(result.summary())  # doctest: +SKIP
    >>> bool(result.model_info['model_type'] == 'zip')
    True

    Notes
    -----
    Log-likelihood for ZIP:

    .. math::
        y_i = 0: \\log[\\pi_i + (1-\\pi_i) e^{-\\mu_i}]
        y_i > 0: \\log(1-\\pi_i) + y_i \\log\\mu_i - \\mu_i - \\log(y_i!)

    where π_i = Λ(z_i'γ) and μ_i = exp(x_i'β).

    See Lambert (1992, *Technometrics*).
    """
    robust, cluster = _parse_se(robust, cluster, "zip_model")
    Y, X_count, X_inflate, count_names, inflate_names, dep_var, df = _build_matrices(
        data, formula, y, x, inflate, "zip_model"
    )
    n = len(Y)
    k_count = X_count.shape[1]
    k_inflate = X_inflate.shape[1]
    k_total = k_count + k_inflate

    # --- Negative log-likelihood ---
    def neg_loglik(theta: np.ndarray) -> float:
        beta = theta[:k_count]
        gamma = theta[k_count:]

        mu = np.exp(np.clip(X_count @ beta, -20, 20))
        pi = _logit(X_inflate @ gamma)

        # y == 0
        zero_mask = Y == 0
        ll = np.zeros(n)
        ll[zero_mask] = np.log(
            np.maximum(
                pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(-mu[zero_mask]), 1e-20
            )
        )
        # y > 0
        pos_mask = ~zero_mask
        ll[pos_mask] = np.log(np.maximum(1 - pi[pos_mask], 1e-20)) + _log_poisson_pmf(
            Y[pos_mask], mu[pos_mask]
        )
        return float(-ll.sum())

    def neg_loglik_obs(theta: np.ndarray) -> np.ndarray:
        """Per-observation negative log-likelihood (for robust SE)."""
        beta = theta[:k_count]
        gamma = theta[k_count:]
        mu = np.exp(np.clip(X_count @ beta, -20, 20))
        pi = _logit(X_inflate @ gamma)
        zero_mask = Y == 0
        ll = np.zeros(n)
        ll[zero_mask] = np.log(
            np.maximum(
                pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(-mu[zero_mask]), 1e-20
            )
        )
        pos_mask = ~zero_mask
        ll[pos_mask] = np.log(np.maximum(1 - pi[pos_mask], 1e-20)) + _log_poisson_pmf(
            Y[pos_mask], mu[pos_mask]
        )
        return _as_float_array(ll)

    # --- Initial values ---
    # Count: log-linear OLS on log(y+1)
    beta0 = np.linalg.lstsq(X_count, np.log(Y + 1), rcond=None)[0]
    gamma0 = np.zeros(k_inflate)

    theta0 = np.concatenate([beta0, gamma0])

    # --- Optimise ---
    result = optimize.minimize(
        neg_loglik,
        theta0,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": tol},
    )

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        """Per-observation ZIP log-likelihood; complex-step safe."""
        mu = np.exp(X_count @ theta[:k_count])
        zg = X_inflate @ theta[k_count:]
        log_pi, log_1mpi = _log_expit(zg), _log_expit(-zg)
        positive = log_1mpi + Y * np.log(mu) - mu - special.gammaln(Y + 1)
        zero = _logaddexp(log_pi, log_1mpi - mu)
        return np.where(Y == 0, zero, positive)

    # Exact Newton steps from the BFGS solution; the same call returns the
    # observed information and per-observation scores by complex-step
    # differentiation, replacing a second-difference Hessian and a central-
    # difference score that each carried ~1e-5 relative error.
    theta_hat, score_obs, H, _ = ml_newton_polish(obs_loglik, _as_float_array(result.x))
    beta_hat = theta_hat[:k_count]
    gamma_hat = theta_hat[k_count:]

    ll_zip = float(np.sum(obs_loglik(theta_hat)))

    # --- Standard errors ---
    clusters = None
    if robust == "cluster":
        if data is None:
            raise ValueError("`data` must be provided for clustered SEs.")
        clusters = (
            df[cluster].values
            if cluster in df.columns
            else data.loc[df.index, cluster].values
        )
        require_two_clusters(clusters, "zip_model / zinb / hurdle")
    se = se_from_vcov(
        ml_vcov(inverse_information(H), score_obs, kind=robust, clusters=clusters)
    )

    # --- Vuong test: ZIP vs plain Poisson ---
    mu_hat = np.exp(np.clip(X_count @ beta_hat, -20, 20))
    pi_hat = _logit(X_inflate @ gamma_hat)

    # The comparison model is the Poisson regression fitted on its own. Before
    # 1.39 the Poisson density was evaluated at the ZIP count coefficients,
    # which is not a maximum of the Poisson likelihood and overstated the
    # statistic in favour of ZIP.
    from .count import _poisson_irls

    ll_zip_obs = np.real(obs_loglik(theta_hat))
    mu_poisson = _poisson_irls(Y, X_count, tol=1e-12)[1]
    ll_poisson_obs = _log_poisson_pmf(Y, mu_poisson)
    vuong = _vuong_test(ll_zip_obs, ll_poisson_obs)

    # --- Predicted values ---
    pred_structural_zero = pi_hat
    pred_count = mu_hat
    pred_overall = (1 - pi_hat) * mu_hat

    # --- Assemble results ---
    all_names = count_names + inflate_names
    params = pd.Series(theta_hat, index=all_names)
    std_errors = pd.Series(se, index=all_names)
    _warn_flat_terms(std_errors, "zip_model")

    model_info = {
        "model_type": "zip",
        "method": "Zero-Inflated Poisson (MLE)",
        "dependent_var": dep_var,
        "ll": ll_zip,
        "aic": -2 * ll_zip + 2 * k_total,
        "bic": -2 * ll_zip + np.log(n) * k_total,
        "vuong_stat": vuong["vuong_stat"],
        "vuong_p": vuong["vuong_p"],
        "converged": robust_convergence(result)[0],
        "robust": robust if cluster is None else f"cluster({cluster})",
        "n_zeros": int((Y == 0).sum()),
        "pct_zeros": float((Y == 0).mean() * 100),
        "omitted": list(df.attrs.get("omitted", [])),
    }

    data_info = {
        "n_obs": n,
        "nobs": n,
        "y": Y,
        "llobs": ll_zip_obs,
        "dependent_var": dep_var,
        "df_resid": n - k_total,
        # Likelihood-based: z / chi2 inference, as Stata's zip / zinb.
        "inference": "z",
        "k_count": k_count,
        "k_inflate": k_inflate,
        "count_names": count_names,
        "inflate_names": inflate_names,
    }

    fitted_arrays = {
        "predicted_structural_zero": pred_structural_zero,
        "predicted_count": pred_count,
        "predicted_overall": pred_overall,
    }
    data_info.update(fitted_arrays)
    diagnostics = _FittedArraysDict(
        {
            "vuong_stat": vuong["vuong_stat"],
            "vuong_p": vuong["vuong_p"],
            "ll": ll_zip,
            "aic": model_info["aic"],
            "bic": model_info["bic"],
            # the same three under the names the other count models use
            "Log-Likelihood": ll_zip,
            "AIC": model_info["aic"],
            "BIC": model_info["bic"],
        },
        fitted_arrays,
    )

    model_info["alpha"] = alpha
    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _compute_zip_score_obs(
    theta: np.ndarray,
    Y: np.ndarray,
    X_count: np.ndarray,
    X_inflate: np.ndarray,
    k_count: int,
    n: int,
) -> np.ndarray:
    """Compute per-observation score for ZIP (numerical)."""
    k_total = len(theta)
    eps = 1e-5
    score = np.zeros((n, k_total))

    beta = theta[:k_count]
    gamma = theta[k_count:]
    mu = np.exp(np.clip(X_count @ beta, -20, 20))
    pi = _logit(X_inflate @ gamma)

    zero_mask = Y == 0
    pos_mask = ~zero_mask

    # Per-obs log-likelihood
    ll_obs = np.zeros(n)
    ll_obs[zero_mask] = np.log(
        np.maximum(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(-mu[zero_mask]), 1e-20)
    )
    ll_obs[pos_mask] = np.log(np.maximum(1 - pi[pos_mask], 1e-20)) + _log_poisson_pmf(
        Y[pos_mask], mu[pos_mask]
    )

    for j in range(k_total):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[j] += eps
        theta_m[j] -= eps

        beta_p = theta_p[:k_count]
        gamma_p = theta_p[k_count:]
        mu_p = np.exp(np.clip(X_count @ beta_p, -20, 20))
        pi_p = _logit(X_inflate @ gamma_p)
        ll_p = np.zeros(n)
        ll_p[zero_mask] = np.log(
            np.maximum(
                pi_p[zero_mask] + (1 - pi_p[zero_mask]) * np.exp(-mu_p[zero_mask]),
                1e-20,
            )
        )
        ll_p[pos_mask] = np.log(
            np.maximum(1 - pi_p[pos_mask], 1e-20)
        ) + _log_poisson_pmf(Y[pos_mask], mu_p[pos_mask])

        beta_m = theta_m[:k_count]
        gamma_m = theta_m[k_count:]
        mu_m = np.exp(np.clip(X_count @ beta_m, -20, 20))
        pi_m = _logit(X_inflate @ gamma_m)
        ll_m = np.zeros(n)
        ll_m[zero_mask] = np.log(
            np.maximum(
                pi_m[zero_mask] + (1 - pi_m[zero_mask]) * np.exp(-mu_m[zero_mask]),
                1e-20,
            )
        )
        ll_m[pos_mask] = np.log(
            np.maximum(1 - pi_m[pos_mask], 1e-20)
        ) + _log_poisson_pmf(Y[pos_mask], mu_m[pos_mask])

        score[:, j] = (ll_p - ll_m) / (2 * eps)

    return _as_float_array(score)


# ===================================================================
# ZINB — Zero-Inflated Negative Binomial
# ===================================================================


@accepts_aliases(vce="robust")
@markout_clusters
def zinb(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    inflate: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Zero-Inflated Negative Binomial (ZINB) regression via MLE.

    Two-part model:
      - Inflate equation: logit for P(structural zero) = Λ(z'γ)
      - Count equation:  NB2 with mean μ = exp(x'β), Var = μ + α·μ²

    Equivalent to Stata's ``zinb y x, inflate(z)``.

    Parameters
    ----------
    formula : str, optional
        Patsy-style formula for the count equation.
    data : pd.DataFrame
        Dataset.
    y : str, optional
        Dependent variable name.
    x : list of str, optional
        Count-equation regressors.
    inflate : list of str, optional
        Inflation-equation regressors. Default: same as count regressors.
    robust : str, default "nonrobust"
        Standard error type.
    cluster : str, optional
        Cluster variable name.
    maxiter : int, default 200
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        Coefficients for count, inflate, and dispersion parameter.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> age = rng.normal(0, 1, n)
    >>> chronic = rng.integers(0, 2, n)
    >>> visits = rng.poisson(np.exp(0.5 + 0.3 * age))
    >>> visits[rng.random(n) < 0.3] = 0  # excess structural zeros
    >>> df = pd.DataFrame({'visits': visits, 'age': age, 'chronic': chronic})
    >>> result = sp.zinb(data=df, y='visits', x=['age'],
    ...                  inflate=['chronic'])
    >>> print(result.summary())  # doctest: +SKIP
    >>> bool(result.model_info['model_type'] == 'zinb')
    True

    Notes
    -----
    The NB2 parameterization uses dispersion parameter α so that
    Var(Y|μ) = μ + α·μ². When α → 0 the model collapses to ZIP.

    See Cameron & Trivedi (2013, Ch. 4).
    """
    robust, cluster = _parse_se(robust, cluster, "zinb")
    Y, X_count, X_inflate, count_names, inflate_names, dep_var, df = _build_matrices(
        data, formula, y, x, inflate, "zinb"
    )
    n = len(Y)
    k_count = X_count.shape[1]
    k_inflate = X_inflate.shape[1]
    # theta = [beta, gamma, log_alpha]
    k_total = k_count + k_inflate + 1

    def neg_loglik(theta: np.ndarray) -> float:
        beta = theta[:k_count]
        gamma = theta[k_count : k_count + k_inflate]
        log_alpha = theta[-1]
        disp = np.exp(np.clip(log_alpha, -10, 10))

        mu = np.exp(np.clip(X_count @ beta, -20, 20))
        pi = _logit(X_inflate @ gamma)

        zero_mask = Y == 0
        ll = np.zeros(n)

        # NB2 pmf at y=0
        nb_zero = _log_nb2_pmf(np.zeros_like(mu[zero_mask]), mu[zero_mask], disp)
        ll[zero_mask] = np.log(
            np.maximum(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(nb_zero), 1e-20)
        )

        pos_mask = ~zero_mask
        ll[pos_mask] = np.log(np.maximum(1 - pi[pos_mask], 1e-20)) + _log_nb2_pmf(
            Y[pos_mask], mu[pos_mask], disp
        )
        return float(-ll.sum())

    def neg_loglik_obs(theta: np.ndarray) -> np.ndarray:
        beta = theta[:k_count]
        gamma = theta[k_count : k_count + k_inflate]
        log_alpha = theta[-1]
        disp = np.exp(np.clip(log_alpha, -10, 10))
        mu = np.exp(np.clip(X_count @ beta, -20, 20))
        pi = _logit(X_inflate @ gamma)
        zero_mask = Y == 0
        ll = np.zeros(n)
        nb_zero = _log_nb2_pmf(np.zeros_like(mu[zero_mask]), mu[zero_mask], disp)
        ll[zero_mask] = np.log(
            np.maximum(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(nb_zero), 1e-20)
        )
        pos_mask = ~zero_mask
        ll[pos_mask] = np.log(np.maximum(1 - pi[pos_mask], 1e-20)) + _log_nb2_pmf(
            Y[pos_mask], mu[pos_mask], disp
        )
        return _as_float_array(ll)

    # Initial values
    beta0 = np.linalg.lstsq(X_count, np.log(Y + 1), rcond=None)[0]
    gamma0 = np.zeros(k_inflate)
    log_alpha0 = np.array([0.0])  # alpha = 1 initial guess
    theta0 = np.concatenate([beta0, gamma0, log_alpha0])

    result = optimize.minimize(
        neg_loglik,
        theta0,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": tol},
    )

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        """Per-observation ZINB (NB2) log-likelihood; complex-step safe."""
        mu = np.exp(X_count @ theta[:k_count])
        zg = X_inflate @ theta[k_count : k_count + k_inflate]
        log_pi, log_1mpi = _log_expit(zg), _log_expit(-zg)
        m = np.exp(-theta[-1])  # 1 / alpha
        log_nb_zero = m * np.log(m / (m + mu))
        log_nb = (
            special.loggamma(Y + m)
            - special.loggamma(m)
            - special.gammaln(Y + 1)
            + log_nb_zero
            + Y * np.log(mu / (m + mu))
        )
        zero = _logaddexp(log_pi, log_1mpi + log_nb_zero)
        return np.where(Y == 0, zero, log_1mpi + log_nb)

    # Exact Newton steps from the BFGS solution; the same call returns the
    # observed information and per-observation scores by complex-step
    # differentiation (see zip_model).
    theta_hat, score_obs, H, _ = ml_newton_polish(obs_loglik, _as_float_array(result.x))
    beta_hat = theta_hat[:k_count]
    gamma_hat = theta_hat[k_count : k_count + k_inflate]
    alpha_hat = float(np.exp(theta_hat[-1]))
    ll_zinb = float(np.sum(obs_loglik(theta_hat)))

    # Standard errors
    clusters = None
    if robust == "cluster":
        if data is None:
            raise ValueError("`data` must be provided for clustered SEs.")
        clusters = (
            df[cluster].values
            if cluster in df.columns
            else data.loc[df.index, cluster].values
        )
        require_two_clusters(clusters, "zip_model / zinb / hurdle")
    se = se_from_vcov(
        ml_vcov(inverse_information(H), score_obs, kind=robust, clusters=clusters)
    )

    # Vuong test: ZINB vs plain NB
    mu_hat = np.exp(np.clip(X_count @ beta_hat, -20, 20))
    # The comparison model is the NB2 regression fitted on its own (see
    # zip_model).
    from ._negbin import negbin_joint
    from .count import _nb2_fit

    ll_zinb_obs = np.real(obs_loglik(theta_hat))
    b_nb, _, a_nb, _, _ = _nb2_fit(Y, X_count)
    nb = negbin_joint(Y, X_count, None, b_nb, a_nb, nb2=True)
    ll_nb_obs = _log_nb2_pmf(Y, nb.mu, float(nb.dispersion))
    vuong = _vuong_test(ll_zinb_obs, ll_nb_obs)

    # Predicted values
    pi_hat = _logit(X_inflate @ gamma_hat)
    pred_structural_zero = pi_hat
    pred_count = mu_hat
    pred_overall = (1 - pi_hat) * mu_hat

    # Assemble
    all_names = count_names + inflate_names + ["ln_alpha"]
    params = pd.Series(theta_hat, index=all_names)
    std_errors = pd.Series(se, index=all_names)
    _warn_flat_terms(std_errors, "zinb")

    model_info = {
        "model_type": "zinb",
        "method": "Zero-Inflated Negative Binomial (MLE)",
        "dependent_var": dep_var,
        "ll": ll_zinb,
        "aic": -2 * ll_zinb + 2 * k_total,
        "bic": -2 * ll_zinb + np.log(n) * k_total,
        "alpha_dispersion": float(alpha_hat),
        "vuong_stat": vuong["vuong_stat"],
        "vuong_p": vuong["vuong_p"],
        "converged": robust_convergence(result)[0],
        "robust": robust if cluster is None else f"cluster({cluster})",
        "n_zeros": int((Y == 0).sum()),
        "pct_zeros": float((Y == 0).mean() * 100),
        "omitted": list(df.attrs.get("omitted", [])),
    }

    data_info = {
        "n_obs": n,
        "nobs": n,
        "y": Y,
        "llobs": ll_zinb_obs,
        "dependent_var": dep_var,
        "df_resid": n - k_total,
        # Likelihood-based: z / chi2 inference, as Stata's zip / zinb.
        "inference": "z",
        "k_count": k_count,
        "k_inflate": k_inflate,
        "count_names": count_names,
        "inflate_names": inflate_names,
    }

    fitted_arrays = {
        "predicted_structural_zero": pred_structural_zero,
        "predicted_count": pred_count,
        "predicted_overall": pred_overall,
    }
    data_info.update(fitted_arrays)
    diagnostics = _FittedArraysDict(
        {
            "alpha_dispersion": float(alpha_hat),
            "vuong_stat": vuong["vuong_stat"],
            "vuong_p": vuong["vuong_p"],
            "ll": ll_zinb,
            "aic": model_info["aic"],
            "bic": model_info["bic"],
            # the same three under the names the other count models use
            "Log-Likelihood": ll_zinb,
            "AIC": model_info["aic"],
            "BIC": model_info["bic"],
        },
        fitted_arrays,
    )

    model_info["alpha"] = alpha
    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _compute_zi_score_obs(
    neg_loglik_obs_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    Y: np.ndarray,
    X_count: np.ndarray,
    X_inflate: np.ndarray,
    k_count: int,
    k_inflate: int,
    n: int,
    nb: bool = False,
) -> np.ndarray:
    """Numerical per-observation score for ZI models."""
    k_total = len(theta)
    eps = 1e-5
    score = np.zeros((n, k_total))

    for j in range(k_total):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[j] += eps
        theta_m[j] -= eps
        ll_p = neg_loglik_obs_fn(theta_p)
        ll_m = neg_loglik_obs_fn(theta_m)
        score[:, j] = (ll_p - ll_m) / (2 * eps)

    return _as_float_array(score)


# ===================================================================
# Hurdle Model
# ===================================================================


@accepts_aliases(vce="robust")
@markout_clusters
def hurdle(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    count_model: str = "poisson",
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Hurdle (two-part) model for count data.

    Part 1 (binary): logit model for P(Y > 0).
    Part 2 (count):  truncated-at-zero Poisson or Negative Binomial for
                     the distribution of Y | Y > 0.

    Unlike zero-inflated models, ALL zeros come from the binary process.

    Equivalent to R's ``pscl::hurdle()``.

    Parameters
    ----------
    formula : str, optional
        Patsy-style formula.
    data : pd.DataFrame
        Dataset.
    y : str, optional
        Dependent variable name.
    x : list of str, optional
        Regressors (used for both hurdle and count parts).
    count_model : str, default "poisson"
        Count distribution: "poisson" or "negbin".
    robust : str, default "nonrobust"
    cluster : str, optional
    maxiter : int, default 200
    tol : float, default 1e-8
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
    >>> age = rng.normal(0, 1, n)
    >>> visits = rng.poisson(np.exp(0.5 + 0.3 * age))
    >>> visits[rng.random(n) < 0.3] = 0  # excess zeros below the hurdle
    >>> df = pd.DataFrame({'visits': visits, 'age': age})
    >>> result = sp.hurdle(data=df, y='visits', x=['age'],
    ...                    count_model='negbin')
    >>> print(result.summary())  # doctest: +SKIP
    >>> bool(result.model_info['model_type'] == 'hurdle')
    True

    Notes
    -----
    The hurdle log-likelihood decomposes as:

    .. math::
        \\ell = \\sum_{y_i=0} \\log(1-p_i) + \\sum_{y_i>0} [\\log p_i
        + \\log f(y_i|\\mu_i) - \\log(1 - f(0|\\mu_i))]

    where p_i = Λ(x_i'δ) is the hurdle probability.

    See Mullahy (1986, *Journal of Econometrics*).
    """
    robust, cluster = _parse_se(robust, cluster, "hurdle")
    if data is None:
        raise ValueError("`data` must be provided.")
    if formula is not None:
        parsed = parse_formula(formula)
        dep_var = str(parsed["dependent"])
        x_vars = [str(name) for name in parsed["exogenous"]]
    else:
        if y is None or x is None:
            raise ValueError("Provide either `formula` or both `y` and `x`.")
        dep_var = y
        x_vars = list(x)

    if str(count_model).lower() not in ("poisson", "negbin", "nb", "nbreg"):
        raise MethodIncompatibility(
            f"hurdle: unknown count_model={count_model!r}. Choose 'poisson' "
            "or 'negbin'.",
            recovery_hint="Use count_model='poisson' or count_model='negbin'.",
            diagnostics={"count_model": count_model, "valid": ["poisson", "negbin"]},
        )
    all_vars = list(set([dep_var] + x_vars))
    _require_plain_columns(data, [dep_var] + x_vars, "hurdle")
    df = data[all_vars].dropna()

    Y = df[dep_var].values.astype(float)
    if np.any(Y < 0) or not np.all(Y == Y.astype(int)):
        raise ValueError("Dependent variable must contain non-negative integers.")
    Y = Y.astype(int)

    X = np.column_stack(
        [np.ones(len(df))] + [df[v].values.astype(float) for v in x_vars]
    )
    var_names = ["const"] + x_vars
    X, var_names, collinear_omitted = _omit_collinear(X, var_names, "hurdle")
    hurdle_names = ["hurdle_const"] + [f"hurdle_{v}" for v in var_names[1:]]
    n, k = X.shape

    zero_mask = Y == 0
    pos_mask = ~zero_mask
    Y_pos = Y[pos_mask]
    X_pos = X[pos_mask]
    # The count equation is a zero-truncated model on the positive
    # outcomes only. With none of them it has no data (it used to be
    # reported as coefficients 0 with standard errors 0), and a regressor
    # that does not vary independently among them has no coefficient there.
    if not pos_mask.any():
        raise DataInsufficient(
            f"hurdle: the outcome '{dep_var}' has no positive counts, so the "
            "count equation is not estimable.",
            recovery_hint=(
                "Check the outcome; with zeros only there is nothing above "
                "the hurdle to model."
            ),
            diagnostics={"n_obs": int(n), "n_positive": 0},
        )
    from ..core._collinear import independent_columns

    _, _pos_dependent = independent_columns(X_pos, var_names)
    if _pos_dependent:
        raise DataInsufficient(
            "hurdle: among the positive outcomes the regressor(s) "
            f"{[o['variable'] for o in _pos_dependent]} are collinear with "
            "the others, so their count-equation coefficients are not "
            "identified.",
            recovery_hint=(
                "Drop those regressors, or check that there are more "
                "positive outcomes than count-equation coefficients."
            ),
            diagnostics={
                "n_positive": int(pos_mask.sum()),
                "dependent": _pos_dependent,
            },
        )

    use_negbin = count_model.lower() in ("negbin", "nb", "nbreg")
    # k_hurdle params + k_count params (+ 1 if negbin for log_alpha)
    k_hurdle = k
    k_count = k
    k_total = k_hurdle + k_count + (1 if use_negbin else 0)

    def neg_loglik(theta: np.ndarray) -> float:
        delta = theta[:k_hurdle]
        beta = theta[k_hurdle : k_hurdle + k_count]

        p = _logit(X @ delta)  # P(Y > 0)
        mu = np.exp(np.clip(X @ beta, -20, 20))

        # Part 1: binary
        ll_binary = np.zeros(n)
        ll_binary[zero_mask] = np.log(np.maximum(1 - p[zero_mask], 1e-20))
        ll_binary[pos_mask] = np.log(np.maximum(p[pos_mask], 1e-20))

        # Part 2: truncated count for positive obs
        if use_negbin:
            disp = np.exp(np.clip(theta[-1], -10, 10))
            log_f = _log_nb2_pmf(Y_pos, mu[pos_mask], disp)
            log_f0 = _log_nb2_pmf(np.zeros(pos_mask.sum()), mu[pos_mask], disp)
        else:
            log_f = _log_poisson_pmf(Y_pos, mu[pos_mask])
            log_f0 = -mu[pos_mask]  # log P(Y=0|mu) for Poisson

        # Truncated: f(y) / (1 - f(0))
        ll_count = log_f - np.log(np.maximum(1 - np.exp(log_f0), 1e-20))

        total = ll_binary.sum() + ll_count.sum()
        return float(-total)

    def neg_loglik_obs(theta: np.ndarray) -> np.ndarray:
        delta = theta[:k_hurdle]
        beta = theta[k_hurdle : k_hurdle + k_count]
        p = _logit(X @ delta)
        mu = np.exp(np.clip(X @ beta, -20, 20))

        ll = np.zeros(n)
        ll[zero_mask] = np.log(np.maximum(1 - p[zero_mask], 1e-20))

        if use_negbin:
            disp = np.exp(np.clip(theta[-1], -10, 10))
            log_f = _log_nb2_pmf(Y[pos_mask], mu[pos_mask], disp)
            log_f0 = _log_nb2_pmf(np.zeros(pos_mask.sum()), mu[pos_mask], disp)
        else:
            log_f = _log_poisson_pmf(Y[pos_mask], mu[pos_mask])
            log_f0 = -mu[pos_mask]

        ll[pos_mask] = (
            np.log(np.maximum(p[pos_mask], 1e-20))
            + log_f
            - np.log(np.maximum(1 - np.exp(log_f0), 1e-20))
        )
        return _as_float_array(ll)

    # Initial values
    delta0 = np.zeros(k_hurdle)
    beta0 = np.linalg.lstsq(X_pos, np.log(Y_pos), rcond=None)[0]
    if use_negbin:
        theta0 = np.concatenate([delta0, beta0, [0.0]])
    else:
        theta0 = np.concatenate([delta0, beta0])

    result = optimize.minimize(
        neg_loglik,
        theta0,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": tol},
    )

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        """Per-observation hurdle log-likelihood; complex-step safe."""
        zd = X @ theta[:k_hurdle]  # index of P(Y > 0)
        mu = np.exp(X @ theta[k_hurdle : k_hurdle + k_count])
        if use_negbin:
            m = np.exp(-theta[-1])  # 1 / alpha
            log_f0 = m * np.log(m / (m + mu))
            log_f = (
                special.loggamma(Y + m)
                - special.loggamma(m)
                - special.gammaln(Y + 1)
                + log_f0
                + Y * np.log(mu / (m + mu))
            )
        else:
            log_f0 = -mu
            log_f = Y * np.log(mu) - mu - special.gammaln(Y + 1)
        # Zero-truncated count density f(y) / (1 - f(0)) above the hurdle.
        positive = _log_expit(zd) + log_f - np.log(-np.expm1(log_f0))
        return np.where(Y == 0, _log_expit(-zd), positive)

    # Exact Newton steps from the BFGS solution; the same call returns the
    # observed information and per-observation scores by complex-step
    # differentiation (see zip_model).
    theta_hat, score_obs, H, _ = ml_newton_polish(obs_loglik, _as_float_array(result.x))
    delta_hat = theta_hat[:k_hurdle]
    beta_hat = theta_hat[k_hurdle : k_hurdle + k_count]
    ll_hurdle = float(np.sum(obs_loglik(theta_hat)))

    # Standard errors
    clusters = None
    if robust == "cluster":
        clusters = (
            df[cluster].values
            if cluster in df.columns
            else data.loc[df.index, cluster].values
        )
        require_two_clusters(clusters, "zip_model / zinb / hurdle")
    se = se_from_vcov(
        ml_vcov(inverse_information(H), score_obs, kind=robust, clusters=clusters)
    )

    # Predicted values
    p_hat = _logit(X @ delta_hat)
    mu_hat = np.exp(np.clip(X @ beta_hat, -20, 20))
    if use_negbin:
        alpha_hat = float(np.exp(theta_hat[-1]))
        f0 = np.exp(_log_nb2_pmf(np.zeros(n), mu_hat, alpha_hat))
    else:
        alpha_hat = None
        f0 = np.exp(-mu_hat)

    pred_hurdle_prob = p_hat  # P(Y > 0)
    # E[Y] = P(Y>0) * E[Y | Y>0] = p * mu / (1 - f(0))
    pred_overall = p_hat * mu_hat / np.maximum(1 - f0, 1e-20)

    # Assemble names
    all_names = hurdle_names + count_names_from_vars(var_names)
    if use_negbin:
        all_names.append("ln_alpha")

    params = pd.Series(theta_hat, index=all_names)
    std_errors = pd.Series(se, index=all_names)
    _warn_flat_terms(std_errors, "hurdle")

    model_info = {
        "model_type": "hurdle",
        "method": f"Hurdle ({count_model.title()}, MLE)",
        "dependent_var": dep_var,
        "count_dist": count_model,
        "ll": ll_hurdle,
        "aic": -2 * ll_hurdle + 2 * k_total,
        "bic": -2 * ll_hurdle + np.log(n) * k_total,
        "converged": robust_convergence(result)[0],
        "robust": robust if cluster is None else f"cluster({cluster})",
        "n_zeros": int(zero_mask.sum()),
        "pct_zeros": float(zero_mask.mean() * 100),
        "omitted": collinear_omitted,
    }
    if use_negbin:
        assert alpha_hat is not None
        model_info["alpha_dispersion"] = float(alpha_hat)

    data_info = {
        "n_obs": n,
        "nobs": n,
        "y": Y,
        "llobs": np.real(obs_loglik(theta_hat)),
        "dependent_var": dep_var,
        "df_resid": n - k_total,
        # Likelihood-based: z / chi2 inference.
        "inference": "z",
        "k_hurdle": k_hurdle,
        "k_count": k_count,
        "hurdle_names": hurdle_names,
        "count_names": count_names_from_vars(var_names),
    }

    fitted_arrays = {
        "predicted_hurdle_prob": pred_hurdle_prob,
        "predicted_count_mean": mu_hat,
        "predicted_overall": pred_overall,
    }
    data_info.update(fitted_arrays)
    diagnostics = _FittedArraysDict(
        {
            "ll": ll_hurdle,
            "aic": model_info["aic"],
            "bic": model_info["bic"],
            # the same three under the names the other count models use
            "Log-Likelihood": ll_hurdle,
            "AIC": model_info["aic"],
            "BIC": model_info["bic"],
        },
        fitted_arrays,
    )
    if use_negbin:
        assert alpha_hat is not None
        diagnostics["alpha_dispersion"] = float(alpha_hat)

    model_info["alpha"] = alpha
    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def count_names_from_vars(var_names: List[str]) -> List[str]:
    """Prefix count-equation variable names."""
    return ["count_" + v for v in var_names]


def _compute_hurdle_score_obs(
    neg_loglik_obs_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    n: int,
) -> np.ndarray:
    """Numerical per-observation score for hurdle models."""
    k_total = len(theta)
    eps = 1e-5
    score = np.zeros((n, k_total))

    for j in range(k_total):
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[j] += eps
        theta_m[j] -= eps
        ll_p = neg_loglik_obs_fn(theta_p)
        ll_m = neg_loglik_obs_fn(theta_m)
        score[:, j] = (ll_p - ll_m) / (2 * eps)

    return _as_float_array(score)
