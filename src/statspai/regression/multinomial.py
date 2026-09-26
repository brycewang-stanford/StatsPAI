"""
Multinomial and ordered discrete-choice models.

Implements:
- Multinomial Logit (McFadden, 1974)
- Ordered Logit / Probit (McKelvey & Zavoina, 1975)
- Conditional Logit (McFadden, 1973)

All models estimated via native numpy/scipy MLE with analytic or
BFGS-approximated Hessians, robust and clustered standard errors.

References
----------
McFadden, D. (1974).
"Conditional Logit Analysis of Qualitative Choice Behavior."
*Frontiers in Econometrics*, 105-142.

McKelvey, R.D. & Zavoina, W. (1975).
"A Statistical Model for the Analysis of Ordinal Level Dependent Variables."
*Journal of Mathematical Sociology*, 4(1), 103-120. [@mckelvey1975statistical]

McFadden, D. (1973).
"Conditional Logit Analysis of Qualitative Choice Behavior."
*Frontiers in Econometrics*, 105-142.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._aliases import accepts_aliases
from ..core._vcov_spec import markout_clusters
from ..core.results import EconometricResults
from ..core.utils import parse_formula
from ..exceptions import MethodIncompatibility
from ._optim_helpers import robust_convergence

LinkFunc = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _parse_inputs(
    formula: Optional[str],
    data: Optional[pd.DataFrame],
    y: Optional[str],
    x: Optional[List[str]],
) -> Tuple[str, List[str]]:
    """Resolve formula / y+x inputs into variable names."""
    if formula is not None:
        parsed = parse_formula(formula)
        y_name = str(parsed["dependent"])
        x_names = [str(name) for name in parsed["exogenous"]]
    else:
        if y is None or x is None:
            raise ValueError("Provide either 'formula' or both 'y' and 'x'.")
        y_name = y
        x_names = list(x)
    return y_name, x_names


def _build_matrices(
    data: Optional[pd.DataFrame],
    y_name: str,
    x_names: List[str],
    add_constant: bool = True,
    extra_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """Return Y (1-d int array), X (n x k float), clean DataFrame."""
    if data is None:
        raise ValueError("'data' must be provided.")
    cols = [y_name] + x_names
    if extra_cols:
        cols = cols + [c for c in extra_cols if c not in cols]
    df = data[cols].dropna().copy()
    Y = df[y_name].values
    if add_constant:
        X = np.column_stack(
            [np.ones(len(df))] + [df[v].values.astype(float) for v in x_names]
        )
        var_names = ["_cons"] + x_names
    else:
        X = np.column_stack([df[v].values.astype(float) for v in x_names])
        var_names = list(x_names)
    return Y, X, df, var_names


def _softmax(Z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax, Z is (n, J)."""
    Z_shift = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return _as_float_array(exp_Z / exp_Z.sum(axis=1, keepdims=True))


#: SE kinds the multinomial / ordered / conditional logit family implements.
_SE_KINDS = ("nonrobust", "robust", "hc0", "hc1", "cluster")


def _parse_se(robust: Any, cluster: Any, function: str) -> Tuple[str, Any]:
    """Resolve ``robust=`` / ``cluster=`` through the shared Stata grammar."""
    from ..core._vcov_spec import parse_se_request

    req = parse_se_request(robust, cluster, function=function, supported=_SE_KINDS)
    return req.kind, req.cluster


def _compute_se(
    score_i: np.ndarray,
    H_inv: np.ndarray,
    kind: str,
    cluster_vals: Optional[np.ndarray],
) -> np.ndarray:
    """Standard errors for a canonical SE ``kind`` under Stata's ML conventions.

    ``H_inv`` is the inverse observed information. ``vce(robust)`` carries
    N/(N-1) and ``vce(cluster)`` G/(G-1) only (see ``core._vcov.ml_vcov``);
    the cluster factor used to be the regress-family G/(G-1)*(N-1)/(N-K).
    """
    from ..core._vcov import ml_vcov

    V = ml_vcov(H_inv, score_i, kind=kind, clusters=cluster_vals)
    return _as_float_array(np.sqrt(np.maximum(np.diag(V), 1e-20)))


def _ordered_logit_cdf(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -500, 500)
    return _as_float_array(1.0 / (1.0 + np.exp(-z_clip)))


def _ordered_logit_pdf(z: np.ndarray) -> np.ndarray:
    p = _ordered_logit_cdf(z)
    return _as_float_array(p * (1.0 - p))


def _ordered_probit_cdf(z: np.ndarray) -> np.ndarray:
    return _as_float_array(stats.norm.cdf(z))


def _ordered_probit_pdf(z: np.ndarray) -> np.ndarray:
    return _as_float_array(stats.norm.pdf(z))


# ====================================================================
# Multinomial Logit
# ====================================================================


@accepts_aliases(vce="robust")
@markout_clusters
def mlogit(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    base: int = 0,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    rrr: bool = False,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Multinomial logit for J > 2 unordered categories via MLE.

    Equivalent to Stata's ``mlogit y x, base(0)`` or ``mlogit y x, rrr``.

    Parameters
    ----------
    formula : str, optional
        Formula ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        Data.
    y : str, optional
        Dependent variable (categorical, integer-coded).
    x : list of str, optional
        Regressors.
    base : int, default 0
        Base / reference category (index into sorted unique values).
    robust : str, default "nonrobust"
        ``"robust"`` / ``"HC1"`` for Huber-White sandwich SE.
    cluster : str, optional
        Cluster variable for clustered SE.
    rrr : bool, default False
        Report Relative Risk Ratios (exp(beta)) instead of coefficients.
    maxiter : int, default 100
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> price = rng.normal(0, 1, n)
    >>> income = rng.normal(0, 1, n)
    >>> eta1 = 0.5 * price - 0.3 * income
    >>> eta2 = -0.4 * price + 0.6 * income
    >>> exps = np.column_stack([np.ones(n), np.exp(eta1), np.exp(eta2)])
    >>> P = exps / exps.sum(axis=1, keepdims=True)
    >>> choice = np.array([rng.choice(3, p=P[i]) for i in range(n)])
    >>> df = pd.DataFrame({'choice': choice, 'price': price, 'income': income})
    >>> result = sp.mlogit('choice ~ price + income', data=df, base=0)
    >>> print(result.summary())  # doctest: +SKIP
    >>> rrr = sp.mlogit(data=df, y='choice', x=['price', 'income'], rrr=True)
    >>> bool(rrr.params is not None)
    True

    Notes
    -----
    Softmax parameterisation: β_j for each category j != base.

    .. math::
        P(Y_i = j | X_i) = \\frac{\\exp(X_i' \\beta_j)}
        {\\sum_{k} \\exp(X_i' \\beta_k)},
        \\quad \\beta_{\\text{base}} = 0.

    McFadden pseudo-R^2 = 1 - LL / LL_0.

    References
    ----------
    mcfadden1974conditional
    """
    # --- Parse inputs ---
    robust, cluster = _parse_se(robust, cluster, "mlogit")
    y_name, x_names = _parse_inputs(formula, data, y, x)
    extra = [c for c in [cluster] if c]
    Y_raw, X, df, var_names = _build_matrices(data, y_name, x_names, extra_cols=extra)
    n, k = X.shape

    categories = np.sort(np.unique(Y_raw))
    J = len(categories)
    if J < 3:
        raise ValueError(f"mlogit requires J >= 3 categories, got {J}.")
    if base < 0 or base >= J:
        raise ValueError(f"base must be in [0, {J - 1}], got {base}.")

    cat_map = {c: j for j, c in enumerate(categories)}
    Y_idx = np.array([cat_map[v] for v in Y_raw])

    # One-hot
    Y_oh = np.zeros((n, J))
    Y_oh[np.arange(n), Y_idx] = 1.0

    # Non-base category indices
    non_base = [j for j in range(J) if j != base]
    n_params = (J - 1) * k

    cluster_vals = df[cluster].values if cluster else None

    # --- Log-likelihood ---
    def _probs(theta: np.ndarray) -> np.ndarray:
        """Return (n, J) probability matrix."""
        V = np.zeros((n, J))
        for idx, j in enumerate(non_base):
            V[:, j] = X @ theta[idx * k : (idx + 1) * k]
        return _softmax(V)

    def neg_loglik(theta: np.ndarray) -> float:
        P = _probs(theta)
        ll = np.sum(Y_oh * np.log(np.maximum(P, 1e-300)))
        return float(-ll)

    def score(theta: np.ndarray) -> np.ndarray:
        """Gradient (vectorised)."""
        P = _probs(theta)
        R = Y_oh - P  # (n, J)
        grad = np.zeros(n_params)
        for idx, j in enumerate(non_base):
            grad[idx * k : (idx + 1) * k] = X.T @ R[:, j]
        return _as_float_array(-grad)

    # --- Optimise ---
    theta0 = np.zeros(n_params)
    res = optimize.minimize(
        neg_loglik,
        theta0,
        jac=score,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": tol},
    )

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        """Per-observation log-likelihood; complex-step safe."""
        V = np.zeros((n, J), dtype=np.result_type(theta, float))
        for idx, j in enumerate(non_base):
            V[:, j] = X @ theta[idx * k : (idx + 1) * k]
        shift = np.max(V.real, axis=1, keepdims=True)
        log_denominator = shift[:, 0] + np.log(np.sum(np.exp(V - shift), axis=1))
        return V[np.arange(n), Y_idx] - log_denominator

    # Exact Newton steps from the BFGS solution; the same call returns the
    # observed information and per-observation scores by complex-step
    # differentiation. The numerical Hessian of the total log-likelihood used
    # before (BFGS ``hess_inv`` before that) carried ~1e-5 relative error.
    from ._optim_helpers import inverse_information, ml_newton_polish

    theta_hat, S_obs, H, _ = ml_newton_polish(obs_loglik, _as_float_array(res.x))
    ll = float(np.sum(obs_loglik(theta_hat)))

    # Null model log-likelihood (intercept only => equal probs)
    freq = np.array([np.sum(Y_idx == j) for j in range(J)]) / n
    ll_0 = float(np.sum(Y_oh * np.log(np.maximum(freq[np.newaxis, :], 1e-300))))

    pseudo_r2 = 1.0 - ll / ll_0
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + np.log(n) * n_params

    # --- Standard errors ---
    H_inv = _as_float_array(inverse_information(H))
    se = _compute_se(S_obs, H_inv, robust, cluster_vals)

    # --- Build results ---
    P_hat = _probs(theta_hat)
    param_names: List[str] = []
    coefs: List[float] = []
    ses: List[float] = []
    for idx, j in enumerate(non_base):
        cat_label = categories[j]
        beta_j = theta_hat[idx * k : (idx + 1) * k]
        se_j = se[idx * k : (idx + 1) * k]
        for vi, vn in enumerate(var_names):
            param_names.append(f"[{cat_label}]{vn}")
            coefs.append(float(beta_j[vi]))
            ses.append(float(se_j[vi]))

    coef_arr = np.asarray(coefs, dtype=float)
    se_arr = np.asarray(ses, dtype=float)

    if rrr:
        # Relative risk ratios: exp(beta), delta-method SE
        rrr_vals = np.exp(coef_arr)
        rrr_se = rrr_vals * se_arr
        params_series = pd.Series(rrr_vals, index=param_names)
        se_series = pd.Series(rrr_se, index=param_names)
    else:
        params_series = pd.Series(coef_arr, index=param_names)
        se_series = pd.Series(se_arr, index=param_names)

    # --- Marginal effects (average) ---
    # dP_j/dx = P_j * (beta_j - sum_k P_k * beta_k)
    me_dict: Dict[Any, Dict[str, float]] = {}
    betas_all = np.zeros((J, k))
    for idx, j in enumerate(non_base):
        betas_all[j] = theta_hat[idx * k : (idx + 1) * k]

    beta_bar = np.einsum("nj,jk->nk", P_hat, betas_all)  # (n, k)
    for idx, j in enumerate(non_base):
        me_j = (P_hat[:, [j]] * (betas_all[j][np.newaxis, :] - beta_bar)).mean(axis=0)
        me_dict[categories[j]] = {
            name: float(value) for name, value in zip(var_names, me_j)
        }
    # base category
    me_base = (P_hat[:, [base]] * (betas_all[base][np.newaxis, :] - beta_bar)).mean(
        axis=0
    )
    me_dict[categories[base]] = {
        name: float(value) for name, value in zip(var_names, me_base)
    }

    # --- IIA test (Hausman-McFadden) ---
    iia_tests: Dict[Any, Dict[str, Any]] = {}
    iia_skipped: List[str] = []
    for drop_j in non_base:
        # Estimate restricted model omitting category drop_j
        restricted_cats = [j for j in range(J) if j != drop_j]
        mask = np.isin(Y_idx, restricted_cats)
        if mask.sum() < k * (len(restricted_cats) - 1) + 10:
            continue

        Y_r = Y_idx[mask]
        X_r = X[mask]
        Y_oh_r = np.zeros((mask.sum(), J))
        Y_oh_r[np.arange(mask.sum()), Y_r] = 1.0
        n_r = int(mask.sum())
        non_base_r = [j for j in restricted_cats if j != base]
        n_params_r = len(non_base_r) * k

        def neg_ll_r(
            theta_r: np.ndarray,
            nb: List[int] = non_base_r,
        ) -> float:
            V_r = np.zeros((n_r, J))
            for idx2, j2 in enumerate(nb):
                V_r[:, j2] = X_r @ theta_r[idx2 * k : (idx2 + 1) * k]
            P_r = _softmax(V_r)
            # Only include restricted cats in likelihood
            ll_r = 0.0
            for j2 in restricted_cats:
                ll_r += np.sum(Y_oh_r[:, j2] * np.log(np.maximum(P_r[:, j2], 1e-300)))
            return float(-ll_r)

        theta0_r = np.zeros(n_params_r)
        # Warm-start from full model
        for idx2, j2 in enumerate(non_base_r):
            full_idx = non_base.index(j2)
            theta0_r[idx2 * k : (idx2 + 1) * k] = theta_hat[
                full_idx * k : (full_idx + 1) * k
            ]

        res_r = optimize.minimize(
            neg_ll_r, theta0_r, method="BFGS", options={"maxiter": maxiter, "gtol": tol}
        )

        # Hausman statistic: (b_r - b_f)' [V_r - V_f]^{-1} (b_r - b_f)
        # Simplified: use the restricted params corresponding to non_base_r
        b_r = _as_float_array(res_r.x)
        b_f = np.concatenate(
            [
                theta_hat[non_base.index(j2) * k : (non_base.index(j2) + 1) * k]
                for j2 in non_base_r
            ]
        )
        diff = b_r - b_f
        df_test = len(diff)
        try:
            V_r_mat = (
                np.asarray(res_r.hess_inv)
                if hasattr(res_r, "hess_inv")
                else np.eye(n_params_r)
            )
            V_f_sub = np.zeros((n_params_r, n_params_r))
            for i1, j1 in enumerate(non_base_r):
                for i2, j2 in enumerate(non_base_r):
                    fi1 = non_base.index(j1)
                    fi2 = non_base.index(j2)
                    V_f_sub[i1 * k : (i1 + 1) * k, i2 * k : (i2 + 1) * k] = H_inv[
                        fi1 * k : (fi1 + 1) * k, fi2 * k : (fi2 + 1) * k
                    ]
            V_diff = V_r_mat - V_f_sub
            eigvals = np.linalg.eigvalsh(V_diff)
            if np.all(eigvals > -1e-6):
                V_diff = V_diff + np.eye(n_params_r) * max(0, -eigvals.min() + 1e-8)
                chi2 = float(diff @ np.linalg.solve(V_diff, diff))
                chi2 = max(chi2, 0.0)
                p_iia = float(stats.chi2.sf(chi2, df_test))
                iia_tests[categories[drop_j]] = {
                    "chi2": chi2,
                    "df": df_test,
                    "pvalue": p_iia,
                }
            else:
                # Hausman regularity (V_r - V_f PSD) failed — common in
                # finite samples. Record the skip instead of silently
                # omitting the test row (CLAUDE.md §7).
                iia_skipped.append(str(categories[drop_j]))
        except np.linalg.LinAlgError:
            iia_skipped.append(str(categories[drop_j]))

    if iia_skipped:
        warnings.warn(
            f"Multinomial IIA (Hausman-McFadden) test could not be computed "
            f"for category/categories {iia_skipped} (singular or non-PSD "
            f"variance difference). These categories are absent from "
            f"`iia_test`; see model_info['iia_skipped'].",
            RuntimeWarning,
            stacklevel=2,
        )

    model_info = {
        "model_type": "Multinomial Logit",
        "method": "MLE (softmax)",
        "base_category": categories[base],
        "n_categories": J,
        "categories": list(categories),
        "log_likelihood": float(ll),
        "log_likelihood_0": float(ll_0),
        "pseudo_r2": float(pseudo_r2),
        "aic": float(aic),
        "bic": float(bic),
        "converged": robust_convergence(res)[0],
        "rrr": rrr,
        "robust": robust if cluster is None else f"cluster({cluster})",
        "iia_skipped": iia_skipped,
    }

    data_info = {
        "dependent_var": y_name,
        "n_obs": n,
        "n_params": n_params,
        "df_resid": n - n_params,
        # Likelihood-based: z / chi2 inference, as Stata's mlogit / ologit.
        "inference": "z",
    }

    diagnostics = {
        "McFadden_pseudo_R2": float(pseudo_r2),
        "Log-Likelihood": float(ll),
        "Log-Likelihood_0": float(ll_0),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_obs": n,
    }

    model_info["alpha"] = alpha
    result = EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )

    # Attach extra attributes
    setattr(
        result,
        "predicted_probs",
        pd.DataFrame(P_hat, columns=categories, index=df.index),
    )
    setattr(result, "marginal_effects", me_dict)
    setattr(result, "iia_test", iia_tests)

    return result


# ====================================================================
# Ordered Logit / Probit
# ====================================================================


def _ordered_model(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    link: str = "logit",
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Internal engine for ordered logit / probit.

    Parameters
    ----------
    link : str
        ``"logit"`` or ``"probit"``.
    """
    robust, cluster = _parse_se(robust, cluster, f"o{link}")
    y_name, x_names = _parse_inputs(formula, data, y, x)
    extra = [c for c in [cluster] if c]
    Y_raw, X_no_const, df, _ = _build_matrices(
        data, y_name, x_names, add_constant=False, extra_cols=extra
    )
    n, k = X_no_const.shape
    var_names = list(x_names)

    categories = np.sort(np.unique(Y_raw))
    J = len(categories)
    if J < 3:
        raise ValueError(f"Ordered model requires J >= 3 categories, got {J}.")

    cat_map = {c: j for j, c in enumerate(categories)}
    Y_idx = np.array([cat_map[v] for v in Y_raw])

    n_cuts = J - 1  # cutpoints kappa_1 < ... < kappa_{J-1}
    n_params = k + n_cuts

    cluster_vals = df[cluster].values if cluster else None

    # CDF
    if link == "logit":
        cdf: LinkFunc = _ordered_logit_cdf
        pdf: LinkFunc = _ordered_logit_pdf
        link_label = "Ordered Logit"
    elif link == "probit":
        cdf = _ordered_probit_cdf
        pdf = _ordered_probit_pdf
        link_label = "Ordered Probit"
    else:
        raise ValueError("link must be 'logit' or 'probit'.")

    # Parameterisation: theta = [beta (k), delta (n_cuts)]
    # where kappa_1 = delta_1, kappa_j = kappa_{j-1} + exp(delta_j) for j >= 2
    # This ensures kappa is strictly increasing.

    def _unpack(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        beta = theta[:k]
        delta = theta[k:]
        kappa = np.empty(n_cuts)
        kappa[0] = delta[0]
        for j in range(1, n_cuts):
            kappa[j] = kappa[j - 1] + np.exp(delta[j])
        return beta, kappa

    def _cum_probs(beta: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        """Return (n, J+1) cumulative probabilities including 0 and 1 boundaries."""
        xb = X_no_const @ beta  # (n,)
        # P(Y <= j) = cdf(kappa_j - xb)
        cum = np.zeros((n, J + 1))
        cum[:, 0] = 0.0
        cum[:, J] = 1.0
        for j in range(n_cuts):
            cum[:, j + 1] = cdf(kappa[j] - xb)
        return _as_float_array(cum)

    def _cat_probs(beta: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        cum = _cum_probs(beta, kappa)
        P = np.diff(cum, axis=1)  # (n, J)
        return _as_float_array(np.maximum(P, 1e-300))

    def neg_loglik(theta: np.ndarray) -> float:
        beta, kappa = _unpack(theta)
        P = _cat_probs(beta, kappa)
        ll = np.sum(np.log(P[np.arange(n), Y_idx]))
        return float(-ll)

    # --- Initial values ---
    # Simple: beta = 0, cutpoints equally spaced
    freq_cum = np.cumsum([np.mean(Y_idx == j) for j in range(J)])
    kappa_init = np.zeros(n_cuts)
    for j in range(n_cuts):
        p = min(max(freq_cum[j], 0.01), 0.99)
        if link == "logit":
            kappa_init[j] = np.log(p / (1 - p))
        else:
            kappa_init[j] = stats.norm.ppf(p)

    delta_init = np.zeros(n_cuts)
    delta_init[0] = kappa_init[0]
    for j in range(1, n_cuts):
        delta_init[j] = np.log(max(kappa_init[j] - kappa_init[j - 1], 0.01))

    theta0 = np.concatenate([np.zeros(k), delta_init])

    # --- Optimise ---
    res = optimize.minimize(
        neg_loglik,
        theta0,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": tol},
    )

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        """Per-observation log-likelihood at (beta, delta); complex-step safe."""
        from scipy import special

        beta = theta[:k]
        delta = theta[k:]
        cuts = [delta[0]]
        for j in range(1, n_cuts):
            cuts.append(cuts[-1] + np.exp(delta[j]))
        xb = X_no_const @ beta
        if link == "logit":

            def log_cdf(z: np.ndarray) -> np.ndarray:
                return -np.log1p(np.exp(-z))

            def cdf_c(z: np.ndarray) -> np.ndarray:
                return 1.0 / (1.0 + np.exp(-z))

        else:
            log_cdf, cdf_c = special.log_ndtr, special.ndtr
        out = np.empty(n, dtype=np.result_type(theta, float))
        for j in range(J):
            m = Y_idx == j
            if not m.any():
                continue
            if j == 0:
                out[m] = log_cdf(cuts[0] - xb[m])
            elif j == n_cuts:
                out[m] = log_cdf(xb[m] - cuts[-1])
            else:
                out[m] = np.log(cdf_c(cuts[j] - xb[m]) - cdf_c(cuts[j - 1] - xb[m]))
        return out

    # Exact Newton steps from the BFGS solution; the same call returns the
    # observed information and per-observation scores by complex-step
    # differentiation, replacing a second-difference Hessian and a forward-
    # difference score that carried ~1e-6 to 1e-5 relative error.
    from ._optim_helpers import inverse_information, ml_newton_polish

    theta_hat, S_obs_exact, H_exact, _ = ml_newton_polish(
        obs_loglik, _as_float_array(res.x)
    )
    beta_hat, kappa_hat = _unpack(theta_hat)
    ll = float(np.sum(obs_loglik(theta_hat)))

    # Null model (beta=0, only cutpoints)
    def neg_loglik_null(delta: np.ndarray) -> float:
        theta_null = np.concatenate([np.zeros(k), delta])
        return neg_loglik(theta_null)

    res_null = optimize.minimize(
        neg_loglik_null,
        delta_init,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": tol},
    )
    ll_0 = float(-res_null.fun)

    pseudo_r2 = 1.0 - ll / ll_0
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + np.log(n) * n_params

    # --- Standard errors ---
    # Observed information and scores from the complex-step derivatives
    # above (BFGS ``hess_inv`` once inflated beta_x SEs by 26%, parity
    # finding #11; the numerical Hessian that replaced it carried ~1e-6).
    H_inv = _as_float_array(inverse_information(H_exact))
    se_all = _compute_se(S_obs_exact, H_inv, robust, cluster_vals)

    # Delta-method for cutpoints: kappa_j SE from delta SE
    # We report beta SE directly and kappa SE via Jacobian
    se_beta = se_all[:k]

    # Jacobian d kappa / d delta
    J_kd = np.zeros((n_cuts, n_cuts))
    J_kd[0, 0] = 1.0
    for j in range(1, n_cuts):
        J_kd[j, 0] = 1.0
        for m in range(1, j + 1):
            J_kd[j, m] = np.exp(theta_hat[k + m])

    V_delta = H_inv[k:, k:]
    V_kappa = J_kd @ V_delta @ J_kd.T
    se_kappa = _as_float_array(np.sqrt(np.maximum(np.diag(V_kappa), 1e-20)))

    # --- Predicted probabilities ---
    P_hat = _cat_probs(beta_hat, kappa_hat)

    # --- Marginal effects (average marginal effects) ---
    # For ordered model: dP(Y=j)/dx_m = [f(kappa_{j-1} - xb) - f(kappa_j - xb)] * beta_m
    xb = X_no_const @ beta_hat
    me_dict: Dict[Any, Dict[str, float]] = {}
    for j in range(J):
        if j > 0:
            f_lower = pdf(kappa_hat[j - 1] - xb)
        else:
            f_lower = np.zeros(n)
        if j < n_cuts:
            f_upper = pdf(kappa_hat[j] - xb)
        else:
            f_upper = np.zeros(n)
        me_j = np.mean(
            (f_lower - f_upper)[:, np.newaxis] * beta_hat[np.newaxis, :], axis=0
        )
        me_dict[categories[j]] = {
            name: float(value) for name, value in zip(var_names, me_j)
        }

    # --- Brant test (parallel regression assumption) ---
    # Compare J-1 binary logits to the constrained ordered model
    brant_test: Dict[str, Dict[str, Any]] = {}
    brant_skipped: List[str] = []
    brant_error = None
    try:
        chi2_total = 0.0
        df_total = 0
        for m in range(k):
            beta_binaries: List[float] = []
            se_binaries: List[float] = []
            for j in range(n_cuts):
                # Binary: Y <= j vs Y > j
                Y_bin = (Y_idx <= j).astype(float)
                X_bin = np.column_stack([np.ones(n), X_no_const])

                def neg_ll_bin(
                    b: np.ndarray,
                    yb: np.ndarray = Y_bin,
                    xb: np.ndarray = X_bin,
                ) -> float:
                    p = 1.0 / (1.0 + np.exp(-np.clip(xb @ b, -500, 500)))
                    ll_bin = yb * np.log(np.maximum(p, 1e-300))
                    ll_bin += (1 - yb) * np.log(np.maximum(1 - p, 1e-300))
                    return float(-np.sum(ll_bin))

                b0 = np.zeros(k + 1)
                res_bin = optimize.minimize(
                    neg_ll_bin, b0, method="BFGS", options={"maxiter": 50}
                )
                # beta for variable m is at index m+1 (skip intercept)
                beta_binaries.append(float(res_bin.x[m + 1]))
                if hasattr(res_bin, "hess_inv"):
                    h_bin = np.asarray(res_bin.hess_inv)
                    se_binaries.append(float(np.sqrt(max(h_bin[m + 1, m + 1], 1e-20))))
                else:
                    se_binaries.append(np.nan)

                beta_binary_arr = np.asarray(beta_binaries, dtype=float)
                se_binary_arr = np.asarray(se_binaries, dtype=float)

                # Test: all beta_j equal (Wald test)
                if n_cuts > 1 and np.all(np.isfinite(se_binary_arr)):
                    beta_mean = float(np.mean(beta_binary_arr))
                    chi2_m = float(
                        np.sum(((beta_binary_arr - beta_mean) / se_binary_arr) ** 2)
                    )
                    df_m = n_cuts - 1
                    p_m = float(stats.chi2.sf(chi2_m, df_m))
                    brant_test[var_names[m]] = {
                        "chi2": chi2_m,
                        "df": df_m,
                        "pvalue": p_m,
                    }
                chi2_total += chi2_m
                df_total += df_m
            else:
                brant_skipped.append(var_names[m])

        if df_total > 0:
            brant_test["_omnibus"] = {
                "chi2": float(chi2_total),
                "df": df_total,
                "pvalue": float(stats.chi2.sf(chi2_total, df_total)),
            }
    except Exception as exc:
        # Don't silently drop the whole parallel-regression diagnostic.
        brant_error = f"{type(exc).__name__}: {exc}"

    if brant_error is not None:
        warnings.warn(
            f"Brant parallel-regression test failed and is omitted from the "
            f"result ({brant_error}). See model_info['brant_error'].",
            RuntimeWarning,
            stacklevel=2,
        )
    elif brant_skipped:
        warnings.warn(
            f"Brant parallel-regression test skipped for {brant_skipped} "
            f"(non-finite binary-logit SE). These rows are absent from "
            f"`brant_test`; see model_info['brant_skipped'].",
            RuntimeWarning,
            stacklevel=2,
        )

    # --- Build results ---
    # Params: beta coefficients + cutpoints
    param_names = var_names + [f"/cut{j + 1}" for j in range(n_cuts)]
    all_coefs = np.concatenate([beta_hat, kappa_hat])
    all_se = np.concatenate([se_beta, se_kappa])

    params_series = pd.Series(all_coefs, index=param_names)
    se_series = pd.Series(all_se, index=param_names)

    model_info = {
        "model_type": link_label,
        "method": f"MLE ({link})",
        "n_categories": J,
        "categories": list(categories),
        "cutpoints": dict(
            zip([f"cut{j + 1}" for j in range(n_cuts)], kappa_hat.tolist())
        ),
        "log_likelihood": float(ll),
        "log_likelihood_0": float(ll_0),
        "pseudo_r2": float(pseudo_r2),
        "aic": float(aic),
        "bic": float(bic),
        "converged": robust_convergence(res)[0],
        "robust": robust if cluster is None else f"cluster({cluster})",
        "brant_skipped": brant_skipped,
        "brant_error": brant_error,
    }

    data_info = {
        "dependent_var": y_name,
        "n_obs": n,
        "n_params": n_params,
        "df_resid": n - n_params,
        # Likelihood-based: z / chi2 inference, as Stata's mlogit / ologit.
        "inference": "z",
    }

    diagnostics = {
        "McFadden_pseudo_R2": float(pseudo_r2),
        "Log-Likelihood": float(ll),
        "Log-Likelihood_0": float(ll_0),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_obs": n,
    }

    model_info["alpha"] = alpha
    result = EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )

    setattr(
        result,
        "predicted_probs",
        pd.DataFrame(P_hat, columns=categories, index=df.index),
    )
    setattr(result, "marginal_effects", me_dict)
    setattr(result, "brant_test", brant_test)
    setattr(result, "cutpoints", kappa_hat)

    return result


@accepts_aliases(vce="robust")
@markout_clusters
def ologit(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Ordered logit (proportional odds) model via MLE.

    Equivalent to Stata's ``ologit y x``.

    Parameters
    ----------
    formula : str, optional
        Formula ``"y ~ x1 + x2"``.
    data : pd.DataFrame
    y : str, optional
        Ordered categorical dependent variable.
    x : list of str, optional
    robust : str, default "nonrobust"
    cluster : str, optional
    maxiter : int, default 100
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        Coefficients (beta) and cutpoints (kappa).
        ``result.predicted_probs`` gives per-category probabilities.
        ``result.marginal_effects`` gives AME per category.
        ``result.brant_test`` gives the Brant parallel-lines test.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 300
    >>> income = rng.normal(0, 1, n)
    >>> age = rng.normal(0, 1, n)
    >>> latent = 0.8 * income + 0.4 * age + rng.logistic(0, 1, n)
    >>> satisfaction = np.digitize(latent, [-0.5, 0.8])  # ordered {0, 1, 2}
    >>> df = pd.DataFrame({'satisfaction': satisfaction,
    ...                    'income': income, 'age': age})
    >>> result = sp.ologit('satisfaction ~ income + age', data=df)
    >>> print(result.summary())  # doctest: +SKIP
    >>> bool('_omnibus' in result.brant_test)  # parallel-regression test
    True

    Notes
    -----
    .. math::
        P(Y \\le j | X) = \\Lambda(\\kappa_j - X'\\beta)

    where :math:`\\Lambda` is the logistic CDF. The parallel regression
    (proportional odds) assumption requires that :math:`\\beta` is the
    same for each cumulative split.

    References
    ----------
    mckelvey1975statistical
    """
    return _ordered_model(
        formula=formula,
        data=data,
        y=y,
        x=x,
        link="logit",
        robust=robust,
        cluster=cluster,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
    )


@accepts_aliases(vce="robust")
@markout_clusters
def oprobit(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Ordered probit model via MLE.

    Equivalent to Stata's ``oprobit y x``.

    Parameters
    ----------
    formula : str, optional
        Formula ``"y ~ x1 + x2"``.
    data : pd.DataFrame
    y : str, optional
        Ordered categorical dependent variable.
    x : list of str, optional
    robust : str, default "nonrobust"
    cluster : str, optional
    maxiter : int, default 100
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        Same structure as :func:`ologit` but with probit link.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(2)
    >>> n = 300
    >>> quality = rng.normal(0, 1, n)
    >>> price = rng.normal(0, 1, n)
    >>> latent = 0.7 * quality - 0.5 * price + rng.normal(0, 1, n)
    >>> rating = np.digitize(latent, [-0.4, 0.6])  # ordered {0, 1, 2}
    >>> df = pd.DataFrame({'rating': rating, 'quality': quality, 'price': price})
    >>> result = sp.oprobit(data=df, y='rating', x=['quality', 'price'])
    >>> print(result.summary())  # doctest: +SKIP
    >>> bool(len(result.marginal_effects) > 0)
    True

    Notes
    -----
    .. math::
        P(Y \\le j | X) = \\Phi(\\kappa_j - X'\\beta)

    where :math:`\\Phi` is the standard normal CDF.

    References
    ----------
    mckelvey1975statistical
    """
    return _ordered_model(
        formula=formula,
        data=data,
        y=y,
        x=x,
        link="probit",
        robust=robust,
        cluster=cluster,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
    )


# ====================================================================
# Conditional Logit
# ====================================================================


@accepts_aliases(vce="robust")
@markout_clusters
def clogit(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    group: Optional[str] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    McFadden's conditional (fixed-effect) logit for choice data.

    Each observation is an alternative within a choice set (group).
    The dependent variable is 1 for the chosen alternative, 0 otherwise.

    Equivalent to Stata's ``clogit y x, group(id)``.

    Parameters
    ----------
    formula : str, optional
        Formula ``"chosen ~ price + quality"``.
    data : pd.DataFrame
        Long-format data with one row per alternative per choice set.
    y : str, optional
        Binary indicator: 1 = chosen, 0 = not chosen.
    x : list of str, optional
        Alternative-specific (and/or individual-specific interacted
        with alternative dummies) covariates.
    group : str
        Variable identifying the choice set / decision-maker.
    robust : str, default "nonrobust"
    cluster : str, optional
    maxiter : int, default 100
    tol : float, default 1e-8
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(2)
    >>> rows = []
    >>> for case in range(150):
    ...     price = rng.normal(0, 1, 3)
    ...     quality = rng.normal(0, 1, 3)
    ...     util = -0.8 * price + 0.9 * quality + rng.gumbel(0, 1, 3)
    ...     chosen_alt = int(np.argmax(util))
    ...     for a in range(3):
    ...         rows.append({'case_id': case, 'chosen': int(a == chosen_alt),
    ...                      'price': price[a], 'quality': quality[a]})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.clogit('chosen ~ price + quality', data=df, group='case_id')
    >>> print(result.summary())  # doctest: +SKIP
    >>> bool(result.params is not None)
    True

    Notes
    -----
    The conditional log-likelihood for group g:

    .. math::
        \\ell_g = X_{g,chosen}'\\beta
        - \\log\\left(\\sum_{j \\in g} \\exp(X_{gj}'\\beta)\\right)

    Only alternative-specific variation identifies beta; the group
    fixed effect is conditioned out (no constant estimated).

    References
    ----------
    mcfadden1974conditional
    """
    if group is None:
        raise ValueError("'group' must be specified for conditional logit.")
    if data is None:
        raise ValueError("'data' must be provided.")
    robust, cluster = _parse_se(robust, cluster, "clogit")

    y_name, x_names = _parse_inputs(formula, data, y, x)
    group_name = group

    cols = [y_name, group_name] + x_names
    if cluster and cluster not in cols:
        cols.append(cluster)
    df = data[cols].dropna().copy()
    Y = df[y_name].values.astype(float)
    G_vals = df[group_name].values
    X = np.column_stack([df[v].values.astype(float) for v in x_names])
    n, k = X.shape
    var_names = list(x_names)

    cluster_vals = df[cluster].values if cluster else None

    # Group structure
    unique_groups = np.unique(G_vals)

    # Map groups to indices
    group_indices: Dict[Any, np.ndarray] = {}
    for g in unique_groups:
        idx = np.where(G_vals == g)[0]
        # Verify exactly one chosen per group
        chosen = Y[idx]
        if chosen.sum() != 1:
            # Skip groups without exactly 1 chosen alternative
            continue
        group_indices[g] = idx

    if len(group_indices) == 0:
        raise ValueError(
            "No valid choice groups (each group needs exactly one chosen alternative)."
        )

    n_groups_valid = len(group_indices)
    groups_list = list(group_indices.keys())

    def neg_loglik(beta: np.ndarray) -> float:
        ll = 0.0
        for g in groups_list:
            idx = group_indices[g]
            xb = X[idx] @ beta
            chosen_mask = Y[idx] == 1
            ll += xb[chosen_mask].sum() - np.log(
                np.sum(np.exp(xb - xb.max())) + np.exp(-xb.max())
            )
            # More stable: logsumexp
        return float(-ll)

    def neg_loglik_stable(beta: np.ndarray) -> float:
        ll = 0.0
        for g in groups_list:
            idx = group_indices[g]
            xb = X[idx] @ beta
            xb_max = xb.max()
            chosen_mask = Y[idx] == 1
            ll += xb[chosen_mask].sum() - (xb_max + np.log(np.sum(np.exp(xb - xb_max))))
        return float(-ll)

    def grad(beta: np.ndarray) -> np.ndarray:
        g_vec = np.zeros(k)
        for g in groups_list:
            idx = group_indices[g]
            xb = X[idx] @ beta
            xb_max = xb.max()
            exp_xb = np.exp(xb - xb_max)
            probs = exp_xb / exp_xb.sum()
            chosen_mask = Y[idx] == 1
            g_vec += X[idx][chosen_mask].sum(axis=0) - (
                probs[:, np.newaxis] * X[idx]
            ).sum(axis=0)
        return _as_float_array(-g_vec)

    def score_obs_clogit(beta: np.ndarray) -> np.ndarray:
        """Per-observation score (one row per group)."""
        S = np.zeros((n_groups_valid, k))
        for gi, g in enumerate(groups_list):
            idx = group_indices[g]
            xb = X[idx] @ beta
            xb_max = xb.max()
            exp_xb = np.exp(xb - xb_max)
            probs = exp_xb / exp_xb.sum()
            chosen_mask = Y[idx] == 1
            S[gi] = X[idx][chosen_mask].sum(axis=0) - (
                probs[:, np.newaxis] * X[idx]
            ).sum(axis=0)
        return _as_float_array(S)

    # --- Optimise ---
    beta0 = np.zeros(k)
    res = optimize.minimize(
        neg_loglik_stable,
        beta0,
        jac=grad,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": tol},
    )
    beta_hat = _as_float_array(res.x)
    ll = float(-res.fun)

    # Null model: beta=0 => each alternative equally likely
    ll_0 = 0.0
    for g in groups_list:
        idx = group_indices[g]
        J_g = len(idx)
        ll_0 += -np.log(J_g)

    pseudo_r2 = 1.0 - ll / ll_0
    aic = -2 * ll + 2 * k
    bic = -2 * ll + np.log(n_groups_valid) * k

    # --- Standard errors ---
    # Inverse *observed* information, analytic: for the conditional logit
    # -d2 ll = sum_g sum_j p_gj (x_gj - xbar_g)(x_gj - xbar_g)', with
    # xbar_g the probability-weighted group mean. This replaces BFGS
    # ``hess_inv``, a quasi-Newton approximation built from the optimisation
    # path rather than the information matrix.
    info = np.zeros((k, k))
    for g in groups_list:
        Xg = X[group_indices[g]]
        xb = Xg @ beta_hat
        p = np.exp(xb - xb.max())
        p = p / p.sum()
        centred = Xg - p @ Xg
        info += (centred * p[:, None]).T @ centred
    try:
        H_inv = _as_float_array(np.linalg.inv(info))
    except np.linalg.LinAlgError:
        H_inv = _as_float_array(np.linalg.pinv(info))

    S_obs = score_obs_clogit(beta_hat)

    # The clogit score is a per-*group* contribution, so clusters are mapped
    # to groups (groups must nest within clusters, as in Stata).
    cluster_group = None
    if cluster_vals is not None:
        cluster_group = np.array(
            [cluster_vals[group_indices[g][0]] for g in groups_list]
        )
        for g in groups_list:
            if len(pd.unique(cluster_vals[group_indices[g]])) > 1:
                raise MethodIncompatibility(
                    f"clogit: group {g!r} spans more than one {cluster!r} "
                    "cluster; groups must be nested within clusters.",
                    recovery_hint=(
                        "Cluster on a variable that is constant within group."
                    ),
                )
    se = _compute_se(S_obs, H_inv, robust, cluster_group)

    # --- Predicted choice probabilities ---
    pred_probs = np.zeros(n)
    for g in groups_list:
        idx = group_indices[g]
        xb = X[idx] @ beta_hat
        xb_max = xb.max()
        exp_xb = np.exp(xb - xb_max)
        pred_probs[idx] = exp_xb / exp_xb.sum()

    # --- Build results ---
    params_series = pd.Series(beta_hat, index=var_names)
    se_series = pd.Series(se, index=var_names)

    model_info = {
        "model_type": "Conditional Logit",
        "method": "MLE (conditional)",
        "group_var": group_name,
        "n_groups": n_groups_valid,
        "log_likelihood": float(ll),
        "log_likelihood_0": float(ll_0),
        "pseudo_r2": float(pseudo_r2),
        "aic": float(aic),
        "bic": float(bic),
        "converged": robust_convergence(res)[0],
        "robust": robust if cluster is None else f"cluster({cluster})",
    }

    data_info = {
        "dependent_var": y_name,
        "n_obs": n,
        "n_params": k,
        "df_resid": n_groups_valid - k,
        # Likelihood-based: z / chi2 inference, as Stata's clogit.
        "inference": "z",
    }

    diagnostics = {
        "McFadden_pseudo_R2": float(pseudo_r2),
        "Log-Likelihood": float(ll),
        "Log-Likelihood_0": float(ll_0),
        "AIC": float(aic),
        "BIC": float(bic),
        "n_obs": n,
        "n_groups": n_groups_valid,
    }

    model_info["alpha"] = alpha
    result = EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )

    setattr(result, "predicted_probs", pd.Series(pred_probs, index=df.index))

    return result
