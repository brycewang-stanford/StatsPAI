"""
Logit, probit, and complementary log-log discrete choice models.

Maximum-likelihood estimation with analytical gradients and Hessians,
robust/clustered standard errors, marginal effects, and full diagnostics.

    P(Y=1|X) = F(X'β)

where F is the logistic CDF (logit), standard normal CDF (probit),
or 1 - exp(-exp(·)) (cloglog).

References
----------
Cameron, A.C. & Trivedi, P.K. (2005).
    *Microeconometrics: Methods and Applications*. Cambridge.

Greene, W.H. (2018).
    *Econometric Analysis*, 8th ed. Pearson.

Hosmer, D.W. & Lemeshow, S. (2000).
    *Applied Logistic Regression*, 2nd ed. Wiley. [@hosmer2000applied]
"""

import functools
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit as _expit

from .._aliases import accepts_aliases
from ..core._vcov_spec import markout_clusters
from ..core.results import EconometricResults
from ..core.utils import create_design_matrices
from ..exceptions import MethodIncompatibility
from ..output._lineage import records_provenance

LinkFunc = Callable[[np.ndarray], np.ndarray]
LinkTriplet = Tuple[LinkFunc, LinkFunc, LinkFunc]


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


# =========================================================================
# Link functions: CDF, PDF, and derivatives
# =========================================================================


def _logit_cdf(z: np.ndarray) -> np.ndarray:
    """Logistic CDF  Λ(z) = 1/(1+exp(-z))  (numerically stable)."""
    # both branches of a np.where are evaluated, so the "stable" two-sided
    # form still overflowed (with a RuntimeWarning) on a large index
    return _as_float_array(_expit(np.asarray(z, dtype=float)))


def _logit_pdf(z: np.ndarray) -> np.ndarray:
    """Logistic PDF  λ(z) = Λ(z)(1-Λ(z))."""
    p = _logit_cdf(z)
    return _as_float_array(p * (1.0 - p))


def _logit_pdf_deriv(z: np.ndarray) -> np.ndarray:
    """d/dz of logistic PDF:  λ'(z) = λ(z)(1-2Λ(z))."""
    p = _logit_cdf(z)
    return _as_float_array(p * (1.0 - p) * (1.0 - 2.0 * p))


def _probit_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF  Φ(z)."""
    return _as_float_array(stats.norm.cdf(z))


def _probit_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal PDF  φ(z)."""
    return _as_float_array(stats.norm.pdf(z))


def _probit_pdf_deriv(z: np.ndarray) -> np.ndarray:
    """d/dz of normal PDF:  φ'(z) = -z φ(z)."""
    return _as_float_array(-z * stats.norm.pdf(z))


def _cloglog_cdf(z: np.ndarray) -> np.ndarray:
    """Complementary log-log CDF  1 - exp(-exp(z))."""
    # Clip to prevent overflow
    z_clip = np.clip(z, -30, 30)
    return _as_float_array(1.0 - np.exp(-np.exp(z_clip)))


def _cloglog_pdf(z: np.ndarray) -> np.ndarray:
    """Complementary log-log PDF  exp(z) * exp(-exp(z))."""
    z_clip = np.clip(z, -30, 30)
    return _as_float_array(np.exp(z_clip) * np.exp(-np.exp(z_clip)))


def _cloglog_pdf_deriv(z: np.ndarray) -> np.ndarray:
    """d/dz of cloglog PDF."""
    z_clip = np.clip(z, -30, 30)
    ez = np.exp(z_clip)
    return _as_float_array(np.exp(-ez) * ez * (1.0 - ez))


_LINKS: Dict[str, LinkTriplet] = {
    "logit": (_logit_cdf, _logit_pdf, _logit_pdf_deriv),
    "probit": (_probit_cdf, _probit_pdf, _probit_pdf_deriv),
    "cloglog": (_cloglog_cdf, _cloglog_pdf, _cloglog_pdf_deriv),
}


# =========================================================================
# Core MLE engine
# =========================================================================


def _log_likelihood(
    beta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    cdf_func: LinkFunc,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Bernoulli log-likelihood  Σ w_i [y_i log F(Xβ) + (1-y_i) log(1-F(Xβ))]."""
    z = X @ beta
    p = cdf_func(z)
    # Clip probabilities for numerical safety
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    ll = y * np.log(p) + (1.0 - y) * np.log(1.0 - p)
    if weights is not None:
        ll = ll * weights
    return float(np.sum(ll))


def _score(
    beta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    cdf_func: LinkFunc,
    pdf_func: LinkFunc,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gradient (score) vector  ∂ℓ/∂β."""
    z = X @ beta
    p = cdf_func(z)
    f = pdf_func(z)
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    # generalized residual
    gen_resid = (y - p) * f / (p * (1.0 - p))
    if weights is not None:
        gen_resid = gen_resid * weights
    return _as_float_array(X.T @ gen_resid)


def _hessian(
    beta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    cdf_func: LinkFunc,
    pdf_func: LinkFunc,
    pdf_deriv_func: LinkFunc,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Analytical Hessian  ∂²ℓ/∂β∂β'."""
    z = X @ beta
    p = cdf_func(z)
    f = pdf_func(z)
    fp = pdf_deriv_func(z)
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)

    pq = p * (1.0 - p)
    # d²ℓ/dz² for each obs
    d2 = (y - p) * (fp * pq - f**2 * (1.0 - 2.0 * p)) / pq**2 - f**2 / pq
    if weights is not None:
        d2 = d2 * weights
    return _as_float_array(X.T @ (d2[:, np.newaxis] * X))


def _warn_if_separated(y: np.ndarray, p_hat: np.ndarray) -> None:
    """Warn on (quasi-)complete separation, where the MLE does not exist.

    Under separation Newton-Raphson does not diverge loudly: it "converges"
    by the step tolerance while coefficients drift toward +/- infinity, so
    the reported estimates and standard errors are artefacts of the stopping
    rule, not a finite optimum.

    Two signatures, either one enough:

    * every observation is classified correctly by the sign of the fitted
      index. That *is* complete separation: a linear index that puts all
      the ones on one side and all the zeros on the other can be scaled up
      without bound, each step raising the likelihood.
    * some fitted probabilities are 0 or 1 to machine precision (R ``glm``
      warns on the same event). A finite maximum of a binary likelihood
      does not put an observation there; it is what quasi-complete
      separation looks like, e.g. a category in which the outcome never
      varies.

    An earlier version also required 99 percent of the fitted probabilities
    to lie within 0.01 of the boundary. With a few dozen observations one
    point near the separating threshold was enough to silence the warning
    while the slope ran into the thousands.
    """
    y_arr = np.asarray(y).ravel()
    p = np.asarray(p_hat).ravel()
    if p.size == 0 or np.unique(y_arr).size < 2:
        return
    complete = np.array_equal((p >= 0.5).astype(int), y_arr.astype(int))
    eps = 10.0 * np.finfo(float).eps
    n_boundary = int(np.sum((p < eps) | (p > 1.0 - eps)))
    if not complete and n_boundary == 0:
        return
    from ..exceptions import ConvergenceWarning
    from ..exceptions import warn as _sp_warn

    if complete:
        what = (
            "Complete separation detected: the linear index classifies every "
            "observation correctly"
        )
    else:
        what = (
            f"Quasi-complete separation suspected: {n_boundary} fitted "
            "probabilities are numerically 0 or 1"
        )
    _sp_warn(
        ConvergenceWarning,
        what + ", so the maximum-likelihood estimates do not exist. The "
        "reported coefficients and standard errors are driven by the "
        "optimizer's stopping rule, not a finite optimum, and should not be "
        "interpreted.",
        recovery_hint=(
            "Use a weakly informative prior "
            "(sp.bayes_regress(..., model='logit', "
            "prior='weakly_informative')), drop the perfectly separating "
            "predictor, or pool sparse categories."
        ),
        stacklevel=3,
    )


def _binary_r2(
    y: np.ndarray,
    p: np.ndarray,
    xb: np.ndarray,
    ll: float,
    ll_null: float,
    link: str,
) -> Dict[str, float]:
    """Goodness-of-fit measures of a binary response model (unweighted).

    No single one is "the" R-squared of a probit or logit; they answer
    different questions and are reported together under their usual names.
    ``mckelvey_zavoina`` is the share of the variance of the latent
    variable explained by the index, with the error variance of the link
    (1 for probit, pi^2 / 3 for logit); it is not defined for cloglog.
    """
    n = float(len(y))
    ybar = float(np.mean(y))
    tss = float(np.sum((y - ybar) ** 2))
    lr = 2.0 * (ll - ll_null)
    cox_snell = 1.0 - np.exp(-lr / n)
    out = {
        "mcfadden": 1.0 - ll / ll_null if ll_null != 0 else np.nan,
        "cox_snell": cox_snell,
        "nagelkerke": cox_snell / (1.0 - np.exp(2.0 * ll_null / n)),
        "efron": 1.0 - float(np.sum((y - p) ** 2)) / tss,
        "tjur": float(np.mean(p[y == 1]) - np.mean(p[y == 0])),
        "estrella": (
            1.0 - (ll / ll_null) ** (-2.0 / n * ll_null) if ll_null != 0 else np.nan
        ),
    }
    err_var = {"probit": 1.0, "logit": np.pi**2 / 3.0}.get(link)
    if err_var is not None:
        v = float(np.var(xb))
        out["mckelvey_zavoina"] = v / (v + err_var)
    return {k: float(val) for k, val in out.items()}


def _newton_raphson(
    y: np.ndarray,
    X: np.ndarray,
    link: str,
    weights: Optional[np.ndarray] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Newton-Raphson MLE for binary choice models.

    Returns
    -------
    beta : parameter vector
    H : Hessian at convergence
    ll : log-likelihood at convergence
    n_iter : iterations used
    """
    cdf_func, pdf_func, pdf_deriv_func = _LINKS[link]
    n, k = X.shape

    # Starting values via OLS on y (clipped to 0.01–0.99)
    y_star = np.clip(y, 0.01, 0.99)
    try:
        beta = np.linalg.lstsq(X, y_star, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(k)

    ll_old = -np.inf
    converged = False

    for iteration in range(maxiter):
        ll_val = _log_likelihood(beta, y, X, cdf_func, weights)
        if np.abs(ll_val - ll_old) < tol:
            converged = True
            break
        ll_old = ll_val

        g = _score(beta, y, X, cdf_func, pdf_func, weights)
        H = _hessian(beta, y, X, cdf_func, pdf_func, pdf_deriv_func, weights)

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]

        # Line search with step halving
        step_size = 1.0
        for _ in range(20):
            beta_new = beta - step_size * step
            ll_new = _log_likelihood(beta_new, y, X, cdf_func, weights)
            if ll_new > ll_val - 1e-4:
                break
            step_size *= 0.5
        beta = beta_new

    if not converged:
        warnings.warn(
            f"Newton-Raphson did not converge after {maxiter} iterations. "
            "Consider increasing maxiter or checking data.",
            stacklevel=3,
        )

    # Final Hessian for variance estimation
    H = _hessian(beta, y, X, cdf_func, pdf_func, pdf_deriv_func, weights)
    ll_val = _log_likelihood(beta, y, X, cdf_func, weights)

    return beta, H, ll_val, iteration + 1


# =========================================================================
# Variance-covariance estimators
# =========================================================================


def _mle_vcov(H: np.ndarray) -> np.ndarray:
    """MLE (observed information) variance: V = -H^{-1}."""
    try:
        return _as_float_array(np.linalg.inv(-H))
    except np.linalg.LinAlgError:
        warnings.warn("Hessian is singular; using pseudo-inverse.", stacklevel=3)
        return _as_float_array(np.linalg.pinv(-H))


def _score_obs(
    beta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    cdf_func: LinkFunc,
    pdf_func: LinkFunc,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-observation score vectors (n × k)."""
    z = X @ beta
    p = cdf_func(z)
    f = pdf_func(z)
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    gen_resid = (y - p) * f / (p * (1.0 - p))
    if weights is not None:
        gen_resid = gen_resid * weights
    return _as_float_array(gen_resid[:, np.newaxis] * X)


# =========================================================================
# Marginal effects
# =========================================================================


def _marginal_effects(
    beta: np.ndarray,
    X: np.ndarray,
    pdf_func: LinkFunc,
    var_names: List[str],
    kind: str = "average",
    at_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute marginal effects ∂P/∂x_j = f(X'β) β_j.

    Parameters
    ----------
    kind : 'average' (AME), 'mean' (MEM), 'at' (MER)
    at_values : dict of variable -> value (for kind='at')
    """
    if kind == "average":
        # AME: average of f(x_i'β) across all obs
        z = X @ beta
        f = pdf_func(z)
        me = np.mean(f) * beta
    elif kind == "mean":
        # MEM: f evaluated at sample means
        x_bar = X.mean(axis=0)
        z_bar = x_bar @ beta
        f_bar = pdf_func(np.array([z_bar]))[0]
        me = f_bar * beta
    elif kind == "at":
        # MER: at representative values
        if at_values is None:
            # Default to means
            x_rep = X.mean(axis=0)
        else:
            x_rep = X.mean(axis=0).copy()
            for vname, val in at_values.items():
                if vname in var_names:
                    idx = var_names.index(vname)
                    x_rep[idx] = val
        z_rep = x_rep @ beta
        f_rep = pdf_func(np.array([z_rep]))[0]
        me = f_rep * beta
    else:
        raise ValueError(f"Unknown marginal_effects kind: {kind}")

    return pd.DataFrame(
        {
            "dy/dx": me,
        },
        index=var_names,
    )


# =========================================================================
# Diagnostics
# =========================================================================


def _hosmer_lemeshow(
    y: np.ndarray, p_hat: np.ndarray, n_groups: int = 10
) -> Tuple[float, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test.

    Returns (chi2_stat, p_value).
    """
    n = len(y)
    order = np.argsort(p_hat)
    y_sorted = y[order]
    p_sorted = p_hat[order]

    # Create groups (approximately equal sized)
    groups = np.array_split(np.arange(n), n_groups)

    chi2 = 0.0
    for grp_idx in groups:
        n_g = len(grp_idx)
        if n_g == 0:
            continue
        o_g = y_sorted[grp_idx].sum()
        e_g = p_sorted[grp_idx].sum()
        pi_g = e_g / n_g
        denom = n_g * pi_g * (1.0 - pi_g)
        if denom > 1e-15:
            chi2 += (o_g - e_g) ** 2 / denom

    df = n_groups - 2
    p_value = stats.chi2.sf(chi2, df) if df > 0 else np.nan
    return chi2, p_value


def _roc_auc(y: np.ndarray, p_hat: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney U statistic)."""
    pos = p_hat[y == 1]
    neg = p_hat[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # Efficient computation via rank sums
    n1 = len(pos)
    n0 = len(neg)
    all_scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(n1), np.zeros(n0)])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(all_scores) + 1, dtype=float)

    # Handle ties
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (ranks[order[i:j]].sum()) / (j - i)
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum = ranks[labels == 1].sum()
    u = rank_sum - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))


def _classification_table(
    y: np.ndarray,
    p_hat: np.ndarray,
    cutoff: float = 0.5,
) -> Dict[str, Any]:
    """Confusion matrix with sensitivity/specificity."""
    y_pred = (p_hat >= cutoff).astype(int)
    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    pcp = (tp + tn) / len(y) * 100

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "pcp": pcp,
        "cutoff": cutoff,
    }


# =========================================================================
# Prediction helper
# =========================================================================


def _predict(
    beta: np.ndarray,
    X: np.ndarray,
    cdf_func: LinkFunc,
    pred_type: str = "response",
    cutoff: float = 0.5,
) -> np.ndarray:
    """
    Predict from a fitted binary choice model.

    Parameters
    ----------
    pred_type : 'response' (probabilities), 'link' (xβ), 'class' (0/1)
    """
    xb = X @ beta
    if pred_type == "link":
        return _as_float_array(xb)
    elif pred_type == "response":
        return _as_float_array(cdf_func(xb))
    elif pred_type == "class":
        return _as_float_array((cdf_func(xb) >= cutoff).astype(int))
    else:
        raise ValueError(f"Unknown predict type: {pred_type}")


# =========================================================================
# Public API
# =========================================================================


def _perfect_prediction_mask(
    y: np.ndarray, X: np.ndarray, var_names: List[str]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Stata's perfect-prediction rule for indicator regressors.

    For a column taking the value 0 and one other value, if the outcome is
    constant among its non-zero rows, those rows carry no information about
    the other coefficients and the column's own coefficient is infinite:
    Stata ``logit`` / ``probit`` report ``"x != 0 predicts failure
    perfectly"``, drop the rows and omit ``x``.  A category of a
    ``C(ind)`` factor in which nobody is treated is the common case.
    Iterated, since dropping rows can create another such column or leave
    a column identically zero (then omitted as collinear).

    Returns ``(keep_rows, dropped_columns, notes)``.
    """
    n, k = X.shape
    keep = np.ones(n, dtype=bool)
    dropped: List[str] = []
    notes: List[str] = []
    changed = True
    while changed:
        changed = False
        for j in range(k):
            if var_names[j] in dropped:
                continue
            col = X[keep, j]
            nz = col != 0
            if nz.all() or not nz.any():
                if not nz.any() and keep.any():
                    dropped.append(var_names[j])
                    notes.append(f"{var_names[j]} omitted (identically zero)")
                    changed = True
                continue
            if np.unique(col[nz]).size != 1:
                continue
            yv = y[keep][nz]
            if yv.min() == yv.max():
                outcome = "success" if yv[0] == 1 else "failure"
                idx = np.flatnonzero(keep)[nz]
                keep[idx] = False
                dropped.append(var_names[j])
                notes.append(
                    f"{var_names[j]} != 0 predicts {outcome} perfectly; "
                    f"{var_names[j]} omitted and {idx.size} obs not used"
                )
                changed = True
    return keep, dropped, notes


def _fit_binary(
    formula: Optional[str],
    data: Optional[pd.DataFrame],
    y: Optional[str],
    x: Optional[List[str]],
    link: str,
    robust: str,
    cluster: Optional[str],
    weights: Optional[str],
    marginal_effects: Optional[str],
    odds_ratio: bool,
    maxiter: int,
    tol: float,
    alpha: float,
    at_values: Optional[Dict[str, float]] = None,
    perfect_prediction: str = "drop",
) -> EconometricResults:
    """
    Internal workhorse for logit / probit / cloglog estimation.
    """
    # ── Prepare data ────────────────────────────────────────────────────
    if formula is not None and data is not None:
        y_df, X_df = create_design_matrices(formula, data)
        y_vec = y_df.values.ravel()
        X_mat = X_df.values
        var_names = list(X_df.columns)
        dep_var = y_df.columns[0]
    elif y is not None and x is not None and data is not None:
        cols = [y] + list(x)
        clean = data[cols].dropna()
        y_vec = clean[y].values.astype(float)
        dep_var = y
        X_raw = clean[x].values.astype(float)
        X_mat = np.column_stack([np.ones(len(X_raw)), X_raw])
        var_names = ["Intercept"] + list(x)
    else:
        raise ValueError("Provide either (formula, data) or (y, x, data).")
    row_index = X_df.index if formula is not None and data is not None else clean.index

    # A dependent regressor leaves the likelihood flat in one direction;
    # omit it (Stata's rule) instead of reporting wherever Newton stopped.
    from ..core._collinear import drop_collinear

    X_mat, var_names, collinear_omitted, _ = drop_collinear(
        np.asarray(X_mat, dtype=float),
        var_names,
        link,
        formula=formula if data is not None else None,
        design_info=(
            getattr(X_df, "design_info", None)
            if formula is not None and data is not None
            else None
        ),
        stacklevel=4,
    )

    # Stata drops indicator regressors that predict the outcome perfectly
    # together with the rows they predict; keeping them leaves an MLE that
    # does not exist and a sample that differs from Stata's (silently).
    pp_notes: List[str] = []
    pp_dropped: List[str] = []
    n_pp = 0
    if perfect_prediction not in ("drop", "keep"):
        raise MethodIncompatibility("perfect_prediction must be 'drop' or 'keep'")
    if perfect_prediction == "drop" and X_mat.shape[1] > 1:
        _keep, pp_dropped, pp_notes = _perfect_prediction_mask(
            np.asarray(y_vec, float), X_mat, var_names
        )
        if pp_dropped:
            n_pp = int((~_keep).sum())
            live = [j for j, v in enumerate(var_names) if v not in pp_dropped]
            y_vec = y_vec[_keep]
            X_mat = X_mat[_keep][:, live]
            var_names = [var_names[j] for j in live]
            row_index = row_index[_keep]
            warnings.warn(
                f"{link}: " + "; ".join(pp_notes) + " (Stata's rule). "
                "Pass perfect_prediction='keep' to keep them.",
                UserWarning,
                stacklevel=3,
            )

    # Validate binary outcome
    unique_vals = np.unique(y_vec)
    if not np.array_equal(np.sort(unique_vals), np.array([0.0, 1.0])):
        if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            pass  # only one category present — unusual but proceed
        else:
            raise ValueError(
                f"Dependent variable must be binary (0/1). "
                f"Found values: {unique_vals[:10]}"
            )

    n, k = X_mat.shape

    # Weights
    w = None
    if weights is not None and data is not None:
        w = data.loc[row_index, weights].values.astype(float)

    # Standard-error request (Stata grammar: vce='robust', 'cluster firm', ...)
    from ..core._vcov import ml_vcov
    from ..core._vcov_spec import parse_se_request

    se_req = parse_se_request(
        robust,
        cluster,
        function=link,
        supported=("nonrobust", "robust", "hc0", "hc1", "cluster"),
    )
    robust, cluster = se_req.kind, se_req.cluster

    # Cluster variable
    cluster_arr = None
    if cluster is not None:
        if data is None:
            raise ValueError("`data` must be provided for clustered SEs.")
        cluster_arr = data.loc[row_index, cluster].values

    cdf_func, pdf_func, pdf_deriv_func = _LINKS[link]

    # ── Estimation ──────────────────────────────────────────────────────
    beta, H, ll, n_iter = _newton_raphson(y_vec, X_mat, link, w, maxiter, tol)

    # ── Variance-covariance ─────────────────────────────────────────────
    # Stata conventions (core._vcov.ml_vcov): vce(robust) carries N/(N-1),
    # vce(cluster) carries G/(G-1) only.
    s_obs = _score_obs(beta, y_vec, X_mat, cdf_func, pdf_func, w)
    vcov = ml_vcov(_mle_vcov(H), s_obs, kind=se_req.kind, clusters=cluster_arr)
    se_type = {
        "nonrobust": "MLE (observed information)",
        "robust": "Robust (sandwich, N/(N-1))",
        "hc0": "HC0 (sandwich, no small-sample factor)",
        "hc1": "HC1 (sandwich, N/(N-K))",
        "cluster": f"Clustered ({cluster})",
    }[se_req.kind]

    std_errors = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    # ── Null model log-likelihood ───────────────────────────────────────
    X_null = np.ones((n, 1))
    _, _, ll_null, _ = _newton_raphson(y_vec, X_null, link, w, maxiter=50, tol=1e-8)

    # ── Diagnostics ─────────────────────────────────────────────────────
    p_hat = cdf_func(X_mat @ beta)
    _warn_if_separated(y_vec, p_hat)
    lr_chi2 = 2.0 * (ll - ll_null)
    lr_df = k - 1
    lr_pvalue = stats.chi2.sf(lr_chi2, lr_df) if lr_df > 0 else np.nan
    pseudo_r2 = 1.0 - ll / ll_null if ll_null != 0 else np.nan
    aic = -2.0 * ll + 2.0 * k
    bic = -2.0 * ll + np.log(n) * k
    cls_table = _classification_table(y_vec, p_hat)
    hl_chi2, hl_pval = _hosmer_lemeshow(y_vec, p_hat)
    auc = _roc_auc(y_vec, p_hat)

    # ── Marginal effects ────────────────────────────────────────────────
    me_df = None
    if marginal_effects is not None:
        kind_map = {"average": "average", "mean": "mean", "at": "at"}
        kind = kind_map.get(marginal_effects, "average")
        me_df = _marginal_effects(beta, X_mat, pdf_func, var_names, kind, at_values)

    # ── Odds ratios (logit only) ────────────────────────────────────────
    or_series = None
    if odds_ratio and link == "logit":
        or_vals = np.exp(beta)
        or_se = or_vals * std_errors  # delta-method
        or_series = pd.DataFrame(
            {
                "OR": or_vals,
                "Std. Err.": or_se,
                f"[{alpha / 2:.3f}": np.exp(
                    beta - stats.norm.ppf(1 - alpha / 2) * std_errors
                ),
                f"{1 - alpha / 2:.3f}]": np.exp(
                    beta + stats.norm.ppf(1 - alpha / 2) * std_errors
                ),
            },
            index=var_names,
        )

    # ── Build result ────────────────────────────────────────────────────
    params = pd.Series(beta, index=var_names)
    se_series = pd.Series(std_errors, index=var_names)

    link_label = {
        "logit": "Logit",
        "probit": "Probit",
        "cloglog": "Complementary log-log",
    }
    model_info = {
        "model_type": link_label[link],
        "method": "Maximum Likelihood (Newton-Raphson)",
        "r2": (
            _binary_r2(y_vec, p_hat, X_mat @ beta, ll, ll_null, link)
            if weights is None
            else None
        ),
        "family": "binomial",
        "link": link,
        "ll": ll,
        "ll_null": ll_null,
        "lr_chi2": lr_chi2,
        "lr_df": lr_df,
        "lr_pvalue": lr_pvalue,
        "pseudo_r2": pseudo_r2,
        "aic": aic,
        "bic": bic,
        "pcp": cls_table["pcp"],
        "robust": robust,
        "cluster": cluster,
        "se_type": se_type,
        "n_iter": n_iter,
        "odds_ratio": or_series,
        "marginal_effects": me_df,
        "classification": cls_table,
        "hosmer_lemeshow": {"chi2": hl_chi2, "p_value": hl_pval},
        "auc": auc,
        "perfect_prediction": perfect_prediction,
        "perfect_prediction_omitted": list(pp_dropped),
        "omitted": collinear_omitted,
        "n_perfect_prediction_dropped": n_pp,
    }

    data_info = {
        "nobs": n,
        "df_model": k - 1,
        "df_resid": n - k,
        "dependent_var": dep_var,
        "fitted_values": p_hat,
        "residuals": y_vec - p_hat,
        "X": X_mat,
        "y": y_vec,
        "var_cov": vcov,
        "var_names": var_names,
        # Likelihood-based: z / chi2 inference, as Stata's logit/probit.
        "inference": "z",
        # Unweighted per-observation log-likelihood (sp.vuong).
        "llobs": np.where(
            y_vec == 1,
            np.log(np.maximum(p_hat, 1e-300)),
            np.log(np.maximum(1.0 - p_hat, 1e-300)),
        ),
    }

    diagnostics = {
        "Pseudo R-squared": pseudo_r2,
        "Log-Likelihood": ll,
        "Log-Lik. (null)": ll_null,
        "LR chi2": lr_chi2,
        "Prob > chi2": lr_pvalue,
        "AIC": aic,
        "BIC": bic,
        "PCP": cls_table["pcp"],
        "AUC (ROC)": auc,
    }
    # Surface the number of clusters so result.violations() flags few-cluster
    # inference (the CRVE is unreliable with few clusters), consistent with
    # sp.regress / sp.panel.
    if cluster_arr is not None:
        model_info["n_clusters"] = int(len(np.unique(cluster_arr)))
    if w is not None:
        from ..core._agent_summary import note_weight_diagnostics

        note_weight_diagnostics(
            model_info,
            w,
            (
                "cluster"
                if cluster_arr is not None
                else ("classical" if robust == "nonrobust" else "robust")
            ),
            entry=f"sp.{link}",
            robust_option="robust='robust' (Stata [pw=])",
            classical_reading="frequency weights (Stata [fw=])",
            classical_assumption=(
                "frequencies (each row standing for w identical observations)"
            ),
            cluster_keys=cluster_arr,
            cluster=cluster,
        )

    model_info["alpha"] = alpha
    # Picklable design recipe for postestimation (sp.margins rebuilds the
    # design, incl. C() factors and I() transforms, from the formula).
    if formula is not None:
        model_info["formula"] = formula
    if isinstance(weights, str):
        model_info["weights"] = weights
    result = EconometricResults(
        params=params,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )

    # functools.partial over module-level helpers, not closures: a closure
    # attribute made every logit/probit result unpicklable, which broke the
    # MCP result cache and replication packs.
    setattr(
        result,
        "predict",
        functools.partial(
            _bound_result_predict,
            beta,
            X_mat,
            cdf_func,
            _names=tuple(str(n) for n in var_names),
            _formula=formula,
            _vcov=np.asarray(vcov, dtype=float),
            _alpha=alpha,
        ),
    )
    setattr(
        result,
        "classification_table",
        functools.partial(
            _bound_classification_table, beta, X_mat, y_vec, p_hat, cdf_func
        ),
    )

    return result


def _bound_result_predict(
    beta: np.ndarray,
    X_mat: np.ndarray,
    cdf_func: Any,
    X_new: Any = None,
    pred_type: str = "response",
    cutoff: float = 0.5,
    *,
    data: Optional[pd.DataFrame] = None,
    what: Optional[str] = None,
    alpha: Optional[float] = None,
    _names: Tuple[str, ...] = (),
    _formula: Optional[str] = None,
    _vcov: Optional[np.ndarray] = None,
    _alpha: float = 0.05,
) -> Any:
    """``result.predict`` for binary-response fits (picklable via partial).

    ``X_new`` (or ``data=``) is a DataFrame holding the variables of the
    model, from which the design -- transforms, factors and interactions
    included -- is rebuilt, or an array that already is the design matrix.
    ``pred_type`` is ``'response'`` (probabilities), ``'link'`` (the index)
    or ``'class'`` (0/1 at ``cutoff``).

    ``what='confidence'`` returns a DataFrame instead, as ``sp.regress``
    and ``sp.glm`` results do: ``yhat`` (the probability), its delta-method
    standard error ``se`` (R ``predict(type = "response", se.fit = TRUE)``)
    and an interval ``lower`` / ``upper`` formed on the index and mapped
    through the link, so it stays inside (0, 1). ``what='mean'`` and
    ``what='link'`` are the probability and the index.
    """
    if what is not None:
        key = str(what).lower()
        if key == "mean":
            pred_type = "response"
        elif key == "link":
            pred_type = "link"
        elif key != "confidence":
            raise MethodIncompatibility(
                f"predict(): what={what!r} is not available for a binary "
                "outcome; use 'mean', 'link' or 'confidence'.",
                recovery_hint="A 0/1 outcome has no prediction interval.",
            )
    if data is not None:
        X_new = data
    if isinstance(X_new, pd.DataFrame):
        from types import SimpleNamespace

        from ..postestimation._design import design_for

        shim = SimpleNamespace(
            params=pd.Series(np.asarray(beta, dtype=float), index=list(_names)),
            model_info={"formula": _formula},
        )
        try:
            X_pred = np.asarray(design_for(shim, X_new).build(X_new), dtype=float)
        except MethodIncompatibility as exc:
            message = str(exc).split("\n")[0].replace("margins: ", "")
            raise MethodIncompatibility(
                f"predict(): {message}",
                recovery_hint="Pass a DataFrame with every variable of the "
                "fitted formula.",
            ) from exc
    else:
        X_pred = X_mat if X_new is None else _as_float_array(X_new)
    if X_pred.ndim != 2 or X_pred.shape[1] != np.asarray(beta).shape[0]:
        raise MethodIncompatibility(
            f"predict(): the design has shape {X_pred.shape} but the model has "
            f"{np.asarray(beta).shape[0]} coefficients.",
            recovery_hint="Pass a DataFrame with the variables of the model, "
            "or the full design matrix including the constant.",
        )
    if what is not None and str(what).lower() == "confidence":
        if _vcov is None:
            raise MethodIncompatibility(
                "predict(what='confidence'): the fit did not keep its "
                "covariance matrix.",
            )
        from scipy import stats as _stats

        level = _alpha if alpha is None else float(alpha)
        xb = X_pred @ np.asarray(beta, dtype=float)
        V = np.asarray(_vcov, dtype=float)
        se_link = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X_pred, V, X_pred), 0.0))
        step = 1e-6
        density = (cdf_func(xb + step) - cdf_func(xb - step)) / (2.0 * step)
        z = float(_stats.norm.ppf(1.0 - level / 2.0))
        index = X_new.index if isinstance(X_new, pd.DataFrame) else None
        return pd.DataFrame(
            {
                "yhat": _as_float_array(cdf_func(xb)),
                "se": _as_float_array(density * se_link),
                "lower": _as_float_array(cdf_func(xb - z * se_link)),
                "upper": _as_float_array(cdf_func(xb + z * se_link)),
            },
            index=index,
        )
    return _predict(beta, X_pred, cdf_func, pred_type, cutoff)


def _bound_classification_table(
    beta: np.ndarray,
    X_mat: np.ndarray,
    y_vec: np.ndarray,
    p_hat: np.ndarray,
    cdf_func: Any,
    cutoff: float = 0.5,
) -> Dict[str, Any]:
    """``result.classification_table`` (picklable via partial)."""
    probs = p_hat if cutoff == 0.5 else cdf_func(X_mat @ beta)
    return _classification_table(y_vec, _as_float_array(probs), cutoff)


# =========================================================================
# Public functions
# =========================================================================


@accepts_aliases(vce="robust")
@records_provenance("sp.logit")
@markout_clusters
def logit(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    marginal_effects: Optional[str] = None,
    odds_ratio: bool = False,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
    at_values: Optional[Dict[str, float]] = None,
    perfect_prediction: str = "drop",
) -> EconometricResults:
    """
    Logit (logistic) regression via maximum likelihood.

    Equivalent to Stata's ``logit y x1 x2`` or ``logistic`` (with ``or=True``).

    Parameters
    ----------
    formula : str, optional
        Formula like ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        Data containing the variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Regressor names (alternative to formula).
    robust : str, default ``'nonrobust'``
        ``'nonrobust'`` for MLE SE, ``'hc1'`` / ``'robust'`` for sandwich SE.
    cluster : str, optional
        Column name for clustered standard errors.
    weights : str, optional
        Column name for frequency/analytic weights.
    marginal_effects : str, optional
        ``'average'`` (AME), ``'mean'`` (MEM), or ``'at'`` (MER).
    odds_ratio : bool, default False
        Report odds ratios instead of log-odds coefficients.
    maxiter : int, default 100
        Maximum Newton-Raphson iterations.
    tol : float, default 1e-8
        Convergence tolerance on log-likelihood change.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    at_values : dict, optional
        Variable values for ``marginal_effects='at'``.
    perfect_prediction : {'drop', 'keep'}, default 'drop'
        An indicator regressor whose non-zero rows all share one outcome
        (e.g. an industry dummy with no treated firm) predicts the outcome
        perfectly: its coefficient is infinite. ``'drop'`` removes those
        rows and omits the regressor, iterated, as Stata does ("x != 0
        predicts failure perfectly"); the omitted names and row count are
        in ``model_info``. ``'keep'`` is the pre-1.33 behaviour.

    Returns
    -------
    EconometricResults
        Fitted model with ``.summary()``, ``.predict()``, diagnostics, etc.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()  # binary `union` outcome
    >>> result = sp.logit("union ~ education + experience", data=df)
    >>> print(result.summary())  # doctest: +SKIP

    >>> # With odds ratios and robust SE
    >>> result = sp.logit("union ~ education + experience", data=df,
    ...                   robust='hc1', odds_ratio=True)

    >>> # Marginal effects at the mean
    >>> result = sp.logit("union ~ education + experience", data=df,
    ...                   marginal_effects='mean')
    >>> me = result.model_info['marginal_effects']
    >>> bool('dy/dx' in me.columns)
    True
    """
    return _fit_binary(
        formula=formula,
        data=data,
        y=y,
        x=x,
        link="logit",
        robust=robust,
        cluster=cluster,
        weights=weights,
        marginal_effects=marginal_effects,
        odds_ratio=odds_ratio,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
        at_values=at_values,
        perfect_prediction=perfect_prediction,
    )


@accepts_aliases(vce="robust")
@records_provenance("sp.probit")
@markout_clusters
def probit(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    marginal_effects: Optional[str] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
    at_values: Optional[Dict[str, float]] = None,
    perfect_prediction: str = "drop",
) -> EconometricResults:
    """
    Probit regression via maximum likelihood.

    Equivalent to Stata's ``probit y x1 x2``.

    Parameters
    ----------
    formula : str, optional
        Formula like ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        Data containing the variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Regressor names (alternative to formula).
    robust : str, default ``'nonrobust'``
        ``'nonrobust'`` for MLE SE, ``'hc1'`` / ``'robust'`` for sandwich SE.
    cluster : str, optional
        Column name for clustered standard errors.
    weights : str, optional
        Column name for frequency/analytic weights.
    marginal_effects : str, optional
        ``'average'`` (AME), ``'mean'`` (MEM), or ``'at'`` (MER).
    maxiter : int, default 100
        Maximum Newton-Raphson iterations.
    tol : float, default 1e-8
        Convergence tolerance on log-likelihood change.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    at_values : dict, optional
        Variable values for ``marginal_effects='at'``.
    perfect_prediction : {'drop', 'keep'}, default 'drop'
        An indicator regressor whose non-zero rows all share one outcome
        (e.g. an industry dummy with no treated firm) predicts the outcome
        perfectly: its coefficient is infinite. ``'drop'`` removes those
        rows and omits the regressor, iterated, as Stata does ("x != 0
        predicts failure perfectly"); the omitted names and row count are
        in ``model_info``. ``'keep'`` is the pre-1.33 behaviour.

    Returns
    -------
    EconometricResults
        Fitted model with ``.summary()``, ``.predict()``, diagnostics, etc.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()  # binary `union` outcome
    >>> result = sp.probit("union ~ education + experience", data=df)
    >>> print(result.summary())  # doctest: +SKIP

    >>> # Average marginal effects with robust SE
    >>> result = sp.probit("union ~ education + experience", data=df,
    ...                    robust='hc1', marginal_effects='average')
    >>> me = result.model_info['marginal_effects']
    >>> bool('dy/dx' in me.columns)
    True
    """
    return _fit_binary(
        formula=formula,
        data=data,
        y=y,
        x=x,
        link="probit",
        robust=robust,
        cluster=cluster,
        weights=weights,
        marginal_effects=marginal_effects,
        odds_ratio=False,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
        at_values=at_values,
        perfect_prediction=perfect_prediction,
    )


@accepts_aliases(vce="robust")
@records_provenance("sp.cloglog")
@markout_clusters
def cloglog(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    marginal_effects: Optional[str] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
    at_values: Optional[Dict[str, float]] = None,
    perfect_prediction: str = "drop",
) -> EconometricResults:
    """
    Complementary log-log regression via maximum likelihood.

    Appropriate when P(Y=1) is small (rare events) or when the latent
    distribution is asymmetric (extreme value type I).

    Equivalent to Stata's ``cloglog y x1 x2``.

    Parameters
    ----------
    formula : str, optional
        Formula like ``"y ~ x1 + x2"``.
    data : pd.DataFrame
        Data containing the variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Regressor names (alternative to formula).
    robust : str, default ``'nonrobust'``
        ``'nonrobust'`` for MLE SE, ``'hc1'`` / ``'robust'`` for sandwich SE.
    cluster : str, optional
        Column name for clustered standard errors.
    weights : str, optional
        Column name for frequency/analytic weights.
    marginal_effects : str, optional
        ``'average'`` (AME), ``'mean'`` (MEM), or ``'at'`` (MER).
    maxiter : int, default 100
        Maximum Newton-Raphson iterations.
    tol : float, default 1e-8
        Convergence tolerance on log-likelihood change.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    at_values : dict, optional
        Variable values for ``marginal_effects='at'``.
    perfect_prediction : {'drop', 'keep'}, default 'drop'
        An indicator regressor whose non-zero rows all share one outcome
        (e.g. an industry dummy with no treated firm) predicts the outcome
        perfectly: its coefficient is infinite. ``'drop'`` removes those
        rows and omits the regressor, iterated, as Stata does ("x != 0
        predicts failure perfectly"); the omitted names and row count are
        in ``model_info``. ``'keep'`` is the pre-1.33 behaviour.

    Returns
    -------
    EconometricResults
        Fitted model with ``.summary()``, ``.predict()``, diagnostics, etc.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()  # binary `union` outcome
    >>> result = sp.cloglog("union ~ education + experience", data=df)
    >>> print(result.summary())  # doctest: +SKIP
    >>> bool(result.model_info['link'] == 'cloglog')
    True
    """
    return _fit_binary(
        formula=formula,
        data=data,
        y=y,
        x=x,
        link="cloglog",
        robust=robust,
        cluster=cluster,
        weights=weights,
        marginal_effects=marginal_effects,
        odds_ratio=False,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
        at_values=at_values,
        perfect_prediction=perfect_prediction,
    )
