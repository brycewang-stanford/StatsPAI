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

import functools
import re
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


_FACTOR_LHS = re.compile(
    r"^\s*(?:as\.)?(?:factor|ordered|C)\(\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\)\s*$"
)


def _bare_outcome(formula: Optional[str]) -> Optional[str]:
    """``factor(y) ~ x`` read as ``y ~ x``.

    R's ``polr`` and ``multinom`` want the outcome wrapped in
    ``factor()``; here the outcome of a categorical model is categorical
    by construction, so the wrapper carries no information. Left in
    place, patsy expanded it into indicator columns and the model saw
    two categories.
    """
    if formula is None or "~" not in formula:
        return formula
    lhs, rhs = formula.split("~", 1)
    match = _FACTOR_LHS.match(lhs)
    if match is None:
        return formula
    return f"{match.group(1)} ~{rhs}"


def _parse_inputs(
    formula: Optional[str],
    data: Optional[pd.DataFrame],
    y: Optional[str],
    x: Optional[List[str]],
) -> Tuple[str, List[str]]:
    """Resolve formula / y+x inputs into variable names."""
    formula = _bare_outcome(formula)
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


def _formula_design(
    formula: Optional[str],
    data: Optional[pd.DataFrame],
    y: Optional[str],
    x: Optional[List[str]],
    add_constant: bool,
    extra_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str], str]:
    """Design of a categorical-outcome model, dependent regressors omitted.

    See :func:`_formula_design_raw` for the parsing. A regressor that is a
    combination of earlier ones (given the constant, which in an ordered
    model is carried by the cutpoints) has no identified coefficient; it is
    omitted with a note, as Stata does, and listed in
    ``df.attrs['omitted']``.
    """
    from ..core._collinear import drop_collinear

    Y, X, df, names, y_name = _formula_design_raw(
        formula, data, y, x, add_constant, extra_cols
    )
    who = "mlogit" if add_constant else "ordered model"
    if add_constant:
        X, names, omitted, _ = drop_collinear(X, names, who, stacklevel=4)
    else:
        full = np.column_stack([np.ones(len(X)), X])
        full, kept, omitted, _ = drop_collinear(
            full, ["_cons"] + list(names), who, stacklevel=4
        )
        X, names = full[:, 1:], [v for v in kept if v != "_cons"]
    df.attrs["omitted"] = omitted
    return Y, X, df, names, y_name


def _formula_design_raw(
    formula: Optional[str],
    data: Optional[pd.DataFrame],
    y: Optional[str],
    x: Optional[List[str]],
    add_constant: bool,
    extra_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str], str]:
    """Y, X, the estimation frame, regressor names and the outcome name.

    Formulas made of bare column names keep the direct path. Anything else
    -- ``C(g)``, ``I(x**2)``, interactions -- is expanded by patsy, so
    ``oprobit("y ~ x + C(g)")`` works as ``oprobit y x i.g`` does (it used to
    fail with ``KeyError: ['C'] not in index``). Ordered models drop the
    intercept (the cutpoints absorb it); the rest call it ``_cons``.
    """
    formula = _bare_outcome(formula)
    y_name, x_names = _parse_inputs(formula, data, y, x)
    if formula is None or all(
        isinstance(v, str) and data is not None and v in data.columns for v in x_names
    ):
        Y, X, df, names = _build_matrices(
            data, y_name, x_names, add_constant=add_constant, extra_cols=extra_cols
        )
        return Y, X, df, names, y_name
    if data is None:
        raise MethodIncompatibility("'data' must be provided.")
    from ..core.utils import create_design_matrices

    extra = [c for c in (extra_cols or []) if c in data.columns]
    frame = data.dropna(subset=extra) if extra else data
    lhs = formula.split("~", 1)[0].strip()
    build = frame
    if lhs in frame.columns and not pd.api.types.is_numeric_dtype(frame[lhs]):
        # a string- or category-labelled outcome ("1", "2-4", "5+"): patsy
        # would expand it into dummies and keep the first as "the" outcome,
        # leaving two categories. Hand it integer codes; the labels are
        # read back from the column below.
        codes = pd.factorize(frame[lhs], sort=True)[0].astype(float)
        codes[codes < 0] = np.nan
        build = frame.assign(**{lhs: codes})
    y_df, X_df = create_design_matrices(formula, build)
    y_name = lhs if build is not frame else str(y_df.columns[0])
    names = [str(c) for c in X_df.columns]
    X = np.asarray(X_df, dtype=float)
    if "Intercept" in names:
        j = names.index("Intercept")
        X = np.delete(X, j, axis=1)
        names.pop(j)
        if add_constant:
            X = np.column_stack([np.ones(len(X)), X])
            names = ["_cons"] + names
    elif add_constant:
        X = np.column_stack([np.ones(len(X)), X])
        names = ["_cons"] + names
    df = frame.loc[y_df.index].copy()
    # patsy returns the outcome as float; category labels come from the
    # column itself when the left-hand side names one.
    Y = (
        df[y_name].to_numpy()
        if y_name in df.columns
        else np.asarray(y_df.iloc[:, 0].to_numpy())
    )
    if not add_constant and X.shape[1] == 0:
        raise MethodIncompatibility("The formula has no regressors.")
    return Y, X, df, names, y_name


def _design_for_new_data(
    data: pd.DataFrame, names: Tuple[str, ...], formula: Optional[str]
) -> np.ndarray:
    """Design columns ``names`` rebuilt on new data (constant excluded)."""
    wanted = [n for n in names if n != "_cons"]
    if all(n in data.columns for n in wanted):
        return np.asarray(data[wanted].to_numpy(dtype=float))
    if formula is None:
        missing = [n for n in wanted if n not in data.columns]
        raise MethodIncompatibility(
            f"predict(): columns {missing} are not in the new data.",
            diagnostics={"missing": missing},
        )
    from patsy import dmatrix

    from ..core.utils import (
        _coerce_string_extension_dtypes,
        formula_eval_env,
        r_formula_idioms,
    )

    rhs = r_formula_idioms(formula.split("~", 1)[1].strip())
    built = dmatrix(
        rhs,
        _coerce_string_extension_dtypes(data),
        eval_env=formula_eval_env(),
        return_type="dataframe",
    )
    cols = {str(c): built[c].to_numpy(dtype=float) for c in built.columns}
    missing = [n for n in wanted if n not in cols]
    if missing:
        raise MethodIncompatibility(
            f"predict(): the new data do not reproduce the model terms "
            f"{missing}; a factor is missing a level the model was fitted "
            "with, or its reference level.",
            recovery_hint=(
                "Predict on rows that include every level of each factor, "
                "or build the dummy columns yourself and fit with x=[...]."
            ),
            diagnostics={"missing": missing},
        )
    return np.column_stack([cols[n] for n in wanted])


def _bound_category_probabilities(
    kind: str,
    coef: np.ndarray,
    cuts: Optional[np.ndarray],
    categories: Tuple[Any, ...],
    names: Tuple[str, ...],
    formula: Optional[str],
    fitted: pd.DataFrame,
    data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """``result.predict`` of the multinomial and ordered models.

    One column of predicted probabilities per outcome category: the
    estimation sample's when ``data`` is omitted (Stata ``predict``, R
    ``predict(type = "probs")``), otherwise evaluated on ``data``. Kept at
    module level and bound with ``functools.partial`` so results pickle.
    """
    if data is None:
        return fitted.copy()
    X = _design_for_new_data(data, names, formula)
    if kind == "mlogit":
        if "_cons" in names:
            X = np.column_stack([np.ones(len(X)), X])
        P = _softmax(X @ coef.T)
    else:
        cdf = _ordered_logit_cdf if kind == "ologit" else _ordered_probit_cdf
        xb = X @ coef
        cum = np.column_stack(
            [np.zeros(len(X))]
            + [cdf(c - xb) for c in np.asarray(cuts, dtype=float)]
            + [np.ones(len(X))]
        )
        P = np.diff(cum, axis=1)
    return pd.DataFrame(P, columns=list(categories), index=data.index)


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
    V = _compute_vcov(score_i, H_inv, kind, cluster_vals)
    return _as_float_array(np.sqrt(np.maximum(np.diag(V), 1e-20)))


def _compute_vcov(
    score_i: np.ndarray,
    H_inv: np.ndarray,
    kind: str,
    cluster_vals: Optional[np.ndarray],
) -> np.ndarray:
    """The covariance matrix behind :func:`_compute_se`. The results keep
    it (``data_info['var_cov']``) so that joint tests and linear
    combinations work after the fit."""
    from ..core._vcov import ml_vcov
    from ._optim_helpers import require_two_clusters

    if cluster_vals is not None:
        require_two_clusters(cluster_vals, "mlogit / ologit / oprobit / clogit")
    return _as_float_array(ml_vcov(H_inv, score_i, kind=kind, clusters=cluster_vals))


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
    extra = [c for c in [cluster] if c]
    Y_raw, X, df, var_names, y_name = _formula_design(
        formula, data, y, x, add_constant=True, extra_cols=extra
    )
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

    # --- Optimise: analytic Newton-Raphson ---
    # Closed-form scores and Hessian (``_ml_newton.mlogit_newton``). The
    # BFGS + complex-step Hessian it replaces cost O(p^2) likelihood
    # evaluations: minutes with a few dozen dummies in three equations.
    from ._ml_newton import mlogit_newton
    from ._optim_helpers import inverse_information

    theta_hat, ll, S_obs, H, n_iter, converged = mlogit_newton(
        X, Y_idx, J, base, maxiter=maxiter, tol=tol
    )
    if not converged:
        warnings.warn(
            f"mlogit: Newton-Raphson did not converge in {maxiter} iterations; "
            "estimates are not an optimum. model_info['converged'] is False.",
            RuntimeWarning,
            stacklevel=3,
        )

    # Null model log-likelihood (intercept only => equal probs)
    freq = np.array([np.sum(Y_idx == j) for j in range(J)]) / n
    ll_0 = float(np.sum(Y_oh * np.log(np.maximum(freq[np.newaxis, :], 1e-300))))

    pseudo_r2 = 1.0 - ll / ll_0
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + np.log(n) * n_params

    # --- Standard errors ---
    H_inv = _as_float_array(inverse_information(H))
    se = _compute_se(S_obs, H_inv, robust, cluster_vals)
    cov_theta = _compute_vcov(S_obs, H_inv, robust, cluster_vals)

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
        # delta method for exp(beta)
        cov_theta = cov_theta * np.outer(rrr_vals, rrr_vals)
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

        # Restricted model: mlogit on the subsample over the remaining
        # categories only (``mlogit y x if y != j``). Through 1.32 the dropped
        # category stayed in the softmax as a phantom zero-utility option and
        # the variance was BFGS's ``hess_inv`` approximation.
        X_r = X[mask]
        remap = {j2: pos for pos, j2 in enumerate(restricted_cats)}
        y_r = np.array([remap[v] for v in Y_idx[mask]])
        non_base_r = [j for j in restricted_cats if j != base]
        n_params_r = len(non_base_r) * k
        theta_r, _, _, H_r, _, conv_r = mlogit_newton(
            X_r, y_r, len(restricted_cats), remap[base], maxiter=maxiter, tol=tol
        )
        if not conv_r:
            iia_skipped.append(str(categories[drop_j]))
            continue

        # Hausman statistic: (b_r - b_f)' [V_r - V_f]^{-1} (b_r - b_f)
        # Simplified: use the restricted params corresponding to non_base_r
        b_r = _as_float_array(theta_r)
        b_f = np.concatenate(
            [
                theta_hat[non_base.index(j2) * k : (non_base.index(j2) + 1) * k]
                for j2 in non_base_r
            ]
        )
        diff = b_r - b_f
        df_test = len(diff)
        try:
            V_r_mat = inverse_information(H_r)
            V_f_sub = np.zeros((n_params_r, n_params_r))
            for i1, j1 in enumerate(non_base_r):
                for i2, j2 in enumerate(non_base_r):
                    fi1 = non_base.index(j1)
                    fi2 = non_base.index(j2)
                    V_f_sub[i1 * k : (i1 + 1) * k, i2 * k : (i2 + 1) * k] = H_inv[
                        fi1 * k : (fi1 + 1) * k, fi2 * k : (fi2 + 1) * k
                    ]
            V_diff = V_r_mat - V_f_sub
            V_diff = (V_diff + V_diff.T) / 2.0
            # Generalised inverse over the non-null eigenvalues and df =
            # rank, as Stata's ``hausman`` reports (``df`` 4 for two
            # equations of three coefficients: the difference is singular).
            # A negative eigenvalue is kept, not skipped; the statistic can
            # then be negative, which Stata also prints (the asymptotic
            # assumptions fail on these data).
            w, U = np.linalg.eigh(V_diff)
            keep_w = np.abs(w) > 1e-8 * np.max(np.abs(w))
            G = (U[:, keep_w] / w[keep_w]) @ U[:, keep_w].T
            chi2 = float(diff @ G @ diff)
            df_test = int(keep_w.sum())
            p_iia = float(stats.chi2.sf(chi2, df_test)) if chi2 >= 0 else 1.0
            iia_tests[categories[drop_j]] = {
                "chi2": chi2,
                "df": df_test,
                "pvalue": p_iia,
                "v_diff_psd": bool(np.all(w[keep_w] > 0)),
            }
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
        "converged": bool(converged),
        "iterations": int(n_iter),
        "rrr": rrr,
        "robust": robust if cluster is None else f"cluster({cluster})",
        "iia_skipped": iia_skipped,
    }

    data_info = {
        "var_cov": cov_theta,
        "var_names": list(param_names),
        "dependent_var": y_name,
        "n_obs": n,
        "n_params": n_params,
        "df_resid": n - n_params,
        "nobs": n,
        "y": Y_idx,
        # Unweighted per-observation log-likelihood (sp.vuong).
        "llobs": np.log(np.maximum(P_hat[np.arange(n), Y_idx], 1e-300)),
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
    setattr(
        result,
        "predict",
        functools.partial(
            _bound_category_probabilities,
            "mlogit",
            betas_all,
            None,
            tuple(categories),
            tuple(str(v) for v in var_names),
            formula,
            pd.DataFrame(P_hat, columns=categories, index=df.index),
        ),
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
    extra = [c for c in [cluster] if c]
    Y_raw, X_no_const, df, x_names, y_name = _formula_design(
        formula, data, y, x, add_constant=False, extra_cols=extra
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

    # --- Initial values ---
    # beta = 0 and the cutpoints of the cutpoint-only model, F^{-1} of the
    # cumulative category shares (every category is observed, so each share
    # is strictly inside (0, 1) and the cutpoints are strictly increasing).
    freq_cum = np.cumsum(np.bincount(Y_idx, minlength=J))[:n_cuts] / n
    if link == "logit":
        kappa_init = np.log(freq_cum / (1.0 - freq_cum))
    else:
        kappa_init = stats.norm.ppf(freq_cum)

    # --- Optimise: analytic Newton-Raphson on (beta, kappa) ---
    # Stata's parameterisation. Scores and the observed information are
    # closed-form (``_ml_newton``); the BFGS + finite-difference fit and
    # complex-step Hessian it replaces took minutes with a few hundred
    # dummies and stopped early on unscaled regressors.
    from ._ml_newton import OrderedLikelihood, fit_ordered
    from ._optim_helpers import inverse_information

    lik = OrderedLikelihood(X_no_const, Y_idx, J, link)
    theta0 = np.concatenate([np.zeros(k), kappa_init])
    theta_hat, ll, S_obs_exact, H_exact, n_iter, converged = fit_ordered(
        lik, theta0, maxiter=maxiter, tol=tol
    )
    beta_hat, kappa_hat = theta_hat[:k], theta_hat[k:]
    if not converged:
        warnings.warn(
            f"o{link}: Newton-Raphson did not converge in {maxiter} "
            "iterations (scaled gradient above tol); estimates are not an "
            "optimum. model_info['converged'] is False.",
            RuntimeWarning,
            stacklevel=3,
        )

    # Null model (cutpoints only) is saturated in the category shares, so
    # its maximum is closed-form: sum_j n_j log(n_j / n).
    counts = np.bincount(Y_idx, minlength=J).astype(float)
    ll_0 = float(np.sum(counts[counts > 0] * np.log(counts[counts > 0] / n)))

    pseudo_r2 = 1.0 - ll / ll_0
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + np.log(n) * n_params

    # --- Standard errors ---
    # One covariance for (beta, kappa) under the requested vce. Through 1.32
    # the cutpoint SEs came from the model-based block whatever ``robust`` /
    # ``cluster`` said.
    H_inv = _as_float_array(inverse_information(H_exact))
    se_all = _compute_se(S_obs_exact, H_inv, robust, cluster_vals)
    cov_all = _compute_vcov(S_obs_exact, H_inv, robust, cluster_vals)
    se_beta = se_all[:k]
    se_kappa = se_all[k:]

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
        from ._ml_newton import logit_newton

        chi2_total = 0.0
        df_total = 0
        X_bin = np.column_stack([np.ones(n), X_no_const])
        fits = [logit_newton(X_bin, (Y_idx <= j).astype(float)) for j in range(n_cuts)]
        for m in range(k):
            beta_binary_arr = np.array([f[0][m + 1] for f in fits])
            se_binary_arr = np.sqrt(
                np.maximum([f[1][m + 1, m + 1] for f in fits], 1e-20)
            )
            # Test: all beta_j equal (Wald test)
            if n_cuts > 1 and np.all(np.isfinite(se_binary_arr)):
                beta_mean = float(np.mean(beta_binary_arr))
                chi2_m = float(
                    np.sum(((beta_binary_arr - beta_mean) / se_binary_arr) ** 2)
                )
                df_m = n_cuts - 1
                brant_test[var_names[m]] = {
                    "chi2": chi2_m,
                    "df": df_m,
                    "pvalue": float(stats.chi2.sf(chi2_m, df_m)),
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
        "converged": bool(converged),
        "iterations": int(n_iter),
        "robust": robust if cluster is None else f"cluster({cluster})",
        "brant_skipped": brant_skipped,
        "brant_error": brant_error,
    }

    data_info = {
        "var_cov": cov_all,
        "var_names": list(param_names),
        "dependent_var": y_name,
        "n_obs": n,
        "n_params": n_params,
        "df_resid": n - n_params,
        "nobs": n,
        "y": Y_idx,
        # Unweighted per-observation log-likelihood (sp.vuong).
        "llobs": np.log(np.maximum(P_hat[np.arange(n), Y_idx], 1e-300)),
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
    setattr(
        result,
        "predict",
        functools.partial(
            _bound_category_probabilities,
            f"o{link}",
            np.asarray(beta_hat, dtype=float),
            np.asarray(kappa_hat, dtype=float),
            tuple(categories),
            tuple(str(v) for v in var_names),
            formula,
            pd.DataFrame(P_hat, columns=categories, index=df.index),
        ),
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

    # Group structure. Rows are worked on in group order, so that every
    # per-group sum is one ``reduceat`` over contiguous segments instead of
    # a Python loop over groups (thousands of choice sets per likelihood
    # evaluation).
    codes_all, _ = pd.factorize(G_vals, sort=True)
    chosen_per_group = np.bincount(codes_all, weights=Y)
    # groups without exactly one chosen alternative carry no information
    valid_rows = chosen_per_group[codes_all] == 1
    if not valid_rows.any():
        raise ValueError(
            "No valid choice groups (each group needs exactly one chosen alternative)."
        )
    rows = np.flatnonzero(valid_rows)
    rows = rows[np.argsort(codes_all[rows], kind="stable")]
    Xs, Ys = X[rows], Y[rows]
    seg_codes = codes_all[rows]
    starts = np.flatnonzero(np.r_[True, seg_codes[1:] != seg_codes[:-1]])
    sizes = np.diff(np.r_[starts, len(rows)])
    member = np.repeat(np.arange(len(starts)), sizes)
    n_groups_valid = len(starts)

    def _probabilities(beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x'b, log of each group's denominator, choice probabilities)."""
        xb = Xs @ beta
        top = np.maximum.reduceat(xb, starts)
        e = np.exp(xb - top[member])
        denom = np.add.reduceat(e, starts)
        return xb, top + np.log(denom), e / denom[member]

    def neg_loglik_stable(beta: np.ndarray) -> float:
        xb, log_denom, _ = _probabilities(beta)
        return float(-(Ys @ xb - log_denom.sum()))

    def grad(beta: np.ndarray) -> np.ndarray:
        p = _probabilities(beta)[2]
        return _as_float_array(-(Xs.T @ (Ys - p)))

    def information(beta: np.ndarray) -> np.ndarray:
        # -d2 ll = sum_g sum_j p_gj (x_gj - xbar_g)(x_gj - xbar_g)', with
        # xbar_g the probability-weighted group mean
        p = _probabilities(beta)[2]
        means = np.add.reduceat(Xs * p[:, None], starts, axis=0)
        return _as_float_array((Xs * p[:, None]).T @ Xs - means.T @ means)

    def score_obs_clogit(beta: np.ndarray) -> np.ndarray:
        """Per-observation score (one row per group)."""
        p = _probabilities(beta)[2]
        return _as_float_array(np.add.reduceat(Xs * (Ys - p)[:, None], starts, axis=0))

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
    # The log likelihood is globally concave: a few Newton steps from the
    # quasi-Newton solution bring the gradient to rounding error.
    for _ in range(5):
        try:
            step = np.linalg.solve(information(beta_hat), -grad(beta_hat))
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        candidate = beta_hat + step
        if neg_loglik_stable(candidate) > neg_loglik_stable(beta_hat):
            break
        beta_hat = candidate
        if np.max(np.abs(step)) < 1e-12 * max(1.0, float(np.max(np.abs(beta_hat)))):
            break
    ll = float(-neg_loglik_stable(beta_hat))

    # Null model: beta=0 => each alternative equally likely
    ll_0 = float(-np.log(sizes).sum())

    pseudo_r2 = 1.0 - ll / ll_0
    aic = -2 * ll + 2 * k
    bic = -2 * ll + np.log(n_groups_valid) * k

    # --- Standard errors ---
    # Inverse *observed* information, analytic (see ``information``). This
    # replaces BFGS ``hess_inv``, a quasi-Newton approximation built from
    # the optimisation path rather than the information matrix.
    info = information(beta_hat)
    try:
        H_inv = _as_float_array(np.linalg.inv(info))
    except np.linalg.LinAlgError:
        H_inv = _as_float_array(np.linalg.pinv(info))

    S_obs = score_obs_clogit(beta_hat)

    # The clogit score is a per-*group* contribution, so clusters are mapped
    # to groups (groups must nest within clusters, as in Stata).
    cluster_group = None
    if cluster_vals is not None:
        cl_sorted = cluster_vals[rows]
        cl_codes = pd.factorize(cl_sorted)[0]
        cluster_group = cl_sorted[starts]
        spans = np.maximum.reduceat(cl_codes, starts) != np.minimum.reduceat(
            cl_codes, starts
        )
        if spans.any():
            bad = G_vals[rows][starts][spans][0]
            raise MethodIncompatibility(
                f"clogit: group {bad!r} spans more than one {cluster!r} "
                "cluster; groups must be nested within clusters.",
                recovery_hint=("Cluster on a variable that is constant within group."),
            )
    se = _compute_se(S_obs, H_inv, robust, cluster_group)
    cov = _compute_vcov(S_obs, H_inv, robust, cluster_group)

    # --- Predicted choice probabilities ---
    pred_probs = np.zeros(n)
    pred_probs[rows] = _probabilities(beta_hat)[2]

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
        "var_cov": cov,
        "var_names": list(var_names),
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
