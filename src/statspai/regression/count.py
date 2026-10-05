"""
Count data models: Poisson, Negative Binomial, and PPML.

Implements native numpy/scipy MLE estimation with robust and clustered standard errors,
following Stata-like API conventions for applied econometric work.

References
----------
- Cameron, A.C. & Trivedi, P.K. (2013). Regression Analysis of Count Data. 2nd ed.
- Santos Silva, J.M.C. & Tenreyro, S. (2006). "The Log of Gravity." REStat.
- Correia, S., Guimaraes, P. & Zylkin, T. (2020). "Fast Poisson estimation with
  high-dimensional fixed effects." Stata Journal. [@cameron2013regression]
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .._aliases import accepts_aliases
from ..core._vcov_spec import markout_clusters
from ..core.results import EconometricResults
from ..core.utils import parse_formula
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability
from ..output._lineage import records_provenance

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_COUNT_ALTERNATIVES = ["sp.poisson", "sp.nbreg", "sp.ppmlhdfe", "sp.xtnbreg"]


def _require_count_dataframe(data: Any, context: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            f"{context} requires data to be a pandas DataFrame.",
            recovery_hint="Pass the model data with data=<DataFrame>.",
            diagnostics={"context": context, "type": type(data).__name__},
            alternative_functions=_COUNT_ALTERNATIVES,
        )
    if data.empty:
        raise DataInsufficient(
            f"{context} received an empty DataFrame.",
            recovery_hint="Provide a non-empty count-regression sample.",
            diagnostics={"context": context, "n_rows": 0},
            alternative_functions=_COUNT_ALTERNATIVES,
        )
    return data


def _require_column_name(name: Any, role: str) -> str:
    if not isinstance(name, str) or not name:
        raise MethodIncompatibility(
            f"{role} must be a non-empty column-name string.",
            recovery_hint="Pass column names as strings.",
            diagnostics={"role": role, "value": repr(name)},
            alternative_functions=_COUNT_ALTERNATIVES,
        )
    return name


def _normalize_column_list(columns: Any, role: str) -> List[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        raw = [columns]
    else:
        try:
            raw = list(columns)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"{role} must be a sequence of column-name strings.",
                recovery_hint=f"Pass {role} as ['x1', 'x2'] or a single string.",
                diagnostics={"role": role, "type": type(columns).__name__},
                alternative_functions=_COUNT_ALTERNATIVES,
            ) from exc
    return [
        _require_column_name(column, f"{role}[{idx}]") for idx, column in enumerate(raw)
    ]


def _require_columns(data: pd.DataFrame, columns: Sequence[str], role: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"{role} references missing column(s): {missing}.",
            recovery_hint="Check spelling or rename columns before fitting the "
            "count model.",
            diagnostics={
                "role": role,
                "missing_columns": missing,
                "available_columns": [str(column) for column in data.columns],
            },
            alternative_functions=_COUNT_ALTERNATIVES,
        )


def _numeric_column(data: pd.DataFrame, column: str, role: str) -> np.ndarray:
    _require_columns(data, [column], role)
    try:
        values = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{role} column {column!r} must be numeric.",
            recovery_hint="Convert the column to numeric values before fitting.",
            diagnostics={"role": role, "column": column},
            alternative_functions=_COUNT_ALTERNATIVES,
        ) from exc
    if not np.all(np.isfinite(values)):
        raise NumericalInstability(
            f"{role} column {column!r} contains NaN or infinite values.",
            recovery_hint="Drop, impute, or recode non-finite values.",
            diagnostics={
                "role": role,
                "column": column,
                "nonfinite_count": int((~np.isfinite(values)).sum()),
            },
            alternative_functions=_COUNT_ALTERNATIVES,
        )
    return np.asarray(values, dtype=np.float64)


def _positive_exposure(data: pd.DataFrame, exposure: str) -> np.ndarray:
    values = _numeric_column(data, exposure, "exposure")
    if np.any(values <= 0):
        raise MethodIncompatibility(
            "exposure must be strictly positive.",
            recovery_hint="Drop or recode non-positive exposure values before "
            "fitting a log-link count model.",
            diagnostics={
                "column": exposure,
                "nonpositive_count": int((values <= 0).sum()),
            },
            alternative_functions=_COUNT_ALTERNATIVES,
        )
    return values


_PATSY_TOKENS = ("(", ":", "*", "**", "/")


def _needs_patsy(formula: str) -> bool:
    """True when the RHS uses factor / transform / interaction syntax.

    The count models' own parser only understands ``y ~ a + b [| fe]``; it
    used to read ``C(g)`` as a column called ``C``. Plain formulas keep the
    original path byte for byte.
    """
    rhs = formula.split("~", 1)[1].split("|", 1)[0] if "~" in formula else ""
    return any(tok in rhs for tok in _PATSY_TOKENS)


def _patsy_count_design(
    formula: str, data: pd.DataFrame, add_constant: bool, fe_vars: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[str], str, List[str], pd.DataFrame]:
    """Design for ``C()`` / ``I()`` / interaction formulas via patsy.

    The intercept keeps the count models' ``_cons`` name; rows patsy drops
    for missing values are dropped from the returned ``data`` too, so
    offset / exposure / cluster columns stay aligned.
    """
    from ..core.utils import create_design_matrices

    lhs, rhs = formula.split("~", 1)
    rhs = rhs.split("|", 1)[0].strip()
    if not add_constant and "-1" not in rhs.replace(" ", ""):
        rhs = rhs + " - 1"
    y_df, X_df = create_design_matrices(f"{lhs.strip()} ~ {rhs}", data)
    var_names = ["_cons" if c == "Intercept" else str(c) for c in X_df.columns]
    kept = data.loc[X_df.index] if len(X_df) != len(data) else data
    return (
        y_df.to_numpy(dtype=np.float64).ravel(),
        X_df.to_numpy(dtype=np.float64),
        var_names,
        str(y_df.columns[0]),
        list(fe_vars),
        kept,
    )


def _complete_cases(data: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Rows with the outcome and every regressor observed (Stata's rule).

    The plain-column path used to pass missing values straight into the
    IRLS, which then failed with ``LinAlgError: SVD did not converge``; the
    patsy path (``C()`` / ``I()`` / interactions) already dropped them. The
    returned frame is what offsets, weights and cluster keys are read
    from, so they stay aligned with the design.
    """
    cols = list(dict.fromkeys(columns))
    observed = data[cols].notna().all(axis=1)
    if bool(observed.all()):
        return data
    if not bool(observed.any()):
        raise DataInsufficient(
            "count regression: no row has the outcome and every regressor " "observed.",
            recovery_hint="Check the columns for missing values: "
            + ", ".join(map(str, cols)),
            diagnostics={"columns": cols},
        )
    return data.loc[observed]


def _parse_formula_or_xy(
    formula: Optional[str],
    data: Any,
    y: Optional[str],
    x: Any,
    add_constant: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], str, List[str], pd.DataFrame]:
    """Parse formula/data or y/x into arrays + variable names."""
    if formula is not None and data is not None:
        data = _require_count_dataframe(data, "count regression")
        parsed = parse_formula(formula)
        dep_var = parsed["dependent"]
        indep_vars = parsed["exogenous"]
        fe_vars = parsed.get("fixed_effects", [])
        has_constant = parsed["has_constant"]
        if _needs_patsy(formula) and dep_var in data.columns:
            return _patsy_count_design(formula, data, add_constant, fe_vars)
        _require_columns(data, [dep_var, *indep_vars], "formula")
        data = _complete_cases(data, [dep_var, *indep_vars])

        y_arr = data[dep_var].values.astype(np.float64)
        X_cols = indep_vars
        X_arr_parts = [data[v].values.astype(np.float64) for v in X_cols]
        if has_constant and add_constant:
            X_arr_parts = [np.ones(len(data))] + X_arr_parts
            var_names = ["_cons"] + X_cols
        else:
            var_names = list(X_cols)
        X_arr = np.column_stack(X_arr_parts) if X_arr_parts else np.ones((len(data), 1))
        if not X_arr_parts:
            var_names = ["_cons"]
        return y_arr, X_arr, var_names, dep_var, fe_vars, data
    elif y is not None and x is not None and data is not None:
        data = _require_count_dataframe(data, "count regression")
        dep_var = _require_column_name(y, "y")
        X_cols = _normalize_column_list(x, "x")
        _require_columns(data, [dep_var, *X_cols], "y/x")
        data = _complete_cases(data, [dep_var, *X_cols])
        y_arr = data[dep_var].values.astype(np.float64)
        X_arr_parts = [data[v].values.astype(np.float64) for v in X_cols]
        if add_constant:
            X_arr_parts = [np.ones(len(data))] + X_arr_parts
            var_names = ["_cons"] + X_cols
        else:
            var_names = list(X_cols)
        X_arr = np.column_stack(X_arr_parts) if X_arr_parts else np.ones((len(data), 1))
        if not X_arr_parts:
            var_names = ["_cons"]
        return y_arr, X_arr, var_names, dep_var, [], data
    else:
        raise MethodIncompatibility(
            "Must provide either (formula, data) or (y, x, data).",
            recovery_hint="Call sp.nbreg('y ~ x', data=df) or "
            "sp.nbreg(y='y', x=['x'], data=df).",
            diagnostics={
                "has_formula": formula is not None,
                "has_data": data is not None,
                "has_y": y is not None,
                "has_x": x is not None,
            },
            alternative_functions=_COUNT_ALTERNATIVES,
        )


def _append_fixed_effect_dummies(
    X: np.ndarray,
    var_names: List[str],
    data: pd.DataFrame,
    fe_vars: Sequence[str],
) -> tuple[np.ndarray, List[str], Dict[str, int]]:
    """Append one-hot fixed-effect columns using a dropped baseline level.

    ``nbreg`` is a nonlinear model, so we cannot absorb fixed effects with
    the within transformation used by OLS. The explicit-dummy route is the
    conservative implementation: transparent, correct for moderate panels,
    and never silently ignores a ``| id`` formula component.
    """
    clean_fe = [str(v).strip() for v in fe_vars if str(v).strip()]
    if not clean_fe:
        return X, list(var_names), {}

    blocks = [X]
    names = list(var_names)
    level_counts: Dict[str, int] = {}

    for fe in clean_fe:
        _require_columns(data, [fe], "fixed effects")
        if data[fe].isna().any():
            raise MethodIncompatibility(
                f"fixed-effect column {fe!r} contains missing values; "
                "drop or impute them before fitting a fixed-effects count model",
                recovery_hint="Drop or impute missing fixed-effect identifiers.",
                diagnostics={
                    "column": fe,
                    "missing_count": int(data[fe].isna().sum()),
                },
                alternative_functions=_COUNT_ALTERNATIVES,
            )

        levels = int(data[fe].nunique(dropna=False))
        level_counts[fe] = levels
        if levels <= 1:
            continue

        dummies = pd.get_dummies(data[fe], drop_first=True, dtype=np.float64)
        if dummies.shape[1] == 0:
            continue
        blocks.append(dummies.to_numpy(dtype=np.float64))
        names.extend([f"C({fe})[{level}]" for level in dummies.columns])

    return np.column_stack(blocks), names, level_counts


def _safe_exp(eta: np.ndarray, cap: float = 700.0) -> np.ndarray:
    """Exponentiate with overflow protection."""
    return np.asarray(np.exp(np.clip(eta, -cap, cap)), dtype=np.float64)


def _obs_weighted(values: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    """``values`` times the observation weights (unchanged without weights)."""
    if weights is None:
        return values
    return np.asarray(values, dtype=np.float64) * np.asarray(weights, dtype=np.float64)


def _sandwich_vcov(
    X: np.ndarray,
    mu: np.ndarray,
    residuals: np.ndarray,
    XtX_inv_bread: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Robust (HC0) sandwich variance-covariance.

    ``weights`` are the observation weights of a weighted fit: they scale
    the information (``w * mu``) and the score (``w * (y - mu)``).
    """
    n, k = X.shape
    W = _obs_weighted(mu, weights)  # Poisson weight
    if XtX_inv_bread is None:
        XtWX = X.T @ (X * W[:, None])
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
    else:
        XtWX_inv = XtX_inv_bread

    # Meat via the canonical core sandwich (CLAUDE.md §4); bread is the
    # GLM-weighted (X'WX)^{-1}. Byte-identical to the prior HC0 sandwich.
    from ..core._vcov import sandwich_vcov

    return sandwich_vcov(
        XtWX_inv,
        X * _obs_weighted(residuals, weights)[:, None],
        correction="none",
    )


def _cluster_vcov(
    X: np.ndarray,
    mu: np.ndarray,
    residuals: np.ndarray,
    cluster_arr: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Clustered sandwich variance-covariance."""
    from ..core._vcov import sandwich_vcov

    W = _obs_weighted(mu, weights)
    residuals = _obs_weighted(residuals, weights)
    XtWX = X.T @ (X * W[:, None])
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)

    # correction='cgm' = n_clusters/(n_clusters-1). Byte-identical for G>=2.
    return sandwich_vcov(
        XtWX_inv, X * residuals[:, None], clusters=cluster_arr, correction="cgm"
    )


def _twoway_cluster_vcov(
    X: np.ndarray,
    mu: np.ndarray,
    residuals: np.ndarray,
    c1_arr: np.ndarray,
    c2_arr: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Two-way cluster sandwich (Cameron-Gelbach-Miller 2011).

    ``V = (G_min/(G_min-1)) · bread · (M1 + M2 - M12) · bread`` on the
    FE-residualised (``X_dm``) design, where each ``M_g`` is the *uncorrected*
    cluster meat ``Σ_g (Σ_{i∈g} X_i r_i)(·)'`` and ``G_min = min(G1, G2)``.
    Byte-identical to Stata ``ppmlhdfe ..., cluster(a b)`` (which applies the
    single ``G_min/(G_min-1)`` small-sample factor to the inclusion-exclusion
    meat — the one-way ``cluster(a)`` path reduces to ``G/(G-1)`` and matches
    ``_cluster_vcov`` exactly).
    """
    W = _obs_weighted(mu, weights)
    XtWX = X.T @ (X * W[:, None])
    try:
        bread = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(XtWX)

    scores = X * _obs_weighted(residuals, weights)[:, None]
    k = X.shape[1]

    def _meat(codes: np.ndarray) -> np.ndarray:
        m = np.zeros((k, k))
        for c in np.unique(codes):
            s = scores[codes == c].sum(axis=0)
            m += np.outer(s, s)
        return m

    a_codes = pd.factorize(c1_arr)[0]
    b_codes = pd.factorize(c2_arr)[0]
    c12 = a_codes * (int(b_codes.max()) + 1) + b_codes
    g_min = min(len(np.unique(c1_arr)), len(np.unique(c2_arr)))
    meat = _meat(c1_arr) + _meat(c2_arr) - _meat(c12)
    vcov = bread @ meat @ bread * (g_min / (g_min - 1.0))
    return vcov


def _poisson_vcov(
    X: np.ndarray,
    mu: np.ndarray,
    residuals: np.ndarray,
    robust: str,
    cluster_arr: Optional[np.ndarray],
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Variance-covariance for Poisson-family models.

    ``weights`` are the observation weights the coefficients were fitted
    with. Until 1.36.0 they were left out here, so a weighted fit paired
    weighted coefficients with the unweighted covariance formula.

    ``robust`` is a canonical SE kind (see ``core._vcov_spec``):

    * ``"nonrobust"`` -- model-based ``(X'WX)^{-1}`` (Var(y) = mu);
    * ``"robust"`` -- Stata ``poisson, vce(robust)``: HC0 times N/(N-1);
    * ``"hc0"`` / ``"hc1"`` -- unscaled and N/(N-K) sandwiches;
    * any kind with ``cluster_arr`` -- Stata ``vce(cluster)``, G/(G-1).

    ``"robust"`` used to be the unscaled HC0, and ``"hc2"`` / ``"hc3"``
    silently returned HC0 as well; both now follow the definitions above
    (hc2/hc3 raise, since no leverage-adjusted Poisson sandwich exists here).
    """
    n, k = X.shape
    if cluster_arr is not None:
        return _cluster_vcov(X, mu, residuals, cluster_arr, weights=weights)

    W = _obs_weighted(mu, weights)
    XtWX = X.T @ (X * W[:, None])
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(XtWX)

    kind = str(robust).lower()
    if kind == "nonrobust":
        return XtWX_inv
    if kind in ("robust", "hc0", "hc1"):
        vcov = _sandwich_vcov(X, mu, residuals, XtWX_inv, weights=weights)
        if kind == "robust":
            vcov = vcov * (n / (n - 1.0))
        elif kind == "hc1":
            vcov = vcov * (n / (n - k))
        return vcov
    raise MethodIncompatibility(
        f"Poisson: Unknown robust option: {robust!r}. Use 'nonrobust', "
        "'robust', 'hc0', 'hc1', or a cluster variable.",
        recovery_hint="Use vce='robust' or vce='cluster <var>'.",
    )


def _poisson_loglik(
    y: np.ndarray, mu: np.ndarray, weights: Optional[np.ndarray] = None
) -> float:
    """Poisson log-likelihood; the weighted sum when ``weights`` is given."""
    return float(np.sum(_obs_weighted(_poisson_llobs(y, mu), weights)))


def _poisson_llobs(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Per-observation Poisson log-likelihood ``y log(mu) - mu - log(y!)``."""
    return np.asarray(y * np.log(np.maximum(mu, 1e-300)) - mu - special.gammaln(y + 1))


# ---------------------------------------------------------------------------
# Poisson IRLS
# ---------------------------------------------------------------------------


def _poisson_irls(
    y: np.ndarray,
    X: np.ndarray,
    offset: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    """
    Poisson regression via Iteratively Reweighted Least Squares.

    Returns (beta, mu, converged, n_iter).
    """
    n, k = X.shape
    if offset is None:
        offset = np.zeros(n)
    if weights is None:
        weights = np.ones(n)

    # Initialize with log(y + 0.5)
    y_init = np.where(y > 0, y, 0.5)
    beta = np.linalg.lstsq(X, np.log(y_init) - offset, rcond=None)[0]

    converged = False
    for it in range(maxiter):
        eta = X @ beta + offset
        mu = _safe_exp(eta)

        # Working variable and weights
        w = weights * mu
        z = eta + (y - mu) / mu - offset  # working response without offset

        # Weighted least squares: (X'WX)^-1 X'Wz
        XtW = X.T * w[None, :]
        XtWX = XtW @ X
        XtWz = XtW @ z
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

        # Check convergence on relative parameter change
        delta = np.max(np.abs(beta_new - beta) / (np.abs(beta) + 1e-12))
        beta = beta_new
        if delta < tol:
            converged = True
            break

    eta = X @ beta + offset
    mu = _safe_exp(eta)
    return beta, mu, converged, it + 1


# ---------------------------------------------------------------------------
# Negative Binomial MLE
# ---------------------------------------------------------------------------


def _nb2_loglik(
    y: np.ndarray, mu: np.ndarray, alpha: float, weights: Optional[np.ndarray] = None
) -> float:
    """NB2 log-likelihood: Var(y) = mu + alpha * mu^2 (weighted sum with weights)."""
    return float(np.sum(_obs_weighted(_nb2_llobs(y, mu, alpha), weights)))


def _nb2_llobs(y: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """Per-observation NB2 log-likelihood."""
    r = 1.0 / max(alpha, 1e-300)
    # lgamma(y + r) - lgamma(r) - lgamma(y+1)
    # plus r*log(r/(r+mu)) + y*log(mu/(r+mu)).
    return np.asarray(
        special.gammaln(y + r)
        - special.gammaln(r)
        - special.gammaln(y + 1)
        + r * np.log(r / (r + mu))
        + y * np.log(np.maximum(mu, 1e-300) / (r + mu))
    )


def _nb1_loglik(
    y: np.ndarray, mu: np.ndarray, delta: float, weights: Optional[np.ndarray] = None
) -> float:
    """NB1 log-likelihood: Var(y) = mu + delta * mu  =>  Var = mu*(1+delta)."""
    return float(np.sum(_obs_weighted(_nb1_llobs(y, mu, delta), weights)))


def _nb1_llobs(y: np.ndarray, mu: np.ndarray, delta: float) -> np.ndarray:
    """Per-observation NB1 log-likelihood."""
    # Parameterize as r = mu/delta  so Var = mu + delta*mu
    delta = max(delta, 1e-300)
    r = mu / delta
    return np.asarray(
        special.gammaln(y + r)
        - special.gammaln(r)
        - special.gammaln(y + 1)
        + r * np.log(r / (r + mu))
        + y * np.log(np.maximum(mu, 1e-300) / (r + mu))
    )


def _nb2_fit(
    y: np.ndarray,
    X: np.ndarray,
    offset: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float, bool, int]:
    """
    NB2 via iterating: Poisson IRLS for beta given alpha, then profile
    likelihood optimization for alpha.

    Returns (beta, mu, alpha, converged, n_iter).
    """
    n, k = X.shape
    if offset is None:
        offset = np.zeros(n)
    if weights is None:
        weights = np.ones(n)

    # Start with Poisson estimates
    beta, mu, _, _ = _poisson_irls(y, X, offset, weights, maxiter=50, tol=1e-6)

    # Initial alpha from moment estimator
    resid = y - mu
    pearson = np.sum(resid**2 / mu) / (n - k)
    alpha = max((pearson - 1) / (np.mean(mu)), 0.01)

    converged = False
    for outer in range(maxiter):
        # Given alpha, IRLS for beta with NB2 weights
        for inner in range(maxiter):
            eta = X @ beta + offset
            mu = _safe_exp(eta)

            # NB2 weight: mu / (1 + alpha*mu)
            w = weights * mu / (1 + alpha * mu)
            z = eta + (y - mu) / mu - offset

            XtW = X.T * w[None, :]
            XtWX = XtW @ X
            XtWz = XtW @ z
            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

            delta_b = np.max(np.abs(beta_new - beta) / (np.abs(beta) + 1e-12))
            beta = beta_new
            if delta_b < tol:
                break

        eta = X @ beta + offset
        mu = _safe_exp(eta)

        # Profile likelihood for alpha
        def neg_profile_ll(log_alpha: float) -> float:
            a = float(np.exp(log_alpha))
            return -_nb2_loglik(y, mu, a)

        res = optimize.minimize_scalar(
            neg_profile_ll,
            bounds=(np.log(1e-8), np.log(1e4)),
            method="bounded",
        )
        alpha_new = np.exp(res.x)

        if abs(alpha_new - alpha) / (alpha + 1e-12) < tol:
            converged = True
            alpha = alpha_new
            break
        alpha = alpha_new

    eta = X @ beta + offset
    mu = _safe_exp(eta)
    return beta, mu, alpha, converged, outer + 1


def _nb1_fit(
    y: np.ndarray,
    X: np.ndarray,
    offset: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    maxiter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float, bool, int]:
    """NB1 fit: Var(y) = mu * (1 + delta)."""
    n, k = X.shape
    if offset is None:
        offset = np.zeros(n)
    if weights is None:
        weights = np.ones(n)

    beta, mu, _, _ = _poisson_irls(y, X, offset, weights, maxiter=50, tol=1e-6)
    resid = y - mu
    pearson = np.sum(resid**2 / mu) / (n - k)
    delta = max(pearson - 1, 0.01)

    converged = False
    for outer in range(maxiter):
        for inner in range(maxiter):
            eta = X @ beta + offset
            mu = _safe_exp(eta)

            w = weights * mu / (1 + delta)
            z = eta + (y - mu) / mu - offset

            XtW = X.T * w[None, :]
            XtWX = XtW @ X
            XtWz = XtW @ z
            try:
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

            delta_b = np.max(np.abs(beta_new - beta) / (np.abs(beta) + 1e-12))
            beta = beta_new
            if delta_b < tol:
                break

        eta = X @ beta + offset
        mu = _safe_exp(eta)

        def neg_profile_ll(log_delta: float) -> float:
            d = float(np.exp(log_delta))
            return -_nb1_loglik(y, mu, d)

        res = optimize.minimize_scalar(
            neg_profile_ll,
            bounds=(np.log(1e-8), np.log(1e4)),
            method="bounded",
        )
        delta_new = np.exp(res.x)

        if abs(delta_new - delta) / (delta + 1e-12) < tol:
            converged = True
            delta = delta_new
            break
        delta = delta_new

    eta = X @ beta + offset
    mu = _safe_exp(eta)
    return beta, mu, delta, converged, outer + 1


# ---------------------------------------------------------------------------
# PPML helpers: fixed-effect absorption via alternating projection
# ---------------------------------------------------------------------------


def _fe_codes(fe_idx: np.ndarray) -> Tuple[np.ndarray, int]:
    """Dense 0..G-1 integer codes for one fixed-effect dimension."""
    codes = pd.factorize(np.asarray(fe_idx))[0].astype(np.intp, copy=False)
    return codes, int(codes.max()) + 1 if codes.size else 0


def _fe_weighted_demean(
    M: np.ndarray,
    fe_indices_list: Sequence[np.ndarray],
    w: np.ndarray,
    maxiter: int = 500,
    tol: float = 1e-10,
) -> np.ndarray:
    """``w``-weighted within-transform of ``M`` by alternating projections.

    Each sweep subtracts the ``w``-weighted group mean of every column, one
    fixed-effect dimension at a time, until the largest group mean removed
    in a sweep is below ``tol``.  Groups within a dimension are disjoint, so
    the per-dimension step is computed for all groups at once with
    ``np.bincount`` -- O(n) per sweep instead of the O(n * G) of a
    per-group boolean-mask loop, which made ``sp.ppmlhdfe`` thousands of
    times slower than ``sp.fepois`` on panels with tens of thousands of
    units.  With one dimension a single sweep is exact.
    """
    Md = np.array(M, dtype=float, copy=True)
    squeeze = Md.ndim == 1
    if squeeze:
        Md = Md[:, None]
    w = np.asarray(w, dtype=float)
    if len(fe_indices_list) >= 2 and Md.shape[1] and np.all(w > 0):
        # Several dimensions: the plain sweep converges linearly and, with
        # five crossed effects on 10^5 rows, still had not reached ``tol``
        # after 500 sweeps (seconds per call, and the result stopped short
        # of the projection).  Reuse the shared accelerated kernel of
        # ``sp.fast.fepois`` -- fused numba (or Rust) sweeps with
        # Irons-Tuck extrapolation, stopping when a sweep moves no entry by
        # more than ``tol`` times the column's scale.
        from ..fast.fepois import _weighted_ap_demean

        codes_list, counts_list = [], []
        for fe_idx in fe_indices_list:
            codes, G = _fe_codes(fe_idx)
            codes_list.append(np.ascontiguousarray(codes, dtype=np.int64))
            counts_list.append(np.bincount(codes, minlength=G))
        out, _iters, _conv = _weighted_ap_demean(
            np.asfortranarray(Md),
            codes_list,
            counts_list,
            np.ascontiguousarray(w),
            max_iter=max(int(maxiter), 10_000),
            tol=float(tol),
        )
        out = np.asarray(out, dtype=float)
        return out[:, 0] if squeeze else out
    blocks = []
    for fe_idx in fe_indices_list:
        codes, G = _fe_codes(fe_idx)
        wsum = np.bincount(codes, weights=w, minlength=G)
        ok = wsum >= 1e-300
        inv = np.zeros(G)
        inv[ok] = 1.0 / wsum[ok]
        blocks.append((codes, G, inv))
    n_sweeps = 1 if len(blocks) == 1 else maxiter
    for _ in range(n_sweeps):
        max_change = 0.0
        for codes, G, inv in blocks:
            for c in range(Md.shape[1]):
                means = np.bincount(codes, weights=w * Md[:, c], minlength=G) * inv
                Md[:, c] -= means[codes]
                if means.size:
                    max_change = max(max_change, float(np.max(np.abs(means))))
        if max_change < tol:
            break
    return Md[:, 0] if squeeze else Md


def _drop_nested_fe(fe_indices_list: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Fixed-effect dimensions minus those nested in another one.

    If every level of dimension B lies inside one level of A (``id`` inside
    ``id x cohort`` in a stacked regression, or a duplicated dimension), A's
    dummies are in B's span: the within-transform is the same without A,
    but alternating projections over redundant dimensions converge far
    more slowly.  Used for the numerical projection only; degree-of-freedom
    bookkeeping keeps every dimension.
    """
    codes = [_fe_codes(fe)[0] for fe in fe_indices_list]
    counts = [int(c.max()) + 1 if c.size else 0 for c in codes]
    keep = [True] * len(codes)
    for a in range(len(codes)):
        for b in range(len(codes)):
            if a == b or not keep[b] or not keep[a]:
                continue
            # A nested in B: each B level maps to a single A level.
            pairs = np.unique(codes[b].astype(np.int64) * max(counts[a], 1) + codes[a])
            if pairs.size == counts[b] and (counts[a] < counts[b] or a > b):
                keep[a] = False
                break
    return [fe for fe, k in zip(fe_indices_list, keep) if k]


def _normalise_ppml_separation(separation: Union[bool, str]) -> str:
    """Map the public ``separation=`` spellings to 'drop' / 'warn' / 'none'."""
    if separation is True:
        return "drop"
    if separation is False or separation is None:
        return "none"
    key = str(separation).strip().lower()
    if key in {"drop", "default", "def", "fe simplex", "fe", "simplex", "on"}:
        return "drop"
    if key in {"warn", "check"}:
        return "warn"
    if key in {"none", "off", "false", "keep"}:
        return "none"
    raise MethodIncompatibility(
        f"separation={separation!r} is not recognised.",
        recovery_hint="Use True/'drop' (default), 'warn', or False.",
        diagnostics={"separation": separation},
    )


def _ppml_separation_mask(
    y: np.ndarray,
    X: np.ndarray,
    fe_indices_list: Optional[Sequence[np.ndarray]] = None,
    maxiter: int = 100,
    drop_singletons: bool = False,
    separation: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Observations PPML can keep after removing separated ones.

    Implements the two cheap, exact rules that Stata ``ppmlhdfe`` applies
    by default (``correia2020fast``; ``separation(fe simplex)``), iterated
    to a fixed point because dropping rows under one rule can create
    separation under the other:

    * **fe** -- a fixed-effect group whose outcomes are all zero.  Its
      effect diverges to minus infinity, the group contributes nothing to
      the slope estimates, and keeping it only stalls IRLS.
    * **simplex (single regressor)** -- a regressor that is zero on every
      ``y > 0`` observation and one-signed on the ``y = 0`` ones: the
      rows where it is non-zero are perfectly predicted zeros.

    The general ReLU rule (separation by a linear combination of several
    regressors) is not implemented; such cases still surface as
    non-convergence.

    With ``drop_singletons=True`` a third rule joins the fixed point, as in
    Stata ``ppmlhdfe`` (and ``reghdfe``): an observation alone in its group
    of any fixed-effect dimension.  Its fitted value equals its outcome, so
    it carries no information about the slopes, but it inflates ``N`` and
    the cluster count.  All three rules are monotone (removing rows never
    undoes one), so the fixed point does not depend on their order; they
    run singletons first, as Stata reports them.  ``separation=False``
    applies the singleton rule alone.

    Returns ``(keep, counts)`` where ``counts`` holds how many observations
    each rule removed.
    """
    keep = np.isfinite(y)
    counts = {"singleton": 0, "fe": 0, "simplex": 0}
    fe_list = list(fe_indices_list or [])
    codes_list = [_fe_codes(fe)[0] for fe in fe_list]
    for _ in range(maxiter):
        changed = False
        if drop_singletons:
            for codes in codes_list:
                G = int(codes.max()) + 1 if codes.size else 0
                nobs = np.bincount(codes[keep], minlength=G)
                drop = keep & (nobs[codes] == 1)
                if drop.any():
                    counts["singleton"] += int(drop.sum())
                    keep &= ~drop
                    changed = True
        if not separation:
            if not changed:
                break
            continue
        for codes in codes_list:
            G = int(codes.max()) + 1 if codes.size else 0
            ysum = np.bincount(codes[keep], weights=y[keep], minlength=G)
            nobs = np.bincount(codes[keep], minlength=G)
            dead = (ysum <= 0) & (nobs > 0)
            if dead.any():
                drop = keep & dead[codes]
                counts["fe"] += int(drop.sum())
                keep &= ~drop
                changed = True
        pos = keep & (y > 0)
        zero = keep & (y == 0)
        for j in range(X.shape[1]):
            col = X[:, j]
            nz = zero & (col != 0)
            if not nz.any() or np.any(col[pos] != 0):
                continue
            vals = col[nz]
            if np.all(vals > 0) or np.all(vals < 0):
                counts["simplex"] += int(nz.sum())
                keep &= ~nz
                zero &= ~nz
                changed = True
        if not changed:
            break
    return keep, counts


def _demean_poisson(
    y: np.ndarray,
    X: np.ndarray,
    fe_indices_list: Sequence[np.ndarray],
    mu: np.ndarray,
    maxiter_demean: int = 500,
    tol_demean: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Demean X (weighted by mu) by absorbing high-dimensional fixed effects
    via alternating projection (Gauss-Seidel on normal equations).

    Parameters
    ----------
    y : ndarray (n,)
    X : ndarray (n, k)
    fe_indices_list : list of ndarray, each (n,) integer-coded FE groups
    mu : ndarray (n,)
        Current Poisson fitted values (used as weights).

    Returns
    -------
    y_tilde, X_tilde : demeaned arrays
    """
    w = mu
    Z = np.column_stack([y.reshape(-1, 1), X]) * w[:, None]
    Z_dm = _fe_weighted_demean(
        Z, fe_indices_list, w, maxiter=maxiter_demean, tol=tol_demean
    )
    return Z_dm[:, 0], Z_dm[:, 1:]


def _detect_separation(
    y: np.ndarray,
    X: np.ndarray,
    fe_indices_list: Optional[Sequence[np.ndarray]] = None,
) -> List[str]:
    """
    Simple separation detection for PPML.

    Checks for regressors that perfectly predict y=0 (i.e., whenever x_j > 0,
    all observations have y=0, or vice versa). Also checks if any FE group has
    all zeros.

    Returns list of warning messages.
    """
    warnings_list = []
    zero_mask = y == 0

    # Check regressors
    for j in range(X.shape[1]):
        col = X[:, j]
        # Skip constant
        if np.all(col == col[0]):
            continue
        # Check if positive values of x perfectly predict y=0
        pos_mask = col > 0
        if pos_mask.sum() > 0 and np.all(zero_mask[pos_mask]):
            warnings_list.append(
                f"Possible separation: column {j} > 0 perfectly predicts y=0"
            )

    # Check FE groups
    if fe_indices_list:
        for fe_i, fe_idx in enumerate(fe_indices_list):
            codes, G = _fe_codes(fe_idx)
            ysum = np.bincount(codes, weights=np.abs(y), minlength=G)
            n_dead = int((ysum == 0).sum())
            if n_dead:
                warnings_list.append(
                    f"Separation: {n_dead} FE group(s) in FE dim {fe_i} have "
                    "all-zero outcomes"
                )

    return warnings_list


# ---------------------------------------------------------------------------
# PPML with HDFE via IRLS + alternating projection
# ---------------------------------------------------------------------------


def _ppml_hdfe_irls(
    y: np.ndarray,
    X: np.ndarray,
    fe_indices_list: Optional[Sequence[np.ndarray]] = None,
    weights: Optional[np.ndarray] = None,
    maxiter: int = 1000,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, bool, int, Optional[np.ndarray]]:
    """
    PPML estimation with high-dimensional fixed effects.

    If fe_indices_list is empty/None, reduces to standard Poisson IRLS.
    Otherwise, absorbs FEs via within-transformation at each IRLS step.

    Returns (beta, mu, converged, n_iter).
    """
    n, k = X.shape
    fe_list = _drop_nested_fe(list(fe_indices_list or []))
    has_fe = len(fe_list) > 0
    if weights is None:
        weights = np.ones(n)

    # Initialize
    y_init = np.where(y > 0, y, 0.5)
    if has_fe:
        beta = np.zeros(k)
        # Initialize FE as group log-means
        eta = np.zeros(n)
        for fe_idx in fe_list:
            codes, G = _fe_codes(fe_idx)
            gm = np.bincount(codes, weights=y_init, minlength=G) / np.maximum(
                np.bincount(codes, minlength=G), 1
            )
            eta += np.log(np.maximum(gm, 0.1))[codes]
        mu = _safe_exp(eta)
    else:
        beta = np.linalg.lstsq(X, np.log(y_init), rcond=None)[0]
        mu = _safe_exp(X @ beta)

    converged = False
    # Tolerance schedule for the inner within-transform (ppmlhdfe's
    # practice): early IRLS steps only need a rough projection, so the
    # demeaning tolerance starts loose and tightens with the coefficient
    # change; convergence is declared only on a step taken at the final
    # tolerance, so the answer is the same as demeaning tightly throughout.
    demean_tol_final = 1e-10
    demean_tol = 1e-6 if len(fe_list) >= 2 else demean_tol_final
    # Warm start of the within-transform.  For any weights, the projection
    # removes everything in the span of the fixed effects, and the previous
    # raw-minus-demeaned columns lie in that span; so projecting
    # ``Zd_prev + (Z - Z_prev)`` gives the same answer as projecting ``Z``
    # but starts next to it.  On a 7-way stacked PPML this cut the tight
    # final projections from thousands of sweeps to a few dozen.
    Z_prev: Optional[np.ndarray] = None
    Zd_prev: Optional[np.ndarray] = None
    for it in range(maxiter):
        # Working variable and weights
        w = weights * mu
        z = (y - mu) / mu  # working residual (without eta, for demeaning)

        if has_fe:
            # Demean z and X by FEs (weighted by w)
            # Working variable on the full linear predictor (slopes and
            # fixed effects); demeaning removes the fixed-effect part.
            z_full = eta + z

            # Demean z_full and X
            Z = np.column_stack([z_full.reshape(-1, 1), X])
            Z_start = Z if Z_prev is None else Zd_prev + (Z - Z_prev)
            Z_dm = _fe_weighted_demean(Z_start, fe_list, w, maxiter=500, tol=demean_tol)
            Z_prev, Zd_prev = Z, Z_dm

            z_dm = Z_dm[:, 0]
            X_dm = Z_dm[:, 1:]

            # WLS on demeaned data
            XtW = X_dm.T * w[None, :]
            XtWX = XtW @ X_dm
            XtWz = XtW @ z_dm
        else:
            eta = X @ beta
            z = eta + (y - mu) / mu
            XtW = X.T * w[None, :]
            XtWX = XtW @ X
            XtWz = XtW @ z

        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]

        # Update the linear predictor.  With absorbed effects this is the
        # IRLS step of Correia, Guimaraes & Zylkin's ppmlhdfe: the fitted
        # working variable is z minus the within residual, so the effects
        # are never solved for separately.  (A Gauss-Seidel solve of the
        # effects' score equations used to sit here; with nested or many
        # crossed effects, e.g. unit x cohort and unit in a stacked
        # regression, it needed thousands of sweeps per iteration.)
        if has_fe:
            resid_w = z_dm - X_dm @ beta_new
            eta = z_full - resid_w
            mu_new = _safe_exp(eta)
        else:
            mu_new = _safe_exp(X @ beta_new)

        delta = np.max(np.abs(beta_new - beta) / (np.abs(beta) + 1e-12)) if k else 0.0
        if has_fe:
            # The effects move too: also require the deviance to settle
            # (ppmlhdfe's criterion), so beta cannot stop while mu drifts.
            dev_new = float(
                2.0
                * np.sum(
                    weights
                    * (
                        np.where(y > 0, y * np.log(np.maximum(y, 1e-300) / mu_new), 0.0)
                        - (y - mu_new)
                    )
                )
            )
            dev_old = float(
                2.0
                * np.sum(
                    weights
                    * (
                        np.where(y > 0, y * np.log(np.maximum(y, 1e-300) / mu), 0.0)
                        - (y - mu)
                    )
                )
            )
            delta = max(delta, abs(dev_new - dev_old) / max(abs(dev_new), 0.1))
        beta = beta_new
        mu = mu_new

        used_tol = demean_tol
        demean_tol = float(np.clip(delta * 1e-3, demean_tol_final, demean_tol))
        if delta < tol and used_tol <= demean_tol_final:
            converged = True
            break

    # For HDFE, recompute the FE-residualised design at the converged
    # mu so the caller can build a Frisch-Waugh-Lovell-correct sandwich:
    # the bread of the robust vcov must use (X' W X) with X projected
    # onto the orthogonal complement of the FE column space, not the
    # raw X. Without this, the (X'WX)^{-1} bread underweights the FE-
    # absorbed variability and the HC1 SE is inflated relative to
    # ppmlhdfe / fixest::fepois (parity finding #6, 2026-05-28).
    if has_fe:
        w_final = weights * mu
        X_start = X if Zd_prev is None else Zd_prev[:, 1:]
        X_dm_final = _fe_weighted_demean(
            X_start, fe_list, w_final, maxiter=500, tol=1e-10
        )
        return beta, mu, converged, it + 1, X_dm_final

    return beta, mu, converged, it + 1, None


# ---------------------------------------------------------------------------
# Cameron-Trivedi overdispersion test
# ---------------------------------------------------------------------------


def _overdispersion_test(
    y: np.ndarray, mu: np.ndarray, df_resid: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Cameron-Trivedi regression-based test for overdispersion.

    Against ``Var(y | x) = mu + alpha * mu^2`` (the NB2 alternative) the
    test is the t statistic on ``mu`` in the regression **without
    intercept** of ``((y - mu)^2 - y) / mu`` on ``mu``; the coefficient
    estimates ``alpha`` [@cameron2013regression]. It is the statistic of
    ``AER::dispersiontest(trafo = 2)`` in R. The p-value is two-sided, from
    the t distribution with ``n - 1`` degrees of freedom, as an OLS fit of
    that regression reports it; ``dispersiontest`` gives the one-sided
    normal p-value by default.

    Before 1.39 the regression was of ``(y - mu)^2 - y`` on a constant and
    ``mu``, tested on the slope. With the constant in, the slope picks up
    only how the excess variance *changes* with the mean: counts with a
    Pearson dispersion of 12 passed with p = 0.44.

    Returns (t statistic, p-value, alpha).
    """
    aux = ((y - mu) ** 2 - y) / mu
    sxx = float(mu @ mu)
    alpha_hat = float(mu @ aux) / sxx
    resid = aux - alpha_hat * mu
    n = len(y)
    s2 = float(resid @ resid) / (n - 1)
    se = np.sqrt(s2 / sxx)
    if not np.isfinite(se) or se <= 0:
        return float("nan"), float("nan"), alpha_hat
    t_stat = alpha_hat / se
    p_val = 2 * stats.t.sf(abs(t_stat), n - 1)
    return float(t_stat), float(p_val), alpha_hat


# ===========================================================================
# Public API
# ===========================================================================


@accepts_aliases(vce="robust")
@records_provenance("sp.poisson")
@markout_clusters
def poisson(
    formula: Optional[str] = None,
    data: pd.DataFrame = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    offset: Optional[str] = None,
    exposure: Optional[str] = None,
    irr: bool = False,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Poisson regression via MLE (IRLS).

    Parameters
    ----------
    formula : str, optional
        Model formula, e.g. "y ~ x1 + x2".
    data : pd.DataFrame
        Data containing all variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Independent variable names (alternative to formula).
    robust : str, default "nonrobust"
        Standard error type: "nonrobust", "robust"/"hc0", "hc1".
    cluster : str, optional
        Variable name for clustered standard errors.
    weights : str, optional
        Observation weight column. Read as Stata reads a weight for the
        variance requested: alone it is ``[iw=w]`` (``[fw=w]`` for integer
        weights), a model-based variance that scales with the weights;
        with ``robust=`` it is ``[pw=w]``; with ``cluster=`` it is
        ``[pw=w], vce(cluster)``. The reported log-likelihood is the
        weighted sum.
    offset : str, optional
        Offset variable (log of exposure already computed).
    exposure : str, optional
        Exposure variable (will be logged and used as offset).
    irr : bool, default False
        If True, report Incidence Rate Ratios (exp(beta)) instead of
        raw coefficients.
    maxiter : int, default 100
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
        Fitted model with params, standard errors, diagnostics.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> math = rng.normal(50, 10, n)
    >>> prog = rng.integers(0, 3, n).astype(float)
    >>> num_awards = rng.poisson(np.exp(-3.0 + 0.06 * math + 0.2 * prog))
    >>> df = pd.DataFrame({'num_awards': num_awards, 'math': math, 'prog': prog})
    >>> res = sp.poisson("num_awards ~ math + prog", data=df)
    >>> list(res.params.index)
    ['_cons', 'math', 'prog']
    >>> res_irr = sp.poisson("num_awards ~ math + prog", data=df,
    ...                      robust="robust", irr=True)
    >>> bool(res_irr.params['math'] > 0)
    True
    """
    y_arr, X, var_names, dep_var, _, data = _parse_formula_or_xy(formula, data, y, x)
    n, k = X.shape

    # Offset / exposure
    offset_arr = np.zeros(n)
    if offset is not None:
        offset_arr = data[offset].values.astype(np.float64)
    if exposure is not None:
        offset_arr = np.log(_positive_exposure(data, exposure))

    # Weights
    w_arr = None
    if weights is not None:
        w_arr = data[weights].values.astype(np.float64)

    # Standard-error request (Stata grammar)
    from ..core._vcov_spec import parse_se_request

    _se = parse_se_request(
        robust,
        cluster,
        function="poisson",
        supported=("nonrobust", "robust", "hc0", "hc1", "cluster"),
    )
    robust, cluster = _se.kind, _se.cluster

    # Cluster variable
    cluster_arr = None
    if cluster is not None:
        cluster_arr = data[cluster].values

    # Fit
    beta, mu, converged, n_iter = _poisson_irls(
        y_arr, X, offset=offset_arr, weights=w_arr, maxiter=maxiter, tol=tol
    )
    if not converged:
        warnings.warn(f"Poisson IRLS did not converge in {maxiter} iterations")

    residuals = y_arr - mu

    # Variance-covariance
    vcov = _poisson_vcov(X, mu, residuals, robust, cluster_arr, weights=w_arr)
    se = np.sqrt(np.diag(vcov))

    # Log-likelihood (the weighted sum under weights, as Stata reports)
    ll = _poisson_loglik(y_arr, mu, w_arr)

    # Null model (intercept only)
    mu_null = np.full(n, np.average(y_arr, weights=w_arr))
    ll_null = _poisson_loglik(y_arr, mu_null, w_arr)

    # LR chi2
    lr_chi2 = 2 * (ll - ll_null)
    lr_pvalue = stats.chi2.sf(lr_chi2, k - 1)

    # Pseudo R-squared (McFadden)
    pseudo_r2 = 1 - ll / ll_null

    # AIC, BIC
    aic = -2 * ll + 2 * k
    bic = -2 * ll + np.log(n) * k

    # Goodness of fit
    deviance = 2 * np.sum(
        np.where(
            y_arr > 0,
            y_arr * np.log(np.maximum(y_arr, 1e-300) / mu),
            0,
        )
        - (y_arr - mu)
    )
    pearson_chi2 = np.sum((y_arr - mu) ** 2 / mu)

    # Overdispersion test
    od_stat, od_pval, od_alpha = _overdispersion_test(y_arr, mu)

    # IRR transform
    if irr:
        params_report = np.exp(beta)
        # Delta method: se(exp(b)) = exp(b) * se(b)
        se_report = params_report * se
        coef_label = "IRR"
    else:
        params_report = beta
        se_report = se
        coef_label = "Coefficient"

    params_series = pd.Series(params_report, index=var_names)
    se_series = pd.Series(se_report, index=var_names)

    model_info = {
        "model_type": "Poisson",
        "family": "Poisson",
        "link": "log",
        "method": "IRLS (MLE)",
        "robust": robust,
        "cluster": cluster,
        "irr": irr,
        "coef_label": coef_label,
        "converged": converged,
        "iterations": n_iter,
        "ll": ll,
        "ll_null": ll_null,
        "lr_chi2": lr_chi2,
        "lr_pvalue": lr_pvalue,
        "pseudo_r2": pseudo_r2,
        "aic": aic,
        "bic": bic,
    }
    if formula is not None:
        model_info["formula"] = formula
    if cluster_arr is not None:
        model_info["n_clusters"] = int(len(np.unique(cluster_arr)))
    if weights is not None:
        from ..core._agent_summary import note_weight_diagnostics

        note_weight_diagnostics(
            model_info,
            w_arr,
            (
                "cluster"
                if cluster_arr is not None
                else ("classical" if robust == "nonrobust" else "robust")
            ),
            entry="sp.poisson",
            robust_option="robust='robust' (Stata [pw=])",
            classical_reading="frequency weights (Stata [fw=])",
            classical_assumption=(
                "frequencies (each row standing for w identical observations)"
            ),
            cluster_keys=cluster_arr,
            cluster=cluster,
        )

    data_info = {
        "nobs": n,
        "df_model": k - 1,
        "df_resid": n - k,
        "dependent_var": dep_var,
        "fitted_values": mu,
        "residuals": residuals,
        "X": X,
        "y": y_arr,
        "var_cov": vcov,
        "var_names": var_names,
        # Likelihood-based: z / chi2 inference, as Stata's poisson.
        "inference": "z",
        # Unweighted per-observation log-likelihood (sp.vuong).
        "llobs": _poisson_llobs(y_arr, mu),
        "offset": offset,
        "exposure": exposure,
        "weights": weights,
    }

    diagnostics = {
        "Log-Likelihood": ll,
        "Log-Lik (null)": ll_null,
        "LR chi2": lr_chi2,
        "Prob > chi2": lr_pvalue,
        "Pseudo R2": pseudo_r2,
        "AIC": aic,
        "BIC": bic,
        "Deviance": deviance,
        "Pearson chi2": pearson_chi2,
        # Pearson chi2 over the residual degrees of freedom: 1 for a
        # Poisson, and the `dispersion` a quasi-Poisson fit would report.
        "Dispersion (Pearson chi2 / df)": float(pearson_chi2)
        / max(len(y_arr) - len(beta), 1),
        "Overdispersion test (C-T)": od_stat,
        "Overdispersion p-value": od_pval,
        "Overdispersion alpha (C-T)": od_alpha,
    }

    model_info["alpha"] = alpha
    return EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


@accepts_aliases(vce="robust")
@records_provenance("sp.nbreg")
@markout_clusters
def nbreg(
    formula: Optional[str] = None,
    data: pd.DataFrame = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    offset: Optional[str] = None,
    exposure: Optional[str] = None,
    irr: bool = False,
    dispersion: str = "mean",
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Negative binomial regression (NB2 or NB1).

    Parameters
    ----------
    formula : str, optional
        Model formula, e.g. "y ~ x1 + x2".
    data : pd.DataFrame
        Data containing all variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Independent variable names (alternative to formula).
    robust : str, default "nonrobust"
        Standard error type: "nonrobust", "robust"/"hc0", "hc1".
    cluster : str, optional
        Variable name for clustered standard errors.
    weights : str, optional
        Weight variable name.
    offset : str, optional
        Offset variable (log of exposure).
    exposure : str, optional
        Exposure variable (will be logged).
    irr : bool, default False
        Report Incidence Rate Ratios.
    dispersion : str, default "mean"
        Dispersion parameterization:
        - "mean" (NB2): Var(y) = mu + alpha * mu^2
        - "constant" (NB1): Var(y) = mu * (1 + delta)
    maxiter : int, default 100
        Maximum iterations.
    tol : float, default 1e-8
        Convergence tolerance.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> math = rng.normal(50, 10, n)
    >>> prog = rng.integers(0, 3, n).astype(float)
    >>> days_absent = rng.negative_binomial(2, 0.3, n)
    >>> df = pd.DataFrame({'days_absent': days_absent, 'math': math, 'prog': prog})
    >>> res = sp.nbreg("days_absent ~ math + prog", data=df, irr=True)
    >>> 'math' in res.params.index
    True
    """
    y_arr, X, var_names, dep_var, formula_fe, data = _parse_formula_or_xy(
        formula, data, y, x
    )
    X, var_names, fe_level_counts = _append_fixed_effect_dummies(
        X, var_names, data, formula_fe
    )
    n, k = X.shape
    n_fe_params = sum(max(v - 1, 0) for v in fe_level_counts.values())

    # Offset / exposure
    offset_arr = np.zeros(n)
    if offset is not None:
        offset_arr = data[offset].values.astype(np.float64)
    if exposure is not None:
        offset_arr = np.log(_positive_exposure(data, exposure))

    w_arr = None
    if weights is not None:
        w_arr = data[weights].values.astype(np.float64)

    cluster_arr = None
    # Standard-error request (Stata grammar)
    from ..core._vcov_spec import parse_se_request

    _se = parse_se_request(
        robust,
        cluster,
        function="nbreg",
        supported=("nonrobust", "robust", "hc0", "hc1", "cluster"),
    )
    robust, cluster = _se.kind, _se.cluster
    if cluster is not None:
        cluster_arr = data[cluster].values

    # Fit
    is_nb2 = dispersion.lower() == "mean"
    if is_nb2:
        beta, mu, disp_param, converged, n_iter = _nb2_fit(
            y_arr,
            X,
            offset=offset_arr,
            weights=w_arr,
            maxiter=maxiter,
            tol=tol,
        )
        disp_label = "alpha"
        nb_label = "NB2"
    else:
        beta, mu, disp_param, converged, n_iter = _nb1_fit(
            y_arr,
            X,
            offset=offset_arr,
            weights=w_arr,
            maxiter=maxiter,
            tol=tol,
        )
        disp_label = "delta"
        nb_label = "NB1"

    if not converged:
        warnings.warn(f"NegBin did not converge in {maxiter} outer iterations")

    # Polish to the joint MLE and take every variance from the joint
    # (beta, ln dispersion) observed information and scores -- what Stata's
    # ``nbreg`` reports. The previous variance used the IRLS weights as the
    # bread (ignoring the beta/dispersion cross-information) and the Poisson
    # residual y - mu as the score (missing the 1/(1 + alpha mu) factor), so
    # its OIM, robust and cluster SEs all disagreed with Stata; robust and
    # cluster by ~5% on moderately overdispersed data.
    from ..core._vcov import ml_vcov
    from ._negbin import negbin_joint

    joint = negbin_joint(
        y_arr, X, offset_arr, beta, disp_param, weights=w_arr, nb2=is_nb2
    )
    beta, disp_param, mu = joint.beta, joint.dispersion, joint.mu
    residuals = y_arr - mu

    vcov_joint = ml_vcov(joint.bread, joint.scores, kind=robust, clusters=cluster_arr)
    vcov = vcov_joint[:k, :k]
    se = np.sqrt(np.diag(vcov))
    se_ln_dispersion = float(np.sqrt(vcov_joint[k, k]))
    # R's MASS::glm.nb reports a different, documented quantity: the
    # model-based SE of beta *conditional on* the dispersion, i.e. the IRLS
    # Fisher information (X' W X)^{-1} with W = mu / Var(y | mu). Kept for
    # R users and for the glm.nb parity test; the reported SE is Stata's.
    nb_var_ratio = (1 + disp_param * mu) if is_nb2 else np.full_like(mu, 1 + disp_param)
    info_conditional = X.T @ (X * (mu / nb_var_ratio)[:, None])
    se_conditional = pd.Series(
        np.sqrt(np.maximum(np.diag(np.linalg.pinv(info_conditional)), 0.0)),
        index=var_names,
    )

    # Log-likelihood
    # Reported likelihoods are the weighted sums under weights (Stata's
    # log pseudolikelihood); they were unweighted until 1.36.0.
    if is_nb2:
        ll = _nb2_loglik(y_arr, mu, disp_param, w_arr)
    else:
        ll = _nb1_loglik(y_arr, mu, disp_param, w_arr)

    # Null model
    mu_null = np.full(n, np.average(y_arr, weights=w_arr))
    if is_nb2:
        # Optimize alpha for null model
        def neg_null_ll(log_a: float) -> float:
            return -_nb2_loglik(y_arr, mu_null, float(np.exp(log_a)), w_arr)

        res_null = optimize.minimize_scalar(
            neg_null_ll, bounds=(np.log(1e-8), np.log(1e4)), method="bounded"
        )
        ll_null = -res_null.fun
    else:

        def neg_null_ll(log_a: float) -> float:
            return -_nb1_loglik(y_arr, mu_null, float(np.exp(log_a)), w_arr)

        res_null = optimize.minimize_scalar(
            neg_null_ll, bounds=(np.log(1e-8), np.log(1e4)), method="bounded"
        )
        ll_null = -res_null.fun

    # Poisson ll for LR test of dispersion
    ll_poisson = _poisson_loglik(y_arr, mu, w_arr)
    lr_alpha = 2 * (ll - ll_poisson)
    # One-sided test (alpha >= 0), use chibar^2 (50:50 mixture of chi2_0 and chi2_1)
    lr_alpha_pvalue = 0.5 * stats.chi2.sf(max(lr_alpha, 0), 1)

    lr_chi2 = 2 * (ll - ll_null)
    lr_pvalue = stats.chi2.sf(lr_chi2, k - 1)
    pseudo_r2 = 1 - ll / ll_null

    aic = -2 * ll + 2 * (k + 1)  # +1 for dispersion
    bic = -2 * ll + np.log(n) * (k + 1)

    # IRR
    if irr:
        params_report = np.exp(beta)
        se_report = params_report * se
        coef_label = "IRR"
    else:
        params_report = beta
        se_report = se
        coef_label = "Coefficient"

    params_series = pd.Series(params_report, index=var_names)
    se_series = pd.Series(se_report, index=var_names)

    model_info = {
        "model_type": f"NegBin ({nb_label})",
        "family": "Negative Binomial",
        "link": "log",
        "method": "MLE (IRLS + profile likelihood, joint Newton polish)",
        "dispersion_type": nb_label,
        f"se_ln{disp_label}": se_ln_dispersion,
        "gradient_norm": joint.gradient_norm,
        "newton_steps": joint.newton_steps,
        # MASS::glm.nb convention (beta SE conditional on the dispersion).
        "se_conditional_on_dispersion": se_conditional,
        "fixed_effects": list(fe_level_counts) or None,
        "n_fe_levels": fe_level_counts or None,
        "n_fe_params": n_fe_params,
        "robust": robust,
        "cluster": cluster,
        "irr": irr,
        "coef_label": coef_label,
        "converged": converged,
        "iterations": n_iter,
        "dispersion": disp_param,
        "dispersion_label": disp_label,
        "ll": ll,
        "ll_null": ll_null,
        "lr_chi2": lr_chi2,
        "lr_pvalue": lr_pvalue,
        "pseudo_r2": pseudo_r2,
        "aic": aic,
        "bic": bic,
    }
    if formula is not None:
        model_info["formula"] = formula
    if cluster_arr is not None:
        model_info["n_clusters"] = int(len(np.unique(cluster_arr)))
    if weights is not None:
        from ..core._agent_summary import note_weight_diagnostics

        note_weight_diagnostics(
            model_info,
            w_arr,
            (
                "cluster"
                if cluster_arr is not None
                else ("classical" if robust == "nonrobust" else "robust")
            ),
            entry="sp.nbreg",
            robust_option="robust='robust' (Stata [pw=])",
            classical_reading="frequency weights (Stata [fw=])",
            classical_assumption=(
                "frequencies (each row standing for w identical observations)"
            ),
            cluster_keys=cluster_arr,
            cluster=cluster,
        )

    data_info = {
        "nobs": n,
        "df_model": k - 1,
        "df_resid": n - k,
        "dependent_var": dep_var,
        "fitted_values": mu,
        "residuals": residuals,
        "X": X,
        "y": y_arr,
        "var_cov": vcov,
        "var_names": var_names,
        # Likelihood-based: z / chi2 inference, as Stata's nbreg.
        "inference": "z",
        # Unweighted per-observation log-likelihood (sp.vuong).
        "llobs": (_nb2_llobs if is_nb2 else _nb1_llobs)(y_arr, mu, disp_param),
        # Coefficients plus the dispersion, which is not in ``params``.
        "n_params": k + 1,
    }

    diagnostics = {
        "Log-Likelihood": ll,
        "Log-Lik (null)": ll_null,
        "LR chi2": lr_chi2,
        "Prob > chi2": lr_pvalue,
        "Pseudo R2": pseudo_r2,
        "AIC": aic,
        "BIC": bic,
        f"Dispersion ({disp_label})": disp_param,
        "LR test vs Poisson (chi2)": lr_alpha,
        "LR test vs Poisson (p)": lr_alpha_pvalue,
    }
    if fe_level_counts:
        diagnostics["N fixed-effect parameters"] = n_fe_params
        diagnostics["Fixed-effect levels"] = fe_level_counts

    model_info["alpha"] = alpha
    return EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


@accepts_aliases(id="entity", vce="robust")
def xtnbreg(
    formula: Optional[str] = None,
    data: pd.DataFrame = None,
    y: Optional[str] = None,
    x: Optional[Sequence[str]] = None,
    entity: Optional[str] = None,
    time: Optional[str] = None,
    model: str = "fe",
    time_effects: bool = False,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    offset: Optional[str] = None,
    exposure: Optional[str] = None,
    irr: bool = False,
    dispersion: str = "mean",
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> Any:
    """
    Panel negative-binomial regression (Stata ``xtnbreg``).

    ``model="fe"`` and ``model="re"`` are the Hausman-Hall-Griliches
    models (the likelihoods printed in Stata's [XT] ``xtnbreg`` Methods and
    formulas) that Stata's ``xtnbreg, fe`` / ``xtnbreg, re`` and R's
    ``pglm(family = negbin, model = "within" / "random")`` estimate: a
    negative binomial whose variance-to-mean ratio ``1 + δ_i`` is constant
    within a panel, with

    * ``fe`` -- the likelihood conditional on each panel's outcome total
      (δ_i drops out; the intercept stays identified; panels with one
      observation or an all-zero outcome carry no information and are
      dropped, as Stata drops them);
    * ``re`` -- ``1/(1 + δ_i) ~ Beta(r, s)``, estimated with ``ln r`` and
      ``ln s``.

    Standard errors are the inverse observed information (``vce(oim)``).

    Two other panel NB estimators remain available under their own names:
    ``model="ufe"`` is the *unconditional* fixed-effects NB-2 (entity
    dummies through :func:`nbreg`; the default of ``model="fe"`` before
    1.28.x), and ``model="normal_re"`` the normal random-intercept NB-2
    GLMM :func:`sp.menbreg` (the default of ``model="re"`` before 1.28.x).
    Neither is what ``xtnbreg`` computes.

    Parameters
    ----------
    formula : str, optional
        Count-model formula. For fixed effects you may pass
        ``"y ~ x1 + x2 | id"`` directly, or pass ``entity=``.
    data : DataFrame
        Long-format panel data.
    y, x : optional
        Alternative to ``formula``.
    entity : str, optional
        Panel/unit identifier. Required when the formula does not contain a
        ``| id`` fixed-effect part.
    time : str, optional
        Time column. Stored as metadata; included as a fixed effect only when
        ``time_effects=True``.
    model : {"fe", "re", "pooled", "ufe", "normal_re"}, default "fe"
        HHG conditional fixed effects, HHG beta random effects, pooled NB,
        unconditional (dummy-variable) fixed-effects NB-2, or the normal
        random-intercept NB-2 GLMM.

    Returns
    -------
    EconometricResults or MEGLMResult
        ``"fe"`` / ``"re"`` / ``"pooled"`` / ``"ufe"`` return
        :class:`EconometricResults`; ``"normal_re"`` returns the multilevel
        :class:`MEGLMResult`.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for uid in range(15):
    ...     fe = rng.normal(0, 0.3)  # entity effect
    ...     for t in range(8):
    ...         x = rng.normal()
    ...         mu = np.exp(0.4 + 0.5 * x + fe)
    ...         rate = rng.gamma(shape=2.0, scale=mu / 2.0)  # NB-2 mixture
    ...         count = rng.poisson(rate)
    ...         rows.append(dict(unit=uid, year=t, count=count, x=x))
    >>> df = pd.DataFrame(rows)
    >>> res = sp.xtnbreg("count ~ x", data=df, entity="unit", model="fe")
    >>> type(res).__name__
    'EconometricResults'
    >>> "x" in res.params.index
    True
    """
    data = _require_count_dataframe(data, "xtnbreg")

    x_list = _normalize_column_list(x, "x")
    if formula is None:
        if y is None:
            raise MethodIncompatibility(
                "xtnbreg requires either `formula` or `y=`.",
                recovery_hint="Call sp.xtnbreg('y ~ x', data=df, entity='id') "
                "or pass y='y' with x=[...].",
                diagnostics={"has_formula": False, "has_y": False},
                alternative_functions=["sp.xtnbreg", "sp.nbreg"],
            )
        y = _require_column_name(y, "y")
        _require_columns(data, [y, *x_list], "xtnbreg y/x")
        rhs = " + ".join(x_list) if x_list else "1"
        formula = f"{y} ~ {rhs}"
    else:
        parsed = parse_formula(formula)
        y = parsed["dependent"]
        x_list = parsed["exogenous"]
        _require_columns(
            data,
            [y, *x_list, *parsed.get("fixed_effects", [])],
            "xtnbreg formula",
        )
        if entity is None and parsed.get("fixed_effects"):
            entity = parsed["fixed_effects"][0]

    if model is None:
        model_key = "fe"
    elif isinstance(model, str):
        model_key = model.lower().replace("-", "_")
    else:
        raise MethodIncompatibility(
            "model must be one of 'fe', 're', 'pooled', 'ufe', or 'normal_re'.",
            recovery_hint="Pass model='fe', 're', 'pooled', 'ufe', or 'normal_re'.",
            diagnostics={"model": repr(model)},
            alternative_functions=["sp.xtnbreg"],
        )
    if model_key in {"fixed", "fixed_effects", "conditional"}:
        model_key = "fe"
    if model_key in {"random", "random_effects"}:
        model_key = "re"

    if model_key in {"fe", "re"}:
        if entity is None:
            kind = "fixed-effects" if model_key == "fe" else "random-effects"
            raise MethodIncompatibility(
                f"{kind} xtnbreg (model={model_key!r}) requires `entity=` or a "
                "formula fixed-effect part such as 'y ~ x | id'.",
                recovery_hint="Pass entity='id'.",
                diagnostics={"model": model_key, "entity": entity},
                alternative_functions=["sp.xtnbreg"],
            )
        return _xtnbreg_hhg(
            model_key,
            data=data,
            y=y,
            x_list=x_list,
            entity=entity,
            time=time,
            time_effects=time_effects,
            robust=robust,
            cluster=cluster,
            weights=weights,
            offset=offset,
            exposure=exposure,
            irr=irr,
            dispersion=dispersion,
            maxiter=maxiter,
            tol=tol,
            alpha=alpha,
        )
    if model_key == "ufe":
        model_key = "fe_dummies"
    if model_key == "normal_re":
        model_key = "re_gaussian"

    if model_key == "pooled":
        pooled_formula = formula.split("|", 1)[0].strip()
        result = nbreg(
            formula=pooled_formula,
            data=data,
            robust=robust,
            cluster=cluster,
            weights=weights,
            offset=offset,
            exposure=exposure,
            irr=irr,
            dispersion=dispersion,
            maxiter=maxiter,
            tol=tol,
            alpha=alpha,
        )
        result.model_info["panel_model"] = "pooled"
        result.model_info["entity"] = entity
        result.model_info["time"] = time
        return result

    if model_key == "fe_dummies":
        fe_formula = formula
        if "|" not in fe_formula:
            if not entity:
                raise MethodIncompatibility(
                    "fixed-effects xtnbreg requires `entity=` or a formula "
                    "fixed-effect part such as 'y ~ x | id'.",
                    recovery_hint="Pass entity='id' or include a fixed-effect "
                    "part in the formula.",
                    diagnostics={"model": model_key, "entity": entity},
                    alternative_functions=["sp.xtnbreg", "sp.nbreg"],
                )
            _require_columns(data, [entity], "xtnbreg entity")
            fe_terms = [entity]
            if time_effects:
                if not time:
                    raise MethodIncompatibility(
                        "time_effects=True requires `time=`.",
                        recovery_hint="Pass the time column name or set "
                        "time_effects=False.",
                        diagnostics={"time_effects": True, "time": time},
                        alternative_functions=["sp.xtnbreg"],
                    )
                _require_columns(data, [time], "xtnbreg time")
                fe_terms.append(time)
            fe_formula = f"{formula} | {' + '.join(fe_terms)}"
        elif entity is None:
            parsed = parse_formula(fe_formula)
            if parsed.get("fixed_effects"):
                entity = parsed["fixed_effects"][0]

        cluster_arg = cluster or entity
        result = nbreg(
            formula=fe_formula,
            data=data,
            robust=robust,
            cluster=cluster_arg,
            weights=weights,
            offset=offset,
            exposure=exposure,
            irr=irr,
            dispersion=dispersion,
            maxiter=maxiter,
            tol=tol,
            alpha=alpha,
        )
        result.model_info["panel_model"] = "unconditional_fixed_effects"
        result.model_info["entity"] = entity
        result.model_info["time"] = time
        result.model_info["time_effects"] = bool(time_effects)
        result.model_info["stata_equivalent"] = "nbreg y x i.entity"
        return result

    if model_key == "re_gaussian":
        if not entity:
            raise MethodIncompatibility(
                "random-effects xtnbreg requires `entity=` or a formula "
                "fixed-effect part whose first term is the panel id.",
                recovery_hint="Pass entity='id' for the random-intercept "
                "negative-binomial model.",
                diagnostics={"model": model_key, "entity": entity},
                alternative_functions=["sp.xtnbreg"],
            )
        _require_columns(data, [entity], "xtnbreg entity")
        if weights is not None:
            warnings.warn(
                "xtnbreg(model='re') does not support weights; ignoring weights"
            )
        if cluster is not None or robust not in (None, False, "nonrobust", "oim"):
            # A warning followed by model-based SEs is a silent change of
            # the requested standard errors; refuse instead.
            raise MethodIncompatibility(
                "xtnbreg(model='re') reports GLMM model-based standard errors "
                f"only; robust={robust!r} / cluster={cluster!r} is not available.",
                recovery_hint=(
                    "Drop robust=/cluster=, or use model='fe' or sp.nbreg with "
                    "vce='cluster <panel id>'."
                ),
            )
        if dispersion.lower() != "mean":
            warnings.warn(
                "xtnbreg(model='re') uses NB2 mean dispersion; " "ignoring dispersion"
            )
        if irr:
            warnings.warn(
                "xtnbreg(model='re') returns coefficients; call "
                "result.incidence_rate_ratios() for IRRs"
            )

        re_data = data
        offset_col = offset
        if exposure is not None:
            exposure_values = _positive_exposure(data, exposure)
            offset_col = "__statspai_log_exposure__"
            re_data = data.copy()
            re_data[offset_col] = np.log(exposure_values)

        from ..multilevel.glmm import menbreg

        return menbreg(
            re_data,
            y,
            x_list,
            entity,
            offset=offset_col,
            maxiter=maxiter,
            tol=tol,
            alpha=alpha,
        )

    raise MethodIncompatibility(
        "model must be one of 'fe', 're', 'pooled', 'ufe', or 'normal_re'.",
        recovery_hint="Pass model='fe' (Stata xtnbreg, fe) or model='re'.",
        diagnostics={"model": model},
        alternative_functions=["sp.xtnbreg"],
    )


def _xtnbreg_hhg(
    model_key: str,
    *,
    data: pd.DataFrame,
    y: str,
    x_list: List[str],
    entity: str,
    time: Optional[str],
    time_effects: bool,
    robust: str,
    cluster: Optional[str],
    weights: Optional[str],
    offset: Optional[str],
    exposure: Optional[str],
    irr: bool,
    dispersion: str,
    maxiter: int,
    tol: float,
    alpha: float,
) -> EconometricResults:
    """Stata ``xtnbreg, fe`` / ``xtnbreg, re`` (HHG 1984); see
    :mod:`._xtnbreg_hhg`."""
    from ._xtnbreg_hhg import fit_hhg

    label = f"xtnbreg(model={model_key!r})"
    if cluster is not None or robust not in (None, False, "nonrobust", "oim"):
        raise MethodIncompatibility(
            f"{label} reports observed-information standard errors only "
            f"(as Stata's xtnbreg, {model_key}); robust={robust!r} / "
            f"cluster={cluster!r} is not available.",
            recovery_hint="Drop robust=/cluster=, or use model='ufe' with "
            "cluster-robust standard errors.",
            diagnostics={"robust": robust, "cluster": cluster},
        )
    if weights is not None:
        raise MethodIncompatibility(
            f"{label} does not support weights.",
            recovery_hint="Drop weights=, or use model='ufe'.",
            diagnostics={"weights": weights},
        )
    if str(dispersion).lower() not in ("mean", "constant"):
        raise MethodIncompatibility(
            f"{label}: the HHG model has a panel-constant dispersion; "
            f"dispersion={dispersion!r} does not apply.",
            diagnostics={"dispersion": dispersion},
        )
    if offset is not None and exposure is not None:
        raise MethodIncompatibility(
            "Pass either offset= or exposure=, not both.",
            diagnostics={"offset": offset, "exposure": exposure},
        )
    if time_effects and not time:
        raise MethodIncompatibility(
            "time_effects=True requires `time=`.",
            diagnostics={"time_effects": True, "time": time},
        )

    cols = [y, *x_list, entity]
    if time_effects:
        cols.append(time)
    if offset is not None:
        cols.append(offset)
    if exposure is not None:
        cols.append(exposure)
    _require_columns(data, list(dict.fromkeys(cols)), label)
    # Listwise deletion on the model's columns, as Stata's estimation sample.
    df = data.loc[data[list(dict.fromkeys(cols))].notna().all(axis=1)].copy()
    if len(df) == 0:
        raise DataInsufficient(
            f"{label}: no complete observations.",
            diagnostics={"columns": cols},
        )

    y_arr = _numeric_column(df, y, "outcome")
    if np.any(y_arr < 0) or np.any(y_arr != np.floor(y_arr)):
        raise MethodIncompatibility(
            f"{label} requires a non-negative integer count outcome.",
            diagnostics={"outcome": y},
        )
    X_cols = [_numeric_column(df, c, "regressor") for c in x_list]
    var_names = list(x_list)
    if time_effects:
        t_levels = sorted(df[time].unique())
        for lv in t_levels[1:]:
            X_cols.append((df[time] == lv).to_numpy(dtype=float))
            var_names.append(f"{time}={lv}")
    X_cols.append(np.ones(len(df)))
    var_names.append("_cons")
    X = np.column_stack(X_cols)
    off = np.zeros(len(df))
    if offset is not None:
        off = _numeric_column(df, offset, "offset")
    if exposure is not None:
        off = np.log(_positive_exposure(df, exposure))

    fit = fit_hhg(
        model_key,
        y_arr,
        X,
        df[entity].to_numpy(),
        off,
        maxiter=max(int(maxiter), 1),
        tol=float(tol),
    )
    if not fit.converged:
        warnings.warn(
            f"{label}: Newton-Raphson did not converge in {fit.iterations} "
            f"iterations (max |score| = {fit.gradient_norm:.2e}).",
            RuntimeWarning,
            stacklevel=3,
        )

    k = X.shape[1]
    beta = fit.params[:k]
    se_beta = np.sqrt(np.maximum(np.diag(fit.vcov)[:k], 0.0))
    names = list(var_names)
    params = list(beta)
    ses = list(se_beta)
    if model_key == "re":
        names += ["/ln_r", "/ln_s"]
        params += list(fit.params[k:])
        ses += list(np.sqrt(np.maximum(np.diag(fit.vcov)[k:], 0.0)))
    params_arr = np.asarray(params)
    ses_arr = np.asarray(ses)
    coef_label = "Coefficient"
    if irr:
        params_arr = params_arr.copy()
        ses_arr = ses_arr.copy()
        params_arr[:k] = np.exp(beta)
        ses_arr[:k] = np.exp(beta) * se_beta
        coef_label = "IRR"

    n_used = int(fit.keep.sum())
    n_par = len(fit.params)
    ll = float(fit.loglik)
    model_info = {
        "model_type": (
            "Conditional FE negative binomial (Hausman-Hall-Griliches)"
            if model_key == "fe"
            else (
                "Random-effects negative binomial, 1/(1+delta_i) ~ Beta(r, s) "
                "(Hausman-Hall-Griliches)"
            )
        ),
        "panel_model": "fixed_effects" if model_key == "fe" else "random_effects",
        "stata_equivalent": f"xtnbreg, {model_key}",
        "family": "Negative Binomial",
        "link": "log",
        "method": "MLE (Newton-Raphson, observed information)",
        "entity": entity,
        "time": time,
        "time_effects": bool(time_effects),
        "n_groups": fit.n_groups,
        "n_groups_dropped": fit.n_dropped_groups,
        "converged": fit.converged,
        "iterations": fit.iterations,
        "gradient_norm": fit.gradient_norm,
        "irr": irr,
        "coef_label": coef_label,
        "ll": ll,
        "aic": -2.0 * ll + 2.0 * n_par,
        "bic": -2.0 * ll + np.log(n_used) * n_par,
    }
    if model_key == "re":
        model_info["r"] = float(np.exp(fit.params[k]))
        model_info["s"] = float(np.exp(fit.params[k + 1]))
    vcov = fit.vcov
    if irr:
        J = np.eye(n_par)
        J[:k, :k] = np.diag(np.exp(beta))
        vcov = J @ vcov @ J.T
    data_info = {
        "nobs": n_used,
        "df_model": k - 1,
        "df_resid": n_used - n_par,
        "dependent_var": y,
        "X": X[fit.keep],
        "y": y_arr[fit.keep],
        "var_cov": vcov,
        "var_names": names,
        "inference": "z",
    }
    diagnostics = {
        "Log-Likelihood": ll,
        "AIC": model_info["aic"],
        "BIC": model_info["bic"],
        "Number of groups": fit.n_groups,
    }
    if fit.n_dropped_groups:
        diagnostics["Groups dropped (single obs or all-zero outcome)"] = (
            fit.n_dropped_groups
        )
    model_info["alpha"] = alpha
    return EconometricResults(
        params=pd.Series(params_arr, index=names),
        std_errors=pd.Series(ses_arr, index=names),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _ppml_front_end(
    formula: Optional[str],
    data: pd.DataFrame,
    y: Optional[str],
    x: Optional[List[str]],
    absorb: Optional[str],
    cluster: Any,
    weights: Optional[str],
) -> Tuple[Optional[str], pd.DataFrame, Optional[str], Any]:
    """Resolve interacted FE / cluster terms and mark out missing rows.

    ``absorb="id + ind^year"`` (fixest) or ``"id + ind#year"`` (Stata) and
    ``cluster="city^year"`` become integer group columns, so users need not
    build ``groupby().ngroup()`` columns by hand.  A formula's ``| fe`` part
    moves into ``absorb`` (unless ``absorb`` is given, which wins, as
    documented).  Rows missing the outcome, a plain regressor, a fixed
    effect, the cluster or the weight are dropped, as Stata's ``markout``
    does -- they used to reach IRLS as NaN.
    """
    from ..core._group_terms import resolve_group_terms

    if data is None or not hasattr(data, "columns"):
        return formula, data, absorb, cluster
    if formula is not None and "|" in formula:
        head, fe_part = formula.split("|", 1)
        formula = head.strip()
        if absorb is None and fe_part.strip():
            absorb = fe_part.strip()
    fe_terms = [t.strip() for t in absorb.split("+") if t.strip()] if absorb else []
    data, fe_names = resolve_group_terms(data, fe_terms)
    absorb = " + ".join(fe_names) if fe_names else None
    if isinstance(cluster, str):
        data, (cluster,) = resolve_group_terms(data, [cluster])
        cl_names = [cluster]
    elif isinstance(cluster, (list, tuple)):
        data, cl_names = resolve_group_terms(data, list(cluster))
        cluster = list(cl_names)
    else:
        cl_names = []

    used: List[str] = list(fe_names) + list(cl_names)
    if isinstance(weights, str):
        used.append(weights)
    if formula is not None and "~" in formula:
        parsed = parse_formula(formula)
        used.append(parsed["dependent"])
        if not _needs_patsy(formula):
            used.extend(parsed["exogenous"])
    else:
        if y is not None:
            used.append(y)
        if x is not None:
            used.extend([x] if isinstance(x, str) else list(x))
    used = [c for c in dict.fromkeys(used) if c in data.columns]
    if used:
        miss = data[used].isna().any(axis=1).to_numpy()
        if miss.any():
            data = data.loc[~miss]
    return formula, data, absorb, cluster


def _ppml_absorbed_columns(
    X: np.ndarray,
    fe_indices_list: Sequence[np.ndarray],
    rtol: float = 1e-9,
) -> List[int]:
    """Indices of regressors collinear with the fixed effects / each other.

    Each column is residualised on the absorbed fixed effects (unweighted
    alternating projections) and then, in order, on the columns already
    kept; a column whose remaining sum of squares is below ``rtol`` times
    its own centred sum of squares is omitted -- ``reghdfe`` / Stata's
    rule, which keeps the first of a collinear set.  A time-invariant
    regressor under a unit fixed effect is the common case: Stata reports
    it as ``(omitted)``, where the solver used to overflow and fail with
    ``LinAlgError``.
    """
    n, k = X.shape
    if k == 0:
        return []
    if fe_indices_list:
        fe_indices_list = _drop_nested_fe(fe_indices_list)
        Xd = _fe_weighted_demean(X, fe_indices_list, np.ones(n), maxiter=2000, tol=1e-9)
    else:
        Xd = np.asarray(X, dtype=float)
    scale = np.sum((X - X.mean(axis=0)) ** 2, axis=0)
    scale = np.where(scale > 0, scale, np.sum(X**2, axis=0))
    omitted: List[int] = []
    kept: List[int] = []
    for j in range(k):
        v = Xd[:, j]
        if kept:
            B = Xd[:, kept]
            coef, *_ = np.linalg.lstsq(B, v, rcond=None)
            v = v - B @ coef
        if scale[j] <= 0 or float(v @ v) <= rtol * float(scale[j]):
            omitted.append(j)
        else:
            kept.append(j)
    return omitted


def _drop_named_columns(
    X: np.ndarray, var_names: List[str], omitted: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """Remove the ``omitted`` regressors from a design (extended vce paths)."""
    if not omitted:
        return X, var_names
    live = [j for j, v in enumerate(var_names) if v not in set(omitted)]
    return X[:, live], [var_names[j] for j in live]


def _ppmlhdfe_design(
    formula: Optional[str],
    data: pd.DataFrame,
    y: Optional[str],
    x: Optional[List[str]],
    absorb: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[np.ndarray], pd.DataFrame]:
    """Parse the PPML design exactly like :func:`ppmlhdfe` does.

    Returns ``(y_arr, X, var_names, fe_indices_list, data)`` with the constant
    already dropped when fixed effects absorb it — the shared front end for
    the extended ``vce=`` paths.
    """
    y_arr, X, var_names, _dep, formula_fe, data = _parse_formula_or_xy(
        formula, data, y, x, add_constant=True
    )
    if absorb is not None:
        fe_names = [v.strip() for v in absorb.split("+")]
    elif formula_fe:
        fe_names = list(formula_fe)
    else:
        fe_names = []
    fe_indices_list = [pd.factorize(data[fv].values)[0] for fv in fe_names]
    if fe_indices_list and var_names[0] == "_cons":
        X = X[:, 1:]
        var_names = var_names[1:]
    return y_arr, X, list(var_names), fe_indices_list, data


def _wfe_demean(
    M: np.ndarray,
    fe_indices_list: Sequence[np.ndarray],
    w: np.ndarray,
    maxiter: int = 2000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Weighted within-transform: alternating w-weighted FE projections of M."""
    return _fe_weighted_demean(M, fe_indices_list, w, maxiter=maxiter, tol=tol)


def _ppmlhdfe_wild(
    formula: Optional[str],
    data: pd.DataFrame,
    y: Optional[str],
    x: Optional[List[str]],
    absorb: Optional[str],
    cluster: str,
    *,
    separation: bool,
    maxiter: int,
    tol: float,
    alpha: float,
    n_boot: int,
    weight_type: str,
    seed: Optional[int],
) -> EconometricResults:
    """Score wild cluster bootstrap p-values for PPML HDFE (boottest convention).

    Runs the restricted score wild cluster bootstrap (Kline-Santos 2012) with
    Stata ``boottest``'s exact studentization, computed **on the FE-absorbed
    design at scale**: the per-cluster one-step contributions ``q_g`` are pure
    linear algebra at the restricted fit, so the weighted
    Frisch-Waugh-Lovell reduction is *exact* (verified to 1e-17 against the
    full-dummy computation) — unlike CR2, whose per-observation leverage does
    not survive absorption. On low-dimensional FE this path equals
    ``sp.fepois(vce="wild")`` (itself bit-exact vs ``boottest``) replication
    for replication; ``boottest`` cannot run after Stata's ``ppmlhdfe`` at
    all (no ``constraints()`` support), so this extends the Stata menu rather
    than mirroring it. Point estimates and cluster-robust SEs stand; p-values
    come from the bootstrap.
    """
    from ..inference.jackknife import score_wild_from_q

    base = ppmlhdfe(
        formula,
        data,
        y=y,
        x=x,
        absorb=absorb,
        cluster=cluster,
        separation=separation,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
    )
    y_arr, X, var_names, fe_list, data = _ppmlhdfe_design(formula, data, y, x, absorb)
    X, var_names = _drop_named_columns(X, var_names, base.model_info.get("omitted", []))
    n, k = X.shape
    if k < 2 and not fe_list:
        raise MethodIncompatibility(
            "ppmlhdfe(vce='wild') needs at least one other regressor or an "
            "absorbed fixed effect to form the restricted (null) model.",
            recovery_hint="Add covariates / absorb= or use cluster= (CRV1).",
        )
    codes = pd.factorize(data[cluster].values)[0]
    G = int(codes.max()) + 1
    c_share = np.array([(codes == c).sum() for c in range(G)]) / float(n)

    pvals: Dict[str, float] = {}
    for j, vname in enumerate(var_names):
        keep = [c for c in range(k) if c != j]
        X_rest = X[:, keep]
        beta_r, mu_r, conv, _it, _xdm = _ppml_hdfe_irls(
            y_arr,
            X_rest,
            fe_indices_list=fe_list or None,
            weights=None,
            maxiter=maxiter,
            tol=tol,
        )
        if not conv:
            warnings.warn(
                f"Restricted PPML for H0: {vname}=0 did not converge; "
                "its wild-bootstrap p-value may be unreliable."
            )
        e = y_arr - mu_r
        W = mu_r
        # weighted-FWL residualization of x_j on [other covariates + FE]
        cols = np.column_stack([X[:, j], X_rest]) if X_rest.size else X[:, [j]]
        cols_d = _wfe_demean(cols, fe_list, W) if fe_list else cols
        xd = cols_d[:, 0]
        Zd = cols_d[:, 1:]
        if Zd.shape[1]:
            gam = np.linalg.solve(Zd.T @ (W[:, None] * Zd), Zd.T @ (W * xd))
            xt = xd - Zd @ gam
        else:
            xt = xd
        denom = float(xt @ (W * xt))
        g_i = xt * e / denom
        q = np.array([g_i[codes == c].sum() for c in range(G)])
        out = score_wild_from_q(
            q, c_share, n_boot=n_boot, weight_type=weight_type, seed=seed
        )
        pvals[vname] = out["p_boot"]

    base.pvalues = pd.Series(
        [pvals[v] for v in base.params.index], index=base.params.index
    )
    base.model_info = dict(base.model_info)
    # These SEs replace the fitted ones and their p-values are normal; mark
    # the fit so conf_int() / tidy() / sp.test use z as well.
    base.data_info = dict(base.data_info, inference="z")
    base.model_info["vcov_type"] = (
        f"score wild cluster bootstrap (Kline-Santos 2012, boottest "
        f"studentization, {weight_type}); SEs remain cluster-robust"
    )
    base.model_info["n_boot"] = n_boot
    return base


def _ppmlhdfe_cr(
    formula: Optional[str],
    data: pd.DataFrame,
    y: Optional[str],
    x: Optional[List[str]],
    absorb: Optional[str],
    cluster: str,
    kind: str,
    *,
    separation: bool,
    maxiter: int,
    tol: float,
    alpha: float,
) -> EconometricResults:
    """CR2/CR3 (clubSandwich glm) SEs for PPML with LOW-dimensional FE.

    The reference-matching CR2/CR3 requires the FE-as-dummies design (the
    weighted projection does not carry the CR2 leverage through absorption —
    the absorbed variant differs ~1% with no published reference), so this
    path builds the dummy design like ``sp.fepois(vce="CR2")`` does and is
    guarded against high-dimensional FE. Matches R
    ``clubSandwich::vcovCR(glm(poisson), type=...)`` exactly; equal to
    ``sp.fepois`` on the same model.
    """
    import statsmodels.api as sm
    from scipy import stats as _stats

    from ..inference.jackknife import glm_cr_vcov

    base = ppmlhdfe(
        formula,
        data,
        y=y,
        x=x,
        absorb=absorb,
        cluster=cluster,
        separation=separation,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
    )
    y_arr, X, var_names, fe_list, data = _ppmlhdfe_design(formula, data, y, x, absorb)
    X, var_names = _drop_named_columns(X, var_names, base.model_info.get("omitted", []))
    n = len(y_arr)
    blocks: List[np.ndarray] = [X]
    if fe_list:
        blocks.append(np.ones((n, 1)))
        for fe_idx in fe_list:
            dmy = pd.get_dummies(pd.Series(fe_idx), drop_first=True).astype(float)
            if dmy.shape[1]:
                blocks.append(dmy.values)
    X_full = np.column_stack(blocks)
    if X_full.shape[1] >= n or X_full.shape[1] > 1000:
        raise MethodIncompatibility(
            f"ppmlhdfe(vce={kind!r}): the fixed-effects dummy design has "
            f"{X_full.shape[1]} columns — too many for the reference-matching "
            "full-dummy CR2/CR3 (the absorbed variant has no published "
            "reference).",
            recovery_hint="Use cluster= (CRV1), cluster=['a','b'] (two-way) "
            "or vce='wild' for high-dimensional fixed effects.",
        )
    codes = pd.factorize(data[cluster].values)[0]
    power = 0.5 if kind == "cr2" else 1.0
    se_full = glm_cr_vcov(X_full, y_arr, sm.families.Poisson(), codes, power=power)
    se = pd.Series(
        [float(se_full[var_names.index(str(v))]) for v in base.params.index],
        index=base.params.index,
    )
    z = base.params / se
    base.std_errors = se
    base.pvalues = pd.Series(2 * _stats.norm.sf(np.abs(z)), index=base.params.index)
    crit = _stats.norm.ppf(1 - alpha / 2)
    base.conf_int_lower = base.params - crit * se
    base.conf_int_upper = base.params + crit * se
    base.model_info = dict(base.model_info)
    # These SEs replace the fitted ones and their p-values are normal; mark
    # the fit so conf_int() / tidy() / sp.test use z as well.
    base.data_info = dict(base.data_info, inference="z")
    base.model_info["vcov_type"] = {
        "cr2": "CR2 cluster-robust (clubSandwich glm, Pustejovsky-Tipton 2018)",
        "cr3": "CR3 cluster-robust (clubSandwich glm jackknife-type)",
        "jackknife": "CR3 cluster-robust (clubSandwich glm jackknife-type)",
    }[kind]
    return base


def _ppmlhdfe_conley(
    formula: Optional[str],
    data: pd.DataFrame,
    y: Optional[str],
    x: Optional[List[str]],
    absorb: Optional[str],
    *,
    conley_lat: Optional[str],
    conley_lon: Optional[str],
    conley_cutoff: Optional[float],
    separation: bool,
    maxiter: int,
    tol: float,
    alpha: float,
) -> EconometricResults:
    """Conley spatial-HAC SEs for PPML (conleyreg spherical convention).

    GLM score sandwich on the FE-as-dummies design with R ``conleyreg``'s
    spherical uniform kernel — the reference implementation for GLM Conley
    (Stata ``acreg`` is OLS/2SLS-only and cannot follow ``ppmlhdfe``).
    Equal to ``sp.fepois(vce="conley")`` on the same model, which is pinned
    to ``conleyreg::conleyreg(model="poisson", kernel="uniform",
    dist_comp="spherical")``. Guarded against high-dimensional FE like the
    CR2/CR3 path (conleyreg's construction is dummy-based).
    """
    import statsmodels.api as sm
    from scipy import stats as _stats

    from ..inference.jackknife import glm_conley_vcov

    if conley_lat is None or conley_lon is None or conley_cutoff is None:
        raise MethodIncompatibility(
            "ppmlhdfe(vce='conley') requires conley_lat=, conley_lon= and "
            "conley_cutoff= (km).",
            recovery_hint="Pass the coordinate columns and distance cutoff.",
        )
    for c in (conley_lat, conley_lon):
        if c not in data.columns:
            raise MethodIncompatibility(
                f"ppmlhdfe(vce='conley'): coordinate column {c!r} not in data."
            )

    base = ppmlhdfe(
        formula,
        data,
        y=y,
        x=x,
        absorb=absorb,
        separation=separation,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
    )
    y_arr, X, var_names, fe_list, data = _ppmlhdfe_design(formula, data, y, x, absorb)
    X, var_names = _drop_named_columns(X, var_names, base.model_info.get("omitted", []))
    n = len(y_arr)
    if data[[conley_lat, conley_lon]].isna().any().any() or len(data) != n:
        raise MethodIncompatibility(
            "ppmlhdfe(vce='conley'): coordinates must be complete and "
            "row-aligned to the fitted sample."
        )
    blocks: List[np.ndarray] = [X]
    if fe_list:
        blocks.append(np.ones((n, 1)))
        for fe_idx in fe_list:
            dmy = pd.get_dummies(pd.Series(fe_idx), drop_first=True).astype(float)
            if dmy.shape[1]:
                blocks.append(dmy.values)
    X_full = np.column_stack(blocks)
    if X_full.shape[1] >= n or X_full.shape[1] > 1000:
        raise MethodIncompatibility(
            f"ppmlhdfe(vce='conley'): the fixed-effects dummy design has "
            f"{X_full.shape[1]} columns — too many for the reference-matching "
            "spatial HAC (conleyreg's construction is dummy-based).",
            recovery_hint="Use cluster= (CRV1) for high-dimensional " "fixed effects.",
        )
    se_full = glm_conley_vcov(
        X_full,
        y_arr,
        sm.families.Poisson(),
        data[conley_lat].to_numpy(dtype=float),
        data[conley_lon].to_numpy(dtype=float),
        float(conley_cutoff),
    )
    se = pd.Series(
        [float(se_full[var_names.index(str(v))]) for v in base.params.index],
        index=base.params.index,
    )
    z = base.params / se
    base.std_errors = se
    base.pvalues = pd.Series(2 * _stats.norm.sf(np.abs(z)), index=base.params.index)
    crit = _stats.norm.ppf(1 - alpha / 2)
    base.conf_int_lower = base.params - crit * se
    base.conf_int_upper = base.params + crit * se
    base.model_info = dict(base.model_info)
    # These SEs replace the fitted ones and their p-values are normal; mark
    # the fit so conf_int() / tidy() / sp.test use z as well.
    base.data_info = dict(base.data_info, inference="z")
    base.model_info["vcov_type"] = (
        f"Conley spatial HAC (conleyreg spherical, {conley_cutoff} km)"
    )
    return base


@markout_clusters
def ppmlhdfe(
    formula: Optional[str] = None,
    data: pd.DataFrame = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    absorb: Optional[str] = None,
    robust: str = "robust",
    cluster: Optional[Union[str, List[str], Tuple[str, str]]] = None,
    weights: Optional[str] = None,
    separation: Union[bool, str] = True,
    maxiter: int = 1000,
    tol: float = 1e-8,
    alpha: float = 0.05,
    vce: Optional[str] = None,
    wild_reps: int = 9999,
    wild_weight_type: str = "rademacher",
    seed: Optional[int] = None,
    conley_lat: Optional[str] = None,
    conley_lon: Optional[str] = None,
    conley_cutoff: Optional[float] = None,
    ssc: str = "stata",
    drop_singletons: bool = True,
) -> EconometricResults:
    """
    Pseudo-Poisson Maximum Likelihood with high-dimensional fixed effects.

    Implements Santos Silva & Tenreyro (2006) PPML estimator, the standard
    approach for gravity models and other trade/economic settings where:
    - The dependent variable has zeros
    - Log-linearization would be inconsistent under heteroskedasticity
    - High-dimensional fixed effects (origin, destination, year) must be absorbed

    Parameters
    ----------
    formula : str, optional
        Model formula. Fixed effects can be specified via ``|``:
        ``"trade ~ dist + contig | origin + destination + year"``
    data : pd.DataFrame
        Data containing all variables.
    y : str, optional
        Dependent variable name (alternative to formula).
    x : list of str, optional
        Independent variable names (alternative to formula).
    absorb : str, optional
        Fixed effects to absorb, e.g. ``"origin + destination + year"``.
        Overrides any FE specification in the formula.  Interacted
        effects are written ``"id + ind^year"`` (fixest) or
        ``"id + ind#year"`` (Stata ``absorb(ind#year)``); both mean one
        effect per observed level combination.  Regressors collinear
        with the absorbed effects (e.g. a time-invariant variable under a
        unit effect) are omitted with a warning and listed in
        ``model_info['omitted']``, as Stata reports them.
    robust : str, default "robust"
        Default is robust SE (as in Stata's ppmlhdfe). Options:
        "robust"/"hc1" (sandwich with the ``ssc`` small-sample factor),
        "hc0" (sandwich, no factor), "nonrobust".
    ssc : {"stata", "fixest", "none"}, default "stata"
        Small-sample factor applied to the heteroskedasticity-robust
        sandwich. ``"stata"`` multiplies by ``N/(N-1)``, the Stata
        ``glm``/``ppmlhdfe vce(robust)`` convention (matches
        ``ppmlhdfe`` at machine precision). ``"fixest"`` multiplies by
        ``(N-1)/(N-K)`` with ``K`` counting the slopes plus the absorbed
        fixed-effect levels (minus one per additional fixed-effect
        dimension for collinearity), the ``fixest::fepois`` default
        ``ssc(adj = TRUE, fixef.K = "full")``. ``"none"`` applies no
        factor. The three differ by less than ``sqrt(N/(N-K))`` and
        are documented, not competing, conventions; point estimates are
        unaffected. Clustered variances keep the ``G/(G-1)`` factor.
    cluster : str, optional
        Variable name for clustered standard errors (recommended for
        gravity models, e.g. cluster on country-pair). A pair
        ``cluster=["a", "b"]`` requests two-way clustering
        (Cameron-Gelbach-Miller 2011 inclusion-exclusion with the single
        ``G_min/(G_min-1)`` small-sample factor — byte-identical to Stata
        ``ppmlhdfe ..., cluster(a b)``).  ``cluster="city^year"`` (or
        ``"city#year"``) clusters on the level combination.
    weights : str, optional
        Weight variable name.
    separation : {True, 'drop', 'warn', False}, default True
        Handling of separated observations (outcomes perfectly predicted to
        be zero, whose maximum-likelihood fitted value is exactly zero).
        ``True`` / ``'drop'`` removes them before estimation with the
        ``fe`` and single-regressor ``simplex`` rules Stata ``ppmlhdfe``
        applies by default (``correia2020fast``), iterated to a fixed
        point; the count is reported as ``model_info['n_separated']`` and a
        warning is issued.  Slopes are unchanged by the removal (the rows
        carry no information about them), but ``N``, the cluster count and
        hence the ``G/(G-1)`` factor match Stata, and IRLS no longer chases
        fixed effects towards minus infinity.  ``'warn'`` is the pre-1.33
        behaviour (detect and warn, keep the rows); ``False`` skips the
        check.  The general ReLU rule is not implemented.
    drop_singletons : bool, default True
        Drop observations that are alone in their group of any absorbed
        fixed effect, iterated jointly with the separation rules to a
        fixed point, as Stata ``ppmlhdfe`` does by default.  Slopes and
        their robust / clustered variance are unchanged (a singleton is
        fitted exactly and has a zero FE-residualised design row), but
        ``N``, the cluster count, the small-sample factors and the pseudo
        R-squared now equal Stata's.  The count is
        ``model_info['n_singletons']``.  ``False`` keeps them, as R
        ``fixest::fepois`` does by default (``fixef.rm = "perfect"``).
    maxiter : int, default 1000
        Maximum IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    vce : str, optional
        Canonical SE-menu keyword. ``"robust"``/``"hc1"``/``"hc0"`` alias the
        ``robust=`` parameter. ``"wild"`` (with ``cluster=``) runs the
        boottest-convention score wild cluster bootstrap on the FE-absorbed
        design — exact at any FE dimensionality (the weighted-FWL reduction
        of the score numerator is exact) and byte-identical to
        ``sp.fepois(vce="wild")`` on low-dimensional FE.
        ``"CR2"``/``"CR3"``/``"jackknife"`` (with ``cluster=``) compute the
        clubSandwich glm bias-reduced SEs on the FE-as-dummies design
        (guarded against high-dimensional FE).
    wild_reps : int, default 9999
        Replications for ``vce="wild"`` (enumerates the 2^G grid when
        ``2**G <= wild_reps``).
    wild_weight_type : str, default "rademacher"
        Wild weight distribution.
    seed : int, optional
        RNG seed for sampled (non-enumerated) wild draws.

    Returns
    -------
    EconometricResults

    Notes
    -----
    PPML is consistent under the assumption E[y|x] = exp(x'beta), regardless
    of the true conditional variance. With robust SE it is a quasi-MLE
    estimator and does not assume Poisson variance.

    References
    ----------
    Santos Silva, J.M.C. & Tenreyro, S. (2006). "The Log of Gravity."
    Review of Economics and Statistics, 88(4), 641-658.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for o in range(6):
    ...     for d in range(6):
    ...         if o == d:
    ...             continue
    ...         for year in (2000, 2001):
    ...             dist = rng.uniform(1.0, 5.0)
    ...             contig = float(rng.integers(0, 2))
    ...             mu = np.exp(2.0 - 0.6 * np.log(dist) + 0.3 * contig
    ...                         + 0.1 * o - 0.1 * d)
    ...             rows.append(dict(trade=rng.poisson(mu), dist=dist,
    ...                              contig=contig, origin=o, dest=d, year=year,
    ...                              pair_id=f"{min(o, d)}_{max(o, d)}"))
    >>> df = pd.DataFrame(rows)
    >>> # Basic gravity model with formula fixed effects
    >>> res = sp.ppmlhdfe("trade ~ dist + contig | origin + dest + year",
    ...                   data=df, cluster="pair_id")
    >>> list(res.params.index)
    ['dist', 'contig']
    >>> # With absorb parameter instead of formula FE
    >>> res2 = sp.ppmlhdfe("trade ~ dist + contig", data=df,
    ...                    absorb="origin + dest + year",
    ...                    cluster="pair_id")
    >>> 'dist' in res2.params.index
    True
    """
    # --- Stata grammar on vce= / robust= (extended SE menu) ------------------
    # One parser for every spelling: True, 'vce(robust)', 'cluster pair_id',
    # 'CR2', 'wild', ... ``robust=True`` used to raise AttributeError and
    # ``vce='cluster'`` was rejected even with cluster= supplied.
    from ..core._vcov_spec import parse_se_request

    _raw = vce if vce is not None else robust
    if isinstance(_raw, str) and _raw.strip().lower() == "hc_robust":
        _raw = "robust"
    _se = parse_se_request(
        _raw,
        cluster,
        function="ppmlhdfe",
        supported=(
            "nonrobust",
            "robust",
            "hc0",
            "hc1",
            "cluster",
            "cr2",
            "cr3",
            "jackknife",
            "wild",
            "conley",
        ),
        multiway=True,
    )
    cluster = _se.cluster
    formula, data, absorb, cluster = _ppml_front_end(
        formula, data, y, x, absorb, cluster, weights
    )

    # Singletons and separated observations are removed from `data`
    # itself, so every SE path below (CRV1, CR2/CR3, wild, Conley) sees
    # the same sample.
    sep_mode = _normalise_ppml_separation(separation)
    sep_counts = {"singleton": 0, "fe": 0, "simplex": 0}
    n_before_sep = None
    if sep_mode == "drop" or (drop_singletons and absorb):
        _y0, _X0, _vn0, _fe0, _data0 = _ppmlhdfe_design(formula, data, y, x, absorb)
        _keep, sep_counts = _ppml_separation_mask(
            _y0,
            _X0,
            _fe0,
            drop_singletons=bool(drop_singletons),
            separation=(sep_mode == "drop"),
        )
        n_before_sep = int(len(_y0))
        n_sep = n_before_sep - int(_keep.sum())
        if n_sep:
            if not _keep.any():
                raise DataInsufficient(
                    "ppmlhdfe: every observation is a singleton or "
                    "separated (all-zero outcomes within each fixed-effect "
                    "group).",
                    recovery_hint="Check the outcome column; PPML needs "
                    "positive outcomes within at least one group.",
                    diagnostics={"n_dropped": n_sep, **sep_counts},
                )
            parts = []
            if sep_counts["singleton"]:
                parts.append(
                    f"{sep_counts['singleton']} singleton(s) of a fixed effect"
                )
            if sep_counts["fe"]:
                parts.append(f"{sep_counts['fe']} in all-zero fixed-effect groups")
            if sep_counts["simplex"]:
                parts.append(
                    f"{sep_counts['simplex']} perfectly predicted by a "
                    "single regressor"
                )
            warnings.warn(
                f"ppmlhdfe: dropped {n_sep} observation(s) "
                f"({'; '.join(parts)}), as Stata ppmlhdfe does by default. "
                "Pass drop_singletons=False / separation='warn' to keep them.",
                stacklevel=2,
            )
            data = _data0.iloc[np.flatnonzero(_keep)]
        if sep_mode == "drop":
            # The extended paths re-enter ppmlhdfe; the data is already clean.
            separation = False
        else:
            separation = sep_mode == "warn"
    elif sep_mode == "none":
        separation = False
    else:  # warn
        separation = True

    if _se.kind in ("cr2", "cr3", "jackknife", "wild", "conley"):
        _vce = _se.kind
    else:
        # One- and two-way CRV1 run through cluster=; the heteroskedastic
        # kinds keep their ssc small-sample convention below.
        _vce = None
        robust = "robust" if _se.kind == "cluster" else _se.kind
    if _vce is not None:
        if weights is not None:
            raise MethodIncompatibility(
                f"ppmlhdfe(vce={vce!r}) does not support weights= — the "
                "extended SE menu is unweighted.",
                recovery_hint="Drop weights= or use cluster= (CRV1).",
            )
        if _vce != "conley" and not isinstance(cluster, str):
            raise MethodIncompatibility(
                f"ppmlhdfe(vce={vce!r}) requires cluster='<one column>' " "(one-way).",
                recovery_hint="Pass cluster='pair_id' (or another id column).",
            )
        if _vce in ("wild", "wildbootstrap", "wild_cluster", "wcr", "boottest"):
            return _ppmlhdfe_wild(
                formula,
                data,
                y,
                x,
                absorb,
                cluster,
                separation=separation,
                maxiter=maxiter,
                tol=tol,
                alpha=alpha,
                n_boot=wild_reps,
                weight_type=wild_weight_type,
                seed=seed,
            )
        if _vce in ("cr2", "cr3", "jackknife"):
            return _ppmlhdfe_cr(
                formula,
                data,
                y,
                x,
                absorb,
                cluster,
                _vce,
                separation=separation,
                maxiter=maxiter,
                tol=tol,
                alpha=alpha,
            )
        if _vce == "conley":
            return _ppmlhdfe_conley(
                formula,
                data,
                y,
                x,
                absorb,
                conley_lat=conley_lat,
                conley_lon=conley_lon,
                conley_cutoff=conley_cutoff,
                separation=separation,
                maxiter=maxiter,
                tol=tol,
                alpha=alpha,
            )
        raise MethodIncompatibility(
            f"ppmlhdfe vce={vce!r} not recognised; use 'robust'/'hc1'/'hc0', "
            "'CR2'/'CR3'/'jackknife', or 'wild'.",
            recovery_hint="See the SE menu in docs/guides/grammar.md.",
        )

    y_arr, X, var_names, dep_var, formula_fe, data = _parse_formula_or_xy(
        formula, data, y, x, add_constant=True
    )
    n, k = X.shape

    # Parse fixed effects
    fe_indices_list = []
    fe_names = []

    # Absorb parameter takes priority over formula-parsed FE
    if absorb is not None:
        fe_names = [v.strip() for v in absorb.split("+")]
    elif formula_fe:
        fe_names = formula_fe

    for fe_var in fe_names:
        codes, _ = pd.factorize(data[fe_var].values)
        fe_indices_list.append(codes)

    # If we have FE, drop the constant from X (absorbed by FE)
    if fe_indices_list and var_names[0] == "_cons":
        X = X[:, 1:]
        var_names = var_names[1:]
        k = X.shape[1]

    # A regressor that is identically zero once separated rows are gone, or
    # that the fixed effects absorb (a time-invariant variable under a unit
    # effect), or that is collinear with earlier regressors is omitted, as
    # Stata reports it, rather than handed to the solver as a singular
    # X'WX (which overflowed and ended in LinAlgError).
    omitted: List[str] = []
    if k:
        dead_cols = _ppml_absorbed_columns(X, fe_indices_list)
        if dead_cols:
            omitted = [var_names[j] for j in dead_cols]
            warnings.warn(
                f"ppmlhdfe: omitted regressor(s) {omitted} — collinear with "
                "the absorbed fixed effects or other regressors (Stata "
                "reports them as omitted).",
                stacklevel=2,
            )
            live = [j for j in range(k) if j not in dead_cols]
            X = X[:, live]
            var_names = [var_names[j] for j in live]
            k = X.shape[1]

    # Weights
    w_arr = None
    if weights is not None:
        w_arr = data[weights].values.astype(np.float64)

    # Cluster variable(s) — a single column (one-way) or a pair (two-way CGM).
    cluster_arr = None
    cluster_pair = None
    if isinstance(cluster, (list, tuple)):
        if len(cluster) != 2:
            raise MethodIncompatibility(
                "ppmlhdfe cluster= accepts a single column name or a pair "
                f"[a, b] for two-way clustering; got {len(cluster)} entries.",
                recovery_hint="Pass cluster='id' or cluster=['a', 'b'].",
            )
        cluster_pair = (data[cluster[0]].values, data[cluster[1]].values)
    elif cluster is not None:
        cluster_arr = data[cluster].values

    # Separation detection
    sep_warnings = []
    if separation:
        sep_warnings = _detect_separation(
            y_arr,
            X,
            fe_indices_list if fe_indices_list else None,
        )
        for sw in sep_warnings:
            warnings.warn(sw)

    # Fit
    beta, mu, converged, n_iter, X_dm = _ppml_hdfe_irls(
        y_arr,
        X,
        fe_indices_list=fe_indices_list if fe_indices_list else None,
        weights=w_arr,
        maxiter=maxiter,
        tol=tol,
    )
    if not converged:
        warnings.warn(f"PPML did not converge in {maxiter} iterations")

    residuals = y_arr - mu

    # Variance-covariance: when FE are absorbed, the FWL-correct
    # sandwich uses the FE-residualised design (X_dm), not the raw X.
    # Using raw X would inflate (X'WX)^{-1} (more apparent regressor
    # variability than is identifying β), yielding HC1 SE 50% larger
    # than fixest::fepois / Stata ppmlhdfe (parity finding #6).
    X_for_vcov = X_dm if (X_dm is not None) else X
    ssc_key = str(ssc).lower()
    if ssc_key not in {"stata", "fixest", "none"}:
        raise ValueError("ssc must be one of 'stata', 'fixest', 'none'")
    # K under the fixest convention: slopes + absorbed FE levels, minus
    # one level per additional FE dimension (the collinear level fixest
    # removes; fixest ssc() documentation, fixef.K = "full").
    fe_levels = [int(len(np.unique(codes))) for codes in fe_indices_list]
    k_fe_fixest = (sum(fe_levels) - (len(fe_levels) - 1)) if fe_levels else 0
    ssc_factor = 1.0
    if cluster_pair is not None:
        vcov = _twoway_cluster_vcov(
            X_for_vcov, mu, residuals, cluster_pair[0], cluster_pair[1], weights=w_arr
        )
    elif cluster_arr is None and robust.lower() in ("robust", "hc1"):
        # Plain sandwich, then the documented small-sample convention.
        vcov = _poisson_vcov(X_for_vcov, mu, residuals, "hc0", None, weights=w_arr)
        n_v, k_v = X_for_vcov.shape
        if ssc_key == "stata":
            ssc_factor = n_v / (n_v - 1.0)
        elif ssc_key == "fixest":
            # fixest::fepois applies N/(N-K) to the GLM sandwich (the
            # (N-1)/(N-K) form is its OLS convention).
            ssc_factor = n_v / max(n_v - (k_v + k_fe_fixest), 1.0)
        vcov = vcov * ssc_factor
    else:
        vcov = _poisson_vcov(
            X_for_vcov, mu, residuals, robust, cluster_arr, weights=w_arr
        )
    se = np.sqrt(np.diag(vcov))

    # Log-likelihood (Poisson quasi-likelihood; weighted under weights)
    ll = _poisson_loglik(y_arr, mu, w_arr)

    # Null model
    mu_null = np.full(n, np.average(y_arr, weights=w_arr))
    ll_null = _poisson_loglik(y_arr, mu_null, w_arr)

    lr_chi2 = 2 * (ll - ll_null)
    lr_pvalue = stats.chi2.sf(lr_chi2, max(k - 1, 1)) if k > 1 else np.nan
    pseudo_r2 = 1 - ll / ll_null

    # Deviance
    deviance = 2 * np.sum(
        np.where(
            y_arr > 0,
            y_arr * np.log(np.maximum(y_arr, 1e-300) / mu),
            0,
        )
        - (y_arr - mu)
    )
    pearson_chi2 = np.sum((y_arr - mu) ** 2 / np.maximum(mu, 1e-300))

    aic = -2 * ll + 2 * k
    bic = -2 * ll + np.log(n) * k

    # Number of FE levels absorbed
    n_fe = sum(len(np.unique(idx)) for idx in fe_indices_list) if fe_indices_list else 0

    params_series = pd.Series(beta, index=var_names)
    se_series = pd.Series(se, index=var_names)

    model_info = {
        "model_type": "PPML" + (" HDFE" if fe_indices_list else ""),
        "family": "Poisson (Pseudo-MLE)",
        "link": "log",
        "method": (
            "IRLS (Quasi-MLE)"
            + (" + Alternating Projection" if fe_indices_list else "")
        ),
        "robust": robust,
        "cluster": cluster,
        "converged": converged,
        "iterations": n_iter,
        "ll": ll,
        "ll_null": ll_null,
        "lr_chi2": lr_chi2,
        "lr_pvalue": lr_pvalue,
        "pseudo_r2": pseudo_r2,
        "aic": aic,
        "bic": bic,
        "absorbed_fe": fe_names if fe_names else None,
        "n_fe_levels": n_fe,
        "separation_warnings": sep_warnings,
        "separation": sep_mode,
        "n_separated": int(sep_counts["fe"] + sep_counts["simplex"]),
        "n_singletons": int(sep_counts["singleton"]),
        "n_separated_by_rule": {
            "fe": int(sep_counts["fe"]),
            "simplex": int(sep_counts["simplex"]),
        },
        "drop_singletons": bool(drop_singletons),
        "omitted": list(omitted),
    }

    if cluster_arr is not None:
        n_cluster = len(np.unique(cluster_arr))
    elif cluster_pair is not None:
        # Two-way CGM inference binds on the smaller cluster dimension.
        n_cluster = min(
            len(np.unique(cluster_pair[0])), len(np.unique(cluster_pair[1]))
        )
    else:
        n_cluster = None
    if w_arr is not None:
        from ..core._agent_summary import note_weight_diagnostics

        _one_way = cluster_arr is not None and len(cluster_arr) == len(w_arr)
        note_weight_diagnostics(
            model_info,
            w_arr,
            (
                "cluster"
                if (cluster_arr is not None or cluster_pair is not None)
                else ("robust" if robust.lower() in ("robust", "hc1") else "classical")
            ),
            entry="sp.ppmlhdfe",
            robust_option="robust='robust' (the default) or cluster=",
            classical_reading="frequency weights (Stata [fw=])",
            classical_assumption=(
                "frequencies (each row standing for w identical observations)"
            ),
            cluster_keys=cluster_arr if _one_way else None,
            cluster=cluster,
        )
    data_info = {
        "nobs": n,
        "df_model": k,
        "df_resid": n - k - n_fe,
        # z / chi2 inference, as Stata's ppmlhdfe.
        "inference": "z",
        "dependent_var": dep_var,
        "fitted_values": mu,
        "residuals": residuals,
        "X": X,
        "y": y_arr,
        "var_cov": vcov,
        "var_names": var_names,
        "n_clusters": n_cluster,
    }

    diagnostics = {
        "Log-Likelihood": ll,
        "Log-Lik (null)": ll_null,
        "Pseudo R2": pseudo_r2,
        "Deviance": deviance,
        "Pearson chi2": pearson_chi2,
        "AIC": aic,
        "BIC": bic,
        "N absorbed FE": n_fe,
    }
    if n_cluster is not None:
        diagnostics["N clusters"] = n_cluster

    model_info["alpha"] = alpha
    return EconometricResults(
        params=params_series,
        std_errors=se_series,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
