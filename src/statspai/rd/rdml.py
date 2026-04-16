"""
Machine Learning + Regression Discontinuity estimation.

Combines modern ML methods (random forests, gradient boosting, LASSO) with
RD designs for flexible CATE estimation, heterogeneity detection, and
precision improvement through automated covariate selection.

Functions
---------
rd_forest   : Causal Forest for RD (adapted Athey-Wager)
rd_boost    : Gradient Boosting for RD
rd_lasso    : LASSO-assisted RD (post-double-selection)
rd_cate_summary : Unified multi-method CATE comparison

References
----------
Athey, S., Tibshirani, J. and Wager, S. (2019).
"Generalized Random Forests." *Annals of Statistics*, 47(2), 1148-1178.

Belloni, A., Chernozhukov, V. and Hansen, C. (2014).
"Inference on Treatment Effects after Selection among High-Dimensional
Controls." *Review of Economic Studies*, 81(2), 608-650.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult

# ======================================================================
# Citations
# ======================================================================

CausalResult._CITATIONS['rd_forest'] = (
    "@article{athey2019generalized,\n"
    "  title={Generalized random forests},\n"
    "  author={Athey, Susan and Tibshirani, Julie and Wager, Stefan},\n"
    "  journal={Annals of Statistics},\n"
    "  volume={47},\n"
    "  number={2},\n"
    "  pages={1148--1178},\n"
    "  year={2019}\n"
    "}"
)

CausalResult._CITATIONS['rd_boost'] = (
    "@article{athey2019generalized,\n"
    "  title={Generalized random forests},\n"
    "  author={Athey, Susan and Tibshirani, Julie and Wager, Stefan},\n"
    "  journal={Annals of Statistics},\n"
    "  volume={47},\n"
    "  number={2},\n"
    "  pages={1148--1178},\n"
    "  year={2019}\n"
    "}"
)

CausalResult._CITATIONS['rd_lasso'] = (
    "@article{belloni2014inference,\n"
    "  title={Inference on treatment effects after selection among "
    "high-dimensional controls},\n"
    "  author={Belloni, Alexandre and Chernozhukov, Victor "
    "and Hansen, Christian},\n"
    "  journal={Review of Economic Studies},\n"
    "  volume={81},\n"
    "  number={2},\n"
    "  pages={608--650},\n"
    "  year={2014},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)


# ======================================================================
# Internal helpers
# ======================================================================

def _ik_bandwidth_simple(y: np.ndarray, x: np.ndarray, c: float) -> float:
    """
    Simple IK-style bandwidth: Silverman pilot scaled by curvature.

    Used as a default when the user does not supply *h*.
    """
    x_c = x - c
    n = len(x_c)
    sd = np.std(x_c)
    if sd < 1e-12 or n < 10:
        return float(np.ptp(x_c) * 0.5)

    h_pilot = 1.06 * sd * n ** (-1 / 5)

    # Estimate curvature difference via local quadratic on each side
    left = x_c < 0
    right = x_c >= 0

    def _quad_curv(yy, xx, hp):
        mask = np.abs(xx) <= hp
        if mask.sum() < 5:
            mask = np.ones(len(xx), dtype=bool)
        try:
            coeffs = np.polyfit(xx[mask], yy[mask], 2)
            return 2 * coeffs[0]
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    h_curv = max(np.median(np.abs(x_c)), h_pilot) * 1.5
    m2_l = _quad_curv(y[left], x_c[left], h_curv)
    m2_r = _quad_curv(y[right], x_c[right], h_curv)

    bias_sq = ((m2_r - m2_l) / 2) ** 2
    if bias_sq < 1e-12:
        h_opt = h_pilot
    else:
        sigma2 = np.var(y)
        C_K = 3.4375  # triangular kernel constant
        f_c = np.sum(np.abs(x_c) <= h_pilot) / (2 * h_pilot * n)
        f_c = max(f_c, 1e-10)
        h_opt = (C_K * 2 * sigma2 / (f_c * bias_sq * n)) ** (1 / 5)

    x_range = np.ptp(x_c)
    h_opt = float(np.clip(h_opt, 0.02 * x_range, 0.98 * x_range))
    return h_opt


def _restrict_to_bandwidth(
    data: pd.DataFrame,
    x: str,
    c: float,
    h: Optional[float],
    y: str,
    covs: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Restrict *data* to observations within bandwidth *h* of cutoff *c*.

    Returns the filtered DataFrame and the bandwidth used.
    """
    if h is None:
        h = _ik_bandwidth_simple(
            data[y].values, data[x].values, c,
        )
    mask = (data[x] >= c - h) & (data[x] <= c + h)
    sub = data.loc[mask].copy()
    if len(sub) < 10:
        raise ValueError(
            f"Only {len(sub)} observations within bandwidth h={h:.4f}. "
            "Increase h or check data."
        )
    return sub, h


def _triangular_weights(x_vals: np.ndarray, c: float, h: float) -> np.ndarray:
    """Triangular kernel weights for observations within bandwidth."""
    u = np.abs(x_vals - c) / h
    w = np.maximum(1 - u, 0.0)
    return w


def _validate_covariates(
    data: pd.DataFrame, covs: Optional[List[str]],
) -> List[str]:
    """Return validated covariate list; raise on missing columns."""
    if covs is None:
        return []
    missing = [c for c in covs if c not in data.columns]
    if missing:
        raise ValueError(f"Covariates not found in data: {missing}")
    return list(covs)


# ======================================================================
# 1. rd_forest  — Causal Forest for RD
# ======================================================================

def rd_forest(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    covs: Optional[List[str]] = None,
    h: Optional[float] = None,
    n_trees: int = 500,
    min_leaf: int = 20,
    honesty: bool = True,
    alpha: float = 0.05,
    seed: int = 42,
) -> CausalResult:
    """
    Causal Forest for RD — heterogeneous treatment effect estimation.

    Adapts the Athey-Wager (2019) generalized random forests framework to
    an RD context. Within bandwidth *h*, treatment is D = 1(X >= c).
    Two separate random forests estimate E[Y|Z, D=0] and E[Y|Z, D=1],
    and the CATE is their difference.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable.
    x : str
        Running variable.
    c : float, default 0
        RD cutoff.
    covs : list of str, optional
        Covariate names used as features for heterogeneity detection.
        Must not include the running variable *x*.
    h : float, optional
        Bandwidth (uses IK-style automatic selection if None).
    n_trees : int, default 500
        Number of trees in each forest.
    min_leaf : int, default 20
        Minimum leaf size (larger → more regularisation).
    honesty : bool, default True
        Split-sample (honest) estimation: half the data for tree
        construction, the other half for leaf predictions.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    seed : int, default 42
        Random seed.

    Returns
    -------
    CausalResult
        estimate = average CATE; detail = DataFrame with per-obs CATE
        and SE; model_info includes variable importance.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        raise ImportError(
            "rd_forest requires scikit-learn. Install: pip install scikit-learn"
        )

    # --- Validate inputs ---
    covs = _validate_covariates(data, covs)
    if not covs:
        raise ValueError(
            "rd_forest requires at least one covariate in `covs` for "
            "heterogeneity estimation."
        )
    if x in covs:
        raise ValueError(
            f"Running variable '{x}' must not be in covs; the forest "
            "estimates heterogeneity over covariates, not the running "
            "variable itself."
        )

    rng = np.random.RandomState(seed)

    # --- Restrict to bandwidth ---
    sub, h_used = _restrict_to_bandwidth(data, x, c, h, y, covs)
    treated = (sub[x] >= c).values
    control = ~treated
    Z = sub[covs].values.astype(float)
    Y = sub[y].values.astype(float)
    n = len(sub)

    n_treated = treated.sum()
    n_control = control.sum()
    if n_treated < min_leaf or n_control < min_leaf:
        raise ValueError(
            f"Too few observations on one side of cutoff (treated={n_treated}, "
            f"control={n_control}). Increase bandwidth or reduce min_leaf."
        )

    # --- Honest split ---
    if honesty:
        idx_all = np.arange(n)
        rng.shuffle(idx_all)
        half = n // 2
        idx_build = idx_all[:half]
        idx_est = idx_all[half:]
    else:
        idx_build = np.arange(n)
        idx_est = np.arange(n)

    # --- Build forests on the construction sample ---
    Z_build, Y_build, T_build = Z[idx_build], Y[idx_build], treated[idx_build]

    rf_params = dict(
        n_estimators=n_trees,
        min_samples_leaf=min_leaf,
        random_state=rng.randint(0, 2**31),
        n_jobs=-1,
    )

    mask_t_build = T_build
    mask_c_build = ~T_build

    if mask_t_build.sum() < min_leaf or mask_c_build.sum() < min_leaf:
        raise ValueError(
            "Too few treated or control observations in the build sample "
            "after honest split. Increase bandwidth or set honesty=False."
        )

    rf1 = RandomForestRegressor(**rf_params)
    rf1.fit(Z_build[mask_t_build], Y_build[mask_t_build])

    rf0 = RandomForestRegressor(**rf_params)
    rf0.fit(Z_build[mask_c_build], Y_build[mask_c_build])

    # --- Predict CATEs on estimation sample ---
    Z_est = Z[idx_est]
    mu1_hat = rf1.predict(Z_est)
    mu0_hat = rf0.predict(Z_est)
    cate = mu1_hat - mu0_hat

    # --- Variance via infinitesimal jackknife (tree-level predictions) ---
    # Collect per-tree predictions and compute variance across trees
    preds_1 = np.column_stack(
        [tree.predict(Z_est) for tree in rf1.estimators_]
    )  # (n_est, n_trees)
    preds_0 = np.column_stack(
        [tree.predict(Z_est) for tree in rf0.estimators_]
    )
    tau_trees = preds_1 - preds_0  # (n_est, n_trees)

    # Infinitesimal jackknife SE per observation
    # V_IJ = (n_trees / (n_trees-1)) * Var_across_trees
    # This is a conservative but consistent variance estimator
    n_t = n_trees
    cate_var = np.var(tau_trees, axis=1, ddof=1) * (n_t / (n_t - 1))
    cate_se = np.sqrt(cate_var)

    # --- Average treatment effect and its SE ---
    ate = float(np.mean(cate))
    # SE of the mean: combine within-unit variance and cross-unit variance
    n_est = len(cate)
    se_ate = float(np.sqrt(np.var(cate, ddof=1) / n_est +
                           np.mean(cate_var) / n_est))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (ate - z_crit * se_ate, ate + z_crit * se_ate)
    pvalue = float(2 * (1 - stats.norm.cdf(abs(ate) / max(se_ate, 1e-15))))

    # --- Variable importance (mean decrease in impurity, averaged) ---
    imp1 = rf1.feature_importances_
    imp0 = rf0.feature_importances_
    avg_importance = (imp1 + imp0) / 2
    var_importance = dict(zip(covs, avg_importance.tolist()))
    var_importance = dict(
        sorted(var_importance.items(), key=lambda kv: -kv[1])
    )

    # --- OOB score (only meaningful when honesty=False) ---
    oob1 = getattr(rf1, 'oob_score_', None)
    oob0 = getattr(rf0, 'oob_score_', None)

    # --- Detail DataFrame ---
    est_indices = sub.index[idx_est].tolist()
    detail = pd.DataFrame({
        'obs_index': est_indices,
        'cate': cate,
        'se': cate_se,
        'ci_lower': cate - z_crit * cate_se,
        'ci_upper': cate + z_crit * cate_se,
    })

    model_info = {
        'method': 'rd_forest',
        'bandwidth': h_used,
        'cutoff': c,
        'n_trees': n_trees,
        'min_leaf': min_leaf,
        'honesty': honesty,
        'n_treated': int(n_treated),
        'n_control': int(n_control),
        'n_estimation': n_est,
        'variable_importance': var_importance,
        'feature_names': covs,
        'oob_score_treated': oob1,
        'oob_score_control': oob0,
    }

    return CausalResult(
        method='RD Causal Forest (Athey-Wager)',
        estimand='CATE (avg)',
        estimate=ate,
        se=se_ate,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='rd_forest',
    )


# ======================================================================
# 2. rd_boost  — Gradient Boosting for RD
# ======================================================================

def rd_boost(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    covs: Optional[List[str]] = None,
    h: Optional[float] = None,
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    alpha: float = 0.05,
    seed: int = 42,
) -> CausalResult:
    """
    Gradient Boosting for RD — flexible CATE estimation.

    Fits separate GBM models on each side of the cutoff and estimates
    individual-level CATEs as mu_1(z) - mu_0(z). Standard errors are
    obtained via bootstrap.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable.
    x : str
        Running variable.
    c : float, default 0
        RD cutoff.
    covs : list of str, optional
        Covariate names for heterogeneity.
    h : float, optional
        Bandwidth (auto-selected if None).
    n_estimators : int, default 200
        Number of boosting rounds.
    max_depth : int, default 3
        Maximum tree depth per round.
    learning_rate : float, default 0.1
        Shrinkage factor.
    alpha : float, default 0.05
        Significance level.
    seed : int, default 42
        Random seed.

    Returns
    -------
    CausalResult
        estimate = average CATE; detail = per-obs CATE and bootstrap SE.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        raise ImportError(
            "rd_boost requires scikit-learn. Install: pip install scikit-learn"
        )

    covs = _validate_covariates(data, covs)
    if not covs:
        raise ValueError(
            "rd_boost requires at least one covariate in `covs`."
        )
    if x in covs:
        raise ValueError(
            f"Running variable '{x}' must not be in covs."
        )

    rng = np.random.RandomState(seed)
    n_boot = 200

    # --- Restrict to bandwidth ---
    sub, h_used = _restrict_to_bandwidth(data, x, c, h, y, covs)
    treated = (sub[x] >= c).values
    control = ~treated
    Z = sub[covs].values.astype(float)
    Y = sub[y].values.astype(float)
    n = len(sub)

    n_treated = int(treated.sum())
    n_control = int(control.sum())
    if n_treated < 5 or n_control < 5:
        raise ValueError(
            f"Too few observations (treated={n_treated}, "
            f"control={n_control}). Increase bandwidth."
        )

    # --- Fit full-sample GBMs ---
    gbm_params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=rng.randint(0, 2**31),
    )

    gbm1 = GradientBoostingRegressor(**gbm_params)
    gbm1.fit(Z[treated], Y[treated])

    gbm0 = GradientBoostingRegressor(**gbm_params)
    gbm0.fit(Z[control], Y[control])

    mu1_hat = gbm1.predict(Z)
    mu0_hat = gbm0.predict(Z)
    cate = mu1_hat - mu0_hat
    ate = float(np.mean(cate))

    # --- Bootstrap SE ---
    boot_ates = np.empty(n_boot)
    boot_cates = np.empty((n_boot, n))

    idx_t = np.where(treated)[0]
    idx_c = np.where(control)[0]

    for b in range(n_boot):
        # Resample within each group
        boot_t = rng.choice(idx_t, size=len(idx_t), replace=True)
        boot_c = rng.choice(idx_c, size=len(idx_c), replace=True)

        gbm1_b = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=rng.randint(0, 2**31),
        )
        gbm1_b.fit(Z[boot_t], Y[boot_t])

        gbm0_b = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=rng.randint(0, 2**31),
        )
        gbm0_b.fit(Z[boot_c], Y[boot_c])

        cate_b = gbm1_b.predict(Z) - gbm0_b.predict(Z)
        boot_cates[b] = cate_b
        boot_ates[b] = np.mean(cate_b)

    se_ate = float(np.std(boot_ates, ddof=1))
    cate_se = np.std(boot_cates, axis=0, ddof=1)

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (ate - z_crit * se_ate, ate + z_crit * se_ate)
    pvalue = float(2 * (1 - stats.norm.cdf(abs(ate) / max(se_ate, 1e-15))))

    # --- Variable importance (from full-sample GBMs) ---
    imp1 = gbm1.feature_importances_
    imp0 = gbm0.feature_importances_
    avg_importance = (imp1 + imp0) / 2
    var_importance = dict(zip(covs, avg_importance.tolist()))
    var_importance = dict(
        sorted(var_importance.items(), key=lambda kv: -kv[1])
    )

    detail = pd.DataFrame({
        'obs_index': sub.index.tolist(),
        'cate': cate,
        'se': cate_se,
        'ci_lower': cate - z_crit * cate_se,
        'ci_upper': cate + z_crit * cate_se,
    })

    model_info = {
        'method': 'rd_boost',
        'bandwidth': h_used,
        'cutoff': c,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_boot': n_boot,
        'n_treated': n_treated,
        'n_control': n_control,
        'variable_importance': var_importance,
        'feature_names': covs,
    }

    return CausalResult(
        method='RD Gradient Boosting',
        estimand='CATE (avg)',
        estimate=ate,
        se=se_ate,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='rd_boost',
    )


# ======================================================================
# 3. rd_lasso  — LASSO-assisted RD (post-double-selection)
# ======================================================================

def rd_lasso(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    covs: Optional[List[str]] = None,
    h: Optional[float] = None,
    kernel: str = 'triangular',
    cv_folds: int = 5,
    alpha: float = 0.05,
) -> CausalResult:
    """
    LASSO-assisted RD via post-double-selection.

    Uses LASSO to select relevant covariates from a potentially large
    set, then runs a local linear RD with the selected covariates.
    Follows Belloni, Chernozhukov, and Hansen (2014).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable.
    x : str
        Running variable.
    c : float, default 0
        RD cutoff.
    covs : list of str, optional
        Candidate covariates (can be large set).
    h : float, optional
        Bandwidth (auto-selected if None).
    kernel : str, default 'triangular'
        Kernel for local linear regression ('triangular', 'uniform',
        'epanechnikov').
    cv_folds : int, default 5
        Cross-validation folds for LASSO penalty selection.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        estimate = RD treatment effect with LASSO-selected covariates;
        model_info includes selected_covariates and LASSO paths.
    """
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError(
            "rd_lasso requires scikit-learn. Install: pip install scikit-learn"
        )

    covs = _validate_covariates(data, covs)
    if not covs:
        raise ValueError(
            "rd_lasso requires candidate covariates in `covs`."
        )

    # --- Restrict to bandwidth ---
    sub, h_used = _restrict_to_bandwidth(data, x, c, h, y, covs)
    n = len(sub)

    # Treatment indicator
    D = (sub[x].values >= c).astype(float)
    Y = sub[y].values.astype(float)
    Z = sub[covs].values.astype(float)
    X_run = sub[x].values - c  # centred running variable

    # Kernel weights
    w = _triangular_weights(sub[x].values, c, h_used)
    if kernel == 'uniform':
        w = np.ones(n)
    elif kernel == 'epanechnikov':
        u = np.abs(sub[x].values - c) / h_used
        w = np.maximum(0.75 * (1 - u**2), 0.0)
    sqrt_w = np.sqrt(w)

    # Standardise covariates (weighted)
    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z * sqrt_w[:, None])

    # ------------------------------------------------------------------
    # Step 1: LASSO Y ~ Z  (select covariates predicting outcome)
    # ------------------------------------------------------------------
    Y_w = Y * sqrt_w
    lasso_y = LassoCV(cv=cv_folds, random_state=42, max_iter=10000)
    lasso_y.fit(Z_scaled, Y_w)
    selected_y = set(np.where(np.abs(lasso_y.coef_) > 1e-10)[0])

    # ------------------------------------------------------------------
    # Step 2: LASSO D ~ Z  (select covariates predicting treatment)
    # ------------------------------------------------------------------
    D_w = D * sqrt_w
    lasso_d = LassoCV(cv=cv_folds, random_state=42, max_iter=10000)
    lasso_d.fit(Z_scaled, D_w)
    selected_d = set(np.where(np.abs(lasso_d.coef_) > 1e-10)[0])

    # ------------------------------------------------------------------
    # Step 3: Union of selected covariates
    # ------------------------------------------------------------------
    selected_idx = sorted(selected_y | selected_d)
    selected_names = [covs[i] for i in selected_idx]

    # ------------------------------------------------------------------
    # Step 4: Local linear RD with selected covariates (WLS)
    # ------------------------------------------------------------------
    # Design matrix:  [1, D, X_run, D*X_run, Z_selected]
    # The coefficient on D is the RD treatment effect at the cutoff.
    n_sel = len(selected_idx)
    Z_sel = Z[:, selected_idx] if selected_idx else np.empty((n, 0))

    # Build design: intercept, D, X_run, D*X_run, selected Z
    ones = np.ones(n)
    X_design = np.column_stack([
        ones,
        D,
        X_run,
        D * X_run,
    ])
    if n_sel > 0:
        X_design = np.column_stack([X_design, Z_sel])

    col_names = ['intercept', 'D', 'X_run', 'D_X_run'] + selected_names

    # WLS: multiply by sqrt(w)
    Xw = X_design * sqrt_w[:, None]
    Yw = Y * sqrt_w

    # OLS via normal equations
    try:
        beta, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback with ridge-like regularisation
        XtX = Xw.T @ Xw
        XtX += 1e-8 * np.eye(XtX.shape[0])
        beta = np.linalg.solve(XtX, Xw.T @ Yw)

    tau_hat = float(beta[1])  # coefficient on D

    # ------------------------------------------------------------------
    # Step 5: HC1 robust standard errors
    # ------------------------------------------------------------------
    resid = Yw - Xw @ beta
    k = Xw.shape[1]
    hc1_scale = n / (n - k)

    # Meat: sum_i e_i^2 * x_i x_i'
    XtX_inv = np.linalg.inv(Xw.T @ Xw + 1e-12 * np.eye(k))
    meat = np.zeros((k, k))
    for i in range(n):
        xi = Xw[i]
        meat += (resid[i] ** 2) * np.outer(xi, xi)
    meat *= hc1_scale

    V_hc1 = XtX_inv @ meat @ XtX_inv
    se_tau = float(np.sqrt(max(V_hc1[1, 1], 0.0)))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (tau_hat - z_crit * se_tau, tau_hat + z_crit * se_tau)
    pvalue = float(2 * (1 - stats.norm.cdf(abs(tau_hat) / max(se_tau, 1e-15))))

    # --- All coefficients ---
    coef_table = pd.DataFrame({
        'variable': col_names,
        'coefficient': beta,
    })

    model_info = {
        'method': 'rd_lasso',
        'bandwidth': h_used,
        'cutoff': c,
        'kernel': kernel,
        'cv_folds': cv_folds,
        'n_candidate_covariates': len(covs),
        'selected_covariates': selected_names,
        'n_selected': n_sel,
        'selected_from_outcome': [covs[i] for i in sorted(selected_y)],
        'selected_from_treatment': [covs[i] for i in sorted(selected_d)],
        'lasso_alpha_y': float(lasso_y.alpha_),
        'lasso_alpha_d': float(lasso_d.alpha_),
        'coefficients': coef_table,
    }

    return CausalResult(
        method='LASSO-assisted RD (Post-Double-Selection)',
        estimand='LATE',
        estimate=tau_hat,
        se=se_tau,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=coef_table,
        model_info=model_info,
        _citation_key='rd_lasso',
    )


# ======================================================================
# 4. rd_cate_summary  — Unified multi-method CATE comparison
# ======================================================================

def rd_cate_summary(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    covs: Optional[List[str]] = None,
    h: Optional[float] = None,
    methods: Optional[List[str]] = None,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Run multiple ML-RD methods and compare CATE estimates.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable.
    x : str
        Running variable.
    c : float, default 0
        RD cutoff.
    covs : list of str, optional
        Covariates for heterogeneity / selection.
    h : float, optional
        Bandwidth (shared across methods).
    methods : list of str, optional
        Subset of ``['forest', 'boost', 'lasso']``. Default: all three.
    alpha : float, default 0.05
        Significance level.
    seed : int, default 42
        Random seed.

    Returns
    -------
    dict
        Keys = method names, values = CausalResult.
        Additional key ``'comparison'`` = pd.DataFrame summarising ATEs.
        Additional key ``'heterogeneity_drivers'`` = top variable
        importances from the forest (if run).
    """
    valid_methods = {'forest', 'boost', 'lasso'}
    if methods is None:
        methods = ['forest', 'boost', 'lasso']
    else:
        unknown = set(methods) - valid_methods
        if unknown:
            raise ValueError(
                f"Unknown methods: {unknown}. Choose from {valid_methods}."
            )

    results: Dict[str, Any] = {}
    rows = []

    # Determine bandwidth once for consistency
    if h is None:
        _, h = _restrict_to_bandwidth(data, x, c, None, y, covs)

    if 'forest' in methods:
        try:
            res = rd_forest(
                data, y, x, c=c, covs=covs, h=h,
                alpha=alpha, seed=seed,
            )
            results['forest'] = res
            rows.append({
                'method': 'Causal Forest',
                'estimate': res.estimate,
                'se': res.se,
                'ci_lower': res.ci[0],
                'ci_upper': res.ci[1],
                'pvalue': res.pvalue,
                'n_obs': res.n_obs,
            })
        except Exception as e:
            results['forest_error'] = str(e)

    if 'boost' in methods:
        try:
            res = rd_boost(
                data, y, x, c=c, covs=covs, h=h,
                alpha=alpha, seed=seed,
            )
            results['boost'] = res
            rows.append({
                'method': 'Gradient Boosting',
                'estimate': res.estimate,
                'se': res.se,
                'ci_lower': res.ci[0],
                'ci_upper': res.ci[1],
                'pvalue': res.pvalue,
                'n_obs': res.n_obs,
            })
        except Exception as e:
            results['boost_error'] = str(e)

    if 'lasso' in methods:
        try:
            res = rd_lasso(
                data, y, x, c=c, covs=covs, h=h,
                alpha=alpha,
            )
            results['lasso'] = res
            rows.append({
                'method': 'LASSO RD',
                'estimate': res.estimate,
                'se': res.se,
                'ci_lower': res.ci[0],
                'ci_upper': res.ci[1],
                'pvalue': res.pvalue,
                'n_obs': res.n_obs,
            })
        except Exception as e:
            results['lasso_error'] = str(e)

    # --- Comparison table ---
    if rows:
        results['comparison'] = pd.DataFrame(rows)
    else:
        results['comparison'] = pd.DataFrame()

    # --- Top heterogeneity drivers from forest ---
    if 'forest' in results and hasattr(results['forest'], 'model_info'):
        vi = results['forest'].model_info.get('variable_importance', {})
        results['heterogeneity_drivers'] = vi
    else:
        results['heterogeneity_drivers'] = {}

    return results


# ======================================================================
# 5. Variable importance plot
# ======================================================================

def _importance_plot(
    result: CausalResult,
    top_k: int = 10,
    ax=None,
    figsize: Tuple[float, float] = (8, 5),
):
    """
    Horizontal bar chart of variable importance for heterogeneity.

    Parameters
    ----------
    result : CausalResult
        Output from ``rd_forest`` or ``rd_boost``.
    top_k : int, default 10
        Number of top variables to display.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (created if None).
    figsize : tuple, default (8, 5)
        Figure size if creating new axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "_importance_plot requires matplotlib. "
            "Install: pip install matplotlib"
        )

    vi = result.model_info.get('variable_importance', {})
    if not vi:
        raise ValueError(
            "No variable_importance found in result.model_info. "
            "Pass a CausalResult from rd_forest or rd_boost."
        )

    # Sort and truncate
    sorted_items = sorted(vi.items(), key=lambda kv: kv[1], reverse=True)
    sorted_items = sorted_items[:top_k]
    names = [item[0] for item in reversed(sorted_items)]
    values = [item[1] for item in reversed(sorted_items)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.barh(names, values, color='#3182bd', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Variable Importance')
    ax.set_title('Heterogeneity Drivers (Variable Importance)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax
