"""
Stepwise regression and LASSO-based variable selection.

Implements forward, backward, and bidirectional stepwise regression using
AIC, BIC, adjusted-R², or p-value criteria, plus LASSO variable selection
with coordinate descent (no sklearn dependency).

References
----------
- Hastie, Tibshirani & Friedman (2009). *Elements of Statistical Learning*.
- Tibshirani (1996). Regression shrinkage and selection via the lasso.
  *JRSS-B*, 58(1), 267-288. [@hastie2009elements]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Result class
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """Result container for variable selection procedures.

    Attributes
    ----------
    selected : list[str]
        Variable names retained in the final model.
    dropped : list[str]
        Variable names excluded from the final model.
    history : pd.DataFrame
        Step-by-step log of the selection procedure.
    final_model : dict
        Summary statistics for the final model (R², adj-R², AIC, BIC, n, k).
    method : str
        Selection method used (``"forward"``, ``"backward"``, ``"both"``,
        ``"lasso_cv"``, ``"lasso_bic"``, ``"lasso_aic"``).
    coefficients : dict | None
        Coefficient estimates from the final model (name -> value).
    lasso_path : dict | None
        For LASSO: ``{"lambdas": array, "coef_paths": dict}`` storing the
        regularisation path.
    """

    selected: List[str]
    dropped: List[str]
    history: pd.DataFrame
    final_model: dict
    method: str = ""
    coefficients: Optional[dict] = None
    lasso_path: Optional[dict] = None

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary and print it."""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"Variable Selection: {self.method}")
        lines.append("=" * 60)

        fm = self.final_model
        lines.append(f"  Observations : {fm.get('n', '?')}")
        lines.append(f"  Variables    : {fm.get('k', '?')}")
        lines.append(f"  R²           : {fm.get('r_squared', 0):.4f}")
        lines.append(f"  Adj. R²      : {fm.get('adj_r_squared', 0):.4f}")
        lines.append(f"  AIC          : {fm.get('aic', 0):.2f}")
        lines.append(f"  BIC          : {fm.get('bic', 0):.2f}")
        lines.append("")
        lines.append(f"  Selected ({len(self.selected)}): "
                      + ", ".join(self.selected) if self.selected else "  Selected: (none)")
        lines.append(f"  Dropped  ({len(self.dropped)}): "
                      + ", ".join(self.dropped) if self.dropped else "  Dropped: (none)")

        if self.coefficients:
            lines.append("")
            lines.append("  Coefficients:")
            for name, val in self.coefficients.items():
                lines.append(f"    {name:20s} {val:12.6f}")

        lines.append("=" * 60)
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, figsize=(10, 6)):
        """Plot selection diagnostics.

        For stepwise: criterion value at each step.
        For LASSO: coefficient path across lambda values.

        Returns
        -------
        (fig, ax) : matplotlib Figure and Axes
        """
        import matplotlib.pyplot as plt

        if self.lasso_path is not None:
            return self._plot_lasso_path(figsize)
        return self._plot_stepwise_history(figsize)

    def _plot_stepwise_history(self, figsize):
        import matplotlib.pyplot as plt

        hist = self.history
        if hist.empty:
            warnings.warn("No history to plot.")
            return None, None

        fig, ax = plt.subplots(figsize=figsize)
        steps = hist["step"].values
        criterion_col = [c for c in hist.columns if c.lower() in ("aic", "bic", "adjr2", "pvalue")]
        if not criterion_col:
            criterion_col = [c for c in hist.columns if c not in ("step", "action", "variable", "r_squared")]
        if not criterion_col:
            warnings.warn("Cannot determine criterion column from history.")
            return fig, ax

        col = criterion_col[0]
        ax.plot(steps, hist[col].values, marker="o", linewidth=2)
        for _, row in hist.iterrows():
            label = row.get("action", "") + " " + row.get("variable", "")
            ax.annotate(label.strip(), (row["step"], row[col]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Step")
        ax.set_ylabel(col.upper())
        ax.set_title(f"Stepwise Selection — {col.upper()} at Each Step")
        fig.tight_layout()
        return fig, ax

    def _plot_lasso_path(self, figsize):
        import matplotlib.pyplot as plt

        lambdas = self.lasso_path["lambdas"]
        paths = self.lasso_path["coef_paths"]

        fig, ax = plt.subplots(figsize=figsize)
        for name, coefs in paths.items():
            ax.plot(np.log10(lambdas), coefs, label=name)
        ax.axvline(np.log10(self.lasso_path.get("lambda_best", lambdas[-1])),
                    color="grey", linestyle="--", linewidth=1, label="selected λ")
        ax.set_xlabel("log₁₀(λ)")
        ax.set_ylabel("Coefficient")
        ax.set_title("LASSO Coefficient Path")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        return fig, ax

    # Jupyter rich display
    def _repr_html_(self) -> str:
        fm = self.final_model
        rows = "".join(
            f"<tr><td style='text-align:left;padding:2px 8px'>{k}</td>"
            f"<td style='text-align:right;padding:2px 8px'>"
            f"{v:.4f if isinstance(v, float) else v}</td></tr>"
            for k, v in fm.items()
        )
        sel = ", ".join(f"<code>{s}</code>" for s in self.selected) or "(none)"
        drp = ", ".join(f"<code>{s}</code>" for s in self.dropped) or "(none)"
        return (
            f"<div style='font-family:monospace'>"
            f"<h4>Variable Selection: {self.method}</h4>"
            f"<table>{rows}</table>"
            f"<p><b>Selected:</b> {sel}</p>"
            f"<p><b>Dropped:</b> {drp}</p>"
            f"</div>"
        )


# ---------------------------------------------------------------------------
# Internal OLS helpers (no sklearn)
# ---------------------------------------------------------------------------

def _ols_fit(X: np.ndarray, y: np.ndarray):
    """OLS via QR decomposition. Returns (beta, residuals, rank)."""
    Q, R = np.linalg.qr(X, mode="reduced")
    beta = np.linalg.solve(R, Q.T @ y)
    residuals = y - X @ beta
    return beta, residuals, np.linalg.matrix_rank(R)


def _model_stats(X: np.ndarray, y: np.ndarray):
    """Compute model statistics: RSS, R², adj-R², AIC, BIC, p-values."""
    n, k = X.shape
    if n <= k:
        return None  # degenerate
    beta, resid, rank = _ols_fit(X, y)
    rss = float(resid @ resid)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else 0.0
    # Log-likelihood (normal errors)
    ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)

    # p-values for each coefficient
    sigma2 = rss / (n - k) if n > k else np.inf
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        t_stats = beta / np.where(se > 0, se, np.inf)
        pvals = 2 * sp_stats.t.sf(np.abs(t_stats), df=n - k)
    except np.linalg.LinAlgError:
        pvals = np.ones(k)

    return {
        "beta": beta,
        "rss": rss,
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "aic": aic,
        "bic": bic,
        "pvalues": pvals,
        "n": n,
        "k": k,
    }


def _build_X(data: pd.DataFrame, cols: list[str], add_intercept: bool = True):
    """Build design matrix from column names."""
    X = data[cols].values.astype(float)
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    return X


# ---------------------------------------------------------------------------
# Stepwise regression
# ---------------------------------------------------------------------------

def stepwise(
    data: pd.DataFrame,
    y: str,
    x: list[str],
    method: Literal["forward", "backward", "both"] = "both",
    criterion: Literal["aic", "bic", "adjr2", "pvalue"] = "bic",
    alpha_in: float = 0.05,
    alpha_out: float = 0.10,
    verbose: bool = True,
) -> SelectionResult:
    """Stepwise variable selection for OLS regression.

    Parameters
    ----------
    data : DataFrame
        Dataset containing all variables.
    y : str
        Name of the dependent variable column.
    x : list[str]
        Candidate independent variable column names.
    method : {"forward", "backward", "both"}
        Selection strategy. Default ``"both"`` (bidirectional).
    criterion : {"aic", "bic", "adjr2", "pvalue"}
        Optimisation criterion. Default ``"bic"``.
    alpha_in : float
        p-value threshold for variable entry (when ``criterion="pvalue"``).
    alpha_out : float
        p-value threshold for variable removal (when ``criterion="pvalue"``).
    verbose : bool
        Print step-by-step progress.

    Returns
    -------
    SelectionResult
    """
    df = data.dropna(subset=[y] + list(x)).reset_index(drop=True)
    y_vec = df[y].values.astype(float)
    candidates = list(x)

    # Validate
    if len(candidates) == 0:
        raise ValueError("x must contain at least one variable name.")
    if y not in df.columns:
        raise ValueError(f"Dependent variable '{y}' not found in data.")
    missing = [c for c in candidates if c not in df.columns]
    if missing:
        raise ValueError(f"Variables not found in data: {missing}")

    # Choose comparator
    if criterion in ("aic", "bic"):
        # lower is better
        def is_better(new, old):
            return new < old
        worse_init = np.inf
    elif criterion == "adjr2":
        # higher is better
        def is_better(new, old):
            return new > old
        worse_init = -np.inf
    else:  # pvalue — use BIC internally for tie-breaking, but gate on p-value
        def is_better(new, old):
            return new < old
        worse_init = np.inf

    def _criterion_value(st):
        if st is None:
            return worse_init
        if criterion == "adjr2":
            return st["adj_r_squared"]
        elif criterion == "pvalue":
            return st["bic"]  # use BIC as tiebreaker
        return st[criterion]

    # Initialise
    if method == "backward":
        included = list(candidates)
    else:
        included = []

    history_rows: list[dict] = []
    step = 0

    def _eval_model(cols):
        if not cols:
            # intercept-only
            X = np.ones((len(y_vec), 1))
        else:
            X = _build_X(df, cols, add_intercept=True)
        return _model_stats(X, y_vec)

    def _log_step(step_num, action, var, st):
        crit_val = _criterion_value(st)
        r2 = st["r_squared"] if st else 0.0
        row = {
            "step": step_num,
            "action": action,
            "variable": var,
            criterion: crit_val,
            "r_squared": r2,
        }
        history_rows.append(row)
        if verbose:
            sign = "+" if action == "add" else "-"
            print(f"Step {step_num}: {sign} {var:20s} "
                  f"{criterion.upper()} = {crit_val:.1f}   "
                  f"R² = {r2:.3f}")

    current_stats = _eval_model(included)
    current_crit = _criterion_value(current_stats)

    improved = True
    while improved:
        improved = False
        best_crit = current_crit
        best_action = None
        best_var = None

        # --- Try adding ---
        if method in ("forward", "both"):
            remaining = [v for v in candidates if v not in included]
            for var in remaining:
                trial = included + [var]
                st = _eval_model(trial)
                if st is None:
                    continue

                if criterion == "pvalue":
                    # Only add if the variable's p-value < alpha_in
                    idx = len(trial)  # index in beta (0 = intercept)
                    pval = st["pvalues"][idx]
                    if pval >= alpha_in:
                        continue

                cval = _criterion_value(st)
                if is_better(cval, best_crit):
                    best_crit = cval
                    best_action = "add"
                    best_var = var

        # --- Try removing ---
        if method in ("backward", "both") and len(included) > 0:
            for var in list(included):
                trial = [v for v in included if v != var]
                st = _eval_model(trial)
                if st is None:
                    continue

                if criterion == "pvalue":
                    # Only remove if that variable's p-value in current model > alpha_out
                    cur_idx = included.index(var) + 1  # +1 for intercept
                    if current_stats is not None and cur_idx < len(current_stats["pvalues"]):
                        pval = current_stats["pvalues"][cur_idx]
                        if pval <= alpha_out:
                            continue
                    else:
                        continue

                cval = _criterion_value(st)
                if is_better(cval, best_crit):
                    best_crit = cval
                    best_action = "remove"
                    best_var = var

        if best_var is not None:
            step += 1
            if best_action == "add":
                included.append(best_var)
            else:
                included.remove(best_var)
            current_stats = _eval_model(included)
            current_crit = _criterion_value(current_stats)
            _log_step(step, best_action, best_var, current_stats)
            improved = True

    if verbose:
        print(f"Step {step + 1}: No improvement possible. Stopping.\n")
        print(f"Selected: {', '.join(included) if included else '(none)'}")
        dropped = [v for v in candidates if v not in included]
        print(f"Dropped:  {', '.join(dropped) if dropped else '(none)'}")

    # Final stats
    final_stats = _eval_model(included)
    fm = {
        "n": final_stats["n"],
        "k": final_stats["k"],
        "r_squared": final_stats["r_squared"],
        "adj_r_squared": final_stats["adj_r_squared"],
        "aic": final_stats["aic"],
        "bic": final_stats["bic"],
    }

    # Coefficients
    names = ["_cons"] + included
    coefs = dict(zip(names, final_stats["beta"]))

    dropped = [v for v in candidates if v not in included]
    _result = SelectionResult(
        selected=included,
        dropped=dropped,
        history=pd.DataFrame(history_rows),
        final_model=fm,
        method=method,
        coefficients=coefs,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.selection.stepwise",
            params={
                "y": y, "x": list(x),
                "method": method, "criterion": criterion,
                "alpha_in": alpha_in, "alpha_out": alpha_out,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ---------------------------------------------------------------------------
# LASSO via coordinate descent
# ---------------------------------------------------------------------------

def _soft_threshold(z: float, lam: float) -> float:
    """Soft-thresholding operator S(z, λ) = sign(z)(|z| - λ)₊."""
    if z > lam:
        return z - lam
    elif z < -lam:
        return z + lam
    return 0.0


def _lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    warm_start: np.ndarray | None = None,
) -> np.ndarray:
    """Fit LASSO via coordinate descent for a single lambda.

    Solves: min (1/2n)||y - Xβ||² + λ||β||₁

    Parameters
    ----------
    X : (n, p) array — design matrix (should be standardised).
    y : (n,) array — response (should be centred).
    lam : float — regularisation parameter.
    max_iter : int
    tol : float — convergence tolerance on max coefficient change.
    warm_start : optional initial beta.

    Returns
    -------
    beta : (p,) array
    """
    n, p = X.shape
    beta = warm_start.copy() if warm_start is not None else np.zeros(p)
    # Pre-compute column norms (denominator)
    col_norm_sq = np.sum(X ** 2, axis=0)  # (p,)

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # Partial residual
            r_j = y - X @ beta + X[:, j] * beta[j]
            z_j = X[:, j] @ r_j / n
            denom = col_norm_sq[j] / n
            if denom == 0:
                beta[j] = 0.0
            else:
                beta[j] = _soft_threshold(z_j, lam) / denom
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


def _lambda_max(X: np.ndarray, y: np.ndarray) -> float:
    """Smallest lambda for which all coefficients are zero."""
    n = X.shape[0]
    return np.max(np.abs(X.T @ y)) / n


def _lambda_grid(X: np.ndarray, y: np.ndarray, n_lambda: int = 100,
                 eps: float = 1e-3) -> np.ndarray:
    """Log-spaced lambda grid from lambda_max to eps * lambda_max."""
    lam_max = _lambda_max(X, y)
    return np.logspace(np.log10(lam_max), np.log10(eps * lam_max), n_lambda)


def _lasso_path(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray,
                max_iter: int = 1000, tol: float = 1e-6):
    """Compute the full LASSO regularisation path.

    Returns
    -------
    coef_matrix : (n_lambda, p) array
    """
    n_lambda = len(lambdas)
    p = X.shape[1]
    coef_matrix = np.zeros((n_lambda, p))
    beta = np.zeros(p)
    for i, lam in enumerate(lambdas):
        beta = _lasso_coordinate_descent(X, y, lam, max_iter, tol,
                                         warm_start=beta)
        coef_matrix[i] = beta
    return coef_matrix


def _kfold_indices(n: int, k: int, seed: int = 42):
    """Generate K-fold train/test index splits."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1
    current = 0
    for fold_size in fold_sizes:
        test_idx = indices[current:current + fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx
        current += fold_size


def lasso_select(
    data: pd.DataFrame,
    y: str,
    x: list[str],
    method: Literal["cv", "bic", "aic"] = "cv",
    n_folds: int = 10,
    n_lambda: int = 100,
    eps: float = 1e-3,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = True,
    seed: int = 42,
) -> SelectionResult:
    """LASSO-based variable selection.

    Solves  min (1/2n)||y - Xβ||² + λ||β||₁  via coordinate descent.
    Selects λ by K-fold cross-validation, BIC, or AIC.

    Parameters
    ----------
    data : DataFrame
    y : str
        Dependent variable column.
    x : list[str]
        Candidate independent variables.
    method : {"cv", "bic", "aic"}
        How to choose the regularisation parameter λ.
    n_folds : int
        Number of cross-validation folds (only for ``method="cv"``).
    n_lambda : int
        Number of λ values in the grid.
    eps : float
        Ratio of lambda_min / lambda_max.
    max_iter : int
        Maximum coordinate descent iterations per λ.
    tol : float
        Convergence tolerance.
    verbose : bool
        Print progress.
    seed : int
        Random seed for CV fold assignment.

    Returns
    -------
    SelectionResult
    """
    df = data.dropna(subset=[y] + list(x)).reset_index(drop=True)
    y_raw = df[y].values.astype(float)
    X_raw = df[x].values.astype(float)
    n, p = X_raw.shape
    var_names = list(x)

    if p == 0:
        raise ValueError("x must contain at least one variable name.")

    # Standardise X and centre y
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0, ddof=0)
    X_std[X_std == 0] = 1.0  # avoid division by zero for constant columns
    X_sc = (X_raw - X_mean) / X_std
    y_mean = y_raw.mean()
    y_c = y_raw - y_mean

    lambdas = _lambda_grid(X_sc, y_c, n_lambda, eps)

    # Compute full path (needed for plot regardless of method)
    coef_matrix = _lasso_path(X_sc, y_c, lambdas, max_iter, tol)

    # Select best lambda
    if method == "cv":
        # K-fold CV
        mse_cv = np.zeros(n_lambda)
        for train_idx, test_idx in _kfold_indices(n, n_folds, seed):
            X_tr, y_tr = X_sc[train_idx], y_c[train_idx]
            X_te, y_te = X_sc[test_idx], y_c[test_idx]
            fold_coefs = _lasso_path(X_tr, y_tr, lambdas, max_iter, tol)
            preds = X_te @ fold_coefs.T  # (n_test, n_lambda)
            mse_cv += np.mean((preds - y_te[:, None]) ** 2, axis=0)
        mse_cv /= n_folds
        best_idx = int(np.argmin(mse_cv))
    else:
        # Information criterion
        scores = np.zeros(n_lambda)
        for i in range(n_lambda):
            resid = y_c - X_sc @ coef_matrix[i]
            rss = float(resid @ resid)
            k_nonzero = int(np.sum(np.abs(coef_matrix[i]) > 1e-10))
            if method == "bic":
                scores[i] = n * np.log(rss / n + 1e-300) + k_nonzero * np.log(n)
            else:  # aic
                scores[i] = n * np.log(rss / n + 1e-300) + 2 * k_nonzero
        best_idx = int(np.argmin(scores))

    best_lambda = lambdas[best_idx]
    best_beta_sc = coef_matrix[best_idx]

    # Un-standardise coefficients
    beta_orig = best_beta_sc / X_std
    intercept = y_mean - float(X_mean @ beta_orig)

    # Identify selected variables
    nonzero_mask = np.abs(best_beta_sc) > 1e-10
    selected = [var_names[j] for j in range(p) if nonzero_mask[j]]
    dropped = [var_names[j] for j in range(p) if not nonzero_mask[j]]

    # Final model stats (OLS on selected variables for clean stats)
    if selected:
        X_final = _build_X(df, selected, add_intercept=True)
        final_st = _model_stats(X_final, y_raw)
    else:
        # intercept-only
        X_final = np.ones((n, 1))
        final_st = _model_stats(X_final, y_raw)

    fm = {
        "n": final_st["n"],
        "k": final_st["k"],
        "r_squared": final_st["r_squared"],
        "adj_r_squared": final_st["adj_r_squared"],
        "aic": final_st["aic"],
        "bic": final_st["bic"],
        "lambda": best_lambda,
    }

    # Coefficients dict
    coefs = {"_cons": intercept}
    for j in range(p):
        if nonzero_mask[j]:
            coefs[var_names[j]] = float(beta_orig[j])

    # Build path dict for plotting (un-standardise)
    path_dict = {}
    for j in range(p):
        path_dict[var_names[j]] = (coef_matrix[:, j] / X_std[j]).tolist()

    # History: one row per lambda showing selected count & criterion
    history_rows = []
    for i, lam in enumerate(lambdas):
        k_nonzero = int(np.sum(np.abs(coef_matrix[i]) > 1e-10))
        resid = y_c - X_sc @ coef_matrix[i]
        rss = float(resid @ resid)
        history_rows.append({
            "step": i + 1,
            "lambda": lam,
            "n_selected": k_nonzero,
            "rss": rss,
        })

    if verbose:
        method_label = f"lasso_{method}"
        print(f"LASSO ({method.upper()}): λ = {best_lambda:.6f}")
        print(f"  Selected ({len(selected)}): {', '.join(selected) if selected else '(none)'}")
        print(f"  Dropped  ({len(dropped)}): {', '.join(dropped) if dropped else '(none)'}")

    _result = SelectionResult(
        selected=selected,
        dropped=dropped,
        history=pd.DataFrame(history_rows),
        final_model=fm,
        method=f"lasso_{method}",
        coefficients=coefs,
        lasso_path={
            "lambdas": lambdas,
            "coef_paths": path_dict,
            "lambda_best": best_lambda,
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.selection.lasso_select",
            params={
                "y": y, "x": list(x),
                "method": method, "n_folds": n_folds,
                "n_lambda": n_lambda, "eps": eps,
                "max_iter": max_iter, "tol": tol, "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
