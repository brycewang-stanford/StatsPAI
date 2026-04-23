"""
Post-Lasso IV with rigorous (data-driven) penalty — Belloni, Chen,
Chernozhukov, Hansen (2012, ECMA).

The received workflow in Python (``sklearn`` LASSO + BIC/CV on the first
stage) lacks the theoretical guarantees developed in BCH 2012:

- Tuning parameter ``λ = 2 c σ √{2 n log(2 p / α)}`` is an *a priori*
  rate-optimal choice that delivers near-oracle sup-norm rates.
- Heteroskedasticity is handled via *per-coefficient penalty loadings*
  (BCH 2012 Algorithm 1), refined by iteration.
- **Post-Lasso** re-estimates the first stage by OLS on the
  LASSO-selected instrument subset to remove the shrinkage bias,
  yielding near-oracle 2SLS behaviour.

This module implements the canonical BCH pipeline with three public
entry points:

- :func:`bch_post_lasso_iv` — full rigorous post-Lasso 2SLS
- :func:`bch_lambda`         — the rigorous penalty level
- :func:`bch_selected`       — returns the selected instrument indices

References
----------
Belloni, A., Chen, D., Chernozhukov, V. and Hansen, C. (2012).
    "Sparse Models and Methods for Optimal Instruments With an
    Application to Eminent Domain." *Econometrica*, 80(6), 2369-2429. [@belloni2011sparse]

Belloni, A., Chernozhukov, V. and Hansen, C. (2014). "Inference on
    Treatment Effects After Selection Among High-Dimensional Controls."
    *Review of Economic Studies*, 81(2), 608-650. [@belloni2014inference]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PostLassoResult:
    """Return of :func:`bch_post_lasso_iv`."""
    beta: pd.Series
    std_errors: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    conf_int: pd.DataFrame
    selected: List[str]
    n_candidate: int
    n_selected: int
    first_stage_f: float
    lambda_used: float
    penalty_loadings: np.ndarray
    iter_taken: int
    n_obs: int
    residuals: np.ndarray
    extra: dict

    def summary(self) -> str:
        lines = [
            "Post-Lasso IV (Belloni-Chen-Chernozhukov-Hansen 2012)",
            "-" * 64,
            f"  Observations         : {self.n_obs}",
            f"  Candidate instruments: {self.n_candidate}",
            f"  Selected instruments : {self.n_selected}"
            f"   [{', '.join(self.selected[:5])}"
            f"{'...' if self.n_selected > 5 else ''}]",
            f"  λ (rigorous)         : {self.lambda_used:.4f}",
            f"  iterations           : {self.iter_taken}",
            f"  post-Lasso F         : {self.first_stage_f:.2f}",
            "",
            "                 coef      std.err      t        P>|t|    95% CI",
        ]
        for name in self.beta.index:
            lo, hi = self.conf_int.loc[name]
            lines.append(
                f"  {name:<14}{self.beta[name]:>10.4f}"
                f"   {self.std_errors[name]:>8.4f}"
                f"  {self.t_stats[name]:>7.3f}"
                f"  {self.p_values[name]:>8.4f}"
                f"  [{lo:.3f}, {hi:.3f}]"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _residualize(M: np.ndarray, W: Optional[np.ndarray]) -> np.ndarray:
    if W is None or W.size == 0 or W.shape[1] == 0:
        return M
    b, *_ = np.linalg.lstsq(W, M, rcond=None)
    return M - W @ b


def _as_matrix(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _grab(v, data, cols=False):
    if isinstance(v, str):
        return data[v].values.astype(float)
    if cols and isinstance(v, list) and all(isinstance(x, str) for x in v):
        return data[v].values.astype(float)
    return np.asarray(v, dtype=float)


def _names(v, prefix, n):
    if isinstance(v, pd.DataFrame):
        return list(v.columns)
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return list(v)
    return [f"{prefix}{i}" for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════
#  Rigorous λ
# ═══════════════════════════════════════════════════════════════════════

def bch_lambda(
    n: int,
    p: int,
    alpha: float = 0.05,
    c: float = 1.1,
) -> float:
    """
    BCH (2012) rigorous penalty level:  λ = 2 · c · √{2 n · log(2 p / α)}.

    Parameters
    ----------
    n : int
    p : int
        Number of candidate instruments.
    alpha : float, default 0.05
        Target confidence level (BCH recommend 0.05 / log(n)).
    c : float, default 1.1
        Slack constant (BCH 2012 recommend 1.1).

    Returns
    -------
    float
    """
    return 2.0 * c * np.sqrt(2.0 * n * np.log(2.0 * p / alpha))


# ═══════════════════════════════════════════════════════════════════════
#  Coordinate-descent Lasso with per-coefficient penalty loadings
# ═══════════════════════════════════════════════════════════════════════

def _lasso_with_loadings(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    psi: np.ndarray,
    tol: float = 1e-7,
    max_iter: int = 10_000,
) -> np.ndarray:
    """
    Solve    (1/n) ||y - X β||² + (λ/n) Σ_j ψ_j |β_j|
    via coordinate descent.

    ``psi`` implements BCH's per-coefficient penalty loadings.
    """
    n, p = X.shape
    beta = np.zeros(p)
    # Precompute column norms
    col_sq = (X ** 2).sum(axis=0)
    col_sq = np.where(col_sq > 0, col_sq, 1.0)
    r = y.copy()

    for _ in range(max_iter):
        max_delta = 0.0
        for j in range(p):
            if psi[j] <= 0 or col_sq[j] <= 0:
                continue
            rj = r + X[:, j] * beta[j]
            rho = X[:, j] @ rj
            thresh = lam * psi[j]
            if rho > thresh:
                new_b = (rho - thresh) / col_sq[j]
            elif rho < -thresh:
                new_b = (rho + thresh) / col_sq[j]
            else:
                new_b = 0.0
            delta = new_b - beta[j]
            if delta != 0.0:
                r = rj - X[:, j] * new_b
                beta[j] = new_b
                max_delta = max(max_delta, abs(delta) * np.sqrt(col_sq[j]))
        if max_delta < tol:
            break
    return beta


def _refine_loadings(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Heteroskedastic per-coef loadings:  ψ_j = √{(1/n) Σ_i X_{ij}² ε̂_i²}."""
    r = y - X @ beta
    n = len(y)
    psi = np.sqrt((X ** 2 * (r ** 2)[:, None]).mean(axis=0))
    psi = np.where(psi > 0, psi, 1.0)
    return psi


# ═══════════════════════════════════════════════════════════════════════
#  BCH first-stage selection
# ═══════════════════════════════════════════════════════════════════════

def bch_selected(
    endog: np.ndarray,
    instruments: np.ndarray,
    exog: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    c: float = 1.1,
    max_refit: int = 15,
) -> Tuple[List[int], np.ndarray, float]:
    """
    BCH first-stage instrument selection.

    Parameters
    ----------
    endog : (n,) array — single endogenous regressor (partialled out of controls).
    instruments : (n, p) array of candidate instruments (partialled out of controls).
    exog : unused; kept for API symmetry.
    alpha, c : penalty-rule parameters.
    max_refit : refit iterations for penalty loadings.

    Returns
    -------
    (sel_indices, psi_final, lam)
    """
    n, p = instruments.shape
    lam = bch_lambda(n, p, alpha=alpha, c=c)

    # Initial loadings: homoskedastic version
    sigma_hat = float(np.std(endog, ddof=1))
    psi = sigma_hat * np.sqrt((instruments ** 2).mean(axis=0))
    psi = np.where(psi > 0, psi, 1.0)

    beta_prev = np.zeros(p)
    iters = 0
    for iters in range(1, max_refit + 1):
        beta_new = _lasso_with_loadings(instruments, endog, lam, psi)
        diff = np.max(np.abs(beta_new - beta_prev))
        beta_prev = beta_new
        psi = _refine_loadings(instruments, endog, beta_new)
        if diff < 1e-6:
            break

    sel = np.where(np.abs(beta_prev) > 1e-10)[0].tolist()
    return sel, psi, lam


# ═══════════════════════════════════════════════════════════════════════
#  Full post-Lasso 2SLS
# ═══════════════════════════════════════════════════════════════════════

def bch_post_lasso_iv(
    y: Union[np.ndarray, pd.Series, str],
    endog: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
    c: float = 1.1,
    add_const: bool = True,
    robust: bool = True,
    ensure_min_instruments: int = 1,
) -> PostLassoResult:
    """
    Post-Lasso 2SLS with rigorous, data-driven penalty.

    Recipe (BCH 2012 §3):

    1. Partial out controls ``exog`` from ``y``, ``endog``, and every
       column of ``instruments``.
    2. Select relevant instruments by LASSO with rigorous penalty
       ``λ = 2 c √{2 n log(2 p / α)}`` and iterated heteroskedastic
       loadings (Algorithm 1).
    3. If fewer than ``ensure_min_instruments`` survive, add the
       instruments with largest univariate first-stage t-stat.
    4. Re-estimate the first stage by OLS on the selected subset —
       *post-Lasso* removes shrinkage bias.
    5. Plug the post-Lasso fitted value into 2SLS for β̂.
    6. Heteroskedasticity-robust (HC1) SEs by default.

    Parameters
    ----------
    y, endog : outcome and the single endogenous regressor.
    instruments : (many) candidate instruments — p may exceed n (the
        method shines precisely there).
    exog : controls (default: intercept only).
    data : DataFrame for string-name inputs.
    alpha : penalty confidence level.
    c : slack constant (BCH default 1.1).
    add_const : whether to include a constant in the exogenous block.
    robust : use HC1 standard errors (default True).
    ensure_min_instruments : if LASSO selects 0, force this many strong ones in.

    Returns
    -------
    PostLassoResult
    """
    Y = _grab(y, data).reshape(-1)
    D = _grab(endog, data).reshape(-1)
    Zraw = _grab(instruments, data, cols=True)
    if Zraw.ndim == 1:
        Zraw = Zraw.reshape(-1, 1)
    z_names = _names(instruments, "z", Zraw.shape[1])

    n = len(Y)
    if exog is None:
        W = np.ones((n, 1)) if add_const else np.empty((n, 0))
    else:
        Wx = _grab(exog, data, cols=True)
        if Wx.ndim == 1:
            Wx = Wx.reshape(-1, 1)
        W = np.column_stack([np.ones(n), Wx]) if add_const else Wx

    # --- 1. Partial out controls --------------------------------------
    Y_t = _residualize(Y.reshape(-1, 1), W).ravel()
    D_t = _residualize(D.reshape(-1, 1), W).ravel()
    Z_t = _residualize(Zraw, W)

    # Standardise instruments for LASSO
    z_sd = np.std(Z_t, axis=0, ddof=1)
    z_sd = np.where(z_sd > 0, z_sd, 1.0)
    Z_std = Z_t / z_sd

    # --- 2. BCH first-stage selection ---------------------------------
    sel, psi, lam = bch_selected(D_t, Z_std, alpha=alpha, c=c)
    n_sel = len(sel)
    iters = 1  # bch_selected internally converges; keep top-line counter

    # --- 3. Ensure we have at least ensure_min_instruments ------------
    if n_sel < ensure_min_instruments:
        # Add instruments with largest univariate |t|-stat
        t_stats = np.zeros(Z_t.shape[1])
        for j in range(Z_t.shape[1]):
            xj = Z_t[:, j]
            b = (xj @ D_t) / (xj @ xj) if (xj @ xj) > 0 else 0
            r = D_t - b * xj
            s = float(np.std(r, ddof=1))
            se = s / np.sqrt(xj @ xj) if (xj @ xj) > 0 else np.inf
            t_stats[j] = b / se if se > 0 else 0
        pending = [j for j in np.argsort(-np.abs(t_stats)) if j not in sel]
        sel = list(sel) + pending[: ensure_min_instruments - n_sel]
        n_sel = len(sel)

    selected_names = [z_names[j] for j in sel]

    # --- 4. Post-Lasso OLS first stage --------------------------------
    Z_sel = Z_t[:, sel]
    if Z_sel.shape[1] == 0:
        raise RuntimeError("No instruments could be selected; all weak.")
    # Post-Lasso: OLS of D on Z_sel (already partialled out)
    pi_hat, *_ = np.linalg.lstsq(Z_sel, D_t, rcond=None)
    D_hat = Z_sel @ pi_hat
    resid_fs = D_t - D_hat
    # First-stage F on the selected subset
    rss_full = float(resid_fs @ resid_fs)
    rss_red = float(D_t @ D_t)
    df_d = max(n - W.shape[1] - Z_sel.shape[1], 1)
    if rss_full > 0:
        first_f = ((rss_red - rss_full) / Z_sel.shape[1]) / (rss_full / df_d)
    else:
        first_f = np.inf

    # --- 5. Second stage: 2SLS β̂ -------------------------------------
    X_hat = D_hat
    X_act = D_t
    XhXh = float(X_hat @ X_hat)
    XhXh_inv = 1.0 / max(XhXh, 1e-12)
    beta = float((X_hat @ Y_t) * XhXh_inv)
    y_resid = Y_t - X_act * beta

    # --- 6. HC1 standard errors ---------------------------------------
    if robust:
        meat = float((X_hat ** 2 * y_resid ** 2).sum())
        scale = n / max(n - 1, 1)
        var_beta = scale * XhXh_inv * meat * XhXh_inv
    else:
        sigma2 = float(y_resid @ y_resid) / max(n - 1, 1)
        var_beta = sigma2 * XhXh_inv
    se_beta = float(np.sqrt(max(var_beta, 0)))

    # Recover intercept/controls via Frisch-Waugh
    intercept_and_controls = {}
    if W.shape[1] > 0:
        resid_Y_on_W = Y - (W @ np.linalg.lstsq(W, Y, rcond=None)[0])
        # β̂ is for partialled-out D; original-scale β is identical
        # Controls: regress (Y - β D) on W
        b_w, *_ = np.linalg.lstsq(W, Y - beta * D, rcond=None)
        if add_const:
            ctl_names = ["Intercept"]
            if exog is not None:
                ctl_names += _names(exog, "w", W.shape[1] - 1)
        else:
            ctl_names = _names(exog, "w", W.shape[1])
        for nm, val in zip(ctl_names, b_w):
            intercept_and_controls[nm] = float(val)

    endog_name = (endog if isinstance(endog, str)
                  else getattr(endog, "name", None) or "endog")

    all_names = [endog_name] + list(intercept_and_controls.keys())
    all_vals = [beta] + list(intercept_and_controls.values())
    ses = [se_beta] + [np.nan] * len(intercept_and_controls)
    params = pd.Series(all_vals, index=all_names)
    ses_s = pd.Series(ses, index=all_names)
    tvals = params / ses_s.replace(0, np.nan)
    pvals = 2 * (1 - stats.norm.cdf(np.abs(tvals.fillna(0).values)))

    z_crit = 1.96
    lo = params - z_crit * ses_s
    hi = params + z_crit * ses_s
    ci = pd.DataFrame({"lower": lo, "upper": hi})

    return PostLassoResult(
        beta=params,
        std_errors=ses_s,
        t_stats=tvals,
        p_values=pd.Series(pvals, index=all_names),
        conf_int=ci,
        selected=selected_names,
        n_candidate=Zraw.shape[1],
        n_selected=n_sel,
        first_stage_f=float(first_f),
        lambda_used=float(lam),
        penalty_loadings=psi,
        iter_taken=iters,
        n_obs=n,
        residuals=y_resid,
        extra={"pi_hat_selected": pi_hat, "selected_idx": sel},
    )


__all__ = [
    "bch_post_lasso_iv",
    "bch_lambda",
    "bch_selected",
    "PostLassoResult",
]
