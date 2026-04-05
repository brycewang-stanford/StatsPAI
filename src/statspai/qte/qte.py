"""
Quantile Treatment Effects (QTE) estimation.

Methods
-------
- **Quantile DID** (Athey & Imbens 2006):
    QTE_DID(τ) = F_{11}^{-1}(τ) - F_{10}^{-1}(τ)
                - [F_{01}^{-1}(τ) - F_{00}^{-1}(τ)]

- **QTE via Quantile Regression** (Firpo 2007):
    For each τ, run quantile regression of Y on D + X; coefficient on D = QTE(τ).

- **QTE via Distribution** (propensity-score reweighting):
    Estimate counterfactual distribution with IPW, compute quantile differences.

References
----------
Athey, S. & Imbens, G. W. (2006). Identification and Inference in Nonlinear
    Difference-in-Differences Models. *Econometrica*, 74(2), 431-497.
Firpo, S. (2007). Efficient Semiparametric Estimation of Quantile Treatment
    Effects. *Econometrica*, 75(1), 259-276.
"""

from __future__ import annotations

from typing import Optional, List, Union

import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════
#  QTEResult
# ══════════════════════════════════════════════════════════════════════

class QTEResult:
    """Container for quantile treatment effect estimates.

    Attributes
    ----------
    quantiles : np.ndarray
        Quantile grid.
    effects : np.ndarray
        QTE point estimates.
    se : np.ndarray
        Bootstrap / analytical standard errors.
    ci_lower, ci_upper : np.ndarray
        Confidence interval bounds.
    ate : float
        Average treatment effect (for comparison).
    method : str
        Estimation method label.
    n_obs : int
        Sample size.
    alpha : float
        Significance level.
    """

    def __init__(
        self,
        quantiles: np.ndarray,
        effects: np.ndarray,
        se: np.ndarray,
        ci_lower: np.ndarray,
        ci_upper: np.ndarray,
        ate: float,
        method: str,
        n_obs: int,
        alpha: float = 0.05,
        model_info: Optional[dict] = None,
    ):
        self.quantiles = np.asarray(quantiles)
        self.effects = np.asarray(effects)
        self.se = np.asarray(se)
        self.ci_lower = np.asarray(ci_lower)
        self.ci_upper = np.asarray(ci_upper)
        self.ate = float(ate)
        self.method = method
        self.n_obs = int(n_obs)
        self.alpha = float(alpha)
        self.model_info = model_info or {}

    # ── pretty printing ──────────────────────────────────────────── #

    @staticmethod
    def _stars(pv: float) -> str:
        if np.isnan(pv):
            return ""
        if pv < 0.01:
            return "***"
        if pv < 0.05:
            return "**"
        if pv < 0.1:
            return "*"
        return ""

    def summary(self) -> str:
        lines = []
        lines.append("━" * 64)
        lines.append(f"  {self.method}")
        lines.append("━" * 64)
        pct = int(100 * (1 - self.alpha))
        lines.append(
            f"  {'τ':>6s}  {'QTE':>10s}  {'SE':>9s}  "
            f"{'[' + str(pct) + '% CI]':>22s}"
        )
        lines.append("  " + "-" * 58)

        for i, tau in enumerate(self.quantiles):
            eff = self.effects[i]
            se_i = self.se[i]
            lo = self.ci_lower[i]
            hi = self.ci_upper[i]
            pv = 2 * (1 - stats.norm.cdf(abs(eff / se_i))) if se_i > 0 else np.nan
            s = self._stars(pv)
            lines.append(
                f"  {tau:6.2f}  {eff:>10.4f}{s:<3s}  ({se_i:.4f})  "
                f"[{lo:.4f}, {hi:.4f}]"
            )

        lines.append("  " + "-" * 58)
        lines.append(f"  ATE (mean):  {self.ate:.4f}")
        lines.append(f"  Observations: {self.n_obs:,}")
        lines.append("━" * 64)
        lines.append("  * p<0.1, ** p<0.05, *** p<0.01")
        out = "\n".join(lines)
        print(out)
        return out

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML rendering."""
        pct = int(100 * (1 - self.alpha))
        rows = ""
        for i, tau in enumerate(self.quantiles):
            eff = self.effects[i]
            se_i = self.se[i]
            lo = self.ci_lower[i]
            hi = self.ci_upper[i]
            pv = 2 * (1 - stats.norm.cdf(abs(eff / se_i))) if se_i > 0 else np.nan
            s = self._stars(pv)
            rows += (
                f"<tr><td>{tau:.2f}</td><td>{eff:.4f}{s}</td>"
                f"<td>({se_i:.4f})</td><td>[{lo:.4f}, {hi:.4f}]</td></tr>\n"
            )
        return (
            f"<h4>{self.method}</h4>"
            f"<table><thead><tr><th>&tau;</th><th>QTE</th><th>SE</th>"
            f"<th>{pct}% CI</th></tr></thead><tbody>{rows}</tbody></table>"
            f"<p>ATE = {self.ate:.4f} &nbsp;|&nbsp; N = {self.n_obs:,}</p>"
        )

    def __repr__(self) -> str:
        return (
            f"QTEResult(method='{self.method}', "
            f"quantiles={list(self.quantiles)}, ate={self.ate:.4f})"
        )

    # ── plot ──────────────────────────────────────────────────────── #

    def plot(self, ax=None):
        """QTE plot with CI bands and ATE reference line.

        Returns (fig, ax).
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        ax.plot(
            self.quantiles, self.effects, "o-",
            color="#2c7bb6", linewidth=2, markersize=5, label="QTE",
        )
        ax.fill_between(
            self.quantiles, self.ci_lower, self.ci_upper,
            alpha=0.2, color="#2c7bb6",
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(
            self.ate, color="#d7191c", linestyle=":",
            linewidth=1.2, label=f"ATE = {self.ate:.4f}",
        )
        ax.set_xlabel("Quantile (τ)")
        ax.set_ylabel("Treatment Effect")
        ax.set_title(self.method)
        ax.legend()
        fig.tight_layout()
        return fig, ax


# ══════════════════════════════════════════════════════════════════════
#  Empirical quantile helpers
# ══════════════════════════════════════════════════════════════════════

def _quantile_func(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    xs = np.sort(x)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    return np.interp(probs, cdf, xs)


# ══════════════════════════════════════════════════════════════════════
#  Quantile DID
# ══════════════════════════════════════════════════════════════════════

def qdid(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    quantiles: Optional[List[float]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> QTEResult:
    """Quantile Difference-in-Differences (Athey & Imbens 2006).

    QTE_DID(τ) = F_{11}^{-1}(τ) - F_{10}^{-1}(τ)
                - [F_{01}^{-1}(τ) - F_{00}^{-1}(τ)]

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome variable.
    group : str
        Binary group indicator (0 = control, 1 = treated).
    time : str
        Binary time indicator (0 = pre, 1 = post).
    quantiles : list of float, optional
        Defaults to ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
    n_boot : int
        Bootstrap replications.
    alpha : float
        Significance level.
    seed : int
        Random seed.

    Returns
    -------
    QTEResult
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    taus = np.asarray(quantiles)

    df = data[[y, group, time]].dropna()
    gv = df[group].astype(int).values
    tv = df[time].astype(int).values
    yv = df[y].values.astype(float)

    y00 = yv[(gv == 0) & (tv == 0)]
    y01 = yv[(gv == 0) & (tv == 1)]
    y10 = yv[(gv == 1) & (tv == 0)]
    y11 = yv[(gv == 1) & (tv == 1)]

    for label, arr in [("control-pre", y00), ("control-post", y01),
                       ("treated-pre", y10), ("treated-post", y11)]:
        if len(arr) < 2:
            raise ValueError(f"Too few observations in {label} cell ({len(arr)}).")

    def _point(y00_, y01_, y10_, y11_, taus_):
        q00 = _quantile_func(y00_, taus_)
        q01 = _quantile_func(y01_, taus_)
        q10 = _quantile_func(y10_, taus_)
        q11 = _quantile_func(y11_, taus_)
        return q11 - q10 - (q01 - q00)

    qte_point = _point(y00, y01, y10, y11, taus)
    ate = float(np.mean(y11) - np.mean(y10) - (np.mean(y01) - np.mean(y00)))

    # Bootstrap
    rng = np.random.RandomState(seed)
    idx00 = np.where((gv == 0) & (tv == 0))[0]
    idx01 = np.where((gv == 0) & (tv == 1))[0]
    idx10 = np.where((gv == 1) & (tv == 0))[0]
    idx11 = np.where((gv == 1) & (tv == 1))[0]

    boot = np.empty((n_boot, len(taus)))
    for b in range(n_boot):
        b00 = yv[rng.choice(idx00, len(idx00), replace=True)]
        b01 = yv[rng.choice(idx01, len(idx01), replace=True)]
        b10 = yv[rng.choice(idx10, len(idx10), replace=True)]
        b11 = yv[rng.choice(idx11, len(idx11), replace=True)]
        boot[b] = _point(b00, b01, b10, b11, taus)

    se = np.std(boot, axis=0, ddof=1)
    ci_lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    ci_hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)

    return QTEResult(
        quantiles=taus,
        effects=qte_point,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ate=ate,
        method="Quantile DID (Athey & Imbens, 2006)",
        n_obs=len(df),
        alpha=alpha,
        model_info={"n_boot": n_boot},
    )


# ══════════════════════════════════════════════════════════════════════
#  QTE via Quantile Regression (Firpo 2007)
# ══════════════════════════════════════════════════════════════════════

def _qreg_coef(y: np.ndarray, X: np.ndarray, tau: float,
               max_iter: int = 500, tol: float = 1e-6) -> np.ndarray:
    """Interior-point quantile regression via iteratively reweighted LS.

    Minimises  sum rho_tau(y - X beta)  where rho_tau(u) = u*(tau - I(u<0)).
    Uses the IRLS algorithm of Koenker & d'Orey (1987).
    """
    n, k = X.shape
    # OLS start
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(max_iter):
        resid = y - X @ beta
        # Weights: avoid division by zero
        w = np.where(resid > 0, tau, 1 - tau) / np.maximum(np.abs(resid), 1e-8)
        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def qte(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    quantiles: Optional[List[float]] = None,
    method: str = "quantile_regression",
    controls: Optional[List[str]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> QTEResult:
    """Quantile Treatment Effect estimation.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome variable.
    treatment : str
        Binary treatment indicator.
    quantiles : list of float, optional
        Defaults to ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
    method : str
        ``'quantile_regression'`` (Firpo 2007) or ``'distribution'``
        (propensity-score reweighting).
    controls : list of str, optional
        Covariates.
    n_boot : int
        Bootstrap replications.
    alpha : float
        Significance level.
    seed : int
        Random seed.

    Returns
    -------
    QTEResult
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    taus = np.asarray(quantiles)

    cols = [y, treatment] + (controls or [])
    df = data[cols].dropna()
    yv = df[y].values.astype(float)
    dv = df[treatment].astype(int).values

    if method == "quantile_regression":
        return _qte_qreg(df, yv, dv, taus, y, treatment, controls, n_boot, alpha, seed)
    elif method == "distribution":
        return _qte_distribution(df, yv, dv, taus, y, treatment, controls, n_boot, alpha, seed)
    else:
        raise ValueError(f"Unknown QTE method '{method}'. Use 'quantile_regression' or 'distribution'.")


def _qte_qreg(
    df, yv, dv, taus, y_col, treat_col, controls, n_boot, alpha, seed,
) -> QTEResult:
    """QTE via quantile regression."""
    # Build design matrix: [intercept, treatment, controls...]
    X_cols = [treat_col] + (controls or [])
    X = np.column_stack([np.ones(len(yv)), df[X_cols].values.astype(float)])
    treat_idx = 1  # treatment is the second column

    # Point estimates
    qte_point = np.empty(len(taus))
    for i, tau in enumerate(taus):
        beta = _qreg_coef(yv, X, tau)
        qte_point[i] = beta[treat_idx]

    ate = float(np.mean(yv[dv == 1]) - np.mean(yv[dv == 0]))

    # Bootstrap
    rng = np.random.RandomState(seed)
    boot = np.empty((n_boot, len(taus)))
    n = len(yv)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb = yv[idx]
        Xb = X[idx]
        for i, tau in enumerate(taus):
            beta = _qreg_coef(yb, Xb, tau)
            boot[b, i] = beta[treat_idx]

    se = np.std(boot, axis=0, ddof=1)
    ci_lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    ci_hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)

    return QTEResult(
        quantiles=taus,
        effects=qte_point,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ate=ate,
        method="QTE via Quantile Regression (Firpo, 2007)",
        n_obs=len(df),
        alpha=alpha,
        model_info={"n_boot": n_boot, "controls": controls},
    )


def _qte_distribution(
    df, yv, dv, taus, y_col, treat_col, controls, n_boot, alpha, seed,
) -> QTEResult:
    """QTE via propensity-score reweighting (distribution method)."""
    # Estimate propensity score with logistic regression
    if controls:
        Xc = df[controls].values.astype(float)
        Xp = np.column_stack([np.ones(len(yv)), Xc])
    else:
        Xp = np.ones((len(yv), 1))

    # Simple logistic via scipy
    from scipy.optimize import minimize

    def _loglik(beta):
        z = Xp @ beta
        z = np.clip(z, -30, 30)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.mean(dv * np.log(p) + (1 - dv) * np.log(1 - p))

    beta0 = np.zeros(Xp.shape[1])
    res = minimize(_loglik, beta0, method="BFGS")
    pscore = 1 / (1 + np.exp(-np.clip(Xp @ res.x, -30, 30)))
    pscore = np.clip(pscore, 0.01, 0.99)

    def _weighted_quantiles(y_, d_, ps_, taus_):
        """IPW-based quantile estimates for treated and counterfactual."""
        # Treated quantiles (unweighted among treated)
        y1 = y_[d_ == 1]
        q1 = _quantile_func(y1, taus_)

        # Counterfactual quantiles via IPW reweighting of controls
        y0 = y_[d_ == 0]
        w0 = ps_[d_ == 0] / (1 - ps_[d_ == 0])
        # Weighted quantile function
        order = np.argsort(y0)
        y0s = y0[order]
        w0s = w0[order]
        wcdf = np.cumsum(w0s) / np.sum(w0s)
        q0 = np.interp(taus_, wcdf, y0s)
        return q1 - q0

    qte_point = _weighted_quantiles(yv, dv, pscore, taus)
    ate = float(np.mean(yv[dv == 1]) - np.sum(yv[dv == 0] * pscore[dv == 0] / (1 - pscore[dv == 0])) / np.sum(pscore[dv == 0] / (1 - pscore[dv == 0])))

    # Bootstrap
    rng = np.random.RandomState(seed)
    boot = np.empty((n_boot, len(taus)))
    n = len(yv)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb = yv[idx]
        db = dv[idx]
        Xpb = Xp[idx]
        # Re-estimate propensity score
        def _ll(beta):
            z = Xpb @ beta
            z = np.clip(z, -30, 30)
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.mean(db * np.log(p) + (1 - db) * np.log(1 - p))

        rb = minimize(_ll, res.x, method="BFGS")
        psb = 1 / (1 + np.exp(-np.clip(Xpb @ rb.x, -30, 30)))
        psb = np.clip(psb, 0.01, 0.99)

        if np.sum(db == 1) < 2 or np.sum(db == 0) < 2:
            boot[b] = np.nan
            continue
        boot[b] = _weighted_quantiles(yb, db, psb, taus)

    se = np.nanstd(boot, axis=0, ddof=1)
    ci_lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    ci_hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)

    return QTEResult(
        quantiles=taus,
        effects=qte_point,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ate=ate,
        method="QTE via Distribution (IPW Reweighting)",
        n_obs=len(df),
        alpha=alpha,
        model_info={"n_boot": n_boot, "controls": controls},
    )


__all__ = ["qdid", "qte", "QTEResult"]
