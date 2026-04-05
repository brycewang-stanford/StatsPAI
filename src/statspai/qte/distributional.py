"""
Distributional Treatment Effects (DTE).

Estimates the full counterfactual distribution F_{Y(0)|D=1} and computes
distributional treatment effects across the entire outcome support.

References
----------
Chernozhukov, V., Fernandez-Val, I. & Melly, B. (2013).
    Inference on Counterfactual Distributions. *Econometrica*, 81(6), 2205-2268.
Athey, S. & Imbens, G. W. (2006).
    Identification and Inference in Nonlinear DID Models. *Econometrica*, 74(2).
"""
from __future__ import annotations
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats


# ══════════════════════════════════════════════════════════════════════
#  DTEResult
# ══════════════════════════════════════════════════════════════════════

class DTEResult:
    """Container for distributional treatment effect estimates."""

    def __init__(self, grid, dte, dte_se, qte_taus, qte_effects, qte_se,
                 cdf_treated, cdf_counterfactual, ks_stat, ks_pvalue,
                 n_obs, method="ipw", alpha=0.05):
        self.grid = np.asarray(grid)
        self.dte = np.asarray(dte)
        self.dte_se = np.asarray(dte_se)
        self.qte_taus = np.asarray(qte_taus)
        self.qte_effects = np.asarray(qte_effects)
        self.qte_se = np.asarray(qte_se)
        self.cdf_treated = np.asarray(cdf_treated)
        self.cdf_counterfactual = np.asarray(cdf_counterfactual)
        self.ks_stat = float(ks_stat)
        self.ks_pvalue = float(ks_pvalue)
        self.n_obs = int(n_obs)
        self.method = method
        self.alpha = float(alpha)

    @staticmethod
    def _stars(pv: float) -> str:
        if np.isnan(pv): return ""
        if pv < 0.01: return "***"
        if pv < 0.05: return "**"
        if pv < 0.1:  return "*"
        return ""

    def summary(self) -> str:
        """Print and return a formatted summary."""
        z = stats.norm.ppf(1 - self.alpha / 2)
        pct = int(100 * (1 - self.alpha))
        lines = [
            "=" * 64,
            f"  Distributional Treatment Effects ({self.method.upper()})",
            "=" * 64,
            f"  KS statistic:  {self.ks_stat:.4f}{self._stars(self.ks_pvalue)}",
            f"  KS p-value:    {self.ks_pvalue:.4f}",
            "",
            f"  {'tau':>6s}  {'QTE':>10s}  {'SE':>9s}  "
            f"{'[' + str(pct) + '% CI]':>22s}",
            "  " + "-" * 58,
        ]
        for i, tau in enumerate(self.qte_taus):
            eff, se_i = self.qte_effects[i], self.qte_se[i]
            lo, hi = eff - z * se_i, eff + z * se_i
            pv = 2 * (1 - stats.norm.cdf(abs(eff / se_i))) if se_i > 0 else np.nan
            lines.append(
                f"  {tau:6.2f}  {eff:>10.4f}{self._stars(pv):<3s}  ({se_i:.4f})  "
                f"[{lo:.4f}, {hi:.4f}]"
            )
        lines += ["  " + "-" * 58, f"  Observations: {self.n_obs:,}",
                   "=" * 64, "  * p<0.1, ** p<0.05, *** p<0.01"]
        out = "\n".join(lines)
        print(out)
        return out

    def __repr__(self) -> str:
        return (f"DTEResult(method='{self.method}', ks_stat={self.ks_stat:.4f}, "
                f"ks_pvalue={self.ks_pvalue:.4f}, n_obs={self.n_obs})")

    # ── plots ────────────────────────────────────────────────────── #

    def plot(self, ax=None):
        """Plot the DTE curve with CI band. Returns (fig, ax)."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()
        z = stats.norm.ppf(1 - self.alpha / 2)
        lo, hi = self.dte - z * self.dte_se, self.dte + z * self.dte_se
        ax.plot(self.grid, self.dte, color="#2c7bb6", linewidth=2, label="DTE")
        ax.fill_between(self.grid, lo, hi, alpha=0.2, color="#2c7bb6",
                         label=f"{int(100 * (1 - self.alpha))}% CI")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("y")
        ax.set_ylabel(r"$F_{Y(1)|D=1}(y) - F_{Y(0)|D=1}(y)$")
        ax.set_title("Distributional Treatment Effect")
        ax.legend()
        return fig, ax

    def plot_cdf(self, ax=None):
        """Plot treated vs. counterfactual CDFs. Returns (fig, ax)."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()
        ax.step(self.grid, self.cdf_treated, color="#d7191c", linewidth=2,
                where="post", label="Treated")
        ax.step(self.grid, self.cdf_counterfactual, color="#2c7bb6", linewidth=2,
                where="post", label="Counterfactual")
        ax.set_xlabel("y"); ax.set_ylabel("CDF")
        ax.set_title("Treated vs. Counterfactual Distribution")
        ax.legend()
        return fig, ax


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def _propensity_score(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Logistic propensity score (near-unpenalised)."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
    clf.fit(X, D)
    return clf.predict_proba(X)[:, 1]


def _weighted_ecdf(vals, w, grid):
    """Weighted empirical CDF on *grid*."""
    ws = w.sum()
    if ws == 0:
        return np.zeros(len(grid))
    return np.array([np.sum(w * (vals <= g)) / ws for g in grid])


def _quantile_from_cdf(grid, cdf, taus):
    """Invert CDF on grid to get quantiles."""
    out = np.empty(len(taus))
    for i, tau in enumerate(taus):
        idx = min(np.searchsorted(cdf, tau, side="left"), len(grid) - 1)
        out[i] = grid[idx]
    return out


def _fit_cond_cdf_ctrl(X_ctrl, Y_ctrl, X_all, grid):
    """Fit P(Y<=y|X,D=0) on controls, predict for all obs. Returns (n, n_grid)."""
    from sklearn.linear_model import LinearRegression
    n, ng = X_all.shape[0], len(grid)
    out = np.empty((n, ng))
    for j, yv in enumerate(grid):
        reg = LinearRegression().fit(X_ctrl, (Y_ctrl <= yv).astype(float))
        out[:, j] = np.clip(reg.predict(X_all), 0, 1)
    return out


# ══════════════════════════════════════════════════════════════════════
#  Core estimators
# ══════════════════════════════════════════════════════════════════════

def _dte_ipw(Y, D, X, grid, taus):
    """IPW estimator for DTE."""
    treated, control = (D == 1), (D == 0)
    ps = _propensity_score(X, D) if X is not None else np.full(len(D), D.mean())
    ps = np.clip(ps, 0.01, 0.99)
    w_ctrl = ps[control] / (1 - ps[control])

    cdf_t = _weighted_ecdf(Y[treated], np.ones(treated.sum()), grid)
    cdf_cf = _weighted_ecdf(Y[control], w_ctrl, grid)
    dte = cdf_t - cdf_cf

    qt = _quantile_from_cdf(grid, cdf_t, taus)
    qcf = _quantile_from_cdf(grid, cdf_cf, taus)
    return dict(cdf_treated=cdf_t, cdf_cf=cdf_cf, dte=dte,
                qte=qt - qcf, ks_stat=float(np.max(np.abs(dte))))


def _dte_dr(Y, D, X, grid, taus):
    """Doubly-robust estimator for DTE."""
    treated, control = (D == 1), (D == 0)
    n1 = treated.sum()
    ps = np.clip(_propensity_score(X, D), 0.01, 0.99)

    # Outcome model: P(Y<=y|X) fitted on controls, predicted for all
    mu = _fit_cond_cdf_ctrl(X[control], Y[control], X, grid)

    cdf_t = _weighted_ecdf(Y[treated], np.ones(n1), grid)

    # DR counterfactual CDF
    cdf_cf = np.zeros(len(grid))
    w_ratio = ps / (1 - ps)
    for j in range(len(grid)):
        ind_y = (Y <= grid[j]).astype(float)
        ipw_term = np.sum((1 - D) * w_ratio * ind_y)
        aug_term = np.sum(D * mu[:, j] - (1 - D) * w_ratio * mu[:, j])
        cdf_cf[j] = (ipw_term + aug_term) / n1
    cdf_cf = np.maximum.accumulate(np.clip(cdf_cf, 0, 1))

    dte = cdf_t - cdf_cf
    qt = _quantile_from_cdf(grid, cdf_t, taus)
    qcf = _quantile_from_cdf(grid, cdf_cf, taus)
    return dict(cdf_treated=cdf_t, cdf_cf=cdf_cf, dte=dte,
                qte=qt - qcf, ks_stat=float(np.max(np.abs(dte))))


def _dte_cic(Y, D, grid, taus):
    """Changes-in-Changes distributional estimator.

    D encoding: 0=control-pre, 1=control-post, 2=treated-pre, 3=treated-post.
    Counterfactual: F_{Y(0)|11}(y) = F_01( Q_00( F_10(y) ) )
    """
    from scipy.interpolate import interp1d
    groups = {g: Y[D == g] for g in range(4)}
    if any(len(v) == 0 for v in groups.values()):
        raise ValueError(
            "CiC requires 4 groups: 0=ctrl-pre, 1=ctrl-post, 2=treat-pre, 3=treat-post.")

    def _ecdf_f(v):
        sv, c = np.sort(v), np.arange(1, len(v) + 1) / len(v)
        return interp1d(sv, c, bounds_error=False, fill_value=(0., 1.))

    def _qf(v):
        sv, c = np.sort(v), np.arange(1, len(v) + 1) / len(v)
        return interp1d(c, sv, bounds_error=False, fill_value=(sv[0], sv[-1]))

    F_00, F_10, F_01 = _ecdf_f(groups[0]), _ecdf_f(groups[2]), _ecdf_f(groups[1])
    Q_00 = _qf(groups[0])

    cdf_t = _weighted_ecdf(groups[3], np.ones(len(groups[3])), grid)
    cdf_cf = np.array([
        float(F_01(Q_00(np.clip(float(F_10(g)), 0.001, 0.999)))) for g in grid
    ])
    cdf_cf = np.maximum.accumulate(np.clip(cdf_cf, 0, 1))

    dte = cdf_t - cdf_cf
    qt = _quantile_from_cdf(grid, cdf_t, taus)
    qcf = _quantile_from_cdf(grid, cdf_cf, taus)
    return dict(cdf_treated=cdf_t, cdf_cf=cdf_cf, dte=dte,
                qte=qt - qcf, ks_stat=float(np.max(np.abs(dte))))


# ══════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════

def distributional_te(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    x: Optional[List[str]] = None,
    method: str = "ipw",
    n_grid: int = 100,
    quantiles: Optional[List[float]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> DTEResult:
    """Estimate distributional treatment effects.

    Parameters
    ----------
    data : DataFrame
    y : str — outcome column.
    treatment : str — treatment column (binary 0/1 for IPW/DR;
        0-3 group encoding for CiC).
    x : list[str], optional — covariates (required for DR).
    method : {'ipw', 'dr', 'cic'}
    n_grid : int — grid points for CDF evaluation.
    quantiles : list[float] — QTE quantile indices.
    n_boot : int — bootstrap replications.
    alpha : float — significance level.
    seed : int, optional — random seed.

    Returns
    -------
    DTEResult
    """
    method = method.lower()
    if method not in ("ipw", "dr", "cic"):
        raise ValueError(f"method must be 'ipw', 'dr', or 'cic', got '{method}'")
    if method == "dr" and x is None:
        raise ValueError("Covariates (x) required for DR method.")

    rng = np.random.default_rng(seed)
    Y_vec = data[y].values.astype(float)
    D_vec = data[treatment].values.astype(int)
    X_mat = data[x].values.astype(float) if x is not None else None
    n = len(Y_vec)

    taus = np.asarray(quantiles if quantiles else [0.1, 0.25, 0.5, 0.75, 0.9])

    # Evaluation grid
    yr = Y_vec if method == "cic" else (Y_vec[D_vec == 1] if np.any(D_vec == 1) else Y_vec)
    margin = 0.01 * np.ptp(yr)
    grid = np.linspace(np.min(yr) - margin, np.max(yr) + margin, n_grid)

    # Dispatch
    _est = {"ipw": _dte_ipw, "dr": _dte_dr, "cic": _dte_cic}
    args = (Y_vec, D_vec, grid, taus) if method == "cic" else (Y_vec, D_vec, X_mat, grid, taus)
    res0 = _est[method](*args)

    # Bootstrap
    boot_dte = np.empty((n_boot, n_grid))
    boot_qte = np.empty((n_boot, len(taus)))
    boot_ks = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        ba = (Y_vec[idx], D_vec[idx], grid, taus) if method == "cic" else \
             (Y_vec[idx], D_vec[idx], X_mat[idx] if X_mat is not None else None, grid, taus)
        try:
            rb = _est[method](*ba)
            boot_dte[b], boot_qte[b], boot_ks[b] = rb["dte"], rb["qte"], rb["ks_stat"]
        except Exception:
            boot_dte[b] = boot_qte[b] = boot_ks[b] = np.nan

    return DTEResult(
        grid=grid, dte=res0["dte"],
        dte_se=np.nanstd(boot_dte, axis=0),
        qte_taus=taus, qte_effects=res0["qte"],
        qte_se=np.nanstd(boot_qte, axis=0),
        cdf_treated=res0["cdf_treated"],
        cdf_counterfactual=res0["cdf_cf"],
        ks_stat=res0["ks_stat"],
        ks_pvalue=float(np.nanmean(boot_ks >= res0["ks_stat"])),
        n_obs=n, method=method, alpha=alpha,
    )
