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

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin

# numpy 2.0 renamed ``trapz`` to ``trapezoid``; the project supports numpy 1.x
# on older Pythons, so bind whichever exists.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ══════════════════════════════════════════════════════════════════════
#  DTEResult
# ══════════════════════════════════════════════════════════════════════


class DTEResult(ResultProtocolMixin):
    """Container for distributional treatment effect estimates.

    Returned by :func:`distributional_te`. Carries the DTE curve over a
    grid, quantile treatment effects, treated/counterfactual CDFs, and a
    Kolmogorov-Smirnov statistic for the null of no distributional effect.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> d = rng.integers(0, 2, n)
    >>> y = 1.0 + 1.5 * d + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "d": d})
    >>> res = sp.distributional_te(
    ...     df, y="y", treatment="d", method="ipw",
    ...     quantiles=[0.25, 0.5, 0.75], n_boot=50, seed=42)
    >>> isinstance(res, sp.DTEResult)
    True
    >>> res.qte_effects.round(2).tolist()  # QTE at each quantile
    [1.63, 1.51, 1.57]
    """

    def __init__(
        self,
        grid: Any,
        dte: Any,
        dte_se: Any,
        qte_taus: Any,
        qte_effects: Any,
        qte_se: Any,
        cdf_treated: Any,
        cdf_counterfactual: Any,
        ks_stat: Any,
        ks_pvalue: Any,
        n_obs: Any,
        method: str = "ipw",
        alpha: float = 0.05,
        cvm_stat: Any = np.nan,
        cvm_pvalue: Any = np.nan,
        n_boot_failed: int = 0,
    ) -> None:
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
        # Cramer-von Mises companion to the KS test: integrated squared
        # deviation rather than the sup, so it is sensitive to broad, shallow
        # distributional shifts that a sup-statistic can miss.
        self.cvm_stat = float(cvm_stat)
        self.cvm_pvalue = float(cvm_pvalue)
        self.n_boot_failed = int(n_boot_failed)
        self.degradations: List[dict] = []

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
        """Print and return a formatted summary."""
        z = stats.norm.ppf(1 - self.alpha / 2)
        pct = int(100 * (1 - self.alpha))
        lines = [
            "=" * 64,
            f"  Distributional Treatment Effects ({self.method.upper()})",
            "=" * 64,
            f"  KS  statistic: {self.ks_stat:.4f}" f"{self._stars(self.ks_pvalue)}",
            f"  KS  p-value:   {self.ks_pvalue:.4f}",
            f"  CvM statistic: {self.cvm_stat:.4f}" f"{self._stars(self.cvm_pvalue)}",
            f"  CvM p-value:   {self.cvm_pvalue:.4f}",
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
        lines += [
            "  " + "-" * 58,
            f"  Observations: {self.n_obs:,}",
            "=" * 64,
            "  * p<0.1, ** p<0.05, *** p<0.01",
        ]
        out = "\n".join(lines)
        print(out)
        return out

    def __repr__(self) -> str:
        return (
            f"DTEResult(method='{self.method}', ks_stat={self.ks_stat:.4f}, "
            f"ks_pvalue={self.ks_pvalue:.4f}, n_obs={self.n_obs})"
        )

    # ── plots ────────────────────────────────────────────────────── #

    def plot(self, ax: Any = None) -> Any:
        """Plot the DTE curve with CI band. Returns (fig, ax)."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()
        z = stats.norm.ppf(1 - self.alpha / 2)
        lo, hi = self.dte - z * self.dte_se, self.dte + z * self.dte_se
        ax.plot(self.grid, self.dte, color="#2c7bb6", linewidth=2, label="DTE")
        ax.fill_between(
            self.grid,
            lo,
            hi,
            alpha=0.2,
            color="#2c7bb6",
            label=f"{int(100 * (1 - self.alpha))}% CI",
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("y")
        ax.set_ylabel(r"$F_{Y(1)|D=1}(y) - F_{Y(0)|D=1}(y)$")
        ax.set_title("Distributional Treatment Effect")
        ax.legend()
        return fig, ax

    def plot_cdf(self, ax: Any = None) -> Any:
        """Plot treated vs. counterfactual CDFs. Returns (fig, ax)."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()
        ax.step(
            self.grid,
            self.cdf_treated,
            color="#d7191c",
            linewidth=2,
            where="post",
            label="Treated",
        )
        ax.step(
            self.grid,
            self.cdf_counterfactual,
            color="#2c7bb6",
            linewidth=2,
            where="post",
            label="Counterfactual",
        )
        ax.set_xlabel("y")
        ax.set_ylabel("CDF")
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
    return np.asarray(clf.predict_proba(X)[:, 1])


def _weighted_ecdf(
    vals: np.ndarray,
    w: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Weighted empirical CDF on *grid*."""
    ws = w.sum()
    if ws == 0:
        return np.zeros(len(grid))
    return np.array([np.sum(w * (vals <= g)) / ws for g in grid])


def _quantile_from_cdf(
    grid: np.ndarray,
    cdf: np.ndarray,
    taus: np.ndarray,
) -> np.ndarray:
    """Invert a CDF tabulated on ``grid``, interpolating between grid points.

    The previous implementation snapped every quantile to the nearest grid
    node, so with the default ``n_grid=100`` each estimate carried a
    discretisation error of up to one grid cell -- a bias that did not shrink
    with the sample size, only with ``n_grid``.  Linear interpolation of the
    CDF removes it.
    """
    grid = np.asarray(grid, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    # np.interp needs an increasing x; the CDF is non-decreasing, and ties on
    # flat stretches resolve to the left-most grid point, which is the
    # left-continuous inverse we want.
    return np.asarray(np.interp(taus, cdf, grid, left=grid[0], right=grid[-1]))


def _fit_cond_cdf_ctrl(
    X_ctrl: np.ndarray,
    Y_ctrl: np.ndarray,
    X_all: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Distribution regression for ``P(Y <= y | X, D = 0)``. Returns (n, n_grid).

    One logit per grid point, i.e. the Chernozhukov, Fernandez-Val & Melly
    (2013) distribution-regression estimator this module's header cites.
    The previous implementation used ``LinearRegression`` -- a linear
    probability model for a CDF -- which is neither bounded in [0, 1] (it was
    clipped after the fact) nor monotone in ``y``.  Monotonicity is restored
    explicitly by rearrangement across the grid.

    Falls back to the empirical control CDF (constant in ``X``) at grid points
    where the outcome indicator is degenerate -- all zeros or all ones -- since
    a logit is unidentified there.

    References
    ----------
    chernozhukov2013inference, chernozhukov2010quantile
    """
    from sklearn.linear_model import LogisticRegression

    n, ng = X_all.shape[0], len(grid)
    out = np.empty((n, ng))
    for j, yv in enumerate(grid):
        ind = (Y_ctrl <= yv).astype(int)
        if ind.min() == ind.max():
            # Degenerate: every control is on one side of this grid point.
            out[:, j] = float(ind[0])
            continue
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
        clf.fit(X_ctrl, ind)
        out[:, j] = clf.predict_proba(X_all)[:, 1]
    # Enforce monotonicity in y for every observation (CFG 2010 rearrangement).
    out = np.sort(out, axis=1)
    return np.clip(out, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════
#  Core estimators
# ══════════════════════════════════════════════════════════════════════


def _dte_ipw(
    Y: np.ndarray,
    D: np.ndarray,
    X: Optional[np.ndarray],
    grid: np.ndarray,
    taus: np.ndarray,
) -> Dict[str, Any]:
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
    return dict(
        cdf_treated=cdf_t,
        cdf_cf=cdf_cf,
        dte=dte,
        qte=qt - qcf,
        ks_stat=float(np.max(np.abs(dte))),
    )


def _dte_dr(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    grid: np.ndarray,
    taus: np.ndarray,
) -> Dict[str, Any]:
    """Doubly-robust estimator for DTE."""
    treated, control = (D == 1), (D == 0)
    n1 = treated.sum()
    ps = np.clip(_propensity_score(X, D), 0.01, 0.99)

    # Outcome model: P(Y<=y|X) fitted on controls, predicted for all
    mu = _fit_cond_cdf_ctrl(X[control], Y[control], X, grid)

    cdf_t = _weighted_ecdf(Y[treated], np.ones(n1), grid)

    # DR counterfactual CDF
    cdf_cf: np.ndarray = np.zeros(len(grid))
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
    return dict(
        cdf_treated=cdf_t,
        cdf_cf=cdf_cf,
        dte=dte,
        qte=qt - qcf,
        ks_stat=float(np.max(np.abs(dte))),
    )


def _dte_cic(
    Y: np.ndarray,
    D: np.ndarray,
    grid: np.ndarray,
    taus: np.ndarray,
) -> Dict[str, Any]:
    """Changes-in-Changes distributional estimator.

    D encoding: 0=control-pre, 1=control-post, 2=treated-pre, 3=treated-post.
    Counterfactual: F_{Y(0)|11}(y) = F_01( Q_00( F_10(y) ) )
    """
    from scipy.interpolate import interp1d

    groups = {g: Y[D == g] for g in range(4)}
    if any(len(v) == 0 for v in groups.values()):
        raise ValueError(
            "CiC requires 4 groups: 0=ctrl-pre, 1=ctrl-post, 2=treat-pre, 3=treat-post."
        )

    def _ecdf_f(v: np.ndarray) -> Any:
        sv, c = np.sort(v), np.arange(1, len(v) + 1) / len(v)
        return interp1d(sv, c, bounds_error=False, fill_value=(0.0, 1.0))

    def _qf(v: np.ndarray) -> Any:
        sv, c = np.sort(v), np.arange(1, len(v) + 1) / len(v)
        return interp1d(c, sv, bounds_error=False, fill_value=(sv[0], sv[-1]))

    F_10, F_01 = _ecdf_f(groups[2]), _ecdf_f(groups[1])
    Q_00 = _qf(groups[0])

    cdf_t = _weighted_ecdf(groups[3], np.ones(len(groups[3])), grid)
    cdf_cf = np.array(
        [float(F_01(Q_00(np.clip(float(F_10(g)), 0.001, 0.999)))) for g in grid]
    )
    cdf_cf = np.maximum.accumulate(np.clip(cdf_cf, 0, 1))

    dte = cdf_t - cdf_cf
    qt = _quantile_from_cdf(grid, cdf_t, taus)
    qcf = _quantile_from_cdf(grid, cdf_cf, taus)
    return dict(
        cdf_treated=cdf_t,
        cdf_cf=cdf_cf,
        dte=dte,
        qte=qt - qcf,
        ks_stat=float(np.max(np.abs(dte))),
    )


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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> d = rng.integers(0, 2, n)
    >>> y = 1.0 + 1.5 * d + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "d": d})
    >>> res = sp.distributional_te(
    ...     df, y="y", treatment="d", method="ipw",
    ...     quantiles=[0.25, 0.5, 0.75], n_boot=50, seed=42)
    >>> res.qte_effects.round(2).tolist()  # QTE at each quantile
    [1.63, 1.51, 1.57]
    >>> round(res.ks_stat, 3)  # Kolmogorov-Smirnov statistic
    0.57
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
    yr = (
        Y_vec
        if method == "cic"
        else (Y_vec[D_vec == 1] if np.any(D_vec == 1) else Y_vec)
    )
    margin = 0.01 * np.ptp(yr)
    grid = np.linspace(np.min(yr) - margin, np.max(yr) + margin, n_grid)

    # Dispatch
    _est: Dict[str, Callable[..., Dict[str, Any]]] = {
        "ipw": _dte_ipw,
        "dr": _dte_dr,
        "cic": _dte_cic,
    }
    args = (
        (Y_vec, D_vec, grid, taus)
        if method == "cic"
        else (Y_vec, D_vec, X_mat, grid, taus)
    )
    res0 = _est[method](*args)

    # Bootstrap
    boot_dte = np.full((n_boot, n_grid), np.nan)
    boot_qte = np.full((n_boot, len(taus)), np.nan)
    n_failed = 0

    result_shell = DTEResult(
        grid=grid,
        dte=res0["dte"],
        dte_se=np.zeros(n_grid),
        qte_taus=taus,
        qte_effects=res0["qte"],
        qte_se=np.zeros(len(taus)),
        cdf_treated=res0["cdf_treated"],
        cdf_counterfactual=res0["cdf_cf"],
        ks_stat=res0["ks_stat"],
        ks_pvalue=np.nan,
        n_obs=n,
        method=method,
        alpha=alpha,
    )

    last_exc: Optional[BaseException] = None
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        ba = (
            (Y_vec[idx], D_vec[idx], grid, taus)
            if method == "cic"
            else (
                Y_vec[idx],
                D_vec[idx],
                X_mat[idx] if X_mat is not None else None,
                grid,
                taus,
            )
        )
        try:
            rb = _est[method](*ba)
            boot_dte[b], boot_qte[b] = rb["dte"], rb["qte"]
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            n_failed += 1
            last_exc = exc

    if n_failed:
        from ..workflow._degradation import record_degradation

        record_degradation(
            result_shell,
            section="distributional_te.bootstrap",
            exc=last_exc if last_exc is not None else RuntimeError("resample failed"),
            detail=(
                f"{n_failed}/{n_boot} bootstrap replications failed for "
                f"method={method!r}; SEs and the KS/CvM p-values use the "
                f"remaining {n_boot - n_failed}."
            ),
        )

    # ── Functional tests of H0: no distributional effect ─────────────── #
    #
    # The bootstrap distribution of sup|DTE_b| is NOT a null distribution --
    # it is centred on the estimate, so comparing it against sup|DTE_hat|
    # gives a quantity that is bounded away from 0 whatever the truth.
    # Measured on a no-effect DGP, the pre-1.21 p-value never fell below
    # 0.565 across 40 seeds and rejected at the 5% level 0% of the time.
    #
    # Recentre: under H0 the sampling distribution of sup|DTE_hat| is
    # approximated by that of sup|DTE_b - DTE_hat|, which is what a valid
    # bootstrap p-value must compare against.
    dte_hat = np.asarray(res0["dte"], dtype=float)
    centered = boot_dte - dte_hat[None, :]
    ok = np.isfinite(centered).all(axis=1)
    if ok.sum() >= 2:
        boot_ks_c = np.max(np.abs(centered[ok]), axis=1)
        ks_p = float(np.mean(boot_ks_c >= res0["ks_stat"]))
        # Cramer-von Mises: integrated squared deviation over the grid.
        cvm_stat = float(_trapz(dte_hat**2, grid))
        boot_cvm_c = _trapz(centered[ok] ** 2, grid, axis=1)
        cvm_p = float(np.mean(boot_cvm_c >= cvm_stat))
    else:
        ks_p = np.nan
        cvm_stat = float(_trapz(dte_hat**2, grid))
        cvm_p = np.nan

    result_shell.dte_se = np.nanstd(boot_dte, axis=0)
    result_shell.qte_se = np.nanstd(boot_qte, axis=0)
    result_shell.ks_pvalue = ks_p
    result_shell.cvm_stat = cvm_stat
    result_shell.cvm_pvalue = cvm_p
    result_shell.n_boot_failed = n_failed
    return result_shell
