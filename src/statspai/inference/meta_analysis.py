"""Study-level meta-analysis (evidence synthesis).

Pools effect sizes (and their standard errors) across studies, the workhorse
of systematic reviews in public health and clinical research. Provides both
the fixed-effect (inverse-variance) and random-effects (DerSimonian-Laird)
models, the standard heterogeneity statistics (Cochran's Q, I^2, tau^2, H^2),
a random-effects prediction interval, and Egger's test for small-study /
funnel-plot asymmetry — plus forest- and funnel-plot helpers.

This is intentionally summary-data meta-analysis (you pass per-study effects
and SEs); it does not fit individual-participant-data models.

References
----------
DerSimonian, R. & Laird, N. (1986). "Meta-analysis in clinical trials."
*Controlled Clinical Trials*, 7(3), 177-188. [@dersimonian1986meta]

Higgins, J.P.T. & Thompson, S.G. (2002). "Quantifying heterogeneity in a
meta-analysis." *Statistics in Medicine*, 21(11), 1539-1558.
[@higgins2002quantifying]

Egger, M., Davey Smith, G., Schneider, M. & Minder, C. (1997). "Bias in
meta-analysis detected by a simple, graphical test." *BMJ*, 315(7109),
629-634. [@egger1997bias]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "MetaAnalysisResult",
    "meta_analysis",
]


@dataclass
class MetaAnalysisResult:
    """Result of a summary-data meta-analysis.

    Attributes
    ----------
    estimate : float
        Pooled effect under the chosen model (random-effects by default).
    se, ci : float, tuple
        Standard error and confidence interval of the pooled effect.
    p_value : float
        Two-sided p-value for the pooled effect being zero.
    method : str
        ``"fixed"`` or ``"DL"`` (DerSimonian-Laird random effects).
    fixed_estimate, fixed_se : float
        Fixed-effect pooled estimate and SE (always reported).
    random_estimate, random_se : float
        Random-effects pooled estimate and SE (always reported).
    tau2, q, q_df, q_pvalue, i2, h2 : float
        Between-study variance and heterogeneity statistics.
    prediction_interval : tuple or None
        Random-effects prediction interval (where a future study's true
        effect is expected to lie); ``None`` when fewer than 3 studies.
    weights : np.ndarray
        Per-study weights under the chosen model (normalised to sum to 1).
    """

    estimate: float
    se: float
    ci: tuple
    p_value: float
    method: str
    fixed_estimate: float
    fixed_se: float
    random_estimate: float
    random_se: float
    tau2: float
    q: float
    q_df: int
    q_pvalue: float
    i2: float
    h2: float
    prediction_interval: Optional[tuple]
    weights: np.ndarray
    effects: np.ndarray
    se_studies: np.ndarray
    labels: List[str]
    alpha: float = 0.05

    def egger_test(self) -> Dict[str, float]:
        """Egger's regression test for funnel-plot asymmetry.

        Regresses the standard normal deviate ``y_i / se_i`` on precision
        ``1 / se_i``; a non-zero intercept indicates small-study effects.
        Requires at least 3 studies.
        """
        y = self.effects
        s = self.se_studies
        k = len(y)
        if k < 3:
            return {"intercept": float("nan"), "se": float("nan"),
                    "t": float("nan"), "p_value": float("nan"), "df": k - 2}
        snd = y / s                 # standard normal deviate
        precision = 1.0 / s
        X = np.column_stack([np.ones(k), precision])
        beta, *_ = np.linalg.lstsq(X, snd, rcond=None)
        resid = snd - X @ beta
        dof = k - 2
        sigma2 = float(resid @ resid) / dof
        xtx_inv = np.linalg.inv(X.T @ X)
        se_intercept = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
        t_stat = float(beta[0] / se_intercept)
        p = float(2 * stats.t.sf(abs(t_stat), dof))
        return {"intercept": float(beta[0]), "se": se_intercept,
                "t": t_stat, "p_value": p, "df": dof}

    def summary(self) -> str:
        z = stats.norm.ppf(1 - self.alpha / 2)
        out = ["=" * 72, "Meta-analysis (summary data)", "=" * 72]
        out.append(f"Studies (k)          : {len(self.effects)}")
        model = ("random-effects (DerSimonian-Laird)"
                 if self.method == "DL" else "fixed-effect (inverse-variance)")
        out.append(f"Model                : {model}")
        out.append("-" * 72)
        fe_lo = self.fixed_estimate - z * self.fixed_se
        fe_hi = self.fixed_estimate + z * self.fixed_se
        re_lo = self.random_estimate - z * self.random_se
        re_hi = self.random_estimate + z * self.random_se
        fe_ci = f"[{fe_lo:.4f}, {fe_hi:.4f}]"
        re_ci = f"[{re_lo:.4f}, {re_hi:.4f}]"
        out.append(f"Fixed-effect pooled  : {self.fixed_estimate:.4f}  {fe_ci}")
        out.append(f"Random-effects pooled: {self.random_estimate:.4f}  {re_ci}")
        if self.prediction_interval is not None:
            pi_lo, pi_hi = self.prediction_interval
            out.append(f"  prediction interval: [{pi_lo:.4f}, {pi_hi:.4f}]")
        out.append("-" * 72)
        out.append("Heterogeneity:")
        out.append(f"  Q({self.q_df}) = {self.q:.4f}, p = {self.q_pvalue:.4g}")
        i2_pct = 100 * self.i2
        out.append(f"  I^2 = {i2_pct:.1f}%   tau^2 = {self.tau2:.4f}"
                   f"   H^2 = {self.h2:.4f}")
        egg = self.egger_test()
        if not np.isnan(egg["p_value"]):
            intercept = egg["intercept"]
            out.append(f"Egger's test: intercept = {intercept:.4f}, "
                       f"p = {egg['p_value']:.4g}")
        out.append("=" * 72)
        return "\n".join(out)

    def forest_plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Forest plot: per-study effects + CIs with the pooled diamond."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(
                figsize=kwargs.pop("figsize", (7, 0.4 * len(self.effects) + 2))
            )
        z = stats.norm.ppf(1 - self.alpha / 2)
        y = self.effects
        s = self.se_studies
        k = len(y)
        positions = np.arange(k, 0, -1)
        ax.errorbar(y, positions, xerr=z * s, fmt="s", color="#333",
                    capsize=3, markersize=5, linestyle="none")
        ax.axvline(0.0, color="grey", lw=0.8, linestyle="--")
        pooled = self.estimate
        p_lo, p_hi = self.ci
        ax.axvline(pooled, color="#c1272d", lw=1.0)
        ax.fill_betweenx([0.2, 0.8], p_lo, p_hi, color="#c1272d", alpha=0.3)
        ax.set_yticks(list(positions) + [0.5])
        ax.set_yticklabels(self.labels + ["Pooled"])
        ax.set_xlabel("Effect size")
        ax.set_title("Forest plot")
        return ax

    def funnel_plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Funnel plot: effect size vs. standard error (precision)."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 5)))
        ax.scatter(self.effects, self.se_studies, color="#333", s=25)
        ax.axvline(self.estimate, color="#c1272d", lw=1.0)
        ax.set_xlabel("Effect size")
        ax.set_ylabel("Standard error")
        ax.invert_yaxis()
        ax.set_title("Funnel plot")
        return ax

    def __repr__(self) -> str:
        return (
            f"<MetaAnalysisResult: k={len(self.effects)}, "
            f"{self.method}, pooled={self.estimate:.4f}, "
            f"I2={100 * self.i2:.0f}%>"
        )


def meta_analysis(
    effects: Sequence[float],
    se: Sequence[float],
    *,
    method: str = "DL",
    labels: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> MetaAnalysisResult:
    """Summary-data meta-analysis with fixed- and random-effects pooling.

    Parameters
    ----------
    effects : sequence of float
        Per-study effect sizes (e.g. log odds ratios, mean differences).
    se : sequence of float
        Per-study standard errors (must be positive).
    method : {"DL", "fixed"}
        Which model the headline ``estimate`` reports: DerSimonian-Laird
        random effects (default) or fixed-effect inverse-variance. Both are
        always computed and available on the result.
    labels : sequence of str, optional
        Study labels for the forest plot.
    alpha : float
        Significance level for confidence/prediction intervals.

    Returns
    -------
    MetaAnalysisResult
        With ``.estimate``, ``.ci``, heterogeneity statistics, ``.summary()``,
        ``.egger_test()``, ``.forest_plot()``, ``.funnel_plot()``.

    Examples
    --------
    >>> import statspai as sp
    >>> # five studies' log odds ratios and SEs
    >>> r = sp.meta_analysis([0.10, 0.25, -0.05, 0.30, 0.15],
    ...                      [0.05, 0.10, 0.08, 0.12, 0.06])
    >>> r.summary()
    """
    y = np.asarray(effects, dtype=float)
    s = np.asarray(se, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("effects and se must be 1-D arrays of equal length.")
    if np.any(s <= 0):
        raise ValueError("all standard errors must be positive.")
    k = len(y)
    if k < 2:
        raise ValueError("meta-analysis needs at least 2 studies.")
    if method not in ("DL", "fixed"):
        raise ValueError("method must be 'DL' or 'fixed'.")
    if labels is None:
        labels = [f"Study {i + 1}" for i in range(k)]
    else:
        labels = list(labels)

    # Fixed-effect inverse-variance.
    w = 1.0 / s ** 2
    fe_est = float(np.sum(w * y) / np.sum(w))
    fe_se = float(np.sqrt(1.0 / np.sum(w)))

    # Cochran's Q and DerSimonian-Laird tau^2.
    q = float(np.sum(w * (y - fe_est) ** 2))
    df = k - 1
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    q_pvalue = float(stats.chi2.sf(q, df)) if df > 0 else float("nan")
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    h2 = q / df if df > 0 else float("nan")

    # Random-effects (DL) weights.
    w_re = 1.0 / (s ** 2 + tau2)
    re_est = float(np.sum(w_re * y) / np.sum(w_re))
    re_se = float(np.sqrt(1.0 / np.sum(w_re)))

    z = stats.norm.ppf(1 - alpha / 2)
    if method == "DL":
        est, se_pooled, weights = re_est, re_se, w_re
    else:
        est, se_pooled, weights = fe_est, fe_se, w

    ci = (est - z * se_pooled, est + z * se_pooled)
    p_value = float(2 * stats.norm.sf(abs(est / se_pooled)))

    # Random-effects prediction interval (Higgins et al. 2009): uses t_{k-2}.
    prediction_interval: Optional[tuple] = None
    if k >= 3:
        t_val = stats.t.ppf(1 - alpha / 2, df=k - 2)
        pi_se = np.sqrt(re_se ** 2 + tau2)
        prediction_interval = (re_est - t_val * pi_se, re_est + t_val * pi_se)

    return MetaAnalysisResult(
        estimate=est,
        se=se_pooled,
        ci=ci,
        p_value=p_value,
        method=method,
        fixed_estimate=fe_est,
        fixed_se=fe_se,
        random_estimate=re_est,
        random_se=re_se,
        tau2=tau2,
        q=q,
        q_df=df,
        q_pvalue=q_pvalue,
        i2=i2,
        h2=h2,
        prediction_interval=prediction_interval,
        weights=weights / np.sum(weights),
        effects=y,
        se_studies=s,
        labels=labels,
        alpha=alpha,
    )
