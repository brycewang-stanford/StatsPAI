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

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from .basic_tests import _check_alpha

__all__ = [
    "MetaAnalysisResult",
    "meta_analysis",
]

#: Printed name of each between-study variance estimator.
_TAU2_NAMES = {
    "DL": "DerSimonian-Laird",
    "REML": "REML",
    "ML": "maximum likelihood",
    "PM": "Paule-Mandel",
}


@dataclass
class MetaAnalysisResult(ResultProtocolMixin):
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
        ``"fixed"``, or random effects with the between-study variance
        estimated by ``"DL"`` (DerSimonian-Laird), ``"REML"`` (restricted
        maximum likelihood), ``"ML"`` (maximum likelihood) or ``"PM"``
        (Paule-Mandel).
    fixed_estimate, fixed_se : float
        Fixed-effect pooled estimate and SE (always reported).
    random_estimate, random_se : float
        Random-effects pooled estimate and SE (always reported), with the
        between-study variance of ``method`` (DerSimonian-Laird under
        ``"fixed"``). With ``hksj=True`` the SE is the Knapp-Hartung one.
    hksj : bool
        Whether the random-effects inference uses the Hartung-Knapp-
        Sidik-Jonkman adjustment (t with ``k - 1`` degrees of freedom).
    tau2, q, q_df, q_pvalue, i2, h2 : float
        Between-study variance and heterogeneity statistics.
    prediction_interval : tuple or None
        Random-effects prediction interval (where a future study's true
        effect is expected to lie); ``None`` when fewer than 3 studies.
    weights : np.ndarray
        Per-study weights under the chosen model (normalised to sum to 1).

    Examples
    --------
    >>> import statspai as sp
    >>> effects = [0.30, 0.45, 0.20, 0.55, 0.35]
    >>> se = [0.12, 0.18, 0.10, 0.22, 0.14]
    >>> result = sp.meta_analysis(effects, se, method="DL")
    >>> type(result).__name__
    'MetaAnalysisResult'
    >>> bool(0.0 <= result.i2 <= 100.0)  # I^2 heterogeneity is a percentage
    True
    >>> bool(result.estimate > 0)  # all studies show a positive effect
    True
    """

    _citation_keys = (
        "dersimonian1986meta",
        "higgins2002quantifying",
        "egger1997bias",
    )

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
    hksj: bool = False

    def egger_test(self) -> Dict[str, float]:
        """Egger's regression test for funnel-plot asymmetry.

        Regresses the standard normal deviate ``y_i / se_i`` on precision
        ``1 / se_i``; a non-zero intercept indicates small-study effects.
        Requires at least 3 studies.
        """
        y = self.effects
        s = self.se_studies
        k = len(y)
        # With equal standard errors the precision is constant and collinear
        # with the intercept: the asymmetry test is not estimable.
        design = np.column_stack([np.ones(k), 1.0 / s])
        if k < 3 or np.linalg.matrix_rank(design) < 2:
            return {
                "intercept": float("nan"),
                "se": float("nan"),
                "t": float("nan"),
                "p_value": float("nan"),
                "df": k - 2,
            }
        snd = y / s  # standard normal deviate
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
        return {
            "intercept": float(beta[0]),
            "se": se_intercept,
            "t": t_stat,
            "p_value": p,
            "df": dof,
        }

    def summary(self) -> str:
        z = stats.norm.ppf(1 - self.alpha / 2)
        k = len(self.effects)
        out = ["=" * 72, "Meta-analysis (summary data)", "=" * 72]
        out.append(f"Studies (k)          : {k}")
        tau_name = _TAU2_NAMES.get(self.method, "DerSimonian-Laird")
        model = (
            "fixed-effect (inverse-variance)"
            if self.method == "fixed"
            else f"random-effects ({tau_name})"
        )
        out.append(f"Model                : {model}")
        if self.hksj:
            out.append(
                f"Inference            : Knapp-Hartung adjustment, t({k - 1}) "
                "for the random-effects pooled effect"
            )
        out.append("-" * 72)
        fe_lo = self.fixed_estimate - z * self.fixed_se
        fe_hi = self.fixed_estimate + z * self.fixed_se
        crit = stats.t.ppf(1 - self.alpha / 2, df=k - 1) if self.hksj else z
        re_lo = self.random_estimate - crit * self.random_se
        re_hi = self.random_estimate + crit * self.random_se
        fe_ci = f"[{fe_lo:.4f}, {fe_hi:.4f}]"
        re_ci = f"[{re_lo:.4f}, {re_hi:.4f}]"
        out.append(f"Fixed-effect pooled  : {self.fixed_estimate:.4f}  {fe_ci}")
        out.append(f"Random-effects pooled: {self.random_estimate:.4f}  {re_ci}")
        if self.prediction_interval is not None:
            pi_lo, pi_hi = self.prediction_interval
            out.append(
                f"  prediction interval: [{pi_lo:.4f}, {pi_hi:.4f}]  "
                f"(t, {k - 2} df; Higgins et al. 2009)"
            )
        out.append("-" * 72)
        out.append("Heterogeneity:")
        out.append(f"  Q({self.q_df}) = {self.q:.4f}, p = {self.q_pvalue:.4g}")
        i2_pct = 100 * self.i2
        out.append(
            f"  I^2 = {i2_pct:.1f}%   tau^2 = {self.tau2:.4f} ({tau_name})"
            f"   H^2 = {self.h2:.4f}"
        )
        egg = self.egger_test()
        if not np.isnan(egg["p_value"]):
            intercept = egg["intercept"]
            out.append(
                f"Egger's test: intercept = {intercept:.4f}, "
                f"p = {egg['p_value']:.4g}"
            )
        out.append("=" * 72)
        return "\n".join(out)

    def trim_and_fill(self, **kwargs: Any) -> Any:
        """Duval-Tweedie trim-and-fill adjustment of this meta-analysis.

        Keyword arguments (``side=``, ``estimator=``, ``maxiter=``) are
        those of :func:`statspai.trim_and_fill`; the model, the Knapp-Hartung
        choice and ``alpha`` are those of this fit.
        """
        from .meta_regression import trim_and_fill

        return trim_and_fill(self, **kwargs)

    def meta_regression(self, moderators: Any, **kwargs: Any) -> Any:
        """Meta-regression of these studies on study-level ``moderators``.

        The between-study variance estimator, the Knapp-Hartung choice and
        ``alpha`` default to those of this fit; other keyword arguments are
        those of :func:`statspai.meta_regression`.
        """
        from .meta_regression import meta_regression

        kwargs.setdefault("method", self.method)
        kwargs.setdefault("hksj", self.hksj)
        kwargs.setdefault("alpha", self.alpha)
        return meta_regression(self.effects, self.se_studies, moderators, **kwargs)

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
        ax.errorbar(
            y,
            positions,
            xerr=z * s,
            fmt="s",
            color="#333",
            capsize=3,
            markersize=5,
            linestyle="none",
        )
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
    hksj: bool = False,
) -> MetaAnalysisResult:
    """Summary-data meta-analysis with fixed- and random-effects pooling.

    Pools study-level effect sizes by inverse variance (fixed effect) and
    under a random-effects model with the DerSimonian-Laird, REML, maximum
    likelihood or Paule-Mandel estimate of the between-study variance, with
    the heterogeneity
    statistics (Cochran's Q, I-squared, tau-squared, H-squared), a
    prediction interval, the optional Knapp-Hartung (HKSJ) adjustment,
    Egger's regression test for funnel-plot asymmetry (small-study effects,
    publication bias) and forest and funnel plots. R ``metafor::rma``,
    Stata ``meta summarize``.

    Parameters
    ----------
    effects : sequence of float
        Per-study effect sizes (e.g. log odds ratios, mean differences).
    se : sequence of float
        Per-study standard errors (must be positive).
    method : {"DL", "REML", "ML", "PM", "fixed"}
        Which model the headline ``estimate`` reports: random effects with
        the DerSimonian-Laird moment estimate of the between-study
        variance (default), with its restricted-maximum-likelihood
        estimate (``"REML"``, the default of R ``metafor::rma`` and Stata
        ``meta``), its maximum-likelihood estimate (``"ML"``) or the
        Paule-Mandel estimate (``"PM"``, the between-study variance at
        which the generalised Q statistic equals its degrees of freedom),
        or fixed-effect inverse-variance. The fixed-effect and a
        random-effects result are always both available on the result.
    labels : sequence of str, optional
        Study labels for the forest plot.
    alpha : float
        Significance level for confidence/prediction intervals.
    hksj : bool, default False
        Hartung-Knapp-Sidik-Jonkman adjustment of the random-effects
        inference (R ``rma(test = "knha")``, Stata ``meta summarize,
        se(khartung)``): the variance of the pooled effect is
        ``sum w_i (y_i - mu)^2 / ((k - 1) sum w_i)`` with the random-
        effects weights, and the interval and p-value use the t
        distribution with ``k - 1`` degrees of freedom. It keeps the
        nominal coverage with few studies, where the normal interval is
        too narrow. Not with ``method="fixed"``.

    Notes
    -----
    I-squared and H-squared are the Q-based ones, ``(Q - df) / Q`` and
    ``Q / df`` (Higgins and Thompson 2002), under every ``method``;
    ``metafor`` reports tau-squared-based versions for ``"REML"``, which
    differ slightly. The prediction interval is
    ``mu +/- t(k - 2) sqrt(se^2 + tau^2)`` (Higgins, Thompson and
    Spiegelhalter 2009), with the model-based ``se``; ``metafor``'s
    default uses the normal quantile. The iterative estimators (``"REML"``,
    ``"ML"``, ``"PM"``) are run to machine precision; ``metafor`` stops at
    a relative change of 1e-5 by default, so its printed ``tau^2`` agrees to
    about five digits unless its ``control`` tolerances are tightened.

    For moderators see :func:`statspai.meta_regression`; for the
    Duval-Tweedie adjustment see :func:`statspai.trim_and_fill` (also
    available as ``result.trim_and_fill()``).

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
    >>> r.method, round(r.estimate, 4)
    ('DL', 0.1305)
    >>> round(r.i2, 3)
    0.549
    >>> report = r.summary()
    """
    y = np.asarray(effects, dtype=float)
    s = np.asarray(se, dtype=float)
    _check_alpha(alpha, "meta_analysis")
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("effects and se must be 1-D arrays of equal length.")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(s))):
        raise MethodIncompatibility(
            "meta_analysis: effects and standard errors must be finite; "
            f"{int((~(np.isfinite(y) & np.isfinite(s))).sum())} study(ies) "
            "have a missing or infinite value.",
            recovery_hint="Drop those studies before pooling.",
        )
    if np.any(s <= 0):
        raise ValueError("all standard errors must be positive.")
    k = len(y)
    if k < 2:
        raise ValueError("meta-analysis needs at least 2 studies.")
    if method not in ("DL", "REML", "ML", "PM", "fixed"):
        raise ValueError("method must be 'DL', 'REML', 'ML', 'PM' or 'fixed'.")
    if hksj and method == "fixed":
        raise MethodIncompatibility(
            "meta_analysis: the Knapp-Hartung adjustment (hksj=True) belongs "
            "to the random-effects model.",
            recovery_hint="Use method='DL' or 'REML' with hksj=True, or "
            "hksj=False with method='fixed'.",
        )
    if labels is None:
        labels = [f"Study {i + 1}" for i in range(k)]
    else:
        labels = list(labels)
        if len(labels) != k:
            raise MethodIncompatibility(
                f"meta_analysis: {len(labels)} labels for {k} studies.",
                recovery_hint="Pass one label per study, or labels=None.",
            )

    # Fixed-effect inverse-variance.
    w = 1.0 / s**2
    fe_est = float(np.sum(w * y) / np.sum(w))
    fe_se = float(np.sqrt(1.0 / np.sum(w)))

    # Cochran's Q and DerSimonian-Laird tau^2.
    q = float(np.sum(w * (y - fe_est) ** 2))
    df = k - 1
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0
    if method == "REML":
        tau2 = _reml_tau2(y, s**2, tau2)
    elif method in ("ML", "PM"):
        from ._meta_core import estimate_tau2

        tau2 = estimate_tau2(y, s**2, np.ones((k, 1)), method)
    q_pvalue = float(stats.chi2.sf(q, df)) if df > 0 else float("nan")
    i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    h2 = q / df if df > 0 else float("nan")

    # Random-effects weights.
    w_re = 1.0 / (s**2 + tau2)
    re_est = float(np.sum(w_re * y) / np.sum(w_re))
    re_se_model = float(np.sqrt(1.0 / np.sum(w_re)))
    re_se = re_se_model
    if hksj:
        # Knapp and Hartung (2003): the weighted residual mean square
        # replaces 1 in the variance of the pooled effect.
        re_se = float(np.sqrt(np.sum(w_re * (y - re_est) ** 2) / (df * np.sum(w_re))))

    z = stats.norm.ppf(1 - alpha / 2)
    if method == "fixed":
        est, se_pooled, weights = fe_est, fe_se, w
    else:
        est, se_pooled, weights = re_est, re_se, w_re

    if hksj:
        crit = float(stats.t.ppf(1 - alpha / 2, df=df))
        p_value = float(2 * stats.t.sf(abs(est / se_pooled), df=df))
    else:
        crit = z
        p_value = float(2 * stats.norm.sf(abs(est / se_pooled)))
    ci = (float(est - crit * se_pooled), float(est + crit * se_pooled))

    # Random-effects prediction interval (Higgins et al. 2009): uses t_{k-2}.
    prediction_interval: Optional[tuple] = None
    if k >= 3:
        t_val = stats.t.ppf(1 - alpha / 2, df=k - 2)
        pi_se = np.sqrt(re_se_model**2 + tau2)
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
        hksj=bool(hksj),
    )


def _reml_tau2(y: np.ndarray, v: np.ndarray, start: float) -> float:
    """REML estimate of the between-study variance, truncated at zero.

    Fisher scoring on the restricted likelihood of the random-effects
    model ``y_i ~ N(mu, v_i + tau2)`` (Viechtbauer 2005): with
    ``w_i = 1 / (v_i + tau2)`` and ``mu`` the weighted mean,

        tau2 <- tau2 + [sum w_i^2 (y_i - mu)^2 - sum w_i + sum w_i^2 / sum w_i]
                       / [sum w_i^2 - 2 sum w_i^3 / sum w_i + (sum w_i^2 / sum w_i)^2]

    The step is halved while it lowers the restricted likelihood, so the
    iteration cannot overshoot, and it stops when the step no longer moves
    ``tau2`` at double precision relative to the within-study variances.
    """

    def _ll(t2: float) -> float:
        w = 1.0 / (v + t2)
        mu = np.sum(w * y) / np.sum(w)
        return float(
            -0.5 * np.sum(np.log(v + t2))
            - 0.5 * np.log(np.sum(w))
            - 0.5 * np.sum(w * (y - mu) ** 2)
        )

    tau2 = max(float(start), 0.0)
    scale = float(np.mean(v))
    for _ in range(500):
        w = 1.0 / (v + tau2)
        sw = np.sum(w)
        mu = np.sum(w * y) / sw
        sw2 = np.sum(w**2)
        score = np.sum(w**2 * (y - mu) ** 2) - sw + sw2 / sw
        info = sw2 - 2.0 * np.sum(w**3) / sw + (sw2 / sw) ** 2
        if not info > 0:
            break
        step = score / info
        new = max(tau2 + step, 0.0)
        ll_old = _ll(tau2)
        for _ in range(60):
            if _ll(new) >= ll_old - 1e-12 * max(1.0, abs(ll_old)):
                break
            new = 0.5 * (new + tau2)
        moved = abs(new - tau2)
        tau2 = new
        if moved <= 1e-14 * max(scale, tau2):
            break
    return float(tau2)
