"""DML-OVB Sensitivity Analysis (Chernozhukov-Cinelli-Newey-Sharma-Syrgkanis 2022).

Implements the "Long Story Short: Omitted Variable Bias in Causal Machine
Learning" formulas for sensitivity analysis of Double / Debiased ML
estimates of partially linear regression coefficients (PLR / IRM ATE).

The bias from a hypothetical unobserved confounder ``Z`` is bounded by

.. math::
    |\\text{bias}| \\le \\sqrt{\\frac{C_Y \\cdot C_D}{1 - C_D}} \\cdot S,

where :math:`C_Y = \\text{Partial-}R^2(Z; Y \\mid D, X)`,
:math:`C_D = \\text{Partial-}R^2(Z; D \\mid X)`, and ``S`` is a
target-specific scaling factor. For the PLR coefficient

.. math::
    S = \\sqrt{\\sigma^2 \\nu^2},\\quad
    \\sigma^2 = E[(Y - \\ell(X) - \\theta(D - m(X)))^2],\\quad
    \\nu^2 = \\frac{1}{E[(D - m(X))^2]},

i.e. the *structural* outcome residual over the treatment residual. The
numerator subtracts the treatment's own contribution; leaving it in
overstates the bias bound and understates the robustness value.

For the IRM average effect the same bound holds with

.. math::
    \\sigma^2 = E[(Y - g(D, X))^2],\\quad
    \\nu^2 = E[2\\,m(W, \\alpha) - \\alpha^2],\\quad
    \\alpha = \\frac{D}{m(X)} - \\frac{1 - D}{1 - m(X)},

the Riesz representer of the ATE (weighted by ``m(X)/P(D=1)`` for the
effect on the treated). Up to 1.38.0 the PLR scaling was applied to IRM
fits, which is not the bound of the paper.

Both bounds are estimates. Their standard errors come from the influence
functions ``psi -/+ c * psi_S`` with ``psi_S = (sigma^2 psi_nu2 + nu^2
psi_sigma2) / (2 S)``, and the reported interval adds a one-sided normal
quantile to each end, the usual interval for an identified set.

The bounds match ``DoubleML``'s ``sensitivity_analysis`` exactly, and so
do the two standard errors, crosswise: ``doubleml`` 0.11.3 reports the
upper bound's standard error for the lower bound and vice versa (see
``tests/external_parity/test_dml_sensitivity_parity.py`` and the
finite-difference check in
``tests/reference_parity/test_dml_sensitivity_bound_scores.py``).

The **robustness value** ``RV_q`` is the value of confounding strength
(assuming :math:`C_Y = C_D = \\text{RV}`) at which the bias just equals
``q * |θ̂|``:

.. math::
    \\text{RV}_q = \\frac{\\sqrt{\\tau^4 + 4\\tau^2} - \\tau^2}{2},
    \\quad \\tau = \\frac{q \\cdot |\\hat\\theta|}{S}.

When ``q = 1``, ``RV_1`` is the strength of an equivalent confounder that
would shrink the estimate exactly to zero. ``RV_{q,\\alpha}`` adjusts for
significance: it is the strength at which the interval for the bound,
not the bound itself, reaches that value.

The reporting interface (robustness values, benchmark covariates,
contour plot) parallels the R ``sensemakr`` package of Cinelli & Hazlett
(2020), but the bound itself is the DML one above, computed from the
cross-fitted residuals :math:`\\tilde Y, \\tilde D`.

References
----------
- Chernozhukov V., Cinelli C., Newey W., Sharma A., Syrgkanis V. (2022).
  "Long Story Short: Omitted Variable Bias in Causal Machine Learning."
  NBER WP 30302; arXiv:2112.13398.
- Cinelli C., Hazlett C. (2020). "Making Sense of Sensitivity:
  Extending Omitted Variable Bias." JRSS B 82(1): 39-67.
  DOI: 10.1111/rssb.12348.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize as _optimize
from scipy import stats as _stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


@dataclass
class DMLSensitivityResult(ResultProtocolMixin):
    """Output of :func:`dml_sensitivity`.

    Attributes
    ----------
    estimate : float
        Original DML point estimate.
    se : float
        Original DML standard error.
    rv_q : float
        Robustness value at the user's q-threshold (default ``q=1`` ⇒
        strength of confounder needed to shrink estimate to zero).
    rv_qa : float
        Confounder strength needed to push the (1-α)·100% lower CI
        across zero. Strictly less than or equal to ``rv_q``.
    bias_bound : float
        Maximum |bias| under the user-specified ``cf_y, cf_d``.
    adjusted_estimate_low : float
    adjusted_estimate_high : float
        Bias-adjusted estimate range under the (cf_y, cf_d) scenario.
    se_low, se_high : float
        Standard errors of the two ends of that range.
    ci_low, ci_high : float
        Interval that covers the range with probability ``1 - alpha``:
        ``adjusted_estimate_low - z * se_low`` and
        ``adjusted_estimate_high + z * se_high`` with the one-sided
        quantile ``z = Phi^{-1}(1 - alpha)``. ``NaN`` for sample-weighted
        fits, which do not store the score elements.
    benchmarks : pd.DataFrame
        For each benchmark covariate ``X_k``: ``cf_y_bench``,
        ``cf_d_bench``, and the implied bias / adjusted estimate when a
        confounder is assumed to be ``k_y, k_d`` times as strong as
        ``X_k`` in the residualised regression. Empty if ``benchmark``
        not provided.
    s : float
        Scaling factor :math:`S = \\sqrt{\\sigma^2\\nu^2}` used in the bias
        formula. For PLR it is the root-mean-square *structural* outcome
        residual :math:`Y-\\ell(X)-\\theta(D-m(X))` over the
        root-mean-square treatment residual :math:`D-m(X)`; for IRM see the
        module notes.
    q : float
    alpha : float
    method : str
        Always ``"DML-OVB (Chernozhukov-Cinelli-Newey 2022)"``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = 0.5 * x1 + rng.normal(size=n)
    >>> y = 1.0 * d + x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> fit = sp.dml(df, y='y', treat='d', covariates=['x1', 'x2'],
    ...              model='plr', ml_g='linear', ml_m='linear', n_folds=2)
    >>> sens = sp.dml_sensitivity(fit, q=1.0, cf_y=0.05, cf_d=0.05)
    >>> isinstance(sens, sp.DMLSensitivityResult)
    True
    >>> round(sens.rv_q, 2)  # confounder strength needed to zero out theta
    0.59
    """

    estimate: float
    se: float
    rv_q: float
    rv_qa: float
    bias_bound: float
    adjusted_estimate_low: float
    adjusted_estimate_high: float
    benchmarks: pd.DataFrame
    s: float
    q: float
    alpha: float
    cf_y: Optional[float] = None
    cf_d: Optional[float] = None
    se_low: float = float("nan")
    se_high: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    method: str = "DML-OVB (Chernozhukov-Cinelli-Newey 2022)"

    def summary(self) -> str:
        lines = [
            self.method,
            "-" * 64,
            f"  Original estimate                : {self.estimate:+.4f} "
            f"(SE {self.se:.4f})",
            f"  Robustness value RV_q (q={self.q})   : "
            f"{self.rv_q:.4f}   "
            f"({self.rv_q * 100:.2f}% partial R^2 to shrink to {1 - self.q:.0%})",
            f"  Robustness value RV_qa (alpha={self.alpha})   : "
            f"{self.rv_qa:.4f}   "
            "(strength to lose significance)",
        ]
        if self.cf_y is not None and self.cf_d is not None:
            lines += [
                f"  Bias bound (cf_y={self.cf_y:.3f}, cf_d={self.cf_d:.3f}) : "
                f"{self.bias_bound:.4f}",
                f"  Adjusted estimate range          : "
                f"[{self.adjusted_estimate_low:+.4f}, "
                f"{self.adjusted_estimate_high:+.4f}]",
            ]
            if np.isfinite(self.ci_low):
                lines += [
                    f"  Std. errors of the two bounds    : "
                    f"{self.se_low:.4f}, {self.se_high:.4f}",
                    f"  {1 - self.alpha:.0%} interval for the range    : "
                    f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}]",
                ]
        if not self.benchmarks.empty:
            lines += [
                "",
                "  Covariate benchmarks (1× as strong as observed X_k):",
                self.benchmarks.round(4).to_string(index=False),
            ]
        lines.append(
            "\n  Reference: Chernozhukov V., Cinelli C., Newey W., Sharma A.,\n"
            "  Syrgkanis V. (2022). 'Long Story Short.' arXiv:2112.13398."
        )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()

    def plot(
        self,
        ax: Any = None,
        levels: Sequence[float] = (0.0, 0.5, 1.0),
        figsize: Any = (6.0, 5.0),
    ) -> Any:
        """Plot bias-contour grid for hypothetical (cf_y, cf_d) pairs.

        Plots the |bias|/|θ̂| contour as a function of the confounder
        strength on Y and D. The user can read off the RV directly.
        Mirrors the ``sensemakr`` plotting convention.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("matplotlib required for plot()") from e

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        grid = np.linspace(0.001, 0.999, 80)
        cy, cd = np.meshgrid(grid, grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            bias = np.sqrt(cy * cd / (1 - cd)) * self.s
        rel_bias = bias / max(abs(self.estimate), 1e-12)

        cs = ax.contour(
            cd, cy, rel_bias, levels=list(levels), colors=["#888", "#1f77b4", "#d62728"]
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
        if not self.benchmarks.empty:
            ax.scatter(
                self.benchmarks["cf_d_bench"],
                self.benchmarks["cf_y_bench"],
                marker="^",
                color="#000",
                s=50,
                zorder=5,
            )
            for _, row in self.benchmarks.iterrows():
                ax.annotate(
                    str(row.get("variable", "")),
                    (row["cf_d_bench"], row["cf_y_bench"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )
        ax.scatter(
            [self.rv_q], [self.rv_q], marker="o", color="#1f77b4", s=80, zorder=5
        )
        ax.annotate(
            f"RV_q={self.rv_q:.3f}",
            (self.rv_q, self.rv_q),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=9,
            color="#1f77b4",
        )
        ax.set_xlabel(r"Partial $R^2$ of confounder with treatment ($C_D$)")
        ax.set_ylabel(r"Partial $R^2$ of confounder with outcome ($C_Y$)")
        ax.set_title("DML-OVB sensitivity (relative bias contours)")
        return fig, ax


class _ScoreBounds:
    """Bias bounds and their sampling error from stored score elements.

    For each cross-fitting repetition, with confounding strength ``c =
    sqrt(cf_y * cf_d / (1 - cf_d))``, the bounds are ``theta -/+ c*S`` with
    ``S = sqrt(sigma^2 nu^2)``, and their influence functions are ``psi
    -/+ c*psi_S`` with ``psi_S = (sigma^2 psi_nu2 + nu^2 psi_sigma2) /
    (2S)``. Repetitions are combined by the median, as ``doubleml`` does;
    the interval adds a one-sided normal quantile to each end, the
    convention for an interval that covers an identified set.
    """

    def __init__(self, reps: List[Dict[str, Any]], level: float) -> None:
        self.reps = reps
        self.level = level
        self._s = [float(np.sqrt(r["sigma2"] * r["nu2"])) for r in reps]
        self._psi_s = [
            (r["sigma2"] * r["psi_nu2"] + r["nu2"] * r["psi_sigma2"]) / (2.0 * s)
            for r, s in zip(reps, self._s)
        ]
        self.max_bias = float(np.median(self._s))

    @staticmethod
    def _se(psi: np.ndarray, cluster: Optional[Dict[str, Any]]) -> float:
        if cluster is None:
            return float(np.sqrt(np.mean(psi**2) / len(psi)))
        codes, tests, sizes = cluster["codes"], cluster["tests"], cluster["sizes"]
        gamma = 0.0
        for test, m in zip(tests, sizes):
            sums = np.bincount(codes[test], weights=psi[test])
            gamma += float(np.sum(sums**2)) / m
        gamma /= len(tests)
        return float(np.sqrt(gamma / (int(codes.max()) + 1)))

    def scenario(self, strength: float) -> Dict[str, float]:
        lows, highs, se_lo, se_hi = [], [], [], []
        for r, s, psi_s in zip(self.reps, self._s, self._psi_s):
            lows.append(r["theta"] - strength * s)
            highs.append(r["theta"] + strength * s)
            se_lo.append(self._se(r["psi"] - strength * psi_s, r.get("cluster")))
            se_hi.append(self._se(r["psi"] + strength * psi_s, r.get("cluster")))
        lo, hi = np.asarray(lows), np.asarray(highs)
        slo, shi = np.asarray(se_lo), np.asarray(se_hi)
        z = float(_stats.norm.ppf(self.level))
        theta_low, theta_high = float(np.median(lo)), float(np.median(hi))
        return {
            "theta_low": theta_low,
            "theta_high": theta_high,
            "se_low": float((np.median(lo + 1.96 * slo) - theta_low) / 1.96),
            "se_high": float((np.median(hi + 1.96 * shi) - theta_high) / 1.96),
            "ci_low": float(np.median(lo - z * slo)),
            "ci_high": float(np.median(hi + z * shi)),
        }

    def robustness_value(self, null: float, what: str, upper_side: bool) -> float:
        """Smallest ``cf_y = cf_d`` at which the bound, or its interval, hits ``null``."""
        side = "high" if upper_side else "low"
        key = f"theta_{side}" if what == "theta" else f"ci_{side}"
        sign = -1.0 if upper_side else 1.0

        def gap(cf: float) -> float:
            return sign * (self.scenario(cf / np.sqrt(1.0 - cf))[key] - null)

        if gap(0.0) <= 0:
            return 0.0
        hi = 1.0 - 1e-12
        if gap(hi) > 0:
            return 1.0  # pragma: no cover
        return float(_optimize.brentq(gap, 0.0, hi, xtol=1e-14, rtol=1e-12))


def _robustness_value(target: float, s: float) -> float:
    """Solve cf² / (1 - cf) = (target / s)² for cf ∈ [0, 1]."""
    if s <= 0 or not np.isfinite(s):
        return float("nan")  # pragma: no cover
    tau = abs(target) / s
    if tau == 0:
        return 0.0  # pragma: no cover
    tau2 = tau * tau
    inside = tau2 * tau2 + 4.0 * tau2
    rv = (np.sqrt(inside) - tau2) / 2.0
    return float(np.clip(rv, 0.0, 1.0))


def dml_sensitivity(
    result: Any,
    q: float = 1.0,
    cf_y: Optional[float] = None,
    cf_d: Optional[float] = None,
    benchmark_covariates: Optional[Sequence[str]] = None,
    k_y: float = 1.0,
    k_d: float = 1.0,
) -> DMLSensitivityResult:
    """Compute DML-OVB sensitivity for a fitted DML CausalResult.

    Parameters
    ----------
    result : CausalResult
        From :func:`statspai.dml.dml` with ``model='plr'`` or
        ``model='irm'`` (ATE or ATTE score), including clustered and
        repeated cross-fitting fits. IV models are refused: the bound is
        for estimands identified by conditional exogeneity.
    q : float, default 1.0
        Bias threshold as a fraction of |θ̂|. ``q=1`` ⇒ confounder needed
        to shrink estimate to zero; ``q=0.5`` ⇒ half the estimate.
    cf_y, cf_d : float, optional
        Hypothesized partial-R² of an unobserved confounder with the
        residualised outcome and treatment. If both are given, the
        report includes a bias bound, the adjusted-estimate range, the
        standard errors of its two ends and an interval for the range.
    benchmark_covariates : list of str, optional
        Subset of the original covariates to benchmark against. For each
        ``X_k``, the benchmark sets ``cf_y_bench, cf_d_bench`` to the
        partial R² that ``X_k`` itself contributes (multiplied by
        ``k_y, k_d`` to express "what if a confounder were k× as strong
        as ``X_k``?").
    k_y, k_d : float
        Multipliers for the benchmark strengths.

    Returns
    -------
    DMLSensitivityResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = 0.5 * x1 + rng.normal(size=n)
    >>> y = 1.0 * d + x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> fit = sp.dml(df, y='y', treat='d', covariates=['x1', 'x2'],
    ...              model='plr', ml_g='linear', ml_m='linear',
    ...              n_folds=2)
    >>> sens = sp.dml_sensitivity(fit, q=1.0, cf_y=0.05, cf_d=0.05)
    >>> round(sens.estimate, 2)
    0.98
    >>> round(sens.rv_q, 2)  # confounder strength to zero out theta
    0.59
    >>> round(sens.bias_bound, 2)  # |bias| under cf_y = cf_d = 0.05
    0.05
    """
    info = result.model_info or {}
    model_tag = str((info or {}).get("dml_model", "")).upper()
    if model_tag in ("PLIV", "IIVM"):
        raise MethodIncompatibility(
            f"dml_sensitivity: not defined for model='{model_tag.lower()}'. The "
            "omitted-variable-bias bound of Chernozhukov et al. (2022) covers "
            "estimands identified by conditional exogeneity (PLR, IRM)."
        )
    theta = float(result.estimate)
    se = float(result.se)
    alpha = getattr(result, "alpha", None)
    alpha = 0.05 if alpha is None else float(alpha)
    y_resid = info.get("_y_resid")
    d_resid = info.get("_d_resid")
    reps = info.get("_sens")

    if reps:
        bounds = _ScoreBounds(reps, level=1.0 - alpha)
        s = bounds.max_bias
    else:
        # Fits that carry residuals but no score elements (sample-weighted
        # PLR): the bound is available, its sampling error is not.
        if model_tag == "IRM":
            raise MethodIncompatibility(
                "dml_sensitivity: the IRM bound needs the Riesz representer, "
                "which sample-weighted fits do not store. Re-fit without "
                "sample_weight."
            )
        if y_resid is None or d_resid is None:
            raise ValueError(
                "dml_sensitivity requires post-fit residuals. Re-fit the DML "
                "model with the current statspai.dml.* implementation, which "
                "stores them under model_info['_y_resid'] / ['_d_resid']."
            )
        y_resid = np.asarray(y_resid, dtype=float).ravel()
        d_resid = np.asarray(d_resid, dtype=float).ravel()
        # S = sqrt(sigma^2 * nu^2) with sigma^2 the second moment of the
        # structural residual Y - l(X) - theta*(D - m(X)) and nu^2 =
        # 1/E[(D - m(X))^2]. Until v1.21 the numerator kept theta*(D - m)
        # in, which overstated the bound (27% on a linear-nuisance fit).
        if float(np.var(d_resid)) <= 0:
            raise ValueError("D residual variance is 0; sensitivity undefined.")
        eps = y_resid - theta * d_resid
        s = float(np.sqrt(np.mean(eps**2)) / np.sqrt(np.mean(d_resid**2)))
        bounds = None

    null = (1.0 - q) * theta
    upper_side = null > theta
    if bounds is not None:
        rv_q = bounds.robustness_value(null, "theta", upper_side)
        rv_qa = bounds.robustness_value(null, "ci", upper_side)
    else:
        crit = float(_stats.norm.ppf(1 - alpha / 2))
        rv_q = _robustness_value(q * abs(theta), s)
        rv_qa = _robustness_value(max(abs(theta) - crit * se, 0.0), s)

    # User-specified scenario (cf_y, cf_d): bias bound, adjusted range and,
    # when the score elements are available, its sampling error.
    se_low = se_high = ci_low = ci_high = float("nan")
    adj_low = adj_high = bias_bound = float("nan")
    if cf_y is not None and cf_d is not None:
        cf_d = float(np.clip(cf_d, 0.0, 0.999))
        cf_y = float(np.clip(cf_y, 0.0, 0.999))
        strength = float(np.sqrt(cf_y * cf_d / (1 - cf_d)))
        bias_bound = strength * s
        if bounds is not None:
            sc = bounds.scenario(strength)
            adj_low, adj_high = sc["theta_low"], sc["theta_high"]
            se_low, se_high = sc["se_low"], sc["se_high"]
            ci_low, ci_high = sc["ci_low"], sc["ci_high"]
        else:
            adj_low, adj_high = theta - bias_bound, theta + bias_bound

    # Benchmark covariates: compute the partial-R² each contributes to
    # the residualised regression. For PLR we need the X matrix, which
    # we recover from result.model_info['_X_design'] if present.
    benchmarks = pd.DataFrame()
    X_design = info.get("_X_design")
    cov_names = info.get("_covariate_names")
    if (
        benchmark_covariates
        and X_design is not None
        and cov_names is not None
        and y_resid is not None
        and d_resid is not None
    ):
        y_resid = np.asarray(y_resid, dtype=float).ravel()
        d_resid = np.asarray(d_resid, dtype=float).ravel()
        X = np.asarray(X_design, dtype=float)
        rows: List[Dict[str, Any]] = []
        for name in benchmark_covariates:
            if name not in cov_names:
                continue  # pragma: no cover
            j = list(cov_names).index(name)
            xk = X[:, j]
            # Partial R²(X_k; D | other X) ≈ corr(X_k, d_resid)²
            # Partial R²(X_k; Y | D, other X) ≈ corr(X_k_resid, y_resid)²
            try:
                cov = np.cov(xk, d_resid)
                r2_d = float(cov[0, 1] ** 2 / max(cov[0, 0] * cov[1, 1], 1e-12))
                cov_y = np.cov(xk, y_resid)
                r2_y = float(cov_y[0, 1] ** 2 / max(cov_y[0, 0] * cov_y[1, 1], 1e-12))
            except Exception as exc:  # pragma: no cover
                from ..core._fallback import warn_fallback

                warn_fallback(
                    f"DML sensitivity benchmark for covariate {name!r}",
                    exc,
                    "it is left out of the benchmark table",
                )
                continue  # pragma: no cover
            cf_y_b = float(np.clip(k_y * r2_y, 0.0, 0.999))
            cf_d_b = float(np.clip(k_d * r2_d, 0.0, 0.999))
            bias_b = float(np.sqrt(cf_y_b * cf_d_b / (1 - cf_d_b)) * s)
            rows.append(
                {
                    "variable": name,
                    "k_y": k_y,
                    "k_d": k_d,
                    "cf_y_bench": cf_y_b,
                    "cf_d_bench": cf_d_b,
                    "bias_bound": bias_b,
                    "adjusted_low": theta - bias_b,
                    "adjusted_high": theta + bias_b,
                }
            )
        benchmarks = pd.DataFrame(rows)

    return DMLSensitivityResult(
        estimate=theta,
        se=se,
        rv_q=rv_q,
        rv_qa=rv_qa,
        bias_bound=bias_bound,
        adjusted_estimate_low=adj_low,
        adjusted_estimate_high=adj_high,
        benchmarks=benchmarks,
        s=s,
        q=q,
        alpha=alpha,
        cf_y=cf_y,
        cf_d=cf_d,
        se_low=se_low,
        se_high=se_high,
        ci_low=ci_low,
        ci_high=ci_high,
    )
