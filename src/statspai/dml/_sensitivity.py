"""DML-OVB Sensitivity Analysis (Chernozhukov-Cinelli-Newey-Sharma-Syrgkanis 2022).

Implements the "Long Story Short: Omitted Variable Bias in Causal Machine
Learning" formulas for sensitivity analysis of Double / Debiased ML
estimates of partially linear regression coefficients (PLR / IRM ATE).

The bias from a hypothetical unobserved confounder ``Z`` is bounded by

.. math::
    |\\text{bias}| \\le \\sqrt{\\frac{C_Y \\cdot C_D}{1 - C_D}} \\cdot S,

where :math:`C_Y = \\text{Partial-}R^2(Z; Y \\mid D, X)`,
:math:`C_D = \\text{Partial-}R^2(Z; D \\mid X)`, and ``S`` is a
target-specific scaling factor (for the PLR coefficient
``S = \\sigma_{Y\\text{ resid}} / \\sigma_{D\\text{ resid}}``).

The **robustness value** ``RV_q`` is the value of confounding strength
(assuming :math:`C_Y = C_D = \\text{RV}`) at which the bias just equals
``q * |θ̂|``:

.. math::
    \\text{RV}_q = \\frac{\\sqrt{\\tau^4 + 4\\tau^2} - \\tau^2}{2},
    \\quad \\tau = \\frac{q \\cdot |\\hat\\theta|}{S}.

When ``q = 1``, ``RV_1`` is the strength of an equivalent confounder that
would shrink the estimate exactly to zero. ``RV_{q,\\alpha}`` adjusts for
significance: it returns the strength required to push the lower CI
across zero.

Implementation parallels the R ``sensemakr`` interface of Cinelli &
Hazlett (2020) but uses the DML residuals :math:`\\tilde Y, \\tilde D`
in place of OLS residuals.

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
from typing import Optional, Sequence, Dict, Any, List

import numpy as np
import pandas as pd


@dataclass
class DMLSensitivityResult:
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
    benchmarks : pd.DataFrame
        For each benchmark covariate ``X_k``: ``cf_y_bench``,
        ``cf_d_bench``, and the implied bias / adjusted estimate when a
        confounder is assumed to be ``k_y, k_d`` times as strong as
        ``X_k`` in the residualised regression. Empty if ``benchmark``
        not provided.
    s : float
        Scaling factor :math:`S = \\sigma_{Y\\text{resid}} /
        \\sigma_{D\\text{resid}}` used in the bias formula.
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
    0.49
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
    method: str = "DML-OVB (Chernozhukov-Cinelli-Newey 2022)"

    def summary(self) -> str:
        lines = [
            self.method,
            "-" * 64,
            f"  Original estimate                : {self.estimate:+.4f} "
            f"(SE {self.se:.4f})",
            f"  Robustness value RV_q (q={self.q})   : "
            f"{self.rv_q:.4f}   "
            f"({self.rv_q*100:.2f}% partial R^2 to shrink to {1-self.q:.0%})",
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
        From :func:`statspai.dml.dml` (PLR or IRM). Must carry the
        post-fit residuals via ``model_info['_y_resid']``,
        ``model_info['_d_resid']``, and the design matrix for benchmarks.
    q : float, default 1.0
        Bias threshold as a fraction of |θ̂|. ``q=1`` ⇒ confounder needed
        to shrink estimate to zero; ``q=0.5`` ⇒ half the estimate.
    cf_y, cf_d : float, optional
        Hypothesized partial-R² of an unobserved confounder with the
        residualised outcome and treatment. If both are given, the
        report includes a bias bound and adjusted-estimate range.
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
    0.49
    >>> round(sens.bias_bound, 2)  # |bias| under cf_y = cf_d = 0.05
    0.07
    """
    info = result.model_info or {}
    y_resid = info.get("_y_resid")
    d_resid = info.get("_d_resid")
    if y_resid is None or d_resid is None:
        raise ValueError(
            "dml_sensitivity requires post-fit residuals. Re-fit the DML "
            "model with the current statspai.dml.* implementation, which "
            "stores them under model_info['_y_resid'] / ['_d_resid']."
        )
    y_resid = np.asarray(y_resid, dtype=float).ravel()
    d_resid = np.asarray(d_resid, dtype=float).ravel()

    theta = float(result.estimate)
    se = float(result.se)

    # Bias scaling factor S = σ(Y_resid) / σ(D_resid) for PLR.
    # Equivalent forms appear in §3 of the paper.
    sigma_y = float(np.std(y_resid, ddof=1))
    sigma_d = float(np.std(d_resid, ddof=1))
    if sigma_d <= 0:
        raise ValueError(
            "D residual variance is 0; sensitivity undefined."
        )  # pragma: no cover
    s = sigma_y / sigma_d

    rv_q = _robustness_value(q * abs(theta), s)

    # RV_qa: strength to push the (1-α)/2 CI lower bound across zero.
    crit = 1.96  # default α=0.05 two-sided; we honour result.alpha below.
    from scipy import stats as _stats

    if hasattr(result, "alpha") and result.alpha is not None:
        crit = float(_stats.norm.ppf(1 - result.alpha / 2))
    target_for_rv_qa = max(abs(theta) - crit * se, 0.0)
    rv_qa = _robustness_value(target_for_rv_qa, s)

    # User-specified scenario (cf_y, cf_d) — bias bound + adjusted range.
    if cf_y is not None and cf_d is not None:
        cf_d = float(np.clip(cf_d, 0.0, 0.999))
        cf_y = float(np.clip(cf_y, 0.0, 0.999))
        bias_bound = float(np.sqrt(cf_y * cf_d / (1 - cf_d)) * s)
    else:
        bias_bound = float("nan")
    adj_low = theta - bias_bound if np.isfinite(bias_bound) else float("nan")
    adj_high = theta + bias_bound if np.isfinite(bias_bound) else float("nan")

    # Benchmark covariates: compute the partial-R² each contributes to
    # the residualised regression. For PLR we need the X matrix, which
    # we recover from result.model_info['_X_design'] if present.
    benchmarks = pd.DataFrame()
    X_design = info.get("_X_design")
    cov_names = info.get("_covariate_names")
    if benchmark_covariates and X_design is not None and cov_names is not None:
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
            except Exception:  # pragma: no cover
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
        alpha=getattr(result, "alpha", 0.05),
        cf_y=cf_y,
        cf_d=cf_d,
    )
