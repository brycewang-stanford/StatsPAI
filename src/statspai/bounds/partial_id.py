"""
Advanced Partial Identification and Treatment Effect Bounds.

Methods
-------
- **Horowitz-Manski Bounds** : Tighter bounds conditioning on covariates (2000)
- **IV Bounds** : Bounds under imperfect instruments (Nevo & Rosen 2012)
- **Oster Delta** : Coefficient stability / identified set (Oster 2019)
- **Selection Bounds** : Lee bounds with covariates (Lee 2009, conditional)
- **Breakdown Frontier** : Assumption robustness frontier (Masten & Poirier 2021)

References
----------
Horowitz, J. L. & Manski, C. F. (2000).
"Nonparametric Analysis of Randomized Experiments with Missing
Covariate and Outcome Data." JASA, 95(449), 77-84.
[@horowitz2000nonparametric]

Nevo, A. & Rosen, A. M. (2012). "Identification with Imperfect Instruments."
Review of Economics and Statistics, 94(3), 659-671.
doi:10.1162/REST_a_00171 (verified via Crossref and cemmap, 2026-10-27).
[@nevo2012identification]

Oster, E. (2019).
"Unobservable Selection and Coefficient Stability: Theory and Evidence."
Journal of Business & Economic Statistics, 37(2), 187-204. [@oster2019unobservable]

Lee, D. S. (2009).
"Training, Wages, and Sample Selection." RES, 76(3), 1071-1102. [@lee2009training]

Masten, M. A. & Poirier, A. (2021).
"Salvaging Falsified Instrumental Variable Models."
Econometrica, 89(3), 1449-1469. [@masten2021salvaging]
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from .._result_serialize import ResultProtocolMixin
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility

# ======================================================================
# BoundsResult — Shared result class for all bounding methods
# ======================================================================


@dataclass
class BoundsResult(ResultProtocolMixin):
    """Result container for partial identification / bounds estimation.

    Attributes
    ----------
    lower : float
        Lower bound of the identified set.
    upper : float
        Upper bound of the identified set.
    se_lower : float
        Standard error of the lower bound.
    se_upper : float
        Standard error of the upper bound.
    ci_lower : tuple
        Confidence interval for the lower bound (lower_lo, lower_hi).
    ci_upper : tuple
        Confidence interval for the upper bound (upper_lo, upper_hi).
    method : str
        Name of the bounding method.
    alpha : float
        Significance level used for confidence intervals.
    n_obs : int
        Number of observations used.
    model_info : dict
        Additional method-specific information.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> age = rng.integers(20, 60, n).astype(float)
    >>> education = rng.integers(8, 18, n).astype(float)
    >>> trained = rng.integers(0, 2, n)
    >>> wage = 10 + 0.3 * age + 0.5 * education + 5 * trained + rng.normal(0, 3, n)
    >>> df = pd.DataFrame(
    ...     {"wage": wage, "trained": trained,
    ...      "age": age, "education": education})
    >>> res = sp.horowitz_manski(
    ...     data=df, y="wage", treatment="trained",
    ...     covariates=["age", "education"], n_boot=50, random_state=0)
    >>> isinstance(res, sp.BoundsResult)
    True
    >>> bool(res.lower <= res.upper)
    True
    >>> bool(res.width >= 0)
    True
    """

    lower: float
    upper: float
    se_lower: float
    se_upper: float
    ci_lower: Tuple[float, float]
    ci_upper: Tuple[float, float]
    method: str
    alpha: float = 0.05
    n_obs: int = 0
    model_info: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        """Width of the identified set."""
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        """Midpoint of the identified set."""
        return (self.lower + self.upper) / 2.0

    def parameter_ci(self) -> Tuple[float, float]:
        """Imbens-Manski (2004) confidence interval for the parameter.

        ``[lower - c se_lower, upper + c se_upper]`` with ``c`` solving
        ``Phi(c + width / max(se)) - Phi(-c) = 1 - alpha``: between the
        one-sided ``z_{1-alpha}`` (wide set) and the two-sided
        ``z_{1-alpha/2}`` (point identified). The same interval
        ``sp.lee_bounds`` reports. Until 2026-10-27 the printed interval was
        labelled Imbens-Manski but used ``z_{1-alpha/2}``, which covers the
        whole identified set and over-covers the parameter.
        """
        from .lee_manski import _imbens_manski_cn

        ses = [v for v in (self.se_lower, self.se_upper) if np.isfinite(v)]
        sigma = max(ses) if ses else float("nan")
        c = _imbens_manski_cn(max(self.width, 0.0), sigma, self.alpha)
        # an open side (a one-sided set) stays open
        lo = (
            self.lower
            if not np.isfinite(self.lower)
            else self.lower - c * self.se_lower
        )
        hi = (
            self.upper
            if not np.isfinite(self.upper)
            else self.upper + c * self.se_upper
        )
        return float(lo), float(hi)

    def includes_zero(self) -> bool:
        """Check whether the identified set includes zero."""
        return self.lower <= 0 <= self.upper

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary string."""
        pct = int((1 - self.alpha) * 100)
        im_lo, im_hi = self.parameter_ci()

        lines = [
            "",
            "\u2501" * 56,
            f"  {self.method}",
            "\u2501" * 56,
            f"  Lower bound:  {self.lower:>10.4f}   (SE: {self.se_lower:.4f})",
            f"  Upper bound:  {self.upper:>10.4f}   (SE: {self.se_upper:.4f})",
            f"  {pct}% CI:       [{im_lo:.4f}, {im_hi:.4f}]",
            "",
        ]

        if self.includes_zero():
            lines.append("  \u2192 Bounds include zero: cannot reject null")
        else:
            direction = "positive" if self.lower > 0 else "negative"
            lines.append(
                f"  \u2192 Bounds exclude zero: treatment effect is {direction}"
            )

        # Method-specific extras
        if "selection_rate_treated" in self.model_info:
            lines.append(
                f"  \u2192 Selection rate: "
                f"{self.model_info['selection_rate_treated']:.1%} (treated), "
                f"{self.model_info['selection_rate_control']:.1%} (control)"
            )
        if "delta_star" in self.model_info:
            lines.append(
                f"  \u2192 \u03b4* for \u03b2=0: "
                f"{self.model_info['delta_star']:.4f}"
            )
        if "r_max" in self.model_info:
            lines.append(
                f"  \u2192 R\u00b2_max assumption: {self.model_info['r_max']:.4f}"
            )

        lines.append("\u2501" * 56)
        lines.append("")

        text = "\n".join(lines)
        print(text)
        return text

    def _repr_html_(self) -> str:
        """Rich HTML for Jupyter notebooks."""
        pct = int((1 - self.alpha) * 100)
        im_lo, im_hi = self.parameter_ci()

        if self.includes_zero():
            verdict = '<span style="color:#e67e22">Bounds include zero</span>'
        else:
            direction = "positive" if self.lower > 0 else "negative"
            verdict = (
                '<span style="color:#27ae60">'
                f"Bounds exclude zero ({direction})</span>"
            )

        return (
            f'<div style="font-family:monospace; padding:12px; '
            f'border:1px solid #ddd; border-radius:6px; max-width:520px">'
            f'<h4 style="margin:0 0 8px 0">{self.method}</h4>'
            f'<table style="border-collapse:collapse">'
            f'<tr><td style="padding:2px 12px 2px 0">Lower bound</td>'
            f"<td><b>{self.lower:.4f}</b> (SE {self.se_lower:.4f})</td></tr>"
            f'<tr><td style="padding:2px 12px 2px 0">Upper bound</td>'
            f"<td><b>{self.upper:.4f}</b> (SE {self.se_upper:.4f})</td></tr>"
            f'<tr><td style="padding:2px 12px 2px 0">{pct}% CI</td>'
            f"<td>[{im_lo:.4f}, {im_hi:.4f}]</td></tr>"
            f"</table>"
            f'<p style="margin:8px 0 0 0">{verdict}</p>'
            f"</div>"
        )

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Interval plot showing the identified set with confidence intervals.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to draw on; created if *None*.
        **kwargs
            Passed to ``ax.errorbar``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3))

        # Identified set as thick bar; an open side (a one-sided set) is
        # drawn to the edge of the finite part of the picture
        finite = [v for v in (self.lower, self.upper) if np.isfinite(v)]
        span = max(1.0, *(abs(v) for v in finite)) if finite else 1.0
        left = self.lower if np.isfinite(self.lower) else min(finite) - span
        right = self.upper if np.isfinite(self.upper) else max(finite) + span
        ax.barh(
            0,
            right - left,
            left=left,
            height=0.3,
            color="#3498db",
            alpha=0.6,
            label="Identified set",
        )

        # CI whiskers
        ci_lo, ci_hi = self.parameter_ci()
        ci_lo = ci_lo if np.isfinite(ci_lo) else left
        ci_hi = ci_hi if np.isfinite(ci_hi) else right
        ax.plot([ci_lo, ci_hi], [0, 0], "k-", linewidth=1)
        ax.plot([ci_lo, ci_lo], [-0.1, 0.1], "k-", linewidth=1)
        ax.plot([ci_hi, ci_hi], [-0.1, 0.1], "k-", linewidth=1)

        # Zero line
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_yticks([])
        ax.set_xlabel("Treatment Effect")
        ax.set_title(self.method)
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        return ax

    def __repr__(self) -> str:
        return (
            f"BoundsResult(method='{self.method}', "
            f"lower={self.lower:.4f}, upper={self.upper:.4f})"
        )


# ======================================================================
# Helper: bootstrap utilities
# ======================================================================


def _bootstrap_bounds(
    compute_fn: Callable[..., tuple[float, float]],
    data: pd.DataFrame,
    n_boot: int,
    alpha: float,
    random_state: int,
    point: Tuple[float, float],
    label: str = "bounds bootstrap",
    **kwargs: Any,
) -> Tuple[float, float, Tuple[float, float], Tuple[float, float], int]:
    """Generic bootstrap for a function that returns (lower, upper).

    Failed replicates (exception or NaN bounds) are dropped from the
    SE/CI computation; a ``RuntimeWarning`` reports the failed fraction
    (same convention as ``core._bootstrap.bootstrap_se``) and the count
    is returned so callers can record it in ``model_info``.

    ``point`` holds the full-sample ``(lower, upper)``: each endpoint's
    interval is ``bound +/- z_{1-alpha/2} se``, centred on the estimate.
    Before 2026-10-27 it was centred on the median of the bootstrap draws,
    which moves with their skew. Fewer than two usable draws give NaN
    standard errors (they used to be reported as 0).

    Returns
    -------
    se_lower, se_upper, ci_lower_tuple, ci_upper_tuple, n_failed
    """
    rng = np.random.RandomState(random_state)
    n = len(data)
    boot_lb = np.zeros(n_boot)
    boot_ub = np.zeros(n_boot)

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        df_b = data.iloc[idx].reset_index(drop=True)
        try:
            boot_lb[b], boot_ub[b] = compute_fn(df_b, **kwargs)
        except Exception:
            boot_lb[b], boot_ub[b] = np.nan, np.nan

    n_failed = int(np.sum(np.isnan(boot_lb) | np.isnan(boot_ub)))
    if n_failed > 0:
        warnings.warn(
            f"{label}: {n_failed}/{n_boot} bootstrap replicates failed; "
            f"SE/CI computed over {n_boot - n_failed} successful "
            f"replicates.",
            RuntimeWarning,
            stacklevel=2,
        )

    z = sp_stats.norm.ppf(1 - alpha / 2)

    def _side(draws: np.ndarray, est: float) -> Tuple[float, Tuple[float, float]]:
        if not np.isfinite(est):  # an open side (e.g. a one-sided bound)
            return float("nan"), (est, est)
        draws = draws[np.isfinite(draws)]
        se = float(np.std(draws, ddof=1)) if len(draws) > 1 else float("nan")
        return se, (est - z * se, est + z * se)

    se_lb, ci_lb = _side(boot_lb, float(point[0]))
    se_ub, ci_ub = _side(boot_ub, float(point[1]))

    return se_lb, se_ub, ci_lb, ci_ub, n_failed


# ======================================================================
# 1. Horowitz-Manski Bounds with Covariates
# ======================================================================


@accepts_aliases(treat="treatment", seed="random_state")
def horowitz_manski(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    covariates: List[str],
    y_lower: Optional[float] = None,
    y_upper: Optional[float] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> BoundsResult:
    """
    Horowitz-Manski (2000) bounds conditioning on covariates.

    Averages the conditional worst-case (Manski) bounds over the
    covariate strata. For the ATE with a common outcome range
    ``[y_lower, y_upper]`` these bounds are linear in the cell means, so
    the average equals the unconditional worst-case bounds exactly
    (``sp.manski_bounds(assumption='none')``); conditioning tightens
    nothing unless the strata carry further restrictions.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treatment : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariates to condition on (discretised via quartiles for
        continuous variables).
    y_lower : float, optional
        Known lower bound of Y. Defaults to observed min.
    y_upper : float, optional
        Known upper bound of Y. Defaults to observed max.
    n_boot : int, default 500
        Bootstrap replications.
    alpha : float, default 0.05
    random_state : int, default 42

    Returns
    -------
    BoundsResult

    Notes
    -----
    Bootstrap replicates that fail are dropped from the SE/CI
    computation; a ``RuntimeWarning`` reports the failed fraction and
    ``model_info['n_boot_failed']`` records the count.

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> trained = rng.binomial(1, 0.5, n)
    >>> wage = 50 + 10 * trained + rng.normal(0, 15, n)
    >>> df = pd.DataFrame({"wage": wage, "trained": trained,
    ...                    "age": rng.normal(40, 10, n)})
    >>> result = sp.horowitz_manski(
    ...     data=df, y="wage", treatment="trained", covariates=["age"],
    ...     y_lower=float(df["wage"].min()), y_upper=float(df["wage"].max()),
    ... )
    >>> result.method
    'Horowitz-Manski Bounds (2000)'
    >>> bool(result.lower < 0 < result.upper)   # worst-case bounds straddle zero
    True
    """
    _check_cols(
        data,
        {"y": y, "treatment": treatment, "covariates": list(covariates)},
        "sp.horowitz_manski",
    )
    df = data.dropna(subset=[y, treatment] + covariates).copy()

    D_all = df[treatment].to_numpy(dtype=np.float64)
    if not np.isin(D_all, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"horowitz_manski: treatment {treatment!r} must be coded 0/1; "
            f"found {sorted(np.unique(D_all).tolist())[:6]}.",
            recovery_hint=f"Recode {treatment!r} to 0/1 (1 = treated).",
        )
    Y = df[y].values.astype(np.float64)

    if y_lower is None:
        y_lower = float(Y.min())
    if y_upper is None:
        y_upper = float(Y.max())

    # Discretise continuous covariates into quartile bins
    strata_col = _create_strata(df, covariates)

    lb, ub = _hm_point(df, y, treatment, strata_col, y_lower, y_upper)

    # Bootstrap
    def _compute(df_b: pd.DataFrame, **kw: Any) -> tuple[float, float]:
        sc = _create_strata(df_b, covariates)
        return _hm_point(df_b, y, treatment, sc, y_lower, y_upper)

    se_lb, se_ub, ci_lb, ci_ub, n_failed = _bootstrap_bounds(
        _compute,
        df,
        n_boot,
        alpha,
        random_state,
        point=(lb, ub),
        label="horowitz_manski",
    )

    return BoundsResult(
        lower=lb,
        upper=ub,
        se_lower=se_lb,
        se_upper=se_ub,
        ci_lower=ci_lb,
        ci_upper=ci_ub,
        method="Horowitz-Manski Bounds (2000)",
        alpha=alpha,
        n_obs=len(df),
        model_info={
            "y_lower": y_lower,
            "y_upper": y_upper,
            "n_strata": int(strata_col.nunique()),
            "covariates": covariates,
            "n_boot_failed": n_failed,
        },
    )


def _hm_point(
    df: pd.DataFrame,
    y: str,
    treatment: str,
    strata: pd.Series,
    y_lo: float,
    y_hi: float,
) -> tuple[float, float]:
    """Compute Horowitz-Manski conditional bounds."""
    Y = df[y].values.astype(np.float64)
    D = df[treatment].values.astype(np.float64)
    S = strata.values
    n = len(Y)

    lb_total = 0.0
    ub_total = 0.0

    for s in np.unique(S):
        mask = S == s
        w = mask.sum() / n  # P(X = s)

        Y_s = Y[mask]
        D_s = D[mask]

        Y1 = Y_s[D_s == 1]
        Y0 = Y_s[D_s == 0]

        p1 = np.mean(D_s)  # P(D=1 | X=s)
        p0 = 1 - p1  # P(D=0 | X=s)

        # A stratum with no treated (or no control) units still carries
        # its mass: the missing arm's conditional mean is simply
        # unrestricted in [y_lo, y_hi], which the formula below encodes
        # once that arm's (zero-weight) sample mean is set to 0. Before
        # 1.29 such strata were skipped, silently dropping their
        # P(X = s) share and shrinking both bounds toward zero.
        e1 = float(np.mean(Y1)) if len(Y1) else 0.0
        e0 = float(np.mean(Y0)) if len(Y0) else 0.0

        # Conditional Manski bounds for ATE at stratum s
        lb_s = e1 * p1 + y_lo * p0 - e0 * p0 - y_hi * p1
        ub_s = e1 * p1 + y_hi * p0 - e0 * p0 - y_lo * p1

        lb_total += w * lb_s
        ub_total += w * ub_s

    return float(lb_total), float(ub_total)


def _create_strata(df: pd.DataFrame, covariates: List[str]) -> pd.Series:
    """Create strata by discretising continuous covariates into quartiles."""
    parts = []
    for c in covariates:
        col = df[c]
        if col.dtype.kind in ("f",) or col.nunique() > 10:
            binned = pd.qcut(col, q=4, labels=False, duplicates="drop")
            # qcut yields NaN when the bin edges collapse (constant column)
            # or the input is missing. Bucket those rows into an explicit
            # sentinel stratum: under pandas>=3.0 astype(str) preserves NaN
            # as missing instead of stringifying to "nan", so without this
            # the masks in _hm_point match nothing and the bounds silently
            # collapse to 0.0.
            binned = pd.Series(binned, index=df.index)
            parts.append(binned.fillna(-1.0).astype(np.int64).astype(str))
        else:
            parts.append(col.fillna(-1).astype(str))
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + "_" + p
    return combined


# ======================================================================
# 2. IV Bounds (Nevo & Rosen 2012)
# ======================================================================


@accepts_aliases(covariates="controls", treat="treatment", seed="random_state")
def iv_bounds(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    instrument: str,
    controls: Optional[List[str]] = None,
    assumption: str = "monotone_iv",
    alpha: float = 0.05,
    n_boot: int = 500,
    random_state: int = 42,
    xu_sign: str = "positive",
) -> BoundsResult:
    """
    Nevo-Rosen (2012) bounds on a linear IV slope with an imperfect instrument.

    The model is ``y = beta x + W delta + u`` with one endogenous regressor
    ``x`` (``treatment``), exogenous ``controls`` W and one instrument ``z``
    that may be correlated with ``u``. Nevo and Rosen replace exogeneity of
    ``z`` with two assumptions:

    * A3 (same direction): ``corr(x, u)`` and ``corr(z, u)`` have the same
      sign;
    * A4 (less endogenous): ``|corr(z, u)| <= |corr(x, u)|``.

    Under A3 and A4, ``V(1) = sd(z) x - sd(x) z`` is a second instrument
    whose bias has the opposite sign, and the sharp identified set for
    ``beta`` has the IV estimates with ``z`` and with ``V(1)`` as endpoints
    (working paper, Proposition 2; Proposition 3 with controls). It is a
    closed interval when the two instruments bound ``beta`` from opposite
    sides, and a half-line otherwise (typically when ``z`` and ``x`` are
    positively correlated). Which side is open follows from the sign of
    ``corr(x, u)``, which the data cannot tell and ``xu_sign`` states.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treatment : str
        Endogenous regressor (continuous or binary).
    instrument : str
        Imperfect instrument (one variable).
    controls : list of str, optional
        Exogenous regressors W, partialled out of ``y``, ``x`` and ``z``
        (they serve as their own instruments).
    assumption : {'monotone_iv', 'less_than_late'}, default 'monotone_iv'
        - ``'monotone_iv'``: Nevo-Rosen A3 and A4, the sharp set above.
        - ``'less_than_late'``: deprecated. It returned
          ``Wald +/- |OLS - Wald|``, an interval with no source in the
          literature that can exclude the true slope under the assumption
          it was described with; it is kept, unchanged up to rounding, until
          the next minor release.
    alpha : float, default 0.05
    n_boot : int, default 500
    random_state : int, default 42
    xu_sign : {'positive', 'negative'}, default 'positive'
        The assumed sign of ``corr(x, u)``: ``'positive'`` when OLS is
        believed to overstate ``beta``. It decides which IV estimate is the
        upper and which the lower endpoint.

    Returns
    -------
    BoundsResult
        ``lower`` / ``upper`` (``-inf`` / ``inf`` for an open side).
        ``model_info`` holds ``beta_iv_z``, ``beta_iv_v1`` (the two
        endpoints), ``ols_estimate``, ``wald_estimate`` (= ``beta_iv_z``),
        ``closed`` and ``first_stage_f`` (the F statistic of ``z`` in the
        regression of ``x`` on ``z`` and the controls, homoskedastic).

    Notes
    -----
    Before 2026-10-27 ``'monotone_iv'`` returned ``[min(OLS, IV),
    max(OLS, IV)]``. When ``z`` and ``x`` are positively correlated both
    estimates are biased in the same direction and the true slope lies
    outside that interval: with ``corr(x, u) > 0`` it lies below both.
    ``first_stage_f`` was ``n corr(z, x)^2``.

    Bootstrap replicates that fail are dropped from the SE/CI
    computation; a ``RuntimeWarning`` reports the failed fraction and
    ``model_info['n_boot_failed']`` records the count. An open side has a
    NaN standard error.

    The formulas are those of the working paper (NBER w14434 and cemmap
    CWP16/08, Section 3, both read 2026-10-27); the published version is
    Review of Economics and Statistics 94(3), 659-671.

    References
    ----------
    nevo2012identification

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> z, v, e = rng.normal(size=(3, n))
    >>> u = 0.2 * z + 0.8 * v + 0.5 * e      # z is mildly endogenous
    >>> x = -z + v                            # negatively correlated with z
    >>> df = pd.DataFrame({"y": x + u, "x": x, "z": z})
    >>> result = sp.iv_bounds(data=df, y="y", treatment="x", instrument="z",
    ...                       n_boot=50)
    >>> result.method
    'Nevo-Rosen IV Bounds (monotone_iv)'
    >>> bool(result.lower <= 1.0 <= result.upper)     # true slope 1
    True
    """
    if assumption not in ("monotone_iv", "less_than_late"):
        raise MethodIncompatibility(
            f"iv_bounds: unknown assumption {assumption!r}.",
            recovery_hint="Use assumption='monotone_iv'.",
        )
    if xu_sign not in ("positive", "negative"):
        raise MethodIncompatibility(
            f"iv_bounds: xu_sign must be 'positive' or 'negative', got {xu_sign!r}.",
            recovery_hint="State the assumed sign of corr(treatment, error).",
        )
    if assumption == "less_than_late":
        warnings.warn(
            "iv_bounds(assumption='less_than_late') is deprecated and will be "
            "removed no earlier than the next minor release: the interval "
            "Wald +/- |OLS - Wald| has no source in the literature and can "
            "exclude the true slope. Use assumption='monotone_iv' (Nevo-Rosen "
            "A3 + A4). See MIGRATION.md.",
            DeprecationWarning,
            stacklevel=2,
        )
    controls = list(controls) if controls else []
    cols = [y, treatment, instrument] + controls
    _check_cols(
        data,
        {
            "y": y,
            "treatment": treatment,
            "instrument": instrument,
            "controls": controls or None,
        },
        "sp.iv_bounds",
    )

    df = data.dropna(subset=cols).reset_index(drop=True)
    sign = 1.0 if xu_sign == "positive" else -1.0

    def _arrays(d: pd.DataFrame) -> tuple:
        Y = d[y].to_numpy(dtype=np.float64)
        X = d[treatment].to_numpy(dtype=np.float64)
        Z = d[instrument].to_numpy(dtype=np.float64)
        W = np.column_stack(
            [np.ones(len(d))] + [d[c].to_numpy(dtype=np.float64) for c in controls]
        )
        return Y, X, Z, W

    def _compute(df_b: pd.DataFrame, **kw: Any) -> tuple[float, float]:
        out = _iv_bounds_point(*_arrays(df_b), assumption, sign)
        return out["lower"], out["upper"]

    point = _iv_bounds_point(*_arrays(df), assumption, sign)
    lb, ub = point["lower"], point["upper"]

    se_lb, se_ub, ci_lb, ci_ub, n_failed = _bootstrap_bounds(
        _compute,
        df,
        n_boot,
        alpha,
        random_state,
        point=(lb, ub),
        label="iv_bounds",
    )

    model_info = {
        "assumption": assumption,
        "xu_sign": xu_sign,
        "beta_iv_z": point["beta_z"],
        "beta_iv_v1": point["beta_v1"],
        "ols_estimate": point["ols"],
        "wald_estimate": point["beta_z"],
        "closed": point["closed"],
        "cov_zy": point["cov_zy"],
        "cov_zd": point["cov_zx"],
        "first_stage_f": point["first_stage_f"],
        "n_boot_failed": n_failed,
    }
    return BoundsResult(
        lower=lb,
        upper=ub,
        se_lower=se_lb,
        se_upper=se_ub,
        ci_lower=ci_lb,
        ci_upper=ci_ub,
        method=f"Nevo-Rosen IV Bounds ({assumption})",
        alpha=alpha,
        n_obs=len(df),
        model_info=model_info,
    )


def _iv_bounds_point(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    assumption: str,
    sign: float = 1.0,
) -> Dict[str, Any]:
    """Nevo-Rosen identified set for one sample.

    ``W`` holds the constant and the exogenous controls; ``Y`` and ``X`` are
    partialled on it (working paper eq. 3.10), and every covariance with a
    residualised variable equals the covariance of the raw variable with it,
    so ``z`` itself needs no residualising. ``V(1) = sd(z) x - sd(x) z`` uses
    the raw standard deviations, because A4 restricts the raw correlations
    with ``u``.
    """

    def _resid(v: np.ndarray) -> np.ndarray:
        r: np.ndarray = v - W @ np.linalg.lstsq(W, v, rcond=None)[0]
        return r

    Yt, Xt, Zt = _resid(Y), _resid(X), _resid(Z)
    n = len(Y)
    cov_zy = float(Zt @ Yt) / (n - 1)
    cov_zx = float(Zt @ Xt) / (n - 1)
    cov_xx = float(Xt @ Xt) / (n - 1)
    cov_xy = float(Xt @ Yt) / (n - 1)
    sd_x = float(np.std(X, ddof=1))
    sd_z = float(np.std(Z, ddof=1))
    cov_vx = sd_z * cov_xx - sd_x * cov_zx  # cov(V(1), X~)
    cov_vy = sd_z * cov_xy - sd_x * cov_zy
    scale = np.sqrt(abs(cov_xx) * float(Zt @ Zt) / (n - 1))
    if abs(cov_zx) <= 1e-12 * max(scale, 1e-300):
        raise MethodIncompatibility(
            "iv_bounds: the instrument is uncorrelated with the treatment "
            "(after the controls); the IV estimate is not defined.",
            recovery_hint="Use an instrument that moves the treatment.",
        )
    beta_z = cov_zy / cov_zx
    ols = cov_xy / cov_xx
    beta_v1 = cov_vy / cov_vx if abs(cov_vx) > 1e-12 * max(scale, 1e-300) else np.nan
    k = W.shape[1]
    r2 = cov_zx**2 / (cov_xx * float(Zt @ Zt) / (n - 1))
    first_stage_f = (n - k - 1) * r2 / (1.0 - r2) if r2 < 1 else float("inf")

    closed = False
    if assumption == "less_than_late":
        diff = abs(ols - beta_z)
        lb, ub = beta_z - diff, beta_z + diff
    else:
        # With sign = sign(corr(x, u)): beta_z - beta has the sign of
        # sign * cov(z, x~), beta_v1 - beta the sign of sign * cov(V1, x~)
        # (A3 / A4 sign the numerators). An endpoint whose bias is positive
        # is an upper bound, otherwise a lower bound.
        uppers, lowers = [], []
        for est, den in ((beta_z, cov_zx), (beta_v1, cov_vx)):
            if not np.isfinite(est):
                continue
            (uppers if sign * den > 0 else lowers).append(est)
        lb = max(lowers) if lowers else -np.inf
        ub = min(uppers) if uppers else np.inf
        closed = bool(lowers and uppers)
        if lb > ub:  # cannot happen in population; sampling noise only
            lb, ub = ub, lb
    return {
        "lower": float(lb),
        "upper": float(ub),
        "beta_z": float(beta_z),
        "beta_v1": float(beta_v1),
        "ols": float(ols),
        "closed": closed,
        "cov_zy": cov_zy,
        "cov_zx": cov_zx,
        "first_stage_f": float(first_stage_f),
    }


# ======================================================================
# 3. Oster Delta (2019) — Coefficient stability / identified set
# ======================================================================


@accepts_aliases(seed="random_state")
def oster_delta(
    data: pd.DataFrame,
    y: str,
    x_base: List[str],
    x_controls: List[str],
    r_max: float = 1.3,
    delta_range: Tuple[float, float] = (-2.0, 2.0),
    n_grid: int = 200,
    alpha: float = 0.05,
    n_boot: int = 500,
    random_state: int = 42,
) -> BoundsResult:
    """
    Oster (2019) coefficient stability bounds and delta* computation.

    Computes the identified set for the treatment coefficient beta under
    the assumption that selection on unobservables is proportional to
    selection on observables (delta), and R-squared would be at most
    ``r_max`` if all unobservables were included.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x_base : list of str
        Key treatment/variable(s) of interest.
    x_controls : list of str
        Additional controls whose inclusion tightens identification.
    r_max : float, default 1.3
        Maximum R-squared assumption. A value in ``(R_full, 1]`` is used as
        R_max itself. A value above 1 cannot be an R-squared and is read as a
        multiplier of the controlled R-squared, so the default 1.3 gives
        Oster's recommended ``R_max = min(1, 1.3 * R_full)``; ``<= 0`` means
        the same. A value in ``(0, R_full]`` falls back to that
        recommendation with a warning.
    delta_range : tuple, default (-2, 2)
        Range of proportional selection parameter delta.
    n_grid : int, default 200
        Grid points for delta in the identified set computation.
    alpha : float, default 0.05
    n_boot : int, default 500
    random_state : int, default 42

    Returns
    -------
    BoundsResult
        lower/upper give the identified set for beta at delta=1 (equal
        selection) and the given r_max. ``model_info['beta_star_delta1']``
        and ``model_info['delta_star']`` are Oster's *exact* solutions
        (the quadratic / cubic of her Proposition 2 as solved by her Stata
        ``psacalc``, which they reproduce to machine precision). The first
        entry of ``x_base`` is the treatment; further ``x_base`` entries
        enter both the short and the long regression (``psacalc``'s
        ``mcontrol()``).

    Notes
    -----
    Bootstrap replicates that fail are dropped from the SE/CI
    computation; a ``RuntimeWarning`` reports the failed fraction and
    ``model_info['n_boot_failed']`` records the count.

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> education = rng.normal(12, 3, n)
    >>> experience = rng.normal(10, 5, n)
    >>> tenure = rng.normal(5, 3, n)
    >>> wage = (2 * education + 1.5 * experience + 0.8 * tenure
    ...         + rng.normal(0, 5, n))
    >>> df = pd.DataFrame({"wage": wage, "education": education,
    ...                    "experience": experience, "tenure": tenure})
    >>> result = sp.oster_delta(
    ...     data=df, y="wage",
    ...     x_base=["education"],
    ...     x_controls=["experience", "tenure"],
    ...     r_max=1.3,
    ... )
    >>> result.method
    'Oster (2019) Coefficient Stability Bounds'
    >>> bool(result.lower <= result.upper)
    True
    """
    from ..diagnostics._oster import oster_beta_exact, oster_delta_exact, oster_inputs

    all_cols = [y] + x_base + x_controls
    _check_cols(
        data,
        {"y": y, "x_base": list(x_base), "x_controls": list(x_controls)},
        "sp.oster_delta",
    )
    df = data.dropna(subset=all_cols).copy()
    n = len(df)
    treat, mcontrol = x_base[0], list(x_base[1:])

    def _resolve_r_max(r2_full: float, warn: bool) -> float:
        # r_max <= 0: Oster's recommendation 1.3 x R_full. r_max > 1 cannot
        # be an R-squared and is read as a multiplier of R_full (so the
        # default 1.3 means min(1, 1.3 R_full), as documented). Until 1.28
        # a value above 1 was used as the R-squared itself.
        if r_max <= 0 or r_max > 1.0:
            mult = 1.3 if r_max <= 0 else r_max
            return min(mult * r2_full, 1.0)
        if r_max <= r2_full:
            if warn:
                warnings.warn(
                    f"oster_delta: r_max={r_max:.6g} does not exceed the "
                    f"controlled R-squared {r2_full:.6g}; using "
                    f"min(1, 1.3 x R_full) instead.",
                    UserWarning,
                    stacklevel=3,
                )
            return min(1.3 * r2_full, 1.0)
        return float(r_max)

    inp = oster_inputs(df, y, treat, list(x_controls), mcontrol)
    beta_short, r2_short = inp["beta_o"], inp["r_o"]
    beta_full, r2_full = inp["beta_t"], inp["r_t"]
    r_max_used = _resolve_r_max(r2_full, warn=True)

    # Exact bias-adjusted coefficient at delta = 1 (Oster 2019; psacalc)
    beta_star = float(
        oster_beta_exact(inp, r_max_used, 1.0)["beta"]  # type: ignore[arg-type]
    )

    # Identified set: [min(beta_full, beta_star), max(beta_full, beta_star)]
    lb = min(beta_full, beta_star)
    ub = max(beta_full, beta_star)

    # delta* such that beta* = 0 (exact)
    delta_star = oster_delta_exact(inp, r_max_used, beta=0.0)

    # Grid of (delta, beta_star) for plotting (exact roots; delta = 0 gives
    # beta_full)
    deltas = np.linspace(delta_range[0], delta_range[1], n_grid)
    betas_grid = np.array(
        [
            (
                float(
                    oster_beta_exact(  # type: ignore[arg-type]
                        inp, r_max_used, float(d)
                    )["beta"]
                )
                if d != 0
                else beta_full
            )
            for d in deltas
        ]
    )

    # Bootstrap
    def _compute(df_b: pd.DataFrame, **kw: Any) -> tuple[float, float]:
        inp_b = oster_inputs(df_b, y, treat, list(x_controls), mcontrol)
        rm = _resolve_r_max(inp_b["r_t"], warn=False)
        bs_star = float(
            oster_beta_exact(inp_b, rm, 1.0)["beta"]  # type: ignore[arg-type]
        )
        return (
            float(min(inp_b["beta_t"], bs_star)),
            float(max(inp_b["beta_t"], bs_star)),
        )

    se_lb, se_ub, ci_lb, ci_ub, n_failed = _bootstrap_bounds(
        _compute,
        df,
        n_boot,
        alpha,
        random_state,
        point=(lb, ub),
        label="oster_delta",
    )
    r_max = r_max_used

    return BoundsResult(
        lower=float(lb),
        upper=float(ub),
        se_lower=se_lb,
        se_upper=se_ub,
        ci_lower=ci_lb,
        ci_upper=ci_ub,
        method="Oster (2019) Coefficient Stability Bounds",
        alpha=alpha,
        n_obs=n,
        model_info={
            "beta_short": float(beta_short),
            "beta_full": float(beta_full),
            "r2_short": float(r2_short),
            "r2_full": float(r2_full),
            "r_max": float(r_max),
            "beta_star_delta1": float(beta_star),
            "delta_star": float(delta_star) if np.isfinite(delta_star) else None,
            "delta_grid": deltas.tolist(),
            "beta_grid": betas_grid.tolist(),
            "n_boot_failed": n_failed,
        },
    )


# ======================================================================
# 4. Selection Bounds — Lee (2009) with covariates
# ======================================================================


@accepts_aliases(treat="treatment", seed="random_state")
def selection_bounds(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    selection: str,
    covariates: Optional[List[str]] = None,
    method: str = "conditional",
    n_boot: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> BoundsResult:
    """
    Lee (2009) bounds for ATE under sample selection, optionally
    conditioning on covariates for tighter bounds.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable (may have NaN when selection=0).
    treatment : str
        Binary treatment (0/1).
    selection : str
        Binary indicator: 1 = outcome observed, 0 = missing.
    covariates : list of str, optional
        Covariates to condition on for tighter (conditional) bounds.
    method : {'conditional', 'unconditional'}, default 'conditional'
        - ``'conditional'``: Lee's (2009) tightened bounds. Each covariate
          stratum is trimmed by its own share and the trimmed means are
          averaged over the covariate distribution of the always-observed
          (the selected units of the less-retained arm). Continuous
          covariates (float, or more than 10 values) are binned into
          quartiles first. Equals
          ``sp.lee_bounds(covariates=<strata>, trimming='quantile')``.
        - ``'unconditional'``: Lee bounds ignoring covariates, equal to
          ``sp.lee_bounds(trimming='quantile')``.
    n_boot : int, default 500
    alpha : float, default 0.05
    random_state : int, default 42

    Returns
    -------
    BoundsResult

    Notes
    -----
    Since 2026-10-27 both methods run the ``sp.lee_bounds`` code. Before,
    the unconditional bounds kept ``floor((1 - p) n)`` observations (one
    fewer than Lee's quantile rule when ``p n`` is not whole, the rule
    ``sp.lee_bounds`` dropped in 1.29), and the conditional bounds
    weighted strata by their share of the whole sample and dropped
    strata missing an arm, instead of weighting by the always-observed.
    Rows with a missing treatment / selection indicator, or selected
    with a missing outcome, are dropped with a warning, as in
    ``sp.lee_bounds``.

    Bootstrap replicates that fail (or yield NaN bounds, e.g. a stratum
    losing all treated or selected units in a resample) are dropped from
    the SE/CI computation; a ``RuntimeWarning`` reports the failed
    fraction and ``model_info['n_boot_failed']`` records the count.

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> trained = rng.binomial(1, 0.5, n)
    >>> employed = rng.binomial(1, 0.8, n)
    >>> wage = 50 + 10 * trained + rng.normal(0, 10, n)
    >>> df = pd.DataFrame({"wage": np.where(employed == 1, wage, np.nan),
    ...                    "trained": trained, "employed": employed,
    ...                    "age": rng.normal(40, 10, n)})
    >>> result = sp.selection_bounds(
    ...     data=df, y="wage", treatment="trained",
    ...     selection="employed",
    ...     covariates=["age"],
    ...     method="conditional",
    ... )
    >>> result.method
    'Lee (2009) Bounds (conditional)'
    >>> bool(result.lower <= result.upper)
    True
    """
    if method not in ("conditional", "unconditional"):
        raise MethodIncompatibility(
            f"selection_bounds: method must be 'conditional' or "
            f"'unconditional', got {method!r}.",
            recovery_hint="Use method='conditional' with covariates=, or "
            "method='unconditional'.",
        )
    _check_cols(
        data,
        {
            "y": y,
            "treatment": treatment,
            "selection": selection,
            "covariates": list(covariates) if covariates else None,
        },
        "sp.selection_bounds",
    )
    # One implementation of Lee (2009): the estimation sample, the trimming
    # rule and the tightened (cell) bounds are those of sp.lee_bounds with
    # trimming='quantile'.
    from ._lee_sample import lee_estimation_sample

    covariates = list(covariates) if covariates else []
    df, sel_col = lee_estimation_sample(data, y, treatment, selection)
    if covariates:
        n_before = len(df)
        df = df.dropna(subset=covariates)
        if len(df) < n_before:
            warnings.warn(
                f"selection_bounds: dropped {n_before - len(df)} row(s) with "
                "missing covariates.",
                UserWarning,
                stacklevel=2,
            )
    df = df.reset_index(drop=True)
    conditional = method == "conditional" and bool(covariates)

    def _compute(df_b: pd.DataFrame, **kw: Any) -> tuple[float, float]:
        return _selection_lee_point(
            df_b, y, treatment, sel_col, covariates, conditional
        )

    lb, ub = _compute(df)
    D = df[treatment].to_numpy(dtype=float)
    S = df[sel_col].to_numpy(dtype=float)
    p1 = float(np.mean(S[D == 1]))
    p0 = float(np.mean(S[D == 0]))

    se_lb, se_ub, ci_lb, ci_ub, n_failed = _bootstrap_bounds(
        _compute,
        df,
        n_boot,
        alpha,
        random_state,
        point=(lb, ub),
        label="selection_bounds",
    )

    return BoundsResult(
        lower=lb,
        upper=ub,
        se_lower=se_lb,
        se_upper=se_ub,
        ci_lower=ci_lb,
        ci_upper=ci_ub,
        method=f"Lee (2009) Bounds ({method})",
        alpha=alpha,
        n_obs=int(np.sum(S)),
        model_info={
            "method": method,
            "selection_rate_treated": float(p1),
            "selection_rate_control": float(p0),
            "trimming_fraction": (
                float(abs(p1 - p0) / max(p1, p0)) if max(p1, p0) > 0 else 0.0
            ),
            "covariates": covariates,
            "trimming": "quantile",
            "n_boot_failed": n_failed,
        },
    )


def _selection_lee_point(
    df: pd.DataFrame,
    y: str,
    treatment: str,
    selection: str,
    covariates: List[str],
    conditional: bool,
) -> tuple[float, float]:
    """Lee bounds through the ``sp.lee_bounds`` primitives (trimming='quantile').

    Unconditional: ``lee_manski._compute_lee_bounds``. Conditional: Lee's
    tightened bounds (``_lee_tight.tightened_lee_bounds``) on the strata of
    ``_create_strata`` (continuous covariates binned into quartiles), i.e.
    ``sp.lee_bounds(covariates=<strata>, trimming='quantile')``. With equal
    retention in the two arms the effect is point identified and both
    branches return the difference in means.
    """
    from ._lee_tight import tightened_lee_bounds
    from .lee_manski import _compute_lee_bounds

    D = df[treatment].to_numpy(dtype=float)
    S = df[selection].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    p1 = float(np.mean(S[D == 1]))
    p0 = float(np.mean(S[D == 0]))
    if p1 == 0 or p0 == 0:
        raise ValueError("One treatment group has zero retention.")
    if conditional and p1 != p0:
        cells = _create_strata(df, covariates).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tight = tightened_lee_bounds(Y, D, S, cells, "quantile")
        return float(tight["lower"]), float(tight["upper"])
    obs = S == 1
    Y1 = Y[obs & (D == 1)]
    Y0 = Y[obs & (D == 0)]
    return _compute_lee_bounds(Y1, Y0, p1, p0, "quantile")


# ======================================================================
# 5. Breakdown Frontier (Masten & Poirier 2021)
# ======================================================================


def breakdown_frontier(
    estimate: float,
    se: float,
    assumption: str = "parallel_trends",
    max_violation: float = 0.1,
    n_grid: int = 100,
    alpha: float = 0.05,
) -> BoundsResult:
    """
    Masten-Poirier (2021) breakdown frontier for qualitative conclusions.

    For a given point estimate and standard error, computes how much an
    identifying assumption can be violated before a qualitative conclusion
    (e.g., "positive treatment effect") breaks down.

    Parameters
    ----------
    estimate : float
        Point estimate of the treatment effect.
    se : float
        Standard error of the estimate.
    assumption : str, default 'parallel_trends'
        Label for the identifying assumption being relaxed.  Currently
        supports a generic linear violation model applicable to
        ``'parallel_trends'``, ``'exclusion_restriction'``, or
        ``'selection_on_observables'``.
    max_violation : float, default 0.1
        Maximum magnitude of the assumption violation to explore.
    n_grid : int, default 100
        Grid resolution for the frontier.
    alpha : float, default 0.05

    Returns
    -------
    BoundsResult
        lower/upper give the identified set at the maximum violation.
        ``model_info`` contains ``'breakdown_point'`` (the violation
        magnitude at which the conclusion reverses) and
        ``'frontier_grid'`` / ``'frontier_bounds'`` for plotting.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.breakdown_frontier(
    ...     estimate=0.05, se=0.02,
    ...     assumption="parallel_trends",
    ...     max_violation=0.1,
    ... )
    >>> round(result.model_info["breakdown_point"], 4)
    0.05
    >>> (round(result.lower, 4), round(result.upper, 4))
    (-0.05, 0.15)
    >>> ax = result.plot()
    """
    if se <= 0:
        raise ValueError("Standard error must be positive.")

    # Breakdown point: the violation c such that sign could flip
    # Under linear violation: identified set = [estimate - c, estimate + c]
    # Conclusion reverses when estimate - c <= 0 (if estimate > 0)
    # => breakdown point c* = |estimate|
    breakdown_point = abs(estimate)

    # Grid of violations
    violations = np.linspace(0, max_violation, n_grid)
    lower_bounds = estimate - violations
    upper_bounds = estimate + violations

    # At max_violation
    lb = float(estimate - max_violation)
    ub = float(estimate + max_violation)

    # SE of the bounds = se (violation is deterministic, not estimated)
    z = sp_stats.norm.ppf(1 - alpha / 2)

    # Breakdown point with CI: includes statistical uncertainty
    # Conclusion breaks down when lower CI crosses zero
    breakdown_ci = max(abs(estimate) - z * se, 0)

    return BoundsResult(
        lower=lb,
        upper=ub,
        se_lower=se,
        se_upper=se,
        ci_lower=(lb - z * se, lb + z * se),
        ci_upper=(ub - z * se, ub + z * se),
        method=f"Breakdown Frontier ({assumption})",
        alpha=alpha,
        n_obs=0,
        model_info={
            "estimate": estimate,
            "se": se,
            "assumption": assumption,
            "breakdown_point": float(breakdown_point),
            "breakdown_point_ci": float(breakdown_ci),
            "max_violation": max_violation,
            "frontier_grid": violations.tolist(),
            "frontier_lower": lower_bounds.tolist(),
            "frontier_upper": upper_bounds.tolist(),
            "conclusion_at_zero_violation": (
                "positive" if estimate > 0 else "negative"
            ),
            "robust_at_max_violation": bool(lb > 0) if estimate > 0 else bool(ub < 0),
        },
    )


# ======================================================================
# Utilities
# ======================================================================


def _check_cols(data: pd.DataFrame, cols: Dict[str, Any], function: str) -> None:
    """Raise :class:`ColumnNotFound` if any named column is absent."""
    check_columns(data, cols, function=function)


# ======================================================================
# Citations
# ======================================================================

CausalResult._CITATIONS["horowitz_manski"] = (
    "@article{horowitz2000nonparametric,\n"
    "  title={Nonparametric Analysis of Randomized Experiments with "
    "Missing Covariate and Outcome Data},\n"
    "  author={Horowitz, Joel L and Manski, Charles F},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={95},\n"
    "  number={449},\n"
    "  pages={77--84},\n"
    "  year={2000}\n"
    "}"
)

CausalResult._CITATIONS["nevo_rosen"] = (
    "@article{nevo2012identification,\n"
    "  title={Identification with Imperfect Instruments},\n"
    "  author={Nevo, Aviv and Rosen, Adam M},\n"
    "  journal={Review of Economics and Statistics},\n"
    "  volume={94},\n"
    "  number={3},\n"
    "  pages={659--671},\n"
    "  year={2012},\n"
    "  doi={10.1162/REST_a_00171}\n"
    "}"
)

CausalResult._CITATIONS["oster_delta"] = (
    "@article{oster2019unobservable,\n"
    "  title={Unobservable Selection and Coefficient Stability: "
    "Theory and Evidence},\n"
    "  author={Oster, Emily},\n"
    "  journal={Journal of Business \\& Economic Statistics},\n"
    "  volume={37},\n"
    "  number={2},\n"
    "  pages={187--204},\n"
    "  year={2019}\n"
    "}"
)

CausalResult._CITATIONS["selection_bounds"] = (
    "@article{lee2009training,\n"
    "  title={Training, Wages, and Sample Selection: Estimating Sharp "
    "Bounds on Treatment Effects},\n"
    "  author={Lee, David S},\n"
    "  journal={The Review of Economic Studies},\n"
    "  volume={76},\n"
    "  number={3},\n"
    "  pages={1071--1102},\n"
    "  year={2009}\n"
    "}"
)

CausalResult._CITATIONS["breakdown_frontier"] = (
    "@article{masten2021salvaging,\n"
    "  title={Salvaging Falsified Instrumental Variable Models},\n"
    "  author={Masten, Matthew A and Poirier, Alexandre},\n"
    "  journal={Econometrica},\n"
    "  volume={89},\n"
    "  number={3},\n"
    "  pages={1449--1469},\n"
    "  year={2021}\n"
    "}"
)
