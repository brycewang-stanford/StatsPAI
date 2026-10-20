"""
MR diagnostics: heterogeneity, pleiotropy, directionality, leave-one-out.

These are the standard MR sanity checks that must accompany every
IVW or Egger estimate in a credible MR analysis.  Implementations
track the R ``MendelianRandomization``/``TwoSampleMR`` packages.

References
----------
Bowden, J. et al. (2017). "Assessing the suitability of summary data
for two-sample Mendelian randomization analyses using MR-Egger
regression: the role of the I2 statistic." *IJE*, 45(6), 1961-1974. [@bowden2016assessing]

Hemani, G. et al. (2017). "Orienting the causal relationship between
imprecisely measured traits using GWAS summary data." *PLoS Genetics*,
13(11). [@hemani2017orienting]

Bowden, J. et al. (2018). "Improving the visualization, interpretation
and analysis of two-sample summary data Mendelian randomization via the
Radial plot and Radial regression." *IJE*, 47(4), 1264-1278. [@bowden2018improving]

Verbanck, M. et al. (2018). "Detection of widespread horizontal
pleiotropy in causal relationships inferred from Mendelian
randomization between complex traits and diseases." *Nature Genetics*,
50(5), 693-698. [@verbanck2018detection]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin

__all__ = [
    "HeterogeneityResult",
    "PleiotropyResult",
    "LeaveOneOutResult",
    "SteigerResult",
    "MRPressoResult",
    "RadialResult",
    "mr_heterogeneity",
    "mr_pleiotropy_egger",
    "mr_leave_one_out",
    "mr_steiger",
    "mr_presso",
    "mr_radial",
]


SampleSizeInput = Union[int, float, Sequence[float], np.ndarray]


# --------------------------------------------------------------------------- #
#  Cochran's Q / Rücker's Q'
# --------------------------------------------------------------------------- #


@dataclass
class HeterogeneityResult(ResultProtocolMixin):
    """Container for Cochran's Q / Rücker's Q' heterogeneity diagnostics.

    Returned by :func:`mr_heterogeneity`.  Holds the Q statistic, its
    degrees of freedom and p-value, the ``I^2`` percentage, and the
    fitting ``method`` (``"ivw"`` or ``"egger"``).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> bx = rng.uniform(0.2, 0.5, n)
    >>> by = 0.3 * bx + rng.normal(0, 0.02, n)
    >>> sy = rng.uniform(0.02, 0.05, n)
    >>> het = sp.mr_heterogeneity(bx, by, sy)
    >>> isinstance(het, sp.HeterogeneityResult)
    True
    >>> het.method
    'ivw'
    >>> bool(het.Q >= 0.0)
    True
    """

    _citation_keys = (
        "bowden2016assessing",
        "hemani2017orienting",
        "bowden2018improving",
        "verbanck2018detection",
    )

    Q: float
    Q_df: int
    Q_p: float
    I2: float
    method: str  # "ivw" | "egger"

    def summary(self) -> str:
        return (
            f"{self.method.upper()} heterogeneity\n"
            f"  Cochran Q = {self.Q:.3f} (df = {self.Q_df}), p = {self.Q_p:.4g}\n"
            f"  I^2 = {self.I2:.1f}%"
        )


def mr_heterogeneity(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_outcome: np.ndarray,
    *,
    method: str = "ivw",
    se_exposure: Optional[np.ndarray] = None,
) -> HeterogeneityResult:
    """Cochran's Q (IVW) or Rücker's Q' (Egger) for pleiotropic heterogeneity.

    Parameters
    ----------
    beta_exposure, beta_outcome : array-like
    se_outcome : array-like
    method : {"ivw", "egger"}
    se_exposure : array-like, optional
        Required for Rücker's Q' (Egger); if omitted we default to
        first-order weights and raise for ``method='egger'``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> beta_exposure = rng.uniform(0.2, 0.5, n)
    >>> beta_outcome = 0.3 * beta_exposure + rng.normal(0, 0.02, n)
    >>> se_outcome = rng.uniform(0.02, 0.05, n)
    >>> het = sp.mr_heterogeneity(beta_exposure, beta_outcome, se_outcome)
    >>> het.method
    'ivw'
    >>> het.Q_df  # n - 1 for IVW
    11
    >>> bool(0.0 <= het.I2 <= 100.0)
    True

    References
    ----------
    [@bowden2016assessing]
    """
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    w = 1.0 / sy**2

    if method == "ivw":
        # Fit IVW through origin; compute residuals weighted by 1/sy^2
        beta_ivw = np.sum(w * bx * by) / np.sum(w * bx**2)
        Q = float(np.sum(w * (by - beta_ivw * bx) ** 2))
        df = len(bx) - 1
    elif method == "egger":
        # Rücker's Q' uses the Egger-fitted intercept+slope, on variants
        # oriented to positive exposure association like the fit itself --
        # the residuals, and so Q', change with allele coding otherwise.
        flip = bx < 0
        bx = np.where(flip, -bx, bx)
        by = np.where(flip, -by, by)
        X = np.column_stack([np.ones(len(bx)), bx])
        W = np.diag(w)
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ by)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ by
        resid = by - X @ beta
        Q = float(np.sum(w * resid**2))
        df = len(bx) - 2
    else:
        raise ValueError("method must be 'ivw' or 'egger'")

    df = max(df, 1)
    p = float(stats.chi2.sf(Q, df))
    I2 = float(max(0.0, (Q - df) / Q * 100.0)) if Q > 0 else 0.0
    _result = HeterogeneityResult(Q=Q, Q_df=df, Q_p=p, I2=I2, method=method)
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.mendelian.mr_heterogeneity",
            params={
                "method": method,
                "n_snps": int(len(beta_exposure)),
            },
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# --------------------------------------------------------------------------- #
#  MR-Egger intercept test (directional pleiotropy)
# --------------------------------------------------------------------------- #


@dataclass
class PleiotropyResult(ResultProtocolMixin):
    """Container for the MR-Egger intercept (directional-pleiotropy) test.

    Returned by :func:`mr_pleiotropy_egger`.  Holds the fitted Egger
    ``intercept``, its standard error ``se``, and the two-sided
    ``p_value`` for H0: intercept == 0.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> bx = rng.uniform(0.2, 0.5, n)
    >>> by = 0.3 * bx + rng.normal(0, 0.02, n)
    >>> sy = rng.uniform(0.02, 0.05, n)
    >>> pleio = sp.mr_pleiotropy_egger(bx, by, sy)
    >>> isinstance(pleio, sp.PleiotropyResult)
    True
    >>> bool(0.0 <= pleio.p_value <= 1.0)
    True
    """

    _citation_keys = (
        "bowden2016assessing",
        "hemani2017orienting",
        "bowden2018improving",
        "verbanck2018detection",
    )

    intercept: float
    se: float
    p_value: float

    def summary(self) -> str:
        direction = (
            "directional pleiotropy"
            if self.p_value < 0.05
            else "no evidence of directional pleiotropy"
        )
        return (
            "MR-Egger Intercept Test\n"
            f"  Intercept = {self.intercept:+.4f}   SE = {self.se:.4f}   "
            f"p = {self.p_value:.4g}\n"
            f"  Interpretation: {direction}"
        )


def mr_pleiotropy_egger(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_outcome: np.ndarray,
) -> PleiotropyResult:
    """Test the MR-Egger intercept for directional (unbalanced) pleiotropy.

    H0: intercept == 0 (no directional pleiotropy).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> beta_exposure = rng.uniform(0.2, 0.5, n)
    >>> beta_outcome = 0.3 * beta_exposure + rng.normal(0, 0.02, n)
    >>> se_outcome = rng.uniform(0.02, 0.05, n)
    >>> pleio = sp.mr_pleiotropy_egger(beta_exposure, beta_outcome, se_outcome)
    >>> bool(pleio.se >= 0.0)
    True
    >>> bool(0.0 <= pleio.p_value <= 1.0)
    True
    """
    # Delegate to mr_egger, the single implementation of the fit: variants
    # oriented to positive exposure association, residual SE floored at 1.
    # This function carried its own copy of the regression without either,
    # so the intercept it tested depended on allele coding -- 38% off on
    # MendelianRandomization's LDL-C / CHD example, 16 of whose 28 variants
    # have negative exposure associations.
    from .mr import mr_egger

    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    n = len(bx)
    eg = mr_egger(bx, by, np.zeros_like(bx), sy)
    intercept = float(eg["intercept"])
    intercept_se = float(eg["intercept_se"])
    if intercept_se > 0:
        t_stat = intercept / intercept_se
        # t(n-2) because sigma^2 is plug-in estimated, as in TwoSampleMR.
        df = max(n - 2, 1)
        p = float(2 * stats.t.sf(abs(t_stat), df=df))
    else:
        p = float("nan")
    return PleiotropyResult(intercept=intercept, se=intercept_se, p_value=p)


# --------------------------------------------------------------------------- #
#  Leave-one-out
# --------------------------------------------------------------------------- #


@dataclass
class LeaveOneOutResult(ResultProtocolMixin):
    """Container for leave-one-out IVW estimates.

    Returned by :func:`mr_leave_one_out`.  Wraps a :class:`pandas.DataFrame`
    ``table`` with one row per dropped SNP and columns ``dropped_snp``,
    ``estimate``, ``se``, ``ci_lower``, ``ci_upper`` and ``p_value``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> bx = rng.uniform(0.2, 0.5, n)
    >>> by = 0.3 * bx + rng.normal(0, 0.02, n)
    >>> sy = rng.uniform(0.02, 0.05, n)
    >>> loo = sp.mr_leave_one_out(bx, by, sy)
    >>> isinstance(loo, sp.LeaveOneOutResult)
    True
    >>> len(loo.table)
    12
    """

    _citation_keys = (
        "bowden2016assessing",
        "hemani2017orienting",
        "bowden2018improving",
        "verbanck2018detection",
    )

    table: pd.DataFrame  # columns: dropped_snp, estimate, se, ci_lower, ci_upper, p

    def summary(self) -> str:
        lines = [
            "Leave-One-Out Analysis (IVW)",
            "-" * 60,
            f"{'Dropped':<20s} {'Estimate':>10s} {'SE':>10s} {'p':>10s}",
        ]
        for _, row in self.table.iterrows():
            lines.append(
                f"{str(row['dropped_snp']):<20s} {row['estimate']:>10.4f} "
                f"{row['se']:>10.4f} {row['p_value']:>10.4g}"
            )
        return "\n".join(lines)


def mr_leave_one_out(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_outcome: np.ndarray,
    *,
    snp_ids: Optional[List[str]] = None,
    alpha: float = 0.05,
    model: str = "default",
) -> LeaveOneOutResult:
    """IVW estimate with each SNP dropped in turn.

    Identifies SNPs that drive the overall estimate disproportionately.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> beta_exposure = rng.uniform(0.2, 0.5, n)
    >>> beta_outcome = 0.3 * beta_exposure + rng.normal(0, 0.02, n)
    >>> se_outcome = rng.uniform(0.02, 0.05, n)
    >>> loo = sp.mr_leave_one_out(beta_exposure, beta_outcome, se_outcome)
    >>> len(loo.table)  # one row per dropped SNP
    12
    >>> sorted(loo.table.columns)
    ['ci_lower', 'ci_upper', 'dropped_snp', 'estimate', 'p_value', 'se']
    """
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    n = len(bx)
    if snp_ids is None:
        snp_ids = [f"SNP_{i}" for i in range(n)]
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Each fit goes through mr_ivw, so a leave-one-out row is exactly the IVW
    # estimate on the remaining variants under the same `model` rule
    # (random effects with more than three variants by default). This loop
    # used to recompute the fixed-effect standard error inline, so its rows
    # were not the IVW results they were labelled as once mr_ivw followed
    # MendelianRandomization's default -- 49% off on the LDL-C example.
    from .mr import mr_ivw

    rows = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        fit = mr_ivw(
            bx[mask], by[mask], np.zeros(mask.sum()), sy[mask], alpha=alpha, model=model
        )
        beta = float(fit["estimate"])
        se = float(fit["se"])
        z = beta / se if se > 0 else 0.0
        p = float(2 * stats.norm.sf(abs(z)))
        rows.append(
            dict(
                dropped_snp=snp_ids[i],
                estimate=beta,
                se=se,
                ci_lower=beta - z_crit * se,
                ci_upper=beta + z_crit * se,
                p_value=p,
            )
        )
    return LeaveOneOutResult(table=pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
#  Steiger directionality test
# --------------------------------------------------------------------------- #


@dataclass
class SteigerResult(ResultProtocolMixin):
    """Container for the Steiger directionality test.

    Returned by :func:`mr_steiger`.  Reports whether the assumed causal
    direction is supported (``correct_direction``), the one-sided
    ``steiger_pvalue``, the variance explained on each trait
    (``r2_exposure``, ``r2_outcome``), and the effective sample sizes.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> bx = rng.uniform(0.2, 0.5, n)
    >>> by = 0.3 * bx + rng.normal(0, 0.02, n)
    >>> sx = rng.uniform(0.02, 0.05, n)
    >>> sy = rng.uniform(0.02, 0.05, n)
    >>> st = sp.mr_steiger(bx, sx, 50000, by, sy, 50000)
    >>> isinstance(st, sp.SteigerResult)
    True
    >>> isinstance(st.correct_direction, bool)
    True
    """

    _citation_keys = (
        "bowden2016assessing",
        "hemani2017orienting",
        "bowden2018improving",
        "verbanck2018detection",
    )

    correct_direction: bool
    steiger_pvalue: float
    r2_exposure: float
    r2_outcome: float
    sample_size_exposure: int
    sample_size_outcome: int

    def summary(self) -> str:
        dir_str = (
            "exposure -> outcome (expected)"
            if self.correct_direction
            else "outcome -> exposure (reverse!)"
        )
        return (
            "Steiger Directionality Test\n"
            f"  R^2 (on exposure) = {self.r2_exposure:.4f}\n"
            f"  R^2 (on outcome)  = {self.r2_outcome:.4f}\n"
            f"  Direction: {dir_str}\n"
            f"  Steiger p = {self.steiger_pvalue:.4g}"
        )


def _r2_from_beta_se(
    beta: np.ndarray,
    se: np.ndarray,
    n: SampleSizeInput,
    eaf: Optional[np.ndarray] = None,
) -> float:
    """Approximate R^2 contributed by a SNP to a trait from GWAS summary.

    Uses the standard two-term approximation:
        R^2 ~ 2 * beta^2 * EAF * (1-EAF) / Var(Y)
    assuming Var(Y) = 1 + beta^2 * 2*EAF*(1-EAF) (per-SNP small).
    Falls back to the t-statistic-based approximation when EAF missing.
    """
    beta = np.asarray(beta, dtype=float)
    se = np.asarray(se, dtype=float)
    n = np.asarray(n, dtype=float)
    if eaf is None:
        # t^2 / (t^2 + n - 2) per SNP
        t2 = (beta / se) ** 2
        r2 = t2 / (t2 + n - 2)
    else:
        eaf = np.asarray(eaf, dtype=float)
        r2 = 2 * beta**2 * eaf * (1 - eaf)
    # Total R^2 across independent SNPs: bounded at 1
    return float(min(1.0, np.sum(r2)))


def mr_steiger(
    beta_exposure: np.ndarray,
    se_exposure: np.ndarray,
    n_exposure: SampleSizeInput,
    beta_outcome: np.ndarray,
    se_outcome: np.ndarray,
    n_outcome: SampleSizeInput,
    *,
    eaf: Optional[np.ndarray] = None,
    alternative: str = "two-sided",
) -> SteigerResult:
    """Steiger test for the direction of the causal effect (Hemani 2017).

    Compares the R^2 of the SNPs on the exposure with their R^2 on the
    outcome.  If R^2_exposure > R^2_outcome the assumed causal
    direction (exposure -> outcome) is supported. Matches
    ``TwoSampleMR::mr_steiger`` with ``r`` from ``get_r_from_bsen``.

    .. versionchanged:: 1.28.0
       The p-value was one-sided and computed as ``1 - Phi(z)``, which is
       exactly 0.0 for z beyond ~8.3. It is now two-sided by default, as
       the method authors' TwoSampleMR reports it, computed from the upper
       tail directly; pass ``alternative="greater"`` for the one-sided test.

    Parameters
    ----------
    alternative : {"two-sided", "greater"}, default "two-sided"
        ``"two-sided"`` is the reference's p-value; ``"greater"`` tests the
        one-sided hypothesis that the exposure R^2 is the larger.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> beta_exposure = rng.uniform(0.2, 0.5, n)
    >>> beta_outcome = 0.3 * beta_exposure + rng.normal(0, 0.02, n)
    >>> se_exposure = rng.uniform(0.02, 0.05, n)
    >>> se_outcome = rng.uniform(0.02, 0.05, n)
    >>> st = sp.mr_steiger(
    ...     beta_exposure, se_exposure, 50000,
    ...     beta_outcome, se_outcome, 50000,
    ... )
    >>> isinstance(st.correct_direction, bool)
    True
    >>> bool(st.r2_exposure >= 0.0 and st.r2_outcome >= 0.0)
    True

    References
    ----------
    [@hemani2017orienting]
    """
    bx = np.asarray(beta_exposure, dtype=float)
    sx = np.asarray(se_exposure, dtype=float)
    nx = np.asarray(n_exposure, dtype=float)
    if nx.ndim == 0:
        nx = np.full(len(bx), float(nx))
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    ny = np.asarray(n_outcome, dtype=float)
    if ny.ndim == 0:
        ny = np.full(len(by), float(ny))

    r2_x = _r2_from_beta_se(bx, sx, nx, eaf)
    r2_y = _r2_from_beta_se(by, sy, ny, eaf)

    # Fisher's z-transform to test H0: r2_x == r2_y
    # Convert R^2 to r and apply Fisher z-transform
    r_x = np.sqrt(max(min(r2_x, 1.0), 0.0))
    r_y = np.sqrt(max(min(r2_y, 1.0), 0.0))
    r_x = min(r_x, 0.9999)
    r_y = min(r_y, 0.9999)
    z_x = 0.5 * np.log((1 + r_x) / (1 - r_x))
    z_y = 0.5 * np.log((1 + r_y) / (1 - r_y))
    # Effective sample size (harmonic)
    n_x_eff = float(np.mean(nx))
    n_y_eff = float(np.mean(ny))
    se_z = np.sqrt(1.0 / (n_x_eff - 3) + 1.0 / (n_y_eff - 3))
    z_stat = (z_x - z_y) / se_z if se_z > 0 else 0.0
    # Two-sided by default, as TwoSampleMR::mr_steiger reports it (the
    # method authors' implementation); "greater" is the one-sided test that
    # the exposure explains more variance than the outcome, which this
    # function reported before 1.28.0. Upper tails come from norm.sf: the
    # old `1 - norm.cdf(z)` returned exactly 0.0 once z passed ~8.3, where
    # the reference reports 1.8e-73 on the LDL-C / CHD example.
    if alternative == "two-sided":
        p = float(2.0 * stats.norm.sf(abs(z_stat)))
    elif alternative == "greater":
        p = float(stats.norm.sf(z_stat))
    else:
        raise ValueError(
            f"alternative must be 'two-sided' or 'greater', got {alternative!r}"
        )

    return SteigerResult(
        correct_direction=bool(r2_x > r2_y),
        steiger_pvalue=p,
        r2_exposure=float(r2_x),
        r2_outcome=float(r2_y),
        sample_size_exposure=int(n_x_eff),
        sample_size_outcome=int(n_y_eff),
    )


# --------------------------------------------------------------------------- #
#  MR-PRESSO (outlier detection + correction)
# --------------------------------------------------------------------------- #


@dataclass
class MRPressoResult(ResultProtocolMixin):
    """Container for MR-PRESSO global test and outlier-corrected estimates.

    Returned by :func:`mr_presso`.  Holds the raw IVW estimate, the
    global-test RSS and p-value, the list of detected ``outliers``, and
    (when outliers are present) the outlier-corrected estimate and
    distortion-test p-value.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> bx = rng.uniform(0.2, 0.5, n)
    >>> by = 0.3 * bx + rng.normal(0, 0.02, n)
    >>> sx = rng.uniform(0.02, 0.05, n)
    >>> sy = rng.uniform(0.02, 0.05, n)
    >>> res = sp.mr_presso(bx, by, sx, sy, n_boot=500, seed=0)
    >>> isinstance(res, sp.MRPressoResult)
    True
    >>> bool(0.0 <= res.global_test_pvalue <= 1.0)
    True
    """

    _citation_keys = (
        "bowden2016assessing",
        "hemani2017orienting",
        "bowden2018improving",
        "verbanck2018detection",
    )

    raw_estimate: float
    raw_se: float
    raw_p: float
    outlier_corrected_estimate: Optional[float]
    outlier_corrected_se: Optional[float]
    outlier_corrected_p: Optional[float]
    outliers: List[int] = field(default_factory=list)
    global_test_rss_obs: float = 0.0
    global_test_pvalue: float = 1.0
    distortion_p: Optional[float] = None
    outlier_pvalues: Optional[List[float]] = None
    distortion_coefficient: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "MR-PRESSO Global Test",
            f"  Observed RSS = {self.global_test_rss_obs:.3f}",
            f"  Global test p-value = {self.global_test_pvalue:.4g}",
            "",
            f"Raw IVW:            beta = {self.raw_estimate:.4f}  "
            f"SE = {self.raw_se:.4f}  p = {self.raw_p:.4g}",
        ]
        if self.outlier_corrected_estimate is not None:
            lines.append(
                f"Outlier-corrected:  beta = {self.outlier_corrected_estimate:.4f}  "
                f"SE = {self.outlier_corrected_se:.4f}  "
                f"p = {self.outlier_corrected_p:.4g}"
            )
            if self.outliers:
                lines.append(f"  Outlier SNP indices: {self.outliers}")
            if self.distortion_p is not None:
                lines.append(
                    f"  Distortion test p = {self.distortion_p:.4g} "
                    f"(H0: raw == corrected)"
                )
        else:
            lines.append("No outliers detected.")
        return "\n".join(lines)


def _ivw_regression(bx: np.ndarray, by: np.ndarray, w: np.ndarray) -> tuple:
    """Weighted no-intercept regression of ``by`` on ``bx`` (weights ``w``).

    Returns ``(slope, se, t, p)`` with the residual-scaled standard error and
    a t reference on ``n - 1`` degrees of freedom (an ordinary weighted
    least-squares fit; Verbanck et al. 2018 build every MR-PRESSO estimate on
    this IVW regression).
    """
    sxx = float(np.sum(w * bx**2))
    slope = float(np.sum(w * bx * by) / sxx)
    df = bx.size - 1
    resid = by - slope * bx
    se = float(np.sqrt(np.sum(w * resid**2) / df / sxx))
    t = slope / se if se > 0 else float("inf") * np.sign(slope)
    p = float(2 * stats.t.sf(abs(t), df))
    return slope, se, float(t), p


def _loo_slopes(bx: np.ndarray, by: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Leave-one-out IVW slopes along the last axis (works row-wise on 2-D)."""
    sxy = np.sum(w * bx * by, axis=-1, keepdims=True)
    sxx = np.sum(w * bx**2, axis=-1, keepdims=True)
    return (sxy - w * bx * by) / (sxx - w * bx**2)


def _presso_rss(bx: np.ndarray, by: np.ndarray, sy: np.ndarray) -> tuple:
    """Observed MR-PRESSO residuals (Verbanck et al. 2018, global test step 2).

    Returns ``(rss_obs, resid2)``: ``resid2[j] = (by_j - b_{-j} bx_j)^2`` with
    ``b_{-j}`` the IVW slope without variant ``j``, and the global statistic
    ``rss_obs = sum_j resid2[j] / se_outcome_j^2``.
    """
    w = 1.0 / sy**2
    resid2 = (by - _loo_slopes(bx, by, w) * bx) ** 2
    return float(np.sum(w * resid2)), resid2


def mr_presso(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_exposure: np.ndarray,
    se_outcome: np.ndarray,
    *,
    n_boot: int = 1000,
    sig_threshold: float = 0.05,
    seed: Optional[int] = None,
) -> MRPressoResult:
    """MR-PRESSO global test + outlier detection + corrected estimate.

    The Mendelian Randomization Pleiotropy RESidual Sum and Outlier method
    of Verbanck et al. (2018), single-exposure version:

    1. **Global test.** With ``b_{-j}`` the IVW slope fitted without variant
       ``j``, the observed statistic is
       ``RSS_obs = sum_j (by_j - b_{-j} bx_j)^2 / se_outcome_j^2``. Its null
       distribution is simulated ``n_boot`` times by drawing
       ``bx*_j ~ N(bx_j, se_exposure_j^2)`` and
       ``by*_j ~ N(b_{-j} bx_j, se_outcome_j^2)`` and recomputing the same
       statistic (leave-one-out slopes included) on the draws. The p-value is
       the share of draws with ``RSS* > RSS_obs``.
    2. **Outlier test** (only when the global test rejects at
       ``sig_threshold``). Variant ``j``'s p-value is the share of draws in
       which its own term exceeds the observed one, multiplied by the number
       of variants (Bonferroni) and capped at 1. Variants below
       ``sig_threshold`` are outliers.
    3. **Outlier-corrected estimate**: the IVW regression without the
       outliers.
    4. **Distortion test**: ``D = 100 (b_raw - b_corrected) / |b_corrected|``
       against a null in which variants are resampled from the non-outliers.

    Parameters
    ----------
    beta_exposure, beta_outcome : array-like
        Variant-exposure and variant-outcome associations.
    se_exposure, se_outcome : array-like
        Their standard errors (strictly positive).
    n_boot : int, default 1000
        Number of simulated null draws. The global p-value resolves ``1 /
        n_boot`` and the Bonferroni-adjusted outlier p-values ``J / n_boot``;
        a warning fires when ``n_boot < J / sig_threshold``.
    sig_threshold : float, default 0.05
        Level of the global and outlier tests.
    seed : int, optional
        Seed for the simulated null draws.

    Returns
    -------
    MRPressoResult
        IVW estimates use the residual-scaled standard error and a t test on
        ``n - 1`` degrees of freedom. Simulated p-values are ``k / n_boot``
        (0 means "below the Monte Carlo resolution"). ``outlier_pvalues`` is
        ``None`` when the outlier test was not run; the corrected estimate and
        the distortion test are ``None`` when no outlier was found.

    Raises
    ------
    ValueError
        On mismatched lengths, fewer than 3 variants, non-finite input,
        non-positive standard errors, ``n_boot < 1`` or ``sig_threshold``
        outside ``(0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> bx = rng.uniform(0.2, 0.5, 15)
    >>> sx = np.full(15, 0.02)
    >>> sy = np.full(15, 0.02)
    >>> by = 0.3 * bx + rng.normal(0, 0.02, 15)
    >>> by[0] += 0.5
    >>> res = sp.mr_presso(bx, by, sx, sy, n_boot=500, seed=0)
    >>> 0 in res.outliers
    True

    Notes
    -----
    Clean-room implementation from Verbanck et al. (2018); the reference
    (R ``MRPRESSO`` 1.0, GPL-3) was used only as a black box. Its
    deterministic outputs (IVW fits, observed RSS, outlier-corrected fit,
    distortion coefficient) are matched exactly; its simulated p-values are
    compared as distributions (``docs/dev/cleanroom/mr_presso.md``).

    References
    ----------
    verbanck2018detection
    """
    bx, by, sx, sy = (
        np.asarray(a, dtype=float).ravel()
        for a in (beta_exposure, beta_outcome, se_exposure, se_outcome)
    )
    n = bx.size
    if not (by.size == sx.size == sy.size == n):
        raise ValueError("mr_presso: all four inputs must have the same length")
    if n < 3:
        raise ValueError(f"mr_presso needs at least 3 variants; got {n}")
    if not all(np.all(np.isfinite(a)) for a in (bx, by, sx, sy)):
        raise ValueError("mr_presso: inputs must be finite")
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("mr_presso: standard errors must be strictly positive")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError(f"mr_presso: n_boot must be >= 1; got {n_boot}")
    if not 0 < sig_threshold < 1:
        raise ValueError(
            f"mr_presso: sig_threshold must lie in (0, 1); got {sig_threshold}"
        )
    if n_boot * sig_threshold < n:
        warnings.warn(
            f"mr_presso: n_boot={n_boot} cannot resolve sig_threshold="
            f"{sig_threshold:g} after the Bonferroni adjustment for {n} variants: "
            f"one exceeding draw already gives an adjusted p of {n}/{n_boot}, so "
            f"only variants that no draw exceeds can be flagged. Use n_boot >= "
            f"{int(np.ceil(n / sig_threshold))}.",
            UserWarning,
            stacklevel=2,
        )

    w = 1.0 / sy**2
    raw_b, raw_se, _, raw_p = _ivw_regression(bx, by, w)
    rss_obs, resid2 = _presso_rss(bx, by, sy)
    rss_obs_j = w * resid2

    # Null draws (global test step 3): exposure around its estimate, outcome
    # around the leave-one-out prediction; the statistic is recomputed on the
    # draws, leave-one-out slopes included.
    rng = np.random.default_rng(seed)
    pred = _loo_slopes(bx, by, w) * bx
    draw_x = bx + sx * rng.standard_normal((n_boot, n))
    draw_y = pred + sy * rng.standard_normal((n_boot, n))
    rss_null_j = w * (draw_y - _loo_slopes(draw_x, draw_y, w) * draw_x) ** 2
    global_p = float(np.mean(rss_null_j.sum(axis=1) > rss_obs))

    outlier_p: Optional[np.ndarray] = None
    outliers: List[int] = []
    if global_p < sig_threshold:
        outlier_p = np.minimum(1.0, n * np.mean(rss_null_j > rss_obs_j, axis=0))
        outliers = [int(i) for i in np.flatnonzero(outlier_p < sig_threshold)]

    cor_b = cor_se = cor_p = dist_p = dist_coef = None
    if outliers:
        keep = np.ones(n, dtype=bool)
        keep[outliers] = False
        if keep.sum() < 2:
            warnings.warn(
                f"mr_presso: {len(outliers)} of {n} variants flagged; fewer than 2 "
                "remain, so no outlier-corrected estimate is reported.",
                UserWarning,
                stacklevel=2,
            )
        else:
            cor_b, cor_se, _, cor_p = _ivw_regression(bx[keep], by[keep], w[keep])
            dist_coef = 100.0 * (raw_b - cor_b) / abs(cor_b)
            dist_p = _distortion_pvalue(bx, by, w, keep, raw_b, dist_coef, n_boot, rng)

    return MRPressoResult(
        raw_estimate=raw_b,
        raw_se=raw_se,
        raw_p=raw_p,
        outlier_corrected_estimate=cor_b,
        outlier_corrected_se=cor_se,
        outlier_corrected_p=cor_p,
        outliers=outliers,
        global_test_rss_obs=rss_obs,
        global_test_pvalue=global_p,
        distortion_p=dist_p,
        outlier_pvalues=None if outlier_p is None else [float(v) for v in outlier_p],
        distortion_coefficient=dist_coef,
    )


def _distortion_pvalue(
    bx: np.ndarray,
    by: np.ndarray,
    w: np.ndarray,
    keep: np.ndarray,
    raw_b: float,
    dist_coef: float,
    n_boot: int,
    rng: np.random.Generator,
) -> float:
    """Simulated p-value of the MR-PRESSO distortion test.

    Verbanck et al. (2018): the null replaces the ``n_out`` outliers by
    ``n_out`` variants drawn with replacement from the non-outliers, keeping
    ``J`` variants. For each draw the IVW slope ``b_null`` of that set plays
    the role of the corrected estimate, ``D_null = 100 (b_raw - b_null) /
    |b_null|``, and the p-value is the share of draws with ``|D_null| >
    |D_obs|``.
    """
    non = np.flatnonzero(keep)
    n_out = bx.size - non.size
    extra = rng.choice(non, size=(n_boot, n_out), replace=True)
    sxy_non = float(np.sum(w[non] * bx[non] * by[non]))
    sxx_non = float(np.sum(w[non] * bx[non] ** 2))
    sxy = sxy_non + np.sum(w[extra] * bx[extra] * by[extra], axis=1)
    sxx = sxx_non + np.sum(w[extra] * bx[extra] ** 2, axis=1)
    b_null = sxy / sxx
    d_null = 100.0 * (raw_b - b_null) / np.abs(b_null)
    return float(np.mean(np.abs(d_null) > abs(dist_coef)))


# --------------------------------------------------------------------------- #
#  Radial MR (Bowden 2018)
# --------------------------------------------------------------------------- #


@dataclass
class RadialResult(ResultProtocolMixin):
    """Container for radial-MR diagnostics.

    Returned by :func:`mr_radial`.  Wraps a per-SNP ``table`` (columns
    ``snp``, ``W``, ``beta_hat``, ``q_contribution``), the ``total_Q``
    statistic with its ``Q_pvalue``, and the list of Bonferroni-flagged
    ``outliers``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> bx = rng.uniform(0.2, 0.5, n)
    >>> by = 0.3 * bx + rng.normal(0, 0.02, n)
    >>> sy = rng.uniform(0.02, 0.05, n)
    >>> rad = sp.mr_radial(bx, by, sy)
    >>> isinstance(rad, sp.RadialResult)
    True
    >>> len(rad.table)
    12
    """

    _citation_keys = (
        "bowden2016assessing",
        "hemani2017orienting",
        "bowden2018improving",
        "verbanck2018detection",
    )

    table: pd.DataFrame  # columns: snp, W, beta_hat, q_contribution
    total_Q: float
    Q_pvalue: float
    outliers: list[int]

    def summary(self) -> str:
        return (
            "Radial MR\n"
            f"  Total Q = {self.total_Q:.3f}   p = {self.Q_pvalue:.4g}\n"
            f"  Number of outlier SNPs (Bonferroni): {len(self.outliers)}"
        )


def mr_radial(
    beta_exposure: np.ndarray,
    beta_outcome: np.ndarray,
    se_outcome: np.ndarray,
    *,
    snp_ids: Optional[List[str]] = None,
    alpha: float = 0.05,
    bonferroni: bool = True,
) -> RadialResult:
    """Radial IVW MR (Bowden et al. 2018).

    Reparameterizes each SNP's Wald ratio as a coordinate in a
    "radial" space.  SNPs whose individual chi-square contribution to
    the Cochran Q exceeds the threshold are flagged as outliers: the
    Bonferroni level ``alpha / n`` by default, or ``alpha`` itself with
    ``bonferroni=False``, which is ``RadialMR::ivw_radial``'s rule. The
    per-variant contributions and total Q match ``RadialMR`` (first-order
    weights) to machine precision either way.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 12
    >>> beta_exposure = rng.uniform(0.2, 0.5, n)
    >>> beta_outcome = 0.3 * beta_exposure + rng.normal(0, 0.02, n)
    >>> se_outcome = rng.uniform(0.02, 0.05, n)
    >>> rad = sp.mr_radial(beta_exposure, beta_outcome, se_outcome)
    >>> len(rad.table)  # one row per SNP
    12
    >>> bool(0.0 <= rad.Q_pvalue <= 1.0)
    True
    >>> isinstance(rad.outliers, list)
    True

    References
    ----------
    [@bowden2018improving]
    """
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    n = len(bx)
    if n < 2:
        raise ValueError("mr_radial requires at least 2 SNPs to compute Cochran Q.")
    if snp_ids is None:
        snp_ids = [f"SNP_{i}" for i in range(n)]

    # Inverse-variance weights
    W = bx**2 / sy**2
    # Radial coords: beta_hat_i = by / bx (ratio), weighted by W
    ratio = by / bx
    beta_ivw = float(np.sum(W * ratio) / np.sum(W))
    # Per-SNP Q contribution
    q_i = W * (ratio - beta_ivw) ** 2
    total_Q = float(q_i.sum())
    df = max(n - 1, 1)
    p = float(stats.chi2.sf(total_Q, df))

    # Per-SNP outlier threshold: Bonferroni by default, RadialMR's
    # uncorrected alpha with bonferroni=False.
    level = alpha / n if bonferroni else alpha
    threshold = stats.chi2.ppf(1 - level, 1)
    outliers = [int(i) for i in range(n) if q_i[i] > threshold]

    table = pd.DataFrame(
        dict(
            snp=snp_ids,
            W=W,
            beta_hat=ratio,
            q_contribution=q_i,
        )
    )
    return RadialResult(
        table=table,
        total_Q=total_Q,
        Q_pvalue=p,
        outliers=outliers,
    )
