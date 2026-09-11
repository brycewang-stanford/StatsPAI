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

    Parameters
    ----------
    alternative : {"two-sided", "greater"}, default "two-sided"
        ``"two-sided"`` is the reference's p-value; ``"greater"`` tests the
        one-sided hypothesis that the exposure R^2 is the larger.

    .. versionchanged:: 1.27.0
       The p-value was one-sided and computed as ``1 - Phi(z)``, which is
       exactly 0.0 for z beyond ~8.3. It is now two-sided by default, as
       the method authors' TwoSampleMR reports it, computed from the upper
       tail directly; pass ``alternative="greater"`` for the one-sided test.

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
    # function reported before 1.27.0. Upper tails come from norm.sf: the
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

    A port of the authors' R package ``MRPRESSO::mr_presso``:

    * variants are oriented to positive exposure associations;
    * the raw and outlier-corrected estimates are weighted least squares
      through the origin (``lm(by ~ -1 + bx, weights = 1 / se_y^2)``), with
      the residual-scaled standard error and t(n - p) p-values;
    * the null simulates each exposure association from N(bx, se_x) and each
      outcome association around its *observed* leave-one-out prediction;
    * a variant's outlier p-value compares its observed squared residual
      from the observed leave-one-out fit with the simulated ones, and is
      Bonferroni-adjusted (multiplied by the number of variants, capped at
      1); the outlier test runs only when the global test rejects;
    * the distortion test resamples non-outlier variants.

    The deterministic outputs -- raw estimate / SE / p, observed RSS, and
    the corrected estimate for a given outlier set -- match the reference
    to machine precision. The simulated p-values cannot (different RNG).
    They follow the reference's ``k / B`` convention, so **a p-value of 0
    means "below the Monte Carlo resolution"**: ``< 1 / B`` for the global
    and distortion tests, ``< n / B`` for a Bonferroni-adjusted outlier
    p-value. The ``(k + 1) / (B + 1)`` convention cannot be used here: with
    Bonferroni its smallest attainable outlier p-value is ``n / (B + 1)``,
    which exceeds 0.05 for 50 variants at B = 1000, so the test could never
    flag anything. A warning is raised when ``n_boot`` is too small to
    resolve ``sig_threshold`` after the Bonferroni adjustment.

    .. versionchanged:: 1.27.0
       Brought in line with the reference: the per-variant outlier p-values
       were not Bonferroni-adjusted (seven outliers flagged on the LDL-C /
       CHD example where MR-PRESSO flags two, moving the corrected estimate
       2.4%), standard errors were fixed-effect rather than
       residual-scaled (half the reference's there), p-values came from
       ``1 - Phi(z)`` and read exactly 0.0, the outlier test ran even when
       the global test did not reject, and the "distortion test" was a
       z-test on the difference of two estimates rather than PRESSO's
       resampling test.

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
    >>> res = sp.mr_presso(
    ...     beta_exposure, beta_outcome, se_exposure, se_outcome,
    ...     n_boot=500, seed=0,
    ... )
    >>> bool(0.0 <= res.global_test_pvalue <= 1.0)
    True
    >>> isinstance(res.outliers, list)
    True

    References
    ----------
    [@verbanck2018detection]
    """
    rng = np.random.default_rng(seed)
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sx = np.asarray(se_exposure, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    sign = np.where(bx < 0, -1.0, 1.0)
    bx, by = bx * sign, by * sign
    w = 1.0 / sy**2
    n = len(bx)
    if n <= 3:
        raise ValueError("mr_presso: not enough instrumental variables (need > 3)")
    if n >= n_boot:
        raise ValueError("mr_presso: n_boot must exceed the number of variants")

    def _wls(mask):
        """lm(by ~ -1 + bx, weights = w): estimate, residual-scaled SE, p."""
        xw, yw = bx[mask] * np.sqrt(w[mask]), by[mask] * np.sqrt(w[mask])
        beta = float(xw @ yw / (xw @ xw))
        df = int(mask.sum()) - 1
        s2 = float(np.sum((yw - beta * xw) ** 2)) / df
        se = float(np.sqrt(s2 / (xw @ xw)))
        pval = float(2.0 * stats.t.sf(abs(beta / se), df)) if se > 0 else 1.0
        return beta, se, pval

    def _loo_betas(x, y):
        xw, yw = x * np.sqrt(w), y * np.sqrt(w)
        sxx, sxy = float(xw @ xw), float(xw @ yw)
        return (sxy - xw * yw) / (sxx - xw * xw)

    def _rss(x, y):
        xw, yw = x * np.sqrt(w), y * np.sqrt(w)
        return float(np.sum((yw - _loo_betas(x, y) * xw) ** 2))

    all_mask = np.ones(n, dtype=bool)
    raw_beta, raw_se, raw_p = _wls(all_mask)
    beta_loo_obs = _loo_betas(bx, by)
    rss_obs = _rss(bx, by)

    # Null data sets: exposure ~ N(bx, sx); outcome_i ~ N(beta_{-i} bx_i, sy_i)
    # around the OBSERVED leave-one-out prediction, as getRandomData builds.
    sim_bx = rng.normal(bx, sx, size=(n_boot, n))
    sim_by = rng.normal(beta_loo_obs * bx, sy, size=(n_boot, n))
    rss_exp = np.array([_rss(sim_bx[b], sim_by[b]) for b in range(n_boot)])
    p_global = float(np.sum(rss_exp > rss_obs) / n_boot)
    if n / n_boot > sig_threshold:
        warnings.warn(
            f"mr_presso: n_boot={n_boot} cannot resolve sig_threshold={sig_threshold} "
            f"after the Bonferroni adjustment for {n} variants "
            f"(resolution {n}/{n_boot}); "
            "only variants that no simulated draw exceeds can be flagged. "
            f"Use n_boot >= {int(np.ceil(n / sig_threshold))}.",
            UserWarning,
            stacklevel=2,
        )

    base = dict(
        raw_estimate=raw_beta,
        raw_se=raw_se,
        raw_p=raw_p,
        global_test_rss_obs=rss_obs,
        global_test_pvalue=p_global,
    )
    if p_global >= sig_threshold:
        return MRPressoResult(
            outlier_corrected_estimate=None,
            outlier_corrected_se=None,
            outlier_corrected_p=None,
            outliers=[],
            **base,
        )

    dif2 = (by - bx * beta_loo_obs) ** 2
    exp2 = (sim_by - sim_bx * beta_loo_obs[None, :]) ** 2
    p_snp = np.sum(exp2 > dif2[None, :], axis=0) / n_boot
    p_snp = np.minimum(p_snp * n, 1.0)
    outliers = [int(i) for i in range(n) if p_snp[i] <= sig_threshold]

    if not outliers or len(outliers) == n:
        return MRPressoResult(
            outlier_corrected_estimate=None,
            outlier_corrected_se=None,
            outlier_corrected_p=None,
            outliers=outliers,
            outlier_pvalues=[float(v) for v in p_snp],
            **base,
        )

    keep = np.ones(n, dtype=bool)
    keep[outliers] = False
    c_beta, c_se, c_p = _wls(keep)

    # Distortion test: resample n - k indices (with replacement) from the
    # non-outliers, refit, and compare relative biases.
    non_out = np.flatnonzero(keep)
    bias_obs = (raw_beta - c_beta) / abs(c_beta)
    bias_exp = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(non_out, size=n - len(outliers), replace=True)
        xw, yw = bx[idx] * np.sqrt(w[idx]), by[idx] * np.sqrt(w[idx])
        br = float(xw @ yw / (xw @ xw))
        bias_exp[b] = (raw_beta - br) / abs(br)
    dist_p = float(np.sum(np.abs(bias_exp) > abs(bias_obs)) / n_boot)

    return MRPressoResult(
        outlier_corrected_estimate=c_beta,
        outlier_corrected_se=c_se,
        outlier_corrected_p=c_p,
        outliers=outliers,
        distortion_p=dist_p,
        outlier_pvalues=[float(v) for v in p_snp],
        distortion_coefficient=float(100.0 * bias_obs),
        **base,
    )


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
