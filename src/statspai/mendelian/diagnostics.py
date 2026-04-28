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

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


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


# --------------------------------------------------------------------------- #
#  Cochran's Q / Rücker's Q'
# --------------------------------------------------------------------------- #


@dataclass
class HeterogeneityResult:
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
    """
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    w = 1.0 / sy ** 2

    if method == "ivw":
        # Fit IVW through origin; compute residuals weighted by 1/sy^2
        beta_ivw = np.sum(w * bx * by) / np.sum(w * bx ** 2)
        Q = float(np.sum(w * (by - beta_ivw * bx) ** 2))
        df = len(bx) - 1
    elif method == "egger":
        # Rücker's Q' uses the Egger-fitted intercept+slope
        X = np.column_stack([np.ones(len(bx)), bx])
        W = np.diag(w)
        try:
            beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ by)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ by
        resid = by - X @ beta
        Q = float(np.sum(w * resid ** 2))
        df = len(bx) - 2
    else:
        raise ValueError("method must be 'ivw' or 'egger'")

    df = max(df, 1)
    p = float(1 - stats.chi2.cdf(Q, df))
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
class PleiotropyResult:
    intercept: float
    se: float
    p_value: float

    def summary(self) -> str:
        direction = "directional pleiotropy" if self.p_value < 0.05 else \
            "no evidence of directional pleiotropy"
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
    """
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    w = 1.0 / sy ** 2
    n = len(bx)

    X = np.column_stack([np.ones(n), bx])
    W = np.diag(w)
    try:
        XtWX_inv = np.linalg.inv(X.T @ W @ X)
    except np.linalg.LinAlgError:
        XtWX_inv = np.linalg.pinv(X.T @ W @ X)
    beta = XtWX_inv @ X.T @ W @ by
    resid = by - X @ beta
    sigma2 = float(np.sum(w * resid ** 2)) / max(n - 2, 1)
    se = np.sqrt(sigma2 * np.diag(XtWX_inv))
    intercept = float(beta[0])
    intercept_se = float(se[0])
    if intercept_se > 0:
        t_stat = intercept / intercept_se
        # Use t(n-2) because sigma^2 is plug-in estimated; matches
        # R's MendelianRandomization / TwoSampleMR.
        df = max(n - 2, 1)
        p = float(2 * stats.t.sf(abs(t_stat), df=df))
    else:
        p = float("nan")
    return PleiotropyResult(intercept=intercept, se=intercept_se, p_value=p)


# --------------------------------------------------------------------------- #
#  Leave-one-out
# --------------------------------------------------------------------------- #


@dataclass
class LeaveOneOutResult:
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
) -> LeaveOneOutResult:
    """IVW estimate with each SNP dropped in turn.

    Identifies SNPs that drive the overall estimate disproportionately.
    """
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    n = len(bx)
    if snp_ids is None:
        snp_ids = [f"SNP_{i}" for i in range(n)]
    z_crit = stats.norm.ppf(1 - alpha / 2)

    rows = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w = 1.0 / sy[mask] ** 2
        beta = float(np.sum(w * bx[mask] * by[mask]) / np.sum(w * bx[mask] ** 2))
        se = float(np.sqrt(1.0 / np.sum(w * bx[mask] ** 2)))
        z = beta / se if se > 0 else 0.0
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
        rows.append(dict(
            dropped_snp=snp_ids[i],
            estimate=beta,
            se=se,
            ci_lower=beta - z_crit * se,
            ci_upper=beta + z_crit * se,
            p_value=p,
        ))
    return LeaveOneOutResult(table=pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
#  Steiger directionality test
# --------------------------------------------------------------------------- #


@dataclass
class SteigerResult:
    correct_direction: bool
    steiger_pvalue: float
    r2_exposure: float
    r2_outcome: float
    sample_size_exposure: int
    sample_size_outcome: int

    def summary(self) -> str:
        dir_str = ("exposure -> outcome (expected)"
                   if self.correct_direction
                   else "outcome -> exposure (reverse!)")
        return (
            "Steiger Directionality Test\n"
            f"  R^2 (on exposure) = {self.r2_exposure:.4f}\n"
            f"  R^2 (on outcome)  = {self.r2_outcome:.4f}\n"
            f"  Direction: {dir_str}\n"
            f"  Steiger p = {self.steiger_pvalue:.4g}"
        )


def _r2_from_beta_se(beta, se, n, eaf=None):
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
        r2 = 2 * beta ** 2 * eaf * (1 - eaf)
    # Total R^2 across independent SNPs: bounded at 1
    return float(min(1.0, np.sum(r2)))


def mr_steiger(
    beta_exposure: np.ndarray,
    se_exposure: np.ndarray,
    n_exposure,
    beta_outcome: np.ndarray,
    se_outcome: np.ndarray,
    n_outcome,
    *,
    eaf: Optional[np.ndarray] = None,
) -> SteigerResult:
    """Steiger test for the direction of the causal effect (Hemani 2017).

    Compares the R^2 of the SNPs on the exposure with their R^2 on the
    outcome.  If R^2_exposure > R^2_outcome the assumed causal
    direction (exposure -> outcome) is supported.
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
    # One-sided: does exposure R^2 exceed outcome R^2?
    p = float(1 - stats.norm.cdf(z_stat))

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
class MRPressoResult:
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


def _ivw(bx, by, sy):
    w = 1.0 / sy ** 2
    beta = float(np.sum(w * bx * by) / np.sum(w * bx ** 2))
    se = float(np.sqrt(1.0 / np.sum(w * bx ** 2)))
    return beta, se


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

    Implementation notes
    --------------------
    Follows Verbanck et al. (2018).  For each SNP i, the "residual sum
    of squares" (RSS_obs) is computed using leave-one-out predictions.
    The null distribution is obtained by simulating SNP-outcome
    effects with the same SEs and no pleiotropy.  SNPs with individual
    test p < ``sig_threshold`` are flagged as outliers and re-analyzed.
    """
    rng = np.random.default_rng(seed)
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sx = np.asarray(se_exposure, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    n = len(bx)

    raw_beta, raw_se = _ivw(bx, by, sy)
    raw_p = float(2 * (1 - stats.norm.cdf(abs(raw_beta / raw_se)))) if raw_se > 0 else 1.0

    # Observed residuals and per-SNP RSS contribution (leave-one-out)
    def _rss_components(bx_, by_, sy_):
        comps = np.empty(len(bx_))
        for i in range(len(bx_)):
            mask = np.arange(len(bx_)) != i
            beta_i, _ = _ivw(bx_[mask], by_[mask], sy_[mask])
            comps[i] = (by_[i] - beta_i * bx_[i]) ** 2 / sy_[i] ** 2
        return comps

    obs_rss_components = _rss_components(bx, by, sy)
    rss_obs = float(obs_rss_components.sum())

    # Null RSS via simulation under no-pleiotropy
    null_rss = np.empty(n_boot)
    null_components = np.empty((n_boot, n))
    for b in range(n_boot):
        # Simulate beta_exposure with measurement error
        bx_sim = bx + rng.normal(0, sx)
        # Simulate beta_outcome under H0: beta_y = beta_ivw * beta_x
        by_sim = raw_beta * bx_sim + rng.normal(0, sy)
        comps = _rss_components(bx_sim, by_sim, sy)
        null_components[b] = comps
        null_rss[b] = comps.sum()

    # MC p-value convention: use (k + 1) / (B + 1) for both the global
    # test and the per-SNP outlier test.  The raw ``mean(null >= obs)``
    # form can return exactly 0 when the observed statistic exceeds
    # every simulated null value, which is inadmissible as a p-value
    # (the true tail probability is bounded below by 1 / (B + 1)).
    # Matches the convention used by the R ``MR-PRESSO`` package and
    # MCRATs 2003 §3.3.
    p_global = float((np.sum(null_rss >= rss_obs) + 1) / (n_boot + 1))
    per_snp_p = np.array([
        (np.sum(null_components[:, i] >= obs_rss_components[i]) + 1)
        / (n_boot + 1)
        for i in range(n)
    ])
    outliers = [int(i) for i in range(n) if per_snp_p[i] < sig_threshold]

    if not outliers:
        return MRPressoResult(
            raw_estimate=raw_beta,
            raw_se=raw_se,
            raw_p=raw_p,
            outlier_corrected_estimate=None,
            outlier_corrected_se=None,
            outlier_corrected_p=None,
            outliers=[],
            global_test_rss_obs=rss_obs,
            global_test_pvalue=p_global,
        )

    keep = [i for i in range(n) if i not in outliers]
    c_beta, c_se = _ivw(bx[keep], by[keep], sy[keep])
    c_p = float(2 * (1 - stats.norm.cdf(abs(c_beta / c_se)))) if c_se > 0 else 1.0

    # Distortion test: how different is corrected from raw?
    dist_z = (raw_beta - c_beta) / np.sqrt(raw_se ** 2 + c_se ** 2)
    dist_p = float(2 * (1 - stats.norm.cdf(abs(dist_z))))

    return MRPressoResult(
        raw_estimate=raw_beta,
        raw_se=raw_se,
        raw_p=raw_p,
        outlier_corrected_estimate=c_beta,
        outlier_corrected_se=c_se,
        outlier_corrected_p=c_p,
        outliers=outliers,
        global_test_rss_obs=rss_obs,
        global_test_pvalue=p_global,
        distortion_p=dist_p,
    )


# --------------------------------------------------------------------------- #
#  Radial MR (Bowden 2018)
# --------------------------------------------------------------------------- #


@dataclass
class RadialResult:
    table: pd.DataFrame   # columns: snp, W, beta_hat, q_contribution
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
) -> RadialResult:
    """Radial IVW MR (Bowden et al. 2018).

    Reparameterizes each SNP's Wald ratio as a coordinate in a
    "radial" space.  SNPs whose individual chi-square contribution to
    the Cochran Q exceeds the Bonferroni threshold at alpha=0.05 are
    flagged as outliers.
    """
    bx = np.asarray(beta_exposure, dtype=float)
    by = np.asarray(beta_outcome, dtype=float)
    sy = np.asarray(se_outcome, dtype=float)
    n = len(bx)
    if n < 2:
        raise ValueError(
            "mr_radial requires at least 2 SNPs to compute Cochran Q."
        )
    if snp_ids is None:
        snp_ids = [f"SNP_{i}" for i in range(n)]

    # Inverse-variance weights
    W = bx ** 2 / sy ** 2
    # Radial coords: beta_hat_i = by / bx (ratio), weighted by W
    ratio = by / bx
    beta_ivw = float(np.sum(W * ratio) / np.sum(W))
    # Per-SNP Q contribution
    q_i = W * (ratio - beta_ivw) ** 2
    total_Q = float(q_i.sum())
    df = max(n - 1, 1)
    p = float(1 - stats.chi2.cdf(total_Q, df))

    # Bonferroni threshold for per-SNP outlier
    threshold = stats.chi2.ppf(1 - 0.05 / n, 1)
    outliers = [int(i) for i in range(n) if q_i[i] > threshold]

    table = pd.DataFrame(dict(
        snp=snp_ids,
        W=W,
        beta_hat=ratio,
        q_contribution=q_i,
    ))
    return RadialResult(
        table=table,
        total_Q=total_Q,
        Q_pvalue=p,
        outliers=outliers,
    )
