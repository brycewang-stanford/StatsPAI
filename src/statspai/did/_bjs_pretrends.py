"""Reference-period conventions for imputation-DiD pre-trend coefficients.

An imputation estimator computes its **post**-treatment event-study
coefficients by averaging ``Y - Ŷ(0)`` over treated cells.  Its
**pre**-treatment coefficients are a separate construction, and the
construction is not the same in every implementation.  Roth (2026) shows
that the choice changes the shape of the plotted event study — a kink or a
jump can appear at the treatment date purely from the construction, with
no treatment effect and with parallel trends violated identically in every
period — so the visual heuristics that applied researchers carry over from
a dynamic TWFE event study do not transfer.

Three constructions are implemented here and in :func:`did_imputation`.

``"bjs"``
    The convention of Borusyak, Jaravel and Spiess (2024) as shipped in
    Stata ``did_imputation, pretrends(k)``.  Fit a dynamic TWFE regression
    on the **untreated** observations only, with lead indicators
    ``1{relative time = -1}, …, 1{relative time = -k}`` and *all earlier
    relative times pooled into the omitted category*.  In the
    non-staggered case this makes the pre-treatment coefficients
    long-differences against the earliest pre-period, while the
    post-treatment coefficients are differences against the *average* of
    all pre-periods — the asymmetry that produces Roth's jump.

``"in-sample"``
    The convention of the ``fect`` and ``did2s`` packages: report
    ``mean(Y - Ŷ(0))`` at pre-treatment relative times too.  Those are
    in-sample prediction errors, because the pre-treatment outcomes of
    eventually-treated units are themselves part of the training data.
    Li and Strezhnev (2025) show, and Roth (2026, appendix A) restates,
    that in the non-staggered case this shrinks the symmetric benchmark by
    exactly ``N0 / N`` — the untreated share of units — so it understates
    pre-trends, severely when most units are treated.

``"symmetric"``
    Roth's (2026) recommended repair, ``β̂^{BJS,new}``: use the average of
    the pre-treatment periods as the reference for *both* the pre- and the
    post-treatment coefficients.  In the non-staggered case the resulting
    path equals the dynamic TWFE event study up to one common vertical
    shift, so the usual visual heuristics apply again.  Note that the
    pre-treatment coefficients then average to zero by construction, so
    only the *relative* movement between coefficients is informative.

References
----------
Borusyak, K., Jaravel, X. and Spiess, J. (2024).  "Revisiting Event-Study
Designs: Robust and Efficient Estimation."  *Review of Economic Studies*,
91(6), 3253-3285.  [@borusyak2024revisiting]

Roth, J. (2026).  "Interpreting Event-Studies from Recent
Difference-in-Differences Methods."  arXiv:2401.12309.
[@roth2026interpreting]

Li, Z. and Strezhnev, A. (2025).  "Benchmarking Parallel Trends Violations
in Regression Imputation Difference-in-Differences."
[@li2025benchmarking]
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import lsqr

from ..exceptions import MethodIncompatibility

__all__ = [
    "EVENT_STUDY_CONVENTION",
    "PRETREND_METHODS",
    "bjs_pretrend_path",
    "symmetric_pretrend_scale",
]

PRETREND_METHODS: Tuple[str, ...] = ("bjs", "in-sample", "symmetric")

#: What each option constructs, in the form a reader of the plot needs.
#: ``twfe_comparable`` answers the only question the visual heuristics
#: depend on: does this path coincide with a dynamic TWFE event study
#: (up to a common vertical shift) when the design is non-staggered?
EVENT_STUDY_CONVENTION: Dict[str, Dict[str, object]] = {
    "bjs": {
        "pre_construction": (
            "auxiliary dynamic TWFE on untreated observations; leads "
            "referenced to the pooled earlier pre-periods"
        ),
        "post_construction": (
            "imputation residuals, referenced to the average of all "
            "pre-treatment periods"
        ),
        "symmetric": False,
        "twfe_comparable": False,
        "matches": "Stata did_imputation, pretrends(k)",
        "caveat": (
            "Pre- and post-treatment coefficients use different reference "
            "periods, so a jump at the treatment date can appear with no "
            "treatment effect. Read the leads as a test, not as a plot."
        ),
    },
    "in-sample": {
        "pre_construction": ("mean of the in-sample imputation residuals at each lead"),
        "post_construction": (
            "imputation residuals, referenced to the average of all "
            "pre-treatment periods"
        ),
        "symmetric": False,
        "twfe_comparable": False,
        "matches": "R fect / did2s",
        "caveat": (
            "Pre-treatment leads are in-sample prediction errors and are "
            "attenuated by the untreated unit share N0/N in the "
            "non-staggered case, so they understate pre-trends."
        ),
    },
    "symmetric": {
        "pre_construction": (
            "in-sample residual means de-attenuated by N/N0 (non-staggered "
            "designs only)"
        ),
        "post_construction": (
            "imputation residuals, referenced to the average of all "
            "pre-treatment periods"
        ),
        "symmetric": True,
        "twfe_comparable": True,
        "matches": "Roth (2026) beta-hat^{BJS,new}",
        "caveat": (
            "Pre-treatment coefficients average to zero by construction, so "
            "compare their movement, not their mean level."
        ),
    },
}

#: Above this many requested leads the partialling-out loop is the
#: dominant cost, and a caller almost certainly wants a joint test rather
#: than 200 separately plotted placebo coefficients.
_MAX_LEADS = 100


def _partial_out(
    x: np.ndarray,
    design: sparse.csr_matrix,
    col_norms: np.ndarray,
    scale: sparse.dia_matrix,
) -> np.ndarray:
    """Residualise ``x`` on the (equilibrated) untreated design."""
    fit = lsqr(
        design @ scale,
        x,
        atol=1e-12,
        btol=1e-12,
        iter_lim=max(2000, 8 * design.shape[1]),
    )
    coef = fit[0] / col_norms
    return x - np.asarray(design @ coef, dtype=float)


def bjs_pretrend_path(
    *,
    design_untreated: sparse.csr_matrix,
    y_untreated: np.ndarray,
    rel_time_untreated: np.ndarray,
    cluster_untreated: np.ndarray,
    n_unit_columns: int,
    leads: List[int],
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Pre-trend coefficients under the BJS / Stata ``did_imputation`` rule.

    Runs the auxiliary dynamic TWFE regression on the untreated
    observations only.  ``design_untreated`` is the same Y(0) design the
    imputation step already built (intercept, fixed effects, controls), so
    the pre-trend regression inherits exactly the model the estimator
    imputes with, and the lead indicators are appended to it.

    Identification of the leads comes from the omitted category: every
    relative time earlier than ``min(leads)`` — plus every never-treated
    observation — stays in the reference group.  Requesting a lead for
    which no untreated observation survives is a caller error rather than
    a silently dropped column, and is reported as such.

    Parameters
    ----------
    design_untreated
        Sparse untreated design matrix, first column the intercept.
    y_untreated
        Outcome on the untreated rows.
    rel_time_untreated
        Relative time on the untreated rows; ``-inf`` for never-treated.
    cluster_untreated
        Cluster identifier on the untreated rows.
    n_unit_columns
        Number of unit fixed-effect columns in ``design_untreated``.
        Used only for the degrees-of-freedom adjustment, which follows the
        absorbed-fixed-effect convention Stata's ``did_imputation`` uses:
        unit effects are absorbed and do not consume degrees of freedom,
        every other non-intercept column does.
    leads
        Negative relative times to estimate, e.g. ``[-3, -2, -1]``.
    alpha
        Two-sided level for the reported confidence intervals.

    Returns
    -------
    (frame, joint_test)
        ``frame`` has one row per lead with ``relative_time``, ``att``,
        ``se``, ``ci_lower``, ``ci_upper``, ``pvalue`` and ``n_obs``.
        ``joint_test`` carries the Wald statistic for the null that every
        lead coefficient is zero, computed from the full cluster-robust
        covariance rather than the diagonal.
    """
    leads = sorted({int(k) for k in leads})
    if not leads:
        raise ValueError("bjs_pretrend_path requires at least one lead.")
    if any(k >= 0 for k in leads):
        raise ValueError(
            f"leads must all be negative relative times, got {leads}. "
            "Post-treatment horizons come from the imputation step."
        )
    if len(leads) > _MAX_LEADS:
        raise MethodIncompatibility(
            f"pretrend_method='bjs' was asked for {len(leads)} lead "
            f"coefficients (cap {_MAX_LEADS}). Each one is partialled out "
            "of the untreated design separately, so the cost is linear in "
            "the count.",
            recovery_hint=(
                "Request fewer leads via pretrends=k or horizon=, or use "
                "sp.bjs_pretrend_joint for a joint pre-trend test without "
                "a per-lead path."
            ),
        )

    y_u = np.asarray(y_untreated, dtype=float)
    rel_u = np.asarray(rel_time_untreated, dtype=float)
    n_obs = y_u.shape[0]

    lead_matrix = np.zeros((n_obs, len(leads)), dtype=float)
    counts: List[int] = []
    for j, k in enumerate(leads):
        mask = rel_u == float(k)
        lead_matrix[:, j] = mask.astype(float)
        counts.append(int(mask.sum()))
    empty = [k for k, c in zip(leads, counts) if c == 0]
    if empty:
        raise MethodIncompatibility(
            "pretrend_method='bjs': no untreated observation sits at "
            f"relative time(s) {empty}, so those lead coefficients are not "
            "identified.",
            recovery_hint=(
                "Reduce the number of requested leads (pretrends=k) so that "
                "every lead has pre-treatment observations, or restrict "
                "horizon= to the leads the panel supports."
            ),
        )

    # The omitted category has to retain observations from *eventually
    # treated* units, not merely from never-treated ones.  If the leads
    # cover every pre-treatment period of a treated cohort, that cohort's
    # lead indicators sum to its unit indicators and the design is exactly
    # rank deficient: the least-squares solve still returns numbers, and
    # they are meaningless.  Never-treated rows do not repair this,
    # because they carry no lead indicator at all.
    earliest = min(leads)
    finite = np.isfinite(rel_u)
    reference_rows = int((finite & (rel_u < earliest)).sum())
    if reference_rows == 0:
        raise MethodIncompatibility(
            f"pretrend_method='bjs': the {len(leads)} requested leads cover "
            "every pre-treatment period of the treated cohort(s), so the "
            "lead indicators are collinear with the unit fixed effects and "
            "no reference period is left.",
            recovery_hint=(
                "Request at most (shortest pre-treatment history - 1) "
                "leads, e.g. pretrends="
                f"{max(len(leads) - 1, 1)}, or trim horizon= accordingly."
            ),
        )

    col_norms = np.sqrt(
        np.asarray(design_untreated.multiply(design_untreated).sum(axis=0)).ravel()
    )
    col_norms[col_norms <= 0] = 1.0
    scale = sparse.diags(1.0 / col_norms)

    # Frisch-Waugh: residualise the leads and the outcome on the Y(0)
    # design, then the K x K problem is dense and small.  Solving the full
    # augmented system instead would need the inverse of a matrix with one
    # column per unit, which is exactly what the sparse solve avoids.
    lead_resid = np.column_stack(
        [
            _partial_out(lead_matrix[:, j], design_untreated, col_norms, scale)
            for j in range(len(leads))
        ]
    )
    y_resid = _partial_out(y_u, design_untreated, col_norms, scale)

    gram = lead_resid.T @ lead_resid
    try:
        gram_inv = np.linalg.inv(gram)
    except np.linalg.LinAlgError:  # pragma: no cover - guarded above
        raise MethodIncompatibility(
            "pretrend_method='bjs': the lead indicators are collinear with "
            "the Y(0) design after partialling out the fixed effects.",
            recovery_hint="Request fewer leads, or use pretrend_method='in-sample'.",
        ) from None
    beta = gram_inv @ (lead_resid.T @ y_resid)
    resid = y_resid - lead_resid @ beta

    cluster = np.asarray(cluster_untreated)
    meat = np.zeros((len(leads), len(leads)), dtype=float)
    _, cluster_codes = np.unique(cluster, return_inverse=True)
    n_clusters = int(cluster_codes.max()) + 1
    for g in range(n_clusters):
        idx = np.flatnonzero(cluster_codes == g)
        score = lead_resid[idx].T @ resid[idx]
        meat += np.outer(score, score)

    # Degrees of freedom: G/(G-1) * (n-1)/(n-p) with p counting every
    # non-intercept column except the absorbed unit effects.  Validated to
    # 1e-14 relative against Stata did_imputation on two non-staggered
    # designs; see tests/test_did_pretrend_conventions.py.
    n_params = design_untreated.shape[1] - 1 - max(n_unit_columns, 0) + len(leads)
    denom = max(n_obs - n_params, 1)
    adjust = (
        (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / denom)
        if n_clusters > 1
        else 1.0
    )
    vcov = gram_inv @ meat @ gram_inv * adjust
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = np.where(se > 0, beta / se, 0.0)
    pvalue = 2.0 * (1.0 - stats.norm.cdf(np.abs(zstat)))

    frame = pd.DataFrame(
        {
            "relative_time": leads,
            "att": beta,
            "se": se,
            "ci_lower": beta - z_crit * se,
            "ci_upper": beta + z_crit * se,
            "pvalue": pvalue,
            "n_obs": counts,
        }
    )

    joint: Dict[str, float] = {}
    try:
        stat = float(beta @ np.linalg.solve(vcov, beta))
    except np.linalg.LinAlgError:  # pragma: no cover
        stat = float("nan")
    if np.isfinite(stat):
        joint = {
            "statistic": stat,
            "df": float(len(leads)),
            "pvalue": float(1 - stats.chi2.cdf(stat, len(leads))),
            "method": "wald-cluster (BJS pretrends auxiliary regression)",
        }
    return frame, joint


def symmetric_pretrend_scale(
    *,
    n_units_total: int,
    n_units_untreated: int,
    n_cohorts_treated: int,
    balanced: bool,
    has_covariates: bool,
) -> float:
    """Exact de-attenuation factor for Roth's symmetric event study.

    In-sample imputation reports ``(N0 / N) * β̂^{BJS,new}`` at
    pre-treatment relative times, where ``N0`` counts never-treated units
    (Roth 2026, appendix A, restating Li and Strezhnev 2025).  Multiplying
    the reported pre-treatment coefficients by ``N / N0`` therefore
    recovers the symmetric path exactly — but the result is only known to
    be exact in the setting the two papers analyse: a single treated
    cohort, a balanced panel, and the plain two-way model.

    Rather than apply the factor outside that setting and hope, this
    raises.  The staggered generalisation is the ``block bias`` of Liu
    (2025) and is not implemented here.
    """
    if n_cohorts_treated != 1:
        raise MethodIncompatibility(
            "pretrend_method='symmetric' is only pinned for non-staggered "
            f"designs; this panel has {n_cohorts_treated} treated cohorts. "
            "The de-attenuation factor N/N0 is exact for a single treated "
            "cohort (Roth 2026, appendix A); the staggered generalisation "
            "is the block-bias construction of Liu (2025) and is not "
            "implemented.",
            recovery_hint=(
                "Use pretrend_method='bjs' for the reference convention, or "
                "sp.callaway_santanna(..., base_period='universal'), whose "
                "event study is symmetric by construction in both "
                "staggered and non-staggered designs."
            ),
        )
    if not balanced:
        raise MethodIncompatibility(
            "pretrend_method='symmetric' requires a balanced panel: the "
            "N/N0 de-attenuation factor is derived for one.",
            recovery_hint="Use pretrend_method='bjs', or balance the panel first.",
        )
    if has_covariates:
        raise MethodIncompatibility(
            "pretrend_method='symmetric' is not pinned once the Y(0) model "
            "carries covariates or non-default fixed effects; the "
            "attenuation factor is no longer N/N0.",
            recovery_hint="Use pretrend_method='bjs'.",
        )
    if n_units_untreated <= 0:
        raise MethodIncompatibility(
            "pretrend_method='symmetric' needs at least one never-treated "
            "unit; with every unit treated the factor N/N0 is undefined.",
            recovery_hint="Use pretrend_method='bjs'.",
        )
    return float(n_units_total) / float(n_units_untreated)
