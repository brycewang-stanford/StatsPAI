"""
Multivariable MR, Mediation MR and MR-BMA (Bayesian Model Averaging).

Three extensions to the single-exposure MR suite:

1. **Multivariable MR** (Sanderson, Davey Smith, Windmeijer & Bowden 2019)
   — estimate direct effects of multiple correlated exposures by regressing
   outcome SNP-associations on *all* exposure SNP-associations jointly.
2. **Mediation MR (two-step MR)** (Burgess et al. 2015) — decompose the
   total effect of an exposure into direct and mediator-mediated paths
   using two IVW regressions.
3. **MR-BMA** (Zuber, Colijn, Staley & Burgess 2020) — Bayesian model
   averaging over the 2^k subsets of exposures to identify which are
   causal when many correlated risk factors are considered.

These satisfy the v0.9.17 road-map's response to the Yao et al. 2026
MR estimand / identification / inference framework review (arXiv:2509.11519).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    "mr_multivariable", "mr_mediation", "mr_bma",
    "MVMRResult", "MediationMRResult", "MRBMAResult",
]


# -------------------------------------------------------------------------
# Result containers
# -------------------------------------------------------------------------


@dataclass
class MVMRResult:
    """Multivariable MR output."""
    exposures: List[str]
    direct_effect: pd.DataFrame  # columns: exposure, estimate, se, ci_low, ci_high, p_value
    conditional_f_stats: Dict[str, float]
    n_snps: int

    def summary(self) -> str:
        lines = [
            "Multivariable Mendelian Randomization",
            "=" * 60,
            f"  n SNPs : {self.n_snps}",
            f"  exposures : {self.exposures}",
            "",
            self.direct_effect.to_string(index=False, float_format="%.4f"),
            "",
            "Conditional F-statistics (strength of each exposure):",
        ]
        for exp, f in self.conditional_f_stats.items():
            flag = " (weak)" if f < 10 else ""
            lines.append(f"  {exp}: {f:.2f}{flag}")
        return "\n".join(lines)


@dataclass
class MediationMRResult:
    """Two-step MR output."""
    exposure: str
    mediator: str
    outcome: str
    total_effect: float
    total_effect_se: float
    direct_effect: float
    direct_effect_se: float
    indirect_effect: float
    indirect_effect_se: float
    proportion_mediated: float

    def summary(self) -> str:
        return (
            f"Two-Step MR Mediation Analysis\n"
            f"{'=' * 60}\n"
            f"  Exposure : {self.exposure}\n"
            f"  Mediator : {self.mediator}\n"
            f"  Outcome  : {self.outcome}\n"
            f"\n"
            f"  Total effect     = {self.total_effect:+.4f} "
            f"(SE {self.total_effect_se:.4f})\n"
            f"  Direct effect    = {self.direct_effect:+.4f} "
            f"(SE {self.direct_effect_se:.4f})\n"
            f"  Indirect effect  = {self.indirect_effect:+.4f} "
            f"(SE {self.indirect_effect_se:.4f})\n"
            f"  % mediated       = {100 * self.proportion_mediated:.1f}%"
        )


@dataclass
class MRBMAResult:
    """MR-BMA (Bayesian Model Averaging) output."""
    exposures: List[str]
    marginal_inclusion: pd.Series  # P(exposure in the causal set)
    best_models: pd.DataFrame
    model_priors: np.ndarray

    def summary(self) -> str:
        lines = [
            "MR-BMA — Bayesian Model Averaging over Exposures",
            "=" * 60,
            "",
            "Marginal inclusion probabilities:",
        ]
        for exp in self.exposures:
            lines.append(f"  {exp}: {self.marginal_inclusion[exp]:.4f}")
        lines += ["", "Top models (by posterior probability):"]
        lines.append(self.best_models.head(10).to_string(index=False))
        return "\n".join(lines)


# -------------------------------------------------------------------------
# Internals
# -------------------------------------------------------------------------


def _check_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")


def _ivw(beta_outcome, se_outcome, beta_exposure):
    """Inverse-variance-weighted estimator for a single exposure."""
    w = 1.0 / (se_outcome ** 2)
    return float(np.sum(w * beta_exposure * beta_outcome) / np.sum(w * beta_exposure ** 2))


# -------------------------------------------------------------------------
# 1) Multivariable MR
# -------------------------------------------------------------------------


def mr_multivariable(
    snp_associations: pd.DataFrame,
    *,
    outcome: str = "beta_y",
    outcome_se: str = "se_y",
    exposures: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> MVMRResult:
    """Multivariable Mendelian Randomization (Sanderson et al. 2019).

    Parameters
    ----------
    snp_associations : DataFrame
        One row per SNP. Must contain:

        - ``outcome`` — outcome-SNP association (β_Y).
        - ``outcome_se`` — its standard error.
        - exposure columns ``beta_X1, beta_X2, ...`` (configurable via
          the ``exposures`` argument).
    outcome, outcome_se : str
    exposures : sequence of str, optional
        Exposure-beta column names. If omitted, all ``beta_*`` columns
        other than the outcome are used.
    alpha : float, default 0.05

    Returns
    -------
    MVMRResult
        Contains direct-effect estimates for each exposure and
        conditional F-statistics (Sanderson-Windmeijer instrument-strength
        diagnostic).

    Notes
    -----
    Model:

        β_Y = Σ_j α_j β_{X_j} + ε,  ε ~ Normal(0, σ_Y²)

    fitted by weighted least squares with weights ``1 / se_y²``. The α_j
    are the direct causal effects, holding other exposures fixed.

    References
    ----------
    Sanderson, E., Davey Smith, G., Windmeijer, F. & Bowden, J. (2019).
    "An examination of multivariable Mendelian randomization in the
    single-sample and two-sample summary data settings." IJE 48(3).
    """
    if exposures is None:
        exposures = [c for c in snp_associations.columns
                     if c.startswith("beta_") and c != outcome]
    exposures = list(exposures)
    if len(exposures) < 2:
        raise ValueError(
            "MVMR requires >= 2 exposures. For a single exposure use "
            "sp.mendelian_randomization() or sp.mr_ivw()."
        )
    _check_columns(snp_associations, [outcome, outcome_se] + exposures)

    df = snp_associations.dropna(
        subset=[outcome, outcome_se] + exposures
    ).reset_index(drop=True)
    n = len(df)
    if n < len(exposures) + 5:
        raise ValueError(
            f"Need at least {len(exposures) + 5} SNPs for MVMR with "
            f"{len(exposures)} exposures; got {n}."
        )

    Y = df[outcome].to_numpy(dtype=float)
    se_y = df[outcome_se].to_numpy(dtype=float)
    X = df[list(exposures)].to_numpy(dtype=float)
    w = 1.0 / (se_y ** 2)
    W = np.diag(w)
    # WLS: alpha_hat = (X' W X)^-1 X' W Y
    XtWX = X.T @ W @ X
    XtWY = X.T @ W @ Y
    alpha_hat = np.linalg.solve(XtWX, XtWY)
    # Sandwich SE
    resid = Y - X @ alpha_hat
    sigma2 = float(np.sum(w * resid ** 2) / max(n - len(exposures), 1))
    var_alpha = sigma2 * np.linalg.inv(XtWX)
    se_alpha = np.sqrt(np.diag(var_alpha))
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for j, exp in enumerate(exposures):
        est = float(alpha_hat[j])
        se = float(se_alpha[j])
        rows.append({
            "exposure": exp,
            "estimate": est,
            "se": se,
            "ci_low": est - z * se,
            "ci_high": est + z * se,
            "p_value": float(2 * (1 - stats.norm.cdf(abs(est) / se))) if se > 0 else np.nan,
        })
    direct = pd.DataFrame(rows)

    # Conditional F-stats (Sanderson-Windmeijer 2016).  For exposure j,
    # regress beta_Xj on the OTHER exposures (weighted by IVW weights).
    # The F-stat is the SS explained by the residual exposure, divided
    # by the SS unexplained, scaled by degrees of freedom.
    #
    # Correct partition uses *mean-centered* SS, not the raw SS of X[:,j].
    # The numerator is the increment from adding X_j to a model that
    # already contains the other exposures (SS_reg_full − SS_reg_others),
    # which equals SS_resid_others − SS_resid_full.  We evaluate SS
    # weighted by ``w`` to match the WLS framework used for ``direct``.
    f_stats: Dict[str, float] = {}
    n_exp = len(exposures)
    sqrt_w = np.sqrt(w)
    for j, exp in enumerate(exposures):
        others = [k for k in range(n_exp) if k != j]
        if not others:
            f_stats[exp] = np.nan
            continue
        # WLS: regress sqrt(w) * X[:, j] on sqrt(w) * X[:, others] (with
        # intercept) so the F-stat matches the MVMR weighting scheme.
        y_j = sqrt_w * X[:, j]
        X_o = sqrt_w[:, None] * np.column_stack([
            np.ones(n), X[:, others]
        ])
        coef, *_ = np.linalg.lstsq(X_o, y_j, rcond=None)
        pred = X_o @ coef
        resid_j = y_j - pred
        # SS explained by full model (weighted): SS_total_centered − SS_resid
        y_mean = float(np.mean(y_j))
        ss_total = float(((y_j - y_mean) ** 2).sum())
        ss_resid = float((resid_j ** 2).sum())
        # F-stat for "X_j has non-zero explanatory power given others":
        #   F = (SS_total - SS_resid) / 1  ÷  (SS_resid / df2)
        df1 = 1
        df2 = max(n - n_exp, 1)
        if ss_resid < 1e-12 or df2 <= 0:
            f_stats[exp] = np.inf
        else:
            ss_explained = max(ss_total - ss_resid, 0.0)
            f_stats[exp] = float(ss_explained / df1 / (ss_resid / df2))

    return MVMRResult(
        exposures=exposures,
        direct_effect=direct,
        conditional_f_stats=f_stats,
        n_snps=n,
    )


# -------------------------------------------------------------------------
# 2) Mediation MR (two-step MR)
# -------------------------------------------------------------------------


def mr_mediation(
    snp_associations: pd.DataFrame,
    *,
    beta_exposure: str = "beta_x",
    se_exposure: str = "se_x",
    beta_mediator: str = "beta_m",
    se_mediator: str = "se_m",
    beta_outcome: str = "beta_y",
    se_outcome: str = "se_y",
    exposure_name: str = "exposure",
    mediator_name: str = "mediator",
    outcome_name: str = "outcome",
) -> MediationMRResult:
    """Two-step MR mediation: decompose total effect into direct + indirect.

    Parameters
    ----------
    snp_associations : DataFrame
        One row per SNP. Must contain SNP-associations with the exposure,
        mediator and outcome (both beta and SE).
    beta_exposure, se_exposure, beta_mediator, se_mediator, beta_outcome, se_outcome : str

    Returns
    -------
    MediationMRResult

    Notes
    -----
    Two-step approach (Burgess et al. 2015):

    1. **Step 1 — total effect (IVW)**: α_total = IVW(β_Y ~ β_X).
    2. **Step 2a — exposure → mediator**: α_XM = IVW(β_M ~ β_X).
    3. **Step 2b — direct effect via MVMR**: α_direct = MVMR(β_Y ~ β_X, β_M).
    4. **Indirect**: α_indirect = α_total − α_direct.

    Delta-method SE for the indirect effect combines SEs from steps 1
    and 3.

    References
    ----------
    Burgess, S., Daniel, R. M., Butterworth, A. S. & Thompson, S. G. (2015).
    "Network Mendelian randomization: using genetic variants as
    instrumental variables to investigate mediation in causal pathways."
    IJE 44(2).
    """
    cols = [beta_exposure, se_exposure, beta_mediator, se_mediator,
            beta_outcome, se_outcome]
    _check_columns(snp_associations, cols)
    df = snp_associations.dropna(subset=cols).reset_index(drop=True)
    n = len(df)
    if n < 5:
        raise ValueError(f"Need >= 5 SNPs for two-step MR; got {n}.")

    bX = df[beta_exposure].to_numpy(dtype=float)
    # β_M enters the direct-effect MVMR via the dataframe columns
    # below; no standalone ndarray needed here.
    bY = df[beta_outcome].to_numpy(dtype=float)
    seY = df[se_outcome].to_numpy(dtype=float)

    # Total effect
    w = 1.0 / seY ** 2
    total = float(np.sum(w * bX * bY) / np.sum(w * bX ** 2))
    total_se = float(np.sqrt(1.0 / np.sum(w * bX ** 2)))

    # Direct effect via MVMR on [β_X, β_M]
    mvmr_df = df[[beta_outcome, se_outcome, beta_exposure, beta_mediator]].rename(
        columns={beta_exposure: "beta_x", beta_mediator: "beta_m"}
    )
    mv = mr_multivariable(
        mvmr_df, outcome=beta_outcome, outcome_se=se_outcome,
        exposures=["beta_x", "beta_m"],
    )
    direct = float(mv.direct_effect.loc[mv.direct_effect["exposure"] == "beta_x", "estimate"].iloc[0])
    direct_se = float(mv.direct_effect.loc[mv.direct_effect["exposure"] == "beta_x", "se"].iloc[0])

    indirect = total - direct
    # Delta-method SE (approx): Var(A - B) = Var(A) + Var(B) (ignoring cov).
    indirect_se = float(np.sqrt(total_se ** 2 + direct_se ** 2))
    prop_mediated = (indirect / total) if abs(total) > 1e-12 else np.nan

    return MediationMRResult(
        exposure=exposure_name, mediator=mediator_name, outcome=outcome_name,
        total_effect=total, total_effect_se=total_se,
        direct_effect=direct, direct_effect_se=direct_se,
        indirect_effect=indirect, indirect_effect_se=indirect_se,
        proportion_mediated=prop_mediated,
    )


# -------------------------------------------------------------------------
# 3) MR-BMA (Zuber et al. 2020)
# -------------------------------------------------------------------------


def mr_bma(
    snp_associations: pd.DataFrame,
    *,
    outcome: str = "beta_y",
    outcome_se: str = "se_y",
    exposures: Optional[Sequence[str]] = None,
    max_model_size: Optional[int] = None,
    prior_inclusion: float = 0.5,
) -> MRBMAResult:
    """Mendelian Randomization with Bayesian Model Averaging.

    Parameters
    ----------
    snp_associations : DataFrame
    outcome, outcome_se, exposures : see :func:`mr_multivariable`
    max_model_size : int, optional
        Maximum number of exposures in any one model. Defaults to
        ``len(exposures)``. Set a smaller value for very high-dimensional
        exposure sets (restricts to at most k at a time).
    prior_inclusion : float, default 0.5
        Prior probability that each exposure is in the true causal model.

    Returns
    -------
    MRBMAResult
        Marginal inclusion probabilities and top posterior models.

    Notes
    -----
    Iterates over the 2^k − 1 non-empty subsets of ``exposures``. For
    each model, fits the MVMR WLS regression and computes the BIC:

        BIC_M = n * log(RSS_M / n) + k_M * log(n)

    Model posterior ∝ exp(-BIC_M / 2) * prior(M).

    Runs quickly for k ≤ 12. For more exposures use ``max_model_size``.

    References
    ----------
    Zuber, V., Colijn, J. M., Staley, J. R. & Burgess, S. (2020).
    "Selecting likely causal risk factors from high-throughput
    experiments using multivariable Mendelian randomization."
    Nature Communications 11, 29.
    """
    if exposures is None:
        exposures = [c for c in snp_associations.columns
                     if c.startswith("beta_") and c != outcome]
    exposures = list(exposures)
    k = len(exposures)
    if k < 2:
        raise ValueError("MR-BMA requires >= 2 exposures.")
    if k > 14:
        raise ValueError(
            f"MR-BMA exhaustive search limited to 14 exposures (got {k}). "
            "Use max_model_size to cap subset size."
        )
    if max_model_size is None:
        max_model_size = k
    _check_columns(snp_associations, [outcome, outcome_se] + exposures)
    df = snp_associations.dropna(
        subset=[outcome, outcome_se] + exposures
    ).reset_index(drop=True)
    n = len(df)
    if n < k + 5:
        raise ValueError(f"Need >= {k + 5} SNPs for MR-BMA with k={k}.")
    Y = df[outcome].to_numpy(dtype=float)
    se_y = df[outcome_se].to_numpy(dtype=float)
    X_all = df[list(exposures)].to_numpy(dtype=float)
    w = 1.0 / se_y ** 2
    log_n = float(np.log(n))

    posterior_scores: List[Tuple[Tuple[int, ...], float, float]] = []
    # Each entry: (model index tuple, bic, log_prior)
    for size in range(1, max_model_size + 1):
        for combo in combinations(range(k), size):
            X_m = X_all[:, list(combo)]
            # Weighted OLS
            XtWX = X_m.T @ (w[:, None] * X_m)
            XtWY = X_m.T @ (w * Y)
            try:
                alpha = np.linalg.solve(XtWX, XtWY)
            except np.linalg.LinAlgError:
                continue
            resid = Y - X_m @ alpha
            rss = float(np.sum(w * resid ** 2))
            # BIC (weighted-least-squares Gaussian approximation)
            bic = n * np.log(rss / n) + size * log_n
            # Beta-Binomial-style prior
            log_prior = (
                size * np.log(prior_inclusion)
                + (k - size) * np.log(1.0 - prior_inclusion)
            )
            posterior_scores.append((combo, bic, log_prior))

    if not posterior_scores:
        raise RuntimeError("MR-BMA: no models could be fit.")

    # Compute posterior probabilities (normalised via log-sum-exp).
    log_scores = np.array([-s[1] / 2.0 + s[2] for s in posterior_scores])
    log_scores -= log_scores.max()
    probs = np.exp(log_scores)
    probs /= probs.sum()

    # Marginal inclusion
    marginal = np.zeros(k)
    for (combo, _, _), p in zip(posterior_scores, probs):
        marginal[list(combo)] += p
    marginal_series = pd.Series(marginal, index=exposures)

    # Top models table
    best_idx = np.argsort(-probs)[:20]
    rows = []
    for i in best_idx:
        combo = posterior_scores[i][0]
        rows.append({
            "model": "+".join(exposures[j] for j in combo),
            "size": len(combo),
            "bic": posterior_scores[i][1],
            "posterior_prob": float(probs[i]),
        })
    best_models = pd.DataFrame(rows)

    return MRBMAResult(
        exposures=exposures,
        marginal_inclusion=marginal_series,
        best_models=best_models,
        model_priors=probs,
    )
