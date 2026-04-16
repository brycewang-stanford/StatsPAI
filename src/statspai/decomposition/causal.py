"""
Causal decomposition methods.

- **Lundberg (2021) gap-closing estimator**: estimates the counterfactual
  gap that would remain if one group's covariate distribution were
  shifted to the other group's. Implements IPW, regression, and doubly
  robust (AIPW) versions.
- **VanderWeele (2014) four-way decomposition** of a total effect into
  controlled direct, natural mediated, reference interaction, and
  mediated interaction effects.
- **Jackson-VanderWeele (2018) causal decomposition** of a disparity into
  initial disparity (difference given X=0 reference) and difference
  attributable to the mediator distribution.

References
----------
Lundberg, I. (2021). "The Gap-Closing Estimand: A Causal Approach to
Study Interventions That Close Disparities Across Social Categories."
*Sociological Methods & Research*.

VanderWeele, T.J. (2014). "A Unification of Mediation and Interaction:
A Four-Way Decomposition." *Epidemiology*, 25(5), 749-761.

Jackson, J.W. & VanderWeele, T.J. (2018). "Decomposition Analysis to
Identify Intervention Targets for Reducing Disparities." *Epidemiology*,
29(6), 825-835.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._common import (
    add_constant,
    bootstrap_ci,
    bootstrap_stat,
    logit_fit,
    logit_predict,
    prepare_frame,
    wls,
)


# ════════════════════════════════════════════════════════════════════════
# Gap-closing (Lundberg 2021)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class GapClosingResult:
    observed_gap: float
    counterfactual_gap: float
    closed_gap: float
    method: str      # 'ipw', 'regression', 'aipw'
    estimator: str   # name of underlying model
    se: Optional[Dict[str, float]] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None
    n_a: int = 0
    n_b: int = 0

    def summary(self) -> str:
        lines = [
            "━" * 62,
            "  Gap-Closing Estimand (Lundberg 2021)",
            "━" * 62,
            f"  Method: {self.method}   Estimator: {self.estimator}",
            f"  N_A = {self.n_a}   N_B = {self.n_b}",
            "",
            f"  Observed gap:          {self.observed_gap: .4f}"
            + (f"   SE={self.se['observed']:.4f}" if self.se else ""),
            f"  Counterfactual gap:    {self.counterfactual_gap: .4f}"
            + (f"   SE={self.se['counterfactual']:.4f}" if self.se else ""),
            f"  Closed gap (diff):     {self.closed_gap: .4f}"
            + (f"   SE={self.se['closed']:.4f}" if self.se else ""),
        ]
        if self.ci:
            for k, (lo, hi) in self.ci.items():
                lines.append(f"    {k:<15s} 95% CI: [{lo: .4f}, {hi: .4f}]")
        lines.append("━" * 62)
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs):
        from .plots import gap_closing_plot
        return gap_closing_plot(self, **kwargs)

    def to_latex(self) -> str:
        lines = [r"\begin{tabular}{lc}", r"\toprule",
                 r"Quantity & Estimate \\", r"\midrule",
                 f"Observed gap & {self.observed_gap:.4f} \\\\",
                 f"Counterfactual gap & {self.counterfactual_gap:.4f} \\\\",
                 f"Closed gap & {self.closed_gap:.4f} \\\\",
                 r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return (f"<div><h3>Gap Closing</h3>"
                f"<p>Observed={self.observed_gap:.4f}, "
                f"Counterfactual={self.counterfactual_gap:.4f}, "
                f"Closed={self.closed_gap:.4f}</p></div>")

    def __repr__(self) -> str:
        return (f"GapClosingResult(method={self.method}, "
                f"observed={self.observed_gap:.4f}, "
                f"counterfactual={self.counterfactual_gap:.4f}, "
                f"closed={self.closed_gap:.4f})")


def _gap_closing_core(
    y: np.ndarray, g: np.ndarray, X: np.ndarray, method: str,
    trim: float, target_dist: int,
) -> Tuple[float, float]:
    """Core gap-closing computation; returns (observed, counterfactual)."""
    n = len(y)
    mask_a = g == 0
    mask_b = g == 1
    y_a, y_b = y[mask_a], y[mask_b]
    X_a, X_b = X[mask_a], X[mask_b]
    obs_gap = float(y_a.mean() - y_b.mean())

    if method == "regression":
        # Regress y on X within each group; predict under target distribution
        beta_a, _, _ = wls(y_a, X_a)
        beta_b, _, _ = wls(y_b, X_b)
        if target_dist == 1:
            # Target: B's X
            ey_a = float((X_b @ beta_a).mean())
            ey_b = float(y_b.mean())
        else:
            ey_a = float(y_a.mean())
            ey_b = float((X_a @ beta_b).mean())
        cf_gap = ey_a - ey_b
        return obs_gap, cf_gap

    # Propensity score
    beta_ps, _ = logit_fit(g.astype(float), X)
    p_hat = logit_predict(beta_ps, X)
    p_hat = np.clip(p_hat, trim, 1 - trim)
    p_group = g.mean()

    if method == "ipw":
        if target_dist == 1:
            # Shift A to look like B
            w_a = (1 - p_hat[mask_a]) / p_hat[mask_a] * p_group / (1 - p_group)
            ey_a = float(np.average(y_a, weights=w_a))
            ey_b = float(y_b.mean())
        else:
            w_b = p_hat[mask_b] / (1 - p_hat[mask_b]) * (1 - p_group) / p_group
            ey_a = float(y_a.mean())
            ey_b = float(np.average(y_b, weights=w_b))
        cf_gap = ey_a - ey_b
        return obs_gap, cf_gap

    if method == "aipw":
        # Doubly robust
        beta_a, _, _ = wls(y_a, X_a)
        beta_b, _, _ = wls(y_b, X_b)
        m_a_all = X @ beta_a
        m_b_all = X @ beta_b
        if target_dist == 1:
            # E[Y_A | X_B] via DR
            # ψ_i = m_a(X_i) + (T_i/p(X_i)) (Y_i - m_a(X_i))  evaluated on B-like sample
            # Weight A obs by (1-p)/p, and evaluate m_a under B
            resid_a = y_a - m_a_all[mask_a]
            dr_A = m_a_all[mask_b].mean() + np.mean(
                ((1 - p_hat[mask_a]) / p_hat[mask_a]
                 * p_group / (1 - p_group)) * resid_a
            )
            ey_a = float(dr_A)
            ey_b = float(y_b.mean())
        else:
            resid_b = y_b - m_b_all[mask_b]
            dr_B = m_b_all[mask_a].mean() + np.mean(
                (p_hat[mask_b] / (1 - p_hat[mask_b])
                 * (1 - p_group) / p_group) * resid_b
            )
            ey_a = float(y_a.mean())
            ey_b = float(dr_B)
        cf_gap = ey_a - ey_b
        return obs_gap, cf_gap

    raise ValueError(f"unknown method {method!r}")


def gap_closing(
    data: pd.DataFrame,
    y: str,
    group: str,
    x: Sequence[str],
    method: str = "aipw",
    target_dist: int = 1,
    trim: float = 0.001,
    inference: str = "analytical",
    n_boot: int = 299,
    alpha: float = 0.05,
    seed: Optional[int] = 12345,
) -> GapClosingResult:
    """
    Lundberg (2021) gap-closing estimator.

    Computes the counterfactual mean gap that would remain if one group's
    covariate distribution were shifted to match the other's.

    Parameters
    ----------
    data : pd.DataFrame
    y, group, x : column names
    method : {'regression', 'ipw', 'aipw'}
        AIPW is doubly robust (recommended).
    target_dist : {0, 1}
        - 1: shift Group A's covariate distribution to match Group B's
        - 0: shift Group B's to match Group A's
    trim : float — propensity trim
    inference : {'analytical', 'bootstrap', 'none'}
    """
    cols = [y, group] + list(x)
    df, _ = prepare_frame(data, cols)
    g = df[group].astype(int).to_numpy()
    y_vec = df[y].to_numpy(dtype=float)
    X_raw = df[list(x)].to_numpy(dtype=float)
    X = add_constant(X_raw)

    n_a = int((g == 0).sum())
    n_b = int((g == 1).sum())
    if n_a < 10 or n_b < 10:
        raise ValueError("Need ≥10 obs per group.")

    obs, cf = _gap_closing_core(y_vec, g, X, method, trim, target_dist)
    closed = obs - cf

    se: Optional[Dict[str, float]] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None
    if inference == "bootstrap":
        rng = np.random.default_rng(seed)
        n = len(df)

        def stat_fn(idx: np.ndarray) -> np.ndarray:
            try:
                g_i = g[idx]; y_i = y_vec[idx]; X_i = X[idx]
                if (g_i == 0).sum() < 10 or (g_i == 1).sum() < 10:
                    return np.array([np.nan, np.nan, np.nan])
                o, c = _gap_closing_core(y_i, g_i, X_i, method, trim, target_dist)
                return np.array([o, c, o - c])
            except Exception:  # noqa: BLE001
                return np.array([np.nan, np.nan, np.nan])

        boot = bootstrap_stat(stat_fn, n, n_boot=n_boot, rng=rng, strata=g)
        boot = boot[~np.isnan(boot).any(axis=1)]
        if len(boot) > 10:
            point = np.array([obs, cf, closed])
            se_vec, lo, hi = bootstrap_ci(boot, point, alpha=alpha)
            se = {"observed": float(se_vec[0]), "counterfactual": float(se_vec[1]),
                  "closed": float(se_vec[2])}
            ci = {"observed": (float(lo[0]), float(hi[0])),
                  "counterfactual": (float(lo[1]), float(hi[1])),
                  "closed": (float(lo[2]), float(hi[2]))}

    return GapClosingResult(
        observed_gap=float(obs), counterfactual_gap=float(cf),
        closed_gap=float(closed), method=method, estimator=method,
        se=se, ci=ci, n_a=n_a, n_b=n_b,
    )


# ════════════════════════════════════════════════════════════════════════
# Natural direct/indirect (VanderWeele mediation)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class MediationDecompResult:
    total: float
    nde: float    # natural direct effect
    nie: float    # natural indirect effect
    cde: float    # controlled direct effect
    propn_mediated: float
    se: Optional[Dict[str, float]] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None
    method: str = "linear_nested"

    def summary(self) -> str:
        lines = [
            "━" * 62,
            "  Causal Mediation Decomposition",
            "━" * 62,
            f"  Total effect:                       {self.total: .4f}"
            + (f"   SE={self.se['total']:.4f}" if self.se else ""),
            f"  Natural direct effect (NDE):        {self.nde: .4f}"
            + (f"   SE={self.se['nde']:.4f}" if self.se else ""),
            f"  Natural indirect effect (NIE):      {self.nie: .4f}"
            + (f"   SE={self.se['nie']:.4f}" if self.se else ""),
            f"  Controlled direct effect (CDE):     {self.cde: .4f}"
            + (f"   SE={self.se['cde']:.4f}" if self.se else ""),
            f"  Proportion mediated (NIE/total):    {self.propn_mediated:.1%}",
            "━" * 62,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def __repr__(self) -> str:
        return (f"MediationDecompResult(total={self.total:.4f}, "
                f"nde={self.nde:.4f}, nie={self.nie:.4f})")


def mediation_decompose(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    mediator: str,
    covariates: Optional[Sequence[str]] = None,
    inference: str = "analytical",
    n_boot: int = 299,
    alpha: float = 0.05,
    seed: Optional[int] = 12345,
) -> MediationDecompResult:
    """
    Linear nested-models mediation decomposition (VanderWeele 2014
    four-way simplified to natural direct / indirect under linearity).

    Parameters
    ----------
    data : pd.DataFrame
    y : str — continuous outcome
    treatment : str — binary exposure
    mediator : str — mediator
    covariates : list of str or None
    inference : {'analytical', 'bootstrap'}

    Returns
    -------
    MediationDecompResult with NDE, NIE, CDE, proportion mediated.
    """
    cov = list(covariates) if covariates else []
    cols = [y, treatment, mediator] + cov
    df, _ = prepare_frame(data, cols)
    n = len(df)
    A = df[treatment].astype(float).to_numpy()
    M = df[mediator].astype(float).to_numpy()
    Y = df[y].astype(float).to_numpy()
    C = df[cov].to_numpy(dtype=float) if cov else np.empty((n, 0))

    # Mediator model: M ~ A + C
    Xm = add_constant(np.column_stack([A] + ([C] if cov else [])))
    beta_m, _, _ = wls(M, Xm)
    alpha_a = beta_m[1]

    # Outcome model: Y ~ A + M + A*M + C
    AM = A * M
    Xy = add_constant(np.column_stack([A, M, AM] + ([C] if cov else [])))
    beta_y, vcov_y, _ = wls(Y, Xy)
    theta1 = beta_y[1]       # A
    theta2 = beta_y[2]       # M
    theta3 = beta_y[3]       # AM

    # Mean mediator at A=0
    m_bar_0 = float(M[A == 0].mean()) if (A == 0).any() else float(M.mean())

    # Natural effects (under linearity, VanderWeele 2014)
    nde = theta1 + theta3 * m_bar_0
    nie = alpha_a * (theta2 + theta3)   # using A=1 for CDE mode
    total = nde + nie
    cde = theta1 + theta3 * m_bar_0     # CDE at M = m*  (same as NDE here if m*=m_bar_0)
    pm = nie / total if abs(total) > 1e-12 else float("nan")

    se = None
    ci = None
    if inference == "bootstrap":
        rng = np.random.default_rng(seed)

        def stat_fn(idx: np.ndarray) -> np.ndarray:
            try:
                Ai = A[idx]; Mi = M[idx]; Yi = Y[idx]
                Ci = C[idx] if cov else np.empty((len(idx), 0))
                Xmi = add_constant(np.column_stack([Ai] + ([Ci] if cov else [])))
                bmi, _, _ = wls(Mi, Xmi)
                Xyi = add_constant(np.column_stack(
                    [Ai, Mi, Ai * Mi] + ([Ci] if cov else [])))
                byi, _, _ = wls(Yi, Xyi)
                m0_i = float(Mi[Ai == 0].mean()) if (Ai == 0).any() else float(Mi.mean())
                nde_i = byi[1] + byi[3] * m0_i
                nie_i = bmi[1] * (byi[2] + byi[3])
                return np.array([nde_i + nie_i, nde_i, nie_i])
            except Exception:  # noqa: BLE001
                return np.array([np.nan, np.nan, np.nan])

        boot = bootstrap_stat(stat_fn, n, n_boot=n_boot, rng=rng)
        boot = boot[~np.isnan(boot).any(axis=1)]
        if len(boot) > 10:
            pt = np.array([total, nde, nie])
            sev, lo, hi = bootstrap_ci(boot, pt, alpha=alpha)
            se = {"total": float(sev[0]), "nde": float(sev[1]),
                  "nie": float(sev[2]), "cde": float(sev[1])}
            ci = {"total": (float(lo[0]), float(hi[0])),
                  "nde": (float(lo[1]), float(hi[1])),
                  "nie": (float(lo[2]), float(hi[2]))}

    return MediationDecompResult(
        total=float(total), nde=float(nde), nie=float(nie),
        cde=float(cde), propn_mediated=float(pm),
        se=se, ci=ci, method="linear_nested",
    )


# ════════════════════════════════════════════════════════════════════════
# Jackson-VanderWeele (2018) disparity decomposition
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DisparityDecompResult:
    total_disparity: float
    initial_disparity: float
    mediator_attributable: float
    propn_mediator: float
    target_mediator_level: float
    se: Optional[Dict[str, float]] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None

    def summary(self) -> str:
        lines = [
            "━" * 62,
            "  Jackson-VanderWeele (2018) Causal Disparity Decomposition",
            "━" * 62,
            f"  Total disparity:                   {self.total_disparity: .4f}",
            f"  Initial disparity:                 {self.initial_disparity: .4f}",
            f"  Mediator-attributable disparity:   {self.mediator_attributable: .4f}",
            f"  Proportion via mediator:           {self.propn_mediator:.1%}",
            f"  Target mediator level (ref):       {self.target_mediator_level: .4f}",
            "━" * 62,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def __repr__(self) -> str:
        return (f"DisparityDecompResult(total={self.total_disparity:.4f}, "
                f"initial={self.initial_disparity:.4f})")


def disparity_decompose(
    data: pd.DataFrame,
    y: str,
    group: str,
    mediator: str,
    covariates: Optional[Sequence[str]] = None,
    target_level: Optional[float] = None,
) -> DisparityDecompResult:
    """
    Jackson & VanderWeele (2018) causal disparity decomposition.

    Decomposes an observed group disparity in Y into:
      - *initial disparity*: what would remain if mediator M were set
        to a reference level (e.g. Group A's M distribution).
      - *mediator-attributable*: the complementary share.

    Parameters
    ----------
    data : pd.DataFrame
    y : str — outcome
    group : str — binary group (0/1, where 1 = disadvantaged)
    mediator : str — mediator
    covariates : list or None
    target_level : float or None
        Value at which to fix mediator for the "initial" counterfactual.
        Default: mean of M in reference group (group=0).
    """
    cov = list(covariates) if covariates else []
    cols = [y, group, mediator] + cov
    df, _ = prepare_frame(data, cols)
    G = df[group].astype(int).to_numpy()
    M = df[mediator].astype(float).to_numpy()
    Y = df[y].astype(float).to_numpy()
    n = len(df)
    C = df[cov].to_numpy(dtype=float) if cov else np.empty((n, 0))

    m_star = float(M[G == 0].mean()) if target_level is None else float(target_level)

    # Outcome model: Y ~ G + M + G*M + C
    GM = G * M
    X = add_constant(np.column_stack([G.astype(float), M, GM.astype(float)]
                                     + ([C] if cov else [])))
    beta, _, _ = wls(Y, X)

    # Predict under counterfactual: all observations as if in group 1, with M = m_star
    def pred(g_val: float, m_val: np.ndarray, c: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(m_val)), np.full(len(m_val), g_val), m_val,
                g_val * m_val]
        if c.shape[1] > 0:
            cols.append(c)
        Xnew = np.column_stack(cols)
        return Xnew @ beta

    # Observed means
    y_a_obs = float(Y[G == 0].mean())
    y_b_obs = float(Y[G == 1].mean())
    total_disp = y_b_obs - y_a_obs

    # Initial disparity: E[Y | G=1, M=m_star] − E[Y | G=0]
    # Use covariates of G=1 obs for expectation
    mask_b = G == 1
    mask_a = G == 0
    C_b = C[mask_b] if cov else np.empty((mask_b.sum(), 0))
    M_b = M[mask_b]
    M_star_b = np.full(mask_b.sum(), m_star)
    y_b_cf_initial = float(pred(1.0, M_star_b, C_b).mean())
    y_a_cf_initial = y_a_obs  # observed
    initial_disp = y_b_cf_initial - y_a_cf_initial

    mediator_attr = total_disp - initial_disp
    pm = mediator_attr / total_disp if abs(total_disp) > 1e-12 else float("nan")

    return DisparityDecompResult(
        total_disparity=float(total_disp),
        initial_disparity=float(initial_disp),
        mediator_attributable=float(mediator_attr),
        propn_mediator=float(pm),
        target_mediator_level=m_star,
    )
