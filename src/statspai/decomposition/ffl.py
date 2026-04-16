"""
Firpo-Fortin-Lemieux (2007, 2009, 2018) two-step distributional decomposition.

Combines DFL reweighting (composition) with RIF regression (structural)
to obtain a *detailed* decomposition of a distributional gap at any
statistic (quantile, variance, Gini, Theil, IQR, log-variance).

Step 1: Reweight one group to match the other's covariate distribution.
Step 2: Run RIF regression on each group and on the reweighted sample.
        Decompose total gap into:
          - composition (explained by X)
          - structure (coefficient differences)
          - specification error (residual from step-1 reweighting)
          - reweighting error (residual from step-2 linearisation)

References
----------
Firpo, S., Fortin, N., & Lemieux, T. (2009). "Unconditional Quantile
Regressions." *Econometrica*, 77(3), 953-973.

Fortin, N., Lemieux, T., & Firpo, S. (2011). "Decomposition Methods in
Economics." *Handbook of Labor Economics*, Vol. 4A, Ch. 1.

Firpo, S., Fortin, N., & Lemieux, T. (2018). "Decomposing Wage
Distributions Using Recentered Influence Function Regressions."
*Econometrics*, 6(2), 28.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ._common import (
    add_constant,
    bootstrap_ci,
    bootstrap_stat,
    logit_fit,
    logit_predict,
    prepare_frame,
    sig_stars,
    weighted_quantile,
    wls,
)
from .dfl import _statistic_value
from .rif import rif_values


# ════════════════════════════════════════════════════════════════════════
# Result container
# ════════════════════════════════════════════════════════════════════════

@dataclass
class FFLResult:
    """Container for Firpo-Fortin-Lemieux two-step decomposition."""
    gap: float
    composition: float
    structure: float
    spec_error: float      # reweighting / specification error
    reweight_error: float  # RIF linearisation error
    stat: str
    tau: float
    detailed_composition: pd.DataFrame   # per-covariate composition
    detailed_structure: pd.DataFrame     # per-covariate structure
    stat_a: float
    stat_b: float
    stat_cf: float
    reference: int
    beta_a: pd.Series
    beta_b: pd.Series
    beta_cf: pd.Series
    se: Optional[Dict[str, float]] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None
    n_a: int = 0
    n_b: int = 0

    def summary(self) -> str:
        name = self.stat + (f" (τ={self.tau})" if self.stat == "quantile" else "")
        w = 68
        lines = [
            "━" * w,
            f"  Firpo-Fortin-Lemieux Two-Step Decomposition — {name}",
            "━" * w,
            f"  Group A (ref={self.reference == 0}): stat = {self.stat_a: .4f}  N={self.n_a}",
            f"  Group B:                          stat = {self.stat_b: .4f}  N={self.n_b}",
            f"  Counterfactual:                   stat = {self.stat_cf: .4f}",
            "",
            f"  Total gap:            {self.gap: .4f}",
            f"  Composition effect:   {self.composition: .4f}"
            + (f"   SE={self.se['composition']:.4f}" if self.se else ""),
            f"  Structure effect:     {self.structure: .4f}"
            + (f"   SE={self.se['structure']:.4f}" if self.se else ""),
            f"  Specification error:  {self.spec_error: .4f}",
            f"  Reweighting error:    {self.reweight_error: .4f}",
            "",
            "  Detailed composition (per covariate):",
            self.detailed_composition.round(4).to_string(index=False),
            "",
            "  Detailed structure (per covariate):",
            self.detailed_structure.round(4).to_string(index=False),
            "━" * w,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs):
        from .plots import ffl_waterfall
        return ffl_waterfall(self, **kwargs)

    def to_latex(self) -> str:
        lines = [
            r"\begin{table}[htbp]", r"\centering",
            f"\\caption{{FFL Two-Step Decomposition — {self.stat}}}",
            r"\begin{tabular}{lc}", r"\toprule",
            r"Component & Estimate \\", r"\midrule",
            f"Total gap & {self.gap:.4f} \\\\",
            f"Composition & {self.composition:.4f} \\\\",
            f"Structure & {self.structure:.4f} \\\\",
            f"Specification error & {self.spec_error:.4f} \\\\",
            f"Reweighting error & {self.reweight_error:.4f} \\\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return (
            "<div style='font-family: monospace;'>"
            f"<h3>FFL — {self.stat}</h3>"
            f"<p>Gap={self.gap:.4f}, Composition={self.composition:.4f}, "
            f"Structure={self.structure:.4f}</p></div>"
        )

    def __repr__(self) -> str:
        return (
            f"FFLResult(stat={self.stat}, gap={self.gap:.4f}, "
            f"composition={self.composition:.4f}, structure={self.structure:.4f})"
        )


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════

def _rif_for_sample(
    y: np.ndarray, w: np.ndarray, stat: str, tau: float,
) -> np.ndarray:
    """
    Compute RIF values of y at the *weighted* distribution's statistic.
    Most RIF formulas are unweighted; we approximate weighted RIF by
    using the weighted statistic as the "true" value and the weighted
    density estimator.
    """
    if stat == "quantile":
        q = float(weighted_quantile(y, tau, w=w))
        # Weighted kernel density at q
        n_eff = (w.sum() ** 2) / (w ** 2).sum()
        sigma = np.sqrt(max(np.cov(y, aweights=w), 1e-12))
        h = 1.06 * sigma * n_eff ** (-0.2)
        h = max(h, 1e-6)
        kern = np.exp(-0.5 * ((y - q) / h) ** 2) / (h * np.sqrt(2 * np.pi))
        f_q = float(np.average(kern, weights=w))
        f_q = max(f_q, 1e-12)
        return q + (tau - (y <= q).astype(float)) / f_q
    if stat == "mean":
        return y.copy()
    if stat == "variance":
        mu = float(np.average(y, weights=w))
        return (y - mu) ** 2
    if stat == "std":
        mu = float(np.average(y, weights=w))
        s2 = float(np.average((y - mu) ** 2, weights=w))
        s = np.sqrt(max(s2, 1e-12))
        return s + ((y - mu) ** 2 - s2) / (2 * s)
    if stat == "log_var":
        ly = np.log(np.clip(y, 1e-12, None))
        mu = float(np.average(ly, weights=w))
        return (ly - mu) ** 2
    if stat == "iqr":
        # RIF for IQR: Q75 − Q25 with two additive RIF pieces
        rif75 = _rif_for_sample(y, w, "quantile", 0.75)
        rif25 = _rif_for_sample(y, w, "quantile", 0.25)
        return rif75 - rif25
    if stat == "gini":
        # Use the unweighted influence-function form
        return rif_values(y, statistic="gini")
    if stat in ("theil_t", "theil_l", "atkinson"):
        # Approximate by numerical influence function
        return _numerical_rif(y, w, stat)
    raise ValueError(f"unknown statistic {stat!r}")


def _numerical_rif(y: np.ndarray, w: np.ndarray, stat: str,
                   eps: float = 1e-4) -> np.ndarray:
    """Numerical influence function: (ν(F + ε δ_{y_i}) − ν(F))/ε + ν(F)."""
    n = len(y)
    v0 = _statistic_value_generic(y, w, stat)
    rif = np.empty(n)
    for i in range(n):
        wi = w.copy()
        wi[i] = wi[i] + eps * w.sum()
        v1 = _statistic_value_generic(y, wi, stat)
        rif[i] = v0 + (v1 - v0) / eps
    return rif


def _statistic_value_generic(y: np.ndarray, w: np.ndarray, stat: str) -> float:
    """Generic weighted statistic (includes Theil/Atkinson)."""
    if stat == "theil_t":
        # T = E[(y/μ) log(y/μ)]
        yp = np.clip(y, 1e-12, None)
        mu = float(np.average(yp, weights=w))
        if mu <= 0:
            return float("nan")
        return float(np.average((yp / mu) * np.log(yp / mu), weights=w))
    if stat == "theil_l":
        # L = E[log(μ/y)] = log(μ) − E[log y]
        yp = np.clip(y, 1e-12, None)
        mu = float(np.average(yp, weights=w))
        return float(np.log(mu) - np.average(np.log(yp), weights=w))
    if stat == "atkinson":
        # Atkinson(ε=1) = 1 − exp(mean(log y)) / μ
        yp = np.clip(y, 1e-12, None)
        mu = float(np.average(yp, weights=w))
        return float(1.0 - np.exp(np.average(np.log(yp), weights=w)) / mu)
    return _statistic_value(y, w, stat)


# ════════════════════════════════════════════════════════════════════════
# Core FFL
# ════════════════════════════════════════════════════════════════════════

def ffl_decompose(
    data: pd.DataFrame,
    y: str,
    group: str,
    x: Sequence[str],
    stat: str = "quantile",
    tau: float = 0.5,
    reference: int = 0,
    weights: Optional[Union[str, np.ndarray]] = None,
    trim: float = 0.001,
    inference: str = "analytical",
    n_boot: int = 299,
    alpha: float = 0.05,
    seed: Optional[int] = 12345,
) -> FFLResult:
    """
    Firpo-Fortin-Lemieux two-step detailed distributional decomposition.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    group : str — binary {0, 1}
    x : Sequence[str]
    stat : {'quantile', 'mean', 'variance', 'std', 'iqr', 'gini',
            'log_var', 'theil_t', 'theil_l', 'atkinson'}
    tau : float (for quantile)
    reference : int {0, 1}
        0: B reweighted to look like A's X (composition = effect of A's X
           on B's outcomes relative to observed B)
    weights : str, array or None
    trim : float — propensity trim
    inference : {'analytical', 'bootstrap', 'none'}
    n_boot : int
    alpha : float
    seed : int or None
    """
    cols = [y, group] + list(x)
    df, w = prepare_frame(data, cols, weights=weights)
    g = df[group].astype(int).to_numpy()
    y_vec = df[y].to_numpy(dtype=float)
    X_raw = df[list(x)].to_numpy(dtype=float)
    X = add_constant(X_raw)

    mask_a = g == 0
    mask_b = g == 1
    y_a, y_b = y_vec[mask_a], y_vec[mask_b]
    X_a, X_b = X[mask_a], X[mask_b]
    w_a, w_b = w[mask_a], w[mask_b]

    if len(y_a) < 10 or len(y_b) < 10:
        raise ValueError("Need at least 10 obs per group.")

    # Step 1: DFL propensity reweighting
    X_pool = np.vstack([X_a, X_b])
    T_pool = np.concatenate([np.ones(len(X_a)), np.zeros(len(X_b))])
    w_pool = np.concatenate([w_a, w_b])
    beta_ps, _ = logit_fit(T_pool, X_pool, w=w_pool)
    p_hat = logit_predict(beta_ps, X_pool)
    p_hat = np.clip(p_hat, trim, 1 - trim)
    p_A = w_a.sum() / (w_a.sum() + w_b.sum())

    if reference == 0:
        p_b_part = p_hat[len(X_a):]
        psi_b = (p_b_part / (1 - p_b_part)) * ((1 - p_A) / p_A)
        w_cf = w_b * psi_b
        X_cf = X_b
        y_cf = y_b
    else:
        p_a_part = p_hat[: len(X_a)]
        psi_a = ((1 - p_a_part) / p_a_part) * (p_A / (1 - p_A))
        w_cf = w_a * psi_a
        X_cf = X_a
        y_cf = y_a

    # Observed stats
    stat_a = _statistic_value_generic(y_a, w_a, stat) if stat in (
        "theil_t", "theil_l", "atkinson"
    ) else _statistic_value(y_a, w_a, stat, tau)
    stat_b = _statistic_value_generic(y_b, w_b, stat) if stat in (
        "theil_t", "theil_l", "atkinson"
    ) else _statistic_value(y_b, w_b, stat, tau)
    stat_cf = _statistic_value_generic(y_cf, w_cf, stat) if stat in (
        "theil_t", "theil_l", "atkinson"
    ) else _statistic_value(y_cf, w_cf, stat, tau)

    gap = stat_a - stat_b

    # Step 2: RIF regressions on each sample
    rif_a = _rif_for_sample(y_a, w_a, stat, tau)
    rif_b = _rif_for_sample(y_b, w_b, stat, tau)
    rif_cf = _rif_for_sample(y_cf, w_cf, stat, tau)

    beta_a, _, _ = wls(rif_a, X_a, w=w_a)
    beta_b, _, _ = wls(rif_b, X_b, w=w_b)
    beta_cf, _, _ = wls(rif_cf, X_cf, w=w_cf)

    # Mean X for each sample (weighted)
    mean_Xa = np.average(X_a, axis=0, weights=w_a)
    mean_Xb = np.average(X_b, axis=0, weights=w_b)
    mean_Xcf = np.average(X_cf, axis=0, weights=w_cf)

    # Detailed FFL decomposition per Firpo-Fortin-Lemieux 2018.
    # When reference = 0, cf is "B reweighted to match A's X":
    #   Composition (X effect)  = (mean_Xcf − mean_Xb)' · β_B     [actual X change under B's structure]
    #   Structure (β effect)    = mean_Xa' · (β_A − β_cf)          [A's structure vs reweighted-B's]
    #   Spec error              = (mean_Xa − mean_Xcf)' · β_cf     [DFL reweighting residual → ~0 if logit OK]
    #   RW error                = mean_Xcf' · (β_cf − β_B)         [RIF linearisation under new weights]
    if reference == 0:
        composition_vec = (mean_Xcf - mean_Xb) * beta_b
        structure_vec = mean_Xa * (beta_a - beta_cf)
        spec_vec = (mean_Xa - mean_Xcf) * beta_cf
        rw_vec = mean_Xcf * (beta_cf - beta_b)
    else:
        # Reference = 1: cf is "A reweighted to match B's X"
        composition_vec = (mean_Xa - mean_Xcf) * beta_a
        structure_vec = mean_Xb * (beta_cf - beta_b)
        spec_vec = (mean_Xcf - mean_Xb) * beta_cf
        rw_vec = mean_Xcf * (beta_cf - beta_a)

    composition = float(composition_vec.sum())
    structure = float(structure_vec.sum())
    spec_error = float(spec_vec.sum())
    reweight_error = float(rw_vec.sum())

    var_names = ["_cons"] + list(x)

    # Build detailed tables (skip constant)
    det_comp = pd.DataFrame({
        "variable": var_names[1:],
        "composition": composition_vec[1:],
    })
    det_struct = pd.DataFrame({
        "variable": var_names[1:],
        "structure": structure_vec[1:],
    })
    # Constants aggregated (intercept contribution)
    det_comp_cons = composition_vec[0]
    det_struct_cons = structure_vec[0]
    det_struct = pd.concat([
        det_struct,
        pd.DataFrame({"variable": ["_cons"], "structure": [det_struct_cons]}),
    ], ignore_index=True)

    # Bootstrap
    se: Optional[Dict[str, float]] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None
    if inference == "bootstrap":
        rng = np.random.default_rng(seed)
        n = len(df)
        strata = np.where(mask_a, 0, 1)

        def stat_fn(idx: np.ndarray) -> np.ndarray:
            try:
                sub = df.iloc[idx]
                g_i = g[idx]
                if (g_i == 0).sum() < 10 or (g_i == 1).sum() < 10:
                    return np.array([np.nan] * 4)
                tmp_res = ffl_decompose(
                    sub, y=y, group=group, x=x, stat=stat, tau=tau,
                    reference=reference, weights=None, trim=trim,
                    inference="none", seed=None,
                )
                return np.array([tmp_res.gap, tmp_res.composition,
                                 tmp_res.structure, tmp_res.spec_error])
            except Exception:  # noqa: BLE001
                return np.array([np.nan] * 4)

        boot = bootstrap_stat(stat_fn, n, n_boot=n_boot, rng=rng, strata=strata)
        boot = boot[~np.isnan(boot).any(axis=1)]
        if len(boot) > 10:
            point = np.array([gap, composition, structure, spec_error])
            se_vec, lo, hi = bootstrap_ci(boot, point, alpha=alpha)
            se = {"gap": float(se_vec[0]), "composition": float(se_vec[1]),
                  "structure": float(se_vec[2]), "spec_error": float(se_vec[3])}
            ci = {"gap": (float(lo[0]), float(hi[0])),
                  "composition": (float(lo[1]), float(hi[1])),
                  "structure": (float(lo[2]), float(hi[2])),
                  "spec_error": (float(lo[3]), float(hi[3]))}

    return FFLResult(
        gap=float(gap), composition=composition, structure=structure,
        spec_error=spec_error, reweight_error=reweight_error,
        stat=stat, tau=tau,
        detailed_composition=det_comp, detailed_structure=det_struct,
        stat_a=float(stat_a), stat_b=float(stat_b), stat_cf=float(stat_cf),
        reference=reference,
        beta_a=pd.Series(beta_a, index=var_names),
        beta_b=pd.Series(beta_b, index=var_names),
        beta_cf=pd.Series(beta_cf, index=var_names),
        se=se, ci=ci, n_a=int(len(y_a)), n_b=int(len(y_b)),
    )
