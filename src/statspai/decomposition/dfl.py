"""
DiNardo-Fortin-Lemieux (1996) reweighting decomposition.

Reweights one group's sample so that its covariate distribution matches
the other group's, creating a counterfactual distribution. The gap
between a statistic (quantile, mean, variance, Gini, CDF, density) in
the observed vs counterfactual distribution identifies the *composition
effect*; the remainder is the *structural* (wage-structure) effect.

References
----------
DiNardo, J., Fortin, N., & Lemieux, T. (1996). "Labor Market Institutions
and the Distribution of Wages, 1973-1992: A Semiparametric Approach."
*Econometrica*, 64(5), 1001-1044.

Fortin, N., Lemieux, T., & Firpo, S. (2011). "Decomposition Methods in
Economics." In *Handbook of Labor Economics*, Vol. 4A, Ch. 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._common import (
    add_constant,
    bootstrap_ci,
    bootstrap_stat,
    logit_fit,
    logit_predict,
    parse_formula,
    prepare_frame,
    sig_stars,
    statistic_value as _statistic_value,
    weighted_ecdf,
    weighted_gini as _weighted_gini,
    weighted_quantile,
)


# ════════════════════════════════════════════════════════════════════════
# Core DFL reweighter
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DFLResult:
    """Container for DFL reweighting decomposition results."""
    gap: float
    composition: float
    structure: float
    stat: str
    tau: float
    stat_a: float
    stat_b: float
    stat_cf: float            # counterfactual: A's X distribution, B's Y conditional
    reference: int
    weights_cf: np.ndarray
    se: Optional[Dict[str, float]] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None
    quantile_grid: Optional[pd.DataFrame] = field(default=None)
    propensity_coef: Optional[pd.Series] = field(default=None)
    n_a: int = 0
    n_b: int = 0

    def summary(self) -> str:
        name = self.stat + (f" (τ={self.tau})" if self.stat == "quantile" else "")
        lines = [
            "━" * 62,
            f"  DFL Reweighting Decomposition — {name}",
            "━" * 62,
            f"  Group A: stat = {self.stat_a: .4f}   N = {self.n_a}",
            f"  Group B: stat = {self.stat_b: .4f}   N = {self.n_b}",
            f"  Counterfactual (A's X, B's structure): {self.stat_cf: .4f}",
            "",
            f"  Total gap (A − B):   {self.gap: .4f}",
        ]
        if self.se:
            lines.append(
                f"  Composition effect:  {self.composition: .4f}   "
                f"SE={self.se.get('composition', float('nan')): .4f}"
            )
            lines.append(
                f"  Structure effect:    {self.structure: .4f}   "
                f"SE={self.se.get('structure', float('nan')): .4f}"
            )
        else:
            lines.append(f"  Composition effect:  {self.composition: .4f}")
            lines.append(f"  Structure effect:    {self.structure: .4f}")
        if self.ci:
            for k, (lo, hi) in self.ci.items():
                lines.append(f"    {k:<14s} 95% CI: [{lo: .4f}, {hi: .4f}]")
        lines.append("━" * 62)
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs):
        """Delegate to plots.dfl_plot()."""
        from .plots import dfl_plot
        return dfl_plot(self, **kwargs)

    def to_latex(self) -> str:
        name = self.stat
        if self.stat == "quantile":
            name = f"quantile (τ={self.tau})"
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{DFL Reweighting Decomposition — {name}}}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Component & Estimate \\",
            r"\midrule",
            f"Total gap & {self.gap:.4f} \\\\",
            f"Composition & {self.composition:.4f} \\\\",
            f"Structure & {self.structure:.4f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return (
            "<div style='font-family: monospace;'>"
            f"<h3>DFL Decomposition — {self.stat}"
            + (f" (τ={self.tau})" if self.stat == "quantile" else "")
            + "</h3>"
            "<table>"
            f"<tr><td>Total gap</td><td>{self.gap:.4f}</td></tr>"
            f"<tr><td>Composition</td><td>{self.composition:.4f}</td></tr>"
            f"<tr><td>Structure</td><td>{self.structure:.4f}</td></tr>"
            "</table></div>"
        )

    def __repr__(self) -> str:
        return (
            f"DFLResult(stat={self.stat}, gap={self.gap:.4f}, "
            f"composition={self.composition:.4f}, structure={self.structure:.4f})"
        )


def _dfl_core(
    y_a: np.ndarray, X_a: np.ndarray, w_a: np.ndarray,
    y_b: np.ndarray, X_b: np.ndarray, w_b: np.ndarray,
    stat: str, tau: float, reference: int,
    trim: float = 0.001,
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """
    Core DFL computation, returning:
      gap, composition, structure, stat_a, stat_b, stat_cf, psi_weights, logit_beta
    """
    # Build pooled design with treatment indicator: T=1 for A, 0 for B
    X_pool = np.vstack([X_a, X_b])
    T_pool = np.concatenate([np.ones(len(X_a)), np.zeros(len(X_b))])
    w_pool = np.concatenate([w_a, w_b])

    beta_ps, _ = logit_fit(T_pool, X_pool, w=w_pool)
    p_hat = logit_predict(beta_ps, X_pool)
    p_hat = np.clip(p_hat, trim, 1 - trim)

    # Mixing proportion
    p_A = w_a.sum() / (w_a.sum() + w_b.sum())

    if reference == 0:
        # Counterfactual: B's outcomes reweighted to look like A's X
        # ψ(X) = P(A|X)/P(B|X) · P(B)/P(A)
        p_b = p_hat[len(X_a):]
        psi = (p_b / (1 - p_b)) * ((1 - p_A) / p_A)
        w_cf = w_b * psi
        y_cf = y_b
    else:
        # Counterfactual: A's outcomes reweighted to look like B's X
        p_a = p_hat[: len(X_a)]
        psi = ((1 - p_a) / p_a) * (p_A / (1 - p_A))
        w_cf = w_a * psi
        y_cf = y_a

    stat_a = _statistic_value(y_a, w_a, stat, tau)
    stat_b = _statistic_value(y_b, w_b, stat, tau)
    stat_cf = _statistic_value(y_cf, w_cf, stat, tau)

    gap = stat_a - stat_b
    if reference == 0:
        # B reweighted to A's X ⇒ cf ≡ F_{Y<1|0>}: A's X, B's structure.
        # Under this cf only β differs between A and cf ⇒ structure effect;
        # only X differs between cf and B ⇒ composition effect.
        structure = stat_a - stat_cf
        composition = stat_cf - stat_b
    else:
        # A reweighted to B's X ⇒ cf ≡ F_{Y<0|1>}: B's X, A's structure.
        # Only X differs between A and cf ⇒ composition effect;
        # only β differs between cf and B ⇒ structure effect.
        composition = stat_a - stat_cf
        structure = stat_cf - stat_b

    return gap, composition, structure, stat_a, stat_b, stat_cf, w_cf, beta_ps


def dfl_decompose(
    data: pd.DataFrame,
    y: str,
    group: str,
    x: Sequence[str],
    stat: str = "mean",
    tau: float = 0.5,
    reference: int = 0,
    weights: Optional[Union[str, np.ndarray]] = None,
    trim: float = 0.001,
    inference: str = "analytical",
    n_boot: int = 299,
    alpha: float = 0.05,
    quantile_grid: Optional[Sequence[float]] = None,
    seed: Optional[int] = 12345,
) -> DFLResult:
    """
    DFL (1996) reweighting decomposition at a chosen distributional statistic.

    Parameters
    ----------
    data : pd.DataFrame
    y : str — outcome variable name
    group : str — binary (0/1) group indicator
    x : Sequence[str] — covariates used for propensity model
    stat : {'mean', 'variance', 'std', 'quantile', 'iqr', 'gini', 'log_var'}
    tau : float — quantile level (when stat='quantile')
    reference : {0, 1}
        - 0: reweight Group B to look like A's X (default). The
          counterfactual is F_{Y<1|0>} — A's X distribution with B's
          outcome structure.
        - 1: reweight Group A to look like B's X. The counterfactual
          is F_{Y<0|1>} — B's X distribution with A's outcome structure.

        .. warning::
           ``reference`` has different economic semantics across method
           families. In DFL, ``reference=0`` yields cf = *A's X, B's β*
           (reweighting approach). In ``machado_mata`` / ``melly`` /
           ``cfm``, ``reference=0`` yields cf = *A's β, B's X*
           (coefficient-substitution approach). These are **opposite**
           counterfactual constructions. Within each method labels are
           internally consistent (DFL structure = A − cf; MM
           composition = A − cf). When comparing estimates across
           methods, read the per-method docstrings carefully.
    weights : str, array or None — sample weights
    trim : float — clip propensity scores to [trim, 1-trim]
    inference : {'none', 'bootstrap', 'analytical'}
    n_boot : int — bootstrap replications
    alpha : float — CI level
    quantile_grid : sequence of τ ∈ (0, 1) or None
        If provided, also compute quantile-process decomposition on this grid.
    seed : int or None

    Returns
    -------
    DFLResult
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

    if len(y_a) < 5 or len(y_b) < 5:
        raise ValueError("Need at least 5 obs per group.")

    gap, comp, struct, s_a, s_b, s_cf, w_cf, beta_ps = _dfl_core(
        y_a, X_a, w_a, y_b, X_b, w_b, stat, tau, reference, trim=trim
    )

    se: Optional[Dict[str, float]] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None

    if inference == "bootstrap":
        rng = np.random.default_rng(seed)
        n = len(df)
        idx_a = np.where(mask_a)[0]
        idx_b = np.where(mask_b)[0]
        strata = np.where(mask_a, 0, 1)

        def stat_fn(idx: np.ndarray) -> np.ndarray:
            g_i = g[idx]
            y_i = y_vec[idx]
            X_i = X[idx]
            w_i = w[idx]
            m_a = g_i == 0
            m_b = g_i == 1
            if m_a.sum() < 5 or m_b.sum() < 5:
                return np.array([np.nan, np.nan, np.nan])
            try:
                _g, _c, _s, *_ = _dfl_core(
                    y_i[m_a], X_i[m_a], w_i[m_a],
                    y_i[m_b], X_i[m_b], w_i[m_b],
                    stat, tau, reference, trim=trim,
                )
                return np.array([_g, _c, _s])
            except Exception:  # noqa: BLE001
                return np.array([np.nan, np.nan, np.nan])

        boot = bootstrap_stat(stat_fn, n, n_boot=n_boot, rng=rng, strata=strata)
        boot = boot[~np.isnan(boot).any(axis=1)]
        if len(boot) > 10:
            point = np.array([gap, comp, struct])
            se_vec, lo, hi = bootstrap_ci(boot, point, alpha=alpha, method="percentile")
            se = {"gap": float(se_vec[0]), "composition": float(se_vec[1]),
                  "structure": float(se_vec[2])}
            ci = {"gap": (float(lo[0]), float(hi[0])),
                  "composition": (float(lo[1]), float(hi[1])),
                  "structure": (float(lo[2]), float(hi[2]))}

    # Optional quantile process
    qproc_df: Optional[pd.DataFrame] = None
    if quantile_grid is not None and stat == "quantile":
        rows = []
        for t in quantile_grid:
            s_a_t = _statistic_value(y_a, w_a, "quantile", t)
            s_b_t = _statistic_value(y_b, w_b, "quantile", t)
            if reference == 0:
                s_cf_t = _statistic_value(y_b, w_cf, "quantile", t)
                struct_t = s_a_t - s_cf_t
                comp_t = s_cf_t - s_b_t
            else:
                s_cf_t = _statistic_value(y_a, w_cf, "quantile", t)
                comp_t = s_a_t - s_cf_t
                struct_t = s_cf_t - s_b_t
            rows.append({"tau": t, "gap": s_a_t - s_b_t,
                         "composition": comp_t, "structure": struct_t,
                         "stat_a": s_a_t, "stat_b": s_b_t, "stat_cf": s_cf_t})
        qproc_df = pd.DataFrame(rows)

    var_names = ["_cons"] + list(x)
    return DFLResult(
        gap=float(gap), composition=float(comp), structure=float(struct),
        stat=stat, tau=tau,
        stat_a=float(s_a), stat_b=float(s_b), stat_cf=float(s_cf),
        reference=reference, weights_cf=w_cf, se=se, ci=ci,
        quantile_grid=qproc_df,
        propensity_coef=pd.Series(beta_ps, index=var_names),
        n_a=int(len(y_a)), n_b=int(len(y_b)),
    )
