"""
Machado-Mata (2005) quantile regression decomposition.

Simulate counterfactual outcome distributions by combining one group's
quantile regression coefficients with another group's covariate
distribution. Decompose quantile gaps into composition (X distribution)
and coefficient (price / structural) effects at each τ.

References
----------
Machado, J.A.F. & Mata, J. (2005). "Counterfactual Decomposition of Changes
in Wage Distributions Using Quantile Regression." *Journal of Applied
Econometrics*, 20(4), 445-465. [@machado2005counterfactual]

Albrecht, Björklund, Vroman (2003). "Is There a Glass Ceiling in Sweden?"
*Journal of Labor Economics*, 21(1), 145-177. [@albrecht2003there]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from ._common import (
    add_constant,
    bootstrap_ci,
    bootstrap_stat,
    parse_formula,
    prepare_frame,
    sig_stars,
    weighted_quantile,
)


# ════════════════════════════════════════════════════════════════════════
# Quantile regression via IRLS (Koenker)
# ════════════════════════════════════════════════════════════════════════

def _qreg_irls(
    y: np.ndarray,
    X: np.ndarray,
    tau: float,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Quantile regression via Iteratively Reweighted Least Squares.

    Uses the weight w_i = 1/max(|resid_i|, eps) and adjusts residuals
    for asymmetry (Koenker's algorithm, adequate but not optimal).
    For higher accuracy one could wire to scipy.optimize or statsmodels.
    """
    n, k = X.shape
    # initialise with OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    eps = 1e-6
    for _ in range(max_iter):
        resid = y - X @ beta
        # asymmetric check function weights
        w = np.where(resid >= 0, tau, 1 - tau) / np.maximum(np.abs(resid), eps)
        # Weighted LS
        WX = X * w[:, None]
        try:
            beta_new = np.linalg.solve(X.T @ WX, X.T @ (w * y))
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(X.T @ WX, X.T @ (w * y), rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _qreg_grid(
    y: np.ndarray,
    X: np.ndarray,
    tau_grid: np.ndarray,
) -> np.ndarray:
    """Run quantile regression at each τ, return (n_tau, k) coefficient matrix."""
    out = np.empty((len(tau_grid), X.shape[1]))
    for i, t in enumerate(tau_grid):
        out[i] = _qreg_irls(y, X, t)
    return out


# ════════════════════════════════════════════════════════════════════════
# Result
# ════════════════════════════════════════════════════════════════════════

@dataclass
class MachadoMataResult:
    """Container for Machado-Mata decomposition."""
    quantile_grid: pd.DataFrame   # τ, q_a, q_b, q_cf, gap, composition, structure
    overall: Dict[str, float]     # aggregated across grid
    reference: int
    n_sim: int
    n_a: int
    n_b: int
    se: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        g = self.quantile_grid
        lines = [
            "━" * 72,
            "  Machado-Mata Quantile Decomposition",
            "━" * 72,
            f"  N_A = {self.n_a}   N_B = {self.n_b}   "
            f"simulations = {self.n_sim}   reference = {self.reference}",
            "",
            f"  {'tau':>6s} {'q_A':>9s} {'q_B':>9s} {'q_CF':>9s} "
            f"{'gap':>9s} {'comp':>9s} {'struct':>9s}",
        ]
        for _, row in g.iterrows():
            lines.append(
                f"  {row['tau']:>6.2f} {row['q_a']:>9.4f} {row['q_b']:>9.4f} "
                f"{row['q_cf']:>9.4f} {row['gap']:>9.4f} "
                f"{row['composition']:>9.4f} {row['structure']:>9.4f}"
            )
        lines.append("━" * 72)
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs):
        from .plots import quantile_process_plot
        return quantile_process_plot(self, **kwargs)

    def to_latex(self) -> str:
        g = self.quantile_grid
        lines = [r"\begin{table}[htbp]", r"\centering",
                 r"\caption{Machado-Mata Decomposition}",
                 r"\begin{tabular}{ccccccc}", r"\toprule",
                 r"$\tau$ & $q_A$ & $q_B$ & $q_{cf}$ & Gap & Comp. & Struct. \\",
                 r"\midrule"]
        for _, row in g.iterrows():
            lines.append(
                f"{row['tau']:.2f} & {row['q_a']:.4f} & {row['q_b']:.4f} & "
                f"{row['q_cf']:.4f} & {row['gap']:.4f} & {row['composition']:.4f} "
                f"& {row['structure']:.4f} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return (
            "<div style='font-family:monospace;'>"
            "<h3>Machado-Mata Decomposition</h3>"
            + self.quantile_grid.round(4).to_html(index=False)
            + "</div>"
        )

    def __repr__(self) -> str:
        return (
            f"MachadoMataResult(n_tau={len(self.quantile_grid)}, "
            f"reference={self.reference}, n_sim={self.n_sim})"
        )


# ════════════════════════════════════════════════════════════════════════
# Core function
# ════════════════════════════════════════════════════════════════════════

def machado_mata(
    data: pd.DataFrame,
    y: str,
    group: str,
    x: Sequence[str],
    tau_grid: Optional[Sequence[float]] = None,
    reference: int = 0,
    n_sim: int = 500,
    n_tau_qr: int = 99,
    inference: str = "none",
    n_boot: int = 199,
    alpha: float = 0.05,
    seed: Optional[int] = 12345,
) -> MachadoMataResult:
    """
    Machado-Mata (2005) quantile decomposition.

    Parameters
    ----------
    data : pd.DataFrame
    y, group, x : column names
    tau_grid : Sequence[float] or None
        τ grid for reporting (default: deciles 0.1..0.9)
    reference : {0, 1}
        0: use Group A's coefficients with Group B's X. The
           counterfactual is F_{Y<0|1>} — A's β on B's X.
        1: use Group B's coefficients with Group A's X.

        .. warning::
           Opposite convention to ``dfl_decompose``. In DFL,
           ``reference=0`` means *A's X, B's β* (reweighting). Here,
           ``reference=0`` means *A's β, B's X* (coefficient swap).
           See ``dfl_decompose`` docstring for the full convention map.
    n_sim : int — number of (τ, obs) draws per counterfactual
    n_tau_qr : int — τ grid resolution for quantile regression estimation
    inference : {'none', 'bootstrap'}
    n_boot : int
    alpha : float
    seed : int or None

    Returns
    -------
    MachadoMataResult
    """
    cols = [y, group] + list(x)
    df, _ = prepare_frame(data, cols)
    g = df[group].astype(int).to_numpy()
    y_vec = df[y].to_numpy(dtype=float)
    X_raw = df[list(x)].to_numpy(dtype=float)

    X_a = add_constant(X_raw[g == 0])
    X_b = add_constant(X_raw[g == 1])
    y_a = y_vec[g == 0]
    y_b = y_vec[g == 1]

    if len(y_a) < 20 or len(y_b) < 20:
        raise ValueError("Need ≥20 obs per group for Machado-Mata.")

    if tau_grid is None:
        tau_grid = np.round(np.arange(0.1, 0.95, 0.1), 2)
    tau_grid = np.asarray(tau_grid, dtype=float)

    tau_qr = np.linspace(0.01, 0.99, n_tau_qr)
    beta_a_grid = _qreg_grid(y_a, X_a, tau_qr)
    beta_b_grid = _qreg_grid(y_b, X_b, tau_qr)

    rng = np.random.default_rng(seed)

    def simulate(beta_grid, X_source, n):
        """Draw n times: random τ, random row from X_source, predict y."""
        n_src = X_source.shape[0]
        t_idx = rng.integers(0, len(tau_qr), size=n)
        r_idx = rng.integers(0, n_src, size=n)
        b = beta_grid[t_idx]        # (n, k)
        xrow = X_source[r_idx]      # (n, k)
        return np.sum(b * xrow, axis=1)

    # Simulated (marginal) distributions
    y_a_sim = simulate(beta_a_grid, X_a, n_sim)
    y_b_sim = simulate(beta_b_grid, X_b, n_sim)
    if reference == 0:
        # counterfactual: A's coefficients, B's X
        y_cf_sim = simulate(beta_a_grid, X_b, n_sim)
    else:
        # counterfactual: B's coefficients, A's X
        y_cf_sim = simulate(beta_b_grid, X_a, n_sim)

    rows = []
    for t in tau_grid:
        q_a = float(np.quantile(y_a_sim, t))
        q_b = float(np.quantile(y_b_sim, t))
        q_cf = float(np.quantile(y_cf_sim, t))
        gap = q_a - q_b
        if reference == 0:
            composition = q_a - q_cf  # effect of X being A-like vs B-like (A's coefs)
            structure = q_cf - q_b    # remaining
        else:
            composition = q_cf - q_b
            structure = q_a - q_cf
        rows.append({"tau": t, "q_a": q_a, "q_b": q_b, "q_cf": q_cf,
                     "gap": gap, "composition": composition, "structure": structure})
    grid_df = pd.DataFrame(rows)

    overall = {
        "mean_gap": float(grid_df["gap"].mean()),
        "mean_composition": float(grid_df["composition"].mean()),
        "mean_structure": float(grid_df["structure"].mean()),
        "median_gap": float(grid_df["gap"].median()),
    }

    se_df = None
    if inference == "bootstrap":
        rng_b = np.random.default_rng(seed)
        boot_list = []
        n_total = len(df)
        strata = g
        for _ in range(n_boot):
            # Stratified bootstrap
            idx_parts = []
            for s in (0, 1):
                s_idx = np.where(strata == s)[0]
                idx_parts.append(rng_b.choice(s_idx, size=len(s_idx), replace=True))
            idx = np.concatenate(idx_parts)
            try:
                g_i = g[idx]
                y_i = y_vec[idx]
                X_i = X_raw[idx]
                X_a_i = add_constant(X_i[g_i == 0])
                X_b_i = add_constant(X_i[g_i == 1])
                y_a_i = y_i[g_i == 0]
                y_b_i = y_i[g_i == 1]
                if len(y_a_i) < 20 or len(y_b_i) < 20:
                    continue
                beta_a_i = _qreg_grid(y_a_i, X_a_i, tau_qr)
                beta_b_i = _qreg_grid(y_b_i, X_b_i, tau_qr)
                ya_sim = simulate(beta_a_i, X_a_i, n_sim)
                yb_sim = simulate(beta_b_i, X_b_i, n_sim)
                if reference == 0:
                    ycf_sim = simulate(beta_a_i, X_b_i, n_sim)
                else:
                    ycf_sim = simulate(beta_b_i, X_a_i, n_sim)
                gaps_b = []
                comps_b = []
                for t in tau_grid:
                    q_a_b = np.quantile(ya_sim, t)
                    q_b_b = np.quantile(yb_sim, t)
                    q_cf_b = np.quantile(ycf_sim, t)
                    gaps_b.append(q_a_b - q_b_b)
                    if reference == 0:
                        comps_b.append(q_a_b - q_cf_b)
                    else:
                        comps_b.append(q_cf_b - q_b_b)
                boot_list.append((gaps_b, comps_b))
            except Exception:  # noqa: BLE001
                continue
        if len(boot_list) > 10:
            gaps_arr = np.array([b[0] for b in boot_list])
            comps_arr = np.array([b[1] for b in boot_list])
            se_df = pd.DataFrame({
                "tau": tau_grid,
                "gap_se": gaps_arr.std(axis=0, ddof=1),
                "composition_se": comps_arr.std(axis=0, ddof=1),
            })

    return MachadoMataResult(
        quantile_grid=grid_df, overall=overall, reference=reference,
        n_sim=n_sim, n_a=int(len(y_a)), n_b=int(len(y_b)),
        se=se_df,
    )
