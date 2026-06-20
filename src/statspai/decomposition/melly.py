"""
Melly (2005, 2006) quantile decomposition.

Improvement over Machado-Mata: uses the *full* grid of quantile
regression coefficients and averages over the covariate distribution
analytically rather than via Monte Carlo simulation. More efficient,
less noisy, and faster for the same accuracy.

The counterfactual unconditional CDF is

    F̂_{Y<A|B>}(y) = (1 / (n_B · J)) Σ_i Σ_j 1{X_{B,i}' β_A(τ_j) ≤ y}

inverted pointwise to obtain the counterfactual quantile function.

References
----------
Melly, B. (2005). "Decomposition of Differences in Distribution Using
Quantile Regression." *Labour Economics*, 12(4), 577-590. [@melly2005decomposition]

Melly, B. (2006). "Estimation of Counterfactual Distributions Using
Quantile Regression." Swiss Institute for International Economics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._common import add_constant, prepare_frame
from ._results import DecompResultMixin
from .machado_mata import _qreg_grid


@dataclass
class MellyResult(DecompResultMixin):
    """Container for Melly quantile decomposition."""

    method_name: ClassVar[str] = "Melly (2005) Quantile Decomposition"
    bib_keys: ClassVar[Tuple[str, ...]] = ("melly2005decomposition",)

    quantile_grid: pd.DataFrame
    overall: Dict[str, float]
    reference: int
    n_tau_qr: int
    n_a: int
    n_b: int

    def summary(self) -> str:
        g = self.quantile_grid
        lines = [
            "━" * 72,
            "  Melly (2005) Quantile Decomposition",
            "━" * 72,
            f"  N_A = {self.n_a}   N_B = {self.n_b}   "
            f"QR grid = {self.n_tau_qr}   reference = {self.reference}",
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

    def plot(self, **kwargs: Any) -> Any:
        from .plots import quantile_process_plot

        return quantile_process_plot(self, **kwargs)

    def to_latex(self) -> str:
        latex: str = self.quantile_grid.round(4).to_latex(index=False)
        return latex

    def _repr_html_(self) -> str:
        html: str = self.quantile_grid.round(4).to_html(index=False)
        return (
            "<div style='font-family:monospace;'>"
            "<h3>Melly Decomposition</h3>" + html + "</div>"
        )

    def __repr__(self) -> str:
        return (
            f"MellyResult(n_tau={len(self.quantile_grid)}, reference={self.reference})"
        )


def _unconditional_quantiles(
    beta_grid: np.ndarray,  # (J, k)
    X_source: np.ndarray,  # (n, k)
    tau_eval: np.ndarray,
) -> np.ndarray:
    """
    Melly's unconditional CDF inversion.

    Returns Q(τ) for τ in tau_eval.
    """
    J, k = beta_grid.shape
    # Predicted conditional quantiles: (J, n)
    preds = beta_grid @ X_source.T
    # Flatten to get unconditional sample
    sample = preds.ravel()
    return np.asarray(np.quantile(sample, tau_eval))


def melly_decompose(
    data: pd.DataFrame,
    y: str,
    group: str,
    x: Sequence[str],
    tau_grid: Optional[Sequence[float]] = None,
    reference: int = 0,
    n_tau_qr: int = 99,
) -> MellyResult:
    """
    Melly (2005) quantile decomposition.

    Parameters
    ----------
    data : pd.DataFrame
    y, group, x : column names
    tau_grid : Sequence[float] or None — reporting τ grid
    reference : {0, 1}
        Same convention as ``machado_mata``: ``reference=0`` uses A's β
        on B's X (coefficient-swap counterfactual F_{Y<0|1>}), opposite
        to ``dfl_decompose`` whose ``reference=0`` uses A's X with B's β.
    n_tau_qr : int — QR estimation grid resolution

    Returns
    -------
    MellyResult

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> r = sp.melly_decompose(df, y='log_wage', group='female',
    ...                        x=['education', 'experience', 'tenure'],
    ...                        tau_grid=[0.25, 0.5, 0.75])
    >>> print(r.summary())  # doctest: +ELLIPSIS
    ...Melly (2005) Quantile Decomposition...
    >>> list(r.quantile_grid['tau'])
    [0.25, 0.5, 0.75]
    >>> set(['gap', 'composition', 'structure']) <= set(r.quantile_grid.columns)
    True

    References
    ----------
    [@melly2005decomposition]
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
        raise ValueError("Need ≥20 obs per group for Melly.")

    if tau_grid is None:
        tau_seq: Sequence[float] | np.ndarray = np.round(np.arange(0.1, 0.95, 0.1), 2)
    else:
        tau_seq = tau_grid
    tau_eval = np.asarray(tau_seq, dtype=float)

    tau_qr = np.linspace(0.01, 0.99, n_tau_qr)
    beta_a_grid = _qreg_grid(y_a, X_a, tau_qr)
    beta_b_grid = _qreg_grid(y_b, X_b, tau_qr)

    q_a = _unconditional_quantiles(beta_a_grid, X_a, tau_eval)
    q_b = _unconditional_quantiles(beta_b_grid, X_b, tau_eval)
    if reference == 0:
        q_cf = _unconditional_quantiles(beta_a_grid, X_b, tau_eval)
    else:
        q_cf = _unconditional_quantiles(beta_b_grid, X_a, tau_eval)

    gap = q_a - q_b
    if reference == 0:
        composition = q_a - q_cf
        structure = q_cf - q_b
    else:
        composition = q_cf - q_b
        structure = q_a - q_cf

    grid_df = pd.DataFrame(
        {
            "tau": tau_eval,
            "q_a": q_a,
            "q_b": q_b,
            "q_cf": q_cf,
            "gap": gap,
            "composition": composition,
            "structure": structure,
        }
    )

    overall = {
        "mean_gap": float(np.mean(gap)),
        "mean_composition": float(np.mean(composition)),
        "mean_structure": float(np.mean(structure)),
    }

    return MellyResult(
        quantile_grid=grid_df,
        overall=overall,
        reference=reference,
        n_tau_qr=n_tau_qr,
        n_a=int(len(y_a)),
        n_b=int(len(y_b)),
    )
