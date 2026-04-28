"""
VCNet-style neural dose-response estimation (Nie, Ye, Zhang, Yang &
Wen 2021, ICLR).

VCNet (Variational Continuous-treatment Network) fits a neural
network that outputs a smooth dose-response curve :math:`\\mu(t, x)`
using a spline-varying coefficient architecture. The key ideas:

1. **Treatment-varying coefficients.** The outcome head has weights
   that are **functions of t** (via B-spline basis evaluated at t),
   so the model can learn non-linear response curves without a fixed
   parametric form.

2. **Density / propensity adjustment.** A concurrent density-estimator
   head models :math:`\\pi(t | x)`, used for inverse-weighting or for
   the targeted update.

For a self-contained implementation without extra dependencies we use
NumPy + SciPy (B-spline basis) to fit a ridge-regularised *varying-
coefficient linear model*:

.. math::

   \\mu(t, x) = \\sum_{b=1}^{B} \\phi_b(t) \\cdot x^\\top \\beta_b,

where :math:`\\phi_b` is a B-spline basis. This captures the
defining property of VCNet (t-varying coefficients) and yields
smooth curves. Users wanting GPU neural fits can plug PyTorch in.

For dose-response inference, we report
:math:`\\hat\\mu(t) = E_n[\\mu(t, X)]` on a treatment grid, plus
bootstrap standard errors.

References
----------
Nie, L., Ye, M., Zhang, L., Yang, Q., & Wen, Q. (2021).
"VCNet and Functional Targeted Regularization For Learning Causal
Effects of Continuous Treatments." *ICLR 2021*. [@nie2021quasi]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline


@dataclass
class VCNetResult:
    t_grid: np.ndarray
    mu_hat: np.ndarray
    se: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    coef_matrix: np.ndarray  # (B, p+1)
    n_obs: int
    n_basis: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        df = pd.DataFrame({
            "t": self.t_grid,
            "mu_hat": self.mu_hat,
            "se": self.se,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
        })
        return "VCNet dose-response curve\n" + df.to_string(index=False)

    def __repr__(self) -> str:  # pragma: no cover
        return f"VCNetResult(n={self.n_obs}, grid_size={len(self.t_grid)})"


def _bspline_basis(t: np.ndarray, n_basis: int = 6, degree: int = 3) -> np.ndarray:
    """B-spline basis matrix evaluated at t (shape: len(t) x n_basis)."""
    t_min, t_max = float(np.min(t)), float(np.max(t))
    if t_min >= t_max:
        t_max = t_min + 1.0
    # interior knots
    n_interior = max(n_basis - degree - 1, 0)
    if n_interior > 0:
        interior = np.linspace(t_min, t_max, n_interior + 2)[1:-1]
    else:
        interior = np.array([])
    knots = np.concatenate([
        [t_min] * (degree + 1),
        interior,
        [t_max] * (degree + 1),
    ])
    basis = np.zeros((len(t), n_basis))
    for b in range(n_basis):
        c = np.zeros(n_basis)
        c[b] = 1.0
        spline = BSpline(knots, c, degree, extrapolate=False)
        vals = spline(t)
        vals = np.where(np.isnan(vals), 0.0, vals)
        basis[:, b] = vals
    return basis


def vcnet(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    covariates: Sequence[str],
    t_grid: Optional[Sequence[float]] = None,
    n_basis: int = 6,
    spline_degree: int = 3,
    ridge: float = 1e-2,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> VCNetResult:
    """
    Varying-coefficient dose-response estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treatment : str
        Continuous treatment / dose column.
    covariates : sequence of str
    t_grid : sequence of float, optional
        Treatment values at which to evaluate the dose-response curve.
        Defaults to 40 equally-spaced points between the observed min/max.
    n_basis : int, default 6
        Number of B-spline basis functions for the t-axis.
    spline_degree : int, default 3
    ridge : float, default 1e-2
        Tikhonov regularisation on the coefficient matrix.
    n_bootstrap : int, default 200
    alpha : float, default 0.05
    random_state : int, default 42

    Returns
    -------
    VCNetResult
    """
    X_cols = list(covariates)
    df = data[[y, treatment] + X_cols].dropna().reset_index(drop=True)
    n = len(df)
    Y = df[y].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=float)
    X = df[X_cols].to_numpy(dtype=float) if X_cols else np.zeros((n, 0))

    if t_grid is None:
        t_grid = np.linspace(T.min(), T.max(), 40)
    t_grid = np.asarray(t_grid, dtype=float)

    # Full design: [phi(t) tensor (1, X)], i.e. n x (B * (p+1))
    B = _bspline_basis(T, n_basis=n_basis, degree=spline_degree)
    X_aug = np.column_stack([np.ones(n), X])  # n x (p+1)
    # outer per-row: shape n x (n_basis * (p+1))
    design = np.einsum("nb,np->nbp", B, X_aug).reshape(n, -1)

    def fit(design, Y):
        G = design.T @ design + ridge * np.eye(design.shape[1])
        rhs = design.T @ Y
        return np.linalg.solve(G, rhs)

    beta = fit(design, Y)
    coef_matrix = beta.reshape(n_basis, X_aug.shape[1])

    # Predict curve on grid
    B_grid = _bspline_basis(t_grid, n_basis=n_basis, degree=spline_degree)

    def predict_curve(coef):
        # For each t, mu(t) = mean_i sum_b phi_b(t) * (x_i dot coef[b])
        # = sum_b phi_b(t) * (mean(x) dot coef[b])
        x_mean = X_aug.mean(axis=0)
        mu_per_basis = coef @ x_mean  # shape (n_basis,)
        return B_grid @ mu_per_basis

    mu_hat = predict_curve(coef_matrix)

    # Bootstrap for SE
    rng = np.random.default_rng(random_state)
    boot_curves = np.zeros((n_bootstrap, len(t_grid)))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        d_boot = design[idx]
        y_boot = Y[idx]
        try:
            beta_b = fit(d_boot, y_boot)
            boot_curves[b] = predict_curve(beta_b.reshape(n_basis, X_aug.shape[1]))
        except np.linalg.LinAlgError:
            boot_curves[b] = mu_hat

    se = boot_curves.std(axis=0, ddof=1)
    q_lo = np.quantile(boot_curves, alpha / 2, axis=0)
    q_hi = np.quantile(boot_curves, 1 - alpha / 2, axis=0)

    _result = VCNetResult(
        t_grid=t_grid,
        mu_hat=mu_hat,
        se=se,
        ci_lo=q_lo,
        ci_hi=q_hi,
        coef_matrix=coef_matrix,
        n_obs=n,
        n_basis=n_basis,
        detail={"ridge": ridge, "degree": spline_degree},
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.dose_response.vcnet",
            params={
                "y": y, "treatment": treatment,
                "covariates": list(covariates),
                "n_basis": n_basis, "spline_degree": spline_degree,
                "ridge": ridge, "n_bootstrap": n_bootstrap,
                "alpha": alpha, "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# --------------------------------------------------------------------
# SCIGAN placeholder — delegate to VCNet-style fit for now, with an
# option to inject adversarial samples via user-provided weights.
# --------------------------------------------------------------------


def scigan(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    covariates: Sequence[str],
    t_grid: Optional[Sequence[float]] = None,
    propensity_weights: Optional[np.ndarray] = None,
    **kwargs,
) -> VCNetResult:
    """
    Adversarial dose-response estimator (Bica et al. 2020).

    The reference SCIGAN architecture is a GAN — too heavy to ship as
    a dependency-free algorithm. This entry point exposes a
    propensity-weighted VCNet fit that captures the essential
    "balancing" behaviour of SCIGAN: residual imbalance along the
    treatment axis is re-weighted using a user-provided ``propensity_weights``
    (``g(T | X)^{-1}``). For the full SCIGAN training loop, plug in
    your own GAN-generated counterfactual samples and pass the
    re-weighting through ``propensity_weights``.
    """
    X_cols = list(covariates)
    df = data[[y, treatment] + X_cols].dropna().reset_index(drop=True)
    if propensity_weights is not None:
        w = np.asarray(propensity_weights, dtype=float)
        if len(w) != len(df):
            raise ValueError("propensity_weights length mismatch")
        df = df.assign(_w=w)
        # Expand (sample with replacement by weight); simplest proxy.
        rng = np.random.default_rng(kwargs.get("random_state", 42))
        probs = np.clip(w, 1e-6, None)
        probs /= probs.sum()
        idx = rng.choice(len(df), size=len(df), replace=True, p=probs)
        df = df.iloc[idx].reset_index(drop=True)
    return vcnet(df, y=y, treatment=treatment, covariates=X_cols, t_grid=t_grid, **kwargs)


__all__ = ["vcnet", "scigan", "VCNetResult"]
