"""Spatial panel data estimators — SAR / SEM / SDM with entity fixed effects.

Balanced panel:  ``y_it = ρ W y_it + X_it β + μ_i + ε_it`` (SAR-FE)
               ``y_it = X_it β + μ_i + u_it;  u_it = λ W u_it + ε_it`` (SEM-FE)
               ``y_it = ρ W y_it + X_it β + W X_it θ + μ_i + ε_it`` (SDM-FE)

Within-transform removes ``μ_i``; concentrated ML on the demeaned variables
gives ρ (or λ), β, σ². Log-determinant is ``T * log|I_N − ρ W|`` (small-N
eigenvalue path).

Two-way fixed effects (``effects='twoways'``) demean along both dimensions.
Random effects are not implemented here (splm's ``spreml``) — open a
separate sub-spec if needed.

References
----------
Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to
  Spatial Panels*. Springer.
Lee, L.-F. & Yu, J. (2010). "Estimation of spatial autoregressive panel
  data models with fixed effects." *JoE*, 154(2), 165-185.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar

from ..weights.core import W as _W
from ..models.ml import _coerce_W


EffectKind = Literal["fe", "twoways"]
ModelKind = Literal["sar", "sem", "sdm"]


# --------------------------------------------------------------------- #
#  Balanced-panel reshaping
# --------------------------------------------------------------------- #

def _balanced_panel_matrix(
    data: pd.DataFrame, entity: str, time: str, var: str
) -> np.ndarray:
    """Return (N, T) matrix: rows = entities (sorted), columns = time (sorted)."""
    pivot = data.pivot(index=entity, columns=time, values=var)
    if pivot.isna().any().any():
        raise ValueError(
            "Panel is unbalanced — every (entity, time) cell must be present."
        )
    return pivot.to_numpy(dtype=float)


def _within_transform(arr: np.ndarray, effects: EffectKind) -> np.ndarray:
    """Demean an (N, T) array within entities (and within time if twoways)."""
    out = arr - arr.mean(axis=1, keepdims=True)          # entity demean
    if effects == "twoways":
        out = out - arr.mean(axis=0, keepdims=True) + arr.mean()
    return out


# --------------------------------------------------------------------- #
#  Result dataclass
# --------------------------------------------------------------------- #

@dataclass
class SpatialPanelResult:
    params: pd.Series            # [x1, x2, …, ρ or λ]
    std_errors: pd.Series
    model: ModelKind
    effects: EffectKind
    spatial_param: str           # "rho" or "lambda"
    spatial_param_value: float
    sigma2: float
    log_likelihood: float
    residuals: np.ndarray        # (N, T)
    N: int
    T: int

    def summary(self) -> str:
        lines = [
            f"Spatial Panel ({self.model.upper()}, {self.effects})",
            "-" * 50,
            f"Entities N : {self.N}",
            f"Periods  T : {self.T}",
            f"NT         : {self.N * self.T}",
            f"σ²         : {self.sigma2: .6f}",
            f"LogLik     : {self.log_likelihood: .4f}",
            "",
            "Coefficients:",
        ]
        for name in self.params.index:
            coef = self.params[name]
            se = self.std_errors[name]
            t = coef / se if se and np.isfinite(se) and se > 0 else np.nan
            lines.append(f"  {name:<15s}  {coef: .4f}   (se={se: .4f}, t={t: .3f})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------- #

def spatial_panel(
    data: pd.DataFrame,
    formula: str,
    entity: str,
    time: str,
    W,
    model: ModelKind = "sar",
    effects: EffectKind = "fe",
    row_normalize: bool = True,
) -> SpatialPanelResult:
    """Fit a spatial panel model by concentrated ML (Elhorst 2014).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel; must be balanced.
    formula : str
        ``"y ~ x1 + x2"``. Constant is dropped (absorbed by the entity FE).
    entity, time : str
        Column names identifying entity and time dimensions.
    W : sparse matrix, ndarray, or :class:`statspai.spatial.weights.W`
        N × N spatial weights matrix aligned with ``sorted(data[entity].unique())``.
    model : {"sar", "sem", "sdm"}
    effects : {"fe", "twoways"}
        ``"fe"`` = entity FE only. ``"twoways"`` = entity + time demeaning.
    """
    if "~" not in formula:
        raise ValueError("formula must be of the form 'y ~ x1 + x2'")
    dep, indep_str = [s.strip() for s in formula.split("~", 1)]
    indep = [v.strip() for v in indep_str.split("+") if v.strip()]

    entities = sorted(data[entity].unique())
    times = sorted(data[time].unique())
    N = len(entities); T = len(times)

    # (N, T) matrices for dependent + each independent
    Y = _balanced_panel_matrix(data, entity, time, dep)
    X_mats = [_balanced_panel_matrix(data, entity, time, v) for v in indep]

    M = _coerce_W(W, n_expected=N, row_normalize=row_normalize)
    if M.shape != (N, N):
        raise ValueError(f"W shape {M.shape} does not match N entities = {N}")

    # Within transform
    Y_w = _within_transform(Y, effects)
    X_w = [_within_transform(x, effects) for x in X_mats]

    # Stack to NT vectors / matrix
    y_vec = Y_w.flatten(order="F")               # time-major (t=0 first, then t=1, …)
    X_stack = np.column_stack([x.flatten(order="F") for x in X_w])

    if model == "sdm":
        # Append W @ X (per period, then stack) — no constant to worry about.
        WX_mats = [M @ x for x in X_w]
        WX_stack = np.column_stack([x.flatten(order="F") for x in WX_mats])
        X_stack = np.column_stack([X_stack, WX_stack])
        indep_names = list(indep) + [f"W_{v}" for v in indep]
    else:
        indep_names = list(indep)

    W_dense = M.toarray()
    eigvals = np.real(np.linalg.eigvals(W_dense))
    lo = max(1.0 / eigvals.min() * 0.99, -0.99)
    hi = min(1.0 / eigvals.max() * 0.99, 0.99)

    def _apply_spatial(rho_or_lam: float, target: np.ndarray) -> np.ndarray:
        """Apply (I - θ W) to each period column of an (N, T) matrix."""
        # target is (N, T); premultiply by (I - θ W)
        out = target - rho_or_lam * (W_dense @ target)
        return out

    if model in ("sar", "sdm"):
        # y_star = (I - ρ W) Y_w  (column-wise); β from OLS on X_w
        def neg_ll(rho: float) -> float:
            Y_star = _apply_spatial(rho, Y_w)
            y_star_vec = Y_star.flatten(order="F")
            XtX_inv = np.linalg.inv(X_stack.T @ X_stack)
            beta = XtX_inv @ (X_stack.T @ y_star_vec)
            e = y_star_vec - X_stack @ beta
            sigma2 = float(e @ e) / (N * T)
            if sigma2 <= 0:
                return 1e20
            # log |I - ρ W| = sum log|1 - ρ λ_i|; panel brings factor T
            ldet = T * float(np.sum(np.log(np.abs(1 - rho * eigvals))))
            return -(-N * T / 2 * np.log(2 * np.pi * sigma2)
                     + ldet - (e @ e) / (2 * sigma2))

        opt = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
        rho_hat = float(opt.x)
        Y_star = _apply_spatial(rho_hat, Y_w)
        y_star_vec = Y_star.flatten(order="F")
        XtX_inv = np.linalg.inv(X_stack.T @ X_stack)
        beta = XtX_inv @ (X_stack.T @ y_star_vec)
        e = y_star_vec - X_stack @ beta
        sigma2 = float(e @ e) / (N * T)
        se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
        # simple numerical SE for ρ
        h = 1e-4
        d2 = (neg_ll(rho_hat + h) - 2 * neg_ll(rho_hat) + neg_ll(rho_hat - h)) / (h * h)
        se_rho = float(1.0 / np.sqrt(max(d2, 1e-10)))
        spatial_name = "rho"
        spatial_value = rho_hat

    else:  # SEM-FE: (I - λW) premultiplied to both sides
        def neg_ll(lam: float) -> float:
            Y_star = _apply_spatial(lam, Y_w)
            y_star_vec = Y_star.flatten(order="F")
            X_star_stack = np.column_stack(
                [_apply_spatial(lam, x).flatten(order="F") for x in X_w]
            )
            XtX_inv = np.linalg.inv(X_star_stack.T @ X_star_stack)
            beta = XtX_inv @ (X_star_stack.T @ y_star_vec)
            e = y_star_vec - X_star_stack @ beta
            sigma2 = float(e @ e) / (N * T)
            if sigma2 <= 0:
                return 1e20
            ldet = T * float(np.sum(np.log(np.abs(1 - lam * eigvals))))
            return -(-N * T / 2 * np.log(2 * np.pi * sigma2)
                     + ldet - (e @ e) / (2 * sigma2))

        opt = minimize_scalar(neg_ll, bounds=(lo, hi), method="bounded")
        lam_hat = float(opt.x)
        Y_star = _apply_spatial(lam_hat, Y_w)
        y_star_vec = Y_star.flatten(order="F")
        X_star_stack = np.column_stack(
            [_apply_spatial(lam_hat, x).flatten(order="F") for x in X_w]
        )
        XtX_inv = np.linalg.inv(X_star_stack.T @ X_star_stack)
        beta = XtX_inv @ (X_star_stack.T @ y_star_vec)
        e = y_star_vec - X_star_stack @ beta
        sigma2 = float(e @ e) / (N * T)
        se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
        h = 1e-4
        d2 = (neg_ll(lam_hat + h) - 2 * neg_ll(lam_hat) + neg_ll(lam_hat - h)) / (h * h)
        se_rho = float(1.0 / np.sqrt(max(d2, 1e-10)))
        spatial_name = "lambda"
        spatial_value = lam_hat

    names = list(indep_names) + [spatial_name]
    params_vec = np.append(beta, spatial_value)
    se_vec = np.append(se_beta, se_rho)
    residuals = e.reshape((N, T), order="F")
    return SpatialPanelResult(
        params=pd.Series(params_vec, index=names),
        std_errors=pd.Series(se_vec, index=names),
        model=model,
        effects=effects,
        spatial_param=spatial_name,
        spatial_param_value=spatial_value,
        sigma2=sigma2,
        log_likelihood=-float(opt.fun),
        residuals=residuals,
        N=N, T=T,
    )
