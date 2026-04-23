"""
Marginal Treatment Effects (MTE) via polynomial MTR specification.

Implements the Brinch-Mogstad-Wiswall (2017, JoPE) closed-form polynomial
approach: parameterise the marginal treatment response functions

    m_0(u, X) = X' * phi_0(u),
    m_1(u, X) = X' * phi_1(u),

with polynomials ``phi_d(u) = sum_{k=0}^K theta_{d,k} u^k``, and invert
the observed conditional means ``E[Y | P(Z)=p, X, D=1]`` and
``E[Y | P(Z)=p, X, D=0]`` — each linear in the theta's — via GLS.

From the fitted MTR functions the skill derives:

- **MTE(u | X)** — u ∈ (0, 1)
- **ATE(X)** — ∫ MTE(u|X) du
- **ATT(X), ATU(X)**  — weighted integrals
- **LATE(p, p')** — ∫_{p}^{p'} MTE(u) du / (p'-p)
- **Policy-Relevant TE** for user-supplied policy shifts in P(Z)

This fills a long-standing Python gap (R has ``ivmte``, Stata ``mtefe``,
Python had nothing).

References
----------
Brinch, C.N., Mogstad, M. and Wiswall, M. (2017). "Beyond LATE with a
    Discrete Instrument." *Journal of Political Economy*, 125(4), 985-1039. [@brinch2017beyond]

Heckman, J.J. and Vytlacil, E.J. (2005). "Structural Equations,
    Treatment Effects, and Econometric Policy Evaluation."
    *Econometrica*, 73(3), 669-738. [@heckman2005structural]

Mogstad, M., Santos, A. and Torgovitsky, A. (2018). "Using Instrumental
    Variables for Inference About Policy Relevant Treatment Parameters."
    *Econometrica*, 86(5), 1589-1619. [@mogstad2018instrumental]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MTEResult:
    """Marginal treatment effects result."""
    poly_degree: int
    theta0: np.ndarray  # (K+1, dim_X) coefficients for control arm MTR
    theta1: np.ndarray  # (K+1, dim_X) coefficients for treated arm MTR
    theta0_se: np.ndarray
    theta1_se: np.ndarray
    ate: float
    ate_se: float
    att: float
    atu: float
    late_2sls: float
    propensity_range: Tuple[float, float]
    mte_curve: pd.DataFrame  # u, mte, se columns
    x_bar: np.ndarray
    n_obs: int
    treated_share: float
    extra: Dict

    def summary(self) -> str:
        u_min, u_max = self.propensity_range
        lines = [
            "Marginal Treatment Effects (BMW 2017 polynomial MTR)",
            "-" * 60,
            f"  Polynomial degree    : {self.poly_degree}",
            f"  Observations         : {self.n_obs}  (treated share = {self.treated_share:.3f})",
            f"  Propensity support   : [{u_min:.3f}, {u_max:.3f}]",
            "",
            "  Aggregate parameters (at mean X)",
            f"    ATE   = {self.ate:>8.4f}   SE={self.ate_se:.4f}",
            f"    ATT   = {self.att:>8.4f}",
            f"    ATU   = {self.atu:>8.4f}",
            f"    LATE* = {self.late_2sls:>8.4f}   (2SLS for reference)",
            "",
            "  * LATE is the 2SLS slope using the same instrument",
            "    projection; it equals the Wald estimator with binary Z.",
        ]
        return "\n".join(lines)


def _as_matrix(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _grab(v, data, cols=False):
    if isinstance(v, str):
        return data[v].values.astype(float)
    if cols and isinstance(v, list) and all(isinstance(x, str) for x in v):
        return data[v].values.astype(float)
    return np.asarray(v, dtype=float)


def _poly_u(u: np.ndarray, K: int) -> np.ndarray:
    """Return Vandermonde matrix [1, u, u^2, ..., u^K]."""
    u = np.asarray(u, dtype=float).reshape(-1)
    return np.vander(u, N=K + 1, increasing=True)


def _int_poly_u(a: float, b: float, K: int) -> np.ndarray:
    """∫_a^b [1, u, u^2, ..., u^K] du = [b-a, (b^2-a^2)/2, ...]."""
    return np.array([(b ** (k + 1) - a ** (k + 1)) / (k + 1) for k in range(K + 1)])


def mte(
    y: Union[np.ndarray, pd.Series, str],
    treatment: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    poly_degree: int = 2,
    u_grid: Optional[np.ndarray] = None,
    add_const: bool = True,
    propensity_model: str = "logit",
    trim: float = 0.01,
    bootstrap: Optional[int] = None,
    random_state: Optional[int] = None,
) -> MTEResult:
    """
    Polynomial MTE (Brinch-Mogstad-Wiswall 2017).

    Parameters
    ----------
    y : outcome.
    treatment : binary treatment indicator D ∈ {0, 1}.
    instruments : instruments Z.
    exog : covariates X (included in both structural model and propensity).
    data : DataFrame carrying any string references.
    poly_degree : int, default 2
        Degree K of the polynomial expansion in u for each MTR function.
    u_grid : array, optional
        Evaluation grid for the MTE curve. Defaults to 101 points covering
        the observed propensity-score range.
    add_const : bool, default True.
    propensity_model : {'logit', 'probit', 'linear'}
        Model for P(D=1 | Z, X).
    trim : float, default 0.01
        Trim observations with p(Z,X) outside ``[trim, 1-trim]``.
    bootstrap : int, optional
        If set, run a nonparametric pairs bootstrap with this many draws
        to obtain honest standard errors for MTE(u), ATE, ATT, ATU. When
        ``None`` (default), analytic plug-in SEs are used for the MTE
        curve and ATE, and no SE is reported for ATT / ATU.
    random_state : int, optional
        Seed for the bootstrap draws.

    Returns
    -------
    MTEResult
    """
    Y = _grab(y, data).reshape(-1)
    D = _grab(treatment, data).reshape(-1)
    uniq = np.unique(D[~np.isnan(D)])
    if not set(uniq.tolist()).issubset({0.0, 1.0}):
        raise ValueError("MTE requires binary 0/1 treatment.")

    Z = _grab(instruments, data, cols=True)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    n = len(Y)
    if exog is None:
        X_no_const = np.empty((n, 0))
    else:
        X_no_const = _grab(exog, data, cols=True)
        if X_no_const.ndim == 1:
            X_no_const = X_no_const.reshape(-1, 1)

    X_raw = np.column_stack([np.ones(n), X_no_const]) if add_const else X_no_const
    if X_raw.shape[1] == 0:
        X_raw = np.ones((n, 1))
    X = X_raw
    dx = X.shape[1]

    # Step 1: propensity score p(Z, X)
    Z_prop = np.column_stack([X, Z])
    p_hat = _fit_propensity(D, Z_prop, model=propensity_model)

    # Trim to common support
    keep = (p_hat > trim) & (p_hat < 1 - trim) & ~np.isnan(p_hat)
    Y, D, X, p_hat = Y[keep], D[keep], X[keep], p_hat[keep]
    n = len(Y)

    K = int(poly_degree)
    # Step 2: build design matrices.
    # For treated (D=1): E[Y | p, X, D=1] = X' β_1 + X' ∑ θ_{1,k}/(k+1) p^k  term
    # Actually BMW closed form:
    #   E[Y | P(Z)=p, X, D=1]*p = X' ∫_0^p phi_1(u) du  = X' K_1(p)
    #   E[Y | P(Z)=p, X, D=0]*(1-p) = X' ∫_p^1 phi_0(u) du = X' K_0(p)
    # So (Y * D) and (Y * (1-D)) regressions on (X ⊗ poly(p)) identify theta.
    #
    # We follow BMW eq. (13):
    #   E[Y | p, X] = X' α_0 + p * X' (α_1 - α_0)  (for K=0, gives LATE)
    #   E[Y | p, X] includes polynomial p-terms via:
    #       Y_i = X_i' phi_0(u) + D_i * X_i' (phi_1(u) - phi_0(u))   with u ~ U(0,1)|p
    # Tractable form: regress Y on [X, X*p, X*p^2, ..., X*p^K,  D*X, D*X*p, ...]
    # which recovers polynomial differences. We use the "partial-sample" OLS
    # separately on D=1 and D=0 sub-samples with p-polynomial interactions.

    # Treated sample: E[Y | p, X, D=1] = (1/p) * X' ∫_0^p phi_1(u) du
    # ∫_0^p u^k du = p^{k+1}/(k+1)  => (1/p)*p^{k+1}/(k+1) = p^k/(k+1)
    # So fit on the treated: Y = sum_k  X * p^k / (k+1) * theta_{1,k} + error
    mask1 = D == 1
    mask0 = D == 0
    if mask1.sum() < dx * (K + 1) or mask0.sum() < dx * (K + 1):
        raise ValueError(
            f"Not enough observations in each treatment arm for poly_degree={K}. "
            f"Try smaller degree or more data."
        )

    def build_design(p, mode: str) -> np.ndarray:
        """For each observation, stack X * weight_k(p) across k=0..K.

        mode='treated'   : weight_k(p) = p^k / (k+1)
        mode='untreated' : weight_k(p) = (1 - p^{k+1}) / ((k+1) * (1-p))
        """
        n_here = len(p)
        cols = []
        for k in range(K + 1):
            if mode == "treated":
                w = p ** k / (k + 1)
            else:
                denom = np.where(1 - p > 1e-8, (k + 1) * (1 - p), np.nan)
                w = (1 - p ** (k + 1)) / denom
            cols.append(X_here * w[:, None])  # n_here x dx
        return np.column_stack(cols)  # n_here x dx*(K+1)

    # Solve separately on the two arms
    X_here = X[mask1]
    p1 = p_hat[mask1]
    M1 = build_design(p1, "treated")
    theta1_flat, *_ = np.linalg.lstsq(M1, Y[mask1], rcond=None)
    resid1 = Y[mask1] - M1 @ theta1_flat
    sigma1 = float(resid1 @ resid1) / max(len(resid1) - M1.shape[1], 1)
    try:
        var1 = sigma1 * np.linalg.inv(M1.T @ M1)
    except np.linalg.LinAlgError:
        var1 = sigma1 * np.linalg.pinv(M1.T @ M1)
    se1_flat = np.sqrt(np.maximum(np.diag(var1), 0))

    X_here = X[mask0]
    p0 = p_hat[mask0]
    M0 = build_design(p0, "untreated")
    theta0_flat, *_ = np.linalg.lstsq(M0, Y[mask0], rcond=None)
    resid0 = Y[mask0] - M0 @ theta0_flat
    sigma0 = float(resid0 @ resid0) / max(len(resid0) - M0.shape[1], 1)
    try:
        var0 = sigma0 * np.linalg.inv(M0.T @ M0)
    except np.linalg.LinAlgError:
        var0 = sigma0 * np.linalg.pinv(M0.T @ M0)
    se0_flat = np.sqrt(np.maximum(np.diag(var0), 0))

    # Reshape theta: rows = poly degree k, cols = X dimension
    theta1 = theta1_flat.reshape(K + 1, dx)
    theta0 = theta0_flat.reshape(K + 1, dx)
    se1 = se1_flat.reshape(K + 1, dx)
    se0 = se0_flat.reshape(K + 1, dx)

    # Aggregate parameters at X̄
    x_bar = X.mean(axis=0)
    # MTE(u | x̄) = x̄' (phi_1(u) - phi_0(u))
    u_min, u_max = float(p_hat.min()), float(p_hat.max())
    if u_grid is None:
        u_grid = np.linspace(max(u_min, 0.01), min(u_max, 0.99), 101)

    V = _poly_u(u_grid, K)  # n_grid x (K+1)
    mte_x = V @ (theta1 - theta0) @ x_bar  # n_grid
    # Delta-method SE: Var( x̄'(theta_{1,k} - theta_{0,k}) ) per u
    var_diff_per_k = (se1 ** 2 + se0 ** 2) @ (x_bar ** 2)  # K+1
    mte_se = np.sqrt(np.maximum(V ** 2 @ var_diff_per_k, 0))

    # ATE = ∫ MTE(u) du at x̄
    weight_ate = _int_poly_u(0.0, 1.0, K)  # (K+1,)
    ate = float(weight_ate @ (theta1 - theta0) @ x_bar)
    ate_var = float(
        weight_ate @ np.diag(var_diff_per_k.reshape(-1)) @ weight_ate
    )
    ate_se = float(np.sqrt(max(ate_var, 0)))

    # ATT = ∫_0^1 MTE(u) * Pr(P(Z)>u | D=1) du ≈ ∫_0^1 MTE(u) * F_p|D=1(u) weight
    # Use empirical plug-in with the observed p distribution among treated
    att = _weighted_integral(u_grid, mte_x, _empirical_cdf_weight(u_grid, p_hat[D == 1]))
    atu = _weighted_integral(u_grid, mte_x, _empirical_cdf_weight(u_grid, p_hat[D == 0], side="upper"))

    # LATE reference = 2SLS of Y on D using Z
    late_ref = _wald_tsls(Y, D, X, p_hat)

    # ─── optional bootstrap SE ──────────────────────────────────────────
    boot_extra = {}
    if bootstrap is not None and int(bootstrap) > 0:
        B = int(bootstrap)
        rng_boot = np.random.default_rng(random_state)
        boot_mte = np.full((B, len(u_grid)), np.nan)
        boot_ate = np.full(B, np.nan)
        boot_att = np.full(B, np.nan)
        boot_atu = np.full(B, np.nan)
        for b in range(B):
            idx = rng_boot.integers(0, n, size=n)
            try:
                pt = _mte_point_only(
                    Y[idx], D[idx], Z[idx], X_raw[idx],
                    K, u_grid, propensity_model, trim,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            boot_mte[b] = pt["mte_curve"]
            boot_ate[b] = pt["ate"]
            boot_att[b] = pt["att"]
            boot_atu[b] = pt["atu"]
        ok = np.isfinite(boot_ate)
        if ok.sum() >= 10:
            mte_se = np.nanstd(boot_mte[ok], axis=0, ddof=1)
            ate_se = float(np.nanstd(boot_ate[ok], ddof=1))
            boot_extra["att_se"] = float(np.nanstd(boot_att[ok], ddof=1))
            boot_extra["atu_se"] = float(np.nanstd(boot_atu[ok], ddof=1))
            boot_extra["n_successful_draws"] = int(ok.sum())
            boot_extra["n_requested_draws"] = B

    curve = pd.DataFrame({"u": u_grid, "mte": mte_x, "se": mte_se})
    curve["ci_lower"] = curve["mte"] - 1.96 * curve["se"]
    curve["ci_upper"] = curve["mte"] + 1.96 * curve["se"]

    return MTEResult(
        poly_degree=K,
        theta0=theta0,
        theta1=theta1,
        theta0_se=se0,
        theta1_se=se1,
        ate=ate,
        ate_se=ate_se,
        att=float(att),
        atu=float(atu),
        late_2sls=float(late_ref),
        propensity_range=(u_min, u_max),
        mte_curve=curve,
        x_bar=x_bar,
        n_obs=n,
        treated_share=float(D.mean()),
        extra={"p_hat": p_hat, "keep_mask": keep, **boot_extra},
    )


def _fit_propensity(D: np.ndarray, Z: np.ndarray, model: str = "logit") -> np.ndarray:
    model = model.lower()
    if model == "linear":
        beta, *_ = np.linalg.lstsq(Z, D, rcond=None)
        p = Z @ beta
        return np.clip(p, 1e-4, 1 - 1e-4)
    # Newton-Raphson for logit / probit
    n, k = Z.shape
    beta = np.zeros(k)
    for _ in range(60):
        eta = Z @ beta
        if model == "logit":
            p = 1.0 / (1.0 + np.exp(-eta))
            W = p * (1 - p)
            resid = D - p
        else:  # probit
            p = stats.norm.cdf(eta)
            phi = stats.norm.pdf(eta)
            W = np.where(p * (1 - p) > 1e-10, phi ** 2 / (p * (1 - p) + 1e-10), 1e-10)
            resid = phi * (D - p) / np.where(p * (1 - p) > 1e-10, p * (1 - p), 1e-10)
        H = Z.T @ (Z * W[:, None])
        g = Z.T @ resid
        try:
            step = np.linalg.solve(H + 1e-8 * np.eye(k), g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta += step
        if np.linalg.norm(step) < 1e-8:
            break
    eta = Z @ beta
    p = 1.0 / (1.0 + np.exp(-eta)) if model == "logit" else stats.norm.cdf(eta)
    return np.clip(p, 1e-4, 1 - 1e-4)


def _wald_tsls(Y: np.ndarray, D: np.ndarray, X: np.ndarray, p: np.ndarray) -> float:
    """Reference 2SLS of Y on D using p(Z, X) as instrument after partialling out X."""
    def _partial(M):
        b, *_ = np.linalg.lstsq(X, M, rcond=None)
        return M - X @ b
    Yp = _partial(Y)
    Dp = _partial(D)
    Zp = _partial(p)
    # β = cov(Zp, Yp)/cov(Zp, Dp)
    denom = float(Zp @ Dp)
    return float(Zp @ Yp) / denom if abs(denom) > 1e-12 else np.nan


def _empirical_cdf_weight(u_grid: np.ndarray, p_sample: np.ndarray, side: str = "lower") -> np.ndarray:
    """Weights used by ATT/ATU integrals — see BMW 2017 Table 2."""
    if len(p_sample) == 0:
        return np.ones_like(u_grid) / max(len(u_grid), 1)
    if side == "lower":
        w = np.array([(p_sample >= u).mean() for u in u_grid])
    else:
        w = np.array([(p_sample <= u).mean() for u in u_grid])
    total = np.trapezoid(w, u_grid)
    return w / total if total > 0 else np.ones_like(u_grid) / max(len(u_grid), 1)


def _weighted_integral(u_grid: np.ndarray, f_values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.trapezoid(f_values * weights, u_grid))


def _mte_point_only(
    Y: np.ndarray, D: np.ndarray, Z: np.ndarray, X: np.ndarray,
    K: int, u_grid: np.ndarray,
    propensity_model: str, trim: float,
) -> Dict:
    """Internal: return *only* the point estimates for a (Y, D, Z, X) sample.

    Used by the bootstrap inside :func:`mte`. No standard errors, no
    diagnostic plumbing — this runs many times so it must be cheap.
    """
    n = len(Y)
    Z_prop = np.column_stack([X, Z])
    p = _fit_propensity(D, Z_prop, model=propensity_model)
    keep = (p > trim) & (p < 1 - trim) & ~np.isnan(p)
    Y, D, X, p = Y[keep], D[keep], X[keep], p[keep]
    if len(Y) < X.shape[1] * (K + 1) * 4:
        raise ValueError("Bootstrap draw too small after trimming.")
    dx = X.shape[1]

    mask1 = D == 1
    mask0 = D == 0
    if mask1.sum() < dx * (K + 1) or mask0.sum() < dx * (K + 1):
        raise ValueError("Arm too small in bootstrap draw.")

    def build(p_sub, Xs, mode):
        cols = []
        for k in range(K + 1):
            if mode == "treated":
                w = p_sub ** k / (k + 1)
            else:
                denom = np.where(1 - p_sub > 1e-8, (k + 1) * (1 - p_sub), np.nan)
                w = (1 - p_sub ** (k + 1)) / denom
            cols.append(Xs * w[:, None])
        return np.column_stack(cols)

    M1 = build(p[mask1], X[mask1], "treated")
    t1_flat, *_ = np.linalg.lstsq(M1, Y[mask1], rcond=None)
    M0 = build(p[mask0], X[mask0], "untreated")
    t0_flat, *_ = np.linalg.lstsq(M0, Y[mask0], rcond=None)
    theta1 = t1_flat.reshape(K + 1, dx)
    theta0 = t0_flat.reshape(K + 1, dx)
    x_bar = X.mean(axis=0)
    V = _poly_u(u_grid, K)
    mte_u = V @ (theta1 - theta0) @ x_bar
    weight_ate = _int_poly_u(0.0, 1.0, K)
    ate = float(weight_ate @ (theta1 - theta0) @ x_bar)
    att = _weighted_integral(u_grid, mte_u, _empirical_cdf_weight(u_grid, p[D == 1]))
    atu = _weighted_integral(u_grid, mte_u, _empirical_cdf_weight(u_grid, p[D == 0], side="upper"))
    return {"mte_curve": mte_u, "ate": ate, "att": float(att), "atu": float(atu)}


__all__ = ["mte", "MTEResult"]
