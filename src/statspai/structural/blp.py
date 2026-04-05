"""
Berry, Levinsohn & Pakes (1995) Demand Estimation
==================================================

Random-coefficients logit model for differentiated products demand,
estimated via Generalized Method of Moments (GMM).

Model
-----
Utility: u_ijt = x_jt'β + α*p_jt + ξ_jt + Σ_k σ_k * v_ik * x_kjt + ε_ijt

Where:
- v_ik ~ N(0,1) are random coefficients
- ε_ijt ~ Type I Extreme Value (logit errors)
- ξ_jt is the unobserved product quality (structural error)

Estimation proceeds in two nested loops:
1. Inner loop: BLP contraction mapping to recover mean utilities δ
2. Outer loop: GMM over nonlinear parameters σ (random coefficient std devs)

References
----------
Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile Prices in Market
Equilibrium. Econometrica, 63(4), 841-890.

Nevo, A. (2000). A Practitioner's Guide to Estimation of Random-Coefficients
Logit Models of Demand. Journal of Economics & Management Strategy, 9(4), 513-548.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Union
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import EconometricResults


# ---------------------------------------------------------------------------
# Halton sequence generator for quasi-Monte Carlo integration
# ---------------------------------------------------------------------------

def _halton_sequence(n: int, dim: int, seed: int | None = None) -> np.ndarray:
    """
    Generate a Halton quasi-random sequence mapped to N(0,1) draws.

    Parameters
    ----------
    n : int
        Number of draws.
    dim : int
        Number of dimensions (one per random coefficient).
    seed : int or None
        If provided, shuffle the sequence for randomization.

    Returns
    -------
    np.ndarray
        Shape (n, dim) array of standard-normal draws.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dim > len(primes):
        raise ValueError(f"x_random has {dim} variables; max supported is {len(primes)}.")

    def _halton_1d(size: int, base: int) -> np.ndarray:
        seq = np.zeros(size)
        for i in range(size):
            f, r = 1.0, 0.0
            idx = i + 1
            while idx > 0:
                f /= base
                r += f * (idx % base)
                idx //= base
            seq[i] = r
        return seq

    rng = np.random.default_rng(seed)
    draws = np.column_stack([_halton_1d(n, primes[d]) for d in range(dim)])

    if seed is not None:
        for d in range(dim):
            draws[:, d] = rng.permutation(draws[:, d])

    # Map uniform to standard normal via inverse CDF
    draws = np.clip(draws, 1e-6, 1 - 1e-6)
    draws = stats.norm.ppf(draws)
    return draws


# ---------------------------------------------------------------------------
# BLP instrument construction
# ---------------------------------------------------------------------------

def _build_blp_instruments(data: pd.DataFrame, x_cols: list,
                           market_id: str, product_id: str) -> np.ndarray:
    """
    Construct standard BLP instruments: for each product j in market t,
    instruments are (1) own characteristics, (2) sum of rival characteristics
    in the same market.

    Parameters
    ----------
    data : pd.DataFrame
    x_cols : list of str
        Characteristic columns (excluding price).
    market_id : str
    product_id : str

    Returns
    -------
    np.ndarray
        Instrument matrix (N, 2*K) where K = len(x_cols).
    """
    N = len(data)
    K = len(x_cols)
    X = data[x_cols].values
    markets = data[market_id].values

    # Sum of all characteristics in market (including own)
    market_sums = np.zeros((N, K))
    unique_markets = np.unique(markets)
    market_indices = {m: np.where(markets == m)[0] for m in unique_markets}

    for m, idx in market_indices.items():
        market_sum = X[idx].sum(axis=0)
        market_sums[idx] = market_sum

    # Rival sum = market sum - own
    rival_sums = market_sums - X

    # Instruments: own X and rival sums
    instruments = np.column_stack([X, rival_sums])
    return instruments


# ---------------------------------------------------------------------------
# Core BLP computation
# ---------------------------------------------------------------------------

def _compute_market_shares(delta: np.ndarray, mu: np.ndarray,
                           market_ids: np.ndarray) -> np.ndarray:
    """
    Compute predicted market shares via Monte Carlo integration.

    Parameters
    ----------
    delta : np.ndarray, shape (J,)
        Mean utilities for all product-market observations.
    mu : np.ndarray, shape (J, R)
        Individual-specific utility deviations, where R = n_draws.
    market_ids : np.ndarray, shape (J,)
        Market identifier for each observation.

    Returns
    -------
    np.ndarray, shape (J,)
        Predicted market shares.
    """
    J = len(delta)
    R = mu.shape[1]
    shares = np.zeros(J)

    unique_markets = np.unique(market_ids)
    market_indices = {m: np.where(market_ids == m)[0] for m in unique_markets}

    for m, idx in market_indices.items():
        # Utility: delta_j + mu_jr for each draw r
        # Shape: (J_m, R)
        v = delta[idx, np.newaxis] + mu[idx]

        # Numerically stable softmax with outside option (utility 0)
        v_max = np.maximum(v.max(axis=0), 0.0)
        exp_v = np.exp(v - v_max)
        exp_outside = np.exp(-v_max)
        denom = exp_v.sum(axis=0) + exp_outside  # (R,)

        # Individual choice probabilities, averaged over draws
        s_ir = exp_v / denom  # (J_m, R)
        shares[idx] = s_ir.mean(axis=1)

    return shares


def _contraction_mapping(s_obs: np.ndarray, delta: np.ndarray,
                         mu: np.ndarray, market_ids: np.ndarray,
                         tol: float, maxiter: int) -> tuple[np.ndarray, bool]:
    """
    BLP contraction mapping: δ_new = δ_old + ln(s_obs) - ln(s_pred).

    Parameters
    ----------
    s_obs : np.ndarray
        Observed market shares.
    delta : np.ndarray
        Initial mean utilities.
    mu : np.ndarray
        Individual-specific deviations (J, R).
    market_ids : np.ndarray
        Market identifiers.
    tol : float
        Convergence tolerance (sup norm).
    maxiter : int
        Maximum iterations.

    Returns
    -------
    delta : np.ndarray
        Converged mean utilities.
    converged : bool
    """
    log_s_obs = np.log(np.maximum(s_obs, 1e-300))
    converged = False

    for _ in range(maxiter):
        s_pred = _compute_market_shares(delta, mu, market_ids)
        s_pred = np.maximum(s_pred, 1e-300)
        delta_new = delta + log_s_obs - np.log(s_pred)

        if np.max(np.abs(delta_new - delta)) < tol:
            converged = True
            delta = delta_new
            break
        delta = delta_new

    return delta, converged


def _compute_mu(X_random: np.ndarray, sigma: np.ndarray,
                draws: np.ndarray) -> np.ndarray:
    """
    Compute individual-specific utility deviations.

    Parameters
    ----------
    X_random : np.ndarray, shape (J, K_r)
        Random coefficient characteristics.
    sigma : np.ndarray, shape (K_r,)
        Standard deviations of random coefficients.
    draws : np.ndarray, shape (R, K_r)
        Standard normal draws.

    Returns
    -------
    np.ndarray, shape (J, R)
        mu_{jr} = Σ_k σ_k * x_{jk} * v_{rk}
    """
    # (J, K_r) @ diag(sigma) @ (K_r, R) = (J, R)
    return (X_random * sigma) @ draws.T


def _iv_regression(delta: np.ndarray, X: np.ndarray,
                   Z: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    IV/2SLS regression of delta on X using instruments Z with weighting matrix W.

    delta = X @ theta + xi
    theta = (X'Z W Z'X)^{-1} X'Z W Z'delta

    Parameters
    ----------
    delta : np.ndarray, shape (N,)
    X : np.ndarray, shape (N, K)
    Z : np.ndarray, shape (N, L)
    W : np.ndarray, shape (L, L)

    Returns
    -------
    theta : np.ndarray, shape (K,)
    xi : np.ndarray, shape (N,)
    """
    ZW = Z.T @ Z if W is None else Z.T
    # Standard 2SLS: theta = (X'Pz X)^{-1} X'Pz delta where Pz = Z(Z'Z)^{-1}Z'
    # With GMM weighting: theta = (X'Z W Z'X)^{-1} X'Z W Z'delta
    ZtX = Z.T @ X
    Ztd = Z.T @ delta
    A = ZtX.T @ W @ ZtX
    b = ZtX.T @ W @ Ztd
    theta = np.linalg.solve(A, b)
    xi = delta - X @ theta
    return theta, xi


def _gmm_objective(sigma: np.ndarray, s_obs: np.ndarray,
                   X_linear: np.ndarray, X_random: np.ndarray,
                   Z: np.ndarray, W: np.ndarray,
                   draws: np.ndarray, market_ids: np.ndarray,
                   delta_init: np.ndarray,
                   tol_inner: float, maxiter_inner: int) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate the GMM objective for given nonlinear parameters sigma.

    Returns
    -------
    obj : float
        GMM objective value.
    xi : np.ndarray
        Structural residuals.
    delta : np.ndarray
        Converged mean utilities.
    """
    sigma = np.abs(sigma)  # sigma must be non-negative
    mu = _compute_mu(X_random, sigma, draws)
    delta, converged = _contraction_mapping(
        s_obs, delta_init, mu, market_ids, tol_inner, maxiter_inner
    )

    if not converged:
        warnings.warn("BLP contraction mapping did not converge.", stacklevel=2)

    # IV regression to get linear params and structural error
    theta, xi = _iv_regression(delta, X_linear, Z, W)

    # GMM objective: xi'Z W Z'xi
    Zxi = Z.T @ xi
    obj = float(Zxi.T @ W @ Zxi)

    return obj, xi, delta


# ---------------------------------------------------------------------------
# Elasticity computation
# ---------------------------------------------------------------------------

def _compute_elasticities(delta: np.ndarray, mu: np.ndarray,
                          prices: np.ndarray, alpha: float,
                          sigma_price: float, draws: np.ndarray,
                          market_ids: np.ndarray) -> dict:
    """
    Compute own- and cross-price elasticities for each market.

    Returns
    -------
    dict
        {market_id: elasticity_matrix} where matrix is (J_m, J_m).
    """
    unique_markets = np.unique(market_ids)
    market_indices = {m: np.where(market_ids == m)[0] for m in unique_markets}
    R = mu.shape[1]
    elasticities = {}

    for m, idx in market_indices.items():
        J_m = len(idx)
        v = delta[idx, np.newaxis] + mu[idx]  # (J_m, R)
        v_max = np.maximum(v.max(axis=0), 0.0)
        exp_v = np.exp(v - v_max)
        exp_outside = np.exp(-v_max)
        denom = exp_v.sum(axis=0) + exp_outside
        s_ir = exp_v / denom  # (J_m, R)  individual choice probs

        # Price coefficient for each draw: alpha + sigma_price * v_r
        # v_r is the draw for the price random coefficient (first dim if present)
        alpha_i = alpha  # scalar; random coefficient adds sigma_price * v_r
        # For simplicity, use mean price coefficient here
        # (full random coefficient on price handled through mu already)

        p = prices[idx]  # (J_m,)
        s_j = s_ir.mean(axis=1)  # (J_m,) predicted shares

        # Elasticity matrix
        E = np.zeros((J_m, J_m))
        for j in range(J_m):
            for k in range(J_m):
                if j == k:
                    # Own elasticity: (α/s_j) * (1/R) Σ_r s_jr(1-s_jr) * p_j
                    deriv = (alpha / R) * np.sum(s_ir[j] * (1 - s_ir[j]))
                    E[j, k] = deriv * p[j] / s_j[j] if s_j[j] > 0 else 0.0
                else:
                    # Cross elasticity: -(α/s_j) * (1/R) Σ_r s_jr*s_kr * p_k
                    deriv = -(alpha / R) * np.sum(s_ir[j] * s_ir[k])
                    E[j, k] = deriv * p[k] / s_j[j] if s_j[j] > 0 else 0.0

        elasticities[m] = E

    return elasticities


# ---------------------------------------------------------------------------
# BLPResult
# ---------------------------------------------------------------------------

class BLPResult:
    """
    Results from BLP demand estimation.

    Attributes
    ----------
    linear_params : pd.Series
        Linear parameter estimates (β, α).
    nonlinear_params : pd.Series
        Nonlinear parameter estimates (σ, random coefficient std devs).
    se_linear : pd.Series
        Standard errors for linear parameters.
    se_nonlinear : pd.Series
        Standard errors for nonlinear parameters.
    mean_utility : pd.Series
        Estimated mean utility δ for each product-market.
    own_elasticities : pd.Series
        Own-price elasticities for each product-market.
    n_markets : int
        Number of markets.
    n_products : int
        Total number of product-market observations.
    gmm_objective : float
        Value of the GMM objective at the optimum.
    converged : bool
        Whether the outer-loop optimization converged.
    """

    def __init__(
        self,
        linear_params: pd.Series,
        nonlinear_params: pd.Series,
        se_linear: pd.Series,
        se_nonlinear: pd.Series,
        mean_utility: pd.Series,
        own_elasticities: pd.Series,
        n_markets: int,
        n_products: int,
        gmm_objective: float,
        converged: bool,
        _elasticity_matrices: dict,
        _market_ids: np.ndarray,
        _product_ids: np.ndarray,
        _data_index: pd.Index,
        alpha: float,
    ):
        self.linear_params = linear_params
        self.nonlinear_params = nonlinear_params
        self.se_linear = se_linear
        self.se_nonlinear = se_nonlinear
        self.mean_utility = mean_utility
        self.own_elasticities = own_elasticities
        self.n_markets = n_markets
        self.n_products = n_products
        self.gmm_objective = gmm_objective
        self.converged = converged
        self._elasticity_matrices = _elasticity_matrices
        self._market_ids = _market_ids
        self._product_ids = _product_ids
        self._data_index = _data_index
        self._alpha = alpha

    # ---- Summary ----------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary of BLP estimation results."""
        width = 72
        lines = []
        lines.append("=" * width)
        lines.append("BLP Random-Coefficients Logit Demand Estimation".center(width))
        lines.append("=" * width)
        lines.append(f"  Markets: {self.n_markets:<10}  Products (total obs): {self.n_products}")
        lines.append(f"  GMM Objective: {self.gmm_objective:.6f}"
                     f"    Converged: {self.converged}")
        lines.append("-" * width)

        # Linear parameters table
        lines.append("Linear Parameters (Mean Utility)")
        lines.append(f"  {'Variable':<20} {'Coef':>10} {'Std.Err':>10} "
                     f"{'z':>8} {'P>|z|':>8} {'[0.025':>8} {'0.975]':>8}")
        lines.append("  " + "-" * 68)
        for name in self.linear_params.index:
            coef = self.linear_params[name]
            se = self.se_linear[name]
            z = coef / se if se > 0 else np.nan
            pval = 2 * (1 - stats.norm.cdf(np.abs(z)))
            ci_lo = coef - 1.96 * se
            ci_hi = coef + 1.96 * se
            lines.append(f"  {name:<20} {coef:>10.4f} {se:>10.4f} "
                         f"{z:>8.3f} {pval:>8.4f} {ci_lo:>8.4f} {ci_hi:>8.4f}")

        lines.append("-" * width)

        # Nonlinear parameters table
        if len(self.nonlinear_params) > 0:
            lines.append("Nonlinear Parameters (Random Coefficient Std. Devs.)")
            lines.append(f"  {'Variable':<20} {'Sigma':>10} {'Std.Err':>10} "
                         f"{'z':>8} {'P>|z|':>8}")
            lines.append("  " + "-" * 50)
            for name in self.nonlinear_params.index:
                sigma = self.nonlinear_params[name]
                se = self.se_nonlinear[name]
                z = sigma / se if se > 0 else np.nan
                pval = 2 * (1 - stats.norm.cdf(np.abs(z)))
                lines.append(f"  {name:<20} {sigma:>10.4f} {se:>10.4f} "
                             f"{z:>8.3f} {pval:>8.4f}")
            lines.append("-" * width)

        # Elasticity summary
        own_e = self.own_elasticities
        lines.append("Own-Price Elasticity Summary")
        lines.append(f"  Mean: {own_e.mean():.4f}   Median: {own_e.median():.4f}   "
                     f"Min: {own_e.min():.4f}   Max: {own_e.max():.4f}")
        lines.append("=" * width)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"BLPResult(n_markets={self.n_markets}, n_products={self.n_products}, "
                f"gmm_obj={self.gmm_objective:.4f}, converged={self.converged})")

    # ---- Elasticity matrix ------------------------------------------------

    def elasticity_matrix(self, market_id=None) -> pd.DataFrame:
        """
        Return the full own- and cross-price elasticity matrix for a market.

        Parameters
        ----------
        market_id : hashable, optional
            Market to return. If None, returns the first market.

        Returns
        -------
        pd.DataFrame
            (J_m x J_m) elasticity matrix with product labels.
        """
        if market_id is None:
            market_id = list(self._elasticity_matrices.keys())[0]

        if market_id not in self._elasticity_matrices:
            raise KeyError(f"Market '{market_id}' not found.")

        E = self._elasticity_matrices[market_id]
        idx = np.where(self._market_ids == market_id)[0]
        labels = self._product_ids[idx]
        return pd.DataFrame(E, index=labels, columns=labels)

    # ---- Diversion ratios -------------------------------------------------

    def diversion_ratios(self, market_id=None) -> pd.DataFrame:
        """
        Compute diversion ratios for a given market.

        Diversion ratio D_{jk} = fraction of consumers leaving product j
        that switch to product k (rather than the outside option or other
        products).  D_{jk} = (ds_k/dp_j) / (-ds_j/dp_j).

        With logit-type models this simplifies to cross-elasticity ratios
        adjusted by shares.

        Parameters
        ----------
        market_id : hashable, optional
            Market to compute for. If None, uses the first market.

        Returns
        -------
        pd.DataFrame
        """
        E_df = self.elasticity_matrix(market_id)
        E = E_df.values
        J = E.shape[0]
        D = np.zeros((J, J))
        for j in range(J):
            own = -E[j, j]
            if own > 0:
                for k in range(J):
                    if k != j:
                        D[j, k] = E[k, j] / own  # cross / own magnitude
            # Diagonal: not defined (set 0)
        return pd.DataFrame(D, index=E_df.index, columns=E_df.columns)

    # ---- Conversion to EconometricResults --------------------------------

    def to_econometric_results(self) -> EconometricResults:
        """Convert to a standard EconometricResults object."""
        all_params = pd.concat([self.linear_params, self.nonlinear_params])
        all_se = pd.concat([self.se_linear, self.se_nonlinear])
        model_info = {
            "model_type": "BLP Random Coefficients Logit",
            "estimation_method": "GMM (two-step)",
            "gmm_objective": self.gmm_objective,
            "converged": self.converged,
        }
        data_info = {
            "n_obs": self.n_products,
            "n_markets": self.n_markets,
            "df_resid": self.n_products - len(all_params),
        }
        diagnostics = {
            "mean_own_elasticity": float(self.own_elasticities.mean()),
            "median_own_elasticity": float(self.own_elasticities.median()),
        }
        return EconometricResults(
            params=all_params,
            std_errors=all_se,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Standard-error computation
# ---------------------------------------------------------------------------

def _compute_standard_errors(
    xi: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    sigma: np.ndarray,
    X_random: np.ndarray,
    delta: np.ndarray,
    mu: np.ndarray,
    s_obs: np.ndarray,
    market_ids: np.ndarray,
    draws: np.ndarray,
    tol_inner: float,
    maxiter_inner: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute asymptotic standard errors for linear and nonlinear parameters.

    For linear params: standard GMM sandwich formula.
    For nonlinear params: numerical gradient of moments w.r.t. sigma.

    Returns
    -------
    se_linear : np.ndarray
    se_nonlinear : np.ndarray
    """
    N = len(xi)
    L = Z.shape[1]

    # --- Linear parameter SEs (GMM) ---
    # Var(theta) = (X'Z W Z'X)^{-1} X'Z W S W Z'X (X'Z W Z'X)^{-1}
    # Under optimal W, simplifies to (X'Z W Z'X)^{-1}
    ZtX = Z.T @ X
    bread = ZtX.T @ W @ ZtX
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        bread_inv = np.linalg.pinv(bread)

    # Robust meat: S = (1/N) Σ_i xi_i^2 z_i z_i'
    S = (Z * xi[:, np.newaxis]).T @ (Z * xi[:, np.newaxis]) / N
    meat = ZtX.T @ W @ S @ W @ ZtX
    var_linear = bread_inv @ meat @ bread_inv
    se_linear = np.sqrt(np.maximum(np.diag(var_linear), 0.0))

    # --- Nonlinear parameter SEs (numerical Jacobian) ---
    K_sigma = len(sigma)
    if K_sigma == 0:
        return se_linear, np.array([])

    # Jacobian of moments g = Z'xi w.r.t. sigma via finite differences
    eps = 1e-5
    g0 = Z.T @ xi  # (L,)
    Jacobian = np.zeros((L, K_sigma))

    for k in range(K_sigma):
        sigma_up = sigma.copy()
        sigma_up[k] += eps
        mu_up = _compute_mu(X_random, np.abs(sigma_up), draws)
        delta_up, _ = _contraction_mapping(
            s_obs, delta.copy(), mu_up, market_ids, tol_inner, maxiter_inner
        )
        _, xi_up = _iv_regression(delta_up, X, Z, W)
        g_up = Z.T @ xi_up
        Jacobian[:, k] = (g_up - g0) / eps

    # Var(sigma) = (G'W G)^{-1} G'W S W G (G'W G)^{-1} / N
    GtWG = Jacobian.T @ W @ Jacobian
    try:
        GtWG_inv = np.linalg.inv(GtWG)
    except np.linalg.LinAlgError:
        GtWG_inv = np.linalg.pinv(GtWG)

    meat_sigma = Jacobian.T @ W @ S @ W @ Jacobian
    var_sigma = GtWG_inv @ meat_sigma @ GtWG_inv / N
    se_nonlinear = np.sqrt(np.maximum(np.diag(var_sigma), 0.0))

    return se_linear, se_nonlinear


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def blp(
    data: pd.DataFrame,
    shares: str,
    prices: str,
    x_linear: list,
    x_random: list | None = None,
    instruments: list | None = None,
    market_id: str = "market_id",
    product_id: str = "product_id",
    n_draws: int = 200,
    method: str = "contraction",
    maxiter: int = 200,
    tol_inner: float = 1e-12,
    tol_outer: float = 1e-6,
    alpha: float = 0.05,
    seed: int | None = None,
) -> BLPResult:
    """
    Estimate a BLP random-coefficients logit demand model.

    Parameters
    ----------
    data : pd.DataFrame
        Panel of product-market observations.
    shares : str
        Column name for observed market shares (0 < s < 1).
    prices : str
        Column name for product prices.
    x_linear : list of str
        Product characteristics entering mean utility linearly.
        These will be included alongside price in the linear part.
    x_random : list of str, optional
        Subset of x_linear that also have random coefficients.
        If None, defaults to all of x_linear.
    instruments : list of str, optional
        Excluded instrument columns. If None, standard BLP instruments
        (functions of own and rival characteristics) are constructed.
    market_id : str, default 'market_id'
        Column identifying markets.
    product_id : str, default 'product_id'
        Column identifying products.
    n_draws : int, default 200
        Number of quasi-Monte Carlo draws for integration.
    method : str, default 'contraction'
        Inner-loop method. Currently only 'contraction' (NFP) is supported.
    maxiter : int, default 200
        Maximum outer-loop iterations.
    tol_inner : float, default 1e-12
        Contraction mapping tolerance.
    tol_outer : float, default 1e-6
        Outer-loop GMM tolerance.
    alpha : float, default 0.05
        Significance level (used in summary output).
    seed : int, optional
        Random seed for reproducibility of Monte Carlo draws.

    Returns
    -------
    BLPResult
        Estimation results including parameters, standard errors,
        mean utilities, and elasticities.

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.structural.blp(
    ...     data=cereal_data,
    ...     shares='share',
    ...     prices='price',
    ...     x_linear=['sugar', 'mushy'],
    ...     x_random=['sugar', 'price'],
    ...     market_id='city_id',
    ...     product_id='brand_id',
    ...     n_draws=500,
    ...     seed=42,
    ... )
    >>> print(result.summary())
    >>> elas = result.elasticity_matrix(market_id=1)
    """
    # ---- Validate inputs --------------------------------------------------
    required_cols = [shares, prices, market_id, product_id] + x_linear
    if instruments is not None:
        required_cols += instruments
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    if x_random is None:
        x_random = list(x_linear)

    for col in x_random:
        if col not in x_linear and col != prices:
            raise ValueError(
                f"x_random variable '{col}' must be in x_linear or be the price column."
            )

    data = data.copy().reset_index(drop=True)
    N = len(data)

    s_obs = data[shares].values.astype(float)
    if np.any(s_obs <= 0) or np.any(s_obs >= 1):
        raise ValueError("Market shares must be strictly between 0 and 1.")

    price_vals = data[prices].values.astype(float)
    market_ids = data[market_id].values
    product_ids = data[product_id].values
    n_markets = len(np.unique(market_ids))

    # ---- Construct matrices -----------------------------------------------
    # Linear regressors: constant, x_linear, price
    X_linear = np.column_stack([
        np.ones(N),
        data[x_linear].values.astype(float),
        price_vals,
    ])
    linear_names = ["const"] + list(x_linear) + [prices]

    # Random coefficient characteristics
    # Determine which columns in X_random correspond to x_random vars
    rc_cols = []
    rc_names = []
    for col in x_random:
        if col == prices:
            rc_cols.append(price_vals)
            rc_names.append(f"sigma_{prices}")
        else:
            rc_cols.append(data[col].values.astype(float))
            rc_names.append(f"sigma_{col}")

    X_random_mat = np.column_stack(rc_cols) if rc_cols else np.zeros((N, 0))
    K_sigma = X_random_mat.shape[1]

    # ---- Instruments ------------------------------------------------------
    if instruments is not None:
        Z_excluded = data[instruments].values.astype(float)
    else:
        # Construct BLP instruments from non-price characteristics
        iv_chars = [c for c in x_linear]
        Z_excluded = _build_blp_instruments(data, iv_chars, market_id, product_id)

    # Full instrument matrix: exogenous regressors + excluded instruments
    X_exog = np.column_stack([np.ones(N), data[x_linear].values.astype(float)])
    Z = np.column_stack([X_exog, Z_excluded])

    # ---- Monte Carlo draws ------------------------------------------------
    draws = _halton_sequence(n_draws, K_sigma, seed=seed)

    # ---- Initial values ---------------------------------------------------
    # Starting delta from logit: ln(s) - ln(s0) where s0 = 1 - sum(s) per market
    delta_init = np.zeros(N)
    for m in np.unique(market_ids):
        idx = np.where(market_ids == m)[0]
        s0 = 1.0 - s_obs[idx].sum()
        s0 = max(s0, 1e-6)
        delta_init[idx] = np.log(s_obs[idx]) - np.log(s0)

    sigma_init = np.ones(K_sigma) * 0.5

    # ---- Initial weighting matrix (identity-like) -------------------------
    W = np.linalg.inv(Z.T @ Z / N)

    # ---- Outer loop: minimize GMM objective over sigma --------------------
    best_delta = delta_init.copy()

    def _obj_wrapper(sigma_vec):
        nonlocal best_delta
        obj, xi, delta = _gmm_objective(
            sigma_vec, s_obs, X_linear, X_random_mat, Z, W,
            draws, market_ids, best_delta, tol_inner, maxiter=1000,
        )
        best_delta = delta.copy()
        return obj

    if K_sigma > 0:
        opt_result = optimize.minimize(
            _obj_wrapper,
            sigma_init,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": tol_outer, "fatol": tol_outer},
        )
        sigma_hat = np.abs(opt_result.x)
        outer_converged = opt_result.success
    else:
        sigma_hat = np.array([])
        outer_converged = True

    # ---- Two-step GMM: update weighting matrix and re-estimate ------------
    # First get residuals at current sigma
    mu_hat = _compute_mu(X_random_mat, sigma_hat, draws) if K_sigma > 0 else np.zeros((N, n_draws))
    delta_hat, inner_converged = _contraction_mapping(
        s_obs, best_delta, mu_hat, market_ids, tol_inner, maxiter=2000,
    )
    theta_hat, xi_hat = _iv_regression(delta_hat, X_linear, Z, W)

    # Optimal weighting matrix: W* = (Z' Omega Z / N)^{-1}
    # Omega = diag(xi^2) for heteroskedasticity-robust
    Omega = (Z * xi_hat[:, np.newaxis]).T @ (Z * xi_hat[:, np.newaxis]) / N
    try:
        W_opt = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        W_opt = np.linalg.pinv(Omega)

    # Re-estimate with optimal W
    best_delta_2 = delta_hat.copy()

    def _obj_wrapper_2(sigma_vec):
        nonlocal best_delta_2
        obj, xi, delta = _gmm_objective(
            sigma_vec, s_obs, X_linear, X_random_mat, Z, W_opt,
            draws, market_ids, best_delta_2, tol_inner, maxiter=1000,
        )
        best_delta_2 = delta.copy()
        return obj

    if K_sigma > 0:
        opt_result_2 = optimize.minimize(
            _obj_wrapper_2,
            sigma_hat,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": tol_outer, "fatol": tol_outer},
        )
        sigma_final = np.abs(opt_result_2.x)
        outer_converged = opt_result_2.success
    else:
        sigma_final = np.array([])

    # Final pass to get delta, theta, xi
    mu_final = _compute_mu(X_random_mat, sigma_final, draws) if K_sigma > 0 else np.zeros((N, n_draws))
    delta_final, _ = _contraction_mapping(
        s_obs, best_delta_2, mu_final, market_ids, tol_inner, maxiter=2000,
    )
    theta_final, xi_final = _iv_regression(delta_final, X_linear, Z, W_opt)
    Zxi = Z.T @ xi_final
    gmm_obj = float(Zxi.T @ W_opt @ Zxi)

    # ---- Standard errors --------------------------------------------------
    se_linear, se_nonlinear = _compute_standard_errors(
        xi_final, X_linear, Z, W_opt, sigma_final, X_random_mat,
        delta_final, mu_final, s_obs, market_ids, draws,
        tol_inner, maxiter_inner=1000,
    )

    # ---- Price coefficient for elasticities -------------------------------
    price_idx = linear_names.index(prices)
    alpha_hat = theta_final[price_idx]

    # ---- Elasticities -----------------------------------------------------
    sigma_price = 0.0
    if prices in x_random:
        sp_idx = x_random.index(prices)
        sigma_price = sigma_final[sp_idx] if K_sigma > 0 else 0.0

    elasticity_matrices = _compute_elasticities(
        delta_final, mu_final, price_vals, alpha_hat, sigma_price,
        draws, market_ids,
    )

    # Own-price elasticities
    own_elast = np.zeros(N)
    for m, idx in {m: np.where(market_ids == m)[0] for m in np.unique(market_ids)}.items():
        E = elasticity_matrices[m]
        for i, j in enumerate(idx):
            own_elast[j] = E[i, i]

    # ---- Package results --------------------------------------------------
    linear_params = pd.Series(theta_final, index=linear_names, name="coef")
    nonlinear_params = pd.Series(sigma_final, index=rc_names, name="sigma")
    se_lin = pd.Series(se_linear, index=linear_names, name="se")
    se_nonlin = pd.Series(se_nonlinear, index=rc_names, name="se")
    mean_util = pd.Series(delta_final, name="delta")
    own_e = pd.Series(own_elast, name="own_elasticity")

    return BLPResult(
        linear_params=linear_params,
        nonlinear_params=nonlinear_params,
        se_linear=se_lin,
        se_nonlinear=se_nonlin,
        mean_utility=mean_util,
        own_elasticities=own_e,
        n_markets=n_markets,
        n_products=N,
        gmm_objective=gmm_obj,
        converged=outer_converged,
        _elasticity_matrices=elasticity_matrices,
        _market_ids=market_ids,
        _product_ids=product_ids,
        _data_index=data.index,
        alpha=alpha_hat,
    )
