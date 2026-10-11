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

Without random coefficients the inversion has the closed form of Berry
(1994) and the model is a linear IV regression:

    ln s_j - ln s_0 = x_j'β + α p_j + ξ_j                      (logit)
    ln s_j - ln s_0 = x_j'β + α p_j + ρ ln s_{j|g} + ξ_j       (nested logit)

``sp.blp(..., x_random=[])`` estimates the first and ``nest=`` the second.

References
----------
Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile Prices in Market
Equilibrium. Econometrica, 63(4), 841-890. [@berry1995automobile]

Berry, S. T. (1994). Estimating Discrete-Choice Models of Product
Differentiation. RAND Journal of Economics, 25(2), 242-262.
[@berry1994estimating]

Nevo, A. (2000). A Practitioner's Guide to Estimation of Random-Coefficients
Logit Models of Demand. Journal of Economics & Management Strategy, 9(4), 513-548. [@nevo2000practitioner]
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._result_serialize import ResultProtocolMixin
from ..core._collinear import independent_columns
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility

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
        raise ValueError(
            f"x_random has {dim} variables; max supported is {len(primes)}."
        )

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

    if dim == 0:
        return np.zeros((n, 0))
    rng = np.random.default_rng(seed)
    draws = np.column_stack([_halton_1d(n, primes[d]) for d in range(dim)])

    if seed is not None:
        for d in range(dim):
            draws[:, d] = rng.permutation(draws[:, d])

    # Map uniform to standard normal via inverse CDF
    draws = np.clip(draws, 1e-6, 1 - 1e-6)
    draws = stats.norm.ppf(draws)
    return np.asarray(draws, dtype=float)


# ---------------------------------------------------------------------------
# BLP instrument construction
# ---------------------------------------------------------------------------


def _group_sum(values: np.ndarray, *keys: np.ndarray) -> np.ndarray:
    """Sum of each column of ``values`` within the cells of ``keys``."""
    frame = pd.DataFrame(values)
    by = [pd.Series(k).reset_index(drop=True) for k in keys]
    return np.asarray(frame.groupby(by, sort=False).transform("sum"), dtype=float)


def blp_instruments(
    data: pd.DataFrame,
    x: list,
    market_id: str = "market_id",
    firm_id: Optional[str] = None,
    nest: Optional[str] = None,
    count: bool = True,
) -> pd.DataFrame:
    """
    Characteristics-based instruments for differentiated-products demand.

    For every characteristic in ``x`` (and the product count), the sums over
    the *other* products a product competes with in its market: the firm's
    own other products and its rivals' products when ``firm_id`` is given
    (Berry, Levinsohn & Pakes 1995, section 5.1), all other products in the
    market otherwise. With ``nest`` the sums over the other products of the
    same nest are added; they move the within-nest share of a nested logit.

    Parameters
    ----------
    data : pd.DataFrame
        One row per product and market.
    x : list of str
        Exogenous product characteristics (not the price).
    market_id : str, default 'market_id'
        Column identifying markets.
    firm_id : str, optional
        Column identifying the firm that sells each product. With it the
        market sum is split into ``own_<x>`` (the firm's other products) and
        ``rival_<x>`` (other firms' products).
    nest : str, optional
        Column identifying the nest of each product; adds ``nest_<x>``, the
        sum over the other products of the same nest and market.
    count : bool, default True
        Also build the sums of a constant, i.e. the number of other
        products (``own_count``, ``rival_count``, ``nest_count``).

    Returns
    -------
    pd.DataFrame
        One column per instrument, indexed like ``data``. The columns can be
        passed to ``sp.blp(instruments=...)`` or written into an ``sp.iv``
        formula. A column that does not vary within the data (a rival count
        when every market has the same number of products, say) is still
        returned; ``sp.blp`` drops redundant instruments itself.

    Notes
    -----
    The instruments are valid when a product's unobserved quality is mean
    independent of the observed characteristics of all products in its
    market. They say nothing about whether that holds; an overidentification
    test after estimation is the usual check.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "market_id": [1, 1, 1, 2, 2],
    ...     "firm": ["a", "a", "b", "a", "b"],
    ...     "size": [1.0, 2.0, 4.0, 3.0, 5.0],
    ... })
    >>> z = sp.blp_instruments(df, ["size"], firm_id="firm")
    >>> z["own_size"].tolist()
    [2.0, 1.0, 0.0, 0.0, 0.0]
    >>> z["rival_size"].tolist()
    [4.0, 4.0, 3.0, 5.0, 3.0]
    >>> z["rival_count"].tolist()
    [1.0, 1.0, 2.0, 1.0, 1.0]

    References
    ----------
    [@berry1995automobile]
    """
    x = list(x)
    needed = [market_id] + x + [c for c in (firm_id, nest) if c is not None]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")
    names = (["count"] if count else []) + [str(c) for c in x]
    if not names:
        raise MethodIncompatibility(
            "blp_instruments needs at least one characteristic in x."
        )
    X = data[x].to_numpy(dtype=float).reshape(len(data), len(x))
    if count:
        X = np.column_stack([np.ones(len(data)), X])
    if not np.all(np.isfinite(X)):
        raise DataInsufficient(
            "blp_instruments: the characteristics contain missing values."
        )
    market = data[market_id].to_numpy()
    market_sum = _group_sum(X, market)
    out = {}
    if firm_id is None:
        for k, name in enumerate(names):
            out[f"rival_{name}"] = market_sum[:, k] - X[:, k]
    else:
        firm_sum = _group_sum(X, market, data[firm_id].to_numpy())
        for k, name in enumerate(names):
            out[f"own_{name}"] = firm_sum[:, k] - X[:, k]
        for k, name in enumerate(names):
            out[f"rival_{name}"] = market_sum[:, k] - firm_sum[:, k]
    if nest is not None:
        nest_sum = _group_sum(X, market, data[nest].to_numpy())
        for k, name in enumerate(names):
            out[f"nest_{name}"] = nest_sum[:, k] - X[:, k]
    return pd.DataFrame(out, index=data.index)


# ---------------------------------------------------------------------------
# Core BLP computation
# ---------------------------------------------------------------------------


def _compute_market_shares(
    delta: np.ndarray, mu: np.ndarray, market_ids: np.ndarray
) -> np.ndarray:
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


def _contraction_mapping(
    s_obs: np.ndarray,
    delta: np.ndarray,
    mu: np.ndarray,
    market_ids: np.ndarray,
    tol: float,
    maxiter: int,
) -> tuple[np.ndarray, bool]:
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


def _compute_mu(
    X_random: np.ndarray, sigma: np.ndarray, draws: np.ndarray
) -> np.ndarray:
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
    return np.asarray((X_random * sigma) @ draws.T, dtype=float)


def _iv_regression(
    delta: np.ndarray, X: np.ndarray, Z: np.ndarray, W: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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
    # Standard 2SLS: theta = (X'Pz X)^{-1} X'Pz delta where Pz = Z(Z'Z)^{-1}Z'
    # With GMM weighting: theta = (X'Z W Z'X)^{-1} X'Z W Z'delta
    ZtX = Z.T @ X
    Ztd = Z.T @ delta
    A = ZtX.T @ W @ ZtX
    b = ZtX.T @ W @ Ztd
    theta = np.linalg.solve(A, b)
    xi = delta - X @ theta
    return theta, xi


def _gmm_objective(
    sigma: np.ndarray,
    s_obs: np.ndarray,
    X_linear: np.ndarray,
    X_random: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
    draws: np.ndarray,
    market_ids: np.ndarray,
    delta_init: np.ndarray,
    tol_inner: float,
    maxiter_inner: int,
) -> tuple[float, np.ndarray, np.ndarray]:
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


def _compute_elasticities(
    delta: np.ndarray,
    mu: np.ndarray,
    prices: np.ndarray,
    alpha: float,
    sigma_price: float,
    draws: np.ndarray,
    market_ids: np.ndarray,
    price_draw_col: Optional[int] = None,
) -> dict:
    """
    Compute own- and cross-price elasticities for each market.

    With a random coefficient on price the consumer-level price coefficient
    is ``alpha_r = alpha + sigma_price * nu_r`` (``nu_r`` the price column of
    ``draws``), and

        ds_j/dp_k = (1/R) sum_r alpha_r s_jr (1{j=k} - s_kr),
        E_jk      = ds_j/dp_k * p_k / s_j.

    Without one, ``alpha_r = alpha`` for every draw.

    Returns
    -------
    dict
        {market_id: elasticity_matrix} where matrix is (J_m, J_m).
    """
    unique_markets = np.unique(market_ids)
    market_indices = {m: np.where(market_ids == m)[0] for m in unique_markets}
    R = mu.shape[1]
    if price_draw_col is not None and sigma_price != 0.0:
        alpha_r = alpha + sigma_price * draws[:, price_draw_col]  # (R,)
    else:
        alpha_r = np.full(R, alpha)
    elasticities = {}

    for m, idx in market_indices.items():
        v = delta[idx, np.newaxis] + mu[idx]  # (J_m, R)
        v_max = np.maximum(v.max(axis=0), 0.0)
        exp_v = np.exp(v - v_max)
        exp_outside = np.exp(-v_max)
        denom = exp_v.sum(axis=0) + exp_outside
        s_ir = exp_v / denom  # (J_m, R)  individual choice probs

        p = prices[idx]  # (J_m,)
        s_j = s_ir.mean(axis=1)  # (J_m,) predicted shares

        a_s = s_ir * alpha_r[np.newaxis, :]  # alpha_r * s_jr
        # dS[j, k] = (1/R) sum_r alpha_r s_jr (1{j=k} - s_kr)
        dS = np.diag(a_s.mean(axis=1)) - (a_s @ s_ir.T) / R
        with np.errstate(divide="ignore", invalid="ignore"):
            E = dS * p[np.newaxis, :] / s_j[:, np.newaxis]
        E[~np.isfinite(E)] = 0.0

        elasticities[m] = E

    return elasticities


def _nested_logit_elasticities(
    shares: np.ndarray,
    within: np.ndarray,
    prices: np.ndarray,
    alpha: float,
    rho: float,
    market_ids: np.ndarray,
    nests: np.ndarray,
) -> dict:
    """
    Price elasticities of the one-level nested logit (Berry 1994).

    With ``alpha`` the coefficient on price, ``s_{j|g}`` the within-nest
    share and ``rho`` the nesting parameter,

        E_jj = alpha p_j [1/(1-rho) - rho/(1-rho) s_{j|g} - s_j]
        E_jk = -alpha p_k [rho/(1-rho) s_{k|g} + s_k]     (k in the nest of j)
        E_jk = -alpha p_k s_k                             (other nests)

    ``rho = 0`` gives the plain logit.
    """
    out = {}
    r = rho / (1.0 - rho)
    for m in np.unique(market_ids):
        idx = np.where(market_ids == m)[0]
        s, w, p = shares[idx], within[idx], prices[idx]
        same = nests[idx][:, None] == nests[idx][None, :]
        E = -alpha * p[None, :] * (s[None, :] + r * w[None, :] * same)
        E[np.diag_indices(len(idx))] = alpha * p * (1.0 / (1.0 - rho) - r * w - s)
        out[m] = E
    return out


# ---------------------------------------------------------------------------
# BLPResult
# ---------------------------------------------------------------------------


_MODEL_TITLES = {
    "random_coefficients": "BLP Random-Coefficients Logit",
    "logit": "Logit (Berry 1994)",
    "nested_logit": "Nested Logit (Berry 1994)",
}


class BLPResult(ResultProtocolMixin):
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
        Value of the GMM objective at the optimum, ``(Z'xi)' W (Z'xi)`` with
        the step-2 weighting matrix -- ``N`` times Hansen's J statistic
        ``N * gbar' W gbar`` (the value pyblp reports).
    converged : bool
        Whether the outer-loop optimization converged.
    model : str
        ``'random_coefficients'``, ``'logit'`` or ``'nested_logit'``.
    rho : float or None
        Nesting parameter of the nested logit (also in ``nonlinear_params``).
    instruments : list of str
        Excluded instruments the moments were built from.
    omitted_instruments : list of dict
        Instruments dropped as redundant, ``{'variable', 'reason'}`` each.
    gmm_steps : int
        1 (weighting matrix ``(Z'Z)^{-1}``, the 2SLS weighting) or 2.
    hansen_j, hansen_df, hansen_p : float, int, float
        Hansen's test of the overidentifying restrictions after two-step
        GMM; ``None`` after one step or when the model is just identified.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for m in range(10):  # 10 markets, 5 products each
    ...     x1 = rng.normal(0, 1, 5)
    ...     x2 = rng.normal(0, 1, 5)
    ...     price = rng.uniform(1, 3, 5)
    ...     delta = x1 + 0.5 * x2 - price + rng.normal(0, 0.2, 5)
    ...     ex = np.exp(delta)
    ...     share = ex / (1 + ex.sum())
    ...     for j in range(5):
    ...         rows.append({"market_id": m, "product_id": j,
    ...                      "share": share[j], "price": price[j],
    ...                      "x1": x1[j], "x2": x2[j]})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.blp(df, shares="share", prices="price",  # doctest: +SKIP
    ...              x_linear=["x1", "x2"], x_random=["x1"],
    ...              n_draws=50, maxiter=50, seed=0)
    >>> type(res).__name__  # doctest: +SKIP
    'BLPResult'
    >>> (res.n_markets, res.n_products)  # doctest: +SKIP
    (10, 50)
    >>> bool("x1" in res.linear_params.index)  # doctest: +SKIP
    True
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
        model: str = "random_coefficients",
        rho: Optional[float] = None,
        instruments: Optional[list] = None,
        omitted_instruments: Optional[list] = None,
        gmm_steps: int = 2,
        hansen_j: Optional[float] = None,
        hansen_df: Optional[int] = None,
        hansen_p: Optional[float] = None,
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
        self.model = model
        self.rho = rho
        self.instruments = list(instruments or [])
        self.omitted_instruments = list(omitted_instruments or [])
        self.gmm_steps = gmm_steps
        self.hansen_j = hansen_j
        self.hansen_df = hansen_df
        self.hansen_p = hansen_p

    # ---- Summary ----------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary of BLP estimation results."""
        width = 72
        lines = []
        lines.append("=" * width)
        lines.append(f"{_MODEL_TITLES[self.model]} Demand Estimation".center(width))
        lines.append("=" * width)
        lines.append(
            f"  Markets: {self.n_markets:<10}  Products (total obs): {self.n_products}"
        )
        lines.append(
            f"  GMM Objective: {self.gmm_objective:.6f}"
            f"    Converged: {self.converged}"
        )
        lines.append(
            f"  GMM steps: {self.gmm_steps:<8}  "
            f"Excluded instruments: {len(self.instruments)}"
        )
        if self.hansen_j is not None:
            lines.append(
                f"  Hansen J: {self.hansen_j:.3f}  (df {self.hansen_df}, "
                f"p = {self.hansen_p:.4f})"
            )
        lines.append("-" * width)

        # Linear parameters table
        lines.append("Linear Parameters (Mean Utility)")
        lines.append(
            f"  {'Variable':<20} {'Coef':>10} {'Std.Err':>10} "
            f"{'z':>8} {'P>|z|':>8} {'[0.025':>8} {'0.975]':>8}"
        )
        lines.append("  " + "-" * 68)
        for name in self.linear_params.index:
            coef = self.linear_params[name]
            se = self.se_linear[name]
            z = coef / se if se > 0 else np.nan
            pval = 2 * stats.norm.sf(np.abs(z))
            ci_lo = coef - 1.96 * se
            ci_hi = coef + 1.96 * se
            lines.append(
                f"  {name:<20} {coef:>10.4f} {se:>10.4f} "
                f"{z:>8.3f} {pval:>8.4f} {ci_lo:>8.4f} {ci_hi:>8.4f}"
            )

        lines.append("-" * width)

        # Nonlinear parameters table
        if len(self.nonlinear_params) > 0:
            lines.append(
                "Nesting Parameter"
                if self.model == "nested_logit"
                else "Nonlinear Parameters (Random Coefficient Std. Devs.)"
            )
            lines.append(
                f"  {'Variable':<20} {'Estimate':>10} {'Std.Err':>10} "
                f"{'z':>8} {'P>|z|':>8}"
            )
            lines.append("  " + "-" * 50)
            for name in self.nonlinear_params.index:
                sigma = self.nonlinear_params[name]
                se = self.se_nonlinear[name]
                z = sigma / se if se > 0 else np.nan
                pval = 2 * stats.norm.sf(np.abs(z))
                lines.append(
                    f"  {name:<20} {sigma:>10.4f} {se:>10.4f} "
                    f"{z:>8.3f} {pval:>8.4f}"
                )
            lines.append("-" * width)

        # Elasticity summary
        own_e = self.own_elasticities
        lines.append("Own-Price Elasticity Summary")
        lines.append(
            f"  Mean: {own_e.mean():.4f}   Median: {own_e.median():.4f}   "
            f"Min: {own_e.min():.4f}   Max: {own_e.max():.4f}"
        )
        lines.append("=" * width)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"BLPResult(n_markets={self.n_markets}, n_products={self.n_products}, "
            f"gmm_obj={self.gmm_objective:.4f}, converged={self.converged})"
        )

    # ---- Elasticity matrix ------------------------------------------------

    def elasticity_matrix(self, market_id: Optional[Any] = None) -> pd.DataFrame:
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

    def diversion_ratios(self, market_id: Optional[Any] = None) -> pd.DataFrame:
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
            "model_type": _MODEL_TITLES[self.model],
            "estimation_method": (
                "GMM (two-step)" if self.gmm_steps == 2 else "GMM (one-step, 2SLS)"
            ),
            "gmm_objective": self.gmm_objective,
            "converged": self.converged,
            "instruments": list(self.instruments),
            "omitted_instruments": list(self.omitted_instruments),
        }
        data_info = {
            "n_obs": self.n_products,
            "n_markets": self.n_markets,
            "df_resid": self.n_products - len(all_params),
        }
        diagnostics: dict[str, Any] = {
            "mean_own_elasticity": float(self.own_elasticities.mean()),
            "median_own_elasticity": float(self.own_elasticities.median()),
        }
        if self.hansen_j is not None:
            diagnostics.update(
                {
                    "Hansen J": self.hansen_j,
                    "Hansen J df": self.hansen_df,
                    "Hansen J p-value": self.hansen_p,
                }
            )
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


def _delta_jacobian(
    delta: np.ndarray,
    mu: np.ndarray,
    X_random: np.ndarray,
    draws: np.ndarray,
    market_ids: np.ndarray,
) -> np.ndarray:
    """Analytic ``d delta / d sigma`` by the implicit function theorem.

    Within market ``m`` the contraction solves ``s(delta, sigma) = s_obs``, so
    ``d delta / d sigma = -(ds/d delta)^{-1} ds/d sigma`` with

        ds_j/d delta_k = (1/R) sum_r s_jr (1{j=k} - s_kr)
        ds_j/d sigma_l = (1/R) sum_r s_jr nu_rl (x_jl - sum_k s_kr x_kl).
    """
    N = len(delta)
    K = X_random.shape[1]
    R = mu.shape[1]
    D = np.zeros((N, K))
    for m in np.unique(market_ids):
        idx = np.where(market_ids == m)[0]
        v = delta[idx, np.newaxis] + mu[idx]
        v_max = np.maximum(v.max(axis=0), 0.0)
        exp_v = np.exp(v - v_max)
        s_ir = exp_v / (exp_v.sum(axis=0) + np.exp(-v_max))  # (J_m, R)
        ds_dd = np.diag(s_ir.mean(axis=1)) - (s_ir @ s_ir.T) / R
        Xm = X_random[idx]  # (J_m, K)
        ds_dsig = np.empty((len(idx), K))
        for c in range(K):
            xbar_r = Xm[:, c] @ s_ir  # (R,) sum_k s_kr x_kc
            ds_dsig[:, c] = (
                s_ir * draws[:, c][np.newaxis, :] * (Xm[:, [c]] - xbar_r[np.newaxis, :])
            ).mean(axis=1)
        D[idx] = -np.linalg.solve(ds_dd, ds_dsig)
    return D


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
    Heteroskedasticity-robust GMM standard errors for ``(beta, sigma)`` jointly.

    Moments ``g(theta) = Z' xi(theta) / N`` with ``xi = delta(sigma) - X beta``.
    Jacobian ``G = [-Z'X, Z' d delta/d sigma] / N`` (``d delta / d sigma`` from
    the implicit function theorem), ``S = sum_i xi_i^2 z_i z_i' / N`` and

        Var(theta) = (G'WG)^{-1} G'W S W G (G'WG)^{-1} / N,

    the covariance pyblp reports with ``se_type="robust"`` and the weighting
    matrix ``W`` of the final GMM step. The linear block therefore carries
    the estimation error of ``sigma``.

    Returns
    -------
    se_linear : np.ndarray
    se_nonlinear : np.ndarray
    """
    N = len(xi)
    K_beta = X.shape[1]
    K_sigma = len(sigma)

    G_beta = -(Z.T @ X) / N
    if K_sigma > 0:
        D = _delta_jacobian(delta, mu, X_random, draws, market_ids)
        # sigma enters as |sigma|; d|sigma|/d sigma = sign(sigma) (1 at sigma > 0)
        D = D * np.where(sigma < 0, -1.0, 1.0)[np.newaxis, :]
        G = np.column_stack([G_beta, (Z.T @ D) / N])
    else:
        G = G_beta

    Zxi = Z * xi[:, np.newaxis]
    S = Zxi.T @ Zxi / N
    bread = G.T @ W @ G
    try:
        bread_inv = np.linalg.inv(bread)
    except np.linalg.LinAlgError:
        bread_inv = np.linalg.pinv(bread)
    V = bread_inv @ (G.T @ W @ S @ W @ G) @ bread_inv / N
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return se[:K_beta], se[K_beta:]


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
    firm_id: str | None = None,
    nest: str | None = None,
    gmm_steps: int = 2,
) -> BLPResult:
    """
    Estimate a BLP random-coefficients logit demand model.

    With ``x_random=[]`` the model is the logit of Berry (1994), a linear
    IV regression of ``ln s_j - ln s_0`` on characteristics and price; with
    ``nest=`` as well it is his nested logit, which adds the log within-nest
    share with coefficient ``rho``.

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
        If None, defaults to all of x_linear. An empty list gives the
        logit (or, with ``nest``, the nested logit) of Berry (1994).
    instruments : list of str, optional
        Excluded instrument columns. If None, the instruments of
        ``sp.blp_instruments(data, x_linear, market_id, firm_id, nest)``
        are used: sums of the characteristics (and the count) of the other
        products in the market. Redundant instruments are dropped, with a
        warning when they were supplied by the caller.
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
    firm_id : str, optional
        Column identifying the firm selling each product. Only used to
        build the default instruments, which then separate the firm's own
        other products from its rivals' (BLP 1995, section 5.1).
    nest : str, optional
        Column identifying the nest of each product. Requires
        ``x_random=[]``: the nested logit is estimated, the log within-nest
        share is treated as endogenous, and its coefficient is reported as
        ``rho`` (0 is the plain logit; towards 1 products of a nest become
        perfect substitutes).
    gmm_steps : int, default 2
        2 for two-step efficient GMM. 1 stops at the weighting matrix
        ``(Z'Z)^{-1}``; without random coefficients the estimates are then
        exactly two-stage least squares, with heteroskedasticity-robust
        (HC0) standard errors.

    Returns
    -------
    BLPResult
        Estimation results including parameters, standard errors,
        mean utilities, and elasticities.

    Notes
    -----
    The logit restricts substitution to be proportional to market shares,
    and the nested logit relaxes that only across the chosen nests; random
    coefficients are what lets similar products be closer substitutes.

    The default instruments rest on the characteristics of all products in
    a market being unrelated to a product's unobserved quality. ``hansen_p``
    on the result tests the overidentifying restrictions; on the automobile
    data of BLP (1995) the logit specification rejects them.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> # Multinomial-logit DGP with endogenous price + two cost instruments.
    >>> rng = np.random.default_rng(1)
    >>> rows = []
    >>> for mkt in range(20):
    ...     x = rng.uniform(0, 2, 4)
    ...     xi = rng.normal(0, 0.2, 4)        # demand shock -> price endogeneity
    ...     cost = rng.uniform(0.5, 2, 4)
    ...     z2 = rng.uniform(0, 1, 4)
    ...     price = 0.8 * cost + 0.5 * z2 + 0.4 * xi + rng.uniform(0, 0.5, 4)
    ...     delta = 1.0 * x - 1.5 * price + xi
    ...     ev = np.exp(delta)
    ...     shares = ev / (1 + ev.sum())      # logit shares with an outside good
    ...     for j in range(4):
    ...         rows.append({"market_id": mkt, "product_id": j,
    ...                      "share": shares[j], "price": price[j],
    ...                      "x1": x[j], "cost": cost[j], "z2": z2[j]})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.blp(
    ...     df, shares='share', prices='price', x_linear=['x1'],
    ...     x_random=['x1'], instruments=['cost', 'z2'], n_draws=30, seed=0,
    ... )
    >>> _ = result.summary()
    >>> elas = result.elasticity_matrix(market_id=1)

    Berry (1994) logit on the same data, as two-stage least squares:

    >>> logit = sp.blp(
    ...     df, shares='share', prices='price', x_linear=['x1'],
    ...     x_random=[], instruments=['cost', 'z2'], gmm_steps=1,
    ... )
    >>> logit.model
    'logit'
    >>> bool(logit.linear_params['price'] < 0)
    True

    References
    ----------
    [@berry1995automobile] [@berry1994estimating] [@nevo2000practitioner]
    """
    # ---- Validate inputs --------------------------------------------------
    x_linear = list(x_linear)
    required_cols = [shares, prices, market_id, product_id] + x_linear
    if instruments is not None:
        instruments = list(instruments)
        required_cols += instruments
    required_cols += [c for c in (firm_id, nest) if c is not None]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    if x_random is None:
        x_random = list(x_linear)
    x_random = list(x_random)
    if gmm_steps not in (1, 2):
        raise MethodIncompatibility(f"gmm_steps must be 1 or 2, got {gmm_steps!r}.")
    if nest is not None and x_random:
        raise MethodIncompatibility(
            "nest= estimates the nested logit of Berry (1994) and needs "
            "x_random=[]; a nested logit with random coefficients is not "
            "implemented.",
            recovery_hint="Pass x_random=[] or drop nest=.",
        )
    model = "random_coefficients" if x_random else "logit"
    if nest is not None:
        model = "nested_logit"

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
    inside = _group_sum(s_obs[:, None], market_ids)[:, 0]
    if np.any(inside >= 1.0):
        raise DataInsufficient(
            "Market shares sum to 1 or more in at least one market; the "
            "model needs an outside good with a positive share."
        )

    # ---- Construct matrices -----------------------------------------------
    # Linear regressors: constant, x_linear, price
    X_linear = np.column_stack(
        [
            np.ones(N),
            data[x_linear].values.astype(float),
            price_vals,
        ]
    )
    linear_names = ["const"] + list(x_linear) + [prices]
    n_endog = 1
    within = np.ones(N)
    if nest is not None:
        nest_ids = data[nest].values
        if pd.isna(nest_ids).any():
            raise DataInsufficient(f"nest column '{nest}' contains missing values.")
        within = s_obs / _group_sum(s_obs[:, None], market_ids, nest_ids)[:, 0]
        if np.all(within > 1.0 - 1e-12):
            raise MethodIncompatibility(
                f"blp: every product is alone in its nest ('{nest}'), so the "
                "within-nest share is 1 throughout and the nesting parameter "
                "is not identified.",
                recovery_hint="Use coarser nests, or drop nest= for the logit.",
            )
        X_linear = np.column_stack([X_linear, np.log(within)])
        linear_names = linear_names + ["rho"]
        n_endog = 2

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
        Z_excluded = data[instruments].values.astype(float).reshape(N, -1)
        z_names = [str(c) for c in instruments]
    else:
        built = blp_instruments(
            data, x_linear, market_id=market_id, firm_id=firm_id, nest=nest
        )
        Z_excluded = built.to_numpy(dtype=float)
        z_names = list(built.columns)

    # Full instrument matrix: exogenous regressors + excluded instruments.
    # Redundant instruments go (the count of rival products is the constant
    # when every market has the same number of products, say): a singular
    # Z'Z has no inverse to weight the moments with.
    X_exog = np.column_stack([np.ones(N), data[x_linear].values.astype(float)])
    n_exog = X_exog.shape[1]
    Z = np.column_stack([X_exog, Z_excluded])
    keep, omitted = independent_columns(Z, ["const"] + x_linear + z_names)
    if any(j not in keep for j in range(n_exog)):
        raise MethodIncompatibility(
            "blp: the characteristics in x_linear are collinear ("
            + "; ".join(f"{o['variable']}: {o['reason']}" for o in omitted)
            + ").",
            recovery_hint="Drop the redundant characteristic from x_linear.",
        )
    if omitted and instruments is not None:
        warnings.warn(
            "blp: "
            + "; ".join(
                f"instrument {o['variable']} omitted ({o['reason']})" for o in omitted
            )
            + ".",
            UserWarning,
            stacklevel=2,
        )
    Z = Z[:, keep]
    z_names = [z_names[j - n_exog] for j in keep if j >= n_exog]
    n_needed = n_endog + len(x_random)
    if len(z_names) < n_needed:
        raise DataInsufficient(
            f"blp: {len(z_names)} independent excluded instrument(s) for "
            f"{n_needed} parameter(s) that need one (price"
            + (", the within-nest share" if nest is not None else "")
            + (f", {len(x_random)} random coefficient(s)" if x_random else "")
            + ").",
            recovery_hint=(
                "Pass more instruments=, or firm_id= so the default "
                "instruments separate own from rival products."
            ),
        )

    # ---- Monte Carlo draws ------------------------------------------------
    # without random coefficients there is nothing to integrate over
    n_nodes = n_draws if K_sigma > 0 else 1
    draws = _halton_sequence(n_nodes, K_sigma, seed=seed)

    # ---- Initial values ---------------------------------------------------
    # Starting delta from logit: ln(s) - ln(s0) where s0 = 1 - sum(s) per market
    delta_init = np.log(s_obs) - np.log(1.0 - inside)

    sigma_init = np.ones(K_sigma) * 0.5

    # ---- Initial weighting matrix (identity-like) -------------------------
    W = np.linalg.inv(Z.T @ Z / N)

    # ---- Outer loop: minimize GMM objective over sigma --------------------
    best_delta = delta_init.copy()

    def _obj_wrapper(sigma_vec: np.ndarray) -> float:
        nonlocal best_delta
        obj, xi, delta = _gmm_objective(
            sigma_vec,
            s_obs,
            X_linear,
            X_random_mat,
            Z,
            W,
            draws,
            market_ids,
            best_delta,
            tol_inner,
            maxiter_inner=1000,
        )
        best_delta = delta.copy()
        return float(obj)

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
    mu_hat = (
        _compute_mu(X_random_mat, sigma_hat, draws)
        if K_sigma > 0
        else np.zeros((N, n_nodes))
    )
    delta_hat, inner_converged = _contraction_mapping(
        s_obs,
        best_delta,
        mu_hat,
        market_ids,
        tol_inner,
        maxiter=2000,
    )
    theta_hat, xi_hat = _iv_regression(delta_hat, X_linear, Z, W)

    # Optimal weighting matrix: W* = (Z' Omega Z / N)^{-1}
    # Omega = diag(xi^2) for heteroskedasticity-robust
    Omega = (Z * xi_hat[:, np.newaxis]).T @ (Z * xi_hat[:, np.newaxis]) / N
    if gmm_steps == 1:
        W_opt = W
    else:
        try:
            W_opt = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            W_opt = np.linalg.pinv(Omega)

    # Re-estimate with optimal W
    best_delta_2 = delta_hat.copy()

    def _obj_wrapper_2(sigma_vec: np.ndarray) -> float:
        nonlocal best_delta_2
        obj, xi, delta = _gmm_objective(
            sigma_vec,
            s_obs,
            X_linear,
            X_random_mat,
            Z,
            W_opt,
            draws,
            market_ids,
            best_delta_2,
            tol_inner,
            maxiter_inner=1000,
        )
        best_delta_2 = delta.copy()
        return float(obj)

    if K_sigma > 0 and gmm_steps == 1:
        sigma_final = sigma_hat
    elif K_sigma > 0:
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
    mu_final = (
        _compute_mu(X_random_mat, sigma_final, draws)
        if K_sigma > 0
        else np.zeros((N, n_nodes))
    )
    delta_final, _ = _contraction_mapping(
        s_obs,
        best_delta_2,
        mu_final,
        market_ids,
        tol_inner,
        maxiter=2000,
    )
    theta_final, xi_final = _iv_regression(delta_final, X_linear, Z, W_opt)
    Zxi = Z.T @ xi_final
    gmm_obj = float(Zxi.T @ W_opt @ Zxi)
    hansen_j = hansen_df = hansen_p = None
    n_overid = Z.shape[1] - X_linear.shape[1] - K_sigma
    if gmm_steps == 2 and n_overid > 0:
        hansen_j = gmm_obj / N
        hansen_df = int(n_overid)
        hansen_p = float(stats.chi2.sf(hansen_j, hansen_df))

    # ---- Standard errors --------------------------------------------------
    se_linear, se_nonlinear = _compute_standard_errors(
        xi_final,
        X_linear,
        Z,
        W_opt,
        sigma_final,
        X_random_mat,
        delta_final,
        mu_final,
        s_obs,
        market_ids,
        draws,
        tol_inner,
        maxiter_inner=1000,
    )

    # ---- Price coefficient for elasticities -------------------------------
    price_idx = linear_names.index(prices)
    alpha_hat = theta_final[price_idx]

    # ---- Elasticities -----------------------------------------------------
    sigma_price = 0.0
    price_draw_col = None
    if prices in x_random:
        price_draw_col = x_random.index(prices)
        sigma_price = float(sigma_final[price_draw_col]) if K_sigma > 0 else 0.0

    rho_hat = None
    if nest is not None:
        rho_hat = float(theta_final[-1])
        if not 0.0 <= rho_hat < 1.0:
            warnings.warn(
                f"blp: the nesting parameter is {rho_hat:.4f}, outside [0, 1); "
                "the nested logit is then not consistent with utility "
                "maximisation and the elasticities are not meaningful. The "
                "nests or the instruments for the within-nest share are the "
                "usual cause.",
                UserWarning,
                stacklevel=2,
            )
        # Berry's mean utility: what is left of ln s_j - ln s_0 once the
        # within-nest term is taken out
        delta_final = delta_final - rho_hat * np.log(within)
        elasticity_matrices = _nested_logit_elasticities(
            s_obs, within, price_vals, alpha_hat, rho_hat, market_ids, nest_ids
        )
    else:
        elasticity_matrices = _compute_elasticities(
            delta_final,
            mu_final,
            price_vals,
            alpha_hat,
            sigma_price,
            draws,
            market_ids,
            price_draw_col=price_draw_col,
        )

    # Own-price elasticities
    own_elast = np.zeros(N)
    for m, idx in {
        m: np.where(market_ids == m)[0] for m in np.unique(market_ids)
    }.items():
        E = elasticity_matrices[m]
        for i, j in enumerate(idx):
            own_elast[j] = E[i, i]

    # ---- Package results --------------------------------------------------
    if nest is not None:
        # rho is estimated in the linear step; it is reported with the
        # nonlinear parameters, which is where a reader looks for it
        rc_names = ["rho"]
        sigma_final, se_nonlinear = theta_final[-1:], se_linear[-1:]
        theta_final, se_linear = theta_final[:-1], se_linear[:-1]
        linear_names = linear_names[:-1]
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
        model=model,
        rho=rho_hat,
        instruments=z_names,
        omitted_instruments=omitted,
        gmm_steps=gmm_steps,
        hansen_j=hansen_j,
        hansen_df=hansen_df,
        hansen_p=hansen_p,
    )
