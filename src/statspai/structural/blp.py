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
1. Inner loop: BLP contraction mapping (accelerated, per market) to recover
   mean utilities δ
2. Outer loop: two-step GMM over nonlinear parameters σ (random coefficient
   std devs, bounded below by zero) with the analytic gradient, the first
   step from several starting points

The numerical core is in ``_blp_core.py``, the result object in
``_blp_result.py``, diversion ratios and markups in ``_blp_post.py``.

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
Logit Models of Demand. Journal of Economics & Management Strategy, 9(4),
513-548. [@nevo2000practitioner]
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._input_validation import check_columns
from ..core._collinear import independent_columns
from ..core._validate import require_numeric_columns
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._blp_core import (  # noqa: F401  (private names tests reach through here)
    _choice_probabilities,
    _compute_elasticities,
    _compute_market_shares,
    _compute_mu,
    _compute_standard_errors,
    _contraction_mapping,
    _delta_jacobian,
    _gmm_objective,
    _group_sum,
    _halton_sequence,
    _iv_regression,
    _market_rows,
    _nested_logit_elasticities,
    _objective_and_gradient,
    _share_jacobians,
)
from ._blp_result import BLPResult

__all__ = ["blp", "blp_instruments", "BLPResult"]

# ---------------------------------------------------------------------------
# BLP instrument construction
# ---------------------------------------------------------------------------


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
    x = [x] if isinstance(x, str) else list(x)
    groups = [market_id] + [c for c in (firm_id, nest) if c is not None]
    check_columns(
        data,
        {"market_id": market_id, "x": x, "firm_id": firm_id, "nest": nest},
        function="sp.blp_instruments",
    )
    require_numeric_columns(data, x, function="blp_instruments", argument="x")
    unlabelled = {c: int(data[c].isna().sum()) for c in groups}
    unlabelled = {c: m for c, m in unlabelled.items() if m}
    if unlabelled:
        # a missing market or firm belongs to no group: its sums are not
        # defined, and the rows of its market would be short of a product
        raise DataInsufficient(
            "blp_instruments: missing values in the grouping column(s) "
            f"{unlabelled} (column: number of rows).",
            recovery_hint="Drop those rows, or fill in the identifiers.",
            diagnostics={"n_missing": unlabelled},
        )
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


# step-1 starting points, in units of 1 / sd(x_k)
_START_SCALES = (0.05, 0.25, 1.0, 2.0)

# a random coefficient whose standard deviation moves utility by less than
# this (sigma_k * sd(x_k)) is at its bound of zero and is set to zero
_SIGMA_AT_BOUND = 1e-6


def _minimise_sigma(
    fun_grad: Any,
    fun: Any,
    sigma0: np.ndarray,
    optimizer: str,
    maxiter: int,
    tol_outer: float,
    scale: float,
) -> tuple[np.ndarray, bool, dict]:
    """Outer loop over ``sigma``. ``scale`` divides the objective (``N``)."""
    if optimizer == "nelder-mead":
        res = optimize.minimize(
            fun,
            sigma0,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": tol_outer, "fatol": tol_outer},
        )
        return np.abs(res.x), bool(res.success), {"n_objective_evaluations": res.nfev}

    def scaled(x: np.ndarray) -> tuple[float, np.ndarray]:
        f, g = fun_grad(x)
        return f / scale, g / scale

    res = optimize.minimize(
        scaled,
        sigma0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * len(sigma0),
        options={"maxiter": maxiter, "gtol": tol_outer, "ftol": 0.0},
    )
    x = np.asarray(res.x, dtype=float)
    # projected gradient: at a zero bound only a negative slope counts
    g = np.asarray(res.jac, dtype=float)
    pg = np.where((x <= 0.0) & (g > 0.0), 0.0, g)
    pg_norm = float(np.max(np.abs(pg))) if len(pg) else 0.0
    # the line search can stall on the noise floor of the inner loop while
    # already at a stationary point; that is convergence, not failure
    converged = bool(res.success) or pg_norm <= max(tol_outer, 1e-8) * 10.0
    info = {"n_objective_evaluations": res.nfev, "projected_gradient_norm": pg_norm}
    return x, converged, info


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
    sigma0: Any = None,
    optimizer: str = "l-bfgs-b",
    n_starts: int = 4,
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
        Variables with a normally distributed random coefficient: any of
        ``x_linear``, the price column, and ``"const"`` for the intercept
        (a random taste for the inside goods as a whole, which governs
        substitution to the outside good). If None, defaults to all of
        x_linear. An empty list gives the logit (or, with ``nest``, the
        nested logit) of Berry (1994).
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
        Outer-loop tolerance: with ``optimizer='l-bfgs-b'``, on the
        projected gradient of the objective divided by ``N``; with
        ``'nelder-mead'``, on the simplex and the objective.
    alpha : float, default 0.05
        Significance level (used in summary output).
    seed : int, optional
        Random seed for reproducibility of Monte Carlo draws.
    firm_id : str, optional
        Column identifying the firm selling each product. Used to build
        the default instruments, which then separate the firm's own other
        products from its rivals' (BLP 1995, section 5.1), and by
        ``result.markups()``.
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
    sigma0 : sequence or dict, optional
        Starting values for the random-coefficient standard deviations, in
        the order of ``x_random`` or keyed by variable name. Default 0.5
        each.
    optimizer : {'l-bfgs-b', 'nelder-mead'}, default 'l-bfgs-b'
        Outer-loop optimizer. ``'l-bfgs-b'`` uses the analytic gradient with
        ``sigma >= 0``. ``'nelder-mead'`` is the derivative-free search used
        before it, kept to reproduce earlier results; it needs a much larger
        ``maxiter`` and many more objective evaluations.
    n_starts : int, default 4
        Number of starting points for the first GMM step. The objective can
        have several local minima, so the step is run from ``sigma0`` and
        from ``c / sd(x_k)`` for ``c = 0.05, 0.25, 1, 2`` (as many as
        ``n_starts - 1``; beyond those, uniform draws on ``[0, 2 / sd(x_k)]``
        fixed by ``seed``), and the lowest objective is kept. The second
        step starts from it. Every start and the objective it reached are
        in ``model_info['starts']``. Ignored with
        ``optimizer='nelder-mead'``.

    Returns
    -------
    BLPResult
        Estimation results including parameters, standard errors,
        mean utilities, and elasticities; ``diversion_ratios()``,
        ``markups()`` and ``marginal_costs()`` are computed from them.

    Notes
    -----
    Random coefficients are normal. With one on price, some simulated
    consumers have a positive price coefficient; when products end up with
    a positive own-price elasticity the fit warns and records the count in
    ``model_info``. Consumer demographics (income in particular) and a
    jointly estimated supply side are not implemented.

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
    ...         rows.append({"market_id": mkt, "product_id": j, "firm": j // 2,
    ...                      "share": shares[j], "price": price[j],
    ...                      "x1": x[j], "cost": cost[j], "z2": z2[j]})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.blp(
    ...     df, shares='share', prices='price', x_linear=['x1'],
    ...     x_random=['x1'], instruments=['cost', 'z2'], n_draws=30, seed=0,
    ...     firm_id='firm',
    ... )
    >>> _ = result.summary()
    >>> elas = result.elasticity_matrix(market_id=1)
    >>> div = result.diversion_ratios(market_id=1, include_outside=True)
    >>> bool(np.allclose(div.sum(axis=1), 1.0))
    True
    >>> markup = result.markups(market_id=1)

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
    if instruments is not None:
        instruments = list(instruments)
    check_columns(
        data,
        {
            "shares": shares,
            "prices": prices,
            "market_id": market_id,
            "product_id": product_id,
            "x_linear": x_linear,
            "instruments": instruments,
            "firm_id": firm_id,
            "nest": nest,
        },
        function="sp.blp",
    )

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
    optimizer = str(optimizer).lower()
    if optimizer not in ("l-bfgs-b", "nelder-mead"):
        raise MethodIncompatibility(
            f"optimizer must be 'l-bfgs-b' or 'nelder-mead'; got {optimizer!r}."
        )
    if int(n_starts) < 1:
        raise MethodIncompatibility(f"n_starts must be at least 1; got {n_starts}.")
    if len(set(x_random)) != len(x_random):
        raise MethodIncompatibility(f"x_random lists a variable twice: {x_random}.")

    for col in x_random:
        if col not in x_linear and col != prices and col != "const":
            raise ValueError(
                f"x_random variable '{col}' must be in x_linear, be the price "
                "column, or be 'const' (the intercept)."
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

    # Random coefficient characteristics ("const" names the intercept unless
    # the data carry a regressor of that name)
    rc_cols = []
    rc_names = []
    for col in x_random:
        if col == prices:
            rc_cols.append(price_vals)
        elif col == "const" and col not in x_linear:
            rc_cols.append(np.ones(N))
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

    if sigma0 is None:
        sigma_init = np.ones(K_sigma) * 0.5
    else:
        if isinstance(sigma0, dict):
            unknown = [k for k in sigma0 if k not in x_random]
            if unknown or len(sigma0) != K_sigma:
                raise MethodIncompatibility(
                    f"sigma0 must give one value per x_random variable {x_random}; "
                    f"got keys {list(sigma0)}."
                )
            sigma_init = np.array([sigma0[c] for c in x_random], dtype=float)
        else:
            sigma_init = np.atleast_1d(np.asarray(sigma0, dtype=float))
        if sigma_init.shape != (K_sigma,):
            raise MethodIncompatibility(
                f"sigma0 has {sigma_init.size} value(s); x_random has {K_sigma}."
            )
        if not np.all(np.isfinite(sigma_init)) or np.any(sigma_init < 0):
            raise MethodIncompatibility(
                "sigma0 must be finite and non-negative (standard deviations)."
            )

    # ---- Initial weighting matrix (identity-like) -------------------------
    W = np.linalg.inv(Z.T @ Z / N)

    # ---- Outer loop: minimize GMM objective over sigma --------------------
    def _step(W_step: np.ndarray, start: np.ndarray, delta0: np.ndarray) -> tuple:
        """One GMM step: returns (sigma, converged, info, last delta)."""
        state = {"delta": delta0.copy()}

        def fun_grad(sigma_vec: np.ndarray) -> tuple[float, np.ndarray]:
            obj, grad, delta = _objective_and_gradient(
                sigma_vec,
                s_obs,
                X_linear,
                X_random_mat,
                Z,
                W_step,
                draws,
                market_ids,
                state["delta"],
                tol_inner,
                maxiter_inner=1000,
            )
            state["delta"] = delta
            return obj, grad

        def fun(sigma_vec: np.ndarray) -> float:
            obj, _, delta = _gmm_objective(
                sigma_vec,
                s_obs,
                X_linear,
                X_random_mat,
                Z,
                W_step,
                draws,
                market_ids,
                state["delta"],
                tol_inner,
                maxiter_inner=1000,
            )
            state["delta"] = delta
            return float(obj)

        sig, ok, info = _minimise_sigma(
            fun_grad, fun, start, optimizer, maxiter, tol_outer, float(N)
        )
        return sig, ok, info, state["delta"]

    opt_info: dict = {"n_objective_evaluations": 0}
    if K_sigma > 0:
        starts = [sigma_init]
        if optimizer == "l-bfgs-b" and int(n_starts) > 1:
            spread = X_random_mat.std(axis=0)
            spread = np.where(spread > 0, spread, 1.0)
            extra = [c / spread for c in _START_SCALES]
            n_random = int(n_starts) - 1 - len(extra)
            if n_random > 0:
                rng = np.random.default_rng([0 if seed is None else int(seed), 1])
                extra += list(rng.uniform(0.0, 2.0, (n_random, K_sigma)) / spread)
            starts += extra[: int(n_starts) - 1]
        tried = []
        best: tuple = ()
        for start in starts:
            sig, ok, info1, d_end = _step(W, np.asarray(start, dtype=float), delta_init)
            opt_info["n_objective_evaluations"] += info1["n_objective_evaluations"]
            d_end, _ = _contraction_mapping(
                s_obs,
                d_end,
                _compute_mu(X_random_mat, sig, draws),
                market_ids,
                tol_inner,
                maxiter=2000,
            )
            _, xi_s = _iv_regression(d_end, X_linear, Z, W)
            obj_s = float((Z.T @ xi_s) @ W @ (Z.T @ xi_s))
            tried.append(
                {
                    "sigma0": np.asarray(start, dtype=float).tolist(),
                    "sigma": sig.tolist(),
                    "objective": obj_s,
                    "converged": ok,
                }
            )
            if not best or obj_s < best[0]:
                best = (obj_s, sig, ok, d_end)
        _, sigma_hat, outer_converged, best_delta = best
        opt_info["starts"] = tried
        objs = np.array([t["objective"] for t in tried])
        opt_info["n_starts_at_best"] = int(
            np.sum(objs <= objs.min() + 1e-6 * max(abs(objs.min()), 1.0))
        )
    else:
        sigma_hat = np.array([])
        outer_converged = True
        best_delta = delta_init.copy()

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
    if K_sigma > 0:
        Zxi_1 = Z.T @ xi_hat
        opt_info["step1_sigma"] = np.asarray(sigma_hat, dtype=float).tolist()
        opt_info["step1_objective"] = float(Zxi_1 @ W @ Zxi_1)

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
    if K_sigma > 0 and gmm_steps == 1:
        sigma_final = sigma_hat
    elif K_sigma > 0:
        sigma_final, outer_converged, info2, best_delta_2 = _step(
            W_opt, sigma_hat, delta_hat
        )
        opt_info["n_objective_evaluations"] += info2["n_objective_evaluations"]
        opt_info.update(
            {k: v for k, v in info2.items() if k == "projected_gradient_norm"}
        )
    else:
        sigma_final = np.array([])
    if K_sigma > 0:
        # an optimiser stops near a zero bound, not on it
        spread = X_random_mat.std(axis=0)
        at_bound = sigma_final * np.where(spread > 0, spread, 1.0) < _SIGMA_AT_BOUND
        sigma_final = np.where(at_bound, 0.0, sigma_final)
        if not outer_converged:
            warnings.warn(
                f"blp: the outer loop ({optimizer}) stopped without converging "
                f"after {opt_info['n_objective_evaluations']} objective "
                "evaluations; raise maxiter or try other sigma0.",
                ConvergenceWarning,
                stacklevel=2,
            )

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
        share_jacobians = None  # rebuilt from the elasticities on demand
    else:
        share_jacobians = _share_jacobians(
            delta_final,
            mu_final,
            alpha_hat,
            sigma_price,
            draws,
            market_ids,
            price_draw_col=price_draw_col,
        )
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
    for m, idx in zip(np.unique(market_ids), _market_rows(market_ids)):
        own_elast[idx] = np.diag(elasticity_matrices[m])

    # ---- How the outer loop went ------------------------------------------
    model_info: dict = {}
    if K_sigma > 0:
        model_info = {
            "optimizer": optimizer,
            "sigma0": sigma_init.tolist(),
            **opt_info,
            "sigma_at_bound": [n for n, v in zip(rc_names, sigma_final) if v == 0.0],
        }
    if price_draw_col is not None:
        share_pos = float(
            np.mean(alpha_hat + sigma_price * draws[:, price_draw_col] > 0)
        )
        n_pos = int(np.sum(own_elast > 0))
        model_info["share_positive_price_coef"] = share_pos
        model_info["n_positive_own_elasticity"] = n_pos
        if n_pos > 0:
            warnings.warn(
                f"blp: {n_pos} of {N} products have a positive own-price "
                f"elasticity. The normal random coefficient on price gives "
                f"{share_pos:.1%} of simulated consumers a positive price "
                "coefficient, and those consumers dominate the demand for "
                "these products. Elasticities, diversion ratios and markups "
                "of the affected products are not meaningful; use more draws, "
                "another seed, or drop the random coefficient on price.",
                UserWarning,
                stacklevel=2,
            )

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
        _share_jacobians=share_jacobians,
        _shares=s_obs,
        _prices=price_vals,
        _firm_ids=data[firm_id].to_numpy() if firm_id is not None else None,
        model_info=model_info,
    )
