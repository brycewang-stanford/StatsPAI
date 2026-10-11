"""Result object of ``sp.blp``."""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility
from ._blp_post import diversion_matrix, firm_markups

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
    model_info : dict
        How the outer loop went (random coefficients only): ``optimizer``,
        ``sigma0``, ``starts`` (each first-step start with the objective it
        reached), ``n_starts_at_best``, ``step1_sigma``,
        ``step1_objective``, ``n_objective_evaluations``,
        ``projected_gradient_norm`` (L-BFGS-B) and ``sigma_at_bound``
        (standard deviations estimated at zero, whose standard error is
        NaN). With a random coefficient on price also
        ``share_positive_price_coef`` and ``n_positive_own_elasticity``.

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
        _share_jacobians: Optional[dict] = None,
        _shares: Optional[np.ndarray] = None,
        _prices: Optional[np.ndarray] = None,
        _firm_ids: Optional[np.ndarray] = None,
        model_info: Optional[dict] = None,
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
        self._share_jacobians = _share_jacobians or {}
        self._shares = _shares
        self._prices = _prices
        self._firm_ids = _firm_ids
        self.model_info = dict(model_info or {})

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
        info = self.model_info
        if len(info.get("starts", [])) > 1:
            lines.append(
                f"  Step-1 starts: {info['n_starts_at_best']} of "
                f"{len(info['starts'])} reached the lowest objective"
            )
        if info.get("sigma_at_bound"):
            lines.append(
                "  At the zero bound (held fixed for the standard errors): "
                + ", ".join(info["sigma_at_bound"])
            )
        if info.get("n_positive_own_elasticity", 0) > 0:
            lines.append(
                f"  WARNING: {info['n_positive_own_elasticity']} products have a "
                "positive own-price elasticity"
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

    def _market_or_first(self, market_id: Optional[Any]) -> Any:
        if market_id is None:
            market_id = list(self._elasticity_matrices.keys())[0]
        if market_id not in self._elasticity_matrices:
            raise KeyError(f"Market '{market_id}' not found.")
        return market_id

    def _jacobian(self, market_id: Any) -> np.ndarray:
        """``dS[j, k] = ds_j / dp_k`` for one market."""
        if market_id in self._share_jacobians:
            return np.asarray(self._share_jacobians[market_id], dtype=float)
        # results built without the Jacobian: undo E_jk = dS_jk p_k / s_j
        if self._shares is None or self._prices is None:
            raise MethodIncompatibility(
                "This BLPResult carries neither share Jacobians nor the shares "
                "and prices needed to rebuild them from the elasticities."
            )
        idx = np.where(self._market_ids == market_id)[0]
        E = self._elasticity_matrices[market_id]
        return np.asarray(
            E * self._shares[idx][:, np.newaxis] / self._prices[idx][np.newaxis, :]
        )

    def diversion_ratios(
        self, market_id: Optional[Any] = None, include_outside: bool = False
    ) -> pd.DataFrame:
        """
        Diversion ratios for a given market.

        ``D[j, k] = (ds_k/dp_j) / (-ds_j/dp_j)``: of the buyers product ``j``
        (row) loses when its price rises, the fraction that switches to
        product ``k`` (column). Each row sums to at most one; the rest goes
        to the outside good.

        Parameters
        ----------
        market_id : hashable, optional
            Market to compute for. If None, uses the first market.
        include_outside : bool, default False
            Append a column ``"outside"`` with the fraction that leaves the
            market, ``1 - sum_k D[j, k]``.

        Returns
        -------
        pd.DataFrame
            ``(J_m, J_m)`` with a zero diagonal, plus the ``"outside"``
            column when requested. Rows of products whose own-price
            derivative is not negative are NaN, with a warning.

        Notes
        -----
        Until October 2026 this method divided cross- by own-price
        *elasticities*, which omits the share ratio ``s_k / s_j``; rows
        could sum to well above one.
        """
        market_id = self._market_or_first(market_id)
        D, outside, undefined = diversion_matrix(self._jacobian(market_id))
        labels = self._product_ids[np.where(self._market_ids == market_id)[0]]
        if undefined.any():
            warnings.warn(
                f"{int(undefined.sum())} product(s) in market {market_id!r} have a "
                "non-negative own-price derivative; their diversion ratios are "
                "undefined and returned as NaN.",
                stacklevel=2,
            )
        out = pd.DataFrame(D, index=labels, columns=labels)
        if include_outside:
            out["outside"] = outside
        return out

    # ---- Markups ----------------------------------------------------------

    def _firms(self, firm_ids: Any) -> np.ndarray:
        if isinstance(firm_ids, str):
            if firm_ids != "single":
                raise MethodIncompatibility(
                    "firm_ids must be an array with one entry per row of the "
                    f"estimation data, or 'single'; got {firm_ids!r}."
                )
            return np.arange(self.n_products)
        if firm_ids is None:
            if self._firm_ids is None:
                raise MethodIncompatibility(
                    "Markups need to know which products each firm prices "
                    "jointly. Fit with sp.blp(..., firm_id=<column>), or pass "
                    "firm_ids=<array> here ('single' for single-product firms)."
                )
            return self._firm_ids
        firms = np.asarray(firm_ids)
        if firms.shape != (self.n_products,):
            raise MethodIncompatibility(
                f"firm_ids has shape {firms.shape}; expected one entry per row "
                f"of the estimation data ({self.n_products},)."
            )
        return firms

    def markups(
        self, market_id: Optional[Any] = None, firm_ids: Any = None
    ) -> pd.Series:
        """
        Price minus marginal cost under multi-product Bertrand pricing.

        Each firm sets the prices of its own products given the others'
        prices; the first-order conditions give
        ``p - mc = -(O * dS')^{-1} s``, with ``dS`` the share Jacobian and
        ``O[j, k] = 1`` when ``j`` and ``k`` belong to the same firm
        [@berry1995automobile]. Demand is the estimated one; no supply side
        is estimated.

        Parameters
        ----------
        market_id : hashable, optional
            One market (labelled by product) or, if None, every
            product-market row (indexed like the estimation data).
        firm_ids : array-like or 'single', optional
            Firm of each row of the estimation data. Defaults to the
            ``firm_id`` column given to ``sp.blp``; ``'single'`` treats
            every product as its own firm.

        Returns
        -------
        pd.Series
            In price units.
        """
        if self._shares is None:
            raise MethodIncompatibility("This BLPResult does not carry shares.")
        firms = self._firms(firm_ids)
        if market_id is not None:
            market_id = self._market_or_first(market_id)
        markets = list(self._elasticity_matrices) if market_id is None else [market_id]
        out = np.full(self.n_products, np.nan)
        for m in markets:
            idx = np.where(self._market_ids == m)[0]
            out[idx] = firm_markups(self._jacobian(m), self._shares[idx], firms[idx])
        if market_id is None:
            return pd.Series(out, index=self._data_index, name="markup")
        idx = np.where(self._market_ids == market_id)[0]
        return pd.Series(out[idx], index=self._product_ids[idx], name="markup")

    def marginal_costs(
        self, market_id: Optional[Any] = None, firm_ids: Any = None
    ) -> pd.Series:
        """Marginal costs implied by :meth:`markups`: ``mc = p - markup``."""
        mk = self.markups(market_id, firm_ids)
        if self._prices is None:
            raise MethodIncompatibility("This BLPResult does not carry prices.")
        if market_id is None:
            price = self._prices
        else:
            price = self._prices[np.where(self._market_ids == market_id)[0]]
        return pd.Series(price - mk.to_numpy(), index=mk.index, name="marginal_cost")

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
            **self.model_info,
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
