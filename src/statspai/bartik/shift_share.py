"""
Shift-Share (Bartik) Instrumental Variables.

The Bartik instrument for region i is:
    B_i = sum_k (s_{ik} * g_k)

where s_{ik} is region i's initial share in industry k, and g_k is
the national growth rate of industry k (excluding region i for
leave-one-out).

Supports:
- Bartik instrument construction
- 2SLS estimation with the Bartik IV
- Rotemberg weight decomposition (GPSS 2020)
- Diagnostics for share vs. shock exogeneity
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult, EconometricResults
from ..exceptions import MethodIncompatibility


@accepts_aliases(vce="robust")
def bartik(
    data: pd.DataFrame,
    y: str,
    endog: str,
    shares: pd.DataFrame,
    shocks: pd.Series,
    covariates: Optional[List[str]] = None,
    leave_one_out: bool = True,
    regional_shocks: Optional[pd.DataFrame] = None,
    robust: str = "hc1",
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Estimate using Shift-Share (Bartik) instrumental variables.

    Parameters
    ----------
    data : pd.DataFrame
        Cross-sectional data with one row per region/unit.
    y : str
        Outcome variable.
    endog : str
        Endogenous regressor (e.g., local employment growth).
    shares : pd.DataFrame
        Share matrix (n_units x n_industries). Rows = regions, cols = industries.
        Row index must align with data index.
    shocks : pd.Series
        National shock vector (n_industries,). Index = industry names.
    covariates : list of str, optional
        Exogenous control variables.
    leave_one_out : bool, default True
        Compute leave-one-out shocks (exclude own region from national
        average). Only takes effect when ``regional_shocks`` is also
        supplied — without the per-region industry growth panel there
        is not enough information to reconstruct ``g_k`` excluding
        region ``i``. When ``leave_one_out=True`` but
        ``regional_shocks`` is not provided, a ``UserWarning`` is
        raised and the estimator falls back to the simple Bartik
        instrument.
    regional_shocks : pd.DataFrame, optional
        Regional industry growth matrix (n_units x n_industries). Row
        ``i``, column ``k`` is the realised growth of industry ``k``
        in region ``i``. When provided with ``leave_one_out=True``,
        the instrument uses
        ``g_k^{-i} = (sum_j g_{jk} - g_{ik}) / (n - 1)`` (Borusyak-
        Hull-Jaravel 2022-style exact leave-one-out). Row index must
        align with ``shares``; columns must be a superset of
        ``shocks.index``.
    robust : {'hc1', 'nonrobust'}, default 'hc1'
        Standard error type: ``'hc1'`` is the 2SLS sandwich with the
        ``n / (n - k)`` factor (Stata ``ivregress 2sls, vce(robust) small``,
        R ``sandwich::vcovHC(type = "HC1")``); ``'nonrobust'`` uses
        ``sigma^2 = RSS / (n - k)`` (``ivregress 2sls, small``). Other values
        raise. For shift-share (AKM) inference pass the result to
        :func:`sp.shift_share_se`.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    EconometricResults
        2SLS results with Bartik IV diagnostics. The Rotemberg weight
        table (columns ``industry``, ``weight`` = alpha_k, ``shock``,
        ``beta`` = the just-identified IV using industry k's share alone,
        as Stata ``bartik_weight`` reports) is in
        ``result.model_info["rotemberg_weights"]``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n, K = 80, 4
    >>> # shares: DataFrame (regions x industries), shocks: Series (industries)
    >>> shares = pd.DataFrame(
    ...     rng.dirichlet(np.ones(K), size=n),
    ...     columns=[f"ind{k}" for k in range(K)],
    ... )
    >>> shocks = pd.Series(rng.normal(0.05, 0.02, K), index=shares.columns)
    >>> emp = shares.values @ shocks.values + rng.normal(0, 0.01, n)
    >>> wage = 1.0 + 2.0 * emp + rng.normal(0, 0.02, n)
    >>> df = pd.DataFrame({"wage_growth": wage, "emp_growth": emp})
    >>> result = sp.bartik(df, y='wage_growth', endog='emp_growth',
    ...                    shares=shares, shocks=shocks, leave_one_out=False)
    >>> _ = result.summary()
    """
    estimator = BartikIV(
        data=data,
        y=y,
        endog=endog,
        shares=shares,
        shocks=shocks,
        covariates=covariates,
        leave_one_out=leave_one_out,
        regional_shocks=regional_shocks,
        robust=robust,
        alpha=alpha,
    )
    _result = estimator.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.bartik",
            params={
                "y": y,
                "endog": endog,
                "covariates": list(covariates) if covariates else None,
                "leave_one_out": leave_one_out,
                "robust": robust,
                "alpha": alpha,
                "shares_shape": (
                    list(shares.shape) if hasattr(shares, "shape") else None
                ),
                "shocks_len": (
                    int(len(shocks)) if hasattr(shocks, "__len__") else None
                ),
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


class BartikIV:
    """
    Bartik Shift-Share IV estimator.

    Constructs the shift-share instrument ``B_i = sum_k s_ik * g_k`` from
    a region-by-industry share matrix and an industry shock vector, then
    runs 2SLS. The convenience wrapper :func:`bartik` builds this object
    and calls :meth:`fit`; use the class directly when you want to hold
    onto the estimator instance.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> from statspai.bartik.shift_share import BartikIV
    >>> rng = np.random.default_rng(0)
    >>> n, K = 60, 4
    >>> shares = pd.DataFrame(
    ...     rng.dirichlet(np.ones(K), size=n),
    ...     columns=[f"ind{k}" for k in range(K)],
    ... )
    >>> shocks = pd.Series(rng.normal(0.05, 0.02, K), index=shares.columns)
    >>> emp = shares.values @ shocks.values + rng.normal(0, 0.01, n)
    >>> wage = 1.5 * emp + rng.normal(0, 0.02, n)
    >>> df = pd.DataFrame({"wage_growth": wage, "emp_growth": emp})
    >>> est = BartikIV(df, y="wage_growth", endog="emp_growth",
    ...                shares=shares, shocks=shocks, leave_one_out=False)
    >>> res = est.fit()
    >>> round(float(res.params["emp_growth"]), 1)  # ~1.5
    1.5
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        endog: str,
        shares: pd.DataFrame,
        shocks: pd.Series,
        covariates: Optional[List[str]] = None,
        leave_one_out: bool = True,
        regional_shocks: Optional[pd.DataFrame] = None,
        robust: str = "hc1",
        alpha: float = 0.05,
    ):
        self.data = data
        self.y = y
        self.endog = endog
        self.shares = shares
        self.shocks = shocks
        self.covariates = covariates or []
        self.leave_one_out = leave_one_out
        self.regional_shocks = regional_shocks
        self.robust = robust
        self.alpha = alpha

        self._validate()

    def _validate(self) -> None:
        for col in [self.y, self.endog] + self.covariates:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")

        if self.shares.shape[0] != len(self.data):
            raise ValueError(
                f"shares has {self.shares.shape[0]} rows but data has "
                f"{len(self.data)} rows"
            )

        # Align industry names
        common = self.shares.columns.intersection(self.shocks.index)
        if len(common) == 0:
            raise ValueError("No common industries between shares and shocks")

        self.shares = self.shares[common]
        self.shocks = self.shocks[common]

        robust = str(self.robust).lower()
        if robust in ("hc1", "robust"):
            self.robust = "hc1"
        elif robust in ("nonrobust", "classical", "unadjusted"):
            self.robust = "nonrobust"
        else:
            raise MethodIncompatibility(
                f"robust={self.robust!r} is not supported by sp.bartik; use "
                "'hc1' (default, = Stata `ivregress ..., vce(robust) small`) "
                "or 'nonrobust' (sigma^2 = RSS/(n-k)). For shift-share "
                "(AKM) inference pass the result to sp.shift_share_se.",
                recovery_hint="Use robust='hc1' or 'nonrobust'.",
            )

        if self.regional_shocks is not None:
            if self.regional_shocks.shape[0] != len(self.data):
                raise ValueError(
                    f"regional_shocks has {self.regional_shocks.shape[0]} "
                    f"rows but data has {len(self.data)} rows"
                )
            missing = set(common) - set(self.regional_shocks.columns)
            if missing:
                raise ValueError(
                    "regional_shocks is missing industries present in "
                    f"shares/shocks: {sorted(missing)}"
                )
            self.regional_shocks = self.regional_shocks[common]

    def _construct_instrument(self) -> np.ndarray:
        """Construct the Bartik instrument.

        - ``B_i = sum_k s_ik * g_k`` (simple Bartik), or
        - ``B_i = sum_k s_ik * g_k^{-i}`` (leave-one-out) when
          ``leave_one_out=True`` and ``regional_shocks`` is provided,
          where ``g_k^{-i} = (sum_j g_{jk} - g_{ik}) / (n - 1)`` is
          the national industry growth rate computed excluding
          region ``i`` (Borusyak-Hull-Jaravel 2022).

        Silent-no-op guard: prior to v0.9.13 ``leave_one_out=True``
        quietly fell through to simple Bartik because the LOO step
        requires per-region industry growth data that the basic API
        does not carry. We now warn loudly instead so users notice
        the fallback.
        """
        S = self.shares.to_numpy(dtype=float)  # (n, K)
        g = self.shocks.to_numpy(dtype=float)  # (K,)

        if not self.leave_one_out:
            return np.asarray(S @ g, dtype=float)  # (n,)

        if self.regional_shocks is None:
            warnings.warn(
                "bartik(leave_one_out=True) requested but "
                "`regional_shocks` was not supplied. Proper "
                "leave-one-out requires per-region industry growth "
                "(n_units x n_industries) to reconstruct g_k^{-i}. "
                "Falling back to the simple Bartik instrument "
                "B_i = sum_k s_ik * g_k; pass `regional_shocks=` or "
                "set `leave_one_out=False` to silence this warning.",
                UserWarning,
                stacklevel=3,
            )
            return np.asarray(S @ g, dtype=float)

        G = self.regional_shocks.to_numpy(dtype=float)  # (n, K)
        n = G.shape[0]
        if n < 2:
            raise ValueError("leave-one-out requires at least 2 regions")
        # g_k^{-i} = (col_sum_k - G[i,k]) / (n - 1)
        col_sum = G.sum(axis=0, keepdims=True)  # (1, K)
        g_loo = (col_sum - G) / (n - 1)  # (n, K)
        return np.asarray(np.einsum("ij,ij->i", S, g_loo), dtype=float)

    def _rotemberg_weights(
        self,
        B: np.ndarray,
        Y: np.ndarray,
        X_endog: np.ndarray,
        X_exog: Optional[np.ndarray],
    ) -> pd.DataFrame:
        """
        Rotemberg weights of the Bartik IV (Goldsmith-Pinkham, Sorkin and
        Swift 2020).

        ``alpha_k = g_k s_k' M_W x / sum_j g_j s_j' M_W x`` and the
        just-identified per-industry IV ``beta_k = s_k' M_W y / s_k' M_W x``,
        with ``M_W`` annihilating the intercept and covariates -- the
        quantities Stata ``bartik_weight`` returns as ``r(alpha)`` /
        ``r(beta)``. The 2SLS estimate equals ``sum_k alpha_k beta_k``.
        """
        S = self.shares.values
        n, K = S.shape

        # Annihilator for the exogenous block, applied to vectors rather than
        # formed (the n x n matrix made Rotemberg weights O(n^2) in memory).
        def _annihilate(v: np.ndarray) -> np.ndarray:
            out: np.ndarray
            if X_exog is not None:
                out = v - X_exog @ np.linalg.lstsq(X_exog, v, rcond=None)[0]
            else:
                out = v - np.mean(v, axis=0)
            return out

        g = self.shocks.values.astype(float)
        Qx = _annihilate(X_endog)
        Qy = _annihilate(Y)
        zx = S.T @ Qx  # Z_k' M_W x
        zy = S.T @ Qy  # Z_k' M_W y
        # alpha_k = g_k Z_k'M_W x / sum_k g_k Z_k'M_W x (GPSS 2020; as Stata
        # bartik_weight / R bartik.weight::bw). The denominator is the Bartik
        # first-stage moment B'M_W x only when B = S g (no leave-one-out).
        denominator = float(g @ zx)
        if abs(denominator) < 1e-10:
            warnings.warn(
                "Rotemberg weights are undefined: sum_k g_k s_k' M_W x is "
                "numerically zero (no first stage); reporting NaN weights.",
                UserWarning,
                stacklevel=3,
            )
            denominator = np.nan
        weights = g * zx / denominator
        with np.errstate(divide="ignore", invalid="ignore"):
            beta_k = zy / zx  # just-identified IV using share k alone

        return (
            pd.DataFrame(
                {
                    "industry": self.shares.columns,
                    "weight": weights,
                    "shock": g,
                    "beta": beta_k,
                }
            )
            .sort_values("weight", ascending=False, key=abs)
            .reset_index(drop=True)
        )

    def fit(self) -> EconometricResults:
        """Fit Bartik IV via 2SLS."""
        n = len(self.data)

        Y = self.data[self.y].values.astype(float)
        X_endog = self.data[self.endog].values.astype(float)
        B = self._construct_instrument()

        # Exogenous regressors (constant + covariates)
        if self.covariates:
            X_exog = np.column_stack(
                [
                    np.ones(n),
                    self.data[self.covariates].values.astype(float),
                ]
            )
            exog_names = ["Intercept"] + self.covariates
        else:
            X_exog = np.ones((n, 1))
            exog_names = ["Intercept"]

        # --- First stage: endog ~ B + exog ---
        Z = np.column_stack([X_exog, B])
        gamma = np.linalg.lstsq(Z, X_endog, rcond=None)[0]
        X_endog_hat = Z @ gamma

        # First-stage F
        resid_full = X_endog - Z @ gamma
        gamma_r = np.linalg.lstsq(X_exog, X_endog, rcond=None)[0]
        resid_restricted = X_endog - X_exog @ gamma_r
        rss_f = resid_full @ resid_full
        rss_r = resid_restricted @ resid_restricted
        df_denom = n - Z.shape[1]
        f_stat = ((rss_r - rss_f) / 1) / (rss_f / df_denom) if df_denom > 0 else np.nan
        f_pvalue = stats.f.sf(f_stat, 1, df_denom) if not np.isnan(f_stat) else np.nan

        # --- Second stage: Y ~ endog_hat + exog ---
        X_2sls = np.column_stack([X_exog, X_endog_hat])
        X_actual = np.column_stack([X_exog, X_endog])
        all_names = exog_names + [self.endog]

        XhXh_inv = np.linalg.inv(X_2sls.T @ X_2sls)
        params = XhXh_inv @ X_2sls.T @ Y

        # Residuals from actual regressors
        fitted = X_actual @ params
        residuals = Y - fitted
        k = len(all_names)

        # Standard errors (HC1)
        if self.robust != "nonrobust":
            weights = (n / (n - k)) * residuals**2
            meat = (X_2sls * (weights)[:, None]).T @ X_2sls
            var_cov = XhXh_inv @ meat @ XhXh_inv
        else:
            sigma2 = np.sum(residuals**2) / (n - k)
            var_cov = sigma2 * XhXh_inv

        std_errors = np.sqrt(np.diag(var_cov))

        # Rotemberg weights
        rotemberg = self._rotemberg_weights(B, Y, X_endog, X_exog)

        # R-squared
        tss = np.sum((Y - np.mean(Y)) ** 2)
        rss = np.sum(residuals**2)
        r_squared = 1 - rss / tss

        # Build results
        params_s = pd.Series(params, index=all_names)
        se_s = pd.Series(std_errors, index=all_names)

        model_info = {
            "model_type": "Bartik IV (2SLS)",
            "method": "Shift-Share IV",
            "robust": self.robust,
        }

        data_info = {
            "nobs": n,
            "df_model": k - 1,
            "df_resid": n - k,
            "dependent_var": self.y,
            "fitted_values": fitted,
            "residuals": residuals,
            # Inputs sp.shift_share_se needs for AKM inference.
            "_shift_share_inputs": {
                "y": Y,
                "endog": X_endog,
                "shift_share": B,
                "controls": X_exog,
            },
        }
        model_info["rotemberg_weights"] = rotemberg

        diagnostics = {
            "R-squared": r_squared,
            "First-stage F": f_stat,
            "First-stage F p-value": f_pvalue,
            "N industries": self.shares.shape[1],
        }

        # Store Rotemberg weights for programmatic access
        self._rotemberg = rotemberg
        self._first_stage_f = f_stat

        return EconometricResults(
            params=params_s,
            std_errors=se_s,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )

    @property
    def rotemberg_weights(self) -> pd.DataFrame:
        """Rotemberg weight decomposition by industry."""
        if not hasattr(self, "_rotemberg"):
            raise ValueError("Must call fit() first")
        return self._rotemberg


# Citation
CausalResult._CITATIONS["bartik"] = (
    "@article{goldsmithpinkham2020bartik,\n"
    "  title={Bartik Instruments: What, When, Why, and How},\n"
    "  author={Goldsmith-Pinkham, Paul and Sorkin, Isaac and Swift, Henry},\n"
    "  journal={American Economic Review},\n"
    "  volume={110},\n"
    "  number={8},\n"
    "  pages={2586--2624},\n"
    "  year={2020},\n"
    "  doi={10.1257/aer.20181047}\n"
    "}"
)
