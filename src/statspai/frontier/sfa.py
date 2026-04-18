"""
Cross-sectional stochastic frontier estimation — :func:`frontier`.

Supports:

* **Distributions**: half-normal, exponential, truncated-normal.
* **Heteroskedastic u**: ``sigma_u_i = exp(w_i' gamma_u)`` via ``usigma=[...]``.
* **Heteroskedastic v**: ``sigma_v_i = exp(r_i' gamma_v)`` via ``vsigma=[...]``.
* **Inefficiency determinants**: ``mu_i = z_i' delta`` for truncated-normal via
  ``emean=[...]``  (Battese-Coelli 1995 cross-sectional analogue,
  Kumbhakar-Ghosh-McGuckin 1991).
* **Cost / production**: ``cost=True`` flips sign of u in composed error.
* **Technical efficiency**: Battese-Coelli (1988) ``E[exp(-u)|eps]`` or JLMS
  ``exp(-E[u|eps])`` via ``te_method``.
* **Specification tests**: LR test against OLS (absence of inefficiency) using
  mixed chi-bar-squared (Kodde-Palm 1986); LR test of half-normal vs
  truncated-normal; residual skewness diagnostic.

Equivalent to (and more general than) Stata's::

    frontier y x1 x2, distribution(hnormal | exponential | tnormal)
    frontier y x1 x2, cost
    frontier y x1 x2, usigma(w1 w2) vsigma(r1)
    frontier y x1 x2, distribution(tnormal) emean(z1 z2)

and R's ``frontier::sfa()`` / ``sfaR::sfacross()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..core.results import EconometricResults
from . import _core as _fc


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class FrontierResult(EconometricResults):
    """Result object returned by :func:`frontier` and :func:`xtfrontier`.

    Extends :class:`~statspai.core.results.EconometricResults` with
    efficiency-score access, LR tests, and bootstrap helpers.
    """

    # ---- Names of diagnostics whose values are per-observation arrays ----
    # The parent class renders every diagnostic literally; we override summary
    # to show an SFA-specific block instead of dumping these arrays.
    _ARRAY_DIAGS = frozenset(
        {
            "sigma_u_i", "sigma_v_i", "mu_i", "eps",
            "efficiency_bc", "efficiency_jlms", "inefficiency_jlms",
            "efficiency_index", "a_it", "group_idx", "unit_ids",
            "efficiency_bc_unit", "efficiency_jlms_unit",
            "efficiency_bc_unit_mean",
            "hessian", "vcov", "spec",
        }
    )

    def summary(self, alpha: float = 0.05) -> str:
        """Publication-ready summary table (Stata-style SFA block).

        Overrides :class:`EconometricResults.summary` to hide per-observation
        diagnostic arrays and surface the SFA-specific scalars.
        """
        # Hide array diagnostics from the parent's generic renderer.
        hidden = {k: self.diagnostics.pop(k) for k in list(self.diagnostics.keys())
                  if k in self._ARRAY_DIAGS}
        try:
            base = EconometricResults.summary(self, alpha=alpha)
        finally:
            self.diagnostics.update(hidden)

        # Build the SFA-specific footer.
        mi = self.model_info
        lines = ["", "Variance decomposition:"]
        lines.append("-" * 20)
        if "sigma_u_mean" in mi:  # cross-sectional
            su, sv = mi["sigma_u_mean"], mi["sigma_v_mean"]
        else:
            su, sv = mi.get("sigma_u", float("nan")), mi.get("sigma_v", float("nan"))
        sigma2 = su**2 + sv**2
        lam = su / sv if sv > 0 else float("nan")
        gamma = su**2 / sigma2 if sigma2 > 0 else float("nan")
        lines.append(f"  sigma_u          : {su:.6f}")
        lines.append(f"  sigma_v          : {sv:.6f}")
        lines.append(f"  sigma            : {np.sqrt(sigma2):.6f}")
        lines.append(f"  lambda = su/sv   : {lam:.4f}")
        lines.append(f"  gamma  = su^2/s^2: {gamma:.4f}")
        mean_bc = mi.get("mean_efficiency_bc")
        mean_jlms = mi.get("mean_efficiency_jlms")
        if mean_bc is not None:
            lines.append(f"  mean TE (BC)     : {mean_bc:.4f}")
        if mean_jlms is not None:
            lines.append(f"  mean TE (JLMS)   : {mean_jlms:.4f}")

        lr = self.diagnostics.get("lr_no_inefficiency")
        if lr is not None:
            from . import _core as _fc
            pval = _fc.mixed_chi_bar_pvalue(float(lr), df_boundary=1)
            lines.append("")
            lines.append(f"LR test vs OLS (H0: sigma_u=0):  chi-bar^2(1)={lr:.4f}  p={pval:.4f}")

        return base + "\n" + "\n".join(lines)

    def efficiency(
        self,
        method: Optional[str] = None,
    ) -> pd.Series:
        """Return unit-level technical efficiency scores.

        Parameters
        ----------
        method : {'bc', 'jlms'}, optional
            'bc' (default) : Battese-Coelli (1988) ``E[exp(-u)|eps]``.
            'jlms'         : Jondrow-Lovell-Materov-Schmidt ``exp(-E[u|eps])``.
            If None, uses the default stored at fit time.
        """
        key = self._efficiency_key(method)
        vals = self.diagnostics.get(key)
        if vals is None:
            raise KeyError(f"Efficiency scores '{key}' not available.")
        idx = self.diagnostics.get("efficiency_index")
        return pd.Series(vals, name=key, index=idx if idx is not None else None)

    def inefficiency(self, method: str = "jlms") -> pd.Series:
        """Return ``E[u|eps]`` (inefficiency), Jondrow et al. (1982)."""
        vals = self.diagnostics.get("inefficiency_jlms")
        if vals is None:
            raise KeyError("Inefficiency scores not available.")
        idx = self.diagnostics.get("efficiency_index")
        return pd.Series(vals, name="u_hat", index=idx if idx is not None else None)

    def _efficiency_key(self, method: Optional[str]) -> str:
        if method is None:
            method = self.model_info.get("te_method", "bc")
        method = method.lower()
        if method in {"bc", "battese-coelli", "battesecoelli"}:
            return "efficiency_bc"
        if method in {"jlms", "jondrow"}:
            return "efficiency_jlms"
        raise ValueError(f"Unknown TE method: {method!r}")

    def lr_test_no_inefficiency(self) -> Dict[str, float]:
        """One-sided LR test ``H0: sigma_u = 0`` (mixed chi-bar squared)."""
        stat = self.diagnostics.get("lr_no_inefficiency")
        if stat is None:
            return {"statistic": np.nan, "pvalue": np.nan, "df": np.nan}
        pval = _fc.mixed_chi_bar_pvalue(stat, df_boundary=1)
        return {"statistic": float(stat), "pvalue": float(pval), "df": 1}

    def efficiency_ci(
        self,
        alpha: float = 0.05,
        B: int = 500,
        method: Optional[str] = None,
        seed: Optional[int] = 0,
    ) -> pd.DataFrame:
        """Parametric-bootstrap CI for unit-level efficiency scores.

        Draws ``(u_b, v_b) ~`` posterior predictive using the fitted
        variance parameters, then recomputes the Jondrow posterior for
        the resampled composed error.  Returns a DataFrame indexed like
        :meth:`efficiency` with columns ``['point', 'lower', 'upper']``.
        """
        point = self.efficiency(method=method).to_numpy()
        sigma_u_i = np.asarray(self.diagnostics.get("sigma_u_i"))
        sigma_v_i = np.asarray(self.diagnostics.get("sigma_v_i"))
        mu_i = self.diagnostics.get("mu_i")
        dist = self.model_info.get("inefficiency_dist", "half-normal")
        sign = self.model_info.get("sign", -1)
        eps = np.asarray(self.diagnostics.get("eps"))
        if eps.size == 0:
            raise RuntimeError("eps not stored; cannot bootstrap.")
        rng = np.random.default_rng(seed)
        n = eps.size
        sims = np.empty((B, n))
        for b in range(B):
            # Redraw posterior predictive u, v and reconstruct eps_b.
            if dist == "half-normal":
                u_sim = np.abs(rng.normal(0.0, sigma_u_i))
            elif dist == "exponential":
                u_sim = rng.exponential(sigma_u_i)
            else:  # truncated-normal
                u_sim = _draw_truncated_normal(mu_i, sigma_u_i, rng)
            v_sim = rng.normal(0.0, sigma_v_i)
            eps_b = v_sim + sign * u_sim
            # Posterior E[u|eps_b] under fitted distribution.
            if dist == "half-normal":
                _, te_bc = _fc.jondrow_halfnormal(eps_b, sigma_v_i, sigma_u_i, sign)
            elif dist == "exponential":
                _, te_bc = _fc.jondrow_exponential(eps_b, sigma_v_i, sigma_u_i, sign)
            else:
                _, te_bc = _fc.jondrow_truncnormal(
                    eps_b, sigma_v_i, sigma_u_i, np.asarray(mu_i), sign
                )
            sims[b] = te_bc
        lower = np.quantile(sims, alpha / 2.0, axis=0)
        upper = np.quantile(sims, 1.0 - alpha / 2.0, axis=0)
        idx = self.diagnostics.get("efficiency_index")
        return pd.DataFrame(
            {"point": point, "lower": lower, "upper": upper},
            index=idx if idx is not None else None,
        )


def _draw_truncated_normal(mu, sigma, rng) -> np.ndarray:
    """Draw u ~ N^+(mu, sigma^2) truncated at 0 (inverse-CDF)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    # Handle broadcasting.
    shape = np.broadcast(mu, sigma).shape
    u = rng.uniform(size=shape)
    lo = stats.norm.cdf(-mu / sigma)
    p = lo + u * (1.0 - lo)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return mu + sigma * stats.norm.ppf(p)


# ---------------------------------------------------------------------------
# Packing / unpacking of parameter vectors
# ---------------------------------------------------------------------------


@dataclass
class _FrontierSpec:
    """Compact description of the parameter vector layout."""
    k_beta: int
    k_gamma_u: int
    k_gamma_v: int
    k_delta_mu: int                 # 0 if no mu (half-normal / exponential)
    has_emean: bool                 # True if mu varies with covariates
    has_usigma: bool
    has_vsigma: bool
    dist: str

    @property
    def k_total(self) -> int:
        return self.k_beta + self.k_gamma_u + self.k_gamma_v + self.k_delta_mu

    def slices(self) -> Tuple[slice, slice, slice, slice]:
        a = slice(0, self.k_beta)
        b = slice(self.k_beta, self.k_beta + self.k_gamma_u)
        c = slice(b.stop, b.stop + self.k_gamma_v)
        d = slice(c.stop, c.stop + self.k_delta_mu)
        return a, b, c, d


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def frontier(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    *,
    dist: str = "half-normal",
    cost: bool = False,
    usigma: Optional[List[str]] = None,
    vsigma: Optional[List[str]] = None,
    emean: Optional[List[str]] = None,
    te_method: str = "bc",
    maxiter: int = 500,
    tol: float = 1e-8,
    alpha: float = 0.05,
    start: Optional[np.ndarray] = None,
) -> FrontierResult:
    """Estimate a cross-sectional stochastic frontier model by ML.

    Parameters
    ----------
    data : pandas.DataFrame
        Cross-sectional data.  Rows with missing values in any referenced
        column are dropped.
    y : str
        Dependent variable (output for production, cost for cost frontier).
    x : list of str
        Frontier regressors (a constant is added automatically).
    dist : {'half-normal', 'exponential', 'truncated-normal'}
        Distribution of the inefficiency term ``u``.
    cost : bool, default False
        If True, estimate a cost frontier (composed error ``v + u``).
    usigma : list of str, optional
        Columns parameterizing ``ln sigma_u_i = gamma_u' [1, w_i]``
        (Caudill-Ford-Gropper 1995).
    vsigma : list of str, optional
        Columns parameterizing ``ln sigma_v_i = gamma_v' [1, r_i]`` (Wang 2002).
    emean : list of str, optional
        Columns parameterizing ``mu_i = delta' [1, z_i]`` for the truncated
        normal (Battese-Coelli 1995; Kumbhakar-Ghosh-McGuckin 1991).  Requires
        ``dist='truncated-normal'``.
    te_method : {'bc', 'jlms'}, default 'bc'
        Default technical-efficiency formula accessed via ``.efficiency()``.
    maxiter : int, default 500
    tol : float, default 1e-8
    alpha : float, default 0.05
    start : ndarray, optional
        User-supplied starting values for the full parameter vector.

    Returns
    -------
    :class:`FrontierResult`

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.frontier(df, y='log_y', x=['log_k', 'log_l'])
    >>> res.efficiency().describe()
    >>> res.lr_test_no_inefficiency()
    >>> sp.frontier(df, y='log_y', x=['log_k', 'log_l'],
    ...             dist='truncated-normal', emean=['firm_age'])
    """
    dist = dist.lower().replace("_", "-")
    if dist not in {"half-normal", "exponential", "truncated-normal"}:
        raise ValueError(f"Unknown distribution: {dist!r}.")
    if emean is not None and dist != "truncated-normal":
        raise ValueError("emean=... requires dist='truncated-normal'.")

    required = [y] + list(x)
    for opt in (usigma, vsigma, emean):
        if opt:
            required += list(opt)
    df = data[required].dropna().copy()
    n = len(df)
    if n < len(x) + 3:
        raise ValueError("Too few observations for frontier estimation.")

    sign = 1 if cost else -1

    y_vec, X_mat, beta_names = _fc.build_design(df, y, x, add_constant=True)
    W_mat, w_names = _fc.build_optional_design(df, usigma, True, prefix="u_")
    R_mat, r_names = _fc.build_optional_design(df, vsigma, True, prefix="v_")
    Z_mat, z_names = _fc.build_optional_design(df, emean, True, prefix="mu_")

    # Parameter layout
    k_beta = X_mat.shape[1]
    k_gamma_u = W_mat.shape[1] if W_mat is not None else 1
    k_gamma_v = R_mat.shape[1] if R_mat is not None else 1
    if dist == "truncated-normal":
        k_delta_mu = Z_mat.shape[1] if Z_mat is not None else 1
    else:
        k_delta_mu = 0

    spec = _FrontierSpec(
        k_beta=k_beta,
        k_gamma_u=k_gamma_u,
        k_gamma_v=k_gamma_v,
        k_delta_mu=k_delta_mu,
        has_emean=emean is not None,
        has_usigma=usigma is not None,
        has_vsigma=vsigma is not None,
        dist=dist,
    )
    sl_beta, sl_gu, sl_gv, sl_dm = spec.slices()

    # ---------------------- Log-likelihood ----------------------

    def _unpack(theta: np.ndarray):
        beta = theta[sl_beta]
        gamma_u = theta[sl_gu]
        gamma_v = theta[sl_gv]
        delta = theta[sl_dm] if k_delta_mu > 0 else None

        sigma_u = _fc.evaluate_sigma(gamma_u, W_mat, gamma_u[0] if W_mat is None else 0.0, n)
        sigma_v = _fc.evaluate_sigma(gamma_v, R_mat, gamma_v[0] if R_mat is None else 0.0, n)
        if delta is not None:
            if Z_mat is None:
                mu = np.full(n, delta[0])
            else:
                mu = Z_mat @ delta
        else:
            mu = None
        return beta, sigma_u, sigma_v, mu

    def neg_loglik(theta):
        if not np.all(np.isfinite(theta)):
            return 1e20
        beta, sigma_u, sigma_v, mu = _unpack(theta)
        # Guard against pathological sigma (optimizer excursions).
        if (
            np.any(sigma_u <= 1e-8)
            or np.any(sigma_v <= 1e-8)
            or np.any(sigma_u > 1e6)
            or np.any(sigma_v > 1e6)
        ):
            return 1e20
        eps = y_vec - X_mat @ beta
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if dist == "half-normal":
                ll = _fc.loglik_halfnormal(eps, sigma_v, sigma_u, sign)
            elif dist == "exponential":
                ll = _fc.loglik_exponential(eps, sigma_v, sigma_u, sign)
            else:
                ll = _fc.loglik_truncated_normal(eps, sigma_v, sigma_u, mu, sign)
        if not np.isfinite(ll).all():
            return 1e20
        return -float(ll.sum())

    # ---------------------- Starting values ----------------------

    if start is None:
        beta0, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        resid0 = y_vec - X_mat @ beta0
        sigma0 = float(np.std(resid0))
        sigma0 = max(sigma0, 1e-3)

        ln_sv0 = np.log(sigma0 * 0.7)
        ln_su0 = np.log(sigma0 * 0.7)

        theta0_parts = [beta0]
        if W_mat is None:
            theta0_parts.append(np.array([ln_su0]))
        else:
            tmp = np.zeros(k_gamma_u)
            tmp[0] = ln_su0
            theta0_parts.append(tmp)
        if R_mat is None:
            theta0_parts.append(np.array([ln_sv0]))
        else:
            tmp = np.zeros(k_gamma_v)
            tmp[0] = ln_sv0
            theta0_parts.append(tmp)
        if k_delta_mu > 0:
            if Z_mat is None:
                theta0_parts.append(np.array([0.0]))
            else:
                theta0_parts.append(np.zeros(k_delta_mu))
        start = np.concatenate(theta0_parts)
    else:
        start = np.asarray(start, dtype=float).copy()
        if start.size != spec.k_total:
            raise ValueError(
                f"start has wrong length: got {start.size}, expected {spec.k_total}."
            )

    # ---------------------- Optimize ----------------------

    # Bounds: loose on betas/mu, tight on log-sigma parameters to keep
    # sigma in a numerically sensible range ~ [e^-12, e^5] ~ [6e-6, 150].
    bounds = []
    for _ in range(k_beta):
        bounds.append((-1e6, 1e6))
    # ln sigma_u block
    bounds.append((-12.0, 5.0))
    for _ in range(k_gamma_u - 1):
        bounds.append((-8.0, 8.0))
    # ln sigma_v block
    bounds.append((-12.0, 5.0))
    for _ in range(k_gamma_v - 1):
        bounds.append((-8.0, 8.0))
    # mu block (truncated-normal only)
    if k_delta_mu > 0:
        bounds.append((-50.0, 50.0))
        for _ in range(k_delta_mu - 1):
            bounds.append((-50.0, 50.0))

    result = minimize(
        neg_loglik,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
    )
    theta_hat = result.x
    ll_val = -neg_loglik(theta_hat)
    beta_hat, sigma_u_i, sigma_v_i, mu_i = _unpack(theta_hat)

    # ---------------------- Standard errors (numerical Hessian) ----------------------

    H = _fc.numerical_hessian(neg_loglik, theta_hat)
    vcov = _fc.safe_invert_hessian(H)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))

    # ---------------------- Efficiency scores ----------------------

    eps_hat = y_vec - X_mat @ beta_hat
    if dist == "half-normal":
        E_u, TE_bc = _fc.jondrow_halfnormal(eps_hat, sigma_v_i, sigma_u_i, sign)
    elif dist == "exponential":
        E_u, TE_bc = _fc.jondrow_exponential(eps_hat, sigma_v_i, sigma_u_i, sign)
    else:
        E_u, TE_bc = _fc.jondrow_truncnormal(eps_hat, sigma_v_i, sigma_u_i, mu_i, sign)
    TE_jlms = np.clip(np.exp(-E_u), 0.0, 1.0)

    # ---------------------- Specification tests ----------------------

    # OLS log-likelihood (H0: no inefficiency).
    resid_ols = y_vec - X_mat @ np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
    sigma_ols = np.std(resid_ols, ddof=0)
    ll_ols = np.sum(stats.norm.logpdf(resid_ols, loc=0.0, scale=max(sigma_ols, 1e-12)))
    lr_stat_noineff = _fc.lr_test_statistic(ll_val, ll_ols)

    # ---------------------- Assemble result ----------------------

    param_names = list(beta_names)
    if W_mat is None:
        param_names.append("ln_sigma_u")
    else:
        param_names.extend(w_names)
    if R_mat is None:
        param_names.append("ln_sigma_v")
    else:
        param_names.extend(r_names)
    if k_delta_mu > 0:
        if Z_mat is None:
            param_names.append("mu")
        else:
            param_names.extend(z_names)

    params = pd.Series(theta_hat, index=param_names)
    std_errors = pd.Series(se, index=param_names)

    # Summary scalars (for display).
    sigma_u_mean = float(np.mean(sigma_u_i))
    sigma_v_mean = float(np.mean(sigma_v_i))
    sigma_total = float(np.sqrt(sigma_u_mean**2 + sigma_v_mean**2))
    lam_mean = sigma_u_mean / sigma_v_mean if sigma_v_mean > 0 else np.nan
    gamma = sigma_u_mean**2 / (sigma_u_mean**2 + sigma_v_mean**2)

    return FrontierResult(
        params=params,
        std_errors=std_errors,
        model_info={
            "model_type": f"Stochastic Frontier ({'Cost' if cost else 'Production'})",
            "method": f"ML, {dist}",
            "inefficiency_dist": dist,
            "cost": cost,
            "sign": sign,
            "te_method": te_method,
            "has_usigma": usigma is not None,
            "has_vsigma": vsigma is not None,
            "has_emean": emean is not None,
            "sigma_u_mean": sigma_u_mean,
            "sigma_v_mean": sigma_v_mean,
            "sigma": sigma_total,
            "lambda": lam_mean,
            "gamma": gamma,
            "mean_efficiency_bc": float(np.mean(TE_bc)),
            "mean_efficiency_jlms": float(np.mean(TE_jlms)),
            "converged": bool(result.success),
        },
        data_info={
            "n_obs": n,
            "dep_var": y,
            "regressors": list(x),
            "df_resid": max(n - spec.k_total, 1),
        },
        diagnostics={
            "log_likelihood": float(ll_val),
            "ll_ols": float(ll_ols),
            "lr_no_inefficiency": float(lr_stat_noineff),
            "aic": float(-2.0 * ll_val + 2.0 * spec.k_total),
            "bic": float(-2.0 * ll_val + np.log(n) * spec.k_total),
            "sigma_u_i": sigma_u_i,
            "sigma_v_i": sigma_v_i,
            "mu_i": mu_i,
            "eps": eps_hat,
            "efficiency_bc": TE_bc,
            "efficiency_jlms": TE_jlms,
            "inefficiency_jlms": E_u,
            "efficiency_index": df.index.to_numpy(),
            "residual_skewness": _fc.ols_residual_skewness(resid_ols),
            "hessian": H,
            "vcov": vcov,
            "spec": spec,
        },
    )


__all__ = ["frontier", "FrontierResult"]
