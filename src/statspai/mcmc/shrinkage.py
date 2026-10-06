"""
Bayesian shrinkage and variable selection in the linear model:
``sp.bayes_shrink``.

* ``prior='lasso'`` -- the Bayesian lasso of Park and Casella (2008): a
  Laplace prior on every slope, written as a scale mixture of normals.
* ``prior='ssvs'``  -- stochastic search variable selection of George and
  McCulloch (1993): every slope is drawn from a narrow "spike" or a wide
  "slab", and the posterior probability of the slab is the inclusion
  probability.

Regressors are standardised for the prior and the coefficients reported
on the original scale. The intercept has a flat prior.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.utils import create_design_matrices
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._core import check_mcmc_args, rinvgamma, rmvnorm_prec, spawn_rngs
from .diagnostics import mcmc_summary
from .regress import BayesRegressResult


class _ShrinkShim:
    #: the predictive side treats the fit as a Gaussian linear model
    name = "normal"

    def __init__(self, X: np.ndarray, names: List[str], y: Any = None):
        self.X = X
        self.k = X.shape[1]
        self.xnames = names
        self.y = None if y is None else np.asarray(y, dtype=float)
        #: column of the draws that holds sigma2 (hyperparameters may follow)
        self.sigma2_index = self.k

    def linear_predictor(self, draws: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.asarray(draws[:, : self.k] @ X.T)

    @staticmethod
    def expected_value(eta: np.ndarray) -> np.ndarray:
        return eta


def bayes_shrink(
    formula: str,
    data: pd.DataFrame,
    prior: str = "lasso",
    lam: Optional[float] = None,
    lam_prior: Tuple[float, float] = (1.0, 1.0),
    spike_sd: float = 0.02,
    slab_sd: float = 1.0,
    inclusion: float = 0.5,
    sigma2_prior: Tuple[float, float] = (0.001, 0.001),
    standardize: bool = True,
    draws: int = 10000,
    burnin: int = 2000,
    thin: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
    global_scale: Optional[float] = None,
    p0: Optional[float] = None,
) -> BayesRegressResult:
    """Linear regression with a shrinkage or a variable-selection prior.

    For many candidate regressors, when the aim is prediction or a
    summary of which ones matter. For model probabilities over subsets
    see :func:`statspai.bma`.

    Parameters
    ----------
    formula : str
        ``'y ~ x1 + x2 + ...'``; the intercept is kept and not shrunk.
    data : DataFrame
    prior : {'lasso', 'ssvs', 'horseshoe'}
        ``'lasso'``: ``beta_j | sigma ~ Laplace(0, sigma / lam)``.
        ``'ssvs'``: ``beta_j ~ (1 - g_j) N(0, spike^2) + g_j N(0, slab^2)``
        with ``g_j ~ Bernoulli(inclusion)``.
        ``'horseshoe'``: ``beta_j | sigma ~ N(0, sigma^2 tau^2 lam_j^2)``
        with half-Cauchy local scales ``lam_j ~ C+(0, 1)`` and global
        scale ``tau ~ C+(0, global_scale)`` (Carvalho, Polson and Scott
        2010). Small coefficients are shrunk hard toward zero and large
        ones almost not at all, which neither the lasso nor a ridge
        prior does.
    lam : float, optional
        Lasso penalty. When omitted it is estimated, with the prior
        ``lam^2 ~ Gamma(shape, rate)`` given by ``lam_prior``.
    lam_prior : (shape, rate), default (1, 1)
    spike_sd, slab_sd : float
        SSVS: standard deviations of the spike and of the slab, in units
        of ``sd(y)`` per standard deviation of the regressor. A slope
        smaller than a few ``spike_sd`` counts as zero.
    inclusion : float, default 0.5
        SSVS: prior probability of the slab.
    sigma2_prior : (alpha0, delta0)
        ``sigma2 ~ InvGamma(alpha0 / 2, delta0 / 2)``.
    standardize : bool, default True
        Scale every regressor to unit standard deviation for the prior.
        Without it the prior treats a regressor in dollars and one in
        thousands of dollars differently.
    draws, burnin, thin, seed, level
        As in :func:`statspai.bayes_regress`.
    global_scale : float, optional
        Horseshoe: scale of the half-Cauchy prior on ``tau``. Default 1
        (the original horseshoe) unless ``p0`` is given.
    p0 : float, optional
        Horseshoe: prior guess of the number of coefficients that are
        far from zero. Sets ``global_scale = p0 / (p - p0) / sqrt(n)``,
        the calibration of Piironen and Vehtari (2017); with many
        regressors and few expected signals this is far smaller than 1
        and keeps the noise coefficients from adding up.

    Returns
    -------
    BayesRegressResult
        ``model='lasso'`` or ``'ssvs'``. Coefficients are on the original
        scale. For SSVS ``table['pip']`` is the posterior inclusion
        probability; for the lasso ``lam`` is among the parameters when
        estimated. For the horseshoe ``tau`` is among the parameters,
        ``table['shrinkage']`` is the posterior mean of each
        coefficient's shrinkage factor (0: untouched, 1: shrunk to zero)
        and ``model_info['m_eff']`` the posterior mean of the effective
        number of unshrunk coefficients. The fit supports
        ``posterior_predict``, ``log_lik`` and so :func:`statspai.loo`.

    Notes
    -----
    The Bayesian lasso shrinks but never sets a coefficient exactly to
    zero: its posterior mean is not sparse. Credible intervals after
    shrinkage are not confidence intervals for the unshrunk effect; for
    inference on one coefficient with many controls use
    ``sp.rlasso_effect`` or ``sp.dml``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.normal(size=(150, 6)), columns=list("abcdef"))
    >>> df["y"] = 1 + 2 * df["a"] - df["c"] + rng.normal(size=150)
    >>> fit = sp.bayes_shrink("y ~ a + b + c + d + e + f", df, prior="ssvs",
    ...                       draws=1000, burnin=300, seed=1)
    >>> bool(fit.table.loc["a", "pip"] > 0.95)
    True
    >>> las = sp.bayes_shrink("y ~ a + b + c + d + e + f", df, draws=1000,
    ...                       burnin=300, seed=1)

    References
    ----------
    park2008bayesian, george1993variable, carvalho2010horseshoe,
    makalic2016simple, piironen2017sparsity
    """
    prior = str(prior).lower()
    if prior in ("hs", "horseshoe"):
        prior = "horseshoe"
    if prior not in ("lasso", "ssvs", "horseshoe"):
        raise MethodIncompatibility("prior must be 'lasso', 'ssvs' or 'horseshoe'.")
    if prior != "horseshoe" and (global_scale is not None or p0 is not None):
        raise MethodIncompatibility("global_scale and p0 belong to prior='horseshoe'.")
    check_mcmc_args(draws, burnin, thin)
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    y_df, X_df = create_design_matrices(formula, data)
    names_all = [str(c) for c in X_df.columns]
    if "Intercept" not in names_all:
        raise MethodIncompatibility(
            "sp.bayes_shrink keeps an unshrunk intercept; write the formula "
            "without '- 1'."
        )
    y = np.asarray(y_df, dtype=float).reshape(-1)
    Xfull = np.asarray(X_df, dtype=float)
    keep = [i for i, c in enumerate(names_all) if c != "Intercept"]
    names = [names_all[i] for i in keep]
    X = Xfull[:, keep]
    n, p = X.shape
    if p < 1:
        raise MethodIncompatibility("There is no regressor to shrink.")
    if n < 4:
        raise DataInsufficient("Too few observations.")
    a0, d0 = (float(v) for v in sigma2_prior)
    if a0 <= 0 or d0 <= 0:
        raise MethodIncompatibility("sigma2_prior must be two positive numbers.")
    xbar = X.mean(axis=0)
    sx = X.std(axis=0, ddof=1) if standardize else np.ones(p)
    if np.any(sx == 0):
        raise MethodIncompatibility("A regressor is constant.")
    Z = (X - xbar) / sx
    ybar = y.mean()
    yc = y - ybar
    sy = float(y.std(ddof=1))
    ZtZ = Z.T @ Z
    Zty = Z.T @ yc
    yty = float(yc @ yc)
    rng = spawn_rngs(seed, 1)[0]
    n_iter = burnin + draws * thin
    est_lam = prior == "lasso" and lam is None
    if prior == "lasso":
        if lam is not None and not lam > 0:
            raise MethodIncompatibility("lam must be positive.")
        r_l, d_l = (float(v) for v in lam_prior)
        lam2 = float(lam) ** 2 if lam is not None else 1.0
    elif prior == "horseshoe":
        if global_scale is not None and p0 is not None:
            raise MethodIncompatibility("Pass global_scale or p0, not both.")
        if p0 is not None:
            if not 0 < p0 < p:
                raise MethodIncompatibility(
                    f"p0 must be between 0 and the number of regressors ({p})."
                )
            tau0 = float(float(p0) / (p - float(p0)) / np.sqrt(n))
        else:
            tau0 = 1.0 if global_scale is None else float(global_scale)
        if not tau0 > 0:
            raise MethodIncompatibility("global_scale must be positive.")
        loc2 = np.ones(p)  # lam_j^2
        nu_aux = np.ones(p)
        glob2 = min(1.0, tau0**2)  # tau^2
        xi_aux = 1.0
        out_k = np.zeros(p)
        m_eff_sum = 0.0
        zvar = np.diag(ZtZ) / n
    else:
        if not (0 < spike_sd < slab_sd):
            raise MethodIncompatibility("Need 0 < spike_sd < slab_sd.")
        if not 0.0 < inclusion < 1.0:
            raise MethodIncompatibility("inclusion must be in (0, 1).")
        v0, v1 = (spike_sd * sy) ** 2, (slab_sd * sy) ** 2
    beta = np.linalg.solve(ZtZ + np.eye(p), Zty)
    s2 = max(float(((yc - Z @ beta) ** 2).sum()) / max(n - 1, 1), 1e-12)
    tau2 = np.ones(p)
    gam = np.ones(p, dtype=bool)
    out_b = np.empty((draws, p))
    out_s = np.empty(draws)
    out_l = np.empty(draws)
    out_g = np.zeros(p)
    out_a = np.empty(draws)
    kept = 0
    for it in range(n_iter):
        if prior == "lasso":
            # beta | . ~ N(A^{-1} Z'y, s2 A^{-1}),  A = Z'Z + diag(1 / tau2)
            beta, _ = rmvnorm_prec(rng, Zty / s2, (ZtZ + np.diag(1.0 / tau2)) / s2)
            ssr = yty - 2.0 * beta @ Zty + beta @ ZtZ @ beta
            # the flat intercept is integrated out: n - 1 degrees of freedom
            s2 = rinvgamma(
                rng,
                (a0 + n - 1 + p) / 2.0,
                (d0 + ssr + float((beta * beta / tau2).sum())) / 2.0,
            )
            mu = np.sqrt(lam2 * s2) / np.maximum(np.abs(beta), 1e-12)
            tau2 = 1.0 / np.maximum(rng.wald(mu, lam2), 1e-12)
            if est_lam:
                lam2 = float(rng.gamma(r_l + p, 1.0 / (d_l + tau2.sum() / 2.0)))
        elif prior == "horseshoe":
            # Makalic and Schmidt (2016): every full conditional is inverse
            # gamma once each half-Cauchy is written as a scale mixture
            scale2 = np.maximum(glob2 * loc2, 1e-300)
            beta, _ = rmvnorm_prec(rng, Zty / s2, (ZtZ + np.diag(1.0 / scale2)) / s2)
            ssr = yty - 2.0 * beta @ Zty + beta @ ZtZ @ beta
            b2 = beta * beta
            s2 = rinvgamma(
                rng,
                (a0 + n - 1 + p) / 2.0,
                (d0 + ssr + float((b2 / scale2).sum())) / 2.0,
            )
            loc2 = (1.0 / nu_aux + b2 / (2.0 * glob2 * s2)) / rng.gamma(1.0, size=p)
            loc2 = np.clip(loc2, 1e-300, 1e300)
            glob2 = float(
                (1.0 / xi_aux + float((b2 / loc2).sum()) / (2.0 * s2))
                / rng.gamma((p + 1.0) / 2.0)
            )
            glob2 = min(max(glob2, 1e-300), 1e300)
            nu_aux = (1.0 + 1.0 / loc2) / rng.gamma(1.0, size=p)
            xi_aux = float((1.0 / tau0**2 + 1.0 / glob2) / rng.gamma(1.0))
        else:
            dvar = np.where(gam, v1, v0)
            beta, _ = rmvnorm_prec(rng, Zty / s2, ZtZ / s2 + np.diag(1.0 / dvar))
            ssr = yty - 2.0 * beta @ Zty + beta @ ZtZ @ beta
            s2 = rinvgamma(rng, (a0 + n - 1) / 2.0, (d0 + ssr) / 2.0)
            l1 = np.log(inclusion) + stats.norm.logpdf(beta, 0.0, np.sqrt(v1))
            l0 = np.log1p(-inclusion) + stats.norm.logpdf(beta, 0.0, np.sqrt(v0))
            gam = np.log(rng.random(p)) < l1 - np.logaddexp(l0, l1)
        if it >= burnin and (it - burnin) % thin == 0:
            out_b[kept] = beta / sx
            out_s[kept] = s2
            out_l[kept] = np.sqrt(lam2) if prior == "lasso" else np.nan
            if prior == "horseshoe":
                out_l[kept] = np.sqrt(glob2)
                kappa = 1.0 / (1.0 + n * zvar * glob2 * loc2)
                out_k += kappa
                m_eff_sum += float((1.0 - kappa).sum())
            # intercept | beta, s2 ~ N(ybar - xbar'b, s2 / n) under the flat prior
            out_a[kept] = (
                ybar - xbar @ out_b[kept] + np.sqrt(s2 / n) * rng.standard_normal()
            )
            if prior == "ssvs":
                out_g += gam
            kept += 1
    cols: Dict[str, np.ndarray] = {"Intercept": out_a}
    for j, nm in enumerate(names):
        cols[nm] = out_b[:, j]
    cols["sigma2"] = out_s
    if est_lam:
        cols["lam"] = out_l
    if prior == "horseshoe":
        cols["tau"] = out_l
    d_df = pd.DataFrame(cols)
    summ = mcmc_summary(d_df, quantiles=())
    lo = (1.0 - level) / 2.0
    table = pd.DataFrame(
        {
            "mean": summ["mean"],
            "sd": summ["sd"],
            "mcse": summ["ts_se"],
            "ess": summ["ess"],
            "lower": d_df.quantile(lo),
            "median": d_df.quantile(0.5),
            "upper": d_df.quantile(1.0 - lo),
            "prob_positive": (d_df > 0).mean(),
        }
    )
    info: Dict[str, Any] = {"prior": prior, "standardize": bool(standardize)}
    if prior == "ssvs":
        pip = pd.Series(np.nan, index=table.index)
        pip.loc[names] = out_g / draws
        table["pip"] = pip
        info.update({"spike_sd": spike_sd, "slab_sd": slab_sd, "inclusion": inclusion})
    elif prior == "horseshoe":
        shrink = pd.Series(np.nan, index=table.index)
        shrink.loc[names] = out_k / draws
        table["shrinkage"] = shrink
        info.update({"global_scale": tau0, "m_eff": m_eff_sum / draws})
    elif lam is not None:
        info["lam"] = float(lam)
    diag: Dict[str, Any] = {"warnings": [], "min_ess": float(table["ess"].min())}
    res = BayesRegressResult(
        model=prior,
        formula=formula,
        params=table["mean"].copy(),
        std_errors=table["sd"].copy(),
        table=table,
        draws=d_df,
        chain=np.zeros(draws, dtype=int),
        n_obs=n,
        n_draws=draws,
        burnin=burnin,
        thin=thin,
        chains=1,
        sampler="Gibbs",
        acceptance_rate=None,
        prior={
            "slopes": prior,
            "intercept": "flat",
            "sigma2": f"InvGamma({a0 / 2:g}, {d0 / 2:g})",
        },
        level=level,
        model_info=info,
        diagnostics_info=diag,
        _model=_ShrinkShim(np.column_stack([np.ones(n), X]), ["Intercept"] + names, y),
        _design_info=getattr(X_df, "design_info", None),
    )
    if diag["min_ess"] < 100:
        text = (
            "The chain mixes slowly: effective sample size "
            f"{diag['min_ess']:.0f}. Increase draws or thin."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    return res
