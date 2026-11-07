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
from ..output._lineage import records_provenance
from ._chains import ChainRun, as_array
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


def _hs_variance(loc2: Any, glob2: Any, slab2: Any) -> Any:
    """Prior variance of a coefficient in units of sigma^2: ``tau^2 lam^2``,
    or its regularized version ``1 / (1 / (tau^2 lam^2) + 1 / c^2)``."""
    plain = glob2 * loc2
    if np.isinf(slab2):
        return plain
    return 1.0 / (1.0 / np.maximum(plain, 1e-300) + 1.0 / slab2)


def _regularized_hs_step(
    rng: np.random.Generator,
    b2: np.ndarray,
    s2: float,
    loc2: np.ndarray,
    glob2: float,
    slab2: float,
    tau0: float,
    slab_df: float,
    slab_scale2: float,
    steps: np.ndarray,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """One Metropolis update of each scale of the regularized horseshoe.

    Random walks on ``log lam_j`` (all coordinates at once: they are
    independent given the rest), ``log tau`` and ``log c^2``. Returns the
    new values and the acceptance rates of the four moves.
    """

    def beta_term(v: Any) -> Any:  # log N(beta; 0, s2 v) up to a constant
        return -0.5 * np.log(v) - b2 / (2.0 * s2 * v)

    p = b2.size
    # local scales: half-Cauchy(0, 1) on lam, Jacobian of the log
    th = 0.5 * np.log(loc2)
    cand = th + steps[0] * rng.standard_normal(p)
    c2 = np.exp(2.0 * cand)
    cur_lp = beta_term(_hs_variance(loc2, glob2, slab2)) - np.log1p(loc2) + th
    new_lp = beta_term(_hs_variance(c2, glob2, slab2)) - np.log1p(c2) + cand
    take = np.log(rng.random(p)) < new_lp - cur_lp
    loc2 = np.clip(np.where(take, c2, loc2), 1e-300, 1e300)
    rate_l = float(take.mean())
    # global scale: half-Cauchy(0, tau0)
    th_t = 0.5 * np.log(glob2)
    cand_t = th_t + steps[1] * rng.standard_normal()
    g2 = float(np.exp(2.0 * cand_t))
    cur = (
        beta_term(_hs_variance(loc2, glob2, slab2)).sum()
        - np.log1p(glob2 / tau0**2)
        + th_t
    )
    new = (
        beta_term(_hs_variance(loc2, g2, slab2)).sum() - np.log1p(g2 / tau0**2) + cand_t
    )
    rate_t = 0.0
    if np.log(rng.random()) < new - cur:
        glob2, rate_t = min(max(g2, 1e-300), 1e300), 1.0
    # tau up and every lam down by the same factor: the products
    # tau * lam_j, and with them the likelihood of the coefficients, do not
    # change, so only the priors decide. This is the direction along which
    # the posterior of tau is most correlated with the local scales.
    delta = steps[3] * rng.standard_normal()
    shifted = loc2 * np.exp(-2.0 * delta)
    g2 = glob2 * float(np.exp(2.0 * delta))
    change = (
        float((np.log1p(loc2) - np.log1p(shifted)).sum())
        - p * delta
        + float(np.log1p(glob2 / tau0**2) - np.log1p(g2 / tau0**2))
        + delta
    )
    rate_j = 0.0
    if np.log(rng.random()) < change:
        loc2 = np.clip(shifted, 1e-300, 1e300)
        glob2, rate_j = min(max(g2, 1e-300), 1e300), 1.0
    # slab: c^2 ~ InvGamma(df / 2, df * scale^2 / 2), Jacobian of the log
    th_c = np.log(slab2)
    cand_c = th_c + steps[2] * rng.standard_normal()
    s2c = float(np.exp(cand_c))
    half = 0.5 * slab_df

    def slab_lp(val: float, log_val: float) -> float:
        return float(
            beta_term(_hs_variance(loc2, glob2, val)).sum()
            - half * log_val
            - half * slab_scale2 / val
        )

    rate_c = 0.0
    if np.log(rng.random()) < slab_lp(s2c, cand_c) - slab_lp(slab2, th_c):
        slab2, rate_c = s2c, 1.0
    return loc2, glob2, slab2, np.array([rate_l, rate_t, rate_c, rate_j])


@records_provenance("sp.bayes_shrink", record_defaults=("seed",))
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
    slab_scale: Optional[float] = None,
    slab_df: float = 4.0,
    family: str = "gaussian",
    inits: Any = None,
    continue_from: Any = None,
    verbose: bool = False,
    keep_burnin: bool = False,
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
        Rows with a missing value in the outcome or a regressor are dropped
        before fitting (listwise deletion), with an ``AssumptionWarning`` that
        counts them; ``n_obs`` on the result is the number of rows used.
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

    slab_scale : float, optional
        Horseshoe: regularize it (Piironen and Vehtari 2017). The local
        scales are replaced by ``c lam_j / sqrt(c^2 + tau^2 lam_j^2)``
        with ``c^2 ~ InvGamma(slab_df / 2, slab_df * slab_scale^2 / 2)``,
        so a coefficient that escapes the shrinkage still has a
        Student-t prior of scale ``slab_scale`` instead of none at all.
        In units of the residual standard deviation per standard
        deviation of the regressor; 2.5 is a wide slab. This is the
        ``hs()`` prior of R ``rstanarm``. It matters when the data
        identify a coefficient weakly, with collinear regressors or few
        observations per coefficient, where the plain horseshoe lets it
        drift.
    slab_df : float, default 4
        Degrees of freedom of the slab.
    family : {'gaussian', 'logit'}, default 'gaussian'
        ``'logit'``: a binary outcome with the horseshoe (plain or
        regularized) on the slopes of a logistic regression, sampled by
        Polya-Gamma data augmentation. There is no residual variance;
        the scales are those of the log odds, and ``p0`` is calibrated
        with a latent standard deviation of 2. Too many regressors for
        the number of observations separate a binary outcome, where the
        maximum-likelihood fit does not exist; the horseshoe keeps the
        fit finite, and with ``slab_scale`` keeps the separated
        coefficients from drifting.
    inits : dict, list of dicts or array, optional
        Starting values of the scale parameters the first step uses:
        ``sigma2`` (Gaussian), ``lam`` (lasso with ``lam`` estimated),
        ``tau`` and ``slab`` (horseshoe). The first step draws the
        coefficients given them, so values for ``Intercept`` and the
        slopes are accepted (``result.params`` can be passed) and not
        used, except with ``family='logit'``, whose first step
        conditions on the coefficients: there they are needed and
        ``sigma2`` does not exist. The latent local scales start at
        one. One chain. ``result.inits`` records the starting state of
        each chain.
    continue_from : BayesRegressResult, optional
        Extend the chains of an earlier ``bayes_shrink`` fit of the
        same data, prior and settings by ``draws`` more draws each,
        from where each chain stopped and with its own generator, with
        no new burn-in. The sampler draws its random numbers iteration
        by iteration, so a fit of ``m`` draws continued by ``m'``
        equals one fit of ``m + m'`` draws with the same seed, draw for
        draw. The shrinkage factors, inclusion probabilities and
        ``m_eff`` then average over both runs.
    verbose : bool, default False
        Print a line to standard output when each chain starts and
        ends. Warnings are not affected.
    keep_burnin : bool, default False
        Keep the burn-in iterations in ``result.burnin_draws`` (shown
        by ``plot(kind='trace')``); they are never used in the
        summaries. The intercept is drawn only at kept iterations, so
        its burn-in rows hold its conditional mean given the slopes,
        not a draw: drawing it would change the kept draws.

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
    makalic2016simple, piironen2017sparsity, polson2013bayesian
    """
    prior = str(prior).lower()
    if prior in ("hs", "horseshoe"):
        prior = "horseshoe"
    if prior not in ("lasso", "ssvs", "horseshoe"):
        raise MethodIncompatibility("prior must be 'lasso', 'ssvs' or 'horseshoe'.")
    if prior != "horseshoe" and (
        global_scale is not None or p0 is not None or slab_scale is not None
    ):
        raise MethodIncompatibility(
            "global_scale, p0 and slab_scale belong to prior='horseshoe'."
        )
    check_mcmc_args(draws, burnin, thin)
    ctl = ChainRun(
        "bayes_shrink",
        inits=inits,
        continue_from=continue_from,
        verbose=verbose,
        keep_burnin=keep_burnin,
    )
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
    family_key = str(family).lower()
    if family_key in ("logit", "logistic", "binomial"):
        if prior != "horseshoe":
            raise MethodIncompatibility(
                "family='logit' is implemented for prior='horseshoe'."
            )
        from ._shrink_logit import shrink_logit

        return shrink_logit(
            formula,
            y,
            X,
            names,
            getattr(X_df, "design_info", None),
            standardize,
            global_scale,
            p0,
            slab_scale,
            slab_df,
            draws,
            burnin,
            thin,
            seed,
            level,
            ctl,
        )
    if family_key not in ("gaussian", "normal"):
        raise MethodIncompatibility("family must be 'gaussian' or 'logit'.")
    a0, d0 = (float(v) for v in sigma2_prior)
    if a0 <= 0 or d0 <= 0:
        raise MethodIncompatibility(
            "sp.bayes_shrink: sigma2_prior must be two positive numbers."
        )
    xbar = X.mean(axis=0)
    sx = X.std(axis=0, ddof=1) if standardize else np.ones(p)
    if np.any(sx == 0):
        raise MethodIncompatibility("sp.bayes_shrink: A regressor is constant.")
    Z = (X - xbar) / sx
    ybar = y.mean()
    yc = y - ybar
    sy = float(y.std(ddof=1))
    ZtZ = Z.T @ Z
    Zty = Z.T @ yc
    yty = float(yc @ yc)
    est_lam = prior == "lasso" and lam is None
    regularized = False
    if prior == "lasso":
        if lam is not None and not lam > 0:
            raise MethodIncompatibility("lam must be positive.")
        r_l, d_l = (float(v) for v in lam_prior)
        lam2 = float(lam) ** 2 if lam is not None else 1.0
    elif prior == "horseshoe":
        if global_scale is not None and p0 is not None:
            raise MethodIncompatibility(
                "sp.bayes_shrink: Pass global_scale or p0, not both."
            )
        if p0 is not None:
            if not 0 < p0 < p:
                raise MethodIncompatibility(
                    f"p0 must be between 0 and the number of regressors ({p})."
                )
            tau0 = float(float(p0) / (p - float(p0)) / np.sqrt(n))
        else:
            tau0 = 1.0 if global_scale is None else float(global_scale)
        if not tau0 > 0:
            raise MethodIncompatibility(
                "sp.bayes_shrink: global_scale must be positive."
            )
        regularized = slab_scale is not None
        slab_s = float(slab_scale) if slab_scale is not None else 0.0
        if regularized and not (slab_s > 0 and slab_df > 0):
            raise MethodIncompatibility(
                "sp.bayes_shrink: slab_scale and slab_df must be positive."
            )
        slab2 = slab_s**2 if regularized else np.inf
        # log-scale steps: lam, tau, c^2, joint shift of tau against lam
        steps = np.array([1.0, 0.5, 1.0, 0.5])
        out_c = np.empty(draws)
        loc2: np.ndarray = np.ones(p)  # lam_j^2
        nu_aux: np.ndarray = np.ones(p)
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
    out_g = np.zeros(p)
    scale_names = (
        ["sigma2"]
        + (["lam"] if est_lam else [])
        + (
            (["tau"] + (["slab"] if regularized else []))
            if prior == "horseshoe"
            else []
        )
    )
    ctl.setup(
        {
            "y": y,
            "X": X,
            "prior": prior,
            "settings": [lam, lam_prior, spike_sd, slab_sd, inclusion, a0, d0],
            "horseshoe": [global_scale, p0, slab_scale, slab_df],
            "standardize": bool(standardize),
        },
        chains=1,
        thin=thin,
        burnin=burnin,
        draws=draws,
        seed=seed,
        default_burnin=2000,
        make_rngs=lambda c: spawn_rngs(seed, 1),
        exact=True,
        init_names=scale_names,
        drawn_first=["Intercept"] + names,
        support=lambda row: bool(np.all(row > 0)),
        support_text="the variance and the scales must be positive",
    )
    thin, run_burnin, n_iter = ctl.thin, ctl.burnin, ctl.n_iter
    rng = ctl.rngs[0]
    ctl.begin(0)
    if ctl.continuing:
        st = ctl.prev_state(0)
        s2, tau2, lam2, gam, out_g = (
            float(st["s2"]),
            as_array(st["tau2"]),
            float(st["lam2"]),
            as_array(st["gam"], bool),
            as_array(st["out_g"]),
        )
        if prior == "horseshoe":
            loc2, nu_aux, steps, out_k = (
                as_array(st[key]) for key in ("loc2", "nu_aux", "steps", "out_k")
            )
            glob2, slab2, xi_aux, m_eff_sum = (
                float(st[key]) for key in ("glob2", "slab2", "xi_aux", "m_eff_sum")
            )
    elif ctl.starts[0] is not None:
        given = dict(zip(scale_names, ctl.starts[0]))
        s2 = float(given["sigma2"])
        if est_lam:
            lam2 = float(given["lam"]) ** 2
        if prior == "horseshoe":
            glob2 = float(given["tau"]) ** 2
            if regularized:
                slab2 = float(given["slab"]) ** 2

    def scale_row() -> List[float]:
        vals = [s2]
        if est_lam:
            vals.append(float(np.sqrt(lam2)))
        if prior == "horseshoe":
            vals.append(float(np.sqrt(glob2)))
            if regularized:
                vals.append(float(np.sqrt(slab2)))
        return vals

    # the coefficients are drawn first
    first = np.array([np.nan] * (p + 1) + scale_row())
    burn = np.empty((run_burnin if ctl.keep_burnin else 0, p + 1 + len(scale_names)))
    out_b = np.empty((draws, p))
    out_s = np.empty(draws)
    out_l = np.empty(draws)
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
            scale2 = np.maximum(_hs_variance(loc2, glob2, slab2), 1e-300)
            beta, _ = rmvnorm_prec(rng, Zty / s2, (ZtZ + np.diag(1.0 / scale2)) / s2)
            ssr = yty - 2.0 * beta @ Zty + beta @ ZtZ @ beta
            b2 = beta * beta
            s2 = rinvgamma(
                rng,
                (a0 + n - 1 + p) / 2.0,
                (d0 + ssr + float((b2 / scale2).sum())) / 2.0,
            )
            if regularized:
                # the slab breaks conjugacy: Metropolis on the log scales,
                # step sizes tuned during burn-in only
                # several cheap passes over the scales per draw of the
                # coefficients: the scales are what mixes slowly
                for _ in range(5):
                    loc2, glob2, slab2, rates = _regularized_hs_step(
                        rng, b2, s2, loc2, glob2, slab2, tau0, slab_df, slab_s**2, steps
                    )
                if it < run_burnin:
                    steps *= np.exp(0.03 * (rates - 0.35))
            else:
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
        if it < len(burn):
            slopes = beta / sx
            # the intercept's conditional mean: a draw here would shift
            # the generator and change the kept draws
            burn[it] = [ybar - xbar @ slopes, *slopes, *scale_row()]
        if it >= run_burnin and (it - run_burnin) % thin == 0:
            out_b[kept] = beta / sx
            out_s[kept] = s2
            out_l[kept] = np.sqrt(lam2) if prior == "lasso" else np.nan
            if prior == "horseshoe":
                out_l[kept] = np.sqrt(glob2)
                out_c[kept] = np.sqrt(slab2)
                kappa = 1.0 / (1.0 + n * zvar * _hs_variance(loc2, glob2, slab2))
                out_k += kappa
                m_eff_sum += float((1.0 - kappa).sum())
            # intercept | beta, s2 ~ N(ybar - xbar'b, s2 / n) under the flat prior
            out_a[kept] = (
                ybar - xbar @ out_b[kept] + np.sqrt(s2 / n) * rng.standard_normal()
            )
            if prior == "ssvs":
                out_g += gam
            kept += 1
    state: Dict[str, Any] = {
        "s2": s2,
        "tau2": tau2.copy(),
        "lam2": lam2 if prior == "lasso" else 0.0,
        "gam": gam.copy(),
        "out_g": out_g.copy(),
    }
    if prior == "horseshoe":
        state.update(
            loc2=loc2.copy(),
            nu_aux=nu_aux.copy(),
            steps=steps.copy(),
            out_k=out_k.copy(),
            glob2=glob2,
            slab2=slab2,
            xi_aux=xi_aux,
            m_eff_sum=m_eff_sum,
        )
    ctl.end(0, state, start=first, burn=burn)
    cols: Dict[str, np.ndarray] = {"Intercept": out_a}
    for j, nm in enumerate(names):
        cols[nm] = out_b[:, j]
    cols["sigma2"] = out_s
    if est_lam:
        cols["lam"] = out_l
    if prior == "horseshoe":
        cols["tau"] = out_l
        if regularized:
            cols["slab"] = out_c
    d_df = pd.DataFrame(cols)
    prev = ctl.prev
    if prev is not None:
        d_df = pd.DataFrame(
            ctl.stack([d_df.to_numpy()], prev.draws.to_numpy()), columns=d_df.columns
        )
    n_kept = len(d_df)
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
        pip.loc[names] = out_g / n_kept
        table["pip"] = pip
        info.update({"spike_sd": spike_sd, "slab_sd": slab_sd, "inclusion": inclusion})
    elif prior == "horseshoe":
        shrink = pd.Series(np.nan, index=table.index)
        shrink.loc[names] = out_k / n_kept
        table["shrinkage"] = shrink
        info.update({"global_scale": tau0, "m_eff": m_eff_sum / n_kept})
        if regularized:
            info.update(
                {
                    "slab_scale": slab_s,
                    "slab_df": float(slab_df),
                    "metropolis_steps": [float(v) for v in steps],
                }
            )
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
        chain=np.zeros(n_kept, dtype=int),
        n_obs=n,
        n_draws=n_kept,
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
    ctl.attach(res, list(d_df.columns), info)
    ctl.done(diag["min_ess"], float("nan"))
    if diag["min_ess"] < 100:
        text = (
            "The chain mixes slowly: effective sample size "
            f"{diag['min_ess']:.0f}. Increase draws or thin."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    return res
