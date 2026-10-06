"""
Bayesian instrumental variables by Gibbs sampling: ``sp.bayes_ivreg``.

The linear model with one endogenous regressor,

    d = z' delta + v
    y = beta d + x' gamma + eps,        (v, eps) ~ N(0, Sigma),

with normal priors on the coefficients and an inverse-Wishart prior on
``Sigma``. All three full conditionals are standard (Rossi, Allenby and
McCulloch 2005, section 5.4), so the sampler needs no tuning and no PyMC.
The PyMC estimator for the same design is :func:`statspai.bayes_iv`.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg

from ..core.utils import create_design_matrices
from ..exceptions import (
    AssumptionWarning,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from ._core import check_mcmc_args, normal_prior, rinvwishart, rmvnorm_prec, spawn_rngs
from .diagnostics import gelman_rubin, mcmc_summary
from .regress import BayesRegressResult

_IV_PART = re.compile(r"\(\s*([^()~]+?)\s*~\s*([^()]+?)\s*\)")


class _IVShim:
    """Model interface for ``predict``: the structural equation."""

    def __init__(self, W: np.ndarray, names: List[str]):
        self.X = W
        self.k = W.shape[1]
        self.xnames = names

    def linear_predictor(self, draws: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.asarray(draws[:, : self.k] @ X.T)

    @staticmethod
    def expected_value(eta: np.ndarray) -> np.ndarray:
        return eta


def _split_formula(formula: str) -> Tuple[str, str, str, str]:
    if "~" not in formula:
        raise MethodIncompatibility(
            "formula must look like 'y ~ x1 + (d ~ z1 + z2)'; got " f"{formula!r}."
        )
    lhs, rhs = formula.split("~", 1)
    found = _IV_PART.findall(rhs)
    if len(found) != 1:
        raise MethodIncompatibility(
            "The formula needs exactly one '(endogenous ~ instruments)' "
            f"part, as in 'y ~ x1 + (d ~ z1 + z2)'; got {formula!r}."
        )
    endog, inst = found[0]
    if "+" in endog:
        raise MethodIncompatibility(
            "sp.bayes_ivreg handles one endogenous regressor; "
            f"got '{endog.strip()}'. For several, use sp.ivreg."
        )
    exog = _IV_PART.sub("", rhs)
    exog = re.sub(r"\+\s*\+", "+", exog).strip().strip("+").strip()
    return lhs.strip(), endog.strip(), inst.strip(), exog or "1"


def bayes_ivreg(
    formula: str,
    data: pd.DataFrame,
    prior_mean: Any = 0.0,
    prior_var: Any = 1000.0,
    first_stage_prior_var: Any = 1000.0,
    sigma_prior: Optional[Tuple[float, Any]] = None,
    draws: int = 10000,
    burnin: int = 2000,
    thin: int = 1,
    chains: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> BayesRegressResult:
    """Bayesian linear instrumental variables by Gibbs sampling.

    The joint model of the first stage and the structural equation,

    ``d = z' delta + v``, ``y = beta d + x' gamma + eps``,
    ``(v, eps) ~ N(0, Sigma)``,

    with ``z`` the exogenous regressors and the instruments. The posterior
    of ``beta`` carries the uncertainty of the first stage and of the
    error covariance. With a strong instrument and weak priors it is
    close to normal around 2SLS; with a weak instrument it is wide and
    skewed, and it then depends on the priors.

    Parameters
    ----------
    formula : str
        ``'y ~ x1 + x2 + (d ~ z1 + z2)'``, the syntax of ``sp.ivreg``:
        one endogenous regressor ``d`` and its excluded instruments in
        parentheses.
    data : DataFrame
    prior_mean, prior_var
        Normal prior of the structural coefficients, in the order
        exogenous terms then the endogenous regressor (scalar, vector or
        matrix as in :func:`statspai.bayes_regress`). Default variance
        1000.
    first_stage_prior_var : float, array or matrix, default 1000
        Prior variance of the first-stage coefficients (mean zero).
    sigma_prior : (df, scale), optional
        ``Sigma ~ InvWishart(df, scale)`` with ``scale`` a number
        (times the identity) or a 2 x 2 matrix. Default ``(3, 0.02)``,
        which adds 0.02 to sums of squared residuals and so is weak on
        any ordinary scale. R ``bayesm::rivGibbs`` defaults to an
        identity scale, ``sigma_prior=(3, 1.0)``.
    draws, burnin, thin, chains, seed, level
        As in :func:`statspai.bayes_regress`.

    Returns
    -------
    BayesRegressResult
        ``model='iv'``. Parameters: the structural coefficients, then
        ``rho`` (the correlation of the two errors: zero means the
        regressor is exogenous), ``sigma2_y`` and ``sigma2_d`` (the error
        variances of the structural equation and of the first stage) and
        the first-stage coefficients, prefixed ``fs:``.
        ``model_info['first_stage_F']`` is the usual F statistic of the
        excluded instruments.

    Notes
    -----
    A ``statspai.AssumptionWarning`` is raised when the first-stage F is
    below 10. The posterior is still the correct posterior of this model,
    but it is far from normal, its mean may not exist, and it is
    sensitive to the priors; report the median and the interval, and how
    they move with ``prior_var``.

    ``fit.prob("rho > 0")`` is the posterior probability of positive
    selection; an interval for ``rho`` that excludes zero is the Bayesian
    counterpart of a Hausman test.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> z, v = rng.normal(size=n), rng.normal(size=n)
    >>> d = z + v
    >>> y = 1 + 0.5 * d + 0.8 * v + 0.6 * rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "z": z})
    >>> fit = sp.bayes_ivreg("y ~ (d ~ z)", df, draws=2000, burnin=500, seed=1)
    >>> list(fit.params.index)[:3]
    ['Intercept', 'd', 'rho']
    >>> bool(fit.prob("rho > 0") > 0.99)
    True

    References
    ----------
    rossi2005bayesian, ramirezhassan2026introduction
    """
    check_mcmc_args(draws, burnin, thin, chains)
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("data must be a pandas DataFrame.")
    ydep, endog, inst, exog = _split_formula(formula)
    if endog not in data.columns:
        raise MethodIncompatibility(
            f"The endogenous regressor {endog!r} must be a column of data."
        )
    # complete cases across both equations
    y_df, X_df = create_design_matrices(f"{ydep} ~ {exog}", data)
    d_df, Z_df = create_design_matrices(f"{endog} ~ {exog} + {inst}", data)
    idx = X_df.index.intersection(Z_df.index)
    if len(idx) < len(X_df) or len(idx) < len(Z_df):
        y_df, X_df = y_df.loc[idx], X_df.loc[idx]
        d_df, Z_df = d_df.loc[idx], Z_df.loc[idx]
    y = np.asarray(y_df, dtype=float).reshape(-1)
    d = np.asarray(d_df, dtype=float).reshape(-1)
    X = np.asarray(X_df, dtype=float)
    Z = np.asarray(Z_df, dtype=float)
    xnames = [str(c) for c in X_df.columns]
    znames = [str(c) for c in Z_df.columns]
    n, kx = X.shape
    kz = Z.shape[1]
    n_inst = kz - kx
    if n_inst < 1:
        raise MethodIncompatibility(
            "No excluded instrument: the terms in parentheses are all "
            "among the exogenous regressors."
        )
    if n <= kz + 2:
        raise DataInsufficient(f"{n} observations for {kz} first-stage coefficients.")
    if np.linalg.matrix_rank(Z) < kz:
        raise MethodIncompatibility(
            "The exogenous regressors and instruments are collinear."
        )
    W = np.column_stack([X, d])
    wnames = xnames + [endog]
    kw = kx + 1
    if np.linalg.matrix_rank(W) < kw:
        raise MethodIncompatibility(
            "The endogenous regressor is collinear with the exogenous ones."
        )

    t0, T0, A_t = normal_prior(kw, prior_mean, prior_var, wnames)
    _, D0, A_d = normal_prior(kz, 0.0, first_stage_prior_var, znames)
    if sigma_prior is None:
        nu0, V0 = 3.0, np.eye(2) * 0.02
    else:
        nu0 = float(sigma_prior[0])
        sc = np.asarray(sigma_prior[1], dtype=float)
        V0 = np.eye(2) * float(sc) if sc.ndim == 0 else 0.5 * (sc + sc.T)
        if V0.shape != (2, 2) or nu0 <= 1:
            raise MethodIncompatibility(
                "sigma_prior must be (df, scale) with df > 1 and scale a "
                "positive number or a 2 x 2 positive definite matrix."
            )
        try:
            linalg.cholesky(V0)
        except linalg.LinAlgError as exc:
            raise MethodIncompatibility(
                "sigma_prior scale must be positive definite."
            ) from exc

    # first-stage strength (classical F of the excluded instruments)
    def rss(M: np.ndarray, t: np.ndarray) -> float:
        e = t - M @ np.linalg.lstsq(M, t, rcond=None)[0]
        return float(e @ e)

    rss_full, rss_rest = rss(Z, d), rss(X, d)
    f_stat = ((rss_rest - rss_full) / n_inst) / (rss_full / (n - kz))

    ZtZ, Ztd, Zty, ZtX = Z.T @ Z, Z.T @ d, Z.T @ y, Z.T @ X
    WtW, Wty, Wtd, WtZ = W.T @ W, W.T @ y, W.T @ d, W.T @ Z
    A_t_t0 = A_t @ t0

    # start at two-stage least squares
    delta_ols = np.linalg.solve(ZtZ, Ztd)
    What = np.column_stack([X, Z @ delta_ols])
    theta_2sls = np.linalg.lstsq(What, y, rcond=None)[0]

    n_iter = burnin + draws * thin
    rngs = spawn_rngs(seed, chains)
    names = wnames + ["rho", "sigma2_y", "sigma2_d"] + [f"fs:{c}" for c in znames]
    pieces = []
    for ch in range(chains):
        rng = rngs[ch]
        delta = delta_ols.copy()
        theta = theta_2sls.copy()
        if chains > 1:
            delta = delta + rng.standard_normal(kz) * 0.1 * np.abs(delta).max()
            theta = theta + rng.standard_normal(kw) * 0.1 * max(np.abs(theta).max(), 1)
        v = d - Z @ delta
        eps = y - W @ theta
        E = np.column_stack([v, eps])
        Sigma = (V0 + E.T @ E) / n
        out = np.empty((draws, len(names)))
        kept = 0
        for it in range(n_iter):
            s11, s12, s22 = Sigma[0, 0], Sigma[0, 1], Sigma[1, 1]
            # 1. structural coefficients given the first stage: eps | v is
            #    N(r v, tau2), so y - r v = W theta + noise
            r = s12 / s11
            tau2 = s22 - s12 * s12 / s11
            Wtv = Wtd - WtZ @ delta
            theta, _ = rmvnorm_prec(
                rng, A_t_t0 + (Wty - r * Wtv) / tau2, A_t + WtW / tau2
            )
            beta, gamma = theta[-1], theta[:-1]
            # 2. first stage given the structural coefficients. Two
            #    equations carry delta: d = Z delta + v and
            #    y - X gamma = beta Z delta + (beta v + eps)
            B = np.array([[1.0, 0.0], [beta, 1.0]])
            Oinv = linalg.inv(B @ Sigma @ B.T)
            c1 = Oinv[0, 0] + beta * Oinv[0, 1]
            c2 = Oinv[0, 1] + beta * Oinv[1, 1]
            scale = Oinv[0, 0] + 2.0 * beta * Oinv[0, 1] + beta * beta * Oinv[1, 1]
            rhs = c1 * Ztd + c2 * (Zty - ZtX @ gamma)
            delta, _ = rmvnorm_prec(rng, rhs, A_d + scale * ZtZ)
            # 3. error covariance
            v = d - Z @ delta
            eps = y - W @ theta
            E = np.column_stack([v, eps])
            Sigma = rinvwishart(rng, nu0 + n, V0 + E.T @ E)
            if it >= burnin and (it - burnin) % thin == 0:
                out[kept, :kw] = theta
                out[kept, kw] = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
                out[kept, kw + 1] = Sigma[1, 1]
                out[kept, kw + 2] = Sigma[0, 0]
                out[kept, kw + 3 :] = delta
                kept += 1
        pieces.append(out)
    arr = np.vstack(pieces)
    d_df2 = pd.DataFrame(arr, columns=names)
    chain_idx = np.repeat(np.arange(chains), draws)
    ess = np.zeros(len(names))
    for ch in range(chains):
        ess += mcmc_summary(d_df2.loc[chain_idx == ch], quantiles=())["ess"].to_numpy()
    lo = (1.0 - level) / 2.0
    sd = d_df2.std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mcse = sd.to_numpy() / np.sqrt(ess)
    table = pd.DataFrame(
        {
            "mean": d_df2.mean(),
            "sd": sd,
            "mcse": mcse,
            "ess": ess,
            "lower": d_df2.quantile(lo),
            "median": d_df2.quantile(0.5),
            "upper": d_df2.quantile(1.0 - lo),
            "prob_positive": (d_df2 > 0).mean(),
        }
    )
    diag_info: Dict[str, Any] = {"warnings": [], "min_ess": float(table["ess"].min())}
    try:
        gr = gelman_rubin(
            [d_df2.loc[chain_idx == ch] for ch in range(chains)], split=True
        )
        diag_info["max_split_rhat"] = float(gr.table["psrf"].max())
    except (MethodIncompatibility, DataInsufficient):
        diag_info["max_split_rhat"] = float("nan")
    info: Dict[str, Any] = {
        "endogenous": endog,
        "instruments": znames[kx:],
        "n_instruments": int(n_inst),
        "first_stage_F": float(f_stat),
        "tsls": {nm: float(val) for nm, val in zip(wnames, theta_2sls)},
    }
    res = BayesRegressResult(
        model="iv",
        formula=formula,
        params=table["mean"].copy(),
        std_errors=table["sd"].copy(),
        table=table,
        draws=d_df2,
        chain=chain_idx,
        n_obs=n,
        n_draws=draws * chains,
        burnin=burnin,
        thin=thin,
        chains=chains,
        sampler="Gibbs (joint normal model of both equations)",
        acceptance_rate=None,
        prior={
            "coefficients": "normal",
            "prior_mean": prior_mean,
            "prior_var": prior_var,
            "first_stage_prior_var": first_stage_prior_var,
            "Sigma": f"InvWishart({nu0:g}, scale)",
            "sigma_scale": V0,
        },
        level=level,
        model_info=info,
        diagnostics_info=diag_info,
        _model=_IVShim(W, wnames),
        _design_info=None,
    )
    if f_stat < 10.0:
        text = (
            f"Weak instruments: the first-stage F statistic is {f_stat:.2f}. "
            "The posterior of the effect is then far from normal, its mean "
            "may not exist and it depends on the priors. Report the median "
            "and the interval, and how they move with prior_var."
        )
        diag_info["warnings"].append(text)
        warnings.warn(text, AssumptionWarning, stacklevel=2)
    msgs = []
    if diag_info["min_ess"] < 100:
        worst = str(table["ess"].idxmin())
        msgs.append(
            f"effective sample size is {diag_info['min_ess']:.0f} for '{worst}'"
        )
    rhat = diag_info["max_split_rhat"]
    if np.isfinite(rhat) and rhat > 1.05:
        msgs.append(f"split potential scale reduction factor is {rhat:.3f}")
    if msgs:
        text = (
            "The chain may not have converged or mixes slowly: "
            + "; ".join(msgs)
            + ". Increase draws / burnin or thin the chain."
        )
        diag_info["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    return res
