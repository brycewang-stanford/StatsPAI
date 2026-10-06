"""
Bayesian seemingly unrelated regressions by Gibbs sampling: ``sp.bayes_sur``.

``y_m = X_m beta_m + e_m`` for equations ``m = 1 .. M`` with errors
correlated across equations, ``(e_1i, ..., e_Mi) ~ N(0, Sigma)``. Normal
prior on the stacked coefficients, inverse-Wishart prior on ``Sigma``;
both full conditionals are standard (Zellner 1962 for the model; the
Gibbs sampler as in Rossi, Allenby and McCulloch 2005, section 3.5). With
the same regressors in every equation it is the multivariate regression
model.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg

from ..core.utils import create_design_matrices
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._core import check_mcmc_args, normal_prior, rinvwishart, rmvnorm_prec, spawn_rngs
from .diagnostics import mcmc_summary
from .regress import BayesRegressResult


class _SURShim:
    def __init__(self, X: np.ndarray, names: List[str]):
        self.X = X
        self.k = X.shape[1]
        self.xnames = names


def bayes_sur(
    formulas: Sequence[str],
    data: pd.DataFrame,
    prior_mean: Any = 0.0,
    prior_var: Any = 1000.0,
    sigma_prior: Optional[Tuple[float, Any]] = None,
    draws: int = 10000,
    burnin: int = 2000,
    thin: int = 1,
    chains: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> BayesRegressResult:
    """Bayesian seemingly unrelated regressions.

    Several linear equations estimated jointly because their errors are
    correlated. The posterior of each coefficient uses the information in
    the other equations' errors, as feasible GLS does, and carries the
    uncertainty of the error covariance.

    Parameters
    ----------
    formulas : list of str
        One formula per equation, ``["y1 ~ x1 + x2", "y2 ~ x1 + z"]``.
        The same regressors in every equation give the multivariate
        regression model.
    data : DataFrame
        Rows with a missing value in any equation are dropped from all.
    prior_mean, prior_var
        Normal prior of the stacked coefficients (equation by equation,
        in the order of ``formulas``): scalars, a vector or a matrix.
    sigma_prior : (df, scale), optional
        ``Sigma ~ InvWishart(df, scale)``, ``scale`` a number (times the
        identity) or an ``M x M`` matrix. Default ``(M + 1, 0.02)``, weak
        on any ordinary scale; R ``bayesm::rsurGibbs`` uses an identity
        scale.
    draws, burnin, thin, chains, seed, level
        As in :func:`statspai.bayes_regress`.

    Returns
    -------
    BayesRegressResult
        ``model='sur'``. Parameters ``<outcome>:<term>`` for the
        coefficients, then ``var(<outcome>)``, ``cov(<a>,<b>)`` and
        ``corr(<a>,<b>)`` for the error covariance.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> e = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], size=n)
    >>> df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    >>> df["y1"] = 1 + 0.5 * df["x"] + e[:, 0]
    >>> df["y2"] = -1 + 0.8 * df["z"] + e[:, 1]
    >>> fit = sp.bayes_sur(["y1 ~ x", "y2 ~ z"], df, draws=1000, burnin=300, seed=1)
    >>> list(fit.params.index)[:4]
    ['y1:Intercept', 'y1:x', 'y2:Intercept', 'y2:z']
    >>> bool(fit.prob("`corr(y1,y2)` > 0") > 0.99)
    True

    References
    ----------
    zellner1962efficient, rossi2005bayesian
    """
    check_mcmc_args(draws, burnin, thin, chains)
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    if isinstance(formulas, str) or len(formulas) < 2:
        raise MethodIncompatibility(
            "formulas must be a list of at least two formulas; for one "
            "equation use sp.bayes_regress."
        )
    ys: List[pd.Series] = []
    Xs: List[pd.DataFrame] = []
    for f in formulas:
        y_df, X_df = create_design_matrices(f, data)
        ys.append(
            pd.Series(np.asarray(y_df, dtype=float).reshape(-1), index=X_df.index)
        )
        Xs.append(X_df)
    idx = Xs[0].index
    for X_df in Xs[1:]:
        idx = idx.intersection(X_df.index)
    ynames = [f.split("~", 1)[0].strip() for f in formulas]
    if len(set(ynames)) != len(ynames):
        raise MethodIncompatibility("Each equation needs a different outcome.")
    Y = np.column_stack([y.loc[idx].to_numpy() for y in ys])
    Xl = [np.asarray(X.loc[idx], dtype=float) for X in Xs]
    n, M = Y.shape
    ks = [X.shape[1] for X in Xl]
    K = int(sum(ks))
    names = [f"{yn}:{c}" for yn, X in zip(ynames, Xs) for c in X.columns]
    if n <= max(ks) + M:
        raise DataInsufficient(f"{n} complete observations for {K} coefficients.")
    for yn, X in zip(ynames, Xl):
        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise MethodIncompatibility(f"The regressors of {yn} are collinear.")
    b0, _, A = normal_prior(K, prior_mean, prior_var, names)
    if sigma_prior is None:
        nu0, V0 = float(M + 1), np.eye(M) * 0.02
    else:
        nu0 = float(sigma_prior[0])
        sc = np.asarray(sigma_prior[1], dtype=float)
        V0 = np.eye(M) * float(sc) if sc.ndim == 0 else 0.5 * (sc + sc.T)
        if V0.shape != (M, M) or nu0 <= M - 1:
            raise MethodIncompatibility(
                f"sigma_prior must be (df, scale) with df > {M - 1} and scale "
                f"a positive number or an {M} x {M} matrix."
            )
    off = np.concatenate([[0], np.cumsum(ks)])
    # cross-products X_m'X_l and X_m'y_l
    XX = [[Xl[m].T @ Xl[q] for q in range(M)] for m in range(M)]
    Xy = [[Xl[m].T @ Y[:, q] for q in range(M)] for m in range(M)]
    A_b0 = A @ b0
    aux = (
        [f"var({a})" for a in ynames]
        + [f"cov({ynames[a]},{ynames[b]})" for a in range(M) for b in range(a + 1, M)]
        + [f"corr({ynames[a]},{ynames[b]})" for a in range(M) for b in range(a + 1, M)]
    )
    pairs = [(a, b) for a in range(M) for b in range(a + 1, M)]
    n_iter = burnin + draws * thin
    rngs = spawn_rngs(seed, chains)
    pieces = []
    for ch in range(chains):
        rng = rngs[ch]
        beta = np.concatenate(
            [np.linalg.lstsq(Xl[m], Y[:, m], rcond=None)[0] for m in range(M)]
        )
        if chains > 1:
            beta = beta + 0.1 * np.abs(beta).max() * rng.standard_normal(K)
        out = np.empty((draws, K + len(aux)))
        kept = 0
        for it in range(n_iter):
            E = np.column_stack(
                [Y[:, m] - Xl[m] @ beta[off[m] : off[m + 1]] for m in range(M)]
            )
            Sigma = rinvwishart(rng, nu0 + n, V0 + E.T @ E)
            Si = linalg.inv(Sigma)
            P = A.copy()
            rhs = A_b0.copy()
            for m in range(M):
                for q in range(M):
                    P[off[m] : off[m + 1], off[q] : off[q + 1]] += Si[m, q] * XX[m][q]
                    rhs[off[m] : off[m + 1]] += Si[m, q] * Xy[m][q]
            beta, _ = rmvnorm_prec(rng, rhs, 0.5 * (P + P.T))
            if it >= burnin and (it - burnin) % thin == 0:
                out[kept, :K] = beta
                sd = np.sqrt(np.diag(Sigma))
                out[kept, K : K + M] = np.diag(Sigma)
                out[kept, K + M : K + M + len(pairs)] = [Sigma[a, b] for a, b in pairs]
                out[kept, K + M + len(pairs) :] = [
                    Sigma[a, b] / (sd[a] * sd[b]) for a, b in pairs
                ]
                kept += 1
        pieces.append(out)
    d_df = pd.DataFrame(np.vstack(pieces), columns=names + aux)
    chain_idx = np.repeat(np.arange(chains), draws)
    ess = np.zeros(d_df.shape[1])
    for ch in range(chains):
        ess += mcmc_summary(d_df.loc[chain_idx == ch], quantiles=())["ess"].to_numpy()
    lo = (1.0 - level) / 2.0
    sd_all = d_df.std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mcse = sd_all.to_numpy() / np.sqrt(ess)
    table = pd.DataFrame(
        {
            "mean": d_df.mean(),
            "sd": sd_all,
            "mcse": mcse,
            "ess": ess,
            "lower": d_df.quantile(lo),
            "median": d_df.quantile(0.5),
            "upper": d_df.quantile(1.0 - lo),
            "prob_positive": (d_df > 0).mean(),
        }
    )
    diag: Dict[str, Any] = {"warnings": [], "min_ess": float(table["ess"].min())}
    res = BayesRegressResult(
        model="sur",
        formula=" ; ".join(formulas),
        params=table["mean"].copy(),
        std_errors=table["sd"].copy(),
        table=table,
        draws=d_df,
        chain=chain_idx,
        n_obs=n,
        n_draws=draws * chains,
        burnin=burnin,
        thin=thin,
        chains=chains,
        sampler="Gibbs",
        acceptance_rate=None,
        prior={
            "coefficients": "normal",
            "prior_mean": prior_mean,
            "prior_var": prior_var,
            "Sigma": f"InvWishart({nu0:g}, scale)",
            "sigma_scale": V0,
        },
        level=level,
        model_info={"equations": list(formulas), "outcomes": ynames},
        diagnostics_info=diag,
        _model=None,
    )
    if diag["min_ess"] < 100:
        text = (
            "The chain mixes slowly: effective sample size "
            f"{diag['min_ess']:.0f}. Increase draws."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    return res
