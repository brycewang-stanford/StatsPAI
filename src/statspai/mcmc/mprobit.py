"""
Multivariate and multinomial probit by Gibbs sampling.

Both models are a system of linear equations in latent normal variables
of which only signs (multivariate probit) or the largest element
(multinomial probit) are observed. The sampler draws the latent variables
from truncated normals and then treats them as the outcomes of a
seemingly unrelated regression.

The scale of the latent variables is not identified. The sampler runs on
the unrestricted error covariance, which keeps every step conjugate, and
the result reports the identified quantities: coefficients divided by the
latent standard deviation, and correlations or the covariance relative to
its first element. The prior is therefore stated for the unrestricted
parameters and only implied for the reported ones.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg

from ..core.utils import create_design_matrices
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._core import (
    check_mcmc_args,
    normal_prior,
    rinvwishart,
    rmvnorm_prec,
    rtruncnorm,
    spawn_rngs,
)
from ._results import posterior_table
from .regress import BayesRegressResult, _ordered_codes


def _draw_latent(
    rng: np.random.Generator,
    w: np.ndarray,
    mu: np.ndarray,
    Si: np.ndarray,
    y: np.ndarray,
    kind: str,
) -> np.ndarray:
    """One Gibbs pass over the columns of the latent matrix ``w``."""
    m = w.shape[1]
    for j in range(m):
        others = [q for q in range(m) if q != j]
        sd = 1.0 / np.sqrt(Si[j, j])
        mean = mu[:, j] - (w[:, others] - mu[:, others]) @ Si[j, others] / Si[j, j]
        if kind == "mv":
            lower = np.where(y[:, j] > 0, 0.0, -np.inf)
            upper = np.where(y[:, j] > 0, np.inf, 0.0)
        else:
            best_other = (
                w[:, others].max(axis=1) if others else np.full(len(w), -np.inf)
            )
            chosen = y == j + 1
            base = y == 0
            lower = np.where(chosen, np.maximum(best_other, 0.0), -np.inf)
            # another alternative was chosen: stay below its utility
            upper = np.where(chosen, np.inf, np.where(base, 0.0, best_other))
        w[:, j] = rtruncnorm(rng, mean, sd, lower, upper)
    return w


def _sweep(
    rng: np.random.Generator,
    w: np.ndarray,
    beta: np.ndarray,
    Sigma: np.ndarray,
    y: np.ndarray,
    Xl: List[np.ndarray],
    XX: List[List[np.ndarray]],
    off: np.ndarray,
    prior: Tuple[np.ndarray, np.ndarray, float, np.ndarray],
    kind: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latent variables, then coefficients, then the error covariance."""
    A, A_b0, nu0, V0 = prior
    m = len(Xl)
    n = w.shape[0]
    Si = linalg.inv(Sigma)
    mu = np.column_stack([Xl[j] @ beta[off[j] : off[j + 1]] for j in range(m)])
    w = _draw_latent(rng, w, mu, Si, y, kind)
    P = A.copy()
    rhs = A_b0.copy()
    for a in range(m):
        for b in range(m):
            P[off[a] : off[a + 1], off[b] : off[b + 1]] += Si[a, b] * XX[a][b]
            rhs[off[a] : off[a + 1]] += Si[a, b] * (Xl[a].T @ w[:, b])
    beta, _ = rmvnorm_prec(rng, rhs, 0.5 * (P + P.T))
    E = w - np.column_stack([Xl[j] @ beta[off[j] : off[j + 1]] for j in range(m)])
    Sigma = rinvwishart(rng, nu0 + n, V0 + E.T @ E)
    return w, beta, Sigma


def _sigma_prior(
    sigma_prior: Optional[Tuple[float, Any]], m: int
) -> Tuple[float, np.ndarray]:
    if sigma_prior is None:
        return float(m + 3), np.eye(m) * float(m + 3)
    nu0 = float(sigma_prior[0])
    sc = np.asarray(sigma_prior[1], dtype=float)
    V0 = np.eye(m) * float(sc) if sc.ndim == 0 else 0.5 * (sc + sc.T)
    if V0.shape != (m, m) or nu0 <= m - 1:
        raise MethodIncompatibility(
            f"sigma_prior must be (df, scale) with df > {m - 1} and scale a "
            f"positive number or an {m} x {m} matrix."
        )
    return nu0, V0


def _run(
    kind: str,
    y: np.ndarray,
    Xl: List[np.ndarray],
    names: List[str],
    labels: List[str],
    prior_mean: Any,
    prior_var: Any,
    sigma_prior: Optional[Tuple[float, Any]],
    draws: int,
    burnin: int,
    thin: int,
    chains: int,
    seed: Optional[int],
    level: float,
    formula: str,
    info: Dict[str, Any],
) -> BayesRegressResult:
    m = len(Xl)
    n = Xl[0].shape[0]
    ks = [X.shape[1] for X in Xl]
    K = int(sum(ks))
    off = np.concatenate([[0], np.cumsum(ks)])
    b0, _, A = normal_prior(K, prior_mean, prior_var, names)
    nu0, V0 = _sigma_prior(sigma_prior, m)
    prior = (A, A @ b0, nu0, V0)
    XX = [[Xl[a].T @ Xl[b] for b in range(m)] for a in range(m)]
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m)]
    if kind == "mv":
        aux = [f"corr({labels[a]},{labels[b]})" for a, b in pairs]
    else:
        aux = [f"var({labels[a]})" for a in range(1, m)] + [
            f"cov({labels[a]},{labels[b]})" for a, b in pairs
        ]
    n_iter = burnin + draws * thin
    rngs = spawn_rngs(seed, chains)
    pieces = []
    for ch in range(chains):
        rng = rngs[ch]
        if kind == "mv":
            w = np.where(y > 0, 0.5, -0.5).astype(float)
        else:
            w = np.full((n, m), -1.0)
            hit = y > 0
            w[np.where(hit)[0], y[hit] - 1] = 1.0
        beta = np.zeros(K)
        Sigma = np.eye(m)
        if chains > 1:
            beta = 0.3 * rng.standard_normal(K)
        out = np.empty((draws, K + len(aux)))
        kept = 0
        for it in range(n_iter):
            w, beta, Sigma = _sweep(rng, w, beta, Sigma, y, Xl, XX, off, prior, kind)
            if it >= burnin and (it - burnin) % thin == 0:
                if kind == "mv":
                    sd = np.sqrt(np.diag(Sigma))
                    out[kept, :K] = beta / np.repeat(sd, ks)
                    out[kept, K:] = [Sigma[a, b] / (sd[a] * sd[b]) for a, b in pairs]
                else:
                    s11 = Sigma[0, 0]
                    out[kept, :K] = beta / np.sqrt(s11)
                    out[kept, K : K + m - 1] = np.diag(Sigma)[1:] / s11
                    out[kept, K + m - 1 :] = [Sigma[a, b] / s11 for a, b in pairs]
                kept += 1
        pieces.append(out)
    d_df = pd.DataFrame(np.vstack(pieces), columns=names + aux)
    chain_idx = np.repeat(np.arange(chains), draws)
    table, diag = posterior_table(d_df, chain_idx, chains, level)
    res = BayesRegressResult(
        model="mvprobit" if kind == "mv" else "mnprobit",
        formula=formula,
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
        sampler="Gibbs with data augmentation, unrestricted covariance",
        acceptance_rate=None,
        prior={
            "coefficients": "normal, on the unrestricted latent scale",
            "prior_mean": prior_mean,
            "prior_var": prior_var,
            "Sigma": f"InvWishart({nu0:g}, scale)",
            "sigma_scale": V0,
        },
        level=level,
        model_info=info,
        diagnostics_info=diag,
        _model=None,
    )
    if diag["min_ess"] < 100:
        text = (
            "The chain mixes slowly: effective sample size "
            f"{diag['min_ess']:.0f} for '{table['ess'].idxmin()}'. Increase "
            "draws; latent-variable samplers of probit systems are slow."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=3)
    return res


def bayes_mvprobit(
    formulas: Sequence[str],
    data: pd.DataFrame,
    prior_mean: Any = 0.0,
    prior_var: Any = 100.0,
    sigma_prior: Optional[Tuple[float, Any]] = None,
    draws: int = 10000,
    burnin: int = 2000,
    thin: int = 1,
    chains: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> BayesRegressResult:
    """Bayesian multivariate probit.

    Several binary outcomes, one probit equation each, with errors that
    are correlated across equations. The correlations say whether the
    outcomes move together beyond what the regressors explain.

    Parameters
    ----------
    formulas : list of str
        One formula per binary outcome, ``["y1 ~ x", "y2 ~ x + z"]``.
        Outcomes must be coded 0 and 1.
    data : DataFrame
    prior_mean, prior_var : float or array, default 0 and 100
        Normal prior of the stacked coefficients on the unrestricted
        latent scale.
    sigma_prior : (df, scale), optional
        Inverse-Wishart prior of the unrestricted error covariance.
        Default ``(M + 3, (M + 3) I)``, centred on unit variances and no
        correlation.
    draws, burnin, thin, chains, seed, level
        As in :func:`statspai.bayes_regress`.

    Returns
    -------
    BayesRegressResult
        ``model='mvprobit'``. Parameters ``<outcome>:<term>`` are probit
        coefficients (error variance one) and ``corr(a,b)`` the error
        correlations.

    Notes
    -----
    The sampler does not restrict the error covariance to a correlation
    matrix. Coefficients and covariance are drawn on an arbitrary scale
    and each draw is normalised afterwards, which is the approach of
    Rossi, Allenby and McCulloch. The prior on the reported quantities is
    the one implied by the stated prior.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> e = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
    >>> df = pd.DataFrame({"x": x, "y1": (0.5 * x + e[:, 0] > 0) * 1,
    ...                    "y2": (-0.5 * x + e[:, 1] > 0) * 1})
    >>> fit = sp.bayes_mvprobit(["y1 ~ x", "y2 ~ x"], df, draws=500, burnin=200, seed=1)
    >>> list(fit.params.index)
    ['y1:Intercept', 'y1:x', 'y2:Intercept', 'y2:x', 'corr(y1,y2)']

    References
    ----------
    rossi2005bayesian, albert1993bayesian
    """
    check_mcmc_args(draws, burnin, thin, chains)
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    if isinstance(formulas, str) or len(formulas) < 2:
        raise MethodIncompatibility(
            "formulas must be a list of at least two formulas; for one "
            "binary outcome use sp.bayes_regress(model='probit')."
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
    if not np.isin(Y, (0.0, 1.0)).all():
        raise MethodIncompatibility("Every outcome must be coded 0 and 1.")
    for j, yn in enumerate(ynames):
        if Y[:, j].min() == Y[:, j].max():
            raise DataInsufficient(f"Outcome {yn} does not vary.")
    Xl = [np.asarray(X.loc[idx], dtype=float) for X in Xs]
    n = Y.shape[0]
    K = sum(X.shape[1] for X in Xl)
    if n <= K:
        raise DataInsufficient(f"{n} complete observations for {K} coefficients.")
    for yn, X in zip(ynames, Xl):
        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise MethodIncompatibility(f"The regressors of {yn} are collinear.")
    names = [f"{yn}:{c}" for yn, X in zip(ynames, Xs) for c in X.columns]
    return _run(
        "mv",
        Y,
        Xl,
        names,
        ynames,
        prior_mean,
        prior_var,
        sigma_prior,
        draws,
        burnin,
        thin,
        chains,
        seed,
        level,
        " ; ".join(formulas),
        {"equations": list(formulas), "outcomes": ynames},
    )


def bayes_mnprobit(
    formula: str,
    data: pd.DataFrame,
    prior_mean: Any = 0.0,
    prior_var: Any = 100.0,
    sigma_prior: Optional[Tuple[float, Any]] = None,
    draws: int = 10000,
    burnin: int = 2000,
    thin: int = 1,
    chains: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> BayesRegressResult:
    """Bayesian multinomial probit.

    An unordered outcome with three or more categories. Each category has
    a latent utility that is linear in the regressors with normal errors,
    correlated across categories, and the category with the largest
    utility is observed. Unlike the multinomial logit the model does not
    impose independence of irrelevant alternatives.

    Parameters
    ----------
    formula : str
        ``"choice ~ x1 + x2"``. The regressors describe the decision
        maker; every non-base category gets its own coefficients.
    data : DataFrame
    prior_mean, prior_var : float or array, default 0 and 100
        Normal prior of the stacked coefficients on the unrestricted
        latent scale.
    sigma_prior : (df, scale), optional
        Inverse-Wishart prior of the covariance of the utility
        differences. Default ``(J + 2, (J + 2) I)`` for ``J`` categories.
    draws, burnin, thin, chains, seed, level
        As in :func:`statspai.bayes_regress`.

    Returns
    -------
    BayesRegressResult
        ``model='mnprobit'``. The first level of the outcome is the base.
        Parameters ``<level>:<term>`` are coefficients of the utility of
        that level relative to the base, in units of the standard
        deviation of the first utility difference. ``var(level)`` and
        ``cov(a,b)`` are the covariance of the utility differences
        relative to its first element.

    Notes
    -----
    Utilities are identified up to location and scale. Location is fixed
    by differencing against the base and scale by dividing each draw by
    the standard deviation of the first difference, the approach of
    McCulloch and Rossi (1994). The error covariance is weakly identified
    in most data: expect wide posteriors for it and a slow chain.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> u = np.column_stack([np.zeros(n), 0.8 * x, -0.5 * x]) + rng.normal(size=(n, 3))
    >>> df = pd.DataFrame({"x": x, "choice": np.array(["a", "b", "c"])[u.argmax(1)]})
    >>> fit = sp.bayes_mnprobit("choice ~ x", df, draws=500, burnin=200, seed=1)
    >>> list(fit.params.index)[:4]
    ['b:Intercept', 'b:x', 'c:Intercept', 'c:x']

    References
    ----------
    mcculloch1994exact, rossi2005bayesian
    """
    check_mcmc_args(draws, burnin, thin, chains)
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    if not isinstance(formula, str) or "~" not in formula:
        raise MethodIncompatibility(
            f"formula must look like 'choice ~ x1 + x2'; got {formula!r}."
        )
    lhs = formula.split("~", 1)[0].strip()
    if lhs not in data.columns:
        raise MethodIncompatibility(
            f"The left-hand side must be a column of data; {lhs!r} is not."
        )
    work = data.loc[data[lhs].notna()].copy()
    codes, levels = _ordered_codes(work[lhs])
    work[lhs] = codes
    y_df, X_df = create_design_matrices(formula, work)
    y = np.asarray(y_df, dtype=float).reshape(-1).astype(int)
    X = np.asarray(X_df, dtype=float)
    J = len(levels)
    if J < 3:
        raise MethodIncompatibility(
            f"The outcome has {J} categories; a multinomial probit needs at "
            "least three. Use sp.bayes_regress(model='probit')."
        )
    if (np.bincount(y, minlength=J) == 0).any():
        raise DataInsufficient("An outcome category has no complete observations.")
    n, k = X.shape
    if n <= k * (J - 1):
        raise DataInsufficient(f"{n} observations for {k * (J - 1)} coefficients.")
    if np.linalg.matrix_rank(X) < k:
        raise MethodIncompatibility("The regressors are collinear.")
    labels = [str(v) for v in levels[1:]]
    names = [f"{lv}:{c}" for lv in labels for c in X_df.columns]
    return _run(
        "mn",
        y,
        [X] * (J - 1),
        names,
        labels,
        prior_mean,
        prior_var,
        sigma_prior,
        draws,
        burnin,
        thin,
        chains,
        seed,
        level,
        formula,
        {"levels": [str(v) for v in levels], "base_level": str(levels[0])},
    )
