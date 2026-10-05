"""
Bayesian hierarchical (random-effects) models for longitudinal data:
``sp.bayes_mixed``.

``y_it = x_it' beta + w_it' b_i + e_it`` with ``b_i ~ N(0, D)``, a normal
prior on ``beta`` and an inverse-Wishart prior on ``D``. Gaussian outcomes
use the blocked Gibbs sampler of Chib and Carlin (1999), which draws
``beta`` with the random effects integrated out. Binary (logit) and count
(Poisson) outcomes use Metropolis steps inside the same scheme.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, special

from ..core.utils import create_design_matrices
from ..exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    StatsPAIWarning,
)
from ._core import (
    check_mcmc_args,
    find_mode,
    normal_prior,
    rinvgamma,
    rinvwishart,
    rmvnorm_prec,
    spawn_rngs,
)
from .diagnostics import gelman_rubin, mcmc_summary
from .regress import BayesRegressResult

_FAMILIES = ("normal", "logit", "poisson")
_ALIASES = {
    "gaussian": "normal",
    "linear": "normal",
    "binomial": "logit",
    "logistic": "logit",
    "count": "poisson",
}


class _MixedShim:
    """Model interface for ``predict`` (fixed part of the model)."""

    def __init__(self, X: np.ndarray, xnames: List[str], family: str):
        self.X = X
        self.k = X.shape[1]
        self.xnames = xnames
        self.family = family

    def linear_predictor(self, draws: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.asarray(draws[:, : self.k] @ X.T)

    def expected_value(self, eta: np.ndarray) -> np.ndarray:
        if self.family == "logit":
            return np.asarray(special.expit(eta))
        if self.family == "poisson":
            return np.asarray(np.exp(eta))
        return eta


@dataclass
class BayesMixedResult(BayesRegressResult):
    """Posterior of a hierarchical model fitted by :func:`bayes_mixed`.

    Everything of :class:`BayesRegressResult`, plus:

    Attributes
    ----------
    random_effects : pd.DataFrame
        Posterior mean and standard deviation of every group's random
        effects (columns ``<term>`` and ``<term>_sd``), indexed by group.
    re_cov : pd.DataFrame
        Posterior mean of the covariance matrix ``D`` of the random
        effects.
    n_groups : int

    Notes
    -----
    ``predict`` returns the fixed part ``x' beta`` (on the outcome scale
    of the family at a random effect of zero); add ``random_effects`` for
    a group-specific prediction.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(30), 6)
    >>> df = pd.DataFrame({"id": g, "x": rng.normal(size=180)})
    >>> df["y"] = 1 + 0.5 * df["x"] + rng.normal(size=30)[g] + rng.normal(size=180)
    >>> fit = sp.bayes_mixed("y ~ x", df, group="id", draws=500, burnin=200, seed=1)
    >>> isinstance(fit, sp.BayesMixedResult)
    True
    >>> list(fit.params.index)
    ['Intercept', 'x', 'sigma2', 'var(Intercept)']
    >>> fit.random_effects.shape
    (30, 2)
    """

    random_effects: pd.DataFrame = field(default_factory=pd.DataFrame)
    re_cov: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_groups: int = 0

    _citation_keys = ("chib1999mcmc",)

    def cite(self, format: str = "keys") -> Any:  # noqa: A002
        """The verified ``paper.bib`` key of the sampler."""
        if format == "json":
            return {
                "citation_keys": list(self._citation_keys),
                "source": "paper.bib",
                "resolve_with": "sp.bibtex(keys=[...])",
            }
        if format != "keys":
            raise MethodIncompatibility(
                f"format must be 'keys' or 'json'; got {format!r}"
            )
        return "\n".join(self._citation_keys)

    def log_marginal_likelihood(
        self, method: Optional[str] = None, alpha: float = 0.05
    ) -> float:
        raise MethodIncompatibility(
            "A marginal likelihood is not implemented for hierarchical "
            "models: it requires integrating the random effects out of "
            "every posterior ordinate."
        )

    def marginal_likelihood_details(
        self, method: Optional[str] = None, alpha: float = 0.05
    ) -> Dict[str, Any]:
        self.log_marginal_likelihood()
        return {}  # pragma: no cover

    def summary(self) -> str:
        base = (
            super()
            .summary()
            .replace(
                f"Bayesian {self.model} regression",
                f"Bayesian hierarchical {self.model} model",
                1,
            )
        )
        sizes = self.model_info.get("group_sizes", {})
        extra = (
            f"Groups: {self.n_groups}"
            f"    Observations per group: min {sizes.get('min')}, "
            f"mean {sizes.get('mean'):.1f}, max {sizes.get('max')}"
        )
        lines = base.split("\n")
        lines.insert(2, extra)
        return "\n".join(lines)

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        out = super().to_dict(*args, **kwargs)
        out["n_groups"] = int(self.n_groups)
        out["re_cov"] = {
            str(r): {str(c): float(v) for c, v in row.items()}
            for r, row in self.re_cov.iterrows()
        }
        return out


def _re_names(wnames: Sequence[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Names and (row, col) positions of the reported elements of D."""
    names, pos = [], []
    q = len(wnames)
    for a in range(q):
        names.append(f"var({wnames[a]})")
        pos.append((a, a))
    for a in range(q):
        for b in range(a + 1, q):
            names.append(f"cov({wnames[a]},{wnames[b]})")
            pos.append((a, b))
    return names, pos


class _Running:
    """Running mean and variance of the random effects over kept draws."""

    def __init__(self, shape: Tuple[int, int]):
        self.n = 0
        self.s1 = np.zeros(shape)
        self.s2 = np.zeros(shape)

    def add(self, b: np.ndarray) -> None:
        self.n += 1
        self.s1 += b
        self.s2 += b * b

    def merge(self, other: "_Running") -> None:
        self.n += other.n
        self.s1 += other.s1
        self.s2 += other.s2

    def moments(self) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.s1 / self.n
        var = np.clip(self.s2 / self.n - mean * mean, 0.0, None)
        return mean, np.sqrt(var * self.n / max(self.n - 1, 1))


def _sample_normal(
    rng: np.random.Generator,
    y: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    gidx: np.ndarray,
    n_groups: int,
    prior: Dict[str, Any],
    n_iter: int,
    burnin: int,
    thin: int,
    jitter: bool,
) -> Tuple[np.ndarray, _Running, Optional[float]]:
    n, k = X.shape
    q = W.shape[1]
    b0, B0inv = prior["b0"], prior["B0inv"]
    a0, d0 = prior["a0"], prior["d0"]
    r0, R0 = prior["r0"], prior["R0"]
    # per-group sufficient statistics
    order = np.argsort(gidx, kind="stable")
    starts = np.searchsorted(gidx[order], np.arange(n_groups))

    def gsum(arr: np.ndarray) -> np.ndarray:
        return np.add.reduceat(arr[order], starts, axis=0)

    WtW = gsum(W[:, :, None] * W[:, None, :])  # N x q x q
    WtX = gsum(W[:, :, None] * X[:, None, :])  # N x q x k
    Wty = gsum(W * y[:, None])  # N x q
    XtX = X.T @ X
    Xty = X.T @ y
    B0inv_b0 = B0inv @ b0

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    s2 = float(e @ e / max(n - k, 1))
    D = np.eye(q) * max(s2, 1e-8)
    if jitter:
        s2 *= float(np.exp(rng.normal(scale=0.5)))
        D = D * float(np.exp(rng.normal(scale=0.5)))
    n_keep = (n_iter - burnin) // thin
    npar = k + 1 + q * (q + 1) // 2
    out = np.empty((n_keep, npar))
    running = _Running((n_groups, q))
    _, pos = _re_names([""] * q)
    kept = 0
    for it in range(n_iter):
        Dinv = linalg.inv(D)
        M = s2 * Dinv[None, :, :] + WtW  # N x q x q
        rhs = np.concatenate([WtX, Wty[:, :, None]], axis=2)
        sol = np.linalg.solve(M, rhs)  # N x q x (k + 1)
        A, a = sol[:, :, :k], sol[:, :, k]
        # beta with the random effects integrated out
        XVX = (XtX - np.einsum("gqk,gql->kl", WtX, A)) / s2
        XVy = (Xty - np.einsum("gqk,gq->k", WtX, a)) / s2
        beta, _ = rmvnorm_prec(rng, B0inv_b0 + XVy, B0inv + XVX)
        # random effects
        mean_b = a - A @ beta
        chol = np.linalg.cholesky(M)
        z = rng.standard_normal((n_groups, q, 1))
        dev = np.linalg.solve(np.swapaxes(chol, 1, 2), z)[:, :, 0]
        b = mean_b + np.sqrt(s2) * dev
        # error variance
        resid = y - X @ beta - np.einsum("nq,nq->n", W, b[gidx])
        s2 = rinvgamma(rng, (a0 + n) / 2.0, (d0 + float(resid @ resid)) / 2.0)
        # covariance of the random effects
        D = rinvwishart(rng, r0 + n_groups, r0 * R0 + b.T @ b)
        if it >= burnin and (it - burnin) % thin == 0:
            out[kept, :k] = beta
            out[kept, k] = s2
            out[kept, k + 1 :] = [D[i, j] for i, j in pos]
            running.add(b)
            kept += 1
    return out, running, None


def _glm_loglik_obs(y: np.ndarray, eta: np.ndarray, family: str) -> np.ndarray:
    if family == "logit":
        return np.asarray(y * eta - np.logaddexp(0.0, eta))
    with np.errstate(over="ignore"):
        return np.asarray(y * eta - np.exp(eta))


def _sample_glmm(
    rng: np.random.Generator,
    y: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    gidx: np.ndarray,
    n_groups: int,
    prior: Dict[str, Any],
    family: str,
    match: List[Tuple[int, int]],
    n_iter: int,
    burnin: int,
    thin: int,
    jitter: bool,
) -> Tuple[np.ndarray, _Running, Optional[float]]:
    n, k = X.shape
    q = W.shape[1]
    b0, B0inv = prior["b0"], prior["B0inv"]
    r0, R0 = prior["r0"], prior["R0"]

    def neg_pooled(bv: np.ndarray) -> float:
        dev = bv - b0
        val = _glm_loglik_obs(y, X @ bv, family).sum() - 0.5 * dev @ B0inv @ dev
        return float(-val) if np.isfinite(val) else np.inf

    start = np.zeros(k)
    const = np.where(np.ptp(X, axis=0) == 0)[0]
    if family == "poisson" and const.size:
        start[const[0]] = np.log(max(y.mean(), 1e-8)) / X[0, const[0]]
    beta, cov_beta = find_mode(neg_pooled, start, what=f"pooled {family} model")
    if jitter:
        beta = rng.multivariate_normal(beta, 4.0 * cov_beta, method="svd")
    chol_beta = linalg.cholesky(cov_beta, lower=True)
    scale_beta = 2.38 / np.sqrt(k)
    b = np.zeros((n_groups, q))
    D = np.eye(q) * 0.5
    scale_b = np.full(n_groups, 1.0)
    acc_b = np.zeros(n_groups)
    acc_beta = 0
    acc_beta_kept = 0
    n_keep = (n_iter - burnin) // thin
    npar = k + q * (q + 1) // 2
    out = np.empty((n_keep, npar))
    running = _Running((n_groups, q))
    _, pos = _re_names([""] * q)
    kept = 0
    wb = np.einsum("nq,nq->n", W, b[gidx])
    eta_fix = X @ beta
    # columns of X that carry the same variable as a random-effect column
    E = np.zeros((k, q))
    for xi, wi in match:
        E[xi, wi] = 1.0
    use = [wi for _, wi in match]

    def group_ll(eta: np.ndarray) -> np.ndarray:
        return np.bincount(
            gidx, weights=_glm_loglik_obs(y, eta, family), minlength=n_groups
        )

    for it in range(n_iter):
        # ---- random effects, all groups at once ---------------------------
        Dinv = linalg.inv(D)
        chol_D = linalg.cholesky(D, lower=True)
        step = (rng.standard_normal((n_groups, q)) @ chol_D.T) * scale_b[:, None]
        cand = b + step
        wb_c = np.einsum("nq,nq->n", W, cand[gidx])
        ll_c = group_ll(eta_fix + wb_c)
        ll_o = group_ll(eta_fix + wb)
        lp_c = -0.5 * np.einsum("gq,qr,gr->g", cand, Dinv, cand)
        lp_o = -0.5 * np.einsum("gq,qr,gr->g", b, Dinv, b)
        accept = np.log(rng.random(n_groups)) < (ll_c + lp_c) - (ll_o + lp_o)
        b = np.where(accept[:, None], cand, b)
        wb = np.where(accept[gidx], wb_c, wb)
        acc_b += accept
        # ---- fixed effects -------------------------------------------------
        cand_beta = beta + scale_beta * (chol_beta @ rng.standard_normal(k))
        eta_c = X @ cand_beta
        dev_c, dev_o = cand_beta - b0, beta - b0
        lr = (
            _glm_loglik_obs(y, eta_c + wb, family).sum()
            - _glm_loglik_obs(y, eta_fix + wb, family).sum()
            - 0.5 * dev_c @ B0inv @ dev_c
            + 0.5 * dev_o @ B0inv @ dev_o
        )
        took = bool(np.log(rng.random()) < lr)
        if took:
            beta, eta_fix = cand_beta, eta_c
            acc_beta += 1
        # ---- location shift between beta and the random effects -----------
        # beta_W + delta and b_i - delta leave the likelihood unchanged, so
        # delta has a normal full conditional
        if use:
            Du = Dinv[np.ix_(use, use)]
            Eu = E[:, use]
            P = n_groups * Du + Eu.T @ B0inv @ Eu
            # cross terms with the random-effect columns that do not shift
            r = (Dinv[use, :] @ b.sum(axis=0)) - Eu.T @ (B0inv @ (beta - b0))
            delta, _ = rmvnorm_prec(rng, r, P)
            beta = beta + Eu @ delta
            b[:, use] -= delta
            eta_fix = X @ beta
            wb = np.einsum("nq,nq->n", W, b[gidx])
        # ---- covariance of the random effects -------------------------------
        D = rinvwishart(rng, r0 + n_groups, r0 * R0 + b.T @ b)
        # ---- tune the proposals during burn-in ------------------------------
        if it < burnin and (it + 1) % 50 == 0:
            rate = acc_b / 50.0
            scale_b *= np.exp(np.clip(rate - 0.35, -0.3, 0.3))
            acc_b[:] = 0
            scale_beta *= float(np.exp(np.clip(acc_beta / 50.0 - 0.25, -0.3, 0.3)))
            acc_beta = 0
        if it >= burnin:
            acc_beta_kept += took
            if (it - burnin) % thin == 0:
                out[kept, :k] = beta
                out[kept, k:] = [D[i, j] for i, j in pos]
                running.add(b)
                kept += 1
    return out, running, acc_beta_kept / max(n_iter - burnin, 1)


def bayes_mixed(
    formula: str,
    data: pd.DataFrame,
    group: str,
    random: Optional[Sequence[str]] = None,
    family: str = "normal",
    random_intercept: bool = True,
    prior_mean: Any = 0.0,
    prior_var: Any = None,
    sigma2_prior: Tuple[float, float] = (0.001, 0.001),
    re_prior: Optional[Tuple[float, Any]] = None,
    draws: int = 10000,
    burnin: int = 2000,
    thin: int = 1,
    chains: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> BayesMixedResult:
    """Bayesian hierarchical model for longitudinal / panel data.

    ``y_it = x_it' beta + w_it' b_i + e_it``, where the random effects of
    group ``i`` are ``b_i ~ N(0, D)``. The fixed effects have a normal
    prior and ``D`` an inverse-Wishart prior. The random effects are
    assumed independent of the regressors (a random-effects model, not a
    fixed-effects one).

    Parameters
    ----------
    formula : str
        Fixed part, ``'y ~ x1 + x2'``.
    data : DataFrame
    group : str
        Column identifying the groups (units of the panel).
    random : list of str, optional
        Columns with a random slope. They should also appear in the
        formula, so that the random slope is a deviation from the fixed
        one.
    family : {'normal', 'logit', 'poisson'}
        ``'normal'``: Gaussian errors with variance ``sigma2``; blocked
        Gibbs sampler of Chib and Carlin (1999). ``'logit'`` /
        ``'poisson'``: the usual generalised linear mixed model, sampled
        by Metropolis steps for the fixed and the random effects.
    random_intercept : bool, default True
    prior_mean, prior_var
        Normal prior of the fixed effects, as in
        :func:`statspai.bayes_regress`. Default variance 1000 (100 for
        logit).
    sigma2_prior : (alpha0, delta0), default (0.001, 0.001)
        ``sigma2 ~ InvGamma(alpha0 / 2, delta0 / 2)``, normal family.
    re_prior : (df, scale), optional
        ``D ~ InvWishart(df, df * scale)``, whose mean is
        ``df * scale / (df - q - 1)``. Default ``df = q + 2`` and the
        identity scale (``q`` the number of random effects), a weak prior
        centred on unit variances. ``scale`` may be a number, a vector of
        variances or a ``q x q`` matrix.
    draws, burnin, thin, chains, seed, level
        As in :func:`statspai.bayes_regress`. Proposals of the Metropolis
        steps are tuned during the burn-in only.

    Returns
    -------
    BayesMixedResult
        The reported parameters are the fixed effects, ``sigma2`` (normal
        family) and the elements of ``D`` (``var(<term>)``,
        ``cov(<term>,<term>)``).

    Notes
    -----
    The default prior of ``D`` is centred on unit variances. That is
    informative when the random effects live on a much smaller or larger
    scale (a log outcome, say), and with few groups even a well-scaled
    prior matters. The fit measures the prior's share of each variance
    component (``model_info['re_prior_share']``) and warns above 25
    percent; set ``re_prior`` and report how the results move.

    R ``MCMCpack::MCMChlogit`` and ``MCMChpoisson`` add an
    observation-level normal error to the linear index. The models here
    do not; they are the ones ``sp.melogit`` / ``sp.mepoisson`` and Stata
    ``melogit`` / ``mepoisson`` fit by maximum likelihood.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(40), 5)
    >>> df = pd.DataFrame({"id": g, "x": rng.normal(size=200)})
    >>> u = rng.normal(size=40)[g]
    >>> df["y"] = 1 + 0.5 * df["x"] + u + rng.normal(size=200)
    >>> fit = sp.bayes_mixed("y ~ x", df, group="id", draws=1000,
    ...                      burnin=300, seed=1)
    >>> bool(fit.prob("x > 0") > 0.99)
    True
    >>> slopes = sp.bayes_mixed("y ~ x", df, group="id", random=["x"],
    ...                         draws=1000, burnin=300, seed=1)
    >>> list(slopes.re_cov.columns)
    ['Intercept', 'x']

    References
    ----------
    chib1999mcmc, ramirezhassan2026introduction
    """
    fam = _ALIASES.get(str(family).lower(), str(family).lower())
    if fam not in _FAMILIES:
        raise MethodIncompatibility(
            f"Unknown family {family!r}. Available: {', '.join(_FAMILIES)}."
        )
    check_mcmc_args(draws, burnin, thin, chains)
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("data must be a pandas DataFrame.")
    if group not in data.columns:
        raise MethodIncompatibility(f"group column {group!r} is not in data.")
    random = list(random or [])
    missing = [c for c in random if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"random= columns not in data: {missing}.")
    if not random and not random_intercept:
        raise MethodIncompatibility(
            "The model has no random effect: pass random= or keep "
            "random_intercept=True. Without one use sp.bayes_regress."
        )
    # complete cases on everything the model touches
    if "~" not in formula:
        raise MethodIncompatibility(
            f"formula must look like 'y ~ x1 + x2'; got {formula!r}."
        )
    work = data.loc[data[[group] + random].notna().all(axis=1)]
    y_df, X_df = create_design_matrices(formula, work)
    idx = X_df.index if hasattr(X_df, "index") else work.index
    work = work.loc[idx]
    y = np.asarray(y_df, dtype=float).reshape(-1)
    X = np.asarray(X_df, dtype=float)
    xnames = [str(c) for c in X_df.columns]
    design_info = getattr(X_df, "design_info", None)
    n, k = X.shape
    wcols: List[np.ndarray] = []
    wnames: List[str] = []
    if random_intercept:
        wcols.append(np.ones(n))
        wnames.append("Intercept")
    for c in random:
        wcols.append(work[c].to_numpy(dtype=float))
        wnames.append(str(c))
    W = np.column_stack(wcols)
    q = W.shape[1]
    codes, uniques = pd.factorize(work[group], sort=True)
    n_groups = len(uniques)
    if n_groups < 3:
        raise DataInsufficient(
            f"Only {n_groups} groups; the covariance of the random effects "
            "cannot be learned from so few."
        )
    if n_groups == n:
        raise DataInsufficient(
            "Every group has a single observation; the random effects are "
            "not separable from the errors."
        )
    if n <= k:
        raise DataInsufficient(f"{n} observations for {k} fixed effects.")
    if np.linalg.matrix_rank(X) < k:
        raise MethodIncompatibility("The fixed-effect regressors are collinear.")
    if fam == "logit" and not np.all(np.isin(np.unique(y), (0.0, 1.0))):
        raise MethodIncompatibility("family='logit' needs a 0/1 outcome.")
    if fam == "poisson" and (np.any(y < 0) or np.any(y != np.round(y))):
        raise MethodIncompatibility("family='poisson' needs non-negative counts.")
    not_fixed = [w for w in wnames if w not in xnames]
    if not_fixed:
        warnings.warn(
            f"The random-effect terms {not_fixed} are not in the fixed part "
            "of the formula; their effects are then centred on zero rather "
            "than on an estimated mean.",
            stacklevel=2,
        )

    if prior_var is None:
        prior_var = 100.0 if fam == "logit" else 1000.0
    b0, B0, B0inv = normal_prior(k, prior_mean, prior_var, xnames)
    a0, d0 = (float(v) for v in sigma2_prior)
    if a0 <= 0 or d0 <= 0:
        raise MethodIncompatibility("sigma2_prior must be two positive numbers.")
    if re_prior is None:
        r0, R0 = float(q + 2), np.eye(q)
    else:
        r0 = float(re_prior[0])
        sc = np.asarray(re_prior[1], dtype=float)
        if sc.ndim == 0:
            R0 = np.eye(q) * float(sc)
        elif sc.ndim == 1 and sc.shape == (q,):
            R0 = np.diag(sc)
        elif sc.shape == (q, q):
            R0 = 0.5 * (sc + sc.T)
        else:
            raise MethodIncompatibility(
                f"re_prior scale must be a number, {q} variances or a "
                f"{q} x {q} matrix."
            )
        if r0 <= q - 1:
            raise MethodIncompatibility(
                f"re_prior degrees of freedom must exceed q - 1 = {q - 1}."
            )
        try:
            linalg.cholesky(R0)
        except linalg.LinAlgError as exc:
            raise MethodIncompatibility(
                "re_prior scale must be positive definite."
            ) from exc
    prior = {"b0": b0, "B0inv": B0inv, "a0": a0, "d0": d0, "r0": r0, "R0": R0}

    n_iter = burnin + draws * thin
    rngs = spawn_rngs(seed, chains)
    pieces = []
    running = _Running((n_groups, q))
    accepts: List[float] = []
    match = [(xnames.index(w), j) for j, w in enumerate(wnames) if w in xnames]
    for ch in range(chains):
        if fam == "normal":
            arr, run, acc = _sample_normal(
                rngs[ch],
                y,
                X,
                W,
                codes,
                n_groups,
                prior,
                n_iter,
                burnin,
                thin,
                chains > 1,
            )
        else:
            arr, run, acc = _sample_glmm(
                rngs[ch],
                y,
                X,
                W,
                codes,
                n_groups,
                prior,
                fam,
                match,
                n_iter,
                burnin,
                thin,
                chains > 1,
            )
        pieces.append(arr)
        running.merge(run)
        if acc is not None:
            accepts.append(acc)
    arr = np.vstack(pieces)
    re_names, pos = _re_names(wnames)
    names = xnames + (["sigma2"] if fam == "normal" else []) + re_names
    d_df = pd.DataFrame(arr, columns=names)
    chain_idx = np.repeat(np.arange(chains), draws)

    ess = np.zeros(len(names))
    for ch in range(chains):
        ess += mcmc_summary(d_df.loc[chain_idx == ch], quantiles=())["ess"].to_numpy()
    lo = (1.0 - level) / 2.0
    sd = d_df.std(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mcse = sd.to_numpy() / np.sqrt(ess)
    table = pd.DataFrame(
        {
            "mean": d_df.mean(),
            "sd": sd,
            "mcse": mcse,
            "ess": ess,
            "lower": d_df.quantile(lo),
            "median": d_df.quantile(0.5),
            "upper": d_df.quantile(1.0 - lo),
            "prob_positive": (d_df > 0).mean(),
        }
    )
    mean_b, sd_b = running.moments()
    re_df = pd.DataFrame(mean_b, index=pd.Index(uniques, name=group), columns=wnames)
    for j, w in enumerate(wnames):
        re_df[f"{w}_sd"] = sd_b[:, j]
    Dbar = np.zeros((q, q))
    for nm, (i, j) in zip(re_names, pos):
        Dbar[i, j] = Dbar[j, i] = table.loc[nm, "mean"]
    re_cov = pd.DataFrame(Dbar, index=wnames, columns=wnames)

    sizes = np.bincount(codes)
    info: Dict[str, Any] = {
        "family": fam,
        "random_terms": wnames,
        "group": group,
        "group_sizes": {
            "min": int(sizes.min()),
            "mean": float(sizes.mean()),
            "max": int(sizes.max()),
        },
    }
    if fam == "normal" and q == 1:
        icc = d_df[re_names[0]] / (d_df[re_names[0]] + d_df["sigma2"])
        info["icc"] = {
            "mean": float(icc.mean()),
            "lower": float(icc.quantile(lo)),
            "upper": float(icc.quantile(1.0 - lo)),
        }
    diag_info: Dict[str, Any] = {"warnings": [], "min_ess": float(table["ess"].min())}
    # How much of each variance component is the prior? The full conditional
    # of D is InvWishart(r0 + G, r0 * R0 + sum_i b_i b_i'): the prior adds
    # r0 * R0_jj to the sum of squares of the group effects.
    ss = (mean_b**2 + sd_b**2).sum(axis=0)
    prior_ss = r0 * np.diag(R0)
    share = prior_ss / (prior_ss + ss)
    info["re_prior_share"] = {w: float(v) for w, v in zip(wnames, share)}
    try:
        gr = gelman_rubin(
            [d_df.loc[chain_idx == ch] for c in range(chains)], split=True
        )
        diag_info["max_split_rhat"] = float(gr.table["psrf"].max())
    except (MethodIncompatibility, DataInsufficient):
        diag_info["max_split_rhat"] = float("nan")
    sampler = (
        "blocked Gibbs (random effects integrated out of the fixed-effect draw)"
        if fam == "normal"
        else "Metropolis within Gibbs, proposals tuned in the burn-in"
    )
    res = BayesMixedResult(
        model=fam,
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
        sampler=sampler,
        acceptance_rate=float(np.mean(accepts)) if accepts else None,
        prior={
            "coefficients": "normal",
            "prior_mean": prior_mean,
            "prior_var": prior_var,
            "random_effects": f"InvWishart({r0:g}, {r0:g} * scale)",
            "re_scale": R0,
            **(
                {"sigma2": f"InvGamma({a0 / 2:g}, {d0 / 2:g})"}
                if fam == "normal"
                else {}
            ),
        },
        level=level,
        model_info=info,
        diagnostics_info=diag_info,
        _model=_MixedShim(X, xnames, fam),
        _design_info=design_info,
        random_effects=re_df,
        re_cov=re_cov,
        n_groups=n_groups,
    )
    msgs = []
    if diag_info["min_ess"] < 100:
        worst = str(table["ess"].idxmin())
        msgs.append(
            f"effective sample size is {diag_info['min_ess']:.0f} for '{worst}'"
        )
    rhat = diag_info["max_split_rhat"]
    if np.isfinite(rhat) and rhat > 1.05:
        msgs.append(f"split potential scale reduction factor is {rhat:.3f}")
    heavy = [
        (w, float(v), float(m))
        for w, v, m in zip(wnames, share, ss / n_groups)
        if v > 0.25
    ]
    if heavy:
        parts = "; ".join(
            f"{w}: {100 * v:.0f}% of the sum of squares comes from the prior, "
            f"the groups' own effects have mean square {m:.3g}"
            for w, v, m in heavy
        )
        text = (
            "The prior on the covariance of the random effects is driving "
            "the variance components ("
            + parts
            + "). Its scale ("
            + ", ".join(f"{v:g}" for v in np.diag(R0))
            + ") is not on the scale of these effects. Pass re_prior=(df, "
            "scale) with a scale near the plausible variance, and report the "
            "sensitivity."
        )
        diag_info["warnings"].append(text)
        warnings.warn(text, StatsPAIWarning, stacklevel=2)
    if msgs:
        text = (
            "The chain may not have converged or mixes slowly: "
            + "; ".join(msgs)
            + ". Increase draws / burnin or thin the chain."
        )
        diag_info["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    return res
