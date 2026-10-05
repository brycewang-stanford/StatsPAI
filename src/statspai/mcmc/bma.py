"""
Bayesian model averaging over the regressors of a (generalised) linear
model: ``sp.bma``.

Two families of posterior model probabilities:

* ``method='bic'``    -- the BIC approximation to the marginal likelihood
  with Occam's window (Raftery 1995; Raftery, Madigan and Hoeting 1997),
  for Gaussian, binomial, Poisson and gamma outcomes. Matches R
  ``BMA::bicreg`` and ``BMA::bic.glm``.
* ``method='gprior'`` -- exact marginal likelihoods under Zellner's
  g-prior with the benchmark choices of Fernandez, Ley and Steel (2001),
  Gaussian outcomes. Matches R ``BMS::bms``. All models are enumerated
  when that is feasible; otherwise the model space is explored by the
  MC3 sampler (Raftery, Madigan and Hoeting 1997).

The models inside Occam's window are found by an exact branch and bound,
not by a "best few of each size" search, so no model inside the window
is missed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg, special

from .._result_serialize import ResultProtocolMixin
from ..core.utils import create_design_matrices
from ..exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)
from ._core import design_for

_FAMILIES = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "inverse",
}
_FAMILY_ALIASES = {
    "normal": "gaussian",
    "linear": "gaussian",
    "logit": "binomial",
    "logistic": "binomial",
}


# --------------------------------------------------------------------------
# GLM fitting by iteratively reweighted least squares
# --------------------------------------------------------------------------


def _glm_parts(family: str, link: str) -> Tuple[Callable, Callable, Callable, Callable]:
    """Inverse link, d mu / d eta, variance function, deviance."""
    if link == "logit":
        inv = special.expit

        def dmu(eta: np.ndarray) -> np.ndarray:
            p = special.expit(eta)
            return np.asarray(p * (1.0 - p))

    elif link == "probit":
        inv = special.ndtr

        def dmu(eta: np.ndarray) -> np.ndarray:
            return np.asarray(np.exp(-0.5 * eta * eta) / np.sqrt(2.0 * np.pi))

    elif link == "log":

        def inv(eta: np.ndarray) -> np.ndarray:
            return np.asarray(np.exp(np.clip(eta, -700.0, 700.0)))

        dmu = inv
    elif link == "inverse":

        def inv(eta: np.ndarray) -> np.ndarray:
            return 1.0 / eta

        def dmu(eta: np.ndarray) -> np.ndarray:
            return np.asarray(-1.0 / (eta * eta))

    elif link == "identity":

        def inv(eta: np.ndarray) -> np.ndarray:
            return eta

        def dmu(eta: np.ndarray) -> np.ndarray:
            return np.ones_like(eta)

    else:
        raise MethodIncompatibility(f"Unknown link {link!r}.")

    if family == "binomial":

        def var(mu: np.ndarray) -> np.ndarray:
            return mu * (1.0 - mu)

        def dev(y: np.ndarray, mu: np.ndarray) -> float:
            mu = np.clip(mu, 1e-15, 1 - 1e-15)
            return float(
                2.0
                * (
                    special.xlogy(y, y / mu) + special.xlogy(1 - y, (1 - y) / (1 - mu))
                ).sum()
            )

    elif family == "poisson":

        def var(mu: np.ndarray) -> np.ndarray:
            return mu

        def dev(y: np.ndarray, mu: np.ndarray) -> float:
            return float(2.0 * (special.xlogy(y, y / mu) - (y - mu)).sum())

    elif family == "gamma":

        def var(mu: np.ndarray) -> np.ndarray:
            return np.asarray(mu * mu)

        def dev(y: np.ndarray, mu: np.ndarray) -> float:
            return float(2.0 * (-np.log(y / mu) + (y - mu) / mu).sum())

    else:

        def var(mu: np.ndarray) -> np.ndarray:
            return np.ones_like(mu)

        def dev(y: np.ndarray, mu: np.ndarray) -> float:
            return float(((y - mu) ** 2).sum())

    return inv, dmu, var, dev


@dataclass
class _GLMFit:
    coef: np.ndarray
    se: np.ndarray
    deviance: float
    dispersion: float
    weights: np.ndarray
    working: np.ndarray
    converged: bool


def _glm_fit(
    y: np.ndarray, X: np.ndarray, family: str, link: str, max_iter: int = 100
) -> _GLMFit:
    inv, dmu, var, dev = _glm_parts(family, link)
    n, p = X.shape
    # starting values as in standard IRLS
    if family == "binomial":
        mu = (y + 0.5) / 2.0
    elif family == "poisson":
        mu = y + 0.1
    else:
        mu = np.where(y > 0, y, y.mean()) if family == "gamma" else y.copy()
    if link == "logit":
        eta = special.logit(mu)
    elif link == "probit":
        eta = special.ndtri(mu)
    elif link == "log":
        eta = np.log(mu)
    elif link == "inverse":
        eta = 1.0 / mu
    else:
        eta = mu.copy()
    dev_old = dev(y, mu)
    coef = np.zeros(p)
    converged = False
    w = np.ones(n)
    z = eta.copy()
    for _ in range(max_iter):
        g = dmu(eta)
        w = g * g / var(mu)
        z = eta + (y - mu) / g
        sw = np.sqrt(w)
        coef_new = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)[0]
        eta_new = X @ coef_new
        mu_new = inv(eta_new)
        dev_new = dev(y, mu_new)
        # step halving when the deviance is not finite or the mean leaves
        # the family's support
        halve = 0
        while (
            not np.isfinite(dev_new)
            or (family in ("poisson", "gamma") and np.any(mu_new <= 0))
        ) and halve < 30:
            coef_new = 0.5 * (coef_new + coef)
            eta_new = X @ coef_new
            mu_new = inv(eta_new)
            dev_new = dev(y, mu_new)
            halve += 1
        coef, eta, mu = coef_new, eta_new, mu_new
        if abs(dev_new - dev_old) / (abs(dev_new) + 0.1) < 1e-10:
            dev_old = dev_new
            converged = True
            break
        dev_old = dev_new
    g = dmu(eta)
    w = g * g / var(mu)
    z = eta + (y - mu) / g
    xtwx = (X * w[:, None]).T @ X
    try:
        cov = linalg.inv(xtwx)
    except linalg.LinAlgError:
        cov = np.linalg.pinv(xtwx)
    if family in ("gamma", "gaussian"):
        disp = float((w * (z - eta) ** 2).sum() / max(n - p, 1))
    else:
        disp = 1.0
    se = np.sqrt(np.clip(np.diag(cov) * disp, 0.0, None))
    return _GLMFit(coef, se, float(dev_old), disp, w, z, converged)


# --------------------------------------------------------------------------
# Branch and bound over term subsets
# --------------------------------------------------------------------------


class _SubsetRSS:
    """Residual sums of squares of every subset of grouped columns, from
    the cross-product matrix of ``[X, y]`` (already residualised on the
    columns that are always in the model)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, groups: List[np.ndarray]):
        self.G = X.T @ X
        self.g = X.T @ y
        self.yy = float(y @ y)
        self.groups = groups
        self.sizes = np.array([len(g) for g in groups])

    def cols(self, members: Sequence[int]) -> np.ndarray:
        if len(members) == 0:
            return np.zeros(0, dtype=int)
        return np.concatenate([self.groups[m] for m in members])

    def rss(self, members: Sequence[int]) -> float:
        c = self.cols(members)
        if c.size == 0:
            return self.yy
        sub = self.G[np.ix_(c, c)]
        rhs = self.g[c]
        try:
            chol = linalg.cholesky(sub, lower=True, check_finite=False)
            z = linalg.solve_triangular(chol, rhs, lower=True, check_finite=False)
            return max(self.yy - float(z @ z), 0.0)
        except linalg.LinAlgError:
            sol = np.linalg.lstsq(sub, rhs, rcond=None)[0]
            return max(self.yy - float(rhs @ sol), 0.0)


def _occam_search(
    sub: _SubsetRSS,
    n: int,
    penalty: float,
    log_prior_in: np.ndarray,
    log_prior_out: np.ndarray,
    window: float,
    max_nodes: int,
    linear_scale: Optional[float] = None,
) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    """All subsets whose criterion is within ``window`` of the best.

    Criterion: ``n log RSS + penalty * columns - 2 log prior``, or with
    ``linear_scale`` (the weighted least squares image of a GLM)
    ``RSS / linear_scale + ...``. A node
    fixes the first ``idx`` terms; any completion has at least the RSS of
    "everything still allowed" and at least the columns already
    included, which bounds its criterion from below.
    """
    G = len(sub.groups)
    # most favourable prior contribution of the undecided terms
    best_prior_tail = np.zeros(G + 1)
    tail = -2.0 * np.maximum(log_prior_in, log_prior_out)
    best_prior_tail[:G] = np.cumsum(tail[::-1])[::-1]

    def crit(rss: float, ncol: int, prior: float) -> float:
        if linear_scale is not None:
            return rss / linear_scale + penalty * ncol + prior
        if rss <= 0.0:
            return -np.inf
        return float(n * np.log(rss) + penalty * ncol + prior)

    # greedy start: forward selection on the criterion
    current: List[int] = []
    cur_prior = -2.0 * log_prior_out.sum()
    best = crit(sub.rss(current), 0, cur_prior)
    improved = True
    while improved:
        improved = False
        for j in range(G):
            if j in current:
                continue
            trial = sorted(current + [j])
            pr = cur_prior - 2.0 * (log_prior_in[j] - log_prior_out[j])
            val = crit(sub.rss(trial), int(sub.sizes[trial].sum()), pr)
            if val < best - 1e-12:
                best, current, cur_prior, improved = val, trial, pr, True
                break

    found: List[Tuple[int, ...]] = []
    values: List[float] = []
    everything = list(range(G))
    stack: List[Tuple[int, Tuple[int, ...], float, int, float]] = [
        (0, (), sub.rss(everything), 0, 0.0)
    ]
    nodes = 0
    while stack:
        idx, inc, rss_u, ncol, prior = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise NumericalInstability(
                f"The search for the models inside Occam's window visited "
                f"more than {max_nodes} nodes. With this many weakly "
                "relevant candidates the window holds too many models to "
                "enumerate: lower occam_ratio, raise max_nodes, or use "
                "method='gprior' with search='mc3'."
            )
        bound = crit(rss_u, ncol, prior + best_prior_tail[idx])
        if bound > best + window:
            continue
        if idx == G:
            found.append(inc)
            values.append(bound)
            if bound < best:
                best = bound
            continue
        rest = list(range(idx + 1, G))
        # exclude term idx
        stack.append(
            (
                idx + 1,
                inc,
                sub.rss(list(inc) + rest),
                ncol,
                prior - 2.0 * log_prior_out[idx],
            )
        )
        # include term idx (explored first)
        stack.append(
            (
                idx + 1,
                inc + (idx,),
                rss_u,
                ncol + int(sub.sizes[idx]),
                prior - 2.0 * log_prior_in[idx],
            )
        )
    vals = np.asarray(values)
    keep = vals <= vals.min() + window
    order = np.argsort(vals[keep], kind="stable")
    models = [m for m, k in zip(found, keep) if k]
    return [models[i] for i in order], vals[keep][order]


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass
class BMAResult(ResultProtocolMixin):
    """Result of :func:`bma`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per coefficient: ``pip`` (posterior inclusion
        probability), ``post_mean`` and ``post_sd`` (averaged over
        models, zero where the coefficient is excluded), ``cond_mean``
        and ``cond_sd`` (given inclusion) and ``sign_certainty``
        (posterior probability, given inclusion, that the coefficient
        has the sign of ``cond_mean``, from the model-wise normal
        approximations).
    models : pd.DataFrame
        The models averaged over, best first: one 0/1 column per term,
        ``n_terms``, the criterion (``bic`` or ``log_marglik``), ``r2``
        (Gaussian) or ``deviance``, and ``post_prob``.
    params : pd.Series
        ``table['post_mean']``.
    std_errors : pd.Series
        ``table['post_sd']``.
    n_obs : int
    n_models : int
        Number of models retained.
    method, family : str

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    >>> df["y"] = 1 + df["a"] - 0.5 * df["c"] + rng.normal(size=200)
    >>> out = sp.bma("y ~ a + b + c + d", df)
    >>> isinstance(out, sp.BMAResult)
    True
    >>> bool(out.table.loc["a", "pip"] > 0.99)
    True
    """

    table: pd.DataFrame
    models: pd.DataFrame
    params: pd.Series
    std_errors: pd.Series
    n_obs: int
    n_models: int
    method: str
    family: str
    link: str
    formula: str
    settings: Dict[str, Any] = field(default_factory=dict)
    _fits: Any = field(default=None, repr=False)
    _design_info: Any = field(default=None, repr=False)

    _citation_keys = (
        "raftery1995bayesian",
        "raftery1997bayesian",
        "hoeting1999bayesian",
    )

    @property
    def pip(self) -> pd.Series:
        """Posterior inclusion probabilities."""
        return self.table["pip"]

    @property
    def coef(self) -> pd.Series:
        return self.params

    def summary(self, top: int = 5) -> str:
        lines = [
            f"Bayesian model averaging ({self.method}, {self.family})    "
            f"{self.formula}",
            f"Observations: {self.n_obs}    Models averaged: {self.n_models}"
            + (
                f"    Cumulative probability of the best {min(top, self.n_models)}: "
                f"{self.models['post_prob'].head(top).sum():.3f}"
            ),
            "",
            self.table.to_string(float_format=lambda v: f"{v:.4g}"),
            "",
            f"Best {min(top, self.n_models)} models:",
            self.models.head(top).to_string(float_format=lambda v: f"{v:.4g}"),
        ]
        for note in self.settings.get("notes", []):
            lines.append("")
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def tidy(self) -> pd.DataFrame:
        return self.table.reset_index().rename(columns={"index": "term"})

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "method": self.method,
            "family": self.family,
            "link": self.link,
            "formula": self.formula,
            "n_obs": int(self.n_obs),
            "n_models": int(self.n_models),
            "coefficients": {
                str(t): {c: float(r[c]) for c in self.table.columns}
                for t, r in self.table.iterrows()
            },
            "top_models": self.models.head(10).to_dict(orient="records"),
            "settings": {
                k: v for k, v in self.settings.items() if not isinstance(v, np.ndarray)
            },
        }

    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Model-averaged prediction of the mean.

        Each retained model's prediction, on the scale of the outcome,
        weighted by its posterior probability.
        """
        if self._fits is None:
            raise MethodIncompatibility("This result cannot predict.")
        if data is None:
            X = self._fits["X"]
        else:
            X = design_for(self._design_info, list(self.table.index), data)
        inv = _glm_parts(self.family, self.link)[0]
        out = np.zeros(X.shape[0])
        for prob, coef in zip(self._fits["probs"], self._fits["coefs"]):
            out += prob * inv(X @ coef)
        return out

    def plot(self) -> Any:
        """Posterior inclusion probabilities, largest first."""
        import matplotlib.pyplot as plt

        t = self.table.loc[self.table["pip"] < 1.0 - 1e-12, "pip"].sort_values()
        if t.empty:
            t = self.table["pip"].sort_values()
        fig, ax = plt.subplots(figsize=(6.0, 0.35 * len(t) + 1.2))
        ax.barh([str(i) for i in t.index], t.to_numpy())
        ax.axvline(0.5, ls="--", lw=0.8, color="grey")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Posterior inclusion probability")
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _term_groups(
    X_df: pd.DataFrame, design_info: Any
) -> Tuple[List[str], List[np.ndarray]]:
    """Columns of the design grouped by formula term."""
    names = [str(c) for c in X_df.columns]
    if design_info is not None and hasattr(design_info, "term_name_slices"):
        terms, groups = [], []
        for term, sl in design_info.term_name_slices.items():
            terms.append(str(term))
            groups.append(np.arange(sl.start, sl.stop))
        return terms, groups
    return names, [np.array([j]) for j in range(len(names))]


def bma(
    formula: str,
    data: pd.DataFrame,
    family: str = "gaussian",
    method: str = "bic",
    link: Optional[str] = None,
    always: Optional[Sequence[str]] = None,
    occam_ratio: float = 20.0,
    strict: bool = False,
    prior_inclusion: Union[float, Dict[str, float]] = 0.5,
    g: Union[str, float] = "benchmark",
    search: str = "auto",
    draws: int = 20000,
    burnin: int = 2000,
    seed: Optional[int] = None,
    max_nodes: int = 2_000_000,
) -> BMAResult:
    """Bayesian model averaging over which regressors enter a model.

    Every subset of the candidate terms is a model. Each gets a posterior
    probability, and coefficients are averaged over models with those
    probabilities, so the reported uncertainty includes not knowing which
    regressors belong.

    Parameters
    ----------
    formula : str
        ``'y ~ x1 + x2 + ...'`` with all candidate terms. The intercept
        is always included. The columns of a categorical term enter or
        leave together.
    data : DataFrame
    family : {'gaussian', 'binomial', 'poisson', 'gamma'}
    method : {'bic', 'gprior'}
        ``'bic'``: posterior model probabilities proportional to
        ``exp(-BIC / 2)``, models outside Occam's window dropped
        (R ``BMA::bicreg`` / ``bic.glm``). ``'gprior'``: exact marginal
        likelihoods under Zellner's g-prior, Gaussian family only
        (R ``BMS::bms``).
    link : str, optional
        Default: identity, logit, log and inverse for the four families.
        Also ``'probit'`` (binomial) and ``'log'`` (gamma).
    always : list of str, optional
        Terms that are in every model, e.g. the regressor of interest
        when averaging over controls.
    occam_ratio : float, default 20
        ``method='bic'``: a model is dropped when the best model is more
        than this many times as probable. ``np.inf`` keeps all models
        (feasible for up to about 20 terms).
    strict : bool, default False
        ``method='bic'``: also drop a model when one of its own
        submodels is more probable.
    prior_inclusion : float or dict, default 0.5
        Prior probability that a term is in the model, common to all or
        by term name. 0.5 is the uniform prior over models.
    g : {'benchmark', 'uip', 'ric'} or float
        ``method='gprior'``: the prior is
        ``beta | sigma2 ~ N(0, g sigma2 (X'X)^{-1})`` on the centred
        regressors. ``'uip'``: ``g = n``; ``'ric'``: ``g = K^2``;
        ``'benchmark'``: ``g = max(n, K^2)`` (Fernandez, Ley and Steel
        2001; ``g = 'BRIC'`` in R BMS).
    search : {'auto', 'enumerate', 'mc3'}
        ``method='gprior'``: enumerate all ``2^K`` models (default up to
        20 terms) or sample the model space with MC3.
    draws, burnin : int
        MC3 iterations kept and discarded.
    seed : int, optional
        For MC3.
    max_nodes : int
        Budget of the branch-and-bound search.

    Returns
    -------
    BMAResult

    Notes
    -----
    BIC for a Gaussian model is ``n log(1 - R^2) + p log n``; for a GLM it
    is the deviance in excess of the null model's plus ``p log n``, the
    deviance of the gamma family being divided by the Pearson dispersion
    of the model with all candidate terms. These are the definitions of
    the R package BMA, which also uses that one dispersion for the
    standard errors of every gamma model. ``BMA::bicreg`` computes its BIC
    from an ``R^2`` rounded to five decimals, so its probabilities agree with the ones
    here to about three digits; rounding ``R^2`` the same way reproduces
    them exactly.

    The standard deviations combine within-model variance and
    between-model spread:
    ``post_sd^2 = sum_m p_m (se_m^2 + b_m^2) - post_mean^2``.

    Model averaging is a tool for prediction and for sensitivity of an
    estimate to the choice of controls. A high inclusion probability is
    not a causal statement.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.normal(size=(300, 5)),
    ...                   columns=["x1", "x2", "x3", "x4", "x5"])
    >>> df["y"] = 1 + df["x1"] + 0.5 * df["x3"] + rng.normal(size=300)
    >>> out = sp.bma("y ~ x1 + x2 + x3 + x4 + x5", df)
    >>> list(out.table.columns)[:3]
    ['pip', 'post_mean', 'post_sd']
    >>> sp.bma("y ~ x1 + x2 + x3 + x4 + x5", df, method="gprior").n_models
    32

    References
    ----------
    raftery1995bayesian, raftery1997bayesian, hoeting1999bayesian,
    fernandez2001benchmark
    """
    family = _FAMILY_ALIASES.get(str(family).lower(), str(family).lower())
    if family not in _FAMILIES:
        raise MethodIncompatibility(
            f"Unknown family {family!r}. Available: {', '.join(_FAMILIES)}."
        )
    method = str(method).lower().replace("-", "").replace("_", "")
    if method not in ("bic", "gprior"):
        raise MethodIncompatibility("method must be 'bic' or 'gprior'.")
    if method == "gprior" and family != "gaussian":
        raise MethodIncompatibility(
            "method='gprior' has closed-form marginal likelihoods only for "
            "family='gaussian'. Use method='bic' for a GLM."
        )
    link = _FAMILIES[family] if link is None else str(link).lower()
    allowed = {
        "gaussian": ("identity",),
        "binomial": ("logit", "probit"),
        "poisson": ("log",),
        "gamma": ("inverse", "log"),
    }[family]
    if link not in allowed:
        raise MethodIncompatibility(
            f"link={link!r} is not available for family='{family}'; "
            f"choose from {allowed}."
        )
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("data must be a pandas DataFrame.")
    if not occam_ratio > 1:
        raise MethodIncompatibility("occam_ratio must exceed 1.")

    y_df, X_df = create_design_matrices(formula, data)
    y = np.asarray(y_df, dtype=float).reshape(-1)
    X = np.asarray(X_df, dtype=float)
    names = [str(c) for c in X_df.columns]
    design_info = getattr(X_df, "design_info", None)
    n = X.shape[0]
    terms, groups = _term_groups(X_df, design_info)
    if "Intercept" not in terms:
        raise MethodIncompatibility(
            "sp.bma keeps an intercept in every model; write the formula "
            "without '- 1'."
        )
    always = list(always or [])
    unknown = [a for a in always if a not in terms]
    if unknown:
        raise MethodIncompatibility(
            f"always= names terms that are not in the formula: {unknown}. "
            f"Terms: {[t for t in terms if t != 'Intercept']}."
        )
    fixed_terms = ["Intercept"] + always
    free_terms = [t for t in terms if t not in fixed_terms]
    if not free_terms:
        raise MethodIncompatibility("There is no candidate term to average over.")
    tindex = {t: i for i, t in enumerate(terms)}
    fixed_cols = np.concatenate([groups[tindex[t]] for t in fixed_terms])
    free_groups = [groups[tindex[t]] for t in free_terms]
    G = len(free_terms)
    k_free = int(sum(len(gr) for gr in free_groups))
    if n <= X.shape[1] + 1:
        raise DataInsufficient(
            f"{n} observations for {X.shape[1]} candidate coefficients; BMA "
            "by BIC or g-prior needs more observations than coefficients."
        )
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise MethodIncompatibility(
            "The candidate regressors are collinear; drop the redundant "
            "terms before averaging."
        )
    if family == "binomial" and not np.all(np.isin(np.unique(y), (0.0, 1.0))):
        raise MethodIncompatibility("family='binomial' needs a 0/1 outcome.")
    if family == "poisson" and (np.any(y < 0) or np.any(y != np.round(y))):
        raise MethodIncompatibility("family='poisson' needs non-negative counts.")
    if family == "gamma" and np.any(y <= 0):
        raise MethodIncompatibility("family='gamma' needs a positive outcome.")

    # prior inclusion probabilities by free term
    if isinstance(prior_inclusion, dict):
        bad = [t for t in prior_inclusion if t not in free_terms]
        if bad:
            raise MethodIncompatibility(
                f"prior_inclusion names terms that are not candidates: {bad}."
            )
        pi = np.array([float(prior_inclusion.get(t, 0.5)) for t in free_terms])
    else:
        pi = np.full(G, float(prior_inclusion))
    if np.any(pi <= 0) or np.any(pi >= 1):
        raise MethodIncompatibility(
            "prior_inclusion must be strictly between 0 and 1 (use always= "
            "for a term that is certainly in)."
        )
    # relative to the uniform prior, so that the criterion is the plain BIC
    # (or marginal likelihood) when prior_inclusion is 0.5
    lpi, lpo = np.log(2.0 * pi), np.log(2.0 * (1.0 - pi))
    notes: List[str] = []

    def member_cols(members: Sequence[int]) -> np.ndarray:
        cols = [fixed_cols] + [free_groups[m] for m in members]
        return np.sort(np.concatenate(cols))

    # ---------------------------------------------------------------------
    # which models, and their weights
    # ---------------------------------------------------------------------
    Xf = X[:, fixed_cols]
    q, _ = np.linalg.qr(Xf)

    def resid(a: np.ndarray) -> np.ndarray:
        return np.asarray(a - q @ (q.T @ a))

    free_all = np.concatenate(free_groups)
    # position of every free column inside the residualised block
    pos = {int(c): i for i, c in enumerate(free_all)}
    local_groups = [np.array([pos[int(c)] for c in gr]) for gr in free_groups]

    crit_name = "bic"
    if method == "bic":
        window = 2.0 * np.log(occam_ratio) if np.isfinite(occam_ratio) else np.inf
        if family == "gaussian":
            sub = _SubsetRSS(resid(X[:, free_all]), resid(y), local_groups)
            rss0 = sub.yy
            models, _ = _occam_search(sub, n, np.log(n), lpi, lpo, window, max_nodes)
            cand = models
        else:
            # candidates from the weighted least squares problem at the
            # full-model fit, with a widened window; exact refits follow
            full = _glm_fit(y, X, family, link)
            if not full.converged:
                raise NumericalInstability(
                    "The full model did not converge; BMA over its "
                    "submodels is not reliable. Check for separation or an "
                    "outcome outside the family's support."
                )
            sw = np.sqrt(full.weights)
            Xw, zw = X * sw[:, None], full.working * sw
            qw, _ = np.linalg.qr(Xw[:, fixed_cols])
            Xr = Xw[:, free_all] - qw @ (qw.T @ Xw[:, free_all])
            zr = zw - qw @ (qw.T @ zw)
            sub = _SubsetRSS(Xr, zr, local_groups)
            wide = window + max(window, 10.0) if np.isfinite(window) else np.inf
            # on the working scale the criterion is RSS / dispersion + p log n
            cand, _ = _occam_search(
                sub,
                n,
                np.log(n),
                lpi,
                lpo,
                wide,
                max_nodes,
                linear_scale=full.dispersion,
            )
    else:
        crit_name = "log_marglik"
        if isinstance(g, str):
            gkey = g.lower()
            if gkey in ("benchmark", "bric"):
                gval = float(max(n, k_free**2))
            elif gkey == "uip":
                gval = float(n)
            elif gkey == "ric":
                gval = float(k_free**2)
            else:
                raise MethodIncompatibility(
                    "g must be 'benchmark', 'uip', 'ric' or a positive number."
                )
        else:
            gval = float(g)
            if not gval > 0:
                raise MethodIncompatibility("g must be positive.")
        sub = _SubsetRSS(resid(X[:, free_all]), resid(y), local_groups)
        rss0 = sub.yy
        shrink = gval / (1.0 + gval)
        df0 = n - len(fixed_cols)  # N - 1 with only an intercept

        def log_ml(members: Sequence[int]) -> float:
            kcol = int(sub.sizes[list(members)].sum()) if len(members) else 0
            r2 = 1.0 - sub.rss(members) / rss0
            prior = float(
                sum(lpi[m] for m in members)
                + sum(lpo[m] for m in range(G) if m not in members)
            )
            fit_term = -0.5 * df0 * np.log1p(-shrink * r2)
            return float(-0.5 * kcol * np.log1p(gval) + fit_term + prior)

        search = str(search).lower()
        if search not in ("auto", "enumerate", "mc3"):
            raise MethodIncompatibility("search must be 'auto', 'enumerate' or 'mc3'.")
        if search == "auto":
            search = "enumerate" if G <= 20 else "mc3"
        if search == "enumerate":
            if G > 25:
                raise MethodIncompatibility(
                    f"Enumerating 2^{G} models is not feasible; use " "search='mc3'."
                )
            cand = []
            for code in range(1 << G):
                cand.append(tuple(j for j in range(G) if (code >> j) & 1))
        else:
            cand = _mc3(log_ml, G, draws, burnin, seed)
            notes.append(
                f"The model space was sampled by MC3 ({draws} iterations "
                f"after {burnin} of burn-in); probabilities are renormalised "
                f"over the {len(cand)} distinct models visited."
            )

    # ---------------------------------------------------------------------
    # fit every candidate
    # ---------------------------------------------------------------------
    p_total = X.shape[1]
    rows_coef: List[np.ndarray] = []
    rows_se: List[np.ndarray] = []
    crit_vals: List[float] = []
    extra: List[float] = []
    kept: List[Tuple[int, ...]] = []
    for members in cand:
        cols = member_cols(members)
        ncol_free = len(cols) - len(fixed_cols)
        coef = np.zeros(p_total)
        se = np.zeros(p_total)
        lprior = float(
            sum(lpi[m] for m in members)
            + sum(lpo[m] for m in range(G) if m not in members)
        )
        if family == "gaussian":
            Xm = X[:, cols]
            b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
            e = y - Xm @ b
            rss = float(e @ e)
            xtx_inv = linalg.inv(Xm.T @ Xm)
            r2 = 1.0 - rss / rss0
            if method == "bic":
                s2 = rss / (n - len(cols))
                coef[cols] = b
                se[cols] = np.sqrt(np.diag(xtx_inv) * s2)
                crit_vals.append(
                    n * np.log1p(-r2) + ncol_free * np.log(n) - 2.0 * lprior
                )
            else:
                # g-prior posterior moments given the model; the always
                # terms carry a flat prior
                free_mask = ~np.isin(cols, fixed_cols)
                bb = b.copy()
                var_scale = rss0 * (1.0 - shrink * r2) / (df0 - 2.0)
                if free_mask.any():
                    Xc = resid(Xm[:, free_mask])
                    bfree = np.linalg.lstsq(Xc, resid(y), rcond=None)[0]
                    bb[free_mask] = shrink * bfree
                    # the fixed coefficients adjust to the shrunk slopes
                    yfix = y - Xm[:, free_mask] @ bb[free_mask]
                    bb[~free_mask] = np.linalg.lstsq(Xf, yfix, rcond=None)[0]
                    vfree = shrink * var_scale * np.diag(linalg.inv(Xc.T @ Xc))
                    se_m: np.ndarray = np.zeros(len(cols))
                    se_m[free_mask] = np.sqrt(vfree)
                    se_m[~free_mask] = np.sqrt(np.diag(xtx_inv)[~free_mask] * var_scale)
                else:
                    se_m = np.sqrt(np.diag(xtx_inv) * var_scale)
                coef[cols] = bb
                se[cols] = se_m
                crit_vals.append(log_ml(members))
            extra.append(r2)
        else:
            fit = _glm_fit(y, X[:, cols], family, link)
            if not fit.converged:
                notes.append(
                    "A candidate model did not converge and was dropped: "
                    + ", ".join(free_terms[m] for m in members)
                )
                continue
            coef[cols] = fit.coef
            se[cols] = fit.se
            if family == "gamma" and fit.dispersion > 0:
                # one dispersion for every model: the full model's
                se[cols] = fit.se * np.sqrt(full.dispersion / fit.dispersion)
            dev = fit.deviance / full.dispersion if family == "gamma" else fit.deviance
            crit_vals.append(dev + ncol_free * np.log(n) - 2.0 * lprior)
            extra.append(fit.deviance)
        rows_coef.append(coef)
        rows_se.append(se)
        kept.append(tuple(members))

    crit_arr = np.asarray(crit_vals)
    if method == "bic":
        if family != "gaussian":
            # exact Occam's window on the refitted candidates; GLM BIC is
            # reported relative to the null model
            null = _glm_fit(y, X[:, fixed_cols], family, link)
            null_dev = (
                null.deviance / full.dispersion if family == "gamma" else null.deviance
            )
            crit_arr = crit_arr - null_dev
        keep = crit_arr <= crit_arr.min() + window
        if strict:
            order0 = np.argsort(crit_arr)
            for a in order0:
                if not keep[a]:
                    continue
                sa = set(kept[a])
                for b_i in range(len(kept)):
                    if b_i != a and keep[b_i] and sa < set(kept[b_i]):
                        if crit_arr[b_i] > crit_arr[a]:
                            keep[b_i] = False
        logw = -0.5 * crit_arr
    else:
        keep = np.ones(crit_arr.size, dtype=bool)
        logw = crit_arr.copy()
    idx = np.where(keep)[0]
    idx = idx[np.argsort(-logw[idx], kind="stable")]
    logw_k = logw[idx]
    probs = np.exp(logw_k - special.logsumexp(logw_k))
    coefs = np.vstack([rows_coef[i] for i in idx])
    ses = np.vstack([rows_se[i] for i in idx])
    members_k = [kept[i] for i in idx]

    # ---------------------------------------------------------------------
    # averaged coefficients
    # ---------------------------------------------------------------------
    incl = np.zeros((len(idx), p_total), dtype=bool)
    for r, members in enumerate(members_k):
        incl[r, member_cols(members)] = True
    pip = probs @ incl
    post_mean = probs @ coefs
    second = probs @ (ses**2 + coefs**2)
    post_sd = np.sqrt(np.clip(second - post_mean**2, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_mean = np.where(pip > 0, post_mean / pip, 0.0)
        cond_sd = np.sqrt(
            np.clip(np.where(pip > 0, second / pip, 0.0) - cond_mean**2, 0.0, None)
        )
        # P(sign(beta) = sign(cond_mean) | included), normal within model
        zsc = np.where(ses > 0, coefs / np.where(ses > 0, ses, 1.0), 0.0)
        ppos = special.ndtr(zsc)
        ppos_avg = np.where(
            pip > 0, (probs @ (ppos * incl)) / np.where(pip > 0, pip, 1.0), np.nan
        )
    sign_cert = np.where(cond_mean >= 0, ppos_avg, 1.0 - ppos_avg)
    table = pd.DataFrame(
        {
            "pip": pip,
            "post_mean": post_mean,
            "post_sd": post_sd,
            "cond_mean": cond_mean,
            "cond_sd": cond_sd,
            "sign_certainty": sign_cert,
        },
        index=names,
    )

    mod = pd.DataFrame(
        [[int(j in members) for j in range(G)] for members in members_k],
        columns=free_terms,
    )
    mod["n_terms"] = mod[free_terms].sum(axis=1)
    mod[crit_name] = (crit_arr if method == "bic" else logw)[idx]
    mod["r2" if family == "gaussian" else "deviance"] = np.asarray(extra)[idx]
    mod["post_prob"] = probs

    settings: Dict[str, Any] = {
        "always": always,
        "prior_inclusion": prior_inclusion,
        "notes": notes,
        "n_candidate_terms": G,
    }
    if method == "bic":
        settings.update({"occam_ratio": occam_ratio, "strict": strict})
    else:
        settings.update({"g": gval, "search": search})
    return BMAResult(
        table=table,
        models=mod,
        params=table["post_mean"].copy(),
        std_errors=table["post_sd"].copy(),
        n_obs=n,
        n_models=len(idx),
        method=method,
        family=family,
        link=link,
        formula=formula,
        settings=settings,
        _fits={"X": X, "probs": probs, "coefs": coefs, "ses": ses},
        _design_info=design_info,
    )


def _mc3(
    log_ml: Callable[[Sequence[int]], float],
    G: int,
    draws: int,
    burnin: int,
    seed: Optional[int],
) -> List[Tuple[int, ...]]:
    """MC3 (Raftery, Madigan and Hoeting 1997): a Metropolis walk on the model space
    that proposes adding or dropping one term. Returns the distinct
    models visited after burn-in."""
    rng = np.random.default_rng(seed)
    state = np.zeros(G, dtype=bool)
    cache: Dict[Tuple[int, ...], float] = {}

    def value(s: np.ndarray) -> float:
        key = tuple(np.flatnonzero(s).tolist())
        if key not in cache:
            cache[key] = log_ml(key)
        return cache[key]

    cur = value(state)
    visited: Dict[Tuple[int, ...], int] = {}
    flips = rng.integers(0, G, size=burnin + draws)
    logu = np.log(rng.random(burnin + draws))
    for it in range(burnin + draws):
        j = flips[it]
        state[j] = ~state[j]
        cand = value(state)
        if logu[it] < cand - cur:
            cur = cand
        else:
            state[j] = ~state[j]
        if it >= burnin:
            key = tuple(np.flatnonzero(state).tolist())
            visited[key] = visited.get(key, 0) + 1
    if len(visited) < 2:
        warnings.warn(
            "MC3 visited a single model; the chain may be stuck. Increase "
            "draws or check the data.",
            ConvergenceWarning,
            stacklevel=3,
        )
    return list(visited.keys())
