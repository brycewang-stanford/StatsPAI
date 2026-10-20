"""Bayesian tree models for causal effects: ``sp.bart_causal``.

Two models on the backfitting engine of :mod:`.trees` (written from the
published models; no BART, bcf, bartCause or stochtree code is used or
ported, those packages are GPL and were used as black boxes only):

``method='bart'`` -- one BART response surface ``f(x, z[, pihat])`` (Hill
2011's approach, the ``bartCause`` default response model). Individual
effects are ``f(x_i, 1) - f(x_i, 0)`` draw by draw. The prior is that of
``dbarts::bart2``, which ``bartCause`` calls, as its documentation states
it: 75 trees, ``k = 2``, cut points equally spaced over each covariate's
range (``useQuantiles = FALSE``).

``method='bcf'`` -- the Bayesian causal forest of Hahn, Murray and Carvalho
(2020) [@hahn2020bayesian]::

    y_i = mu(x_i, pihat_i) + tau(x_i) b_{z_i} + e_i,   e_i ~ N(0, sigma^2)

with a prognostic forest ``mu`` (200 trees, base 0.95, power 2) and a
smaller, more strongly regularised effect forest ``tau`` (50 trees, base
0.25, power 3). Scales: ``mu = eta * mu~`` with ``eta`` half-Cauchy whose
median is ``sd_control`` (2 standard deviations of ``y``), sampled as a
normal scale mixture; ``tau`` enters through ``b_0, b_1 ~ N(0, 1/2)`` so
that the treatment effect ``(b_1 - b_0) tau~(x)`` has a half-normal scale
with median ``sd_moderate`` (one standard deviation of ``y``). The tau
forest sees the residual ``(y - mu) / b_z`` with precision weights
``b_z^2``. The error variance has the scaled inverse chi-square prior
calibrated on a linear fit (``nu = 3``, ``q = 0.9``). These are the
defaults of R ``bcf``'s documentation; details of that package's sampler
(e.g. its exact hyperprior updates) are not reproduced, so the two are
different implementations of one model and are compared as stochastic
screens, never as parity.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ..exceptions import warn as _sp_warn
from .trees import _cutpoints, _Forest

_Q_HALF_NORMAL_MEDIAN = 0.6744897501960817  # median of |N(0, 1)|


class BayesCausalResult(ResultProtocolMixin):
    """Posterior of a Bayesian tree causal model (:func:`sp.bart_causal`).

    Attributes
    ----------
    estimate, se : float
        Posterior mean and standard deviation of the estimand (the sample
        average of the individual effects over the target units).
    ci : tuple
        Equal-tailed credible interval at ``level``.
    draws : pd.Series
        Posterior draws of the estimand (all chains, in chain order).
    cate : pd.DataFrame
        Per-row posterior mean, sd and interval of ``tau(x_i)``.
    diagnostics : dict
        ``rhat`` and ``ess`` of the estimand across chains, mean leaves
        per tree, the propensity model, number of chains.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x = rng.normal(size=n)
    >>> z = rng.binomial(1, 1 / (1 + np.exp(-x)))
    >>> y = x + 1.0 * z + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({'y': y, 'z': z, 'x': x})
    >>> r = sp.bart_causal(df, 'y', 'z', ['x'], method='bcf', draws=200,
    ...                    burnin=200, chains=1, seed=1)
    >>> bool(r.ci[0] < 1.0 < r.ci[1])
    True
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("hahn2020bayesian",)

    def __init__(
        self,
        *,
        method: str,
        estimand: str,
        draws: pd.Series,
        cate: pd.DataFrame,
        level: float,
        n_obs: int,
        diagnostics: Dict[str, Any],
        tau_draws: np.ndarray,
    ) -> None:
        self.method = method
        self.estimand = estimand
        self.draws = draws
        self.estimate = float(draws.mean())
        self.se = float(draws.std(ddof=1))
        lo = (1.0 - level) / 2.0
        self.ci = (float(draws.quantile(lo)), float(draws.quantile(1.0 - lo)))
        self.alpha = 1.0 - level
        self.level = level
        self.pvalue = None
        self.cate = cate
        self.n_obs = n_obs
        self.diagnostics = diagnostics
        self.model_info = {"diagnostics": diagnostics, "method": method}
        self._tau_draws = tau_draws

    @property
    def tau_draws(self) -> np.ndarray:
        """``draws x n`` posterior draws of the individual effects."""
        return self._tau_draws

    def summary(self) -> str:
        d = self.diagnostics
        name = "causal forest" if self.method == "bcf" else "additive regression trees"
        lines = [
            f"Bayesian {name} -- {self.estimand}",
            "=" * 60,
            f"Posterior mean {self.estimate:.4f}  sd {self.se:.4f}  "
            f"{100 * self.level:.0f}% interval [{self.ci[0]:.4f}, {self.ci[1]:.4f}]",
            f"Observations {self.n_obs}; chains {d['chains']}; draws per chain "
            f"{d['draws']}; R-hat {d['rhat']:.3f}; ESS {d['ess']:.0f}",
            f"Propensity: {d['propensity']}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _uniform_cutpoints(x: np.ndarray, numcut: int) -> np.ndarray:
    """``numcut`` points equally spaced inside the range of ``x`` (the
    midpoints between its values when it has at most ``numcut + 1``)."""
    u = np.unique(x)
    if u.size <= 1:
        return np.empty(0)
    if u.size <= numcut + 1:
        return np.asarray(0.5 * (u[1:] + u[:-1]))
    lo, hi = float(u[0]), float(u[-1])
    return lo + (hi - lo) * np.arange(1, numcut + 1) / (numcut + 1)


def _cut_design(
    X: np.ndarray, numcut: int, uniform: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    grid = _uniform_cutpoints if uniform else _cutpoints
    cuts = [grid(X[:, v], int(numcut)) for v in range(X.shape[1])]
    ncut = np.array([c.size for c in cuts], dtype=np.int64)
    xcut = np.column_stack(
        [np.searchsorted(c, X[:, v], side="left") for v, c in enumerate(cuts)]
    )
    return xcut, ncut, cuts


def _sigma_prior(
    ys: np.ndarray, Z: np.ndarray, nu: float, q: float
) -> Tuple[float, float]:
    n = len(ys)
    if n > Z.shape[1] + 1 and np.linalg.matrix_rank(Z) == Z.shape[1]:
        res = ys - Z @ np.linalg.lstsq(Z, ys, rcond=None)[0]
        s2 = float(res @ res / (n - Z.shape[1]))
    else:  # pragma: no cover - more columns than rows
        s2 = float(ys.var(ddof=1))
    return s2, s2 * stats.chi2.ppf(1.0 - q, nu) / nu


def _propensity(
    data: pd.DataFrame, rows: np.ndarray, z: np.ndarray, X: np.ndarray, propensity: Any
) -> Tuple[Optional[np.ndarray], str]:
    if propensity is None or (isinstance(propensity, str) and propensity == "none"):
        return None, "none"
    if isinstance(propensity, str) and propensity == "logit":
        from ..matching._binary_fit import fit_binary_index

        fit = fit_binary_index(X, z, "logit")
        eta = np.column_stack([np.ones(len(z)), X]) @ fit["beta"]
        return 1.0 / (1.0 + np.exp(-eta)), "logit on the covariates"
    if isinstance(propensity, str):
        if propensity not in data.columns:
            raise MethodIncompatibility(
                f"bart_causal: propensity column {propensity!r} not found.",
                recovery_hint="Pass 'logit', 'none' or a column of scores.",
            )
        ps = data.loc[rows, propensity].to_numpy(dtype=float)
        src = f"column {propensity!r}"
    else:
        ps = np.asarray(propensity, dtype=float).ravel()
        if len(ps) != len(data):
            raise MethodIncompatibility("bart_causal: propensity has the wrong length.")
        ps = ps[rows]
        src = "user array"
    if not np.all((ps > 0) & (ps < 1)):
        raise MethodIncompatibility(
            "bart_causal: propensity scores must lie strictly in (0, 1).",
            recovery_hint="Pass fitted probabilities, not labels.",
        )
    return ps, src


def _rhat_ess(chains: np.ndarray) -> Tuple[float, float]:
    """Split-R-hat and a simple ESS (batch-means-free autocorrelation sum)
    of a ``chains x draws`` array."""
    m, n = chains.shape
    if n < 4:
        return float("nan"), float("nan")
    half = n // 2
    split = np.vstack([chains[:, :half], chains[:, half : 2 * half]])
    mm, nn = split.shape
    means = split.mean(axis=1)
    W = split.var(axis=1, ddof=1).mean()
    B = nn * means.var(ddof=1)
    var_hat = (nn - 1) / nn * W + B / nn
    rhat = float(np.sqrt(var_hat / W)) if W > 0 else float("nan")
    ess = 0.0
    for c in chains:
        x = c - c.mean()
        v = float(x @ x) / n
        if v <= 0:
            continue
        rho_sum = 0.0
        for lag in range(1, n // 2):
            rho = float(x[:-lag] @ x[lag:]) / (n * v)
            if rho < 0.05:
                break
            rho_sum += rho
        ess += n / (1.0 + 2.0 * rho_sum)
    return rhat, ess


def bart_causal(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    *,
    method: str = "bcf",
    estimand: str = "ATE",
    propensity: Any = "logit",
    draws: int = 1000,
    burnin: int = 1000,
    thin: int = 1,
    chains: int = 4,
    n_trees_control: Optional[int] = None,
    n_trees_moderate: int = 50,
    sd_control: float = 2.0,
    sd_moderate: float = 1.0,
    numcut: int = 100,
    min_leaf: int = 5,
    seed: Optional[int] = None,
    level: float = 0.95,
    cut_grid: str = "auto",
) -> BayesCausalResult:
    """Bayesian tree models for a binary treatment: BCF or BART.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Continuous outcome.
    treat : str
        0/1 treatment.
    covariates : sequence of str
        Numeric confounders / moderators.
    method : {'bcf', 'bart'}, default 'bcf'
        ``'bcf'``: Bayesian causal forest (separate prognostic and
        effect forests, effect forest strongly regularised).
        ``'bart'``: one BART response surface of the outcome on the
        covariates, the treatment and (by default) the propensity score;
        effects are differences of its predictions (``bartCause``'s
        default response model).
    estimand : {'ATE', 'ATT', 'ATC'}, default 'ATE'
        The sample average of the individual effects over all units, the
        treated or the controls.
    propensity : {'logit', 'none'}, str or array, default 'logit'
        The propensity score used as a covariate of the prognostic forest
        (``'bcf'``) or the response surface (``'bart'``): a logit on the
        covariates, none, a column name, or one score per row.
    draws, burnin, thin, chains : int
        Posterior draws kept per chain, burn-in sweeps, thinning, and
        independent chains (each with its own seed; R-hat needs >= 2).
    n_trees_control, n_trees_moderate : int, default None, 50
        Trees in the prognostic (``'bart'``: the only) forest and in the
        effect forest. ``None``: 200 for ``'bcf'`` (R ``bcf``), 75 for
        ``'bart'`` (``dbarts::bart2``, which ``bartCause`` uses).
    sd_control, sd_moderate : float, default 2.0, 1.0
        Prior medians of the scale of the prognostic and effect functions,
        in standard deviations of ``y`` (``'bcf'``).
    numcut : int, default 100
        Maximum number of cut points per covariate.
    min_leaf : int, default 5
        Smallest number of observations in a leaf.
    seed : int, optional
    level : float, default 0.95
    cut_grid : {'auto', 'quantile', 'uniform'}, default 'auto'
        Candidate split points of a covariate with more than ``numcut + 1``
        distinct values: ``numcut`` sample quantiles, or ``numcut`` points
        equally spaced over its range (fewer distinct values: the
        midpoints between them). ``'auto'`` is ``'quantile'`` for
        ``'bcf'`` and ``'uniform'`` for ``'bart'`` (``dbarts``'s
        ``useQuantiles = FALSE``). The grid is part of the prior: with
        quantile cuts and 200 trees the BART posterior mean of the ATE sat
        0.21 posterior SDs from ``bartCause``'s, with its own prior 0.02.

    Returns
    -------
    BayesCausalResult

    Notes
    -----
    Evidence. The samplers are checked against known truth on simulated
    designs (``tests/test_bart_causal.py``: posterior intervals cover the
    true ATE and the BCF effect forest recovers effect heterogeneity), and
    against R ``bcf`` 2.0.2 / ``bartCause`` 1.0.10 by seed replication on a
    fixed data set: several seeds on each side, comparing the
    distributions of the posterior means. That is a stochastic screen
    (CLAUDE.md §5.1, tier S), not parity; the two sides implement the same
    model with different samplers.

    ``sp.bcf`` (the random-forest / gradient-boosting approximation) is
    unchanged; this function is the posterior-sampling BCF.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(2)
    >>> n = 200
    >>> x = rng.normal(size=n)
    >>> z = rng.binomial(1, 0.5, size=n)
    >>> y = x + z * (1 + (x > 0)) + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({'y': y, 'z': z, 'x': x})
    >>> r = sp.bart_causal(df, 'y', 'z', ['x'], method='bart', draws=150,
    ...                    burnin=150, chains=1, seed=3)
    >>> r.cate.shape[0]
    200

    References
    ----------
    [@hahn2020bayesian]
    """
    method = str(method).lower()
    if cut_grid not in ("auto", "quantile", "uniform"):
        raise MethodIncompatibility(
            f"bart_causal: cut_grid must be 'auto', 'quantile' or 'uniform'; "
            f"got {cut_grid!r}.",
            recovery_hint="Use cut_grid='auto'.",
        )
    if method not in ("bcf", "bart"):
        raise MethodIncompatibility(
            f"bart_causal: method must be 'bcf' or 'bart'; got {method!r}.",
            recovery_hint="Use method='bcf' or 'bart'.",
        )
    estimand = str(estimand).upper()
    if estimand not in ("ATE", "ATT", "ATC"):
        raise MethodIncompatibility(
            f"bart_causal: estimand must be 'ATE', 'ATT' or 'ATC'; got {estimand!r}.",
            recovery_hint="Use estimand='ATE'.",
        )
    if draws < 10 or burnin < 0 or thin < 1 or chains < 1:
        raise MethodIncompatibility(
            "bart_causal: draws >= 10, burnin >= 0, thin >= 1, chains >= 1.",
            recovery_hint="Use the defaults (draws=1000, burnin=1000, chains=4).",
        )
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(
            f"bart_causal: level must be in (0, 1); got {level}."
        )
    if sd_control <= 0 or sd_moderate <= 0:
        raise MethodIncompatibility(
            "bart_causal: sd_control and sd_moderate must be positive."
        )
    covariates = list(covariates)
    missing = [c for c in [y, treat] + covariates if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"bart_causal: columns {missing} not found.",
            diagnostics={"missing": missing},
        )
    rows = data[[y, treat] + covariates].notna().all(axis=1).to_numpy()
    df = data.loc[rows]
    yv = df[y].to_numpy(dtype=float)
    z = df[treat].to_numpy(dtype=float)
    if not np.isin(z, (0.0, 1.0)).all() or z.min() == z.max():
        raise MethodIncompatibility(
            "bart_causal: the treatment must be 0/1 with both values present.",
            recovery_hint="Recode the treatment to 0/1.",
        )
    X = df[covariates].to_numpy(dtype=float)
    n = len(yv)
    if n < 20:
        raise DataInsufficient(f"bart_causal: {n} complete rows are too few.")
    if np.ptp(yv) == 0:
        raise MethodIncompatibility("bart_causal: the outcome does not vary.")
    ps, ps_src = _propensity(data, rows, z, X, propensity)
    if n_trees_control is None:
        n_trees_control = 200 if method == "bcf" else 75
    uniform = cut_grid == "uniform" or (cut_grid == "auto" and method == "bart")
    target = {"ATE": np.ones(n, bool), "ATT": z == 1, "ATC": z == 0}[estimand]

    rng = np.random.default_rng(seed)
    chain_seeds = rng.integers(0, 2**31 - 1, size=int(chains))
    tau_all: List[np.ndarray] = []
    est_chains = np.empty((int(chains), int(draws)))
    leaves: List[float] = []
    for c, cs in enumerate(chain_seeds):
        crng = np.random.default_rng(int(cs))
        if method == "bcf":
            tau, nl = _bcf_chain(
                yv,
                z,
                X,
                ps,
                crng,
                int(draws),
                int(burnin),
                int(thin),
                int(n_trees_control),
                int(n_trees_moderate),
                float(sd_control),
                float(sd_moderate),
                int(numcut),
                int(min_leaf),
                uniform,
            )
        else:
            tau, nl = _bart_chain(
                yv,
                z,
                X,
                ps,
                crng,
                int(draws),
                int(burnin),
                int(thin),
                int(n_trees_control),
                int(numcut),
                int(min_leaf),
                uniform,
            )
        tau_all.append(tau)
        est_chains[c] = tau[:, target].mean(axis=1)
        leaves.append(nl)
    tau_draws = np.vstack(tau_all)
    # split R-hat (a single chain is split in halves)
    rhat, ess = _rhat_ess(est_chains)
    lo = (1.0 - level) / 2.0
    cate = pd.DataFrame(
        {
            "mean": tau_draws.mean(axis=0),
            "sd": tau_draws.std(axis=0, ddof=1),
            "lower": np.quantile(tau_draws, lo, axis=0),
            "upper": np.quantile(tau_draws, 1.0 - lo, axis=0),
        },
        index=df.index,
    )
    diag = {
        "rhat": rhat,
        "ess": ess,
        "chains": int(chains),
        "draws": int(draws),
        "burnin": int(burnin),
        "thin": int(thin),
        "leaves_per_tree": float(np.mean(leaves)),
        "propensity": ps_src,
        "n_dropped": int((~rows).sum()),
        "n_trees_control": int(n_trees_control),
        "cut_grid": "uniform" if uniform else "quantile",
    }
    if np.isfinite(rhat) and rhat > 1.05:
        _sp_warn(
            ConvergenceWarning,
            f"bart_causal: R-hat of the {estimand} is {rhat:.3f} across chains.",
            recovery_hint="Increase burnin and draws.",
            stacklevel=2,
        )
    return BayesCausalResult(
        method=method,
        estimand=estimand,
        draws=pd.Series(est_chains.ravel(), name=estimand),
        cate=cate,
        level=level,
        n_obs=n,
        diagnostics=diag,
        tau_draws=tau_draws,
    )


def _bcf_chain(
    yv,
    z,
    X,
    ps,
    rng,
    draws,
    burnin,
    thin,
    m_mu,
    m_tau,
    sd_control,
    sd_moderate,
    numcut,
    min_leaf,
    uniform=False,
) -> Tuple[np.ndarray, float]:
    n = len(yv)
    ybar, sy = float(yv.mean()), float(yv.std(ddof=1))
    ys = (yv - ybar) / sy
    Xmu = X if ps is None else np.column_stack([X, ps])
    xc_mu, nc_mu, _ = _cut_design(Xmu, numcut, uniform)
    xc_tau, nc_tau, _ = _cut_design(X, numcut, uniform)
    f_mu = _Forest(xc_mu, nc_mu, m_mu, 1.0 / m_mu, 0.95, 2.0, 8, min_leaf)
    tau2_tau = (sd_moderate / _Q_HALF_NORMAL_MEDIAN) ** 2 / m_tau
    f_tau = _Forest(xc_tau, nc_tau, m_tau, tau2_tau, 0.25, 3.0, 8, min_leaf)
    f_mu.k["seed"](int(rng.integers(0, 2**31 - 1)))
    nu, q = 3.0, 0.9
    s2_hat, lam = _sigma_prior(ys, np.column_stack([np.ones(n), X, z]), nu, q)
    sigma2 = s2_hat
    eta, v = 1.0, sd_control**2
    b = np.array([-0.5, 0.5])
    zi = z.astype(int)
    out = np.empty((draws, n))
    nl = 0.0
    kept = 0
    for it in range(burnin + draws * thin):
        bz = b[zi]
        # prognostic forest on (y - b_z tau~) / eta, noise sigma^2 / eta^2
        r_mu = ys - bz * f_tau.total
        f_mu.sweep(r_mu / eta, sigma2 / eta**2)
        fm = f_mu.total
        prec = float(fm @ fm) / sigma2 + 1.0 / v
        eta = rng.normal(float(fm @ r_mu) / sigma2 / prec, 1.0 / math.sqrt(prec))
        v = 0.5 * (eta**2 + sd_control**2) / rng.gamma(1.0)
        # effect forest on (y - mu) / b_z with precision weights b_z^2
        r_tau = ys - eta * fm
        f_tau.sweep(r_tau / bz, sigma2, bz**2)
        t = f_tau.total
        for g in (0, 1):
            sel = zi == g
            pg = float(t[sel] @ t[sel]) / sigma2 + 2.0
            b[g] = rng.normal(
                float(t[sel] @ r_tau[sel]) / sigma2 / pg, 1.0 / math.sqrt(pg)
            )
        resid = ys - eta * fm - b[zi] * t
        sigma2 = (nu * lam + float(resid @ resid)) / rng.chisquare(nu + n)
        if it >= burnin and (it - burnin) % thin == 0:
            out[kept] = (b[1] - b[0]) * t * sy
            nl += float(f_mu.nleaf.mean() + f_tau.nleaf.mean()) / 2.0
            kept += 1
    return out, nl / max(kept, 1)


def _bart_chain(
    yv, z, X, ps, rng, draws, burnin, thin, m, numcut, min_leaf, uniform=False
) -> Tuple[np.ndarray, float]:
    n = len(yv)
    cols = [X, z[:, None]] + ([] if ps is None else [ps[:, None]])
    Xf = np.column_stack(cols)
    xcut, ncut, cuts = _cut_design(Xf, numcut, uniform)
    zcol = X.shape[1]
    xc1 = xcut.copy()
    xc0 = xcut.copy()
    xc1[:, zcol] = np.searchsorted(cuts[zcol], np.ones(n), side="left")
    xc0[:, zcol] = np.searchsorted(cuts[zcol], np.zeros(n), side="left")
    xc1 = np.ascontiguousarray(xc1, dtype=np.int64)
    xc0 = np.ascontiguousarray(xc0, dtype=np.int64)
    scale = float(np.ptp(yv))
    center = float(0.5 * (yv.max() + yv.min()))
    ys = (yv - center) / scale
    k = 2.0
    forest = _Forest(
        xcut, ncut, m, (0.5 / (k * math.sqrt(m))) ** 2, 0.95, 2.0, 8, min_leaf
    )
    forest.k["seed"](int(rng.integers(0, 2**31 - 1)))
    nu, q = 3.0, 0.9
    s2_hat, lam = _sigma_prior(ys, np.column_stack([np.ones(n), Xf]), nu, q)
    sigma2 = s2_hat
    pred = forest.k["predict"]
    out = np.empty((draws, n))
    nl = 0.0
    kept = 0
    for it in range(burnin + draws * thin):
        forest.sweep(ys, sigma2)
        resid = ys - forest.total
        sigma2 = (nu * lam + float(resid @ resid)) / rng.chisquare(nu + n)
        if it >= burnin and (it - burnin) % thin == 0:
            out[kept] = scale * (
                pred(xc1, forest.var, forest.cut, forest.val)
                - pred(xc0, forest.var, forest.cut, forest.val)
            )
            nl += float(forest.nleaf.mean())
            kept += 1
    return out, nl / max(kept, 1)
