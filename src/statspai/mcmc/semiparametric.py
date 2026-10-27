"""
Locally robust, semiparametrically efficient Bayesian inference:
``sp.semiparametric_bayes``.

A baseline parametric model ``p(y | theta)``, ``theta = (gamma, zeta)``,
whose parameter of interest ``gamma`` keeps its meaning in a larger
semiparametric model (a Student-t regression inside the linear model with
``E[e | x] = 0``; a Weibull duration model inside ``E[y | x] = exp(x'b)``).
Under local misspecification of the baseline the posterior of ``gamma`` is
biased by as much as its own spread. Mueller and Norets (2026) augment the
likelihood with an auxiliary parameter ``delta`` of the dimension of
``gamma``::

    q(y | theta, delta) = c(theta, delta) k0(delta' ldot_d(y)) p(y | theta)
    ldot_d = ldot_gamma - ltilde_gamma,       k0(z) = 2 / (1 + exp(-2 z))

``ldot_gamma`` is the baseline score of ``gamma`` and ``ltilde_gamma`` the
efficient score of the semiparametric model. ``delta = 0`` is the baseline
model; the marginal posterior of ``gamma`` in the augmented model is
asymptotically normal, centred at a semiparametrically efficient estimator
with variance at the efficiency bound, under any local misspecification.

The normalising constant ``c(theta, delta)`` is avoided as in Rao, Lin and
Dunson (2016): each observation is treated as the accepted draw of an
acceptance sampler with proposal ``p(. | theta)`` and acceptance
probability ``k0 / 2``, and the rejected proposals are latent variables.
The sampler alternates (i) Metropolis moves of ``(theta, delta)`` given the
rejected proposals and (ii) a fresh run of the acceptance sampler for every
observation. The paper moves block (i) with Hamiltonian Monte Carlo; here
it is a random walk whose covariance is tuned during burn-in only, so the
retained chain is a fixed-kernel Markov chain with the same target.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, special

from ..core.utils import create_design_matrices
from ..exceptions import (
    AssumptionWarning,
    ConvergenceFailure,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from ._core import (
    check_mcmc_args,
    find_mode,
    is_default,
    metropolis_mixture,
    note_default_prior,
    random_walk_metropolis,
    rows_by_position,
    spawn_rngs,
)
from ._results import posterior_table
from ._semiparametric_models import (
    _SCORE_HINT,
    _custom_spec,
    _gamma_qmle,
    _log_k0,
    _log_one_minus_half_k0,
    _norm_logpdf,
    _orthogonalise,
    _Spec,
    _student_t_spec,
    _weibull_spec,
)
from .regress import BayesRegressResult

#: Metropolis moves of (theta, delta) per refresh of the rejected proposals
_MH_STEPS = 6
#: the acceptance sampler gives up after this many rejections per observation
_MAX_REJECT_PER_OBS = 50
#: proposals drawn per pending observation in each round of that sampler
_BATCH = 4


class SemiparametricBayesResult(BayesRegressResult):
    """Posterior of :func:`semiparametric_bayes`.

    A :class:`BayesRegressResult` whose ``draws`` hold the baseline
    parameters (natural scale) followed by the augmentation parameters
    ``delta:<name>``. ``diagnostics_info['comparison']`` sets the
    augmented posterior of the parameters of interest beside the baseline
    posterior and the semiparametrically efficient estimator.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=200)})
    >>> df["y"] = 1 + df["x"] + rng.standard_t(4, size=200)
    >>> fit = sp.semiparametric_bayes("y ~ x", df, draws=300, burnin=150,
    ...                               seed=1, baseline=False)
    >>> type(fit).__name__
    'SemiparametricBayesResult'
    """

    _citation_keys = ("mueller2026locally", "rao2016data")

    @property
    def method(self) -> str:
        return (
            "Locally robust semiparametric Bayes "
            f"({self.model_info.get('baseline_model', self.model)})"
        )


# --------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------


def _rejected_proposals(
    spec: _Spec, u: np.ndarray, d: np.ndarray, X: np.ndarray, rng
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the acceptance sampler once per observation; return the row
    index of every rejected proposal and the proposals themselves.

    Each round draws ``_BATCH`` proposals per pending row in one call and
    keeps those before the first acceptance: the same sequence of
    independent trials, with the draws after the acceptance unused.
    """
    n = X.shape[0]
    pending = np.arange(n)
    ys: List[np.ndarray] = []
    idx: List[np.ndarray] = []
    total = 0
    while pending.size:
        rows = np.repeat(pending, _BATCH)
        Xp = X[rows]
        cand = spec.simulate(u, Xp, rng)
        z = spec.ldelta(u, cand, Xp) @ d
        accept = (rng.random(rows.size) < special.expit(2.0 * z)).reshape(-1, _BATCH)
        done = accept.any(axis=1)
        first = np.where(done, accept.argmax(axis=1), _BATCH)
        keep = (np.arange(_BATCH)[None, :] < first[:, None]).reshape(-1)
        ys.append(cand[keep])
        idx.append(rows[keep])
        total += int(keep.sum())
        if total > _MAX_REJECT_PER_OBS * n:
            raise ConvergenceFailure(
                "The acceptance sampler of the augmented model rejects almost "
                "every proposal: the augmentation parameter delta has moved "
                "far from zero, so the baseline model is grossly misspecified.",
                recovery_hint=(
                    "Revisit the baseline model, or shrink delta with a smaller "
                    "delta_prior_sd."
                ),
                diagnostics={"delta": d.tolist()},
            )
        pending = pending[~done]
    return np.concatenate(idx), np.concatenate(ys)


def _aug_log_post(spec, v, m, n, y_all, X_all, delta_sd) -> float:
    """Log posterior of (theta, delta) given the rejected proposals.

    ``y_all`` stacks the ``n`` observations and then the rejected
    proposals, ``X_all`` their design rows.
    """
    u, d = v[:m], v[m:]
    lp = spec.log_prior(u) + _norm_logpdf(d, 0.0, delta_sd)
    if not np.isfinite(lp):
        return -np.inf
    with np.errstate(all="ignore"):
        z = spec.ldelta(u, y_all, X_all) @ d
        val = (
            spec.loglik(u, y_all, X_all).sum()
            + _log_k0(z[:n]).sum()
            + _log_one_minus_half_k0(z[n:]).sum()
        )
    val = float(val) + lp
    return val if np.isfinite(val) else -np.inf


def _initial_cov(spec, u0, y, X, prior_sd, delta_sd) -> np.ndarray:
    """Outer product of the per-observation scores of (theta, delta) at
    (u0, 0), plus the prior precision: the starting proposal."""
    m = u0.size
    G = np.empty((y.shape[0], m))
    for j in range(m):
        h = 1e-5 * max(1.0, abs(u0[j]))
        e = np.zeros(m)
        e[j] = h
        G[:, j] = (spec.loglik(u0 + e, y, X) - spec.loglik(u0 - e, y, X)) / (2 * h)
    S = np.column_stack([G, spec.ldelta(u0, y, X)])
    J = S.T @ S
    dim = m + spec.k
    J[np.arange(m), np.arange(m)] += 1.0 / prior_sd**2
    J[np.arange(m, dim), np.arange(m, dim)] += 1.0 / delta_sd**2
    try:
        cov = linalg.inv(0.5 * (J + J.T))
        linalg.cholesky(cov, lower=True)
    except linalg.LinAlgError:
        cov = np.diag(1.0 / np.maximum(np.diag(J), 1e-12))
    return 0.5 * (cov + cov.T)


def _augmented_chain(
    spec,
    rng,
    start,
    y,
    X,
    n_iter,
    burnin,
    thin,
    draws,
    cov0,
    delta_sd,
    mh_steps=_MH_STEPS,
) -> Tuple[np.ndarray, float, float]:
    m = spec.start.size
    n_obs = y.shape[0]
    dim = start.size
    cur = start.copy()
    scale = 2.38 / np.sqrt(dim)
    cov = cov0
    out = np.empty((draws, dim))
    kept = 0
    acc_sum, acc_n, rej_sum = 0.0, 0, 0.0
    hist = np.empty((max(burnin, 1), dim))
    window_acc: List[float] = []
    checkpoints = (
        {int(burnin * f) for f in (0.2, 0.4, 0.6, 0.8)} if burnin >= 50 else set()
    )
    for it in range(n_iter):
        ridx, yt = _rejected_proposals(spec, cur[:m], cur[m:], X, rng)
        y_all = np.concatenate([y, yt])
        X_all = np.vstack([X, X[ridx]])

        def target(v):
            return _aug_log_post(spec, v, m, n_obs, y_all, X_all, delta_sd)

        states, acc = random_walk_metropolis(rng, target, cur, scale**2 * cov, mh_steps)
        cur = states[-1]
        if it < burnin:
            hist[it] = cur
            window_acc.append(acc)
            if it + 1 in checkpoints:
                rate = float(np.mean(window_acc))
                window_acc = []
                if rate < 0.15:
                    scale *= 0.7
                elif rate > 0.45:
                    scale *= 1.3
                emp = np.cov(hist[: it + 1][(it + 1) // 2 :].T)
                if np.all(np.isfinite(emp)) and np.linalg.matrix_rank(emp) == dim:
                    cov = 0.5 * (emp + emp.T) + 1e-10 * np.eye(dim)
            continue
        acc_sum += acc
        acc_n += 1
        rej_sum += yt.size / y.shape[0]
        if (it - burnin) % thin == 0 and kept < draws:
            out[kept] = cur
            kept += 1
    return out, acc_sum / max(acc_n, 1), rej_sum / max(acc_n, 1)


# --------------------------------------------------------------------------
# Efficient estimators for the comparison
# --------------------------------------------------------------------------


def _efficient_estimate(
    model, spec, y, X0, names, base_mean_u
) -> Tuple[np.ndarray, np.ndarray, str]:
    n, p = X0.shape
    if model == "student_t":
        b, *_ = np.linalg.lstsq(X0, y, rcond=None)
        r = y - X0 @ b
        s2 = float(r @ r) / (n - p)
        V = s2 * linalg.inv(X0.T @ X0)
        pos = [names.index(nm) for nm in spec.interest]
        return (
            b[pos],
            np.sqrt(np.diag(V)[pos]),
            ("OLS with the homoskedastic variance s^2 (X'X)^-1"),
        )
    if model == "weibull":
        b = _gamma_qmle(y, X0)
        r = y * np.exp(-X0 @ b) - 1.0
        c = float(r @ r) / (n - p)
        V = c * linalg.inv(X0.T @ X0)
        pos = [names.index(nm) for nm in spec.interest]
        return (
            b[pos],
            np.sqrt(np.diag(V)[pos]),
            (
                "solution of sum x (y exp(-x'b) - 1) = 0 with variance c (X'X)^-1, "
                "c the mean square of y exp(-x'b) - 1"
            ),
        )
    # one Newton step on the efficient score from the baseline posterior
    # mean, Jacobian by central differences, sandwich variance
    ipos = spec.interest_pos_u
    L = spec.efficient_score(base_mean_u, y, X0)
    J = np.empty((len(ipos), len(ipos)))
    for c, j in enumerate(ipos):
        h = 1e-5 * max(1.0, abs(base_mean_u[j]))
        e = np.zeros(base_mean_u.size)
        e[j] = h
        up = spec.efficient_score(base_mean_u + e, y, X0).sum(axis=0)
        dn = spec.efficient_score(base_mean_u - e, y, X0).sum(axis=0)
        J[:, c] = -(up - dn) / (2 * h)
    try:
        Ji = linalg.inv(J)
    except linalg.LinAlgError as exc:
        raise MethodIncompatibility(
            "The efficient score does not depend on the parameters of "
            "interest at the baseline posterior mean (singular Jacobian).",
            recovery_hint=_SCORE_HINT,
        ) from exc
    est = base_mean_u[ipos] + Ji @ L.sum(axis=0)
    V = Ji @ (L.T @ L) @ Ji.T
    return (
        est,
        np.sqrt(np.diag(V)),
        (
            "one Newton step on the efficient score from the baseline posterior "
            "mean, with the sandwich variance"
        ),
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def semiparametric_bayes(
    formula: str,
    data: pd.DataFrame,
    model: Literal["student_t", "weibull", "custom"] = "student_t",
    interest: Optional[Sequence[Any]] = None,
    prior_sd: float = 10.0,
    delta_prior_sd: float = 0.2,
    loglik: Optional[Callable[..., Any]] = None,
    score: Optional[Callable[..., Any]] = None,
    efficient_score: Optional[Callable[..., Any]] = None,
    simulate: Optional[Callable[..., Any]] = None,
    log_prior: Optional[Callable[..., Any]] = None,
    theta_start: Any = None,
    param_names: Optional[Sequence[str]] = None,
    draws: int = 4000,
    burnin: int = 1500,
    thin: int = 1,
    chains: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
    baseline: bool = True,
) -> SemiparametricBayesResult:
    """Locally robust, semiparametrically efficient Bayesian inference.

    Bayesian inference in a parametric baseline model (Student-t or
    Weibull regression, or any model you supply) made robust to local
    misspecification. The likelihood is augmented by a factor in the
    baseline and efficient scores of the parameters of interest and an
    auxiliary parameter ``delta`` of the same dimension (Mueller and
    Norets 2026). In large samples the posterior of the parameters of
    interest is normal, centred at the semiparametrically efficient
    estimator with variance at the efficiency bound, whether or not the
    baseline distribution is exactly right, while the rest of the model
    (tails, quantiles, predictions) still comes from the baseline.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2"``. The outcome and the linear index (or, for
        ``model='custom'``, the design ``X`` handed to your callables).
    data : DataFrame
    model : {'student_t', 'weibull', 'custom'}
        ``'student_t'``: ``y = x'b + sigma t_nu``, embedded in the linear
        model with ``E[e | x] = 0`` and homoskedastic errors (efficient
        estimator OLS). ``'weibull'``: Weibull durations with
        ``E[y | x] = exp(x'b)``, embedded in that conditional mean
        restriction (efficient estimator the weighted nonlinear least
        squares of the paper). ``'custom'``: your own callables.
    interest : list, optional
        Coefficients of interest (names or positions); default all
        coefficients. The other coefficients are nuisance parameters: the
        regressors of interest are replaced by their residuals on the
        others while sampling (the index is unchanged) and every
        coefficient is reported on its original scale. For
        ``model='custom'``, the names (``param_names``) or positions in
        ``theta`` of the parameters whose scores you return; default the
        first ``k``.
    prior_sd : float
        Built-in models: ``N(0, prior_sd^2)`` priors on the coefficients
        and on ``log(sigma)`` / ``log(alpha)``; ``log(nu - 2) ~
        N(log 2, 1)``. The paper's choices.
    delta_prior_sd : float
        ``delta ~ N(0, delta_prior_sd^2)`` per coordinate (paper: 0.2). A
        tiny value gives back the baseline posterior.
    loglik, score, efficient_score, simulate, log_prior : callable
        ``model='custom'`` only. ``loglik(theta, y, X)`` returns the
        baseline log density of every row; ``score(theta, y, X)`` and
        ``efficient_score(theta, y, X)`` the ``(n, k)`` baseline and
        efficient scores of the parameters of interest;
        ``simulate(theta, X, rng)`` one baseline draw per row of ``X``;
        ``log_prior(theta)`` a scalar. ``y`` may be the observed outcome
        or simulated proposals, with the matching rows of ``X``.
    theta_start : array, optional
        ``model='custom'``: starting value (the baseline posterior mode is
        searched from it).
    param_names : list of str, optional
        ``model='custom'``: names of the entries of ``theta``.
    draws, burnin, thin, chains, seed, level
        As in :func:`statspai.bayes_regress`. The proposal covariance is
        tuned during burn-in only.
    baseline : bool
        Also sample the baseline (non-robust) posterior, for the
        comparison in ``diagnostics_info``.

    Returns
    -------
    SemiparametricBayesResult
        Posterior table of the baseline parameters and of
        ``delta:<name>``. ``diagnostics_info`` has ``comparison``
        (efficient estimate and standard error, baseline and augmented
        posterior mean and standard deviation of every parameter of
        interest), ``efficient_method``, ``delta_excludes_zero`` (a
        credible interval of ``delta`` that excludes zero is evidence that
        the baseline is misspecified), ``rejected_per_obs``,
        ``max_split_rhat``, ``min_ess`` and ``warnings``.

    Notes
    -----
    The posterior of the augmented model is sampled with the rejected
    proposals of an acceptance sampler as latent variables (Rao, Lin and
    Dunson 2016), so the normalising constant of the augmented density is
    never computed. The paper moves ``(theta, delta)`` by Hamiltonian
    Monte Carlo; this implementation uses a random walk with a covariance
    tuned in burn-in. The guarantee is asymptotic and local: under gross
    misspecification ``delta`` drifts away from zero, which is reported
    rather than corrected. For a custom model the efficient estimator in
    the comparison is one Newton step on the efficient score from the
    baseline posterior mean, with the efficiency bound estimated by the
    outer product of the efficient score.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=200)})
    >>> df["y"] = 1 + df["x"] + rng.standard_t(4, size=200)
    >>> fit = sp.semiparametric_bayes("y ~ x", df, draws=300, burnin=150,
    ...                               seed=1, baseline=False)
    >>> list(fit.params.index)
    ['Intercept', 'x', 'sigma', 'nu', 'delta:Intercept', 'delta:x']
    >>> list(fit.diagnostics_info["comparison"].columns)[:2]
    ['efficient', 'efficient_se']

    References
    ----------
    [@mueller2026locally], [@rao2016data]
    """
    check_mcmc_args(draws, burnin, thin, chains)
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    if model not in ("student_t", "weibull", "custom"):
        raise MethodIncompatibility(
            f"model must be 'student_t', 'weibull' or 'custom'; got {model!r}."
        )
    for nm, val in (("prior_sd", prior_sd), ("delta_prior_sd", delta_prior_sd)):
        if not (np.isfinite(val) and val > 0):
            raise MethodIncompatibility(f"{nm} must be positive; got {val!r}.")
    data = rows_by_position(data)
    y_df, X_df = create_design_matrices(formula, data)
    y = np.asarray(y_df, dtype=float).reshape(-1)
    X0 = np.asarray(X_df, dtype=float)
    names = [str(c) for c in X_df.columns]
    n, p = X0.shape
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(X0))):
        raise MethodIncompatibility(
            "The outcome or a regressor has infinite values.",
            recovery_hint="Drop or recode the rows with infinite values.",
        )
    if model != "custom":
        if n <= p + 2:
            raise DataInsufficient(f"{n} observations for {p} coefficients.")
        if np.linalg.matrix_rank(X0) < p:
            raise MethodIncompatibility(
                "The regressors are collinear.",
                recovery_hint="Drop the redundant regressors.",
            )
    if model == "weibull" and np.any(y <= 0):
        raise MethodIncompatibility(
            "model='weibull' needs a positive outcome (a duration).",
            recovery_hint="Drop the non-positive durations or use model='student_t'.",
        )
    X = X0
    if model == "custom":
        spec = _custom_spec(
            y,
            X0,
            loglik,
            score,
            efficient_score,
            simulate,
            log_prior,
            theta_start,
            param_names,
            interest,
        )
    else:
        X, pos, Pi = _orthogonalise(X0, names, interest)
        maker = _student_t_spec if model == "student_t" else _weibull_spec
        spec = maker(y, X, names, pos, Pi, prior_sd)

    def neg_base(u):
        with np.errstate(all="ignore"):
            v = -(spec.loglik(u, y, X).sum() + spec.log_prior(u))
        return float(v) if np.isfinite(v) else 1e300

    mode, lap_cov = find_mode(neg_base, spec.start, what="baseline posterior")
    m = mode.size
    n_iter = burnin + draws * thin
    rngs = spawn_rngs(seed, chains + 1)
    cov0 = _initial_cov(spec, mode, y, X, prior_sd, delta_prior_sd)
    pieces, accs, rejs = [], [], []
    for ch in range(chains):
        start = np.concatenate([mode, np.zeros(spec.k)])
        if ch > 0:
            start = start + rngs[ch].multivariate_normal(np.zeros(start.size), cov0)
        out, acc, rej = _augmented_chain(
            spec,
            rngs[ch],
            start,
            y,
            X,
            n_iter,
            burnin,
            thin,
            draws,
            cov0,
            delta_prior_sd,
        )
        pieces.append(out)
        accs.append(acc)
        rejs.append(rej)
    U = np.vstack(pieces)
    rep = spec.to_report(U[:, :m])
    dnames = [f"delta:{nm}" for nm in spec.interest]
    d_df = pd.DataFrame(
        np.column_stack([rep, U[:, m:]]), columns=spec.report_names + dnames
    )
    chain_idx = np.repeat(np.arange(chains), draws)
    table, diag = posterior_table(d_df, chain_idx, chains, level)

    base_mean_u = mode
    base_tab = None
    if baseline:
        bu, _rw, _ind = metropolis_mixture(
            rngs[chains],
            lambda u: -neg_base(u),
            mode,
            mode,
            lap_cov,
            2.38 / np.sqrt(m),
            burnin + draws,
        )
        bu = bu[burnin:]
        base_mean_u = bu.mean(axis=0)
        brep = pd.DataFrame(spec.to_report(bu), columns=spec.report_names)
        base_tab = pd.DataFrame({"mean": brep.mean(), "sd": brep.std(ddof=1)})
    est, est_se, how = _efficient_estimate(model, spec, y, X0, names, base_mean_u)
    comp = pd.DataFrame(
        {
            "efficient": est,
            "efficient_se": est_se,
            "baseline_mean": (
                base_tab.loc[spec.interest, "mean"].to_numpy()
                if base_tab is not None
                else np.nan
            ),
            "baseline_sd": (
                base_tab.loc[spec.interest, "sd"].to_numpy()
                if base_tab is not None
                else np.nan
            ),
            "augmented_mean": table.loc[spec.interest, "mean"].to_numpy(),
            "augmented_sd": table.loc[spec.interest, "sd"].to_numpy(),
        },
        index=spec.interest,
    )
    excl = (table.loc[dnames, "lower"] > 0) | (table.loc[dnames, "upper"] < 0)
    excl.index = spec.interest
    diag.update(
        {
            "comparison": comp,
            "baseline_posterior": base_tab,
            "efficient_method": how,
            "delta_excludes_zero": excl,
            "acceptance_rate": float(np.mean(accs)),
            "rejected_per_obs": float(np.mean(rejs)),
        }
    )
    res = SemiparametricBayesResult(
        model=f"semiparametric_{model}",
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
        sampler="Metropolis-within-Gibbs with rejected-proposal augmentation",
        acceptance_rate=float(np.mean(accs)),
        prior={
            "coefficients": (
                f"N(0, {prior_sd:g}^2)" if model != "custom" else "log_prior"
            ),
            "delta": f"N(0, {delta_prior_sd:g}^2)",
            "prior_sd": prior_sd,
            "delta_prior_sd": delta_prior_sd,
        },
        level=level,
        model_info={
            "baseline_model": model,
            "interest": list(spec.interest),
            "augmentation": (
                "k0(z) = 2 / (1 + exp(-2 z)), z = delta' (score - efficient score)"
            ),
            "sampler_scale": list(spec.names_u),
        },
        diagnostics_info=diag,
    )
    rhat = diag["max_split_rhat"]
    if diag["min_ess"] < 100:
        text = (
            "The chain mixes slowly: effective sample size "
            f"{diag['min_ess']:.0f}. Increase draws."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    elif np.isfinite(rhat) and rhat > 1.05:
        text = (
            "The chain may not have converged: split potential scale "
            f"reduction factor is {rhat:.3f}. Increase draws / burnin."
        )
        diag["warnings"].append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    if bool(excl.any()):
        text = (
            "The credible interval of delta excludes zero for "
            + ", ".join(map(str, excl.index[excl.to_numpy()]))
            + ": the data reject the baseline distribution. The coefficients "
            "of interest are protected only against local departures, so "
            "consider a better baseline model."
        )
        diag["warnings"].append(text)
        warnings.warn(text, AssumptionWarning, stacklevel=2)
    if model != "custom" and is_default(prior_sd, 10.0):
        note_default_prior(
            diag["warnings"],
            names,
            table.loc[names, "mean"].to_numpy(),
            table.loc[names, "sd"].to_numpy(),
            0.0,
            prior_sd,
        )
    return res
