"""Bootstrap likelihood-ratio test of the number of Markov-switching regimes.

Under the hypothesis of fewer regimes the transition probabilities of the
extra regime are not identified and the parameters sit on the boundary of
the larger model, so twice the gain in log likelihood is not chi-squared.
:func:`mswitch_lrtest` refers it to its distribution in series simulated
from the fitted smaller model instead.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility, NumericalInstability
from . import _mswitch_core as core
from .mswitch import ArrayLike, MarkovSwitchingResult, _setup, mswitch

__all__ = ["mswitch_lrtest", "MarkovSwitchingLRTest"]

_LEVELS = (0.10, 0.05, 0.01)
_FIT_ERRORS = (np.linalg.LinAlgError, FloatingPointError, ValueError)

_STATEMENT = (
    "The p-value is the share of series simulated from the fitted "
    "{k0}-regime model whose likelihood-ratio statistic, computed by the "
    "same search, is at least the observed one. A small p-value says that "
    "the fitted {k0}-regime model with normal errors does not produce a "
    "gain in likelihood this large; it does not say that the departure is "
    "a Markov chain with {k1} regimes (outliers, a break, or conditional "
    "heteroskedasticity also raise the statistic), and a large p-value "
    "does not establish {k0} regime(s). The reference distribution is "
    "conditional on the estimated null parameters and the validity of the "
    "bootstrap for this non-regular problem rests on simulation evidence, "
    "not on a theorem. The chi-squared p-value is shown for comparison "
    "only; it is not a valid reference."
)


@dataclass
class MarkovSwitchingLRTest(ResultProtocolMixin):
    """Result of :func:`mswitch_lrtest`.

    Attributes
    ----------
    statistic : float
        ``2 * (loglik_alt - loglik_null)``, not below zero.
    pvalue : float
        ``(1 + #{LR* >= LR}) / (n_valid + 1)``.
    critical_values : pd.Series
        Bootstrap critical values at 10%, 5% and 1%: the
        ``ceil((n_valid + 1)(1 - level))``-th smallest ``LR*``; ``inf``
        when there are too few replicates for the level.
    reject : bool
        ``pvalue <= alpha``.
    loglik_null, loglik_alt : float
    states, null_states, n_obs, reps, n_valid : int
    naive_df, naive_pvalue : int, float
        Difference in the number of parameters and the chi-squared tail
        probability with it. Not a valid reference; for comparison only
        (see the Notes of :func:`mswitch_lrtest`).
    replicates : pd.DataFrame
        One row per simulated series: ``lr``, both log likelihoods, and
        the flags that ``diagnostics`` counts.
    diagnostics : dict
        ``n_failed`` (no statistic; left out of the p-value),
        ``n_floored`` (alternative fit below the null fit; statistic set to
        zero), ``n_multiple_maxima`` (the starts of the alternative ended at
        different local maxima), ``n_alt_not_converged``,
        ``n_variance_floor``, the same flags for the observed series, and
        the layout of the starting values.
    null_params : pd.Series
        Estimates of the null model on the estimation scale; the series
        are simulated from them.
    fit_null : MarkovSwitchingResult or None
        ``None`` for a single regime (see ``null_params``).
    fit_alt : MarkovSwitchingResult
    interpretation : str
        What the test does and does not establish.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> s = np.zeros(120, dtype=int)
    >>> for t in range(1, 120):
    ...     s[t] = s[t - 1] if rng.uniform() < 0.95 else 1 - s[t - 1]
    >>> y = np.where(s == 0, -1.5, 1.5) + rng.normal(size=120)
    >>> test = sp.mswitch_lrtest(y, states=2, reps=19, starts=3, seed=0)
    >>> type(test).__name__
    'MarkovSwitchingLRTest'
    >>> bool(test.reject)
    True
    """

    statistic: float
    pvalue: float
    critical_values: pd.Series
    reject: bool
    loglik_null: float
    loglik_alt: float
    states: int
    null_states: int
    n_obs: int
    reps: int
    n_valid: int
    naive_df: int
    naive_pvalue: float
    replicates: pd.DataFrame
    diagnostics: Dict[str, Any]
    null_params: pd.Series
    fit_null: Optional[MarkovSwitchingResult]
    fit_alt: MarkovSwitchingResult
    interpretation: str
    method: str = "bootstrap"
    alpha: float = 0.05
    model_info: Dict[str, Any] = field(default_factory=dict)

    _citation_keys = ("hamilton1989new",)

    def summary(self) -> str:
        d = self.diagnostics
        crit = "   ".join(
            f"{int(round(100 * float(lv)))}%: {cv:.4f}"
            for lv, cv in self.critical_values.items()
        )
        lines = [
            f"Likelihood-ratio test: {self.null_states} against {self.states} "
            "Markov-switching regime(s), parametric bootstrap",
            f"Observations: {self.n_obs}    Replicates: {self.n_valid} valid "
            f"of {self.reps}",
            f"Log likelihood: null {self.loglik_null:.5f}    alternative "
            f"{self.loglik_alt:.5f}",
            f"LR = {self.statistic:.4f}    bootstrap p-value = {self.pvalue:.4f}"
            f"    reject at {self.alpha:g}: {self.reject}",
            f"Bootstrap critical values   {crit}",
            f"(not valid, for comparison) chi2({self.naive_df}) p-value = "
            f"{self.naive_pvalue:.4g}",
            f"Replicates: {d['n_failed']} failed, {d['n_floored']} set to "
            f"zero, {d['n_multiple_maxima']} with several local maxima, "
            f"{d['n_alt_not_converged']} alternative fits not converged",
            "",
            self.interpretation,
        ]
        lines += [f"Note: {note}" for note in self.model_info.get("notes", [])]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        scalar = {k: v for k, v in self.diagnostics.items() if np.isscalar(v)}
        return {
            "method": self.method,
            "states": int(self.states),
            "null_states": int(self.null_states),
            "n_obs": int(self.n_obs),
            "statistic": float(self.statistic),
            "pvalue": float(self.pvalue),
            "reject": bool(self.reject),
            "alpha": float(self.alpha),
            "critical_values": {
                str(k): float(v) for k, v in self.critical_values.items()
            },
            "loglik_null": float(self.loglik_null),
            "loglik_alt": float(self.loglik_alt),
            "reps": int(self.reps),
            "n_valid": int(self.n_valid),
            "naive_df": int(self.naive_df),
            "naive_pvalue": float(self.naive_pvalue),
            "diagnostics": scalar,
            "interpretation": self.interpretation,
        }


def _parts(spec: core.Spec, theta: np.ndarray) -> Dict[str, np.ndarray]:
    return {key: v[0] for key, v in core.unpack(spec, theta[None, :]).items()}


def _simulate(
    spec: core.Spec, data: core.Data, theta: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One series from the model at ``theta``.

    Regressors and the first ``ar`` observations are those of the sample;
    the chain starts from its ergodic distribution at the first date.
    """
    part = _parts(spec, theta)
    k, p, n_all = spec.k, spec.p, len(data.y)
    trans = part["P"]
    state = np.zeros(n_all, dtype=int)
    if k > 1:
        cum = np.cumsum(trans, axis=1)
        u = rng.uniform(size=n_all)
        pi = core.ergodic(trans[None])[0]
        state[0] = min(int(np.searchsorted(np.cumsum(pi), u[0])), k - 1)
        for t in range(1, n_all):
            state[t] = min(int(np.searchsorted(cum[state[t - 1]], u[t])), k - 1)
    shock = rng.standard_normal(n_all) * np.exp(part["lnsig"])[state]
    mean = part["mu"][state] + data.x @ part["a"]
    if spec.nz:
        mean = mean + np.einsum("tz,tz->t", data.z, part["b"][state])
    phi = part["phi"][state]
    y = data.y.astype(float).copy()
    dev = y - mean
    for t in range(p, n_all):
        if spec.model == "ar":
            dev[t] = phi[t] @ dev[t - p : t][::-1] + shock[t]
            y[t] = mean[t] + dev[t]
        else:
            y[t] = mean[t] + phi[t] @ y[t - p : t][::-1] + shock[t]
    return y


def _one_state(
    spec: core.Spec, data: core.Data, maxiter: int, tol: float
) -> Tuple[np.ndarray, float]:
    """Gaussian ML of the model with a single regime.

    Least squares on the dynamic-regression form (the exact maximiser of
    the conditional likelihood for ``model='dr'``), followed for
    ``model='ar'`` by Newton steps on the same likelihood expression that
    the switching model uses.
    """
    dr = core.Spec(1, "dr", spec.p, spec.nx, spec.nz, spec.const, False, False)
    design = core.design_dr(dr, data)[:, 0, :]
    yv = data.y[spec.p :]
    beta = np.linalg.lstsq(design, yv, rcond=None)[0] if dr.n_mean else np.zeros(0)
    floor = 1e-8 * float(np.var(data.y))
    s2 = max(float(np.mean((yv - design @ beta) ** 2)), floor)
    theta = core.dr_to_model(spec, np.concatenate([beta, [0.5 * np.log(s2)]]))
    ll = core.loglik(spec, data, theta)
    if spec.model == "ar":
        th, ll_new, _, _ = core.maximise(
            spec, data, theta, maxiter, tol, 0.5 * float(np.log(floor))
        )
        if np.isfinite(ll_new) and ll_new >= ll:
            theta, ll = th, ll_new
    return theta, ll


def _split(
    spec0: core.Spec,
    spec1: core.Spec,
    data: core.Data,
    theta0: np.ndarray,
    stay: float,
    size: float,
) -> np.ndarray:
    """The null estimate written as a model with more, near-identical regimes.

    The most visited regime is split in two with the same parameters; with
    ``size = 0`` the likelihood equals that of the null model exactly (the
    chain is lumpable). ``size`` moves the two copies apart.
    """
    part = _parts(spec0, theta0)
    trans = part["P"]
    for k in range(spec0.k, spec1.k):
        j = int(np.argmax(core.ergodic(trans[None])[0]))
        sd = float(np.exp(part["lnsig"][j]))
        for name in ("mu", "b", "phi", "lnsig"):
            part[name] = np.concatenate([part[name], part[name][j : j + 1]], axis=0)
        if spec1.const == "switch":
            part["mu"][[j, k]] += size * sd * np.array([-1.0, 1.0])
        if spec1.nz:
            zsd = np.maximum(data.z[spec1.p :].std(axis=0), 1e-12)
            part["b"][[j, k]] += size * sd / zsd * np.array([[-1.0], [1.0]])
        if spec1.sw_ar:
            part["phi"][[j, k]] += 0.1 * size * np.array([[-1.0], [1.0]])
        if spec1.sw_var:
            part["lnsig"][[j, k]] += 0.2 * size * np.array([-1.0, 1.0])
        new = np.zeros((k + 1, k + 1))
        new[:k, :k] = trans
        new[k, :k] = trans[j]
        into = new[:, j].copy()
        new[j, j], new[j, k] = into[j] * stay, into[j] * (1.0 - stay)
        new[k, k], new[k, j] = into[k] * stay, into[k] * (1.0 - stay)
        others = [i for i in range(k) if i != j]
        new[others, j], new[others, k] = into[others] * 0.5, into[others] * 0.5
        trans = new
    trans = np.clip(trans, 1e-10, None)
    part["P"] = trans / trans.sum(axis=1, keepdims=True)
    return core.pack(spec1, part)


def _plan(
    spec: core.Spec,
    data: core.Data,
    seq: np.random.SeedSequence,
    starts: int,
    nested: Sequence[np.ndarray],
) -> Tuple[int, int, List[np.ndarray]]:
    """Starting values of one fit: EM starts, their seed, direct starts."""
    em_seq, direct_seq = seq.spawn(2)
    n_em = 2 if starts >= 4 else 1
    rng = np.random.default_rng(direct_seq)
    extra = list(nested)
    for _ in range(max(starts - n_em - len(extra), 0)):
        extra.append(core.dr_to_model(spec, core.start_values(spec, data, rng)))
    return n_em, int(em_seq.generate_state(1)[0]), extra


def _flags(table: pd.DataFrame, best: int) -> Dict[str, Any]:
    ll = float(table.loc[best, "loglik"])
    ok = table["converged"]
    done = ok if ok.any() else table["loglik"] == ll
    lls = np.sort(table.loc[done, "loglik"].to_numpy())
    n_max = 1 + int((np.diff(lls) > 1e-5 * (1.0 + np.abs(lls[1:]))).sum())
    return {"converged": bool(ok[best]), "n_maxima": n_max}


def _pair(
    spec0: core.Spec,
    spec1: core.Spec,
    data: core.Data,
    seq: np.random.SeedSequence,
    starts: int,
    maxiter: int,
    tol: float,
) -> Dict[str, Any]:
    """Fit the null and the alternative to one series by the common search."""
    null_seq, alt_seq = seq.spawn(2)
    if spec0.k == 1:
        theta0, ll0 = _one_state(spec0, data, maxiter, tol)
    else:
        n_em, seed, extra = _plan(spec0, data, null_seq, starts, [])
        table, fits, best = core.fit_starts(
            spec0, data, n_em, np.random.default_rng(seed), maxiter, tol, extra
        )
        theta0, ll0 = fits[best], float(table.loc[best, "loglik"])
    if not np.isfinite(ll0):
        raise NumericalInstability("the null likelihood is not finite")
    nested = [
        _split(spec0, spec1, data, theta0, 0.9, 0.5),
        _split(spec0, spec1, data, theta0, 0.5, 0.25),
    ]
    n_em, seed, extra = _plan(spec1, data, alt_seq, starts, nested)
    table, fits, best = core.fit_starts(
        spec1, data, n_em, np.random.default_rng(seed), maxiter, tol, extra
    )
    ll1 = float(table.loc[best, "loglik"])
    if not np.isfinite(ll1):
        raise NumericalInstability("the alternative likelihood is not finite")
    sig = fits[best][spec1.n_mean : spec1.n_mean + spec1.n_sig]
    floor = 0.5 * float(np.log(1e-8 * float(np.var(data.y))))
    out: Dict[str, Any] = {"loglik_null": ll0, "loglik_alt": ll1}
    out["lr_raw"] = 2.0 * (ll1 - ll0)
    out.update(_flags(table, best))
    out["variance_floor"] = bool((sig < floor + 1e-6).any())
    out["theta_null"] = theta0
    return out


def _replicate(job: Tuple[Any, ...]) -> List[Dict[str, Any]]:
    """Simulate from the null and refit, for a block of replicates."""
    spec0, spec1, data, theta0, seqs, starts, maxiter, tol = job
    rows: List[Dict[str, Any]] = []
    for seq in seqs:
        sim_seq, fit_seq = seq.spawn(2)
        row: Dict[str, Any] = {"failed": False, "error": ""}
        try:
            with np.errstate(all="ignore"):
                y = _simulate(spec0, data, theta0, np.random.default_rng(sim_seq))
                sim = core.Data(y, data.x, data.z, data.p)
                res = _pair(spec0, spec1, sim, fit_seq, starts, maxiter, tol)
            res.pop("theta_null")
            row.update(res)
        except _FIT_ERRORS as exc:  # counted in diagnostics['n_failed']
            row.update({"failed": True, "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    return rows


def mswitch_lrtest(
    y: Union[str, np.ndarray, pd.Series],
    x: ArrayLike = None,
    *,
    data: Optional[pd.DataFrame] = None,
    states: int = 2,
    null_states: Optional[int] = None,
    model: str = "dr",
    ar: int = 0,
    switch: ArrayLike = None,
    switch_ar: bool = False,
    switch_variance: bool = False,
    constant: Union[bool, str] = True,
    method: str = "bootstrap",
    reps: int = 199,
    starts: int = 10,
    seed: Optional[int] = None,
    maxiter: int = 500,
    tol: float = 1e-9,
    alpha: float = 0.05,
    n_jobs: int = 1,
) -> MarkovSwitchingLRTest:
    """Likelihood-ratio test of the number of regimes in :func:`mswitch`.

    Tests ``null_states`` regimes against ``states`` regimes. The statistic
    ``LR = 2 (log L_alt - log L_null)`` has no chi-squared limit, because
    under the null the transition probabilities of the extra regime are not
    identified. Its reference distribution is obtained by parametric
    bootstrap: ``reps`` series are simulated from the fitted null model,
    both models are refitted to each, and the p-value is
    ``(1 + #{LR* >= LR}) / (reps + 1)``.

    Parameters
    ----------
    y, x, data, model, ar, switch, switch_ar, switch_variance, constant
        The model, as in :func:`mswitch`.
    states : int, default 2
        Number of regimes under the alternative.
    null_states : int, optional
        Number of regimes under the null; ``states - 1`` by default. With
        one regime the null is the linear model in which everything that
        switches under the alternative is constant.
    method : {'bootstrap'}, default 'bootstrap'
        The only reference distribution offered.
    reps : int, default 199
        Number of simulated series. Choose it so that
        ``alpha * (reps + 1)`` is an integer.
    starts : int, default 10
        Starting values of every switching fit, at least 3. Two (one when
        ``starts < 4``) are the first starts of :func:`mswitch` and go
        through EM. Two are the null estimate split into near-identical
        regimes. The others are random and go straight to Newton
        iterations.
    seed : int, optional
        Seed of the simulations and of the starting values. The result
        does not depend on ``n_jobs``.
    maxiter, tol
        As in :func:`mswitch`.
    alpha : float, default 0.05
        Level of ``reject``.
    n_jobs : int, default 1
        Number of processes for the replicates (started with ``spawn``).

    Returns
    -------
    MarkovSwitchingLRTest

    Raises
    ------
    MethodIncompatibility
        Another ``method``, ``null_states`` outside ``1 .. states - 1``,
        ``reps < 1``, ``starts < 3``, ``n_jobs < 1``, or anything
        :func:`mswitch` rejects.
    DataInsufficient
        Too few observations for the alternative model.

    Notes
    -----
    *One likelihood.* With one regime the null is fitted by Gaussian
    maximum likelihood conditional on the first ``ar`` observations, the
    same expression as the switching likelihood with a single state: least
    squares for ``model='dr'``, and for ``model='ar'`` the model with
    autoregressive deviations from ``mu + x'a``.

    *The search is part of the statistic.* The likelihood of a switching
    model has several local maxima, most of all when the data have one
    regime. A search that misses the highest one understates ``LR``. If
    that happened in the replicates only, the p-value would be too small;
    in the observed series only, too large. The observed series and every
    replicate are therefore fitted by the same search (``starts`` and its
    layout), so that the bootstrap reproduces the law of the statistic as
    it is computed and its size does not depend on the search finding the
    global maximum; a weak search costs power. A fit of the alternative
    that ends below the null fit is replaced by the null value
    (``LR = 0``), which is a point of the alternative model.
    ``diagnostics`` counts these, the replicates in which the starts
    disagreed, and those that failed.

    *Simulation.* Regressors are held fixed and the first ``ar``
    observations are those of the sample; errors are normal. With two or
    more regimes under the null the chain is drawn from its ergodic
    distribution.

    *Simulation evidence.* With one regime against two (switching
    constant) and a true null, the test rejected at about the nominal rate
    for white noise with 100 observations and for an AR(1) with 150. A
    chi-squared reference with one degree of freedom, the number of
    restricted means, rejected 37% of the time at the 5% level; with the
    difference in the number of parameters (``naive_df``) it was close to
    nominal in these two designs, which nothing guarantees elsewhere. The
    numbers are in ``tests/reference_parity/_fixtures/mswitch_lrtest_mc.json``.

    *Unbounded likelihood.* With ``switch_variance=True`` the likelihood
    grows without bound where a regime fits one observation; the statistic
    then depends on the lower bound on the variances
    (``diagnostics['n_variance_floor']`` counts such replicates).

    References
    ----------
    [@hamilton1989new],
    [@mclachlan1987bootstrapping],
    [@hansen1992likelihood],
    [@garcia1998asymptotic]

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> s = np.zeros(120, dtype=int)
    >>> for t in range(1, 120):
    ...     s[t] = s[t - 1] if rng.uniform() < 0.95 else 1 - s[t - 1]
    >>> y = np.where(s == 0, -1.5, 1.5) + rng.normal(size=120)
    >>> test = sp.mswitch_lrtest(y, states=2, reps=19, starts=3, seed=0)
    >>> bool(test.statistic > test.critical_values[0.05])
    True
    >>> float(test.pvalue)
    0.05
    """
    if method != "bootstrap":
        raise MethodIncompatibility(
            f"mswitch_lrtest: method must be 'bootstrap', got {method!r}.",
            recovery_hint="No asymptotic reference is offered: the "
            "chi-squared law does not apply to this test.",
        )
    k0 = int(states) - 1 if null_states is None else null_states
    if not isinstance(k0, (int, np.integer)) or isinstance(k0, bool):
        raise MethodIncompatibility("mswitch_lrtest: null_states must be an integer.")
    if reps < 1 or starts < 3 or n_jobs < 1:
        raise MethodIncompatibility(
            "mswitch_lrtest: need reps >= 1, starts >= 3 and n_jobs >= 1."
        )
    spec1, dat, _, xn, zn = _setup(
        y,
        x,
        data,
        states,
        model,
        ar,
        switch,
        switch_ar,
        switch_variance,
        constant,
        "oim",
        starts,
        alpha,
    )
    if not 1 <= k0 < spec1.k:
        raise MethodIncompatibility(
            f"mswitch_lrtest: null_states must be between 1 and {spec1.k - 1}, "
            f"got {k0}.",
        )
    one = k0 == 1
    spec0 = replace(spec1, k=int(k0), sw_ar=spec1.sw_ar and not one)
    spec0 = replace(spec0, sw_var=spec1.sw_var and not one)
    seqs = np.random.SeedSequence(seed).spawn(int(reps) + 1)
    null_seq, alt_seq = seqs[0].spawn(2)
    kwargs: Dict[str, Any] = dict(
        data=data,
        model=model,
        ar=ar,
        switch=switch,
        switch_ar=switch_ar,
        switch_variance=switch_variance,
        constant=constant,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
    )
    notes: List[str] = []
    fit_null: Optional[MarkovSwitchingResult] = None
    if k0 == 1:
        theta0, ll0 = _one_state(spec0, dat, maxiter, tol)
        names = [c for c in ["const"] * (spec0.const != "none")]
        names += xn + zn + [f"ar.L{j + 1}" for j in range(spec0.p)] + ["lnsigma"]
    else:
        n_em, em_seed, extra = _plan(spec0, dat, null_seq, starts, [])
        fit_null = mswitch(
            y, x, states=k0, starts=n_em, seed=em_seed, start_params=extra, **kwargs
        )
        theta0, ll0 = fit_null.theta.to_numpy(), float(fit_null.loglik)
        names = list(fit_null.theta.index)
    nested = [
        _split(spec0, spec1, dat, theta0, 0.9, 0.5),
        _split(spec0, spec1, dat, theta0, 0.5, 0.25),
    ]
    n_em, em_seed, extra = _plan(spec1, dat, alt_seq, starts, nested)
    fit_alt = mswitch(
        y, x, states=states, starts=n_em, seed=em_seed, start_params=extra, **kwargs
    )
    ll1 = float(fit_alt.loglik)
    lr_raw = 2.0 * (ll1 - ll0)
    if lr_raw < 0.0:
        notes.append(
            "the alternative fit ended below the null fit; LR is set to 0, "
            "the value at the null model. Raise starts."
        )
    lr = max(lr_raw, 0.0)

    size = -(-int(reps) // (4 * int(n_jobs))) if n_jobs > 1 else int(reps)
    blocks = [seqs[1:][i : i + size] for i in range(0, int(reps), size)]
    jobs = [(spec0, spec1, dat, theta0, b, starts, maxiter, tol) for b in blocks]
    if n_jobs > 1:
        import multiprocessing as mp

        with ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn")) as pool:
            done = list(pool.map(_replicate, jobs))
    else:
        done = [_replicate(job) for job in jobs]
    table = pd.DataFrame([row for block in done for row in block])
    for col in ("loglik_null", "loglik_alt", "lr_raw"):
        if col not in table:
            table[col] = np.nan
    ok = ~table["failed"].to_numpy(dtype=bool)
    if not ok.any():
        raise MethodIncompatibility(
            "mswitch_lrtest: no bootstrap replicate could be fitted.",
            diagnostics={"errors": table["error"].unique().tolist()[:5]},
        )
    raw = table["lr_raw"].to_numpy(dtype=float)
    table["floored"] = ok & (raw < 0.0)
    table["lr"] = np.where(ok, np.maximum(raw, 0.0), np.nan)
    table = table.drop(columns="lr_raw")
    lr_star = np.sort(table.loc[ok, "lr"].to_numpy())
    n_valid = int(len(lr_star))
    pvalue = (1.0 + float((lr_star >= lr).sum())) / (n_valid + 1.0)
    ranks = [int(np.ceil((n_valid + 1) * (1.0 - lv) - 1e-9)) for lv in _LEVELS]
    crit = pd.Series(
        [float(lr_star[r - 1]) if r <= n_valid else np.inf for r in ranks],
        index=pd.Index(_LEVELS, name="level"),
        name="critical_value",
    )
    sig = fit_alt.theta.to_numpy()[spec1.n_mean : spec1.n_mean + spec1.n_sig]
    floor = 0.5 * float(np.log(1e-8 * float(np.var(dat.y))))

    def count(col: str) -> int:
        if col not in table:
            return 0
        return int(table.loc[ok, col].astype(bool).sum())

    diagnostics: Dict[str, Any] = {
        "n_failed": int((~ok).sum()),
        "n_floored": count("floored"),
        "n_multiple_maxima": int((table.loc[ok, "n_maxima"] > 1).sum()),
        "n_alt_not_converged": n_valid - count("converged"),
        "n_variance_floor": count("variance_floor"),
        "observed_floored": bool(lr_raw < 0.0),
        "observed_alt_converged": bool(fit_alt.converged),
        "observed_n_maxima": int(fit_alt.model_info["n_distinct_maxima"]),
        "observed_variance_floor": bool((sig < floor + 1e-6).any()),
        "starts_em": int(n_em),
        "starts_null_split": 2,
        "starts_random": int(len(extra) - 2),
        "errors": sorted(set(table.loc[~ok, "error"])),
    }
    if diagnostics["n_failed"]:
        notes.append(
            f"{diagnostics['n_failed']} of {int(reps)} replicates could not "
            "be fitted and are left out of the p-value"
        )
        warnings.warn(f"mswitch_lrtest: {notes[-1]}.", RuntimeWarning, stacklevel=2)
    if spec1.sw_var:
        notes.append(
            "switch_variance=True: the likelihood is unbounded; the statistic "
            "depends on the lower bound on the variances "
            f"({diagnostics['n_variance_floor']} replicates ended on it)"
        )
    if alpha * (n_valid + 1) < 1.0:
        notes.append(
            f"{n_valid} replicates cannot reject at alpha = {alpha:g}; the "
            f"smallest attainable p-value is {1.0 / (n_valid + 1):.4g}"
        )
    df = int(spec1.n_par - spec0.n_par)
    return MarkovSwitchingLRTest(
        statistic=float(lr),
        pvalue=float(pvalue),
        critical_values=crit,
        reject=bool(pvalue <= alpha),
        loglik_null=float(ll0),
        loglik_alt=float(ll1),
        states=int(spec1.k),
        null_states=int(k0),
        n_obs=int(dat.n),
        reps=int(reps),
        n_valid=n_valid,
        naive_df=df,
        naive_pvalue=float(stats.chi2.sf(lr, df)),
        replicates=table,
        diagnostics=diagnostics,
        null_params=pd.Series(theta0, index=names, name="estimate"),
        fit_null=fit_null,
        fit_alt=fit_alt,
        interpretation=_STATEMENT.format(k0=int(k0), k1=int(spec1.k)),
        alpha=float(alpha),
        model_info={"notes": notes, "seed": seed, "starts": int(starts)},
    )
