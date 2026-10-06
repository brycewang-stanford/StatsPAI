"""Markov-switching regression and autoregression (Hamilton filter).

:func:`mswitch` fits by maximum likelihood a regression whose constant,
some coefficients, autoregressive coefficients or error variance depend on
an unobserved state that follows a first-order Markov chain.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from . import _mswitch_core as core

__all__ = ["mswitch", "MarkovSwitchingResult"]

ArrayLike = Union[str, Sequence[str], np.ndarray, pd.Series, pd.DataFrame, None]


@dataclass
class MarkovSwitchingResult(ResultProtocolMixin):
    """Result of :func:`mswitch`.

    Attributes
    ----------
    params : pd.DataFrame
        One row per coefficient: ``state`` (``'all'`` for a coefficient
        common to the states), ``term``, ``estimate``, ``se``, ``z``,
        ``pvalue``, ``ci_lower``, ``ci_upper``. The interval of a standard
        deviation is built on the log scale.
    transition : pd.DataFrame
        ``transition.loc[i, j]`` is the probability of moving from state
        ``i`` to state ``j``; rows sum to one.
    transition_table : pd.DataFrame
        The same probabilities with delta-method standard errors and
        intervals built on the logit scale.
    durations : pd.DataFrame
        Expected duration ``1 / (1 - p_ii)`` of each state.
    filtered, smoothed, predicted : pd.DataFrame
        Probability of each state given the data up to the date, the whole
        sample (Kim smoother), and the previous date.
    fitted, resid : pd.Series
        One-step predictions ``E[y_t | y_{t-1}, ...]`` (the state means
        weighted by the predicted probabilities) and their errors.
    theta, vcov : pd.Series, pd.DataFrame
        Estimates and covariance on the estimation scale (log standard
        deviations, transition logits ``q`` as in Stata).
    loglik, aic, bic, hqic, n_obs, n_params, states, model, ar, vce
    converged : bool
    starts : pd.DataFrame
        Log likelihood reached from each starting value; ``degenerate``
        marks an end point where one state is never visited.
    model_info : dict
        ``order_rule``, ``n_distinct_maxima``, ``ergodic`` and notes.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> s = np.zeros(300, dtype=int)
    >>> for t in range(1, 300):
    ...     stay = 0.95 if s[t - 1] == 0 else 0.9
    ...     s[t] = s[t - 1] if rng.uniform() < stay else 1 - s[t - 1]
    >>> y = np.where(s == 0, -1.0, 2.0) + rng.normal(size=300)
    >>> fit = sp.mswitch(y, states=2, seed=0)
    >>> type(fit).__name__
    'MarkovSwitchingResult'
    >>> list(fit.smoothed.columns)
    ['state1', 'state2']
    """

    params: pd.DataFrame
    transition: pd.DataFrame
    transition_table: pd.DataFrame
    durations: pd.DataFrame
    filtered: pd.DataFrame
    smoothed: pd.DataFrame
    predicted: pd.DataFrame
    fitted: pd.Series
    resid: pd.Series
    theta: pd.Series
    vcov: pd.DataFrame
    loglik: float
    aic: float
    bic: float
    hqic: float
    n_obs: int
    n_params: int
    states: int
    model: str
    ar: int
    vce: str
    converged: bool
    starts: pd.DataFrame
    alpha: float = 0.05
    model_info: Dict[str, Any] = field(default_factory=dict)
    _state: Dict[str, Any] = field(default_factory=dict, repr=False)

    _citation_keys = ("hamilton1989new", "kim1994dynamic")

    def summary(self) -> str:
        kind = {"dr": "dynamic regression", "ar": "autoregression"}[self.model]
        info = self.model_info

        def fmt(v: float) -> str:
            return f"{v:.6g}"

        lines = [
            f"Markov-switching {kind}: {self.states} states, ar = {self.ar}",
            f"Observations: {self.n_obs}    Log likelihood: {self.loglik:.5f}",
            f"AIC {self.aic:.4f}    BIC {self.bic:.4f}    HQIC {self.hqic:.4f}"
            f"    vce: {self.vce}",
            f"States ordered by {info['order_rule']}; initial distribution: "
            "ergodic probabilities",
            "",
            self.params.to_string(index=False, float_format=fmt),
            "",
            "Transition probabilities (row: from, column: to)",
            self.transition_table.to_string(index=False, float_format=fmt),
            "",
            "Expected durations",
            self.durations.to_string(index=False, float_format=fmt),
            "",
            f"Starting values: {len(self.starts)}; distinct local maxima "
            f"reached: {info['n_distinct_maxima']}; converged: {self.converged}",
        ]
        lines += [f"Note: {note}" for note in info.get("notes", [])]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "model": self.model,
            "states": int(self.states),
            "ar": int(self.ar),
            "n_obs": int(self.n_obs),
            "n_params": int(self.n_params),
            "loglik": float(self.loglik),
            "aic": float(self.aic),
            "bic": float(self.bic),
            "hqic": float(self.hqic),
            "vce": self.vce,
            "converged": bool(self.converged),
            "params": self.params.to_dict(orient="records"),
            "transition": self.transition.to_numpy().tolist(),
            "durations": self.durations.to_dict(orient="records"),
            "order_rule": self.model_info["order_rule"],
            "n_distinct_maxima": int(self.model_info["n_distinct_maxima"]),
        }

    def plot(self, which: str = "smoothed") -> Any:
        """Probability of each state over time (``'smoothed'``/``'filtered'``)."""
        if which not in ("smoothed", "filtered", "predicted"):
            raise MethodIncompatibility(
                f"plot: which must be 'smoothed', 'filtered' or 'predicted', "
                f"got {which!r}."
            )
        import matplotlib.pyplot as plt

        prob: pd.DataFrame = getattr(self, which)
        fig, axes = plt.subplots(
            self.states, 1, figsize=(7.0, 1.9 * self.states), sharex=True, squeeze=False
        )
        for ax, col in zip(axes[:, 0], prob.columns):
            ax.fill_between(prob.index, 0.0, prob[col].to_numpy(), alpha=0.35)
            ax.plot(prob.index, prob[col].to_numpy(), lw=1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel(f"P({col})")
        axes[0, 0].set_title(f"{which.capitalize()} state probabilities")
        fig.tight_layout()
        return fig

    def forecast(self, steps: int = 1) -> pd.DataFrame:
        """Forecasts of ``y`` and of the state probabilities.

        Exact conditional expectations given the sample and the estimates;
        available without exogenous regressors, for ``model='dr'`` with
        ``ar=0`` and for ``model='ar'`` with common AR coefficients.
        """
        st = self._state
        spec: core.Spec = st["spec"]
        if steps < 1:
            raise MethodIncompatibility("forecast: steps must be at least 1.")
        linear = (spec.model == "dr" and spec.p == 0) or (
            spec.model == "ar" and not spec.sw_ar
        )
        if spec.nx or spec.nz or not linear:
            raise MethodIncompatibility(
                "forecast: the conditional mean is a closed form only without "
                "exogenous regressors, for model='dr' with ar=0 or model='ar' "
                "with common AR coefficients.",
                recovery_hint="Use the fitted transition matrix and "
                "coefficients to simulate paths for other models.",
            )
        k, p = spec.k, spec.p
        mu, phi, trans = st["mu"], st["phi"][0], st["P"]
        last = st["filt_exp"][-1].reshape((k,) * (spec.p_exp + 1))
        # deviations y_t - E[mu_{s_t} | sample] of the last p dates
        dev: List[float] = []
        for lag in range(p - 1, -1, -1):
            axes = tuple(a for a in range(p + 1) if a != lag)
            dev.append(
                float(st["y"][len(st["y"]) - 1 - lag] - last.sum(axis=axes) @ mu)
            )
        prob = last.reshape(k, -1).sum(axis=1)
        rows = []
        for h in range(1, steps + 1):
            prob = prob @ trans
            u = float(sum(phi[j] * dev[-1 - j] for j in range(p)))
            dev.append(u)
            rows.append([h, float(prob @ mu) + u] + [float(v) for v in prob])
        cols = ["step", "forecast"] + [f"state{s + 1}" for s in range(k)]
        return pd.DataFrame(rows, columns=cols)


def _columns(
    obj: ArrayLike, data: Optional[pd.DataFrame], n: int, stem: str
) -> Tuple[np.ndarray, List[str]]:
    if obj is None:
        return np.empty((n, 0)), []
    if isinstance(obj, str) or (
        isinstance(obj, (list, tuple)) and all(isinstance(c, str) for c in obj)
    ):
        cols = [obj] if isinstance(obj, str) else list(obj)
        if data is None:
            raise MethodIncompatibility(
                f"mswitch: column names {cols} need data=.",
                recovery_hint="Pass the DataFrame as data=, or pass arrays.",
            )
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise MethodIncompatibility(f"mswitch: column(s) {missing} not in data.")
        return data[cols].to_numpy(dtype=float).reshape(n, len(cols)), cols
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy(dtype=float), [str(c) for c in obj.columns]
    arr = np.asarray(obj, dtype=float)
    arr = arr.reshape(len(arr), -1)
    if isinstance(obj, pd.Series) and obj.name is not None:
        return arr, [str(obj.name)]
    return arr, [f"{stem}{j + 1}" for j in range(arr.shape[1])]


def _layout(
    spec: core.Spec, xn: List[str], zn: List[str]
) -> List[Tuple[str, str, str]]:
    """(name, state, term) of every entry of the parameter vector."""
    k = spec.k
    out: List[Tuple[str, str, str]] = []
    if spec.const == "switch":
        out += [(f"state{s + 1}:const", f"state{s + 1}", "const") for s in range(k)]
    elif spec.const == "common":
        out.append(("const", "all", "const"))
    out += [(c, "all", c) for c in xn]
    out += [(f"state{s + 1}:{c}", f"state{s + 1}", c) for s in range(k) for c in zn]
    lags = [f"ar.L{j + 1}" for j in range(spec.p)]
    if spec.sw_ar:
        out += [
            (f"state{s + 1}:{c}", f"state{s + 1}", c) for s in range(k) for c in lags
        ]
    else:
        out += [(c, "all", c) for c in lags]
    if spec.sw_var:
        out += [(f"lnsigma{s + 1}", f"state{s + 1}", "sigma") for s in range(k)]
    else:
        out.append(("lnsigma", "all", "sigma"))
    out += [(f"q{i + 1}{j + 1}", "", "") for i in range(k) for j in range(k - 1)]
    return out


def _tables(
    spec: core.Spec,
    theta: np.ndarray,
    vcov: np.ndarray,
    layout: List[Tuple[str, str, str]],
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k = spec.k
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    se = np.sqrt(np.diag(vcov))
    rows = []
    for i, (_, state, term) in enumerate(layout[: spec.n_mean + spec.n_sig]):
        est, s = float(theta[i]), float(se[i])
        if term == "sigma":
            sig = float(np.exp(est))
            rows.append(
                [state, term, sig, sig * s, np.nan, np.nan]
                + [sig * np.exp(-crit * s), sig * np.exp(crit * s)]
            )
        else:
            zstat = est / s
            pval = 2.0 * float(stats.norm.sf(abs(zstat)))
            rows.append(
                [state, term, est, s, zstat, pval, est - crit * s, est + crit * s]
            )
    cols = ["state", "term", "estimate", "se", "z", "pvalue", "ci_lower", "ci_upper"]
    params = pd.DataFrame(rows, columns=cols)
    order = {"all": 0, **{f"state{s + 1}": s + 1 for s in range(k)}}
    params = params.assign(
        _s=(params["term"] == "sigma").astype(int), _o=params["state"].map(order)
    )
    params = params.sort_values(["_s", "_o"], kind="stable").drop(columns=["_s", "_o"])
    params = params.reset_index(drop=True)

    nq = k * (k - 1)
    pert = np.tile(theta, (nq, 1)).astype(complex)
    pert[np.arange(nq), len(theta) - nq + np.arange(nq)] += 1e-30j
    jac = core.unpack(spec, pert)["P"].imag.reshape(nq, k * k).T / 1e-30
    trans = core.unpack(spec, theta[None, :])["P"][0]
    vq = vcov[-nq:, -nq:]
    p_se: np.ndarray
    if np.isnan(vq).all():
        p_se = np.full(k * k, np.nan)
    else:
        p_se = np.sqrt(np.einsum("ij,jk,ik->i", jac, np.nan_to_num(vq), jac))
        edge = (trans.ravel() < 1e-6) | (trans.ravel() > 1.0 - 1e-6)
        p_se = np.where(edge, np.nan, p_se)
    names = [f"state{s + 1}" for s in range(k)]
    trows, drows = [], []
    for i in range(k):
        for j in range(k):
            pij, sij = float(trans[i, j]), float(p_se[i * k + j])
            with np.errstate(divide="ignore", invalid="ignore"):
                half = crit * sij / (pij * (1.0 - pij))
                logit = np.log(pij / (1.0 - pij))
                lo = 1.0 / (1.0 + np.exp(-(logit - half)))
                hi = 1.0 / (1.0 + np.exp(-(logit + half)))
            trows.append([names[i], names[j], pij, sij, float(lo), float(hi)])
            if i == j:
                drows.append(
                    [names[i], 1.0 / (1.0 - pij), sij / (1.0 - pij) ** 2]
                    + [float(1.0 / (1.0 - lo)), float(1.0 / (1.0 - hi))]
                )
    tcols = ["from", "to", "estimate", "se", "ci_lower", "ci_upper"]
    dcols = ["state", "estimate", "se", "ci_lower", "ci_upper"]
    return (
        params,
        pd.DataFrame(trans, index=names, columns=names),
        pd.DataFrame(trows, columns=tcols),
        pd.DataFrame(drows, columns=dcols),
    )


def _setup(
    y: Union[str, np.ndarray, pd.Series],
    x: ArrayLike,
    data: Optional[pd.DataFrame],
    states: int,
    model: str,
    ar: int,
    switch: ArrayLike,
    switch_ar: bool,
    switch_variance: bool,
    constant: Union[bool, str],
    vce: str,
    starts: int,
    alpha: float,
) -> Tuple[core.Spec, core.Data, pd.Series, List[str], List[str]]:
    """Check the arguments of :func:`mswitch`; the model shape and the data."""
    if model not in ("dr", "ar"):
        raise MethodIncompatibility(
            f"mswitch: model must be 'dr' or 'ar', got {model!r}."
        )
    if vce not in ("oim", "robust"):
        raise MethodIncompatibility(
            f"mswitch: vce must be 'oim' or 'robust', got {vce!r}."
        )
    if constant not in (True, False, "common"):
        raise MethodIncompatibility(
            f"mswitch: constant must be True, False or 'common', got {constant!r}."
        )
    if not isinstance(states, (int, np.integer)) or states < 2:
        raise MethodIncompatibility("mswitch: states must be an integer of at least 2.")
    if not isinstance(ar, (int, np.integer)) or ar < 0:
        raise MethodIncompatibility("mswitch: ar must be a non-negative integer.")
    if starts < 1 or not 0.0 < alpha < 1.0:
        raise MethodIncompatibility("mswitch: need starts >= 1 and 0 < alpha < 1.")
    if model == "ar" and ar > 0 and (states > 3 or ar > 4):
        raise MethodIncompatibility(
            "mswitch: model='ar' is verified for states <= 3 and ar <= 4 only.",
            recovery_hint="Use model='dr', where lags of y are regressors.",
            diagnostics={"states": int(states), "ar": int(ar)},
        )
    if ar == 0 and switch_ar:
        raise MethodIncompatibility("mswitch: switch_ar=True needs ar >= 1.")

    if isinstance(y, str):
        if data is None or y not in data.columns:
            raise MethodIncompatibility(
                f"mswitch: column {y!r} needs data= holding it."
            )
        ys = data[y]
    else:
        ys = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float))
    yv = ys.to_numpy(dtype=float).ravel()
    n_all = len(yv)
    xv, xn = _columns(x, data, n_all, "x")
    zv, zn = _columns(switch, data, n_all, "z")
    if len(xv) != n_all or len(zv) != n_all:
        raise MethodIncompatibility("mswitch: y, x and switch differ in length.")
    if not (np.isfinite(yv).all() and np.isfinite(xv).all() and np.isfinite(zv).all()):
        raise MethodIncompatibility(
            "mswitch: missing or infinite values in y, x or switch.",
            recovery_hint="The filter needs a gapless series; drop or fill first.",
        )
    const = {True: "switch", False: "none", "common": "common"}[constant]
    spec = core.Spec(
        int(states),
        model if ar else "dr",
        int(ar),
        len(xn),
        len(zn),
        const,
        bool(switch_ar),
        bool(switch_variance),
    )
    if not (const == "switch" or zn or switch_ar or switch_variance):
        raise MethodIncompatibility(
            "mswitch: nothing depends on the state.",
            recovery_hint="Keep constant=True, or set switch=, switch_ar=True "
            "or switch_variance=True.",
        )
    n = n_all - spec.p
    if n < max(3 * spec.n_par, 20):
        raise DataInsufficient(
            f"mswitch: {n} usable observations for {spec.n_par} parameters.",
            diagnostics={"n_obs": int(n), "n_params": int(spec.n_par)},
        )
    dat = core.Data(yv, xv, zv, spec.p)
    return spec, dat, ys, xn, zn


def mswitch(
    y: Union[str, np.ndarray, pd.Series],
    x: ArrayLike = None,
    *,
    data: Optional[pd.DataFrame] = None,
    states: int = 2,
    model: str = "dr",
    ar: int = 0,
    switch: ArrayLike = None,
    switch_ar: bool = False,
    switch_variance: bool = False,
    constant: Union[bool, str] = True,
    vce: str = "oim",
    starts: int = 5,
    seed: Optional[int] = None,
    maxiter: int = 500,
    tol: float = 1e-9,
    alpha: float = 0.05,
    start_params: Optional[Sequence[np.ndarray]] = None,
) -> MarkovSwitchingResult:
    """Markov-switching regression by maximum likelihood.

    The state ``s_t`` follows a first-order Markov chain with ``states``
    values. With ``model='dr'`` (dynamic regression) ::

        y_t = mu_{s_t} + x_t'a + z_t'b_{s_t} + sum_k phi_k y_{t-k} + e_t

    so the level adjusts at once when the state changes and lags of ``y``
    are ordinary regressors. With ``model='ar'`` (Hamilton's
    autoregression) the deviations from the state-dependent mean are
    autoregressive ::

        y_t - m_t(s_t) = sum_k phi_k [y_{t-k} - m_{t-k}(s_{t-k})] + e_t,
        m_t(s) = mu_s + x_t'a + z_t'b_s

    and the likelihood runs over the ``states**(ar + 1)`` histories
    ``(s_t, ..., s_{t-ar})``. ``e_t`` is normal with standard deviation
    ``sigma`` or ``sigma_{s_t}``.

    Parameters
    ----------
    y : str or array-like
        Outcome in time order, or its column name in ``data``.
    x : str, list of str or array-like, optional
        Regressors with coefficients common to the states.
    data : pandas.DataFrame, optional
        Needed when variables are given by name.
    states : int, default 2
        Number of states (2 or 3 for ``model='ar'``).
    model : {'dr', 'ar'}, default 'dr'
        Dynamic regression or autoregression; they coincide for ``ar=0``.
    ar : int, default 0
        Number of lags of ``y`` (lags ``1..ar``; at most 4 for
        ``model='ar'``). The first ``ar`` observations are conditioned on.
    switch : str, list of str or array-like, optional
        Regressors with state-dependent coefficients.
    switch_ar : bool, default False
        State-dependent autoregressive coefficients.
    switch_variance : bool, default False
        State-dependent error variance.
    constant : {True, False, 'common'}, default True
        ``True``: a constant in each state. ``'common'``: one constant.
        ``False``: none.
    vce : {'oim', 'robust'}, default 'oim'
        Inverse observed information, or the sandwich with the outer
        product of the scores (times ``n / (n - 1)``).
    starts : int, default 5
        Number of starting values: one from the quantiles of the pooled
        residuals and ``starts - 1`` random ones. Each runs EM and then
        Newton iterations; the highest likelihood is kept.
    seed : int, optional
        Seed of the random starting values.
    maxiter : int, default 500
        Limit on EM iterations and on quasi-Newton iterations per start.
    tol : float, default 1e-9
        Convergence when the Newton decrement ``g'(-H)^{-1}g`` is below it.
    alpha : float, default 0.05
        Intervals have level ``1 - alpha``.
    start_params : sequence of arrays, optional
        Further starting values, each a full parameter vector on the
        estimation scale and in the order of ``result.theta``. They are
        tried after the ``starts`` built-in ones and skip the EM phase.

    Returns
    -------
    MarkovSwitchingResult

    Raises
    ------
    MethodIncompatibility
        Unknown options, nothing that switches, missing values, or a
        ``model='ar'`` specification outside ``states <= 3``, ``ar <= 4``.
    DataInsufficient
        Fewer usable observations than ``max(20, 3 * n_params)``.

    Notes
    -----
    The initial state distribution is the ergodic distribution of the
    chain (Stata's ``p0(transition)``).

    The likelihood is invariant to relabelling the states, so they are
    ordered after estimation: by increasing constant when it switches,
    otherwise by the first switching coefficient, the first AR coefficient
    or the standard deviation, in that order (``model_info['order_rule']``).

    The likelihood of a switching model has several local maxima, and with
    ``switch_variance=True`` it is unbounded where one state fits a single
    observation. ``result.starts`` shows where each start ended and
    ``model_info['n_distinct_maxima']`` counts the different end points;
    raise ``starts`` when they disagree. Variances are bounded below at
    ``1e-8`` times the sample variance.

    Information criteria are on the ``-2 log L`` scale; Stata's ``mswitch``
    reports them divided by the number of observations.

    Intervals of transition probabilities are built on the logit scale and
    those of expected durations are the transformed intervals of ``p_ii``,
    for any number of states. Stata does the same with two states and
    switches to ``estimate +/- z * se`` for durations with three or more.

    ``fitted`` weights the state means by ``P(s_t | y_{t-1}, ...)``, which
    is what Stata's ``predict, yhat smethod(filter)`` returns; Stata's
    default ``predict, yhat`` applies the transition matrix once more.

    References
    ----------
    [@hamilton1989new],
    [@kim1994dynamic]

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> s = np.zeros(300, dtype=int)
    >>> for t in range(1, 300):
    ...     stay = 0.95 if s[t - 1] == 0 else 0.9
    ...     s[t] = s[t - 1] if rng.uniform() < stay else 1 - s[t - 1]
    >>> y = np.where(s == 0, -1.0, 2.0) + rng.normal(size=300)
    >>> fit = sp.mswitch(y, states=2, seed=0)
    >>> fit.transition.shape
    (2, 2)
    >>> bool((fit.smoothed["state2"] > 0.5).eq(s == 1).mean() > 0.9)
    True
    """
    spec, dat, ys, xn, zn = _setup(
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
        vce,
        starts,
        alpha,
    )
    yv, n = dat.y, dat.n
    extra = [np.asarray(v, dtype=float).ravel() for v in (start_params or [])]
    if any(len(v) != spec.n_par or not np.isfinite(v).all() for v in extra):
        raise MethodIncompatibility(
            f"mswitch: each start_params vector needs {spec.n_par} finite "
            "entries, in the order of result.theta.",
        )
    lnsig_floor = 0.5 * float(np.log(1e-8 * float(np.var(yv))))
    table, fits, best = core.fit_starts(
        spec, dat, int(starts), np.random.default_rng(seed), maxiter, tol, extra
    )
    if not np.isfinite(table["loglik"]).any():
        raise MethodIncompatibility(
            "mswitch: the likelihood could not be evaluated from any start.",
            recovery_hint="Check the scale of the data; try more starts.",
        )
    theta, order_rule = core.reorder(spec, fits[best])
    loglik = float(table.loc[best, "loglik"])
    converged = bool(table.loc[best, "converged"])
    done = table["converged"] if table["converged"].any() else table["loglik"] == loglik
    lls = np.sort(table.loc[done, "loglik"].to_numpy())
    n_max = 1 + int((np.diff(lls) > 1e-5 * (1.0 + np.abs(lls[1:]))).sum())

    vcov, notes = core.covariance(spec, dat, theta, vce)
    if not converged:
        notes.append("the Newton iterations did not converge at the reported maximum")
        warnings.warn(
            "mswitch: no start converged; the reported estimates are the best "
            "point reached. Try more starts or a simpler model.",
            ConvergenceWarning,
            stacklevel=2,
        )
    if n_max > 1:
        notes.append(
            f"the starts converged to {n_max} different local maxima; the "
            "highest is reported"
        )
    n_fail = int((~table["converged"]).sum())
    if n_fail and converged:
        notes.append(
            f"{n_fail} of {len(table)} starts did not reach a maximum "
            f"({int(table['degenerate'].sum())} ended with a state that is "
            "never visited)"
        )
    if np.isnan(vcov).any():
        warnings.warn(f"mswitch: {notes[0]}.", ConvergenceWarning, stacklevel=2)
    sig_at_floor = theta[spec.n_mean : spec.n_mean + spec.n_sig] < lnsig_floor + 1e-6
    if sig_at_floor.any():
        notes.append("a state variance is at its lower bound (degenerate state)")
        warnings.warn(
            "mswitch: a state variance collapsed to its lower bound; the "
            "likelihood is unbounded there. Use switch_variance=False or "
            "fewer states.",
            ConvergenceWarning,
            stacklevel=2,
        )

    layout = _layout(spec, xn, zn)
    params, trans, ttable, durations = _tables(spec, theta, vcov, layout, alpha)
    fs = core.filter_smooth(spec, dat, theta)
    index = ys.index[spec.p :]
    names = [f"state{s + 1}" for s in range(spec.k)]
    yhat = (fs["pred_exp"] * (yv[spec.p :, None] - fs["resid_exp"])).sum(axis=1)
    parts = {key: v[0] for key, v in core.unpack(spec, theta[None, :]).items()}
    pnames = [nm for nm, _, _ in layout]
    k_par = spec.n_par
    return MarkovSwitchingResult(
        params=params,
        transition=trans,
        transition_table=ttable,
        durations=durations,
        filtered=pd.DataFrame(fs["filt"], index=index, columns=names),
        smoothed=pd.DataFrame(fs["smooth"], index=index, columns=names),
        predicted=pd.DataFrame(fs["pred"], index=index, columns=names),
        fitted=pd.Series(yhat, index=index, name="fitted"),
        resid=pd.Series(yv[spec.p :] - yhat, index=index, name="resid"),
        theta=pd.Series(theta, index=pnames),
        vcov=pd.DataFrame(vcov, index=pnames, columns=pnames),
        loglik=loglik,
        aic=-2.0 * loglik + 2.0 * k_par,
        bic=-2.0 * loglik + k_par * float(np.log(n)),
        hqic=-2.0 * loglik + 2.0 * k_par * float(np.log(np.log(n))),
        n_obs=int(n),
        n_params=int(k_par),
        states=int(spec.k),
        model=model,
        ar=int(ar),
        vce=vce,
        converged=converged,
        starts=table,
        alpha=float(alpha),
        model_info={
            "order_rule": order_rule,
            "n_distinct_maxima": n_max,
            "ergodic": core.ergodic(fs["P"][None])[0].tolist(),
            "initial": "ergodic",
            "notes": notes,
        },
        _state={
            "spec": spec,
            "mu": parts["mu"],
            "phi": parts["phi"],
            "P": fs["P"],
            "filt_exp": fs["filt_exp"],
            "y": yv,
            "data": dat,
        },
    )
