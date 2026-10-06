"""
Dynamic linear models: regression with time-varying coefficients.

``sp.dlm`` fits

    y_t    = x_t' beta_t + e_t,          e_t ~ N(0, V)
    beta_t = beta_{t-1} + w_t,           w_t ~ N(0, diag(W))

by the Kalman filter and smoother, with the variances estimated by
maximum likelihood or sampled by Gibbs with forward filtering and
backward sampling (Carter and Kohn 1994; Fruhwirth-Schnatter 1994). With
``y ~ 1`` it is the local level model. A coefficient whose state variance
is fixed at zero is constant, so ordinary regression is the special case
with every state variance zero.

Filter, smoother and likelihood follow Petris, Petrone and Campagnoli
(2009) and agree with the R package ``dlm`` on the same variances.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from .._result_serialize import ResultProtocolMixin
from ..core.utils import create_design_matrices
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility

_LOG_2PI = float(np.log(2.0 * np.pi))


def _filter_py(
    y: np.ndarray,
    X: np.ndarray,
    V: float,
    W: np.ndarray,
    m0: np.ndarray,
    C0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Kalman filter for a random-walk coefficient regression.

    Returns filtered means ``m`` and covariances ``C`` (index 0 is the
    prior), one-step state covariances ``R``, forecasts ``f``, their
    variances ``Q`` and the log likelihood. A missing ``y_t`` (NaN) is
    skipped: the state is propagated and nothing is added to the
    likelihood.
    """
    n, k = X.shape
    m = np.zeros((n + 1, k))
    C = np.zeros((n + 1, k, k))
    R = np.zeros((n, k, k))
    f = np.zeros(n)
    Q = np.zeros(n)
    m[0] = m0
    C[0] = C0
    ll = 0.0
    for t in range(n):
        Rt = C[t].copy()
        for j in range(k):
            Rt[j, j] += W[j]
        R[t] = Rt
        x = X[t]
        Rx = Rt @ x
        ft = 0.0
        for j in range(k):
            ft += x[j] * m[t, j]
        Qt = V
        for j in range(k):
            Qt += x[j] * Rx[j]
        f[t] = ft
        Q[t] = Qt
        if np.isnan(y[t]):
            m[t + 1] = m[t]
            C[t + 1] = Rt
            continue
        e = y[t] - ft
        ll += -0.5 * (_LOG_2PI + np.log(Qt) + e * e / Qt)
        # Joseph form of the covariance update, (I - K x') R (I - K x')' +
        # V K K': the textbook R - K Q K' loses most of its digits under a
        # diffuse prior (R of order 1e7)
        K = Rx / Qt
        A = np.eye(k)
        for i in range(k):
            m[t + 1, i] = m[t, i] + K[i] * e
            for j in range(k):
                A[i, j] -= K[i] * x[j]
        Cn = A @ Rt @ A.T
        for i in range(k):
            for j in range(k):
                Cn[i, j] += V * K[i] * K[j]
        C[t + 1] = 0.5 * (Cn + Cn.T)
    return m, C, R, f, Q, ll


def _smooth_py(
    m: np.ndarray, C: np.ndarray, R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel smoother for random-walk states."""
    n = R.shape[0]
    k = m.shape[1]
    s = np.zeros((n + 1, k))
    S = np.zeros((n + 1, k, k))
    s[n] = m[n]
    S[n] = C[n]
    for t in range(n - 1, -1, -1):
        Rinv = np.linalg.inv(R[t])
        J = C[t] @ Rinv
        s[t] = m[t] + J @ (s[t + 1] - m[t])
        S[t] = C[t] + J @ (S[t + 1] - R[t]) @ J.T
    return s, S


def _backward_sample_py(
    m: np.ndarray, C: np.ndarray, R: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """One draw of the whole state path given the filter output (FFBS).

    ``z`` holds standard normal draws, shape ``(n + 1, k)``.
    """
    n = R.shape[0]
    k = m.shape[1]
    out = np.zeros((n + 1, k))
    L = np.linalg.cholesky(0.5 * (C[n] + C[n].T) + 1e-14 * np.eye(k))
    out[n] = m[n] + L @ z[n]
    for t in range(n - 1, -1, -1):
        Rinv = np.linalg.inv(R[t])
        J = C[t] @ Rinv
        h = m[t] + J @ (out[t + 1] - m[t])
        H = C[t] - J @ R[t] @ J.T
        H = 0.5 * (H + H.T)
        # H can be singular where a state variance is zero
        w, U = np.linalg.eigh(H)
        for j in range(k):
            if w[j] < 0.0:
                w[j] = 0.0
        out[t] = h + U @ (np.sqrt(w) * z[t])
    return out


_KERNELS: Dict[str, Any] = {}


def _kernels() -> Dict[str, Any]:
    """The three recursions, compiled by numba on first use.

    numba is imported here and not at module import: ``import statspai``
    must stay light, and this module is loaded with the package.
    """
    if not _KERNELS:
        from numba import njit  # type: ignore[import-untyped]

        _KERNELS["filter"] = njit(cache=True)(_filter_py)
        _KERNELS["smooth"] = njit(cache=True)(_smooth_py)
        _KERNELS["backward"] = njit(cache=True)(_backward_sample_py)
    return _KERNELS


@dataclass
class DLMResult(ResultProtocolMixin):
    """Result of :func:`dlm`.

    Attributes
    ----------
    variances : pd.DataFrame
        The observation variance (``obs``) and the state variance of
        every coefficient (``state:<term>``): the estimate, and for
        ``method='gibbs'`` the posterior sd, interval and effective
        sample size.
    smoothed : pd.DataFrame
        The coefficient paths given the whole sample, one column per
        term, with ``<term>_sd`` (and ``<term>_lower`` / ``<term>_upper``).
    filtered : pd.DataFrame
        The coefficient paths given the data up to each date.
    fitted : pd.Series
        One-step-ahead forecasts of ``y``.
    loglik : float
        Log likelihood at the variances (maximum likelihood), or at
        their posterior means (Gibbs).
    params : pd.Series
        The smoothed coefficients at the last date.
    draws : pd.DataFrame or None
        Gibbs draws of the variances.
    method, n_obs, level

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> T = 150
    >>> x = rng.normal(size=T)
    >>> slope = 1 + np.cumsum(0.1 * rng.normal(size=T))
    >>> df = pd.DataFrame({"x": x, "y": 0.5 + slope * x + 0.3 * rng.normal(size=T)})
    >>> fit = sp.dlm("y ~ x", df)
    >>> isinstance(fit, sp.DLMResult)
    True
    >>> list(fit.smoothed.columns)[:2]
    ['Intercept', 'x']
    >>> fit.forecast(2, pd.DataFrame({"x": [0.0, 1.0]})).shape
    (2, 3)
    """

    variances: pd.DataFrame
    smoothed: pd.DataFrame
    filtered: pd.DataFrame
    fitted: pd.Series
    loglik: float
    params: pd.Series
    method: str
    formula: str
    n_obs: int
    level: float = 0.95
    draws: Optional[pd.DataFrame] = None
    model_info: Dict[str, Any] = field(default_factory=dict)
    _state: Dict[str, Any] = field(default_factory=dict, repr=False)

    _citation_keys = ("petris2009dynamic", "carter1994gibbs", "fruhwirth1994data")

    @property
    def coef(self) -> pd.Series:
        return self.params

    def summary(self) -> str:
        names = self._state["names"]
        lines = [
            f"Dynamic linear model ({self.method})    {self.formula}",
            f"Observations: {self.n_obs}    Log likelihood: {self.loglik:.4f}",
            "",
            "Variances",
            self.variances.to_string(float_format=lambda v: f"{v:.5g}"),
            "",
            "Smoothed coefficients (first, middle and last date)",
        ]
        idx = [0, len(self.smoothed) // 2, len(self.smoothed) - 1]
        lines.append(
            self.smoothed.iloc[idx][names].to_string(float_format=lambda v: f"{v:.5g}")
        )
        for note in self.model_info.get("notes", []):
            lines.append("")
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "method": self.method,
            "formula": self.formula,
            "n_obs": int(self.n_obs),
            "loglik": float(self.loglik),
            "variances": {
                str(i): {c: float(r[c]) for c in self.variances.columns}
                for i, r in self.variances.iterrows()
            },
            "last_coefficients": {str(i): float(v) for i, v in self.params.items()},
            "model_info": {
                k: v
                for k, v in self.model_info.items()
                if not isinstance(v, np.ndarray)
            },
        }

    def forecast(
        self, steps: int = 1, data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Forecast ``y`` beyond the sample.

        Parameters
        ----------
        steps : int
            Horizon.
        data : DataFrame, optional
            Regressor values for the forecast dates (``steps`` rows).
            Not needed for a model with only an intercept.

        Returns
        -------
        pd.DataFrame
            ``mean``, ``lower`` and ``upper``: the coefficients stay at
            their last filtered mean and their uncertainty grows by the
            state variances each step. The interval is for the outcome
            (it includes the observation variance). With
            ``method='gibbs'`` the variances are at their posterior means.
        """
        st = self._state
        names: List[str] = st["names"]
        k = len(names)
        if data is None:
            if names != ["Intercept"]:
                raise MethodIncompatibility(
                    "Pass data= with the regressors of the forecast dates."
                )
            Xf = np.ones((steps, 1))
        else:
            cols = [c for c in names if c != "Intercept"]
            missing = [c for c in cols if c not in data.columns]
            if missing:
                raise MethodIncompatibility(f"data lacks the regressors {missing}.")
            if len(data) < steps:
                raise MethodIncompatibility(
                    f"data has {len(data)} rows; {steps} steps were requested."
                )
            parts = [
                np.ones(steps) if c == "Intercept" else data[c].to_numpy(float)[:steps]
                for c in names
            ]
            Xf = np.column_stack(parts)
        from scipy import stats

        z = stats.norm.ppf(0.5 + self.level / 2.0)
        m, C = st["m_last"], st["C_last"].copy()
        V, W = st["V"], st["W"]
        rows = []
        for h in range(steps):
            C = C + np.diag(W)
            x = Xf[h]
            mean = float(x @ m)
            sd = float(np.sqrt(x @ C @ x + V))
            rows.append((mean, mean - z * sd, mean + z * sd))
        _ = k
        return pd.DataFrame(rows, columns=["mean", "lower", "upper"])

    def plot(self, terms: Optional[Sequence[str]] = None) -> Any:
        """Smoothed coefficient paths with their intervals."""
        import matplotlib.pyplot as plt

        names = list(terms) if terms is not None else self._state["names"]
        fig, axes = plt.subplots(
            len(names), 1, figsize=(7.0, 2.4 * len(names)), squeeze=False
        )
        t = np.arange(len(self.smoothed))
        for ax, nm in zip(axes[:, 0], names):
            ax.plot(t, self.smoothed[nm].to_numpy(), lw=1.2)
            ax.fill_between(
                t,
                self.smoothed[f"{nm}_lower"].to_numpy(),
                self.smoothed[f"{nm}_upper"].to_numpy(),
                alpha=0.25,
            )
            ax.set_title(str(nm))
        fig.tight_layout()
        return fig


def _paths(
    mean: np.ndarray, sd: np.ndarray, names: List[str], index: Any, z: float
) -> pd.DataFrame:
    out = pd.DataFrame(mean, columns=names, index=index)
    for j, nm in enumerate(names):
        out[f"{nm}_sd"] = sd[:, j]
        out[f"{nm}_lower"] = mean[:, j] - z * sd[:, j]
        out[f"{nm}_upper"] = mean[:, j] + z * sd[:, j]
    return out


def dlm(
    formula: str,
    data: pd.DataFrame,
    method: str = "mle",
    time: Optional[str] = None,
    constant: Optional[Sequence[str]] = None,
    obs_var: Optional[float] = None,
    state_var: Optional[Any] = None,
    m0: Any = 0.0,
    C0: Any = 1e7,
    obs_var_prior: Tuple[float, float] = (0.001, 0.001),
    state_var_prior: Tuple[float, float] = (0.001, 0.001),
    draws: int = 5000,
    burnin: int = 1000,
    thin: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> DLMResult:
    """Dynamic linear model: regression with time-varying coefficients.

    ``y_t = x_t' beta_t + e_t`` with every coefficient a random walk,
    ``beta_t = beta_{t-1} + w_t``. Returns the filtered and smoothed
    coefficient paths, one-step forecasts and the variances of ``e`` and
    ``w``. ``'y ~ 1'`` is the local level model.

    Parameters
    ----------
    formula : str
        ``'y ~ x1 + x2'``. Rows are taken in the order of ``data`` unless
        ``time=`` is given.
    data : DataFrame
    method : {'mle', 'gibbs'}
        ``'mle'``: the variances maximise the Kalman-filter likelihood.
        ``'gibbs'``: inverse-gamma priors on the variances, states drawn
        by forward filtering and backward sampling.
    time : str, optional
        Column to sort by.
    constant : list of str, optional
        Terms whose coefficient does not vary (state variance fixed at
        zero).
    obs_var, state_var : float / array, optional
        Fix the observation variance and / or the state variances
        instead of estimating them (``state_var``: one number, or one per
        term). With both given the function only filters and smooths.
    m0, C0 : float, array or matrix
        Mean and covariance of the coefficients before the first
        observation. The default covariance ``1e7 I`` is the diffuse
        choice of R ``dlm::dlmModReg``.
    obs_var_prior, state_var_prior : (alpha0, delta0)
        ``method='gibbs'``: each variance is
        ``InvGamma(alpha0 / 2, delta0 / 2)``, as in
        :func:`statspai.bayes_regress`.
    draws, burnin, thin, seed
        Gibbs settings; ``draws`` are kept after ``burnin``.
    level : float, default 0.95
        Coverage of the intervals.

    Returns
    -------
    DLMResult

    Notes
    -----
    The first dates are dominated by the diffuse prior: with ``k``
    coefficients the first ``k`` one-step forecasts have enormous
    variance and contribute a constant to the likelihood. This is the
    convention of R ``dlm``; likelihoods are comparable across models
    with the same ``k`` and ``C0`` only.

    A state variance estimated at (numerically) zero means the data do
    not ask for that coefficient to move; the result notes it.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> level = np.cumsum(0.3 * rng.normal(size=120))
    >>> df = pd.DataFrame({"y": level + rng.normal(size=120)})
    >>> fit = sp.dlm("y ~ 1", df)                      # local level
    >>> list(fit.variances.index)
    ['obs', 'state:Intercept']
    >>> fc = fit.forecast(3)
    >>> g = sp.dlm("y ~ 1", df, method="gibbs", draws=500, burnin=200, seed=1)

    References
    ----------
    petris2009dynamic, carter1994gibbs, fruhwirth1994data, kalman1960new
    """
    method = str(method).lower()
    if method not in ("mle", "gibbs"):
        raise MethodIncompatibility("method must be 'mle' or 'gibbs'.")
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("data must be a pandas DataFrame.")
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    work = data
    if time is not None:
        if time not in data.columns:
            raise MethodIncompatibility(f"time column {time!r} is not in data.")
        work = data.sort_values(time, kind="stable")
    y_df, X_df = create_design_matrices(formula, work)
    if len(X_df) < len(work):
        raise MethodIncompatibility(
            f"{len(work) - len(X_df)} rows have missing values in the model's "
            "variables. Dropping them would join dates that are not "
            "adjacent; fill or remove them explicitly first."
        )
    y = np.ascontiguousarray(np.asarray(y_df, dtype=float).reshape(-1))
    X = np.ascontiguousarray(np.asarray(X_df, dtype=float))
    names = [str(c) for c in X_df.columns]
    index = work.loc[X_df.index, time] if time is not None else X_df.index
    n, k = X.shape
    if n < k + 3:
        raise DataInsufficient(f"{n} observations for {k} time-varying coefficients.")
    constant = list(constant or [])
    unknown = [c for c in constant if c not in names]
    if unknown:
        raise MethodIncompatibility(
            f"constant= names terms that are not in the model: {unknown}. "
            f"Terms: {names}."
        )
    free = np.array([nm not in constant for nm in names])

    m0v = np.asarray(m0, dtype=float)
    if m0v.ndim == 0:
        m0v = np.full(k, float(m0v))
    if m0v.shape != (k,):
        raise MethodIncompatibility(f"m0 must be a scalar or have {k} entries.")
    C0m = np.asarray(C0, dtype=float)
    if C0m.ndim == 0:
        C0m = np.eye(k) * float(C0m)
    elif C0m.ndim == 1 and C0m.shape == (k,):
        C0m = np.diag(C0m)
    if C0m.shape != (k, k):
        raise MethodIncompatibility(
            f"C0 must be a scalar, {k} variances or a {k} x {k} matrix."
        )

    W_fixed: Optional[np.ndarray] = None
    if state_var is not None:
        sv = np.asarray(state_var, dtype=float)
        W_fixed = np.full(k, float(sv)) if sv.ndim == 0 else sv.astype(float)
        if W_fixed.shape != (k,) or np.any(W_fixed < 0):
            raise MethodIncompatibility(
                f"state_var must be a non-negative number or {k} of them."
            )
        W_fixed = np.where(free, W_fixed, 0.0)
    if obs_var is not None and not obs_var > 0:
        raise MethodIncompatibility("obs_var must be positive.")
    from scipy import stats

    z = float(stats.norm.ppf(0.5 + level / 2.0))
    notes: List[str] = []
    draws_df: Optional[pd.DataFrame] = None
    var_names = ["obs"] + [f"state:{nm}" for nm in names]

    def run(V: float, W: np.ndarray) -> Tuple[Any, ...]:
        w_arr = np.ascontiguousarray(W, dtype=float)
        out: Tuple[Any, ...] = _kernels()["filter"](y, X, float(V), w_arr, m0v, C0m)
        return out

    if method == "mle":
        scale = float(np.nanvar(y)) or 1.0
        n_w = int(free.sum()) if W_fixed is None else 0
        n_v = 0 if obs_var is not None else 1

        def unpack(par: np.ndarray) -> Tuple[float, np.ndarray]:
            V = float(obs_var) if obs_var is not None else float(np.exp(par[0]))
            if W_fixed is not None:
                W = W_fixed
            else:
                W = np.zeros(k)
                W[free] = np.exp(par[n_v:])
            return V, W

        if n_v + n_w > 0:

            def nll(par: np.ndarray) -> float:
                V, W = unpack(par)
                val = -run(V, W)[5]
                return float(val) if np.isfinite(val) else 1e300

            best = None
            for start_w in (-2.0, -5.0, 0.0):
                p0 = np.concatenate(
                    [
                        np.full(n_v, np.log(scale * 0.5)),
                        np.full(n_w, np.log(scale) + start_w),
                    ]
                )
                res = optimize.minimize(
                    nll,
                    p0,
                    method="L-BFGS-B",
                    bounds=[(-40.0, 40.0)] * p0.size,
                    options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000},
                )
                if best is None or res.fun < best.fun:
                    best = res
            assert best is not None
            # the likelihood is flat near its maximum: polish
            pol = optimize.minimize(
                nll,
                best.x,
                method="Nelder-Mead",
                options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 4000},
            )
            if pol.fun < best.fun:
                pol.success, pol.message = True, "CONVERGENCE (polished)"
                best = pol
            if not best.success and "CONVERGENCE" not in str(best.message).upper():
                warnings.warn(
                    f"The likelihood optimiser stopped early: {best.message}",
                    ConvergenceWarning,
                    stacklevel=2,
                )
            V, W = unpack(best.x)
        else:
            assert W_fixed is not None and obs_var is not None
            V, W = float(obs_var), W_fixed
        tiny = [nm for nm, w_j, fr in zip(names, W, free) if fr and w_j < 1e-10 * scale]
        if tiny and W_fixed is None:
            notes.append(
                "the state variance of "
                + ", ".join(tiny)
                + " is estimated at zero: the data do not ask for "
                "this coefficient to vary"
            )
        var_table = pd.DataFrame({"estimate": np.append(V, W)}, index=var_names)
        m, C, R, f, Q, ll = run(V, W)
        s, S = _kernels()["smooth"](m, C, R)
        sm_mean, sm_sd = s[1:], np.sqrt(np.clip(np.einsum("tii->ti", S[1:]), 0.0, None))
    else:
        from ..mcmc._core import check_mcmc_args
        from ..mcmc.diagnostics import mcmc_summary

        check_mcmc_args(draws, burnin, thin)
        a_v, d_v = (float(v) for v in obs_var_prior)
        a_w, d_w = (float(v) for v in state_var_prior)
        if min(a_v, d_v, a_w, d_w) <= 0:
            raise MethodIncompatibility(
                "obs_var_prior and state_var_prior must be positive pairs."
            )
        rng = np.random.default_rng(seed)
        scale = float(np.var(y)) or 1.0
        V = float(obs_var) if obs_var is not None else 0.5 * scale
        W = W_fixed.copy() if W_fixed is not None else np.where(free, 0.01 * scale, 0.0)
        n_iter = burnin + draws * thin
        kept = np.empty((draws, 1 + k))
        s1 = np.zeros((n, k))
        s2 = np.zeros((n, k))
        lo_q = (1.0 - level) / 2.0
        store = np.empty((draws, n, k)) if draws * n * k <= 20_000_000 else None
        j = 0
        backward = _kernels()["backward"]
        for it in range(n_iter):
            m, C, R, f, Q, _ll = run(V, W)
            z_draw = rng.standard_normal((n + 1, k))
            theta = backward(m, C, R, z_draw)
            if obs_var is None:
                e = y - np.einsum("tk,tk->t", X, theta[1:])
                V = (d_v + float(e @ e)) / 2.0 / rng.gamma((a_v + n) / 2.0)
            if W_fixed is None:
                dth = np.diff(theta, axis=0)
                ssw = (dth * dth).sum(axis=0)
                for q in range(k):
                    if free[q]:
                        W[q] = (d_w + ssw[q]) / 2.0 / rng.gamma((a_w + n) / 2.0)
            if it >= burnin and (it - burnin) % thin == 0:
                kept[j, 0] = V
                kept[j, 1:] = W
                s1 += theta[1:]
                s2 += theta[1:] ** 2
                if store is not None:
                    store[j] = theta[1:]
                j += 1
        draws_df = pd.DataFrame(kept, columns=var_names)
        summ = mcmc_summary(draws_df, quantiles=(lo_q, 0.5, 1.0 - lo_q))
        var_table = pd.DataFrame(
            {
                "estimate": summ["mean"],
                "sd": summ["sd"],
                "lower": summ.iloc[:, -3],
                "median": summ.iloc[:, -2],
                "upper": summ.iloc[:, -1],
                "ess": summ["ess"],
            }
        )
        sm_mean = s1 / draws
        sm_sd = np.sqrt(np.clip(s2 / draws - sm_mean**2, 0.0, None))
        V, W = float(kept[:, 0].mean()), kept[:, 1:].mean(axis=0)
        m, C, R, f, Q, ll = run(V, W)
        moving = var_table["ess"].to_numpy()[np.append(obs_var is None, free)]
        if moving.size and np.nanmin(moving) < 100:
            warnings.warn(
                "The Gibbs sampler mixes slowly for the variances (effective "
                f"sample size {np.nanmin(moving):.0f}). Increase draws or "
                "thin.",
                ConvergenceWarning,
                stacklevel=2,
            )
    smoothed = _paths(sm_mean, sm_sd, names, index, z)
    if method == "gibbs" and store is not None:
        lo_q = (1.0 - level) / 2.0
        ql = np.quantile(store, lo_q, axis=0)
        qu = np.quantile(store, 1.0 - lo_q, axis=0)
        for q, nm in enumerate(names):
            smoothed[f"{nm}_lower"] = ql[:, q]
            smoothed[f"{nm}_upper"] = qu[:, q]
    filt_sd = np.sqrt(np.clip(np.einsum("tii->ti", C[1:]), 0.0, None))
    filtered = _paths(m[1:], filt_sd, names, index, z)
    fitted = pd.Series(f, index=index, name="one_step_forecast")
    info: Dict[str, Any] = {
        "notes": notes,
        "terms": names,
        "constant_terms": constant,
        "forecast_variance": Q,
    }
    return DLMResult(
        variances=var_table,
        smoothed=smoothed,
        filtered=filtered,
        fitted=fitted,
        loglik=float(ll),
        params=pd.Series(sm_mean[-1], index=names),
        method=method,
        formula=formula,
        n_obs=int(n),
        level=level,
        draws=draws_df,
        model_info=info,
        _state={
            "names": names,
            "m_last": m[-1].copy(),
            "C_last": C[-1].copy(),
            "V": float(V),
            "W": np.asarray(W, dtype=float).copy(),
        },
    )
