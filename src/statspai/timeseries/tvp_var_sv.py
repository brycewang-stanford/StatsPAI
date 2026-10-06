"""Bayesian TVP-VAR with stochastic volatility by Gibbs sampling.

``y_t = c_t + A_{1,t} y_{t-1} + ... + A_{p,t} y_{t-p} + u_t`` with
``u_t = A_t^{-1} diag(sigma_t) eps_t``: the coefficients, the free
elements of the unit lower-triangular ``A_t`` and the log standard
deviations ``log sigma_t`` of the orthogonal shocks all follow random
walks. :func:`statspai.tvp_var` keeps the error covariance constant (or
discounts it); here it moves, and everything comes with posterior bands.

The kernels are in ``_tvp_var_sv_core.py``, the result class in
``_tvp_var_sv_results.py``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    StatsPAIWarning,
)
from ._tvp_var_sv_core import kernels, lag_matrix, mixture, simple_prior, training_prior
from ._tvp_var_sv_results import TVPVARSVResult

__all__ = ["tvp_var_sv", "TVPVARSVResult"]

_PRIOR_KEYS = {
    "b0": "b0",
    "b0_var": "PB",
    "a0": "a0",
    "a0_var": "PA",
    "logsig0": "h0",
    "logsig0_var": "PH",
    "Q_scale": "Qs",
    "Q_df": "Qdf",
    "S_scale": "Ss",
    "S_df": "Sdf",
    "W_scale": "Ws",
    "W_df": "Wdf",
}


def _apply_user_prior(base: Dict[str, Any], user: Dict[str, Any], K: int) -> None:
    """Overwrite entries of ``base`` with the user's, checking shapes."""
    unknown = sorted(set(user) - set(_PRIOR_KEYS))
    if unknown:
        raise MethodIncompatibility(
            f"tvp_var_sv: unknown prior entries {unknown}.",
            recovery_hint=f"Allowed: {sorted(_PRIOR_KEYS)}.",
        )
    for key, value in user.items():
        name = _PRIOR_KEYS[key]
        ref = np.asarray(base[name], dtype=float)
        val = np.asarray(value, dtype=float)
        if ref.ndim == 0:
            if val.ndim != 0:
                raise MethodIncompatibility(f"tvp_var_sv: prior[{key!r}] is a scalar.")
            base[name] = float(val)
            continue
        if ref.ndim == 2 and val.ndim == 0:
            val = float(val) * np.eye(ref.shape[0])
        elif ref.ndim == 2 and val.ndim == 1:
            val = np.diag(val)
        elif ref.ndim == 1 and val.ndim == 0:
            val = np.full(ref.shape, float(val))
        if val.shape != ref.shape:
            raise MethodIncompatibility(
                f"tvp_var_sv: prior[{key!r}] has shape {val.shape}; expected "
                f"{ref.shape} (a scalar or a diagonal are also accepted)."
            )
        base[name] = np.ascontiguousarray(val, dtype=float)
    for name in ("PB", "PA", "PH", "Qs", "Ws"):
        mat = base[name]
        if mat.size and (
            not np.allclose(mat, mat.T) or np.linalg.eigvalsh(mat).min() <= 0.0
        ):
            raise MethodIncompatibility(
                f"tvp_var_sv: the prior matrix {name} is not symmetric "
                "positive definite."
            )
    m = base["b0"].shape[0]
    if base["Qdf"] <= m + 1 or base["Wdf"] <= K - 1:
        raise MethodIncompatibility(
            "tvp_var_sv: the inverse-Wishart degrees of freedom are too small "
            f"(Q_df must exceed {m + 1}, W_df must exceed {K - 1}).",
        )
    pos = 0
    for i in range(1, K):
        blk = base["Ss"][pos : pos + i, pos : pos + i]
        if np.linalg.eigvalsh(blk).min() <= 0.0 or base["Sdf"][i - 1] <= i - 1:
            raise MethodIncompatibility(
                f"tvp_var_sv: the S prior of equation {i + 1} is not proper."
            )
        pos += i


def tvp_var_sv(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    lags: int = 1,
    time: Optional[str] = None,
    training: Optional[int] = 40,
    prior: Optional[Dict[str, Any]] = None,
    k_Q: float = 0.01,
    k_S: float = 0.1,
    k_W: float = 0.01,
    k_B: float = 4.0,
    k_A: float = 4.0,
    k_sig: float = 1.0,
    draws: int = 5000,
    burnin: int = 2000,
    thin: int = 1,
    seed: Optional[int] = None,
    stationary: bool = False,
    max_tries: int = 100,
    offset: float = 0.001,
    alpha: float = 0.05,
) -> TVPVARSVResult:
    """Time-varying-parameter VAR with stochastic volatility (Gibbs sampler).

    ``y_t = c_t + A_{1,t} y_{t-1} + ... + A_{p,t} y_{t-p} + u_t`` with
    ``A_t u_t = diag(sigma_t) eps_t``. The coefficients, the free elements
    of the unit lower-triangular ``A_t`` and ``log sigma_t`` are random
    walks with innovation covariances ``Q``, ``S`` (one block per
    equation) and ``W``. The error covariance
    ``Omega_t = A_t^{-1} diag(sigma_t^2) A_t^{-1}'`` therefore changes in
    both size and shape, and a change in the size of the shocks is
    separated from a change in how the economy responds to them.

    Parameters
    ----------
    data : DataFrame
        Rows in time order (or sorted by ``time=``), no missing values.
    variables : list of str, optional
        Default: every numeric column (except ``time``). Their order is
        the recursive ordering: a shock to a variable moves the variables
        after it within the period, not those before it.
    lags : int, default 1
    time : str, optional
        Column to sort by; its values label the dates.
    training : int or None, default 40
        Number of initial rows used only to set the prior: a
        constant-coefficient VAR is fitted to them by least squares and
        the posterior is computed on the remaining rows. ``None``: no
        training sample, a weakly informative prior instead (see Notes),
        and all rows after the first ``lags`` are used.
    prior : dict, optional
        Overrides single elements of the prior: ``b0``, ``b0_var``
        (coefficients at the first date, stacked equation by equation in
        the order of ``terms``), ``a0``, ``a0_var`` (free elements of
        ``A`` row by row), ``logsig0``, ``logsig0_var``, and the
        inverse-Wishart ``Q_scale``, ``Q_df``, ``S_scale``, ``S_df`` (one
        per equation from the second on), ``W_scale``, ``W_df``. A scalar
        or a vector stands for a diagonal matrix.
    k_Q, k_S, k_W : float, default 0.01, 0.1, 0.01
        Prior beliefs about the amount of time variation in the
        coefficients, the covariance states and the log volatilities: the
        prior scale matrices are proportional to their squares.
    k_B, k_A, k_sig : float, default 4, 4, 1
        Prior variance of the states at the first date, as multiples of
        the least-squares covariance (coefficients, covariance states) or
        of the identity (log volatilities). Used with ``training`` only.
    draws : int, default 5000
        Posterior draws kept.
    burnin : int, default 2000
    thin : int, default 1
        Keep every ``thin``-th sweep.
    seed : int, optional
    stationary : bool, default False
        Reject coefficient paths for which the VAR frozen at some date
        has an explosive root. The default keeps every draw, as the
        original sampler does, and ``.stability()`` reports how many are
        explosive at each date. With ``True`` up to ``max_tries`` paths
        are drawn per sweep and the previous path is kept when all are
        explosive; the hyperparameter updates are those of the
        unrestricted model, which is the usual practice and an
        approximation.
    max_tries : int, default 100
    offset : float, default 0.001
        Added to the squared orthogonal residuals before taking logs, so
        that a residual of zero does not send the volatility to zero. It
        is in squared units of the data and is meant to be negligible
        next to the shock variances: the default suits series in percent.
        A warning is issued when a prior shock variance is below 100
        times the offset; rescale the data or lower the offset then.
    alpha : float, default 0.05
        Bands are equal-tailed with pointwise coverage ``1 - alpha``.

    Returns
    -------
    TVPVARSVResult

    Raises
    ------
    DataInsufficient
        Too few rows in the training sample or after it.
    MethodIncompatibility
        Bad arguments, missing values, fewer than two variables, an
        explosive starting point with ``stationary=True``.

    Notes
    -----
    *Sampler.* One sweep draws (1) the coefficient path from its Gaussian
    conditional by forward filtering and backward sampling, (2) the
    covariance states one equation at a time (given the residuals each
    equation is a regression with random-walk coefficients on the
    residuals of the equations above), (3) the mixture indicators and
    then the log volatilities, (4) ``Q``, the blocks of ``S`` and ``W``
    from their inverse-Wishart conditionals. Step (3) replaces the log
    chi-square(1) error of the log squared orthogonal residuals by a
    mixture of seven normals, the approximation of
    :func:`statspai.stochvol`.

    *Ordering.* The indicators are drawn immediately before the
    volatilities and are used by nothing else. They are a device to draw
    the volatilities and must not be carried across the other blocks: the
    coefficient and covariance steps do not condition on them, so an
    indicator drawn before those steps is no longer a draw from its
    conditional when the volatilities use it. The sampler of the 2005
    paper drew the indicators after the volatilities and kept them for
    the next sweep; that is not a Gibbs sampler for this model, and the
    corrigendum to the paper prescribes the order used here. The test
    suite runs a joint-distribution test of both orders on a small model:
    this one passes and the original one fails.

    *Remaining approximation.* Steps (1) and (2) use the Gaussian model
    and step (3) the mixture, so the chain targets the posterior up to the
    error of the seven-normal approximation; no reweighting is done.

    *Priors.* With a training sample of ``tau`` rows and least-squares
    estimates ``B``, ``V(B)``, ``A``, ``V(A)``, ``sigma``: coefficients at
    the first date ``N(B, k_B V(B))``, covariance states
    ``N(A, k_A V(A))``, log volatilities ``N(log sigma, k_sig I)``,
    ``Q ~ IW(k_Q^2 tau V(B), tau)``, block ``i`` of ``S`` (dimension
    ``i``) ``IW(k_S^2 (i + 1) V(A_i), i + 1)`` and
    ``W ~ IW(k_W^2 (K + 1) I, K + 1)``. If ``tau`` is below the number of
    coefficients plus two, that number replaces ``tau`` for ``Q`` (the
    prior must be proper) and a note says so. The priors apply to the
    states of the first estimation date. Without a training sample the
    states start at zero with variance 10 (log volatilities: at the log
    standard deviation of each variable's first difference), and the
    scale matrices are ``k^2 df I``, which only makes sense for variables
    on comparable scales of order one.

    The chain is a single one. Convergence statistics for a set of
    summary functionals are in ``.diagnostics()``; a warning is issued
    when the smallest effective sample size is below 100.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros((140, 2))
    >>> for t in range(1, 140):
    ...     sd = 1.0 if t < 90 else 2.0
    ...     y[t, 0] = 0.5 * y[t - 1, 0] + sd * rng.normal()
    ...     y[t, 1] = 0.3 * y[t - 1, 1] + 0.4 * y[t, 0] + rng.normal()
    >>> df = pd.DataFrame(y, columns=["x", "z"])
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")  # a chain this short mixes badly
    ...     fit = sp.tvp_var_sv(df, lags=1, training=40, draws=300,
    ...                         burnin=200, seed=1)
    >>> fit.volatility().columns.tolist()
    ['date', 'variable', 'median', 'mean', 'lower', 'upper']
    >>> fit.irf(at=[10, 90], periods=8).shape
    (72, 8)

    References
    ----------
    [@primiceri2005time],
    [@delnegro2015time],
    [@kim1998stochastic],
    [@carter1994gibbs],
    [@fruhwirth1994data],
    [@cogley2005drifts],
    [@geweke2004getting]
    """
    from ..mcmc._core import check_mcmc_args, spawn_rngs

    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("tvp_var_sv: data must be a pandas DataFrame.")
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(f"tvp_var_sv: alpha={alpha} is not in (0, 1).")
    if not isinstance(lags, (int, np.integer)) or isinstance(lags, bool) or lags < 1:
        raise MethodIncompatibility("tvp_var_sv: lags must be a positive integer.")
    check_mcmc_args(draws, burnin, thin)
    for nm, val in (("k_Q", k_Q), ("k_S", k_S), ("k_W", k_W), ("k_B", k_B)):
        if not val > 0.0:
            raise MethodIncompatibility(f"tvp_var_sv: {nm} must be positive.")
    if not (k_A > 0.0 and k_sig > 0.0 and offset >= 0.0 and max_tries >= 1):
        raise MethodIncompatibility(
            "tvp_var_sv: k_A, k_sig and max_tries must be positive and "
            "offset non-negative."
        )
    work = data
    if time is not None:
        if time not in data.columns:
            raise MethodIncompatibility(f"tvp_var_sv: time column {time!r} not found.")
        work = data.sort_values(time, kind="stable")
    if variables is None:
        names = [
            str(c) for c in work.select_dtypes(include=[np.number]).columns if c != time
        ]
    else:
        names = [str(v) for v in variables]
        missing = [v for v in names if v not in work.columns]
        if missing:
            raise MethodIncompatibility(
                f"tvp_var_sv: columns {missing} are not in data.",
                recovery_hint="Check the names passed to variables=.",
            )
    if len(names) < 2 or len(set(names)) != len(names):
        raise MethodIncompatibility(
            "tvp_var_sv: variables must name at least two distinct numeric " "columns.",
            recovery_hint="For one series use sp.stochvol or sp.dlm.",
        )
    Yall = work[names].to_numpy(dtype=float)
    if not np.all(np.isfinite(Yall)):
        raise MethodIncompatibility(
            "tvp_var_sv: the variables contain missing or infinite values.",
            recovery_hint="Dropping rows would join dates that are not "
            "adjacent; fill or trim the sample first.",
        )
    K, p = len(names), int(lags)
    k = K * p + 1
    m = K * k
    notes: List[str] = []
    if training is not None:
        if isinstance(training, bool) or not isinstance(training, (int, np.integer)):
            raise MethodIncompatibility("tvp_var_sv: training is an integer or None.")
        tau = int(training)
        if tau - p < k + K + 1:
            raise DataInsufficient(
                f"tvp_var_sv: a training sample of {tau} rows leaves "
                f"{max(tau - p, 0)} observations for {k} coefficients per "
                f"equation; at least {k + K + 1 + p} rows are needed.",
                recovery_hint="Lengthen training=, reduce lags, or pass "
                "training=None with a prior.",
            )
        pr = training_prior(
            Yall[:tau],
            p,
            float(k_B),
            float(k_A),
            float(k_sig),
            float(k_Q),
            float(k_S),
            float(k_W),
        )
        if pr["Qdf"] != tau:
            notes.append(
                f"Q prior: {pr['Qdf']:g} degrees of freedom instead of the "
                f"training size {tau}, which is too small for {m} coefficients"
            )
        start = tau
    else:
        pr = simple_prior(Yall, p, float(k_Q), float(k_S), float(k_W))
        start = p
    if prior is not None:
        if not isinstance(prior, dict):
            raise MethodIncompatibility("tvp_var_sv: prior must be a dict.")
        _apply_user_prior(pr, prior, K)
    small = [names[i] for i in range(K) if np.exp(2.0 * pr["h0"][i]) < 100.0 * offset]
    if small:
        text = (
            f"tvp_var_sv: the prior shock variance of {small} is below 100 "
            f"times offset={offset:g}, so the offset is not negligible and "
            "the volatilities are biased upwards. Rescale the data (for "
            "example to percent) or pass a smaller offset."
        )
        notes.append(text)
        warnings.warn(text, StatsPAIWarning, stacklevel=2)
    Y, X = lag_matrix(Yall[start - p :], p)
    T = Y.shape[0]
    if T < max(k + 3, 10):
        raise DataInsufficient(
            f"tvp_var_sv: {max(T, 0)} dates are left for estimation; at "
            f"least {max(k + 3, 10)} are needed.",
            recovery_hint="Shorten training=, reduce lags, or use a longer sample.",
        )
    labels = work[time] if time is not None else work.index
    index = pd.Index(labels[start:])
    terms = [f"L{lag}.{v}" for lag in range(1, p + 1) for v in names] + ["_cons"]
    Z = np.zeros((T, K, m))
    for i in range(K):
        Z[:, i, i * k : (i + 1) * k] = X
    na = K * (K - 1) // 2
    kern = kernels()
    B = np.ascontiguousarray(np.tile(pr["b0"], (T, 1)))
    if stationary and kern["explosive"](B[:1], K, p):
        raise MethodIncompatibility(
            "tvp_var_sv: the prior mean of the coefficients is an explosive "
            "VAR, so stationary=True has no admissible starting point.",
            recovery_hint="Use stationary=False, difference the trending "
            "variables, or pass prior={'b0': ...}.",
        )
    a = np.ascontiguousarray(np.tile(pr["a0"], (T, 1)))
    h = np.ascontiguousarray(np.tile(pr["h0"], (T, 1)))
    s = np.full((T, K), 4, dtype=np.int64)
    Qc = np.ascontiguousarray(pr["Qs"] / pr["Qdf"])
    Sc = np.zeros((na, na))
    pos = 0
    for i in range(1, K):
        Sc[pos : pos + i, pos : pos + i] = (
            pr["Ss"][pos : pos + i, pos : pos + i] / pr["Sdf"][i - 1]
        )
        pos += i
    Wc = np.ascontiguousarray(pr["Ws"] / pr["Wdf"])
    q, mm, vv = mixture()
    rng = spawn_rngs(seed, 1)[0]
    Bd, ad, hd, Qd, Sd, Wd, stuck, paths = kern["run"](
        rng,
        int(burnin),
        int(draws),
        int(thin),
        Y,
        Z,
        B,
        a,
        h,
        s,
        Qc,
        Sc,
        Wc,
        pr["b0"],
        np.ascontiguousarray(pr["PB"]),
        pr["a0"],
        np.ascontiguousarray(pr["PA"]),
        pr["h0"],
        np.ascontiguousarray(pr["PH"]),
        np.ascontiguousarray(pr["Qs"]),
        float(pr["Qdf"]),
        np.ascontiguousarray(pr["Ss"]),
        np.ascontiguousarray(pr["Sdf"], dtype=float),
        np.ascontiguousarray(pr["Ws"]),
        float(pr["Wdf"]),
        q,
        mm,
        vv,
        float(offset),
        0,
        bool(stationary),
        p,
        int(max_tries),
    )
    if not (np.isfinite(Bd).all() and np.isfinite(hd).all()):
        raise MethodIncompatibility(
            "tvp_var_sv: the chain produced non-finite draws.",
            recovery_hint="Rescale the variables to comparable units, or "
            "tighten the prior (smaller k_Q, k_W).",
        )
    n_sweeps = int(draws) * int(thin)
    if stationary:
        notes.append(
            f"stationary=True: {paths / n_sweeps:.2f} coefficient paths drawn "
            f"per sweep after burn-in; in {stuck} of {n_sweeps} sweeps none "
            "was stable and the previous path was kept"
        )
    res = TVPVARSVResult(
        coef_draws=Bd.reshape(int(draws), T, K, k),
        a_draws=ad,
        logsig_draws=hd,
        q_diag_draws=Qd,
        s_draws=Sd,
        w_draws=Wd,
        index=index,
        var_names=names,
        terms=terms,
        lags=p,
        n_obs=T,
        alpha=float(alpha),
        model_info={
            "training": None if training is None else int(training),
            "prior": {
                key: np.asarray(pr[val]).copy() for key, val in _PRIOR_KEYS.items()
            },
            "k": {"k_Q": k_Q, "k_S": k_S, "k_W": k_W},
            "draws": int(draws),
            "burnin": int(burnin),
            "thin": int(thin),
            "seed": seed,
            "stationary": bool(stationary),
            "offset": float(offset),
            "stuck_sweeps": int(stuck),
            "sampler": "Gibbs: coefficients, covariances, mixture indicators, "
            "volatilities, hyperparameters (indicators drawn immediately "
            "before the volatilities)",
            "notes": notes,
        },
    )
    diag = res.diagnostics()
    worst = float(diag["ess"].min())
    if worst < 100.0:
        text = (
            f"tvp_var_sv: effective sample size {worst:.0f} for "
            f"'{diag['ess'].idxmin()}'; the chain mixes slowly. Increase "
            "draws (and thin to save memory)."
        )
        notes.append(text)
        warnings.warn(text, ConvergenceWarning, stacklevel=2)
    return res
