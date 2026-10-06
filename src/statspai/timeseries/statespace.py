"""Linear state space models: Kalman filter, smoother and maximum likelihood.

The model, in the notation of Neusser (2016, ch. 17) and Hamilton (1994,
ch. 13), is

    state:        X_t = F_t X_{t-1} + V_t,        V_t ~ (0, Q_t)
    observation:  Y_t = A_t + G_t X_t + W_t,      W_t ~ (0, R_t)

for ``t = 1, ..., T`` with ``X_0 ~ (x0, P0)``, and ``V``, ``W`` and ``X_0``
mutually uncorrelated. ``sp.kalman_filter`` runs the recursions for system
matrices the user writes down; ``sp.statespace`` estimates the parameters
those matrices depend on by maximum likelihood.

Timing. Entry ``t`` of a time-varying ``F`` or ``Q`` is the transition
*into* date ``t``. The initial moments belong to ``X_0``, the state one
step before the first observation, so the first predicted state is
``F_1 x0`` with covariance ``F_1 P0 F_1' + Q_1``. Software that initialises
the first *predicted* state instead (statsmodels, R ``KFAS``) takes these
two as its ``a1`` and ``P1``; under ``init='stationary'`` the two coincide.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from . import _statespace_core as core
from ._statespace_results import KalmanResult, StateSpaceResult, inference, package

__all__ = ["kalman_filter", "statespace", "KalmanResult", "StateSpaceResult"]

_SYSTEM = ("A", "G", "F", "Q", "R")


def _observations(
    y: Any, data: Optional[pd.DataFrame]
) -> Tuple[np.ndarray, List[str], pd.Index]:
    """Observations as a ``(T, n)`` float array with names and an index."""
    if data is not None:
        cols = [y] if isinstance(y, str) else list(y)
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise MethodIncompatibility(
                f"Columns {missing} are not in data.",
                recovery_hint="Check the column names passed as y.",
            )
        frame = data[cols]
    elif isinstance(y, pd.DataFrame):
        frame = y
    elif isinstance(y, pd.Series):
        frame = y.to_frame(name=y.name if y.name is not None else "y")
    else:
        arr = np.array(y, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise MethodIncompatibility("y must be one- or two-dimensional.")
        names = (
            ["y"] if arr.shape[1] == 1 else [f"y{i + 1}" for i in range(arr.shape[1])]
        )
        frame = pd.DataFrame(arr, columns=names)
    values = frame.to_numpy(dtype=float)
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise DataInsufficient("y is empty.")
    if np.any(np.isinf(values)):
        raise MethodIncompatibility("y contains infinite values.")
    return np.ascontiguousarray(values), [str(c) for c in frame.columns], frame.index


def _system(mats: Dict[str, Any], T: int, n: int) -> Tuple[Dict[str, np.ndarray], int]:
    """Validate system matrices and give each a leading time axis."""
    absent = [k for k in ("F", "G", "Q", "R") if mats.get(k) is None]
    if absent:
        raise MethodIncompatibility(
            f"System matrices {absent} are missing.",
            recovery_hint="F, G, Q and R are required; A defaults to zero.",
        )
    Fa = np.array(mats["F"], dtype=float)
    m = 1 if Fa.ndim == 0 else int(Fa.shape[-1])
    out = {
        "F": core.as_stack(mats["F"], (m, m), T, "F"),
        "G": core.as_stack(mats["G"], (n, m), T, "G"),
        "Q": core.as_stack(mats["Q"], (m, m), T, "Q", symmetric=True),
        "R": core.as_stack(mats["R"], (n, n), T, "R", symmetric=True),
    }
    A = mats.get("A")
    out["A"] = core.as_stack(np.zeros(n) if A is None else A, (n,), T, "A")
    return out, m


def _run(
    yv: np.ndarray,
    mats: Dict[str, Any],
    init: str,
    kappa: float,
    smooth: bool,
    compiled: bool,
) -> Dict[str, Any]:
    """Filter (and smooth) once; raises ``LinAlgError`` on a singular step."""
    T, n = yv.shape
    sysm, m = _system(mats, T, n)
    x0, P0, rule = core.initial_state(
        sysm["F"], sysm["Q"], mats.get("x0"), mats.get("P0"), init, kappa
    )
    mask = np.isfinite(yv)
    y0 = np.where(mask, yv, 0.0)
    kf, ks = core.kernels(compiled)
    out = kf(y0, mask, sysm["A"], sysm["G"], sysm["F"], sysm["Q"], sysm["R"], x0, P0)
    res: Dict[str, Any] = dict(
        zip(("xp", "Pp", "xf", "Pf", "v", "S", "Sinv", "K", "e", "ll"), out)
    )
    res.update(sys=sysm, mask=mask, x0=x0, P0=P0, rule=rule, m=m)
    if smooth:
        res["xs"], res["Ps"] = ks(
            res["xp"],
            res["Pp"],
            res["v"],
            res["Sinv"],
            res["K"],
            sysm["G"],
            mask,
            sysm["F"],
        )
    return res


def _singular(exc: Exception) -> MethodIncompatibility:
    return MethodIncompatibility(
        "A prediction-error covariance G P G' + R is not positive definite.",
        recovery_hint=(
            "An observable is an exact linear function of the others or of "
            "past data. Drop it, or give it measurement noise in R."
        ),
        diagnostics={"error": str(exc)},
    )


def kalman_filter(
    y: Any,
    *,
    F: Any,
    G: Any,
    Q: Any,
    R: Any,
    A: Optional[Any] = None,
    x0: Optional[Any] = None,
    P0: Optional[Any] = None,
    init: str = "auto",
    kappa: float = 1e7,
    smooth: bool = True,
    burn: int = 0,
    data: Optional[pd.DataFrame] = None,
    state_names: Optional[Sequence[str]] = None,
) -> KalmanResult:
    """Kalman filter and smoother for a linear state space model.

    The model is ``X_t = F_t X_{t-1} + V_t`` with ``Var(V_t) = Q_t`` and
    ``Y_t = A_t + G_t X_t + W_t`` with ``Var(W_t) = R_t``. All system
    matrices are taken as known; :func:`statespace` estimates them.

    Parameters
    ----------
    y : array-like, Series, DataFrame, or column name(s) with ``data``
        Observations, ``T`` rows and ``n`` columns, in time order. NaN is a
        missing value: a missing element drops its row of the observation
        equation at that date, and a date with nothing observed is a pure
        prediction step.
    F : array-like, ``(m, m)`` or ``(T, m, m)``
        Transition matrix. With a time axis, entry ``t`` moves the state
        from date ``t - 1`` to date ``t``.
    G : array-like, ``(n, m)`` or ``(T, n, m)``
        Loadings of the observables on the state. With one observable,
        ``(m,)`` and ``(T, m)`` are accepted, so a regression with
        time-varying coefficients passes its regressor matrix.
    Q : array-like, ``(m, m)`` or ``(T, m, m)``
        Covariance of the state disturbance. May be singular.
    R : array-like, ``(n, n)`` or ``(T, n, n)``
        Covariance of the measurement error. May be singular or zero.
    A : array-like, ``(n,)`` or ``(T, n)``, optional
        Intercept of the observation equation. Default zero.
    x0, P0 : array-like, optional
        Mean and covariance of the state one step before the first
        observation. ``x0`` defaults to zero; ``P0`` follows ``init``.
    init : {'auto', 'stationary', 'diffuse'}, default 'auto'
        Rule for ``P0`` when it is not given. ``'stationary'`` solves
        ``P0 = F P0 F' + Q`` and needs constant ``F`` and ``Q`` with every
        eigenvalue of ``F`` inside the unit circle. ``'diffuse'`` sets
        ``P0 = kappa I``. ``'auto'`` is stationary when ``F`` is stable
        and diffuse otherwise.
    kappa : float, default 1e7
        Variance of the diffuse initial state.
    smooth : bool, default True
        Also run the fixed-interval smoother.
    burn : int, default 0
        Leading dates left out of ``loglik`` (the filter still uses them).
    data : DataFrame, optional
        Source of the columns named by ``y``.
    state_names : sequence of str, optional
        Names of the states. Default ``x1, x2, ...``.

    Returns
    -------
    KalmanResult
        Predicted, filtered and smoothed states with their covariances,
        prediction errors, the log-likelihood, ``.states()``,
        ``.forecast()``, ``.summary()`` and ``.plot()``.

    Raises
    ------
    MethodIncompatibility
        System matrices of the wrong shape, an unstable ``F`` under
        ``init='stationary'``, or a prediction-error covariance that is not
        positive definite.
    DataInsufficient
        ``y`` is empty.

    Notes
    -----
    The log-likelihood is the Gaussian prediction-error decomposition,
    summed over every observed element of ``y`` from date ``burn + 1``:
    date ``t`` contributes ``-0.5 [n_t log(2 pi) + log det S_t + v_t'
    S_t^{-1} v_t]`` with ``n_t`` the number of elements observed and
    ``S_t`` the covariance of their prediction errors. Missing elements
    contribute nothing.

    The diffuse initial state is the large-variance approximation, not the
    exact diffuse filter: the first prediction errors have variance of
    order ``kappa`` and enter the likelihood with it, and state moments at
    the first dates lose about ``log10(kappa)`` digits. For a model whose
    transient is short this does not affect later dates. Use ``burn`` to
    leave the first dates out of the likelihood.

    The covariance update uses the Joseph form, and the smoother the
    backward recursion that inverts only prediction-error covariances, so
    singular ``Q`` and ``R`` need no special treatment. It gives the same
    moments as the Rauch-Tung-Striebel recursion ``J_t = P_{t|t} F'
    P_{t+1|t}^{-1}``, ``P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t})
    J_t'``.

    Examples
    --------
    A local level model:

    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> level = np.cumsum(0.5 * rng.normal(size=100))
    >>> y = level + rng.normal(size=100)
    >>> out = sp.kalman_filter(y, F=1.0, G=1.0, Q=0.25, R=1.0)
    >>> out.smoothed_state.shape
    (100, 1)

    An AR(2) in companion form; the likelihood is the exact Gaussian one:

    >>> F = np.array([[0.5, 0.3], [1.0, 0.0]])
    >>> Q = np.diag([1.0, 0.0])
    >>> out = sp.kalman_filter(y, F=F, G=[1.0, 0.0], Q=Q, R=0.0)
    >>> out.init
    'stationary'
    >>> out.forecast(2)["obs"].shape
    (2, 1)

    References
    ----------
    [@kalman1960new],
    [@neusser2016time]
    """
    yv, names, index = _observations(y, data)
    mats = {"F": F, "G": G, "Q": Q, "R": R, "A": A, "x0": x0, "P0": P0}
    try:
        res = _run(yv, mats, init, kappa, bool(smooth), False)
    except np.linalg.LinAlgError as exc:
        raise _singular(exc) from exc
    return package(res, names, index, burn, state_names, kappa)


def statespace(
    y: Any,
    build: Callable[[np.ndarray], Dict[str, Any]],
    start: Any,
    *,
    param_names: Optional[Sequence[str]] = None,
    transform: Optional[Callable[[np.ndarray], Any]] = None,
    data: Optional[pd.DataFrame] = None,
    init: str = "auto",
    kappa: float = 1e7,
    burn: int = 0,
    method: str = "bfgs",
    vce: str = "hessian",
    maxiter: int = 1000,
    tol: float = 1e-5,
    alpha: float = 0.05,
    state_names: Optional[Sequence[str]] = None,
    engine: str = "auto",
) -> StateSpaceResult:
    """Maximum-likelihood estimation of a user-written state space model.

    ``build`` maps a parameter vector to the system matrices of
    ``X_t = F_t X_{t-1} + V_t``, ``Y_t = A_t + G_t X_t + W_t``; the
    Gaussian likelihood from the Kalman filter is maximised over it.

    Parameters
    ----------
    y : array-like, Series, DataFrame, or column name(s) with ``data``
        Observations; NaN marks a missing value (see
        :func:`kalman_filter`).
    build : callable
        ``build(theta)`` returns a dict with keys ``F``, ``G``, ``Q``,
        ``R`` and optionally ``A``, ``x0``, ``P0``, each as
        :func:`kalman_filter` takes it. Restrictions are the caller's
        business: write a variance as ``exp(theta[i])`` to keep it
        positive.
    start : array-like
        Starting values.
    param_names : sequence of str, optional
        Names of the parameters. Default ``theta1, theta2, ...``.
    transform : callable, optional
        ``transform(theta)`` returns quantities of interest (an array, or
        a dict or Series to name them), reported with delta-method
        standard errors in ``.transformed``, for instance
        ``lambda th: {"sigma2": np.exp(th[0])}``.
    data : DataFrame, optional
        Source of the columns named by ``y``.
    init, kappa, burn
        As in :func:`kalman_filter`, applied when ``build`` returns no
        ``P0``. ``init='auto'`` is settled at ``start`` and then kept: if
        ``F`` is stable there, the stationary initial state is used
        throughout and parameter values with an unstable ``F`` have
        likelihood minus infinity.
    method : {'bfgs', 'l-bfgs-b', 'nelder-mead'}, default 'bfgs'
        First optimiser. Gradients are central differences.
    vce : {'hessian', 'opg', 'robust'}, default 'hessian'
        Covariance of the estimates: inverse of the numerical Hessian of
        the negative log-likelihood, inverse of the outer product of the
        per-date scores, or the sandwich of the two.
    maxiter : int, default 1000
        Iteration limit of each optimiser run.
    tol : float, default 1e-5
        Bound on the scaled gradient ``max_i |g_i| max(|theta_i|, 1) /
        max(|loglik|, 1)`` for the fit to count as converged.
    alpha : float, default 0.05
        Level of the confidence intervals.
    state_names : sequence of str, optional
        Names of the states.
    engine : {'auto', 'numpy', 'numba'}, default 'auto'
        How the filter runs inside the optimiser. ``'numba'`` compiles the
        recursion on first use (several seconds, cached afterwards) and
        falls back to NumPy when numba is not installed; ``'auto'`` asks
        for it when ``T`` times the number of parameters is at least 400.
        The two agree to rounding error.

    Returns
    -------
    StateSpaceResult
        Estimates with standard errors, ``loglik``, ``aic``, ``bic``,
        ``converged``, the filter and smoother output at the optimum in
        ``.filter``, ``.summary()``, ``.forecast()`` and ``.plot()``.

    Raises
    ------
    MethodIncompatibility
        ``build`` does not return the system matrices, they have the wrong
        shape, or the likelihood is not finite at ``start``.
    DataInsufficient
        Fewer observations than parameters.

    Notes
    -----
    The search runs in rescaled parameters, each divided by the inverse
    square root of the curvature of the likelihood in its own direction at
    ``start``, which makes the first steps of BFGS sensible when parameters
    differ by orders of magnitude. Estimates, gradient and Hessian are
    reported in the original parameters.

    After the first optimiser, the scaled gradient is checked. If it is
    not below ``tol``, a Nelder-Mead search is started from that point and
    its result polished by BFGS; the better point is kept. Every run is
    listed in ``model_info['steps']``. A ``ConvergenceWarning`` is issued
    when the gradient is still not small or the Hessian is not positive
    definite; in the latter case standard errors are NaN under
    ``vce='hessian'`` and ``'robust'``. A likelihood with several local
    maxima is not detected by any of this: start from more than one point.

    The Hessian is a central-difference approximation with step
    ``1e-4 max(|theta_i|, 1)``. Standard errors refer to ``theta`` as
    ``build`` receives it; use ``transform`` for functions of it.

    A parameter value at which a prediction-error covariance is not
    positive definite is given likelihood minus infinity.

    Examples
    --------
    A regression whose slope follows a random walk:

    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(2)
    >>> T = 120
    >>> x = rng.normal(size=T)
    >>> beta = 1.0 + np.cumsum(0.1 * rng.normal(size=T))
    >>> y = beta * x + 0.5 * rng.normal(size=T)
    >>> def build(th):
    ...     return {"F": 1.0, "G": x[:, None], "Q": np.exp(th[0]),
    ...             "R": np.exp(th[1])}
    >>> fit = sp.statespace(y, build, [0.0, 0.0], init="diffuse", burn=1,
    ...                     param_names=["log_q", "log_r"],
    ...                     transform=lambda th: {"q": np.exp(th[0])})
    >>> bool(fit.converged)
    True
    >>> fit.filter.smoothed_state.shape
    (120, 1)

    References
    ----------
    [@kalman1960new],
    [@neusser2016time]
    """
    yv, names, index = _observations(y, data)
    theta0 = np.array(start, dtype=float).reshape(-1)
    k = theta0.size
    if k == 0 or not np.all(np.isfinite(theta0)):
        raise MethodIncompatibility("start must be a non-empty finite vector.")
    pnames = (
        [f"theta{i + 1}" for i in range(k)]
        if param_names is None
        else list(param_names)
    )
    if len(pnames) != k:
        raise MethodIncompatibility(f"param_names must have {k} entries.")
    methods = {"bfgs": "BFGS", "l-bfgs-b": "L-BFGS-B", "nelder-mead": "Nelder-Mead"}
    if method not in methods:
        raise MethodIncompatibility(f"method={method!r} is not one of {list(methods)}.")
    if vce not in ("hessian", "opg", "robust"):
        raise MethodIncompatibility(f"vce={vce!r} is not 'hessian', 'opg' or 'robust'.")
    if engine not in ("auto", "numpy", "numba"):
        raise MethodIncompatibility(
            f"engine={engine!r} is not 'auto', 'numpy' or 'numba'."
        )
    T = yv.shape[0]
    compiled = engine == "numba" or (engine == "auto" and T * k >= 400)
    if not 0 <= int(burn) < T:
        raise MethodIncompatibility(f"burn must be between 0 and {T - 1}.")
    burn = int(burn)
    n_obs = int(np.isfinite(yv[burn:]).sum())
    if n_obs <= k:
        raise DataInsufficient(
            f"{n_obs} observations cannot identify {k} parameters.",
            diagnostics={"n_obs": n_obs, "n_params": k},
        )

    def mats_at(theta: np.ndarray) -> Dict[str, Any]:
        out = build(np.array(theta, dtype=float))
        if not isinstance(out, dict):
            raise MethodIncompatibility(
                "build(theta) must return a dict of system matrices.",
                recovery_hint="Keys: F, G, Q, R and optionally A, x0, P0.",
            )
        extra = sorted(set(out) - set(_SYSTEM) - {"x0", "P0"})
        if extra:
            raise MethodIncompatibility(f"build returned unknown keys {extra}.")
        return out

    # init='auto' is settled once, at the starting values: switching between
    # a stationary and a diffuse initial state as F crosses the unit circle
    # would put a jump in the likelihood
    rule = init
    if init == "auto":
        try:
            first = _run(yv, mats_at(theta0), init, kappa, False, False)
        except np.linalg.LinAlgError as exc:
            raise _singular(exc) from exc
        rule = first["rule"] if first["rule"] != "user" else init

    def contributions(theta: np.ndarray) -> np.ndarray:
        try:
            res = _run(yv, mats_at(theta), rule, kappa, False, compiled)
        except (np.linalg.LinAlgError, core.NotStationary):
            return np.full(T - burn, -np.inf)
        return np.asarray(res["ll"][burn:])

    def nll(theta: np.ndarray) -> float:
        val = -float(contributions(theta).sum())
        return val if np.isfinite(val) else 1e12

    f0 = nll(theta0)
    if f0 >= 1e12:
        raise MethodIncompatibility(
            "The likelihood is not finite at the starting values.",
            recovery_hint="Choose start so that every covariance is valid.",
        )

    opt = core.maximise(nll, contributions, theta0, methods[method], vce, maxiter, tol)
    best, cov, notes = opt["theta"], opt["cov"], opt["notes"]
    if notes:
        warnings.warn(
            "statespace: " + " ".join(notes), ConvergenceWarning, stacklevel=2
        )

    try:
        res = _run(yv, mats_at(best), rule, kappa, True, False)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by nll
        raise _singular(exc) from exc
    filt = package(res, names, index, burn, state_names, kappa)
    table = inference(best, cov, pnames, alpha)
    transformed = None
    if transform is not None:
        raw = transform(best)
        if isinstance(raw, dict):
            raw = pd.Series(raw, dtype=float)
        tnames = (
            [str(i) for i in raw.index]
            if isinstance(raw, pd.Series)
            else [f"g{i + 1}" for i in range(np.size(raw))]
        )

        def as_vector(theta: np.ndarray) -> np.ndarray:
            out = transform(theta)
            if isinstance(out, dict):
                out = list(out.values())
            return np.asarray(out, dtype=float).reshape(-1)

        J = core.num_jacobian(as_vector, best)
        transformed = inference(as_vector(best), J @ cov @ J.T, tnames, alpha)
    loglik = -float(opt["fun"])
    n_dates = filt.n_dates
    info = {key: opt[key] for key in core.REPORTED}
    info.update(vce=vce, start_loglik=-f0, notes=notes)
    return StateSpaceResult(
        params=pd.Series(best, index=pnames, name="estimate"),
        se=table["se"].copy(),
        table=table,
        cov=pd.DataFrame(cov, index=pnames, columns=pnames),
        transformed=transformed,
        loglik=loglik,
        aic=float(-2.0 * loglik + 2.0 * k),
        bic=float(-2.0 * loglik + k * np.log(n_dates)),
        n_obs=n_obs,
        n_dates=n_dates,
        converged=bool(opt["converged"]),
        filter=filt,
        alpha=alpha,
        model_info=info,
    )
