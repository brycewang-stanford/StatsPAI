"""
Simulated method of moments (SMM).

Chooses ``theta`` so that moments computed on data simulated from the
model match the same moments computed on the observed data:

    g(theta) = m_hat - m_tilde(theta),
    m_tilde(theta) = (1/S) sum_s m_tilde_s(theta),
    theta_hat = argmin g(theta)' W g(theta).

Why this is not ``sp.gmm(method=...)``
--------------------------------------
``sp.gmm`` takes a per-observation moment function ``g_i(theta)`` and
estimates the moment covariance from those rows at ``theta_hat``. In SMM
the data enter only through ``m_hat``, which does not depend on
``theta``; the model enters only through a simulator, whose output carries
simulation noise of its own. The variance therefore has two independent
parts (data and simulation), and the efficient weight is known before any
estimate exists when the data moment covariance is. Bolting that onto
``sp.gmm`` would give one argument (``moment_fn``) two incompatible
meanings, and ``sp.gmm``'s ``method`` already means the weight-updating
scheme. The two estimators share the moment-covariance estimator
(``_omega``: iid / HAC / cluster) and the safe inverse from
``general_gmm``; everything else is specific to simulation.

Variance with the simulation correction
---------------------------------------
Let ``Sigma = Var(m_hat) = Omega / N`` (``N`` observations, ``Omega`` the
covariance of one observation's moment contribution) and let each of the
``S`` simulated data sets have ``N_s`` observations, drawn independently
of the data and of each other. Under correct specification the simulated
contributions at the true ``theta`` have the same covariance ``Omega``,
so ``Var(m_tilde) = Omega / (S N_s)`` and

    Var(g(theta_0)) = c * Sigma,   c = 1 + N / (S N_s),

which is the familiar ``1 + 1/S`` when ``N_s = N`` [@mcfadden1989method;
@pakes1989simulation; @lee1991simulation for time series]. With
``D = d g / d theta'`` the sandwich is

    V = (D'WD)^-1 D'W (c Sigma) W D (D'WD)^-1,

and at the efficient weight ``W = (c Sigma)^-1`` it collapses to
``(D' (c Sigma)^-1 D)^-1``: the GMM variance for the same moments scaled
by ``c``. Hansen's J is ``g' (c Sigma)^-1 g`` and is chi2(q - k) only at
that weight. The factor requires (i) simulation draws independent of the
data, (ii) the same draws reused at every ``theta`` (common random
numbers -- otherwise the objective is not even continuous), and (iii) the
model to be correctly specified (it uses ``Omega`` for the simulated
contributions too).

References
----------
[@mcfadden1989method; @pakes1989simulation; @lee1991simulation]
[@hansen1982large]
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from .._aliases import accepts_aliases
from ..core._call_record import records_call
from ..core.results import EconometricResults
from ..exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    NumericalInstability,
)
from ._smm_core import (
    Simulator,
    as_float_array,
    check_cov,
    check_weight,
    finite_difference_jacobian,
    gauss_newton,
    parse_bounds,
)
from .general_gmm import _omega, _safe_inv

_VALID_OPTIMIZERS = ("L-BFGS-B", "BFGS", "Nelder-Mead", "Powell")
_VALID_VCOV = ("iid", "hac", "cluster")


@records_call("smm")
@accepts_aliases(vce="vcov")
def smm(
    simulate: Callable[[np.ndarray, np.random.Generator], Any],
    data_moments: Any,
    theta0: Any,
    *,
    n_sim: int = 10,
    W: Union[str, np.ndarray] = "optimal",
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    seed: Optional[int] = None,
    data_moment_cov: Optional[Any] = None,
    n_obs: Optional[int] = None,
    sim_n_obs: Optional[int] = None,
    vcov: str = "iid",
    cluster: Optional[Any] = None,
    hac_bandwidth: Optional[int] = None,
    jac_step: float = 1e-5,
    optimizer: str = "L-BFGS-B",
    maxiter: int = 2000,
    tol: float = 1e-10,
    param_names: Optional[List[str]] = None,
    moment_names: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Simulated method of moments with common random numbers.

    Minimises ``g(theta)' W g(theta)`` with ``g = m_hat - m_tilde(theta)``,
    the gap between the observed moments and the average of the same
    moments over ``n_sim`` data sets simulated from the model. The
    simulation draws are fixed once (``seed``) and reused at every
    ``theta``, so the objective is a deterministic, and for a smooth
    simulator differentiable, function of ``theta``.

    Parameters
    ----------
    simulate : callable
        ``simulate(theta, rng) -> moments``. Simulate one data set from the
        model at ``theta`` using only the numpy ``Generator`` ``rng``, and
        return either its moment vector, shape ``(q,)``, or its
        per-observation moment contributions, shape ``(N_s, q)`` (averaged
        here; ``N_s`` then enters the simulation correction). It must draw
        the same random numbers whatever ``theta`` is (transform the draws;
        do not draw a ``theta``-dependent number of them) and must not touch
        the global numpy RNG -- this is checked at ``theta0``.
    data_moments : array-like
        Shape ``(q,)``: the observed moments; then ``data_moment_cov`` gives
        their covariance, or it is simulated (see Notes). Shape ``(N, q)``:
        each observation's moment contribution; the moments are the column
        means and their covariance is estimated from the rows (``vcov``).
    theta0 : array-like
        Starting values, shape ``(k,)``.
    n_sim : int, default 10
        ``S``, the number of simulated data sets.
    W : {'optimal', 'identity'} or ndarray, default 'optimal'
        Weighting matrix (named as in :func:`sp.gmm`). ``'optimal'``:
        ``W = (c Sigma)^-1``, efficient, and the only weight
        under which Hansen's J is reported. When ``Sigma`` comes from the
        data it does not depend on ``theta`` and one step is efficient; when
        it is simulated, step one uses the identity and step two the
        efficient weight at the step-one estimate. ``'identity'`` or a
        ``q x q`` positive-definite matrix: one step, sandwich variance.
    bounds : sequence of (low, high), optional
        Box constraints; ``None`` in a pair leaves that side open.
    seed : int, optional
        Seeds the common random numbers (one child ``SeedSequence`` per
        replication). ``None`` draws fresh entropy, recorded in
        ``model_info['seed_entropy']`` so the fit can be repeated.
    data_moment_cov : array-like, optional
        ``Sigma = Var(m_hat)``, the covariance of the *moment vector* (not
        of one observation's contribution), for ``(q,)`` data moments.
    n_obs : int, optional
        ``N``, the data sample size, when ``data_moments`` has shape
        ``(q,)``. Needed when ``Sigma`` is simulated, or when ``N_s`` and
        ``N`` may differ.
    sim_n_obs : int, optional
        ``N_s`` when ``simulate`` returns a moment vector. Default: ``N``
        (correction ``1 + 1/S``); recorded in ``model_info``.
    vcov : {'iid', 'hac', 'cluster'}, default 'iid'
        Estimator of ``Omega`` from contributions (data rows, or simulated
        rows when ``Sigma`` is simulated): the same estimators as
        :func:`sp.gmm` (centred). Use ``'hac'`` for time-series moments.
    cluster : array-like, optional
        Cluster labels of the data rows (``vcov='cluster'``).
    hac_bandwidth : int, optional
        Bartlett bandwidth (``L + 1`` weights ``L`` lags) for ``'hac'``;
        default ``L = floor(4 (N/100)^(2/9))``.
    jac_step : float, default 1e-5
        Relative finite-difference step for ``d m_tilde / d theta``:
        ``h_j = jac_step * max(1, |theta_j|)``, central differences
        (one-sided at a bound). Common random numbers make the simulated
        moments a smooth function of ``theta`` only if the simulator is
        smooth; a frequency simulator (``mean(y_sim > 0)``) is a step
        function and gives a zero derivative, which is caught.
    optimizer : {'L-BFGS-B', 'BFGS', 'Nelder-Mead', 'Powell'}, default 'L-BFGS-B'
        ``scipy.optimize.minimize`` method; a Gauss-Newton finish follows
        (projected on the bounds) and certifies convergence.
    maxiter : int, default 2000
    tol : float, default 1e-10
    param_names, moment_names : list of str, optional
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params`` / ``std_errors`` (z inference, ``vcov()`` available);
        ``model_info`` records ``n_sim``, ``correction_factor`` (``c``),
        ``cov_source`` (``'supplied'`` / ``'data'`` / ``'simulated'``),
        ``weight`` (``'optimal'`` / ``'identity'`` / ``'user'``),
        ``seed_entropy``, ``jac_step`` and ``jacobian_scheme``;
        ``diagnostics`` carries ``J_stat`` / ``J_df`` / ``J_p`` (``None``
        with ``J_note`` unless ``W='optimal'``), ``converged``,
        ``optimizer_success``, ``optimizer_message``, ``n_iter``,
        ``n_simulator_calls``, ``gradient_norm``, ``at_bound`` and
        ``moment_fit`` (data, simulated, gap and the gap's standard error
        ``sqrt(c Sigma_jj)``).

    Notes
    -----
    Covariance sources, in order of precedence: ``data_moments`` of shape
    ``(N, q)`` (estimated from the data rows); ``data_moment_cov``
    (supplied); otherwise simulated at the step-one estimate (``Omega``
    from the pooled simulated contributions, or from the spread of the
    ``S`` replicate moment vectors times ``N_s``; needs ``n_obs``). The
    simulated route relies on correct specification twice -- for ``Sigma``
    and for ``c`` -- and is the only option when the data are published
    aggregates.

    Asymptotic standard errors take ``c = 1 + N / (S N_s)``. They assume
    independent simulation draws, reused across ``theta``, and a correctly
    specified model; at a parameter on a bound the normal approximation
    fails (``at_bound`` lists them and a warning is issued).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> y = rng.normal(1.0, 2.0, size=2000)
    >>> contrib = np.column_stack([y, y**2, y**3])
    >>> def simulate(theta, rng):
    ...     z = rng.standard_normal(2000)
    ...     ys = theta[0] + theta[1] * z
    ...     return np.column_stack([ys, ys**2, ys**3])
    >>> res = sp.smm(simulate, contrib, theta0=[0.0, 1.0], n_sim=10,
    ...              bounds=[(None, None), (1e-6, None)], seed=0,
    ...              param_names=["mu", "sigma"])
    >>> round(res.model_info["correction_factor"], 2)
    1.1
    >>> bool(res.diagnostics["converged"])
    True

    References
    ----------
    [@mcfadden1989method; @pakes1989simulation; @lee1991simulation]
    [@hansen1982large]
    """
    # ---- argument validation ---------------------------------------------
    if not callable(simulate):
        raise MethodIncompatibility(
            "smm: `simulate` must be callable as simulate(theta, rng).",
            recovery_hint="Wrap the model simulator in a function of (theta, rng).",
        )
    if isinstance(n_sim, bool) or not isinstance(n_sim, (int, np.integer)):
        raise MethodIncompatibility(
            f"smm: n_sim must be a positive integer, got {n_sim!r}.",
            recovery_hint="n_sim is S, the number of simulated data sets.",
        )
    S = int(n_sim)
    if S < 1:
        raise MethodIncompatibility(
            f"smm: n_sim must be >= 1, got {S}.",
            recovery_hint="Use n_sim >= 1 (10 or more is typical).",
        )
    if optimizer not in _VALID_OPTIMIZERS:
        raise MethodIncompatibility(
            f"smm: optimizer must be one of {_VALID_OPTIMIZERS}, got {optimizer!r}.",
            recovery_hint="'L-BFGS-B' for smooth simulators; 'Nelder-Mead' "
            "when the objective is rough.",
        )
    if vcov not in _VALID_VCOV:
        raise MethodIncompatibility(
            f"smm: vcov must be one of {_VALID_VCOV}, got {vcov!r}.",
            recovery_hint="'iid' for cross-sections, 'hac' for time series.",
        )
    if not (np.isfinite(jac_step) and 0 < float(jac_step) < 1):
        raise MethodIncompatibility(
            f"smm: jac_step must be in (0, 1), got {jac_step!r}.",
            recovery_hint="1e-5 suits smooth simulators.",
        )
    if bounds is not None and optimizer == "BFGS":
        raise MethodIncompatibility(
            "smm: optimizer='BFGS' does not accept bounds.",
            recovery_hint="Use 'L-BFGS-B', 'Nelder-Mead' or 'Powell' with bounds.",
        )

    theta0 = as_float_array(theta0, "theta0").ravel()
    k = theta0.size
    lower, upper = parse_bounds(bounds, k)
    if np.any(theta0 < lower) or np.any(theta0 > upper):
        raise MethodIncompatibility(
            "smm: theta0 lies outside the bounds.",
            recovery_hint="Start inside the box.",
        )

    dm = as_float_array(data_moments, "data_moments")
    N: Optional[int]
    if dm.ndim == 2:
        N = int(dm.shape[0])
        q = int(dm.shape[1])
        if n_obs is not None and int(n_obs) != N:
            raise MethodIncompatibility(
                f"smm: n_obs={n_obs} but data_moments has {N} rows.",
                recovery_hint="Omit n_obs when passing per-observation rows.",
            )
        if not np.all(np.isfinite(dm)):
            raise DataInsufficient(
                "smm: data_moments contains missing or infinite values.",
                recovery_hint="Drop or impute the incomplete rows first.",
            )
        m_hat = dm.mean(axis=0)
    elif dm.ndim == 1:
        q = int(dm.size)
        N = None if n_obs is None else int(n_obs)
        m_hat = dm
        if not np.all(np.isfinite(m_hat)):
            raise DataInsufficient(
                "smm: data_moments contains missing or infinite values.",
                recovery_hint="Every moment must be finite.",
            )
    else:
        raise MethodIncompatibility(
            f"smm: data_moments must have shape (q,) or (N, q); got {dm.shape}.",
            recovery_hint="Pass the moment vector or per-observation rows.",
        )
    if N is not None and N < 2:
        raise DataInsufficient(
            f"smm: need at least 2 observations, got {N}.",
            recovery_hint="SMM needs a sample to estimate the moment covariance.",
        )
    if q < k:
        raise IdentificationFailure(
            f"smm: {q} moment(s) for {k} parameter(s): under-identified.",
            recovery_hint="Add moments or fix some parameters.",
            diagnostics={"n_moments": q, "n_params": k},
        )
    if param_names is None:
        param_names = [f"theta_{j}" for j in range(k)]
    if moment_names is None:
        moment_names = [f"m{j}" for j in range(q)]
    if len(param_names) != k or len(moment_names) != q:
        raise MethodIncompatibility(
            "smm: param_names / moment_names have the wrong length.",
            recovery_hint=f"Give {k} parameter names and {q} moment names.",
        )

    W_user: Optional[np.ndarray] = None
    if isinstance(W, str):
        if W not in ("optimal", "identity"):
            raise MethodIncompatibility(
                f"smm: W must be 'optimal', 'identity' or a matrix; " f"got {W!r}.",
                recovery_hint="W='optimal' gives the efficient estimator "
                "and the J test.",
            )
        weight_label = W
    else:
        W_user = check_weight(as_float_array(W, "W"), q)
        weight_label = "user"

    # ---- common random numbers --------------------------------------------
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(S)
    sim = Simulator(simulate, children, q)

    first = sim.one(theta0, 0)
    again = sim.one(theta0, 0)
    if first.shape != again.shape or not np.array_equal(first, again):
        raise MethodIncompatibility(
            "smm: simulate(theta, rng) returned different values for the same "
            "theta and the same rng seed, so common random numbers cannot be "
            "held fixed and the objective is not a function of theta.",
            recovery_hint="Draw every random number from the `rng` argument "
            "(not np.random.* or a generator created inside simulate).",
        )
    if not np.all(np.isfinite(first)):
        raise NumericalInstability(
            "smm: simulate returned non-finite moments at theta0.",
            recovery_hint="Choose a theta0 at which the model can be simulated.",
        )

    # ---- correction factor --------------------------------------------------
    N_s: Optional[int] = sim.rows if first.ndim == 2 else sim_n_obs
    if first.ndim == 2 and sim_n_obs is not None and int(sim_n_obs) != sim.rows:
        raise MethodIncompatibility(
            f"smm: sim_n_obs={sim_n_obs} but simulate returned {sim.rows} rows.",
            recovery_hint="Omit sim_n_obs when simulate returns rows.",
        )
    sim_size_assumed = False
    if N_s is None:
        if N is None:
            # N_s = N is the stated default; neither is needed for c.
            ratio = 1.0
        else:
            N_s = N
            ratio = 1.0
        sim_size_assumed = True
    else:
        N_s = int(N_s)
        if N is None:
            raise MethodIncompatibility(
                "smm: the simulated sample size is known but the data sample "
                "size is not, so the simulation correction 1 + N/(S N_s) "
                "cannot be formed.",
                recovery_hint="Pass n_obs=N (the observed sample size).",
            )
        ratio = N / N_s
    c = 1.0 + ratio / S

    # ---- data-moment covariance --------------------------------------------
    cov_source: str
    Sigma: Optional[np.ndarray] = None
    hac_lags: Optional[int] = None

    def _hac_bw(n_rows: int) -> Optional[int]:
        nonlocal hac_lags
        if vcov != "hac":
            return None
        if hac_bandwidth is None:
            hac_lags = int(np.floor(4 * (n_rows / 100) ** (2 / 9)))
            return hac_lags + 1
        if int(hac_bandwidth) < 1:
            raise MethodIncompatibility(
                f"smm: hac_bandwidth must be >= 1, got {hac_bandwidth!r}.",
                recovery_hint="hac_bandwidth = L + 1 weights L lags.",
            )
        hac_lags = int(hac_bandwidth) - 1
        return int(hac_bandwidth)

    cl = None
    if vcov == "cluster":
        if cluster is None or dm.ndim != 2:
            raise MethodIncompatibility(
                "smm: vcov='cluster' needs per-observation data_moments and a "
                "`cluster` array.",
                recovery_hint="Pass (N, q) data_moments and cluster labels.",
            )
        cl = np.asarray(cluster).ravel()
        if cl.size != N or pd.isna(pd.Series(cl)).any():
            raise MethodIncompatibility(
                "smm: `cluster` must give a label for every data row.",
                recovery_hint="One non-missing cluster label per row.",
            )

    if dm.ndim == 2:
        assert N is not None
        Omega = _omega(
            dm,
            vcov=vcov,
            center=True,
            cluster=cl,
            hac_bandwidth=_hac_bw(N),
        )
        Sigma = Omega / N
        cov_source = "data"
    elif data_moment_cov is not None:
        Sigma = as_float_array(data_moment_cov, "data_moment_cov")
        cov_source = "supplied"
    else:
        if N is None:
            raise MethodIncompatibility(
                "smm: no data covariance: pass (N, q) data_moments, "
                "data_moment_cov, or n_obs so it can be simulated.",
                recovery_hint="Give n_obs=N to estimate the covariance by "
                "simulation at the step-one estimate.",
            )
        if vcov == "cluster":
            raise MethodIncompatibility(
                "smm: a simulated covariance cannot be clustered.",
                recovery_hint="Use vcov='iid' or 'hac'.",
            )
        cov_source = "simulated"
    if Sigma is not None:
        check_cov(Sigma, q, cov_source)
        Sigma = (Sigma + Sigma.T) / 2.0

    def _simulated_sigma(theta: np.ndarray) -> np.ndarray:
        reps = sim.replicates(theta)
        assert N is not None and N_s is not None
        if reps[0].ndim == 2:
            # Omega from each replication's own rows (centred within the
            # replication; HAC within a simulated path), averaged over S.
            Om: np.ndarray = np.zeros((q, q))
            for r in reps:
                Om += _omega(
                    r,
                    vcov=vcov,
                    center=True,
                    cluster=None,
                    hac_bandwidth=_hac_bw(r.shape[0]),
                )
            Om /= len(reps)
        else:
            if S <= q:
                raise DataInsufficient(
                    f"smm: simulating the moment covariance from {S} "
                    f"replicate moment vectors needs n_sim > q = {q}.",
                    recovery_hint="Raise n_sim, return per-observation rows "
                    "from simulate, or pass data_moment_cov.",
                )
            V = np.cov(np.array(reps), rowvar=False, ddof=1)
            Om = np.asarray(np.atleast_2d(V), dtype=float) * N_s
        Sig = Om / N
        check_cov(Sig, q, "simulated")
        return np.asarray((Sig + Sig.T) / 2.0)

    # ---- objective and optimisation ----------------------------------------
    m_fn = sim.mean

    def gap(theta: np.ndarray) -> np.ndarray:
        return np.asarray(m_hat - m_fn(theta))

    def objective(theta: np.ndarray, W: np.ndarray) -> float:
        g = gap(theta)
        val = float(g @ W @ g)
        return val if np.isfinite(val) else 1e300

    def jac(theta: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        Dm, schemes = finite_difference_jacobian(
            m_fn, theta, float(jac_step), lower, upper
        )
        return -Dm, schemes  # D = d g / d theta'

    use_bounds = optimizer in ("L-BFGS-B", "Nelder-Mead", "Powell")
    sp_bounds = (
        list(
            zip(
                [None if np.isinf(v) else v for v in lower],
                [None if np.isinf(v) else v for v in upper],
            )
        )
        if (bounds is not None and use_bounds)
        else None
    )

    opt_options = {
        "L-BFGS-B": {"maxiter": maxiter, "ftol": tol * 1e-2, "gtol": tol},
        "BFGS": {"maxiter": maxiter, "gtol": tol},
        "Nelder-Mead": {"maxiter": maxiter, "xatol": tol, "fatol": tol * 1e-2},
        "Powell": {"maxiter": maxiter, "xtol": tol, "ftol": tol * 1e-2},
    }[optimizer]

    def solve(W: np.ndarray, start: np.ndarray) -> Dict[str, Any]:
        res = minimize(
            lambda t: objective(t, W),
            start,
            method=optimizer,
            bounds=sp_bounds,
            options=opt_options,
        )
        theta, gn_steps, gn_ok = gauss_newton(
            np.asarray(res.x, float),
            W,
            gap=gap,
            jac=jac,
            objective=objective,
            m_hat=m_hat,
            lower=lower,
            upper=upper,
        )
        return {
            "theta": theta,
            "optimizer_success": bool(res.success),
            "optimizer_message": str(res.message),
            "n_iter": int(getattr(res, "nit", 0) or 0) + gn_steps,
            "gn_converged": gn_ok,
        }

    if W_user is not None:
        W_est = W_user
    elif weight_label == "identity":
        W_est = np.eye(q)
    elif Sigma is not None:
        W_est = _safe_inv(c * Sigma)
    else:
        W_est = np.eye(q)  # step one of the simulated-covariance route

    fit = solve(W_est, theta0)
    steps = 1
    step1_theta: Optional[np.ndarray] = None
    if Sigma is None:
        step1_theta = fit["theta"].copy()
        Sigma = _simulated_sigma(step1_theta)
        if weight_label == "optimal":
            W_est = _safe_inv(c * Sigma)
            fit2 = solve(W_est, step1_theta)
            fit2["n_iter"] += fit["n_iter"]
            fit = fit2
            steps = 2

    theta_hat = fit["theta"]
    g_hat = gap(theta_hat)
    if not np.all(np.isfinite(g_hat)):
        raise NumericalInstability(
            "smm: the simulated moments are not finite at the estimate.",
            recovery_hint="Add bounds that keep the simulator well defined.",
        )

    # ---- variance -----------------------------------------------------------
    D, schemes = jac(theta_hat)
    zero_cols = [
        param_names[j]
        for j in range(k)
        if np.max(np.abs(D[:, j])) <= 1e-12 * max(1.0, np.max(np.abs(m_hat)))
    ]
    if zero_cols or np.linalg.matrix_rank(D) < k:
        raise NumericalInstability(
            "smm: the derivative of the simulated moments is (numerically) "
            f"rank deficient at the estimate (flat in {zero_cols or 'a combination'}). "
            "Either the moments do not identify the parameters, or the "
            "simulator is a step function of theta (e.g. simulated "
            "frequencies), whose finite-difference derivative is zero.",
            recovery_hint="Smooth the simulator (e.g. a logit-kernel smoothed "
            "frequency), raise jac_step, or add moments that move with "
            "the flat parameter.",
            diagnostics={"jacobian": D.tolist(), "jac_step": float(jac_step)},
        )
    cSigma = c * Sigma
    DtW = D.T @ W_est
    bread = _safe_inv(DtW @ D)
    V = bread @ DtW @ cSigma @ DtW.T @ bread
    V = (V + V.T) / 2.0
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))

    # ---- J test -------------------------------------------------------------
    J_df = q - k
    J_stat: Optional[float]
    J_p: Optional[float]
    J_note: Optional[str] = None
    if weight_label == "optimal":
        J_stat = float(g_hat @ _safe_inv(cSigma) @ g_hat)
        J_p = float(stats.chi2.sf(J_stat, J_df)) if J_df > 0 else None
        if J_df == 0:
            J_note = "Exactly identified: J is zero by construction, no test."
    else:
        J_stat = J_p = None
        J_note = (
            "Hansen's J is reported only at the efficient weight "
            "(W='optimal'); at another weight it is not chi2(q - k)."
        )

    # ---- convergence diagnostics ------------------------------------------
    grad = 2.0 * D.T @ W_est @ g_hat
    at_bound = [
        param_names[j]
        for j in range(k)
        if np.isclose(theta_hat[j], lower[j], rtol=0, atol=1e-10 * (1 + abs(lower[j])))
        or np.isclose(theta_hat[j], upper[j], rtol=0, atol=1e-10 * (1 + abs(upper[j])))
    ]
    interior = [j for j in range(k) if param_names[j] not in at_bound]
    grad_norm = float(np.max(np.abs(grad[interior]))) if interior else 0.0
    obj = float(g_hat @ W_est @ g_hat)
    converged = bool(fit["gn_converged"] or fit["optimizer_success"])
    if not converged:
        warnings.warn(
            ConvergenceWarning(
                "smm: neither the optimiser nor the Gauss-Newton finish "
                f"certified convergence ({fit['optimizer_message']}); the "
                "estimate may not be a minimum.",
                recovery_hint="Try another theta0 or optimizer, raise maxiter, "
                "or check the simulator is smooth in theta.",
                diagnostics={"gradient_norm": grad_norm, "objective": obj},
            ),
            stacklevel=2,
        )
    if at_bound:
        warnings.warn(
            ConvergenceWarning(
                f"smm: {at_bound} at a bound; their standard errors use the "
                "interior asymptotics, which do not hold on the boundary.",
                recovery_hint="Treat those intervals as unreliable or "
                "reparametrise.",
                diagnostics={"at_bound": at_bound},
            ),
            stacklevel=2,
        )
    if sim.n_nonfinite:
        warnings.warn(
            ConvergenceWarning(
                f"smm: the simulator returned non-finite moments at "
                f"{sim.n_nonfinite} trial value(s) of theta during the search.",
                recovery_hint="Bounds that keep the model well defined help "
                "the optimiser.",
                diagnostics={"n_nonfinite": sim.n_nonfinite},
            ),
            stacklevel=2,
        )

    m_sim = m_fn(theta_hat)
    moment_fit = pd.DataFrame(
        {
            "data": m_hat,
            "simulated": m_sim,
            "gap": m_hat - m_sim,
            "gap_se": np.sqrt(np.diag(cSigma)),
        },
        index=moment_names,
    )

    names = list(param_names)
    data_info: Dict[str, Any] = {
        "n_obs": N,
        "inference": "z",
        "var_cov": V,
    }
    if N is not None:
        data_info["df_resid"] = max(N - k, 1)
    return EconometricResults(
        params=pd.Series(theta_hat, index=names),
        std_errors=pd.Series(se, index=names),
        model_info={
            "alpha": alpha,
            "model_type": "SMM",
            "se_type": f"SMM sandwich x simulation correction c = {c:.6g}",
            "n_sim": S,
            "correction_factor": c,
            "sim_n_obs": N_s,
            "sim_size_assumed_equal_to_data": sim_size_assumed,
            "cov_source": cov_source,
            "vcov": vcov,
            "hac_lags": hac_lags,
            "weight": weight_label,
            "weight_steps": steps,
            "step1_params": None if step1_theta is None else step1_theta.tolist(),
            "seed_entropy": ss.entropy,
            "jac_step": float(jac_step),
            "jacobian_scheme": dict(zip(names, schemes)),
            "optimizer": optimizer,
            "n_moments": q,
            "n_params": k,
            "overidentified": q > k,
        },
        data_info=data_info,
        diagnostics={
            "J_stat": J_stat,
            "J_df": J_df,
            "J_p": J_p,
            "J_note": J_note,
            "smm_objective": obj,
            "converged": converged,
            "optimizer_success": fit["optimizer_success"],
            "optimizer_message": fit["optimizer_message"],
            "n_iter": fit["n_iter"],
            "n_simulator_calls": sim.n_calls,
            "gradient_norm": grad_norm,
            "at_bound": at_bound,
            "moment_fit": moment_fit,
            "jacobian": pd.DataFrame(D, index=moment_names, columns=names),
            "moment_cov": pd.DataFrame(
                cSigma, index=moment_names, columns=moment_names
            ),
        },
    )
