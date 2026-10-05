"""
Shared primitives for the MCMC samplers in :mod:`statspai.mcmc`.

Random draws that every sampler needs (truncated normal, multivariate
normal from a precision matrix, inverse gamma, inverse Wishart), prior
parsing, and the random-walk Metropolis kernel. Kept in one place so the
samplers do not each carry their own copy.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import linalg, special

from ..exceptions import MethodIncompatibility

ArrayLike = Union[float, Sequence[float], np.ndarray]


# --------------------------------------------------------------------------
# Random draws
# --------------------------------------------------------------------------


def rtruncnorm(
    rng: np.random.Generator,
    mean: np.ndarray,
    sd: Union[float, np.ndarray],
    lower: Union[float, np.ndarray] = -np.inf,
    upper: Union[float, np.ndarray] = np.inf,
) -> np.ndarray:
    """Draw from N(mean, sd^2) truncated to (lower, upper), vectorised.

    Inverse-CDF sampling carried out on the log scale and always in the
    lower tail (an interval that lies above the mean is reflected), so it
    stays accurate when the interval is many standard deviations away.
    """
    mean = np.asarray(mean, dtype=float)
    sd_arr = np.broadcast_to(np.asarray(sd, dtype=float), mean.shape)
    a = (np.broadcast_to(np.asarray(lower, dtype=float), mean.shape) - mean) / sd_arr
    b = (np.broadcast_to(np.asarray(upper, dtype=float), mean.shape) - mean) / sd_arr
    # reflect intervals in the upper half so that both ends are <= 0 or the
    # interval straddles zero with the bulk in reach
    flip = a > 0
    lo = np.where(flip, -b, a)
    hi = np.where(flip, -a, b)
    log_plo = special.log_ndtr(lo)
    log_phi = special.log_ndtr(hi)
    u = rng.random(mean.shape)
    # log( Phi(lo) + u (Phi(hi) - Phi(lo)) )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_width = log_phi + np.log1p(-np.exp(np.minimum(log_plo - log_phi, 0.0)))
        log_p = np.logaddexp(log_plo, np.log(u) + log_width)
    z = special.ndtri_exp(log_p)
    # numerical guard: stay inside the interval
    z = np.minimum(np.maximum(z, lo), hi)
    z = np.where(flip, -z, z)
    return np.asarray(mean + sd_arr * z)


def rmvnorm_prec(
    rng: np.random.Generator, rhs: np.ndarray, prec: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw from N(P^{-1} rhs, P^{-1}) given the precision matrix P.

    Returns the draw and the mean ``P^{-1} rhs``.
    """
    chol = linalg.cholesky(prec, lower=True, check_finite=False)
    mean = linalg.cho_solve((chol, True), rhs, check_finite=False)
    z = rng.standard_normal(rhs.shape[0])
    draw = mean + linalg.solve_triangular(chol.T, z, lower=False, check_finite=False)
    return draw, mean


def rinvgamma(rng: np.random.Generator, shape: float, rate: float) -> float:
    """One draw from the inverse gamma with density
    ``rate^shape / G(shape) x^{-shape-1} exp(-rate / x)``."""
    return float(rate / rng.gamma(shape))


def rwishart(rng: np.random.Generator, df: float, scale: np.ndarray) -> np.ndarray:
    """One draw from the Wishart W(df, scale), mean ``df * scale``.

    Bartlett decomposition; ``df`` may be any real number above
    ``dim - 1``.
    """
    p = scale.shape[0]
    chol = linalg.cholesky(scale, lower=True, check_finite=False)
    a = np.zeros((p, p))
    for i in range(p):
        a[i, i] = np.sqrt(rng.chisquare(df - i))
        if i:
            a[i, :i] = rng.standard_normal(i)
    la = chol @ a
    return np.asarray(la @ la.T)


def rinvwishart(rng: np.random.Generator, df: float, scale: np.ndarray) -> np.ndarray:
    """One draw from the inverse Wishart IW(df, scale), with density
    proportional to ``|S|^{-(df + p + 1)/2} exp(-tr(scale S^{-1}) / 2)``
    and mean ``scale / (df - p - 1)``."""
    inv_scale = linalg.inv(scale, check_finite=False)
    inv_scale = 0.5 * (inv_scale + inv_scale.T)
    w = rwishart(rng, df, inv_scale)
    out = linalg.inv(w, check_finite=False)
    return np.asarray(0.5 * (out + out.T))


# --------------------------------------------------------------------------
# Priors
# --------------------------------------------------------------------------


def normal_prior(
    k: int, mean: ArrayLike, var: ArrayLike, names: Optional[Sequence[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a N(mean, var) prior for a ``k``-vector.

    ``mean`` is a scalar or a ``k``-vector; ``var`` is a scalar, a
    ``k``-vector of variances or a ``k x k`` covariance matrix. Returns
    the mean, the covariance and the precision.
    """
    m = np.asarray(mean, dtype=float)
    if m.ndim == 0:
        m = np.full(k, float(m))
    if m.shape != (k,):
        raise MethodIncompatibility(
            f"prior_mean must be a scalar or have one entry per coefficient "
            f"({k}); got shape {m.shape}."
            + (f" Coefficients: {list(names)}." if names is not None else "")
        )
    v = np.asarray(var, dtype=float)
    if v.ndim == 0:
        cov = np.eye(k) * float(v)
    elif v.ndim == 1:
        if v.shape != (k,):
            raise MethodIncompatibility(
                f"prior_var as a vector must have one variance per "
                f"coefficient ({k}); got {v.shape[0]}."
            )
        cov = np.diag(v)
    elif v.shape == (k, k):
        cov = 0.5 * (v + v.T)
    else:
        raise MethodIncompatibility(
            f"prior_var must be a scalar, a length-{k} vector or a {k} x {k} "
            f"matrix; got shape {v.shape}."
        )
    try:
        chol = linalg.cholesky(cov, lower=True)
    except linalg.LinAlgError as exc:
        raise MethodIncompatibility(
            "prior_var must be positive definite (a proper prior)."
        ) from exc
    prec = linalg.cho_solve((chol, True), np.eye(k))
    prec = 0.5 * (prec + prec.T)
    return m, cov, prec


def log_mvnorm(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Log density of N(mean, cov) at ``x``."""
    k = mean.shape[0]
    chol = linalg.cholesky(cov, lower=True, check_finite=False)
    z = linalg.solve_triangular(chol, x - mean, lower=True, check_finite=False)
    return float(
        -0.5 * k * np.log(2.0 * np.pi) - np.log(np.diag(chol)).sum() - 0.5 * z @ z
    )


def log_invgamma(x: float, shape: float, rate: float) -> float:
    """Log density of the inverse gamma (shape, rate) at ``x``."""
    return float(
        shape * np.log(rate)
        - special.gammaln(shape)
        - (shape + 1.0) * np.log(x)
        - rate / x
    )


def log_gamma(x: float, shape: float, rate: float) -> float:
    """Log density of the gamma (shape, rate) at ``x``."""
    return float(
        shape * np.log(rate)
        - special.gammaln(shape)
        + (shape - 1.0) * np.log(x)
        - rate * x
    )


# --------------------------------------------------------------------------
# Mode finding and random-walk Metropolis
# --------------------------------------------------------------------------


def numerical_hessian(
    f: Callable[[np.ndarray], float], x: np.ndarray, rel_step: float = 1e-4
) -> np.ndarray:
    """Central-difference Hessian of a scalar function."""
    k = x.size
    h = rel_step * np.maximum(np.abs(x), 1.0)
    hess = np.empty((k, k))
    f0 = f(x)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        hess[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(k)
            ej[j] = h[j]
            val = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4.0 * h[i] * h[j])
            hess[i, j] = hess[j, i] = val
    return hess


def find_mode(
    neg_log_post: Callable[[np.ndarray], float],
    start: np.ndarray,
    what: str = "posterior",
) -> Tuple[np.ndarray, np.ndarray]:
    """Maximise a log density and return the mode and the inverse of the
    negative Hessian there (the Laplace covariance)."""
    from scipy import optimize

    best = optimize.minimize(neg_log_post, start, method="BFGS")
    # polish: BFGS can stop early on flat ridges
    polished = optimize.minimize(
        neg_log_post,
        best.x,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 400 * start.size},
    )
    if polished.fun < best.fun:
        best = optimize.minimize(neg_log_post, polished.x, method="BFGS")
        if polished.fun < best.fun:
            best = polished
    mode = np.asarray(best.x, dtype=float)
    if not np.isfinite(best.fun):
        raise MethodIncompatibility(
            f"Could not find the mode of the {what}: the log density is not "
            "finite at the optimiser's solution. Check the data for "
            "separation, collinearity or outcome values outside the "
            "model's support."
        )
    hess = numerical_hessian(neg_log_post, mode)
    hess = 0.5 * (hess + hess.T)
    try:
        chol = linalg.cholesky(hess, lower=True)
        cov = linalg.cho_solve((chol, True), np.eye(mode.size))
    except linalg.LinAlgError as exc:
        raise MethodIncompatibility(
            f"The {what} is not locally concave at its mode (the Hessian is "
            "not positive definite). The usual causes are perfectly "
            "collinear regressors or, in binary and ordered models, "
            "separation. A tighter prior (smaller prior_var) restores a "
            "proper posterior."
        ) from exc
    return mode, 0.5 * (cov + cov.T)


def random_walk_metropolis(
    rng: np.random.Generator,
    log_post: Callable[[np.ndarray], float],
    start: np.ndarray,
    prop_cov: np.ndarray,
    n_iter: int,
) -> Tuple[np.ndarray, float]:
    """Random-walk Metropolis with a fixed Gaussian proposal.

    Returns all ``n_iter`` states and the acceptance rate.
    """
    k = start.size
    chol = linalg.cholesky(prop_cov, lower=True)
    out = np.empty((n_iter, k))
    cur = np.array(start, dtype=float)
    lp = log_post(cur)
    accepted = 0
    steps = rng.standard_normal((n_iter, k)) @ chol.T
    log_u = np.log(rng.random(n_iter))
    for it in range(n_iter):
        cand = cur + steps[it]
        lp_c = log_post(cand)
        if log_u[it] < lp_c - lp:
            cur, lp = cand, lp_c
            accepted += 1
        out[it] = cur
    return out, accepted / n_iter


def check_mcmc_args(draws: int, burnin: int, thin: int, chains: int = 1) -> None:
    for nm, val, lo in (
        ("draws", draws, 1),
        ("burnin", burnin, 0),
        ("thin", thin, 1),
        ("chains", chains, 1),
    ):
        if not isinstance(val, (int, np.integer)) or val < lo:
            raise MethodIncompatibility(
                f"{nm} must be an integer >= {lo}; got {val!r}."
            )


def spawn_rngs(seed: Any, n: int) -> list:
    """Independent generators, one per chain, from one seed."""
    if isinstance(seed, np.random.Generator):
        seq: Any = seed.bit_generator.seed_seq
        return [np.random.default_rng(s) for s in seq.spawn(n)]
    ss = np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in ss.spawn(n)]


def design_for(design_info: Any, names: Sequence[str], data: Any) -> np.ndarray:
    """Rebuild a design matrix for new data.

    Uses the patsy design when the formula needed one; a formula of plain
    numeric columns is parsed without patsy, and its design is the named
    columns with an optional leading intercept.
    """
    if design_info is not None:
        from patsy import build_design_matrices

        return np.asarray(build_design_matrices([design_info], data)[0], dtype=float)
    cols = [c for c in names if c != "Intercept"]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"The new data lack the regressors {missing}.")
    pieces = {c: data[c].to_numpy(dtype=float) for c in cols}
    n = len(data)
    out = [np.ones(n) if c == "Intercept" else pieces[c] for c in names]
    return np.column_stack(out) if out else np.zeros((n, 0))
