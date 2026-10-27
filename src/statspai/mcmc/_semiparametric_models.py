"""Baseline models of ``sp.semiparametric_bayes``.

Each model is a :class:`_Spec`: the baseline log density, a simulator, the
baseline and efficient scores of the parameters of interest, the prior and
the map from the sampler's unconstrained scale to the reported one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg, special

from ..exceptions import MethodIncompatibility

_SCORE_HINT = (
    "Each callable takes (theta, y, X): theta the parameter vector, y a 1-d "
    "outcome array and X the matching rows of the design matrix. loglik "
    "returns one value per row, score and efficient_score an (n, k) array, "
    "simulate(theta, X, rng) one draw per row."
)


@dataclass
class _Spec:
    names_u: List[str]  # sampler-scale parameter names
    report_names: List[str]  # natural-scale names
    interest: List[str]  # names of gamma in report_names
    loglik: Callable[..., np.ndarray]
    simulate: Callable[..., np.ndarray]
    ldelta: Callable[..., np.ndarray]  # baseline minus efficient score
    efficient_score: Callable[..., np.ndarray]
    log_prior: Callable[[np.ndarray], float]
    to_report: Callable[[np.ndarray], np.ndarray]
    start: np.ndarray
    k: int
    interest_pos_u: List[int]  # positions of gamma in theta


_LOG_2PI = float(np.log(2.0 * np.pi))


def _norm_logpdf(x: Any, mean: float, sd: float) -> float:
    """Sum of N(mean, sd^2) log densities (scipy's is slow in the loop)."""
    z = (np.asarray(x, dtype=float) - mean) / sd
    return float(-0.5 * np.sum(z * z) - z.size * (np.log(sd) + 0.5 * _LOG_2PI))


def _log_k0(z: np.ndarray) -> np.ndarray:
    # log(2 / (1 + exp(-2z)))
    return np.log(2.0) - np.logaddexp(0.0, -2.0 * z)


def _log_one_minus_half_k0(z: np.ndarray) -> np.ndarray:
    # log(1 - k0(z) / 2) = log(1 / (1 + exp(2z)))
    return -np.logaddexp(0.0, 2.0 * z)


# --------------------------------------------------------------------------
# Built-in baseline models
# --------------------------------------------------------------------------


def _orthogonalise(
    X: np.ndarray, names: List[str], interest: Optional[Sequence[Any]]
) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Replace the columns of interest by their residuals on the others.

    The linear index is unchanged, ``x_g b_g + W b_w = xt b_g + W b_w*``
    with ``b_w* = b_w + Pi b_g``, so the coefficients of interest keep
    their meaning; the nuisance coefficients are mapped back after
    sampling. This is how the paper treats nuisance regressors (its
    incumbency application).
    """
    p = X.shape[1]
    if interest is None:
        return X, list(range(p)), np.zeros((0, 0))
    pos = []
    for item in interest:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            j = int(item)
        elif item in names:
            j = names.index(item)
        else:
            raise MethodIncompatibility(
                f"interest entry {item!r} is not a coefficient; the "
                f"coefficients are {names}.",
                recovery_hint="Pass names from the design, e.g. interest=['x'].",
            )
        if not 0 <= j < p or j in pos:
            raise MethodIncompatibility(
                f"interest entry {item!r} is out of range or repeated."
            )
        pos.append(j)
    pos = sorted(pos)
    other = [j for j in range(p) if j not in pos]
    if not other:
        return X, pos, np.zeros((0, 0))
    W = X[:, other]
    Pi = np.linalg.lstsq(W, X[:, pos], rcond=None)[0]
    Xo = X.copy()
    Xo[:, pos] = X[:, pos] - W @ Pi
    return Xo, pos, Pi


def _back_transform(
    B: np.ndarray, pos: List[int], Pi: np.ndarray, p: int
) -> np.ndarray:
    """Coefficients of the orthogonalised index -> original coefficients."""
    if Pi.size == 0:
        return B
    other = [j for j in range(p) if j not in pos]
    out = B.copy()
    out[:, other] = B[:, other] - B[:, pos] @ Pi.T
    return out


def _student_t_spec(
    y: np.ndarray, X: np.ndarray, names: List[str], pos: List[int], Pi, prior_sd
) -> _Spec:
    p = X.shape[1]

    def unpack(u):
        return u[:p], np.exp(u[p]), 2.0 + np.exp(u[p + 1])

    def loglik(u, yy, XX):
        b, s, nu = unpack(u)
        e = (yy - XX @ b) / s
        return (
            special.gammaln((nu + 1) / 2)
            - special.gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi)
            - (nu + 1) / 2 * np.log1p(e * e / nu)
            - np.log(s)
        )

    def simulate(u, XX, rng):
        b, s, nu = unpack(u)
        return XX @ b + s * rng.standard_t(nu, size=XX.shape[0])

    def efficient_score(u, yy, XX):
        b, s, nu = unpack(u)
        e = yy - XX @ b
        # var(e) = s^2 nu / (nu - 2) under the baseline (eq. 11 of the paper)
        return XX[:, pos] * (e * (nu - 2) / (nu * s * s))[:, None]

    def ldelta(u, yy, XX):
        b, s, nu = unpack(u)
        e = yy - XX @ b
        w = (nu + 1) / (nu * s * s + e * e) - (nu - 2) / (nu * s * s)
        return XX[:, pos] * (e * w)[:, None]

    def log_prior(u):
        return _norm_logpdf(u[: p + 1], 0.0, prior_sd) + _norm_logpdf(
            u[p + 1], float(np.log(2.0)), 1.0
        )

    def to_report(U):
        B = _back_transform(U[:, :p], pos, Pi, p)
        return np.column_stack([B, np.exp(U[:, p]), 2.0 + np.exp(U[:, p + 1])])

    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b0
    start = np.concatenate([b0, [np.log(max(np.std(r), 1e-8)), np.log(2.0)]])
    return _Spec(
        names_u=names + ["log_sigma", "log_nu_minus_2"],
        report_names=names + ["sigma", "nu"],
        interest=[names[j] for j in pos],
        loglik=loglik,
        simulate=simulate,
        ldelta=ldelta,
        efficient_score=efficient_score,
        log_prior=log_prior,
        to_report=to_report,
        start=start,
        k=len(pos),
        interest_pos_u=list(pos),
    )


def _gamma_qmle(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Solve sum_i x_i (y_i exp(-x_i'b) - 1) = 0 by Newton's method."""
    b = np.linalg.lstsq(X, np.log(y), rcond=None)[0]
    for _ in range(200):
        r = y * np.exp(-X @ b)
        g = X.T @ (r - 1.0)
        H = (X * r[:, None]).T @ X
        step = linalg.solve(H, g, assume_a="pos")
        b = b + step
        if np.max(np.abs(step)) < 1e-12 * (1.0 + np.max(np.abs(b))):
            break
    return b


def _weibull_c(alpha):
    return (
        np.exp(special.gammaln(1 + 2 / alpha) - 2 * special.gammaln(1 + 1 / alpha)) - 1
    )


def _weibull_spec(
    y: np.ndarray, X: np.ndarray, names: List[str], pos: List[int], Pi, prior_sd
) -> _Spec:
    p = X.shape[1]

    def log_lam(u, XX):
        alpha = np.exp(u[p])
        return XX @ u[:p] - special.gammaln(1 + 1 / alpha), alpha

    def loglik(u, yy, XX):
        ll, alpha = log_lam(u, XX)
        d = np.log(yy) - ll
        return u[p] - ll + (alpha - 1) * d - np.exp(alpha * d)

    def simulate(u, XX, rng):
        ll, alpha = log_lam(u, XX)
        return np.exp(ll) * rng.weibull(alpha, size=XX.shape[0])

    def efficient_score(u, yy, XX):
        alpha = np.exp(u[p])
        r = yy * np.exp(-(XX @ u[:p])) - 1.0
        return XX[:, pos] * (r / _weibull_c(alpha))[:, None]

    def ldelta(u, yy, XX):
        ll, alpha = log_lam(u, XX)
        z = np.exp(alpha * (np.log(yy) - ll))
        base = alpha * (z - 1.0)
        eff = (yy * np.exp(-(XX @ u[:p])) - 1.0) / _weibull_c(alpha)
        return XX[:, pos] * (base - eff)[:, None]

    def log_prior(u):
        return _norm_logpdf(u, 0.0, prior_sd)

    def to_report(U):
        B = _back_transform(U[:, :p], pos, Pi, p)
        return np.column_stack([B, np.exp(U[:, p])])

    start = np.concatenate([_gamma_qmle(y, X), [0.0]])
    return _Spec(
        names_u=names + ["log_alpha"],
        report_names=names + ["alpha"],
        interest=[names[j] for j in pos],
        loglik=loglik,
        simulate=simulate,
        ldelta=ldelta,
        efficient_score=efficient_score,
        log_prior=log_prior,
        to_report=to_report,
        start=start,
        k=len(pos),
        interest_pos_u=list(pos),
    )


def _custom_spec(
    y,
    X,
    loglik,
    score,
    efficient_score,
    simulate,
    log_prior,
    theta_start,
    param_names,
    interest,
) -> _Spec:
    missing = [
        nm
        for nm, f in (
            ("loglik", loglik),
            ("score", score),
            ("efficient_score", efficient_score),
            ("simulate", simulate),
            ("log_prior", log_prior),
        )
        if not callable(f)
    ]
    if missing or theta_start is None:
        raise MethodIncompatibility(
            "model='custom' needs the callables "
            + ", ".join(missing or ["theta_start"])
            + ("" if theta_start is not None or not missing else " and theta_start")
            + ".",
            recovery_hint=_SCORE_HINT,
        )
    t0 = np.asarray(theta_start, dtype=float).reshape(-1)
    m = t0.size
    n = y.shape[0]
    names = (
        list(param_names)
        if param_names is not None
        else [f"theta{j}" for j in range(m)]
    )
    if len(names) != m:
        raise MethodIncompatibility(
            f"param_names has {len(names)} entries for {m} parameters."
        )

    def as_scores(f, label):
        def g(u, yy, XX):
            s = np.asarray(f(u, yy, XX), dtype=float)
            if s.ndim == 1:
                s = s[:, None]
            if s.ndim != 2 or s.shape[0] != yy.shape[0]:
                raise MethodIncompatibility(
                    f"{label} returned shape {s.shape}; expected "
                    f"({yy.shape[0]}, k).",
                    recovery_hint=_SCORE_HINT,
                )
            return s

        return g

    sc = as_scores(score, "score")
    ef = as_scores(efficient_score, "efficient_score")
    s0, e0 = sc(t0, y, X), ef(t0, y, X)
    if s0.shape != e0.shape:
        raise MethodIncompatibility(
            f"score has shape {s0.shape} but efficient_score {e0.shape}; both "
            "are the scores of the same parameters of interest.",
            recovery_hint=_SCORE_HINT,
        )
    k = s0.shape[1]
    l0 = np.asarray(loglik(t0, y, X), dtype=float)
    if l0.shape != (n,) or not np.all(np.isfinite(l0)):
        raise MethodIncompatibility(
            f"loglik must return {n} finite values at theta_start; got shape "
            f"{l0.shape}.",
            recovery_hint=_SCORE_HINT,
        )
    y0 = np.asarray(simulate(t0, X, np.random.default_rng(0)), dtype=float)
    if y0.shape != (n,):
        raise MethodIncompatibility(
            f"simulate must return one draw per row ({n}); got shape {y0.shape}.",
            recovery_hint=_SCORE_HINT,
        )
    if not np.isfinite(float(log_prior(t0))):
        raise MethodIncompatibility("log_prior is not finite at theta_start.")
    if interest is None:
        ipos = list(range(k))
    else:
        ipos = [names.index(i) if i in names else int(i) for i in interest]
    if len(ipos) != k:
        raise MethodIncompatibility(
            f"The scores have {k} columns but interest names {len(ipos)} "
            "parameters; the scores are those of the parameters of interest.",
            recovery_hint=_SCORE_HINT,
        )

    def ldelta(u, yy, XX):
        return sc(u, yy, XX) - ef(u, yy, XX)

    return _Spec(
        names_u=names,
        report_names=names,
        interest=[names[j] for j in ipos],
        loglik=lambda u, yy, XX: np.asarray(loglik(u, yy, XX), dtype=float),
        simulate=lambda u, XX, rng: np.asarray(simulate(u, XX, rng), dtype=float),
        ldelta=ldelta,
        efficient_score=ef,
        log_prior=lambda u: float(log_prior(u)),
        to_report=lambda U: U,
        start=t0,
        k=k,
        interest_pos_u=ipos,
    )
