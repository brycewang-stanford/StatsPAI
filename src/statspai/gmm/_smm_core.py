"""Helpers for :func:`statspai.gmm.smm.smm` (simulator wrapper, Jacobian,
input checks and the Gauss-Newton finish)."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..exceptions import MethodIncompatibility, NumericalInstability


def as_float_array(x: Any, name: str) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"smm: `{name}` must be numeric.",
            recovery_hint=f"Pass `{name}` as a float array.",
        ) from exc
    return arr


class Simulator:
    """Evaluate the averaged simulated moments with common random numbers.

    Each replication ``s`` owns a fixed child ``SeedSequence``; every call
    builds a fresh ``Generator`` from it, so the draws are identical at
    every ``theta`` as long as the user's simulator consumes the generator
    the same way each time.
    """

    def __init__(
        self,
        simulate: Callable[[np.ndarray, np.random.Generator], Any],
        children: Sequence[np.random.SeedSequence],
        q: int,
    ) -> None:
        self.simulate = simulate
        self.children = list(children)
        self.q = q
        self.n_calls = 0
        self.n_nonfinite = 0
        self.rows: Optional[int] = None  # simulated N_s when 2-D output

    def one(self, theta: np.ndarray, s: int) -> np.ndarray:
        out = as_float_array(
            self.simulate(
                np.array(theta, dtype=float),
                np.random.default_rng(self.children[s]),
            ),
            "simulate(...)",
        )
        self.n_calls += 1
        if out.ndim == 1:
            if out.size != self.q:
                raise MethodIncompatibility(
                    f"smm: simulate returned {out.size} moments, the data "
                    f"have {self.q}.",
                    recovery_hint="Return the same moments, in the same "
                    "order, as `data_moments`.",
                )
            return out
        if out.ndim == 2 and out.shape[1] == self.q:
            if self.rows is None:
                self.rows = int(out.shape[0])
            elif out.shape[0] != self.rows:
                raise MethodIncompatibility(
                    "smm: simulate returned a different number of rows across "
                    f"calls ({self.rows} then {out.shape[0]}).",
                    recovery_hint="Simulate data sets of a fixed size.",
                )
            return out
        raise MethodIncompatibility(
            f"smm: simulate must return shape ({self.q},) or (N_s, {self.q}); "
            f"got {out.shape}.",
            recovery_hint="Return the moment vector of one simulated data "
            "set, or its per-observation contributions.",
        )

    def mean(self, theta: np.ndarray) -> np.ndarray:
        reps = [self.one(theta, s) for s in range(len(self.children))]
        vec = np.array([r.mean(axis=0) if r.ndim == 2 else r for r in reps])
        m = vec.mean(axis=0)
        if not np.all(np.isfinite(m)):
            self.n_nonfinite += 1
        return np.asarray(m)

    def replicates(self, theta: np.ndarray) -> List[np.ndarray]:
        return [self.one(theta, s) for s in range(len(self.children))]


def finite_difference_jacobian(
    m_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    rel_step: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Finite-difference ``d m_tilde / d theta'`` (q x k).

    Step ``h_j = rel_step * max(1, |theta_j|)``; central where both points
    are inside the bounds, one-sided (forward or backward) otherwise. The
    scheme used for each parameter is returned.
    """
    k = theta.size
    cols = []
    schemes = []
    for j in range(k):
        h = rel_step * max(1.0, abs(float(theta[j])))
        e = np.zeros(k)
        e[j] = h
        up_ok = theta[j] + h <= upper[j]
        dn_ok = theta[j] - h >= lower[j]
        if up_ok and dn_ok:
            cols.append((m_fn(theta + e) - m_fn(theta - e)) / (2.0 * h))
            schemes.append("central")
        elif up_ok:
            cols.append((m_fn(theta + e) - m_fn(theta)) / h)
            schemes.append("forward")
        elif dn_ok:
            cols.append((m_fn(theta) - m_fn(theta - e)) / h)
            schemes.append("backward")
        else:
            raise MethodIncompatibility(
                f"smm: the bounds of parameter {j} are narrower than the "
                f"finite-difference step {h:.3g}.",
                recovery_hint="Widen the bounds or lower `jac_step`.",
            )
    return np.column_stack(cols), schemes


def parse_bounds(
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]], k: int
) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.full(k, -np.inf)
    upper = np.full(k, np.inf)
    if bounds is None:
        return lower, upper
    if len(bounds) != k:
        raise MethodIncompatibility(
            f"smm: `bounds` has {len(bounds)} pairs for {k} parameters.",
            recovery_hint="Give one (low, high) pair per parameter; None = open.",
        )
    for j, (lo, hi) in enumerate(bounds):
        lower[j] = -np.inf if lo is None else float(lo)
        upper[j] = np.inf if hi is None else float(hi)
        if lower[j] >= upper[j]:
            raise MethodIncompatibility(
                f"smm: bounds for parameter {j} are empty ({lo}, {hi}).",
                recovery_hint="Each pair must satisfy low < high.",
            )
    return lower, upper


def check_weight(W: np.ndarray, q: int) -> np.ndarray:
    if W.shape != (q, q):
        raise MethodIncompatibility(
            f"smm: weight matrix has shape {W.shape}; expected ({q}, {q}).",
            recovery_hint="Pass a q x q symmetric positive-definite matrix.",
        )
    if not np.allclose(W, W.T, rtol=1e-10, atol=1e-12 * np.abs(W).max()):
        raise MethodIncompatibility(
            "smm: weight matrix is not symmetric.",
            recovery_hint="Pass (W + W.T) / 2.",
        )
    ev = np.linalg.eigvalsh(W)
    if ev.min() <= 0:
        raise MethodIncompatibility(
            "smm: weight matrix is not positive definite "
            f"(smallest eigenvalue {ev.min():.3g}).",
            recovery_hint="Use a positive-definite weight, or weight='optimal'.",
        )
    return W


def check_cov(Sigma: np.ndarray, q: int, source: str) -> None:
    if Sigma.shape != (q, q) or not np.all(np.isfinite(Sigma)):
        raise MethodIncompatibility(
            f"smm: the moment covariance ({source}) must be a finite "
            f"({q}, {q}) matrix; got shape {Sigma.shape}.",
            recovery_hint="Check `data_moment_cov` / the data contributions.",
        )
    ev = np.linalg.eigvalsh((Sigma + Sigma.T) / 2.0)
    if ev.min() <= 1e-12 * max(ev.max(), 1e-300):
        raise NumericalInstability(
            f"smm: the moment covariance ({source}) is singular: some "
            "moments are (nearly) linear combinations of others.",
            recovery_hint="Drop redundant moments (e.g. a constant moment, or "
            "a share that sums to one with the others).",
            diagnostics={"eigenvalues": ev.tolist()},
        )


def gauss_newton(
    theta: np.ndarray,
    W: np.ndarray,
    *,
    gap: Callable[[np.ndarray], np.ndarray],
    jac: Callable[[np.ndarray], Tuple[np.ndarray, List[str]]],
    objective: Callable[[np.ndarray, np.ndarray], float],
    m_hat: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, int, bool]:
    """Projected Gauss-Newton ``theta <- theta - (D'WD)^-1 D'W g``.

    Parameters sitting on a bound with the step pointing outwards are
    held fixed (active set). Convergence is certified by the
    Gauss-Newton decrement ``delta = (D'Wg)' (D'WD)^-1 (D'Wg)``, the
    predicted fall in the objective: ``delta <= 1e-14 * f``, or below the
    objective's rounding floor (moments perturbed by ``1e-12`` of their
    size) for a perfect fit. When no step decreases the objective any more
    (rounding), ``delta <= 1e-8 * f`` still certifies. Unlike a step-size
    rule this does not depend on the units of ``theta``. Returns
    ``(theta, steps, converged)``.
    """
    theta = np.clip(theta, lower, upper)
    f0 = objective(theta, W)
    e = 1e-12 * np.maximum(np.abs(m_hat), 1e-300)
    f_floor = float(e @ np.abs(W) @ e)
    for it in range(0, 51):
        g = gap(theta)
        try:
            D, _ = jac(theta)
        except MethodIncompatibility:
            return theta, it, False
        DtW = D.T @ W
        grad = DtW @ g  # half the gradient of the objective
        k = theta.size
        free = np.ones(k, dtype=bool)
        for _ in range(k):
            if not free.any():
                # every parameter on a bound, each pushed outwards: a KKT point
                return theta, it, True
            Hf = (DtW @ D)[np.ix_(free, free)]
            if np.linalg.matrix_rank(Hf) < free.sum():
                return theta, it, False
            step = np.zeros(k)
            step[free] = -np.linalg.solve(Hf, grad[free])
            out = free & (
                ((theta <= lower) & (step < 0)) | ((theta >= upper) & (step > 0))
            )
            if not out.any():
                break
            free &= ~out
        delta = float(-grad @ step)
        if delta <= max(1e-14 * f0, f_floor):
            return theta, it, True
        if it == 50:
            break
        t = 1.0
        accepted = False
        for _ in range(40):
            cand = np.clip(theta + t * step, lower, upper)
            f1 = objective(cand, W)
            if f1 < f0:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            # no decrease along a descent direction: stationary to the
            # precision the objective can be evaluated at, or rough
            return theta, it, bool(delta <= max(1e-8 * f0, 100.0 * f_floor))
        theta, f0 = cand, f1
    return theta, 50, False
