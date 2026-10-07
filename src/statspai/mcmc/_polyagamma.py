"""
Polya-Gamma draws, for Gibbs sampling of logistic models.

If ``omega ~ PG(1, psi)`` then, for a binary outcome with log odds
``psi``, the likelihood contribution ``exp(psi)^y / (1 + exp(psi))`` is
proportional to ``exp(kappa psi) E[exp(-omega psi^2 / 2)]`` with
``kappa = y - 1/2`` (Polson, Scott and Windle 2013). Conditional on the
``omega`` the coefficients of a logit have a normal full conditional, so
any prior that is conditionally normal (ridge, horseshoe) gives a Gibbs
sampler with no tuning.

The sampler is the exact accept-reject algorithm of that paper for
``PG(1, z)``: an exponentially tilted Jacobi distribution, proposed from a
truncated inverse Gaussian below ``t = 0.64`` and an exponential above,
and accepted by the alternating-series method. It is written from the
paper and vectorised over the observations.
"""

from __future__ import annotations

import numpy as np
from scipy import special

_T = 0.64
_PI = float(np.pi)


def _a_coef(n: int, x: np.ndarray) -> np.ndarray:
    """Terms of the alternating series of the Jacobi density."""
    k = n + 0.5
    small = x <= _T
    out = np.empty_like(x)
    xs = x[small]
    out[small] = _PI * k * (2.0 / (_PI * xs)) ** 1.5 * np.exp(-2.0 * k * k / xs)
    xl = x[~small]
    out[~small] = _PI * k * np.exp(-0.5 * k * k * _PI * _PI * xl)
    return out


def _inverse_gaussian_cdf_at_t(z: np.ndarray) -> np.ndarray:
    """P(X <= t) for X inverse Gaussian with mean 1 / z and shape 1."""
    root = 1.0 / np.sqrt(_T)
    a = special.ndtr(root * (_T * z - 1.0))
    b = np.exp(2.0 * z + special.log_ndtr(-root * (_T * z + 1.0)))
    return np.asarray(a + b)


def _truncated_inverse_gaussian(rng: np.random.Generator, z: np.ndarray) -> np.ndarray:
    """Inverse Gaussian(1 / z, 1) restricted to (0, t)."""
    out = np.empty_like(z)
    todo = np.ones(z.size, dtype=bool)
    with np.errstate(divide="ignore"):
        mu = np.where(z > 0, 1.0 / np.maximum(z, 1e-300), np.inf)
    big = mu > _T
    while todo.any():
        # mean beyond the truncation point: propose from the tail of the
        # limiting Levy distribution, accept with exp(-z^2 x / 2)
        idx = np.flatnonzero(todo & big)
        if idx.size:
            m = idx.size
            x = np.empty(m)
            need = np.ones(m, dtype=bool)
            while need.any():
                e1 = rng.exponential(size=int(need.sum()))
                e2 = rng.exponential(size=e1.size)
                ok = e1 * e1 <= 2.0 * e2 / _T
                pos = np.flatnonzero(need)[ok]
                x[pos] = _T / (1.0 + _T * e1[ok]) ** 2
                need[pos] = False
            accept = rng.random(m) < np.exp(-0.5 * z[idx] ** 2 * x)
            out[idx[accept]] = x[accept]
            todo[idx[accept]] = False
        # mean inside: ordinary inverse Gaussian draw, kept if below t
        idx = np.flatnonzero(todo & ~big)
        if idx.size:
            m_ = mu[idx]
            y = rng.standard_normal(idx.size) ** 2
            first = (
                m_
                + 0.5 * m_ * m_ * y
                - 0.5 * m_ * np.sqrt(4.0 * m_ * y + (m_ * y) ** 2)
            )
            flip = rng.random(idx.size) > m_ / (m_ + first)
            draw = np.where(flip, m_ * m_ / first, first)
            accept = draw <= _T
            out[idx[accept]] = draw[accept]
            todo[idx[accept]] = False
    return out


def rpolyagamma(rng: np.random.Generator, psi: np.ndarray) -> np.ndarray:
    """One draw of ``PG(1, psi_i)`` for each element of ``psi``."""
    psi = np.asarray(psi, dtype=float)
    z = 0.5 * np.abs(psi).reshape(-1)
    n = z.size
    out = np.empty(n)
    todo = np.arange(n)
    big_k = 0.125 * _PI * _PI + 0.5 * z * z
    p = 0.5 * _PI / big_k * np.exp(-big_k * _T)
    q = 2.0 * np.exp(-z) * _inverse_gaussian_cdf_at_t(z)
    share = p / (p + q)
    while todo.size:
        zt = z[todo]
        from_tail = rng.random(todo.size) < share[todo]
        x = np.empty(todo.size)
        if from_tail.any():
            x[from_tail] = (
                _T + rng.exponential(size=int(from_tail.sum())) / big_k[todo[from_tail]]
            )
        if (~from_tail).any():
            x[~from_tail] = _truncated_inverse_gaussian(rng, zt[~from_tail])
        s = _a_coef(0, x)
        y = rng.random(todo.size) * s
        undecided = np.ones(todo.size, dtype=bool)
        accepted = np.zeros(todo.size, dtype=bool)
        term = 0
        while undecided.any():
            term += 1
            live = np.flatnonzero(undecided)
            a = _a_coef(term, x[live])
            if term % 2 == 1:
                s[live] -= a
                hit = y[live] <= s[live]
                accepted[live[hit]] = True
                undecided[live[hit]] = False
            else:
                s[live] += a
                miss = y[live] > s[live]
                undecided[live[miss]] = False
        out[todo[accepted]] = 0.25 * x[accepted]
        todo = todo[~accepted]
    return out.reshape(psi.shape)


def polyagamma_mean(psi: np.ndarray) -> np.ndarray:
    """``E[PG(1, psi)] = tanh(psi / 2) / (2 psi)``, ``1 / 4`` at zero."""
    psi = np.asarray(psi, dtype=float)
    safe = np.where(np.abs(psi) < 1e-8, 1.0, psi)
    return np.asarray(
        np.where(np.abs(psi) < 1e-8, 0.25, np.tanh(0.5 * safe) / (2.0 * safe))
    )


__all__ = ["rpolyagamma", "polyagamma_mean"]
