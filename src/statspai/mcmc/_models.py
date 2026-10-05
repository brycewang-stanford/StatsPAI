"""
The single-equation models behind :func:`statspai.bayes_regress`.

Each model is a small class with the same four pieces:

* ``names``            -- the reported parameters, in order;
* ``sample``           -- its sampler (Gibbs, data augmentation or
  random-walk Metropolis), returning draws of the reported parameters;
* ``to_u`` / ``log_kernel`` -- the reported parameters mapped to an
  unconstrained vector ``u`` and the log of likelihood x prior x Jacobian
  there. Mode finding, the Laplace and Gelfand-Dey marginal likelihoods
  and the tests' quadrature checks all work on this one function, so a
  sampler and its target can be compared without sharing code;
* ``linear_index``     -- which reported parameters multiply the design.

Samplers are written from the full conditionals in the cited papers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg, special

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._core import (
    find_mode,
    log_gamma,
    log_invgamma,
    log_mvnorm,
    normal_prior,
    random_walk_metropolis,
    rinvgamma,
    rmvnorm_prec,
    rtruncnorm,
)

_LOG_2PI = float(np.log(2.0 * np.pi))


def _log_diff_ndtr(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """log(Phi(hi) - Phi(lo)) for lo < hi, accurate in both tails."""
    flip = lo > 0
    a = np.where(flip, -hi, lo)
    b = np.where(flip, -lo, hi)
    lb = special.log_ndtr(b)
    la = special.log_ndtr(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(lb + np.log1p(-np.exp(np.minimum(la - lb, 0.0))))


class _Model:
    """Common state: data, the normal prior on the coefficients."""

    name = ""
    #: sampler family, reported to the user
    sampler = ""
    #: marginal-likelihood methods the model supports beyond the generic ones
    has_chib = False

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        xnames: Sequence[str],
        prior_mean: Any,
        prior_var: Any,
    ) -> None:
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.n, self.k = self.X.shape
        self.xnames = [str(c) for c in xnames]
        self.b0, self.B0, self.B0inv = normal_prior(
            self.k, prior_mean, prior_var, self.xnames
        )
        self.B0inv_b0 = self.B0inv @ self.b0
        self.aux_names: List[str] = []
        self._mode: Optional[np.ndarray] = None
        self._mode_cov: Optional[np.ndarray] = None

    # -- interface ---------------------------------------------------------
    @property
    def names(self) -> List[str]:
        return self.xnames + self.aux_names

    @property
    def n_par(self) -> int:
        return self.k + len(self.aux_names)

    def to_u(self, draws: np.ndarray) -> np.ndarray:
        return np.asarray(draws, dtype=float)

    def from_u(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(u, dtype=float)

    def log_lik(self, theta: np.ndarray) -> float:
        raise MethodIncompatibility(f"model='{self.name}' does not define this step.")

    def log_prior(self, theta: np.ndarray) -> float:
        return log_mvnorm(theta[: self.k], self.b0, self.B0)

    def log_jac(self, u: np.ndarray) -> float:
        return 0.0

    def log_kernel(self, u: np.ndarray) -> float:
        theta = self.from_u(u)
        val = self.log_lik(theta) + self.log_prior(theta) + self.log_jac(u)
        return float(val) if np.isfinite(val) else -np.inf

    def start_u(self) -> np.ndarray:
        raise MethodIncompatibility(f"model='{self.name}' does not define this step.")

    def mode(self) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mode on the unconstrained scale and its Laplace
        covariance."""
        if self._mode is None:
            self._mode, self._mode_cov = find_mode(
                lambda u: -self.log_kernel(u),
                self.start_u(),
                what=f"{self.name} posterior",
            )
        assert self._mode is not None and self._mode_cov is not None
        return self._mode, self._mode_cov

    def sample(
        self, rng: np.random.Generator, n_iter: int, jitter: bool
    ) -> Dict[str, Any]:
        raise MethodIncompatibility(f"model='{self.name}' does not define this step.")

    def linear_predictor(self, draws: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.asarray(draws[:, : self.k] @ X.T)

    def expected_value(self, eta: np.ndarray) -> np.ndarray:
        return eta

    # helpers --------------------------------------------------------------
    def _ols(self) -> np.ndarray:
        return np.asarray(np.linalg.lstsq(self.X, self.y, rcond=None)[0])


# ==========================================================================
# Gaussian linear model, independent normal / inverse-gamma prior
# ==========================================================================


class NormalModel(_Model):
    name = "normal"
    sampler = "Gibbs"
    has_chib = True

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        a0: float,
        d0: float,
    ) -> None:
        super().__init__(y, X, xnames, prior_mean, prior_var)
        if a0 <= 0 or d0 <= 0:
            raise MethodIncompatibility(
                "sigma2_prior must be two positive numbers (alpha0, delta0): "
                "sigma^2 ~ InvGamma(alpha0 / 2, delta0 / 2)."
            )
        self.a0, self.d0 = float(a0), float(d0)
        self.aux_names = ["sigma2"]
        self.XtX = self.X.T @ self.X
        self.Xty = self.X.T @ self.y
        self.yty = float(self.y @ self.y)

    def to_u(self, draws: np.ndarray) -> np.ndarray:
        u = np.array(draws, dtype=float)
        u[..., -1] = np.log(u[..., -1])
        return u

    def from_u(self, u: np.ndarray) -> np.ndarray:
        t = np.array(u, dtype=float)
        t[..., -1] = np.exp(t[..., -1])
        return t

    def _ssr(self, beta: np.ndarray) -> float:
        return float(self.yty - 2.0 * beta @ self.Xty + beta @ self.XtX @ beta)

    def log_lik(self, theta: np.ndarray) -> float:
        beta, s2 = theta[: self.k], theta[-1]
        return float(
            -0.5 * self.n * (_LOG_2PI + np.log(s2)) - 0.5 * self._ssr(beta) / s2
        )

    def log_prior(self, theta: np.ndarray) -> float:
        return super().log_prior(theta) + log_invgamma(
            theta[-1], self.a0 / 2.0, self.d0 / 2.0
        )

    def log_jac(self, u: np.ndarray) -> float:
        return float(u[-1])

    def start_u(self) -> np.ndarray:
        b = self._ols()
        s2 = max(self._ssr(b) / max(self.n - self.k, 1), 1e-12)
        return np.append(b, np.log(s2))

    def beta_conditional(self, s2: float) -> Tuple[np.ndarray, np.ndarray]:
        """Mean and precision of beta | sigma2, y."""
        prec = self.B0inv + self.XtX / s2
        rhs = self.B0inv_b0 + self.Xty / s2
        chol = linalg.cholesky(prec, lower=True, check_finite=False)
        return linalg.cho_solve((chol, True), rhs, check_finite=False), prec

    def sample(
        self, rng: np.random.Generator, n_iter: int, jitter: bool
    ) -> Dict[str, Any]:
        out = np.empty((n_iter, self.k + 1))
        beta = self._ols()
        if jitter:
            s2_0 = max(self._ssr(beta) / max(self.n - self.k, 1), 1e-12)
            cov = s2_0 * np.linalg.pinv(self.XtX)
            beta = rng.multivariate_normal(beta, 4.0 * cov, method="svd")
        an = self.a0 + self.n
        for it in range(n_iter):
            s2 = rinvgamma(rng, an / 2.0, (self.d0 + self._ssr(beta)) / 2.0)
            beta, _ = rmvnorm_prec(
                rng, self.B0inv_b0 + self.Xty / s2, self.B0inv + self.XtX / s2
            )
            out[it, : self.k] = beta
            out[it, -1] = s2
        return {"draws": out, "accept": None, "extras": {}}

    def chib(self, draws: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Chib (1995) log marginal likelihood from the Gibbs output."""
        beta_s = draws[:, : self.k].mean(axis=0)
        # pi(beta* | y) = E_{sigma2 | y} N(beta*; b_n(s2), B_n(s2))
        logs = np.empty(draws.shape[0])
        for g, s2 in enumerate(draws[:, -1]):
            mean, prec = self.beta_conditional(float(s2))
            chol = linalg.cholesky(prec, lower=True, check_finite=False)
            dev = chol.T @ (beta_s - mean)
            logs[g] = (
                -0.5 * self.k * _LOG_2PI + np.log(np.diag(chol)).sum() - 0.5 * dev @ dev
            )
        log_ord_beta = float(special.logsumexp(logs) - np.log(logs.size))
        # sigma2 | beta*, y is inverse gamma: take its mode as sigma2*
        shape = (self.a0 + self.n) / 2.0
        rate = (self.d0 + self._ssr(beta_s)) / 2.0
        s2_s = rate / (shape + 1.0)
        log_ord_s2 = log_invgamma(s2_s, shape, rate)
        theta = np.append(beta_s, s2_s)
        lml = self.log_lik(theta) + self.log_prior(theta) - log_ord_beta - log_ord_s2
        return float(lml), {"log_posterior_ordinate": log_ord_beta + log_ord_s2}


# ==========================================================================
# Student-t errors (scale mixture of normals), fixed degrees of freedom
# ==========================================================================


class StudentTModel(NormalModel):
    name = "t"
    sampler = "Gibbs (scale mixture of normals)"
    has_chib = False

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        a0: Any,
        d0: Any,
        df: float,
    ) -> None:
        super().__init__(y, X, xnames, prior_mean, prior_var, a0, d0)
        if not df > 0:
            raise MethodIncompatibility(f"dof must be positive; got {df}.")
        self.df = float(df)

    def log_lik(self, theta: np.ndarray) -> float:
        beta, s2 = theta[: self.k], theta[-1]
        e2 = (self.y - self.X @ beta) ** 2
        v = self.df
        const = (
            special.gammaln((v + 1.0) / 2.0)
            - special.gammaln(v / 2.0)
            - 0.5 * np.log(v * np.pi * s2)
        )
        return float(self.n * const - 0.5 * (v + 1.0) * np.log1p(e2 / (v * s2)).sum())

    def sample(
        self, rng: np.random.Generator, n_iter: int, jitter: bool
    ) -> Dict[str, Any]:
        out = np.empty((n_iter, self.k + 1))
        beta = self._ols()
        s2 = max(self._ssr(beta) / max(self.n - self.k, 1), 1e-12)
        if jitter:
            cov = s2 * np.linalg.pinv(self.XtX)
            beta = rng.multivariate_normal(beta, 4.0 * cov, method="svd")
        v = self.df
        an = self.a0 + self.n
        X, y = self.X, self.y
        for it in range(n_iter):
            e = y - X @ beta
            lam = rng.gamma((v + 1.0) / 2.0, 2.0 / (v + e * e / s2))
            Xw = X * lam[:, None]
            beta, _ = rmvnorm_prec(
                rng, self.B0inv_b0 + Xw.T @ y / s2, self.B0inv + Xw.T @ X / s2
            )
            e = y - X @ beta
            s2 = rinvgamma(rng, an / 2.0, (self.d0 + float(lam @ (e * e))) / 2.0)
            out[it, : self.k] = beta
            out[it, -1] = s2
        return {"draws": out, "accept": None, "extras": {}}


# ==========================================================================
# Probit: Albert and Chib (1993) data augmentation
# ==========================================================================


def _check_binary(y: np.ndarray, model: str) -> None:
    vals = np.unique(y)
    if not np.all(np.isin(vals, (0.0, 1.0))):
        raise MethodIncompatibility(
            f"model='{model}' needs a 0/1 outcome; found values "
            f"{vals[:6].tolist()}{'...' if vals.size > 6 else ''}."
        )
    if vals.size < 2:
        raise DataInsufficient(
            f"The outcome is constant (all {int(vals[0])}); a binary model "
            "is not identified."
        )


class ProbitModel(_Model):
    name = "probit"
    sampler = "Gibbs (data augmentation)"
    has_chib = True

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
    ) -> None:
        super().__init__(y, X, xnames, prior_mean, prior_var)
        _check_binary(self.y, "probit")
        self.q = 2.0 * self.y - 1.0
        prec = self.B0inv + self.X.T @ self.X
        self.Bn_chol = linalg.cholesky(prec, lower=True)
        self.prec = prec

    def log_lik(self, theta: np.ndarray) -> float:
        return float(special.log_ndtr(self.q * (self.X @ theta[: self.k])).sum())

    def start_u(self) -> np.ndarray:
        return np.zeros(self.k)

    def expected_value(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(special.ndtr(eta))

    def sample(
        self, rng: np.random.Generator, n_iter: int, jitter: bool
    ) -> Dict[str, Any]:
        k = self.k
        out = np.empty((n_iter, k))
        cond_means = np.empty((n_iter, k))
        beta: np.ndarray = np.zeros(k)
        if jitter:
            beta = rng.standard_normal(k) * 0.1
        lower = np.where(self.y == 1.0, 0.0, -np.inf)
        upper = np.where(self.y == 1.0, np.inf, 0.0)
        X = self.X
        for it in range(n_iter):
            z = rtruncnorm(rng, X @ beta, 1.0, lower, upper)
            mean = linalg.cho_solve(
                (self.Bn_chol, True), self.B0inv_b0 + X.T @ z, check_finite=False
            )
            beta = mean + linalg.solve_triangular(
                self.Bn_chol.T, rng.standard_normal(k), lower=False, check_finite=False
            )
            out[it] = beta
            cond_means[it] = mean
        return {"draws": out, "accept": None, "extras": {"cond_mean": cond_means}}

    def chib(
        self, draws: np.ndarray, cond_mean: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        beta_s = draws.mean(axis=0)
        dev = (beta_s[None, :] - cond_mean) @ self.Bn_chol
        logs = (
            -0.5 * self.k * _LOG_2PI
            + np.log(np.diag(self.Bn_chol)).sum()
            - 0.5 * (dev * dev).sum(axis=1)
        )
        log_ord = float(special.logsumexp(logs) - np.log(logs.size))
        lml = self.log_lik(beta_s) + self.log_prior(beta_s) - log_ord
        return float(lml), {"log_posterior_ordinate": log_ord}


# ==========================================================================
# Random-walk Metropolis models: logit, Poisson, negative binomial
# ==========================================================================


class _RWModel(_Model):
    sampler = "random-walk Metropolis"

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        tune: Optional[float],
    ) -> None:
        super().__init__(y, X, xnames, prior_mean, prior_var)
        self.tune = tune

    def _tune(self) -> float:
        if self.tune is not None:
            if not self.tune > 0:
                raise MethodIncompatibility(f"tune must be positive; got {self.tune}.")
            return float(self.tune)
        return float(2.38 / np.sqrt(self.n_par))

    def start_u(self) -> np.ndarray:
        return np.zeros(self.n_par)

    def sample(
        self, rng: np.random.Generator, n_iter: int, jitter: bool
    ) -> Dict[str, Any]:
        mode, cov = self.mode()
        start = mode
        if jitter:
            start = rng.multivariate_normal(mode, 4.0 * cov, method="svd")
        tune = self._tune()
        u, acc = random_walk_metropolis(
            rng, self.log_kernel, start, tune * tune * cov, n_iter
        )
        return {"draws": self.from_u(u), "accept": acc, "extras": {"tune": tune}}


class LogitModel(_RWModel):
    name = "logit"

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        tune: Any,
    ) -> None:
        super().__init__(y, X, xnames, prior_mean, prior_var, tune)
        _check_binary(self.y, "logit")

    def log_lik(self, theta: np.ndarray) -> float:
        eta = self.X @ theta[: self.k]
        return float(self.y @ eta - np.logaddexp(0.0, eta).sum())

    def expected_value(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(special.expit(eta))


def _check_count(y: np.ndarray, model: str) -> None:
    if np.any(y < 0) or np.any(np.abs(y - np.round(y)) > 1e-9):
        raise MethodIncompatibility(
            f"model='{model}' needs a non-negative integer outcome."
        )


class PoissonModel(_RWModel):
    name = "poisson"

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        tune: Any,
    ) -> None:
        super().__init__(y, X, xnames, prior_mean, prior_var, tune)
        _check_count(self.y, "poisson")
        self._lgam = float(special.gammaln(self.y + 1.0).sum())

    def log_lik(self, theta: np.ndarray) -> float:
        eta = self.X @ theta[: self.k]
        with np.errstate(over="ignore"):
            return float(self.y @ eta - np.exp(eta).sum() - self._lgam)

    def start_u(self) -> np.ndarray:
        u = np.zeros(self.k)
        # log of the mean on the intercept, when there is one
        const = np.where(np.ptp(self.X, axis=0) == 0)[0]
        if const.size:
            u[const[0]] = np.log(max(self.y.mean(), 1e-8)) / self.X[0, const[0]]
        return u

    def expected_value(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(np.exp(eta))


class NegBinModel(_RWModel):
    """Negative binomial, variance ``mu + alpha mu^2``.

    The prior is on the size ``1 / alpha``: Gamma(shape, rate).
    """

    name = "negbin"

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        tune: Any,
        shape: Any,
        rate: Any,
    ) -> None:
        super().__init__(y, X, xnames, prior_mean, prior_var, tune)
        _check_count(self.y, "negbin")
        if shape <= 0 or rate <= 0:
            raise MethodIncompatibility(
                "size_prior must be two positive numbers (shape, rate) of the "
                "Gamma prior on the size 1 / alpha."
            )
        self.sh, self.rt = float(shape), float(rate)
        self.aux_names = ["alpha"]
        self._lgam = float(special.gammaln(self.y + 1.0).sum())

    # u = (beta, log size); reported = (beta, alpha = 1 / size)
    def to_u(self, draws: np.ndarray) -> np.ndarray:
        u = np.array(draws, dtype=float)
        u[..., -1] = -np.log(u[..., -1])
        return u

    def from_u(self, u: np.ndarray) -> np.ndarray:
        t = np.array(u, dtype=float)
        t[..., -1] = np.exp(-t[..., -1])
        return t

    def log_lik(self, theta: np.ndarray) -> float:
        beta, r = theta[: self.k], 1.0 / theta[-1]
        eta = self.X @ beta
        with np.errstate(over="ignore"):
            mu = np.exp(eta)
        y = self.y
        val = (
            special.gammaln(y + r).sum()
            - self.n * special.gammaln(r)
            - self._lgam
            + self.n * r * np.log(r)
            + y @ eta
            - ((y + r) * np.log(r + mu)).sum()
        )
        return float(val)

    def log_prior(self, theta: np.ndarray) -> float:
        return super().log_prior(theta) + log_gamma(1.0 / theta[-1], self.sh, self.rt)

    def log_jac(self, u: np.ndarray) -> float:
        # prior and likelihood are stated in the size r = exp(u_last)
        return float(u[-1])

    def start_u(self) -> np.ndarray:
        u = np.zeros(self.k + 1)
        const = np.where(np.ptp(self.X, axis=0) == 0)[0]
        if const.size:
            u[const[0]] = np.log(max(self.y.mean(), 1e-8)) / self.X[0, const[0]]
        return u

    def expected_value(self, eta: np.ndarray) -> np.ndarray:
        return np.asarray(np.exp(eta))


# ==========================================================================
# Tobit: data augmentation for the censored observations
# ==========================================================================


class TobitModel(NormalModel):
    name = "tobit"
    sampler = "Gibbs (data augmentation)"
    has_chib = False

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        a0: Any,
        d0: Any,
        lower: Any,
        upper: Any,
    ) -> None:
        super().__init__(y, X, xnames, prior_mean, prior_var, a0, d0)
        self.lower = -np.inf if lower is None else float(lower)
        self.upper = np.inf if upper is None else float(upper)
        if not self.lower < self.upper:
            raise MethodIncompatibility(
                f"lower ({lower}) must be below upper ({upper})."
            )
        if np.isneginf(self.lower) and np.isposinf(self.upper):
            raise MethodIncompatibility(
                "model='tobit' needs a censoring point: pass lower= and / or "
                "upper=. Without one the model is model='normal'."
            )
        self.left = self.y <= self.lower
        self.right = self.y >= self.upper
        self.unc = ~(self.left | self.right)
        if self.unc.sum() <= self.k:
            raise DataInsufficient(
                f"Only {int(self.unc.sum())} uncensored observations for "
                f"{self.k} coefficients."
            )

    def log_lik(self, theta: np.ndarray) -> float:
        beta, s2 = theta[: self.k], theta[-1]
        s = np.sqrt(s2)
        mu = self.X @ beta
        e = self.y[self.unc] - mu[self.unc]
        val = -0.5 * self.unc.sum() * (_LOG_2PI + np.log(s2)) - 0.5 * (e @ e) / s2
        if self.left.any():
            val += special.log_ndtr((self.lower - mu[self.left]) / s).sum()
        if self.right.any():
            val += special.log_ndtr((mu[self.right] - self.upper) / s).sum()
        return float(val)

    def start_u(self) -> np.ndarray:
        Xu, yu = self.X[self.unc], self.y[self.unc]
        b = np.linalg.lstsq(Xu, yu, rcond=None)[0]
        e = yu - Xu @ b
        return np.append(b, np.log(max(e @ e / max(yu.size - self.k, 1), 1e-12)))

    def sample(
        self, rng: np.random.Generator, n_iter: int, jitter: bool
    ) -> Dict[str, Any]:
        out = np.empty((n_iter, self.k + 1))
        start = self.start_u()
        beta, s2 = start[: self.k], float(np.exp(start[-1]))
        if jitter:
            beta = beta + rng.standard_normal(self.k) * 2.0 * np.sqrt(
                np.diag(s2 * np.linalg.pinv(self.XtX))
            )
        an = self.a0 + self.n
        X = self.X
        z = self.y.copy()
        left, right = self.left, self.right
        XtX = self.XtX
        for it in range(n_iter):
            mu = X @ beta
            s = np.sqrt(s2)
            if left.any():
                z[left] = rtruncnorm(rng, mu[left], s, -np.inf, self.lower)
            if right.any():
                z[right] = rtruncnorm(rng, mu[right], s, self.upper, np.inf)
            beta, _ = rmvnorm_prec(
                rng, self.B0inv_b0 + X.T @ z / s2, self.B0inv + XtX / s2
            )
            e = z - X @ beta
            s2 = rinvgamma(rng, an / 2.0, (self.d0 + float(e @ e)) / 2.0)
            out[it, : self.k] = beta
            out[it, -1] = s2
        return {"draws": out, "accept": None, "extras": {}}

    def expected_value(self, eta: np.ndarray) -> np.ndarray:
        # the latent mean; the censored mean needs sigma and is left to
        # the user
        return eta


# ==========================================================================
# Quantile regression: asymmetric Laplace likelihood,
# Kozumi and Kobayashi (2011) Gibbs sampler
# ==========================================================================


class QuantileModel(_Model):
    name = "quantile"
    sampler = "Gibbs (normal-exponential mixture)"

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        quantile: Any,
        n0: Any,
        s0: Any,
        scale: Any,
    ) -> None:
        super().__init__(y, X, xnames, prior_mean, prior_var)
        if not 0.0 < quantile < 1.0:
            raise MethodIncompatibility(f"quantile must be in (0, 1); got {quantile}.")
        self.p = float(quantile)
        self.theta = (1.0 - 2.0 * self.p) / (self.p * (1.0 - self.p))
        self.tau2 = 2.0 / (self.p * (1.0 - self.p))
        self.fixed_scale: Optional[float] = None
        if scale is not None:
            if not scale > 0:
                raise MethodIncompatibility(f"scale must be positive; got {scale}.")
            self.fixed_scale = float(scale)
        else:
            if n0 <= 0 or s0 <= 0:
                raise MethodIncompatibility(
                    "scale_prior must be two positive numbers (n0, s0): "
                    "sigma ~ InvGamma(n0 / 2, s0 / 2)."
                )
            self.aux_names = ["sigma"]
        self.n0, self.s0 = float(n0), float(s0)

    def to_u(self, draws: np.ndarray) -> np.ndarray:
        u = np.array(draws, dtype=float)
        if self.fixed_scale is None:
            u[..., -1] = np.log(u[..., -1])
        return u

    def from_u(self, u: np.ndarray) -> np.ndarray:
        t = np.array(u, dtype=float)
        if self.fixed_scale is None:
            t[..., -1] = np.exp(t[..., -1])
        return t

    def log_lik(self, theta: np.ndarray) -> float:
        beta = theta[: self.k]
        sig = self.fixed_scale if self.fixed_scale is not None else theta[-1]
        e = (self.y - self.X @ beta) / sig
        check = e * (self.p - (e < 0))
        return float(
            self.n * (np.log(self.p * (1.0 - self.p)) - np.log(sig)) - check.sum()
        )

    def log_prior(self, theta: np.ndarray) -> float:
        val = super().log_prior(theta)
        if self.fixed_scale is None:
            val += log_invgamma(theta[-1], self.n0 / 2.0, self.s0 / 2.0)
        return val

    def log_jac(self, u: np.ndarray) -> float:
        return float(u[-1]) if self.fixed_scale is None else 0.0

    def start_u(self) -> np.ndarray:
        b = self._ols()
        if self.fixed_scale is not None:
            return b
        e = self.y - self.X @ b
        return np.append(b, np.log(max(np.abs(e).mean() / 2.0, 1e-8)))

    def mode(self) -> Tuple[np.ndarray, np.ndarray]:
        # the check loss is not differentiable: use a smooth surrogate of
        # the posterior only for a Laplace-type covariance
        raise MethodIncompatibility(
            "The asymmetric Laplace posterior is not differentiable, so a "
            "Laplace approximation is not available for model='quantile'."
        )

    def sample(
        self, rng: np.random.Generator, n_iter: int, jitter: bool
    ) -> Dict[str, Any]:
        k = self.k
        est_scale = self.fixed_scale is None
        out = np.empty((n_iter, k + (1 if est_scale else 0)))
        X, y = self.X, self.y
        beta = self._ols()
        e0 = y - X @ beta
        sig: float = max(float(np.abs(e0).mean()) / 2.0, 1e-8)
        if self.fixed_scale is not None:
            sig = self.fixed_scale
        if jitter:
            beta = beta + rng.standard_normal(k) * 2.0 * np.sqrt(
                np.diag(np.var(e0) * np.linalg.pinv(X.T @ X))
            )
        th, tau2 = self.theta, self.tau2
        for it in range(n_iter):
            e = y - X @ beta
            # v_i | . ~ GIG(1/2, delta_i^2, gamma^2): 1 / v_i is inverse Gaussian
            delta = np.maximum(np.abs(e) / np.sqrt(tau2 * sig), 1e-10)
            gam2 = 2.0 / sig + th * th / (tau2 * sig)
            v = 1.0 / rng.wald(np.sqrt(gam2) / delta, gam2)
            v = np.maximum(v, 1e-12)
            w = 1.0 / (tau2 * sig * v)
            Xw = X * w[:, None]
            beta, _ = rmvnorm_prec(
                rng, self.B0inv_b0 + Xw.T @ (y - th * v), self.B0inv + Xw.T @ X
            )
            out[it, :k] = beta
            if est_scale:
                r = y - X @ beta - th * v
                rate = self.s0 / 2.0 + v.sum() + float((r * r / v).sum()) / (2.0 * tau2)
                sig = rinvgamma(rng, self.n0 / 2.0 + 1.5 * self.n, rate)
                out[it, -1] = sig
        return {"draws": out, "accept": None, "extras": {}}


# ==========================================================================
# Ordered probit: data augmentation, unconstrained cutpoint increments
# ==========================================================================


class OrderedProbitModel(_Model):
    """Ordered probit.

    Internally: an intercept, the first cutpoint fixed at 0 and the
    others ``c_j = c_{j-1} + exp(d_j)``. Reported: the slopes and the
    cutpoints of the model without intercept (``cut_j = c_j - intercept``),
    the parameterisation of ``sp.oprobit`` and Stata.
    """

    name = "oprobit"
    sampler = "Gibbs with a Metropolis step for the cutpoints"

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        tune: Any,
        cut_prior_var: Any,
        levels: Any,
    ) -> None:
        # X arrives with the intercept in column 0
        super().__init__(y, X, xnames, prior_mean, prior_var)
        self.levels = list(levels)
        self.J = len(self.levels)
        if self.J < 3:
            raise MethodIncompatibility(
                "model='oprobit' needs an outcome with at least three "
                f"ordered categories; found {self.J}. Use model='probit' for "
                "a binary outcome."
            )
        counts = np.bincount(self.y.astype(int), minlength=self.J)
        if (counts == 0).any():
            raise DataInsufficient("An outcome category has no observations.")
        self.nd = self.J - 2
        if not cut_prior_var > 0:
            raise MethodIncompatibility("cut_prior_var must be positive.")
        self.dvar = float(cut_prior_var)
        self.tune = tune
        self.yi = self.y.astype(int)
        self.slope_names = self.xnames[1:]
        self.cut_names = [f"cut{j}" for j in range(1, self.J)]
        self.XtX = self.X.T @ self.X
        prec = self.B0inv + self.XtX
        self.Bn_chol = linalg.cholesky(prec, lower=True)

    @property
    def names(self) -> List[str]:
        return self.slope_names + self.cut_names

    @property
    def n_par(self) -> int:
        return self.k + self.nd

    # internal vector: (intercept, slopes, d_2 .. d_{J-1})
    def _cuts(self, d: np.ndarray) -> np.ndarray:
        """Internal cutpoints c_0 = -inf, c_1 = 0, ..., c_J = +inf."""
        inner = np.concatenate(([0.0], np.cumsum(np.exp(d))))
        return np.concatenate(([-np.inf], inner, [np.inf]))

    def to_u(self, draws: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(np.asarray(draws, dtype=float))
        ks = self.k - 1
        slopes = d[:, :ks]
        cuts = d[:, ks:]
        inter = -cuts[:, :1]
        dd = np.log(np.diff(cuts, axis=1))
        u = np.hstack([inter, slopes, dd])
        return u if np.ndim(draws) == 2 else u[0]

    def from_u(self, u: np.ndarray) -> np.ndarray:
        uu = np.atleast_2d(np.asarray(u, dtype=float))
        inter = uu[:, :1]
        slopes = uu[:, 1 : self.k]
        inner = np.hstack(
            [np.zeros((uu.shape[0], 1)), np.cumsum(np.exp(uu[:, self.k :]), axis=1)]
        )
        t = np.hstack([slopes, inner - inter])
        return t if np.ndim(u) == 2 else t[0]

    def _loglik_internal(self, b: np.ndarray, d: np.ndarray) -> float:
        cuts = self._cuts(d)
        eta = self.X @ b
        lo = cuts[self.yi] - eta
        hi = cuts[self.yi + 1] - eta
        return float(_log_diff_ndtr(lo, hi).sum())

    def log_kernel(self, u: np.ndarray) -> float:
        b, d = u[: self.k], u[self.k :]
        with np.errstate(over="ignore"):
            val = (
                self._loglik_internal(b, d)
                + log_mvnorm(b, self.b0, self.B0)
                - 0.5 * self.nd * (_LOG_2PI + np.log(self.dvar))
                - 0.5 * float(d @ d) / self.dvar
            )
        return float(val) if np.isfinite(val) else -np.inf

    def log_lik(self, theta: np.ndarray) -> float:
        u = self.to_u(theta)
        return self._loglik_internal(u[: self.k], u[self.k :])

    def start_u(self) -> np.ndarray:
        u = np.zeros(self.k + self.nd)
        # spread the cutpoints by the marginal frequencies
        cum = np.cumsum(np.bincount(self.yi, minlength=self.J))[:-1] / self.n
        q = special.ndtri(np.clip(cum, 1e-4, 1 - 1e-4))
        u[0] = -q[0]
        u[self.k :] = np.log(np.maximum(np.diff(q), 1e-3))
        return u

    def linear_predictor(self, draws: np.ndarray, X: np.ndarray) -> np.ndarray:
        # X carries the intercept column; the reported model has none
        return np.asarray(draws[:, : self.k - 1] @ X[:, 1:].T)

    def sample(
        self, rng: np.random.Generator, n_iter: int, jitter: bool
    ) -> Dict[str, Any]:
        mode, cov = self.mode()
        k, nd = self.k, self.nd
        b = mode[:k].copy()
        d = mode[k:].copy()
        if jitter:
            start = rng.multivariate_normal(mode, 4.0 * cov, method="svd")
            b, d = start[:k].copy(), start[k:].copy()
        tune = self.tune if self.tune is not None else 2.38 / np.sqrt(nd)
        if not tune > 0:
            raise MethodIncompatibility(f"tune must be positive; got {tune}.")
        # proposal for d: conditional Laplace covariance given the
        # coefficients, i.e. the inverse of the d-block of the precision
        prec_full = linalg.inv(cov)
        prop_cov = tune * tune * linalg.inv(prec_full[k:, k:])
        prop_chol = linalg.cholesky(0.5 * (prop_cov + prop_cov.T), lower=True)
        out_u = np.empty((n_iter, k + nd))
        X = self.X
        accepted = 0

        def lp_d(dd: np.ndarray, eta: np.ndarray) -> float:
            with np.errstate(over="ignore"):
                cuts = self._cuts(dd)
            lo = cuts[self.yi] - eta
            hi = cuts[self.yi + 1] - eta
            val = _log_diff_ndtr(lo, hi).sum() - 0.5 * float(dd @ dd) / self.dvar
            return float(val) if np.isfinite(val) else -np.inf

        for it in range(n_iter):
            eta = X @ b
            # 1. cutpoints given the coefficients, latent data integrated out
            cand = d + prop_chol @ rng.standard_normal(nd)
            if np.log(rng.random()) < lp_d(cand, eta) - lp_d(d, eta):
                d = cand
                accepted += 1
            # 2. latent data given coefficients and cutpoints
            cuts = self._cuts(d)
            z = rtruncnorm(rng, eta, 1.0, cuts[self.yi], cuts[self.yi + 1])
            # 3. coefficients given the latent data
            mean = linalg.cho_solve(
                (self.Bn_chol, True), self.B0inv_b0 + X.T @ z, check_finite=False
            )
            b = mean + linalg.solve_triangular(
                self.Bn_chol.T, rng.standard_normal(k), lower=False, check_finite=False
            )
            out_u[it, :k] = b
            out_u[it, k:] = d
        return {
            "draws": self.from_u(out_u),
            "accept": accepted / n_iter,
            "extras": {"tune": tune},
        }

    def category_probabilities(self, draws: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Posterior mean probability of each category, rows of ``X``."""
        ks = self.k - 1
        eta = draws[:, :ks] @ X[:, 1:].T  # S x n
        cuts = draws[:, ks:]  # S x (J-1)
        cdf = special.ndtr(cuts[:, None, :] - eta[:, :, None])  # S x n x (J-1)
        cdf = np.concatenate(
            [np.zeros(cdf.shape[:2] + (1,)), cdf, np.ones(cdf.shape[:2] + (1,))], axis=2
        )
        return np.asarray(np.diff(cdf, axis=2).mean(axis=0))


MODELS = (
    "normal",
    "conjugate",
    "t",
    "logit",
    "probit",
    "oprobit",
    "poisson",
    "negbin",
    "tobit",
    "quantile",
)
