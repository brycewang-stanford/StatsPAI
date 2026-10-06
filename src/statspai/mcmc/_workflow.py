"""
Pieces of the Bayesian regression workflow that sit around the samplers
of :mod:`statspai.mcmc._models`: weakly informative default priors on
the scale of the data, the pointwise log-likelihood of each model, and
draws from its predictive distribution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import special, stats

from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _models as M
from ._core import rinvgamma, rmvnorm_prec

_LOG_2PI = float(np.log(2.0 * np.pi))

#: likelihoods with a pointwise log-likelihood and predictive draws here
PREDICTIVE_MODELS = (
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
    "mlogit",
)

#: models that accept ``prior='weakly_informative'``
WEAK_PRIOR_MODELS = ("normal", "logit", "probit", "poisson", "negbin")


# ---------------------------------------------------------------------
# weakly informative priors
# ---------------------------------------------------------------------


def weakly_informative_prior(
    model_key: str,
    y: np.ndarray,
    X: np.ndarray,
    xnames: List[str],
    scale: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Prior mean and covariance of the coefficients, scaled to the data.

    Independent normal priors, ``N(0, (scale * s_y / s_x)^2)`` on each
    slope and ``N(m_y, (scale * s_y)^2)`` on the intercept *after the
    regressors are centred*, with ``s_y = sd(y)``, ``m_y = mean(y)`` for
    the Gaussian model and ``s_y = 1``, ``m_y = 0`` for the others. This
    is the default of R ``rstanarm`` (2.21 and later) and the practice of
    Gelman, Hill and Vehtari (2020).

    A prior on the centred intercept is a correlated prior on the raw
    coefficients: with ``alpha_c = alpha + xbar' beta`` the raw intercept
    is ``alpha_c - xbar' beta``. The covariance returned is that of the
    raw coefficients, so the samplers need no centring.
    """
    if model_key not in WEAK_PRIOR_MODELS:
        raise MethodIncompatibility(
            "prior='weakly_informative' is defined for models "
            f"{', '.join(WEAK_PRIOR_MODELS)}; got '{model_key}'. Pass "
            "prior_mean= and prior_var= for this model."
        )
    n, k = X.shape
    gaussian = model_key == "normal"
    s_y = float(np.std(y, ddof=1)) if gaussian else 1.0
    m_y = float(np.mean(y)) if gaussian else 0.0
    if gaussian and not s_y > 0:
        raise DataInsufficient("The outcome does not vary.")
    icpt = xnames.index("Intercept") if "Intercept" in xnames else None
    slopes = [j for j in range(k) if j != icpt]
    s_x = np.std(X, axis=0, ddof=1)
    if np.any(s_x[slopes] <= 0):
        flat = [xnames[j] for j in slopes if s_x[j] <= 0]
        raise MethodIncompatibility(
            f"The regressors {flat} do not vary; a prior scaled by their "
            "standard deviation is undefined."
        )
    sd = np.zeros(k)
    sd[slopes] = scale * s_y / s_x[slopes]
    mean_c = np.zeros(k)
    info: Dict[str, Any] = {
        "coefficients": "normal(0, scale), independent",
        "scale": {xnames[j]: float(sd[j]) for j in slopes},
    }
    transform = np.eye(k)
    if icpt is not None:
        sd[icpt] = scale * s_y
        mean_c[icpt] = m_y
        xbar = X.mean(axis=0)
        transform[icpt, slopes] = -xbar[slopes]
        info["intercept"] = (
            f"normal({m_y:.5g}, {sd[icpt]:.5g}) on the intercept with the "
            "regressors centred"
        )
    cov = transform @ np.diag(sd**2) @ transform.T
    cov = 0.5 * (cov + cov.T)
    return transform @ mean_c, cov, info


class NormalExpSigmaModel(M.NormalModel):
    """Gaussian linear model with ``sigma ~ Exponential(rate)``.

    The coefficients are drawn from their normal full conditional. The
    full conditional of the variance is the flat-prior inverse gamma
    times ``exp(-rate * sigma)``; a draw from the inverse gamma is
    accepted with probability ``exp(-rate (sigma' - sigma))``, an exact
    independence Metropolis step whose acceptance rate is close to one
    because the prior is weak next to the likelihood.
    """

    name = "normal"
    sampler = "Gibbs (independence Metropolis step for sigma)"
    has_chib = False

    def __init__(
        self,
        y: Any,
        X: Any,
        xnames: Any,
        prior_mean: Any,
        prior_var: Any,
        rate: float,
    ) -> None:
        super().__init__(y, X, xnames, prior_mean, prior_var, 1.0, 1.0)
        if not rate > 0:
            raise MethodIncompatibility("The exponential rate must be positive.")
        if self.n < 3:
            raise DataInsufficient("At least three observations are needed.")
        self.rate = float(rate)

    def log_prior(self, theta: np.ndarray) -> float:
        s = float(np.sqrt(theta[-1]))
        # density of sigma^2 when sigma is exponential
        return float(
            M._Model.log_prior(self, theta)
            + np.log(self.rate)
            - self.rate * s
            - np.log(2.0 * s)
        )

    def sample(
        self, rng: np.random.Generator, n_iter: int, jitter: bool
    ) -> Dict[str, Any]:
        out = np.empty((n_iter, self.k + 1))
        beta = self._ols()
        s2 = max(self._ssr(beta) / max(self.n - self.k, 1), 1e-12)
        if jitter:
            cov = s2 * np.linalg.pinv(self.XtX)
            beta = rng.multivariate_normal(beta, 4.0 * cov, method="svd")
        shape = (self.n - 1.0) / 2.0
        accepted = 0
        for it in range(n_iter):
            proposal = rinvgamma(rng, shape, self._ssr(beta) / 2.0)
            log_ratio = -self.rate * (np.sqrt(proposal) - np.sqrt(s2))
            if it == 0 or np.log(rng.uniform()) < log_ratio:
                s2 = proposal
                accepted += 1
            beta, _ = rmvnorm_prec(
                rng, self.B0inv_b0 + self.Xty / s2, self.B0inv + self.XtX / s2
            )
            out[it, : self.k] = beta
            out[it, -1] = s2
        return {
            "draws": out,
            "accept": None,
            "extras": {"sigma_accept": accepted / n_iter},
        }


# ---------------------------------------------------------------------
# linear index, pointwise log-likelihood, predictive draws
# ---------------------------------------------------------------------


def linear_index(model: Any, draws: np.ndarray, X: np.ndarray, offset: Any) -> Any:
    """``draws x observations`` matrix of the linear index."""
    name = model.name
    if name == "mlogit":
        raise MethodIncompatibility(
            "A multinomial logit has one index per category; use "
            "predict(what='probabilities')."
        )
    if name == "oprobit":
        return np.asarray(draws[:, : model.k - 1] @ X[:, 1:].T)
    return np.asarray(draws[:, : model.k] @ X.T) + offset


def _scale(model: Any, draws: np.ndarray) -> np.ndarray:
    """Residual standard deviation (or ALD scale) by draw, as a column."""
    if model.name == "quantile":
        if model.fixed_scale is not None:
            return np.full((draws.shape[0], 1), float(model.fixed_scale))
        return draws[:, [-1]]
    return np.asarray(np.sqrt(draws[:, [getattr(model, "sigma2_index", -1)]]))


def _ordered_cuts(model: Any, draws: np.ndarray) -> np.ndarray:
    cuts = draws[:, model.k - 1 :]
    pad = np.full((draws.shape[0], 1), np.inf)
    return np.hstack([-pad, cuts, pad])


def _mlogit_eta(model: Any, draws: np.ndarray, X: np.ndarray) -> np.ndarray:
    coef = draws[:, : model.k].reshape(draws.shape[0], model.J - 1, model.kx)
    eta = np.einsum("sjk,nk->snj", coef, X)
    zero = np.zeros(eta.shape[:2] + (1,))
    return np.concatenate([zero, eta], axis=2)


def pointwise_log_lik(
    model: Any, draws: np.ndarray, y: np.ndarray, X: np.ndarray, offset: Any = 0.0
) -> np.ndarray:
    """Log density of each observation under each draw: ``draws x n``."""
    name = model.name
    y = np.asarray(y, dtype=float)
    if name == "mlogit":
        eta = _mlogit_eta(model, draws, X)
        own = np.take_along_axis(
            eta, y.astype(int)[None, :, None].repeat(eta.shape[0], axis=0), axis=2
        )[:, :, 0]
        return np.asarray(own - special.logsumexp(eta, axis=2))
    eta = linear_index(model, draws, X, offset)
    if name == "oprobit":
        cuts = _ordered_cuts(model, draws)
        yi = y.astype(int)
        lo = cuts[:, yi] - eta
        hi = cuts[:, yi + 1] - eta
        return np.asarray(M._log_diff_ndtr(lo, hi))
    if name in ("normal", "conjugate"):
        s = _scale(model, draws)
        return np.asarray(-0.5 * _LOG_2PI - np.log(s) - 0.5 * ((y - eta) / s) ** 2)
    if name == "t":
        s = _scale(model, draws)
        return np.asarray(stats.t.logpdf((y - eta) / s, model.df) - np.log(s))
    if name == "tobit":
        s = _scale(model, draws)
        out = -0.5 * _LOG_2PI - np.log(s) - 0.5 * ((y - eta) / s) ** 2
        left = y <= model.lower
        right = y >= model.upper
        if left.any():
            out[:, left] = special.log_ndtr((model.lower - eta[:, left]) / s)
        if right.any():
            out[:, right] = special.log_ndtr((eta[:, right] - model.upper) / s)
        return np.asarray(out)
    if name == "quantile":
        s = _scale(model, draws)
        e = (y - eta) / s
        p = model.p
        return np.asarray(np.log(p * (1.0 - p)) - np.log(s) - e * (p - (e < 0)))
    if name == "probit":
        return np.asarray(special.log_ndtr(np.where(y > 0.5, eta, -eta)))
    if name == "logit":
        return np.asarray(y * eta - np.logaddexp(0.0, eta))
    if name == "poisson":
        with np.errstate(over="ignore"):
            return np.asarray(y * eta - np.exp(eta) - special.gammaln(y + 1.0))
    if name == "negbin":
        r = 1.0 / draws[:, [-1]]
        with np.errstate(over="ignore"):
            mu = np.exp(eta)
        return np.asarray(
            special.gammaln(y + r)
            - special.gammaln(r)
            - special.gammaln(y + 1.0)
            + r * np.log(r)
            + y * eta
            - (y + r) * np.log(r + mu)
        )
    raise MethodIncompatibility(f"model='{name}' has no pointwise log-likelihood here.")


def predictive_draws(
    model: Any,
    draws: np.ndarray,
    X: np.ndarray,
    rng: np.random.Generator,
    offset: Any = 0.0,
) -> np.ndarray:
    """One outcome per draw and observation from the predictive
    distribution: ``draws x n``."""
    name = model.name
    n_draws, n = draws.shape[0], X.shape[0]
    if name == "mlogit":
        prob = special.softmax(_mlogit_eta(model, draws, X), axis=2)
        u = rng.uniform(size=(n_draws, n, 1))
        return np.asarray((u > np.cumsum(prob, axis=2)).sum(axis=2), dtype=float)
    eta = linear_index(model, draws, X, offset)
    if name == "oprobit":
        latent = eta + rng.standard_normal((n_draws, n))
        cuts = draws[:, model.k - 1 :]
        return np.asarray(
            (latent[:, :, None] > cuts[:, None, :]).sum(axis=2), dtype=float
        )
    if name in ("normal", "conjugate"):
        return np.asarray(
            eta + _scale(model, draws) * rng.standard_normal((n_draws, n))
        )
    if name == "t":
        noise = rng.standard_t(model.df, size=(n_draws, n))
        return np.asarray(eta + _scale(model, draws) * noise)
    if name == "tobit":
        latent = eta + _scale(model, draws) * rng.standard_normal((n_draws, n))
        return np.asarray(np.clip(latent, model.lower, model.upper))
    if name == "quantile":
        z = rng.exponential(size=(n_draws, n))
        u = rng.standard_normal((n_draws, n))
        mix = model.theta * z + np.sqrt(model.tau2 * z) * u
        return np.asarray(eta + _scale(model, draws) * mix)
    if name == "probit":
        return np.asarray(rng.uniform(size=(n_draws, n)) < special.ndtr(eta), float)
    if name == "logit":
        return np.asarray(rng.uniform(size=(n_draws, n)) < special.expit(eta), float)
    with np.errstate(over="ignore"):
        mu = np.exp(eta)
    if name == "poisson":
        return np.asarray(rng.poisson(mu), dtype=float)
    if name == "negbin":
        r = np.broadcast_to(1.0 / draws[:, [-1]], mu.shape)
        return np.asarray(rng.poisson(rng.gamma(r, mu / r)), dtype=float)
    raise MethodIncompatibility(f"model='{name}' has no predictive distribution here.")


def residual_variance(
    model: Any, draws: np.ndarray, mu: np.ndarray
) -> Optional[np.ndarray]:
    """Expected residual variance under the model, by draw.

    ``None`` for the models whose outcome scale has no variance to
    explain (ordered and multinomial outcomes, quantile regression,
    censored outcomes).
    """
    name = model.name
    if name in ("normal", "conjugate"):
        return np.asarray(draws[:, getattr(model, "sigma2_index", -1)])
    if name == "t":
        if model.df <= 2:
            return None
        return np.asarray(draws[:, -1] * model.df / (model.df - 2.0))
    if name in ("logit", "probit"):
        return np.asarray((mu * (1.0 - mu)).mean(axis=1))
    if name == "poisson":
        return np.asarray(mu.mean(axis=1))
    if name == "negbin":
        return np.asarray((mu + draws[:, [-1]] * mu**2).mean(axis=1))
    return None
