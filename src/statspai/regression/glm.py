"""
Generalized Linear Model (GLM) implementation with IRLS estimation

Supports binomial, poisson, gamma, gaussian, inverse_gaussian, and negative_binomial
families with flexible link functions, robust/clustered standard errors, and
marginal effects computation.

References
----------
- McCullagh, P. and Nelder, J.A. (1989). Generalized Linear Models, 2nd ed.
- Hardin, J.W. and Hilbe, J.M. (2007). Generalized Linear Models and Extensions.
- Cameron, A.C. and Trivedi, P.K. (2005). Microeconometrics: Methods and Applications. [@mccullagh1989generalized]
"""

from typing import Optional, Union, Dict, Any, List, Callable, Tuple
import pandas as pd
import numpy as np
from scipy import stats, optimize, special
import warnings

from ..core.base import BaseModel, BaseEstimator
from ..core.results import EconometricResults
from ..core.utils import parse_formula, create_design_matrices, prepare_data


# ---------------------------------------------------------------------------
# Link functions
# ---------------------------------------------------------------------------

class LinkFunction:
    """Base class for GLM link functions."""

    def link(self, mu: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, eta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def deriv(self, mu: np.ndarray) -> np.ndarray:
        """d eta / d mu"""
        raise NotImplementedError


class IdentityLink(LinkFunction):
    name = "identity"

    def link(self, mu):
        return mu

    def inverse(self, eta):
        return eta

    def deriv(self, mu):
        return np.ones_like(mu)


class LogLink(LinkFunction):
    name = "log"

    def link(self, mu):
        return np.log(np.clip(mu, 1e-20, None))

    def inverse(self, eta):
        return np.exp(np.clip(eta, -500, 500))

    def deriv(self, mu):
        return 1.0 / np.clip(mu, 1e-20, None)


class LogitLink(LinkFunction):
    name = "logit"

    def link(self, mu):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return np.log(mu / (1 - mu))

    def inverse(self, eta):
        eta = np.clip(eta, -500, 500)
        return 1.0 / (1.0 + np.exp(-eta))

    def deriv(self, mu):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return 1.0 / (mu * (1 - mu))


class ProbitLink(LinkFunction):
    name = "probit"

    def link(self, mu):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return stats.norm.ppf(mu)

    def inverse(self, eta):
        return stats.norm.cdf(eta)

    def deriv(self, mu):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return 1.0 / stats.norm.pdf(stats.norm.ppf(mu))


class InverseLink(LinkFunction):
    name = "inverse"

    def link(self, mu):
        return 1.0 / np.clip(mu, 1e-20, None)

    def inverse(self, eta):
        return 1.0 / np.clip(eta, 1e-20, None)

    def deriv(self, mu):
        return -1.0 / np.clip(mu ** 2, 1e-40, None)


class CLogLogLink(LinkFunction):
    name = "cloglog"

    def link(self, mu):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return np.log(-np.log(1 - mu))

    def inverse(self, eta):
        eta = np.clip(eta, -500, 500)
        return 1.0 - np.exp(-np.exp(eta))

    def deriv(self, mu):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return 1.0 / ((1 - mu) * (-np.log(1 - mu)))


class PowerLink(LinkFunction):
    name = "power"

    def __init__(self, power: float = 1.0):
        self.power = power

    def link(self, mu):
        if self.power == 0:
            return np.log(np.clip(mu, 1e-20, None))
        return np.clip(mu, 1e-20, None) ** self.power

    def inverse(self, eta):
        if self.power == 0:
            return np.exp(np.clip(eta, -500, 500))
        return np.clip(eta, 1e-20, None) ** (1.0 / self.power)

    def deriv(self, mu):
        if self.power == 0:
            return 1.0 / np.clip(mu, 1e-20, None)
        return self.power * np.clip(mu, 1e-20, None) ** (self.power - 1)


class SqrtLink(LinkFunction):
    name = "sqrt"

    def link(self, mu):
        return np.sqrt(np.clip(mu, 0, None))

    def inverse(self, eta):
        return np.clip(eta, 0, None) ** 2

    def deriv(self, mu):
        return 0.5 / np.sqrt(np.clip(mu, 1e-20, None))


LINK_FUNCTIONS = {
    "identity": IdentityLink,
    "log": LogLink,
    "logit": LogitLink,
    "probit": ProbitLink,
    "inverse": InverseLink,
    "cloglog": CLogLogLink,
    "power": PowerLink,
    "sqrt": SqrtLink,
}


# ---------------------------------------------------------------------------
# Family distributions
# ---------------------------------------------------------------------------

class Family:
    """Base class for exponential family distributions."""

    name: str = "family"
    canonical_link: str = "identity"

    def variance(self, mu: np.ndarray) -> np.ndarray:
        """Variance function V(mu)."""
        raise NotImplementedError

    def deviance_residuals(self, y: np.ndarray, mu: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Unit deviance d(y, mu)."""
        raise NotImplementedError

    def deviance(self, y: np.ndarray, mu: np.ndarray, weights: np.ndarray) -> float:
        """Total deviance = sum(w * d(y, mu))."""
        d = self.deviance_residuals(y, mu, weights)
        return np.sum(weights * d)

    def log_likelihood(self, y: np.ndarray, mu: np.ndarray, weights: np.ndarray,
                       scale: float) -> float:
        """Log-likelihood (may be overridden for exact form)."""
        raise NotImplementedError

    def initialize_mu(self, y: np.ndarray) -> np.ndarray:
        """Starting values for mu."""
        return y.copy()

    def dispersion(self, y: np.ndarray, mu: np.ndarray, weights: np.ndarray,
                   df_resid: int) -> float:
        """Estimate dispersion (phi). 1.0 for binomial/poisson."""
        return self.deviance(y, mu, weights) / df_resid


class Gaussian(Family):
    name = "gaussian"
    canonical_link = "identity"

    def variance(self, mu):
        return np.ones_like(mu)

    def deviance_residuals(self, y, mu, weights):
        return (y - mu) ** 2

    def log_likelihood(self, y, mu, weights, scale):
        n = len(y)
        return -0.5 * (n * np.log(2 * np.pi * scale) + np.sum(weights * (y - mu) ** 2) / scale)


class Binomial(Family):
    name = "binomial"
    canonical_link = "logit"

    def variance(self, mu):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return mu * (1 - mu)

    def deviance_residuals(self, y, mu, weights):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        # Use safe log to avoid 0*log(0) warnings
        y_safe = np.clip(y, 1e-15, None)
        omy_safe = np.clip(1 - y, 1e-15, None)
        d = np.where(y > 0, 2 * y * np.log(y_safe / mu), 0.0) + \
            np.where(y < 1, 2 * (1 - y) * np.log(omy_safe / (1 - mu)), 0.0)
        return d

    def log_likelihood(self, y, mu, weights, scale):
        mu = np.clip(mu, 1e-15, 1 - 1e-15)
        return np.sum(weights * (y * np.log(mu) + (1 - y) * np.log(1 - mu)))

    def initialize_mu(self, y):
        return np.clip((y + 0.5) / 2.0, 0.01, 0.99)

    def dispersion(self, y, mu, weights, df_resid):
        return 1.0


class Poisson(Family):
    name = "poisson"
    canonical_link = "log"

    def variance(self, mu):
        return np.clip(mu, 1e-20, None)

    def deviance_residuals(self, y, mu, weights):
        mu = np.clip(mu, 1e-20, None)
        y_safe = np.clip(y, 1e-20, None)
        d = np.where(y > 0, 2 * (y * np.log(y_safe / mu) - (y - mu)), 2 * mu)
        return d

    def log_likelihood(self, y, mu, weights, scale):
        mu = np.clip(mu, 1e-20, None)
        return np.sum(weights * (y * np.log(mu) - mu - special.gammaln(y + 1)))

    def initialize_mu(self, y):
        return np.clip(y, 0.1, None)

    def dispersion(self, y, mu, weights, df_resid):
        return 1.0


class Gamma(Family):
    name = "gamma"
    canonical_link = "inverse"

    def variance(self, mu):
        return np.clip(mu, 1e-20, None) ** 2

    def deviance_residuals(self, y, mu, weights):
        mu = np.clip(mu, 1e-20, None)
        y = np.clip(y, 1e-20, None)
        return 2 * (-np.log(y / mu) + (y - mu) / mu)

    def log_likelihood(self, y, mu, weights, scale):
        mu = np.clip(mu, 1e-20, None)
        y = np.clip(y, 1e-20, None)
        nu = 1.0 / scale  # shape parameter
        return np.sum(weights * (
            nu * np.log(nu) - special.gammaln(nu)
            + (nu - 1) * np.log(y) - nu * y / mu - nu * np.log(mu)
        ))

    def initialize_mu(self, y):
        return np.clip(y, 0.1, None)


class InverseGaussian(Family):
    name = "inverse_gaussian"
    canonical_link = "inverse"

    def variance(self, mu):
        return np.clip(mu, 1e-20, None) ** 3

    def deviance_residuals(self, y, mu, weights):
        mu = np.clip(mu, 1e-20, None)
        y = np.clip(y, 1e-20, None)
        return (y - mu) ** 2 / (y * mu ** 2)

    def log_likelihood(self, y, mu, weights, scale):
        mu = np.clip(mu, 1e-20, None)
        y = np.clip(y, 1e-20, None)
        lam = 1.0 / scale
        return np.sum(weights * (
            0.5 * np.log(lam / (2 * np.pi * y ** 3))
            - lam * (y - mu) ** 2 / (2 * mu ** 2 * y)
        ))

    def initialize_mu(self, y):
        return np.clip(y, 0.1, None)


class NegativeBinomial(Family):
    """
    Negative Binomial (NB2) family.

    Variance = mu + alpha * mu^2 where alpha is the overdispersion parameter.
    Alpha is estimated via MLE as part of the fitting procedure.
    """
    name = "negative_binomial"
    canonical_link = "log"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def variance(self, mu):
        return np.clip(mu, 1e-20, None) + self.alpha * np.clip(mu, 1e-20, None) ** 2

    def deviance_residuals(self, y, mu, weights):
        mu = np.clip(mu, 1e-20, None)
        y_safe = np.clip(y, 1e-20, None)
        alpha = self.alpha
        if alpha < 1e-20:
            # Degenerate to Poisson
            return np.where(y > 0, 2 * (y * np.log(y_safe / mu) - (y - mu)), 2 * mu)
        inv_alpha = 1.0 / alpha
        d = 2 * (
            np.where(y > 0, y * np.log(y_safe / mu), 0.0)
            - (y + inv_alpha) * np.log((1 + alpha * y) / (1 + alpha * mu))
        )
        return d

    def log_likelihood(self, y, mu, weights, scale):
        mu = np.clip(mu, 1e-20, None)
        alpha = self.alpha
        inv_alpha = 1.0 / alpha
        ll = (
            special.gammaln(y + inv_alpha) - special.gammaln(inv_alpha) - special.gammaln(y + 1)
            + inv_alpha * np.log(inv_alpha / (inv_alpha + mu))
            + y * np.log(mu / (inv_alpha + mu))
        )
        return np.sum(weights * ll)

    def initialize_mu(self, y):
        return np.clip(y, 0.1, None)

    def dispersion(self, y, mu, weights, df_resid):
        return 1.0


FAMILIES = {
    "gaussian": Gaussian,
    "binomial": Binomial,
    "poisson": Poisson,
    "gamma": Gamma,
    "inverse_gaussian": InverseGaussian,
    "negative_binomial": NegativeBinomial,
}


def _get_family(family: str) -> Family:
    key = family.lower().replace("-", "_").replace(" ", "_")
    if key not in FAMILIES:
        raise ValueError(
            f"Unknown family '{family}'. Choose from: {', '.join(FAMILIES.keys())}"
        )
    return FAMILIES[key]()


def _get_link(link: Optional[str], family: Family) -> LinkFunction:
    if link is None:
        link = family.canonical_link
    key = link.lower()
    if key not in LINK_FUNCTIONS:
        raise ValueError(
            f"Unknown link function '{link}'. Choose from: {', '.join(LINK_FUNCTIONS.keys())}"
        )
    return LINK_FUNCTIONS[key]()


# ---------------------------------------------------------------------------
# GLM Estimator
# ---------------------------------------------------------------------------

class GLMEstimator(BaseEstimator):
    """
    Generalized Linear Model estimator using IRLS

    Implements Iteratively Reweighted Least Squares for maximum likelihood
    estimation of GLM parameters.
    """

    def estimate(
        self,
        y: np.ndarray,
        X: np.ndarray,
        family: Family,
        link: LinkFunction,
        robust: str = "nonrobust",
        cluster: Optional[pd.Series] = None,
        weights: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        maxiter: int = 100,
        tol: float = 1e-8,
        alpha_nb: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Estimate GLM parameters via IRLS.

        Parameters
        ----------
        y : np.ndarray
            Response variable (n,).
        X : np.ndarray
            Design matrix (n, k) including intercept if desired.
        family : Family
            Distribution family instance.
        link : LinkFunction
            Link function instance.
        robust : str
            Standard-error type ('nonrobust', 'hc0', 'hc1', 'hc2', 'hc3', 'hac').
        cluster : pd.Series, optional
            Cluster variable.
        weights : np.ndarray, optional
            Prior / frequency weights.
        offset : np.ndarray, optional
            Known offset added to the linear predictor.
        maxiter : int
            Maximum IRLS iterations.
        tol : float
            Convergence tolerance on deviance.
        alpha_nb : float, optional
            If family is NB, initial alpha for joint estimation.

        Returns
        -------
        Dict[str, Any]
        """
        n, k = X.shape
        if weights is None:
            weights = np.ones(n)
        if offset is None:
            offset = np.zeros(n)

        # --- initialise mu and eta ---
        mu = family.initialize_mu(y)
        eta = link.link(mu) + offset

        dev_old = np.inf
        params_old = None
        converged = False

        for iteration in range(maxiter):
            # Variance and link derivative
            V = family.variance(mu)
            g_prime = link.deriv(mu)  # d eta / d mu

            # Working weight and working response (IRLS)
            W = weights / (V * g_prime ** 2)
            W = np.clip(W, 1e-20, 1e20)
            z = (eta - offset) + (y - mu) * g_prime  # working / pseudo response

            # Weighted least squares: solve (X' W X) beta = X' W z
            sqrt_W = np.sqrt(W)
            Xw = X * sqrt_W[:, np.newaxis]
            zw = z * sqrt_W

            try:
                # QR decomposition for numerical stability
                Q, R = np.linalg.qr(Xw, mode="reduced")
                params = np.linalg.solve(R, Q.T @ zw)
            except np.linalg.LinAlgError:
                warnings.warn("Singular matrix in IRLS; using pseudo-inverse")
                params = np.linalg.lstsq(Xw, zw, rcond=None)[0]

            # Update eta, mu
            eta = X @ params + offset
            mu = link.inverse(eta)

            # Bound mu for safety (family-dependent)
            if isinstance(family, Binomial):
                mu = np.clip(mu, 1e-15, 1 - 1e-15)
            elif isinstance(family, Gaussian):
                pass  # Gaussian mu can be any real number
            else:
                # Poisson, Gamma, InverseGaussian, NB: mu must be positive
                mu = np.clip(mu, 1e-15, 1e15)

            dev_new = family.deviance(y, mu, weights)

            # Check convergence: deviance criterion or parameter criterion
            if np.isfinite(dev_old):
                if np.abs(dev_new - dev_old) / (np.abs(dev_old) + 0.1) < tol:
                    converged = True
                    break
            elif params_old is not None:
                # Fallback: parameter convergence (useful when first deviance is inf)
                param_change = np.max(np.abs(params - params_old)) / (np.max(np.abs(params)) + 1e-10)
                if param_change < tol:
                    converged = True
                    break

            dev_old = dev_new
            params_old = params.copy()

            # --- NB2: update alpha via MLE profile ---
            if isinstance(family, NegativeBinomial) and iteration > 0 and iteration % 5 == 0:
                family.alpha = self._estimate_nb_alpha(y, mu, weights, family.alpha)

        if not converged:
            warnings.warn(
                f"IRLS did not converge after {maxiter} iterations. "
                f"Last deviance change: {np.abs(dev_new - dev_old):.2e}"
            )

        # Final NB alpha update
        if isinstance(family, NegativeBinomial):
            family.alpha = self._estimate_nb_alpha(y, mu, weights, family.alpha)

        # ----------------------------------------------------------------
        # Variance-covariance matrix
        # ----------------------------------------------------------------
        V = family.variance(mu)
        g_prime = link.deriv(mu)
        W = weights / (V * g_prime ** 2)
        W = np.clip(W, 1e-20, 1e20)

        # (X' W X)^{-1}
        XtWX = X.T @ (X * W[:, np.newaxis])
        try:
            XtWX_inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError:
            XtWX_inv = np.linalg.pinv(XtWX)
            warnings.warn("Singular X'WX, using pseudo-inverse for covariance")

        # Pearson residuals
        pearson_resid = (y - mu) / np.sqrt(V)

        # Scale (dispersion)
        df_resid = n - k
        phi = family.dispersion(y, mu, weights, df_resid)

        if cluster is not None:
            var_cov = self._cluster_cov(X, y, mu, V, g_prime, weights, XtWX_inv, cluster, n, k)
        elif robust == "nonrobust":
            var_cov = phi * XtWX_inv
        elif robust.lower() in ("hc0", "hc1", "hc2", "hc3"):
            var_cov = self._robust_cov(X, y, mu, V, g_prime, weights, XtWX_inv, robust.lower(), n, k)
        elif robust.lower() == "hac":
            lags = kwargs.get("lags", None)
            var_cov = self._hac_cov(X, y, mu, V, g_prime, weights, XtWX_inv, n, k, lags)
        else:
            raise ValueError(f"Unknown robust option: {robust}")

        std_errors = np.sqrt(np.diag(var_cov))

        # Fitted values & residuals
        fitted_values = mu
        residuals = y - mu

        # Deviance residuals (signed)
        unit_dev = family.deviance_residuals(y, mu, weights)
        deviance_resid = np.sign(y - mu) * np.sqrt(np.clip(unit_dev, 0, None))

        # Deviance, log-likelihood, information criteria
        deviance = family.deviance(y, mu, weights)
        ll = family.log_likelihood(y, mu, weights, phi)
        null_mu = np.full(n, np.average(y, weights=weights))
        if isinstance(family, Binomial):
            null_mu = np.clip(null_mu, 1e-15, 1 - 1e-15)
        null_dev = family.deviance(y, null_mu, weights)
        ll_null = family.log_likelihood(y, null_mu, weights, phi)

        aic = -2 * ll + 2 * k
        bic = -2 * ll + np.log(n) * k
        pseudo_r2 = 1.0 - deviance / null_dev if null_dev > 0 else np.nan
        pearson_chi2 = np.sum(weights * pearson_resid ** 2)

        return {
            "params": params,
            "std_errors": std_errors,
            "var_cov": var_cov,
            "fitted_values": fitted_values,
            "residuals": residuals,
            "pearson_residuals": pearson_resid,
            "deviance_residuals": deviance_resid,
            "nobs": n,
            "df_model": k - 1,
            "df_resid": df_resid,
            "deviance": deviance,
            "null_deviance": null_dev,
            "pearson_chi2": pearson_chi2,
            "log_likelihood": ll,
            "log_likelihood_null": ll_null,
            "aic": aic,
            "bic": bic,
            "pseudo_r2": pseudo_r2,
            "dispersion": phi,
            "converged": converged,
            "n_iter": iteration + 1,
        }

    # ------------------------------------------------------------------
    # Robust / clustered covariance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_obs(X, y, mu, V, g_prime, weights):
        """Per-observation score contributions (n, k)."""
        r = (y - mu) / V
        w = weights / g_prime
        return X * (r * w)[:, np.newaxis]

    def _robust_cov(self, X, y, mu, V, g_prime, weights, bread, robust_type, n, k):
        """Sandwich HC0-HC3 covariance."""
        S = self._score_obs(X, y, mu, V, g_prime, weights)

        if robust_type == "hc0":
            meat = S.T @ S
        elif robust_type == "hc1":
            meat = (n / (n - k)) * S.T @ S
        elif robust_type == "hc2":
            XtWX_inv = bread
            W_diag = weights / (V * g_prime ** 2)
            h = np.sum((X * W_diag[:, np.newaxis]) * (X @ XtWX_inv), axis=1)
            S_adj = S / np.sqrt(np.clip(1 - h, 1e-10, None))[:, np.newaxis]
            meat = S_adj.T @ S_adj
        elif robust_type == "hc3":
            XtWX_inv = bread
            W_diag = weights / (V * g_prime ** 2)
            h = np.sum((X * W_diag[:, np.newaxis]) * (X @ XtWX_inv), axis=1)
            S_adj = S / np.clip(1 - h, 1e-10, None)[:, np.newaxis]
            meat = S_adj.T @ S_adj
        else:
            raise ValueError(f"Unknown HC type: {robust_type}")

        return bread @ meat @ bread

    def _cluster_cov(self, X, y, mu, V, g_prime, weights, bread, cluster, n, k):
        """Clustered (sandwich) covariance."""
        S = self._score_obs(X, y, mu, V, g_prime, weights)
        cluster_arr = np.asarray(cluster)
        clusters = np.unique(cluster_arr)
        n_clusters = len(clusters)

        meat = np.zeros((k, k))
        for cid in clusters:
            idx = cluster_arr == cid
            s_c = S[idx].sum(axis=0)
            meat += np.outer(s_c, s_c)

        correction = n_clusters / (n_clusters - 1) * (n - 1) / (n - k)
        return correction * bread @ meat @ bread

    def _hac_cov(self, X, y, mu, V, g_prime, weights, bread, n, k, lags=None):
        """Newey-West HAC covariance."""
        S = self._score_obs(X, y, mu, V, g_prime, weights)
        if lags is None:
            lags = int(np.floor(4 * (n / 100) ** (2 / 9)))

        gamma_0 = S.T @ S / n
        gamma_sum = gamma_0.copy()
        for j in range(1, lags + 1):
            gamma_j = S[j:].T @ S[:-j] / n
            w = 1 - j / (lags + 1)
            gamma_sum += w * (gamma_j + gamma_j.T)

        return bread @ gamma_sum @ bread

    # ------------------------------------------------------------------
    # NB alpha estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_nb_alpha(y, mu, weights, alpha_init):
        """Profile MLE for NB2 overdispersion parameter alpha."""
        def neg_ll(log_alpha):
            alpha = np.exp(log_alpha)
            inv_a = 1.0 / alpha
            ll = np.sum(weights * (
                special.gammaln(y + inv_a) - special.gammaln(inv_a) - special.gammaln(y + 1)
                + inv_a * np.log(inv_a / (inv_a + mu))
                + y * np.log(mu / (inv_a + mu))
            ))
            return -ll

        try:
            res = optimize.minimize_scalar(
                neg_ll,
                bounds=(np.log(1e-6), np.log(1e4)),
                method="bounded",
            )
            return np.exp(res.x)
        except Exception:
            return alpha_init


# ---------------------------------------------------------------------------
# GLM Model class
# ---------------------------------------------------------------------------

class GLMRegression(BaseModel):
    """
    Generalized Linear Model with IRLS estimation.

    Parameters
    ----------
    formula : str, optional
        Model formula (e.g. ``"y ~ x1 + x2"``).
    data : pd.DataFrame, optional
        Data frame containing the variables.
    y : np.ndarray, optional
        Response array (alternative to formula).
    X : np.ndarray, optional
        Design matrix (alternative to formula).
    var_names : list of str, optional
        Variable names when using ``y``/``X`` directly.
    family : str
        Distribution family.
    link : str or None
        Link function (``None`` selects canonical link).
    """

    def __init__(
        self,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        y: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        var_names: Optional[List[str]] = None,
        family: str = "gaussian",
        link: Optional[str] = None,
    ):
        super().__init__()
        self.formula = formula
        self.data = data
        self.y = y
        self.X = X
        self.var_names = var_names
        self.family_name = family
        self.link_name = link
        self.family = _get_family(family)
        self.link = _get_link(link, self.family)
        self.estimator = GLMEstimator()

    def fit(
        self,
        robust: str = "nonrobust",
        cluster: Optional[str] = None,
        weights: Optional[str] = None,
        offset: Optional[str] = None,
        exposure: Optional[str] = None,
        maxiter: int = 100,
        tol: float = 1e-8,
        alpha: float = 0.05,
        **kwargs,
    ) -> EconometricResults:
        """
        Fit the GLM.

        Parameters
        ----------
        robust : str
            Standard-error type.
        cluster : str, optional
            Cluster variable name.
        weights : str, optional
            Weight variable name.
        offset : str, optional
            Offset variable name.
        exposure : str, optional
            Exposure variable name (log added as offset).
        maxiter : int
            Maximum IRLS iterations.
        tol : float
            Convergence tolerance.
        alpha : float
            Significance level for confidence intervals.

        Returns
        -------
        EconometricResults
        """
        # --- prepare design matrices ---
        if self.formula is not None and self.data is not None:
            y_df, X_df = create_design_matrices(self.formula, self.data)
            self.y = y_df.values.ravel()
            self.X = X_df.values
            self.var_names = list(X_df.columns)
            self.dependent_var = y_df.columns[0]
        elif self.y is not None and self.X is not None:
            if self.var_names is None:
                self.var_names = [f"x{i}" for i in range(self.X.shape[1])]
            self.dependent_var = "y"
        else:
            raise ValueError("Must provide either (formula, data) or (y, X)")

        n = len(self.y)

        # Weights, offset, exposure
        w = None
        if weights and self.data is not None:
            w = self.data[weights].values.astype(float)

        off = np.zeros(n)
        if offset and self.data is not None:
            off = self.data[offset].values.astype(float)
        if exposure and self.data is not None:
            off = off + np.log(self.data[exposure].values.astype(float))

        cluster_var = None
        if cluster and self.data is not None:
            cluster_var = self.data[cluster]

        # --- estimate ---
        results = self.estimator.estimate(
            y=self.y,
            X=self.X,
            family=self.family,
            link=self.link,
            robust=robust,
            cluster=cluster_var,
            weights=w,
            offset=off,
            maxiter=maxiter,
            tol=tol,
            **kwargs,
        )

        # --- marginal effects (AME) ---
        marginal_effects = self._compute_marginal_effects(
            results["params"], self.X, self.link, self.family
        )
        me_series = pd.Series(marginal_effects, index=self.var_names)

        # --- build EconometricResults ---
        params = pd.Series(results["params"], index=self.var_names)
        std_errors = pd.Series(results["std_errors"], index=self.var_names)

        se_label = robust if robust != "nonrobust" else ("cluster" if cluster else "nonrobust")

        model_info = {
            "model_type": "GLM",
            "method": "IRLS (MLE)",
            "family": self.family.name,
            "link": self.link.name,
            "robust": se_label,
            "cluster": cluster,
            "converged": results["converged"],
            "n_iter": results["n_iter"],
        }

        data_info = {
            "nobs": results["nobs"],
            "df_model": results["df_model"],
            "df_resid": results["df_resid"],
            "dependent_var": self.dependent_var,
            "fitted_values": results["fitted_values"],
            "residuals": results["residuals"],
            "pearson_residuals": results["pearson_residuals"],
            "deviance_residuals": results["deviance_residuals"],
            "X": self.X,
            "y": self.y,
            "var_cov": results["var_cov"],
            "var_names": self.var_names,
            "marginal_effects": me_series,
            "family_obj": self.family,
            "link_obj": self.link,
        }

        diagnostics = {
            "Deviance": results["deviance"],
            "Null deviance": results["null_deviance"],
            "Pearson chi2": results["pearson_chi2"],
            "Dispersion": results["dispersion"],
            "Log-Likelihood": results["log_likelihood"],
            "Log-Likelihood (null)": results["log_likelihood_null"],
            "AIC": results["aic"],
            "BIC": results["bic"],
            "Pseudo R-squared": results["pseudo_r2"],
        }

        if isinstance(self.family, NegativeBinomial):
            diagnostics["NB alpha"] = self.family.alpha

        self._results = EconometricResults(
            params=params,
            std_errors=std_errors,
            model_info=model_info,
            data_info=data_info,
            diagnostics=diagnostics,
        )

        self.is_fitted = True
        return self._results

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        type: str = "response",
        offset: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate predictions from the fitted model.

        Parameters
        ----------
        data : pd.DataFrame, optional
            New data for prediction. Uses training data if ``None``.
        type : str
            ``'response'`` (mean), ``'link'`` (linear predictor), or
            ``'variance'`` (variance function evaluated at predicted mu).
        offset : np.ndarray, optional
            Offset for new data.

        Returns
        -------
        np.ndarray
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if data is not None:
            _, X_new = create_design_matrices(self.formula, data)
            X_pred = X_new.values
        else:
            X_pred = self.X

        eta = X_pred @ self._results.params.values
        if offset is not None:
            eta = eta + offset

        if type == "link":
            return eta
        mu = self.link.inverse(eta)
        if type == "response":
            return mu
        if type == "variance":
            return self.family.variance(mu)
        raise ValueError(f"Unknown prediction type '{type}'. Choose 'response', 'link', or 'variance'.")

    # ------------------------------------------------------------------
    # Marginal effects
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_marginal_effects(
        params: np.ndarray,
        X: np.ndarray,
        link: LinkFunction,
        family: Family,
    ) -> np.ndarray:
        """
        Average Marginal Effects (AME).

        For each covariate j:  AME_j = (1/n) * sum_i [ d mu / d x_j ]_i
        where  d mu / d x_j = beta_j / g'(mu_i)
        """
        eta = X @ params
        mu = link.inverse(eta)
        g_prime = link.deriv(mu)  # d eta / d mu
        # d mu / d eta = 1 / g'(mu)
        dmu_deta = 1.0 / g_prime
        # AME_j = mean( beta_j * dmu_deta )
        ame = params * np.mean(dmu_deta)
        return ame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def glm(
    formula: str = None,
    data: pd.DataFrame = None,
    y: str = None,
    x: list = None,
    family: str = "gaussian",
    link: str = None,
    robust: str = "nonrobust",
    cluster: str = None,
    weights: str = None,
    offset: str = None,
    exposure: str = None,
    maxiter: int = 100,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Fit a Generalized Linear Model.

    Provides a Stata-like interface for GLM estimation with IRLS,
    supporting robust and clustered standard errors.

    Parameters
    ----------
    formula : str, optional
        Model formula (e.g. ``"y ~ x1 + x2"``).
    data : pd.DataFrame, optional
        Data frame containing the variables.
    y : str, optional
        Name of the dependent variable (alternative to formula).
    x : list of str, optional
        Names of independent variables (alternative to formula).
    family : str, default ``"gaussian"``
        Distribution family. One of ``"gaussian"``, ``"binomial"``,
        ``"poisson"``, ``"gamma"``, ``"inverse_gaussian"``,
        ``"negative_binomial"``.
    link : str or None
        Link function. If ``None`` the canonical link for the chosen
        family is used. Options: ``"identity"``, ``"log"``, ``"logit"``,
        ``"probit"``, ``"inverse"``, ``"cloglog"``, ``"power"``,
        ``"sqrt"``.
    robust : str, default ``"nonrobust"``
        Standard-error type (``"nonrobust"``, ``"hc0"``-``"hc3"``,
        ``"hac"``).
    cluster : str, optional
        Variable name for clustered standard errors.
    weights : str, optional
        Variable name for observation weights.
    offset : str, optional
        Variable name for offset.
    exposure : str, optional
        Variable name for exposure (``log(exposure)`` is added as offset).
    maxiter : int, default 100
        Maximum number of IRLS iterations.
    tol : float, default 1e-8
        Convergence tolerance on the relative change in deviance.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
        Fitted model results with coefficients, standard errors,
        diagnostics, and marginal effects.

    Examples
    --------
    Logistic regression:

    >>> results = glm("admit ~ gre + gpa + rank", data=df, family="binomial")
    >>> print(results.summary())

    Poisson with exposure and robust SE:

    >>> results = glm("counts ~ x1 + x2", data=df, family="poisson",
    ...               exposure="pop", robust="hc1")

    Negative binomial:

    >>> results = glm("y ~ x1 + x2", data=df, family="negative_binomial")

    Gamma with log link and clustered SE:

    >>> results = glm("cost ~ age + severity", data=df,
    ...               family="gamma", link="log", cluster="hospital_id")
    """
    # Handle y/x style specification
    if formula is None and y is not None and x is not None and data is not None:
        formula = f"{y} ~ {' + '.join(x)}"

    if formula is None or data is None:
        raise ValueError("Must provide (formula, data) or (y, x, data)")

    model = GLMRegression(formula=formula, data=data, family=family, link=link)
    return model.fit(
        robust=robust,
        cluster=cluster,
        weights=weights,
        offset=offset,
        exposure=exposure,
        maxiter=maxiter,
        tol=tol,
        alpha=alpha,
    )
