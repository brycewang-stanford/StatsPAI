"""
Generalized linear mixed models (GLMM).

Estimation
----------
Two estimation paths are exposed via the ``nAGQ`` argument:

    nAGQ = 1   →  Laplace approximation (default; lme4's default).
                  Stata ``intmethod(laplace)``, ``glmer(nAGQ = 1)``,
                  ``glmmTMB``.
    nAGQ ≥ 2   →  Adaptive Gauss-Hermite quadrature (AGHQ) with ``nAGQ``
                  nodes centred at the conditional mode and scaled by the
                  curvature there: Stata ``intmethod(mcaghermite)
                  intpoints(k)``, ``glmer(nAGQ = k)``.  Stata's *default*
                  (``mvaghermite``, 7 points, nodes re-centred at the
                  posterior mean) is a different rule and is not
                  implemented.  Restricted to a single scalar random effect
                  (``q = 1``) -- the same restriction lme4 imposes.

The curvature of the log integrand at the mode is the observed
information by default (``curvature='observed'``; Stata, glmmTMB,
ordinal::clmm).  ``curvature='expected'`` uses the Fisher weights, as
lme4's PIRLS does; the two coincide for canonical links.

Families
--------
Five exponential / dispersion families are supported; each has its
canonical link plus the most common practical link reported by Stata
``meglm``:

    * ``gaussian``  identity link, residual variance σ² estimated by ML
      (``meglm`` Gaussian; Laplace is exact, so this is the ML LMM).
    * ``binomial``  logit link  (``melogit``).  Bernoulli or counts/trials.
    * ``poisson``   log link    (``mepoisson``).
    * ``gamma``     log link    — dispersion ``φ`` estimated by ML.
    * ``nbinomial`` log link    — NB-2 (mean-dispersion ``α`` estimated).

Ordinal-logit GLMM (``meologit``) lives in :mod:`._ordinal`; it shares
the result class but uses a dedicated fitter because of its threshold
parameters.

The integrated log-likelihood

    ℓ(β, θ, ψ) = Σ_j log ∫ f(y_j | β, ψ, u_j) φ(u_j; 0, G(θ)) du_j

has no closed form for non-Gaussian families.  Each integral is
approximated either by a Laplace expansion around the conditional mode
û_j (``nAGQ=1``) or by adaptive Gauss-Hermite quadrature recentred at
û_j with curvature H_j (``nAGQ>1``).  The optimum is found by L-BFGS-B
and finished with Newton steps on the numerical gradient and Hessian;
the fixed-effect covariance is the corresponding block of the inverse
numerical Hessian of the approximated marginal log-likelihood over all
parameters (β, covariance parameters, dispersion) -- Stata's ``vce(oim)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.optimize import minimize

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _glmm_ri as _ri
from ._core import (
    _group_blocks,
    _GroupBlock,
    _initial_theta,
    _n_cov_params,
    _prepare_frame,
    _unpack_G,
)


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise MethodIncompatibility(
            f"`{name}` must be a string.",
            diagnostics={name: repr(value)},
        )
    return value


def _require_dataframe(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise MethodIncompatibility(
            f"`{name}` must be a pandas DataFrame.",
            diagnostics={name: value.__class__.__name__},
        )
    return value


def _require_open_unit_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must be a number in (0, 1).",
            diagnostics={name: repr(value)},
        ) from exc
    if not np.isfinite(out) or not 0.0 < out < 1.0:
        raise MethodIncompatibility(
            f"`{name}` must be in (0, 1).",
            diagnostics={name: out},
        )
    return out


def _require_positive_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must be a positive finite number.",
            diagnostics={name: repr(value)},
        ) from exc
    if not np.isfinite(out) or out <= 0.0:
        raise MethodIncompatibility(
            f"`{name}` must be a positive finite number.",
            diagnostics={name: out},
        )
    return out


def _require_int_at_least(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise MethodIncompatibility(
            f"`{name}` must be an integer >= {minimum}.",
            diagnostics={name: repr(value), "minimum": minimum},
        )
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must be an integer >= {minimum}.",
            diagnostics={name: repr(value), "minimum": minimum},
        ) from exc
    if out != value or out < minimum:
        raise MethodIncompatibility(
            f"`{name}` must be an integer >= {minimum}.",
            diagnostics={name: repr(value), "minimum": minimum},
        )
    return out


def _coerce_column_list(value: Any, name: str) -> List[str]:
    if isinstance(value, str):
        cols = [value]
    else:
        try:
            cols = list(value)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"`{name}` must be a column name or list of column names.",
                diagnostics={name: repr(value)},
            ) from exc
    bad = [c for c in cols if not isinstance(c, str) or not c]
    if bad:
        raise MethodIncompatibility(
            f"`{name}` must contain only non-empty string column names.",
            diagnostics={name: cols, "invalid_columns": bad},
        )
    return cols


def _coerce_optional_column_list(value: Any, name: str) -> List[str]:
    if value is None:
        return []
    return _coerce_column_list(value, name)


# ---------------------------------------------------------------------------
# Exponential / dispersion families
# ---------------------------------------------------------------------------
#
# Each family advertises ``n_disp_params``: 0 for Gaussian/Binomial/Poisson
# (no extra parameter beyond β and G), 1 for Gamma (log φ) and NegBin (log α).
# The packed parameter is appended to θ after the covariance parameters and
# converted via ``parse_dispersion`` to its natural-scale value.
#
# The four hot-loop methods each take ``dispersion`` (None when n=0):
#
#     inv_link(eta)                              μ = g⁻¹(η)
#     irls_weight(mu, w, dispersion)             −∂²log f/∂η² (Fisher info)
#     score_eta(y, mu, w, dispersion)            ∂log f/∂η  per observation
#     log_lik(y, mu, w, dispersion)              Σ log f(y_i; μ_i, …)
#
# ``w`` carries either trial counts (binomial) or an observation weight
# (other families); for the latter it is conventionally a vector of ones.

_LOG_2PI = float(np.log(2.0 * np.pi))
_EPS = 1e-12


class _Family:
    """Base class — concrete families override the four hot-loop methods."""

    name: str
    link: str
    n_disp_params: int = 0

    # ---- dispersion plumbing -----------------------------------------------

    @classmethod
    def parse_dispersion(cls, packed: np.ndarray) -> Optional[float]:
        """Return the natural-scale dispersion from its packed value(s)."""
        if cls.n_disp_params == 0:
            return None
        return float(np.exp(packed[0]))

    @classmethod
    def initial_dispersion(cls) -> np.ndarray:
        """Sensible starting value for the packed dispersion vector."""
        if cls.n_disp_params == 0:
            return np.zeros(0)
        return np.zeros(cls.n_disp_params)  # log φ = 0 → φ = 1

    # ---- core hot-loop methods -- subclasses implement -----------------------

    @staticmethod
    def inv_link(eta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def irls_weight(
        mu: np.ndarray, w: np.ndarray, dispersion: Optional[float]
    ) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def score_eta(
        y: np.ndarray, mu: np.ndarray, w: np.ndarray, dispersion: Optional[float]
    ) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def obs_weight(
        cls,
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        """Observed information on η, ``−∂² log f / ∂η²`` per observation.

        Equal to the Fisher weight :meth:`irls_weight` for canonical links
        (Gaussian-identity, binomial-logit, Poisson-log); the non-canonical
        families (gamma-log, NB-2-log) override it.
        """
        return cls.irls_weight(mu, w, dispersion)

    @staticmethod
    def log_lik_vec(
        y: np.ndarray, mu: np.ndarray, w: np.ndarray, dispersion: Optional[float]
    ) -> np.ndarray:
        """Per-observation log density ``log f(y_i; μ_i, …)``."""
        raise NotImplementedError

    @classmethod
    def log_lik(
        cls,
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> float:
        return float(np.sum(cls.log_lik_vec(y, mu, w, dispersion)))


class _Gaussian(_Family):
    """
    Gaussian family, identity link (Stata ``meglm, family(gaussian)``).

    The residual variance ``σ²`` is estimated by ML jointly with (β, G)
    through the packed parameter ``log σ²`` and reported as
    ``var(Residual)``.  With an identity link the Laplace approximation
    is exact, so the fit is the ML linear mixed model (``sp.mixed(...,
    method='ml')``, ``lme4::lmer(REML = FALSE)``).
    """

    name = "gaussian"
    link = "identity"
    n_disp_params = 1

    @staticmethod
    def inv_link(eta: np.ndarray) -> np.ndarray:
        return np.asarray(eta, dtype=float)

    @staticmethod
    def irls_weight(
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        s2 = dispersion if dispersion is not None else 1.0
        return np.asarray((w / max(s2, _EPS)) * np.ones_like(mu), dtype=float)

    @staticmethod
    def score_eta(
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        s2 = dispersion if dispersion is not None else 1.0
        return np.asarray(w * (y - mu) / max(s2, _EPS), dtype=float)

    @staticmethod
    def log_lik_vec(
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        s2 = max(dispersion if dispersion is not None else 1.0, _EPS)
        return -0.5 * w * ((y - mu) ** 2 / s2 + _LOG_2PI + np.log(s2))


class _Binomial(_Family):
    name = "binomial"
    link = "logit"

    @staticmethod
    def inv_link(eta: np.ndarray) -> np.ndarray:
        out = np.empty_like(eta, dtype=float)
        pos = eta >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
        ex = np.exp(eta[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    @staticmethod
    def irls_weight(
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        return np.asarray(w * mu * (1.0 - mu), dtype=float)

    @staticmethod
    def score_eta(
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        return np.asarray(y - w * mu, dtype=float)

    @staticmethod
    def log_lik_vec(
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        mu = np.clip(mu, _EPS, 1.0 - _EPS)
        # Include the log-binomial-coefficient constant so the
        # Bernoulli (w=1) and trials-binomial (w>1) cases yield the full
        # log-likelihood — this matters for AIC comparability with other
        # families fit to the same y.  For w=1 the constant is 0 and the
        # expression reduces to the Bernoulli log-lik.
        log_coef = (
            special.gammaln(w + 1.0)
            - special.gammaln(y + 1.0)
            - special.gammaln(w - y + 1.0)
        )
        return log_coef + y * np.log(mu) + (w - y) * np.log(1.0 - mu)


class _Poisson(_Family):
    name = "poisson"
    link = "log"

    @staticmethod
    def inv_link(eta: np.ndarray) -> np.ndarray:
        return np.asarray(np.exp(np.clip(eta, -30.0, 30.0)), dtype=float)

    @staticmethod
    def irls_weight(
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        return np.asarray(mu, dtype=float)

    @staticmethod
    def score_eta(
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        return np.asarray(y - mu, dtype=float)

    @staticmethod
    def log_lik_vec(
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        mu = np.clip(mu, 1e-300, None)
        # Include -log(y!) so AIC is comparable to negative-binomial fits
        # on the same y (NB collapses to Poisson + log(y!) as α → 0).
        return y * np.log(mu) - mu - special.gammaln(y + 1.0)


class _Gamma(_Family):
    """
    Gamma family with log link (``meglm`` ``family(gamma) link(log)``).

    Density (mean-dispersion form):

        f(y; μ, φ) = (1/(y Γ(1/φ))) · (y / (μ φ))^(1/φ) · exp(-y/(μ φ)),
        E[Y] = μ,   Var(Y) = φ μ².

    Dispersion ``φ`` is estimated jointly with (β, θ) through the packed
    parameter ``log φ``.  The expected (Fisher) information on η is
    ``W = 1/φ``; the observed information is ``W = y/(μ φ)``, positive for
    every y > 0.  The Laplace / AGHQ curvature uses the observed form by
    default (``curvature='observed'``), the Fisher form on request.  The
    score on η is the canonical-Pearson residual scaled by 1/φ.
    """

    name = "gamma"
    link = "log"
    n_disp_params = 1

    @staticmethod
    def inv_link(eta: np.ndarray) -> np.ndarray:
        return np.asarray(np.exp(np.clip(eta, -30.0, 30.0)), dtype=float)

    @staticmethod
    def irls_weight(
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        phi = dispersion if dispersion is not None else 1.0
        return np.asarray((w / max(phi, _EPS)) * np.ones_like(mu), dtype=float)

    @staticmethod
    def score_eta(
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        phi = dispersion if dispersion is not None else 1.0
        return np.asarray(w * (y - mu) / (mu * max(phi, _EPS)), dtype=float)

    @classmethod
    def obs_weight(
        cls,
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        # log f = -(η + y e^{-η})/φ + const  ⇒  −∂²/∂η² = y / (μ φ) > 0.
        phi = dispersion if dispersion is not None else 1.0
        return np.asarray(
            w * y / (np.clip(mu, _EPS, None) * max(phi, _EPS)), dtype=float
        )

    @staticmethod
    def log_lik_vec(
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        phi = dispersion if dispersion is not None else 1.0
        phi = max(phi, _EPS)
        inv_phi = 1.0 / phi
        # Drop terms independent of (μ, φ) gradients?  Keep the full kernel
        # so AIC/BIC are interpretable and so AGHQ's quadrature is exact.
        ll = (
            (inv_phi - 1.0) * np.log(np.clip(y, _EPS, None))
            - special.gammaln(inv_phi)
            - inv_phi * (np.log(np.clip(mu, _EPS, None)) + np.log(phi))
            - y / (np.clip(mu, _EPS, None) * phi)
        )
        return w * ll


class _NegBin(_Family):
    """
    Negative binomial (NB-2) family with log link (``menbreg``).

    Parameterisation:  Var(Y) = μ + α μ², α > 0.  Density

        f(y; μ, α) = Γ(y + 1/α) / (Γ(1/α) Γ(y+1))
                     · (1/(1+αμ))^(1/α) · (αμ/(1+αμ))^y.

    α → 0 reduces to Poisson; the score and Fisher weight collapse
    accordingly.  We pack ``log α`` so the optimiser has unconstrained
    real support.
    """

    name = "nbinomial"
    link = "log"
    n_disp_params = 1

    @staticmethod
    def inv_link(eta: np.ndarray) -> np.ndarray:
        return np.asarray(np.exp(np.clip(eta, -30.0, 30.0)), dtype=float)

    @staticmethod
    def irls_weight(
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        alpha = dispersion if dispersion is not None else 0.0
        return np.asarray(w * mu / (1.0 + alpha * mu), dtype=float)

    @staticmethod
    def score_eta(
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        alpha = dispersion if dispersion is not None else 0.0
        return np.asarray(w * (y - mu) / (1.0 + alpha * mu), dtype=float)

    @classmethod
    def obs_weight(
        cls,
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        # ∂ log f/∂η = (y − μ)/(1 + αμ);  −∂²/∂η² = μ(1 + αy)/(1 + αμ)² > 0.
        alpha = dispersion if dispersion is not None else 0.0
        return np.asarray(
            w * mu * (1.0 + alpha * y) / (1.0 + alpha * mu) ** 2, dtype=float
        )

    @staticmethod
    def log_lik_vec(
        y: np.ndarray,
        mu: np.ndarray,
        w: np.ndarray,
        dispersion: Optional[float],
    ) -> np.ndarray:
        alpha = dispersion if dispersion is not None else 0.0
        if alpha <= _EPS:
            # Poisson limit
            mu_c = np.clip(mu, 1e-300, None)
            return w * (y * np.log(mu_c) - mu_c - special.gammaln(y + 1))
        inv_a = 1.0 / alpha
        am = alpha * np.clip(mu, _EPS, None)
        ll = (
            special.gammaln(y + inv_a)
            - special.gammaln(inv_a)
            - special.gammaln(y + 1.0)
            - inv_a * np.log1p(am)
            + y * (np.log(am) - np.log1p(am))
        )
        return w * ll


_FAMILIES: Dict[str, _Family] = {
    "gaussian": _Gaussian(),
    "binomial": _Binomial(),
    "poisson": _Poisson(),
    "gamma": _Gamma(),
    "nbinomial": _NegBin(),
}

# Aliases accepted from the user but normalised to the canonical key.
_FAMILY_ALIASES: Dict[str, str] = {
    "negbin": "nbinomial",
    "negbinomial": "nbinomial",
    "negative_binomial": "nbinomial",
    "nb": "nbinomial",
    "nb2": "nbinomial",
}


def _resolve_family(name: str) -> _Family:
    name = _require_string(name, "family")
    key = name.lower()
    key = _FAMILY_ALIASES.get(key, key)
    if key not in _FAMILIES:
        raise MethodIncompatibility(
            f"family must be one of {sorted(_FAMILIES)} (with aliases "
            f"{sorted(_FAMILY_ALIASES)}); got {name!r}.",
            recovery_hint=(
                "Use family='gaussian', 'binomial', 'poisson', 'gamma', "
                "or 'nbinomial'."
            ),
            diagnostics={
                "family": name,
                "valid": sorted(_FAMILIES),
                "aliases": sorted(_FAMILY_ALIASES),
            },
        )
    return _FAMILIES[key]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MEGLMResult(ResultProtocolMixin):
    """Container for GLMM fits (``meglm``, ``melogit``, ``mepoisson``,
    ``megamma``, ``menbreg``, ``meologit``).

    Exposes the fitted ``fixed_effects`` / ``random_effects``,
    ``variance_components``, BLUPs, and information criteria (``aic`` /
    ``bic``). Family-specific helpers include :meth:`odds_ratios`
    (binomial), :meth:`incidence_rate_ratios` (Poisson / NB), plus
    :meth:`summary`, :meth:`predict`, :meth:`to_latex`, and :meth:`plot`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(20), 12)
    >>> x = rng.normal(size=240)
    >>> u = rng.normal(0, 0.7, 20)[g]
    >>> p = 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * x + u)))
    >>> y = (rng.uniform(size=240) < p).astype(float)
    >>> df = pd.DataFrame({"y": y, "x": x, "gid": g})
    >>> res = sp.melogit(df, y="y", x_fixed=["x"], group="gid")
    >>> isinstance(res, sp.MEGLMResult)
    True
    >>> bool(np.isfinite(res.aic))
    True
    """

    fixed_effects: pd.Series
    random_effects: pd.DataFrame
    variance_components: Dict[str, float]
    blups: Dict[Any, np.ndarray]
    n_obs: int
    n_groups: int
    log_likelihood: float

    family: str = "gaussian"
    link: str = "identity"

    _se_fixed: pd.Series = field(default=None, repr=False)
    _cov_fixed: Optional[np.ndarray] = field(default=None, repr=False)
    _cov_fixed_conditional: Optional[np.ndarray] = field(default=None, repr=False)
    _cov_full: Optional[np.ndarray] = field(default=None, repr=False)
    _vce_method: str = field(default="oim", repr=False)
    _G: Optional[np.ndarray] = field(default=None, repr=False)
    _x_fixed: List[str] = field(default_factory=list, repr=False)
    _x_random: List[str] = field(default_factory=list, repr=False)
    _group_col: str = field(default="", repr=False)
    _fixed_names: List[str] = field(default_factory=list, repr=False)
    _random_names: List[str] = field(default_factory=list, repr=False)
    _y_name: str = field(default="", repr=False)
    _converged: bool = field(default=True, repr=False)
    _method: str = field(default="laplace", repr=False)
    _cov_type: str = field(default="unstructured", repr=False)
    _alpha: float = field(default=0.05, repr=False)
    _n_cov_params: int = field(default=0, repr=False)
    _offset_name: Optional[str] = field(default=None, repr=False)
    _dispersion: Optional[float] = field(default=None, repr=False)
    # Ordinal-logit specific — None for non-ordinal models.
    thresholds: Optional[pd.Series] = field(default=None, repr=False)
    thresholds_se: Optional[pd.Series] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def params(self) -> pd.Series:
        return self.fixed_effects

    @property
    def bse(self) -> pd.Series:
        return self._se_fixed

    @property
    def tvalues(self) -> pd.Series:
        return self.fixed_effects / self._se_fixed

    @property
    def pvalues(self) -> pd.Series:
        z = self.tvalues.abs()
        return 2.0 * stats.norm.sf(z)

    @property
    def n_fixed(self) -> int:
        return len(self.fixed_effects)

    @property
    def n_thresholds(self) -> int:
        return 0 if self.thresholds is None else len(self.thresholds)

    @property
    def n_dispersion_params(self) -> int:
        return 0 if self._dispersion is None else 1

    @property
    def n_params(self) -> int:
        return (
            self.n_fixed
            + self._n_cov_params
            + self.n_thresholds
            + self.n_dispersion_params
        )

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        return float(self.n_params * np.log(self.n_obs) - 2.0 * self.log_likelihood)

    @property
    def dispersion(self) -> Optional[float]:
        """Estimated dispersion (φ for gamma, α for nbinomial); None otherwise."""
        return self._dispersion

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        alpha = _require_open_unit_float(alpha, "alpha")
        z = stats.norm.ppf(1 - alpha / 2)
        lo = self.fixed_effects - z * self._se_fixed
        hi = self.fixed_effects + z * self._se_fixed
        return pd.DataFrame({"lower": lo, "upper": hi})

    def odds_ratios(self) -> pd.DataFrame:
        """Exponentiated fixed effects with Wald CIs (binomial / ordinal)."""
        if self.family not in ("binomial", "ordinal"):
            raise MethodIncompatibility(
                "odds_ratios() is meaningful for binomial and ordinal " "GLMMs only.",
                recovery_hint=("Use odds_ratios() only after melogit/meologit fits."),
                diagnostics={"family": self.family},
            )
        ci = self.conf_int()
        return pd.DataFrame(
            {
                "OR": np.exp(self.fixed_effects),
                "lower": np.exp(ci["lower"]),
                "upper": np.exp(ci["upper"]),
            }
        )

    def incidence_rate_ratios(self) -> pd.DataFrame:
        """Exponentiated coefficients for log-link count GLMMs (Poisson / NB)."""
        if self.family not in ("poisson", "nbinomial"):
            raise MethodIncompatibility(
                "IRR is meaningful for Poisson and negative-binomial GLMMs " "only.",
                recovery_hint=(
                    "Use incidence_rate_ratios() only after mepoisson or "
                    "menbreg fits."
                ),
                diagnostics={"family": self.family},
            )
        ci = self.conf_int()
        return pd.DataFrame(
            {
                "IRR": np.exp(self.fixed_effects),
                "lower": np.exp(ci["lower"]),
                "upper": np.exp(ci["upper"]),
            }
        )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines: List[str] = []
        w = 76
        lines.append("=" * w)
        title = f"Mixed-effects {self.family.upper()} GLMM (link: {self.link})"
        lines.append(title.center(w))
        lines.append("=" * w)

        lines.append(f"  Method:          {self._method}")
        lines.append(f"  Cov(random):     {self._cov_type}")
        lines.append(f"  No. obs:         {self.n_obs}")
        lines.append(f"  No. groups:      {self.n_groups}")
        lines.append(f"  Log-likelihood:  {self.log_likelihood:.4f}")
        lines.append(f"  AIC / BIC:       {self.aic:.3f}  /  {self.bic:.3f}")
        lines.append(f"  Converged:       {self._converged}")
        if self._dispersion is not None:
            disp_label = {
                "gamma": "phi",
                "nbinomial": "alpha",
                "gaussian": "sigma2",
            }.get(self.family, "dispersion")
            lines.append(f"  Dispersion ({disp_label}): {self._dispersion:.6f}")
        lines.append("-" * w)

        z_crit = stats.norm.ppf(1 - self._alpha / 2)
        lines.append("Fixed effects:")
        hdr = (
            f"{'':>18s} {'Coef':>10s} {'Std.Err':>10s} "
            f"{'z':>8s} {'P>|z|':>8s}  [{100 * (1 - self._alpha):.0f}% CI]"
        )
        lines.append(hdr)
        lines.append("-" * w)
        for var in self.fixed_effects.index:
            b = self.fixed_effects[var]
            se = self._se_fixed[var] if self._se_fixed is not None else np.nan
            z = b / se if se and se > 0 else np.nan
            p = 2 * stats.norm.sf(abs(z)) if z == z else np.nan
            lo, hi = b - z_crit * se, b + z_crit * se
            lines.append(
                f"{var:>18s} {b:10.4f} {se:10.4f} {z:8.3f} {p:8.4f}  "
                f"[{lo:8.4f}, {hi:8.4f}]"
            )
        if self.thresholds is not None and len(self.thresholds) > 0:
            lines.append("-" * w)
            lines.append("Thresholds (cutpoints):")
            for name, val in self.thresholds.items():
                lines.append(f"{name:>18s} {val:10.4f}")
        lines.append("-" * w)
        lines.append("Variance components:")
        for name, val in self.variance_components.items():
            lines.append(f"  {name:24s}  {val:.6f}")
        lines.append("=" * w)
        return "\n".join(lines)

    # Own layout without the base class's digits= / fmt= precision options;
    # passing them raises TypeError rather than being ignored.
    def to_markdown(self) -> str:  # type: ignore[override]
        rows = []
        for var in self.fixed_effects.index:
            b = self.fixed_effects[var]
            se = self._se_fixed[var]
            z = b / se if se else float("nan")
            p = 2 * stats.norm.sf(abs(z)) if z == z else float("nan")
            rows.append(f"| {var} | {b:.4f} | {se:.4f} | {z:.3f} | {p:.4f} |")
        vc_rows = [f"| {n} | {v:.6f} |" for n, v in self.variance_components.items()]
        return (
            f"# {self.family.capitalize()} GLMM (link: {self.link})\n\n"
            f"**N = {self.n_obs}, Groups = {self.n_groups}, "
            f"LogL = {self.log_likelihood:.3f}**\n\n"
            "## Fixed effects\n\n"
            "| Variable | Coef | SE | z | P>|z| |\n"
            "|----------|-----:|---:|---:|-----:|\n"
            + "\n".join(rows)
            + "\n\n## Variance components\n\n"
            + "| Component | Estimate |\n|-----------|---------:|\n"
            + "\n".join(vc_rows)
        )

    def _repr_html_(self) -> str:
        rows_fixed = "".join(
            f"<tr><td>{v}</td><td>{self.fixed_effects[v]:.4f}</td>"
            f"<td>{self._se_fixed[v]:.4f}</td>"
            f"<td>{self.fixed_effects[v] / self._se_fixed[v]:.3f}</td></tr>"
            for v in self.fixed_effects.index
        )
        rows_vc = "".join(
            f"<tr><td>{n}</td><td>{val:.6f}</td></tr>"
            for n, val in self.variance_components.items()
        )
        return (
            "<div style='font-family: monospace'>"
            f"<h4>{self.family.capitalize()} GLMM &mdash; link: {self.link}</h4>"
            f"<p>N = {self.n_obs}, groups = {self.n_groups}, "
            f"LogL = {self.log_likelihood:.3f}, "
            f"AIC = {self.aic:.2f}, BIC = {self.bic:.2f}</p>"
            "<table><thead><tr><th>Variable</th><th>Coef</th>"
            "<th>SE</th><th>z</th></tr></thead>"
            f"<tbody>{rows_fixed}</tbody></table>"
            "<table><thead><tr><th>Variance component</th>"
            "<th>Estimate</th></tr></thead>"
            f"<tbody>{rows_vc}</tbody></table>"
            "</div>"
        )

    def cite(self, format: str = "bibtex") -> str:
        return (
            "@article{breslow1993,\n"
            "  author  = {Breslow, N. E. and Clayton, D. G.},\n"
            "  title   = {Approximate Inference in Generalized Linear Mixed Models},\n"
            "  journal = {Journal of the American Statistical Association},\n"
            "  year    = {1993},\n"
            "  volume  = {88},\n"
            "  pages   = {9--25}\n"
            "}\n"
        )

    # ------------------------------------------------------------------
    # LaTeX / plot
    # ------------------------------------------------------------------

    # Own layout without the base class's digits= / fmt= precision options;
    # passing them raises TypeError rather than being ignored.
    def to_latex(  # type: ignore[override]
        self,
        *,
        caption: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        """Booktabs LaTeX fragment; mirrors ``MixedResult.to_latex``."""
        caption_text = caption or f"{self.family.capitalize()} GLMM ({self.link} link)"
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{caption_text}}}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Variable & Coef. & Std.\ Err. & $z$ & $P>|z|$ \\",
            r"\midrule",
        ]
        if label is not None:
            lines.insert(3, rf"\label{{{label}}}")
        for var in self.fixed_effects.index:
            b = self.fixed_effects[var]
            se = self._se_fixed[var]
            z = b / se if se else float("nan")
            p = 2 * stats.norm.sf(abs(z)) if z == z else float("nan")
            lines.append(f"{var} & {b:.4f} & {se:.4f} & {z:.3f} & {p:.4f} \\\\")
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{l}{\textit{Variance components}} \\")
        for name, val in self.variance_components.items():
            safe = name.replace("_", r"\_")
            lines.append(f"{safe} & \\multicolumn{{4}}{{r}}{{{val:.6f}}} \\\\")
        lines.append(r"\bottomrule")
        lines.append(
            rf"\multicolumn{{5}}{{l}}{{\footnotesize $N={self.n_obs}$, "
            rf"groups $={self.n_groups}$, LogL $={self.log_likelihood:.3f}$, "
            rf"AIC $={self.aic:.2f}$.}} \\"
        )
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def plot(
        self,
        kind: str = "caterpillar",
        variable: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """
        Diagnostic plot for the GLMM fit.

        Only ``kind='caterpillar'`` is currently supported: a forest plot
        of the BLUPs for one random effect (default: the intercept).
        Posterior SEs are not returned by ``meglm`` at present, so the
        error bars are omitted.
        """
        import matplotlib.pyplot as plt  # local import to keep plotting optional

        kind = _require_string(kind, "kind")
        if kind != "caterpillar":
            raise MethodIncompatibility(
                f"unknown plot kind {kind!r}",
                recovery_hint="Use kind='caterpillar'.",
                diagnostics={"kind": kind, "valid": ["caterpillar"]},
            )

        name = variable if variable is not None else self._random_names[0]
        if name not in self.random_effects.columns:
            raise MethodIncompatibility(
                f"random effect {name!r} not in model",
                recovery_hint=(
                    "Choose a random-effect name present in "
                    "result.random_effects.columns."
                ),
                diagnostics={
                    "random_effect": name,
                    "available": list(self.random_effects.columns),
                },
            )
        u = self.random_effects[name].copy().sort_values()
        fig, ax = plt.subplots(**{"figsize": (6, 0.2 * len(u) + 1), **kwargs})
        y_pos = np.arange(len(u))
        ax.plot(u.values, y_pos, "o", ms=3)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(i) for i in u.index], fontsize=7)
        ax.set_xlabel(f"BLUP of {name}")
        ax.set_title(f"Caterpillar plot ({self.family} GLMM): random {name}")
        fig.tight_layout()
        return fig, ax

    def predict(
        self,
        data: Optional[pd.DataFrame] = None,
        include_random: bool = True,
        type: str = "response",
    ) -> pd.Series:
        """
        Predict response or linear predictor.

        ``type='response'`` returns μ = g⁻¹(η); ``type='linear'`` returns
        η itself.  ``include_random=True`` adds Z û for groups seen at
        fit time.
        """
        if data is None:
            raise MethodIncompatibility(
                "predict() requires a dataframe for GLMMs.",
                recovery_hint="Pass the training data or new prediction data.",
                diagnostics={"data": None},
            )
        if not isinstance(data, pd.DataFrame):
            raise MethodIncompatibility(
                "predict() requires a pandas DataFrame for GLMMs.",
                diagnostics={"data_type": data.__class__.__name__},
            )
        valid_types = {"response", "linear"}
        if not isinstance(type, str) or type not in valid_types:
            raise MethodIncompatibility(
                "`type` must be 'response' or 'linear'.",
                recovery_hint="Use type='response' or type='linear'.",
                diagnostics={"type": repr(type), "valid": sorted(valid_types)},
            )
        required = list(self._x_fixed)
        if include_random:
            required.extend(self._x_random or [])
            required.append(self._group_col)
        if self._offset_name:
            required.append(self._offset_name)
        missing = [c for c in dict.fromkeys(required) if c not in data.columns]
        if missing:
            raise MethodIncompatibility(
                f"predict(): missing column(s) {missing}",
                recovery_hint=(
                    "Pass fixed-effect columns for marginal prediction; add "
                    "random-effect, group, and offset columns when the fitted "
                    "model uses them."
                ),
                diagnostics={
                    "missing_columns": missing,
                    "include_random": bool(include_random),
                    "offset": self._offset_name,
                },
            )
        try:
            X = np.column_stack(
                [np.ones(len(data))]
                + [data[c].to_numpy(dtype=float) for c in self._x_fixed]
            )
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "GLMM prediction fixed-effect columns must be numeric.",
                diagnostics={"columns": list(self._x_fixed), "error": str(exc)},
            ) from exc
        if not np.isfinite(X).all():
            raise MethodIncompatibility(
                "GLMM prediction fixed-effect columns must be finite.",
                diagnostics={"columns": list(self._x_fixed)},
            )
        eta = X @ self.fixed_effects.values
        if include_random:
            try:
                Z = np.column_stack(
                    [np.ones(len(data))]
                    + [data[c].to_numpy(dtype=float) for c in self._x_random]
                )
            except (TypeError, ValueError) as exc:
                raise MethodIncompatibility(
                    "GLMM prediction random-effect columns must be numeric.",
                    diagnostics={"columns": list(self._x_random), "error": str(exc)},
                ) from exc
            if not np.isfinite(Z).all():
                raise MethodIncompatibility(
                    "GLMM prediction random-effect columns must be finite.",
                    diagnostics={"columns": list(self._x_random)},
                )
            keys = list(data[self._group_col].values)
            u_mat = np.zeros_like(Z)
            for i, k in enumerate(keys):
                u = self.blups.get(k)
                if u is not None:
                    u_mat[i, :] = u
            eta = eta + np.einsum("ij,ij->i", Z, u_mat)
        if self._offset_name:
            try:
                offset_values = data[self._offset_name].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise MethodIncompatibility(
                    "GLMM prediction offset column must be numeric.",
                    diagnostics={"column": self._offset_name, "error": str(exc)},
                ) from exc
            if not np.isfinite(offset_values).all():
                raise MethodIncompatibility(
                    "GLMM prediction offset column must be finite.",
                    diagnostics={"column": self._offset_name},
                )
            eta = eta + offset_values
        if type == "linear":
            return pd.Series(eta, index=data.index, name="eta")
        fam = _resolve_family(self.family)
        return pd.Series(fam.inv_link(eta), index=data.index, name="mu")


# ---------------------------------------------------------------------------
# Inner mode-finder (Newton on u_j given β, G, dispersion)
# ---------------------------------------------------------------------------


def _find_mode(
    block: _GroupBlock,
    beta: np.ndarray,
    G: np.ndarray,
    Ginv: np.ndarray,
    family: _Family,
    weights: np.ndarray,
    offset: np.ndarray,
    u0: np.ndarray,
    dispersion: Optional[float],
    max_inner: int = 50,
    tol: float = 1e-10,
    observed: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Newton iteration for the conditional mode û_j with step damping.

    Returns ``(û_j, H_j, log|H_j|, converged)`` where ``H_j`` is the
    curvature of the negative log integrand at û_j: the observed
    information ``Z'W_obs Z + G⁻¹`` (``observed=True``, the Laplace
    approximation proper) or the Fisher-weight version ``Z'W Z + G⁻¹``
    (``observed=False``, lme4's PIRLS convention).  The two coincide for
    canonical links.  ``converged`` is ``False`` when the tolerance is
    not met within ``max_inner`` iterations or when the Hessian cannot be
    factorised — the outer optimiser can then record a warning rather
    than silently using a stale mode.

    Once the step criterion is met one further Newton step is taken:
    Newton converges quadratically, so this puts û_j at machine precision.
    The outer objective (and its finite-difference gradient / Hessian)
    depends on û_j through log|H_j|, which is not stationary in u, so a
    mode that is only accurate to the stopping tolerance would inject that
    tolerance into every numerical derivative.
    """
    weight_fn = family.obs_weight if observed else None

    def _curv(mu: np.ndarray) -> np.ndarray:
        if weight_fn is not None:
            return weight_fn(block.y, mu, weights, dispersion)
        return family.irls_weight(mu, weights, dispersion)

    u = u0.copy()
    converged = False
    for _ in range(max_inner):
        eta = block.X @ beta + block.Z @ u + offset
        mu = family.inv_link(eta)
        W = _curv(mu)
        s = family.score_eta(block.y, mu, weights, dispersion)

        grad = block.Z.T @ s - Ginv @ u
        H = block.Z.T @ (W[:, None] * block.Z) + Ginv

        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break

        # Step damping: keep ‖step‖ ≤ 5·max(‖u‖, 1) to avoid blow-up
        # when the mode is far from the current iterate.
        step_norm = float(np.linalg.norm(step))
        u_norm = float(np.linalg.norm(u)) + 1e-12
        if step_norm > 5.0 * max(u_norm, 1.0):
            step = step * (5.0 * max(u_norm, 1.0) / step_norm)
            step_norm = float(np.linalg.norm(step))

        u = u + step
        if converged:
            break  # the extra quadratic-convergence step has been taken
        if step_norm < tol * (1 + u_norm):
            converged = True

    eta = block.X @ beta + block.Z @ u + offset
    mu = family.inv_link(eta)
    W = _curv(mu)
    H = block.Z.T @ (W[:, None] * block.Z) + Ginv
    sign, logdet_H = np.linalg.slogdet(H)
    if sign <= 0:
        logdet_H = np.inf
        converged = False
    return u, H, logdet_H, converged


# ---------------------------------------------------------------------------
# Adaptive Gauss-Hermite quadrature (q = 1 only)
# ---------------------------------------------------------------------------
#
# Single random intercept ⇒ q = 1.  For a group j with conditional mode û_j,
# Hessian H_j and observed unit variance σ̂_j² = 1/H_j, AGHQ approximates
#
#     L_j = ∫ exp(h(u)) du
#         ≈ Σ_k √2 σ̂_j exp(x_k²) w_k · exp(h(û_j + √2 σ̂_j x_k)),
#
# where (x_k, w_k) are the standard Gauss-Hermite nodes/weights for
# ∫ exp(-x²) g(x) dx.  ``log L_j`` is computed via logsumexp for stability.
#
# h(u) = log f(y_j | u) − ½ u²/σ² − ½ log(2π σ²)  (with q = 1, G = σ²).
#
# For nAGQ = 1 the formula collapses to the Laplace approximation, which we
# verify in the unit tests.


def _aghq_log_lik(
    block: _GroupBlock,
    beta: np.ndarray,
    sigma2: float,
    family: _Family,
    weights: np.ndarray,
    offset: np.ndarray,
    u_hat: float,
    H: float,
    nodes: np.ndarray,
    log_weights: np.ndarray,
    dispersion: Optional[float],
) -> float:
    """Per-group AGHQ log-likelihood for a scalar random intercept."""
    sigma_hat = 1.0 / np.sqrt(max(H, _EPS))
    # u_k = û + √2 σ̂ x_k
    u_grid = u_hat + np.sqrt(2.0) * sigma_hat * nodes
    # log f(y_j | u_k) for each node
    log_lik_vals = np.empty(nodes.shape[0])
    for k, u_k in enumerate(u_grid):
        eta_k = block.X @ beta + block.Z[:, 0] * u_k + offset
        mu_k = family.inv_link(eta_k)
        log_lik_vals[k] = family.log_lik(block.y, mu_k, weights, dispersion)
    # log φ(u_k; 0, σ²) = -½ log(2π σ²) - u_k²/(2σ²)
    log_prior = -0.5 * (_LOG_2PI + np.log(max(sigma2, _EPS))) - 0.5 * u_grid**2 / max(
        sigma2, _EPS
    )
    # log integrand at node k: log f + log prior + x_k² + log w_k + ½ log(2 σ̂²)
    log_terms = (
        log_lik_vals
        + log_prior
        + nodes**2
        + log_weights
        + 0.5 * (np.log(2.0) + 2.0 * np.log(sigma_hat))
    )
    return float(special.logsumexp(log_terms))


def _gh_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Standard Gauss-Hermite nodes/weights for ∫ exp(-x²) g(x) dx."""
    nodes, weights = special.roots_hermite(n)
    return nodes, weights


# ---------------------------------------------------------------------------
# Outer NLL: combines Laplace (nAGQ=1) and AGHQ (nAGQ>=2) paths
# ---------------------------------------------------------------------------


def _glmm_nll(
    theta: np.ndarray,
    blocks: List[_GroupBlock],
    weights_list: List[np.ndarray],
    offsets_list: List[np.ndarray],
    p_fixed: int,
    q_random: int,
    cov_type: str,
    family: _Family,
    u_cache: List[np.ndarray],
    nAGQ: int,
    gh_nodes: Optional[np.ndarray],
    gh_log_weights: Optional[np.ndarray],
    observed: bool = True,
) -> float:
    """
    Negative integrated log-likelihood for a GLMM.

    Layout of ``theta``:

        [ β  (p_fixed)
        , cov_params                   (n_cov)
        , dispersion (log φ or log α)  (0 or 1) ]
    """
    beta = theta[:p_fixed]
    n_cov = _n_cov_params(q_random, cov_type)
    cov_params = theta[p_fixed : p_fixed + n_cov]
    disp_packed = theta[p_fixed + n_cov :]
    dispersion = family.parse_dispersion(disp_packed) if family.n_disp_params else None

    G = _unpack_G(cov_params, q_random, cov_type)
    try:
        sign, logdet_G = np.linalg.slogdet(G)
        if sign <= 0:
            return 1e12
        Ginv = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return 1e12

    use_aghq = nAGQ > 1
    if use_aghq and q_random != 1:
        # Defensive — should be rejected at the public-API boundary.
        raise MethodIncompatibility(
            "AGHQ supports only q=1 random-intercept models.",
            diagnostics={"nAGQ": nAGQ, "q_random": q_random},
        )
    sigma2 = float(G[0, 0]) if use_aghq else None

    nll = 0.0
    for j, block in enumerate(blocks):
        w = weights_list[j]
        off = offsets_list[j]
        u_hat, H_j, logdet_H, _ = _find_mode(
            block,
            beta,
            G,
            Ginv,
            family,
            w,
            off,
            u_cache[j],
            dispersion,
            observed=observed,
        )
        u_cache[j] = u_hat

        if use_aghq:
            if sigma2 is None or gh_nodes is None or gh_log_weights is None:
                raise RuntimeError("AGHQ nodes must be initialized for nAGQ > 1.")
            # H_j is a 1×1 in q=1, so its scalar value is H_j[0,0].
            ll_j = _aghq_log_lik(
                block,
                beta,
                sigma2,
                family,
                w,
                off,
                float(u_hat[0]),
                float(H_j[0, 0]),
                gh_nodes,
                gh_log_weights,
                dispersion,
            )
        else:
            eta = block.X @ beta + block.Z @ u_hat + off
            mu = family.inv_link(eta)
            ll_data = family.log_lik(block.y, mu, w, dispersion)
            quad = float(u_hat @ Ginv @ u_hat)
            # Laplace log-integrand (constants in 2π cancel out): see module docstring.
            ll_j = ll_data - 0.5 * logdet_G - 0.5 * quad - 0.5 * logdet_H
        nll -= ll_j
    return nll


def _glmm_nll_ri(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    off: np.ndarray,
    gidx: np.ndarray,
    n_groups: int,
    p_fixed: int,
    cov_type: str,
    family: _Family,
    u_cache: np.ndarray,
    nAGQ: int,
    gh_nodes: Optional[np.ndarray],
    gh_log_weights: Optional[np.ndarray],
    observed: bool = True,
) -> float:
    """:func:`_glmm_nll` for a single random intercept, vectorised over
    groups (see :mod:`._glmm_ri`).  Same parameter layout."""
    beta = theta[:p_fixed]
    n_cov = _n_cov_params(1, cov_type)
    G = _unpack_G(theta[p_fixed : p_fixed + n_cov], 1, cov_type)
    sigma2 = float(G[0, 0])
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return 1e12
    disp_packed = theta[p_fixed + n_cov :]
    dispersion = family.parse_dispersion(disp_packed) if family.n_disp_params else None
    eta_fixed = X @ beta + off

    def curv_score(eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = family.inv_link(eta)
        if observed:
            W = family.obs_weight(y, mu, w, dispersion)
        else:
            W = family.irls_weight(mu, w, dispersion)
        return W, family.score_eta(y, mu, w, dispersion)

    def loglik_vec(eta: np.ndarray) -> np.ndarray:
        return family.log_lik_vec(y, family.inv_link(eta), w, dispersion)

    ll = _ri.ri_marginal_loglik(
        eta_fixed,
        gidx,
        n_groups,
        sigma2,
        curv_score,
        loglik_vec,
        u_cache,
        gh_nodes if nAGQ > 1 else None,
        gh_log_weights if nAGQ > 1 else None,
    )
    if not np.isfinite(ll):
        return 1e12
    return -ll


def _numerical_hessian(nll: Any, theta: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of ``nll`` at ``theta``.

    The second-order stencil has O(step^2) truncation error; with
    ``step=1e-4`` on parameters of order one and the inner mode solved to
    machine precision (see :func:`_find_mode`) the error is ~1e-8
    relative.
    """
    theta = np.asarray(theta, dtype=float)
    k = theta.size
    f0 = float(nll(theta))
    H = np.zeros((k, k))
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = step
        fp = float(nll(theta + ei))
        fm = float(nll(theta - ei))
        H[i, i] = (fp - 2.0 * f0 + fm) / step**2
        for j in range(i + 1, k):
            ej = np.zeros(k)
            ej[j] = step
            fpp = float(nll(theta + ei + ej))
            fpm = float(nll(theta + ei - ej))
            fmp = float(nll(theta - ei + ej))
            fmm = float(nll(theta - ei - ej))
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * step**2)
    return H


def _numerical_gradient(nll: Any, theta: np.ndarray, step: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of ``nll`` at ``theta``."""
    theta = np.asarray(theta, dtype=float)
    g = np.zeros(theta.size)
    for i in range(theta.size):
        ei = np.zeros(theta.size)
        ei[i] = step
        g[i] = (float(nll(theta + ei)) - float(nll(theta - ei))) / (2.0 * step)
    return g


def _newton_polish(
    nll: Any, theta: np.ndarray, max_steps: int = 8, xtol: float = 1e-9
) -> Tuple[np.ndarray, bool]:
    """Finish a quasi-Newton optimum with full Newton steps.

    L-BFGS-B on a finite-difference gradient stops where its *relative
    function change* falls below ``ftol``; on an integrated likelihood of
    order 10^3 that leaves the parameters ~1e-4 from the optimum, which
    is the whole disagreement budget of a cross-package comparison.
    Newton steps on the central-difference gradient and Hessian converge
    quadratically from there to the noise floor of the numerical
    derivatives (~1e-10 in the parameters).  Each step is backtracked so
    the objective never increases.  Returns ``(theta, converged)``;
    ``converged`` is ``False`` when the Hessian is not positive definite
    (a variance component drifting to the boundary, a flat likelihood).
    ``theta`` is then the last iterate at which the Hessian *was*
    positive definite -- the starting point if there was none -- so the
    covariance evaluated there is well defined, as it was before
    polishing.
    """
    theta = np.asarray(theta, dtype=float).copy()
    f0 = float(nll(theta))
    last_pd = theta.copy()
    for _ in range(max_steps):
        g = _numerical_gradient(nll, theta)
        H = _numerical_hessian(nll, theta)
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            return last_pd, False
        last_pd = theta.copy()
        step = np.linalg.solve(H, g)
        t = 1.0
        accepted = False
        for _ in range(30):
            cand = theta - t * step
            f1 = float(nll(cand))
            if np.isfinite(f1) and f1 <= f0 + 1e-12 * max(abs(f0), 1.0):
                accepted = True
                break
            t *= 0.5
        if not accepted:
            # No decrease along the Newton direction: already at the
            # numerical-derivative noise floor.
            return theta, True
        theta, f0 = cand, f1
        if float(np.max(np.abs(t * step))) < xtol:
            return theta, True
    return theta, True


def _numerical_oim_cov(
    nll: Any, theta: np.ndarray, step: float = 1e-4
) -> Optional[np.ndarray]:
    """Inverse of the central-difference Hessian of ``nll`` at ``theta``.

    Returns ``None`` when the Hessian is not positive definite.
    """
    H = _numerical_hessian(nll, theta, step=step)
    k = H.shape[0]
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return None
    Linv = np.linalg.solve(L, np.eye(k))
    return Linv.T @ Linv


# ---------------------------------------------------------------------------
# IRLS warm-start for β (and dispersion seed for gamma/nbreg)
# ---------------------------------------------------------------------------


def _irls_init(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    off: np.ndarray,
    fam: _Family,
    maxiter: int = 50,
    tol: float = 1e-8,
) -> np.ndarray:
    """Plain GLM via Fisher scoring — used only to warm-start β.

    Dispersion (φ for gamma, α for nbreg) is left at its default of 1
    here; the outer optimiser will refine it jointly with (β, θ).
    """
    p = X.shape[1]
    beta = np.zeros(p)
    # Smarter intercept seeds for non-identity links.
    if fam.name == "binomial":
        pbar = np.clip(np.sum(y) / max(np.sum(w), 1.0), 1e-3, 1 - 1e-3)
        beta[0] = np.log(pbar / (1.0 - pbar))
    elif fam.name in ("poisson", "nbinomial"):
        beta[0] = np.log(max(np.mean(y), 1e-6))
    elif fam.name == "gamma":
        beta[0] = np.log(max(np.mean(y), 1e-6))

    disp_seed = 1.0 if fam.n_disp_params else None

    for _ in range(maxiter):
        eta = X @ beta + off
        mu = fam.inv_link(eta)
        W = fam.irls_weight(mu, w, disp_seed)
        grad = X.T @ fam.score_eta(y, mu, w, disp_seed)
        H = X.T @ (W[:, None] * X) + 1e-8 * np.eye(p)
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta += step
        if np.linalg.norm(step) < tol * (1 + np.linalg.norm(beta)):
            break
    return beta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def meglm(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group: str,
    family: str = "gaussian",
    x_random: Optional[Sequence[str]] = None,
    cov_type: str = "unstructured",
    trials: Optional[str] = None,
    offset: Optional[str] = None,
    nAGQ: int = 1,
    maxiter: int = 300,
    tol: float = 1e-8,
    alpha: float = 0.05,
    curvature: str = "observed",
) -> MEGLMResult:
    """
    Fit a generalised linear mixed model.

    Parameters
    ----------
    data
        Long-format dataframe.
    y
        Outcome column.  For binomial models this is the number of
        successes; pair it with ``trials=`` to model proportions.
    x_fixed
        Fixed-effect regressors (intercept added automatically).
    group
        Grouping variable for random effects.
    family
        ``'gaussian'``, ``'binomial'``, ``'poisson'``, ``'gamma'``, or
        ``'nbinomial'`` (alias ``'negbin'``).
    x_random
        Random-slope variables; defaults to random intercept only.
    cov_type
        Random-effect covariance: ``'unstructured'`` (default),
        ``'diagonal'``, ``'identity'``.
    trials
        Column of trial counts for binomial responses.  Defaults to 1
        (Bernoulli).
    offset
        Column of fixed offsets added to the linear predictor (e.g.
        ``log(exposure)`` for Poisson rate models).
    nAGQ
        Number of adaptive Gauss-Hermite quadrature points per scalar
        random effect.  ``1`` (default) ≡ Laplace approximation.  Use
        ``nAGQ=7`` to match Stata ``meglm intpoints(7)``; values
        ``> 1`` require a single scalar random effect (no random slopes).
    maxiter, tol, alpha
        Optimisation controls / CI width.  ``maxiter`` / ``tol`` drive the
        L-BFGS-B stage; for AGHQ (``nAGQ > 1``) the default budget is
        internally tightened to ``maxiter=5000`` and ``tol=1e-12``
        (explicit user-supplied controls are respected).  Either way the
        optimum is then finished with full Newton steps on the numerical
        gradient and Hessian, so the reported estimates sit at the
        optimum to ~1e-9 rather than at L-BFGS-B's relative-function-change
        stopping point (which on these likelihoods is ~1e-4 away).
    curvature
        Curvature of the log integrand used by the Laplace approximation
        and to scale the AGHQ nodes.  ``'observed'`` (default) is the
        second derivative of the conditional log-likelihood at the mode —
        the Laplace approximation proper, and what Stata ``me*``
        (``intmethod(laplace)`` / ``mcaghermite``), ``glmmTMB`` and
        ``ordinal::clmm`` compute.  ``'expected'`` substitutes the Fisher
        information (the PIRLS weights of ``lme4::glmer`` /
        ``glmer.nb``).  The two coincide for canonical links (Gaussian,
        binomial-logit, Poisson-log) and differ for gamma-log and
        NB-2-log, where they are two different approximations of the same
        integral.

    Returns
    -------
    MEGLMResult

    Notes
    -----
    Stata's ``me*`` commands default to mean-variance adaptive quadrature
    with 7 points (``intmethod(mvaghermite) intpoints(7)``); this function
    defaults to the Laplace approximation (``nAGQ=1``, lme4's default).
    To reproduce Stata's ``intmethod(laplace)`` pass ``nAGQ=1``; for
    ``intmethod(mcaghermite) intpoints(k)`` pass ``nAGQ=k``.  Mean-variance
    adaptive nodes are not implemented; they agree with mode-curvature
    nodes only up to the quadrature error at ``k`` points.

    References
    ----------
    breslow1993

    Examples
    --------
    Random-intercept logistic GLMM on simulated grouped binary data:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(20), 12)
    >>> x = rng.normal(size=240)
    >>> u = rng.normal(0, 0.7, 20)[g]
    >>> p = 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * x + u)))
    >>> y = (rng.uniform(size=240) < p).astype(float)
    >>> df = pd.DataFrame({"y": y, "x": x, "gid": g})
    >>> res = sp.meglm(df, y="y", x_fixed=["x"], group="gid", family="binomial")
    >>> res.family, res.link
    ('binomial', 'logit')
    >>> bool(res.fixed_effects["x"] > 0)  # recovers the positive slope
    True
    """
    data = _require_dataframe(data, "data")
    y = _require_string(y, "y")
    fam = _resolve_family(family)
    fam_key = fam.name
    cov_type = _require_string(cov_type, "cov_type")
    if cov_type not in ("unstructured", "diagonal", "identity"):
        raise MethodIncompatibility(
            f"unknown cov_type {cov_type!r}",
            recovery_hint=("Use cov_type='unstructured', 'diagonal', or 'identity'."),
            diagnostics={
                "cov_type": cov_type,
                "valid": ["unstructured", "diagonal", "identity"],
            },
        )
    if isinstance(group, (list, tuple)):
        if len(group) != 1:
            raise MethodIncompatibility(
                "meglm() currently supports a single grouping variable; "
                "collapse nested levels into one key first.",
                recovery_hint=(
                    "Create a single grouped key column before calling " "meglm()."
                ),
                diagnostics={"group": list(group)},
            )
        group = group[0]
    if not isinstance(group, str):
        raise MethodIncompatibility(
            "`group` must be a column name string.",
            diagnostics={"group": repr(group)},
        )

    nAGQ = _require_int_at_least(nAGQ, "nAGQ", 1)
    maxiter = _require_int_at_least(maxiter, "maxiter", 1)
    tol = _require_positive_float(tol, "tol")
    alpha = _require_open_unit_float(alpha, "alpha")
    curvature = _require_string(curvature, "curvature")
    if curvature not in ("observed", "expected"):
        raise MethodIncompatibility(
            f"curvature must be 'observed' or 'expected'; got {curvature!r}.",
            recovery_hint=(
                "Use curvature='observed' (Laplace proper; Stata, glmmTMB) "
                "or 'expected' (Fisher weights; lme4)."
            ),
            diagnostics={"curvature": curvature},
        )
    observed = curvature == "observed"

    x_fixed = _coerce_column_list(x_fixed, "x_fixed")
    x_random_cols = _coerce_optional_column_list(x_random, "x_random")
    if nAGQ > 1 and len(x_random_cols) > 0:
        raise MethodIncompatibility(
            "AGHQ (nAGQ > 1) currently supports only random-intercept models "
            "(empty x_random).  Use nAGQ=1 (Laplace) for random-slope models.",
            recovery_hint="Use nAGQ=1 or remove x_random slopes.",
            diagnostics={"nAGQ": nAGQ, "x_random": x_random_cols},
        )

    extra_cols = []
    if trials:
        trials = _require_string(trials, "trials")
        extra_cols.append(trials)
    if offset:
        offset = _require_string(offset, "offset")
        extra_cols.append(offset)

    required_cols = [y] + x_fixed + [group] + x_random_cols + extra_cols
    missing = [c for c in dict.fromkeys(required_cols) if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"meglm(): missing column(s) {missing}",
            recovery_hint=(
                "Add the response, fixed-effect, random-effect, group, "
                "trials, or offset columns referenced by the model."
            ),
            diagnostics={"missing_columns": missing},
        )
    try:
        df = _prepare_frame(data, y, x_fixed + extra_cols, [group], x_random_cols)
    except (KeyError, TypeError) as exc:
        raise MethodIncompatibility(
            str(exc),
            recovery_hint=(
                "Check that all GLMM columns exist and that group values "
                "are hashable."
            ),
            diagnostics={"error": str(exc)},
        ) from exc
    if len(df) == 0:
        raise DataInsufficient(
            "No rows remain after dropping missing GLMM inputs.",
            recovery_hint=(
                "Provide at least one complete row for response, fixed, "
                "random, group, trials, and offset columns."
            ),
            diagnostics={"required_columns": required_cols},
        )
    blocks, fixed_names, random_names = _group_blocks(
        df, y, x_fixed, x_random_cols, group
    )

    # Per-group weight & offset vectors.
    weights_list: List[np.ndarray] = []
    offsets_list: List[np.ndarray] = []
    for block in blocks:
        sub_mask = df[group] == block.key
        if trials:
            w = df.loc[sub_mask, trials].to_numpy(dtype=float)
        else:
            w = np.ones(block.n)
        if offset:
            off = df.loc[sub_mask, offset].to_numpy(dtype=float)
        else:
            off = np.zeros(block.n)
        weights_list.append(w)
        offsets_list.append(off)

    p_fixed = 1 + len(x_fixed)
    q_random = 1 + len(x_random_cols)
    n_cov_pars = _n_cov_params(q_random, cov_type)
    n_disp = fam.n_disp_params

    # GLM warm-start for β (no random effects, plain IRLS).
    X_all = np.vstack([b.X for b in blocks])
    y_all = np.concatenate([b.y for b in blocks])
    w_all = np.concatenate(weights_list)
    off_all = np.concatenate(offsets_list)
    beta0 = _irls_init(X_all, y_all, w_all, off_all, fam)

    theta_cov0 = _initial_theta(q_random, cov_type, s2_init=0.3)
    theta_disp0 = fam.initial_dispersion()
    if fam_key == "gaussian":
        # Seed log σ² at the pooled residual variance of the warm start.
        r0 = y_all - (X_all @ beta0 + off_all)
        theta_disp0 = np.array([np.log(max(float(np.var(r0)), 1e-6))])
    theta0 = np.concatenate([beta0, theta_cov0, theta_disp0])

    u_cache = [np.zeros(q_random) for _ in blocks]
    gh_nodes, gh_log_weights = (None, None)
    if nAGQ > 1:
        nodes, weights = _gh_nodes(nAGQ)
        gh_nodes = nodes
        gh_log_weights = np.log(weights)

    opt_maxiter = maxiter
    opt_tol = tol
    if nAGQ > 1:
        if maxiter == 300:
            opt_maxiter = 5000
        if tol == 1e-8:
            opt_tol = 1e-12

    if q_random == 1:
        # Vectorised random-intercept kernel (same numbers, no group loop).
        gidx = np.repeat(np.arange(len(blocks)), [b.n for b in blocks])
        u_arr = np.zeros(len(blocks))
        ri_args = (
            X_all,
            y_all,
            w_all,
            off_all,
            gidx,
            len(blocks),
            p_fixed,
            cov_type,
            fam,
            u_arr,
            nAGQ,
            gh_nodes,
            gh_log_weights,
            observed,
        )

        def _objective(th: np.ndarray) -> float:
            return _glmm_nll_ri(th, *ri_args)

    else:
        u_arr = None
        nll_args_blocks = (
            blocks,
            weights_list,
            offsets_list,
            p_fixed,
            q_random,
            cov_type,
            fam,
            u_cache,
            nAGQ,
            gh_nodes,
            gh_log_weights,
            observed,
        )

        def _objective(th: np.ndarray) -> float:
            return _glmm_nll(th, *nll_args_blocks)

    res = minimize(
        _objective,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": opt_maxiter, "ftol": opt_tol, "gtol": opt_tol},
    )
    outer_converged = bool(res.success)

    theta_hat, polish_ok = _newton_polish(_objective, res.x)
    fun_hat = float(_objective(theta_hat))
    if u_arr is not None:
        u_cache = [np.array([v]) for v in u_arr]
    if polish_ok:
        # A Newton-certified optimum supersedes an L-BFGS-B "ABNORMAL"
        # line-search exit, which is common at a flat optimum.
        outer_converged = True

    beta_hat = theta_hat[:p_fixed]
    cov_hat = theta_hat[p_fixed : p_fixed + n_cov_pars]
    disp_hat_packed = theta_hat[p_fixed + n_cov_pars :]
    G_hat = _unpack_G(cov_hat, q_random, cov_type)
    Ginv = np.linalg.inv(G_hat)
    dispersion = fam.parse_dispersion(disp_hat_packed) if n_disp else None

    # Fixed-effect covariance: the beta block of the inverse observed
    # information of the *marginal* (Laplace / AGHQ) log-likelihood,
    # differentiated numerically over the full parameter vector
    # (beta, covariance parameters, dispersion). This is what Stata
    # melogit's default vce(oim) and lme4's vcov() report, and it carries
    # the uncertainty in the variance components that the conditional
    # Schur-complement formula below leaves out (which understated the
    # fixed-effect SEs by up to ~2% on the parity fixtures).
    cov_full = _numerical_oim_cov(_objective, theta_hat)
    if u_arr is not None:
        u_cache = [np.array([v]) for v in u_arr]

    # BLUPs and fixed-effect info matrix at the optimum.
    blup_rows: List[Dict[str, float]] = []
    blup_dict: Dict[Any, np.ndarray] = {}
    keys: List[Any] = []

    info = np.zeros((p_fixed, p_fixed))
    inner_failures = 0
    for j, (block, w, off) in enumerate(zip(blocks, weights_list, offsets_list)):
        u0 = u_cache[j]
        u_hat, H_j, _, inner_ok = _find_mode(
            block, beta_hat, G_hat, Ginv, fam, w, off, u0, dispersion, observed=observed
        )
        if not inner_ok:
            inner_failures += 1
        eta = block.X @ beta_hat + block.Z @ u_hat + off
        mu = fam.inv_link(eta)
        W = fam.irls_weight(mu, w, dispersion)
        XtWX = block.X.T @ (W[:, None] * block.X)
        XtWZ = block.X.T @ (W[:, None] * block.Z)
        ZtWX = block.Z.T @ (W[:, None] * block.X)
        info += XtWX - XtWZ @ np.linalg.solve(H_j, ZtWX)

        blup_dict[block.key] = u_hat
        blup_rows.append(dict(zip(random_names, u_hat)))
        keys.append(block.key)

    try:
        cov_beta_conditional = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov_beta_conditional = np.full((p_fixed, p_fixed), np.nan)
    if cov_full is not None and np.all(np.isfinite(cov_full[:p_fixed, :p_fixed])):
        cov_beta = cov_full[:p_fixed, :p_fixed]
        vce_method = "oim"
    else:
        # Fall back to the conditional information if the numerical
        # Hessian is not positive definite (flat likelihood, boundary
        # variance component); say so rather than hide it.
        cov_beta = cov_beta_conditional
        vce_method = "conditional_information"
        warnings.warn(
            "GLMM observed-information Hessian is not positive definite at "
            "the optimum; fixed-effect standard errors fall back to the "
            "conditional (variance-components-fixed) information and may be "
            "understated.",
            RuntimeWarning,
            stacklevel=2,
        )
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    if inner_failures > 0:
        warnings.warn(
            f"GLMM inner Newton failed to converge for "
            f"{inner_failures}/{len(blocks)} clusters; standard errors "
            "and log-likelihood may be unreliable for those groups.",
            RuntimeWarning,
            stacklevel=2,
        )

    random_effects_df = pd.DataFrame(blup_rows, index=keys)
    random_effects_df.index.name = group

    vc: Dict[str, float] = {}
    for i, name in enumerate(random_names):
        vc[f"var({name})"] = float(G_hat[i, i])
    if cov_type == "unstructured" and q_random >= 2:
        for i in range(q_random):
            for j in range(i):
                denom = np.sqrt(G_hat[i, i] * G_hat[j, j])
                corr = G_hat[i, j] / denom if denom > 0 else np.nan
                vc[f"cov({random_names[j]},{random_names[i]})"] = float(G_hat[i, j])
                vc[f"corr({random_names[j]},{random_names[i]})"] = float(corr)
    if dispersion is not None:
        if fam_key == "gaussian":
            vc["var(Residual)"] = float(dispersion)
        else:
            disp_label = "phi" if fam_key == "gamma" else "alpha"
            vc[f"dispersion({disp_label})"] = float(dispersion)

    method = "laplace" if nAGQ == 1 else f"AGHQ(nAGQ={nAGQ})"

    return MEGLMResult(
        fixed_effects=pd.Series(beta_hat, index=fixed_names),
        random_effects=random_effects_df,
        variance_components=vc,
        blups=blup_dict,
        n_obs=int(np.sum(w_all) if fam_key == "binomial" and trials else len(df)),
        n_groups=len(blocks),
        log_likelihood=float(-fun_hat),
        family=fam_key,
        link=fam.link,
        _se_fixed=pd.Series(se_beta, index=fixed_names),
        _cov_fixed=cov_beta,
        _cov_fixed_conditional=cov_beta_conditional,
        _cov_full=cov_full,
        _vce_method=vce_method,
        _G=G_hat,
        _x_fixed=x_fixed,
        _x_random=x_random_cols,
        _group_col=group,
        _fixed_names=fixed_names,
        _random_names=random_names,
        _y_name=y,
        _converged=outer_converged and inner_failures == 0,
        _method=method,
        _cov_type=cov_type,
        _alpha=alpha,
        _n_cov_params=n_cov_pars,
        _offset_name=offset,
        _dispersion=dispersion,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers — keep parity with Stata command names
# ---------------------------------------------------------------------------


def melogit(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group: str,
    x_random: Optional[Sequence[str]] = None,
    trials: Optional[str] = None,
    nAGQ: int = 1,
    **kw: Any,
) -> MEGLMResult:
    """Random-effects logistic regression (Stata ``melogit``).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(20), 12)
    >>> x = rng.normal(size=240)
    >>> u = rng.normal(0, 0.7, 20)[g]
    >>> p = 1.0 / (1.0 + np.exp(-(-0.3 + 0.8 * x + u)))
    >>> y = (rng.uniform(size=240) < p).astype(float)
    >>> df = pd.DataFrame({"y": y, "x": x, "gid": g})
    >>> res = sp.melogit(df, y="y", x_fixed=["x"], group="gid")
    >>> res.family
    'binomial'
    >>> bool(res.fixed_effects["x"] > 0)
    True
    """
    return meglm(
        data,
        y,
        x_fixed,
        group,
        family="binomial",
        x_random=x_random,
        trials=trials,
        nAGQ=nAGQ,
        **kw,
    )


def mepoisson(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group: str,
    x_random: Optional[Sequence[str]] = None,
    offset: Optional[str] = None,
    nAGQ: int = 1,
    **kw: Any,
) -> MEGLMResult:
    """Random-effects Poisson regression (Stata ``mepoisson``).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(20), 12)
    >>> x = rng.normal(size=240)
    >>> u = rng.normal(0, 0.7, 20)[g]
    >>> y = rng.poisson(np.exp(0.2 + 0.5 * x + u)).astype(float)
    >>> df = pd.DataFrame({"y": y, "x": x, "gid": g})
    >>> res = sp.mepoisson(df, y="y", x_fixed=["x"], group="gid")
    >>> res.family, res.link
    ('poisson', 'log')
    >>> bool(res.fixed_effects["x"] > 0)
    True
    """
    return meglm(
        data,
        y,
        x_fixed,
        group,
        family="poisson",
        x_random=x_random,
        offset=offset,
        nAGQ=nAGQ,
        **kw,
    )


def menbreg(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group: str,
    x_random: Optional[Sequence[str]] = None,
    offset: Optional[str] = None,
    nAGQ: int = 1,
    **kw: Any,
) -> MEGLMResult:
    """Random-effects negative-binomial regression (Stata ``menbreg``).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(20), 12)
    >>> x = rng.normal(size=240)
    >>> u = rng.normal(0, 0.7, 20)[g]
    >>> y = rng.poisson(np.exp(0.2 + 0.5 * x + u)).astype(float)
    >>> df = pd.DataFrame({"y": y, "x": x, "gid": g})
    >>> res = sp.menbreg(df, y="y", x_fixed=["x"], group="gid")
    >>> res.family
    'nbinomial'
    >>> res.dispersion is not None  # NB-2 overdispersion alpha
    True
    """
    return meglm(
        data,
        y,
        x_fixed,
        group,
        family="nbinomial",
        x_random=x_random,
        offset=offset,
        nAGQ=nAGQ,
        **kw,
    )


def megamma(
    data: pd.DataFrame,
    y: str,
    x_fixed: Sequence[str],
    group: str,
    x_random: Optional[Sequence[str]] = None,
    offset: Optional[str] = None,
    nAGQ: int = 1,
    **kw: Any,
) -> MEGLMResult:
    """Random-effects Gamma GLMM with log link (Stata ``meglm`` ``family(gamma)``).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(20), 12)
    >>> x = rng.normal(size=240)
    >>> u = rng.normal(0, 0.7, 20)[g]
    >>> y = np.exp(0.3 + 0.4 * x + u) * rng.gamma(4.0, 0.25, size=240)
    >>> df = pd.DataFrame({"y": y, "x": x, "gid": g})
    >>> res = sp.megamma(df, y="y", x_fixed=["x"], group="gid")
    >>> res.family, res.link
    ('gamma', 'log')
    >>> res.dispersion is not None  # phi estimated by ML
    True
    """
    return meglm(
        data,
        y,
        x_fixed,
        group,
        family="gamma",
        x_random=x_random,
        offset=offset,
        nAGQ=nAGQ,
        **kw,
    )


__all__ = [
    "meglm",
    "melogit",
    "mepoisson",
    "menbreg",
    "megamma",
    "MEGLMResult",
]
