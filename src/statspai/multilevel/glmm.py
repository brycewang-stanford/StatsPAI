"""
Generalized linear mixed models (GLMM).

Estimation
----------
Two estimation paths are exposed via the ``nAGQ`` argument:

    nAGQ = 1   →  Laplace approximation (default).  Fast; equivalent to
                  lme4's ``glmer(nAGQ = 1)`` and Stata ``meglm`` default.
    nAGQ ≥ 2   →  Adaptive Gauss-Hermite quadrature (AGHQ) with ``nAGQ``
                  nodes per scalar random effect.  Required for unbiased
                  small-cluster binary outcomes; matches lme4
                  ``glmer(nAGQ = k)`` and Stata ``meglm intpoints(k)``.
                  Currently restricted to single scalar random effects
                  (``q = 1``, e.g. random intercept only) — same
                  restriction lme4 imposes; tensor-product AGHQ over
                  random slopes is not enabled because the cost grows
                  as nAGQ^q.

Families
--------
Five exponential / dispersion families are supported; each has its
canonical link plus the most common practical link reported by Stata
``meglm``:

    * ``gaussian``  identity link (``meglm`` Gaussian-with-RE).
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
approximated either by a first-order Laplace expansion around the
conditional mode û_j (``nAGQ=1``) or by adaptive Gauss-Hermite
quadrature recentered at û_j with curvature 1/H_j (``nAGQ>1``).

Fixed-effect covariance uses the standard GLMM observed-information
formula

    Cov(β̂) = ( Σ_j X_j' W_j X_j
              − Σ_j (X_j' W_j Z_j) H_j⁻¹ (Z_j' W_j X_j) )⁻¹,

evaluated at the optimum.  This expression is the leading-order
observed information of the marginal log-likelihood and is identical
for nAGQ ≥ 1; only the point estimates β̂, θ̂ change with nAGQ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import warnings

import numpy as np
import pandas as pd
from scipy import special, stats
from scipy.optimize import minimize

from ._core import (
    _GroupBlock,
    _group_blocks,
    _initial_theta,
    _n_cov_params,
    _prepare_frame,
    _unpack_G,
)


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

    @staticmethod
    def log_lik(
        y: np.ndarray, mu: np.ndarray, w: np.ndarray, dispersion: Optional[float]
    ) -> float:
        raise NotImplementedError


class _Gaussian(_Family):
    name = "gaussian"
    link = "identity"

    @staticmethod
    def inv_link(eta):
        return eta

    @staticmethod
    def irls_weight(mu, w, dispersion):
        return w

    @staticmethod
    def score_eta(y, mu, w, dispersion):
        return w * (y - mu)

    @staticmethod
    def log_lik(y, mu, w, dispersion):
        # Use a unit residual variance scale; the LMM path is preferred for
        # production Gaussian fits.  ``w`` here carries an observation weight
        # (typically 1.0) and serves the same role as σ⁻² in a known-variance
        # working likelihood.  This keeps the GLMM Gaussian path useful as a
        # cross-check against ``mixed()`` without re-introducing residual
        # variance into the optimizer state.
        return float(-0.5 * np.sum(w * (y - mu) ** 2))


class _Binomial(_Family):
    name = "binomial"
    link = "logit"

    @staticmethod
    def inv_link(eta):
        out = np.empty_like(eta, dtype=float)
        pos = eta >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
        ex = np.exp(eta[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out

    @staticmethod
    def irls_weight(mu, w, dispersion):
        return w * mu * (1.0 - mu)

    @staticmethod
    def score_eta(y, mu, w, dispersion):
        return y - w * mu

    @staticmethod
    def log_lik(y, mu, w, dispersion):
        mu = np.clip(mu, _EPS, 1.0 - _EPS)
        # Include the log-binomial-coefficient constant so the
        # Bernoulli (w=1) and trials-binomial (w>1) cases yield the full
        # log-likelihood — this matters for AIC comparability with other
        # families fit to the same y.  For w=1 the constant is 0 and the
        # expression reduces to the Bernoulli log-lik.
        log_coef = special.gammaln(w + 1.0) - special.gammaln(y + 1.0) \
                   - special.gammaln(w - y + 1.0)
        return float(np.sum(
            log_coef + y * np.log(mu) + (w - y) * np.log(1.0 - mu)
        ))


class _Poisson(_Family):
    name = "poisson"
    link = "log"

    @staticmethod
    def inv_link(eta):
        return np.exp(np.clip(eta, -30.0, 30.0))

    @staticmethod
    def irls_weight(mu, w, dispersion):
        return mu

    @staticmethod
    def score_eta(y, mu, w, dispersion):
        return y - mu

    @staticmethod
    def log_lik(y, mu, w, dispersion):
        mu = np.clip(mu, 1e-300, None)
        # Include -log(y!) so AIC is comparable to negative-binomial fits
        # on the same y (NB collapses to Poisson + log(y!) as α → 0).
        return float(np.sum(y * np.log(mu) - mu - special.gammaln(y + 1.0)))


class _Gamma(_Family):
    """
    Gamma family with log link (``meglm`` ``family(gamma) link(log)``).

    Density (mean-dispersion form):

        f(y; μ, φ) = (1/(y Γ(1/φ))) · (y / (μ φ))^(1/φ) · exp(-y/(μ φ)),
        E[Y] = μ,   Var(Y) = φ μ².

    Dispersion ``φ`` is estimated jointly with (β, θ) through the packed
    parameter ``log φ``.  IRLS weight uses the **expected** Fisher
    information ``W = 1/φ`` per observation (Fisher scoring), which is
    always positive — observed info ``W = y/(μ φ)`` would lose definiteness
    when y < μ.  The score on η is the canonical-Pearson residual scaled
    by 1/φ.
    """

    name = "gamma"
    link = "log"
    n_disp_params = 1

    @staticmethod
    def inv_link(eta):
        return np.exp(np.clip(eta, -30.0, 30.0))

    @staticmethod
    def irls_weight(mu, w, dispersion):
        phi = dispersion if dispersion is not None else 1.0
        return (w / max(phi, _EPS)) * np.ones_like(mu)

    @staticmethod
    def score_eta(y, mu, w, dispersion):
        phi = dispersion if dispersion is not None else 1.0
        return w * (y - mu) / (mu * max(phi, _EPS))

    @staticmethod
    def log_lik(y, mu, w, dispersion):
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
        return float(np.sum(w * ll))


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
    def inv_link(eta):
        return np.exp(np.clip(eta, -30.0, 30.0))

    @staticmethod
    def irls_weight(mu, w, dispersion):
        alpha = dispersion if dispersion is not None else 0.0
        return w * mu / (1.0 + alpha * mu)

    @staticmethod
    def score_eta(y, mu, w, dispersion):
        alpha = dispersion if dispersion is not None else 0.0
        return w * (y - mu) / (1.0 + alpha * mu)

    @staticmethod
    def log_lik(y, mu, w, dispersion):
        alpha = dispersion if dispersion is not None else 0.0
        if alpha <= _EPS:
            # Poisson limit
            mu_c = np.clip(mu, 1e-300, None)
            return float(np.sum(w * (y * np.log(mu_c) - mu_c - special.gammaln(y + 1))))
        inv_a = 1.0 / alpha
        am = alpha * np.clip(mu, _EPS, None)
        ll = (
            special.gammaln(y + inv_a)
            - special.gammaln(inv_a)
            - special.gammaln(y + 1.0)
            - inv_a * np.log1p(am)
            + y * (np.log(am) - np.log1p(am))
        )
        return float(np.sum(w * ll))


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
    key = name.lower()
    key = _FAMILY_ALIASES.get(key, key)
    if key not in _FAMILIES:
        raise ValueError(
            f"family must be one of {sorted(_FAMILIES)} (with aliases "
            f"{sorted(_FAMILY_ALIASES)}); got {name!r}."
        )
    return _FAMILIES[key]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MEGLMResult:
    """Container for GLMM fits (``meglm``, ``melogit``, ``mepoisson``,
    ``megamma``, ``menbreg``, ``meologit``)."""

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
    _cov_fixed: np.ndarray = field(default=None, repr=False)
    _G: np.ndarray = field(default=None, repr=False)
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
        return 2.0 * (1.0 - stats.norm.cdf(z))

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
        return self.n_params * np.log(self.n_obs) - 2.0 * self.log_likelihood

    @property
    def dispersion(self) -> Optional[float]:
        """Estimated dispersion (φ for gamma, α for nbinomial); None otherwise."""
        return self._dispersion

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        z = stats.norm.ppf(1 - alpha / 2)
        lo = self.fixed_effects - z * self._se_fixed
        hi = self.fixed_effects + z * self._se_fixed
        return pd.DataFrame({"lower": lo, "upper": hi})

    def odds_ratios(self) -> pd.DataFrame:
        """Exponentiated fixed effects with Wald CIs (binomial / ordinal)."""
        if self.family not in ("binomial", "ordinal"):
            raise ValueError(
                "odds_ratios() is meaningful for binomial and ordinal GLMMs only"
            )
        ci = self.conf_int()
        return pd.DataFrame({
            "OR": np.exp(self.fixed_effects),
            "lower": np.exp(ci["lower"]),
            "upper": np.exp(ci["upper"]),
        })

    def incidence_rate_ratios(self) -> pd.DataFrame:
        """Exponentiated coefficients for log-link count GLMMs (Poisson / NB)."""
        if self.family not in ("poisson", "nbinomial"):
            raise ValueError(
                "IRR is meaningful for Poisson and negative-binomial GLMMs only"
            )
        ci = self.conf_int()
        return pd.DataFrame({
            "IRR": np.exp(self.fixed_effects),
            "lower": np.exp(ci["lower"]),
            "upper": np.exp(ci["upper"]),
        })

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
            disp_label = {"gamma": "phi", "nbinomial": "alpha"}.get(
                self.family, "dispersion"
            )
            lines.append(f"  Dispersion ({disp_label}): {self._dispersion:.6f}")
        lines.append("-" * w)

        z_crit = stats.norm.ppf(1 - self._alpha / 2)
        lines.append("Fixed effects:")
        hdr = (
            f"{'':>18s} {'Coef':>10s} {'Std.Err':>10s} "
            f"{'z':>8s} {'P>|z|':>8s}  [{100*(1-self._alpha):.0f}% CI]"
        )
        lines.append(hdr)
        lines.append("-" * w)
        for var in self.fixed_effects.index:
            b = self.fixed_effects[var]
            se = self._se_fixed[var] if self._se_fixed is not None else np.nan
            z = b / se if se and se > 0 else np.nan
            p = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else np.nan
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

    def to_markdown(self) -> str:
        rows = []
        for var in self.fixed_effects.index:
            b = self.fixed_effects[var]
            se = self._se_fixed[var]
            z = b / se if se else float("nan")
            p = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else float("nan")
            rows.append(f"| {var} | {b:.4f} | {se:.4f} | {z:.3f} | {p:.4f} |")
        vc_rows = [
            f"| {n} | {v:.6f} |" for n, v in self.variance_components.items()
        ]
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

    def cite(self) -> str:
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

    def to_latex(self) -> str:
        """Booktabs LaTeX fragment; mirrors ``MixedResult.to_latex``."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{self.family.capitalize()} GLMM ({self.link} link)}}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Variable & Coef. & Std.\ Err. & $z$ & $P>|z|$ \\",
            r"\midrule",
        ]
        for var in self.fixed_effects.index:
            b = self.fixed_effects[var]
            se = self._se_fixed[var]
            z = b / se if se else float("nan")
            p = 2 * (1 - stats.norm.cdf(abs(z))) if z == z else float("nan")
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

    def plot(self, kind: str = "caterpillar", variable: Optional[str] = None, **kwargs):
        """
        Diagnostic plot for the GLMM fit.

        Only ``kind='caterpillar'`` is currently supported: a forest plot
        of the BLUPs for one random effect (default: the intercept).
        Posterior SEs are not returned by ``meglm`` at present, so the
        error bars are omitted.
        """
        import matplotlib.pyplot as plt  # local import to keep plotting optional

        if kind != "caterpillar":
            raise ValueError(f"unknown plot kind {kind!r}")

        name = variable if variable is not None else self._random_names[0]
        if name not in self.random_effects.columns:
            raise ValueError(f"random effect {name!r} not in model")
        u = self.random_effects[name].copy().sort_values()
        fig, ax = plt.subplots(**{"figsize": (6, 0.2 * len(u) + 1), **kwargs})
        y_pos = np.arange(len(u))
        ax.plot(u.values, y_pos, "o", ms=3)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(i) for i in u.index], fontsize=7)
        ax.set_xlabel(f"BLUP of {name}")
        ax.set_title(
            f"Caterpillar plot ({self.family} GLMM): random {name}"
        )
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
            raise ValueError("predict() requires a dataframe for GLMMs")
        X = np.column_stack(
            [np.ones(len(data))] + [data[c].to_numpy(dtype=float) for c in self._x_fixed]
        )
        eta = X @ self.fixed_effects.values
        if include_random:
            Z = np.column_stack(
                [np.ones(len(data))]
                + [data[c].to_numpy(dtype=float) for c in self._x_random]
            )
            keys = list(data[self._group_col].values)
            u_mat = np.zeros_like(Z)
            for i, k in enumerate(keys):
                u = self.blups.get(k)
                if u is not None:
                    u_mat[i, :] = u
            eta = eta + np.einsum("ij,ij->i", Z, u_mat)
        if self._offset_name and self._offset_name in data.columns:
            eta = eta + data[self._offset_name].to_numpy(dtype=float)
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
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """
    Newton iteration for the conditional mode û_j with step damping.

    Returns ``(û_j, H_j, log|H_j|, converged)``.  ``converged`` is
    ``False`` when the tolerance is not met within ``max_inner`` iterations
    or when the Hessian cannot be factorised — the outer optimiser can
    then record a warning rather than silently using a stale mode.
    """
    u = u0.copy()
    converged = False
    for _ in range(max_inner):
        eta = block.X @ beta + block.Z @ u + offset
        mu = family.inv_link(eta)
        W = family.irls_weight(mu, weights, dispersion)
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

        u_new = u + step
        if step_norm < tol * (1 + u_norm):
            u = u_new
            converged = True
            break
        u = u_new

    eta = block.X @ beta + block.Z @ u + offset
    mu = family.inv_link(eta)
    W = family.irls_weight(mu, weights, dispersion)
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
    log_prior = -0.5 * (_LOG_2PI + np.log(max(sigma2, _EPS))) - 0.5 * u_grid ** 2 / max(sigma2, _EPS)
    # log integrand at node k: log f + log prior + x_k² + log w_k + ½ log(2 σ̂²)
    log_terms = (
        log_lik_vals
        + log_prior
        + nodes ** 2
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
):
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
        raise ValueError("AGHQ supports only q=1 random-intercept models")
    sigma2 = float(G[0, 0]) if use_aghq else None

    nll = 0.0
    for j, block in enumerate(blocks):
        w = weights_list[j]
        off = offsets_list[j]
        u_hat, H_j, logdet_H, _ = _find_mode(
            block, beta, G, Ginv, family, w, off, u_cache[j], dispersion
        )
        u_cache[j] = u_hat

        if use_aghq:
            # H_j is a 1×1 in q=1, so its scalar value is H_j[0,0].
            ll_j = _aghq_log_lik(
                block, beta, sigma2, family, w, off,
                float(u_hat[0]), float(H_j[0, 0]),
                gh_nodes, gh_log_weights, dispersion,
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
    tol: float = 1e-6,
    alpha: float = 0.05,
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
        Optimisation controls / CI width.

    Returns
    -------
    MEGLMResult
    """
    fam = _resolve_family(family)
    fam_key = fam.name
    if cov_type not in ("unstructured", "diagonal", "identity"):
        raise ValueError(f"unknown cov_type {cov_type!r}")
    if isinstance(group, (list, tuple)):
        if len(group) != 1:
            raise ValueError(
                "meglm() currently supports a single grouping variable; "
                "collapse nested levels into one key first."
            )
        group = group[0]
    if not isinstance(group, str):
        raise TypeError("`group` must be a column name string")

    nAGQ = int(nAGQ)
    if nAGQ < 1:
        raise ValueError(f"nAGQ must be >= 1, got {nAGQ}")

    x_fixed = list(x_fixed)
    x_random_cols: List[str] = list(x_random) if x_random is not None else []
    if nAGQ > 1 and len(x_random_cols) > 0:
        raise ValueError(
            "AGHQ (nAGQ > 1) currently supports only random-intercept models "
            "(empty x_random).  Use nAGQ=1 (Laplace) for random-slope models."
        )

    extra_cols = []
    if trials:
        extra_cols.append(trials)
    if offset:
        extra_cols.append(offset)

    df = _prepare_frame(
        data, y, x_fixed + extra_cols, [group], x_random_cols
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
    theta0 = np.concatenate([beta0, theta_cov0, theta_disp0])

    u_cache = [np.zeros(q_random) for _ in blocks]
    gh_nodes, gh_log_weights = (None, None)
    if nAGQ > 1:
        nodes, weights = _gh_nodes(nAGQ)
        gh_nodes = nodes
        gh_log_weights = np.log(weights)

    res = minimize(
        _glmm_nll,
        theta0,
        args=(
            blocks, weights_list, offsets_list,
            p_fixed, q_random, cov_type, fam, u_cache,
            nAGQ, gh_nodes, gh_log_weights,
        ),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol},
    )
    outer_converged = bool(res.success)

    beta_hat = res.x[:p_fixed]
    cov_hat = res.x[p_fixed : p_fixed + n_cov_pars]
    disp_hat_packed = res.x[p_fixed + n_cov_pars :]
    G_hat = _unpack_G(cov_hat, q_random, cov_type)
    Ginv = np.linalg.inv(G_hat)
    dispersion = fam.parse_dispersion(disp_hat_packed) if n_disp else None

    # BLUPs and fixed-effect info matrix at the optimum.
    blup_rows: List[Dict[str, float]] = []
    blup_dict: Dict[Any, np.ndarray] = {}
    keys: List[Any] = []

    info = np.zeros((p_fixed, p_fixed))
    inner_failures = 0
    for j, (block, w, off) in enumerate(zip(blocks, weights_list, offsets_list)):
        u0 = u_cache[j]
        u_hat, H_j, _, inner_ok = _find_mode(
            block, beta_hat, G_hat, Ginv, fam, w, off, u0, dispersion
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
        cov_beta = np.linalg.inv(info)
        se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
    except np.linalg.LinAlgError:
        cov_beta = np.full((p_fixed, p_fixed), np.nan)
        se_beta = np.full(p_fixed, np.nan)

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
        log_likelihood=float(-res.fun),
        family=fam_key,
        link=fam.link,
        _se_fixed=pd.Series(se_beta, index=fixed_names),
        _cov_fixed=cov_beta,
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
    """Random-effects logistic regression (Stata ``melogit``)."""
    return meglm(
        data, y, x_fixed, group,
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
    """Random-effects Poisson regression (Stata ``mepoisson``)."""
    return meglm(
        data, y, x_fixed, group,
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
    """Random-effects negative-binomial regression (Stata ``menbreg``)."""
    return meglm(
        data, y, x_fixed, group,
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
    """Random-effects Gamma GLMM with log link (Stata ``meglm`` ``family(gamma)``)."""
    return meglm(
        data, y, x_fixed, group,
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
